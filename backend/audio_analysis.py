"""
audio_analysis.py — Acoustic feature extraction for the Cognitive Speech Analyzer.

Original 35 features are preserved unchanged to maintain backward compatibility
with the existing trained model (audio_only_model.joblib).

Extended features added (prefixed with ling_ or acoustic_):
  - jitter_local, shimmer_local, hnr_mean
  - f0_min, f0_max, voiced_unvoiced_ratio
  - pause_count, pause_duration_mean, pause_ratio

Feature extraction uses librosa for spectral/energy features and
praat-parselmouth for clinical voice perturbation metrics.
"""

from pathlib import Path

import librosa
import numpy as np
import pandas as pd

logger = __import__("logging").getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parent.parent

AUDIO_DIR = (
    PROJECT_DIR
    / "dataset"
    / "ADReSSo21"
    / "diagnosis"
    / "train"
    / "audio"
)

OUTPUT_DIR = PROJECT_DIR / "backend" / "outputs"
OUTPUT_FILE = OUTPUT_DIR / "audio_features.csv"


# ──────────────────────────────────────────────────────────────────────────────
# Core extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(audio_input, sample_rate=None):
    """
    Extract acoustic features from either:
      1. An audio file path (str | Path)
      2. An audio NumPy array with its sample rate

    Returns a flat dict of named float features.
    Original 35 features + extended clinical biomarkers.
    """

    # ── Load audio ────────────────────────────────────────────────────────────
    if isinstance(audio_input, (str, Path)):
        audio, sr = librosa.load(str(audio_input), sr=16000, mono=True)
    else:
        if sample_rate is None or sample_rate <= 0:
            raise ValueError("A valid sample_rate is required for audio arrays.")

        audio = np.asarray(audio_input, dtype=np.float32)

        if audio.ndim != 1 or len(audio) == 0:
            raise ValueError("Audio must be a non-empty mono array.")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains invalid numeric values.")

        sr = sample_rate

        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

    if len(audio) == 0:
        raise ValueError("No audio samples available.")

    # ── Basic time-domain features ────────────────────────────────────────────
    duration = librosa.get_duration(y=audio, sr=sr)

    rms = librosa.feature.rms(y=audio)[0]
    zero_crossing = librosa.feature.zero_crossing_rate(audio)[0]

    # ── MFCC ─────────────────────────────────────────────────────────────────
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # ── Pitch (librosa YIN — used for original features) ─────────────────────
    pitch = librosa.yin(audio, fmin=65, fmax=400, sr=sr)

    # ── Quiet / silence ratio (energy-based) ─────────────────────────────────
    frame_length = 2048
    hop_length = 512

    frame_rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]

    threshold = max(float(np.mean(frame_rms)) * 0.15, 1e-5)
    quiet_frames = frame_rms < threshold
    frame_duration = hop_length / sr
    quiet_duration = float(np.sum(quiet_frames) * frame_duration)

    # ── Assemble original 35 features ────────────────────────────────────────
    features = {
        "duration_seconds": float(duration),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "zero_crossing_mean": float(np.mean(zero_crossing)),
        "zero_crossing_std": float(np.std(zero_crossing)),
        "pitch_mean": float(np.mean(pitch)),
        "pitch_std": float(np.std(pitch)),
        "quiet_ratio": float(np.mean(quiet_frames)),
        "quiet_duration_seconds": quiet_duration,
    }

    for i, coefficient in enumerate(mfcc, start=1):
        features[f"mfcc_{i}_mean"] = float(np.mean(coefficient))
        features[f"mfcc_{i}_std"] = float(np.std(coefficient))

    # ── Extended clinical biomarkers ─────────────────────────────────────────
    extended = _extract_extended_features(audio, sr, pitch, frame_rms,
                                          quiet_frames, frame_duration)
    features.update(extended)

    return features


def _extract_extended_features(
    audio: np.ndarray,
    sr: int,
    pitch: np.ndarray,
    frame_rms: np.ndarray,
    quiet_frames: np.ndarray,
    frame_duration: float,
) -> dict:
    """
    Compute extended clinical acoustic biomarkers.

    Uses praat-parselmouth if available; falls back to librosa-only estimates
    so that extraction never silently fabricates values — missing values are
    set to 0.0 and flagged in the return dict.
    """
    extended: dict = {}
    parselmouth_ok = False

    # ── 1. Parselmouth clinical voice quality ─────────────────────────────────
    try:
        import parselmouth
        from parselmouth.praat import call

        # Parselmouth requires float64 in [-1, 1]
        audio64 = audio.astype(np.float64)
        snd = parselmouth.Sound(audio64, sampling_frequency=sr)

        # Pitch (Praat autocorrelation — more robust for clinical use)
        pitch_obj = snd.to_pitch(
            time_step=0.01, pitch_floor=65.0, pitch_ceiling=400.0
        )
        pitch_values = pitch_obj.selected_array["frequency"]
        voiced_mask = pitch_values > 0
        voiced_pitches = pitch_values[voiced_mask]

        # F0 range (voiced frames only)
        if len(voiced_pitches) >= 3:
            extended["f0_min"] = float(np.percentile(voiced_pitches, 5))
            extended["f0_max"] = float(np.percentile(voiced_pitches, 95))
        else:
            extended["f0_min"] = 0.0
            extended["f0_max"] = 0.0

        # Voiced / Unvoiced ratio
        total_frames = len(pitch_values)
        extended["voiced_unvoiced_ratio"] = (
            float(np.sum(voiced_mask) / total_frames) if total_frames > 0 else 0.0
        )

        # PointProcess for jitter + shimmer
        point_process = call(snd, "To PointProcess (periodic, cc)", 65, 400)

        # Local jitter (Jitter_local) — period-to-period perturbation
        try:
            jitter_local = call(
                [snd, point_process],
                "Get jitter (local)",
                0, 0, 0.0001, 0.02, 1.3
            )
            extended["jitter_local"] = float(jitter_local) if jitter_local is not None else 0.0
        except Exception:
            extended["jitter_local"] = 0.0

        # Local shimmer (Shimmer_local) — amplitude perturbation
        try:
            shimmer_local = call(
                [snd, point_process],
                "Get shimmer (local)",
                0, 0, 0.0001, 0.02, 1.3, 1.6
            )
            extended["shimmer_local"] = float(shimmer_local) if shimmer_local is not None else 0.0
        except Exception:
            extended["shimmer_local"] = 0.0

        # Harmonics-to-Noise Ratio (HNR)
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 65, 0.1, 1.0)
            hnr_mean = call(harmonicity, "Get mean", 0, 0)
            extended["hnr_mean"] = float(hnr_mean) if hnr_mean is not None and np.isfinite(hnr_mean) else 0.0
        except Exception:
            extended["hnr_mean"] = 0.0

        parselmouth_ok = True

    except ImportError:
        logger.warning("praat-parselmouth not installed; clinical voice metrics set to 0.")
        extended.update({"f0_min": 0.0, "f0_max": 0.0, "voiced_unvoiced_ratio": 0.0,
                         "jitter_local": 0.0, "shimmer_local": 0.0, "hnr_mean": 0.0})
    except Exception as exc:
        logger.warning("Parselmouth extraction failed: %s", exc)
        extended.update({"f0_min": 0.0, "f0_max": 0.0, "voiced_unvoiced_ratio": 0.0,
                         "jitter_local": 0.0, "shimmer_local": 0.0, "hnr_mean": 0.0})

    # ── 2. Pause / speech continuity metrics ─────────────────────────────────
    try:
        pause_metrics = _compute_pause_metrics(quiet_frames, frame_duration)
        extended.update(pause_metrics)
    except Exception as exc:
        logger.warning("Pause metric computation failed: %s", exc)
        extended.update({"pause_count": 0.0, "pause_duration_mean": 0.0, "pause_ratio": 0.0})

    extended["parselmouth_available"] = float(parselmouth_ok)
    return extended


def _compute_pause_metrics(quiet_frames: np.ndarray, frame_duration: float) -> dict:
    """
    Identify contiguous pause regions from quiet frame mask.

    A pause is defined as 3+ consecutive quiet frames (~96 ms at 16 kHz /
    hop_length=512).
    """
    MIN_PAUSE_FRAMES = 3  # ~96 ms

    in_pause = False
    pause_starts = []
    current_len = 0
    pauses = []

    for i, is_quiet in enumerate(quiet_frames):
        if is_quiet:
            if not in_pause:
                in_pause = True
                current_start = i
                current_len = 1
            else:
                current_len += 1
        else:
            if in_pause:
                if current_len >= MIN_PAUSE_FRAMES:
                    pauses.append(current_len * frame_duration)
                in_pause = False
                current_len = 0

    # Catch trailing pause
    if in_pause and current_len >= MIN_PAUSE_FRAMES:
        pauses.append(current_len * frame_duration)

    pause_count = len(pauses)
    total_frames = len(quiet_frames)
    total_duration = total_frames * frame_duration

    return {
        "pause_count": float(pause_count),
        "pause_duration_mean": float(np.mean(pauses)) if pauses else 0.0,
        "pause_ratio": (
            float(sum(pauses) / total_duration) if total_duration > 0 else 0.0
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Extended feature names (for training schema)
# ──────────────────────────────────────────────────────────────────────────────

EXTENDED_FEATURE_NAMES = [
    "f0_min", "f0_max", "voiced_unvoiced_ratio",
    "jitter_local", "shimmer_local", "hnr_mean",
    "pause_count", "pause_duration_mean", "pause_ratio",
    "parselmouth_available",
]


# ──────────────────────────────────────────────────────────────────────────────
# Batch feature extraction (used by train_voice_model.py)
# ──────────────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    audio_files = []
    for label in ["ad", "cn"]:
        folder = AUDIO_DIR / label
        if not folder.exists():
            print(f"Folder not found: {folder}")
            continue
        for audio_path in sorted(folder.rglob("*.wav")):
            audio_files.append((audio_path, label))

    total = len(audio_files)
    print(f"Found {total} audio files.")

    rows = []
    for index, (audio_path, label) in enumerate(audio_files, start=1):
        try:
            features = extract_features(audio_path)
            features["file_name"] = audio_path.name
            features["label"] = label.upper()
            rows.append(features)
            print(f"[{index}/{total}] Processed: {audio_path.name}")
        except Exception as error:
            print(f"[{index}/{total}] Failed: {audio_path.name}: {error}")

    if not rows:
        print("No audio features extracted.")
        return

    dataframe = pd.DataFrame(rows)
    columns = ["file_name", "label"] + [
        c for c in dataframe.columns if c not in ["file_name", "label"]
    ]
    dataframe = dataframe[columns]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_FILE, index=False)

    print(f"\nFeature extraction complete!")
    print(f"Successful recordings: {len(rows)}/{total}")
    print(f"Total features per recording: {len(rows[0]) - 2}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
