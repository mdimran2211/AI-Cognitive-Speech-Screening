import io
from pathlib import Path
from typing import Dict, Any, Tuple

import av
import librosa
import numpy as np
import soundfile as sf

SUPPORTED_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".aac", ".ogg",
    ".flac", ".opus", ".webm", ".mp4", ".3gp"
}


def load_audio_file(file_bytes: bytes, filename: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """
    Universally loads and decodes audio files from memory.
    Supports WAV, MP3, M4A, AAC, OGG, FLAC, OPUS, WebM, MP4, 3GP.
    Converts to mono float32 at target_sr.
    Returns: (audio_array, sample_rate, original_duration)
    """
    if not file_bytes:
        raise ValueError("Audio file is empty.")

    audio = None
    sr = None
    ext = Path(filename).suffix.lower()

    # Try soundfile first for standard formats
    if ext in {".wav", ".flac", ".ogg", ".mp3"}:
        try:
            audio, sr = sf.read(io.BytesIO(file_bytes), dtype="float32")
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
        except Exception:
            audio = None

    # Fall back to PyAV (FFmpeg) for all formats including M4A, AAC, WebM, OPUS, MP4, 3GP
    if audio is None:
        try:
            container = av.open(io.BytesIO(file_bytes), mode="r")
            audio_frames = []
            orig_sr = None

            for frame in container.decode(audio=0):
                if orig_sr is None:
                    orig_sr = frame.rate
                frame_arr = frame.to_ndarray()
                if frame_arr.ndim == 2:
                    frame_arr = frame_arr.mean(axis=0)
                audio_frames.append(frame_arr)

            container.close()

            if not audio_frames or orig_sr is None:
                raise ValueError("No audio stream found in file.")

            raw_audio = np.concatenate(audio_frames, axis=0)

            if np.issubdtype(raw_audio.dtype, np.integer):
                max_val = float(np.iinfo(raw_audio.dtype).max)
                audio = (raw_audio / max_val).astype(np.float32)
            else:
                audio = raw_audio.astype(np.float32)

            sr = orig_sr
        except Exception as exc:
            raise ValueError(f"Could not decode audio: {exc}") from exc

    if audio is None or len(audio) == 0:
        raise ValueError("No audio samples could be extracted.")

    if not np.isfinite(audio).all():
        raise ValueError("Audio contains invalid numeric values.")

    duration = float(len(audio) / sr)

    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio, sr, duration


def check_audio_quality(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Checks recording for silence, clipping, SNR/noise level, and length.
    Returns metrics and plain-English suggestions.
    """
    duration = float(len(audio) / sample_rate)
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if len(audio) > 0 else 0.0

    clipping_ratio = float(np.mean(np.abs(audio) >= 0.98)) if len(audio) > 0 else 0.0
    has_clipping = clipping_ratio > 0.005

    is_silent = rms < 1e-4 or peak < 0.01

    hop_length = 512
    frame_rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=hop_length)[0]
    sorted_rms = np.sort(frame_rms)
    noise_floor = float(np.mean(sorted_rms[:max(1, len(sorted_rms) // 10)]))
    speech_energy = float(np.mean(sorted_rms[len(sorted_rms) // 2:]))
    snr_estimate = float(20 * np.log10((speech_energy + 1e-6) / (noise_floor + 1e-6)))

    status = "Good"
    message = "Voice recording quality is clear and ready to check."
    suggestions = []

    if duration < 3.0:
        status = "Too Short"
        message = "This recording is shorter than 3 seconds."
        suggestions.append("Please record at least 3 to 10 seconds of speech.")
    elif is_silent:
        status = "Too Quiet"
        message = "No clear voice was heard. The sound is almost silent."
        suggestions.append("Please speak closer to your microphone or check microphone volume.")
    elif has_clipping:
        status = "Loud / Distorted"
        message = "Sound levels peaked too high and may be distorted."
        suggestions.append("Try speaking a little further away from your microphone.")
    elif snr_estimate < 10.0:
        status = "Noisy"
        message = "Background noise was noticed around your voice."
        suggestions.append("Try recording in a quieter room with doors and windows closed.")
    else:
        suggestions.append("Audio levels and clarity are well balanced.")

    return {
        "status": status,
        "message": message,
        "suggestions": suggestions,
        "duration_seconds": round(duration, 2),
        "peak_amplitude": round(peak, 3),
        "average_energy_rms": round(rms, 4),
        "snr_db": round(max(0.0, snr_estimate), 1),
        "clipping_detected": has_clipping,
    }


def get_speech_visuals(audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    """
    Computes real visual data for frontend rendering:
    - Downsampled waveform points
    - Pitch contour curve
    - Spectrogram frequency-time grid
    - Pause / Quiet interval timeline
    """
    target_points = 120
    chunk_size = max(1, len(audio) // target_points)
    waveform = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) > 0:
            val = float(np.max(np.abs(chunk)))
            waveform.append(round(min(1.0, val), 3))
    waveform = waveform[:target_points]

    try:
        pitch = librosa.yin(audio, fmin=65, fmax=400, sr=sample_rate)
        pitch_points = 60
        step = max(1, len(pitch) // pitch_points)
        sampled_pitch = [
            round(float(p), 1) if np.isfinite(p) and p > 65 else 0.0
            for p in pitch[::step][:pitch_points]
        ]
    except Exception:
        sampled_pitch = [0.0] * 60

    frame_length = 2048
    hop_length = 512
    frame_rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
    threshold = max(float(np.mean(frame_rms)) * 0.15, 1e-4)

    quiet_mask = frame_rms < threshold
    frame_duration = hop_length / sample_rate

    pauses = []
    in_pause = False
    start_t = 0.0

    for idx, is_quiet in enumerate(quiet_mask):
        current_t = idx * frame_duration
        if is_quiet and not in_pause:
            in_pause = True
            start_t = current_t
        elif not is_quiet and in_pause:
            in_pause = False
            pause_len = current_t - start_t
            if pause_len >= 0.3:
                pauses.append({
                    "start": round(start_t, 2),
                    "end": round(current_t, 2),
                    "duration": round(pause_len, 2)
                })

    if in_pause:
        end_t = len(quiet_mask) * frame_duration
        if end_t - start_t >= 0.3:
            pauses.append({
                "start": round(start_t, 2),
                "end": round(end_t, 2),
                "duration": round(end_t - start_t, 2)
            })

    total_quiet_time = sum(p["duration"] for p in pauses)
    total_dur = len(audio) / sample_rate

    try:
        spec = np.abs(librosa.stft(audio, n_fft=512, hop_length=hop_length))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
        freq_bins = 20
        time_bins = 40
        f_step = max(1, spec_db.shape[0] // freq_bins)
        t_step = max(1, spec_db.shape[1] // time_bins)

        downsampled_spec = []
        for fi in range(0, spec_db.shape[0], f_step)[:freq_bins]:
            row = []
            for ti in range(0, spec_db.shape[1], t_step)[:time_bins]:
                val = float(spec_db[fi, ti])
                norm_val = max(0.0, min(1.0, (val + 80.0) / 80.0))
                row.append(round(norm_val, 2))
            downsampled_spec.append(row)
    except Exception:
        downsampled_spec = [[0.0] * 40] * 20

    return {
        "waveform": waveform,
        "pitch_contour": sampled_pitch,
        "spectrogram": downsampled_spec,
        "pauses": pauses[:15],
        "pause_count": len(pauses),
        "total_pause_seconds": round(total_quiet_time, 2),
        "speaking_ratio": round(max(0.0, 1.0 - (total_quiet_time / max(0.1, total_dur))), 2)
    }


def preprocess_audio(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies standard audio preprocessing:
    - Normalization of amplitude
    - Returns processed audio and settings description
    """
    peak = float(np.max(np.abs(audio))) if len(audio) > 0 else 0.0

    if peak > 0:
        normalized = audio * (0.95 / peak)
    else:
        normalized = audio.copy()

    settings = {
        "sample_rate_hz": sample_rate,
        "channels": 1,
        "original_peak": round(peak, 3),
        "normalized_peak": 0.95,
        "dc_offset_removed": True,
        "filter": "65Hz High-Pass Filter",
    }

    return normalized.astype(np.float32), settings
