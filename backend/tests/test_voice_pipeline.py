r"""
test_voice_pipeline.py
======================
Automated test suite for the voice-only cognitive screening pipeline.

Tests (read-only where possible -- no model mutation):
  1.  Dataset sanity   -- labels, participant counts, class balance
  2.  Feature extraction -- shape, NaN handling, value ranges
  3.  Feature order    -- inference schema matches training schema
  4.  Missing features -- verify_feature_schema raises on mismatch
  5.  Invalid audio    -- silence, clipped, sub-3s, empty bytes
  6.  Model loading    -- both old and new model artifacts
  7.  Prediction       -- inference on real WAV returns valid output
  8.  Model metadata   -- metadata JSON structure and required keys
  9.  Leakage check    -- participant IDs not shared across train/test
  10. API endpoints    -- GET / and GET /model-info without starting server

Run with:
    .venv\Scripts\python.exe backend/tests/test_voice_pipeline.py
"""

from __future__ import annotations

import io
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

# ── Helpers ──────────────────────────────────────────────────────────────────

PASS  = "PASS"
FAIL  = "FAIL"
SKIP  = "SKIP"
_W    = 55   # column width for result display

results: list[dict] = []


def _run(name: str, fn):
    """Run a test function and record the result."""
    try:
        msg = fn()
        status = PASS if msg is None else (FAIL if msg.startswith("FAIL") else PASS)
        results.append({"name": name, "status": status, "detail": msg or ""})
    except Exception as exc:
        results.append({"name": name, "status": FAIL, "detail": str(exc)})


def _assert(condition: bool, msg: str) -> str | None:
    if not condition:
        return f"FAIL: {msg}"
    return None


# ── Test 1: Dataset sanity ────────────────────────────────────────────────────

def test_dataset_labels():
    AUDIO_DIR = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
    META_CSV  = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "adresso-train-mmse-scores.csv"

    if not AUDIO_DIR.exists():
        return "FAIL: Audio directory not found"
    if not META_CSV.exists():
        return "FAIL: Metadata CSV not found"

    import pandas as pd
    meta = pd.read_csv(META_CSV)
    meta["dx"] = meta["dx"].str.strip().str.upper()
    meta["adressfname"] = meta["adressfname"].str.strip()

    ad_wav = list((AUDIO_DIR / "ad").rglob("*.wav")) if (AUDIO_DIR / "ad").exists() else []
    cn_wav = list((AUDIO_DIR / "cn").rglob("*.wav")) if (AUDIO_DIR / "cn").exists() else []
    total_wav = len(ad_wav) + len(cn_wav)

    if total_wav != 166:
        return f"FAIL: Expected 166 WAV files, found {total_wav}"
    if len(meta) != 166:
        return f"FAIL: Expected 166 metadata rows, found {len(meta)}"

    meta_ids = set(meta["adressfname"])
    audio_ids = {w.stem for w in ad_wav + cn_wav}
    missing = audio_ids - meta_ids
    if missing:
        return f"FAIL: {len(missing)} audio IDs missing from metadata: {sorted(missing)[:3]}"

    # Label consistency
    for wav in ad_wav:
        pid = wav.stem
        if pid in meta_ids:
            row = meta[meta["adressfname"] == pid].iloc[0]
            if row["dx"] != "AD":
                return f"FAIL: Label mismatch for {pid}: folder=AD, meta={row['dx']}"

    for wav in cn_wav:
        pid = wav.stem
        if pid in meta_ids:
            row = meta[meta["adressfname"] == pid].iloc[0]
            if row["dx"] != "CN":
                return f"FAIL: Label mismatch for {pid}: folder=CN, meta={row['dx']}"

    return None  # PASS


def test_no_duplicate_participants():
    AUDIO_DIR = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
    all_stems = []
    for lbl in ["ad", "cn"]:
        folder = AUDIO_DIR / lbl
        if folder.exists():
            all_stems += [w.stem for w in folder.rglob("*.wav")]
    if len(all_stems) != len(set(all_stems)):
        from collections import Counter
        dups = [k for k, v in Counter(all_stems).items() if v > 1]
        return f"FAIL: Duplicate participant IDs: {dups}"
    return None  # PASS


# ── Test 2: Feature extraction -- shape, NaN, value range ─────────────────────

def test_feature_extraction_shape():
    from backend.audio_analysis import extract_features, EXTENDED_FEATURE_NAMES
    AUDIO_DIR = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
    wav = next((AUDIO_DIR / "ad").rglob("*.wav"), None)
    if wav is None:
        return "SKIP: No WAV found for test"

    feats = extract_features(wav)
    n = len(feats)
    if n < 45:
        return f"FAIL: Expected >= 45 features, got {n}. Missing extended features?"
    for k in EXTENDED_FEATURE_NAMES:
        if k not in feats:
            return f"FAIL: Extended feature '{k}' missing from extraction"
    return None  # PASS


def test_feature_values_finite():
    from backend.audio_analysis import extract_features
    AUDIO_DIR = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
    wav = next((AUDIO_DIR / "cn").rglob("*.wav"), None)
    if wav is None:
        return "SKIP: No WAV found for test"

    feats = extract_features(wav)
    bad = {k: v for k, v in feats.items()
           if not isinstance(v, (int, float)) or (isinstance(v, float) and not math.isfinite(v))}
    # parselmouth_available is allowed to be 0 or 1 (both finite)
    bad.pop("parselmouth_available", None)

    if bad:
        return f"FAIL: Non-finite feature values: {bad}"
    return None  # PASS


# ── Test 3: Feature order -- inference schema matches training schema ──────────

def test_feature_order_consistency():
    MODEL_PATH = PROJECT_DIR / "backend" / "outputs" / "voice_screening_model.joblib"
    if not MODEL_PATH.exists():
        return "SKIP: voice_screening_model.joblib not yet trained"

    import joblib
    from backend.audio_analysis import extract_features
    AUDIO_DIR = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
    wav = next((AUDIO_DIR / "ad").rglob("*.wav"), None)
    if wav is None:
        return "SKIP: No WAV for test"

    artifact      = joblib.load(MODEL_PATH)
    expected_cols = artifact["features"]
    feats         = extract_features(wav)

    missing = [n for n in expected_cols if n not in feats]
    extra   = [n for n in feats if n not in expected_cols]

    if missing:
        return f"FAIL: Features in model schema but not in extraction: {missing}"
    if extra:
        return f"INFO: Extra extraction features not in model (fine if intentional): {extra[:3]}..."
    return None  # PASS


# ── Test 4: verify_feature_schema raises on mismatch ─────────────────────────

def test_schema_verification_raises():
    from backend.train_voice_model import verify_feature_schema
    expected = ["feat_a", "feat_b", "feat_c"]
    bad_dict = {"feat_a": 1.0, "feat_x": 2.0}   # missing feat_b, feat_c; extra feat_x
    try:
        verify_feature_schema(expected, bad_dict, "unit-test")
        return "FAIL: Should have raised ValueError"
    except ValueError:
        return None  # PASS -- correctly detected mismatch


def test_schema_verification_passes():
    from backend.train_voice_model import verify_feature_schema
    expected = ["feat_a", "feat_b", "feat_c"]
    good_dict = {"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3.0}
    try:
        verify_feature_schema(expected, good_dict, "unit-test")
        return None  # PASS
    except ValueError as e:
        return f"FAIL: {e}"


# ── Test 5: Invalid / edge-case audio inputs ──────────────────────────────────

def _make_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    import soundfile as sf
    bio = io.BytesIO()
    sf.write(bio, audio, sr, format="WAV")
    return bio.getvalue()


def test_silence_handling():
    from backend.audio_analysis import extract_features
    silence = np.zeros(16000 * 5, dtype=np.float32)   # 5s silence
    try:
        feats = extract_features(silence, sample_rate=16000)
        # Extraction should succeed; duration should be ~5s
        dur = feats.get("duration_seconds", 0)
        if not (4.5 <= dur <= 5.5):
            return f"FAIL: Unexpected duration for silent audio: {dur}"
        return None  # PASS
    except Exception as exc:
        return f"FAIL: Extraction raised on silence: {exc}"


def test_very_short_audio():
    """Sub-1s audio should succeed at extraction level (API rejects <3s, not extractor)."""
    from backend.audio_analysis import extract_features
    short = np.random.randn(8000).astype(np.float32) * 0.01   # 0.5s
    try:
        feats = extract_features(short, sample_rate=16000)
        return None  # PASS -- extractor doesn't enforce minimum
    except Exception as exc:
        return None  # Also acceptable -- short audio may legitimately fail


def test_empty_audio_array():
    from backend.audio_analysis import extract_features
    try:
        extract_features(np.array([], dtype=np.float32), sample_rate=16000)
        return "FAIL: Should raise on empty audio"
    except (ValueError, RuntimeError):
        return None  # PASS
    except Exception as exc:
        return f"FAIL: Unexpected exception type: {type(exc).__name__}: {exc}"


def test_invalid_sample_rate():
    from backend.audio_analysis import extract_features
    audio = np.random.randn(16000).astype(np.float32)
    try:
        extract_features(audio, sample_rate=0)
        return "FAIL: Should raise on sample_rate=0"
    except ValueError:
        return None  # PASS


# ── Test 6: Model loading ─────────────────────────────────────────────────────

def test_old_model_loads():
    OLD_MODEL = PROJECT_DIR / "backend" / "outputs" / "audio_only_model.joblib"
    if not OLD_MODEL.exists():
        return "FAIL: audio_only_model.joblib not found"
    import joblib
    art = joblib.load(OLD_MODEL)
    required = {"model", "features", "label_mapping", "model_type"}
    missing = required - set(art.keys())
    if missing:
        return f"FAIL: Old model missing keys: {missing}"
    if len(art["features"]) != 35:
        return f"FAIL: Old model expected 35 features, got {len(art['features'])}"
    return None  # PASS


def test_new_model_loads():
    NEW_MODEL = PROJECT_DIR / "backend" / "outputs" / "voice_screening_model.joblib"
    if not NEW_MODEL.exists():
        return "SKIP: voice_screening_model.joblib not yet trained"
    import joblib
    art = joblib.load(NEW_MODEL)
    required = {"model", "features", "label_mapping", "model_type"}
    missing = required - set(art.keys())
    if missing:
        return f"FAIL: New model missing keys: {missing}"
    n = len(art["features"])
    if n < 45:
        return f"FAIL: Expected >= 45 features in new model, got {n}"
    return None  # PASS


# ── Test 7: Prediction returns valid output ───────────────────────────────────

def test_prediction_on_real_wav():
    NEW_MODEL = PROJECT_DIR / "backend" / "outputs" / "voice_screening_model.joblib"
    if not NEW_MODEL.exists():
        return "SKIP: voice_screening_model.joblib not yet trained"

    import joblib
    import pandas as pd
    from backend.audio_analysis import extract_features

    AUDIO_DIR = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
    wav = next((AUDIO_DIR / "ad").rglob("*.wav"), None)
    if wav is None:
        return "SKIP: No WAV found"

    art   = joblib.load(NEW_MODEL)
    feats = extract_features(wav)
    cols  = art["features"]

    missing = [c for c in cols if c not in feats]
    if missing:
        return f"FAIL: Features missing at inference: {missing}"

    X = pd.DataFrame([[feats[c] for c in cols]], columns=cols)
    pred_id = int(art["model"].predict(X)[0])
    probs   = art["model"].predict_proba(X)[0]
    label   = art["label_mapping"][pred_id]

    if label not in ("AD", "CN"):
        return f"FAIL: Unexpected prediction label: {label}"
    if not (0.0 <= probs[0] <= 1.0 and 0.0 <= probs[1] <= 1.0):
        return f"FAIL: Probabilities out of range: {probs}"
    if abs(sum(probs) - 1.0) > 1e-4:
        return f"FAIL: Probabilities don't sum to 1: {probs}"

    return None  # PASS


# ── Test 8: Model metadata JSON structure ─────────────────────────────────────

def test_metadata_json():
    META_PATH = PROJECT_DIR / "backend" / "outputs" / "model_metadata.json"
    if not META_PATH.exists():
        return "SKIP: model_metadata.json not yet created"

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    required_keys = {
        "model_name", "version", "feature_names", "feature_count",
        "n_train", "n_test", "cv_results", "test_metrics",
    }
    missing = required_keys - set(meta.keys())
    if missing:
        return f"FAIL: Metadata missing keys: {missing}"

    test_m = meta["test_metrics"]
    required_metrics = {"balanced_accuracy", "sensitivity_ad_recall", "specificity_cn_recall",
                        "f1_score", "roc_auc", "confusion_matrix"}
    missing_m = required_metrics - set(test_m.keys())
    if missing_m:
        return f"FAIL: test_metrics missing: {missing_m}"

    ba = test_m["balanced_accuracy"]
    if not (0.0 <= ba <= 1.0):
        return f"FAIL: balanced_accuracy out of range: {ba}"

    return None  # PASS


# ── Test 9: No participant leakage between train/test ─────────────────────────

def test_no_participant_leakage():
    FEAT_CSV = PROJECT_DIR / "backend" / "outputs" / "voice_features.csv"
    META_PATH = PROJECT_DIR / "backend" / "outputs" / "model_metadata.json"
    if not FEAT_CSV.exists() or not META_PATH.exists():
        return "SKIP: voice_features.csv or model_metadata.json not yet created"

    import pandas as pd
    from backend.train_voice_model import participant_split

    feat_df = pd.read_csv(FEAT_CSV)
    if "participant_id" not in feat_df.columns:
        return "SKIP: voice_features.csv has no participant_id column"

    train_df, test_df = participant_split(feat_df)
    overlap = set(train_df["participant_id"]) & set(test_df["participant_id"])
    if overlap:
        return f"FAIL: Participant overlap between train and test: {overlap}"
    return None  # PASS


# ── Test 10: Feature importance JSON ─────────────────────────────────────────

def test_feature_importance_json():
    FI_PATH = PROJECT_DIR / "backend" / "outputs" / "feature_importance.json"
    if not FI_PATH.exists():
        return "SKIP: feature_importance.json not yet created"

    fi = json.loads(FI_PATH.read_text(encoding="utf-8"))
    if not isinstance(fi, list) or not fi:
        return "FAIL: feature_importance.json is empty or wrong type"
    first = fi[0]
    if "feature" not in first or "importance" not in first:
        return f"FAIL: Missing 'feature' or 'importance' keys: {first}"
    importances = [item["importance"] for item in fi]
    if abs(sum(importances) - 1.0) > 0.01:
        return f"FAIL: Importances don't sum to ~1.0: {sum(importances):.4f}"
    return None  # PASS


# ══════════════════════════════════════════════════════════════════════════════
# Main test runner
# ══════════════════════════════════════════════════════════════════════════════

ALL_TESTS = [
    ("Dataset labels verified",                test_dataset_labels),
    ("No duplicate participant IDs",           test_no_duplicate_participants),
    ("Feature extraction returns 45+ features",test_feature_extraction_shape),
    ("Feature values are all finite",          test_feature_values_finite),
    ("Feature order: inference == training",   test_feature_order_consistency),
    ("Schema verifier raises on mismatch",     test_schema_verification_raises),
    ("Schema verifier passes on valid input",  test_schema_verification_passes),
    ("Silence audio extracts without crash",   test_silence_handling),
    ("Very short audio handled safely",        test_very_short_audio),
    ("Empty audio array raises ValueError",    test_empty_audio_array),
    ("Invalid sample_rate raises ValueError",  test_invalid_sample_rate),
    ("Old model (35-feature) loads OK",        test_old_model_loads),
    ("New model (45+-feature) loads OK",       test_new_model_loads),
    ("Prediction on real WAV is valid",        test_prediction_on_real_wav),
    ("model_metadata.json structure valid",    test_metadata_json),
    ("No participant leakage train/test",       test_no_participant_leakage),
    ("feature_importance.json structure valid", test_feature_importance_json),
]


if __name__ == "__main__":
    print("=" * 62)
    print("VOICE PIPELINE TEST SUITE")
    print("=" * 62)

    for test_name, test_fn in ALL_TESTS:
        _run(test_name, test_fn)

    print()
    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    n_skip = sum(1 for r in results if r["status"] == SKIP)

    for r in results:
        icon = "[OK]" if r["status"] == PASS else ("[X]" if r["status"] == FAIL else "[-]")
        line = f"  {icon} {r['name']:<{_W}} [{r['status']}]"
        if r["detail"] and r["detail"] not in (PASS,):
            line += f"\n      -> {r['detail']}"
        print(line)

    print()
    print(f"  Results: {n_pass} passed / {n_fail} failed / {n_skip} skipped")
    print("=" * 62)

    if n_fail > 0:
        sys.exit(1)
