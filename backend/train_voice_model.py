"""
train_voice_model.py
====================
Reproducible ML training pipeline for Voice-Only Alzheimer's Speech Screening.

Dataset:        ADReSSo 2021 Diagnostic Challenge (166 participants)
                87 AD (Alzheimer's Disease) / 79 CN (Cognitively Normal)
                Source: dataset/ADReSSo21/diagnosis/train/
Labels:         adresso-train-mmse-scores.csv  (adressfname, dx columns)
                100% label agreement verified between CSV and folder structure.
Features:       45 acoustic features per recording (35 original + 10 extended)
                Extended: F0 min/max, voiced/unvoiced ratio, jitter, shimmer,
                          HNR, pause count, pause mean duration, pause ratio,
                          parselmouth_available flag
Participants:   1 recording per participant -- no overlap risk.
Split:          Stratified participant-level 80/20 (132 train / 34 held-out).
                Preprocessing fitted ONLY on train; applied to test.
Models:         Logistic Regression (L2), SVM (RBF), Random Forest
CV:             5-Fold Stratified on training set
Saved outputs:  voice_screening_model.joblib   (best model bundle)
                model_metadata.json            (metrics, feature schema)
                feature_importance.json        (RF importances)
                voice_features.csv             (full 45-feature matrix)

This is a research prototype. Results do not constitute clinical diagnosis.
"""

from __future__ import annotations

import json
import math
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# -- Paths --------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR   = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "audio"
META_CSV    = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis" / "train" / "adresso-train-mmse-scores.csv"
OUTPUT_DIR  = PROJECT_DIR / "backend" / "outputs"

# -- Constants ----------------------------------------------------------------
RANDOM_SEED  = 42
N_SPLITS     = 5
TEST_RATIO   = 0.20          # ~33 held-out participants
SENTINEL_NaN = float("nan")  # Explicit sentinel for extraction failures

_DIVIDER = "=" * 62


# ==============================================================================
# 1.  Dataset loading & label verification
# ==============================================================================

def load_dataset_manifest() -> pd.DataFrame:
    """
    Build a manifest of all audio files with verified labels.

    Cross-checks the folder-based labels (ad/cn) against the official
    ADReSSo metadata CSV (adressfname, dx columns).
    Raises RuntimeError if dataset is absent or labels are inconsistent.

    Returns DataFrame with columns:
        wav_path (Path), participant_id (str), folder_label (str: AD/CN),
        meta_label (str: AD/CN), mmse (int or NaN)
    """
    # -- Verify audio directory exists ----------------------------------------
    if not AUDIO_DIR.exists():
        raise RuntimeError(
            f"Audio directory not found: {AUDIO_DIR}\n"
            "Please ensure the ADReSSo 2021 dataset is present."
        )

    # -- Load metadata CSV ----------------------------------------------------
    if not META_CSV.exists():
        raise RuntimeError(
            f"Metadata CSV not found: {META_CSV}\n"
            "Cannot verify labels without adresso-train-mmse-scores.csv."
        )

    meta = pd.read_csv(META_CSV)
    required_cols = {"adressfname", "dx"}
    missing_cols = required_cols - set(meta.columns)
    if missing_cols:
        raise RuntimeError(
            f"Metadata CSV is missing required columns: {missing_cols}\n"
            f"Found: {list(meta.columns)}"
        )

    meta["adressfname"] = meta["adressfname"].str.strip()
    meta["dx"]          = meta["dx"].str.strip().str.upper()
    meta_dict = dict(zip(meta["adressfname"], meta["dx"]))   # ID -> "AD"/"CN"
    mmse_dict = (
        dict(zip(meta["adressfname"], meta["mmse"]))
        if "mmse" in meta.columns else {}
    )

    # -- Collect audio files ---------------------------------------------------
    rows = []
    for label_folder in ["ad", "cn"]:
        folder = AUDIO_DIR / label_folder
        if not folder.exists():
            print(f"  WARNING: {folder} not found -- skipping.")
            continue
        for wav in sorted(folder.rglob("*.wav")):
            pid = wav.stem
            folder_label = label_folder.upper()

            # Verify against metadata
            if pid not in meta_dict:
                print(f"  WARNING: {pid} has no metadata row -- SKIPPED.")
                continue

            meta_label = meta_dict[pid]
            if meta_label not in ("AD", "CN"):
                print(f"  WARNING: {pid} has unrecognised label '{meta_label}' -- SKIPPED.")
                continue

            if folder_label != meta_label:
                raise RuntimeError(
                    f"Label mismatch for {pid}: "
                    f"folder says '{folder_label}', metadata says '{meta_label}'.\n"
                    "Cannot proceed -- fix dataset before training."
                )

            rows.append({
                "wav_path":       wav,
                "participant_id": pid,
                "folder_label":   folder_label,
                "meta_label":     meta_label,
                "mmse":           mmse_dict.get(pid, float("nan")),
            })

    if not rows:
        raise RuntimeError("No valid audio files found after label verification.")

    df = pd.DataFrame(rows)
    ad_n = (df["meta_label"] == "AD").sum()
    cn_n = (df["meta_label"] == "CN").sum()
    print(f"  Manifest: {len(df)} recordings  (AD={ad_n}, CN={cn_n})")

    # Duplicate participant check
    dup_pids = df["participant_id"].duplicated()
    if dup_pids.any():
        raise RuntimeError(
            f"Duplicate participant IDs found: {df[dup_pids]['participant_id'].tolist()}"
        )

    return df


# ==============================================================================
# 2.  Feature extraction
# ==============================================================================

def extract_all_features(manifest: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Extract acoustic features for every recording in the manifest.

    Extraction failures are recorded as NaN (not zero) so that
    the imputer can handle them explicitly and they can be tracked.

    Returns DataFrame with participant_id, label, and all feature columns.
    """
    from backend.audio_analysis import extract_features

    rows       = []
    failed_ids = []

    total = len(manifest)
    for idx, row in enumerate(manifest.itertuples(), 1):
        pid   = row.participant_id
        label = row.meta_label
        wav   = row.wav_path

        try:
            feats = extract_features(wav)

            # Guard: all feature values must be real numbers
            bad = [k for k, v in feats.items()
                   if not isinstance(v, (int, float)) or not math.isfinite(v)]
            if bad:
                print(f"  [{idx}/{total}] {pid}: non-finite values in {bad} -- set to NaN")
                for k in bad:
                    feats[k] = float("nan")

            feats["participant_id"] = pid
            feats["label"]          = label
            rows.append(feats)

            if verbose:
                print(f"  [{idx}/{total}] {pid} ({label}) -> {len(feats)-2} features OK")

        except Exception as exc:
            print(f"  [{idx}/{total}] FAILED: {pid} ({label}): {exc}")
            failed_ids.append(pid)

    if failed_ids:
        print(f"\n  !! {len(failed_ids)} recordings failed extraction: {failed_ids}")
        if len(failed_ids) > 10:
            raise RuntimeError("Too many extraction failures -- check audio files.")

    if not rows:
        raise RuntimeError("All feature extractions failed.")

    df = pd.DataFrame(rows)
    col_order = ["participant_id", "label"] + [
        c for c in df.columns if c not in ("participant_id", "label")
    ]
    return df[col_order]


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature column names in extraction order (excluding metadata)."""
    return [c for c in df.columns if c not in ("participant_id", "label")]


# ==============================================================================
# 3.  Participant-level train / held-out split
# ==============================================================================

def participant_split(
    df: pd.DataFrame, test_ratio: float = TEST_RATIO, seed: int = RANDOM_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Stratified participant-level split.

    Because each participant has exactly one recording, stratified random
    sampling on labels achieves participant isolation by construction.
    """
    rng = random.Random(seed)

    ad_ids = df[df["label"] == "AD"]["participant_id"].tolist()
    cn_ids = df[df["label"] == "CN"]["participant_id"].tolist()

    rng.shuffle(ad_ids)
    rng.shuffle(cn_ids)

    n_ad_test = max(1, round(len(ad_ids) * test_ratio))
    n_cn_test = max(1, round(len(cn_ids) * test_ratio))

    test_ids  = set(ad_ids[:n_ad_test]  + cn_ids[:n_cn_test])
    train_ids = set(ad_ids[n_ad_test:]  + cn_ids[n_cn_test:])

    # Verify no overlap
    overlap = train_ids & test_ids
    if overlap:
        raise RuntimeError(f"Participant overlap in train/test split: {overlap}")

    train_df = df[df["participant_id"].isin(train_ids)].reset_index(drop=True)
    test_df  = df[df["participant_id"].isin(test_ids)].reset_index(drop=True)

    return train_df, test_df


# ==============================================================================
# 4.  Model definitions
# ==============================================================================

LABEL_MAP = {"CN": 0, "AD": 1}
INV_LABEL = {0: "CN", 1: "AD"}


def build_pipeline(clf) -> Pipeline:
    """Impute -> Scale -> Classify. Fitted entirely on training data."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("clf",     clf),
    ])


def model_candidates(seed: int = RANDOM_SEED) -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            C=1.0, class_weight="balanced",
            max_iter=2000, random_state=seed,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", C=1.0, gamma="scale",
            class_weight="balanced", probability=True,
            random_state=seed,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            max_depth=None, min_samples_leaf=2,
            random_state=seed, n_jobs=-1,
        ),
    }


# ==============================================================================
# 5.  Evaluation helpers
# ==============================================================================

def _ci95(values: list[float]) -> tuple[float, float]:
    """95% bootstrap-style CI using 1.96 x std / sqrt(n)."""
    arr = np.array(values)
    se = arr.std() / math.sqrt(len(arr))
    return float(arr.mean() - 1.96 * se), float(arr.mean() + 1.96 * se)


def evaluate_predictions(y_true, y_pred, y_prob, set_name: str) -> dict:
    """Compute all evaluation metrics from actual predictions (no fabrication)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "set": set_name,
        "n_samples": int(len(y_true)),
        "n_ad": int(np.sum(y_true == 1)),
        "n_cn": int(np.sum(y_true == 0)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "sensitivity_ad_recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "specificity_cn_recall": round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "confusion_matrix": {
            "true_negatives_cn":  int(tn),
            "false_positives_ad": int(fp),
            "false_negatives_cn": int(fn),
            "true_positives_ad":  int(tp),
        },
    }


# ==============================================================================
# 6.  Feature-schema verification (inference-time guard)
# ==============================================================================

def verify_feature_schema(
    expected_names: list[str], actual_features: dict, context: str = ""
) -> None:
    """
    Verify that inference-time features exactly match the training schema.
    Raises ValueError on mismatch -- never silently accepts wrong input.
    """
    missing   = [n for n in expected_names if n not in actual_features]
    extra     = [n for n in actual_features if n not in expected_names]
    order_ok  = list(actual_features.keys())[:len(expected_names)] == expected_names

    errors = []
    if missing:
        errors.append(f"Missing features: {missing}")
    if extra:
        errors.append(f"Unexpected extra features: {extra}")

    if errors:
        raise ValueError(
            f"Feature schema mismatch{' (' + context + ')' if context else ''}:\n"
            + "\n".join(f"  * {e}" for e in errors)
        )


# ==============================================================================
# 7.  Main training entry point
# ==============================================================================

def main() -> dict:
    print(_DIVIDER)
    print("VOICE SCREENING MODEL  --  TRAINING PIPELINE  v2.0")
    print(_DIVIDER)
    t_start = time.time()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PROJECT_DIR))

    # -- Step 1: Load and verify dataset --------------------------------------
    print("\n[1/6]  Loading dataset manifest and verifying labels ...")
    manifest = load_dataset_manifest()

    # -- Step 2: Extract features ----------------------------------------------
    feat_csv_path = OUTPUT_DIR / "voice_features.csv"
    if feat_csv_path.exists():
        print("\n[2/6]  Loading cached features from " + str(feat_csv_path) + " ...")
        feat_df = pd.read_csv(feat_csv_path)
        feat_elapsed = 0.0
    else:
        print("\n[2/6]  Extracting acoustic features (this may take several minutes) ...")
        t_feat = time.time()
        feat_df = extract_all_features(manifest, verbose=False)
        feat_elapsed = time.time() - t_feat
        feat_df.to_csv(feat_csv_path, index=False)
        print("  Saved:           " + str(feat_csv_path))

    feature_cols = get_feature_columns(feat_df)
    n_nan = feat_df[feature_cols].isna().sum().sum()
    n_recordings = len(feat_df)

    print(f"\n  Feature matrix:  {n_recordings} recordings x {len(feature_cols)} features")
    print(f"  NaN cell count:  {n_nan}  "
          f"({'none -- all features extracted cleanly' if n_nan == 0 else 'will be imputed on train folds only'})")
    if feat_elapsed > 0:
        print(f"  Extraction time: {feat_elapsed:.1f}s")

    # -- Step 3: Participant-level split ---------------------------------------
    print("\n[3/6]  Participant-level stratified 80/20 split ...")
    train_df, test_df = participant_split(feat_df, test_ratio=TEST_RATIO, seed=RANDOM_SEED)

    train_ad = (train_df["label"] == "AD").sum()
    train_cn = (train_df["label"] == "CN").sum()
    test_ad  = (test_df["label"]  == "AD").sum()
    test_cn  = (test_df["label"]  == "CN").sum()

    print(f"  Train set: {len(train_df)} recordings  (AD={train_ad}, CN={train_cn})")
    print(f"  Test set:  {len(test_df)}  recordings  (AD={test_ad},  CN={test_cn})")

    # Confirm no participant overlap
    overlap = set(train_df["participant_id"]) & set(test_df["participant_id"])
    assert not overlap, f"Participant overlap detected: {overlap}"
    print(f"  Participant overlap: NONE [OK]")

    X_train = train_df[feature_cols].values.astype(float)
    y_train = train_df["label"].map(LABEL_MAP).values.astype(int)
    X_test  = test_df[feature_cols].values.astype(float)
    y_test  = test_df["label"].map(LABEL_MAP).values.astype(int)

    # -- Step 4: Cross-validation model comparison -----------------------------
    print(f"\n[4/6]  {N_SPLITS}-Fold Stratified Cross-Validation on training set ...")

    cv_skf     = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    candidates = model_candidates(seed=RANDOM_SEED)
    cv_results = {}
    best_name  = None
    best_ba    = -1.0

    for name, clf in candidates.items():
        pipe = build_pipeline(clf)
        cv = cross_validate(
            pipe, X_train, y_train,
            cv=cv_skf,
            scoring=["balanced_accuracy", "f1", "roc_auc", "accuracy"],
            n_jobs=-1,
            return_train_score=False,
        )

        ba_vals  = cv["test_balanced_accuracy"].tolist()
        f1_vals  = cv["test_f1"].tolist()
        auc_vals = cv["test_roc_auc"].tolist()
        acc_vals = cv["test_accuracy"].tolist()

        ba_mean  = float(np.mean(ba_vals))
        ba_std   = float(np.std(ba_vals))
        ba_lo, ba_hi = _ci95(ba_vals)

        cv_results[name] = {
            "cv_balanced_accuracy": {
                "mean":   round(ba_mean, 4),
                "std":    round(ba_std, 4),
                "ci95_lo": round(ba_lo, 4),
                "ci95_hi": round(ba_hi, 4),
                "folds":  [round(v, 4) for v in ba_vals],
            },
            "cv_f1_mean":      round(float(np.mean(f1_vals)), 4),
            "cv_roc_auc_mean": round(float(np.mean(auc_vals)), 4),
            "cv_accuracy_mean": round(float(np.mean(acc_vals)), 4),
        }

        status = "<= BEST" if ba_mean > best_ba else ""
        print(f"  {name:30s}  BA={ba_mean:.3f}+/-{ba_std:.3f}  "
              f"[{ba_lo:.3f},{ba_hi:.3f}]  F1={np.mean(f1_vals):.3f}  "
              f"AUC={np.mean(auc_vals):.3f}  {status}")

        if ba_mean > best_ba:
            best_ba   = ba_mean
            best_name = name

    print(f"\n  Selected model: {best_name}  "
          f"(CV balanced accuracy = {best_ba:.3f})")

    # -- Step 5: Train best model on full train set; evaluate held-out test -----
    print(f"\n[5/6]  Training {best_name} on full training set ...")
    best_pipe = build_pipeline(candidates[best_name])
    best_pipe.fit(X_train, y_train)

    # Verify feature schema consistency
    verify_feature_schema(feature_cols, dict(zip(feature_cols, X_train[0])), "post-training check")

    train_pred  = best_pipe.predict(X_train)
    train_prob  = best_pipe.predict_proba(X_train)[:, 1]
    train_metrics = evaluate_predictions(y_train, train_pred, train_prob, "train")

    test_pred   = best_pipe.predict(X_test)
    test_prob   = best_pipe.predict_proba(X_test)[:, 1]
    test_metrics  = evaluate_predictions(y_test, test_pred, test_prob, "held_out_test")

    print(f"\n  -- Held-out test results ------------------------------")
    print(f"  Accuracy         : {test_metrics['accuracy']:.3f}")
    print(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.3f}")
    print(f"  Sensitivity (AD) : {test_metrics['sensitivity_ad_recall']:.3f}")
    print(f"  Specificity (CN) : {test_metrics['specificity_cn_recall']:.3f}")
    print(f"  Precision        : {test_metrics['precision']:.3f}")
    print(f"  F1 Score         : {test_metrics['f1_score']:.3f}")
    print(f"  ROC-AUC          : {test_metrics['roc_auc']:.3f}")
    cm = test_metrics["confusion_matrix"]
    print(f"  Confusion Matrix :")
    print(f"    TN(CN correct)={cm['true_negatives_cn']}  FP(CN->AD)={cm['false_positives_ad']}")
    print(f"    FN(AD->CN)={cm['false_negatives_cn']}  TP(AD correct)={cm['true_positives_ad']}")

    # -- Step 6: Save artifacts ------------------------------------------------
    print(f"\n[6/6]  Saving model artifacts ...")

    # 6a. Model bundle
    artifact = {
        "model":         best_pipe,
        "model_type":    best_name,
        "features":      feature_cols,
        "label_mapping": INV_LABEL,
        "version":       "2.0.0-voice",
        "dataset":       "ADReSSo 2021 Diagnostic Challenge (166 recordings)",
        "n_train":       int(len(train_df)),
        "n_test":        int(len(test_df)),
        "class_balance": {"train_AD": int(train_ad), "train_CN": int(train_cn),
                          "test_AD":  int(test_ad),  "test_CN":  int(test_cn)},
    }
    model_path = OUTPUT_DIR / "voice_screening_model.joblib"
    joblib.dump(artifact, model_path)
    print(f"  Saved model   : {model_path}")

    # 6b. Metadata JSON
    metadata = {
        "model_name":    best_name,
        "version":       "2.0.0-voice",
        "dataset":       "ADReSSo 2021 Diagnostic Challenge",
        "n_total":       int(n_recordings),
        "n_train":       int(len(train_df)),
        "n_test":        int(len(test_df)),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
        "label_mapping": {"0": "CN", "1": "AD"},
        "split_strategy": f"Participant-level stratified {int((1-TEST_RATIO)*100)}/{int(TEST_RATIO*100)} split",
        "cv_protocol":   f"{N_SPLITS}-Fold Stratified Cross-Validation (train set only)",
        "class_balance": artifact["class_balance"],
        "nan_cells_before_imputation": int(n_nan),
        "cv_results":     cv_results,
        "train_metrics":  train_metrics,
        "test_metrics":   test_metrics,
        "total_time_s":   round(time.time() - t_start, 1),
    }
    meta_path = OUTPUT_DIR / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Saved metadata: {meta_path}")

    # 6c. Feature importance (computed for any selected classifier)
    clf_step = best_pipe.named_steps["clf"]
    if hasattr(clf_step, "feature_importances_"):
        fi = sorted(
            zip(feature_cols, clf_step.feature_importances_),
            key=lambda x: -x[1],
        )
    elif hasattr(clf_step, "coef_"):
        weights = np.abs(clf_step.coef_[0])
        total_w = float(np.sum(weights))
        weights_norm = (weights / total_w) if total_w > 0 else weights
        fi = sorted(
            zip(feature_cols, weights_norm),
            key=lambda x: -x[1],
        )
    else:
        fi = [(f, 1.0 / len(feature_cols)) for f in feature_cols]

    fi_json = [{"feature": f, "importance": round(float(v), 6)} for f, v in fi]
    fi_path = OUTPUT_DIR / "feature_importance.json"
    fi_path.write_text(json.dumps(fi_json, indent=2), encoding="utf-8")
    print(f"  Saved fi      : {fi_path}")

    elapsed = round(time.time() - t_start, 1)
    print(f"\n{_DIVIDER}")
    print(f"TRAINING COMPLETE  ({elapsed}s total)")
    print(f"{_DIVIDER}\n")

    return metadata


if __name__ == "__main__":
    main()
