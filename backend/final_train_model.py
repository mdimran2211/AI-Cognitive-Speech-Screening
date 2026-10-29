
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

DATA_PATH = OUT_DIR / "prepared_dataset_with_turns.csv"
MODEL_PATH = OUT_DIR / "final_speech_model.joblib"

df = pd.read_csv(DATA_PATH)

audio_features = [
    col for col in df.columns
    if col.startswith(("mfcc_", "rms_", "zero_crossing_"))
    or col in [
        "duration_seconds",
        "pitch_mean",
        "pitch_std",
        "quiet_ratio",
        "quiet_duration_seconds",
    ]
]

turn_features = [
    "participant_turn_count",
    "interviewer_turn_count",
    "participant_mean_turn_sec",
    "participant_median_turn_sec",
    "participant_total_segment_sec",
    "interviewer_mean_turn_sec",
    "participant_turn_ratio",
]

features = audio_features + turn_features

y = df["dx"].str.lower().map({"ad": 1, "cn": 0})
X = df[features].apply(pd.to_numeric, errors="raise")

if y.isna().any():
    raise ValueError("Unexpected or missing labels.")

if X.isna().any().any():
    raise ValueError("Missing feature values found.")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        min_samples_leaf=1,
        n_jobs=-1,
    )),
])

model.fit(X, y)

artifact = {
    "model": model,
    "features": features,
    "label_mapping": {0: "CN", 1: "AD"},
    "model_type": "RandomForestClassifier",
}

joblib.dump(artifact, MODEL_PATH)

print("Final model training complete.")
print("Training recordings:", len(df))
print("Audio features:", len(audio_features))
print("Turn features:", len(turn_features))
print("Total features:", len(features))
print("Saved model:", MODEL_PATH)
print("\nNote: This model is experimental and not clinically validated.")
