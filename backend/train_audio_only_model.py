
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, balanced_accuracy_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "audio_only_model.joblib"

df = pd.read_csv(DATA_PATH)

# Use only the 35 audio features.
feature_columns = [
    column
    for column in df.columns
    if column.startswith((
        "duration_",
        "rms_",
        "zero_crossing_",
        "mfcc_",
        "pitch_",
        "quiet_",
    ))
]

X = df[feature_columns]
y = df["dx"].str.lower().map({"cn": 0, "ad": 1})

if y.isna().any():
    raise ValueError("Unexpected or missing diagnosis labels.")

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42,
)

scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "balanced_accuracy": make_scorer(balanced_accuracy_score),
        "f1": "f1",
    },
    n_jobs=-1,
)

print("Audio features:", len(feature_columns))
print(f"CV accuracy: {scores['test_accuracy'].mean():.3f}")
print(
    "CV balanced accuracy: "
    f"{scores['test_balanced_accuracy'].mean():.3f}"
)
print(f"CV F1: {scores['test_f1'].mean():.3f}")

# Train the final model on the available training dataset.
pipeline.fit(X, y)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": pipeline,
    "features": feature_columns,
    "label_mapping": {0: "CN", 1: "AD"},
    "model_type": "RandomForestClassifier",
}, OUTPUT_PATH)

print(f"Saved audio-only model to: {OUTPUT_PATH}")
