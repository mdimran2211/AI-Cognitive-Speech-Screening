
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset_with_turns.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "turn_model_comparison.csv"

df = pd.read_csv(DATA_PATH)

# Original audio features
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

# Turn-related features, excluding missing-speaker flags
turn_features = [
    "participant_turn_count",
    "interviewer_turn_count",
    "participant_mean_turn_sec",
    "participant_median_turn_sec",
    "participant_total_segment_sec",
    "interviewer_mean_turn_sec",
    "participant_turn_ratio",
]

y = df["dx"].str.lower().map({"ad": 1, "cn": 0})

if y.isna().any():
    raise ValueError("Unexpected or missing class labels.")

if len(audio_features) != 35:
    raise ValueError(
        f"Expected 35 audio features, found {len(audio_features)}"
    )

feature_groups = {
    "Audio only": audio_features,
    "Audio + turn features": audio_features + turn_features,
}

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42,
)

results = []

for model_name, features in feature_groups.items():
    X = df[features].apply(pd.to_numeric, errors="raise")

    scores = {
        "accuracy": [],
        "balanced_accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for train_idx, test_idx in cv.split(X, y):
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            min_samples_leaf=1,
            n_jobs=-1,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        scores["accuracy"].append(
            accuracy_score(y_test, predictions)
        )
        scores["balanced_accuracy"].append(
            balanced_accuracy_score(y_test, predictions)
        )
        scores["precision"].append(
            precision_score(y_test, predictions, zero_division=0)
        )
        scores["recall"].append(
            recall_score(y_test, predictions, zero_division=0)
        )
        scores["f1"].append(
            f1_score(y_test, predictions, zero_division=0)
        )

    row = {"model": model_name, "feature_count": len(features)}

    for metric, values in scores.items():
        row[f"{metric}_mean"] = sum(values) / len(values)
        row[f"{metric}_std"] = pd.Series(values).std(ddof=1)

    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_PATH, index=False)

print("\nModel comparison (50 CV evaluations per model):")
print(results_df.round(3).to_string(index=False))
print("\nSaved:", OUTPUT_PATH)
