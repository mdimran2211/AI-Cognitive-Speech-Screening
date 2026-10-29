
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "feature_group_comparison.csv"

df = pd.read_csv(DATA_PATH)

feature_groups = {
    "Duration + Energy": [
        "duration_", "rms_", "zero_crossing_"
    ],
    "MFCC": [
        "mfcc_"
    ],
    "Pitch + Quiet": [
        "pitch_", "quiet_"
    ],
    "All Features": [
        "duration_", "rms_", "zero_crossing_",
        "mfcc_", "pitch_", "quiet_"
    ],
}

y = df["dx"].str.lower().map({"ad": 1, "cn": 0})

if y.isna().any():
    raise ValueError("Unexpected labels found in the dx column.")

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

results = []

for group_name, prefixes in feature_groups.items():
    feature_columns = [
        col for col in df.columns
        if col.startswith(tuple(prefixes))
    ]

    if not feature_columns:
        print(f"Skipping {group_name}: no features found.")
        continue

    X = df[feature_columns]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy", "f1"],
        n_jobs=-1
    )

    result = {
        "feature_group": group_name,
        "number_of_features": len(feature_columns),
        "accuracy_mean": scores["test_accuracy"].mean(),
        "balanced_accuracy_mean": scores[
            "test_balanced_accuracy"
        ].mean(),
        "balanced_accuracy_std": scores[
            "test_balanced_accuracy"
        ].std(),
        "f1_mean": scores["test_f1"].mean(),
    }

    results.append(result)

    print(f"\nFeature group: {group_name}")
    print("Number of features:", len(feature_columns))
    print("Accuracy:", round(result["accuracy_mean"], 3))
    print(
        "Balanced accuracy:",
        round(result["balanced_accuracy_mean"], 3)
    )
    print(
        "Balanced accuracy std:",
        round(result["balanced_accuracy_std"], 3)
    )
    print("F1-score:", round(result["f1_mean"], 3))

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "balanced_accuracy_mean",
    ascending=False
)

results_df.to_csv(OUTPUT_PATH, index=False)

print("\n--- FINAL COMPARISON ---")
print(results_df.round(3).to_string(index=False))
print("\nResults saved to:", OUTPUT_PATH)
