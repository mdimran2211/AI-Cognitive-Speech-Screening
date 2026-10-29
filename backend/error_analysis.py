
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset.csv"
OUTPUT_PATH = BASE_DIR / "outputs" / "error_analysis.csv"

df = pd.read_csv(DATA_PATH)

feature_columns = [
    col for col in df.columns
    if col.startswith((
        "duration_",
        "rms_",
        "zero_crossing_",
        "mfcc_",
        "pitch_",
        "quiet_"
    ))
]

X = df[feature_columns]
y = df["dx"].str.lower().map({"ad": 1, "cn": 0})

if y.isna().any():
    raise ValueError("Unexpected labels found in the dx column.")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    n_jobs=-1
)

print("\n--- MODEL PERFORMANCE ---")
print("Accuracy:", round(accuracy_score(y, predictions), 3))
print(
    "Balanced accuracy:",
    round(balanced_accuracy_score(y, predictions), 3)
)

print("\n--- CLASSIFICATION REPORT ---")
print(
    classification_report(
        y,
        predictions,
        labels=[0, 1],
        target_names=["CN", "AD"],
        zero_division=0
    )
)

print("--- CONFUSION MATRIX ---")
print("Rows = actual labels; columns = predicted labels")
print("Order: CN, AD")
print(confusion_matrix(y, predictions, labels=[0, 1]))

results = pd.DataFrame({
    "file_name": df["file_name"],
    "actual_label": y.map({0: "CN", 1: "AD"}),
    "predicted_label": pd.Series(predictions).map(
        {0: "CN", 1: "AD"}
    ),
})

results["correct"] = (
    results["actual_label"] == results["predicted_label"]
)

results["error_type"] = results.apply(
    lambda row: (
        "Correct"
        if row["correct"]
        else f'{row["actual_label"]} predicted as '
             f'{row["predicted_label"]}'
    ),
    axis=1
)

results.to_csv(OUTPUT_PATH, index=False)

print("\n--- ERROR COUNTS ---")
print(results["error_type"].value_counts().to_string())
print("\nSaved predictions to:", OUTPUT_PATH)
