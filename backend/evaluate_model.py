
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_validate,
    cross_val_predict
)
from sklearn.metrics import confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(BASE_DIR / "outputs" / "prepared_dataset.csv")

feature_columns = [
    col for col in df.columns
    if col.startswith((
        "duration_", "rms_", "zero_crossing_",
        "mfcc_", "pitch_", "quiet_"
    ))
]

X = df[feature_columns]
y = df["dx"].str.lower().map({"ad": 1, "cn": 0})

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# 5-fold CV, repeated 10 times
cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

# Calculate average metrics across repeated folds
scores = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "balanced_accuracy"],
    n_jobs=-1
)

print("Accuracy:", round(scores["test_accuracy"].mean(), 3))
print(
    "Balanced accuracy:",
    round(scores["test_balanced_accuracy"].mean(), 3)
)

# Use ordinary 5-fold CV for a confusion matrix
# Each recording receives exactly one out-of-fold prediction.
from sklearn.model_selection import StratifiedKFold

single_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=single_cv,
    n_jobs=-1
)

tn, fp, fn, tp = confusion_matrix(
    y,
    predictions,
    labels=[0, 1]
).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) else 0
specificity = tn / (tn + fp) if (tn + fp) else 0

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions, labels=[0, 1]))

print("\nSensitivity (AD recall):", round(sensitivity, 3))
print("Specificity (CN recall):", round(specificity, 3))
print("True negatives:", tn)
print("False positives:", fp)
print("False negatives:", fn)
print("True positives:", tp)
