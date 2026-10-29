
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

# Load prepared dataset
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset.csv"

df = pd.read_csv(DATA_PATH)

# Select audio features
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

# Calculate feature importance across 5 folds
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

importance_scores = []

for train_index, test_index in cv.split(X, y):
    X_train = X.iloc[train_index]
    y_train = y.iloc[train_index]

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    importance_scores.append(model.feature_importances_)

# Average importance across folds
average_importance = pd.Series(
    sum(importance_scores) / len(importance_scores),
    index=feature_columns
).sort_values(ascending=False)

print("Total audio features:", len(feature_columns))
print("\nTop 10 important features:\n")
print(average_importance.head(10).round(4))

# Save results
output_path = BASE_DIR / "outputs" / "feature_importance.csv"
average_importance.to_csv(
    output_path,
    header=["importance"]
)

print("\nFull results saved to:", output_path)
