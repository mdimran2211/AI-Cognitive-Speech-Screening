
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import balanced_accuracy_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset.csv"

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

parameter_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 10],
    "min_samples_leaf": [1, 2]
}

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []
best_parameters = []

for fold, (train_index, test_index) in enumerate(
    outer_cv.split(X, y), start=1
):
    X_train = X.iloc[train_index]
    X_test = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=fold
    )

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=1
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="balanced_accuracy",
        cv=inner_cv,
        n_jobs=-1,
        refit=True
    )

    search.fit(X_train, y_train)

    predictions = search.best_estimator_.predict(X_test)
    score = balanced_accuracy_score(y_test, predictions)

    results.append(score)
    best_parameters.append(search.best_params_)

    print(f"\nOuter fold {fold}")
    print("Best parameters:", search.best_params_)
    print("Inner CV balanced accuracy:",
          round(search.best_score_, 3))
    print("Outer test balanced accuracy:", round(score, 3))

print("\n--- NESTED CROSS-VALIDATION RESULTS ---")
print("Mean balanced accuracy:", round(sum(results) / len(results), 3))

import numpy as np

print("Standard deviation:", round(np.std(results), 3))
print("Fold scores:", [round(score, 3) for score in results])

print("\nBest parameters selected in each fold:")
for fold, params in enumerate(best_parameters, start=1):
    print(f"Fold {fold}: {params}")
