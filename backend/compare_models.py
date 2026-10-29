
from pathlib import Path

import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "backend" / "outputs" / "prepared_dataset.csv"


def main():
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
        raise ValueError("Unknown or missing labels found in dx column.")

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42
            ))
        ]),

        "SVM": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="rbf",
                class_weight="balanced"
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42
            ))
        ])
    }

    cv = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall"
    }

    summary = []

    print("Total recordings:", len(df))
    print("Audio features:", len(feature_columns))
    print("Evaluation: 5 folds x 10 repeats")

    for name, model in models.items():
        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1
        )

        row = {"Model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_std"] = values.std()

        summary.append(row)

    results = pd.DataFrame(summary)
    results = results.sort_values(
        "balanced_accuracy_mean",
        ascending=False
    )

    print("\nMODEL COMPARISON")
    print(results.round(3).to_string(index=False))

    output_path = (
        PROJECT_DIR / "backend" / "outputs" / "model_comparison.csv"
    )
    results.to_csv(output_path, index=False)

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
