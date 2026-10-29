
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "backend" / "outputs" / "prepared_dataset.csv"
)

MODEL_PATH = (
    PROJECT_DIR / "backend" / "outputs" / "speech_model.joblib"
)


def main():
    df = pd.read_csv(DATA_PATH)

    # Select audio features only
    feature_columns = [
        column for column in df.columns
        if column.startswith(("duration_", "rms_", "zero_crossing_", "mfcc_"))
    ]

    X = df[feature_columns]
    y = df["dx"].str.lower().map({"ad": 1, "cn": 0})

    if y.isna().any():
        raise ValueError("Unexpected or missing class labels found.")

    print("Total recordings:", len(df))
    print("Audio features:", len(feature_columns))
    print("\nClass counts:")
    print(y.value_counts().rename(index={1: "AD", 0: "CN"}))

    # Keep a separate test set
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Scaling is fitted only on training data
    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced", random_state=42),
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\nTest accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["CN", "AD"],
            zero_division=0,
        )
    )

    print("Confusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "class_names": {0: "CN", 1: "AD"},
        },
        MODEL_PATH,
    )

    print("\nModel saved to:", MODEL_PATH)


if __name__ == "__main__":
    main()
