
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_DIR / "backend" / "outputs" / "audio_features.csv"


def main():
    if not CSV_PATH.exists():
        print("Feature CSV not found:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)

    print("\n--- DATASET OVERVIEW ---")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Labels:\n", df["label"].value_counts())

    print("\n--- MISSING VALUES ---")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.any() else "No missing values found.")

    print("\n--- DUPLICATE FILENAMES ---")
    print("Duplicates:", df["file_name"].duplicated().sum())

    print("\n--- FEATURE SUMMARY ---")
    numeric = df.select_dtypes(include="number")
    print(numeric.describe().round(3).to_string())

    print("\n--- FEATURES BY CLASS ---")
    print(
        df.groupby("label")[
            ["duration_seconds", "rms_mean", "zero_crossing_mean"]
        ].mean().round(3)
    )


if __name__ == "__main__":
    main()
