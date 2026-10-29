
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    PROJECT_DIR
    / "dataset"
    / "ADReSSo21"
    / "diagnosis"
    / "train"
    / "adresso-train-mmse-scores.csv"
)


def main():
    if not CSV_PATH.exists():
        print("Metadata CSV not found:")
        print(CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)

    print("\n--- METADATA OVERVIEW ---")
    print("Shape:", df.shape)

    print("\n--- COLUMN NAMES ---")
    print(df.columns.tolist())

    print("\n--- FIRST 10 ROWS ---")
    print(df.head(10).to_string(index=False))

    print("\n--- MISSING VALUES ---")
    print(df.isnull().sum().to_string())

    print("\n--- DATA TYPES ---")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
