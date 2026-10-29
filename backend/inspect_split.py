
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_DIR / "backend" / "outputs" / "prepared_dataset.csv"
)


def main():
    df = pd.read_csv(DATA_PATH)

    # Check whether participant IDs are unique
    print("Total rows:", len(df))
    print("Unique participant IDs:", df["adressfname"].nunique())
    print(
        "Duplicate participant IDs:",
        df["adressfname"].duplicated().sum()
    )

    # Check whether any participant has conflicting labels
    label_counts = df.groupby("adressfname")["dx"].nunique()
    print(
        "Participants with conflicting labels:",
        (label_counts > 1).sum()
    )

    print("\nClass distribution:")
    print(df["dx"].value_counts())

    print("\nDataset folders available:")
    dataset_dir = PROJECT_DIR / "dataset" / "ADReSSo21" / "diagnosis"

    for path in sorted(dataset_dir.iterdir()):
        if path.is_dir():
            print("-", path.name)


if __name__ == "__main__":
    main()
