
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = (
    PROJECT_DIR / "backend" / "outputs" / "audio_features.csv"
)

METADATA_PATH = (
    PROJECT_DIR
    / "dataset"
    / "ADReSSo21"
    / "diagnosis"
    / "train"
    / "adresso-train-mmse-scores.csv"
)


def main():
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    print("Feature rows:", len(features))
    print("Metadata rows:", len(metadata))

    print("\nFeature columns:")
    print(features.columns.tolist())

    print("\nMetadata ID examples:")
    print(metadata["adressfname"].head().tolist())

    # Check whether audio filenames match metadata IDs
    feature_ids = set(
        features["file_name"].str.replace(".wav", "", regex=False)
    )
    metadata_ids = set(metadata["adressfname"])

    print("\nMatching IDs:", len(feature_ids & metadata_ids))
    print("Only in features:", sorted(feature_ids - metadata_ids)[:10])
    print("Only in metadata:", sorted(metadata_ids - feature_ids)[:10])


if __name__ == "__main__":
    main()
