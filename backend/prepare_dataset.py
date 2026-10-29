
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

OUTPUT_PATH = (
    PROJECT_DIR / "backend" / "outputs" / "prepared_dataset.csv"
)


def main():
    features = pd.read_csv(FEATURES_PATH)
    metadata = pd.read_csv(METADATA_PATH)

    # Extract participant ID from the audio filename
    features["adressfname"] = (
        features["file_name"].str.replace(".wav", "", regex=False)
    )

    # Keep only the required metadata columns
    metadata = metadata[["adressfname", "mmse", "dx"]]

    # Merge audio features with metadata
    df = features.merge(
        metadata,
        on="adressfname",
        how="inner",
        validate="one_to_one"
    )

    # Check whether the two label columns agree
    df["labels_match"] = (
    df["label"].str.lower() == df["dx"].str.lower()
    )
    print("Prepared rows:", len(df))
    print("Label agreement:", df["labels_match"].all())
    
    print("\nMismatched rows:")
    print(
        df.loc[
            df["labels_match"] == False,
            ["file_name", "label", "dx"]
        ].to_string(index=False)
    )

    print("\nAudio feature label counts:")
    print(df["label"].value_counts())

    print("\nMetadata label counts:")
    print(df["dx"].value_counts())

    print("\nClass counts:")
    print(df["dx"].value_counts())
    print("\nMMSE score summary:")
    print(df.groupby("dx")["mmse"].describe().round(2))

    # Save the prepared dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print("\nSaved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
