
from pathlib import Path
import pandas as pd

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

forbidden_columns = {
    "dx",
    "mmse",
    "labels_match",
    "file_name",
    "adressfname"
}

print("--- DATA LEAKAGE CHECK ---")
print("Total rows:", len(df))
print("Unique filenames:", df["file_name"].nunique())
print("Duplicate filenames:", df["file_name"].duplicated().sum())

print("\nFeature count:", len(feature_columns))
print("Forbidden columns used as features:",
      sorted(set(feature_columns) & forbidden_columns))

print("\nLabels:")
print(df["dx"].value_counts().to_string())

print("\nMissing feature values:", df[feature_columns].isna().sum().sum())

print("\nFeature columns:")
for column in feature_columns:
    print(column)

if set(feature_columns) & forbidden_columns:
    print("\nWARNING: Potential target/metadata leakage detected.")
else:
    print("\nPASS: Target and metadata columns are excluded from features.")
