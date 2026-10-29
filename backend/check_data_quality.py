
from pathlib import Path

import numpy as np
import pandas as pd

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

features = df[feature_columns]

print("===== DATA QUALITY REPORT =====")
print("Total recordings:", len(df))
print("Total audio features:", len(feature_columns))

# Check missing values
missing_count = features.isnull().sum().sum()
print("\nMissing feature values:", missing_count)

# Check infinite values
infinite_count = np.isinf(
    features.to_numpy(dtype=float)
).sum()
print("Infinite feature values:", infinite_count)

# Check duplicate recordings
duplicate_count = df["file_name"].duplicated().sum()
print("Duplicate filenames:", duplicate_count)

# Check invalid labels
print("\nLabel distribution:")
print(df["dx"].value_counts())

# Check unusual constant features
constant_features = [
    col for col in feature_columns
    if features[col].nunique(dropna=False) <= 1
]

print("\nConstant features:", constant_features)

# Show features with highest and lowest values
print("\nFeature ranges:")
print(
    features.describe()
    .loc[["min", "max"]]
    .round(4)
    .to_string()
)

print("\nData quality check complete.")
