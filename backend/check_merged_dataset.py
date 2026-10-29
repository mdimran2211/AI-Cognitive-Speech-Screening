
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
path = BASE_DIR / "outputs" / "prepared_dataset_with_turns.csv"

df = pd.read_csv(path)

turn_columns = [
    col for col in df.columns
    if col.endswith("_turn_count")
    or col.endswith("_turn_sec")
    or col.endswith("_turn_ratio")
    or col.startswith("missing_")
    or col == "participant_total_segment_sec"
]

print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Duplicate recordings:", df["file_name"].duplicated().sum())

print("\nLabels:")
print(df["dx"].value_counts(dropna=False))

print("\nTurn feature columns:")
print(turn_columns)

print("\nMissing values in turn features:")
print(df[turn_columns].isna().sum())

numeric = df[turn_columns].select_dtypes(include="number")
print("\nInfinite numeric values:", 
      int(np.isinf(numeric.to_numpy()).sum()))

print("\nTurn feature summary:")
print(df[turn_columns].describe().round(3).to_string())

print("\nQuality check complete.")
