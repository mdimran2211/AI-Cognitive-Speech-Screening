
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs"

prepared = pd.read_csv(OUT_DIR / "prepared_dataset.csv")
turns = pd.read_csv(OUT_DIR / "turn_features.csv")

# Normalize IDs: adrso024.wav -> adrso024
prepared["recording_id"] = (
    prepared["file_name"].astype(str).str.replace(
        r"\.wav$", "", regex=True, case=False
    )
)

turns["recording_id"] = (
    turns["file_name"].astype(str).str.replace(
        r"\.wav$", "", regex=True, case=False
    )
)

if prepared["recording_id"].duplicated().any():
    raise ValueError("Duplicate IDs in prepared dataset")

if turns["recording_id"].duplicated().any():
    raise ValueError("Duplicate IDs in turn features")

print(
    "Matching IDs:",
    len(set(prepared["recording_id"]) & set(turns["recording_id"]))
)

merged = prepared.merge(
    turns.drop(columns=["file_name"]),
    on="recording_id",
    how="left",
    validate="one_to_one",
    indicator=True
)

print("\nMerge status:")
print(merged["_merge"].value_counts())

if not (merged["_merge"] == "both").all():
    raise ValueError("Some recordings did not match. Check IDs.")

merged = merged.drop(columns=["_merge", "recording_id"])

output_path = OUT_DIR / "prepared_dataset_with_turns.csv"
merged.to_csv(output_path, index=False)

print("\nMerged rows:", len(merged))
print("Merged columns:", len(merged.columns))
print("Saved:", output_path)
