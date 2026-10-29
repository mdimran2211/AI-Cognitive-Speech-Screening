
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = (
    BASE_DIR.parent
    / "dataset"
    / "ADReSSo21"
    / "diagnosis"
    / "train"
    / "segmentation"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

results = []

for csv_path in sorted(DATA_DIR.rglob("*.csv")):
    df = pd.read_csv(csv_path)

    if not {"speaker", "begin", "end"}.issubset(df.columns):
        print("Skipping invalid columns:", csv_path.name)
        continue

    df = df.dropna(subset=["speaker", "begin", "end"]).copy()
    df["speaker"] = df["speaker"].astype(str).str.strip()

    # Convert milliseconds to seconds.
    df["turn_duration_sec"] = (
        df["end"] - df["begin"]
    ) / 1000.0

    # Keep only valid, non-negative durations.
    df = df[df["turn_duration_sec"] >= 0]

    par = df[df["speaker"] == "PAR"]
    inv = df[df["speaker"] == "INV"]

    par_count = len(par)
    inv_count = len(inv)

    results.append({
        "file_name": csv_path.stem,
        "participant_turn_count": par_count,
        "interviewer_turn_count": inv_count,
        "participant_mean_turn_sec": (
            par["turn_duration_sec"].mean()
            if par_count else 0.0
        ),
        "participant_median_turn_sec": (
            par["turn_duration_sec"].median()
            if par_count else 0.0
        ),
        "participant_total_segment_sec": (
            par["turn_duration_sec"].sum()
        ),
        "interviewer_mean_turn_sec": (
            inv["turn_duration_sec"].mean()
            if inv_count else 0.0
        ),
        "participant_turn_ratio": (
            par_count / (par_count + inv_count)
            if par_count + inv_count else 0.0
        ),
        "missing_participant_turns": int(par_count == 0),
        "missing_interviewer_turns": int(inv_count == 0)
    })

features = pd.DataFrame(results)
output_path = OUTPUT_DIR / "turn_features.csv"
features.to_csv(output_path, index=False)

print("Recordings processed:", len(features))
print("Output:", output_path)
print("\nMissing participant turns:", 
      int(features["missing_participant_turns"].sum()))
print("Missing interviewer turns:",
      int(features["missing_interviewer_turns"].sum()))
print("\nFeature summary:")
print(features.describe().round(3))
print("\nExtraction complete.")
