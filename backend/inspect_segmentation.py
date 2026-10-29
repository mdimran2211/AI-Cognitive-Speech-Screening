
from pathlib import Path
import pandas as pd
import soundfile as sf

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "dataset" / "ADReSSo21" / "diagnosis" / "train"

seg_dir = DATA_DIR / "segmentation"
audio_dir = DATA_DIR / "audio"

results = []

for csv_path in sorted(seg_dir.rglob("*.csv")):
    audio_matches = list(audio_dir.rglob(csv_path.stem + ".wav"))

    if len(audio_matches) != 1:
        print("Audio match issue:", csv_path.stem, len(audio_matches))
        continue

    df = pd.read_csv(csv_path)
    audio_info = sf.info(audio_matches[0])

    required = {"speaker", "begin", "end"}
    if not required.issubset(df.columns) or df.empty:
        print("Invalid CSV:", csv_path.name)
        continue

    invalid = (
        (df["begin"] < 0)
        | (df["end"] < df["begin"])
    )

    duration_sec = audio_info.duration
    last_timestamp_sec = df["end"].max() / 1000

    results.append({
        "file_name": csv_path.stem,
        "rows": len(df),
        "participant_turns": int((df["speaker"] == "PAR").sum()),
        "interviewer_turns": int((df["speaker"] == "INV").sum()),
        "invalid_rows": int(invalid.sum()),
        "audio_duration_sec": duration_sec,
        "last_timestamp_sec": last_timestamp_sec,
        "duration_difference_sec": duration_sec - last_timestamp_sec
    })

out = pd.DataFrame(results)

print("Files checked:", len(out))
if not out.empty:
    print("Files with invalid rows:", int((out["invalid_rows"] > 0).sum()))
    print(
        "Files with unknown speaker labels:",
        int(out["file_name"].isna().sum())
    )
    print(
        "Mean absolute duration difference (seconds):",
        round(out["duration_difference_sec"].abs().mean(), 3)
    )
    print("\nSpeaker labels and turn counts:")
    print(out[["participant_turns", "interviewer_turns"]].describe())

    output_path = BASE_DIR / "outputs" / "segmentation_quality.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print("\nSaved:", output_path)

print("\nValidation complete.")
