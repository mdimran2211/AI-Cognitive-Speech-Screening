
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "outputs" / "prepared_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "feature_plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)
df["dx"] = df["dx"].str.lower()

features_to_plot = [
    "duration_seconds",
    "pitch_mean",
    "pitch_std",
    "quiet_ratio",
    "quiet_duration_seconds",
    "rms_mean",
    "zero_crossing_mean",
    "mfcc_1_mean",
]

for feature in features_to_plot:
    plt.figure(figsize=(7, 4))

    for label in ["cn", "ad"]:
        values = df.loc[df["dx"] == label, feature].dropna()
        plt.hist(values, bins=15, alpha=0.5, label=label.upper())

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of recordings")
    plt.legend()
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{feature}.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print("Saved:", output_path)

print("\nFeature plots generated successfully.")
