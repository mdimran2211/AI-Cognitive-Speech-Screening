
from pathlib import Path
import wave

# Project ke main folder ka path
PROJECT_DIR = Path(__file__).resolve().parent.parent

# Dataset ke audio folders
AUDIO_DIR = (
    PROJECT_DIR
    / "dataset"
    / "ADReSSo21"
    / "diagnosis"
    / "train"
    / "audio"
)


def check_audio_dataset():
    print("=" * 45)
    print("  COGNITIVE SPEECH ANALYZER")
    print("  Dataset Verification")
    print("=" * 45)

    if not AUDIO_DIR.exists():
        print("ERROR: Audio dataset folder nahi mila!")
        print(f"Expected path: {AUDIO_DIR}")
        return

    total_files = 0

    for label in ["ad", "cn"]:
        folder = AUDIO_DIR / label

        if not folder.exists():
            print(f"\nERROR: '{label}' folder nahi mila.")
            continue

        audio_files = list(folder.rglob("*.wav"))
        total_files += len(audio_files)

        print(f"\nClass: {label.upper()}")
        print(f"Total WAV files: {len(audio_files)}")

        if audio_files:
            sample_file = audio_files[0]
            print(f"Sample file: {sample_file.name}")

            try:
                with wave.open(str(sample_file), "rb") as audio:
                    duration = audio.getnframes() / audio.getframerate()

                    print(f"Sample rate: {audio.getframerate()} Hz")
                    print(f"Channels: {audio.getnchannels()}")
                    print(f"Duration: {duration:.2f} seconds")

            except (wave.Error, OSError, ZeroDivisionError) as error:
                print(f"Audio metadata read nahi hua: {error}")

    print("\n" + "=" * 45)
    print(f"TOTAL WAV FILES: {total_files}")
    print("Dataset verification complete.")
    print("=" * 45)


if __name__ == "__main__":
    check_audio_dataset()

