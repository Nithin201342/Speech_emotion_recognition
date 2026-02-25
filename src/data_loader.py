# Loads all RAVDESS audio file paths and their emotion labels into a DataFrame.

import os
import logging
from typing import Optional

import pandas as pd

# Maps RAVDESS emotion codes to human-readable labels
EMOTION_MAP: dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def extract_emotion_from_filename(filename: str) -> Optional[str]:
    # Reads the emotion code from a RAVDESS filename and returns the label.
    try:
        parts = os.path.splitext(filename)[0].split("-")
        if len(parts) < 3:
            return None
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code)
    except Exception:
        return None


def load_ravdess_data(data_dir: str) -> pd.DataFrame:
    # Walks data_dir and returns a DataFrame of file paths and emotion labels.
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    records: list[dict[str, str]] = []
    skipped = 0

    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            if not fname.lower().endswith(".wav"):
                continue
            emotion = extract_emotion_from_filename(fname)
            if emotion is None:
                logger.warning("Skipping unrecognised file: %s", fname)
                skipped += 1
                continue
            records.append({
                "file_path": os.path.join(root, fname),
                "emotion_label": emotion,
            })

    df = pd.DataFrame(records)
    logger.info("Loaded %d files (%d skipped) from %s", len(df), skipped, data_dir)
    return df


# Quick sanity check when run directly
if __name__ == "__main__":
    import sys
    directory = sys.argv[1] if len(sys.argv) > 1 else "data"
    dataframe = load_ravdess_data(directory)
    print(dataframe.head(10))
    print(f"\nEmotion distribution:\n{dataframe['emotion_label'].value_counts()}")
