"""
data_loader.py — RAVDESS Dataset Loader

Provides utilities to load and parse the RAVDESS (Ryerson Audio-Visual Database
of Emotional Speech and Song) dataset.  The loader walks a root directory, reads
every `.wav` file, extracts the emotion label encoded in the filename, and
returns a tidy pandas DataFrame ready for analysis and modelling.

RAVDESS filename convention (7 numerical identifiers separated by hyphens):
    Modality - Vocal channel - Emotion - Intensity - Statement - Repetition - Actor
    e.g.  03-01-06-01-02-01-12.wav
              ^--- emotion code (position 3, 1-indexed)

Emotion codes
    01 = neutral   05 = angry
    02 = calm      06 = fearful
    03 = happy     07 = disgust
    04 = sad       08 = surprised
"""

import os
import logging
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Emotion mapping
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_emotion_from_filename(filename: str) -> Optional[str]:
    """Extract the emotion label from a RAVDESS-formatted filename.

    Parameters
    ----------
    filename : str
        The basename of a RAVDESS `.wav` file, e.g. ``03-01-06-01-02-01-12.wav``.

    Returns
    -------
    str or None
        The human-readable emotion string (e.g. ``"fearful"``), or ``None``
        if the filename does not match the expected format.

    Examples
    --------
    >>> extract_emotion_from_filename("03-01-06-01-02-01-12.wav")
    'fearful'
    """
    try:
        # Strip extension, split on hyphens
        parts = os.path.splitext(filename)[0].split("-")
        if len(parts) < 3:
            return None
        emotion_code = parts[2]
        return EMOTION_MAP.get(emotion_code)
    except Exception:
        return None


def load_ravdess_data(data_dir: str) -> pd.DataFrame:
    """Walk `data_dir` and build a DataFrame of RAVDESS audio files.

    Parameters
    ----------
    data_dir : str
        Root directory that contains the RAVDESS actor sub-folders
        (e.g. ``Actor_01/``, ``Actor_02/``, …).

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns:

        * ``file_path`` — absolute path to each ``.wav`` file
        * ``emotion_label`` — human-readable emotion string

    Raises
    ------
    FileNotFoundError
        If `data_dir` does not exist.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    records: list[dict[str, str]] = []
    skipped = 0

    for root, _dirs, files in os.walk(data_dir):
        for fname in sorted(files):
            # Only process WAV files
            if not fname.lower().endswith(".wav"):
                continue

            emotion = extract_emotion_from_filename(fname)
            if emotion is None:
                logger.warning("Skipping unrecognised file: %s", fname)
                skipped += 1
                continue

            records.append(
                {
                    "file_path": os.path.join(root, fname),
                    "emotion_label": emotion,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        "Loaded %d files (%d skipped) from %s", len(df), skipped, data_dir
    )
    return df


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    directory = sys.argv[1] if len(sys.argv) > 1 else "data"
    dataframe = load_ravdess_data(directory)
    print(dataframe.head(10))
    print(f"\nEmotion distribution:\n{dataframe['emotion_label'].value_counts()}")
