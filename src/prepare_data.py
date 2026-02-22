"""
prepare_data.py - Prepare features and labels for training

Steps:
  1. Load all audio file paths and emotion labels using data_loader.py
  2. Extract features from every audio file using extract_features.py
  3. Encode emotion labels as numbers (e.g. "happy" → 2)
  4. Split into train set (80%) and test set (20%)
  5. Scale features so they are on the same range
  6. Save everything as .npy files in data/processed/
  7. Save the scaler to models/scaler.pkl for use in prediction
"""

import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add the src folder to the path so we can import our own modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_ravdess_data
from extract_features import extract_features


# -----------------------------------------------------------------------
# Settings - change these to match your setup
# -----------------------------------------------------------------------

# Path to the raw dataset (RAVDESS actors folder)
DATA_DIR = "data"

# Where to save the processed .npy files
OUTPUT_DIR = "data/processed"


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition - Data Preparation")
    print("=" * 55)

    # -----------------------------------------------------------------------
    # Step 1: Load file paths and labels from the dataset
    # -----------------------------------------------------------------------
    print("\n[1] Loading dataset file paths and labels...")
    df = load_ravdess_data(DATA_DIR)
    print(f"    Total audio files found: {len(df)}")

    # -----------------------------------------------------------------------
    # Step 2: Extract features from every audio file
    # -----------------------------------------------------------------------
    print("\n[2] Extracting features from audio files...")
    print("    (This may take a few minutes)")

    X = []  # will store feature vectors
    y = []  # will store emotion labels

    for i, row in df.iterrows():
        file_path = row["file_path"]
        label = row["emotion_label"]

        # Progress update every 50 files
        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1} / {len(df)} files...")

        # Extract features for this file
        features = extract_features(file_path)

        # Only add to dataset if extraction was successful
        if features is not None:
            X.append(features)
            y.append(label)

    print(f"\n    Successfully extracted features from {len(X)} files")
    print(f"    Skipped files: {len(df) - len(X)}")

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    print(f"\n    Feature matrix shape: {X.shape}")
    print(f"    Labels shape: {y.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Encode labels
    # Convert text labels ("happy", "sad") to numbers (0, 1, 2...)
    # -----------------------------------------------------------------------
    print("\n[3] Encoding emotion labels as numbers...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("    Emotion → Number mapping:")
    for number, emotion in enumerate(label_encoder.classes_):
        print(f"      {number}  →  {emotion}")

    # -----------------------------------------------------------------------
    # Step 4: Split into train and test sets
    # 80% for training, 20% for testing
    # random_state=42 means we get the same split every time
    # -----------------------------------------------------------------------
    print("\n[4] Splitting data into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded   # keep emotion balance in both splits
    )

    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Testing samples:  {X_test.shape[0]}")

    # -----------------------------------------------------------------------
    # Step 5: Scale features using StandardScaler
    # This makes every feature have mean=0 and std=1
    # Without scaling, some features (like Mel Spectrogram values) can be
    # much larger than others (like ZCR), which confuses the model
    # -----------------------------------------------------------------------
    print("\n[5] Scaling features (StandardScaler)...")
    scaler = StandardScaler()

    # Fit the scaler on TRAINING data only
    # (we then apply the same scaling to test data - avoid data leakage)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("    Features scaled successfully.")

    # Save the scaler so predict.py can apply the exact same scaling to new audio
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("    Scaler saved to: models/scaler.pkl")

    # -----------------------------------------------------------------------
    # Step 6: Save everything to the output folder
    # -----------------------------------------------------------------------
    print(f"\n[6] Saving processed data to '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"), X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"), y_test)

    # Also save the label encoder classes so we can decode predictions later
    np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), label_encoder.classes_)

    print("    Saved:")
    print(f"      {OUTPUT_DIR}/X_train.npy")
    print(f"      {OUTPUT_DIR}/X_test.npy")
    print(f"      {OUTPUT_DIR}/y_train.npy")
    print(f"      {OUTPUT_DIR}/y_test.npy")
    print(f"      {OUTPUT_DIR}/label_classes.npy")

    # -----------------------------------------------------------------------
    # Final summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    print(f"  Total samples used:   {len(X)}")
    print(f"  Feature vector size:  {X.shape[1]}")
    print(f"  Training set shape:   {X_train.shape}")
    print(f"  Testing set shape:    {X_test.shape}")

    print("\n  Samples per emotion:")
    unique, counts = np.unique(y, return_counts=True)
    for emotion, count in zip(unique, counts):
        print(f"    {emotion:<12} {count} samples")

    print("\n  Data preparation complete!")
    print("  You can now use this data to train a model.")
    print("=" * 55)


if __name__ == "__main__":
    main()
