# Prepares audio features and labels, then saves them as .npy files for training.

import os
import sys
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Allow importing from the src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_ravdess_data
from extract_features import extract_features

DATA_DIR   = "data"
OUTPUT_DIR = "data/processed"


def main():
    print("=" * 55)
    print("  Speech Emotion Recognition - Data Preparation")
    print("=" * 55)

    # Load all file paths and emotion labels from the dataset
    print("\n[1] Loading dataset file paths and labels...")
    df = load_ravdess_data(DATA_DIR)
    print(f"    Total audio files found: {len(df)}")

    # Extract features from every audio file
    print("\n[2] Extracting features from audio files...")
    print("    (This may take a few minutes)")

    X = []
    y = []

    for i, row in df.iterrows():
        file_path = row["file_path"]
        label     = row["emotion_label"]

        if (i + 1) % 50 == 0:
            print(f"    Processed {i + 1} / {len(df)} files...")

        features = extract_features(file_path)
        if features is not None:
            X.append(features)
            y.append(label)

    print(f"\n    Successfully extracted features from {len(X)} files")
    print(f"    Skipped files: {len(df) - len(X)}")

    X = np.array(X)
    y = np.array(y)

    print(f"\n    Feature matrix shape: {X.shape}")
    print(f"    Labels shape: {y.shape}")

    # Encode text labels to numbers (e.g. "happy" -> 2)
    print("\n[3] Encoding emotion labels as numbers...")
    label_encoder = LabelEncoder()
    y_encoded     = label_encoder.fit_transform(y)

    print("    Emotion -> Number mapping:")
    for number, emotion in enumerate(label_encoder.classes_):
        print(f"      {number}  ->  {emotion}")

    # Split into 80% train and 20% test, keeping emotion balance
    print("\n[4] Splitting data into train (80%) and test (20%)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print(f"    Training samples: {X_train.shape[0]}")
    print(f"    Testing samples:  {X_test.shape[0]}")

    # Scale features so they all have the same range
    print("\n[5] Scaling features (StandardScaler)...")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    print("    Features scaled successfully.")

    # Save the scaler so predict.py can apply the same scaling to new audio
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    print("    Scaler saved to: models/scaler.pkl")

    # Save all processed arrays to the output folder
    print(f"\n[6] Saving processed data to '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    np.save(os.path.join(OUTPUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUTPUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUTPUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUTPUT_DIR, "y_test.npy"),  y_test)
    np.save(os.path.join(OUTPUT_DIR, "label_classes.npy"), label_encoder.classes_)

    print("    Saved:")
    print(f"      {OUTPUT_DIR}/X_train.npy")
    print(f"      {OUTPUT_DIR}/X_test.npy")
    print(f"      {OUTPUT_DIR}/y_train.npy")
    print(f"      {OUTPUT_DIR}/y_test.npy")
    print(f"      {OUTPUT_DIR}/label_classes.npy")

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
    print("=" * 55)


if __name__ == "__main__":
    main()
