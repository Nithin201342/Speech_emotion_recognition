# Predicts the emotion from a single WAV file using the trained model.

import os
import sys
import numpy as np
import joblib

# Allow importing from the src folder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from extract_features import extract_features


def predict_emotion(audio_path):
    # Returns (predicted_emotion, probabilities, label_classes) or None on failure.

    if not os.path.isfile(audio_path):
        print(f"  [ERROR] File not found: {audio_path}")
        return None

    if not os.path.isfile("models/model.pkl"):
        print("  [ERROR] Model not found at models/model.pkl")
        print("  Please run src/train_model.py first.")
        return None

    if not os.path.isfile("models/scaler.pkl"):
        print("  [ERROR] Scaler not found at models/scaler.pkl")
        print("  Please run src/prepare_data.py first.")
        return None

    # Load model, scaler, and emotion label names
    model         = joblib.load("models/model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    label_classes = np.load("data/processed/label_classes.npy", allow_pickle=True)

    # Extract features from the audio file
    print(f"\n  Extracting features from: {os.path.basename(audio_path)}")
    features = extract_features(audio_path)

    if features is None:
        print("  [ERROR] Could not extract features from this file.")
        return None

    # Reshape for model input and apply the same scaler used during training
    features        = features.reshape(1, -1)
    features_scaled = scaler.transform(features)

    # Predict the emotion
    predicted_number  = model.predict(features_scaled)[0]
    predicted_emotion = label_classes[predicted_number]
    probabilities     = model.predict_proba(features_scaled)[0]

    return predicted_emotion, probabilities, label_classes


def main():
    print("=" * 50)
    print("  Speech Emotion Recognition — Predict")
    print("=" * 50)

    if len(sys.argv) < 2:
        print("\n  Usage: python src/predict.py <path_to_wav>")
        print("  Example: python src/predict.py data/Actor_01/03-01-01-01-01-01-01.wav")
        sys.exit(1)

    audio_path = sys.argv[1]
    result     = predict_emotion(audio_path)

    if result is None:
        sys.exit(1)

    predicted_emotion, probabilities, label_classes = result

    print(f"\n  Predicted Emotion:  {predicted_emotion.upper()}")

    # Print all emotion probabilities sorted from highest to lowest
    print("\n  Emotion Probabilities:")
    sorted_indices = np.argsort(probabilities)[::-1]
    for i in sorted_indices:
        bar_length = int(probabilities[i] * 30)
        bar = "#" * bar_length
        print(f"    {label_classes[i]:<12} {bar:<30}  {probabilities[i]*100:.1f}%")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
