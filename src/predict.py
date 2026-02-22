"""
predict.py - Predict emotion from a single audio file

Usage:
    python src/predict.py <path_to_wav_file>

Example:
    python src/predict.py data/Actor_01/03-01-01-01-01-01-01.wav

Steps:
  1. Load the trained model (models/model.pkl)
  2. Load the scaler (models/scaler.pkl)
  3. Extract features from the given audio file
  4. Scale the features using the saved scaler
  5. Predict the emotion and print the result
"""

import os
import sys
import numpy as np
import joblib

# Add src folder to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))

from extract_features import extract_features


# Emotion emojis to make output more fun
EMOTION_EMOJI = {
    "angry":     "😠",
    "calm":      "😌",
    "disgust":   "🤢",
    "fearful":   "😨",
    "happy":     "😊",
    "neutral":   "😐",
    "sad":       "😢",
    "surprised": "😲",
}


def predict_emotion(audio_path):
    """
    Load a .wav file, extract features, and predict the emotion.

    Parameters:
        audio_path (str): Path to a .wav audio file

    Returns:
        str: Predicted emotion label, or None if prediction failed
    """

    # Check the file exists
    if not os.path.isfile(audio_path):
        print(f"  [ERROR] File not found: {audio_path}")
        return None

    # Load the trained Random Forest model
    if not os.path.isfile("models/model.pkl"):
        print("  [ERROR] Model not found at models/model.pkl")
        print("  Please run src/train_model.py first.")
        return None

    # Load the scaler that was used during training
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

    # Reshape to 2D array (model expects multiple rows, we have one sample)
    features = features.reshape(1, -1)

    # Scale the features using the SAME scaler used during training
    # This is very important — without this the prediction will be wrong
    features_scaled = scaler.transform(features)

    # Predict the emotion label (number)
    predicted_number = model.predict(features_scaled)[0]

    # Get the name of the emotion from the label number
    predicted_emotion = label_classes[predicted_number]

    # Get the probability for each emotion
    probabilities = model.predict_proba(features_scaled)[0]

    return predicted_emotion, probabilities, label_classes


def main():
    print("=" * 50)
    print("  Speech Emotion Recognition — Predict")
    print("=" * 50)

    # Get the audio file path from command line
    if len(sys.argv) < 2:
        print("\n  Usage: python src/predict.py <path_to_wav>")
        print("  Example: python src/predict.py data/Actor_01/03-01-01-01-01-01-01.wav")
        sys.exit(1)

    audio_path = sys.argv[1]

    # Make prediction
    result = predict_emotion(audio_path)

    if result is None:
        sys.exit(1)

    predicted_emotion, probabilities, label_classes = result
    emoji = EMOTION_EMOJI.get(predicted_emotion, "🎙️")

    # Print the main result clearly
    print(f"\n  Predicted Emotion:  {predicted_emotion.upper()}  {emoji}")

    # Print probabilities for all emotions (sorted highest first)
    print("\n  Emotion Probabilities:")
    sorted_indices = np.argsort(probabilities)[::-1]  # sort descending
    for i in sorted_indices:
        bar_length = int(probabilities[i] * 30)
        bar = "█" * bar_length
        print(f"    {label_classes[i]:<12} {bar:<30}  {probabilities[i]*100:.1f}%")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
