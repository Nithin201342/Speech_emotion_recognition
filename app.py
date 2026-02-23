"""
app.py - Simple Streamlit Web App for Speech Emotion Recognition
"""

import os
import sys
import numpy as np
import joblib
import streamlit as st

# Add src to path to import feature extraction
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from extract_features import extract_features

@st.cache_resource
def load_model_and_scaler():
    """Load model, scaler, and label classes."""
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
    return model, scaler, label_classes

def predict_emotion(audio_path, model, scaler, label_classes):
    """Extract features and predict emotion."""
    try:
        features = extract_features(audio_path)
        if features is None:
            return None, None

        features_scaled = scaler.transform(features.reshape(1, -1))
        probabilities = model.predict_proba(features_scaled)[0]
        predicted_index = np.argmax(probabilities)
        predicted_label = label_classes[predicted_index]

        return predicted_label, probabilities
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# App UI
st.title("Speech Emotion Recognition")
st.write("Upload a .wav file to test the model.")
st.write("**Model Test Accuracy: 49%**")

try:
    model, scaler, label_classes = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload Audio", type=["wav"])

if uploaded_file is not None:
    # Save temp file
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio(temp_path, format="audio/wav")

    # Predict
    predicted_emotion, probabilities = predict_emotion(temp_path, model, scaler, label_classes)

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if predicted_emotion:
        st.subheader("Prediction")
        st.write(f"**Emotion:** {predicted_emotion.capitalize()}")
        
        # Display confidence percentage
        confidence = probabilities[list(label_classes).index(predicted_emotion)] * 100
        st.write(f"**Confidence:** {confidence:.2f}%")

        st.subheader("Probability Scores")
        for label, prob in zip(label_classes, probabilities):
            st.write(f"- {label.capitalize()}: {prob * 100:.2f}%")
    else:
        st.error("Error processing audio file. Please try a different .wav file.")
