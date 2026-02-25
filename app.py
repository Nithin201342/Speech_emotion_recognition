# Streamlit app for Speech Emotion Recognition with Bootstrap UI.

import os
import sys
import numpy as np
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from extract_features import extract_features

st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

# Load Bootstrap 5 and Poppins font, then apply minimal custom overrides
HEAD = """
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp { font-family: 'Poppins', sans-serif !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 720px; }
</style>
"""
st.markdown(HEAD, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_scaler():
    # Load model, scaler, and label classes once and cache them.
    model         = joblib.load("models/model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    label_classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
    return model, scaler, label_classes

try:
    model, scaler, label_classes = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


def read_model_info():
    # Read model info from models/model_info.txt and return as a dict.
    info = {}
    info_path = "models/model_info.txt"
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    info[key.strip()] = value.strip()
    return info


def predict_emotion(audio_path, model, scaler, label_classes):
    # Extract features from audio and return predicted emotion with probabilities.
    try:
        features = extract_features(audio_path)
        if features is None:
            return None, None
        features_scaled = scaler.transform(features.reshape(1, -1))
        probabilities   = model.predict_proba(features_scaled)[0]
        predicted_index = np.argmax(probabilities)
        return label_classes[predicted_index], probabilities
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None


# Set up session state to track if result should be cleared
if "clear" not in st.session_state:
    st.session_state.clear = False


# Page header
st.markdown("""
<div style="padding:1rem 0 0.5rem 0; border-bottom:2px solid #dee2e6; margin-bottom:1.25rem;">
  <h4 style="font-weight:700; color:#1a1a2e; margin:0;">Speech Emotion Recognition</h4>
  <p style="color:#6c757d; font-size:0.9rem; margin:4px 0 0 0;">
    Upload a .wav audio file to detect the emotion from speech.
  </p>
</div>
""", unsafe_allow_html=True)

# File uploader — hidden if clear was just clicked
uploaded_file = None
if not st.session_state.clear:
    st.markdown('<p style="font-size:0.75rem; font-weight:600; text-transform:uppercase; '
                'letter-spacing:0.07em; color:#6c757d; margin-bottom:4px;">Upload Audio</p>',
                unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Reset the clear flag since a new file is loaded
    st.session_state.clear = False

    # Save the uploaded file temporarily
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Audio preview
    st.markdown('<p style="font-size:0.75rem; font-weight:600; text-transform:uppercase; '
                'letter-spacing:0.07em; color:#6c757d; margin:1rem 0 4px 0;">Audio Preview</p>',
                unsafe_allow_html=True)
    st.audio(temp_path, format="audio/wav")

    # Run prediction
    with st.spinner("Analysing audio..."):
        predicted_emotion, probabilities = predict_emotion(temp_path, model, scaler, label_classes)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if predicted_emotion:
        confidence = probabilities[list(label_classes).index(predicted_emotion)] * 100

        # Prediction result card
        st.markdown(f"""
<div class="card mb-3" style="border-left:4px solid #0d6efd;">
  <div class="card-body">
    <p style="font-size:0.7rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.08em; color:#6c757d; margin-bottom:6px;">Prediction Result</p>
    <h3 style="font-weight:700; color:#0d6efd; margin:0;">{predicted_emotion.capitalize()}</h3>
    <p style="font-size:1rem; font-weight:500; color:#198754; margin:4px 0 0 0;">
      Confidence: {confidence:.1f}%
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

        # Probability breakdown card with progress bars for each emotion
        sorted_pairs = sorted(zip(label_classes, probabilities), key=lambda x: x[1], reverse=True)

        rows = ""
        for label, prob in sorted_pairs:
            pct       = prob * 100
            bar_color = "#0d6efd" if label == predicted_emotion else "#ced4da"
            rows += f"""
<div class="d-flex align-items-center gap-2 mb-2">
  <span style="width:90px; font-size:0.82rem; font-weight:500; color:#343a40;">{label.capitalize()}</span>
  <div class="flex-grow-1 bg-light rounded" style="height:9px;">
    <div style="width:{pct:.1f}%; height:9px; background:{bar_color}; border-radius:4px;"></div>
  </div>
  <span style="width:42px; font-size:0.8rem; color:#6c757d; text-align:right;">{pct:.1f}%</span>
</div>"""

        st.markdown(f"""
<div class="card mb-3">
  <div class="card-body">
    <p style="font-size:0.7rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.08em; color:#6c757d; margin-bottom:10px;">Probability Breakdown</p>
    {rows}
  </div>
</div>
""", unsafe_allow_html=True)

        # Clear Result button — resets the app to default state
        if st.button("🗑️ Clear Result"):
            st.session_state.clear = True
            st.rerun()

    else:
        st.error("Could not process the audio file. Please try a different .wav file.")

elif st.session_state.clear:
    # Show a message and a button to upload again after clearing
    st.info("Result cleared. Upload a new audio file to start again.")
    if st.button("Upload New File"):
        st.session_state.clear = False
        st.rerun()

# Model Information card
model_info = read_model_info()
model_type    = model_info.get("model_type", "Random Forest")
n_trees       = model_info.get("n_estimators", "100")
test_accuracy = model_info.get("test_accuracy", "N/A")

st.markdown(f"""
<div class="card mb-3" style="margin-top:1.5rem; border-left:4px solid #198754;">
  <div class="card-body">
    <p style="font-size:0.7rem; font-weight:600; text-transform:uppercase;
              letter-spacing:0.08em; color:#6c757d; margin-bottom:10px;">Model Information</p>
    <table style="width:100%; font-size:0.88rem; border-collapse:collapse;">
      <tr>
        <td style="padding:4px 0; color:#495057; font-weight:500; width:150px;">Model Type</td>
        <td style="padding:4px 0; color:#212529;">{model_type}</td>
      </tr>
      <tr>
        <td style="padding:4px 0; color:#495057; font-weight:500;">Number of Trees</td>
        <td style="padding:4px 0; color:#212529;">{n_trees}</td>
      </tr>
      <tr>
        <td style="padding:4px 0; color:#495057; font-weight:500;">Test Accuracy</td>
        <td style="padding:4px 0; color:#198754; font-weight:600;">{test_accuracy}%</td>
      </tr>
    </table>
    <p style="font-size:0.8rem; color:#6c757d; margin:10px 0 0 0;">
      This model was trained using extracted audio features like MFCC and Chroma.
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#adb5bd;
            border-top:1px solid #dee2e6; padding-top:1rem; margin-top:1.5rem;">
  Speech Emotion Recognition &nbsp;&bull;&nbsp; RAVDESS Dataset &nbsp;&bull;&nbsp; Random Forest Model
</div>
""", unsafe_allow_html=True)
