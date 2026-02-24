"""
app.py - Streamlit Web App for Speech Emotion Recognition
Clean Bootstrap 5 UI
"""

import os
import sys
import numpy as np
import joblib
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from extract_features import extract_features

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Speech Emotion Recognition", layout="centered")

# ── Single-block head injection (Bootstrap + font + custom CSS) ───────────────────
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


# ── Load model ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_scaler():
    model         = joblib.load("models/model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    label_classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
    return model, scaler, label_classes

try:
    model, scaler, label_classes = load_model_and_scaler()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


def predict_emotion(audio_path, model, scaler, label_classes):
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


# ── Page header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.5rem 0; border-bottom:2px solid #dee2e6; margin-bottom:1.25rem;">
  <h4 style="font-weight:700; color:#1a1a2e; margin:0;">Speech Emotion Recognition</h4>
  <p style="color:#6c757d; font-size:0.9rem; margin:4px 0 0 0;">
    Upload a .wav audio file to detect the emotion from speech.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Model info badge row ────────────────────────────────────────────────────────────
st.markdown("""
<div class="d-flex align-items-center gap-2 mb-4 p-2 rounded bg-white border" style="font-size:0.875rem;">
  <span class="badge bg-primary" style="font-size:0.75rem; padding:4px 8px;">Model Info</span>
  <span class="text-secondary">Random Forest &nbsp;&bull;&nbsp; RAVDESS Dataset &nbsp;&bull;&nbsp;
    Test Accuracy: <strong class="text-dark">49%</strong>
  </span>
</div>
""", unsafe_allow_html=True)

# ── File uploader ──────────────────────────────────────────────────────────────────
st.markdown('<p style="font-size:0.75rem; font-weight:600; text-transform:uppercase; '
            'letter-spacing:0.07em; color:#6c757d; margin-bottom:4px;">Upload Audio</p>',
            unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Save temp
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Audio preview
    st.markdown('<p style="font-size:0.75rem; font-weight:600; text-transform:uppercase; '
                'letter-spacing:0.07em; color:#6c757d; margin:1rem 0 4px 0;">Audio Preview</p>',
                unsafe_allow_html=True)
    st.audio(temp_path, format="audio/wav")

    # Predict
    with st.spinner("Analysing audio..."):
        predicted_emotion, probabilities = predict_emotion(temp_path, model, scaler, label_classes)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    if predicted_emotion:
        confidence = probabilities[list(label_classes).index(predicted_emotion)] * 100

        # ── Prediction result card ──────────────────────────────────────────────
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

        # ── Probability breakdown card ──────────────────────────────────────────
        sorted_pairs = sorted(
            zip(label_classes, probabilities),
            key=lambda x: x[1], reverse=True
        )

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

    else:
        st.error("Could not process the audio file. Please try a different .wav file.")

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; font-size:0.75rem; color:#adb5bd;
            border-top:1px solid #dee2e6; padding-top:1rem; margin-top:1.5rem;">
  Speech Emotion Recognition &nbsp;&bull;&nbsp; RAVDESS Dataset &nbsp;&bull;&nbsp; Random Forest Model
</div>
""", unsafe_allow_html=True)
