"""
app.py - Streamlit Web App for Speech Emotion Recognition

Usage:
    streamlit run app.py

Upload a .wav audio file and the app will predict the emotion.
"""

import os
import sys
import numpy as np
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Add src to path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from extract_features import extract_features

# -----------------------------------------------------------------------
# Emotion emojis for display
# -----------------------------------------------------------------------
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

EMOTION_COLOR = {
    "angry":     "#FF5252",
    "calm":      "#66BB6A",
    "disgust":   "#AB47BC",
    "fearful":   "#FF7043",
    "happy":     "#FFD600",
    "neutral":   "#78909C",
    "sad":       "#42A5F5",
    "surprised": "#26C6DA",
}


@st.cache_resource
def load_model_and_scaler():
    """Load the model and scaler once and cache them."""
    model         = joblib.load("models/model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    label_classes = np.load("data/processed/label_classes.npy", allow_pickle=True)
    return model, scaler, label_classes


def predict_from_file(audio_path, model, scaler, label_classes):
    """Extract features, scale, and predict emotion from an audio file."""
    features = extract_features(audio_path)
    if features is None:
        return None, None

    features_scaled = scaler.transform(features.reshape(1, -1))
    predicted_index = model.predict(features_scaled)[0]
    probabilities   = model.predict_proba(features_scaled)[0]
    predicted_label = label_classes[predicted_index]

    return predicted_label, probabilities


# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️",
    layout="centered",
)

# -----------------------------------------------------------------------
# Title
# -----------------------------------------------------------------------
st.title("🎙️ Speech Emotion Recognition")
st.markdown("Upload a `.wav` audio file and the model will detect the emotion in the speech.")
st.markdown("---")

# -----------------------------------------------------------------------
# Load model
# -----------------------------------------------------------------------
try:
    model, scaler, label_classes = load_model_and_scaler()
    st.success("✅ Model loaded successfully", icon="✅")
except FileNotFoundError as e:
    st.error(f"Model or scaler not found. Please run `prepare_data.py` and `train_model.py` first.\n\nError: {e}")
    st.stop()

# -----------------------------------------------------------------------
# File uploader
# -----------------------------------------------------------------------
st.subheader("Upload Audio File")
uploaded_file = st.file_uploader(
    "Choose a .wav file",
    type=["wav"],
    help="Upload a short speech audio clip (3–5 seconds works best)"
)

if uploaded_file is not None:
    # Save the uploaded file temporarily so librosa can read it
    temp_path = "temp_audio.wav"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.read())

    # Play the audio in the browser
    st.audio(temp_path, format="audio/wav")

    # Predict
    with st.spinner("Analysing audio..."):
        predicted_emotion, probabilities = predict_from_file(
            temp_path, model, scaler, label_classes
        )

    # Clean up temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)

    if predicted_emotion is None:
        st.error("Could not extract features from this file. Please try a different audio clip.")
    else:
        emoji = EMOTION_EMOJI.get(predicted_emotion, "🎙️")
        color = EMOTION_COLOR.get(predicted_emotion, "#607D8B")

        st.markdown("---")

        # ---------------------------------------------------------------
        # Main result
        # ---------------------------------------------------------------
        st.subheader("Prediction Result")
        st.markdown(
            f"""
            <div style="
                background-color: {color}22;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 24px;
                text-align: center;
            ">
                <div style="font-size: 64px;">{emoji}</div>
                <div style="font-size: 32px; font-weight: bold; color: {color}; margin-top: 8px;">
                    {predicted_emotion.upper()}
                </div>
                <div style="color: #888; margin-top: 4px;">Detected Emotion</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # ---------------------------------------------------------------
        # Probability bar chart
        # ---------------------------------------------------------------
        st.subheader("Emotion Probabilities")

        fig, ax = plt.subplots(figsize=(8, 4))
        bar_colors = [EMOTION_COLOR.get(e, "#607D8B") for e in label_classes]
        bars = ax.barh(label_classes, probabilities * 100, color=bar_colors, edgecolor="none")

        # Highlight the predicted emotion
        pred_idx = list(label_classes).index(predicted_emotion)
        bars[pred_idx].set_edgecolor("black")
        bars[pred_idx].set_linewidth(2)

        # Add percentage labels
        for bar, prob in zip(bars, probabilities):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{prob * 100:.1f}%",
                va="center",
                fontsize=9,
            )

        ax.set_xlabel("Probability (%)")
        ax.set_xlim(0, 110)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# -----------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:13px;'>"
    "Built with ❤️ using Random Forest + librosa features | MCA Hackathon"
    "</div>",
    unsafe_allow_html=True,
)
