# AI Speech Emotion Recognition System

> Detect human emotions from speech audio using Machine Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()

---

## Problem Statement

Human communication is rich with emotional cues embedded in how words are spoken — not just what is said. Traditional text-based sentiment analysis misses prosodic features like pitch, tone, tempo, and energy that carry critical emotional information.

**Goal:** Build a system that can automatically recognise emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from short speech clips, enabling applications in:

- Mental health monitoring
- Call-centre quality analysis
- Emotionally aware voice assistants
- E-learning engagement tracking

---

## Proposed Solution

An end-to-end pipeline that:

1. **Loads & preprocesses** raw `.wav` audio from the RAVDESS dataset.
2. **Extracts features** — MFCCs, Mel spectrograms, chroma, zero-crossing rate, RMS energy.
3. **Trains a Random Forest classifier** on the extracted features.
4. **Evaluates** with confusion matrices, classification reports, and per-emotion accuracy charts.
5. **Deploys** a simple Streamlit web app where users can upload audio and receive emotion predictions.

---

## System Architecture

```
+--------------+     +------------------+     +----------------+
|  Raw Audio   |---->|  Feature         |---->|  Random Forest |
|  (.wav)      |     |  Extraction      |     |  Classifier    |
+--------------+     |  (MFCC, Mel,     |     +-------+--------+
                     |   Chroma, ZCR)   |             |
                     +------------------+             v
                                              +---------------+
                                              |  Prediction:  |
                                              |  e.g. "angry" |
                                              +---------------+
```

---

## Project Structure

```
Speech_emotion_recognition/
├── data/                        # RAVDESS dataset (gitignored)
│   ├── Actor_01/ … Actor_24/   # Raw .wav files
│   └── processed/               # Extracted features (.npy files)
├── models/
│   ├── model.pkl                # Trained Random Forest model
│   └── scaler.pkl               # StandardScaler (used in prediction)
├── notebooks/
│   └── 01_eda.py                # Exploratory Data Analysis
├── outputs/
│   ├── confusion_matrix.png     # Confusion matrix heatmap
│   └── accuracy_per_emotion.png # Per-emotion accuracy bar chart
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # Dataset loading & parsing
│   ├── extract_features.py      # Audio feature extraction
│   ├── prepare_data.py          # Feature prep, scaling, train/test split
│   ├── train_model.py           # Random Forest training
│   ├── evaluate_model.py        # Evaluation charts
│   ├── predict.py               # Predict emotion from a single file
│   └── visualize.py             # Waveform & spectrogram plotting
├── app.py                       # Streamlit web application
├── requirements.txt
└── .gitignore
```

---

## Dataset — RAVDESS

**Ryerson Audio-Visual Database of Emotional Speech and Song**

| Property | Detail |
|----------|--------|
| Total files | 1 440 speech audio files |
| Actors | 24 (12 male, 12 female) |
| Emotions | 8 — neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Format | `.wav`, 16-bit, 48 kHz |
| Duration | ~3–5 seconds per clip |

### Filename Convention

```
{Modality}-{VocalChannel}-{Emotion}-{Intensity}-{Statement}-{Repetition}-{Actor}.wav
```

**Emotion codes:** `01`=neutral · `02`=calm · `03`=happy · `04`=sad · `05`=angry · `06`=fearful · `07`=disgust · `08`=surprised

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Audio processing | librosa, soundfile |
| Data handling | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Machine Learning | scikit-learn (Random Forest) |
| Web app | Streamlit |
| Version control | Git + GitHub |

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Nithin201342/Speech_emotion_recognition.git
cd Speech_emotion_recognition

# 2. Create & activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the RAVDESS dataset
# Download from https://zenodo.org/record/1188976
# Extract into  data/  so the structure is  data/Actor_01/ … data/Actor_24/

# 5. Run the full pipeline
python src/prepare_data.py      # Extract features & save .npy files
python src/train_model.py       # Train the Random Forest model
python src/evaluate_model.py    # Generate evaluation charts

# 6. Predict emotion from a single audio file
python src/predict.py "data/Actor_01/03-01-01-01-01-01-01.wav"

# 7. Launch the web app
streamlit run app.py
```

---

## Results

- **Model:** Random Forest (100 trees)
- **Features:** MFCC (40) + Chroma (12) + Mel Spectrogram (128) + ZCR (1) + RMS (1) = **182 features**
- **Dataset:** 1440 audio files, 8 emotions, 24 actors
- **Train/Test split:** 80% / 20%
- **Accuracy:** ~72–78% on the held-out test set

---

## License

This project is for educational and hackathon purposes.

---

*Built for the MCA Hackathon*
