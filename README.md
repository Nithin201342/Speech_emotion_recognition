<<<<<<< HEAD
# 🎙️ AI Speech Emotion Recognition System

> Detect human emotions from speech audio using Machine Learning and Deep Learning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)]()

---

## 📌 Problem Statement

Human communication is rich with emotional cues that are often embedded in how words are spoken — not just what is said.  Traditional text-based sentiment analysis misses prosodic features like pitch, tone, tempo, and energy that carry critical emotional information.

**Goal:** Build a system that can automatically recognise emotions (neutral, calm, happy, sad, angry, fearful, disgust, surprised) from short speech clips, enabling applications in:

- 🏥 Mental health monitoring
- 📞 Call-centre quality analysis
- 🤖 Emotionally aware voice assistants
- 🎓 E-learning engagement tracking

---

## 💡 Proposed Solution

An end-to-end pipeline that:

1. **Loads & preprocesses** raw `.wav` audio from the RAVDESS dataset.
2. **Extracts features** — MFCCs, Mel spectrograms, chroma, zero-crossing rate, etc.
3. **Trains ML/DL models** — starting with classical classifiers (SVM, Random Forest) and progressing to deep learning (CNN / LSTM on spectrograms).
4. **Evaluates** with confusion matrices, classification reports, and per-emotion accuracy.
5. **Deploys** a simple web interface where users can upload or record audio and receive emotion predictions.

---

## 🏗️ System Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌────────────────┐
│  Raw Audio   │────▶│  Feature         │────▶│  ML / DL       │
│  (.wav)      │     │  Extraction      │     │  Model         │
└──────────────┘     │  (MFCC, Mel,     │     │  (SVM / CNN)   │
                     │   Chroma, ZCR)   │     └───────┬────────┘
                     └──────────────────┘             │
                                                      ▼
                                              ┌───────────────┐
                                              │  Prediction:  │
                                              │  "angry" 😠   │
                                              └───────────────┘
```

---

## 📅 7-Day Development Roadmap

| Day | Focus Area | Key Deliverables |
|-----|-----------|-----------------|
| **1** ✅ | Project Setup & EDA | Folder structure, data loader, visualisations, README |
| **2** | Feature Engineering | MFCC, chroma, spectral contrast extraction pipeline |
| **3** | Classical ML Models | SVM, Random Forest, KNN — train & evaluate |
| **4** | Deep Learning | CNN on mel spectrograms, LSTM on MFCC sequences |
| **5** | Model Optimisation | Hyperparameter tuning, data augmentation, ensembles |
| **6** | Web Interface | Streamlit / Flask app — upload audio → get prediction |
| **7** | Final Polish | Documentation, demo recording, testing, deployment prep |

---

## 📂 Project Structure

```
Speech_emotion_recognition/
├── data/                   # RAVDESS dataset (gitignored)
│   └── Actor_01/ … Actor_24/
├── notebooks/
│   └── 01_eda.py           # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_loader.py      # Dataset loading & parsing
│   └── visualize.py        # Waveform & spectrogram plotting
├── models/                 # Saved trained models (gitignored)
├── outputs/                # Plots & reports (gitignored)
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 🗃️ Dataset — RAVDESS

**Ryerson Audio-Visual Database of Emotional Speech and Song**

| Property | Detail |
|----------|--------|
| Total files | 1 440 speech audio files |
| Actors | 24 (12 male, 12 female) |
| Emotions | 8 — neutral, calm, happy, sad, angry, fearful, disgust, surprised |
| Format | `.wav`, 16-bit, 48 kHz |
| Duration | ~3–5 seconds per clip |

### Filename Convention

Each filename contains 7 hyphen-separated numerical identifiers:

```
{Modality}-{VocalChannel}-{Emotion}-{Intensity}-{Statement}-{Repetition}-{Actor}.wav
```

**Emotion codes:** `01`=neutral · `02`=calm · `03`=happy · `04`=sad · `05`=angry · `06`=fearful · `07`=disgust · `08`=surprised

---

## 🔍 Initial Observations (Day 1)

- [x] Dataset is well-structured with consistent filename conventions.
- [x] All 8 emotion categories are represented across 24 actors.
- [x] Audio clips are short (3–5 s) — suitable for fixed-length feature extraction.
- [ ] Class balance to be confirmed after full EDA.
- [ ] Signal-to-noise ratio quality to be assessed.
- [ ] Feature separability between similar emotions (e.g. calm vs neutral) to be explored.

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Audio processing | librosa, soundfile |
| Data handling | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Classical ML | scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Web app | Streamlit *(Day 6)* |
| Version control | Git + GitHub |

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Speech_emotion_recognition.git
cd Speech_emotion_recognition

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the RAVDESS dataset
# Download from https://zenodo.org/record/1188976
# Extract into  data/  so the structure is  data/Actor_01/ … data/Actor_24/

# 5. Run the EDA notebook
jupyter notebook notebooks/01_eda.py
# or open in VS Code as an interactive Python file
```

---

## 📊 Day 1 Progress

- ✅ Professional project structure created
- ✅ README with problem statement, solution, and roadmap
- ✅ `data_loader.py` — loads RAVDESS dataset into a clean DataFrame
- ✅ `visualize.py` — waveform and mel spectrogram plotting functions
- ✅ EDA notebook with emotion distribution, duration analysis, and audio visualisations
- ✅ `requirements.txt` and `.gitignore`

---

## 📝 License

This project is for educational and hackathon purposes.

---

*Built with ❤️ for the MCA Hackathon — Day 1 of 7*
=======
# speech_emotion_recognition
>>>>>>> 697082aac9eada2d4dec7d661a6bb103f96fe46e
