# %% [markdown]
# # 🎙️ Exploratory Data Analysis — RAVDESS Speech Emotion Dataset
#
# **Objective:** Understand the structure, distribution, and audio characteristics
# of the RAVDESS dataset before feature engineering and model training.
#
# **Sections:**
# 1. Setup & Imports
# 2. Load Dataset
# 3. Dataset Overview
# 4. Emotion Distribution
# 5. Audio Duration Analysis
# 6. Waveform Visualisation
# 7. Mel Spectrogram Visualisation
# 8. Key Takeaways

# %% [markdown]
# ---
# ## 1. Setup & Imports

# %%
import sys
import os

# Add project root to path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa

from src.data_loader import load_ravdess_data
from src.visualize import plot_waveform, plot_mel_spectrogram, plot_audio_overview

# Plotting defaults
plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")
plt.rcParams.update({"figure.dpi": 100, "font.size": 11})

print("✅ All imports successful!")

# %% [markdown]
# ---
# ## 2. Load Dataset
#
# We use our custom `data_loader.py` module to walk the `data/` folder,
# parse RAVDESS filenames, and return a tidy DataFrame.

# %%
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

df = load_ravdess_data(DATA_DIR)
print(f"Total audio files loaded: {len(df)}")

# %% [markdown]
# ---
# ## 3. Dataset Overview
#
# A quick look at the first few rows and basic statistics.

# %%
print("=== First 10 rows ===")
print(df.head(10).to_string(index=False))

print(f"\nShape: {df.shape}")
print(f"Unique emotions: {df['emotion_label'].nunique()}")
print(f"Emotion labels: {sorted(df['emotion_label'].unique())}")

# %%
df.info()

# %% [markdown]
# ---
# ## 4. Emotion Distribution
#
# How many samples do we have per emotion?  A balanced dataset is important
# for training a fair classifier.

# %%
emotion_counts = df["emotion_label"].value_counts().sort_index()

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(
    emotion_counts.index,
    emotion_counts.values,
    color=sns.color_palette("husl", n_colors=len(emotion_counts)),
    edgecolor="black",
    linewidth=0.8,
)

# Add count labels on top of each bar
for bar, count in zip(bars, emotion_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        str(count),
        ha="center",
        va="bottom",
        fontweight="bold",
        fontsize=11,
    )

ax.set_title("Emotion Distribution in RAVDESS Dataset", fontsize=15, fontweight="bold")
ax.set_xlabel("Emotion", fontsize=12)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_ylim(0, emotion_counts.max() * 1.15)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "outputs", "emotion_distribution.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"\n{emotion_counts.to_string()}")

# %% [markdown]
# ---
# ## 5. Audio Duration Analysis
#
# Let's check how long each audio clip is.  Consistent durations simplify
# feature extraction (we can pad/truncate to a fixed length).

# %%
def get_duration(file_path: str) -> float:
    """Return the duration of a WAV file in seconds."""
    try:
        y, sr = librosa.load(file_path, sr=None)
        return librosa.get_duration(y=y, sr=sr)
    except Exception:
        return np.nan


print("⏳ Computing audio durations (this may take a moment)...")
df["duration_sec"] = df["file_path"].apply(get_duration)

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(df["duration_sec"].dropna(), bins=30, color="#4C72B0", edgecolor="black", alpha=0.85)
axes[0].set_title("Distribution of Audio Durations", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Duration (seconds)")
axes[0].set_ylabel("Count")
axes[0].axvline(df["duration_sec"].mean(), color="red", linestyle="--", label=f"Mean: {df['duration_sec'].mean():.2f}s")
axes[0].legend()

# Box plot per emotion
df.boxplot(column="duration_sec", by="emotion_label", ax=axes[1], grid=False)
axes[1].set_title("Duration by Emotion", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Emotion")
axes[1].set_ylabel("Duration (seconds)")
plt.suptitle("")  # Remove auto-generated title

plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "outputs", "duration_analysis.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"\nDuration statistics:\n{df['duration_sec'].describe().to_string()}")

# %% [markdown]
# ---
# ## 6. Waveform Visualisation
#
# Waveforms show the raw amplitude of the audio signal over time.
# Let's visualise one sample from each emotion to see if there are
# visible differences in energy and pattern.

# %%
# Pick one random sample per emotion for reproducibility
samples = df.groupby("emotion_label", group_keys=False).apply(lambda x: x.sample(1, random_state=42)).reset_index(drop=True)

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
axes = axes.flatten()

for i, (_, row) in enumerate(samples.iterrows()):
    plot_waveform(row["file_path"], emotion=row["emotion_label"], ax=axes[i])

fig.suptitle("Waveforms — One Sample per Emotion", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "outputs", "waveforms.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Observations:**
# - Angry and fearful samples tend to have higher amplitude (more energy).
# - Calm and neutral samples show lower, more uniform amplitude.
# - The temporal envelope varies noticeably across emotions.

# %% [markdown]
# ---
# ## 7. Mel Spectrogram Visualisation
#
# Mel spectrograms represent the frequency content of audio over time,
# scaled to the mel frequency axis which approximates human hearing.
# These are commonly used as inputs to CNN-based audio classifiers.

# %%
fig, axes = plt.subplots(4, 2, figsize=(16, 16))
axes = axes.flatten()

for i, (_, row) in enumerate(samples.iterrows()):
    plot_mel_spectrogram(row["file_path"], emotion=row["emotion_label"], ax=axes[i])

fig.suptitle("Mel Spectrograms — One Sample per Emotion", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(__file__), "..", "outputs", "mel_spectrograms.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Observations:**
# - High-energy emotions (angry, surprised) show stronger energy across
#   a broader frequency range.
# - Calm and neutral have energy concentrated in lower frequencies.
# - Spectrogram patterns should provide useful features for a CNN classifier.

# %% [markdown]
# ---
# ## 8. Key Takeaways
#
# | Finding | Detail |
# |---------|--------|
# | Dataset size | ~1 440 WAV files across 8 emotions |
# | Balance | Roughly balanced; neutral has fewer samples than others |
# | Duration | Consistent ~3–5 seconds; minimal padding needed |
# | Waveforms | Visible energy differences across emotions |
# | Spectrograms | Clear frequency-pattern differences — promising for CNNs |
#
# **Next steps (Day 2):**
# - Extract MFCC, chroma, and spectral contrast features
# - Build a feature matrix for classical ML models
# - Begin training baseline classifiers (SVM, Random Forest)
