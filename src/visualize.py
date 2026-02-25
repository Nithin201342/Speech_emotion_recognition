# Helper functions for plotting waveforms and mel spectrograms.

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def plot_waveform(file_path: str, emotion: str = "", sr: int = 22050, ax: plt.Axes | None = None) -> plt.Figure | None:
    # Plots the waveform of an audio file. Returns a figure if no axes were passed.
    y, sr = librosa.load(file_path, sr=sr)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
        created_fig = True

    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4", alpha=0.8)
    ax.set_title(f"Waveform — {emotion}" if emotion else "Waveform", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


def plot_mel_spectrogram(file_path: str, emotion: str = "", sr: int = 22050, n_mels: int = 128, ax: plt.Axes | None = None) -> plt.Figure | None:
    # Plots the mel spectrogram of an audio file. Returns a figure if no axes were passed.
    y, sr  = librosa.load(file_path, sr=sr)
    S      = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB   = librosa.power_to_db(S, ref=np.max)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        created_fig = True

    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma")
    ax.set_title(f"Mel Spectrogram — {emotion}" if emotion else "Mel Spectrogram", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency")

    if created_fig:
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()
        return fig
    return None


def plot_audio_overview(file_path: str, emotion: str = "", sr: int = 22050) -> plt.Figure:
    # Plots waveform and mel spectrogram side by side for a single file.
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    plot_waveform(file_path, emotion=emotion, sr=sr, ax=axes[0])
    plot_mel_spectrogram(file_path, emotion=emotion, sr=sr, ax=axes[1])
    fig.suptitle(
        f"Audio Overview — {emotion}" if emotion else "Audio Overview",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return fig
