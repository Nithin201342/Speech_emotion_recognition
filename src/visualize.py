"""
visualize.py — Audio Visualisation Utilities

Provides helper functions to generate waveform and mel-spectrogram plots
for individual audio files.  Used by the EDA notebook and can be reused
throughout the project for quick audio inspection.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Waveform
# ---------------------------------------------------------------------------
def plot_waveform(
    file_path: str,
    emotion: str = "",
    sr: int = 22050,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the waveform of an audio file.

    Parameters
    ----------
    file_path : str
        Path to the ``.wav`` audio file.
    emotion : str, optional
        Emotion label used in the plot title.
    sr : int, optional
        Target sampling rate (default 22 050 Hz).
    ax : matplotlib.axes.Axes or None, optional
        If provided, the waveform is drawn on this axes object.
        Otherwise a new figure is created and returned.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if a new figure was created, else ``None``.
    """
    y, sr = librosa.load(file_path, sr=sr)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
        created_fig = True

    librosa.display.waveshow(y, sr=sr, ax=ax, color="#1f77b4", alpha=0.8)
    title = f"Waveform — {emotion}" if emotion else "Waveform"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    if created_fig:
        fig.tight_layout()
        return fig
    return None


# ---------------------------------------------------------------------------
# Mel Spectrogram
# ---------------------------------------------------------------------------
def plot_mel_spectrogram(
    file_path: str,
    emotion: str = "",
    sr: int = 22050,
    n_mels: int = 128,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot the mel spectrogram of an audio file.

    Parameters
    ----------
    file_path : str
        Path to the ``.wav`` audio file.
    emotion : str, optional
        Emotion label used in the plot title.
    sr : int, optional
        Target sampling rate (default 22 050 Hz).
    n_mels : int, optional
        Number of mel bands (default 128).
    ax : matplotlib.axes.Axes or None, optional
        If provided, the spectrogram is drawn on this axes object.
        Otherwise a new figure is created and returned.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure object if a new figure was created, else ``None``.
    """
    y, sr = librosa.load(file_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
        created_fig = True

    img = librosa.display.specshow(
        S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap="magma"
    )
    title = f"Mel Spectrogram — {emotion}" if emotion else "Mel Spectrogram"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel Frequency")

    if created_fig:
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()
        return fig

    return None


# ---------------------------------------------------------------------------
# Convenience: side-by-side waveform + spectrogram
# ---------------------------------------------------------------------------
def plot_audio_overview(
    file_path: str,
    emotion: str = "",
    sr: int = 22050,
) -> plt.Figure:
    """Plot waveform and mel spectrogram side by side for a single file.

    Parameters
    ----------
    file_path : str
        Path to the ``.wav`` audio file.
    emotion : str, optional
        Emotion label used in the plot titles.
    sr : int, optional
        Target sampling rate (default 22 050 Hz).

    Returns
    -------
    matplotlib.figure.Figure
        The combined figure.
    """
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
