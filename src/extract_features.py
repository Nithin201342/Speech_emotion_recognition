"""
extract_features.py - Extract audio features from a .wav file

We extract 5 types of features from each audio file:
  1. MFCC (Mel Frequency Cepstral Coefficients)
  2. Chroma
  3. Mel Spectrogram
  4. Zero Crossing Rate
  5. RMS Energy

All features are averaged across time so every file gives one fixed-length vector.
"""

import numpy as np
import librosa


def extract_features(file_path):
    """
    Load an audio file and extract features from it.

    Parameters:
        file_path (str): Path to the .wav audio file

    Returns:
        numpy array: A 1D array of features, or None if the file is broken
    """
    try:
        # Load the audio file
        # sr=22050 is the default sample rate (22050 samples per second)
        # duration=3 means we only read the first 3 seconds
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=3)

        # ---------------------------------------------------------------
        # 1. MFCC - Mel Frequency Cepstral Coefficients
        # MFCCs describe the shape of the sound spectrum (like a fingerprint of the voice)
        # They capture how the voice sounds - tone, texture, and timbre
        # We take 40 coefficients and average across time → 40 values
        # ---------------------------------------------------------------
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)  # shape: (40,)

        # ---------------------------------------------------------------
        # 2. Chroma - Chroma Short-Time Fourier Transform
        # Chroma features represent the 12 pitch classes (like piano keys: C, C#, D, ...)
        # They capture the harmonic and melodic content of the audio
        # We take mean across time → 12 values
        # ---------------------------------------------------------------
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)  # shape: (12,)

        # ---------------------------------------------------------------
        # 3. Mel Spectrogram
        # This shows how energy is distributed across frequencies over time
        # (Mel scale is closer to how human ears perceive sound)
        # We use 128 mel bands and take mean across time → 128 values
        # ---------------------------------------------------------------
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_spec_mean = np.mean(mel_spec, axis=1)  # shape: (128,)

        # ---------------------------------------------------------------
        # 4. Zero Crossing Rate (ZCR)
        # Counts how many times the audio signal crosses zero (changes sign)
        # High ZCR = noisy / unvoiced sound (like 's', 'f')
        # Low ZCR = voiced sound (like vowels)
        # We take the mean → 1 value
        # ---------------------------------------------------------------
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr)  # shape: scalar

        # ---------------------------------------------------------------
        # 5. RMS Energy - Root Mean Square Energy
        # Measures how loud the audio is at each point in time
        # Angry/fearful speech tends to have higher energy than calm/neutral
        # We take the mean → 1 value
        # ---------------------------------------------------------------
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)  # shape: scalar

        # ---------------------------------------------------------------
        # Combine all features into one flat vector
        # Total size: 40 + 12 + 128 + 1 + 1 = 182 features
        # ---------------------------------------------------------------
        features = np.hstack([
            mfcc_mean,        # 40 values
            chroma_mean,      # 12 values
            mel_spec_mean,    # 128 values
            [zcr_mean],       # 1 value
            [rms_mean]        # 1 value
        ])

        return features

    except Exception as e:
        # If the file is broken or unreadable, skip it and return None
        print(f"  [WARNING] Could not process file: {file_path}")
        print(f"  Reason: {e}")
        return None


# Quick test when running this file directly
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_features.py <path_to_wav_file>")
    else:
        path = sys.argv[1]
        result = extract_features(path)
        if result is not None:
            print(f"Feature vector shape: {result.shape}")
            print(f"First 10 values: {result[:10]}")
        else:
            print("Feature extraction failed.")
