# Extracts audio features from a WAV file and returns a flat feature vector.

import numpy as np
import librosa


def extract_features(file_path):
    # Returns a 1D numpy array of features, or None if the file can't be read.
    try:
        # Load audio, use first 3 seconds at 22050 Hz
        audio, sample_rate = librosa.load(file_path, sr=22050, duration=3)

        # MFCC: captures voice tone and texture (40 values)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Chroma: captures pitch class distribution (12 values)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean = np.mean(chroma, axis=1)

        # Mel spectrogram: energy across frequency bands (128 values)
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128)
        mel_spec_mean = np.mean(mel_spec, axis=1)

        # Zero crossing rate: how often the signal crosses zero (1 value)
        zcr = librosa.feature.zero_crossing_rate(y=audio)
        zcr_mean = np.mean(zcr)

        # RMS energy: average loudness of the audio (1 value)
        rms = librosa.feature.rms(y=audio)
        rms_mean = np.mean(rms)

        # Combine all features into one vector (40 + 12 + 128 + 1 + 1 = 182)
        features = np.hstack([
            mfcc_mean,
            chroma_mean,
            mel_spec_mean,
            [zcr_mean],
            [rms_mean]
        ])

        return features

    except Exception as e:
        print(f"  [WARNING] Could not process file: {file_path}")
        print(f"  Reason: {e}")
        return None


# Run a quick test if this file is called directly
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
