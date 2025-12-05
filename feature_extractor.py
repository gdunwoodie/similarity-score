import numpy as np
import librosa

def extract_features_from_path(path, sr=22050, n_mfcc=20, n_chroma=12):
    """
    Load an audio file and return a dict of features compatible with your dataset:
      mfcc_0 .. mfcc_{n_mfcc-1},
      chroma_0 .. chroma_{n_chroma-1},
      tempo_bpm (float)
    Each MFCC/chroma is the mean across frames.
    """
    # Load audio (mono)
    y, sr_actual = librosa.load(path, sr=sr, mono=True)

    # Optional: trim leading/trailing silence (uncomment if desired)
    # y, _ = librosa.effects.trim(y)

    # MFCCs: shape (n_mfcc, frames)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_means = np.mean(mfccs, axis=1)  # length n_mfcc

    # Chroma (use harmonic component for more stable chroma)
    y_harmonic = librosa.effects.harmonic(y)
    chroma = librosa.feature.chroma_stft(y=y_harmonic, sr=sr, n_chroma=n_chroma)
    chroma_means = np.mean(chroma, axis=1)  # length n_chroma

    # Tempo estimate (BPM) using onset envelope + tempo detection
    # librosa.beat.beat_track returns tempo and beats
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Build dict
    out = {}
    for i, val in enumerate(mfcc_means):
        out[f"mfcc_{i}"] = float(val)
    for i, val in enumerate(chroma_means):
        out[f"chroma_{i}"] = float(val)

    out["tempo_bpm"] = float(tempo)

    return out