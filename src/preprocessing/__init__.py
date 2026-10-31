"""
Audio preprocessing and feature extraction modules for ESP32 KWS.
"""

from .mel_features import AudioFeatureExtractor, KWSAudioDataset, create_mel_filterbank, create_dct_matrix

# Alias for intuitive discovery
LogMelSpectrogramExtractor = AudioFeatureExtractor

__all__ = [
    "AudioFeatureExtractor",
    "LogMelSpectrogramExtractor",
    "KWSAudioDataset",
    "create_mel_filterbank",
    "create_dct_matrix",
]
