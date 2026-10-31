"""
Audio Feature Extraction and Dataset Utilities for Keyword Spotting (KWS).
Implements exact preprocessing pipelines for:
1. Arm "Hello Edge" Baseline: 10 MFCCs from 40ms window, 20ms stride (49 x 10)
2. Proposed Progressive DS-CNN: 40 Log-Mel filterbanks with CMN (49 x 40)
3. Pete Warden / Arm SHA-1 Speaker Hash Partitioning
4. Augmentations: Background noise mixing, Time shifting, SpecAugment, Pitch warping.
"""

import os
import re
import glob
import math
import hashlib
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import scipy.io.wavfile as wav
import torch
from torch.utils.data import Dataset

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1
SILENCE_LABEL = '_silence_'
UNKNOWN_WORD_LABEL = '_unknown_'

# Standard 10 Target Keywords from Arm Hello Edge Paper
ARM_TARGET_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
SIH_TARGET_WORDS = ['go', 'stop', 'left', 'right']

def which_set(filename: str, validation_percentage: int = 10, testing_percentage: int = 10) -> str:
    """
    Determines partition using Pete Warden's official deterministic SHA-1 speaker hash.
    Groups utterances by speaker (characters before '_nohash_').
    """
    base_name = os.path.basename(filename)
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    hash_name_hashed = hashlib.sha1(hash_name.encode('utf-8')).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        return 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        return 'testing'
    else:
        return 'training'


def create_mel_filterbank(num_bins: int = 40, fft_size: int = 1024, sample_rate: int = 16000,
                          low_freq: float = 60.0, high_freq: float = 7800.0) -> np.ndarray:
    """Creates standard triangular Mel filterbank matrix."""
    def hz_to_mel(hz): return 2595.0 * np.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel): return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    
    mel_min = hz_to_mel(low_freq)
    mel_max = hz_to_mel(high_freq)
    mel_points = np.linspace(mel_min, mel_max, num_bins + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size // 2 + 1) * hz_points / (sample_rate / 2)).astype(int)
    
    filterbank = np.zeros((num_bins, fft_size // 2 + 1), dtype=np.float32)
    for m in range(1, num_bins + 1):
        f_m_minus = bin_points[m - 1]
        f_m = bin_points[m]
        f_m_plus = bin_points[m + 1]
        for k in range(f_m_minus, f_m):
            filterbank[m - 1, k] = (k - bin_points[m - 1]) / max(1, f_m - bin_points[m - 1])
        for k in range(f_m, f_m_plus):
            filterbank[m - 1, k] = (bin_points[m + 1] - k) / max(1, bin_points[m + 1] - f_m)
    return filterbank


def create_dct_matrix(num_mfcc: int = 10, num_mel_bins: int = 40) -> np.ndarray:
    """Computes DCT-II matrix to transform log-mel energies to MFCCs."""
    dct_mat = np.zeros((num_mfcc, num_mel_bins), dtype=np.float32)
    for i in range(num_mfcc):
        for j in range(num_mel_bins):
            dct_mat[i, j] = np.cos(np.pi * i * (j + 0.5) / num_mel_bins)
    return dct_mat


class AudioFeatureExtractor:
    """
    Standard KWS Feature Extractor:
    - 16 kHz audio, 1.0 second duration (16,000 samples)
    - 40ms frame length (640 samples)
    - 20ms frame stride (320 samples)
    - 49 time frames
    Supports:
    - 'mfcc10': 10 MFCC coefficients (Arm Paper standard) -> shape: (1, 49, 10)
    - 'mel40': 40 Log-Mel filterbanks -> shape: (1, 49, 40)
    - 'mel40_cmn': 40 Log-Mel filterbanks with Cepstral Mean Normalization (CMN) -> shape: (1, 49, 40)
    """
    def __init__(self, feature_type: str = 'mfcc10', sample_rate: int = 16000,
                 window_size_ms: float = 40.0, window_stride_ms: float = 20.0):
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.frame_length = int(sample_rate * window_size_ms / 1000.0) # 640
        self.frame_step = int(sample_rate * window_stride_ms / 1000.0)   # 320
        self.fft_length = 1024
        self.num_samples = 16000
        self.num_frames = 1 + (self.num_samples - self.frame_length) // self.frame_step # 49
        
        self.hanning_window = np.hanning(self.frame_length).astype(np.float32)
        
        if feature_type.startswith('mel'):
            # Parse number of mel bins, e.g. 'mel40', 'mel20_cmn', 'mel30_cmn'
            clean_name = feature_type.split('_')[0]
            self.num_mel_bins = int(clean_name.replace('mel', ''))
        else:
            self.num_mel_bins = 40
            
        self.mel_filterbank = create_mel_filterbank(self.num_mel_bins, self.fft_length, sample_rate)
        self.dct_matrix = create_dct_matrix(10, self.num_mel_bins)

    def get_feature_shape(self):
        """Returns (channels, time_frames, freq_bins)."""
        if self.feature_type == 'mfcc10':
            return (1, self.num_frames, 10)
        else:
            return (1, self.num_frames, self.num_mel_bins)

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """Extracts 2D feature tensor from 1D audio waveform."""
        if len(audio) < self.num_samples:
            audio = np.pad(audio, (0, self.num_samples - len(audio)), mode='constant')
        elif len(audio) > self.num_samples:
            audio = audio[:self.num_samples]
            
        frames = [audio[i * self.frame_step : i * self.frame_step + self.frame_length] * self.hanning_window
                  for i in range(self.num_frames)]
        fft_frames = np.abs(np.fft.rfft(frames, n=self.fft_length, axis=1)).astype(np.float32)
        
        # Power spectrogram as in TensorFlow contrib_audio
        power_spectrogram = np.square(fft_frames)
        mel_energies = np.dot(power_spectrogram, self.mel_filterbank.T)
        log_mel = np.log(np.maximum(mel_energies, 1e-6))
        
        if self.feature_type == 'mfcc10':
            # DCT-II over Mel energies
            mfcc = np.dot(log_mel, self.dct_matrix.T) # Shape: (49, 10)
            return mfcc[np.newaxis, :, :].astype(np.float32) # (1, 49, 10)
        elif self.feature_type.endswith('_cmn'):
            # Cepstral Mean Normalization across time
            mean_val = np.mean(log_mel, axis=0, keepdims=True)
            std_val = np.std(log_mel, axis=0, keepdims=True) + 1e-5
            cmn_mel = (log_mel - mean_val) / std_val
            return cmn_mel[np.newaxis, :, :].astype(np.float32) # (1, 49, num_mel_bins)
        elif self.feature_type.startswith('mel'):
            return log_mel[np.newaxis, :, :].astype(np.float32) # (1, 49, num_mel_bins)
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")


def load_wav_file(file_path: str, target_samples: int = 16000) -> np.ndarray:
    """Reads 16-bit WAV file and returns normalized float32 waveform [-1.0, 1.0] of length target_samples."""
    try:
        sr, audio = wav.read(file_path)
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
    except Exception:
        audio = np.zeros(target_samples, dtype=np.float32)

    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio


def apply_time_shift(audio: np.ndarray, max_shift_samples: int = 1600) -> np.ndarray:
    """Applies random time shifting up to max_shift_samples (100ms at 16kHz)."""
    shift = random.randint(-max_shift_samples, max_shift_samples)
    if shift > 0:
        audio = np.pad(audio[shift:], (0, shift), mode='constant')
    elif shift < 0:
        audio = np.pad(audio[:shift], (-shift, 0), mode='constant')
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
    return audio[:16000]


def apply_background_noise(audio: np.ndarray, noises: List[np.ndarray],
                           noise_prob: float = 0.8, max_vol: float = 0.1) -> np.ndarray:
    """Mixes a random background noise slice into the audio."""
    if not noises or random.random() > noise_prob:
        return audio
    if len(audio) < 16000:
        audio = np.pad(audio, (0, 16000 - len(audio)), mode='constant')
    elif len(audio) > 16000:
        audio = audio[:16000]
    n = random.choice(noises)
    vol = random.uniform(0.0, max_vol)
    if len(n) > 16000:
        start = random.randint(0, len(n) - 16000)
        noise_slice = n[start : start + 16000]
    else:
        noise_slice = np.pad(n, (0, max(0, 16000 - len(n))), mode='wrap')[:16000]
    mixed = audio + vol * noise_slice
    return np.clip(mixed, -1.0, 1.0)


def apply_pitch_warp(audio: np.ndarray, min_scale: float = 0.85, max_scale: float = 1.20) -> np.ndarray:
    """Simulates vocal pitch stretching and compression."""
    if len(audio) < 1000:
        return audio
    pf = random.uniform(min_scale, max_scale)
    orig_idx = np.arange(len(audio))
    new_len = int(len(audio) / pf)
    warped = np.interp(np.linspace(0, len(audio) - 1, new_len), orig_idx, audio)
    if len(warped) < 16000:
        return np.pad(warped, (0, 16000 - len(warped)), mode='constant')
    return warped[:16000]


def apply_specaugment_batch(specs: torch.Tensor, max_t_mask: int = 6, max_f_mask: int = 4) -> torch.Tensor:
    """Applies online SpecAugment (Time & Frequency masking) on a batch of spectrograms."""
    b, c, t_steps, f_bins = specs.shape
    specs = specs.clone()
    for i in range(b):
        if random.random() < 0.60:
            t_len = random.randint(1, max_t_mask)
            t_start = random.randint(0, max(0, t_steps - t_len))
            specs[i, :, t_start : t_start + t_len, :] = 0.0
        if random.random() < 0.60:
            f_len = random.randint(1, max_f_mask)
            f_start = random.randint(0, max(0, f_bins - f_len))
            specs[i, :, :, f_start : f_start + f_len] = 0.0
    return specs


def prepare_kws_data_indices(data_dir: str, wanted_words: List[str],
                             silence_percentage: float = 10.0,
                             unknown_percentage: float = 10.0,
                             validation_percentage: int = 10,
                             testing_percentage: int = 10,
                             include_silence_class: bool = True,
                             seed: int = 42) -> Tuple[Dict[str, List[Dict]], List[str], Dict[str, int]]:
    """
    Indexes files into train, val, and test splits following the official paper algorithm.
    If include_silence_class=True (12-class):
      Label 0: _silence_, Label 1: _unknown_, Label 2..N+1: wanted_words
    If include_silence_class=False (5-class):
      Label 0..N-1: wanted_words, Label N: _unknown_ (with silence mixed into unknown)
    """
    random.seed(seed)
    if include_silence_class:
        words_list = [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words
    else:
        words_list = wanted_words + [UNKNOWN_WORD_LABEL]
        
    word_to_index = {w: i for i, w in enumerate(words_list)}
    
    data_index = {'training': [], 'validation': [], 'testing': []}
    unknown_index = {'training': [], 'validation': [], 'testing': []}
    
    subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for word in subdirs:
        if word == '_background_noise_':
            continue
        word_lower = word.lower()
        wav_files = glob.glob(os.path.join(data_dir, word, '*.wav'))
        is_wanted = word_lower in wanted_words
        
        for wav_path in wav_files:
            set_name = which_set(wav_path, validation_percentage, testing_percentage)
            item = {'file': wav_path, 'word': word_lower}
            if is_wanted:
                item['label'] = word_to_index[word_lower]
                data_index[set_name].append(item)
            else:
                item['label'] = word_to_index[UNKNOWN_WORD_LABEL]
                unknown_index[set_name].append(item)
                
    # Add Silence and Unknown to each partition
    sample_file = data_index['training'][0]['file']
    for s in ['training', 'validation', 'testing']:
        current_size = len(data_index[s])
        
        if include_silence_class:
            silence_count = int(math.ceil(current_size * silence_percentage / 100.0))
            for _ in range(silence_count):
                data_index[s].append({'file': sample_file, 'word': SILENCE_LABEL, 'label': word_to_index[SILENCE_LABEL], 'is_silence': True})
        else:
            # When silence is not a separate class, add small amount of silence to unknown
            silence_count = int(math.ceil(current_size * 0.05))
            for _ in range(silence_count):
                unknown_index[s].append({'file': sample_file, 'word': UNKNOWN_WORD_LABEL, 'label': word_to_index[UNKNOWN_WORD_LABEL], 'is_silence': True})

        random.shuffle(unknown_index[s])
        unknown_count = int(math.ceil(current_size * unknown_percentage / 100.0)) if include_silence_class else int(math.ceil(current_size * 0.25))
        data_index[s].extend(unknown_index[s][:unknown_count])
        random.shuffle(data_index[s])
        
    return data_index, words_list, word_to_index


class KWSAudioDataset(Dataset):
    """PyTorch Dataset that loads audio, applies on-the-fly distortions, and extracts features."""
    def __init__(self, data_items: List[Dict], feature_extractor: AudioFeatureExtractor,
                 noise_files: List[str] = None, is_training: bool = False,
                 use_specaugment: bool = False, use_pitch_warp: bool = False):
        self.data_items = data_items
        self.extractor = feature_extractor
        self.is_training = is_training
        self.use_specaugment = use_specaugment
        self.use_pitch_warp = use_pitch_warp
        
        self.noises = []
        if noise_files:
            for nf in noise_files:
                try:
                    self.noises.append(load_wav_file(nf))
                except Exception:
                    pass

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        item = self.data_items[idx]
        is_silence = item.get('is_silence', False)
        
        if is_silence:
            audio = np.zeros(16000, dtype=np.float32)
            # Silence is pure background noise at volume
            if self.noises:
                n = random.choice(self.noises)
                vol = random.uniform(0.0, 0.2)
                start = random.randint(0, max(0, len(n) - 16000))
                audio = vol * n[start : start + 16000]
        else:
            audio = load_wav_file(item['file'])
            
            if self.is_training:
                # 1. Pitch warping (if enabled)
                if self.use_pitch_warp and random.random() < 0.35:
                    audio = apply_pitch_warp(audio)
                # 2. Time shift (+- 100ms)
                audio = apply_time_shift(audio, max_shift_samples=1600)
                # 3. Background noise mixing (Arm standard: 80% prob, 0.1 vol)
                if self.noises:
                    audio = apply_background_noise(audio, self.noises, noise_prob=0.8, max_vol=0.1)

        feature = self.extractor.extract(audio) # shape: (1, 49, F)
        return torch.from_numpy(feature), torch.tensor(item['label'], dtype=torch.long)
