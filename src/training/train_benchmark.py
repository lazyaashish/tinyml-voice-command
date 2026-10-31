"""
Rigorous Keyword Spotting (KWS) Training Engine
Supports:
- Track 1: 12-Class Academic Reproduction (Arm Paper Benchmark)
- Track 2: 5-Class Project Benchmark (SIH 2026 Problem Statement 172)
- Track 3: Full Proposed Pipeline with SpecAugment + Pitch Warping + CMN
"""

import os
import sys
import time
import json
import glob
import random
import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.audio_features import (
    AudioFeatureExtractor, prepare_kws_data_indices,
    load_wav_file, apply_time_shift, apply_background_noise,
    apply_pitch_warp, apply_specaugment_batch,
    ARM_TARGET_WORDS, SIH_TARGET_WORDS
)
from src.models.baseline_dscnn import ArmSmallDSCNN
from src.models.proposed_progressive_dscnn import ProgressiveDSCNN

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cache_features(data_items: List[Dict], feature_type: str,
                   noise_files: List[str] = None, is_training: bool = False,
                   use_pitch_warp: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts features for all items into memory for ultra-fast training iterations.
    Applies audio-domain augmentations during caching.
    """
    extractor = AudioFeatureExtractor(feature_type=feature_type)
    num_samples = len(data_items)
    f_dim = 10 if feature_type == 'mfcc10' else 40
    
    X = np.zeros((num_samples, 1, 49, f_dim), dtype=np.float32)
    y = np.zeros(num_samples, dtype=np.int64)
    
    noises = []
    if noise_files:
        for nf in noise_files:
            try:
                noises.append(load_wav_file(nf))
            except Exception:
                pass
                
    t0 = time.time()
    for i, item in enumerate(data_items):
        is_silence = item.get('is_silence', False)
        if is_silence:
            audio = np.zeros(16000, dtype=np.float32)
            if noises:
                n = random.choice(noises)
                vol = random.uniform(0.0, 0.15)
                start = random.randint(0, max(0, len(n) - 16000))
                audio = vol * n[start : start + 16000]
        else:
            audio = load_wav_file(item['file'])
            if is_training:
                if use_pitch_warp and random.random() < 0.35:
                    audio = apply_pitch_warp(audio)
                audio = apply_time_shift(audio, max_shift_samples=1600)
                if noises:
                    audio = apply_background_noise(audio, noises, noise_prob=0.8, max_vol=0.1)
                    
        feat = extractor.extract(audio)
        X[i] = feat
        y[i] = item['label']
        
    elapsed = time.time() - t0
    print(f"  Extracted {num_samples} samples ({feature_type}) in {elapsed:.2f}s ({num_samples/elapsed:.1f} samples/s)")
    return torch.from_numpy(X), torch.from_numpy(y)


def run_training(model_name: str, task: str, feature_type: str,
                 epochs: int = 25, batch_size: int = 100, lr: float = 5e-4,
                 use_specaugment: bool = False, use_pitch_warp: bool = False,
                 output_dir: str = "results", seed: int = 42) -> Dict:
    """
    Main training function.
    """
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"STARTING TRAINING: Model={model_name} | Task={task} | Features={feature_type}")
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size} | LR: {lr} | Seed: {seed}")
    print(f"SpecAugment: {use_specaugment} | Pitch Warp: {use_pitch_warp}")
    print("=" * 80)
    
    # 1. Dataset Indexing
    data_dir = "dataset/speech_commands_v0.02"
    if task == "12_class":
        wanted_words = ARM_TARGET_WORDS
        num_classes = 12
        include_silence = True
    elif task == "5_class":
        wanted_words = SIH_TARGET_WORDS
        num_classes = 5
        include_silence = False
    else:
        raise ValueError(f"Unknown task: {task}")
        
    print("\n1. Indexing Dataset Partitions (Pete Warden / Arm SHA-1 Speaker Hash)...")
    data_index, words_list, word_to_index = prepare_kws_data_indices(
        data_dir, wanted_words, silence_percentage=10.0, unknown_percentage=10.0,
        include_silence_class=include_silence, seed=seed
    )
    print(f"Classes ({num_classes}): {words_list}")
    print(f"Train: {len(data_index['training'])} | Val: {len(data_index['validation'])} | Test: {len(data_index['testing'])}")
    
    noise_files = glob.glob(os.path.join(data_dir, "_background_noise_", "*.wav"))
    
    # 2. Extract and Cache Features
    print("\n2. Caching Features in Memory...")
    print("Caching Train set...")
    X_train, y_train = cache_features(
        data_index['training'], feature_type, noise_files, is_training=True, use_pitch_warp=use_pitch_warp
    )
    print("Caching Validation set...")
    X_val, y_val = cache_features(
        data_index['validation'], feature_type, noise_files, is_training=False
    )
    print("Caching Test set...")
    X_test, y_test = cache_features(
        data_index['testing'], feature_type, noise_files, is_training=False
    )
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    
    # 3. Model Instantiation
    freq_bins = 10 if feature_type == 'mfcc10' else 40
    if model_name == "baseline_dscnn":
        model = ArmSmallDSCNN(num_classes=num_classes, in_channels=1, freq_bins=freq_bins).to(device)
    elif model_name == "proposed_progressive":
        model = ProgressiveDSCNN(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
        
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n3. Model Initialized: {model_name} (Parameters: {total_params:,})")
    
    # 4. Optimizer & LR Scheduler (Arm paper uses Adam, initial 5e-4, then 1e-4)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate step decay at 50% of epochs matching Arm paper
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 2), gamma=0.2)
    
    best_val_acc = 0.0
    history = {
        'epoch': [], 'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'lr': []
    }
    
    best_checkpoint_path = os.path.join(output_dir, f"{model_name}_{task}_{feature_type}_best.pt")
    
    print("\n4. Training Progress:")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10} {'Time':<8}")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        t_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for specs, labels in train_loader:
            if use_specaugment:
                specs = apply_specaugment_batch(specs)
            specs, labels = specs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * specs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)
            
        cur_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        train_epoch_loss = train_loss / train_total
        train_epoch_acc = (train_correct / train_total) * 100.0
        
        # Validation Pass
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * specs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = (val_correct / val_total) * 100.0
        elapsed = time.time() - t_start
        
        is_best = val_epoch_acc > best_val_acc
        if is_best:
            best_val_acc = val_epoch_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_epoch_acc,
                'model_name': model_name,
                'task': task,
                'feature_type': feature_type,
                'words_list': words_list
            }, best_checkpoint_path)
            
        best_tag = " [BEST]" if is_best else ""
        print(f"{epoch:<6} {train_epoch_loss:<12.4f} {train_epoch_acc:<11.2f}% {val_epoch_loss:<12.4f} {val_epoch_acc:<11.2f}% {cur_lr:<10.6f} {elapsed:<6.1f}s{best_tag}")
        
        history['epoch'].append(epoch)
        history['train_loss'].append(train_epoch_loss)
        history['train_acc'].append(train_epoch_acc)
        history['val_loss'].append(val_epoch_loss)
        history['val_acc'].append(val_epoch_acc)
        history['lr'].append(cur_lr)
        
    # Save training history
    history_path = os.path.join(output_dir, f"{model_name}_{task}_{feature_type}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Saved best checkpoint: {best_checkpoint_path}")
    print(f"Saved training history: {history_path}")
    
    return {
        'best_val_acc': best_val_acc,
        'checkpoint': best_checkpoint_path,
        'history_path': history_path,
        'words_list': words_list
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train KWS Model")
    parser.add_argument("--model", type=str, required=True, choices=["baseline_dscnn", "proposed_progressive"])
    parser.add_argument("--task", type=str, required=True, choices=["12_class", "5_class"])
    parser.add_argument("--features", type=str, default="mfcc10", choices=["mfcc10", "mel40", "mel40_cmn"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--specaugment", action="store_true")
    parser.add_argument("--pitch_warp", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    run_training(
        model_name=args.model,
        task=args.task,
        feature_type=args.features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_specaugment=args.specaugment,
        use_pitch_warp=args.pitch_warp,
        output_dir=args.output_dir,
        seed=args.seed
    )
