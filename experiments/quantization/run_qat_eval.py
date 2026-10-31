"""
Master Experiment Runner for Low-SRAM Controlled Recovery:
1. SE Placement Ablation (6 models: b2, b3, b4, b2+b3, b2+b4, b3+b4)
2. Anti-Aliased Frequency Downsampling (BlurPool [1, 2, 1]/4)
3. Delayed Temporal Downsampling (preserves 24 time frames through Block 2)
"""

import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from src.preprocessing.audio_features import (
    AudioFeatureExtractor, prepare_kws_data_indices,
    load_wav_file, apply_time_shift, apply_background_noise,
    apply_specaugment_batch, ARM_TARGET_WORDS
)
from src.models.low_sram_recovery_models import (
    LowSRAM_ConfigurableSE, LowSRAM_AntiAlias, LowSRAM_DelayedTime
)
from src.evaluation.evaluate import (
    compute_model_profile, measure_inference_latency,
    estimate_esp32_latency
)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cache_features_ram(data_items, noise_files, is_training=False):
    extractor = AudioFeatureExtractor(feature_type='mel40_cmn')
    n = len(data_items)
    X = np.zeros((n, 1, 49, 40), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    
    noises = []
    if noise_files:
        for nf in noise_files:
            try:
                noises.append(load_wav_file(nf))
            except Exception:
                pass
                
    t0 = time.time()
    for i, item in enumerate(data_items):
        if item.get('is_silence', False):
            audio = np.zeros(16000, dtype=np.float32)
            if noises:
                noi = random.choice(noises)
                vol = random.uniform(0.0, 0.15)
                start = random.randint(0, max(0, len(noi) - 16000))
                audio = vol * noi[start : start + 16000]
        else:
            audio = load_wav_file(item['file'])
            if is_training:
                audio = apply_time_shift(audio, max_shift_samples=1600)
                if noises:
                    audio = apply_background_noise(audio, noises, noise_prob=0.8, max_vol=0.1)
                    
        X[i] = extractor.extract(audio)
        y[i] = item['label']
        
    elapsed = time.time() - t0
    print(f"    Cached {n} samples in {elapsed:.2f}s ({n/elapsed:.1f} samples/s)")
    return torch.from_numpy(X), torch.from_numpy(y)


def plot_confusion_matrix(cm, classes, title, save_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_model(exp_id, exp_name, model_fn,
                train_loader, val_loader, words_list,
                device, output_dir, epochs=25, lr=1e-3):
    set_seed(42)
    print("\n" + "=" * 80)
    print(f"TRAINING EXPERIMENT: {exp_id} ({exp_name})")
    print("=" * 80)
    
    model = model_fn().to(device)
    prof = compute_model_profile(model, (1, 49, 40))
    print(f"Parameters: {prof['total_params']:,} | MACs: {prof['total_macs']:,} | Peak Arena: {prof['peak_arena_kb']:.2f} KB")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, f"{exp_id}_best.pt")
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
    
    print("\nTraining Progress:")
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<12} {'LR':<10} {'Time':<8}")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for specs, labels in train_loader:
            specs = apply_specaugment_batch(specs, max_t_mask=6, max_f_mask=4)
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
            
        scheduler.step()
        epoch_time = time.time() - t0
        train_loss = train_loss / train_total
        train_acc = (train_correct / train_total) * 100.0
        
        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for v_specs, v_labels in val_loader:
                v_specs, v_labels = v_specs.to(device), v_labels.to(device)
                v_out = model(v_specs)
                v_loss = F.cross_entropy(v_out, v_labels)
                val_loss += v_loss.item() * v_specs.size(0)
                v_preds = v_out.argmax(dim=1)
                val_correct += (v_preds == v_labels).sum().item()
                val_total += v_labels.size(0)
                
        val_loss = val_loss / val_total
        val_acc = (val_correct / val_total) * 100.0
        curr_lr = optimizer.param_groups[0]['lr']
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(curr_lr)
        
        is_best = val_acc > best_val_acc
        mark = " [BEST]" if is_best else ""
        if is_best:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'exp_id': exp_id,
                'exp_name': exp_name,
                'words_list': words_list
            }, best_ckpt)
            
        print(f"{epoch:<6} {train_loss:<12.4f} {train_acc:<10.2f}% {val_loss:<12.4f} {val_acc:<10.2f}% {curr_lr:<10.6f} {epoch_time:<6.1f}s{mark}")
        
    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%. Saved to {best_ckpt}")
    with open(os.path.join(output_dir, f"{exp_id}_history.json"), 'w') as f:
        json.dump(history, f, indent=2)
        
    return best_val_acc, best_ckpt, prof


def evaluate_model(exp_id, exp_name, model_fn, ckpt_path, prof,
                   test_loader, words_list, device, output_dir):
    print(f"\nEvaluating on untouched 4,890 test set: {exp_id}...")
    model = model_fn().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
    model.eval()
    
    y_true, y_pred = [], []
    with torch.no_grad():
        for specs, labels in test_loader:
            specs = specs.to(device)
            outputs = model(specs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())
            
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    test_acc = float(np.mean(y_true == y_pred) * 100.0)
    print(f"--> {exp_id} Final Untouched Test Accuracy: {test_acc:.2f}%")
    
    report = classification_report(y_true, y_pred, target_names=words_list, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    lat_meas = measure_inference_latency(model, (1, 49, 40), iterations=1000, warmup=50)
    lat_theo = estimate_esp32_latency(prof['total_macs'])
    
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cm_path = os.path.join(fig_dir, f"{exp_id}_cm.png")
    plot_confusion_matrix(cm, words_list, f"Confusion Matrix: {exp_id} ({test_acc:.2f}%)", cm_path)
    
    res = {
        'exp_id': exp_id,
        'exp_name': exp_name,
        'test_accuracy': test_acc,
        'hardware_profile': {
            'total_params': prof['total_params'],
            'total_macs': prof['total_macs'],
            'total_mops': prof['total_mops'],
            'flash_int8_kb': prof['flash_int8_kb'],
            'peak_arena_kb': prof['peak_arena_kb'],
            'peak_arena_bytes': prof['peak_arena_bytes']
        },
        'measured_latency': lat_meas,
        'theoretical_latency': lat_theo,
        'per_class': {
            w: {
                'precision': float(report[w]['precision'] * 100.0),
                'recall': float(report[w]['recall'] * 100.0),
                'f1_score': float(report[w]['f1-score'] * 100.0),
                'support': int(report[w]['support'])
            } for w in words_list
        }
    }
    with open(os.path.join(output_dir, f"{exp_id}_eval.json"), 'w') as f:
        json.dump(res, f, indent=2)
        
    return res


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"STARTING LOW-SRAM RECOVERY EXPERIMENTS ON DEVICE: {device} | SEED: 42")
    print("=" * 80)
    
    data_dir = "dataset/speech_commands_v0.02"
    noise_files = [os.path.join(data_dir, "_background_noise_", f) for f in os.listdir(os.path.join(data_dir, "_background_noise_")) if f.endswith('.wav')]
    
    # 1. Dataset Indexing
    print("\n1. Indexing Dataset Partitions...")
    data_index, words_list, _ = prepare_kws_data_indices(
        data_dir, ARM_TARGET_WORDS, silence_percentage=10.0, unknown_percentage=10.0,
        include_silence_class=True, seed=42
    )
    print(f"Classes (12): {words_list}")
    print(f"Train: {len(data_index['training'])} | Val: {len(data_index['validation'])} | Test: {len(data_index['testing'])}")
    
    # 2. Cache Features in RAM
    print("\n2. Caching 40 Log-Mel CMN Features in RAM...")
    X_train, y_train = cache_features_ram(data_index['training'], noise_files, is_training=True)
    X_val, y_val = cache_features_ram(data_index['validation'], noise_files, is_training=False)
    X_test, y_test = cache_features_ram(data_index['testing'], noise_files, is_training=False)
    
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=100, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=100, shuffle=False)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=100, shuffle=False)
    
    output_dir = "results/low_sram_recovery"
    os.makedirs(output_dir, exist_ok=True)
    
    experiments = [
        # SECTION 1: SE PLACEMENT ABLATION
        ("se_b2", "SE_Block2_Only", lambda: LowSRAM_ConfigurableSE(se_b2=True, se_b3=False, se_b4=False)),
        ("se_b3", "SE_Block3_Only", lambda: LowSRAM_ConfigurableSE(se_b2=False, se_b3=True, se_b4=False)),
        ("se_b4", "SE_Block4_Only", lambda: LowSRAM_ConfigurableSE(se_b2=False, se_b3=False, se_b4=True)),
        ("se_b2_b3", "SE_Blocks_2_3", lambda: LowSRAM_ConfigurableSE(se_b2=True, se_b3=True, se_b4=False)),
        ("se_b2_b4", "SE_Blocks_2_4", lambda: LowSRAM_ConfigurableSE(se_b2=True, se_b3=False, se_b4=True)),
        ("se_b3_b4", "SE_Blocks_3_4", lambda: LowSRAM_ConfigurableSE(se_b2=False, se_b3=True, se_b4=True)),
        # SECTION 2: INFORMATION-PRESERVING DOWNSAMPLING
        ("downsample_antialias", "LowSRAM_AntiAlias", lambda: LowSRAM_AntiAlias(num_classes=12)),
        # SECTION 3: TEMPORAL INFORMATION PRESERVATION
        ("temporal_delayed_time", "LowSRAM_DelayedTime", lambda: LowSRAM_DelayedTime(num_classes=12)),
    ]
    
    # PHASE 1: SCREENING BY VALIDATION ACCURACY
    print("\n" + "=" * 80)
    print("PHASE 1: SCREENING BY VALIDATION ACCURACY")
    print("=" * 80)
    
    val_results = {}
    for exp_id, exp_name, model_fn in experiments:
        best_val, best_ckpt, prof = train_model(
            exp_id, exp_name, model_fn,
            train_loader, val_loader, words_list,
            device, output_dir, epochs=25, lr=1e-3
        )
        val_results[exp_id] = {
            'exp_name': exp_name,
            'model_fn': model_fn,
            'best_val_acc': best_val,
            'best_ckpt': best_ckpt,
            'prof': prof
        }
        
    print("\n" + "=" * 80)
    print("PHASE 1 SCREENING RESULTS (VALIDATION ACCURACY RANKING):")
    print("=" * 80)
    sorted_exps = sorted(val_results.items(), key=lambda x: x[1]['best_val_acc'], reverse=True)
    for rank, (eid, info) in enumerate(sorted_exps, 1):
        print(f"Rank {rank}: {eid:<25} ({info['exp_name']:<25}) | Val Acc: {info['best_val_acc']:.2f}% | Params: {info['prof']['total_params']:,} | MACs: {info['prof']['total_macs']:,}")
        
    # PHASE 2: FINAL TEST SET EVALUATION
    print("\n" + "=" * 80)
    print("PHASE 2: EVALUATING FINALISTS ON UNTOUCHED TEST SET")
    print("=" * 80)
    
    final_summary = {}
    for exp_id, info in sorted_exps:
        eval_res = evaluate_model(
            exp_id, info['exp_name'], info['model_fn'],
            info['best_ckpt'], info['prof'],
            test_loader, words_list, device, output_dir
        )
        eval_res['val_accuracy'] = info['best_val_acc']
        final_summary[exp_id] = eval_res
        
    summary_path = os.path.join(output_dir, "recovery_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(final_summary, f, indent=2)
        
    print("\n" + "=" * 80)
    print(f"ALL RECOVERY EXPERIMENTS COMPLETED! Summary saved to {summary_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
