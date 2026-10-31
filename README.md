# SIH ESP32 Keyword Spotting

High-efficiency, on-device Voice Keyword Spotting (KWS) system for low-power embedded deployment on Espressif ESP32 microcontrollers.

---

## Overview

This repository contains the engineering implementation and architectural research for an offline, real-time voice activation system developed for **Smart India Hackathon (SIH 2026)**. The system runs fully on-device with zero cloud connectivity, enabling responsive voice control for resource-constrained robotics and edge hardware.

---

## Problem

Edge microcontrollers operate under strict physical constraints:
- **Tight Memory Budgets:** Typical edge microcontrollers feature limited on-chip SRAM (from 16 KB up to a few hundred kilobytes) shared between application firmware, networking buffers, and neural network tensor arenas.
- **Compute & Power Caps:** Battery-operated systems must execute speech inference within tens of milliseconds while maintaining low average power consumption to enable extended operational lifetimes.
- **Real-Time Responsiveness:** Latency must remain imperceptible (<20 ms) while rejecting background acoustic noise and out-of-vocabulary speech without false activations.

---

## Objective

Design, optimize, and deploy a production-grade keyword spotter capable of:
1. Recognizing targeted navigation keywords with high precision (>95%).
2. Minimizing peak SRAM tensor allocations and arithmetic operations (MACs).
3. Exporting bit-exact 8-bit quantized weights for bare-metal C/C++ firmware running on ESP32.

---

## Approach

The system employs a tightly coupled audio DSP and neural network pipeline:
- **Audio Preprocessing:** 16 kHz mono audio sampled into 1-second frames with a pre-emphasis filter ($\alpha = 0.97$) to compensate for high-frequency glottal roll-off.
- **Spectral Representation:** 40-channel Log-Mel filterbank spectrograms (30 ms window, 20 ms hop) normalized via running Cepstral Mean Normalization (CMN).
- **Cascaded Micro-VAD:** A time-domain zero-crossing rate and energy detector gates spectrogram generation, skipping neural inference during silence and reducing active compute by up to 78%.
- **Compact Architecture:** Depthwise-separable convolutions featuring asymmetric phoneme receptive fields and progressive channel scaling ($28 \to 32 \to 48 \to 64$).
- **Quantization-Aware Optimization:** 8-bit symmetric integer quantization with calibrated scaling to preserve floating-point accuracy within a fraction of a percent.

---

## Target Commands

The production model classifies speech into 5 discrete command categories:
- `GO`
- `STOP`
- `LEFT`
- `RIGHT`
- `UNKNOWN` (out-of-vocabulary speech, background noise, and silence)

---

## Model

The deployed edge model (**`TinyML_DSCNN`**) uses an asymmetric progressive depthwise-separable architecture:

```text
Input: (1, 49, 40) Log-Mel Spectrogram
  │
  ▼
Initial Conv (10×4 Phoneme Kernel, Stride 2×2, 28 Channels) + BatchNorm + ReLU6
  │
  ▼
DS Block 1: 3×3 DW Conv (Stride 1) + 1×1 PW Conv (32 Channels)
  │
  ▼
DS Block 2: 3×3 DW Conv (Stride 2) + 1×1 PW Conv (48 Channels)
  │
  ▼
DS Block 3: 3×3 DW Conv (Stride 1) + 1×1 PW Conv (48 Channels)
  │
  ▼
DS Block 4: 3×3 DW Conv (Stride 1) + 1×1 PW Conv (64 Channels)
  │
  ▼
Adaptive Average Pooling (1×1) + Linear Classifier (64 → 5 Classes)
```

- **Parameters:** 11,409
- **INT8 Model Payload:** 13.36 KB (13,680 bytes)
- **Arithmetic Compute:** 2.06M MACs per inference

---

## Optimization

To establish the theoretical Pareto frontier for microcontroller deployment, multiple architectural variants and compression strategies were evaluated alongside standard academic baselines (such as Arm Small DS-CNN on Google Speech Commands v0.02):

- **Memory Arena Exploration:** Systematically bounded intermediate activation tensor footprints across layer boundaries to determine architectures capable of operating within strict static memory ceilings ($\le 16\text{ KB}$ to $30\text{ KB}$ SRAM arenas).
- **Progressive Channel Sizing:** Allocated narrower channel depths in early, high-resolution layers and wider depths in deep semantic layers to eliminate memory spikes without sacrificing representational capacity.
- **Quantization-Aware Training (QAT):** Straight-Through Estimators (STE) were applied during fine-tuning to model hardware truncation effects and prevent quantization degradation.

---

## Results

### Verified Performance Summary

| Metric | Arm Research Baseline (Zhang et al.) | Deployed SIH Edge Model |
|:---|:---:|:---:|
| **Parameters** | 24,100 | **11,409** |
| **Multiply-Accumulate Ops** | ~5.40M MACs | **2.06M MACs** (61.9% reduction) |
| **INT8 Flash Footprint** | ~38.60 KB | **13.36 KB** |
| **Peak SRAM Tensor Arena** | ~28.00 KB | **28.12 KB** (8.8% of ESP32 SRAM) |
| **Latency @ 160 MHz (ESP32)** | 17.52 ms | **7.08 ms** |
| **Latency @ 240 MHz (ESP32)** | 11.68 ms | **4.72 ms** |
| **Held-Out Test Accuracy** | 94.40% (12 classes) | **96.40%** (5 classes) |

---

## ESP32 Deployment

The trained model is exported directly to a standard C byte array (`sih_model_data.h`) and linked into an ESP32 firmware sketch (`esp32_voice_receiver.ino`):
1. **Audio Capture:** I2S MEMS microphone (e.g. INMP441) streams 16 kHz PCM audio via DMA buffers.
2. **Feature Extraction:** Pre-emphasis, windowing, FFT, and triangular Mel filterbank computation execute on Core 0.
3. **Inference Execution:** Fixed-point depthwise-separable convolution executes on Core 1 in 7.08 ms.
4. **Command Output:** Classified commands trigger GPIO signals and UART telemetry for connected robotics.

---

## Repository Structure

```text
sih-esp32-kws/
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
├── configs/                            # Model and audio configurations
│   ├── sih_5class.yaml                 # SIH production configuration
│   └── benchmark_16k_sram.yaml         # Academic 16 KB SRAM benchmark config
├── src/                                # Modular source code
│   ├── models/                         # Model definitions (TinyML_DSCNN, CandF, Arm)
│   ├── preprocessing/                  # Log-Mel and MFCC feature extractors
│   ├── quantization/                   # QAT engine and fake-quantization modules
│   ├── training/                       # Training and distillation loops
│   └── inference/                      # Real-time microphone demo
├── deployment/                         # Embedded firmware and exported weights
│   ├── esp32_firmware/                 # Arduino/ESP-IDF C++ sketch
│   ├── headers/                        # Bit-exact C weight headers
│   └── export_c_header.py              # Export tool: PyTorch -> C header
├── experiments/                        # Academic research and memory audits
│   ├── baseline/                       # Arm DS-CNN benchmark evaluation
│   ├── sram_optimization/             # Static SRAM arena calculator
│   └── quantization/                  # INT8 QAT evaluation
├── scripts/                            # CLI execution scripts
│   ├── train.py                        # Training CLI
│   ├── evaluate.py                     # Checkpoint evaluation CLI
│   └── download_dataset.py             # Speech Commands dataset downloader
├── checkpoints/                        # Canonical pre-trained weights
│   ├── sih_final_5class.pt             # Deployed SIH 5-class model
│   ├── cand_f_44ch_champion_fp32.pt    # 16 KB SRAM benchmark FP32 model
│   └── cand_f_qat_champion_int8.pt     # 16 KB SRAM benchmark INT8 model
├── docs/                               # Detailed technical reports and figures
├── results/                            # Benchmark comparisons and verification logs
└── tests/                              # Automated unit tests
```

---

## Training

### Quick Smoke Test (Synthetic Data)
```bash
python scripts/train.py --smoke-test --model-type sih
```

### Full Training
1. Download dataset (Google Speech Commands v0.02):
   ```bash
   python scripts/download_dataset.py
   ```
2. Train the SIH model:
   ```bash
   python scripts/train.py --config configs/sih_5class.yaml --model-type sih
   ```

---

## Evaluation

Evaluate pre-trained checkpoints directly:

```bash
# Evaluate final deployed SIH 5-class model
python scripts/evaluate.py --checkpoint checkpoints/sih_final_5class.pt --model-type sih

# Evaluate 16 KB SRAM benchmark champion (CandF)
python scripts/evaluate.py --checkpoint checkpoints/cand_f_44ch_champion_fp32.pt --model-type cand_f
```

Run test suite:
```bash
python -m unittest discover tests
```

---

## Deployment

Export any checkpoint to an aligned C byte array for firmware:
```bash
python deployment/export_c_header.py \
    --checkpoint checkpoints/sih_final_5class.pt \
    --model-type sih \
    --output deployment/headers/sih_model_data.h
```

Flash firmware to ESP32:
1. Open `deployment/esp32_firmware/esp32_voice_receiver.ino` in Arduino IDE or ESP-IDF.
2. Select your board (`ESP32 Dev Module`, CPU Frequency `240 MHz` or `160 MHz`).
3. Connect an I2S microphone (e.g. INMP441: BCLK=GPIO14, LRCL=GPIO15, DOUT=GPIO32).
4. Build and flash.

---

## Research Evolution

The project progressed through a systematic engineering lifecycle:
1. **Baseline Formulation:** Characterized published edge KWS architectures (Arm "Hello Edge" DS-CNN) and identified memory arena bottlenecks on microcontrollers.
2. **Architectural Exploration:** Designed asymmetric receptive field stems and progressive channel topologies to prevent early layer memory spikes.
3. **Memory Bounding & Quantization:** Proven static memory arena calculations and applied quantization-aware fine-tuning to prevent loss of precision in INT8.
4. **Final Selection & Hardware Integration:** Selected the 11,409-parameter progressive DS-CNN, implemented time-domain micro-VAD, and validated end-to-end inference on physical ESP32 hardware.

---

## Limitations

- **Acoustic Noise Floor:** Performance degrades at signal-to-noise ratios below $-5\text{ dB}$ without directional beamforming or dedicated acoustic echo cancellation.
- **Microphone Calibration:** Differences in microphone sensitivity across hardware revisions may require calibrating the zero-crossing micro-VAD threshold.
- **Vocabulary Scope:** The model is optimized for single-keyword activation rather than continuous conversational speech.

---

## License

This project is licensed under the [MIT License](LICENSE).
