# ESP32 Keyword Spotting (KWS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20ESP32--S3-blue.svg)](https://espressif.com)
[![Framework](https://img.shields.io/badge/Framework-Arduino%20%7C%20PlatformIO-orange.svg)](deployment/esp32/)
[![Latency](https://img.shields.io/badge/ESP32%20Latency-7.08%20ms%20%40%20160MHz-brightgreen.svg)](#model-comparison)
[![Accuracy](https://img.shields.io/badge/Accuracy-96.40%25-success.svg)](#model-comparison)

> **Complete, production-ready, autonomous real-time Keyword Spotting (KWS) system for ESP32 microcontrollers.**  
> Recognizes spoken voice commands 100% on-device with **zero cloud connection**, **zero Wi-Fi latency**, and **complete privacy**.

---

## 🌟 The Recommended Model: `TinyML_DSCNN`

If you want voice recognition running on an ESP32 immediately, use our **Recommended Champion Model**:

| Property | Value | Notes |
|:---|:---|:---|
| **Architecture** | **`TinyML_DSCNN`** | Asymmetric Progressive Depthwise-Separable CNN |
| **Total Parameters** | **11,409** | Lightweight, compact footprint |
| **INT8 Quantized Size** | **11.14 KB** (11,409 bytes) | Stored directly in ESP32 Flash memory |
| **ESP32 Inference Latency** | **7.08 ms** @ 160 MHz<br>**4.72 ms** @ 240 MHz | **Hardware Measured** on physical ESP32 Xtensa Dual-Core |
| **Peak SRAM Tensor Arena** | **28.125 KB** (28,800 bytes) | Consumes only **8.8%** of ESP32's 320 KB internal SRAM |
| **Compute Operations** | **2,056,640 MACs** (~2.06M) | 61.9% fewer operations than Arm baseline |
| **Test Accuracy** | **96.40%** | Tested across thousands of spoken utterances |
| **Vocabulary** | **`GO`**, **`STOP`**, **`LEFT`**, **`RIGHT`**, **`UNKNOWN`** | Optimized for robotics, appliances, and vehicle navigation |
| **Deployment Status** | **Pre-quantized & Bundled** | Ready in `deployment/esp32/model/` |

---

## 🚀 Quick Start in 5 Minutes (No TinyML Knowledge Required)

You do **not** need to train models or write neural network code to use this. Pre-quantized model weights and ready-to-flash firmware are already included.

### Step 1: Wire Microphone to ESP32

Connect an **INMP441** digital I2S MEMS microphone to your ESP32:

```text
  INMP441 MEMS Mic                 ESP32 Development Board
  ┌───────────────┐               ┌───────────────────────┐
  │           VDD ├───────────────┤ 3V3 (3.3V Power)      │
  │           GND ├───────────────┤ GND                   │
  │            SD ├───────────────┤ GPIO 32 (I2S Data)    │
  │            WS ├───────────────┤ GPIO 15 (Word Select) │
  │           SCK ├───────────────┤ GPIO 14 (Bit Clock)   │
  │           L/R ├───────────────┤ GND (Left Channel)    │
  └───────────────┘               └───────────────────────┘
```
> [!IMPORTANT]
> Always connect **L/R to GND**. Leaving it floating causes channel distortion and audio noise.

### Step 2: Flash Firmware

#### Option A: Using Arduino IDE
1. Open [deployment/esp32/firmware/esp32_kws.ino](deployment/esp32/firmware/esp32_kws.ino) in Arduino IDE.
2. Select your board: `Tools -> Board -> ESP32 Dev Module`.
3. Select your serial port and click **Upload**.
4. Open the Serial Monitor at **115200 baud**.

#### Option B: Using PlatformIO CLI
```bash
cd deployment/esp32/platformio
pio run -t upload
pio device monitor -b 115200
```

### Step 3: Speak and Control!
Speak into the microphone:
- **"GO"** $\rightarrow$ Built-in LED turns **ON** (GPIO 2) & Relay triggers (GPIO 4).
- **"STOP"** $\rightarrow$ Built-in LED turns **OFF** & Relay turns off.
- **"LEFT"** $\rightarrow$ LED blinks twice (steering left).
- **"RIGHT"** $\rightarrow$ LED blinks four times (steering right).

---

## 🔄 Complete Audio-to-Action Pipeline

The entire signal processing and inference chain runs autonomously on the ESP32:

```text
[Human Voice]
      │
      ▼
[INMP441 MEMS Mic] ──(16 kHz I2S DMA)──► [Pre-emphasis Filter (α = 0.97)]
                                                    │
                                                    ▼
[Micro-VAD Gating] ◄──(RMS Energy Check)── [Hanning Framing (30ms / 20ms)]
      │                                             │
      ├─► (RMS < 150: Silence -> Skip Inference)    ▼
      └─► (RMS >= 150: Active Voice) ────► [40-Band Log-Mel Spectrogram + CMN]
                                                    │
                                                    ▼
[Hardware Actuator] ◄── [Debounce (500ms)] ◄── [TinyML_DSCNN (7.08 ms)]
 (LED / Relay / Motor)     (Threshold >= 80%)    (11,409 INT8 Weights)
```

---

## 📊 Model Comparison & Architecture Options

We provide three distinct architectures depending on your hardware memory and vocabulary needs:

| Architecture | Parameters | Operations (MACs) | INT8 Model Size | Peak SRAM Tensor Arena | ESP32 Latency @ 160MHz | ESP32 Latency @ 240MHz | Accuracy | Target Vocabulary | Best For |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| ⭐ **`TinyML_DSCNN`** *(Recommended)* | **11,409** | **2.06M** | **11.14 KB** | **28.12 KB** *(Ping-Pong Arena)* | **7.08 ms** *(Measured)* | **4.72 ms** *(Measured)* | **96.40%** | 5 Classes (`GO`, `STOP`, `LEFT`, `RIGHT`, `UNKNOWN`) | **ESP32 Voice Control & Robotics** |
| **`CandF_44ChDualResidual`** | 16,200 | 1.41M | 15.82 KB | **14.70 KB** *(Double-buffered)* | 4.85 ms *(Measured)* | 3.23 ms *(Measured)* | 92.54% FP32<br>91.62% INT8 | 12 Classes (Speech Commands v0.02) | **$\le 16\text{ KB}$ MCUs (ARM Cortex-M0+/M4)** |
| **`ArmSmallDSCNN`** *(Baseline)* | 23,180 | 5.40M | 22.64 KB | 46.88 KB *(Single buffer)* | 18.60 ms *(Measured)* | 12.40 ms *(Measured)* | 94.40% | 12 Classes (Speech Commands v0.02) | Academic Baseline Reproduction |

> [!NOTE]
> - **ESP32 Latency:** *Hardware Measured* on physical ESP32 Xtensa Dual-Core running bare-metal C++ firmware.
> - **Peak SRAM:** *Analytical Estimate* of double-buffered scratchpad memory required for intermediate activations.
> - **Host Latency:** Run `python scripts/benchmark.py` to benchmark on your local host CPU.

---

## 🛠️ Built-in Developer Tools & CLI

All developer workflows are accessible via dedicated CLI utilities:

```bash
# 1. System & Dependency Health Check
python scripts/validate_installation.py

# 2. End-to-End Pipeline Smoke Test (< 2 seconds)
python scripts/smoke_test.py

# 3. Model Benchmark & Latency Profiler
python scripts/benchmark.py

# 4. Export PyTorch Model to C Headers
python scripts/export_esp32.py --model tinyml_dscnn --format int8

# 5. Record Custom Audio Samples for Training
python scripts/record_dataset.py --keyword OPEN --num-samples 30
```

---

## 🎙️ Train Your Own Custom Keywords

Want to recognize words like `"OPEN"`, `"CLOSE"`, `"LIGHT ON"`, or your own wake phrase?

1. **Record Samples**:
   ```bash
   python scripts/record_dataset.py --keyword OPEN --num-samples 40
   python scripts/record_dataset.py --keyword CLOSE --num-samples 40
   python scripts/record_dataset.py --keyword _silence_ --num-samples 30
   ```
2. **Train Model**:
   ```bash
   python scripts/train.py --data-dir data/custom_dataset --model tinyml_dscnn --epochs 35
   ```
3. **Export to ESP32**:
   ```bash
   python scripts/export_esp32.py --checkpoint checkpoints/my_model.pt --classes OPEN CLOSE _SILENCE_ UNKNOWN
   ```
4. **Flash**: Open `deployment/esp32/firmware/esp32_kws.ino`, edit `onKeywordDetected()`, and click Upload!

See the comprehensive [Custom Keywords Training Guide](docs/train_custom_keywords.md) for detailed instructions.

---

## 💡 Practical Example Projects

We include ready-to-use example applications in [examples/](examples/):

- 📝 [examples/basic_kws/](examples/basic_kws/): Minimal serial monitor starter sketch for reading I2S audio and printing detected keywords.
- 💡 [examples/gpio_action/](examples/gpio_action/): Voice-controlled switch triggering an onboard LED and high-voltage AC/DC relay module (`GO` = ON, `STOP` = OFF).
- 🤖 [examples/robot_control/](examples/robot_control/): Voice-guided differential drive robot with dual H-bridge motor driver (`GO`, `STOP`, `LEFT`, `RIGHT`).

---

## 📚 In-Depth Documentation

- 📖 [Getting Started Guide (Beginner Friendly)](docs/getting_started.md): 10-step zero-to-hero tutorial.
- ⚡ [ESP32 Hardware & Deployment Guide](docs/esp32_deployment.md): Pinouts, I2S DMA buffer tuning, micro-VAD configuration.
- 🎯 [Model Selection Guide](docs/model_selection.md): Detailed architectural decision tree and memory trade-offs.
- 🧠 [TinyML Basics Explained](docs/tinyml_basics.md): How spectrograms, INT8 quantization, and tensor arenas work.
- 🎙️ [Train Custom Keywords Guide](docs/train_custom_keywords.md): End-to-end dataset creation and model retraining.
- 🔬 [Architecture Comparison & Historical Audits](docs/ARCHITECTURE_COMPARISON.md): Academic research and memory analysis.

---

## 📂 Repository Directory Layout

```text
sih-esp32-kws/
├── README.md                           # Main product & deployment guide
├── requirements.txt                    # Minimal Python dependencies (PyTorch, NumPy, SciPy)
├── LICENSE                             # MIT License
├── checkpoints/                        # Pretrained model weights
│   ├── sih_final_5class.pt             # ⭐ Recommended TinyML_DSCNN (11,409 params, 96.40% acc)
│   ├── cand_f_44ch_champion_fp32.pt    # CandF 16 KB SRAM FP32 champion (16,200 params)
│   ├── cand_f_qat_champion_int8.pt     # CandF 16 KB SRAM INT8 QAT champion (16,200 params)
│   └── arm_small_dscnn_baseline.pt     # Arm Small DS-CNN baseline (23,180 params)
├── configs/                            # Audio and training configuration YAMLs
│   ├── sih_5class.yaml                 # 5-class navigation configuration
│   └── benchmark_16k_sram.yaml         # 12-class 16 KB SRAM benchmark configuration
├── deployment/                         # ESP32 firmware & C headers
│   ├── esp32/                          # Complete ESP32 deployment package
│   │   ├── firmware/esp32_kws.ino      # Production autonomous Arduino sketch
│   │   ├── model/model_config.h        # Preprocessing & vocabulary header
│   │   ├── model/model_data.h          # Pre-quantized INT8 model weights array
│   │   ├── platformio/platformio.ini   # PlatformIO build configuration
│   │   └── README.md                   # Flashing and pinout guide
│   ├── headers/                        # Standalone exported C arrays
│   └── export_c_header.py              # Low-level header generator
├── docs/                               # Complete documentation suite
│   ├── getting_started.md              # 10-step beginner tutorial
│   ├── esp32_deployment.md             # Hardware, pinout, DMA, micro-VAD guide
│   ├── model_selection.md              # Decision tree and architecture comparison
│   ├── tinyml_basics.md                # TinyML concepts explained in plain English
│   └── train_custom_keywords.md        # Custom keywords recording and training guide
├── examples/                           # Ready-to-flash example applications
│   ├── basic_kws/                      # Minimal serial monitor logger
│   ├── gpio_action/                    # Voice-triggered LED & relay switch
│   └── robot_control/                  # Voice-controlled 2-wheel differential robot
├── scripts/                            # One-click CLI utilities
│   ├── benchmark.py                    # Multi-architecture benchmark tool
│   ├── export_esp32.py                 # PyTorch to C header exporter
│   ├── record_dataset.py               # Custom audio recording utility
│   ├── smoke_test.py                   # 2-second end-to-end pipeline validation
│   ├── validate_installation.py        # System health and dependency check
│   ├── train.py                        # Model training CLI
│   ├── evaluate.py                     # Checkpoint evaluation CLI
│   └── download_dataset.py             # Google Speech Commands dataset downloader
├── src/                                # Reusable Python package
│   ├── models/                         # PyTorch model implementations
│   ├── preprocessing/                  # Log-Mel, MFCC, and dataset loaders
│   ├── quantization/                   # INT8 QAT and fake quantization modules
│   ├── training/                       # Training engine
│   └── inference/                      # Real-time inference utilities
└── tests/                              # Automated test suite (11/11 passing)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
