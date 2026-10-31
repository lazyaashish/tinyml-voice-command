# ESP32 Keyword Spotting (KWS)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-ESP32%20%7C%20ESP32--S3-blue.svg)](https://espressif.com)
[![Framework](https://img.shields.io/badge/Framework-Arduino%20%7C%20PlatformIO-orange.svg)](deployment/esp32/)
[![Accuracy](https://img.shields.io/badge/12--Class%20Accuracy-95.0%25-success.svg)](#model-comparison)
[![Latency](https://img.shields.io/badge/ESP32%20Latency-7.08%20ms%20%40%20160MHz-brightgreen.svg)](#model-comparison)

> **Complete, production-ready, autonomous real-time Keyword Spotting (KWS) system for ESP32 microcontrollers.**  
> Recognizes spoken voice commands 100% on-device with **zero cloud connection**, **zero Wi-Fi latency**, and **complete privacy**.

---

## 🌟 Supported Champion Models

Depending on whether your project requires an ultracompact voice-navigation system (e.g. for robots or appliances) or a full 12-keyword vocabulary with $\ge 94\%$ accuracy, the repository provides two pre-trained champion models:

### Option A: `TinyML_DSCNN` (5-Class Robotics Navigation Champion)
- **Target Application**: Robotics navigation, toys, voice switches, and home automation.
- **Vocabulary (5 Classes)**: `GO`, `STOP`, `LEFT`, `RIGHT`, `UNKNOWN`.
- **Parameters**: **11,409** (11.14 KB INT8 model weights).
- **Compute**: **2,056,640 MACs** (~2.06M Operations).
- **Peak SRAM Tensor Arena**: **28.125 KB** (Consumes only **8.8%** of ESP32 SRAM).
- **ESP32 Inference Latency**: **7.08 ms** @ 160 MHz / **4.72 ms** @ 240 MHz *(Analytical Cycle Estimate)*.
- **Test Accuracy**: **91.66%** on 2,027 held-out test clips (*96.40% average confidence on physical trials*).

### Option B: `Ext_Mel40_StridedStem_W76` (12-Class High-Accuracy Champion)
- **Target Application**: Full-featured voice control with standard benchmark vocabulary.
- **Vocabulary (12 Classes)**: `YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN`.
- **Parameters**: **38,360** (37.46 KB INT8 model weights).
- **Compute**: **3,986,688 MACs** (~3.99M Operations).
- **Peak SRAM Tensor Arena**: **20.000 KB** (Fits within standard 25 KB microcontroller limits).
- **ESP32 Inference Latency**: **13.72 ms** @ 160 MHz / **9.15 ms** @ 240 MHz *(Analytical Cycle Estimate)*.
- **Test Accuracy**: **94.99% (~95.0%)** on 4,890 untouched test clips (**95.41%** QAT validation).

---

## 🚀 Quick Start in 5 Minutes (No TinyML Knowledge Required)

Pre-quantized model weights and ready-to-flash firmware are already bundled. You do not need to train anything to get started.

### Step 1: Wire Microphone & OLED to ESP32

Connect your **INMP441** digital I2S MEMS microphone and **SSD1306** I2C OLED display to your classic ESP32 DevKit:

```text
  INMP441 MEMS Mic                 Classic ESP32 DevKit
  ┌───────────────┐               ┌───────────────────────┐
  │           VDD ├───────────────┤ 3V3 (3.3V Power)      │
  │           GND ├───────────────┤ GND                   │
  │     SCK(BCLK) ├───────────────┤ GPIO 26 (Bit Clock)   │
  │     WS(LRCLK) ├───────────────┤ GPIO 25 (Word Select) │
  │      SD(DATA) ├───────────────┤ GPIO 33 (I2S Data)    │
  │           L/R ├───────────────┤ GND (Left Channel)    │
  └───────────────┘               │                       │
  SSD1306 OLED                    │                       │
  ┌───────────────┐               │                       │
  │           VCC ├───────────────┤ 3V3 (3.3V Power)      │
  │           GND ├───────────────┤ GND                   │
  │           SDA ├───────────────┤ GPIO 21 (I2C Data)    │
  │           SCL ├───────────────┤ GPIO 22 (I2C Clock)   │
  └───────────────┘               └───────────────────────┘
```
> [!IMPORTANT]
> - Always connect INMP441 **L/R to GND**. Leaving it floating causes channel distortion and audio noise.
> - **Note on Model Files:** PyTorch `.pt` checkpoint files cannot run directly on the ESP32. The script `scripts/export_esp32.py` converts the checkpoint into bit-exact C/C++ array headers (`model_data.h` and `model_config.h`) which are compiled into the firmware.

### Step 2: Flash Firmware & Run Hardware Tests

1. Export your desired model (e.g. W76 12-class champion):
   ```bash
   python scripts/export_esp32.py --model ext_mel40_w76 --format int8
   ```
2. Open [`deployment/esp32/firmware/esp32_kws.ino`](deployment/esp32/firmware/esp32_kws.ino) in Arduino IDE or PlatformIO.
3. Select board: `Tools -> Board -> ESP32 Dev Module`, select your COM port, and click **Upload**.
4. Open the Serial Monitor at **115200 baud**.
5. Interactive Hardware Diagnostic Commands:
   - Type `1` or `oled`  -> Runs OLED Display Test.
   - Type `2` or `mic`   -> Runs INMP441 5-Second Real-Time Audio VU Meter.
   - Type `3` or `bench` -> Runs W76 Physical Hardware Benchmark (Latency, Free Heap, Flash).
   - Type `4` or `kws`   -> Resumes Continuous Listening Mode.

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
[Hardware Actuator] ◄── [Debounce (500ms)] ◄── [Quantized Neural Net]
 (LED / Relay / Motor)     (Threshold >= 80%)    (7.08 ms - 13.72 ms)
```

---

## 📊 Model Comparison & Architecture Options

| Architecture | Vocabulary | Classes | Parameters | Operations (MACs) | INT8 Model Size | Peak SRAM Arena | ESP32 Latency @ 160MHz | Accuracy & Test Split | Target Hardware & Use Case |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---|
| ⭐ **`TinyML_DSCNN`** | `GO`, `STOP`, `LEFT`, `RIGHT`, `UNKNOWN` | 5 | **11,409** | **2,056,640** (~2.06M) | **11.14 KB** | 28.125 KB *(Ping-Pong Arena)* | **7.08 ms** *(Estimated)* | **91.66%** *(Held-Out Test, 2,027 clips)*<br>96.40% *(Avg Stop Conf, 40 clips)* | **ESP32 Robotics & Voice Navigation** |
| 🏆 **`Ext_Mel40_StridedStem_W76`** | 10 Words + Silence + Unknown | 12 | **38,360** | **3,986,688** (~3.99M) | **37.46 KB** | **20.000 KB** *(Peak In-Out Buffer)* | **13.72 ms** *(Estimated)* | **94.99% (~95.0%)** *(True INT8 Test, 4,890 clips)*<br>95.41% *(QAT Validation)* | **High-Accuracy 12-Class KWS on ESP32 & MCUs** |
| ⚡ **`CandF_44ChDualResidual`** | 10 Words + Silence + Unknown | 12 | **16,200** | **1,406,672** (~1.41M) | **15.82 KB** | **14.703 KB** *(Guaranteed ≤ 16 KB)* | **4.85 ms** *(Estimated)* | **92.54%** FP32 / **91.62%** INT8 *(4,890 test clips)* | **Ultra-low SRAM MCUs (ARM Cortex-M0+/M4)** |
| 📖 **`ArmSmallDSCNN`** | 10 Words + Silence + Unknown | 12 | **23,180** | **5,400,000** (~5.40M) | **22.64 KB** | 46.875 KB *(Single-buffer)* | **18.60 ms** *(Estimated)* | **94.40%** Published / **93.95%** Modern Recipe | Academic Baseline Reproduction |

> [!NOTE]
> - **ESP32 Latency:** *Analytical Cycle Estimate* derived from theoretical layer-by-layer cycle counts divided by clock frequency ($1,132,320\text{ cycles} / 160\text{ MHz} = 7.08\text{ ms}$).
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
python scripts/benchmark.py --iterations 100

# 4. Export Any PyTorch Model to C Headers
python scripts/export_esp32.py --model tinyml_dscnn --format int8
python scripts/export_esp32.py --model ext_mel40_w76 --format int8

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
│   ├── sih_final_5class.pt             # ⭐ Recommended TinyML_DSCNN (11,409 params, 5-class navigation)
│   ├── ext_mel40_w76_champion_95.pt    # 🏆 High-Accuracy Ext_Mel40_W76 (38,360 params, 12-class 95% acc)
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
└── tests/                              # Automated test suite (12/12 passing)
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
