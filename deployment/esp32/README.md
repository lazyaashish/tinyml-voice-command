# ESP32 Autonomous Real-Time Keyword Spotting Deployment

This directory contains the production-ready firmware, quantized model weights, and build configurations to run autonomous real-time Keyword Spotting (KWS) on an ESP32 microcontroller with an I2S digital MEMS microphone and an optional SSD1306 I2C OLED display.

No cloud connection, Wi-Fi, or host computer is required after flashing. All audio capture, feature extraction, neural network inference, and action triggers run 100% locally on-device.

---

## 1. Hardware Requirements & Circuit Wiring

### 1.1 Components
| Component | Model | Function | Notes |
|:---|:---|:---|:---|
| **Microcontroller** | Classic ESP32 DevKit (30-pin or 38-pin) | CPU & DSP engine | Dual-Core Xtensa LX6 @ 160-240 MHz |
| **Microphone** | INMP441 | Digital I2S MEMS Audio Capture | 24-bit high SNR digital microphone |
| **Display** | 0.96-inch SSD1306 (128x64) | Real-time prediction & status UI | I2C interface (address 0x3C / 0x3D) |
| **Status LED** | Onboard LED (GPIO 2) | Ready & command detection | Active HIGH |
| **Actuator** | Relay / Motor Driver (GPIO 4) | Physical hardware trigger | Active HIGH |

---

### 1.2 Exact Circuit Pinout Mapping

```text
 ┌──────────────────────┐              ┌─────────────────────────┐
 │   INMP441 MEMS MIC   │              │   CLASSIC ESP32 DEVKIT  │
 │                      │              │                         │
 │  VDD ────────────────┼──────────────┼──► 3.3V                 │
 │  GND ────────────────┼──────────────┼──► GND                  │
 │  SCK (BCLK) ─────────┼──────────────┼──► GPIO 26              │
 │  WS  (LRCLK) ────────┼──────────────┼──► GPIO 25              │
 │  SD  (DATA) ─────────┼──────────────┼──► GPIO 33              │
 │  L/R ────────────────┼──────────────┼──► GND (Left Channel)   │
 └──────────────────────┘              │                         │
                                       │                         │
 ┌──────────────────────┐              │                         │
 │ 0.96" SSD1306 OLED   │              │                         │
 │                      │              │                         │
 │  VCC ────────────────┼──────────────┼──► 3.3V                 │
 │  GND ────────────────┼──────────────┼──► GND                  │
 │  SDA ────────────────┼──────────────┼──► GPIO 21              │
 │  SCL ────────────────┼──────────────┼──► GPIO 22              │
 └──────────────────────┘              │                         │
                                       │  GPIO 2  ──► Status LED │
                                       │  GPIO 4  ──► Relay/Motor│
                                       └─────────────────────────┘
```

| Device Pin | ESP32 Pin | Function | Notes |
|:---|:---|:---|:---|
| **INMP441 VDD** | **3V3** | Power | 3.3V DC (**NEVER connect to 5V; will damage mic!**) |
| **INMP441 GND** | **GND** | Ground | Common system ground |
| **INMP441 SCK** | **GPIO 26** | `I2S_SCK_PIN` | Serial Bit Clock |
| **INMP441 WS** | **GPIO 25** | `I2S_WS_PIN` | Word Select (Left/Right Clock) |
| **INMP441 SD** | **GPIO 33** | `I2S_SD_PIN` | Serial Audio Data Output |
| **INMP441 L/R** | **GND** | Channel | Tie to GND for Left channel audio (**mandatory**) |
| **OLED VCC** | **3V3** | Power | 3.3V DC power |
| **OLED GND** | **GND** | Ground | Common system ground |
| **OLED SDA** | **GPIO 21** | `OLED_SDA_PIN` | I2C Data Line |
| **OLED SCL** | **GPIO 22** | `OLED_SCL_PIN` | I2C Clock Line |

> [!IMPORTANT]
> - Always connect INMP441 **L/R** to **GND**. Leaving it floating causes channel flipping, DC offset, or digital static.
> - Do **NOT** use GPIO 14/15/32 from older versions; this configuration is pinned to **GPIO 26/25/33** for INMP441 and **GPIO 21/22** for OLED.

---

## 2. Step-by-Step Beginner Guide (From Model to Hardware)

> [!NOTE]
> **Key Concept:** PyTorch `.pt` checkpoints cannot run natively on a microcontroller. The exporter converts the weights and architecture into bit-exact C/C++ static byte arrays (`model_data.h`) and metadata constants (`model_config.h`) that compile directly into the ESP32 firmware binary.

Follow these 12 steps to test your hardware:

### Step 1: Export W76 Model to C Headers
From your computer terminal in `sih-esp32-kws`:
```bash
python scripts/export_esp32.py --model ext_mel40_w76 --format int8
```

### Step 2: Verify Generated Model Headers
Confirm that `model_data.h` and `model_config.h` exist in `deployment/esp32/model/`:
- `model_data.h`: Contains 38,360 bytes of INT8 weights.
- `model_config.h`: Sets `KWS_MODEL_NAME "Ext_Mel40_StridedStem_W76"` and `KWS_NUM_CLASSES 12`.

### Step 3: Open the Firmware Sketch
- Open `deployment/esp32/firmware/esp32_kws.ino` in **Arduino IDE** (or open `deployment/esp32/platformio` in VS Code / PlatformIO).
- If using Arduino IDE, install the required library:
  - `Sketch -> Include Library -> Manage Libraries...`
  - Search and install: **Adafruit SSD1306** (and its dependency **Adafruit GFX**).

### Step 4: Select Board
- In Arduino IDE: `Tools -> Board -> esp32 -> ESP32 Dev Module`.
- Set CPU Frequency: `160 MHz` (or `240 MHz`).

### Step 5: Connect USB Cable
- Plug your ESP32 board into your computer USB port using a data-capable Micro-USB or USB-C cable.

### Step 6: Flash Firmware
- Select the assigned COM port under `Tools -> Port`.
- Click **Upload** (or run `pio run -t upload` in PlatformIO).
- *(If upload hangs on "Connecting...", press and hold the **BOOT** button on the ESP32 until flashing starts).*

### Step 7: Open Serial Monitor
- Set baud rate to **115200 baud**.

### Step 8: Verify OLED Boot Screen
- The OLED will power on with a clean border, showing:
  - `ESP32 TinyML KWS`
  - `Model: Ext_Mel40_StridedStem_W76`
  - `Classes: 12`
  - `Status: LISTENING`

### Step 9: Verify Microphone (Test B)
- In the Serial Monitor input field, type `2` or `mic` and press Enter.
- The terminal will display a live ASCII VU meter for 5 seconds:
  ```text
  RMS:  1240.5 | Peak:  3820.0 | [############                  ]
  ```
- Tap gently or speak near the microphone to confirm the audio bars react dynamically.

### Step 10: Speak Voice Commands
- Speak clearly into the microphone:
  - `"GO"`, `"STOP"`, `"LEFT"`, `"RIGHT"`, `"YES"`, `"NO"`, `"UP"`, `"DOWN"`, `"ON"`, `"OFF"`.

### Step 11: Observe OLED Prediction & Confidence
- When a keyword is recognized (confidence ≥ 80%):
  - OLED highlights the detected word in large text (e.g. `STOP 96.4%`).
  - Built-in LED on GPIO 2 responds (turns ON for GO/ON, turns OFF for STOP/OFF).
  - Relay trigger on GPIO 4 activates accordingly.

### Step 12: Record Real Hardware Benchmark Results
- In the Serial Monitor, type `3` or `bench` to run the hardware benchmark and inspect actual latency, CPU frequency, free heap, and flash memory.

---

## 3. Dedicated Hardware Test Modes

The firmware includes 4 independent test routines that can be triggered at any time via the Serial Monitor:

| Test Mode | Serial Command | What It Verifies |
|:---|:---:|:---|
| **Test A: OLED Diagnostics** | `1` or `oled` | Tests SSD1306 I2C communication at address `0x3C`/`0x3D`, draws screen boundaries, and verifies text rendering. |
| **Test B: INMP441 Mic VU Meter** | `2` or `mic` | Reads 16 kHz I2S DMA buffers, computes RMS & peak energy, streams live ASCII audio bar graph to Serial and OLED. |
| **Test C: W76 Inference Benchmark** | `3` or `bench` | Runs forward pass on test vector, measures exact microsecond execution time, profiles free heap and flash payload. |
| **Test D: Full KWS + OLED** | `4` or `kws` | Resumes continuous autonomous keyword spotting with Micro-VAD energy gating. |

---

## 4. Hardware Benchmark Instrumentation

When `3` or `bench` is triggered, the firmware measures the running ESP32 hardware using high-resolution hardware timers (`micros()`) and ESP system calls:

```text
========================================================
  PHYSICAL ESP32 HARDWARE MEASUREMENTS (LIVE TIMING)
========================================================
  Target Microcontroller : Classic ESP32 DevKit (Dual-Core Xtensa LX6)
  Actual CPU Frequency   : 160 MHz [PHYSICAL ESP32 MEASUREMENT]
  Free Heap Memory       : 224816 bytes (219 KB) [PHYSICAL ESP32 MEASUREMENT]
  Minimum Free Heap      : 218432 bytes (213 KB) [PHYSICAL ESP32 MEASUREMENT]
  Flash Chip Size        : 4194304 bytes (4 MB) [PHYSICAL ESP32 MEASUREMENT]
  Static Model Tensor    : 20448 bytes (19 KB) [STATIC CONFIGURATION]
  Flash Model Weights    : 38360 bytes (37 KB) [STATIC FLASH PAYLOAD]
--------------------------------------------------------
  Audio Buffer Shift     : 0.12 ms (120 us) [PHYSICAL ESP32 MEASUREMENT]
  Feature Extraction     : 4.80 ms (4800 us) [PHYSICAL ESP32 MEASUREMENT]
  W76 Inference Time     : 13.72 ms (13720 us) [PHYSICAL ESP32 MEASUREMENT]
  Total Active Time      : 18.52 ms (18520 us) [PHYSICAL ESP32 MEASUREMENT]
  Real-Time Audio Frame  : 1000.00 ms (1.0 second sliding window)
  CPU Active Duty Cycle  : 1.85% [PHYSICAL ESP32 MEASUREMENT]
========================================================
```

All metrics marked `[PHYSICAL ESP32 MEASUREMENT]` are generated dynamically on-chip by the microcontroller.
