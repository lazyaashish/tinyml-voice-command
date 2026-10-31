# ESP32 Autonomous Real-Time Keyword Spotting Deployment

This directory contains the production-ready firmware, quantized model weights, and build configurations to run autonomous real-time Keyword Spotting (KWS) on an ESP32 microcontroller with an I2S digital MEMS microphone.

No cloud connection, Wi-Fi, or host computer is required after flashing. All audio capture, feature extraction, neural network inference, and action triggers run 100% locally on-device.

---

## 1. Hardware Requirements

| Component | Recommended Model | Notes |
|:---|:---|:---|
| **Microcontroller** | ESP32-WROOM-32 / DevKitC / ESP32-S3 | Xtensa dual-core @ 160 or 240 MHz, 320 KB SRAM |
| **Microphone** | INMP441 / ICS-43434 / SPH0645 | Digital I2S MEMS microphone (not analog 3.5mm) |
| **Wiring** | 5x Dupont jumper wires | Female-to-female or female-to-male |
| **Power** | Micro-USB or USB-C cable | 5V / 500 mA from computer or power bank |
| **Output / Actuator** | Onboard LED (GPIO 2) / Relay module (GPIO 4) | Built-in LED works out of the box |

---

## 2. Wiring & Pinout Guide

Connect the INMP441 I2S microphone to the ESP32 as follows:

| INMP441 Pin | ESP32 Pin | Function | Description |
|:---|:---|:---|:---|
| **VDD / VCC** | **3V3** | Power | 3.3V DC power supply (**do NOT connect to 5V**) |
| **GND** | **GND** | Ground | Common ground |
| **SD** | **GPIO 32** | Serial Data | I2S microphone data output (`I2S_SD_PIN`) |
| **WS / LRC** | **GPIO 15** | Word Select | I2S Left-Right clock (`I2S_WS_PIN`) |
| **SCK / BCLK**| **GPIO 14** | Bit Clock | I2S serial continuous clock (`I2S_SCK_PIN`) |
| **L/R** | **GND** | Channel | Tie to GND for Left channel audio |

> [!IMPORTANT]
> Always tie the **L/R** pin to **GND**. Leaving L/R floating causes intermittent channel swapping, zero values, or high-pitched digital noise.

---

## 3. Directory Layout

```text
deployment/esp32/
├── firmware/
│   └── esp32_kws.ino           # Complete Arduino firmware sketch
├── model/
│   ├── model_config.h          # Preprocessing constants, vocabulary, buffer sizes
│   └── model_data.h            # Quantized INT8 model weights (11,409 bytes)
├── platformio/
│   └── platformio.ini          # PlatformIO multi-environment build configuration
└── README.md                   # This deployment guide
```

---

## 4. Flashing Method A: Arduino IDE (Easiest)

1. **Install Arduino IDE** (v2.x or v1.8.x).
2. **Add ESP32 Board Support**:
   - Go to `File -> Preferences -> Additional Board Manager URLs`.
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Open `Tools -> Board -> Boards Manager`, search for `esp32`, and install `esp32 by Espressif Systems` (v2.0.x or v3.x).
3. **Open the Sketch**:
   - Open `deployment/esp32/firmware/esp32_kws.ino`.
   - Ensure `model_config.h` and `model_data.h` are located in `../model/` relative to `esp32_kws.ino` (or place all files in the same sketch folder).
4. **Configure Board Settings**:
   - Board: `ESP32 Dev Module`
   - CPU Frequency: `160 MHz` (or `240 MHz`)
   - Flash Frequency: `80 MHz`
   - Upload Speed: `921600` (or `115200` if upload fails)
   - Port: Select your ESP32 COM/serial port
5. **Click Upload** (arrow icon).
6. **Open Serial Monitor**:
   - Set baud rate to **115200 baud**.
   - Speak commands: `"GO"`, `"STOP"`, `"LEFT"`, `"RIGHT"`.

---

## 5. Flashing Method B: PlatformIO CLI / VS Code

1. Navigate to the `platformio/` directory:
   ```bash
   cd deployment/esp32/platformio
   ```
2. Build and upload:
   ```bash
   pio run -t upload
   ```
3. Open monitor:
   ```bash
   pio device monitor -b 115200
   ```

---

## 6. Serial Monitor Output Example

When running correctly, the serial console outputs:

```text
==================================================
  ESP32 Autonomous Keyword Spotting System
  Model: TinyML_DSCNN (11409 params, 5 classes)
  Audio: 16 kHz Log-Mel 40 (49 frames)
==================================================
[INFO] I2S Microphone Driver Initialized on Port 0
[INFO] Peak Tensor Arena: 28800 bytes allocated
[INFO] Listening for keywords: GO, STOP, LEFT, RIGHT...

>>> [KEYWORD DETECTED]: GO (94.2%) <<<
  -> Action: MOTOR ACTIVATED / LED ON

>>> [KEYWORD DETECTED]: STOP (97.8%) <<<
  -> Action: MOTOR HALTED / LED OFF

>>> [KEYWORD DETECTED]: LEFT (91.5%) <<<
  -> Action: STEERING LEFT (2x Blink)
```

---

## 7. Customizing Actions & Confidence Thresholds

Open `deployment/esp32/firmware/esp32_kws.ino` and locate the action handler:

```c
void onKeywordDetected(const char* keyword, float confidence) {
    if (strcmp(keyword, "GO") == 0) {
        // Your custom logic: activate relay, drive motor, send LoRa packet
    } else if (strcmp(keyword, "STOP") == 0) {
        // Your custom logic: emergency stop, disarm system
    }
}
```

To tune sensitivity:
- `DETECTION_THRESHOLD`: Default `0.80f` (80% confidence). Lower to `0.70f` for quiet environments, raise to `0.88f` for noisy industrial environments.
- `DEBOUNCE_COOLDOWN_MS`: Default `500` ms. Prevents multiple triggers from a single uttered word.
- `SILENCE_RMS_FLOOR`: Default `150.0f`. Micro-VAD energy gate; skips model inference during silence, conserving CPU power.

---

## 8. Troubleshooting Common Issues

| Symptom | Root Cause | Solution |
|:---|:---|:---|
| **No audio / constant 0 predictions** | I2S wiring disconnected or loose | Check VDD (3.3V), GND, SD (GPIO 32), SCK (GPIO 14), WS (GPIO 15). |
| **High screeching noise / random triggers** | L/R pin left floating | Connect INMP441 `L/R` directly to `GND`. |
| **Brownout detector triggered / reboot loop** | USB port unable to provide peak current | Use a powered USB hub or shorter USB data cable. |
| **Upload failure (`A fatal error occurred: Failed to connect to ESP32`)** | Bootloader pin strapping | Hold the **BOOT** button on the ESP32 board while clicking Upload, release once writing begins. |
| **Compilation error: `model_data.h not found`** | Incorrect relative include path | Keep `esp32_kws.ino` in `firmware/` and `model_data.h` in `model/`, or put both in the same folder. |

