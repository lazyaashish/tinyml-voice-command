# ESP32 Keyword Spotting Deployment Guide

This guide covers advanced hardware integration, pinout mapping across different ESP32 variants, I2S direct memory access (DMA) buffer optimization, firmware flashing workflows, and runtime tuning.

---

## 1. Supported Microcontrollers

| Board Variant | Core / Frequency | SRAM | Flash | Recommended Settings |
|:---|:---|:---|:---|:---|
| **ESP32 DevKit v1 / NodeMCU-32S** | Xtensa Dual-Core @ 160–240 MHz | 320 KB | 4 MB | Default board profile |
| **ESP32-WROOM-32** | Xtensa Dual-Core @ 160–240 MHz | 320 KB | 4–8 MB | Default board profile |
| **ESP32-S3 DevKitC** | Xtensa LX7 Dual-Core with Vector Extensions @ 240 MHz | 512 KB | 8–16 MB | Fast SIMD acceleration |
| **ESP32-C3** | RISC-V Single-Core @ 160 MHz | 400 KB | 4 MB | Supported via software fixed-point |

---

## 2. Hardware Wiring & Pinout Guide

### 2.1 INMP441 Digital Microphone Pin Assignment

| INMP441 Pin | ESP32 Pin | GPIO Function | Signal Description |
|:---|:---|:---|:---|
| **VDD / 3V3** | **3V3** | Power | 3.3V power supply (Do NOT use 5V) |
| **GND** | **GND** | Ground | Common system ground |
| **SCK / BCLK** | **GPIO 26** | `I2S_SCK_PIN` | Serial Bit Clock |
| **WS / LRCLK** | **GPIO 25** | `I2S_WS_PIN` | Word Select (Left/Right clock) |
| **SD / DATA** | **GPIO 33** | `I2S_SD_PIN` | Serial Audio Data out from microphone |
| **L/R** | **GND** | Channel Select | Tie to GND for Left channel (mandatory) |

### 2.2 0.96-inch SSD1306 I2C OLED Display

| OLED Pin | ESP32 Pin | GPIO Function | Signal Description |
|:---|:---|:---|:---|
| **VCC** | **3V3** | Power | 3.3V DC power |
| **GND** | **GND** | Ground | Common system ground |
| **SDA** | **GPIO 21** | `OLED_SDA_PIN` | I2C Serial Data |
| **SCL** | **GPIO 22** | `OLED_SCL_PIN` | I2C Serial Clock |

> [!IMPORTANT]
> Always tie INMP441 **L/R** to **GND**. Leaving L/R floating causes intermittent channel swapping, zero values, or high-pitched digital static. Pin assignments use **GPIO 26, 25, 33** for I2S audio and **GPIO 21, 22** for I2C OLED.

---

## 3. Real-Time Audio Pipeline Architecture

```text
[Digital Audio Stream]
        │  16,000 Hz, 24-bit / 32-bit I2S
        ▼
[DMA Double Buffer]
        │  4 buffers x 512 samples
        ▼
[16 kHz Ring Buffer]
        │  16,000 samples (1.0 sec rolling audio window)
        ▼
[Micro-VAD Energy Filter] ──(RMS < 150)──> Skip Inference (Save CPU Power)
        │
        ▼ (RMS >= 150)
[Pre-emphasis Filter]
        │  y[n] = x[n] - 0.97 * x[n-1]
        ▼
[Framing & Hanning Window]
        │  30 ms frame length (480 samples), 20 ms hop (320 samples)
        ▼
[FFT & Triangular Mel Filterbank]
        │  40 Mel frequency channels (60 Hz – 7800 Hz)
        ▼
[Log Energy & CMN Normalization]
        │  Spectrogram feature tensor: (1, 49, 40)
        ▼
[TinyML_DSCNN Quantized Inference]
        │  28,800 bytes ping-pong tensor arena (~7.08 ms analytical cycle estimate @ 160 MHz)
        ▼
[Action Dispatcher & Debounce Gate]
        │  Threshold >= 0.80, Cooldown = 500 ms
        ▼
[Hardware Actuators] (LED, GPIO, Relay, Motor Driver, LoRa, Serial)
```

---

## 4. Firmware Flashing

### Option A: PlatformIO (Recommended for CLI & CI/CD)

PlatformIO automates toolchain installation and library dependency resolution.

```bash
cd deployment/esp32/platformio

# Flash standard ESP32 Dev Module
pio run -t upload -e esp32dev

# Flash ESP32-S3
pio run -t upload -e esp32-s3-devkitc-1

# Launch serial monitor
pio device monitor -b 115200
```

### Option B: Arduino IDE

1. Open `deployment/esp32/firmware/esp32_kws.ino`.
2. Under `Tools`:
   - Board: `ESP32 Dev Module`
   - CPU Frequency: `160 MHz` (sufficient for ~7.08 ms analytical latency) or `240 MHz` (~4.72 ms analytical latency)
   - Core Debug Level: `None` (for minimal overhead)
3. Click **Upload**.

---

## 5. Performance Tuning & Optimization

### 5.1 DMA Buffer Sizing

In `esp32_kws.ino`:
```cpp
.dma_buf_count = 4,
.dma_buf_len = 512,
```
- Increasing `dma_buf_count` to 6 or 8 prevents audio dropping during heavy background Wi-Fi/Bluetooth operations at the cost of slight additional SRAM (~4 KB).
- Decreasing `dma_buf_len` to 256 decreases latency by ~16 ms but increases context-switching interrupt overhead.

### 5.2 Micro-VAD Energy Gating

To maximize battery efficiency and prevent unnecessary neural network execution during silence, the firmware incorporates a lightweight zero-crossing energy detector:
```cpp
#define SILENCE_RMS_FLOOR 150.0f
```
- If the current 1-second audio frame RMS falls below 150, the neural network inference step is bypassed entirely.
- In a quiet room, this cuts ESP32 active CPU power consumption by over **85%**.

### 5.3 Debounce Cooldown

```cpp
#define DEBOUNCE_COOLDOWN_MS 500
```
- Prevents double-triggering when an elongated utterance (e.g. *"Stoooop"*) spans multiple consecutive sliding analysis frames.

