# Basic Keyword Spotting Serial Output Example

A minimal starter sketch demonstrating I2S digital microphone input and serial monitor logging on ESP32.

---

## Hardware Setup

| INMP441 Pin | ESP32 Pin | Description |
|:---|:---|:---|
| **VDD** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SD** | **GPIO 32** | Serial Data In |
| **WS** | **GPIO 15** | Word Select (LRC) |
| **SCK** | **GPIO 14** | Bit Clock (BCLK) |
| **L/R** | **GND** | Left channel |

---

## How to Run

1. Open `basic_kws.ino` in the Arduino IDE.
2. Select Board: `ESP32 Dev Module`.
3. Select your serial COM port.
4. Click Upload.
5. Open Serial Monitor at **115200 baud**.
6. Speak towards the INMP441 microphone and observe voice activity detections printed in real time.

