# Basic Keyword Spotting Serial Output Example

A minimal starter sketch demonstrating I2S digital microphone input and serial monitor logging on ESP32.

---

## Hardware Setup

| INMP441 Pin | ESP32 Pin | Description |
|:---|:---|:---|
| **VDD** | **3V3** | 3.3V Power |
| **GND** | **GND** | Ground |
| **SCK** | **GPIO 26** | Bit Clock (BCLK) |
| **WS** | **GPIO 25** | Word Select (LRC) |
| **SD** | **GPIO 33** | Serial Data In |
| **L/R** | **GND** | Left channel |

---

## How to Run

1. Open `basic_kws.ino` in the Arduino IDE.
2. Select Board: `ESP32 Dev Module`.
3. Select your serial COM port.
4. Click Upload.
5. Open Serial Monitor at **115200 baud**.
6. Speak towards the INMP441 microphone and observe voice activity detections printed in real time.

