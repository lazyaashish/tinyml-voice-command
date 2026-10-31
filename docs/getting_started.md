# Getting Started: ESP32 Voice Recognition in 10 Minutes

Welcome! This guide is written for anyone—even if you have never worked with machine learning, neural networks, or TinyML before. By following these 10 simple steps, you will have your ESP32 listening for voice commands and triggering real hardware actions.

---

## 1. What You Need (Hardware Shopping List)

You only need two inexpensive parts:

1. **ESP32 Development Board** (~$3–$5)
   - Any standard 30-pin or 38-pin ESP32 board (e.g. ESP32-WROOM-32, NodeMCU-32S, ESP32 DevKit v1, or ESP32-S3).
2. **INMP441 I2S Digital MEMS Microphone Module** (~$2–$3)
   - Do **not** use an analog 3.5mm mic module or analog sound sensor with a potentiometer. You need a digital I2S microphone (INMP441, SPH0645, or ICS-43434).
3. **5 Female-to-Female Jumper Wires**
4. **Micro-USB (or USB-C) cable** to connect ESP32 to your computer.

---

## 2. Wire the Microphone to the ESP32

Connect the 5 pins of your INMP441 microphone to the ESP32 pins using your jumper wires:

| INMP441 Mic Pin | ESP32 Board Pin | What It Does |
|:---|:---|:---|
| **VDD / VCC** | **3V3** (3.3 Volts) | Powers the microphone. (*Never connect to 5V!*) |
| **GND** | **GND** (Ground) | Electrical common ground. |
| **SD** | **GPIO 32** | Sends recorded digital audio bytes to ESP32. |
| **WS / LRC** | **GPIO 15** | Left/Right word clock signal. |
| **SCK / BCLK** | **GPIO 14** | High-speed bit clock signal. |
| **L/R** | **GND** | Forces audio to Left channel. (*Tie directly to GND!*) |

> [!TIP]
> Tying **L/R to GND** is very important. If left unconnected (floating), your audio may have screeching static or random silence.

---

## 3. Clone or Download This Repository

Open your terminal or command prompt and clone the repository:

```bash
git clone https://github.com/your-username/sih-esp32-kws.git
cd sih-esp32-kws
```

---

## 4. Install Python Dependencies (If Modifying Models)

If you want to run Python scripts or train your own models:

```bash
pip install -r requirements.txt
```

Verify your environment with one simple command:

```bash
python scripts/validate_installation.py
```

You should see `[SUCCESS] All system validation checks passed!`.

---

## 5. Install the Arduino IDE

1. Download and install [Arduino IDE 2.x](https://www.arduino.cc/en/software).
2. Open Arduino IDE and go to **File -> Preferences**.
3. In the **Additional boards manager URLs** field, paste:
   ```text
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Click **OK**.
5. Go to **Tools -> Board -> Boards Manager**, search for `esp32`, and click **Install**.

---

## 6. Open the Firmware Sketch

In Arduino IDE:
1. Click **File -> Open**.
2. Select `deployment/esp32/firmware/esp32_kws.ino`.
3. Notice that `model_config.h` and `model_data.h` in `../model/` are automatically included.

---

## 7. Select Your Board and Port

Under the **Tools** menu in Arduino IDE:
- **Board**: Select `ESP32 Dev Module` (or your specific ESP32 variant).
- **CPU Frequency**: `160 MHz` or `240 MHz`.
- **Upload Speed**: `921600` (or `115200` if upload fails).
- **Port**: Select the COM port connected to your ESP32.

---

## 8. Flash the Code to the ESP32

1. Click the **Upload** button (the right arrow in the top left).
2. The Arduino IDE will compile the C++ firmware, bundle the pre-quantized INT8 model weights, and flash the ESP32.
3. *If you see `Connecting........_____.....` in the console, press and hold the physical **BOOT** button on your ESP32 board for 2 seconds until the upload starts, then release it.*

---

## 9. Open the Serial Monitor and Test

1. Click the **Serial Monitor** icon (top right corner of Arduino IDE).
2. Set the baud rate dropdown in the bottom right corner of the Serial Monitor to **115200 baud**.
3. Press the **EN / RESET** button on your ESP32.
4. You will see:
   ```text
   ==================================================
     ESP32 Autonomous Keyword Spotting System
     Model: TinyML_DSCNN (11409 params, 5 classes)
     Audio: 16 kHz Log-Mel 40 (49 frames)
   ==================================================
   [INFO] I2S Microphone Driver Initialized on Port 0
   [INFO] Listening for keywords: GO, STOP, LEFT, RIGHT...
   ```
5. Speak clearly into the microphone:
   - Say: **"GO"** -> The onboard LED (GPIO 2) turns ON!
   - Say: **"STOP"** -> The onboard LED turns OFF!
   - Say: **"LEFT"** -> The onboard LED blinks twice!
   - Say: **"RIGHT"** -> The onboard LED blinks four times!

---

## 10. Next Steps: Making It Your Own

Now that your ESP32 is recognizing speech:
- **Wire a Relay or Motor**: Check out [examples/gpio_action/](../examples/gpio_action/README.md) and [examples/robot_control/](../examples/robot_control/README.md).
- **Train Custom Words**: Want the ESP32 to respond to `"LIGHTS ON"`, `"ALARM"`, or your own name? Follow the [Custom Keywords Training Guide](train_custom_keywords.md).
- **Learn the Technology**: Curious how an 11 KB neural network runs on a microcontroller without the internet? Read [TinyML Basics Explained](tinyml_basics.md).
