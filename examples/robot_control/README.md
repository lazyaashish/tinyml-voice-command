# Voice-Controlled Robot Example (Differential Drive)

Control a mobile robot base using spoken commands (`GO`, `STOP`, `LEFT`, `RIGHT`) processed locally on an ESP32.

---

## Hardware Connection

| Driver Pin (L298N) | ESP32 Pin | Function |
|:---|:---|:---|
| **IN1** | **GPIO 18** | Left Motor Forward |
| **IN2** | **GPIO 19** | Left Motor Reverse |
| **IN3** | **GPIO 22** | Right Motor Forward |
| **IN4** | **GPIO 23** | Right Motor Reverse |
| **VCC** | **External 7-12V Battery** | Motor Power |
| **GND** | **ESP32 GND & Battery GND** | Common Ground |

---

## Spoken Commands & Motions

- **"GO"**: Drives forward at full speed.
- **"STOP"**: Immediately stops both wheels.
- **"LEFT"**: Pivots 90 degrees counter-clockwise and resumes.
- **"RIGHT"**: Pivots 90 degrees clockwise and resumes.
