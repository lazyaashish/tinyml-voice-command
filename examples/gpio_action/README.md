# Voice-Triggered GPIO & Relay Actuator Example

Demonstrates controlling hardware actuators (relays, lamps, motors, solenoids) via voice commands using ESP32.

---

## Wiring Diagram

| Actuator Pin | ESP32 Pin | Description |
|:---|:---|:---|
| **Relay IN** | **GPIO 4** | Control signal line |
| **Relay VCC** | **5V / VIN** | Relay coil power |
| **Relay GND** | **GND** | Relay ground |
| **Status LED** | **GPIO 2** | Built-in ESP32 LED indicator |

---

## Behavior

- **Say "GO"** (confidence $\ge 75\%$): GPIO 4 goes HIGH, energizing the relay and turning ON the appliance. Status LED illuminates.
- **Say "STOP"** (confidence $\ge 75\%$): GPIO 4 goes LOW, de-energizing the relay and turning OFF the appliance. Status LED turns off.
