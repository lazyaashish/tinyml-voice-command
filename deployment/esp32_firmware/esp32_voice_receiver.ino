/*
 * ESP32 Voice AI Command Receiver (Universal Pin Driver)
 * Listens on Serial (115200 baud) for voice commands from voice_demo.py
 * Commands: "go", "stop", "left", "right", "unknown"
 */

// Common ESP32 GPIO pins used for built-in LEDs or external LEDs/Motors
const int LED_PINS[] = {2, 4, 5, 12, 13, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 27};
const int NUM_PINS = sizeof(LED_PINS) / sizeof(LED_PINS[0]);

void setAllLEDs(bool state) {
  for (int i = 0; i < NUM_PINS; i++) {
    digitalWrite(LED_PINS[i], state ? HIGH : LOW);
  }
}

void blinkAllLEDs(int count, int delayMs) {
  for (int c = 0; c < count; c++) {
    setAllLEDs(true);
    delay(delayMs);
    setAllLEDs(false);
    delay(delayMs);
  }
}

void setup() {
  Serial.begin(115200);

  // Initialize all candidate LED output pins
  for (int i = 0; i < NUM_PINS; i++) {
    pinMode(LED_PINS[i], OUTPUT);
    digitalWrite(LED_PINS[i], LOW);
  }

  // Diagnostic boot blink: 3 quick blinks across all pins
  blinkAllLEDs(3, 120);

  Serial.println("ESP32 Edge Voice Activator Ready [115200 Baud]");
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd.length() == 0) return;

    if (cmd.equalsIgnoreCase("go")) {
      setAllLEDs(true);  // Turn controllable LEDs ON
      Serial.println("GO RECEIVED - MOTOR FORWARD (LED ON)");
    } 
    else if (cmd.equalsIgnoreCase("stop")) {
      setAllLEDs(false); // Turn controllable LEDs OFF
      Serial.println("STOP RECEIVED - MOTOR HALTED (LED OFF)");
    } 
    else if (cmd.equalsIgnoreCase("left")) {
      blinkAllLEDs(2, 90);
      Serial.println("LEFT RECEIVED - STEERING LEFT (BLINK 2X)");
    } 
    else if (cmd.equalsIgnoreCase("right")) {
      blinkAllLEDs(4, 70);
      Serial.println("RIGHT RECEIVED - STEERING RIGHT (BLINK 4X)");
    } 
    else if (cmd.equalsIgnoreCase("unknown")) {
      Serial.println("UNKNOWN RECEIVED - STANDBY");
    }
  }
}


