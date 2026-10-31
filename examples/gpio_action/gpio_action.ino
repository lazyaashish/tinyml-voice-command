/*
 * ==============================================================================
 * Voice-Triggered GPIO & Relay Switch Example for ESP32
 * ==============================================================================
 * Triggers onboard LED and external relay module based on voice commands:
 *   - "GO"   -> Turns LED & Relay ON
 *   - "STOP" -> Turns LED & Relay OFF
 *
 * Microcontroller: ESP32 / ESP32-S3
 * Output Pins:     GPIO 2 (LED), GPIO 4 (Relay)
 * ==============================================================================
 */

#include <Arduino.h>

#define PIN_LED      2   // Built-in LED
#define PIN_RELAY    4   // Active-HIGH Relay Module

void setup() {
    Serial.begin(115200);
    pinMode(PIN_LED, OUTPUT);
    pinMode(PIN_RELAY, OUTPUT);

    // Initial state: OFF
    digitalWrite(PIN_LED, LOW);
    digitalWrite(PIN_RELAY, LOW);

    Serial.println("\n==============================================");
    Serial.println("  ESP32 KWS: GPIO / Relay Voice Controller    ");
    Serial.println("==============================================");
    Serial.println("Commands:");
    Serial.println("  'GO'   -> Turn Relay ON");
    Serial.println("  'STOP' -> Turn Relay OFF");
    Serial.println("System Ready.\n");
}

void handleVoiceCommand(const char* command, float confidence) {
    if (confidence < 0.75f) return;

    Serial.printf("[COMMAND RECEIVED]: %s (Confidence: %.1f%%)\n", command, confidence * 100.0f);

    if (strcmp(command, "GO") == 0) {
        digitalWrite(PIN_LED, HIGH);
        digitalWrite(PIN_RELAY, HIGH);
        Serial.println(">>> Action Executed: RELAY ACTIVATED (ON)");
    } 
    else if (strcmp(command, "STOP") == 0) {
        digitalWrite(PIN_LED, LOW);
        digitalWrite(PIN_RELAY, LOW);
        Serial.println(">>> Action Executed: RELAY DEACTIVATED (OFF)");
    }
}

void loop() {
    // Demonstration loop: In production, handleVoiceCommand() is called directly
    // from the audio inference pipeline in deployment/esp32/firmware/esp32_kws.ino
    delay(100);
}

