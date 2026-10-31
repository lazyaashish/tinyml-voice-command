/*
 * ==============================================================================
 * Voice-Controlled Mobile Robot (Differential Drive) for ESP32
 * ==============================================================================
 * Interprets 4 navigation keywords to drive a differential 2-motor robot:
 *   - "GO"    -> Forward (both motors forward)
 *   - "STOP"  -> Halt (both motors braked)
 *   - "LEFT"  -> Turn Left (left motor reverse, right motor forward)
 *   - "RIGHT" -> Turn Right (left motor forward, right motor reverse)
 *
 * Microcontroller: ESP32 / ESP32-S3
 * Motor Driver:    L298N / TB6612FNG Dual H-Bridge
 * ==============================================================================
 */

#include <Arduino.h>

// Motor Driver H-Bridge Pins
#define MOTOR_LEFT_IN1   18
#define MOTOR_LEFT_IN2   19
#define MOTOR_RIGHT_IN3  22
#define MOTOR_RIGHT_IN4  23

void setMotors(int left_speed, int right_speed) {
    // Left motor
    if (left_speed > 0) {
        digitalWrite(MOTOR_LEFT_IN1, HIGH);
        digitalWrite(MOTOR_LEFT_IN2, LOW);
    } else if (left_speed < 0) {
        digitalWrite(MOTOR_LEFT_IN1, LOW);
        digitalWrite(MOTOR_LEFT_IN2, HIGH);
    } else {
        digitalWrite(MOTOR_LEFT_IN1, LOW);
        digitalWrite(MOTOR_LEFT_IN2, LOW);
    }

    // Right motor
    if (right_speed > 0) {
        digitalWrite(MOTOR_RIGHT_IN3, HIGH);
        digitalWrite(MOTOR_RIGHT_IN4, LOW);
    } else if (right_speed < 0) {
        digitalWrite(MOTOR_RIGHT_IN3, LOW);
        digitalWrite(MOTOR_RIGHT_IN4, HIGH);
    } else {
        digitalWrite(MOTOR_RIGHT_IN3, LOW);
        digitalWrite(MOTOR_RIGHT_IN4, LOW);
    }
}

void onVoiceNavigation(const char* keyword) {
    if (strcmp(keyword, "GO") == 0) {
        Serial.println(">>> Robot: FORWARD");
        setMotors(255, 255);
    } 
    else if (strcmp(keyword, "STOP") == 0) {
        Serial.println(">>> Robot: HALT");
        setMotors(0, 0);
    } 
    else if (strcmp(keyword, "LEFT") == 0) {
        Serial.println(">>> Robot: PIVOT LEFT");
        setMotors(-200, 200);
        delay(400); // timed pivot turn
        setMotors(255, 255);
    } 
    else if (strcmp(keyword, "RIGHT") == 0) {
        Serial.println(">>> Robot: PIVOT RIGHT");
        setMotors(200, -200);
        delay(400); // timed pivot turn
        setMotors(255, 255);
    }
}

void setup() {
    Serial.begin(115200);
    pinMode(MOTOR_LEFT_IN1, OUTPUT);
    pinMode(MOTOR_LEFT_IN2, OUTPUT);
    pinMode(MOTOR_RIGHT_IN3, OUTPUT);
    pinMode(MOTOR_RIGHT_IN4, OUTPUT);

    setMotors(0, 0);

    Serial.println("\n==============================================");
    Serial.println("  ESP32 KWS: Autonomous Voice-Driven Robot    ");
    Serial.println("==============================================");
    Serial.println("Awaiting voice commands: GO, STOP, LEFT, RIGHT...\n");
}

void loop() {
    // Demonstration loop: In production, onVoiceNavigation() is called
    // directly by the real-time inference loop in deployment/esp32/firmware/esp32_kws.ino
    delay(100);
}
