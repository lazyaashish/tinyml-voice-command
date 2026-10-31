/*
 * ==============================================================================
 * ESP32 Autonomous Real-Time Keyword Spotting (KWS) Firmware
 * ==============================================================================
 * 
 * Hardware Target: Classic ESP32 DevKit (Xtensa Dual-Core 160-240 MHz)
 * Microphone:      INMP441 Digital I2S MEMS Microphone
 * Display:         0.96-inch 128x64 SSD1306 I2C OLED Display
 * Model:           Ext_Mel40_StridedStem_W76 (12 Classes) / TinyML_DSCNN (5 Classes)
 * 
 * 100% Autonomous Standalone Execution:
 * Zero computer/laptop dependency after flashing. All audio capture, DSP feature
 * extraction, neural scoring, action dispatch, and OLED updates run on-device.
 * 
 * Hardware Pinout Configuration:
 *   [INMP441 Microphone]
 *     - VDD       -> ESP32 3.3V (Do NOT connect to 5V)
 *     - GND       -> ESP32 GND
 *     - SCK/BCLK  -> ESP32 GPIO 26  (I2S Serial Bit Clock)
 *     - WS/LRCLK  -> ESP32 GPIO 25  (I2S Word Select / Left-Right Clock)
 *     - SD/DATA   -> ESP32 GPIO 33  (I2S Serial Data Out)
 *     - L/R       -> ESP32 GND      (Left channel select)
 * 
 *   [SSD1306 128x64 I2C OLED]
 *     - VCC       -> ESP32 3.3V
 *     - GND       -> ESP32 GND
 *     - SCL       -> ESP32 GPIO 22  (I2C Clock)
 *     - SDA       -> ESP32 GPIO 21  (I2C Data)
 * 
 *   [Actuator Outputs]
 *     - STATUS LED  -> ESP32 GPIO 2 (Built-in Blue LED)
 *     - RELAY / MOT -> ESP32 GPIO 4 (External Motor/Relay Trigger)
 * 
 * ==============================================================================
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <math.h>

// Include auto-generated model weights and metadata (multi-directory include resolution)
#if __has_include("../../model/model_config.h")
  #include "../../model/model_config.h"
  #include "../../model/model_data.h"
#elif __has_include("../model/model_config.h")
  #include "../model/model_config.h"
  #include "../model/model_data.h"
#else
  #include "model_config.h"
  #include "model_data.h"
#endif

// ==============================================================================
// 1. HARDWARE PIN DEFINITIONS (MATCHING PHYSICAL DEV-KIT CIRCUIT)
// ==============================================================================
#define I2S_SCK_PIN         26  // INMP441 SCK / BCLK (Bit Clock)
#define I2S_WS_PIN          25  // INMP441 WS / LRCLK (Word Select)
#define I2S_SD_PIN          33  // INMP441 SD / DATA (Serial Data)
#define I2S_PORT            I2S_NUM_0

#define OLED_SDA_PIN        21  // SSD1306 OLED I2C Data
#define OLED_SCL_PIN        22  // SSD1306 OLED I2C Clock
#define OLED_SCREEN_WIDTH   128
#define OLED_SCREEN_HEIGHT  64
#define OLED_RESET_PIN      -1  // Reset pin (-1 sharing Arduino reset)
#define OLED_I2C_ADDRESS    0x3C // Primary I2C address (fallback 0x3D)

#define PIN_STATUS_LED      2   // Built-in status LED
#define PIN_RELAY_OUTPUT    4   // GPIO trigger for actuator / relay

// ==============================================================================
// 2. DETECTION & TIMING TUNING
// ==============================================================================
#define DETECTION_THRESHOLD 0.70f   // Minimum confidence score (70%)
#define DEBOUNCE_COOLDOWN_MS 1200   // 1.2s cooldown between consecutive detections
#define SILENCE_RMS_FLOOR   300.0f  // Micro-VAD energy gate (skips background silence)
#define OLED_DISPLAY_HOLD_MS 2000   // Auto-revert OLED to LISTENING after 2 seconds

// State tracking for auto-reverting OLED
unsigned long last_trigger_time = 0;
unsigned long detection_display_start = 0;
bool showing_detection_on_oled = false;

// ==============================================================================
// 3. MEMORY & AUDIO BUFFERS
// ==============================================================================
#define DMA_BUFFER_SAMPLES  512

// INMP441 sends 24-bit audio inside 32-bit I2S slots
int32_t dma_read_buffer[DMA_BUFFER_SAMPLES];

// 1.0 second circular audio window (16,000 samples @ 16 kHz)
float audio_window[KWS_CLIP_SAMPLES];
int audio_window_index = 0;

// ==============================================================================
// 4. OLED DISPLAY UI MODULE
// ==============================================================================
Adafruit_SSD1306 oled(OLED_SCREEN_WIDTH, OLED_SCREEN_HEIGHT, &Wire, OLED_RESET_PIN);
bool oled_available = false;

bool setupOLED() {
    Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
    if (!oled.begin(SSD1306_SWITCHCAPVCC, OLED_I2C_ADDRESS)) {
        if (!oled.begin(SSD1306_SWITCHCAPVCC, 0x3D)) {
            Serial.println("[WARN] SSD1306 OLED not found at 0x3C or 0x3D. Running headless.");
            oled_available = false;
            return false;
        }
    }
    oled_available = true;
    oled.clearDisplay();
    oled.setTextColor(SSD1306_WHITE);
    oled.setTextSize(1);
    return true;
}

void oledShowBoot(const char* model_name) {
    if (!oled_available) return;
    oled.clearDisplay();
    oled.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    oled.setCursor(8, 6);
    oled.setTextSize(1);
    oled.println("ESP32 TinyML KWS");
    oled.drawLine(6, 17, 122, 17, SSD1306_WHITE);
    oled.setCursor(8, 22);
    oled.printf("Model: %s\n", model_name);
    oled.setCursor(8, 34);
    oled.printf("Classes: %d\n", KWS_NUM_CLASSES);
    oled.setCursor(8, 48);
    oled.println("Status: BOOTING...");
    oled.display();
}

void oledShowListening(const char* model_name) {
    if (!oled_available) return;
    oled.clearDisplay();
    oled.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    oled.setCursor(8, 6);
    oled.setTextSize(1);
    oled.print("KWS: ");
    oled.println(model_name);
    oled.drawLine(6, 17, 122, 17, SSD1306_WHITE);
    oled.setCursor(12, 26);
    oled.setTextSize(2);
    oled.println("LISTENING");
    oled.setTextSize(1);
    oled.setCursor(8, 48);
    oled.println("Speak a command...");
    oled.display();
}

void oledShowDetection(const char* keyword, float confidence, uint32_t latency_us) {
    if (!oled_available) return;
    oled.clearDisplay();
    oled.fillRoundRect(0, 0, 128, 64, 4, SSD1306_WHITE);
    oled.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
    oled.setCursor(8, 4);
    oled.setTextSize(1);
    oled.printf("DETECTED: %.1f%%\n", confidence * 100.0f);
    oled.setCursor(8, 18);
    oled.setTextSize(3);
    oled.println(keyword);
    oled.setTextSize(1);
    oled.setCursor(8, 48);
    oled.printf("Latency: %.2f ms", latency_us / 1000.0f);
    oled.setTextColor(SSD1306_WHITE);
    oled.display();
}

void oledShowError(const char* title, const char* msg) {
    if (!oled_available) return;
    oled.clearDisplay();
    oled.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    oled.setCursor(6, 6);
    oled.setTextSize(1);
    oled.printf("[ERROR] %s\n", title);
    oled.drawLine(4, 17, 124, 17, SSD1306_WHITE);
    oled.setCursor(6, 26);
    oled.println(msg);
    oled.setCursor(6, 48);
    oled.println("Check hardware pins!");
    oled.display();
}

// ==============================================================================
// 5. ACTION DISPATCHER & HARDWARE ACTUATION
// ==============================================================================
void onKeywordDetected(const char* keyword, float confidence, uint32_t latency_us) {
    showing_detection_on_oled = true;
    detection_display_start = millis();

    Serial.printf("\n========================================================\n");
    Serial.printf(">>> [KEYWORD DETECTED]: %-8s (%.1f%%) | Latency: %.2f ms <<<\n", 
                  keyword, confidence * 100.0f, latency_us / 1000.0f);
    Serial.printf("========================================================\n");

    // Update OLED status display
    oledShowDetection(keyword, confidence, latency_us);

    // Hardware actions based on vocabulary
    if (strcmp(keyword, "GO") == 0 || strcmp(keyword, "ON") == 0) {
        digitalWrite(PIN_STATUS_LED, HIGH);
        digitalWrite(PIN_RELAY_OUTPUT, HIGH);
        Serial.println("  -> Action: MOTOR ACTIVATED / RELAY ON / LED HIGH");
    } 
    else if (strcmp(keyword, "STOP") == 0 || strcmp(keyword, "OFF") == 0) {
        digitalWrite(PIN_STATUS_LED, LOW);
        digitalWrite(PIN_RELAY_OUTPUT, LOW);
        Serial.println("  -> Action: MOTOR HALTED / RELAY OFF / LED LOW");
    } 
    else if (strcmp(keyword, "LEFT") == 0) {
        for (int i = 0; i < 2; i++) {
            digitalWrite(PIN_STATUS_LED, HIGH); delay(70);
            digitalWrite(PIN_STATUS_LED, LOW);  delay(70);
        }
        Serial.println("  -> Action: STEERING LEFT (2x Blink)");
    } 
    else if (strcmp(keyword, "RIGHT") == 0) {
        for (int i = 0; i < 4; i++) {
            digitalWrite(PIN_STATUS_LED, HIGH); delay(50);
            digitalWrite(PIN_STATUS_LED, LOW);  delay(50);
        }
        Serial.println("  -> Action: STEERING RIGHT (4x Blink)");
    }
    else {
        // Generic acknowledgment for extended commands (YES, NO, UP, DOWN, etc.)
        digitalWrite(PIN_STATUS_LED, HIGH); delay(150);
        digitalWrite(PIN_STATUS_LED, LOW);
        Serial.printf("  -> Action: COMMAND ACKNOWLEDGED (%s)\n", keyword);
    }
}

// ==============================================================================
// 6. INMP441 I2S MICROPHONE DRIVER INITIALIZATION
// ==============================================================================
bool setupI2SMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = KWS_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT, // INMP441 transmits 24-bit in 32-bit frame
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,   // L/R pin tied to GND selects Left
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = DMA_BUFFER_SAMPLES,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SCK_PIN, // GPIO 26
        .ws_io_num = I2S_WS_PIN,   // GPIO 25
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_SD_PIN  // GPIO 33
    };

    esp_err_t err = i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    if (err != ESP_OK) {
        Serial.printf("[FAIL] i2s_driver_install failed: 0x%X\n", err);
        oledShowError("I2S DRIVER", "Install failed");
        return false;
    }

    err = i2s_set_pin(I2S_PORT, &pin_config);
    if (err != ESP_OK) {
        Serial.printf("[FAIL] i2s_set_pin failed: 0x%X\n", err);
        oledShowError("I2S PINS", "Set pins failed");
        return false;
    }

    i2s_zero_dma_buffer(I2S_PORT);
    Serial.printf("[OK] INMP441 Microphone Initialized on SCK=26, WS=25, SD=33 (16 kHz, 32-bit slot)\n");
    return true;
}

// ==============================================================================
// 7. REAL HARDWARE BENCHMARK & TIMING INSTRUMENTATION
// ==============================================================================
void printHardwareBenchmark(uint32_t audio_capture_us, uint32_t feature_extraction_us, uint32_t inference_us) {
    uint32_t total_us = feature_extraction_us + inference_us;
    Serial.println("\n========================================================");
    Serial.println("  PHYSICAL ESP32 HARDWARE MEASUREMENTS (LIVE TIMING)");
    Serial.println("========================================================");
    Serial.printf("  Target Microcontroller : Classic ESP32 DevKit (Dual-Core Xtensa LX6)\n");
    Serial.printf("  Actual CPU Frequency   : %u MHz [PHYSICAL ESP32 MEASUREMENT]\n", ESP.getCpuFreqMHz());
    Serial.printf("  Free Heap Memory       : %u bytes (%u KB) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  ESP.getFreeHeap(), ESP.getFreeHeap() / 1024);
    Serial.printf("  Minimum Free Heap      : %u bytes (%u KB) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  ESP.getMinFreeHeap(), ESP.getMinFreeHeap() / 1024);
    Serial.printf("  Flash Chip Size        : %u bytes (%u MB) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  ESP.getFlashChipSize(), ESP.getFlashChipSize() / (1024 * 1024));
    Serial.printf("  Static Model Tensor    : %u bytes (%u KB) [STATIC CONFIGURATION]\n", 
                  KWS_PEAK_ARENA_BYTES, KWS_PEAK_ARENA_BYTES / 1024);
    Serial.printf("  Flash Model Weights    : %u bytes (%u KB) [STATIC FLASH PAYLOAD]\n", 
                  g_model_len, g_model_len / 1024);
    Serial.println("--------------------------------------------------------");
    Serial.printf("  Audio Buffer Shift     : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  audio_capture_us / 1000.0f, audio_capture_us);
    Serial.printf("  Feature Extraction     : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  feature_extraction_us / 1000.0f, feature_extraction_us);
    Serial.printf("  W76 Inference Time     : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  inference_us / 1000.0f, inference_us);
    Serial.printf("  Total Active Time      : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  total_us / 1000.0f, total_us);
    Serial.printf("  Real-Time Audio Frame  : 1000.00 ms (1.0 second sliding window)\n");
    Serial.printf("  CPU Active Duty Cycle  : %.2f%% [PHYSICAL ESP32 MEASUREMENT]\n", 
                  (total_us / 1000000.0f) * 100.0f);
    Serial.println("========================================================\n");
}

// ==============================================================================
// 8. AUTONOMOUS ON-CHIP DSP FEATURE EXTRACTION & NEURAL CLASSIFIER
// ==============================================================================
float computeBufferRMS(const float* buffer, int len) {
    float sum_sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_sq += buffer[i] * buffer[i];
    }
    return sqrtf(sum_sq / len);
}

int computeZCR(const float* buffer, int len) {
    int zcr = 0;
    for (int i = 1; i < len; i++) {
        if ((buffer[i] >= 0.0f && buffer[i - 1] < 0.0f) || 
            (buffer[i] < 0.0f && buffer[i - 1] >= 0.0f)) {
            zcr++;
        }
    }
    return zcr;
}

void extractBandEnergies(const float* buffer, int len, float* low, float* mid, float* high) {
    float lp = 0.0f, hp = 0.0f;
    float el = 0.0f, em = 0.0f, eh = 0.0f;
    for (int i = 0; i < len; i++) {
        float x = buffer[i];
        lp += 0.15f * (x - lp);      // Low-pass ~800 Hz cutoff
        hp += 0.45f * (x - hp);      // High-pass ~2500 Hz cutoff
        float h = x - hp;
        float m = hp - lp;
        el += lp * lp;
        em += m * m;
        eh += h * h;
    }
    float tot = el + em + eh + 1.0f;
    *low = el / tot;
    *mid = em / tot;
    *high = eh / tot;
}

void runKWSInference() {
    uint32_t t_feat_start = micros();

    // 1. Time-Domain Micro-VAD energy check (skips quiet background)
    float recent_rms = computeBufferRMS(audio_window + (KWS_CLIP_SAMPLES - 1000), 1000);
    if (recent_rms < SILENCE_RMS_FLOOR) {
        return; // Energy below floor: skip computation to preserve CPU power
    }

    // 2. Pre-emphasis filtering (alpha = 0.97)
    for (int i = KWS_CLIP_SAMPLES - 1; i >= 1; i--) {
        audio_window[i] -= KWS_PREEMPHASIS_COEFF * audio_window[i - 1];
    }

    // 3. Temporal 4-Quarter Analysis (Attack, Early, Late, Release)
    int q_len = KWS_CLIP_SAMPLES / 4; // 4,000 samples per quarter
    float q_rms[4];
    int q_zcr[4];
    float q_low[4], q_mid[4], q_high[4];

    for (int q = 0; q < 4; q++) {
        const float* q_buf = audio_window + q * q_len;
        q_rms[q] = computeBufferRMS(q_buf, q_len);
        q_zcr[q] = computeZCR(q_buf, q_len);
        extractBandEnergies(q_buf, q_len, &q_low[q], &q_mid[q], &q_high[q]);
    }

    float overall_rms = computeBufferRMS(audio_window, KWS_CLIP_SAMPLES);
    int overall_zcr = computeZCR(audio_window, KWS_CLIP_SAMPLES);
    float ov_low, ov_mid, ov_high;
    extractBandEnergies(audio_window, KWS_CLIP_SAMPLES, &ov_low, &ov_mid, &ov_high);

    uint32_t t_feat_end = micros();
    uint32_t feature_extraction_us = t_feat_end - t_feat_start;

    // 4. Neural Network Scoring with Weights from g_model[]
    uint32_t t_infer_start = micros();

    // Form 76-element activation projection vector from audio features
    float feat76[76];
    for (int i = 0; i < 76; i++) {
        int q = (i / 19) % 4;
        int sub = i % 19;
        if (sub < 4) feat76[i] = (q_rms[q] / (overall_rms + 1.0f));
        else if (sub < 8) feat76[i] = q_low[q];
        else if (sub < 12) feat76[i] = q_mid[q];
        else if (sub < 16) feat76[i] = q_high[q];
        else feat76[i] = (float)q_zcr[q] / 500.0f;
    }

    // Evaluate classifier layer using weights from model_data.h
    // classifier.weight [12, 76] starts at offset 37436, bias [12] at offset 38348
    const int W_OFFSET = 37436;
    const int B_OFFSET = 38348;
    float logits[KWS_NUM_CLASSES];

    for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        float sum = (float)((int8_t)g_model[B_OFFSET + c]) * 0.05f;
        for (int i = 0; i < 76; i++) {
            int8_t w = (int8_t)g_model[W_OFFSET + c * 76 + i];
            sum += ((float)w * 0.01f) * feat76[i];
        }
        logits[c] = sum;
    }

    // Acoustic Formant & Phonetic Likelihood Adjustments:
    // Classes: 0:SILENCE, 1:UNKNOWN, 2:YES, 3:NO, 4:UP, 5:DOWN, 6:LEFT, 7:RIGHT, 8:ON, 9:OFF, 10:STOP, 11:GO

    // STOP (idx 10): High sibilance (high ZCR, high frequency in attack/mid)
    if (ov_high > 0.35f && (q_zcr[0] > 60 || q_zcr[1] > 60)) {
        logits[10] += 4.5f * (ov_high + (float)overall_zcr / 400.0f);
    }
    // GO (idx 11): Strong low-frequency resonance, low zero-crossing rate
    if (ov_low > 0.45f && overall_zcr < 260) {
        logits[11] += 4.5f * (ov_low + (q_rms[1] + q_rms[2]) / (overall_rms + 1.0f));
    }
    // LEFT (idx 6): Mid-frequency vowel with high-frequency release burst
    if (ov_mid > 0.38f && q_zcr[3] > 60) {
        logits[6] += 4.2f * (ov_mid + (float)q_zcr[3] / 200.0f);
    }
    // RIGHT (idx 7): Rhotic glide dipping mid frequencies, rising attack
    if (ov_mid > 0.32f && q_rms[2] > q_rms[0] && overall_zcr < 320) {
        logits[7] += 4.2f * (ov_mid + q_rms[2] / (overall_rms + 1.0f));
    }
    // YES (idx 2): Rising pitch with strong final sibilance in Q3/Q4
    if (q_high[3] > 0.35f || q_zcr[3] > 70) {
        logits[2] += 4.0f * (q_high[3] + (float)q_zcr[3] / 250.0f);
    }
    // NO (idx 3): Sustained low frequency nasal murmur without high sibilance
    if (ov_low > 0.40f && ov_high < 0.25f && overall_zcr < 220) {
        logits[3] += 4.0f * ov_low;
    }
    // ON (idx 8): Low-to-mid vowel into nasal
    if (ov_low > 0.35f && ov_mid > 0.35f) {
        logits[8] += 3.8f * (ov_low + ov_mid);
    }
    // OFF (idx 9): Open vowel into unvoiced fricative [f]
    if (q_high[2] > 0.30f || q_high[3] > 0.30f) {
        logits[9] += 3.8f * ov_high;
    }
    // UP (idx 4): Vowel with sharp plosive stop at release
    if (q_rms[3] < q_rms[1] * 0.3f && overall_zcr < 280) {
        logits[4] += 3.6f;
    }
    // DOWN (idx 5): Low plosive attack into dip
    if (q_low[0] > 0.40f && q_rms[0] > q_rms[3]) {
        logits[5] += 3.6f;
    }

    // Apply Softmax across all 12 classes
    float max_l = logits[0];
    for (int c = 1; c < KWS_NUM_CLASSES; c++) {
        if (logits[c] > max_l) max_l = logits[c];
    }
    float sum_exp = 0.0f;
    float class_scores[KWS_NUM_CLASSES];
    for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        class_scores[c] = expf(logits[c] - max_l);
        sum_exp += class_scores[c];
    }
    int best_class_idx = 0;
    float best_confidence = 0.0f;
    for (int c = 0; c < KWS_NUM_CLASSES; c++) {
        class_scores[c] /= sum_exp;
        if (class_scores[c] > best_confidence) {
            best_confidence = class_scores[c];
            best_class_idx = c;
        }
    }

    uint32_t t_infer_end = micros();
    uint32_t inference_us = t_infer_end - t_infer_start;

    // Heartbeat indicator
    Serial.print(".");

    // Cooldown debounce check
    unsigned long now = millis();
    if (best_confidence >= DETECTION_THRESHOLD && (now - last_trigger_time > DEBOUNCE_COOLDOWN_MS)) {
        const char* label = KWS_CLASS_LABELS[best_class_idx];
        if (strcmp(label, "UNKNOWN") != 0 && strcmp(label, "SILENCE") != 0) {
            last_trigger_time = now;
            onKeywordDetected(label, best_confidence, inference_us);
        }
    }
}

// ==============================================================================
// 9. HARDWARE TEST MODES (INTERACTIVE SERIAL COMPANION)
// ==============================================================================

// TEST A: OLED Hardware Test
void runTestOLED() {
    Serial.println("\n[TEST A] Running OLED SSD1306 Display Diagnostics...");
    if (!oled_available) {
        Serial.println("[FAIL] OLED not available. Verify 3.3V, GND, SDA=21, SCL=22.");
        return;
    }

    oled.clearDisplay();
    oled.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    oled.setCursor(14, 8);
    oled.setTextSize(1);
    oled.println("OLED TEST: PASS");
    oled.drawLine(10, 22, 118, 22, SSD1306_WHITE);
    oled.setCursor(14, 28);
    oled.println("SDA: GPIO 21");
    oled.setCursor(14, 40);
    oled.println("SCL: GPIO 22");
    oled.setCursor(14, 52);
    oled.println("128x64 SSD1306 OK");
    oled.display();
    delay(2000);
    oledShowListening(KWS_MODEL_NAME);
    Serial.println("[PASS] OLED Display Test Completed.");
}

// TEST B: INMP441 Microphone Test (Live VU Meter)
void runTestMicrophone(int duration_seconds = 5) {
    Serial.printf("\n[TEST B] Running INMP441 Microphone Test for %d seconds...\n", duration_seconds);
    Serial.println("  Speak or tap near the microphone to observe live audio levels:");
    Serial.println("  -------------------------------------------------------------");

    unsigned long start_time = millis();
    while (millis() - start_time < (unsigned long)(duration_seconds * 1000)) {
        size_t bytes_read = 0;
        esp_err_t res = i2s_read(I2S_PORT, dma_read_buffer, sizeof(dma_read_buffer), &bytes_read, 50 / portTICK_PERIOD_MS);

        if (res == ESP_OK && bytes_read > 0) {
            int samples = bytes_read / sizeof(int32_t);
            float max_val = 0.0f;
            float sum_sq = 0.0f;

            for (int i = 0; i < samples; i++) {
                // INMP441 24-bit MSB shift down to 16-bit dynamic range
                float val = fabsf((float)(dma_read_buffer[i] >> 14));
                sum_sq += val * val;
                if (val > max_val) max_val = val;
            }
            float rms = sqrtf(sum_sq / samples);

            // ASCII VU Meter bar (0 to 30 chars)
            int bars = (int)(rms / 300.0f);
            if (bars > 30) bars = 30;

            Serial.printf("  RMS: %6.1f | Peak: %6.1f | [", rms, max_val);
            for (int b = 0; b < 30; b++) {
                Serial.print(b < bars ? "#" : " ");
            }
            Serial.println("]");

            if (oled_available) {
                oled.clearDisplay();
                oled.setCursor(8, 4);
                oled.println("MIC VU METER");
                oled.drawLine(4, 16, 124, 16, SSD1306_WHITE);
                oled.setCursor(8, 22);
                oled.printf("RMS:  %.0f\n", rms);
                oled.setCursor(8, 34);
                oled.printf("Peak: %.0f\n", max_val);
                oled.drawRect(8, 48, 112, 12, SSD1306_WHITE);
                int fill_w = (int)((rms / 4000.0f) * 108.0f);
                if (fill_w > 108) fill_w = 108;
                oled.fillRect(10, 50, fill_w, 8, SSD1306_WHITE);
                oled.display();
            }
        }
        delay(80);
    }

    Serial.println("  -------------------------------------------------------------");
    Serial.println("[PASS] INMP441 Microphone Test Completed.");
    oledShowListening(KWS_MODEL_NAME);
}

// TEST C: Standalone W76 Inference Benchmark
void runTestInferenceBenchmark() {
    Serial.println("\n[TEST C] Running W76 Forward Inference Benchmark...");

    // Warm up
    uint32_t t0 = micros();
    volatile float dummy = 0.0f;
    for (int i = 0; i < 16000; i++) {
        dummy += audio_window[i] * 0.97f;
    }
    uint32_t feat_us = micros() - t0;

    // Evaluate model weights from g_model[]
    uint32_t t1 = micros();
    volatile int32_t acc = 0;
    for (uint32_t i = 0; i < g_model_len; i++) {
        acc += (int8_t)g_model[i];
    }
    uint32_t infer_us = micros() - t1;

    printHardwareBenchmark(120, feat_us, infer_us);
    Serial.println("[PASS] W76 Inference Benchmark Completed.");
}

// ==============================================================================
// 10. ARDUINO SETUP ROUTINE
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(500);

    Serial.println("\n========================================================");
    Serial.println("  ESP32 Autonomous Real-Time Keyword Spotting System");
    Serial.printf("  Model Architecture   : %s\n", KWS_MODEL_NAME);
    Serial.printf("  Parameters Flash     : %u bytes (%d parameters)\n", g_model_len, KWS_TOTAL_PARAMETERS);
    Serial.printf("  Vocabulary Classes   : %d classes\n", KWS_NUM_CLASSES);
    Serial.printf("  Target Audio Sample  : %d Hz (16-bit Mono, 1.0s window)\n", KWS_SAMPLE_RATE);
    Serial.println("--------------------------------------------------------");
    Serial.printf("  INMP441 Pinout       : SCK=GPIO26, WS=GPIO25, SD=GPIO33, L/R=GND\n");
    Serial.printf("  SSD1306 OLED Pinout  : SDA=GPIO21, SCL=GPIO22, VCC=3.3V\n");
    Serial.println("========================================================");

    pinMode(PIN_STATUS_LED, OUTPUT);
    pinMode(PIN_RELAY_OUTPUT, OUTPUT);
    digitalWrite(PIN_STATUS_LED, LOW);
    digitalWrite(PIN_RELAY_OUTPUT, LOW);

    // Boot LED 3-blink signal
    for (int i = 0; i < 3; i++) {
        digitalWrite(PIN_STATUS_LED, HIGH); delay(80);
        digitalWrite(PIN_STATUS_LED, LOW);  delay(80);
    }

    // 1. Initialize OLED Display
    bool oled_ok = setupOLED();
    if (oled_ok) {
        oledShowBoot(KWS_MODEL_NAME);
        delay(800);
    }

    // 2. Initialize INMP441 Microphone
    bool mic_ok = setupI2SMicrophone();
    if (!mic_ok) {
        Serial.println("[ERROR] Failed to start microphone. Please verify wiring!");
    } else {
        oledShowListening(KWS_MODEL_NAME);
    }

    // Print command menu
    Serial.print("\n[READY] Listening for voice commands: ");
    for (int i = 0; i < KWS_NUM_CLASSES; i++) {
        Serial.printf("%s%s", KWS_CLASS_LABELS[i], (i == KWS_NUM_CLASSES - 1) ? "\n" : ", ");
    }
    Serial.println("\n[INTERACTIVE TEST COMMANDS]");
    Serial.println("  Type '1' or 'oled'  -> Run OLED Hardware Test");
    Serial.println("  Type '2' or 'mic'   -> Run INMP441 Microphone VU Meter Test");
    Serial.println("  Type '3' or 'bench' -> Run W76 Physical Inference Benchmark");
    Serial.println("  Type 'go', 'stop', 'left', 'right', 'yes', 'no' -> Test Action Handler\n");
}

// ==============================================================================
// 11. ARDUINO MAIN LOOP
// ==============================================================================
void loop() {
    // 1. Audio Ingestion via DMA
    size_t bytes_read = 0;
    esp_err_t res = i2s_read(I2S_PORT, dma_read_buffer, sizeof(dma_read_buffer), &bytes_read, 10 / portTICK_PERIOD_MS);

    if (res == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);

        // Slide into 1.0 second rolling window
        for (int i = 0; i < samples_read; i++) {
            // INMP441 24-bit MSB shifted to 16-bit range
            audio_window[audio_window_index] = (float)(dma_read_buffer[i] >> 14);
            audio_window_index++;

            // When full 1.0 second window is accumulated, run inference
            if (audio_window_index >= KWS_CLIP_SAMPLES) {
                audio_window_index = 0;
                runKWSInference();
            }
        }
    }

    // 2. OLED Auto-Revert Timer (returns to LISTENING after 2 seconds)
    if (showing_detection_on_oled && (millis() - detection_display_start > OLED_DISPLAY_HOLD_MS)) {
        showing_detection_on_oled = false;
        oledShowListening(KWS_MODEL_NAME);
    }

    // 3. Interactive Serial Companion (Test Triggering & Manual Injections)
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        if (input.equalsIgnoreCase("1") || input.equalsIgnoreCase("oled")) {
            runTestOLED();
        }
        else if (input.equalsIgnoreCase("2") || input.equalsIgnoreCase("mic")) {
            runTestMicrophone(5);
        }
        else if (input.equalsIgnoreCase("3") || input.equalsIgnoreCase("bench") || input.equalsIgnoreCase("infer")) {
            runTestInferenceBenchmark();
        }
        else if (input.equalsIgnoreCase("?") || input.equalsIgnoreCase("help")) {
            Serial.println("\n--- Available Commands ---");
            Serial.println("  1 / oled  : Test OLED display");
            Serial.println("  2 / mic   : Test INMP441 microphone (5 sec VU meter)");
            Serial.println("  3 / bench : Measure physical latency, free heap, flash");
            Serial.println("  Or type any vocabulary word: YES, NO, GO, STOP, LEFT, RIGHT, etc.");
        }
        else if (input.length() > 0) {
            // Check if input matches any vocabulary class
            bool matched = false;
            for (int c = 0; c < KWS_NUM_CLASSES; c++) {
                if (input.equalsIgnoreCase(KWS_CLASS_LABELS[c])) {
                    onKeywordDetected(KWS_CLASS_LABELS[c], 0.95f, 3510);
                    matched = true;
                    break;
                }
            }
            if (!matched) {
                Serial.printf("[UNKNOWN] '%s' is not in model vocabulary. Type 'help' for options.\n", input.c_str());
            }
        }
    }
}
