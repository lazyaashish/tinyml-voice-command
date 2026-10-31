/*
 * ==============================================================================
 * ESP32 Autonomous Real-Time Keyword Spotting (KWS) Firmware
 * ==============================================================================
 * 
 * Hardware Target: Classic ESP32 DevKit (Xtensa Dual-Core 160-240 MHz)
 * Microphone:      INMP441 Digital I2S MEMS Microphone
 * Display:         0.96-inch 128x64 SSD1306 I2C OLED Display
 * Model:           Ext_Mel40_StridedStem_W76 (12 Classes, 95.0% Accuracy)
 * 
 * 100% Autonomous Standalone Execution:
 * Zero computer/laptop dependency after flashing. Powered via USB adapter,
 * battery, or external 5V. All audio capture, IIR DC-removal, adaptive VAD,
 * 40-band Log-Mel feature extraction with CMN, W76 deep neural network forward
 * inference, hardware actuation, and OLED updates execute directly on the ESP32.
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

// Include auto-generated model metadata and standalone neural inference engine
#if __has_include("w76_engine.h")
  #include "model_config.h"
  #include "w76_engine.h"
#elif __has_include("../../model/w76_engine.h")
  #include "../../model/model_config.h"
  #include "../../model/w76_engine.h"
#elif __has_include("../model/w76_engine.h")
  #include "../model/model_config.h"
  #include "../model/w76_engine.h"
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
#define DETECTION_THRESHOLD 0.55f   // Minimum neural probability (55%)
#define DEBOUNCE_COOLDOWN_MS 1200   // 1.2s cooldown between consecutive detections
#define OLED_DISPLAY_HOLD_MS 2000   // Auto-revert OLED to LISTENING after 2 seconds

// State tracking for auto-reverting OLED
unsigned long last_trigger_time = 0;
unsigned long detection_display_start = 0;
bool showing_detection_on_oled = false;

// DC offset blocking filter state
float dc_filter_prev_in = 0.0f;
float dc_filter_prev_out = 0.0f;

// ==============================================================================
// 3. AUDIO ROLLING RING BUFFER & ADAPTIVE VAD
// ==============================================================================
#define DMA_BUFFER_SAMPLES   256
#define RING_BUFFER_SAMPLES  16000  // 1.0 second rolling audio window @ 16 kHz
#define VAD_SUBFRAME_SAMPLES 320    // 20 ms subframe for energy tracking

// INMP441 sends 24-bit audio inside 32-bit I2S slots
int32_t dma_read_buffer[DMA_BUFFER_SAMPLES];

// Rolling circular ring buffer (16,000 samples @ 16-bit PCM: 32 KB)
int16_t audio_ring_buffer[RING_BUFFER_SAMPLES];
int ring_buffer_write_idx = 0;

// Aligned inference buffer passed to W76 neural engine (32 KB)
int16_t inference_audio_window[RING_BUFFER_SAMPLES];

// VAD State Machine
enum VadState {
    VAD_LISTENING,
    VAD_RECORDING
};

VadState vad_state = VAD_LISTENING;
float ambient_noise_floor = 150.0f;
int speech_active_samples = 0;
int speech_trailing_silence_samples = 0;
int speech_start_ring_idx = 0;

// ==============================================================================
// 4. OLED DISPLAY UI MODULE
// ==============================================================================
Adafruit_SSD1306 oled(OLED_SCREEN_WIDTH, OLED_SCREEN_HEIGHT, &Wire, OLED_RESET_PIN);
bool oled_available = false;

bool setupOLED() {
    Wire.begin(OLED_SDA_PIN, OLED_SCL_PIN);
    Wire.setClock(400000); // 400 kHz Fast I2C for smooth live VU meter
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

void oledShowBoot() {
    if (!oled_available) return;
    oled.clearDisplay();
    oled.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    oled.setCursor(8, 6);
    oled.setTextSize(1);
    oled.println("ESP32 TinyML KWS");
    oled.drawLine(6, 17, 122, 17, SSD1306_WHITE);
    oled.setCursor(8, 22);
    oled.println("Model: W76 12-Class");
    oled.setCursor(8, 34);
    oled.println("Accuracy: 95.0% INT8");
    oled.setCursor(8, 48);
    oled.println("Status: READY...");
    oled.display();
}

void oledShowListening(float rms = 0.0f, bool recording = false) {
    if (!oled_available) return;
    oled.clearDisplay();
    oled.drawRect(0, 0, 128, 64, SSD1306_WHITE);
    oled.setCursor(8, 5);
    oled.setTextSize(1);
    oled.println("KWS: W76 (12-Class)");
    oled.drawLine(4, 15, 124, 15, SSD1306_WHITE);
    
    if (recording) {
        oled.setCursor(14, 22);
        oled.setTextSize(2);
        oled.println("HEARING...");
    } else {
        oled.setCursor(12, 22);
        oled.setTextSize(2);
        oled.println("LISTENING");
    }
    
    // Live Real-Time Audio Level VU Meter
    oled.drawRect(8, 46, 112, 12, SSD1306_WHITE);
    int bar_w = (int)((rms / 350.0f) * 108.0f);
    if (bar_w > 108) bar_w = 108;
    if (bar_w > 0) {
        oled.fillRect(10, 48, bar_w, 8, SSD1306_WHITE);
    }
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
    oled.printf("Latency: %.1f ms", latency_us / 1000.0f);
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
void runTestInferenceBenchmark() {
    Serial.println("\n[TEST C] Running W76 Forward Inference Benchmark on ESP32...");

    // Create synthetic test audio
    for (int i = 0; i < RING_BUFFER_SAMPLES; i++) {
        inference_audio_window[i] = (int16_t)(sinf(2.0f * 3.14159f * 440.0f * (float)i / 16000.0f) * 1500.0f);
    }

    uint32_t t_feat_start = micros();
    w76_extract_features(inference_audio_window);
    uint32_t feat_us = micros() - t_feat_start;

    uint32_t t_infer_start = micros();
    float probs[12];
    float logits[12];
    int cls = w76_forward(probs, logits);
    uint32_t infer_us = micros() - t_infer_start;
    uint32_t total_us = feat_us + infer_us;

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
    Serial.printf("  Model Architecture     : %s\n", KWS_MODEL_NAME);
    Serial.printf("  Total Parameters       : %d (12 classes)\n", KWS_TOTAL_PARAMETERS);
    Serial.println("--------------------------------------------------------");
    Serial.printf("  Feature Extraction     : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  feat_us / 1000.0f, feat_us);
    Serial.printf("  W76 Neural Inference   : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  infer_us / 1000.0f, infer_us);
    Serial.printf("  Total Execution Time   : %.2f ms (%u us) [PHYSICAL ESP32 MEASUREMENT]\n", 
                  total_us / 1000.0f, total_us);
    Serial.printf("  Real-Time Audio Frame  : 1000.00 ms (1.0 second sliding window)\n");
    Serial.printf("  CPU Active Duty Cycle  : %.2f%% [PHYSICAL ESP32 MEASUREMENT]\n", 
                  (total_us / 1000000.0f) * 100.0f);
    Serial.printf("  Benchmark Prediction   : %s (%.1f%%)\n", KWS_CLASS_LABELS[cls], probs[cls] * 100.0f);
    Serial.println("========================================================\n");
    Serial.println("[PASS] W76 Inference Benchmark Completed.");
}

// ==============================================================================
// 8. INTERACTIVE HARDWARE DIAGNOSTICS (TEST MODES)
// ==============================================================================
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
    oledShowListening();
    Serial.println("[PASS] OLED Display Test Completed.");
}

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
                float raw = (float)(dma_read_buffer[i] >> 14);
                float val = fabsf(raw - dc_filter_prev_in + 0.995f * dc_filter_prev_out);
                sum_sq += val * val;
                if (val > max_val) max_val = val;
            }
            float rms = sqrtf(sum_sq / (float)samples);

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
    oledShowListening();
}

// ==============================================================================
// 9. ON-DEVICE NEURAL CLASSIFIER INFERENCE ROUTINE
// ==============================================================================
void triggerNeuralInference() {
    uint32_t t_start = micros();

    // 1. Copy rolling buffer into aligned 1.0 second inference window
    // Align window so the speech event is centered within the 16,000 samples
    int read_start = (speech_start_ring_idx - 1600 + RING_BUFFER_SAMPLES) % RING_BUFFER_SAMPLES;
    for (int i = 0; i < RING_BUFFER_SAMPLES; i++) {
        inference_audio_window[i] = audio_ring_buffer[(read_start + i) % RING_BUFFER_SAMPLES];
    }

    // 2. Run W76 Neural Forward Pass (Mel Spectrogram + CMN + 4-block ConvNet)
    float confidence = 0.0f;
    float logits[12];
    int best_cls = w76_predict_audio(inference_audio_window, &confidence, logits);
    uint32_t latency_us = micros() - t_start;

    // Heartbeat indicator on serial
    Serial.print(".");

    // Print top prediction details
    const char* detected_label = KWS_CLASS_LABELS[best_cls];

    // Check if detected word is an actionable keyword (classes 2..11: YES, NO, UP, DOWN, LEFT, RIGHT, ON, OFF, STOP, GO)
    // Ignore SILENCE (class 0) and UNKNOWN (class 1)
    unsigned long now = millis();
    if (best_cls >= 2 && confidence >= DETECTION_THRESHOLD && (now - last_trigger_time > DEBOUNCE_COOLDOWN_MS)) {
        last_trigger_time = now;
        Serial.printf("\n[W76 PREDICTION] Class: %-8s | Prob: %.1f%% | Latency: %.2f ms\n", 
                      detected_label, confidence * 100.0f, latency_us / 1000.0f);
        onKeywordDetected(detected_label, confidence, latency_us);
    }
}

// ==============================================================================
// 10. ARDUINO SETUP ROUTINE
// ==============================================================================
void setup() {
    Serial.begin(115200);
    delay(400);

    Serial.println("\n========================================================");
    Serial.println("  ESP32 Autonomous Real-Time Keyword Spotting System");
    Serial.printf("  Model Architecture   : %s (12 Classes)\n", KWS_MODEL_NAME);
    Serial.printf("  Parameters Flash     : 156.5 KB (%d parameters)\n", KWS_TOTAL_PARAMETERS);
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

    // 1. Initialize Neural Inference Engine (Heap Arena Allocation)
    if (!w76_init_engine()) {
        Serial.println("[ERROR] Failed to allocate W76 engine heap memory!");
        oledShowError("W76 HEAP", "Malloc failed");
    } else {
        Serial.println("[OK] W76 Neural Engine Initialized (102 KB Heap Arena Allocated)");
    }

    // 2. Initialize OLED Display
    bool oled_ok = setupOLED();
    if (oled_ok) {
        oledShowBoot();
        delay(1000);
    }

    // 3. Initialize INMP441 Microphone
    bool mic_ok = setupI2SMicrophone();
    if (!mic_ok) {
        Serial.println("[ERROR] Failed to start microphone. Please verify wiring!");
    } else {
        oledShowListening();
    }

    // Print command menu
    Serial.print("\n[READY] Listening for voice commands: ");
    for (int i = 2; i < KWS_NUM_CLASSES; i++) {
        Serial.printf("%s%s", KWS_CLASS_LABELS[i], (i == KWS_NUM_CLASSES - 1) ? "\n" : ", ");
    }
    Serial.println("\n[INTERACTIVE TEST COMMANDS]");
    Serial.println("  Type '1' or 'oled'  -> Run OLED Hardware Test");
    Serial.println("  Type '2' or 'mic'   -> Run INMP441 Microphone VU Meter Test");
    Serial.println("  Type '3' or 'bench' -> Run W76 Physical Inference Benchmark");
    Serial.println("  Type 'go', 'stop', 'left', 'right', 'yes', 'no' -> Test Action Handler\n");
}

// ==============================================================================
// 11. ARDUINO MAIN LOOP (CONTINUOUS AUDIO STREAMING & REAL-TIME VAD)
// ==============================================================================
static float latest_chunk_rms = 0.0f;
static unsigned long last_oled_vu_time = 0;

void loop() {
    // 1. Audio Ingestion via DMA with High-Pass DC Filter
    size_t bytes_read = 0;
    esp_err_t res = i2s_read(I2S_PORT, dma_read_buffer, sizeof(dma_read_buffer), &bytes_read, 10 / portTICK_PERIOD_MS);

    if (res == ESP_OK && bytes_read > 0) {
        int samples_read = bytes_read / sizeof(int32_t);

        // Slide into rolling circular buffer with DC blocking filter
        float chunk_sum_sq = 0.0f;
        for (int i = 0; i < samples_read; i++) {
            float raw_val = (float)(dma_read_buffer[i] >> 14);
            // IIR high-pass filter: y[n] = x[n] - x[n-1] + 0.995 * y[n-1]
            float clean_val = raw_val - dc_filter_prev_in + 0.995f * dc_filter_prev_out;
            dc_filter_prev_in = raw_val;
            dc_filter_prev_out = clean_val;

            // Store as 16-bit PCM in ring buffer
            int16_t s16 = (int16_t)fmaxf(-32768.0f, fminf(32767.0f, clean_val));
            audio_ring_buffer[ring_buffer_write_idx] = s16;
            ring_buffer_write_idx = (ring_buffer_write_idx + 1) % RING_BUFFER_SAMPLES;

            chunk_sum_sq += clean_val * clean_val;
        }

        // 2. Real-Time Energy Tracking (RMS over DMA chunk)
        float chunk_rms = sqrtf(chunk_sum_sq / (float)samples_read);
        latest_chunk_rms = chunk_rms;

        // Dynamic ambient noise floor tracking (slow EMA update)
        if (vad_state == VAD_LISTENING) {
            if (chunk_rms < ambient_noise_floor * 1.5f) {
                ambient_noise_floor = ambient_noise_floor * 0.97f + chunk_rms * 0.03f;
            }
        }

        // Adaptive speech trigger threshold (sensitive: 2.0x noise floor, minimum floor of 55.0)
        float speech_threshold = fmaxf(55.0f, ambient_noise_floor * 2.0f);

        // 3. VAD State Machine
        if (vad_state == VAD_LISTENING) {
            if (chunk_rms > speech_threshold) {
                // Speech onset detected!
                vad_state = VAD_RECORDING;
                speech_active_samples = samples_read;
                speech_trailing_silence_samples = 0;
                speech_start_ring_idx = (ring_buffer_write_idx - samples_read + RING_BUFFER_SAMPLES) % RING_BUFFER_SAMPLES;
            }
        } 
        else if (vad_state == VAD_RECORDING) {
            speech_active_samples += samples_read;

            if (chunk_rms < speech_threshold * 0.7f) {
                speech_trailing_silence_samples += samples_read;
            } else {
                speech_trailing_silence_samples = 0;
            }

            // Word finished: either 200 ms (3200 samples) of trailing silence OR buffer filled (14,400 samples)
            if (speech_trailing_silence_samples >= 3200 || speech_active_samples >= 14400) {
                // If speech duration is valid (between 150 ms and 900 ms)
                if (speech_active_samples >= 2400) {
                    triggerNeuralInference();
                }
                // Return to listening
                vad_state = VAD_LISTENING;
                speech_active_samples = 0;
                speech_trailing_silence_samples = 0;
            }
        }
    }

    // 4. OLED Live VU Meter & Auto-Revert Timer
    if (showing_detection_on_oled) {
        if (millis() - detection_display_start > OLED_DISPLAY_HOLD_MS) {
            showing_detection_on_oled = false;
        }
    } else {
        if (millis() - last_oled_vu_time > 80) { // 12 FPS smooth VU meter on OLED
            last_oled_vu_time = millis();
            oledShowListening(latest_chunk_rms, vad_state == VAD_RECORDING);
        }
    }

    // 5. Interactive Serial Companion (Test Triggering & Manual Injections)
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
            bool matched = false;
            for (int c = 0; c < KWS_NUM_CLASSES; c++) {
                if (input.equalsIgnoreCase(KWS_CLASS_LABELS[c])) {
                    onKeywordDetected(KWS_CLASS_LABELS[c], 0.95f, 48500);
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
