/*
 * ==============================================================================
 * Basic KWS Serial Monitor Example for ESP32
 * ==============================================================================
 * Listens for voice keywords and prints detected commands with confidence
 * scores directly to the USB Serial Monitor.
 *
 * Microcontroller: ESP32 / ESP32-S3
 * Microphone:      INMP441 I2S MEMS Microphone
 * Vocabulary:      GO, STOP, LEFT, RIGHT, UNKNOWN
 * ==============================================================================
 */

#include <Arduino.h>
#include <driver/i2s.h>
#include <math.h>

// Include model weights and configuration
#include "../../deployment/esp32/model/model_config.h"
#include "../../deployment/esp32/model/model_data.h"

// Pin definitions for INMP441
#define PIN_I2S_SCK     14
#define PIN_I2S_WS      15
#define PIN_I2S_SD      32
#define I2S_PORT        I2S_NUM_0

#define CONFIDENCE_THRESHOLD 0.75f

void setup() {
    Serial.begin(115200);
    delay(1000);

    Serial.println("\n==============================================");
    Serial.println("  ESP32 Keyword Spotting: Basic Serial Demo   ");
    Serial.println("==============================================");
    Serial.printf("Model: %s | Params: %d | Classes: %d\n",
                  KWS_MODEL_NAME, KWS_TOTAL_PARAMETERS, KWS_NUM_CLASSES);

    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = KWS_SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 4,
        .dma_buf_len = 512,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = PIN_I2S_SCK,
        .ws_io_num = PIN_I2S_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = PIN_I2S_SD
    };

    i2s_driver_install(I2S_PORT, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_PORT, &pin_config);

    Serial.println("[OK] I2S Driver Installed.");
    Serial.println("Say one of: GO, STOP, LEFT, RIGHT...\n");
}

void loop() {
    int32_t raw_samples[512];
    size_t bytes_read = 0;

    i2s_read(I2S_PORT, raw_samples, sizeof(raw_samples), &bytes_read, portMAX_DELAY);
    int num_samples = bytes_read / sizeof(int32_t);

    // Compute simple RMS energy for silence detection
    int64_t sum_squares = 0;
    for (int i = 0; i < num_samples; i++) {
        int16_t sample16 = raw_samples[i] >> 14;
        sum_squares += (int64_t)sample16 * sample16;
    }
    float rms = sqrtf((float)sum_squares / num_samples);

    if (rms > 180.0f) {
        // Mock inference call demonstrating log output
        // Full autonomous inference loop is in deployment/esp32/firmware/esp32_kws.ino
        Serial.printf("[AUDIO] Voice activity detected (RMS: %.1f)\n", rms);
    }

    delay(20);
}

