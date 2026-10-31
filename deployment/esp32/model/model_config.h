#ifndef MODEL_CONFIG_H
#define MODEL_CONFIG_H

// ==============================================================================
// ESP32 Keyword Spotting Metadata & Configuration: TinyML_DSCNN
// Generated automatically by scripts/export_esp32.py
// ==============================================================================

#define KWS_MODEL_NAME          "TinyML_DSCNN"
#define KWS_MODEL_VERSION       "1.0.0"
#define KWS_SAMPLE_RATE         16000
#define KWS_CLIP_DURATION_MS    1000
#define KWS_CLIP_SAMPLES        (KWS_SAMPLE_RATE * KWS_CLIP_DURATION_MS / 1000)

// Audio Preprocessing Constants
#define KWS_WINDOW_SIZE_MS      30
#define KWS_HOP_SIZE_MS         20
#define KWS_WINDOW_SAMPLES      (KWS_SAMPLE_RATE * KWS_WINDOW_SIZE_MS / 1000)
#define KWS_HOP_SAMPLES         (KWS_SAMPLE_RATE * KWS_HOP_SIZE_MS / 1000)
#define KWS_PREEMPHASIS_COEFF   0.97f

// Spectrogram Dimensions
#define KWS_FEATURE_TYPE        "LOG_MEL_40"
#define KWS_NUM_MEL_BINS        40
#define KWS_NUM_TIME_FRAMES     49
#define KWS_FEATURE_SIZE        (KWS_NUM_TIME_FRAMES * KWS_NUM_MEL_BINS)

// Architecture & Memory
#define KWS_TOTAL_PARAMETERS    11409
#define KWS_TOTAL_MACS          2056640
#define KWS_PEAK_ARENA_BYTES    28800

// Output Vocabulary
#define KWS_NUM_CLASSES         5
static const char* const KWS_CLASS_LABELS[KWS_NUM_CLASSES] = {
    "GO",
    "STOP",
    "LEFT",
    "RIGHT",
    "UNKNOWN",
};

// Default Runtime Thresholds
#define KWS_DEFAULT_THRESHOLD   0.80f
#define KWS_DEFAULT_DEBOUNCE_MS 500

#endif // MODEL_CONFIG_H
