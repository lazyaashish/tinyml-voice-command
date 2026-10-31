# Training Your Own Custom Voice Commands

Want your ESP32 to recognize custom trigger phrases like `"OPEN"`, `"CLOSE"`, `"LIGHT ON"`, `"ALARM"`, or your own name? This guide walks you through collecting audio data, training a compact model, and exporting it to C headers for your ESP32.

---

## 1. Overview of the Workflow

```text
[ 1. Record Custom Audio ] ──► [ 2. Organize Folders ] ──► [ 3. Train Model ]
                                                                   │
[ 5. Flash ESP32 Firmware ] ◄── [ 4. Export C Headers ] ◄──────────┘
```

---

## 2. Step 1: Record Custom Audio Clips

Use our built-in recording tool to capture 1-second 16 kHz audio samples directly from your computer microphone.

For reliable recognition, record **30 to 50 samples** of each keyword spoken with natural variations (different distances, speeds, pitches, and background noise):

```bash
# Record 40 samples of "OPEN"
python scripts/record_dataset.py --keyword open --num-samples 40

# Record 40 samples of "CLOSE"
python scripts/record_dataset.py --keyword close --num-samples 40

# Record 30 samples of background silence / room ambient noise
python scripts/record_dataset.py --keyword _silence_ --num-samples 30
```

> [!TIP]
> Always include a `_silence_` class containing ambient room noise (air conditioning, computer fans, typing) so the model learns not to trigger when nobody is speaking.

---

## 3. Step 2: Verify Your Dataset Structure

The recording tool automatically places WAV files in `data/custom_dataset/`:

```text
data/custom_dataset/
├── open/
│   ├── open_1730000000_001.wav
│   ├── open_1730000000_002.wav
│   └── ...
├── close/
│   ├── close_1730000000_001.wav
│   └── ...
└── _silence_/
    ├── silence_1730000000_001.wav
    └── ...
```

All audio files must be **16 kHz, 16-bit mono PCM WAV** files of approximately 1.0 second in length.

---

## 4. Step 3: Train the Neural Network

Train a `TinyML_DSCNN` model on your dataset:

```bash
python scripts/train.py \
    --data-dir data/custom_dataset \
    --model tinyml_dscnn \
    --epochs 35 \
    --batch-size 32 \
    --lr 0.003 \
    --output-checkpoint checkpoints/my_custom_model.pt
```

Training runs fast (usually under 2 minutes on CPU, under 30 seconds on GPU). You will see epoch-by-epoch loss and validation accuracy.

---

## 5. Step 4: Evaluate Model Performance

Evaluate your trained checkpoint to check precision, recall, and confusion matrix:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/my_custom_model.pt \
    --model tinyml_dscnn \
    --data-dir data/custom_dataset
```

Aim for $\ge 92\%$ validation accuracy before deploying to hardware.

---

## 6. Step 5: Export to ESP32 C Headers

Convert your trained PyTorch checkpoint into ready-to-flash C headers:

```bash
python scripts/export_esp32.py \
    --checkpoint checkpoints/my_custom_model.pt \
    --model tinyml_dscnn \
    --classes OPEN CLOSE _SILENCE_ UNKNOWN \
    --output-dir deployment/esp32/model
```

This generates:
- `deployment/esp32/model/model_data.h`: Quantized INT8 weight arrays.
- `deployment/esp32/model/model_config.h`: Preprocessing configuration and class label mappings.

---

## 7. Step 6: Update Firmware Actions & Flash

1. Open `deployment/esp32/firmware/esp32_kws.ino`.
2. Locate `onKeywordDetected()` and bind your actions to your new keywords:
   ```cpp
   void onKeywordDetected(const char* keyword, float confidence) {
       if (strcmp(keyword, "OPEN") == 0) {
           Serial.println("Action: Motor Opening Gate...");
           // Add your hardware code here
       } else if (strcmp(keyword, "CLOSE") == 0) {
           Serial.println("Action: Motor Closing Gate...");
           // Add your hardware code here
       }
   }
   ```
3. Flash to the ESP32 using Arduino IDE or PlatformIO:
   ```bash
   cd deployment/esp32/platformio
   pio run -t upload
   ```
4. Open the Serial Monitor and test your new custom voice commands!
