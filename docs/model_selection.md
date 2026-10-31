# Model Selection Guide: Choosing the Right KWS Architecture

This guide breaks down the trade-offs between parameter counts, peak SRAM usage, inference latency, vocabulary size, and scientific accuracy across all evaluated Keyword Spotting (KWS) models.

---

## 1. Quick Decision Tree

```text
What is your target application?
  │
  ├─► Mobile Robot / Hardware Voice Navigation (5 Commands)
  │    └─► Select: ⭐ TinyML_DSCNN (Navigation Champion)
  │         • Vocabulary: 5 classes (GO, STOP, LEFT, RIGHT, UNKNOWN)
  │         • Parameters: 11,409 (11.14 KB INT8)
  │         • Operations: 2.06M MACs (Fastest computation)
  │         • Accuracy: 91.66% Held-Out Test (2,027 clips) | 96.40% STOP Confidence
  │         • Peak Arena: 28.125 KB (8.8% of ESP32 SRAM)
  │         • ESP32 Latency: 7.08 ms @ 160 MHz (Analytical Cycle Estimate)
  │
  ├─► Full Vocabulary Voice Control with Maximum Accuracy (12 Commands, ≥ 94% Accuracy)
  │    └─► Select: 🏆 Ext_Mel40_StridedStem_W76 (12-Class 95% Champion)
  │         • Vocabulary: 12 classes (Pete Warden Google Speech Commands v0.02)
  │         • Parameters: 38,360 (37.46 KB INT8)
  │         • Operations: 3.99M MACs
  │         • Accuracy: 94.99% (~95.0%) True INT8 Test (4,890 untouched clips)
  │         • Peak Arena: 20.000 KB (Fits easily in ESP32 320 KB and 25KB MCU limits)
  │         • ESP32 Latency: 13.72 ms @ 160 MHz (Analytical Cycle Estimate)
  │
  ├─► Constrained to ≤ 16 KB SRAM (e.g. ARM Cortex-M0+ / Cortex-M4 / STM32F4)
  │    └─► Select: ⚡ CandF_44ChDualResidual (16KB Champion)
  │         • Vocabulary: 12 classes
  │         • Parameters: 16,200 (15.82 KB INT8)
  │         • Operations: 1.41M MACs
  │         • Accuracy: 92.54% FP32 / 91.62% INT8 QAT (4,890 test clips)
  │         • Peak Arena: 14.703 KB (Guaranteed ≤ 16 KB SRAM!)
  │         • ESP32 Latency: 4.85 ms @ 160 MHz (Analytical Cycle Estimate)
  │
  └─► Academic Baseline Comparison
       └─► Select: 📖 ArmSmallDSCNN (Zhang et al. 2017 Reproduction)
            • Vocabulary: 12 classes
            • Parameters: 23,180
            • Accuracy: 94.40% Published / 93.95% Modern Recipe
            • Peak Arena: 46.875 KB (Single-buffer)
```

---

## 2. Complete Architecture Comparison Table

| Architecture | Classes | Vocabulary | Parameters | Operations (MACs) | INT8 Model Size | Peak SRAM Arena | ESP32 Latency @ 160MHz | Accuracy & Split | Target Hardware & Use Case |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|:---|
| ⭐ **`TinyML_DSCNN`** | 5 | `GO`, `STOP`, `LEFT`, `RIGHT`, `UNKNOWN` | **11,409** | **2,056,640** (~2.06M) | **11.14 KB** | 28.125 KB *(Ping-Pong Arena)* | **7.08 ms** *(Estimated)* | **91.66%** *(Held-Out Test, 2,027 clips)*<br>96.40% *(Avg Stop Conf, 40 clips)* | **ESP32 Robotics & Voice Navigation** |
| 🏆 **`Ext_Mel40_StridedStem_W76`** | 12 | `YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN` | **38,360** | **3,986,688** (~3.99M) | **37.46 KB** | **20.000 KB** *(Peak In-Out Buffer)* | **13.72 ms** *(Estimated)* | **94.99% (~95.0%)** *(True INT8 Test, 4,890 clips)*<br>95.41% *(QAT Validation)* | **High-Accuracy 12-Class KWS on ESP32 & MCUs** |
| ⚡ **`CandF_44ChDualResidual`** | 12 | `YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN` | **16,200** | **1,406,672** (~1.41M) | **15.82 KB** | **14.703 KB** *(Guaranteed ≤ 16 KB)* | **4.85 ms** *(Estimated)* | **92.54%** FP32 / **91.62%** INT8 *(4,890 test clips)* | **Ultra-low SRAM MCUs (ARM Cortex-M0+/M4)** |
| 📖 **`ArmSmallDSCNN`** | 12 | `YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN` | **23,180** | **5,400,000** (~5.40M) | **22.64 KB** | 46.875 KB *(Single-buffer)* | **18.60 ms** *(Estimated)* | **94.40%** Published / **93.95%** Modern Recipe | Academic Baseline Reproduction |

---

## 3. Evidence-Based Audit: De-mystifying the Metrics

### Why Was 96.40% Cited for `TinyML_DSCNN`?
In early technical summaries, **96.40%** was cited as the accuracy for `TinyML_DSCNN`. A forensic investigation of the original laboratory report (`references/sih_report_text.txt`, Table 4) reveals the provenance:
- **Table 4 (40-clip physical test set)**:
  - `STOP`: 10/10 correct (**100.0%** accuracy, **96.4%** average confidence score)
  - `RIGHT`: 10/10 correct (100.0% accuracy, 94.0% average confidence score)
  - `LEFT`: 9/10 correct (90.0% accuracy, 91.4% average confidence score)
  - `GO`: 8/10 correct (80.0% accuracy, 88.7% average confidence score)
- The **96.4% confidence score** for the `STOP` keyword was copied into summary tables and propagated as the model's overall test accuracy.
- **The true test-set performance**:
  - On the official **held-out test split of 2,027 untouched clips**, `TinyML_DSCNN` achieves **91.66% test accuracy** (with SpecAugment and pitch warping) and **89.20%** without augmentation.
  - In `train_edge_model.ipynb`, validation accuracy on an 85/15 random split reached **94.39%**.

### Why Are ESP32 Latency Numbers Labeled as "Analytical Cycle Estimates"?
The 7.08 ms latency figure originates from `proposed_progressive_5_class_mel40_cmn_eval.json`:
$$\text{Latency} = \frac{1,132,320\text{ cycles}}{160\text{ MHz}} = 7.077\text{ ms} \approx 7.08\text{ ms}$$
$$\text{Latency @ 240 MHz} = \frac{1,132,320\text{ cycles}}{240\text{ MHz}} = 4.718\text{ ms} \approx 4.72\text{ ms}$$
This is an **analytical clock cycle estimate** derived from the layer-by-layer multiply-accumulate profile and FFT cycle requirements, rather than a physical timer probe.

---

## 4. Comparing Problem Complexities: 5-Class vs 12-Class

It is scientifically invalid to compare raw accuracy percentages across different tasks:
- **5-Class Task** (`GO`, `STOP`, `LEFT`, `RIGHT`, `UNKNOWN`):
  - Random guessing baseline is **20.0%**.
  - Words were chosen specifically for acoustic distinctiveness (e.g. sibilant fricative in *"stop"*, liquid consonant in *"left"*, rhotic glide in *"right"*).
- **12-Class Task** (`YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN`):
  - Random guessing baseline is **8.33%**.
  - Contains acoustically confusable phoneme pairs (*"no"* vs *"go"*, *"up"* vs *"off"*).
  - Achieving **94.99% on 12 classes** (`Ext_Mel40_StridedStem_W76`) represents a significantly more powerful acoustic classification capability than 91.66% on 5 classes.
