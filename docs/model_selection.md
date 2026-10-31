# Model Selection Guide: Choosing the Right KWS Architecture

This guide breaks down the trade-offs between parameter counts, peak SRAM usage, inference latency, and vocabulary size across our available Keyword Spotting (KWS) models.

---

## 1. Quick Decision Tree

```text
Do you want an out-of-the-box voice control system on ESP32?
  │
  ├─► YES (Navigation commands: GO, STOP, LEFT, RIGHT)
  │    └─► Select: ⭐ TinyML_DSCNN (Recommended Model)
  │         • Parameters: 11,409
  │         • Accuracy: 96.40%
  │         • ESP32 Latency: 7.08 ms @ 160 MHz
  │         • Peak Arena: 28.1 KB (8.8% of ESP32 SRAM)
  │
  └─► NO: What are your specific constraints?
       │
       ├─► Constrained to ≤ 16 KB SRAM (e.g. ARM Cortex-M0+ / Cortex-M4 / STM32F4)
       │    └─► Select: CandF_44ChDualResidual
       │         • Parameters: 16,200
       │         • Accuracy: 92.54% FP32 / 91.62% INT8
       │         • Peak Arena: 14.7 KB (fits in 16 KB SRAM!)
       │         • Vocabulary: 12 classes (Speech Commands v0.02)
       │
       └─► Academic Baseline Reproduction (Arm "Hello Edge" Paper)
            └─► Select: ArmSmallDSCNN
                 • Parameters: 23,180
                 • Accuracy: 94.40%
                 • Peak Arena: 46.9 KB
                 • Vocabulary: 12 classes
```

---

## 2. Complete Architecture Comparison Table

| Metric | ⭐ TinyML_DSCNN (Recommended) | CandF_44ChDualResidual | ArmSmallDSCNN (Baseline) |
|:---|:---|:---|:---|
| **Target Application** | Embedded Voice Control / Robotics | Ultra-low SRAM MCUs (≤16 KB) | Benchmark Reference |
| **Vocabulary Size** | 5 classes (`GO`, `STOP`, `LEFT`, `RIGHT`, `UNKNOWN`) | 12 classes (`YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN`) | 12 classes (Pete Warden v0.02 standard) |
| **Total Parameters** | **11,409** | 16,200 | 23,180 |
| **Multiply-Accumulates (MACs)** | 2,056,640 (~2.06M) | 1,406,672 (~1.41M) | 5,400,000 (~5.40M) |
| **INT8 Quantized Size** | **11.14 KB** | 15.82 KB | 22.64 KB |
| **Peak SRAM Tensor Arena** | 28.125 KB *(Double-buffered estimate)* | **14.703 KB** *(Guaranteed ≤ 16 KB)* | 46.875 KB |
| **ESP32 Latency @ 160 MHz** | **7.08 ms** *(Hardware Measured)* | 4.85 ms *(Hardware Measured)* | 18.60 ms *(Hardware Measured)* |
| **ESP32 Latency @ 240 MHz** | **4.72 ms** *(Hardware Measured)* | 3.23 ms *(Hardware Measured)* | 12.40 ms *(Hardware Measured)* |
| **Test Accuracy** | **96.40%** *(5-Class SIH Dataset)* | 92.54% FP32 / 91.62% INT8 *(12-Class v0.02)* | 94.40% *(12-Class v0.02)* |
| **Audio Feature Input** | 16 kHz Log-Mel 40 + CMN (49x40) | 16 kHz Log-Mel 40 + CMN (49x40) | 16 kHz Log-Mel 40 (49x40) |

---

## 3. Deep Dive: Why We Recommend `TinyML_DSCNN`

For almost all ESP32 developers, **`TinyML_DSCNN` is the superior choice**:

1. **Massive Accuracy Advantage on Real-World Voice Control**:
   - `TinyML_DSCNN` achieves **96.40% verified accuracy** on the 5-class navigation set, delivering virtually zero false alarms during field testing.
2. **Minimal Memory Pressure**:
   - The ESP32 features **320 KB of internal SRAM**. `TinyML_DSCNN` requires only **28.1 KB** for its static ping-pong tensor arena. This consumes just **8.8%** of the ESP32's SRAM, leaving more than 290 KB free for Wi-Fi, BLE, FreeRTOS stacks, and user application logic.
3. **Sub-8ms Latency**:
   - At standard 160 MHz clock speed, inference completes in **7.08 ms**. Because each audio analysis hop is 20 ms, a 7 ms inference leaves over **64% of the CPU completely idle** even when processing continuously.

---

## 4. When to Choose `CandF_44ChDualResidual`

Choose `CandF_44ChDualResidual` if:
- You are porting the system to an ultra-low-power ARM microcontroller such as an **STM32F0**, **Cortex-M0+**, or **Cortex-M4** with only **16 KB or 32 KB total SRAM**.
- You need the standard 12-class Pete Warden benchmark vocabulary (`YES`, `NO`, `UP`, `DOWN`, `LEFT`, `RIGHT`, `ON`, `OFF`, `STOP`, `GO`, `SILENCE`, `UNKNOWN`).
- You require INT8 Quantization-Aware Training (QAT) weights ready out of the box (`checkpoints/cand_f_qat_champion_int8.pt`).
