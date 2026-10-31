# TinyML Basics Explained: How Speech Recognition Fits on a $3 Chip

If you are new to machine learning on embedded hardware (TinyML), you might wonder:
> *"How can a tiny microcontroller with only 320 KB of RAM and no internet connection recognize human speech in real time?"*

This guide explains the complete pipeline in plain English.

---

## 1. What is TinyML?

Traditional AI (like ChatGPT or Siri) runs on massive server farms in the cloud with thousands of gigabytes of RAM. When you speak to Siri, your audio is recorded, compressed, sent over the internet to a data center, processed by large neural networks, and the result is sent back.

**TinyML (Tiny Machine Learning)** flips this paradigm:
- The entire neural network lives **inside the microcontroller's flash memory**.
- All calculations occur **locally on the microcontroller's CPU**.
- It requires **no internet**, **no cloud subscription**, has **zero network latency**, and **100% user privacy** (your voice never leaves the chip).

---

## 2. The 4-Stage Audio-to-Action Pipeline

When you speak `"GO"` into the microphone, four things happen in sequence:

```text
Human Voice ──► [ 1. I2S Digital Mic ] ──► [ 2. Audio Preprocessing ]
                                                      │
[ 4. Hardware Action ] ◄── [ 3. Quantized Neural Net ] ◄┘
```

### Stage 1: Digital Audio Capture (I2S)
An I2S digital MEMS microphone (like the INMP441) samples sound waves 16,000 times per second (16 kHz). Each sample is converted into a 16-bit number representing air pressure at that microsecond. Over 1 second, this produces an array of 16,000 numbers.

### Stage 2: Audio Preprocessing (Spectrograms)
Raw audio waveforms are difficult for small neural networks to understand directly. Instead, we convert sound into a visual picture called a **Log-Mel Spectrogram**:
1. **Pre-emphasis**: High frequencies are amplified so quiet consonants (like *"s"* or *"t"*) are not lost.
2. **Windowing**: The 1-second audio is sliced into short 30 ms chunks called frames.
3. **Mel Filterbank**: Human ears do not hear frequencies linearly; we are much better at distinguishing low pitches than high pitches. A Mel filterbank groups audio frequencies into 40 bands mirroring human perception.
4. The result is a small **2D image of sound**: 49 time steps by 40 frequency bands (1,960 numbers total).

### Stage 3: Neural Network Inference (DS-CNN)
The 2D spectrogram image is fed into a **Depthwise Separable Convolutional Neural Network (DS-CNN)**:
- Standard convolution multiplies all input channels simultaneously, requiring millions of calculations.
- Depthwise separable convolution splits the operation into two smaller steps:
  1. A **depthwise** filter that scans across time and frequency independently.
  2. A **pointwise** (1x1) filter that combines the channel information.
- This architectural trick slashes computational requirements and memory consumption by over **85%** without sacrificing accuracy!

### Stage 4: Prediction & Hardware Trigger
The final layer outputs probabilities for each possible word:
- `GO`: 94.2%
- `STOP`: 2.1%
- `LEFT`: 1.8%
- `RIGHT`: 0.9%
- `UNKNOWN`: 1.0%

Because `GO` exceeds our confidence threshold (80%), the ESP32 code immediately executes our custom action (e.g. flips a relay pin from LOW to HIGH).

---

## 3. Core TinyML Concepts Explained

### What is a "MAC"? (Multiply-Accumulate)
A Multiply-Accumulate operation is the fundamental math step in neural networks:
$$\text{Output} = \text{Input} \times \text{Weight} + \text{Bias}$$
When you see that `TinyML_DSCNN` requires **2.06M MACs**, that means the chip performs approximately 2 million multiplications and additions to analyze one second of speech. At 160 MHz, the ESP32 crunches through these 2 million operations in just **7.08 milliseconds**!

### What is INT8 Quantization?
In Python and PyTorch on your PC, numbers are stored as **32-bit floating point (FP32)** numbers (e.g. `0.84920194`).
On microcontrollers, storing 32-bit floats wastes precious flash and RAM:
- By converting every weight to an **8-bit integer (INT8)** (an integer from -128 to +127), we shrink model storage size by **4x** (from 44 KB down to 11 KB).
- Integer arithmetic is also significantly faster and consumes less power than floating-point math.

### What is a "Tensor Arena"?
On a microcontroller, you cannot use dynamic memory allocation (`malloc()` or `new`) during real-time audio processing without risking memory fragmentation and crashes.
Instead, TinyML uses a single static block of memory called a **Tensor Arena**:
- Two scratchpad buffers (Buffer A and Buffer B) are pre-allocated in SRAM.
- Layer 1 reads from Buffer A and writes its results into Buffer B.
- Layer 2 reads from Buffer B and writes its results back into Buffer A.
- This "ping-pong" mechanism guarantees that peak SRAM usage never exceeds **28.1 KB**, leaving the rest of the ESP32's 320 KB SRAM completely free.
