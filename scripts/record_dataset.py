#!/usr/bin/env python3
"""
scripts/record_dataset.py
-------------------------
Interactive audio recording CLI to record custom voice command datasets.
Records 16 kHz 16-bit mono 1-second WAV files organized into class folders.

Usage:
    python scripts/record_dataset.py --keyword OPEN --num-samples 20
    python scripts/record_dataset.py --keyword CLOSE --num-samples 20
    python scripts/record_dataset.py --keyword _silence_ --num-samples 20
    python scripts/record_dataset.py --synthetic --keyword TEST --num-samples 5  # For headless / automated CI
"""

import os
import sys
import time
import argparse
import wave
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "custom_dataset")

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
NUM_SAMPLES_PER_CLIP = int(SAMPLE_RATE * DURATION_SEC)


def record_with_sounddevice(duration: float, sample_rate: int):
    """Records audio clip using sounddevice library."""
    try:
        import sounddevice as sd
    except ImportError:
        raise ImportError(
            "Microphone recording requires 'sounddevice'.\n"
            "Install it via: pip install sounddevice\n"
            "Or run with '--synthetic' to generate test samples without physical microphone hardware."
        )

    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    return audio.flatten()


def generate_synthetic_audio(duration: float, sample_rate: int):
    """Generates synthetic audio for automated testing or headless environments."""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate mixture of harmonic sine waves + low background noise
    freq = np.random.uniform(200, 800)
    signal = 0.4 * np.sin(2 * np.pi * freq * t) + 0.1 * np.random.normal(0, 0.05, len(t))
    int16_signal = (signal * 32767).clip(-32768, 32767).astype(np.int16)
    return int16_signal


def save_wav(filepath: str, audio_data: np.ndarray, sample_rate: int):
    """Writes mono 16-bit PCM WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Record custom voice command audio dataset for ESP32 KWS.")
    parser.add_argument("--keyword", type=str, required=True, help="Command keyword name (e.g., OPEN, CLOSE, LIGHT_ON)")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of audio samples to record (default: 20)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output root directory for dataset")
    parser.add_argument("--duration", type=float, default=1.0, help="Sample duration in seconds (default: 1.0s)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic test audio without physical microphone")
    args = parser.parse_args()

    class_dir = os.path.join(args.output_dir, args.keyword.strip().lower())
    os.makedirs(class_dir, exist_ok=True)

    print("==================================================")
    print("  Custom Voice Command Audio Recorder")
    print("==================================================")
    print(f"  Target Keyword   : {args.keyword.upper()}")
    print(f"  Samples to Record: {args.num_samples}")
    print(f"  Sample Rate      : {SAMPLE_RATE} Hz (16-bit Mono)")
    print(f"  Clip Duration    : {args.duration} s")
    print(f"  Output Directory : {class_dir}")
    print(f"  Mode             : {'Synthetic (No Mic)' if args.synthetic else 'Live Microphone'}")
    print("==================================================")

    if not args.synthetic:
        try:
            import sounddevice as sd
            print("[INFO] Audio device available: " + str(sd.query_devices(kind='input')['name']))
        except Exception as e:
            print(f"\n[NOTE] Live microphone recording unavailable: {e}")
            print("Tip: Install 'sounddevice' (pip install sounddevice) or run with '--synthetic' for testing.\n")
            if not sys.stdin.isatty():
                print("[INFO] Non-interactive environment detected. Falling back to synthetic generation.")
                args.synthetic = True

    for i in range(1, args.num_samples + 1):
        filename = f"{args.keyword.lower()}_{int(time.time())}_{i:03d}.wav"
        filepath = os.path.join(class_dir, filename)

        if not args.synthetic:
            input(f"\n[{i}/{args.num_samples}] Press ENTER and speak '{args.keyword.upper()}'... ")
            print("  -> Recording... (1.0 second)")
            audio = record_with_sounddevice(args.duration, SAMPLE_RATE)
            print("  -> Captured!")
        else:
            audio = generate_synthetic_audio(args.duration, SAMPLE_RATE)

        save_wav(filepath, audio, SAMPLE_RATE)
        print(f"  Saved: {filename} ({len(audio)} samples)")

    print(f"\n[SUCCESS] Completed recording {args.num_samples} samples for '{args.keyword.upper()}'.")
    print(f"Files saved in: {class_dir}")
    print("\nTo train your model with this dataset:")
    print(f"  python scripts/train.py --data-dir {args.output_dir} --epochs 30\n")


if __name__ == "__main__":
    main()
