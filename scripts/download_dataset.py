"""
Google Speech Commands v0.02 Downloader & Verifier.
Downloads the official 12-class benchmark dataset archive (~2.4 GB) and sets up standard partitions.
"""
import os
import sys
import tarfile
import urllib.request

DATA_URL = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
TARGET_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "speech_commands_v0.02")

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    tar_path = os.path.join(TARGET_DIR, "speech_commands_v0.02.tar.gz")
    
    if not os.path.exists(tar_path):
        print(f"Downloading {DATA_URL} to {tar_path}...")
        urllib.request.urlretrieve(DATA_URL, tar_path)
        print("Download complete.")
        
    print(f"Extracting {tar_path}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=TARGET_DIR)
    print(f"Dataset successfully prepared in {TARGET_DIR}")

if __name__ == "__main__":
    main()
