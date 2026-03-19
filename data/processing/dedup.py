"""
Phase 0.2 (Step 4): Deduplication.

Perceptual hashing (pHash) on screenshots to remove near-duplicates.

Usage:
    python -m data.processing.dedup --input_dir ./data/verified --output_dir ./data/deduped --hash_threshold 8
"""
import json
import os
import glob
import argparse
import cv2
import numpy as np


def compute_phash(image_bytes: bytes, hash_size: int = 16) -> np.ndarray:
    """Compute perceptual hash of an image."""
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    # DCT-based pHash: compare adjacent pixels
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


def dedup(input_dir: str, output_dir: str, hash_threshold: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    verified_files = sorted(glob.glob(os.path.join(input_dir, "widget-*.json")))
    print(f"Found {len(verified_files)} verified samples")

    # compute hashes
    hashes = []
    samples = []
    for path in verified_files:
        with open(path) as f:
            sample = json.load(f)

        with open(sample["screenshot_path"], "rb") as f:
            img_bytes = f.read()

        h = compute_phash(img_bytes)
        hashes.append(h)
        samples.append((path, sample))

    # greedy dedup: keep first, discard any within threshold
    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if not keep[j]:
                continue
            if hamming_distance(hashes[i], hashes[j]) < hash_threshold:
                keep[j] = False

    kept = 0
    for i, (path, sample) in enumerate(samples):
        if not keep[i]:
            continue
        out_path = os.path.join(output_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(sample, f)
        kept += 1

    removed = len(samples) - kept
    print(f"Done! kept: {kept}, removed as duplicates: {removed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/verified")
    parser.add_argument("--output_dir", type=str, default="./data/deduped")
    parser.add_argument("--hash_threshold", type=int, default=8)
    args = parser.parse_args()
    dedup(args.input_dir, args.output_dir, args.hash_threshold)
