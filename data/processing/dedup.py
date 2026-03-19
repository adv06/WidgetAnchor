"""
Phase 0.2 (Step 4): Deduplication.

Perceptual hashing (dHash) on screenshots to remove near-duplicates.
Uses multi-probe bucketing to avoid O(n^2) comparisons.

Usage:
    python -m data.processing.dedup --input_dir ./output/verified --output_dir ./output/deduped --hash_threshold 8
"""
import json
import os
import glob
import argparse
import cv2
import numpy as np
from collections import defaultdict


def compute_dhash(image_bytes: bytes, hash_size: int = 16) -> np.ndarray:
    """Compute difference hash of an image."""
    img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    resized = cv2.resize(img, (hash_size + 1, hash_size))
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def hash_to_int(h: np.ndarray) -> int:
    """Convert boolean hash array to integer for bucketing."""
    result = 0
    for bit in h:
        result = (result << 1) | int(bit)
    return result


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

        h = compute_dhash(img_bytes)
        hashes.append(h)
        samples.append((path, sample))

    # band-based dedup: split hash into bands, bucket by band value
    # items sharing a band bucket are candidate duplicates
    hash_bits = len(hashes[0]) if hashes else 256
    num_bands = max(1, hash_bits // hash_threshold)  # each band ~ threshold bits
    band_size = hash_bits // num_bands

    # build buckets per band
    candidates = defaultdict(set)  # maps index -> set of indices to compare against
    for band_idx in range(num_bands):
        buckets = defaultdict(list)
        start = band_idx * band_size
        end = start + band_size
        for i, h in enumerate(hashes):
            band_key = hash_to_int(h[start:end])
            buckets[band_key].append(i)
        for bucket in buckets.values():
            if len(bucket) > 1:
                for i in range(len(bucket)):
                    for j in range(i + 1, len(bucket)):
                        candidates[bucket[i]].add(bucket[j])
                        candidates[bucket[j]].add(bucket[i])

    # greedy dedup using only candidate pairs
    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in candidates.get(i, set()):
            if j <= i or not keep[j]:
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
    parser.add_argument("--input_dir", type=str, default="./output/verified")
    parser.add_argument("--output_dir", type=str, default="./output/deduped")
    parser.add_argument("--hash_threshold", type=int, default=8)
    args = parser.parse_args()
    dedup(args.input_dir, args.output_dir, args.hash_threshold)
