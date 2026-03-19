"""
Phase 0: Final split into train/val/test (90/5/5).

Assembles all annotated samples into a single dataset JSON ready for SFT training.
Each entry has: screenshot_path, cot (structured reasoning + code), difficulty, widget_id.

Usage:
    python -m data.processing.split_data --input_dir ./data/annotated --output_dir ./data/final
"""
import json
import os
import glob
import random
import argparse


def split(input_dir: str, output_dir: str, seed: int = 42):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "widget-*.json")))
    print(f"Found {len(files)} annotated samples")

    dataset = []
    for path in files:
        with open(path) as f:
            sample = json.load(f)

        dataset.append({
            "widget_id": sample["widget_id"],
            "screenshot_path": sample["screenshot_path"],
            "html": sample["html"],
            "cot": sample["cot"],
            "difficulty": sample.get("difficulty", "unknown"),
            "token_count": sample.get("token_count", 0),
            "ssim_score": sample.get("ssim_score", 0),
        })

    random.seed(seed)
    random.shuffle(dataset)

    n = len(dataset)
    train_end = int(n * 0.90)
    val_end = int(n * 0.95)

    train = dataset[:train_end]
    val = dataset[train_end:val_end]
    test = dataset[val_end:]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # difficulty breakdown
    for name, data in [("train", train), ("val", val), ("test", test)]:
        counts = {}
        for d in data:
            diff = d["difficulty"]
            counts[diff] = counts.get(diff, 0) + 1
        print(f"  {name}: {len(data)} samples — {counts}")

    print(f"\nDone! Saved to {output_dir}/")
    print(f"  train: {len(train)}, val: {len(val)}, test: {len(test)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/annotated")
    parser.add_argument("--output_dir", type=str, default="./data/final")
    args = parser.parse_args()
    split(args.input_dir, args.output_dir)
