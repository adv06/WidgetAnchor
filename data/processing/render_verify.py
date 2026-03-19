"""
Phase 0.2 (Step 3): Render verification.

Re-render each HTML with Playwright, screenshot the result,
compute SSIM against original widget screenshot, discard pairs below threshold.

Usage:
    python -m data.processing.render_verify --input_dir ./data/raw --output_dir ./data/verified --threshold 0.85
"""
import json
import os
import glob
import argparse
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from reward.programmatic import render_tsx_to_image


def compute_ssim(ref_bytes: bytes, rendered_bytes: bytes) -> float:
    ref = cv2.imdecode(np.frombuffer(ref_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    gen = cv2.imdecode(np.frombuffer(rendered_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if ref is None or gen is None:
        return 0.0
    h = min(ref.shape[0], gen.shape[0])
    w = min(ref.shape[1], gen.shape[1])
    ref = cv2.resize(ref, (w, h))
    gen = cv2.resize(gen, (w, h))
    score, _ = ssim(ref, gen, full=True, channel_axis=2)
    return score


def verify(input_dir: str, output_dir: str, threshold: float = 0.85):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "rendered"), exist_ok=True)

    raw_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(raw_files)} raw samples to verify")

    kept = 0
    discarded = 0
    render_failed = 0

    for i, path in enumerate(raw_files):
        with open(path) as f:
            sample = json.load(f)

        widget_id = sample["widget_id"]
        tsx = sample["tsx"]
        screenshot_path = sample["screenshot_path"]

        # render the generated TSX
        try:
            rendered_bytes = render_tsx_to_image(tsx)
        except Exception:
            render_failed += 1
            continue

        # load original screenshot
        with open(screenshot_path, "rb") as f:
            ref_bytes = f.read()

        score = compute_ssim(ref_bytes, rendered_bytes)

        if score < threshold:
            discarded += 1
            continue

        # save verified sample
        rendered_path = os.path.join(output_dir, "rendered", f"{widget_id}.png")
        with open(rendered_path, "wb") as f:
            f.write(rendered_bytes)

        sample["ssim_score"] = score
        sample["rendered_path"] = rendered_path

        out_path = os.path.join(output_dir, f"{widget_id}.json")
        with open(out_path, "w") as f:
            json.dump(sample, f)

        kept += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(raw_files)} | kept: {kept} | discarded: {discarded} | render_failed: {render_failed}")

    print(f"\nDone! kept: {kept}, discarded (SSIM < {threshold}): {discarded}, render_failed: {render_failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/raw")
    parser.add_argument("--output_dir", type=str, default="./data/verified")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    verify(args.input_dir, args.output_dir, args.threshold)
