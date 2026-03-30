"""
Evaluate SFT model using widget-factory's evaluation pipeline.

1. Load SFT checkpoint
2. For each GT image in widget-factory test_split, generate TSX → render → PNG
3. Save in widget-factory's expected directory structure
4. Run widget-factory's evaluation script

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m evaluation.run_wf_eval \
        --checkpoint /shared/advey/checkpoints/sft_step_2200 \
        --num_samples 500
"""
import json
import os
import glob
import random
import argparse
import torch
from inference.generate import load_model, generate, extract_code
from reward.programmatic import render_tsx_to_image
from training.sft import MODEL_NAME

WF_DIR = "/home/advey/widget-factory"


def run_eval(checkpoint: str, model_name: str, num_samples: int, output_name: str, gt_dir: str, device: str):
    results_dir = os.path.join(WF_DIR, "results", output_name)
    os.makedirs(results_dir, exist_ok=True)

    model, processor = load_model(checkpoint, model_name=model_name, device=device)

    # find all GT images (supports gt_*.png and image_*.png naming)
    gt_images = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    print(f"Found {len(gt_images)} GT images in {gt_dir}")

    # sample if requested
    if num_samples and num_samples < len(gt_images):
        random.seed(42)
        gt_images = random.sample(gt_images, num_samples)
        gt_images.sort()
        print(f"Sampled {num_samples} images for evaluation")

    succeeded = 0
    failed_gen = 0
    failed_render = 0

    for i, gt_path in enumerate(gt_images):
        # image_0001.png -> image_0001, gt_synthetic-00000.png -> synthetic-00000
        basename = os.path.splitext(os.path.basename(gt_path))[0]
        widget_id = basename.removeprefix("gt_")
        widget_dir = os.path.join(results_dir, widget_id)
        output_png = os.path.join(widget_dir, "output.png")

        if os.path.exists(output_png):
            succeeded += 1
            continue

        print(f"[{i+1}/{len(gt_images)}] {basename}")

        try:
            text = generate(model, processor, gt_path)
            tsx = extract_code(text)

            if tsx is None:
                print(f"  No <code> block found, skipping")
                failed_gen += 1
                continue

            # save TSX for debugging
            os.makedirs(widget_dir, exist_ok=True)
            with open(os.path.join(widget_dir, "generated.tsx"), "w") as f:
                f.write(tsx)

            # render TSX -> PNG
            png_bytes = render_tsx_to_image(tsx)
            with open(output_png, "wb") as f:
                f.write(png_bytes)

            # also copy input for reference
            import shutil
            shutil.copy2(gt_path, os.path.join(widget_dir, "input.png"))

            succeeded += 1

        except Exception as e:
            print(f"  Error: {e}")
            failed_render += 1

        if (i + 1) % 25 == 0:
            print(f"  Progress: {i+1}/{len(gt_images)} | ok: {succeeded} | gen_fail: {failed_gen} | render_fail: {failed_render}")

    print(f"\nGeneration done: {succeeded} succeeded, {failed_gen} gen failures, {failed_render} render failures")
    print(f"Results saved to: {results_dir}")
    print(f"\nNow run widget-factory evaluation:")
    print(f"  cd {WF_DIR}")
    print(f"  ./scripts/evaluation/run_evaluation.sh results/{output_name} -g test_split -w 8 --cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/shared/advey/checkpoints/sft_step_2200")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--output_name", type=str, default="sft-v3-hf-eval")
    parser.add_argument("--gt_dir", type=str, default="/home/advey/WidgetAnchor/data/widget2code-benchmark/test")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_eval(args.checkpoint, args.model_name, args.num_samples, args.output_name, args.gt_dir, args.device)
