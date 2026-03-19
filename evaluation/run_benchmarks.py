"""
Phase 4: Evaluation on Widget2Code benchmark.

Benchmark: https://github.com/Djanghao/widget2code
Dataset: https://huggingface.co/datasets/Djanghao/widget2code-benchmark

Setup:
    1. Clone widget2code repo:
       git clone https://github.com/Djanghao/widget2code.git ./evaluation/widget2code
    2. Download benchmark dataset:
       huggingface-cli download Djanghao/widget2code-benchmark --repo-type dataset --local-dir ./data/widget2code-benchmark
    3. Set GT_DIR in .env:
       GT_DIR=./data/widget2code-benchmark/test

Usage:
    # Generate our model's outputs for the benchmark
    python -m evaluation.run_benchmarks --checkpoint /shared/advey/checkpoints/grpo_final --output_dir ./output/benchmark_results

    # Then run widget2code evaluation:
    cd evaluation/widget2code
    ./scripts/evaluation/run_evaluation.sh ../../output/benchmark_results -g ../../data/widget2code-benchmark/test --cuda -w 16
"""
import json
import os
import glob
import argparse
import torch
from inference.generate import load_model, generate, extract_code
from inference.polish import generate_with_polish
from reward.programmatic import render_tsx_to_image, compute_reward_code
from training.sft import MODEL_NAME


def run_benchmark(checkpoint: str, model_name: str, gt_dir: str, output_dir: str,
                  use_polish: bool = False, n: int = 4, polish_rounds: int = 3,
                  device: str = "cuda:0"):
    os.makedirs(output_dir, exist_ok=True)

    model, processor = load_model(checkpoint, model_name=model_name, device=device)

    widget_images = sorted(glob.glob(os.path.join(gt_dir, "**", "*.png"), recursive=True))
    print(f"Found {len(widget_images)} benchmark widgets in {gt_dir}")

    results = []
    for i, img_path in enumerate(widget_images):
        widget_id = os.path.splitext(os.path.basename(img_path))[0]
        tsx_out_path = os.path.join(output_dir, f"{widget_id}.tsx")

        if os.path.exists(tsx_out_path):
            continue

        print(f"[{i+1}/{len(widget_images)}] {widget_id}")

        with open(img_path, "rb") as f:
            ref_image = f.read()

        try:
            if use_polish:
                tsx, score = generate_with_polish(model, processor, img_path, ref_image, n=n, polish_rounds=polish_rounds)
            else:
                text = generate(model, processor, img_path)
                tsx = extract_code(text)
                score = compute_reward_code(ref_image, tsx) if tsx else 0.0

            if tsx is None:
                print(f"  Failed to generate code")
                continue

            # save TSX
            with open(tsx_out_path, "w") as f:
                f.write(tsx)

            # render and save screenshot
            try:
                rendered = render_tsx_to_image(tsx)
                png_path = os.path.join(output_dir, f"{widget_id}.png")
                with open(png_path, "wb") as f:
                    f.write(rendered)
            except Exception:
                pass

            results.append({"widget_id": widget_id, "score": score})

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # save summary
    with open(os.path.join(output_dir, "results_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nDone! Generated {len(results)} outputs in {output_dir}")
    print(f"Run widget2code evaluation with:")
    print(f"  cd evaluation/widget2code")
    print(f"  ./scripts/evaluation/run_evaluation.sh ../../{output_dir} -g ../../{gt_dir} --cuda -w 16")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--gt_dir", type=str, default="./data/widget2code-benchmark/test")
    parser.add_argument("--output_dir", type=str, default="./output/benchmark_results")
    parser.add_argument("--polish", action="store_true", help="Use best-of-N + polishing")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--polish_rounds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_benchmark(args.checkpoint, args.model_name, args.gt_dir, args.output_dir,
                  use_polish=args.polish, n=args.n, polish_rounds=args.polish_rounds,
                  device=args.device)
