"""
Phase 3.1: Best-of-N sampling.

Generate N candidates per widget, score all with the reward function, select the best.
Typically gives 2-5 point improvement with no additional training.

Usage:
    python -m inference.best_of_n --checkpoint /shared/advey/checkpoints/grpo_final --image widget.png --n 4
"""
import argparse
import torch
from inference.generate import load_model, generate, extract_code
from reward.programmatic import compute_reward_code, render_html_to_image
from training.sft import SYSTEM_PROMPT


def best_of_n(model, tokenizer, prompt: str, ref_image: bytes, n: int = 4,
              temperature: float = 0.8) -> tuple[str, float]:
    """Generate N candidates, return the best (html, score) pair."""
    candidates = []

    for i in range(n):
        text = generate(model, tokenizer, prompt, temperature=temperature)
        html = extract_code(text)
        if html is None:
            continue
        try:
            score = compute_reward_code(ref_image, html)
            candidates.append((html, score))
        except Exception:
            continue

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Reference widget screenshot path")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.checkpoint)
    prompt = SYSTEM_PROMPT + "\nRecreate the widget shown in the reference image."

    with open(args.image, "rb") as f:
        ref_image = f.read()

    html, score = best_of_n(model, tokenizer, prompt, ref_image, n=args.n, temperature=args.temperature)
    if html:
        print(f"Best score: {score:.4f}")
        print(html)
    else:
        print("All candidates failed.")
