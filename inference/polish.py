"""
Phase 3.2: Iterative polishing.

Takes a reference screenshot + initial code, renders it, then asks the model
to improve the code by comparing the render against the reference.
UI2Code^N showed +12% improvement with 4 rounds.

Usage:
    python -m inference.polish --checkpoint /shared/advey/checkpoints/grpo_final --image widget.png --rounds 3
"""
import re
import argparse
import torch
from inference.generate import load_model, generate, extract_code
from inference.best_of_n import best_of_n
from reward.programmatic import compute_reward_code, render_html_to_image
from training.sft import SYSTEM_PROMPT


POLISH_PROMPT = (
    "You are given:\n"
    "1. A reference widget screenshot (the target design)\n"
    "2. Your previous HTML/CSS code attempt\n"
    "3. A screenshot of how your code currently renders\n\n"
    "Compare the rendered result against the reference. Identify differences in:\n"
    "- Layout (spacing, alignment, sizes)\n"
    "- Colors (palette, gradients, backgrounds)\n"
    "- Typography (font sizes, weights, contrast)\n"
    "- Missing or extra elements\n\n"
    "Then output an improved version.\n"
    "Format: <think>[comparison reasoning]</think><code>[corrected HTML/CSS]</code>"
)


def polish(model, tokenizer, ref_image: bytes, initial_html: str, rounds: int = 3) -> list[tuple[str, float]]:
    """
    Iteratively refine HTML code over N rounds.
    Returns list of (html, score) for each round.
    """
    current_html = initial_html
    history = []

    initial_score = compute_reward_code(ref_image, current_html)
    history.append((current_html, initial_score))
    print(f"  Round 0 (initial): score={initial_score:.4f}")

    for r in range(rounds):
        # build polishing prompt with the current code
        prompt = (
            POLISH_PROMPT + "\n\n"
            f"Previous HTML code:\n```html\n{current_html}\n```\n\n"
            "Generate an improved version that more closely matches the reference."
        )

        text = generate(model, tokenizer, prompt, temperature=0.5)
        html = extract_code(text)

        if html is None:
            print(f"  Round {r+1}: no <code> block, keeping previous version")
            history.append(history[-1])
            continue

        score = compute_reward_code(ref_image, html)
        print(f"  Round {r+1}: score={score:.4f}")

        # only keep if it improved
        if score > history[-1][1]:
            current_html = html
            history.append((html, score))
        else:
            print(f"    -> no improvement, keeping previous version")
            history.append(history[-1])

    return history


def generate_with_polish(model, tokenizer, ref_image: bytes, n: int = 4,
                         polish_rounds: int = 3) -> tuple[str, float]:
    """
    Phase 3.3 combined strategy:
    1. Generate N candidates (best-of-N)
    2. Select best candidate
    3. Run M polishing rounds on the best
    """
    prompt = SYSTEM_PROMPT + "\nRecreate the widget shown in the reference image."

    # Step 1-2: best-of-N
    html, score = best_of_n(model, tokenizer, prompt, ref_image, n=n)
    if html is None:
        return None, 0.0
    print(f"Best-of-{n} score: {score:.4f}")

    # Step 3: polish
    history = polish(model, tokenizer, ref_image, html, rounds=polish_rounds)
    return history[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--n", type=int, default=4, help="Best-of-N candidates")
    parser.add_argument("--rounds", type=int, default=3, help="Polishing rounds")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.checkpoint)

    with open(args.image, "rb") as f:
        ref_image = f.read()

    html, score = generate_with_polish(model, tokenizer, ref_image, n=args.n, polish_rounds=args.rounds)
    if html:
        print(f"\nFinal score: {score:.4f}")
        print(html)
    else:
        print("Generation failed.")
