# Usage: python -m inference.polish --checkpoint /shared/advey/checkpoints/grpo_final --image widget.png --rounds 3

# continuous refinement
import argparse
from inference.generate import load_model, extract_code
from inference.best_of_n import best_of_n
from reward.programmatic import compute_reward_code
from training.sft import _get_image_size
import torch


POLISH_PROMPT = (
    "You are a high-fidelity UI reproduction expert. You are given a reference widget screenshot "
    "and your previous React+Tailwind attempt. Compare carefully against the reference and fix:\n"
    "- Layout misalignment (flex/grid direction, spacing, padding, gaps)\n"
    "- Wrong or missing colors (use exact hex values via `bg-[#hex]` syntax)\n"
    "- Typography differences (text size, font weight, leading)\n"
    "- Missing or incorrect text content (must be character-perfect)\n"
    "- Missing elements (icons from lucide-react, borders, shadows, dividers)\n"
    "- Charts: use recharts components\n\n"
    "Format: <think>[comparison reasoning]</think><code>[corrected React component]</code>"
)


def polish(model, processor, image_path: str, ref_image: bytes, initial_tsx: str,
           rounds: int = 3) -> list[tuple[str, float]]:
    """Iteratively refine TSX code over N rounds."""
    current_tsx = initial_tsx
    history = []

    initial_score = compute_reward_code(ref_image, current_tsx)
    history.append((current_tsx, initial_score))
    print(f"  Round 0 (initial): score={initial_score:.4f}")

    w, h = _get_image_size(image_path)

    for r in range(rounds):
        # build polishing prompt with reference image + current code
        messages = [
            {"role": "system", "content": [{"type": "text", "text": POLISH_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": f"Widget dimensions: {w}x{h}px.\n\nHere is my previous React component attempt:\n```tsx\n{current_tsx}\n```\n\nImprove it to better match the reference widget."},
            ]},
        ]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(**inputs, max_new_tokens=2048, temperature=0.5, do_sample=True)
        prompt_len = inputs["input_ids"].shape[1]
        text = processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

        tsx = extract_code(text)
        if tsx is None:
            print(f"  Round {r+1}: no <code> block, keeping previous version")
            history.append(history[-1])
            continue

        score = compute_reward_code(ref_image, tsx)
        print(f"  Round {r+1}: score={score:.4f}")

        if score > history[-1][1]:
            current_tsx = tsx
            history.append((tsx, score))
        else:
            print(f"    -> no improvement, keeping previous version")
            history.append(history[-1])

    return history


def generate_with_polish(model, processor, image_path: str, ref_image: bytes,
                         n: int = 4, polish_rounds: int = 3) -> tuple[str, float]:
    """Combined: best-of-N then polish the winner."""
    tsx, score = best_of_n(model, processor, image_path, ref_image, n=n)
    if tsx is None:
        return None, 0.0
    print(f"Best-of-{n} score: {score:.4f}")

    history = polish(model, processor, image_path, ref_image, tsx, rounds=polish_rounds)
    return history[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint)

    with open(args.image, "rb") as f:
        ref_image = f.read()

    tsx, score = generate_with_polish(model, processor, args.image, ref_image, n=args.n, polish_rounds=args.rounds)
    if tsx:
        print(f"\nFinal score: {score:.4f}")
        print(tsx)
    else:
        print("Generation failed.")
