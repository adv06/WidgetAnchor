# Usage: python -m inference.best_of_n --checkpoint /shared/advey/checkpoints/grpo_final --image widget.png --n 4

import argparse
from inference.generate import load_model, generate, extract_code
from reward.programmatic import compute_reward_code


def best_of_n(model, processor, image_path: str, ref_image: bytes, n: int = 4,
              temperature: float = 0.8) -> tuple[str, float]:
    """Generate N candidates, return the best (tsx, score) pair."""
    candidates = []

    for i in range(n):
        text = generate(model, processor, image_path, temperature=temperature)
        tsx = extract_code(text)
        if tsx is None:
            continue
        try:
            score = compute_reward_code(ref_image, tsx)
            candidates.append((tsx, score))
        except Exception:
            continue

    if not candidates:
        return None, 0.0

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint)

    with open(args.image, "rb") as f:
        ref_image = f.read()

    tsx, score = best_of_n(model, processor, args.image, ref_image, n=args.n, temperature=args.temperature)
    if tsx:
        print(f"Best score: {score:.4f}")
        print(tsx)
    else:
        print("All candidates failed.")
