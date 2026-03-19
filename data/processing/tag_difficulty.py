"""
Phase 0.2 (Step 5): Difficulty tagging.

Tag each sample as simple/medium/complex based on HTML token count.
- Simple: <50 tokens
- Medium: 50-200 tokens
- Complex: >200 tokens

Usage:
    python -m data.processing.tag_difficulty --input_dir ./data/deduped --output_dir ./data/tagged
"""
import json
import os
import glob
import argparse
import re


def count_tokens(html: str) -> int:
    """Rough token count: split on whitespace and punctuation boundaries."""
    tokens = re.findall(r'\w+|[^\w\s]', html)
    return len(tokens)


def tag_difficulty(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "widget-*.json")))
    print(f"Found {len(files)} samples to tag")

    counts = {"simple": 0, "medium": 0, "complex": 0}

    for path in files:
        with open(path) as f:
            sample = json.load(f)

        token_count = count_tokens(sample["html"])

        if token_count < 50:
            difficulty = "simple"
        elif token_count <= 200:
            difficulty = "medium"
        else:
            difficulty = "complex"

        sample["difficulty"] = difficulty
        sample["token_count"] = token_count
        counts[difficulty] += 1

        out_path = os.path.join(output_dir, os.path.basename(path))
        with open(out_path, "w") as f:
            json.dump(sample, f)

    print(f"Done! simple: {counts['simple']}, medium: {counts['medium']}, complex: {counts['complex']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/deduped")
    parser.add_argument("--output_dir", type=str, default="./data/tagged")
    args = parser.parse_args()
    tag_difficulty(args.input_dir, args.output_dir)
