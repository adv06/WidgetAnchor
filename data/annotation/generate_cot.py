"""
Overall goal is to output annotations given produced image and ground truth
Usage:
    python -m data.annotation.generate_cot --input_dir ./data/tagged --output_dir ./data/annotated --workers 8
"""
import json
import os
import glob
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

COT_SYSTEM_PROMPT = (
    "You are given:\n"
    "1. A screenshot of a UI widget\n"
    "2. The ground-truth React+Tailwind component code that produces this widget\n\n"
    "Generate a structured reasoning trace that decomposes the widget into:\n"
    "1. Structure Analysis — what type of widget, component hierarchy\n"
    "2. Layout Plan — flex/grid, dimensions, padding, margins, gaps (Tailwind classes)\n"
    "3. Color & Style — exact hex colors via `bg-[#hex]`, border-radius, shadows\n"
    "4. Typography — text sizes, font weights, leading, tracking\n"
    "5. Implementation Plan — React component structure outline\n\n"
    "Then include the ground-truth code.\n\n"
    "Format your output EXACTLY as:\n"
    "<think>\n"
    "## 1. Structure Analysis\n"
    "[your analysis]\n\n"
    "## 2. Layout Plan\n"
    "[your analysis]\n\n"
    "## 3. Color & Style Extraction\n"
    "[your analysis]\n\n"
    "## 4. Typography & Legibility\n"
    "[your analysis]\n\n"
    "## 5. Implementation Plan\n"
    "[your analysis]\n"
    "</think>\n"
    "<code>\n"
    "[the ground-truth React component]\n"
    "</code>\n\n"
    "Be precise with hex color values, Tailwind classes, and component props. "
    "The reasoning should directly map to the code that follows."
)


def generate_cot(screenshot_path: str, tsx: str, model: str = "gpt-4o") -> str:
    with open(screenshot_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode() # convert image to base 64 encoded string

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": f"Here is the ground-truth React component:\n\n```tsx\n{tsx}\n```\n\nGenerate the structured reasoning trace for this widget."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    return response.choices[0].message.content.strip()


def process_one(path: str, output_dir: str, model: str) -> dict | None:
    with open(path) as f:
        sample = json.load(f) # path contains ground truth tsx and widget screenshot

    widget_id = sample["widget_id"]
    out_path = os.path.join(output_dir, f"{widget_id}.json")

    # resumability
    if os.path.exists(out_path):
        return None

    try:
        cot_output = generate_cot(sample["screenshot_path"], sample["tsx"], model=model)

        # validate that output has both <think> and <code> tags
        if "<think>" not in cot_output or "<code>" not in cot_output:
            print(f"  Bad format for {widget_id}, skipping")
            return None

        sample["cot"] = cot_output

        with open(out_path, "w") as f:
            json.dump(sample, f)

        return sample
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def annotate(input_dir: str, output_dir: str, model: str = "gpt-4o", workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(files)} samples to annotate")

    already_done = len(glob.glob(os.path.join(output_dir, "*.json")))
    print(f"Already annotated: {already_done}")

    succeeded = already_done
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool: # multiple threads to generate cot
        futures = {pool.submit(process_one, p, output_dir, model): p for p in files}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                succeeded += 1
            elif not os.path.exists(os.path.join(output_dir, os.path.basename(futures[future]))):
                failed += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(files)} | annotated: {succeeded} | failed: {failed}")

    print(f"\nDone! {succeeded} annotated, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/tagged")
    parser.add_argument("--output_dir", type=str, default="./data/annotated")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    annotate(args.input_dir, args.output_dir, args.model, args.workers)
