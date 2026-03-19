"""

reverse-engineer self-contained HTML/CSS using a SOTA LLM.
Usage:
    python -m data.collection.collect_widget_factory --num_samples 50000 --output_dir ./data/raw
    python -m data.collection.collect_widget_factory --num_samples 100 --output_dir ./data/raw  # test run
"""
import json
import os
import glob
import random
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

WIDGET2CODE_DATA_DIR = "/shared/houston/widget2code/widget2code-sft/widget-factory-sft-models/50000-allicons-qwen3-max-all"

SYSTEM_PROMPT = (
    "You are an expert frontend developer. Given a screenshot of a UI widget, "
    "generate a React functional component with Tailwind CSS that recreates the widget as closely as possible.\n\n"
    "Requirements:\n"
    "- Output a single default-exported React component: `export default function Widget() { ... }`\n"
    "- Use Tailwind CSS utility classes for all styling\n"
    "- Colors: use exact hex values via arbitrary-value syntax (e.g. `bg-[#3B82F6]`)\n"
    "- Charts/gauges/progress: use recharts (import from 'recharts')\n"
    "- Icons: use lucide-react (import from 'lucide-react')\n"
    "- Match spacing, padding, margins, border-radius, shadows precisely\n"
    "- Match font sizes, weights, and line heights\n"
    "- The widget should be centered on the page\n"
    "- Return ONLY the component code, no markdown, no explanation, no code fences"
)

USER_PROMPT = (
    "Recreate this widget as a React functional component with Tailwind CSS. "
    "Match the layout, colors, typography, spacing, and visual appearance as closely as possible."
)


def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def load_widget_paths(data_dir: str, num_samples: int = None) -> list[dict]:
    widget_dirs = sorted(glob.glob(os.path.join(data_dir, "widget-*")))
    samples = []
    for d in widget_dirs:
        screenshot = os.path.join(d, "6-output.png")
        if os.path.exists(screenshot):
            samples.append({
                "widget_id": os.path.basename(d),
                "screenshot_path": screenshot,
            })
    if num_samples and num_samples < len(samples):
        random.seed(42)
        samples = random.sample(samples, num_samples)
    return samples


def generate_tsx_from_image(image_path: str, model: str = "gpt-4o") -> str:
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]},
        ],
        temperature=0.3,
        max_tokens=4096,
    )
    code = response.choices[0].message.content.strip()
    return strip_code_fences(code)


def process_one(sample: dict, output_dir: str, model: str) -> dict | None:
    widget_id = sample["widget_id"]
    screenshot_path = sample["screenshot_path"]
    output_path = os.path.join(output_dir, f"{widget_id}.json")

    # skip if already processed (for resumability)
    if os.path.exists(output_path):
        return None

    try:
        tsx = generate_tsx_from_image(screenshot_path, model=model)

        # basic sanity: must contain export default
        if "export default" not in tsx:
            return None

        result = {
            "widget_id": widget_id,
            "screenshot_path": screenshot_path,
            "tsx": tsx,
        }

        with open(output_path, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def collect(num_samples: int, output_dir: str, model: str = "gpt-4o", workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading widget screenshots from {WIDGET2CODE_DATA_DIR}...")
    samples = load_widget_paths(WIDGET2CODE_DATA_DIR, num_samples)
    print(f"Found {len(samples)} widgets to process")

    # count already done
    already_done = len(glob.glob(os.path.join(output_dir, "widget-*.json")))
    print(f"Already processed: {already_done}, remaining: {len(samples) - already_done}")

    succeeded = already_done
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, s, output_dir, model): s for s in samples}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                succeeded += 1
            else:
                # could be skipped (already done) or failed
                if not os.path.exists(os.path.join(output_dir, f"{futures[future]['widget_id']}.json")):
                    failed += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(samples)} | succeeded: {succeeded} | failed: {failed}")

    print(f"\nDone! {succeeded} succeeded, {failed} failed")
    print(f"Raw data saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=50000)
    parser.add_argument("--output_dir", type=str, default="./data/raw")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    collect(args.num_samples, args.output_dir, args.model, args.workers)
