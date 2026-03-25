"""
Extract training data from the HuggingFace widget2code-benchmark train split.

Uses local git-cloned images at data/widget2code-benchmark/train/*.png,
sends each to GPT-5.3 to reverse-engineer React+Tailwind TSX.

Usage:
    python -m data.collection.collect_hf_extraction --num_samples 1820 --output_dir ./output/raw --workers 8
"""
import json
import os
import glob
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

HF_TRAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "widget2code-benchmark", "train")

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


def generate_tsx_from_image(image_path: str, model: str = "gemini-3-flash-preview") -> str:
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    response = client.models.generate_content(
        model=model,
        contents=[
            SYSTEM_PROMPT + "\n\n" + USER_PROMPT,
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    code = response.text.strip()
    return strip_code_fences(code)


def process_one(image_path: str, output_dir: str, model: str = "gemini-3-flash-preview") -> dict | None:
    basename = os.path.splitext(os.path.basename(image_path))[0]
    widget_id = f"hf-{basename}"
    output_path = os.path.join(output_dir, f"{widget_id}.json")

    if os.path.exists(output_path):
        return None

    try:
        tsx = generate_tsx_from_image(image_path, model=model)

        if "export default" not in tsx:
            print(f"  Bad output for {widget_id}, skipping")
            return None

        result = {
            "widget_id": widget_id,
            "screenshot_path": os.path.abspath(image_path),
            "tsx": tsx,
        }

        with open(output_path, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def extract(num_samples: int, output_dir: str, model: str = "gemini-3-flash-preview", workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    hf_dir = os.path.abspath(HF_TRAIN_DIR)
    all_images = sorted(glob.glob(os.path.join(hf_dir, "*.png")))
    print(f"Found {len(all_images)} images in {hf_dir}")

    if not all_images:
        print("No images found. Clone the dataset first:")
        print("  git clone https://huggingface.co/datasets/Djanghao/widget2code-benchmark data/widget2code-benchmark")
        return

    if num_samples < len(all_images):
        random.seed(42)
        all_images = random.sample(all_images, num_samples)

    already_done = len([f for f in os.listdir(output_dir) if f.startswith("hf-") and f.endswith(".json")])
    print(f"Already processed: {already_done}, total to process: {len(all_images)}")

    succeeded = already_done
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, img, output_dir, model): img for img in all_images}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                succeeded += 1
            else:
                img = futures[future]
                basename = os.path.splitext(os.path.basename(img))[0]
                widget_id = f"hf-{basename}"
                if not os.path.exists(os.path.join(output_dir, f"{widget_id}.json")):
                    failed += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(all_images)} | succeeded: {succeeded} | failed: {failed}")

    print(f"\nDone! {succeeded} succeeded, {failed} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=1820)
    parser.add_argument("--output_dir", type=str, default="./output/raw")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    extract(args.num_samples, args.output_dir, args.model, args.workers)
