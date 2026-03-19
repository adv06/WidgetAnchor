"""
Generate React code from widget2code dataset screenshots using GPT-5.4 vision.

Pipeline:
1. Load widget screenshots from widget2code synthetic dataset
2. Feed each screenshot to GPT-5.4 vision → self-contained React component
3. Render the generated React code via Playwright → screenshot
4. Validate with Gemini — send original + rendered screenshot, filter low-quality pairs
5. Save image-code pairs as train/val JSON
"""
import json
import os
import random
import base64
import glob
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from reward.programmatic import render_react_to_image

load_dotenv()

client = OpenAI()
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

WIDGET2CODE_DATA_DIR = "/shared/houston/widget2code/widget2code-sft/widget-factory-synthetic-data"

SYSTEM_PROMPT = (
    "You are an expert React developer. Given a screenshot of a UI widget, "
    "generate a single self-contained React functional component named App that recreates the widget as closely as possible. "
    "Use inline style objects for all CSS. Do not include import statements (React is available globally). "
    "Use only standard HTML elements (div, span, button, input, etc.) — no external component libraries. "
    "The component should be the default export. "
    "Return ONLY the component code, no markdown, no explanation, no code fences."
)

USER_PROMPT = "Recreate this widget as a self-contained React component. Match the layout, colors, typography, and spacing as closely as possible."


def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def load_widget_paths(data_dir: str, num_samples: int = None) -> list[dict]:
    """Find all widget directories with both a screenshot and JSX file."""
    widget_dirs = sorted(glob.glob(os.path.join(data_dir, "widget-*")))
    samples = []
    for d in widget_dirs:
        screenshot = os.path.join(d, "6-output.png")
        if os.path.exists(screenshot):
            samples.append({
                "widget_dir": d,
                "widget_id": os.path.basename(d),
                "screenshot_path": screenshot,
            })
    if num_samples and num_samples < len(samples):
        random.seed(42)
        samples = random.sample(samples, num_samples)
    return samples


def generate_react_from_image(image_path: str) -> str:
    """Send a widget screenshot to GPT-4o vision and get React code back."""
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = client.chat.completions.create(
        model="gpt-5.4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]},
        ],
        temperature=0.7,
        max_tokens=4096,
    )
    code = response.choices[0].message.content.strip()
    return strip_code_fences(code)


GEMINI_VALIDATION_PROMPT = (
    "You are evaluating how well a generated React widget matches the original design. "
    "The first image is the ORIGINAL widget screenshot. The second image is the RENDERED output of generated React code. "
    "Score how closely the second image matches the first on a scale of 0.0 to 1.0, considering: "
    "layout structure, color scheme, component placement, typography, and overall visual appearance. "
    "Ignore minor differences like exact font rendering or slight spacing variations. "
    "0.0 = completely different, 0.5 = same type but different style, 1.0 = nearly identical. "
    "Reply with ONLY a decimal number between 0.0 and 1.0."
)

GEMINI_QUALITY_THRESHOLD = 0.4


def validate_with_gemini(original_image_path: str, rendered_image_bytes: bytes) -> float:
    """Use Gemini to score how well the rendered React matches the original widget."""
    import re

    with open(original_image_path, "rb") as f:
        original_bytes = f.read()

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Content(role="user", parts=[
                types.Part.from_text(GEMINI_VALIDATION_PROMPT),
                types.Part.from_bytes(data=original_bytes, mime_type="image/png"),
                types.Part.from_bytes(data=rendered_image_bytes, mime_type="image/png"),
            ]),
        ],
    )

    score_text = response.text.strip()
    try:
        score = float(score_text)
    except ValueError:
        match = re.search(r"\d+\.?\d*", score_text)
        score = float(match.group()) if match else 0.0
    return max(0.0, min(1.0, score))


def generate_dataset(num_samples=500, output_dir="./data"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    print(f"Loading widget screenshots from {WIDGET2CODE_DATA_DIR}...")
    widget_samples = load_widget_paths(WIDGET2CODE_DATA_DIR, num_samples)
    print(f"Found {len(widget_samples)} widgets to process")

    dataset = []
    for i, sample in enumerate(widget_samples):
        print(f"[{i+1}/{len(widget_samples)}] Processing: {sample['widget_id']}")

        try:
            # Generate React code from the screenshot
            react_code = generate_react_from_image(sample["screenshot_path"])

            # Render the generated React code to get a new screenshot
            rendered_bytes = render_react_to_image(react_code)

            # Validate with Gemini: compare original vs rendered
            gemini_score = validate_with_gemini(sample["screenshot_path"], rendered_bytes)
            print(f"  Gemini validation score: {gemini_score:.2f}", end="")

            if gemini_score < GEMINI_QUALITY_THRESHOLD:
                print(" — SKIPPED (below threshold)")
                continue
            print(" — OK")

            rendered_path = f"{output_dir}/images/{sample['widget_id']}.png"
            with open(rendered_path, "wb") as f:
                f.write(rendered_bytes)

            # The prompt for training: describe the task
            prompt = (
                "Generate a single self-contained React component that recreates "
                "the widget shown in the reference image. Use inline styles, no imports. "
                "Export as default function App."
            )

            dataset.append({
                "id": i,
                "prompt": prompt,
                "react_code": react_code,
                "image_path": rendered_path,
                "source_image_path": sample["screenshot_path"],
                "widget_id": sample["widget_id"],
                "gemini_score": gemini_score,
            })
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # train/val split (90/10)
    random.seed(42)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.9)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(f"{output_dir}/val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    num_filtered = len(widget_samples) - len(dataset)
    avg_score = sum(d["gemini_score"] for d in dataset) / len(dataset) if dataset else 0
    print(f"\nDone! {len(train_data)} train, {len(val_data)} val samples saved to {output_dir}/")
    print(f"Filtered {num_filtered}/{len(widget_samples)} samples below Gemini threshold ({GEMINI_QUALITY_THRESHOLD})")
    print(f"Average Gemini score of kept samples: {avg_score:.3f}")
    return train_data, val_data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=500, help="Number of widgets to process")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory")
    args = parser.parse_args()
    generate_dataset(num_samples=args.num_samples, output_dir=args.output_dir)
