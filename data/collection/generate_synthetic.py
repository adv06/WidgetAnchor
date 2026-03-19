"""
Uses GPT-4o to generate React+Tailwind TSX components for random widget categories,
then renders them to screenshots. Mirrors the UI2Code "reversed" strategy.

Usage:
    python -m data.collection.generate_synthetic --num_samples 10000 --output_dir ./output/raw --workers 8
    python -m data.collection.generate_synthetic --num_samples 3 --output_dir /tmp/test_syn --workers 1  # test run
"""
import json
import os
import random
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv
from reward.programmatic import render_tsx_to_image

load_dotenv()

client = OpenAI()

WIDGET_CATEGORIES = [
    "dashboard stats card",
    "line chart widget",
    "bar chart widget",
    "pie/donut chart widget",
    "stats panel with KPIs",
    "pricing table card",
    "notification/alert card",
    "settings toggle panel",
    "user profile card",
    "progress tracker/stepper",
    "calendar widget",
    "weather card",
    "file upload dropzone",
    "chat message bubble",
    "activity feed/timeline",
    "data table with pagination",
    "login/signup form",
    "search bar with filters",
    "navigation sidebar snippet",
    "breadcrumb navigation",
    "testimonial/review card",
    "feature comparison card",
    "onboarding checklist",
    "analytics summary card",
    "social media post card",
    "music/media player controls",
    "order summary card",
    "countdown timer widget",
    "rating/feedback widget",
    "kanban board column",
]

STYLE_PALETTES = [
    "modern minimal with neutral grays and a blue accent",
    "dark mode with deep navy background and cyan highlights",
    "vibrant gradient with purple-to-pink accents",
    "clean corporate with white background and green accents",
    "warm tones with orange and amber highlights",
    "pastel soft with rounded corners and light shadows",
    "high contrast with black background and yellow accents",
    "glassmorphism with semi-transparent layers and blur",
]

DENSITY_VARIANTS = ["sparse with lots of whitespace", "moderate density", "dense and information-rich"]

LAYOUT_VARIANTS = ["single column", "two-column side by side", "card with header and body sections", "stacked rows"]

SYSTEM_PROMPT = (
    "You are an expert frontend developer. Generate a React functional component with Tailwind CSS "
    "that implements a UI widget.\n\n"
    "Requirements:\n"
    "- Output a single default-exported React component: `export default function Widget() { ... }`\n"
    "- Use Tailwind CSS utility classes for all styling\n"
    "- Colors: use exact hex values via arbitrary-value syntax (e.g. `bg-[#3B82F6]`)\n"
    "- Charts/gauges/progress: use recharts (import from 'recharts')\n"
    "- Icons: use lucide-react (import from 'lucide-react')\n"
    "- Use realistic placeholder data (names, numbers, dates)\n"
    "- The widget should be self-contained and visually polished\n"
    "- The widget should be centered on the page\n"
    "- Return ONLY the component code, no markdown, no explanation, no code fences"
)


def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def generate_widget(category: str, style: str, density: str, layout: str, model: str = "gpt-4o") -> str:
    prompt = (
        f"Create a {category} widget.\n"
        f"Style: {style}\n"
        f"Density: {density}\n"
        f"Layout: {layout}\n"
        f"Make it visually complete and realistic with sample data."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
        max_tokens=4096,
    )
    code = response.choices[0].message.content.strip()
    return strip_code_fences(code)


def process_one(sample_id: int, output_dir: str, model: str) -> dict | None:
    widget_id = f"synthetic-{sample_id:05d}"
    output_path = os.path.join(output_dir, f"{widget_id}.json")

    if os.path.exists(output_path):
        return None

    category = random.choice(WIDGET_CATEGORIES)
    style = random.choice(STYLE_PALETTES)
    density = random.choice(DENSITY_VARIANTS)
    layout = random.choice(LAYOUT_VARIANTS)

    try:
        tsx = generate_widget(category, style, density, layout, model=model)

        if "export default" not in tsx:
            return None

        # Render to screenshot
        screenshots_dir = os.path.join(output_dir, "screenshots")
        screenshot_path = os.path.join(screenshots_dir, f"{widget_id}.png")

        png_bytes = render_tsx_to_image(tsx)
        with open(screenshot_path, "wb") as f:
            f.write(png_bytes)

        result = {
            "widget_id": widget_id,
            "screenshot_path": screenshot_path,
            "tsx": tsx,
            "category": category,
        }

        with open(output_path, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def generate(num_samples: int, output_dir: str, model: str = "gpt-4o", workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

    already_done = len([f for f in os.listdir(output_dir) if f.startswith("synthetic-") and f.endswith(".json")])
    print(f"Generating {num_samples} synthetic widgets (already done: {already_done})")

    succeeded = already_done
    failed = 0

    random.seed(42)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, i, output_dir, model): i for i in range(num_samples)}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                succeeded += 1
            else:
                sample_id = futures[future]
                widget_id = f"synthetic-{sample_id:05d}"
                if not os.path.exists(os.path.join(output_dir, f"{widget_id}.json")):
                    failed += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{num_samples} | succeeded: {succeeded} | failed: {failed}")

    print(f"\nDone! {succeeded} succeeded, {failed} failed")
    print(f"Synthetic data saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="./output/raw")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    generate(args.num_samples, args.output_dir, args.model, args.workers)
