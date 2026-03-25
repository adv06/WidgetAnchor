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
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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


def generate_widget(category: str, style: str, density: str, layout: str, model: str = "gemini-3-flash-preview") -> str:
    prompt = (
        f"Create a {category} widget.\n"
        f"Style: {style}\n"
        f"Density: {density}\n"
        f"Layout: {layout}\n"
        f"Make it visually complete and realistic with sample data."
    )

    response = client.models.generate_content(
        model=model,
        contents=SYSTEM_PROMPT + "\n\n" + prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=4096,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    code = response.text.strip()
    return strip_code_fences(code)


def generate_one(sample_id: int, output_dir: str, model: str = "gemini-3-flash-preview") -> dict | None:
    """Generate TSX via API (thread-safe). Rendering is deferred to render_verify."""
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

        result = {
            "widget_id": widget_id,
            "tsx": tsx,
            "category": category,
            "needs_render": True,
        }

        with open(output_path, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def render_batch(output_dir: str):
    """Render screenshots for all synthetic samples that need it (sequential, Playwright not thread-safe)."""
    from reward.programmatic import render_tsx_to_image

    screenshots_dir = os.path.join(output_dir, "screenshots")
    os.makedirs(screenshots_dir, exist_ok=True)

    json_files = sorted([f for f in os.listdir(output_dir) if f.startswith("synthetic-") and f.endswith(".json")])
    rendered = 0
    failed = 0

    for jf in json_files:
        path = os.path.join(output_dir, jf)
        with open(path) as f:
            sample = json.load(f)

        screenshot_path = os.path.join(screenshots_dir, f"{sample['widget_id']}.png")

        if os.path.exists(screenshot_path) and "screenshot_path" in sample:
            continue

        try:
            png_bytes = render_tsx_to_image(sample["tsx"])
            with open(screenshot_path, "wb") as f:
                f.write(png_bytes)

            sample["screenshot_path"] = os.path.abspath(screenshot_path)
            sample.pop("needs_render", None)
            with open(path, "w") as f:
                json.dump(sample, f)
            rendered += 1
        except Exception as e:
            print(f"  Render failed {sample['widget_id']}: {e}")
            failed += 1

        if (rendered + failed) % 100 == 0:
            print(f"  Rendered: {rendered} | failed: {failed} / {rendered + failed}")

    print(f"Rendering done: {rendered} succeeded, {failed} failed")


def generate(num_samples: int, output_dir: str, model: str = "gemini-3-flash-preview", workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "screenshots"), exist_ok=True)

    already_done = len([f for f in os.listdir(output_dir) if f.startswith("synthetic-") and f.endswith(".json")])
    print(f"Generating {num_samples} synthetic widgets (already done: {already_done})")

    succeeded = already_done
    failed = 0

    random.seed(42)

    # Phase 1: parallel API calls (thread-safe)
    print(f"Phase 1: Generating TSX via API ({workers} workers)...")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, i, output_dir, model): i for i in range(num_samples)}
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

    print(f"\nAPI generation done: {succeeded} succeeded, {failed} failed")

    # Phase 2: sequential rendering (Playwright not thread-safe)
    print(f"Phase 2: Rendering screenshots (sequential)...")
    render_batch(output_dir)

    print(f"Synthetic data saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--output_dir", type=str, default="./output/raw")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    generate(args.num_samples, args.output_dir, args.model, args.workers)
