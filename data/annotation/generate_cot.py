"""
Overall goal is to output annotations given produced image and ground truth.
Uses multi-provider fallback: Gemini models + OpenAI gpt-5.3.

Usage:
    python -m data.annotation.generate_cot --input_dir ./data/tagged --output_dir ./data/annotated --workers 8
"""
import json
import os
import glob
import time
import re
import base64
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai import types
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
openai_client = OpenAI()

# providers to rotate through: (provider, model)
PROVIDERS = [
    ("gemini", "gemini-2.5-flash"),
    ("openai", "gpt-5.3-chat-latest"),
]
_provider_idx = 0

COT_SYSTEM_PROMPT = (
    "You are an expert at analyzing UI widgets as information display systems.\n\n"
    "A widget is NOT merely a spatial arrangement of components — it is a compact presentation of information. "
    "A UI tree like `Column > Text('AAPL') > Text('$184') > Text('+1.2%')` loses the semantics: "
    "Entity=AAPL, Metric=price, Metric=change. Your job is to recover that semantic layer.\n\n"
    "Given a widget screenshot and the ground-truth React+Tailwind code that produces it, "
    "generate a structured reasoning trace that FIRST infers the widget's information specification, "
    "THEN plans the presentation realization, THEN includes the code.\n\n"
    "## Reasoning structure\n\n"
    "### 1. Widget Information Specification (s)\n\n"
    "**1a. Widget Type (t)** — the presentation archetype / communicative role. "
    "NOT a visual class, but a presentation class. Examples: metric-summary, weather-summary, "
    "stock-summary, progress-summary, schedule-summary, chart-summary, media-control-summary, "
    "feed-summary, form-capture, comparison-summary, notification-alert.\n\n"
    "**1b. Semantic Fields (F)** — the set of information fields the widget communicates. "
    "These encode WHAT is being communicated, not WHERE it sits on screen. "
    "Use semantic roles like: primary-metric, entity-label, trend-indicator, status-indicator, "
    "supporting-qualifier, temporal-anchor, action-trigger, data-series, category-label, "
    "progress-value, threshold, comparison-baseline. "
    "For each field, give: {name, semantic_role, value, data_type}.\n\n"
    "**1c. Relations (R)** — typed semantic relations between fields. "
    "Relation types: entity-of, qualifier-of, trend-of, supports, compares-to, "
    "grouped-with, repeated-as, visualizes, aggregates, conditions. "
    "Example: '+1.2% is the trend-of AAPL price'. These are SEMANTIC, not structural.\n\n"
    "**1d. Salience (π)** — field priority for glanceability. "
    "Widgets are designed for at-a-glance comprehension, so not all fields are equally important. "
    "Assign each field a salience score 0.0–1.0. The primary metric or key information should be highest.\n\n"
    "**1e. Grouping (g)** — partition of fields into meaningful presentation units. "
    "NOT a component tree or DOM structure. Group by semantic meaning. "
    "Example: {header-group: [entity-label, status-icon]}, {metric-group: [primary-metric, trend-indicator]}.\n\n"
    "**1f. Presentation Intent (κ)** — the compactness regime and emphasis strategy.\n"
    "  - Density: compact / balanced / relaxed\n"
    "  - Focus: single-focus / multi-signal\n"
    "  - Lead: metric-led / icon-led / trend-led / chart-led / content-led\n\n"
    "### 2. Presentation Realization (z)\n\n"
    "**2a. Layout Topology** — how semantic groups map to spatial structure "
    "(vertical stack, horizontal split, grid, card-with-header-body, sidebar+main, etc.)\n\n"
    "**2b. Geometric Allocation** — spacing, padding, sizing strategy in Tailwind classes.\n\n"
    "**2c. Style Tokens** — exact colors (hex via `bg-[#hex]`), border-radius, shadows, "
    "font sizes, weights, line heights.\n\n"
    "**2d. Field-to-Visual Binding** — how each semantic field maps to a visual element "
    "(text node, icon, chart element, badge, progress bar, etc.).\n\n"
    "### 3. Code\n"
    "The ground-truth React component.\n\n"
    "## Output format\n"
    "Wrap everything EXACTLY as:\n"
    "<think>\n"
    "## 1. Widget Information Specification\n"
    "### 1a. Widget Type\n[...]\n"
    "### 1b. Semantic Fields\n[...]\n"
    "### 1c. Relations\n[...]\n"
    "### 1d. Salience\n[...]\n"
    "### 1e. Grouping\n[...]\n"
    "### 1f. Presentation Intent\n[...]\n\n"
    "## 2. Presentation Realization\n"
    "### 2a. Layout Topology\n[...]\n"
    "### 2b. Geometric Allocation\n[...]\n"
    "### 2c. Style Tokens\n[...]\n"
    "### 2d. Field-to-Visual Binding\n[...]\n"
    "</think>\n"
    "<code>\nYOU MUST INCLUDE THE COMPLETE REACT COMPONENT CODE HERE — copy every line from the ground-truth code provided. DO NOT use a placeholder or summary.\n</code>\n\n"
    "IMPORTANT:\n"
    "- The <code> block MUST contain the FULL ground-truth React component, copied verbatim. "
    "NEVER write a placeholder like '[the ground-truth React component]'.\n"
    "- Semantic fields must use SEMANTIC ROLES (primary-metric, entity-label, trend-indicator), "
    "NOT component labels (title text, subtitle text, left icon).\n"
    "- Infer the INFORMATION STRUCTURE the widget communicates, not just its visual layout.\n"
    "- The same information specification could yield multiple valid presentations — "
    "acknowledge this one-to-many mapping.\n"
    "- Be precise with hex colors, Tailwind classes, and salience scores."
)


def _next_provider():
    """Round-robin provider selection to spread rate limits."""
    global _provider_idx
    provider, model = PROVIDERS[_provider_idx % len(PROVIDERS)]
    _provider_idx += 1
    return provider, model


def _call_gemini(model: str, img_bytes: bytes, user_text: str) -> str:
    response = gemini_client.models.generate_content(
        model=model,
        contents=[
            COT_SYSTEM_PROMPT + "\n\n" + user_text,
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
        ],
        config=types.GenerateContentConfig(
            max_output_tokens=8192,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    text = response.text
    if text is None:
        raise RuntimeError("Gemini returned None")
    return text.strip()


def _call_openai(model: str, img_bytes: bytes, user_text: str) -> str:
    img_b64 = base64.b64encode(img_bytes).decode()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": user_text},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
            ]},
        ],
        max_completion_tokens=8192,
    )
    return response.choices[0].message.content.strip()


def generate_cot(screenshot_path: str, tsx: str) -> str:
    with open(screenshot_path, "rb") as f:
        img_bytes = f.read()

    user_text = (
        f"Here is the ground-truth React component:\n\n```tsx\n{tsx}\n```\n\n"
        "Analyze this widget screenshot and generate the structured reasoning trace "
        "following the Widget Information Specification framework. "
        "First infer what information the widget communicates and its semantic structure, "
        "then describe how that information is realized visually, then include the code."
    )

    # try each provider, cycling on rate limits
    last_error = None
    for attempt in range(len(PROVIDERS) * 2):
        provider, model = _next_provider()
        try:
            if provider == "gemini":
                return _call_gemini(model, img_bytes, user_text)
            else:
                return _call_openai(model, img_bytes, user_text)
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                wait = min(30, 2 ** (attempt // len(PROVIDERS)) * 2)
                print(f"    Rate limited on {provider}/{model}, trying next provider (wait {wait}s)...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"All providers exhausted: {last_error}")


def _normalize_format(cot_output: str) -> str:
    """Normalize output format: convert markdown fences to <code> tags if needed."""
    if "<code>" not in cot_output and "```" in cot_output:
        fence_match = re.search(r"```(?:tsx|jsx|javascript|react)?\s*\n(.*?)```", cot_output, re.DOTALL)
        if fence_match:
            code_content = fence_match.group(1).strip()
            cot_output = cot_output[:fence_match.start()] + "<code>\n" + code_content + "\n</code>" + cot_output[fence_match.end():]
    return cot_output


def process_one(path: str, output_dir: str) -> dict | None:
    with open(path) as f:
        sample = json.load(f)

    widget_id = sample["widget_id"]
    out_path = os.path.join(output_dir, f"{widget_id}.json")

    if os.path.exists(out_path):
        return None

    try:
        cot_output = generate_cot(sample["screenshot_path"], sample["tsx"])
        cot_output = _normalize_format(cot_output)

        think_open = cot_output.find("<think>")
        think_close = cot_output.find("</think>")
        code_open = cot_output.find("<code>")
        code_close = cot_output.find("</code>")

        if any(pos == -1 for pos in [think_open, think_close, code_open, code_close]):
            print(f"  Bad format for {widget_id}: missing tags, skipping")
            return None
        if not (think_open < think_close < code_open < code_close):
            print(f"  Bad format for {widget_id}: tags not properly nested, skipping")
            return None

        # validate that <code> contains actual code, not a placeholder
        code_content = cot_output[code_open + len("<code>"):code_close].strip()
        if "import" not in code_content and "export default" not in code_content:
            print(f"  Bad code for {widget_id}: placeholder detected, patching with original tsx")
            # replace placeholder with the actual tsx from the sample
            cot_output = cot_output[:code_open] + "<code>\n" + sample["tsx"] + "\n</code>" + cot_output[code_close + len("</code>"):]

        sample["cot"] = cot_output

        with open(out_path, "w") as f:
            json.dump(sample, f)

        return sample
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def annotate(input_dir: str, output_dir: str, workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    print(f"Found {len(files)} samples to annotate")
    print(f"Providers: {[(p,m) for p,m in PROVIDERS]}")

    already_done = len(glob.glob(os.path.join(output_dir, "*.json")))
    print(f"Already annotated: {already_done}")

    succeeded = already_done
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, p, output_dir): p for p in files}
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
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    annotate(args.input_dir, args.output_dir, args.workers)
