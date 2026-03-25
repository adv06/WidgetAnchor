"""
Quick CLI to send an image to an SGLang-served model and print the raw output.

Usage:
    python -m inference.run_inference \
        --model-url http://localhost:8000/v1 \
        --model-name /shared/advey/glm-4.1v-9b-thinking-sft-merged \
        --image /path/to/image.png
"""

import argparse
import base64
import struct
import requests


SYSTEM_PROMPT = (
    "You are a high-fidelity UI reproduction expert. Given a screenshot of a UI widget, "
    "generate a React functional component with Tailwind CSS that visually matches it as closely as possible.\n\n"
    "A widget is NOT merely a spatial arrangement of components — it is a compact presentation of information. "
    "First infer what information the widget communicates, then plan how to realize it visually.\n\n"
    "## Rules\n"
    "- Output a single default-exported React component: `export default function Widget() { ... }`\n"
    "- Use Tailwind CSS utility classes for all styling\n"
    "- Colors: use exact hex values via arbitrary-value syntax (e.g. `bg-[#3B82F6]`, `text-[#1F2937]`)\n"
    "- Match layout exactly: flex/grid direction, alignment, spacing, gaps, padding, margins\n"
    "- Match typography: text size, font weight, tracking, leading\n"
    "- Match border-radius, shadow, opacity, gradients\n"
    "- Charts/gauges/progress: use recharts (import from 'recharts')\n"
    "- Icons: use lucide-react (import from 'lucide-react')\n"
    "- The root container must match the widget dimensions given in the user message\n"
    "- Text content must be character-perfect — copy every word and number exactly\n\n"
    "## Output format\n"
    "<think>\n"
    "## 1. Widget Information Specification\n"
    "### 1a. Widget Type\n[presentation archetype: metric-summary, chart-summary, feed-summary, etc.]\n"
    "### 1b. Semantic Fields\n[{name, semantic_role, value, data_type} for each field — use roles like primary-metric, entity-label, trend-indicator, status-indicator, supporting-qualifier, temporal-anchor, action-trigger, data-series]\n"
    "### 1c. Relations\n[typed semantic relations: entity-of, qualifier-of, trend-of, supports, compares-to, grouped-with, repeated-as, visualizes]\n"
    "### 1d. Salience\n[field priority scores 0.0-1.0 for glanceability]\n"
    "### 1e. Grouping\n[partition fields into meaningful presentation units]\n"
    "### 1f. Presentation Intent\n[density: compact/balanced/relaxed, focus: single-focus/multi-signal, lead: metric-led/icon-led/trend-led/chart-led/content-led]\n\n"
    "## 2. Presentation Realization\n"
    "### 2a. Layout Topology\n[how semantic groups map to spatial structure]\n"
    "### 2b. Geometric Allocation\n[spacing, padding, sizing in Tailwind classes]\n"
    "### 2c. Style Tokens\n[exact hex colors, border-radius, shadows, font sizes/weights]\n"
    "### 2d. Field-to-Visual Binding\n[how each semantic field maps to a visual element]\n"
    "</think>\n"
    "<code>[complete React component]</code>"
)


def get_image_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", header[16:24])
    return (800, 600)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-url", type=str, required=True,
                        help="SGLang API base URL (e.g. http://localhost:8000/v1)")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name as registered in SGLang")
    parser.add_argument("--image", type=str, required=True,
                        help="Path to widget screenshot")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    w, h = get_image_size(args.image)

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    payload = {
        "model": args.model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": f"Widget dimensions: {w}x{h}px. Recreate this widget as a React component with Tailwind CSS."},
                ],
            },
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    resp = requests.post(f"{args.model_url}/chat/completions", json=payload, timeout=300)
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    print(text)


if __name__ == "__main__":
    main()
