"""
SFT Benchmark: evaluate SFT (or base) model on widget-factory test_split.

Two-phase pipeline:
  Phase 1 (this script): Inference via SGLang API → extract TSX → render PNG
  Phase 2 (subprocess):  widget-factory eval.py for metrics

Usage:
    # Phase 1+2 combined:
    python -m evaluation.run_sft_benchmark \
        --model-url http://localhost:8000/v1 \
        --model-name /shared/advey/glm-4.1v-9b-thinking-sft-merged \
        --test-dir /home/advey/widget-factory/test_split \
        --output-dir /shared/advey/benchmark-results/sft-benchmark \
        --limit 10

    # Phase 1 only (inference + render):
    python -m evaluation.run_sft_benchmark --skip-eval ...

    # Phase 2 only (evaluate existing renders):
    python -m evaluation.run_sft_benchmark --eval-only \
        --test-dir /home/advey/widget-factory/test_split \
        --output-dir /shared/advey/benchmark-results/sft-benchmark
"""

import argparse
import asyncio
import base64
import glob
import json
import os
import re
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ── Reuse WidgetAnchor's rendering pipeline ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reward.programmatic import render_tsx_to_image

# ── Build set of valid lucide-react exports for import validation ────────────
_LUCIDE_EXPORTS: set[str] | None = None

def _get_lucide_exports() -> set[str]:
    global _LUCIDE_EXPORTS
    if _LUCIDE_EXPORTS is None:
        lucide_esm = Path(__file__).resolve().parent.parent / "render" / "node_modules" / "lucide-react" / "dist" / "esm" / "lucide-react.js"
        if lucide_esm.exists():
            text = lucide_esm.read_text()
            _LUCIDE_EXPORTS = set(re.findall(r"export \{ default as (\w+)", text))
        else:
            _LUCIDE_EXPORTS = set()
    return _LUCIDE_EXPORTS

# ── Prompt format (from training/sft.py) ─────────────────────────────────────
# ── System prompt: MUST match training exactly (from training/sft.py) ─────────
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


def _get_image_size(path: str) -> tuple[int, int]:
    """Read PNG width/height from IHDR chunk."""
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", header[16:24])
    return (800, 600)


def _unwrap_chat_template(text: str) -> str:
    """Extract raw text from chat template dict format if present."""
    match = re.search(r"\[?\{['\"]type['\"]:\s*['\"]text['\"],\s*['\"]text['\"]:\s*['\"](.+)", text, re.DOTALL)
    if match:
        inner = match.group(1)
        inner = inner.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n")
        inner = re.sub(r"['\"]?\s*\}?\]?\s*$", "", inner)
        return inner
    return text


def _is_real_code(code: str) -> bool:
    """Return True if the code looks like a real React component, not a placeholder."""
    code_stripped = code.strip()
    placeholder_patterns = [
        r"^\[.*(?:provided|above|given|component).*\]$",
        r"^<\s*the\s+",
        r"^(?:the|your|same)\s+(?:provided|above|given|complete)",
        r"^\.\.\.",
    ]
    for pat in placeholder_patterns:
        if re.match(pat, code_stripped, re.IGNORECASE):
            return False
    if "export" not in code and "function" not in code and "=>" not in code:
        return False
    return True


def extract_code(text: str) -> str | None:
    """Extract TSX/JSX code from model output.

    Tries in order:
    1. <code>...</code> tags — try ALL blocks, pick best real one
    2. React component inside <script> tags (base model HTML format)
    3. Markdown fenced code blocks (any language tag)
    4. Unclosed <code> tag (truncated output)

    Then applies cleanup to fix common issues.
    """
    text = _unwrap_chat_template(text)
    # Unescape literal \\n, \\', \\" that SGLang may return
    text = text.replace("\\n", "\n").replace("\\'", "'").replace('\\"', '"')

    code = None

    # 1. Try ALL closed <code>...</code> blocks — pick the best real one
    code_blocks = re.findall(r"<code>(.*?)</code>", text, re.DOTALL)
    if code_blocks:
        # Filter to real code blocks (not placeholders)
        real_blocks = [b.strip() for b in code_blocks if _is_real_code(b.strip())]
        if real_blocks:
            # Prefer the one with "export default", else longest
            export_blocks = [b for b in real_blocks if "export default" in b]
            code = max(export_blocks or real_blocks, key=len)

    # 2. Try extracting React component from <script> tags (base model HTML output)
    if code is None:
        scripts = re.findall(r"<script[^>]*>(.*?)</script>", text, re.DOTALL)
        for script in scripts:
            # Look for a component function
            if ("function Widget" in script or "function App" in script or
                    "export default function" in script or "const Widget" in script):
                # Extract the component — rewrite lucide destructuring to import
                component = script.strip()
                # Replace `const { Icon1, Icon2 } = lucide;` with proper import
                component = re.sub(
                    r"(?:const|let|var)\s+\{([^}]+)\}\s*=\s*lucide\s*;?",
                    lambda m: f"import {{ {m.group(1).strip()} }} from 'lucide-react';",
                    component,
                )
                # Remove ReactDOM.render / createRoot calls
                component = re.sub(r"ReactDOM\.(?:render|createRoot)\(.*$", "", component, flags=re.DOTALL)
                # Ensure it has export default
                if "export default" not in component:
                    component = re.sub(
                        r"((?:const|function)\s+(?:Widget|App)\b)",
                        r"export default \1",
                        component,
                        count=1,
                    )
                if _is_real_code(component):
                    code = component
                    break

    # 3. Try markdown fenced code blocks (any language tag including html, none, etc.)
    if code is None:
        fenced = re.findall(r"```\w*\s*\n(.*?)```", text, re.DOTALL)
        if fenced:
            # Prefer the block that has "export default"
            candidates = [b.strip() for b in fenced if _is_real_code(b.strip())]
            if candidates:
                export_blocks = [b for b in candidates if "export default" in b]
                code = max(export_blocks or candidates, key=len)

    # 4. Handle truncated <code> (no closing tag)
    if code is None:
        match = re.search(r"<code>(.*)", text, re.DOTALL)
        if match:
            c = match.group(1).strip()
            if len(c) > 50 and _is_real_code(c):
                code = c

    if code is None:
        return None

    # Legacy check (kept for safety)
    if "export" not in code and "function" not in code and "=>" not in code:
        return None

    # ── Post-processing cleanup ──────────────────────────────────
    code = _cleanup_tsx(code)
    return code


def _strip_centering_wrapper(code: str) -> str:
    """Neutralize the outer centering wrapper that SFT adds around widgets.

    Replace dark bg colors on the outermost min-h-screen div with transparent,
    so the widget renders against a white background like the GT.
    Keep min-h-screen and flex centering intact so sizing is preserved.
    """
    return code


def _cleanup_tsx(code: str) -> str:
    """Fix common issues in extracted TSX code."""
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        # Remove CSS file imports (esbuild can't resolve them)
        if re.match(r"""^import\s+['"]\.\/.*\.css['"];?\s*//.*$""", line):
            continue
        if re.match(r"""^import\s+['"]\.\/.*\.css['"];?\s*$""", line):
            continue
        cleaned.append(line)
    code = "\n".join(cleaned)

    # Fix unescaped apostrophes in JSX string literals
    code = re.sub(
        r"'([^']*?[a-zA-Z])'(m|s|t|d|ll|re|ve)\b",
        r"'\1&apos;\2",
        code,
    )

    # Fix template literal classNames that confuse esbuild
    # e.g., className={`foo ${bar} baz`} with unclosed backticks
    # Replace with simple string concatenation where possible
    code = re.sub(
        r'className=\{`([^`]*)\b(data-\w+)=\{`([^`]*)`\}',
        lambda m: f'className="{m.group(1)}" {m.group(2)}="{m.group(3)}"',
        code,
    )

    # ── Strip min-h-screen centering wrapper ────────────────────
    # SFT often wraps the widget in a full-screen dark centering div.
    # Remove it so the inner widget renders at correct size.
    code = _strip_centering_wrapper(code)

    # ── Fix invalid lucide-react imports ─────────────────────────
    valid = _get_lucide_exports()
    if valid:
        code = _fix_lucide_imports(code, valid)

    # ── Fix missing recharts imports ──────────────────────────────
    code = _fix_recharts_imports(code)

    return code


def _fix_lucide_imports(code: str, valid_exports: set[str]) -> str:
    """Remove invalid/duplicate lucide-react imports and stub removed names."""
    lines = code.split("\n")
    fixed = []
    removed_names: list[str] = []
    seen_imports: set[str] = set()  # track all lucide names already imported

    for line in lines:
        match = re.match(r"^(import\s+\{)(.*?)(\}\s+from\s+['\"]lucide-react['\"];?\s*)$", line)
        if match:
            prefix, imports_str, suffix = match.groups()
            names = [n.strip() for n in imports_str.split(",") if n.strip()]
            # Filter: valid AND not already imported (dedup across multiple import lines)
            valid_names = [n for n in names if n in valid_exports and n not in seen_imports]
            invalid_names = [n for n in names if n not in valid_exports]
            dup_names = [n for n in names if n in valid_exports and n in seen_imports]
            removed_names.extend(invalid_names)
            seen_imports.update(valid_names)
            if not valid_names:
                fixed.append(f"// {line}  // removed: all imports invalid/duplicate")
                continue
            if invalid_names or dup_names:
                fixed.append(f"{prefix} {', '.join(valid_names)} {suffix}// removed: {', '.join(invalid_names + dup_names)}")
            else:
                fixed.append(line)
        else:
            fixed.append(line)

    # Inject stub components for removed icon names so JSX references don't crash
    # Render as a visible circle placeholder instead of empty span
    stub_names = [n for n in set(removed_names) if n not in seen_imports]
    if stub_names:
        stub_line = (
            "// Stubs for invalid lucide-react icons\n"
            + "\n".join(
                f"const {name} = (props) => <svg width={{props.size||24}} height={{props.size||24}} viewBox=\"0 0 24 24\" fill=\"none\" stroke=\"currentColor\" strokeWidth=\"2\" strokeLinecap=\"round\" strokeLinejoin=\"round\"><circle cx=\"12\" cy=\"12\" r=\"10\"/></svg>;"
                for name in sorted(stub_names)
            )
        )
        last_import_idx = -1
        for i, line in enumerate(fixed):
            if line.startswith("import ") or (line.startswith("//") and "import" in line):
                last_import_idx = i
        if last_import_idx >= 0:
            fixed.insert(last_import_idx + 1, stub_line)
        else:
            fixed.insert(0, stub_line)

    return "\n".join(fixed)


# Known recharts components for auto-import
_RECHARTS_COMPONENTS = {
    "ResponsiveContainer", "LineChart", "Line", "BarChart", "Bar",
    "AreaChart", "Area", "PieChart", "Pie", "Cell", "RadarChart", "Radar",
    "PolarGrid", "PolarAngleAxis", "PolarRadiusAxis", "ScatterChart", "Scatter",
    "XAxis", "YAxis", "CartesianGrid", "Tooltip", "Legend",
    "RadialBarChart", "RadialBar", "Treemap", "Sankey",
    "ComposedChart", "FunnelChart", "Funnel",
}


def _fix_recharts_imports(code: str) -> str:
    """Auto-add missing recharts imports by scanning JSX for known component names."""
    # Find which recharts components are already imported
    already_imported: set[str] = set()
    for m in re.finditer(r"import\s+\{([^}]+)\}\s+from\s+['\"]recharts['\"]", code):
        for name in m.group(1).split(","):
            already_imported.add(name.strip())

    # Find which recharts components are used as JSX tags
    used_tags = set(re.findall(r"<(\w+)", code))
    needed = (used_tags & _RECHARTS_COMPONENTS) - already_imported

    if not needed:
        return code

    import_line = f"import {{ {', '.join(sorted(needed))} }} from 'recharts';"

    # Insert after the last import line
    lines = code.split("\n")
    last_import_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") or (stripped.startswith("//") and "import" in stripped):
            last_import_idx = i
    if last_import_idx >= 0:
        lines.insert(last_import_idx + 1, import_line)
    else:
        lines.insert(0, import_line)

    return "\n".join(lines)


async def call_model(session, model_url: str, model_name: str,
                     image_path: str, temperature: float,
                     max_tokens: int) -> str:
    """Send image to SGLang OpenAI-compatible API and return model output text."""
    w, h = _get_image_size(image_path)

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    data_url = f"data:image/png;base64,{img_b64}"

    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": f"Widget dimensions: {w}x{h}px. Recreate this widget as a React component with Tailwind CSS."},
                ],
            },
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    async with session.post(f"{model_url}/chat/completions", json=payload) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"API error {resp.status}: {body[:500]}")
        data = await resp.json()

    return data["choices"][0]["message"]["content"]


async def infer_sample(sem: asyncio.Semaphore, session,
                       model_url: str, model_name: str,
                       gt_path: str, widget_id: str, output_dir: str,
                       temperature: float, max_tokens: int,
                       max_retries: int = 2) -> dict:
    """Phase 1a: call model and extract TSX (concurrent-safe)."""
    sample_dir = os.path.join(output_dir, widget_id)
    tsx_path = os.path.join(sample_dir, "component.tsx")
    pred_png_path = os.path.join(sample_dir, "pred.png")

    # Resume: skip if pred.png already exists
    if os.path.exists(pred_png_path) and os.path.exists(tsx_path):
        return {"id": widget_id, "status": "skipped", "gt_path": gt_path}

    os.makedirs(sample_dir, exist_ok=True)

    async with sem:
        for attempt in range(1, max_retries + 1):
            try:
                raw_output_path = os.path.join(sample_dir, "raw_output.txt")
                raw_output = await call_model(session, model_url, model_name,
                                              gt_path, temperature, max_tokens)
                with open(raw_output_path, "w") as f:
                    f.write(raw_output)

                tsx = extract_code(raw_output)
                if tsx is None:
                    if attempt < max_retries:
                        print(f"  [{widget_id}] No valid code found (attempt {attempt}/{max_retries}), retrying...")
                        continue
                    print(f"  [{widget_id}] No valid code found after {max_retries} attempts")
                    return {"id": widget_id, "status": "no_code", "gt_path": gt_path}

                with open(tsx_path, "w") as f:
                    f.write(tsx)

                print(f"  [{widget_id}] Inferred {len(tsx)} chars")
                return {"id": widget_id, "status": "inferred", "gt_path": gt_path}

            except Exception as e:
                if attempt < max_retries:
                    print(f"  [{widget_id}] Error on attempt {attempt}/{max_retries}: {e}, retrying...")
                    for f_path in [tsx_path, pred_png_path]:
                        if os.path.exists(f_path):
                            os.unlink(f_path)
                    continue
                print(f"  [{widget_id}] Error after {max_retries} attempts: {e}")
                return {"id": widget_id, "status": "error", "error": str(e), "gt_path": gt_path}


def render_all(results: list[dict], output_dir: str):
    """Phase 1b: render all TSX files to PNG (serial — Playwright not thread-safe)."""
    to_render = [r for r in results if r["status"] == "inferred"]
    print(f"\nRendering {len(to_render)} components...")

    ok = 0
    errors = 0
    for i, r in enumerate(to_render):
        widget_id = r["id"]
        sample_dir = os.path.join(output_dir, widget_id)
        tsx_path = os.path.join(sample_dir, "component.tsx")
        pred_png_path = os.path.join(sample_dir, "pred.png")

        try:
            tsx = open(tsx_path).read()
            w, h = _get_image_size(r["gt_path"])
            rendered_bytes = render_tsx_to_image(tsx, w, h)
            with open(pred_png_path, "wb") as f:
                f.write(rendered_bytes)
            r["status"] = "ok"
            ok += 1
            if (i + 1) % 50 == 0 or i == len(to_render) - 1:
                print(f"  Rendered {i+1}/{len(to_render)} ({ok} ok, {errors} errors)")
        except Exception as e:
            r["status"] = "render_error"
            r["error"] = str(e)
            errors += 1
            print(f"  [{widget_id}] Render error: {e}")

    print(f"  Rendering done: {ok} ok, {errors} errors")


def run_evaluation(test_dir: str, output_dir: str):
    """
    Run widget-factory eval.py as a subprocess to avoid torch/OpenBLAS conflicts.

    eval.py expects:
      GT dir:   gt_*.png files
      Pred dir: {id}/pred.png  (our output structure matches this)
    """
    eval_script = "/home/advey/widget-factory/tools/evaluation/eval.py"

    cmd = [
        sys.executable, eval_script,
        "--gt_dir", test_dir,
        "--baseline_dir", output_dir,
        "--workers", "4",
    ]

    print(f"\nRunning evaluation: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"Evaluation exited with code {result.returncode}")

    # Collect per-sample evaluations into a summary
    collect_summary(test_dir, output_dir)


def collect_summary(test_dir: str, output_dir: str):
    """Collect per-sample evaluation.json files into an aggregate summary."""
    results = []
    for eval_file in sorted(glob.glob(os.path.join(output_dir, "*/evaluation.json"))):
        try:
            with open(eval_file) as f:
                data = json.load(f)
            if "PerceptualScore" in data:  # valid evaluation
                results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue

    if not results:
        print("No evaluation results found to aggregate.")
        return

    # Aggregate
    metric_groups = ["LayoutScore", "LegibilityScore", "StyleScore", "PerceptualScore", "Geometry"]
    avg = {}
    for group in metric_groups:
        vals = [r[group] for r in results if group in r]
        if not vals:
            continue
        if isinstance(vals[0], dict):
            avg[group] = {}
            for key in vals[0]:
                sub_vals = [v[key] for v in vals if key in v]
                avg[group][key] = round(float(np.mean(sub_vals)), 3)
        else:
            avg[group] = round(float(np.mean(vals)), 3)

    # Count statuses
    all_dirs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
    has_pred = [d for d in all_dirs if os.path.exists(os.path.join(output_dir, d, "pred.png"))]
    no_code = [d for d in all_dirs if os.path.exists(os.path.join(output_dir, d, "raw_output.txt"))
               and not os.path.exists(os.path.join(output_dir, d, "component.tsx"))]

    summary = {
        "total": len(all_dirs),
        "rendered": len(has_pred),
        "evaluated": len(results),
        "no_code": len(no_code),
        "widget_factory_metrics": avg,
    }

    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS ({len(results)} samples)")
    print(f"{'='*60}")
    for group, val in avg.items():
        if isinstance(val, dict):
            print(f"  {group}:")
            for k, v in val.items():
                print(f"    {k:20s}: {v:.3f}")
        else:
            print(f"  {group:20s}: {val:.3f}")
    print(f"\nSummary saved to {summary_path}")


async def run_inference(args):
    """Phase 1: inference (concurrent) + rendering (serial)."""
    gt_files = sorted(glob.glob(os.path.join(args.test_dir, "gt_*.png")))
    if not gt_files:
        print(f"No gt_*.png files found in {args.test_dir}")
        sys.exit(1)

    if args.limit > 0:
        gt_files = gt_files[:args.limit]

    print(f"Found {len(gt_files)} test images in {args.test_dir}")
    print(f"Model: {args.model_name} @ {args.model_url}")
    print(f"Output: {args.output_dir}")
    print(f"Concurrency: {args.concurrency}")
    os.makedirs(args.output_dir, exist_ok=True)

    sem = asyncio.Semaphore(args.concurrency)

    # Phase 1a: concurrent inference
    import aiohttp
    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for gt_path in gt_files:
            fname = os.path.basename(gt_path)
            widget_id = fname.replace("gt_", "").replace(".png", "")
            tasks.append(
                infer_sample(sem, session, args.model_url, args.model_name,
                             gt_path, widget_id, args.output_dir,
                             args.temperature, args.max_tokens)
            )

        start = time.time()
        results = await asyncio.gather(*tasks)
        infer_elapsed = time.time() - start

    inferred = sum(1 for r in results if r["status"] == "inferred")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    no_code = sum(1 for r in results if r["status"] == "no_code")
    errors = sum(1 for r in results if r["status"] == "error")

    print(f"\nInference done in {infer_elapsed:.1f}s — {inferred} to render, {skipped} skipped, {no_code} no_code, {errors} errors")

    return results


async def main():
    parser = argparse.ArgumentParser(description="SFT Benchmark: evaluate model on widget-factory test_split")
    parser.add_argument("--model-url", type=str, default="",
                        help="SGLang OpenAI-compatible API base URL")
    parser.add_argument("--model-name", type=str, default="",
                        help="Model name for the API")
    parser.add_argument("--test-dir", type=str, default="/home/advey/widget-factory/test_split",
                        help="Directory containing gt_*.png files")
    parser.add_argument("--output-dir", type=str, default="/shared/advey/benchmark-results/sft-benchmark",
                        help="Output directory for results")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max concurrent API requests")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of samples (0 = all)")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Only run inference + rendering, skip evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Only run evaluation on existing renders")
    args = parser.parse_args()

    results = None
    if not args.eval_only:
        if not args.model_url or not args.model_name:
            parser.error("--model-url and --model-name are required unless --eval-only")
        results = await run_inference(args)

    return args, results


def sync_main():
    args, results = asyncio.run(main())

    # Phase 1b: render OUTSIDE asyncio loop (Playwright sync API requirement)
    if results is not None:
        render_start = time.time()
        render_all(results, args.output_dir)
        render_elapsed = time.time() - render_start

        ok = sum(1 for r in results if r["status"] == "ok")
        skipped = sum(1 for r in results if r["status"] == "skipped")
        no_code = sum(1 for r in results if r["status"] == "no_code")
        errors = sum(1 for r in results if r["status"] in ("error", "render_error"))
        print(f"\nPhase 1 complete (render {render_elapsed:.1f}s)")
        print(f"  OK: {ok}  Skipped: {skipped}  No code: {no_code}  Errors: {errors}")

    if not args.skip_eval:
        run_evaluation(args.test_dir, args.output_dir)


if __name__ == "__main__":
    sync_main()
