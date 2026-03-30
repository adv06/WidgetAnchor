"""
Generate CoT + TSX code from widget images using GPT via OpenAI API.
Single-pass: generates both the reasoning trace and React component in one call.

Usage:
    python -m data.collection.generate_from_images --input_dir ./data/widget2code-benchmark/test --output_dir ./output/raw --workers 8
"""
import json
import os
import re
import struct
import base64
import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

ICON_LIST = (
    "Star, Heart, Home, Search, Settings, User, Mail, Phone, MapPin, Clock, Calendar, "
    "Bell, ChevronRight, ChevronLeft, ChevronDown, ChevronUp, ArrowRight, ArrowLeft, "
    "ArrowUp, ArrowDown, Plus, Minus, X, Check, AlertTriangle, AlertCircle, Info, "
    "Eye, EyeOff, Download, Upload, Share2, Bookmark, Tag, Filter, BarChart3, "
    "LineChart, PieChart, TrendingUp, TrendingDown, DollarSign, CreditCard, "
    "ShoppingCart, Package, Truck, Globe, Wifi, Battery, Zap, Sun, Moon, Cloud, "
    "Thermometer, Droplets, Wind, Umbrella, Play, Pause, SkipForward, SkipBack, "
    "Volume2, Mic, Camera, Image, Video, Music, File, FileText, Folder, Trash2, "
    "Pencil, Copy, Clipboard, Link, ExternalLink, MoreHorizontal, MoreVertical, "
    "Menu, Grid, List, Layout, Layers, RefreshCw, RotateCw, Loader, Send, "
    "MessageSquare, MessageCircle, ThumbsUp, ThumbsDown, Award, Target, Flame, "
    "Sparkles, Crown, Shield, Lock, Unlock, Key, LogIn, LogOut, UserPlus, Users, "
    "Building, Landmark, Map, Navigation, Compass, Flag, Crosshair, Activity, "
    "Cpu, Database, Server, HardDrive, Monitor, Smartphone, Tablet, Watch, "
    "Headphones, Speaker, Printer, Scan, QrCode, Fingerprint, CircleDot, "
    "Timer, Hourglass, Repeat2, Shuffle, Maximize2, Minimize2, Move"
)

SYSTEM_PROMPT = (
    "You are a high-fidelity UI reproduction expert. Given a screenshot of a UI widget, "
    "analyze it and generate a React functional component with Tailwind CSS that visually matches it.\n\n"
    "## Rules\n"
    "- Output a single default-exported React component: `export default function Widget() { ... }`\n"
    "- Use Tailwind CSS utility classes for all styling\n"
    "- Colors: use exact hex values via arbitrary-value syntax (e.g. `bg-[#3B82F6]`, `text-[#1F2937]`)\n"
    "- Match layout exactly: flex/grid direction, alignment, spacing, gaps, padding, margins\n"
    "- Match typography: text size, font weight, tracking, leading\n"
    "- Match border-radius, shadow, opacity, gradients\n"
    "- Charts/gauges/progress: use recharts (import from 'recharts')\n"
    "- Icons: ONLY use lucide-react (import from 'lucide-react'). Valid names: " + ICON_LIST + ".\n"
    "- NEVER invent icon names. If unsure, use Circle or Square.\n"
    "- Use DOUBLE QUOTES for all JavaScript strings containing apostrophes\n"
    "- Text content must be character-perfect — copy every word and number exactly\n"
    "- The root container must match the widget dimensions given in the user message\n\n"
    "## Output format\n"
    "<think>\n"
    "## 1. Widget Information Specification\n"
    "### 1a. Widget Type\n[presentation archetype]\n"
    "### 1b. Semantic Fields\n[information fields with semantic roles]\n"
    "### 1c. Relations\n[typed semantic relations between fields]\n"
    "### 1d. Salience\n[field priority scores 0.0-1.0]\n"
    "### 1e. Grouping\n[meaningful presentation units]\n"
    "### 1f. Presentation Intent\n[density, focus, lead]\n\n"
    "## 2. Presentation Realization\n"
    "### 2a. Layout Topology\n[spatial structure]\n"
    "### 2b. Geometric Allocation\n[spacing, padding, sizing in Tailwind]\n"
    "### 2c. Style Tokens\n[exact hex colors, borders, shadows, fonts]\n"
    "### 2d. Field-to-Visual Binding\n[semantic field → visual element mapping]\n"
    "</think>\n"
    "<code>\n[complete React component]\n</code>"
)


def _get_image_size(path: str) -> tuple[int, int]:
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", header[16:24])
    return (800, 600)


def strip_code_fences(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1]
    if code.endswith("```"):
        code = code.rsplit("```", 1)[0]
    return code.strip()


def _normalize_format(text: str) -> str:
    """Normalize: convert markdown code fences inside output to <code> tags if needed."""
    if "<code>" not in text and "```" in text:
        fence_match = re.search(r"```(?:tsx|jsx|javascript|react)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            code_content = fence_match.group(1).strip()
            text = text[:fence_match.start()] + "<code>\n" + code_content + "\n</code>" + text[fence_match.end():]
    return text


def generate_from_image(image_path: str, model: str = "gpt-5.4") -> tuple[str, str]:
    """Send image to GPT and get CoT + TSX back. Returns (cot, tsx)."""
    with open(image_path, "rb") as f:
        img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode()

    w, h = _get_image_size(image_path)
    user_text = f"Widget dimensions: {w}x{h}px. Recreate this widget as a React component with Tailwind CSS."

    last_error = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    ]},
                ],
                max_completion_tokens=8192,
            )
            text = response.choices[0].message.content
            if text is None:
                raise RuntimeError("API returned None")

            text = _normalize_format(text.strip())

            # Extract tsx from <code> block
            code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
            if code_match:
                tsx = strip_code_fences(code_match.group(1).strip())
            else:
                tsx = strip_code_fences(text)

            return text, tsx
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str or "rate" in err_str.lower() or "quota" in err_str.lower():
                wait = min(60, 2 ** attempt * 5)
                print(f"    Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"All retries exhausted: {last_error}")


def generate_one(image_path: str, output_dir: str, model: str = "gpt-5.4") -> dict | None:
    """Generate CoT + TSX for a single image."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    widget_id = f"hf-{basename}"
    output_path = os.path.join(output_dir, f"{widget_id}.json")

    if os.path.exists(output_path):
        return None  # already done

    try:
        cot, tsx = generate_from_image(image_path, model=model)

        if "export default" not in tsx:
            print(f"  Bad output for {widget_id}: no export default")
            return None

        # Validate <think> and <code> tags
        if "<think>" not in cot or "<code>" not in cot:
            print(f"  Bad format for {widget_id}: missing tags")
            return None

        result = {
            "widget_id": widget_id,
            "tsx": tsx,
            "cot": cot,
            "category": "widget2code-benchmark",
            "screenshot_path": os.path.abspath(image_path),
        }

        with open(output_path, "w") as f:
            json.dump(result, f)

        return result
    except Exception as e:
        print(f"  Failed {widget_id}: {e}")
        return None


def generate(input_dir: str, output_dir: str, model: str = "gpt-5.4", workers: int = 8):
    os.makedirs(output_dir, exist_ok=True)

    images = sorted([
        os.path.join(input_dir, f) for f in os.listdir(input_dir)
        if f.endswith(".png")
    ])
    print(f"Found {len(images)} images to process")

    already_done = len([
        f for f in os.listdir(output_dir)
        if f.startswith("hf-") and f.endswith(".json")
    ])
    print(f"Already done: {already_done}")

    succeeded = already_done
    failed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(generate_one, img, output_dir, model): img for img in images}
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            if result is not None:
                succeeded += 1
            else:
                img_path = futures[future]
                basename = os.path.splitext(os.path.basename(img_path))[0]
                widget_id = f"hf-{basename}"
                if not os.path.exists(os.path.join(output_dir, f"{widget_id}.json")):
                    failed += 1

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(images)} | succeeded: {succeeded} | failed: {failed}")

    print(f"\nDone! {succeeded} succeeded, {failed} failed")
    print(f"Output saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="./data/widget2code-benchmark/test")
    parser.add_argument("--output_dir", type=str, default="./output/raw")
    parser.add_argument("--model", type=str, default="gpt-5.4")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    generate(args.input_dir, args.output_dir, args.model, args.workers)
