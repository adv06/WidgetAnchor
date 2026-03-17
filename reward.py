import torch
import torch.nn as nn
import tempfile
import os
from openai import OpenAI
import base64

from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import sync_playwright
import cv2
import numpy as np

# keep browser alive across calls — launching chromium is ~1-2s per call
_playwright_instance = None
_browser = None

def _get_browser():
    global _playwright_instance, _browser
    if _browser is None:
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=True)
    return _browser


def render_html_to_image(html_code: str, width=800, height=600) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html_code)
        tmp_path = f.name

    try:
        browser = _get_browser()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"file:///{tmp_path}")
        page.wait_for_load_state("networkidle")
        png_bytes = page.screenshot()
        page.close()
        return png_bytes
    finally:
        os.unlink(tmp_path)


def compute_reward_image(ref_image: bytes, base_image: bytes) -> float:
    image1 = cv2.imdecode(np.frombuffer(ref_image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imdecode(np.frombuffer(base_image, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    h = min(image1.shape[0], image2.shape[0])
    w = min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (w, h))
    image2 = cv2.resize(image2, (w, h))

    score, _ = ssim(image1, image2, full=True)
    return score


def compute_html_validity(html_code: str) -> float:
    from html.parser import HTMLParser

    tags_opened = []
    tag_count = 0

    class _Parser(HTMLParser):
        def handle_starttag(self, tag, attrs):
            nonlocal tag_count
            tags_opened.append(tag)
            tag_count += 1

        def handle_endtag(self, tag):
            if tags_opened and tags_opened[-1] == tag:
                tags_opened.pop()

    try:
        _Parser().feed(html_code)
    except Exception:
        return 0.0

    if tag_count == 0:
        return 0.0

    unclosed = len(tags_opened)
    return max(0.0, 1.0 - (unclosed / tag_count))

def compute_vllm_validity(ref_image: bytes, base_image: bytes) -> float:
    client = OpenAI()
    img1 = base64.b64encode(ref_image).decode()
    img2 = base64.b64encode(base_image).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "These are two rendered HTML/CSS widget screenshots. "
                    "Score how closely the second widget matches the first, considering layout structure, "
                    "color scheme, component placement, and overall visual appearance. "
                    "Ignore minor browser rendering differences like slight font variations. "
                    "Use this scale: 0.0 = completely different, 0.5 = same type of widget but different style, 1.0 = nearly identical. "
                    "Reply with only a decimal number between 0.0 and 1.0."
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2}"}}
            ]
        }]
    )

    score_text = response.choices[0].message.content.strip()
    try:
        score = float(score_text)
    except ValueError:
        import re
        match = re.search(r"\d+\.?\d*", score_text)
        score = float(match.group()) if match else 0.0
    return max(0.0, min(1.0, score))

_clip_model = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # keep on CPU — SIGFPE on GPU
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


def compute_clip_similarity(ref_image: bytes, gen_image: bytes) -> float:
    from PIL import Image
    import io

    model, processor = _get_clip()

    img1 = Image.open(io.BytesIO(ref_image)).convert("RGB")
    img2 = Image.open(io.BytesIO(gen_image)).convert("RGB")

    inputs = processor(images=[img1, img2], return_tensors="pt", padding=True)
    # inputs stay on CPU — CLIP triggers SIGFPE on this GPU
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    features = features / features.norm(dim=-1, keepdim=True)
    similarity = (features[0] @ features[1]).item()
    return max(0.0, similarity) # cosine similarity, clamp negative


def compute_reward_code(target_image: bytes, generated_html: str, ssim_weight=0.5, validity_weight=0.2, clip_weight=0.3):
    """Reward = SSIM + validity + CLIP (no VLM — too slow for training, 25k GPT-4o calls)"""
    validity = compute_html_validity(generated_html)

    ssim_score = 0.0
    clip_score = 0.0
    try:
        rendered_image = render_html_to_image(generated_html)
        ssim_score = compute_reward_image(target_image, rendered_image)
        clip_score = compute_clip_similarity(target_image, rendered_image)
    except Exception:
        pass

    return ssim_weight * ssim_score + validity_weight * validity + clip_weight * clip_score
