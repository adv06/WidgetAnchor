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


def render_html_to_image(html_code: str, width=800, height=600) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html_code)
        tmp_path = f.name

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height})
            page.goto(f"file:///{tmp_path}")
            page.wait_for_load_state("networkidle")
            png_bytes = page.screenshot()
            browser.close()
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
    
def compute_reward_code(ref_code, base_code, ssim_weight=0.5, validity_weight=0.2, vlm_weight=0.3):
    validity = compute_html_validity(base_code)

    ssim_score = 0.0
    vlm_score = 0.0
    try:
        ref_image = render_html_to_image(ref_code)
        base_image = render_html_to_image(base_code)
        ssim_score = compute_reward_image(ref_image, base_image)
        vlm_score = compute_vllm_validity(ref_image, base_image)
    except Exception:
        pass

    return ssim_weight * ssim_score + validity_weight * validity + vlm_weight * vlm_score

