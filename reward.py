import torch
import torch.nn as nn
import tempfile
import os

from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import sync_playwright
import cv2
import numpy as np 


def render_html_to_image(html_code: str, width=800, height=600):
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
            img_array = np.frombuffer(png_bytes, dtype=np.uint8)
            browser.close()
            return img_array
    finally:
        os.unlink(tmp_path)


def compute_reward_image(ref_image, base_image):
    image1 = cv2.imdecode(ref_image, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imdecode(base_image, cv2.IMREAD_GRAYSCALE)
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


def compute_reward_code(ref_code, base_code, ssim_weight=0.7, validity_weight=0.3):
    validity = compute_html_validity(base_code)

    try:
        ref_image = render_html_to_image(ref_code)
        base_image = render_html_to_image(base_code)
        ssim_score = compute_reward_image(ref_image, base_image)
    except Exception:
        ssim_score = 0.0

    return ssim_weight * ssim_score + validity_weight * validity

