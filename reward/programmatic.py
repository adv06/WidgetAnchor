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

from PIL import Image
import io
from sklearn.cluster import KMeans
from scipy import optimize
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# keep browser alive across calls — launching chromium is ~1-2s per call
_playwright_instance = None
_browser = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    image1 = cv2.imdecode(np.frombuffer(ref_image, dtype=np.uint8), cv2.IMREAD_COLOR) # take the raw bytes and convert it into an HxWx3 array of pixel value, color axis is the third channel
    image2 = cv2.imdecode(np.frombuffer(base_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    h = min(image1.shape[0], image2.shape[0])
    w = min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (w, h))
    image2 = cv2.resize(image2, (w, h))  # reshape to min width and min height

    score, _ = ssim(image1, image2, full=True, channel_axis=2)
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

_clip_model = None
_clip_processor = None

_lpips_model = None 

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # keep on CPU — SIGFPE on GPU
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

def _get_lpips():
    global _lpips_model
    if _lpips_model is None:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex') # alexnet
        _lpips_model.eval() # eval mode 
        _lpips_model.to(device)
    return _lpips_model 
            

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

def compute_polarity(img1, img2):
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
    
    L1, _, _ = cv2.split(lab1)
    L2, _, _ = cv2.split(lab2)
    
    # build a histogram - how many pixels have each brightness value
    hist1 = cv2.calcHist([L1], [0], None, [256], [0, 256]) # 256 buckets, range from 0 to 256 (0 --> bucket 0 etc)
    hist2 = cv2.calcHist([L2], [0], None, [256], [0, 256])
    hist1 = hist1 / hist1.sum()
    hist2 = hist2 / hist2.sum()
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(score, 0)
    
def compute_lpips(img1: bytes, img2: bytes):
    import torchvision.transforms as T 
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3) # one per channel, we do -0.5/0.5 to put in the range [-1, 1]
    ])
    model = _get_lpips()
    img1 = Image.open(io.BytesIO(img1)).convert("RGB")
    img2 = Image.open(io.BytesIO(img2)).convert("RGB")
    img1 = transform(img1).unsqueeze(0).to(device)
    img2 = transform(img2).unsqueeze(0).to(device)
    distance = model(img1, img2)
    
    return 1-distance.item()

def compute_palette_distance(img1: np.array, img2: np.array):
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float) # remember -1 means figure this dimension out automatically
    k = 5
    labels1 = KMeans(n_clusters=k, random_state=0, n_init=10).fit(img1).cluster_centers_
    labels2 = KMeans(n_clusters=k, random_state=0, n_init=10).fit(img2).cluster_centers_
    D = cdist(labels1, labels2)
    row_ind, col_ind = linear_sum_assignment(D)
    
    return max(0.0, 1.0 - D[row_ind, col_ind].mean() / 100)
    
def compute_contrast_score(img1: bytes, img2: bytes):
    img1 = cv2.imdecode(np.frombuffer(img1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(img2, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    std1 = np.std(img1)
    std2 = np.std(img2)
    
    return 1.0 - abs(std1 - std2) / max(std1, std2)
    
def _extract_bounding_boxes(html_code: str, width=800, height=600) -> list[dict]:
    """Render HTML in Playwright and extract bounding boxes of all visible elements."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
        f.write(html_code)
        tmp_path = f.name

    try:
        browser = _get_browser()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"file:///{tmp_path}")
        page.wait_for_load_state("networkidle")

        elements = page.query_selector_all("*")
        boxes = []
        for el in elements:
            bb = el.bounding_box()
            if bb and bb["width"] > 0 and bb["height"] > 0:
                boxes.append({
                    "x": bb["x"],
                    "y": bb["y"],
                    "width": bb["width"],
                    "height": bb["height"],
                    "tag": el.evaluate("el => el.tagName.toLowerCase()"),
                })
        page.close()
        return boxes
    finally:
        os.unlink(tmp_path)


def compute_layout_score(ref_html: str, gen_html: str, width=800, height=600) -> float:
    """Compare layout similarity by extracting element bounding boxes from both renders."""
    ref_boxes = _extract_bounding_boxes(ref_html, width, height)
    gen_boxes = _extract_bounding_boxes(gen_html, width, height)

    if not ref_boxes or not gen_boxes:
        return 0.0
    
    # IoU
    IoU =  [[0 for j in range(len(ref_boxes))] for i in range(len(gen_boxes))]
    
    for i in range(len(gen_boxes)):
        for j in range(len(ref_boxes)):
            xleft = max(gen_boxes[i]['x'], ref_boxes[j]['x'])
            yup = max(gen_boxes[i]['y'], ref_boxes[j]['y'])
            xright = min(gen_boxes[i]['x']+gen_boxes[i]['width'], ref_boxes[j]['x']+ref_boxes[j]['width'])
            ydown = min(gen_boxes[i]['y']+gen_boxes[i]['height'], ref_boxes[j]['y']+ref_boxes[j]['height'])
            inter = max(xright-xleft, 0) * max(ydown-yup, 0)
            union = gen_boxes[i]['width']*gen_boxes[i]['height'] + ref_boxes[j]['width'] * ref_boxes[j]['height'] - inter 
            IoU[i][j] = inter / union if union > 0 else 0 
    IoU = np.array(IoU)
    
    rows, cols = linear_sum_assignment(-IoU)
    return IoU[rows, cols].mean()

    
def compute_reward_code(target_image: bytes, generated_html: str) -> float:
    ssim_score = 0.0
    lpips_score = 0.0
    palette_score = 0.0
    contrast_score = 0.0
    layout_score = 0.0
    polarity_score = 0.0

    try:
        rendered_image = render_html_to_image(generated_html)

        # decode both images to numpy for functions that need arrays
        ref_np = cv2.imdecode(np.frombuffer(target_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        gen_np = cv2.imdecode(np.frombuffer(rendered_image, dtype=np.uint8), cv2.IMREAD_COLOR)
        h = min(ref_np.shape[0], gen_np.shape[0])
        w = min(ref_np.shape[1], gen_np.shape[1])
        ref_np = cv2.resize(ref_np, (w, h))
        gen_np = cv2.resize(gen_np, (w, h))

        ssim_score = compute_reward_image(target_image, rendered_image)
        lpips_score = compute_lpips(target_image, rendered_image)
        palette_score = compute_palette_distance(ref_np, gen_np)
        contrast_score = compute_contrast_score(target_image, rendered_image)
        polarity_score = compute_polarity(ref_np, gen_np)
        layout_score = compute_layout_score(generated_html, generated_html)  # TODO: need ref HTML too
    except Exception:
        pass

    return (0.15 * ssim_score +
            0.15 * lpips_score +
            0.25 * palette_score +
            0.15 * contrast_score +
            0.20 * layout_score +
            0.10 * polarity_score)
