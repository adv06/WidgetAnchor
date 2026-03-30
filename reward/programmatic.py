import torch
import tempfile
import os
import shutil
import subprocess
from pathlib import Path

from skimage.metrics import structural_similarity as ssim
from playwright.sync_api import sync_playwright
import cv2
import numpy as np

from PIL import Image
import io
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

# keep browser alive across calls — launching chromium is ~1-2s per call
_playwright_instance = None
_browser = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths for the esbuild render pipeline
_RENDER_DIR = Path(__file__).resolve().parent.parent / "render"
_NODE_MODULES = _RENDER_DIR / "node_modules"
_TEMPLATE_HTML = _RENDER_DIR / "template.html"
_TAILWIND_JS = _RENDER_DIR / "tailwind.js"
_ESBUILD_BIN = _NODE_MODULES / ".bin" / "esbuild"

def _get_browser():
    global _playwright_instance, _browser
    if _browser is None:
        _playwright_instance = sync_playwright().start()
        _browser = _playwright_instance.chromium.launch(headless=True)
    return _browser


def render_tsx_to_image(tsx_code: str, width=800, height=600) -> bytes:
    # Render a React+Tailwind TSX component to a PNG screenshot via esbuild + Playwright.
    tmp_dir = tempfile.mkdtemp(prefix="widget_render_")
    try:
        # write the user's component
        comp_path = os.path.join(tmp_dir, "component.tsx")
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write(tsx_code)

        # write entry point that mounts the component
        entry_path = os.path.join(tmp_dir, "entry.tsx")
        with open(entry_path, "w", encoding="utf-8") as f:
            f.write(
                'import React from "react";\n'
                'import { createRoot } from "react-dom/client";\n'
                'import Widget from "./component";\n'
                'createRoot(document.getElementById("root")!).render(<Widget />);\n'
            )

        # symlink node_modules so esbuild can resolve packages
        nm_link = os.path.join(tmp_dir, "node_modules")
        os.symlink(str(_NODE_MODULES), nm_link)

        # bundle with esbuild
        result = subprocess.run(
            [
                str(_ESBUILD_BIN), "entry.tsx",
                "--bundle",
                "--outfile=bundle.js",
                "--format=iife",
                "--jsx=automatic",
                "--loader:.tsx=tsx",
            ],
            cwd=tmp_dir,
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"esbuild failed:\n{result.stderr}")

        # copy the HTML template and local tailwind.js
        shutil.copy2(str(_TEMPLATE_HTML), os.path.join(tmp_dir, "index.html"))
        if _TAILWIND_JS.exists():
            shutil.copy2(str(_TAILWIND_JS), os.path.join(tmp_dir, "tailwind.js"))

        # render with Playwright
        browser = _get_browser()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"file://{os.path.join(tmp_dir, 'index.html')}")
        page.wait_for_load_state("networkidle")

        # auto-resize viewport to the #root bounding box
        root = page.query_selector("#root")
        if root:
            bb = root.bounding_box()
            if bb and bb["width"] > 0 and bb["height"] > 0:
                new_w = min(int(bb["width"]) + 2, 1920)
                new_h = min(int(bb["height"]) + 2, 1080)
                page.set_viewport_size({"width": new_w, "height": new_h})
                page.wait_for_timeout(100)

        png_bytes = page.screenshot()
        page.close()
        return png_bytes
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)



def compute_reward_image(ref_image: bytes, base_image: bytes) -> float:
    image1 = cv2.imdecode(np.frombuffer(ref_image, dtype=np.uint8), cv2.IMREAD_COLOR) # take the raw bytes and convert it into an HxWx3 array of pixel value, color axis is the third channel
    image2 = cv2.imdecode(np.frombuffer(base_image, dtype=np.uint8), cv2.IMREAD_COLOR)
    h = min(image1.shape[0], image2.shape[0])
    w = min(image1.shape[1], image2.shape[1])
    image1 = cv2.resize(image1, (w, h))
    image2 = cv2.resize(image2, (w, h))  # reshape to min width and min height

    score, _ = ssim(image1, image2, full=True, channel_axis=2)
    return score



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
    labels2 = KMeans(n_clusters=k, random_state=0, n_init=10).fit(img2).cluster_centers_ # k means, cluster centers is the different palette colors (K, 3) i think..
    D = cdist(labels1, labels2)
    row_ind, col_ind = linear_sum_assignment(D)
    
    return max(0.0, 1.0 - D[row_ind, col_ind].mean() / 100)
    
def compute_contrast_score(img1: bytes, img2: bytes):
    img1 = cv2.imdecode(np.frombuffer(img1, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imdecode(np.frombuffer(img2, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    std1 = np.std(img1)
    std2 = np.std(img2)
    
    if max(std1, std2) == 0:
        return 1.0
    return 1.0 - abs(std1 - std2) / max(std1, std2)
    
def _extract_bounding_boxes(tsx_code: str, width=800, height=600) -> list[dict]:
    """Render TSX in Playwright and extract bounding boxes of all visible elements."""
    tmp_dir = tempfile.mkdtemp(prefix="widget_bbox_")
    try:
        # write component
        with open(os.path.join(tmp_dir, "component.tsx"), "w", encoding="utf-8") as f:
            f.write(tsx_code)

        # write entry point
        with open(os.path.join(tmp_dir, "entry.tsx"), "w", encoding="utf-8") as f:
            f.write(
                'import React from "react";\n'
                'import { createRoot } from "react-dom/client";\n'
                'import Widget from "./component";\n'
                'createRoot(document.getElementById("root")!).render(<Widget />);\n'
            )

        os.symlink(str(_NODE_MODULES), os.path.join(tmp_dir, "node_modules"))

        result = subprocess.run(
            [
                str(_ESBUILD_BIN), "entry.tsx",
                "--bundle", "--outfile=bundle.js",
                "--format=iife", "--jsx=automatic", "--loader:.tsx=tsx",
            ],
            cwd=tmp_dir, capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            return []

        shutil.copy2(str(_TEMPLATE_HTML), os.path.join(tmp_dir, "index.html"))
        if _TAILWIND_JS.exists():
            shutil.copy2(str(_TAILWIND_JS), os.path.join(tmp_dir, "tailwind.js"))

        browser = _get_browser()
        page = browser.new_page(viewport={"width": width, "height": height})
        page.goto(f"file://{os.path.join(tmp_dir, 'index.html')}")
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
        shutil.rmtree(tmp_dir, ignore_errors=True)


def compute_layout_score(ref_tsx: str, gen_tsx: str, width=800, height=600) -> float:
    # layout score by IoU
    ref_boxes = _extract_bounding_boxes(ref_tsx, width, height)
    gen_boxes = _extract_bounding_boxes(gen_tsx, width, height)

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
            inter = max(xright-xleft, 0) * max(ydown-yup, 0) # think about it this is just intersecting rectangle
            union = gen_boxes[i]['width']*gen_boxes[i]['height'] + ref_boxes[j]['width'] * ref_boxes[j]['height'] - inter 
            IoU[i][j] = inter / union if union > 0 else 0 
    IoU = np.array(IoU)
    
    rows, cols = linear_sum_assignment(-IoU)
    return IoU[rows, cols].mean()

    
def _get_wf_metrics():
    """Lazy-load widget2code evaluation metrics."""
    import sys
    if "/home/advey/widget-factory/tools/evaluation" not in sys.path:
        sys.path.insert(0, "/home/advey/widget-factory/tools/evaluation")
    from widget_quality.layout import compute_layout
    from widget_quality.legibility import compute_legibility
    from widget_quality.perceptual import compute_perceptual
    from widget_quality.style import compute_style
    from widget_quality.geometry import compute_aspect_dimensionality_fidelity
    from widget_quality.composite import (
        handling_layout, handling_legibility, handling_style, handling_perceptual
    )
    return {
        "compute_layout": compute_layout,
        "compute_legibility": compute_legibility,
        "compute_perceptual": compute_perceptual,
        "compute_style": compute_style,
        "compute_geo": compute_aspect_dimensionality_fidelity,
        "handling_layout": handling_layout,
        "handling_legibility": handling_legibility,
        "handling_style": handling_style,
        "handling_perceptual": handling_perceptual,
    }


_wf_metrics = None


def compute_reward_wf(target_image: bytes, rendered_image: bytes) -> float:
    """Compute reward using widget2code evaluation metrics. Returns [0, 1]."""
    global _wf_metrics
    if _wf_metrics is None:
        _wf_metrics = _get_wf_metrics()

    from PIL import Image as PILImage
    ref_img = np.array(PILImage.open(io.BytesIO(target_image)).convert("RGB"))
    gen_img = np.array(PILImage.open(io.BytesIO(rendered_image)).convert("RGB"))

    h = min(ref_img.shape[0], gen_img.shape[0])
    w = min(ref_img.shape[1], gen_img.shape[1])
    ref_img = np.array(PILImage.fromarray(ref_img).resize((w, h)))
    gen_img = np.array(PILImage.fromarray(gen_img).resize((w, h)))

    geo = _wf_metrics["compute_geo"](ref_img, gen_img)
    layout = _wf_metrics["compute_layout"](ref_img, gen_img)
    legibility = _wf_metrics["compute_legibility"](ref_img, gen_img)
    perceptual = _wf_metrics["compute_perceptual"](ref_img, gen_img)
    style = _wf_metrics["compute_style"](ref_img, gen_img)

    ls = _wf_metrics["handling_layout"](layout)
    leg = _wf_metrics["handling_legibility"](legibility)
    st = _wf_metrics["handling_style"](style)
    ps = _wf_metrics["handling_perceptual"](perceptual)
    gs = 100 * np.clip(geo, 0, 1)

    # Weighted combination matching widget2code benchmark emphasis
    score = (
        0.08 * ls["MarginAsymmetry"] +
        0.08 * ls["ContentAspectDiff"] +
        0.04 * ls["AreaRatioDiff"] +
        0.15 * leg["TextJaccard"] +
        0.05 * leg["ContrastDiff"] +
        0.05 * leg["ContrastLocalDiff"] +
        0.10 * st["PaletteDistance"] +
        0.05 * st["Vibrancy"] +
        0.05 * st["PolarityConsistency"] +
        0.10 * (ps["ssim"] * 100) +
        0.10 * ((1 - ps["lp"]) * 100) +
        0.10 * gs +
        0.05 * 100  # render success bonus
    ) / 100.0

    return float(np.clip(score, 0, 1))


def compute_reward_code(target_image: bytes, generated_tsx: str, rendered_image: bytes = None,
                        ref_tsx: str = None) -> float:
    # render if needed — let render failures propagate so callers can handle them
    if rendered_image is None:
        rendered_image = render_tsx_to_image(generated_tsx)

    # --- simple metrics (fast, stable baseline signal) ---
    ssim_score = 0.0
    lpips_score = 0.0
    palette_score = 0.0
    contrast_score = 0.0
    polarity_score = 0.0

    try:
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
    except Exception as e:
        import sys
        print(f"[reward] simple metrics failed: {e}", file=sys.stderr)

    simple_score = (0.20 * ssim_score +
                    0.20 * lpips_score +
                    0.25 * palette_score +
                    0.20 * contrast_score +
                    0.15 * polarity_score)

    # --- wf metrics (benchmark-aligned, covers layout/text/geometry that simple metrics miss) ---
    wf_score = 0.0
    try:
        wf_score = compute_reward_wf(target_image, rendered_image)
    except Exception as e:
        import sys
        print(f"[reward] wf metrics failed, using simple only: {e}", file=sys.stderr)
        return simple_score

    # 70% wf (what the benchmark scores) + 30% simple (stable gradient signal)
    return 0.7 * wf_score + 0.3 * simple_score
