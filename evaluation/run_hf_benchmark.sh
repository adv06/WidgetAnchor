#!/bin/bash
# Full benchmark pipeline on Djanghao/widget2code-benchmark test split
# Safe to run via: nohup bash run_hf_benchmark.sh > benchmark.log 2>&1 &

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
BENCHMARK_DIR="/shared/advey/hf-benchmark"
GT_DIR="$BENCHMARK_DIR/gt"
SFT_DIR="$BENCHMARK_DIR/sft"
BASE_DIR="$BENCHMARK_DIR/base"

SFT_URL="http://localhost:8000/v1"
SFT_NAME="/shared/advey/glm-4.1v-9b-thinking-sft-merged"
BASE_URL="http://localhost:8001/v1"
BASE_NAME="zai-org/GLM-4.1V-9B-Thinking"

MAX_TOKENS=16384
TEMPERATURE=0.7
CONCURRENCY=4

export HF_HOME="/shared/advey/hf_cache"
export HF_DATASETS_CACHE="/shared/advey/hf_cache/datasets"

echo "============================================================"
echo "Widget2Code HF Benchmark"
echo "Started: $(date)"
echo "============================================================"

# ── Step 1: Download test split ──────────────────────────────────
if [ -d "$GT_DIR" ] && [ "$(ls "$GT_DIR"/gt_*.png 2>/dev/null | wc -l)" -ge 100 ]; then
    echo "GT images already downloaded, skipping..."
else
    echo "Downloading Djanghao/widget2code-benchmark test split..."
    mkdir -p "$GT_DIR"
    python3 - "$GT_DIR" <<'PYEOF'
import sys
from datasets import load_dataset
from pathlib import Path

out_dir = Path(sys.argv[1])
ds = load_dataset("Djanghao/widget2code-benchmark", split="test")
print(f"Test split has {len(ds)} samples")

for i, sample in enumerate(ds):
    img = sample["image"]
    # Convert to RGB if needed (some PNGs have alpha)
    if img.mode != "RGB":
        img = img.convert("RGB")
    fname = out_dir / f"gt_test-{i:05d}.png"
    img.save(fname)
    if (i + 1) % 100 == 0:
        print(f"  Saved {i+1}/{len(ds)} images")

print(f"Done: saved {len(ds)} images to {out_dir}")
PYEOF
fi

N_IMAGES=$(ls "$GT_DIR"/gt_*.png 2>/dev/null | wc -l)
echo "GT images ready: $N_IMAGES"

# ── Step 2: Run SFT model ───────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 1: SFT model inference + rendering"
echo "============================================================"
cd "$REPO_DIR"
python3 -m evaluation.run_sft_benchmark \
    --model-url "$SFT_URL" \
    --model-name "$SFT_NAME" \
    --test-dir "$GT_DIR" \
    --output-dir "$SFT_DIR" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    --concurrency "$CONCURRENCY" \
    --limit 500 \
    --skip-eval

# ── Step 3: Base model runs in parallel tmux session (hf-base) ──
echo "Base model running in parallel tmux session 'hf-base'"

# ── Step 4: Evaluate SFT ────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 3: Evaluate SFT"
echo "============================================================"
python3 -m evaluation.run_sft_benchmark \
    --eval-only \
    --test-dir "$GT_DIR" \
    --output-dir "$SFT_DIR"

# ── Step 5: Evaluate Base ────────────────────────────────────────
echo ""
echo "============================================================"
echo "Phase 4: Evaluate Base"
echo "============================================================"
python3 -m evaluation.run_sft_benchmark \
    --eval-only \
    --test-dir "$GT_DIR" \
    --output-dir "$BASE_DIR"

# ── Step 6: Print comparison ─────────────────────────────────────
echo ""
echo "============================================================"
echo "FINAL COMPARISON"
echo "============================================================"
python3 - "$SFT_DIR/summary.json" "$BASE_DIR/summary.json" <<'PYEOF'
import json, sys

with open(sys.argv[1]) as f:
    sft = json.load(f)
with open(sys.argv[2]) as f:
    base = json.load(f)

print(f"{'Metric':<25} {'SFT':>10} {'Base':>10} {'Winner':>8}")
print("-" * 58)

sm = sft.get("widget_factory_metrics", {})
bm = base.get("widget_factory_metrics", {})

for group in ["LayoutScore", "LegibilityScore", "StyleScore", "PerceptualScore", "Geometry"]:
    sg = sm.get(group, {})
    bg = bm.get(group, {})
    if isinstance(sg, dict):
        for k in sg:
            sv, bv = sg.get(k, 0), bg.get(k, 0)
            # For LPIPS (lp), lower is better
            if k == "lp":
                winner = "SFT" if sv < bv else "Base"
            else:
                winner = "SFT" if sv > bv else "Base"
            print(f"  {k:<23} {sv:>10.3f} {bv:>10.3f} {winner:>8}")
    else:
        sv, bv = sg, bg
        winner = "SFT" if sv > bv else "Base"
        print(f"  {group:<23} {sv:>10.3f} {bv:>10.3f} {winner:>8}")

print()
print(f"SFT: {sft['rendered']}/{sft['total']} rendered, {sft['evaluated']} evaluated")
print(f"Base: {base['rendered']}/{base['total']} rendered, {base['evaluated']} evaluated")
PYEOF

echo ""
echo "Completed: $(date)"
