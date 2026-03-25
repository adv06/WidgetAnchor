#!/bin/bash
# Run SFT vs Base benchmark comparison.
#
# Usage:
#   bash evaluation/run_benchmark.sh              # full run (558 images)
#   bash evaluation/run_benchmark.sh --limit 10   # quick test (10 images)
#
# Prerequisites:
#   - SGLang servers running (or this script starts them)
#   - pip install aiohttp  (for async API calls)

set -euo pipefail
cd "$(dirname "$0")/.."  # project root

LIMIT_FLAG="${1:-}"
LIMIT_VAL="${2:-}"
EXTRA_ARGS=""
if [ "$LIMIT_FLAG" = "--limit" ] && [ -n "$LIMIT_VAL" ]; then
    EXTRA_ARGS="--limit $LIMIT_VAL"
fi

# ── Configuration ────────────────────────────────────────────────────────────
SFT_MODEL_PATH="/shared/advey/checkpoints/sft_final_v2"
BASE_MODEL_NAME="zai-org/GLM-4.1V-9B-Thinking"

SFT_PORT=8000
BASE_PORT=8001
SFT_GPU=3
BASE_GPU=4

SFT_URL="http://localhost:${SFT_PORT}/v1"
BASE_URL="http://localhost:${BASE_PORT}/v1"

TEST_DIR="/home/advey/widget-factory/test_split"
SFT_OUTPUT="results/sft-benchmark"
BASE_OUTPUT="results/base-benchmark"

# ── Helper: check if server is up ────────────────────────────────────────────
wait_for_server() {
    local url="$1"
    local name="$2"
    local max_wait=300  # 5 minutes
    local waited=0

    echo "Waiting for $name at $url ..."
    while ! curl -s "${url}/models" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if [ $waited -ge $max_wait ]; then
            echo "ERROR: $name did not start within ${max_wait}s"
            exit 1
        fi
    done
    echo "$name is ready (waited ${waited}s)"
}

# ── Start SGLang servers if not running ──────────────────────────────────────
start_server() {
    local port="$1"
    local gpu="$2"
    local model="$3"
    local name="$4"
    local url="http://localhost:${port}/v1"

    if curl -s "${url}/models" > /dev/null 2>&1; then
        echo "$name already running on port $port"
        return
    fi

    echo "Starting $name on GPU $gpu, port $port ..."
    CUDA_VISIBLE_DEVICES=$gpu python -m sglang.launch_server \
        --model-path "$model" \
        --port "$port" \
        --tp 1 \
        --dtype bfloat16 \
        --trust-remote-code \
        > "logs/${name}.log" 2>&1 &

    wait_for_server "$url" "$name"
}

mkdir -p logs

echo "============================================================"
echo "  SFT vs Base Benchmark"
echo "============================================================"

# Optionally start servers (comment out if you manage them externally)
# start_server $SFT_PORT $SFT_GPU "$SFT_MODEL_PATH" "sft-server"
# start_server $BASE_PORT $BASE_GPU "$BASE_MODEL_NAME" "base-server"

# ── Run SFT benchmark ───────────────────────────────────────────────────────
echo ""
echo ">>> Running SFT model benchmark ..."
python -m evaluation.run_sft_benchmark \
    --model-url "$SFT_URL" \
    --model-name sft \
    --test-dir "$TEST_DIR" \
    --output-dir "$SFT_OUTPUT" \
    --concurrency 4 \
    --cuda \
    $EXTRA_ARGS

# ── Run Base benchmark ──────────────────────────────────────────────────────
echo ""
echo ">>> Running Base model benchmark ..."
python -m evaluation.run_sft_benchmark \
    --model-url "$BASE_URL" \
    --model-name base \
    --test-dir "$TEST_DIR" \
    --output-dir "$BASE_OUTPUT" \
    --concurrency 4 \
    --cuda \
    $EXTRA_ARGS

# ── Print comparison ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  COMPARISON"
echo "============================================================"

python3 -c "
import json, sys

def load_summary(path):
    with open(path) as f:
        return json.load(f)

sft = load_summary('$SFT_OUTPUT/summary.json')
base = load_summary('$BASE_OUTPUT/summary.json')

print(f'{'Metric':>30s} | {'SFT':>10s} | {'Base':>10s} | {'Delta':>10s}')
print('-' * 68)

# Reward
sr, br = sft['reward_mean'], base['reward_mean']
print(f'{\"Reward\":>30s} | {sr:10.3f} | {br:10.3f} | {sr-br:+10.3f}')

# Widget-factory metrics
for group in ['PerceptualScore', 'LayoutScore', 'LegibilityScore', 'StyleScore', 'Geometry']:
    sg = sft['widget_factory_metrics'].get(group, {})
    bg = base['widget_factory_metrics'].get(group, {})
    if isinstance(sg, dict):
        for k in sg:
            sv, bv = sg.get(k, 0), bg.get(k, 0)
            print(f'{group+\"/\"+k:>30s} | {sv:10.3f} | {bv:10.3f} | {sv-bv:+10.3f}')
    else:
        print(f'{group:>30s} | {sg:10.3f} | {bg:10.3f} | {sg-bg:+10.3f}')

print()
print(f'SFT: {sft[\"successful\"]}/{sft[\"total\"]} successful ({sft[\"errors\"]} errors)')
print(f'Base: {base[\"successful\"]}/{base[\"total\"]} successful ({base[\"errors\"]} errors)')
"

echo ""
echo "Done! Results saved to:"
echo "  SFT:  $SFT_OUTPUT/summary.json"
echo "  Base: $BASE_OUTPUT/summary.json"
