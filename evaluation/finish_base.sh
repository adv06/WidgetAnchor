#!/bin/bash
# Wait for base inference to finish, then render + evaluate
set -e
cd /home/advey/WidgetAnchor

echo "Waiting for base inference to complete..."
while true; do
    # Check if hf-base tmux session is still running a python process
    if ! tmux has-session -t hf-base 2>/dev/null; then
        echo "hf-base session ended"
        break
    fi
    count=$(ls /shared/advey/hf-benchmark/base/*/raw_output.txt 2>/dev/null | wc -l)
    echo "  Base inferred: $count/500 ($(date +%H:%M))"
    sleep 120
done

echo ""
echo "=== Re-extracting and rendering base ==="
python -m evaluation.rerender /shared/advey/hf-benchmark/base /shared/advey/hf-benchmark/gt

echo ""
echo "=== Evaluating base ==="
python -m evaluation.run_sft_benchmark \
    --eval-only \
    --test-dir /shared/advey/hf-benchmark/gt \
    --output-dir /shared/advey/hf-benchmark/base

echo ""
echo "=== COMPARISON ==="
python3 - /shared/advey/hf-benchmark/sft/summary.json /shared/advey/hf-benchmark/base/summary.json <<'PYEOF'
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
            if k == "lp":
                winner = "SFT" if sv < bv else "Base"
            else:
                winner = "SFT" if sv > bv else "Base"
            print(f"  {k:<23} {sv:>10.3f} {bv:>10.3f} {winner:>8}")

print()
print(f"SFT: {sft['rendered']}/{sft['total']} rendered, {sft['evaluated']} evaluated")
print(f"Base: {base['rendered']}/{base['total']} rendered, {base['evaluated']} evaluated")
PYEOF

echo ""
echo "Done: $(date)"
