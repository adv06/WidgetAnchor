# WidgetAnchor

Fine-tunes GLM-4.1V-9B-Thinking (a 9B-param VLM) with LoRA to reverse-engineer UI widgets: screenshot → React+Tailwind TSX component. Pipeline: data collection → SFT → GRPO → inference → evaluation.

## Overview

WidgetAnchor fine-tunes `zai-org/GLM-4.1V-9B-Thinking` in two phases:

1. **Supervised Fine-Tuning (SFT)** — cross-entropy on structured chain-of-thought completions (5-section reasoning trace + React component)
2. **Group Relative Policy Optimization (GRPO)** — custom PPO-clip implementation with hybrid reward: programmatic image similarity metrics + VLM pairwise judging via round-robin tournament

The model takes a widget screenshot as input and outputs `<think>CoT reasoning</think><code>React+Tailwind TSX</code>`.

## Project Structure

```
data/
  collection/
    collect_widget_factory.py   # Reverse-engineer 10K widget screenshots via GPT-4o
    generate_synthetic.py       # Generate 10K+ synthetic widgets from random categories
  processing/
    render_verify.py            # Render TSX, filter by SSIM >= 0.85
    dedup.py                    # dHash + LSH deduplication
    tag_difficulty.py           # simple/medium/complex by token count
    split_data.py               # 90/5/5 train/val/test split
  annotation/
    generate_cot.py             # GPT-4o generates structured CoT traces
  run_pipeline.sh               # Master script for full data pipeline

training/
  sft.py                        # SFT loop (defines SYSTEM_PROMPT, shared constants)
  training_loop_grpo.py         # Custom GRPO with reference adapter trick
  pipeline.py                   # Orchestrates SFT → GRPO
  run_grpo_only.py              # Resume GRPO from SFT checkpoint

reward/
  programmatic.py               # 5-metric blend (SSIM, LPIPS, palette, contrast, polarity)
  vlm_reward.py                 # GPT-4o absolute + pairwise scoring
  round_robin.py                # Tournament ranking for GRPO rollouts
  composite.py                  # 40% programmatic + 60% VLM hybrid

inference/
  generate.py                   # Single-shot generation
  best_of_n.py                  # N candidates → highest scoring
  polish.py                     # Iterative multi-round refinement

evaluation/
  run_benchmarks.py             # Generate TSX + PNG for Widget2Code benchmark
  setup_benchmark.sh            # Download benchmark data

render/                         # esbuild + Playwright TSX → PNG pipeline
```

## Setup

```bash
pip install torch transformers peft einops flask playwright openai scikit-image opencv-python lpips
playwright install chromium
cd render && npm install
```

Requires an `.env` file with `OPENAI_API_KEY` for GPT-4o calls (data collection, CoT annotation, VLM reward).

## Usage

### 1. Collect and generate training data

Two parallel data collection paths writing to `output/raw/` with non-colliding ID prefixes (`widget-*` and `synthetic-*`):

```bash
# Full pipeline (collection + processing + annotation + split)
bash data/run_pipeline.sh

# Or run individually:
python -m data.collection.collect_widget_factory --num_samples 10000 --output_dir ./output/raw --workers 8
python -m data.collection.generate_synthetic --num_samples 10000 --output_dir ./output/raw --workers 8
```

### 2. Train

Runs SFT then GRPO. Checkpoints saved to `checkpoints/`.

```bash
python -m training.pipeline
```

Or resume GRPO from an existing SFT checkpoint:

```bash
python -m training.run_grpo_only
```

### 3. Inference

```bash
# Single-shot
python -m inference.generate --checkpoint /path/to/checkpoint --image widget.png

# Best-of-N + iterative polish
python -m inference.polish --checkpoint /path/to/checkpoint --image widget.png
```

### 4. Evaluate

```bash
bash evaluation/setup_benchmark.sh
python -m evaluation.run_benchmarks
```

## Reward Signal

### Programmatic (`compute_reward_code`) — weighted blend of 5 metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| SSIM | 0.15 | Structural similarity |
| LPIPS | 0.15 | Perceptual distance (AlexNet) |
| Palette | 0.20 | K-means color clustering + Hungarian matching |
| Contrast | 0.15 | Grayscale std deviation similarity |
| Polarity | 0.10 | Lightness histogram correlation (LAB space) |
| Layout   | 0.20 | Intersection over Union | 

### GRPO reward — round-robin tournament:
1. Pre-filter: programmatic score < 0.3 → eliminated
2. All-pairs VLM pairwise comparison on survivors
3. Win counts = reward signal

## Key Hyperparameters

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Learning rate | 1e-4 | 1e-5 |
| LoRA rank / alpha | 16 / 32 | 16 / 32 |
| Clipping (eps) | - | 0.2 |
| KL penalty (beta) | - | 0.05 |
| Generations per prompt (n) | - | 5 |
| Inner epochs | - | 4 |

## Model

- **Base**: `zai-org/GLM-4.1V-9B-Thinking` (9B param VLM)
- **Adaptation**: LoRA r=16, α=32, dropout=0.05, targets: q/k/v/o projections
- **Precision**: bfloat16 with gradient checkpointing
