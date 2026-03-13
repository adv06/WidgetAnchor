# WidgetAnchor V1

Train a language model to generate HTML/CSS widgets from natural language descriptions using SFT + GRPO reinforcement learning.

## Overview

WidgetAnchor fine-tunes Qwen2.5-7B (with LoRA) in two phases:

1. **Supervised Fine-Tuning (SFT)** — teaches the model the HTML/CSS generation format using teacher forcing against ground truth code
2. **Group Relative Policy Optimization (GRPO)** — improves visual quality by rewarding generations that visually match target widget screenshots (SSIM + HTML validity + VLM scoring)

The GRPO implementation is written from scratch (not using TRL) with the following optimizations:
- Per-token PPO clipping with Schulman KL approximation
- vLLM for fast generation (PagedAttention, continuous batching)
- LoRA with merge/unmerge for vLLM weight sync
- Flash Attention 2, bf16 mixed precision, torch.compile
- Gradient accumulation with cosine LR scheduling

## Project Structure

```
generate_data.py          # Dataset generation (GPT-4o + Playwright rendering)
sft.py                    # Supervised fine-tuning loop
training_loop_grpo.py     # Custom GRPO training loop
training_loop_trl.py      # Alternative TRL-based GRPO (for reference)
pipeline.py               # Orchestrates SFT -> GRPO training
reward.py                 # Reward model (SSIM, HTML validity, VLM)
serve.py                  # Flask inference server with live preview
```

## Setup

```bash
pip install torch transformers peft vllm einops flask playwright openai
playwright install chromium
```

## Usage

### 1. Generate training data

Uses GPT-4o to create HTML widgets across 40 widget types and 10 style variants, then renders them to screenshots with Playwright.

```bash
python generate_data.py
```

This creates `./data/train.json` and `./data/val.json` with prompt-HTML-image triples.

### 2. Train

Runs SFT (500 steps) then GRPO (1000 steps). Checkpoints are saved to `./checkpoints/`.

```bash
python pipeline.py
```

### 3. Serve

Launches a local web UI where you can type widget descriptions and see generated HTML rendered live.

```bash
python serve.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Reward Signal

The reward for each generated HTML widget is a weighted combination of:

| Component | Weight | Description |
|-----------|--------|-------------|
| SSIM | 0.5 | Structural similarity between rendered screenshot and target image |
| HTML validity | 0.2 | Whether the generated HTML parses correctly |
| VLM score | 0.3 | Vision-language model assessment of visual fidelity |

## Configuration

Key hyperparameters in `pipeline.py`:

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Learning rate | 1e-4 | 1e-5 |
| Training steps | 500 | 1000 |
| LoRA rank | 16 | 16 |
| Grad accumulation | - | 4 |
| Clipping (eps) | - | 0.2 |
| KL penalty (beta) | - | 0.05 |
| Generations per prompt (n) | - | 5 |
