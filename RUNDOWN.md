# Codebase Rundown

## Data Pipeline (`data/`)

### `data/collection/collect_widget_factory.py`
Takes ~50K widget screenshots from `/shared/houston/widget2code/widget2code-sft/widget-factory-synthetic-data`, sends each to GPT-4o vision to reverse-engineer self-contained HTML/CSS. Saves one JSON per widget. Supports concurrent workers and is resumable (skips already-processed widgets).

### `data/processing/render_verify.py`
Re-renders each collected HTML sample via Playwright, computes SSIM against the original widget screenshot, discards samples below threshold (default 0.85). Ensures the generated HTML actually reproduces the original widget.

### `data/processing/dedup.py`
Removes near-duplicate widgets using difference hashing (dHash). Uses band-based bucketing (locality-sensitive hashing) to avoid O(n^2) comparisons. Samples with Hamming distance below threshold are considered duplicates.

### `data/processing/tag_difficulty.py`
Tags each sample as simple (<50 tokens), medium (50-200 tokens), or complex (>200 tokens) based on rough HTML token count. Used for curriculum sampling during SFT.

### `data/annotation/generate_cot.py`
Uses GPT-4o with both the screenshot and ground-truth HTML to generate structured chain-of-thought traces. Output format: `<think>` with 5 sections (structure analysis, layout plan, color/style, typography, implementation plan) followed by `<code>` with the HTML. Concurrent and resumable.

### `data/processing/split_data.py`
Assembles all annotated samples into train/val/test JSON files (90/5/5 split) ready for SFT training.

### `data/run_pipeline.sh`
Master script that runs all 6 steps sequentially. All output goes to `output/` (gitignored).

---

## Reward System (`reward/`)

### `reward/programmatic.py`
Core programmatic reward function. Contains:
- `render_html_to_image()` — renders HTML via Playwright headless browser, returns PNG bytes. Browser instance is kept alive across calls.
- `compute_reward_image()` — SSIM between two images.
- `compute_lpips()` — learned perceptual distance (AlexNet backbone), inverted so higher = better.
- `compute_palette_distance()` — k-means clustering on LAB colors, Hungarian-matched CIEDE2000 distance.
- `compute_contrast_score()` — compares grayscale standard deviations.
- `compute_polarity()` — lightness histogram correlation in LAB space.
- `compute_layout_score()` — bounding box IoU via Playwright DOM extraction (currently unused in `compute_reward_code` because we lack reference HTML).
- `compute_clip_similarity()` — CLIP cosine similarity (available but not in the reward blend since CLIP degrades RL per UI2Code^N).
- `compute_html_validity()` — checks unclosed HTML tags.
- `compute_reward_code()` — weighted blend of 5 metrics (SSIM 0.20, LPIPS 0.20, palette 0.25, contrast 0.20, polarity 0.15). Accepts optional pre-rendered image to avoid double-rendering.

### `reward/vlm_reward.py`
Decomposed VLM scoring via OpenAI API. Two functions:
- `compute_vlm_reward()` — asks GPT-4o to score a reference vs rendered image on 4 dimensions (layout, color, typography, overall) with `\boxed{}` format. Returns weighted total (color and layout weighted highest at 0.30 each).
- `compute_vlm_comparison()` — pairwise comparison of two candidates against a reference. Returns per-dimension scores and a winner ("A", "B", or "tie"). Used by round-robin.

### `reward/composite.py`
Hybrid reward: renders HTML once, passes to both programmatic (40% weight) and VLM (60% weight) scorers. Returns dict with sub-scores and total.

### `reward/round_robin.py`
Tournament ranking for GRPO rollouts. Pre-filters candidates with programmatic reward (threshold 0.3) to skip expensive VLM calls on bad rollouts. Survivors enter pairwise VLM comparison. Accumulates win counts as the reward signal.

---

## Training (`training/`)

### `training/sft.py`
Supervised fine-tuning loop. Cross-entropy loss on completion tokens only (prompt tokens masked out). Cosine LR schedule with warmup. Gradient clipping at 1.0. Checkpoints every 200 steps. Defines `SYSTEM_PROMPT` used across the project.

### `training/training_loop_grpo.py`
Custom GRPO implementation. For each step:
1. Generates N completions per prompt with temperature sampling.
2. Extracts `<code>...</code>` blocks from model output.
3. Renders HTML and scores with round-robin ranking.
4. Normalizes advantages within each prompt group.
5. Runs multiple inner epochs of PPO-clip loss with KL penalty (Schulman approximation) against a frozen reference adapter.
Attention mask is built from known sequence lengths (not token identity) to avoid PAD/EOS collision issues. Logs loss, reward, KL, clip fraction. Saves training curves as PNG.

### `training/pipeline.py`
Full training orchestrator: loads data from `output/final/train.json`, sets up model with LoRA (r=16, alpha=32), runs SFT then GRPO sequentially. Saves checkpoints after each phase.

### `training/run_grpo_only.py`
Resumes GRPO from an existing SFT checkpoint. Loads the base model + SFT LoRA adapter, then runs the GRPO loop.

### `training/training_loop_trl.py`
Alternative GRPO implementation using TRL library's `GRPOTrainer`. Not currently maintained — has broken imports and an unset dataset. Kept for reference.

---

## Inference (`inference/`)

### `inference/generate.py`
Single-shot generation: loads a LoRA-finetuned model, generates text from a prompt, extracts HTML from `<code>...</code>` tags.

### `inference/best_of_n.py`
Best-of-N sampling (Phase 3.1). Generates N candidates, scores each with `compute_reward_code`, returns the highest-scoring HTML. Typically gives 2-5 point improvement.

### `inference/polish.py`
Iterative polishing (Phase 3.2). Takes initial HTML, asks the model to improve it by comparing against the reference over M rounds. Only keeps improvements. Also provides `generate_with_polish()` which combines best-of-N + polishing (Phase 3.3).

### `inference/serve.py`
Flask web demo. Text input generates a widget, shows live preview in an iframe and the raw HTML.

---

## Evaluation (`evaluation/`)

### `evaluation/run_benchmarks.py`
Generates HTML for each Widget2Code benchmark widget using our model. Supports single-shot or best-of-N + polishing modes. Saves HTML files and rendered PNGs. Resumable. Prints instructions for running the Widget2Code evaluation scripts.

### `evaluation/setup_benchmark.sh`
Downloads the Widget2Code repo, benchmark dataset (from HuggingFace), and reference benchmark results (from Google Drive).

---

## Config Files

### `mission.md`
Full research plan: 4 phases (data pipeline, SFT, GRPO with hybrid reward, test-time scaling), benchmark targets, ablation studies, compute estimates, risk mitigation.

### `.gitignore`
Ignores `.env`, `__pycache__/`, `checkpoints/`, `plots/`, `output/`, `evaluation/widget2code/`, `evaluation/benchmarks_backup/`.
