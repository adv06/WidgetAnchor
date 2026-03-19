# MISSION: Widget-to-Code Generation Research

## Goal
Train a vision-language model (VLM) that takes a screenshot of a UI widget and generates the corresponding HTML/CSS code. Beat the current SOTA (Widget2Code) on a benchmark that evaluates Layout, Legibility, Style, Perceptual similarity, and Geometry.

## Key Differentiators vs Prior Work (UI2Code^N)
1. **Widget-focused data** — we train on isolated widgets, not full webpages
2. **Structured chain-of-thought** — our CoT decomposes into layout → style → typography → code (not generic `<think>` traces)
3. **Hybrid VLM + programmatic reward** — we combine a fine-tuned VLM judge with deterministic metrics (SSIM, LPIPS, palette distance, contrast ratios) instead of a single holistic VLM score
4. **No pretraining stage** — we skip the expensive continual pretraining by starting from a strong instruct-tuned VLM

## Benchmark Targets (Widget2Code scores to beat)

| Category | Metric | Widget2Code Score | Our Target |
|----------|--------|------------------|------------|
| Layout | Margin | 72.15 | 74+ |
| Layout | Content | 66.08 | 68+ |
| Layout | Area | 82.24 | 84+ |
| Legibility | Text | 70.6 | 72+ |
| Legibility | Contrast | 66.2 | 68+ |
| Legibility | LocCon | 64.06 | 66+ |
| Style | Palette | 58.09 | 62+ |
| Style | Vibrancy | 51.38 | 55+ |
| Style | Polarity | 63.28 | 66+ |
| Perceptual | SSIM | 0.721 | 0.74+ |
| Perceptual | LPIPS↓ | 0.335 | 0.31 |
| Perceptual | CLIP | 0.838 | 0.85+ |
| Geometry | Geometry | 100 | 100 |

**Weakest areas (biggest opportunity):** Style metrics (Palette, Vibrancy, Polarity) — weight reward function heavily here.

---

## ARCHITECTURE OVERVIEW

```
Phase 0: Data Pipeline ──→ Phase 1: SFT ──→ Phase 2: RL (GRPO) ──→ Phase 3: Test-Time Scaling
   (Weeks 1-3)              (Weeks 3-5)       (Weeks 5-8)              (Weeks 8-9)
```

---

## PHASE 0: DATA PIPELINE

### 0.1 Widget Collection

**Sources (priority order):**
1. **Component libraries** (MUI, Ant Design, Chakra, shadcn/ui, Bootstrap) — scrape each component variant: buttons, cards, modals, inputs, dropdowns, tables, navbars, date pickers, tabs, accordions, alerts, tooltips. Est: 10K-20K pairs.
2. **Production sites via headless browser** — use Playwright to screenshot individual DOM components by detecting bounding boxes via DOM traversal. Target: e-commerce, dashboards, SaaS apps, landing pages. Est: 20K-40K pairs.
3. **Storybook instances** — many companies publish Storybook docs. Scrape component screenshots + source code directly. Est: 5K-15K pairs.
4. **Synthetic generation** — use a strong LLM to generate widget specs with randomized styles (colors, fonts, spacing, border-radius, shadows), then render with headless browser. Est: 20K-50K pairs.
5. **Figma community files** — export widget designs as screenshots, pair with implementation code. Est: 5K-10K pairs.

**Target: 60K-135K raw (screenshot, code) pairs. 50K+ usable after filtering.**

### 0.2 Data Processing Pipeline

For each collected widget, run these steps in order:

```python
# Step 1: Screenshot standardization
# - Render at consistent viewport width (1280px)
# - Capture at 2x DPI
# - Crop to widget bounding box with 8-16px padding
# - Save as PNG

# Step 2: Code normalization
# - Convert all code to self-contained HTML with inline CSS
# - No external dependencies
# - Strip unnecessary wrappers, normalize whitespace, remove comments
# - Ensure code actually renders when opened in browser

# Step 3: Render verification
# - Re-render normalized code with Playwright
# - Screenshot the result
# - Compute SSIM against original screenshot
# - DISCARD pairs where SSIM < 0.85

# Step 4: Deduplication
# - Perceptual hashing (pHash) on screenshots to remove near-duplicates
# - Code similarity dedup (AST-level or token-level Jaccard)

# Step 5: Difficulty tagging
# - Simple: single element, <50 tokens of code
# - Medium: compound widget, 50-200 tokens
# - Complex: multi-component, >200 tokens
```

### 0.3 Chain-of-Thought Annotation

Use a strong model (Claude Sonnet or GPT-4o) with BOTH the screenshot and ground-truth code as input. Generate structured reasoning traces.

**CoT output format (this is what the model will learn to produce):**

```
<think>
## 1. Structure Analysis
Identify the widget type. Describe the component hierarchy.
Example: "This is a card component with a header, image area, body text,
and a footer with two action buttons arranged horizontally."

## 2. Layout Plan
Specify flex/grid direction, approximate dimensions (px), padding,
margins, gaps between sections.
Maps to benchmark metrics: Margin, Content, Area.

## 3. Color & Style Extraction
List exact colors: background, primary text, secondary text, accent,
borders, shadows. Note border-radius, box-shadow values.
Maps to benchmark metrics: Palette, Vibrancy, Polarity.

## 4. Typography & Legibility
Specify font family, sizes, weights, line-heights, letter-spacing.
Check contrast ratios between text and background.
Maps to benchmark metrics: Text, Contrast, LocCon.

## 5. Implementation Plan
Outline the HTML structure and key CSS properties before writing code.
</think>
<code>
<!DOCTYPE html>
<html>
<!-- full self-contained HTML/CSS here -->
</html>
</code>
```

**CoT annotation prompt template (for generating training data):**

```
You are given:
1. A screenshot of a UI widget
2. The ground-truth HTML/CSS code that produces this widget

Generate a structured reasoning trace that decomposes the widget into:
1. Structure Analysis — what type of widget, component hierarchy
2. Layout Plan — flex/grid, dimensions, padding, margins, gaps
3. Color & Style — exact hex colors, border-radius, shadows
4. Typography — font family, sizes, weights, line-heights, contrast
5. Implementation Plan — HTML structure outline

Then include the ground-truth code.

Format your output as:
<think>[structured reasoning]</think>
<code>[the ground-truth code]</code>
```

**Quality control:** Sample 500 annotations, manually verify that color values match screenshot, layout descriptions are accurate, and structure analysis is correct. Reject batches with >20% error rate.

**Final deliverable:** 50K-100K triples of (screenshot, structured_CoT, code). Split 90/5/5 train/val/test.

---

## PHASE 1: SUPERVISED FINE-TUNING (SFT)

### 1.1 Base Model

**Recommended:** Qwen3-VL-7B-Instruct or GLM-4.1V-9B-Thinking

| Model | Params | Why |
|-------|--------|-----|
| Qwen3-VL-7B | 7B | Strong UI understanding, practical for RL, good baseline scores |
| GLM-4.1V-9B | 9B | Same base as UI2Code^N (fair comparison), scores 64.7 on Design2Code already |

Using an instruct-tuned model means we can skip continual pretraining (UI2Code^N started from a base model and needed 20M samples of pretraining).

### 1.2 Training Configuration

```yaml
input: widget screenshot + system prompt
output: "<think>[structured CoT]</think><code>[HTML/CSS]</code>"
sequence_length: 32768
learning_rate: 5e-6
batch_size: 256 (with packing)
epochs: 5
precision: bf16
optimizer: AdamW
```

### 1.3 Curriculum Strategy (optional)

- **Epochs 1-2:** All data mixed, oversample simple widgets 2x
- **Epochs 3-4:** Uniform sampling across difficulty levels
- **Epoch 5:** Oversample complex widgets 2x

### 1.4 Polishing Data (needed for Phase 3)

Include 10K-20K polishing samples in SFT training alongside generation data:

```
# How to create polishing training data:
1. Take ground-truth (screenshot, code) pairs
2. Use your SFT model (or other VLMs) to generate imperfect renderings
3. Use a strong model to generate comparison reasoning
4. Package as: input=(reference_screenshot, buggy_code, buggy_rendering)
              output=<think>[comparison reasoning]</think><code>[corrected code]</code>
```

### 1.5 Validation Gates

After each epoch, evaluate on held-out val set + Design2Code. Track:
- Render success rate (target: >95%)
- SSIM between rendered output and reference (target: >0.6)
- CLIP similarity
- Qualitative inspection of 20 random samples

**Proceed to Phase 2 when:** render success rate >95% AND SSIM >0.6 on val set.

**Fallback:** If SFT plateaus below expectations (<60 on Design2Code), do lightweight continued pretraining on 500K-1M widget pairs with DOM-level bounding box alignment loss before retrying SFT.

---

## PHASE 2: REINFORCEMENT LEARNING (GRPO)

### 2.1 Key Finding from UI2Code^N

**CLIP reward DEGRADES performance** (72.3 → 62.0 on Design2Code). VLM reward improves it (72.3 → 74.6). But they only used a single holistic 0-100 VLM score. We do better with a hybrid.

### 2.2 RL Data Pool

Separate widgets NOT seen during SFT. Only reference screenshots needed (no ground-truth code).
- 12K real-world widgets (from production sites not in SFT data)
- 18K+ synthetically generated widgets
- Total: 30K+

### 2.3 GRPO Hyperparameters

```yaml
algorithm: GRPO (Group Relative Policy Optimization)
group_size: 8-16  # UI2Code^N used 16. Start with 8 if compute-limited.
batch_size: 64
learning_rate: 1e-6
training_steps: 400-600
sampling_temperature: 0.7-0.9  # need diverse rollouts
kl_regularization: none  # UI2Code^N explicitly dropped KL to raise ceiling
max_sequence_length: 32768
```

### 2.4 Reward Function (OUR KEY INNOVATION)

For each rollout, the reward computation pipeline is:

```
Step 1: Extract code from <code>...</code> tags
Step 2: Render with Playwright in headless browser, screenshot result
Step 3: If render fails → reward = -1, stop
Step 4: Compute programmatic reward components
Step 5: Compute VLM reward components
Step 6: Combine into final reward
Step 7: Apply round-robin ranking across group
```

#### Programmatic Reward Components (R_prog)

```python
def compute_programmatic_reward(reference_img, rendered_img, reference_dom, rendered_dom):
    """
    All components normalized to [0, 1].
    """
    # 1. SSIM — structural similarity
    #    Library: scikit-image structural_similarity()
    #    Targets benchmark: SSIM
    ssim_score = structural_similarity(reference_img, rendered_img, multichannel=True)

    # 2. LPIPS — learned perceptual distance (lower is better, so invert)
    #    Library: pip install lpips, AlexNet backbone
    #    Targets benchmark: LPIPS
    lpips_score = 1.0 - lpips_model(reference_img, rendered_img)

    # 3. Palette distance — k-means on pixel colors, CIEDE2000 between clusters
    #    Library: sklearn KMeans + colormath
    #    Targets benchmark: Palette, Vibrancy
    ref_palette = extract_palette(reference_img, k=5)  # k-means clustering
    gen_palette = extract_palette(rendered_img, k=5)
    palette_score = 1.0 - normalized_ciede2000(ref_palette, gen_palette)

    # 4. Contrast ratio — extract text elements via DOM, compute WCAG ratios
    #    Library: Playwright DOM extraction + WCAG formula
    #    Targets benchmark: Contrast, Text
    contrast_score = compute_wcag_contrast_similarity(reference_dom, rendered_dom)

    # 5. Layout score — compare bounding boxes of all elements
    #    Library: Playwright element.boundingBox()
    #    Targets benchmark: Margin, Content, Area, LocCon
    layout_score = compute_bbox_similarity(reference_dom, rendered_dom)

    # 6. Polarity — lightness distribution similarity
    #    Library: convert to LAB, histogram on L channel
    #    Targets benchmark: Polarity
    polarity_score = compute_lightness_histogram_similarity(reference_img, rendered_img)

    # Weighted combination (emphasize style — biggest opportunity)
    R_prog = (0.15 * ssim_score +
              0.15 * lpips_score +
              0.25 * palette_score +  # HIGH weight — style is weakest
              0.15 * contrast_score +
              0.20 * layout_score +
              0.10 * polarity_score)

    return R_prog
```

#### VLM Reward Component (R_vlm)

Use a **fine-tuned** VLM (see Section 2.5). Unlike UI2Code^N's single holistic score, ask for decomposed scores:

```
VLM REWARD PROMPT:

You will be given two images:
- Image 1: the reference widget (target design)
- Image 2: a code rendering generated from the reference

Score EACH dimension independently (0-100):
1. Layout fidelity (positions, sizes, spacing, alignment)
2. Color accuracy (palette match, vibrancy, dark/light mode correctness)
3. Typography fidelity (font sizes, weights, contrast, readability)
4. Overall visual similarity

Strictly output in this format:
\boxed{layout: X, color: Y, typo: Z, overall: W}
```

```python
def compute_vlm_reward(reference_img, rendered_img, vlm_model):
    scores = vlm_model.score(reference_img, rendered_img)  # returns dict
    R_vlm = (0.30 * scores['layout'] / 100 +
             0.30 * scores['color'] / 100 +    # HIGH weight — style
             0.20 * scores['typo'] / 100 +
             0.20 * scores['overall'] / 100)
    return R_vlm
```

#### Composite Reward

```python
def compute_reward(reference_img, rendered_img, reference_dom, rendered_dom, vlm_model):
    # Render failure check already done upstream (reward = -1)

    R_prog = compute_programmatic_reward(reference_img, rendered_img, reference_dom, rendered_dom)
    R_vlm = compute_vlm_reward(reference_img, rendered_img, vlm_model)

    # Hybrid combination
    R_total = 0.4 * R_prog + 0.6 * R_vlm

    return R_total
```

#### Round-Robin Ranking (adopted from UI2Code^N Algo 6, optimized)

UI2Code^N's best reward algorithm uses tournament-style pairwise comparison. We adopt it with cost optimization:

```python
def round_robin_reward(candidates, reference_img, vlm_model):
    """
    candidates: list of (rendered_img, rendered_dom) for each rollout in group
    """
    # Step 1: Pre-filter with programmatic score (CHEAP)
    # Only send candidates to expensive VLM round-robin if R_prog > 0.3
    pool = []
    rewards = [-1] * len(candidates)  # default for render failures

    for i, (img, dom) in enumerate(candidates):
        if img is None:  # render failed
            rewards[i] = -1
            continue
        r_prog = compute_programmatic_reward(reference_img, img, reference_dom, dom)
        if r_prog < 0.3:
            rewards[i] = 0  # below threshold, skip VLM
        else:
            pool.append(i)
            rewards[i] = 1  # base score for being in pool

    # Step 2: Round-robin VLM comparison among pool candidates
    for i in pool:
        for j in pool:
            if i >= j:
                continue
            # Single VLM call compares both candidates against reference
            score_i, score_j = vlm_comparator(reference_img, candidates[i][0], candidates[j][0])
            if score_i > score_j:
                rewards[i] += 1
            elif score_j > score_i:
                rewards[j] += 1
            else:
                rewards[i] += 0.5
                rewards[j] += 0.5

    return rewards
```

### 2.5 Fine-Tune Your Own Verifier VLM

**CRITICAL:** UI2Code^N found that off-the-shelf VLMs were unreliable as comparators (degraded by 3%). You MUST fine-tune a verifier.

```
# Verifier training data:
1. Use your SFT model to generate multiple candidates per widget
2. Rank candidates using SSIM/LPIPS (ground truth ranking)
3. Create 5K-10K triplets: (reference, good_rendering, bad_rendering)
4. Fine-tune Qwen2.5-VL-7B on the comparator task:
   Input: reference + two candidates
   Output: decomposed scores for each candidate
5. Validate on 500 held-out triplets — verifier should agree with
   SSIM ranking >85% of the time
```

**VLM COMPARATOR PROMPT (for round-robin):**

```
You will be given three images:
- Image 1: the reference widget (target design)
- Image 2: candidate A rendering
- Image 3: candidate B rendering

For each candidate, score these dimensions (0-100):
1. Layout fidelity
2. Color accuracy
3. Typography fidelity
4. Overall similarity

Then state which candidate is closer to the reference.

Output format:
Candidate A: layout=X, color=Y, typo=Z, overall=W
Candidate B: layout=X, color=Y, typo=Z, overall=W
\boxed{Candidate A is better} or \boxed{Candidate B is better}
```

---

## PHASE 3: TEST-TIME SCALING

### 3.1 Best-of-N Sampling

At inference, generate N=4-8 candidates per widget. Score all with the reward function. Select the best. Typically gives 2-5 point improvement with no additional training.

### 3.2 Iterative Polishing

```
Round 1: screenshot → model → code_v1
Round 2: (screenshot, code_v1, render_v1) → model → code_v2
Round 3: (screenshot, code_v2, render_v2) → model → code_v3
...repeat for N=3-5 rounds
```

UI2Code^N showed +12% improvement with 4 rounds. Performance on simple widgets saturates at N=3; complex widgets keep improving through N=5.

### 3.3 Combined Strategy

```
1. Generate N=4 candidates (best-of-N)
2. Select best candidate using reward function
3. Run M=3 polishing rounds on the best candidate
4. Final output = polished version of best candidate
```

---

## PHASE 4: EVALUATION & ABLATIONS

### 4.1 Benchmarks to Evaluate On

1. **Widget2Code benchmark** (from our target table — Layout, Legibility, Style, Perceptual, Geometry)
2. **Design2Code** (Si et al., 2024)
3. **Flame-React-Eval** (Ge et al., 2025)
4. **Web2Code** (Yun et al., 2024)

### 4.2 Required Ablation Studies

Run all from the same SFT checkpoint:

| Ablation | What It Proves |
|----------|---------------|
| SFT only (no RL) | Baseline — how much RL helps |
| RL + CLIP reward only | Replicates UI2Code^N finding that CLIP degrades performance |
| RL + VLM reward only (holistic single score) | Replicates UI2Code^N approach |
| RL + programmatic reward only | Tests if deterministic metrics alone suffice |
| RL + hybrid reward (OURS) | Should be best — main contribution |
| SFT with structured CoT vs no CoT | Shows value of reasoning |
| Structured CoT vs generic CoT | Shows our decomposition > generic `<think>` |
| Round-robin vs vanilla verifier | Confirms UI2Code^N finding |

### 4.3 Key Analysis

- Which reward component (prog vs VLM) helps which metric most?
- Per-metric breakdown: where do we beat Widget2Code and where don't we?
- Qualitative examples of style details we get right that others miss
- Cost-benefit analysis of round-robin with pre-filter vs full round-robin

---

## COMPUTE ESTIMATES (7-9B model)

| Component | GPUs | Duration |
|-----------|------|----------|
| SFT (50K samples, 5 epochs, seq_len 32K) | 4-8 A100 80GB | 3-5 days |
| Verifier VLM fine-tuning (10K samples, 3 epochs) | 2 A100 | 1 day |
| GRPO (400-600 steps, G=8-16, batch=64) | 8-16 A100 | 5-7 days |
| VLM reward calls during RL | Additional GPUs for verifier | concurrent |

**RL is the most expensive** due to: generation + rendering + VLM scoring per rollout.
Estimated ~700K VLM inference calls total during RL (400 steps × 64 batch × ~28 calls per sample with pre-filter + G=8 round-robin).

---

## RISK MITIGATION

| Risk | Mitigation |
|------|-----------|
| SFT can't generate valid HTML reliably (<95% render rate) | Add lightweight continued pretraining on 500K widget pairs, or switch to stronger base model |
| Reward hacking during RL | Monitor qualitative samples every 50 steps. Add format penalty. Consider light KL penalty. |
| VLM verifier too noisy | Invest more in verifier fine-tuning. Fallback: programmatic-only reward. |
| Can't beat Widget2Code on style metrics | Increase palette/vibrancy weight in reward. Add color augmentation to RL data. |
| Compute budget exceeded | Reduce G to 4, fewer RL steps (200), prioritize top 3 ablations. |

---

## PAPER FRAMING

**Suggested title:** "WidgetCoder: Structured Reasoning and Metric-Decomposed Rewards for High-Fidelity Widget-to-Code Generation"

**Four novelty claims:**
1. Widget-focused data layer (vs full webpages)
2. Structured CoT decomposition mirroring evaluation axes
3. Hybrid VLM + programmatic metric-decomposed reward
4. Efficient two-stage pipeline (no pretraining needed)

**Key experiments reviewers will want:**
- Hybrid reward > VLM-only > CLIP (ablation table)
- Structured CoT > generic CoT > no CoT (ablation table)
- Per-component reward analysis (which component helps which metric)
- 2-stage vs 3-stage comparison with same base model
- Qualitative style-detail examples

---

## FILE STRUCTURE (suggested repo layout)

```
widget-coder/
├── README.md
├── MISSION.md                    # this file
├── data/
│   ├── collection/
│   │   ├── scrape_component_libs.py
│   │   ├── scrape_production_sites.py
│   │   ├── scrape_storybook.py
│   │   └── generate_synthetic.py
│   ├── processing/
│   │   ├── normalize_code.py
│   │   ├── render_verify.py      # Playwright render + SSIM check
│   │   ├── dedup.py
│   │   └── tag_difficulty.py
│   ├── annotation/
│   │   ├── generate_cot.py       # calls Claude/GPT-4o for CoT
│   │   └── quality_check.py
│   └── polishing/
│       └── generate_polish_data.py
├── training/
│   ├── sft/
│   │   ├── config.yaml
│   │   ├── train_sft.py
│   │   └── validate_sft.py
│   ├── verifier/
│   │   ├── create_verifier_data.py
│   │   ├── train_verifier.py
│   │   └── validate_verifier.py
│   └── rl/
│       ├── config.yaml
│       ├── train_grpo.py
│       ├── reward/
│       │   ├── programmatic.py   # SSIM, LPIPS, palette, contrast, layout, polarity
│       │   ├── vlm_reward.py     # decomposed VLM scoring
│       │   ├── composite.py      # hybrid combination
│       │   └── round_robin.py    # tournament ranking
│       └── render_pipeline.py    # Playwright rendering for rollouts
├── inference/
│   ├── generate.py               # single-shot generation
│   ├── best_of_n.py              # best-of-N sampling
│   └── polish.py                 # iterative polishing loop
├── evaluation/
│   ├── run_benchmarks.py
│   ├── ablation_runner.py
│   └── metrics/
│       ├── layout_metrics.py
│       ├── style_metrics.py
│       ├── perceptual_metrics.py
│       └── legibility_metrics.py
└── paper/
    ├── figures/
    └── tables/
```

---

## CHECKLIST

- [ ] **PHASE 0:** Widget scraping pipeline built and tested
- [ ] **PHASE 0:** 50K+ (screenshot, code) pairs collected and verified
- [ ] **PHASE 0:** CoT annotations generated and quality-checked
- [ ] **PHASE 0:** Train/val/test split created (90/5/5)
- [ ] **PHASE 1:** Base model selected and environment configured
- [ ] **PHASE 1:** SFT training completed (5 epochs)
- [ ] **PHASE 1:** Render success rate >95% on val set
- [ ] **PHASE 1:** Baseline benchmark scores recorded
- [ ] **PHASE 2:** RL widget pool prepared (30K+ separate widgets)
- [ ] **PHASE 2:** Programmatic reward pipeline built and tested
- [ ] **PHASE 2:** Verifier VLM fine-tuned and validated (>85% agreement)
- [ ] **PHASE 2:** GRPO training completed (400-600 steps)
- [ ] **PHASE 2:** RL model beats SFT model on all metrics
- [ ] **PHASE 3:** Best-of-N sampling implemented and tested
- [ ] **PHASE 3:** Iterative polishing pipeline working
- [ ] **PHASE 3:** Polishing improves scores over N=3-5 rounds
- [ ] **PHASE 4:** Full benchmark evaluation completed
- [ ] **PHASE 4:** All ablation studies run
- [ ] **PHASE 4:** Widget2Code beaten on majority of metrics
- [ ] **PHASE 4:** Paper draft completed
