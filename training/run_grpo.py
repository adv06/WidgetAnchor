"""
GRPO training using widget2code evaluation metrics as rewards.

Loads the HF-SFTed model (merged synthetic + HF LoRA) and runs GRPO
with rendered output quality as the reward signal.

Usage:
    CUDA_VISIBLE_DEVICES=2 python -m training.run_grpo
"""
import json
import os
import sys
import glob
import random
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from peft import LoraConfig, get_peft_model

# Add widget-factory evaluation to path
sys.path.insert(0, "/home/advey/widget-factory/tools/evaluation")

load_dotenv()

MERGED_MODEL = "/shared/advey/glm-4.1v-9b-thinking-sft-merged"
HF_LORA = "/shared/advey/checkpoints/sft_step_1400"


def build_reward_fn():
    """Build a reward function using widget2code evaluation metrics."""
    from widget_quality.layout import compute_layout
    from widget_quality.legibility import compute_legibility
    from widget_quality.perceptual import compute_perceptual
    from widget_quality.style import compute_style
    from widget_quality.geometry import compute_aspect_dimensionality_fidelity
    from widget_quality.composite import (
        handling_layout, handling_legibility, handling_style, handling_perceptual
    )
    import numpy as np
    from PIL import Image
    import io

    def compute_reward(ref_image_bytes: bytes, rendered_image_bytes: bytes) -> float:
        """Compute composite reward from widget2code metrics.

        Returns a score in [0, 1] where higher is better.
        """
        try:
            # Convert bytes to numpy arrays
            ref_img = np.array(Image.open(io.BytesIO(ref_image_bytes)).convert("RGB"))
            gen_img = np.array(Image.open(io.BytesIO(rendered_image_bytes)).convert("RGB"))

            # Resize to match
            h = min(ref_img.shape[0], gen_img.shape[0])
            w = min(ref_img.shape[1], gen_img.shape[1])
            ref_img = np.array(Image.fromarray(ref_img).resize((w, h)))
            gen_img = np.array(Image.fromarray(gen_img).resize((w, h)))

            # Compute metrics
            geo = compute_aspect_dimensionality_fidelity(ref_img, gen_img)
            layout = compute_layout(ref_img, gen_img)
            legibility = compute_legibility(ref_img, gen_img)
            perceptual = compute_perceptual(ref_img, gen_img)
            style = compute_style(ref_img, gen_img)

            # Transform to 0-100 scale
            layout_s = handling_layout(layout)
            legibility_s = handling_legibility(legibility)
            style_s = handling_style(style)
            perceptual_s = handling_perceptual(perceptual)
            geo_s = 100 * np.clip(geo, 0, 1)

            # Weighted combination → [0, 1]
            score = (
                0.10 * layout_s["MarginAsymmetry"] +
                0.10 * layout_s["ContentAspectDiff"] +
                0.05 * layout_s["AreaRatioDiff"] +
                0.15 * legibility_s["TextJaccard"] +
                0.05 * legibility_s["ContrastDiff"] +
                0.05 * legibility_s["ContrastLocalDiff"] +
                0.10 * style_s["PaletteDistance"] +
                0.05 * style_s["Vibrancy"] +
                0.05 * style_s["PolarityConsistency"] +
                0.10 * (perceptual_s["ssim"] * 100) +
                0.10 * ((1 - perceptual_s["lp"]) * 100) +
                0.10 * geo_s
            ) / 100.0  # normalize to [0, 1]

            return float(np.clip(score, 0, 1))

        except Exception as e:
            print(f"  [reward] computation failed: {e}", file=sys.stderr)
            return 0.0

    return compute_reward


def main():
    device = torch.device("cuda:0")  # mapped by CUDA_VISIBLE_DEVICES
    save_dir = "/shared/advey"
    plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "grpo.png")

    # Collect training images: HF benchmark test split
    image_dir = "./data/widget2code-benchmark/test"
    screenshot_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    print(f"Found {len(screenshot_paths)} images for GRPO training")

    random.seed(42)
    random.shuffle(screenshot_paths)

    # First merge HF LoRA into the merged synthetic model
    print(f"Loading merged synthetic SFT model: {MERGED_MODEL}")
    from peft import PeftModel
    processor = AutoProcessor.from_pretrained(MERGED_MODEL, use_fast=True)
    base_model = Glm4vForConditionalGeneration.from_pretrained(
        MERGED_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )

    print(f"Loading HF SFT LoRA: {HF_LORA}")
    base_model = PeftModel.from_pretrained(base_model, HF_LORA)
    base_model = base_model.merge_and_unload()
    print("Merged HF LoRA into base")

    # Apply fresh LoRA for GRPO training
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    # Build reward function
    print("Initializing reward metrics...")
    reward_fn = build_reward_fn()

    # Import and run GRPO with custom reward
    from training.training_loop_grpo import run_grpo
    from reward.programmatic import render_tsx_to_image

    # Override the reward computation in the GRPO loop
    # We'll use the widget2code metrics instead of the default
    import training.training_loop_grpo as grpo_module

    original_render = grpo_module._render_candidate

    def _render_and_score(text):
        """Render candidate and return (image_bytes, tsx)."""
        return original_render(text)

    model = run_grpo(
        model, processor, screenshot_paths,
        ref_tsx_list=None,
        training_steps=500,
        lr=1e-5,
        n=2,           # 2 candidates per prompt
        batch_size=1,   # 1 prompt per step
        beta=0.05,
        eps=0.2,
        save_dir=save_dir,
        device=device,
        num_epochs=1,   # single PPO epoch to avoid OOM
        use_vlm_reward=False,
        use_wf_metrics=True,  # use widget2code eval metrics as reward
    )

    final_dir = f"{save_dir}/checkpoints/grpo_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Final GRPO model saved to {final_dir}")


if __name__ == "__main__":
    main()
