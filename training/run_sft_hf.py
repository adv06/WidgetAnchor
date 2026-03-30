"""Continue SFT on HF widget2code-benchmark data, starting from merged synthetic SFT model.

Loads the merged model (base + synthetic LoRA already baked in), applies a fresh LoRA,
then trains on HF-generated data.

Usage:
    CUDA_VISIBLE_DEVICES=0 python -m training.run_sft_hf
"""
import json
import os
import glob
import random
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from peft import LoraConfig, get_peft_model
from training.sft import run_sft

load_dotenv()

MERGED_MODEL = "/shared/advey/glm-4.1v-9b-thinking-sft-merged"


def load_hf_data(raw_dir: str = "./output/raw") -> list[dict]:
    """Load generated CoT+TSX data for HF benchmark images."""
    files = sorted(glob.glob(os.path.join(raw_dir, "hf-*.json")))
    samples = []
    for f in files:
        with open(f) as fh:
            sample = json.load(fh)
        if all(k in sample for k in ("widget_id", "screenshot_path", "cot")):
            if "export default" in sample.get("tsx", ""):
                samples.append(sample)
    return samples


def main():
    save_dir = "/shared/advey"
    plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sft_hf.png")
    device = torch.device("cuda:0")

    # Load HF data
    hf_data = load_hf_data()
    print(f"Loaded {len(hf_data)} HF benchmark samples")
    if len(hf_data) == 0:
        print("No HF data found! Run generate_from_images.py first.")
        return

    random.seed(42)
    random.shuffle(hf_data)

    # Load merged model (base + synthetic SFT already baked in)
    print(f"Loading merged SFT model: {MERGED_MODEL}")
    processor = AutoProcessor.from_pretrained(MERGED_MODEL, use_fast=True)
    model = Glm4vForConditionalGeneration.from_pretrained(
        MERGED_MODEL, torch_dtype=torch.bfloat16, device_map={"": device}
    )

    # Fresh LoRA for HF fine-tuning
    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    # 3 epochs, lower LR since we're fine-tuning an already-SFTed model
    epochs = 3
    training_steps = len(hf_data) * epochs
    lr = 5e-5
    print(f"Training for {training_steps} steps (~{epochs} epochs, {training_steps // 4} optimizer steps)")
    print(f"LR: {lr}")

    model = run_sft(
        model, processor, hf_data,
        training_steps=training_steps,
        lr=lr,
        save_dir=save_dir,
        plot_path=plot_path,
        device=device,
    )

    final_dir = f"{save_dir}/checkpoints/sft_hf_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Final HF-tuned model saved to {final_dir}")


if __name__ == "__main__":
    main()
