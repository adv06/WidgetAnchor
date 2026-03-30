"""Fresh SFT training on validated data with fixed LR scheduler.

Usage:
    CUDA_VISIBLE_DEVICES=1 python -m training.run_sft_fresh
"""
import json
import os
import random
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from peft import LoraConfig, get_peft_model
from training.sft import MODEL_NAME, run_sft

load_dotenv()


def main():
    data_dir = "./output/final"
    save_dir = "/shared/advey"
    plot_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sft.png")
    device = torch.device("cuda:0")  # mapped by CUDA_VISIBLE_DEVICES

    with open(f"{data_dir}/train.json") as f:
        train_data = json.load(f)

    # filter to synthetic-only to avoid data poisoning from non-synthetic sources
    train_data = [s for s in train_data if s["widget_id"].startswith("synthetic-")]
    print(f"Filtered to {len(train_data)} synthetic-only samples")

    # shuffle for better training
    random.seed(42)
    random.shuffle(train_data)

    print(f"Loaded {len(train_data)} training samples")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device}
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.print_trainable_parameters()

    # 3 full epochs over 9873 samples with accum_steps=4 -> ~7400 optimizer steps
    # but each "step" in run_sft is one sample, so training_steps = samples * epochs
    training_steps = len(train_data) * 3
    print(f"Training for {training_steps} steps (~3 epochs, {training_steps // 4} optimizer steps)")

    model = run_sft(
        model, processor, train_data,
        training_steps=training_steps,
        lr=1e-4,
        save_dir=save_dir,
        plot_path=plot_path,
        device=device,
    )

    final_dir = f"{save_dir}/checkpoints/sft_final"
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Final model saved to {final_dir}")


if __name__ == "__main__":
    main()
