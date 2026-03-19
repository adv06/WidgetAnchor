"""Resume from SFT checkpoint and run GRPO only."""
import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from peft import PeftModel
from training.training_loop_grpo import run_grpo
from training.sft import MODEL_NAME


def main():
    load_dotenv()

    data_dir = "./output/final"
    save_dir = "/shared/advey"
    device = torch.device("cuda:0")

    with open(f"{data_dir}/train.json") as f:
        train_data = json.load(f)

    screenshot_paths = [s["screenshot_path"] for s in train_data]
    ref_tsx_list = [s["tsx"] for s in train_data]

    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    base_model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model = PeftModel.from_pretrained(base_model, f"{save_dir}/checkpoints/sft_final", is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("=" * 60)
    print("Phase 2: GRPO Reinforcement Learning (resumed from SFT checkpoint)")
    print("=" * 60)

    model = run_grpo(model, processor, screenshot_paths, ref_tsx_list=ref_tsx_list, model_name=MODEL_NAME,
                     training_steps=1000, lr=5e-6, n=3, batch_size=2, num_epochs=2,
                     save_dir=save_dir, device=device, use_vlm_reward=False)
    model.save_pretrained(f"{save_dir}/checkpoints/grpo_final")
    print("GRPO complete. Final model saved.")

if __name__ == "__main__":
    main()
