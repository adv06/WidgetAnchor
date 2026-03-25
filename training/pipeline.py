import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from peft import LoraConfig, get_peft_model
from training.sft import run_sft, MODEL_NAME
from training.training_loop_grpo import run_grpo

def main():
    load_dotenv()

    data_dir = "./output/final"
    save_dir = "/shared/advey"
    device = torch.device("cuda:0")  # use CUDA_VISIBLE_DEVICES to select physical GPU

    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)

    # ============================================================
    # Load data
    # ============================================================
    with open(f"{data_dir}/train.json") as f:
        train_data = json.load(f)

    screenshot_paths = [s["screenshot_path"] for s in train_data]
    ref_tsx_list = [s["tsx"] for s in train_data]

    # ============================================================
    # Model setup — GLM-4.1V-9B-Thinking (VLM)
    # ============================================================
    processor = AutoProcessor.from_pretrained(MODEL_NAME, use_fast=True)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = Glm4vForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config)

    # SFT
    print("=" * 60)
    print(f"Phase 1: Supervised Fine-Tuning ({MODEL_NAME})")
    print("=" * 60)

    # scale training steps with dataset size
    num_samples = len(train_data)
    sft_steps = max(500, num_samples * 3)  # ~3 epochs over data
    grpo_steps = max(400, num_samples * 2)  # ~2 epochs

    print(f"Training data: {num_samples} samples")
    print(f"SFT steps: {sft_steps} | GRPO steps: {grpo_steps}")
    model = run_sft(model, processor, train_data, training_steps=sft_steps, lr=1e-4, save_dir=save_dir, device=device)
    model.save_pretrained(f"{save_dir}/checkpoints/sft_final")
    print("SFT complete.\n")

    # GRPO
    print("=" * 60)
    print("Phase 2: GRPO Reinforcement Learning")
    print("=" * 60)

    model = run_grpo(model, processor, screenshot_paths, ref_tsx_list=ref_tsx_list, model_name=MODEL_NAME,
                     training_steps=grpo_steps, lr=1e-5, n=4, batch_size=1, num_epochs=4,
                     save_dir=save_dir, device=device, use_vlm_reward=False)
    model.save_pretrained(f"{save_dir}/checkpoints/grpo_final")
    print("GRPO complete. Final model saved.")

if __name__ == "__main__":
    main()
