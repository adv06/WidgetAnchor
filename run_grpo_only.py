"""Resume from SFT checkpoint and run GRPO only."""
import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training_loop_grpo import run_grpo

def main():
    load_dotenv()

    model_name = "Qwen/Qwen2.5-1.5B"
    data_dir = "./data"
    save_dir = "/shared/advey"
    device = torch.device("cuda:0")

    with open(f"{data_dir}/train.json") as f:
        train_data = json.load(f)

    grpo_prompts = [s["prompt"] for s in train_data]
    grpo_targets = []
    for sample in train_data:
        with open(sample["image_path"], "rb") as f:
            grpo_targets.append(f.read())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # load base model + SFT LoRA checkpoint
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="sdpa", dtype="auto", device_map={"": "cuda:0"}
    )
    model = PeftModel.from_pretrained(base_model, f"{save_dir}/checkpoints/sft_final", is_trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print("=" * 60)
    print("Phase 2: GRPO Reinforcement Learning (resumed from SFT checkpoint)")
    print("=" * 60)

    model = run_grpo(model, tokenizer, grpo_prompts, grpo_targets, model_name=model_name,
                     training_steps=1000, lr=5e-6, n=3, batch_size=2, num_epochs=2,
                     save_dir=save_dir, device=device)
    model.save_pretrained(f"{save_dir}/checkpoints/grpo_final")
    print("GRPO complete. Final model saved.")

if __name__ == "__main__":
    main()
