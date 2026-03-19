"""Resume from SFT checkpoint and run GRPO only."""
import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training.training_loop_grpo import run_grpo
from training.sft import SYSTEM_PROMPT


def main():
    load_dotenv()

    model_name = "Qwen/Qwen2.5-1.5B"
    data_dir = "./output/final"
    save_dir = "/shared/advey"
    device = torch.device("cuda:0")

    with open(f"{data_dir}/train.json") as f:
        train_data = json.load(f)

    # for GRPO, prompts are the task instruction; targets are reference screenshot bytes
    grpo_prompts = [SYSTEM_PROMPT + "\nRecreate the widget shown in the reference image." for _ in train_data]

    grpo_targets = []
    for sample in train_data:
        with open(sample["screenshot_path"], "rb") as f:
            grpo_targets.append(f.read())

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="sdpa", torch_dtype="auto", device_map={"": "cuda:0"}
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
