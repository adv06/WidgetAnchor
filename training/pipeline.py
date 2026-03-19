import json
import os
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from training.sft import run_sft, SYSTEM_PROMPT
from training.training_loop_grpo import run_grpo

def main():
    load_dotenv()

    # ============================================================
    # Config
    # ============================================================
    model_name = "Qwen/Qwen2.5-1.5B"
    data_dir = "./output/final"
    save_dir = "/shared/advey"
    device = torch.device("cuda:0")

    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/plots", exist_ok=True)

    # ============================================================
    # Load data
    # ============================================================
    with open(f"{data_dir}/train.json") as f:
        train_data = json.load(f)

    # SFT: input is system prompt + task, target is CoT + code
    sft_prompts = [SYSTEM_PROMPT + "\nRecreate the widget shown in the reference image." for _ in train_data]
    sft_targets = [s["cot"] for s in train_data]

    # GRPO: same prompts, targets are reference screenshot bytes for reward
    grpo_prompts = sft_prompts
    grpo_targets = []
    for sample in train_data:
        with open(sample["screenshot_path"], "rb") as f:
            grpo_targets.append(f.read())

    # ============================================================
    # Model setup
    # ============================================================
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="sdpa",
        torch_dtype="auto",
        device_map="auto"
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, lora_config)

    # SFT
    print("=" * 60)
    print("Phase 1: Supervised Fine-Tuning")
    print("=" * 60)

    model = run_sft(model, tokenizer, sft_prompts, sft_targets, training_steps=500, lr=1e-4, save_dir=save_dir, device=device)
    model.save_pretrained(f"{save_dir}/checkpoints/sft_final")
    print("SFT complete.\n")

    # GRPO
    print("=" * 60)
    print("Phase 2: GRPO Reinforcement Learning")
    print("=" * 60)

    model = run_grpo(model, tokenizer, grpo_prompts, grpo_targets, model_name=model_name,
                     training_steps=1000, lr=1e-5, n=3, batch_size=2, num_epochs=4,
                     save_dir=save_dir, device=device)
    model.save_pretrained(f"{save_dir}/checkpoints/grpo_final")
    print("GRPO complete. Final model saved.")

if __name__ == "__main__":
    main()
