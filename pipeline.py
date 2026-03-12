import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from sft import run_sft
from training_loop_grpo import run_grpo

# ============================================================
# Config
# ============================================================
model_name = "Qwen/Qwen2.5-7B"
data_dir = "./data"
device = torch.device("cuda:0")

# ============================================================
# Load data
# ============================================================
with open(f"{data_dir}/train.json") as f:
    train_data = json.load(f)

sft_prompts = [s["prompt"] for s in train_data]
sft_code_gt = [s["html"] for s in train_data]
grpo_prompts = [s["prompt"] for s in train_data]

# load target images for GRPO reward
grpo_targets = []
for sample in train_data:
    with open(sample["image_path"], "rb") as f:
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
    attn_implementation="flash_attention_2",
    torch_dtype="auto",
    device_map="auto"
)
model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)
model = torch.compile(model)

# ============================================================
# Phase 1: SFT
# ============================================================
print("=" * 60)
print("Phase 1: Supervised Fine-Tuning")
print("=" * 60)

model = run_sft(model, tokenizer, sft_prompts, sft_code_gt, training_steps=500, lr=1e-4, device=device)
model.save_pretrained("./checkpoints/sft_final")
print("SFT complete.\n")

# ============================================================
# Phase 2: GRPO
# ============================================================
print("=" * 60)
print("Phase 2: GRPO Reinforcement Learning")
print("=" * 60)

model = run_grpo(model, tokenizer, grpo_prompts, grpo_targets, model_name=model_name,
                 training_steps=1000, lr=1e-5, device=device)
model.save_pretrained("./checkpoints/grpo_final")
print("GRPO complete. Final model saved.")
