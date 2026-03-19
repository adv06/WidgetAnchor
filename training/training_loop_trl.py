from trl import GRPOConfig, GRPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from reward.programmatic import compute_reward_code


model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

def reward_fn(completions, ground_truth, **kwargs):
    rewards = []
    for completion, gt in zip(completions, ground_truth):
        # extract the assistant's text from the conversation format
        text = completion[0]["content"] if isinstance(completion, list) else completion
        rewards.append(compute_reward_code(text, gt))
    return rewards

training_args = GRPOConfig(
    output_dir="./grpo_output",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    weight_decay=0.01,
    max_completion_length=500,
    num_generations=5,
    beta=0.05,
    epsilon=0.2,
    loss_type="grpo",
    bf16=True,
    gradient_checkpointing=True,
    logging_steps=10,
    save_steps=100,
    max_grad_norm=1.0,
    report_to="wandb",
)

# dataset must have a "prompt" column (and optionally "ground_truth" for our reward fn)
# example format:
# dataset = Dataset.from_dict({
#     "prompt": ["Generate an HTML button with blue background", ...],
#     "ground_truth": ["<button style='background:blue'>Click</button>", ...]
# })
dataset = None  # replace with your dataset

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_fn,
    peft_config=lora_config,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("./grpo_final")
