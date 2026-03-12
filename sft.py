import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from einops import rearrange

def selective_log_softmax(logits, targets):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
PAD_TOKEN_ID = tokenizer.pad_token_id
device = torch.device("cuda:0")

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


prompts = []
code_gt = []

training_steps = 1000
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps)

for i in range(training_steps):
    tokenized_gt = tokenizer(prompts[i] + code_gt[i], return_tensors="pt").to(device)
    prompt_len = len(tokenizer.encode(prompts[i]))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(**tokenized_gt).logits[:, prompt_len-1:-1, :]
        targets = tokenized_gt["input_ids"][:, prompt_len:]
        loss = F.cross_entropy(rearrange(logits, "B T D -> (B T) D"), rearrange(targets, "B T -> (B T)"))# cross entropy first, already does mean, only 2 dimensions for target
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
     
    