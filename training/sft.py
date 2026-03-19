import torch
import torch.nn.functional as F
import os
from transformers import get_cosine_schedule_with_warmup
from einops import rearrange


SYSTEM_PROMPT = (
    "You are a UI widget-to-code expert. Given a description of a UI widget, "
    "analyze its structure, layout, colors, and typography, then generate self-contained HTML/CSS code.\n"
    "Output format:\n"
    "<think>[structured reasoning]</think>\n"
    "<code>[complete HTML/CSS]</code>"
)


def selective_log_softmax(logits, targets):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def run_sft(model, tokenizer, prompts, targets, training_steps=1000, lr=1e-4, save_dir="/shared/advey", device=torch.device("cuda:0")):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps)

    for i in range(training_steps):
        idx = i % len(prompts)
        full_text = prompts[idx] + targets[idx] + tokenizer.eos_token
        tokenized = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=32768).to(device)
        prompt_len = len(tokenizer.encode(prompts[idx]))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**tokenized).logits[:, prompt_len-1:-1, :]
            target_ids = tokenized["input_ids"][:, prompt_len:]
            loss = F.cross_entropy(rearrange(logits, "B T D -> (B T) D"), rearrange(target_ids, "B T -> (B T)")) # does softmax automatically I think 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if (i+1) % 10 == 0:
            print(f"SFT step {i+1}/{training_steps} | loss: {loss.item():.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        if (i+1) % 200 == 0:
            ckpt_dir = f"{save_dir}/checkpoints/sft_step_{i+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir) 
            print(f"SFT checkpoint saved at step {i+1}")

    return model
