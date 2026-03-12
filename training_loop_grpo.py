import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from reward import compute_reward_code

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from vllm import LLM, SamplingParams

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
model = torch.compile(model) # what this does is instead of executing operations seperately and storing in memory, it does a lot of operations at once and stores it a all in memory

old_model = copy.deepcopy(model) # store the old for policy updates

for param in old_model.parameters():
    param.requires_grad = False

# vLLM engine for fast generation — uses PagedAttention, continuous batching, optimized CUDA kernels
# runs on a separate GPU (or same GPU in colocate mode)
llm = LLM(model=model_name, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.7, max_tokens=500, n=5)

n = 5
training_steps = 1000
prompts = [] # training steps prompts
ground_truths = []

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps)
grad_accum_steps = 4

for i in range(training_steps):
    completion_scores = []
    generations = []

    vllm_outputs = llm.generate([prompts[i]], sampling_params)
    prompt_len = len(vllm_outputs[0].prompt_token_ids)

    completion_lengths = [] # track actual completion lengths since pad_token == eos_token
    for completion in vllm_outputs[0].outputs:
        text = completion.text
       
        full_ids = vllm_outputs[0].prompt_token_ids + list(completion.token_ids)
        generations.append(torch.tensor(full_ids, device=device))
        completion_lengths.append(len(completion.token_ids))
        completion_scores.append(compute_reward_code(text, ground_truths[i]))
        
    completion_scores = torch.tensor(completion_scores, device=device)
    # if all generations scored the same, std=0 makes advantages explode, zero out advantages instead of skipping
    reward_std = torch.std(completion_scores)
    if reward_std < 1e-6:
        completion_reg = torch.zeros_like(completion_scores)
    else:
        completion_reg = (completion_scores-completion_scores.mean())/(reward_std + 1e-8)
    
    beta = 0.05
    
    tokens = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True, padding_value=tokenizer.pad_token_id) # batch generations
    
    attention_mask = (tokens != PAD_TOKEN_ID).long() # 1 for real tokens, 0 for fake padding (my homies hate padding)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(tokens, attention_mask=attention_mask)
        with torch.no_grad():
            model.disable_adapter_layers()
            outputs_ref = model(tokens, attention_mask=attention_mask)
            model.enable_adapter_layers()
            outputs_old = old_model(tokens, attention_mask=attention_mask)
            
        
        # build mask from actual completion lengths (not EOS search, since pad_token == eos_token)
        completion_ids = tokens[:, prompt_len:] # completion_lengths gives the real length so we chillin no need to calc eos
        comp_lens = torch.tensor(completion_lengths, device=device) # (n,)
        seq_indices = torch.arange(completion_ids.size(1), device=device).unsqueeze(0) # (1, max_comp_len), crazy broadcast look at next line
        mask = (seq_indices < comp_lens.unsqueeze(1)).float() # (n, max_comp_len), 1 for real tokens, 0 for padding
        
        adv = completion_reg.detach().unsqueeze(1) # should not carry gradients, (n, 1) — broadcasts with (n, seq_len)
        
        logits = outputs.logits # batch, seq, tokens
        logits_ref = outputs_ref.logits
        logits_old = outputs_old.logits
        
        logits = logits[:, :-1] 
        logits_ref = logits_ref[:, :-1] 
        logits_old = logits_old[:, :-1] 
        
        targets = tokens[:, 1:] # batch, token align the logits and targets, remember that logits has an extra dimension
        
        token_log_probs = selective_log_softmax(logits, targets)
        token_log_probs_ref = selective_log_softmax(logits_ref, targets)
        token_log_probs_old = selective_log_softmax(logits_old, targets)

        completion_log_probs = token_log_probs[:, prompt_len-1: ] # get rid of prompt batch, tokens
        completion_log_probs_ref = token_log_probs_ref[:, prompt_len-1:] 
        completion_log_probs_old = token_log_probs_old[:, prompt_len-1:]
        
        log_ratio = (completion_log_probs_ref - completion_log_probs) # KL divergence
        KL = torch.exp(log_ratio) - log_ratio - 1  # schulman approximation, always non negative as opposed to the log_ratio
        eps = 0.2
        
        ratio = torch.exp(completion_log_probs - completion_log_probs_old)
        clipped = torch.clamp(ratio, 1-eps, 1+eps)
        per_token_loss = -torch.min(adv * ratio,  adv * clipped) + beta * KL
        loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean()   # average per token across sequences
        
    loss = loss / grad_accum_steps # do the math bro --> need to scale it down --> (a+b+c)/n + (d+e+f)/n = (a+b+c+d+e+f)/n but we want (a+b+c+d+e+f)/(2*n) look at the loss --> grad accumulation step
    loss.backward()
    
    if (i+1) % grad_accum_steps == 0:                                                                                                                                                                                                                                                                                               
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)                                                                                                                                                                                                                                                      
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        old_model.load_state_dict(model.state_dict()) # snapshot AFTER update, as a copy not a reference
        # sync updated weights to vLLM — merge LoRA into base weights temporarily
        # merge_and_unload() is destructive (can't train after), so use merge/unmerge instead
        model.merge_adapter()  # folds lora_A @ lora_B into base weights
        llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(model.state_dict()) # merge LoRA into base 
        model.unmerge_adapter()  # restores base weights + separate LoRA for continued training

    if (i+1) % 10 == 0:
        print(f"step {i+1} | loss: {loss.item():.4f} | mean_reward: {completion_scores.mean().item():.4f} | KL: {KL.mean().item():.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

    if (i+1) % 200 == 0:
        model.save_pretrained(f"./grpo_checkpoint_step_{i+1}")
        print(f"checkpoint saved at step {i+1}")
