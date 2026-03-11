import torch 
import torch.nn.functional as F 
import torch.nn as nn
import copy
from reward import compute_reward_code

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


model_name = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)


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
    torch_dtype="auto",
    device_map="auto"
)
frozen = copy.deepcopy(model) # for KL later

model.gradient_checkpointing_enable()
model = get_peft_model(model, lora_config)

old_model = copy.deepcopy(model) # store the old for policy updates
for param in frozen.parameters():
    param.requires_grad = False # no grad update 

for param in old_model.parameters():
    param.requires_grad = False

n = 5
training_steps = 1000
prompts = [] # training steps prompts
ground_truths = []

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

for i in range(training_steps):
    completion_scores = []
    inputs = tokenizer(prompts[i], return_tensors="pt").to(model.device)
    generations = []
    for j in range(n):
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True,
                return_dict_in_generate=True 
            )
        
        prompt_len = inputs["input_ids"].shape[1] 
        tokens = output.sequences[0]
         
        text = tokenizer.decode(tokens[prompt_len:], skip_special_tokens=True)
        generations.append(tokens)
        completion_scores.append(compute_reward_code(text, ground_truths[i]))
        
    completion_scores = torch.tensor(completion_scores, device=model.device)
    completion_reg = (completion_scores-completion_scores.mean())/(torch.std(completion_scores) + 1e-8)
    prompt_len = inputs["input_ids"].shape[1] # number of tokens in the prompt
    
    losses = []
    beta = 0.05
    for tokens, adv in zip(generations, completion_reg):
        tokens = tokens.unsqueeze(0) # expects a batch input
        
        completion_ids = tokens[:, prompt_len: ]
        eos_token_id = tokenizer.eos_token_id # eos token id
        is_eos = (completion_ids == eos_token_id) # boolean grid of true and false
        
#         No, it gives a boolean tensor the same shape as completion_ids — True at every position where the token is EOS, False everywhere else.
                                                                                                                                                                                                                                                                                                
#         # Example: completion_ids = [42, 15, EOS, 7, EOS]                                                                                                                                                                                                                                               
#         # is_eos            =       [F,  F,  T,  F,  T ]   

        eos_idx = torch.full((completion_ids.size(0),), completion_ids.size(1), device=tokens.device) # (batch,) fill each number completion_ids.size(1)
        
        # stop at the first eof, update the size
        if is_eos.any(): # check eos anywhere in the matrix
            eos_idx[is_eos.any(dim=1)] = is_eos.argmax(dim=1)[is_eos.any(dim=1)] # think about this one, its tricky, lots of parallelize
        
        mask = torch.arange(completion_ids.size(1), device=tokens.device).unsqueeze(0)
        mask = (mask <= eos_idx.unsqueeze(1)).float() # get the stop column for each row, mask is broadcasted shape: (batch, seq_len)
        # tokens *= mask why is this wrong? Exercise for the reader
        
        adv = adv.detach() # should not carry gradients
        outputs = model(tokens)
        with torch.no_grad():
            outputs_ref = frozen(tokens)
            outputs_old = old_model(tokens)
        
        logits = outputs.logits # batch, seq, tokens
        logits_ref = outputs_ref.logits
        logits_old = outputs_old.logits
        
        logits = logits[:, :-1] 
        logits_ref = logits_ref[:, :-1] 
        logits_old = logits_old[:, :-1] 
        
        targets = tokens[:, 1:] # batch, token align the logits and targets, remember that logits has an extra dimension
        log_probs = F.log_softmax(logits, dim=-1)    
        log_probs_ref = F.log_softmax(logits_ref, dim=-1)
        log_probs_old = F.log_softmax(logits_old, dim = -1)
        
        token_log_probs = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_log_probs_ref = log_probs_ref.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        token_log_probs_old = log_probs_old.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    
        completion_log_probs = token_log_probs[:, prompt_len-1: ] # get rid of prompt batch, tokens
        completion_log_probs_ref = token_log_probs_ref[:, prompt_len-1:] 
        completion_log_probs_old = token_log_probs_old[:, prompt_len-1:]
        
        log_ratio = (completion_log_probs_ref - completion_log_probs) # KL divergence
        KL = torch.exp(log_ratio) - log_ratio - 1  # schulman approximation, always non negative as opposed to the log_ration
        eps = 0.2
        
        ratio = torch.exp(completion_log_probs - completion_log_probs_old)
        clipped = torch.clamp(ratio, 1-eps, 1+eps)
        per_token_loss = -torch.min(adv * ratio,  adv * clipped) + beta * KL
        loss = (per_token_loss*mask).sum() / mask.sum() # average over real tokens
        
        losses.append(loss)
    
    loss = torch.stack(losses).mean() # preserve computation graph
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    old_model.load_state_dict(model.state_dict()) # snapshot AFTER update, as a copy not a reference
