import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from reward import compute_reward_code
from einops import rearrange
from peft import set_peft_model_state_dict
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup

def selective_log_softmax(logits, targets):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def run_grpo(model, tokenizer, prompts, ground_truths, model_name="Qwen/Qwen2.5-1.5B",
             training_steps=1000, lr=1e-5, n=5, batch_size=4, grad_accum_steps=4, beta=0.05, eps=0.2,
             save_dir="/shared/advey", device=torch.device("cuda:0"), num_epochs=4):

    sft_state = deepcopy(model.peft_config["default"])
    model.add_adapter("reference", sft_state)
    ref_weights = {k: v.clone() for k, v in model.get_adapter_state_dict("default").items()} # get the sft weights
    set_peft_model_state_dict(model, ref_weights, adapter_name="reference") # clone into reference that wont be touched by optimizer
    
    PAD_TOKEN_ID = tokenizer.pad_token_id
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps * num_epochs)
    history = {"loss": [], "reward": [], "kl": [], "clip_frac": []}
    for i in range(training_steps):
        completion_scores = []
        raw_scores = []
        generations = []

        idxs = [(i*batch_size+b) % len(prompts) for b in range(batch_size)] # indexes in the batch
        batch_prompts = [prompts[k] for k in idxs]
    
        old_probs = []
        prompt_lengths = []
        completion_lengths = [] # track actual completion lengths since pad_token == eos_token
        model.eval() # get rid of the dropout noise added by lora
        for b in range(batch_size):
            completion_interim = []
            prompt_ids = tokenizer(batch_prompts[b], return_tensors="pt")["input_ids"].to(device)
            prompt_len = prompt_ids.shape[1]
            prompt_lengths.extend([prompt_len] * n)
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gen_out = model.generate(prompt_ids, output_logits=True, return_dict_in_generate=True, temperature=0.7, do_sample=True, max_new_tokens=1024, pad_token_id=PAD_TOKEN_ID, num_return_sequences=n)
            stacked = torch.stack(gen_out.logits) # initially is a tuple, stack it to make a tensor
            reshaped_logits = rearrange(stacked, "S N V -> N S V") # sequence, n generations, vocab length (softmax)
            for j in range(n):
                # reconstruct full token sequence (prompt + completion) for HF model forward pass
                full_ids = gen_out.sequences[j]
                completion_ids = full_ids[prompt_len: ]
                completion_length = len(completion_ids)
                completion_lengths.append(completion_length)
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                score = compute_reward_code(ground_truths[idxs[b]], text) # ground_truths[idx] is target image bytes, text is generated HTML
                raw_scores.append(score)
                completion_interim.append(score)
                # extract old logprobs from model 
                prob_distr_old = torch.tensor(selective_log_softmax(reshaped_logits[j], completion_ids))
                old_probs.append(prob_distr_old.detach().cpu())
                generations.append(torch.tensor(full_ids, device=device))
            # normalize advantages within this prompt's group
            group = torch.tensor(completion_interim, device=device)
            reward_std = group.std()
            # if all generations scored the same, std=0 makes advantages explode, zero out advantages instead of skipping
            if reward_std < 1e-6:
                group_adv = torch.zeros_like(group)
            else:
                group_adv = (group - group.mean()) / (reward_std + 1e-8)
            completion_scores.extend(group_adv.tolist())
            
        tokens = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True, padding_value=PAD_TOKEN_ID) # batch generations
        old_probs = torch.nn.utils.rnn.pad_sequence(old_probs, batch_first=True, padding_value=0.0) # (B*n, max_comp_len)
        attention_mask = (tokens != PAD_TOKEN_ID).long() # 1 for real tokens, 0 for fake padding
        
        model.train() # train mode
        for epoch in range(num_epochs):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(tokens, attention_mask=attention_mask)
                with torch.no_grad():
                    model.set_adapter("reference")
                    outputs_ref = model(tokens, attention_mask=attention_mask) # lora is only added in the forward pass so we can disable and use it as the frozen model
                    model.set_adapter("default")

                adv = torch.tensor(completion_scores, device=device).detach().unsqueeze(1) # (B*n, 1), broadcasts with (B*n, seq_len)

                logits = outputs.logits[:, :-1] # batch, seq, tokens
                logits_ref = outputs_ref.logits[:, :-1]
                targets = tokens[:, 1:] # batch, token align the logits and targets, remember that logits has an extra dimension

                token_log_probs = selective_log_softmax(logits, targets)
                token_log_probs_ref = selective_log_softmax(logits_ref, targets)

                completion_log_probs = []
                completion_log_probs_ref = []
                
                for s in range(len(generations)) :
                    pl = prompt_lengths[s]
                    completion_log_probs.append(token_log_probs[s, pl-1:]) # get rid of prompt
                    completion_log_probs_ref.append(token_log_probs_ref[s, pl-1:])
                
                completion_log_probs = torch.nn.utils.rnn.pad_sequence(completion_log_probs, batch_first=True, padding_value=0.0)
                completion_log_probs_ref = torch.nn.utils.rnn.pad_sequence(completion_log_probs_ref, batch_first=True, padding_value=0.0)

                # build mask from actual completion lengths (not EOS search, since pad_token == eos_token)
                comp_lens = torch.tensor(completion_lengths, device=device) # (B*n,)
                seq_len = min(completion_log_probs.shape[1], old_probs.shape[1])
                completion_log_probs = completion_log_probs[:, :seq_len]
                completion_log_probs_ref = completion_log_probs_ref[:, :seq_len]
                old_probs_aligned = old_probs[:, :seq_len]
                seq_indices = torch.arange(seq_len, device=device).unsqueeze(0) # broadcast and broadcast in next line
                mask = (seq_indices < comp_lens.unsqueeze(1)).float() # (B*n, seq_len), 1 for real tokens, 0 for padding
                # tokens *= mask why is this wrong? Exercise for the reader

                log_ratio = completion_log_probs_ref - completion_log_probs # KL divergence
                KL = torch.exp(log_ratio) - log_ratio - 1 # schulman approximation, always non negative as opposed to the log_ratio
                
                ratio = torch.exp(completion_log_probs - old_probs_aligned)
                clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
                # clip fraction: how often the ratio was clipped — if too high, policy is changing too fast
                clip_frac = ((ratio - 1).abs() > eps).float().mean().item()
                per_token_loss = -torch.min(adv * ratio, adv * clipped) + beta * KL
                loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean() # average per token across sequences
            # loss = loss / grad_accum_steps # need to scale it down --> (a+b+c)/n + (d+e+f)/n = (a+b+c+d+e+f)/n but we want (a+b+c+d+e+f)/(2*n) look at the loss --> grad accumulation step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # track metrics
        history["loss"].append(loss.item()) # unscaled loss
        history["reward"].append(torch.tensor(raw_scores).mean().item())
        history["kl"].append(KL.mean().item())
        history["clip_frac"].append(clip_frac)

            

        if (i+1) % 10 == 0:
            print(f"GRPO step {i+1}/{training_steps} | loss: {history['loss'][-1]:.4f} | mean_reward: {history['reward'][-1]:.4f} | KL: {history['kl'][-1]:.4f} | clip_frac: {clip_frac:.3f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        if (i+1) % 200 == 0:
            ckpt_dir = f"{save_dir}/checkpoints/grpo_step_{i+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"GRPO checkpoint saved at step {i+1}")

    # save training curves
    plot_dir = f"{save_dir}/plots"
    os.makedirs(plot_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, values) in zip(axes.flat, history.items()):
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("step")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/grpo_training.png", dpi=150)
    plt.close(fig)
    print(f"Training curves saved to {plot_dir}/grpo_training.png")

    return model
