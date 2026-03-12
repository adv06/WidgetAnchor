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


def run_grpo(model, tokenizer, prompts, ground_truths, model_name="Qwen/Qwen2.5-7B",
             training_steps=1000, lr=1e-5, n=5, grad_accum_steps=4, beta=0.05, eps=0.2,
             device=torch.device("cuda:0")):

    PAD_TOKEN_ID = tokenizer.pad_token_id

    old_model = copy.deepcopy(model) # store the old for policy updates

    for param in old_model.parameters():
        param.requires_grad = False

    # vLLM engine for fast generation — uses PagedAttention, continuous batching, optimized CUDA kernels
    # runs on a separate GPU (or same GPU in colocate mode)
    llm = LLM(model=model_name, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=500, n=n)

    # sync current weights to vLLM
    model.merge_adapter()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(model.state_dict())
    model.unmerge_adapter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps)

    for i in range(training_steps):
        completion_scores = []
        generations = []

        # vLLM generation, much faster than model.generate()
        idx = i % len(prompts)
        vllm_outputs = llm.generate([prompts[idx]], sampling_params)
        prompt_len = len(vllm_outputs[0].prompt_token_ids)

        completion_lengths = [] # track actual completion lengths since pad_token == eos_token
        for completion in vllm_outputs[0].outputs:
            text = completion.text
            # reconstruct full token sequence (prompt + completion) for HF model forward pass
            full_ids = vllm_outputs[0].prompt_token_ids + list(completion.token_ids)
            generations.append(torch.tensor(full_ids, device=device))
            completion_lengths.append(len(completion.token_ids))
            completion_scores.append(compute_reward_code(ground_truths[idx], text)) # ground_truths[idx] is target image bytes, text is generated HTML

        completion_scores = torch.tensor(completion_scores, device=device)
        # if all generations scored the same, std=0 makes advantages explode, zero out advantages instead of skipping
        reward_std = torch.std(completion_scores)
        if reward_std < 1e-6:
            completion_reg = torch.zeros_like(completion_scores)
        else:
            completion_reg = (completion_scores - completion_scores.mean()) / (reward_std + 1e-8)

        tokens = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True, padding_value=PAD_TOKEN_ID) # batch generations

        attention_mask = (tokens != PAD_TOKEN_ID).long() # 1 for real tokens, 0 for fake padding (my homies hate padding)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(tokens, attention_mask=attention_mask)
            with torch.no_grad():
                model.disable_adapter_layers()
                outputs_ref = model(tokens, attention_mask=attention_mask) # lora is only added in the forward pass so we can disable and use it as the frozen model
                model.enable_adapter_layers()
                outputs_old = old_model(tokens, attention_mask=attention_mask)

            # build mask from actual completion lengths (not EOS search, since pad_token == eos_token)
            completion_ids = tokens[:, prompt_len:] # completion_lengths gives the real length so we chillin no need to calc eos
            comp_lens = torch.tensor(completion_lengths, device=device) # (n,)
            seq_indices = torch.arange(completion_ids.size(1), device=device).unsqueeze(0) # (1, max_comp_len), crazy broadcast look at next line
            mask = (seq_indices < comp_lens.unsqueeze(1)).float() # (n, max_comp_len), 1 for real tokens, 0 for padding
            # tokens *= mask why is this wrong? Exercise for the reader

            adv = completion_reg.detach().unsqueeze(1) # should not carry gradients, (n, 1) — broadcasts with (n, seq_len)

            logits = outputs.logits[:, :-1] # batch, seq, tokens
            logits_ref = outputs_ref.logits[:, :-1]
            logits_old = outputs_old.logits[:, :-1]
            targets = tokens[:, 1:] # batch, token align the logits and targets, remember that logits has an extra dimension

            token_log_probs = selective_log_softmax(logits, targets)
            token_log_probs_ref = selective_log_softmax(logits_ref, targets)
            token_log_probs_old = selective_log_softmax(logits_old, targets)

            completion_log_probs = token_log_probs[:, prompt_len-1:] # get rid of prompt batch, tokens
            completion_log_probs_ref = token_log_probs_ref[:, prompt_len-1:]
            completion_log_probs_old = token_log_probs_old[:, prompt_len-1:]

            log_ratio = completion_log_probs_ref - completion_log_probs # KL divergence
            KL = torch.exp(log_ratio) - log_ratio - 1 # schulman approximation, always non negative as opposed to the log_ratio

            ratio = torch.exp(completion_log_probs - completion_log_probs_old)
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            per_token_loss = -torch.min(adv * ratio, adv * clipped) + beta * KL
            loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean() # average per token across sequences

        loss = loss / grad_accum_steps # need to scale it down --> (a+b+c)/n + (d+e+f)/n = (a+b+c+d+e+f)/n but we want (a+b+c+d+e+f)/(2*n) look at the loss --> grad accumulation step
        loss.backward()

        if (i+1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            old_model.load_state_dict(model.state_dict()) # snapshot AFTER update, as a copy not a reference
            # sync updated weights to vLLM — merge LoRA into base weights temporarily
            # merge_and_unload() is destructive (can't train after), so use merge/unmerge instead
            model.merge_adapter() # folds lora_A @ lora_B into base weights
            llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(model.state_dict()) # merge LoRA into base
            model.unmerge_adapter() # restores base weights + separate LoRA for continued training

        if (i+1) % 10 == 0:
            print(f"GRPO step {i+1}/{training_steps} | loss: {loss.item():.4f} | mean_reward: {completion_scores.mean().item():.4f} | KL: {KL.mean().item():.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        if (i+1) % 200 == 0:
            model.save_pretrained(f"./checkpoints/grpo_step_{i+1}")
            print(f"GRPO checkpoint saved at step {i+1}")

    return model
