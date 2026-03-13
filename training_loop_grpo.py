import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from reward import compute_reward_code

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from vllm import LLM, SamplingParams

def selective_log_softmax(logits, targets):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def run_grpo(model, tokenizer, prompts, ground_truths, model_name="Qwen/Qwen2.5-1.5B",
             training_steps=1000, lr=1e-5, n=5, batch_size=4, grad_accum_steps=4, beta=0.05, eps=0.2,
             save_dir="/shared/advey", device=torch.device("cuda:0")):

    PAD_TOKEN_ID = tokenizer.pad_token_id

    # vLLM engine for fast generation — uses PagedAttention, continuous batching, optimized CUDA kernels
    # runs on a separate GPU (or same GPU in colocate mode)
    llm = LLM(model=model_name, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=500, n=n, logprobs=1)
    # sync current weights to vLLM
    model.merge_adapter()
    llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(model.state_dict())
    model.unmerge_adapter()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps)
    history = {"loss": [], "reward": [], "kl": [], "clip_frac": []}

    for i in range(training_steps):
        completion_scores = []
        raw_scores = []
        generations = []

        # vLLM generation, much faster than model.generate()
        idxs = [(i*batch_size+b) % len(prompts) for b in range(batch_size)] # indexes in the batch
        batch_prompts = [prompts[k] for k in idxs]
        vllm_outputs = llm.generate(batch_prompts, sampling_params)
        prompt_lens = [len(vllm_outputs[b].prompt_token_ids) for b in range(batch_size)]
        min_prompt_len = min(prompt_lens)
        old_probs = []
        completion_lengths = [] # track actual completion lengths since pad_token == eos_token

        for b in range(batch_size):
            completion_interim = []
            for completion in vllm_outputs[b].outputs:
                text = completion.text
                # reconstruct full token sequence (prompt + completion) for HF model forward pass
                full_ids = vllm_outputs[b].prompt_token_ids + list(completion.token_ids)
                completion_lengths.append(len(completion.token_ids))
                score = compute_reward_code(ground_truths[idxs[b]], text) # ground_truths[idx] is target image bytes, text is generated HTML
                raw_scores.append(score)
                completion_interim.append(score)
                # extract old logprobs from vLLM — already completion-only
                prob_distr_old = torch.tensor([completion.logprobs[j][token_id].logprob for j, token_id in enumerate(completion.token_ids)], device=device)
                old_probs.append(prob_distr_old)
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

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(tokens, attention_mask=attention_mask)
            with torch.no_grad():
                model.disable_adapter_layers()
                outputs_ref = model(tokens, attention_mask=attention_mask) # lora is only added in the forward pass so we can disable and use it as the frozen model
                model.enable_adapter_layers()

            adv = torch.tensor(completion_scores, device=device).detach().unsqueeze(1) # (B*n, 1) — broadcasts with (B*n, seq_len)

            logits = outputs.logits[:, :-1] # batch, seq, tokens
            logits_ref = outputs_ref.logits[:, :-1]
            targets = tokens[:, 1:] # batch, token align the logits and targets, remember that logits has an extra dimension

            token_log_probs = selective_log_softmax(logits, targets)
            token_log_probs_ref = selective_log_softmax(logits_ref, targets)

            completion_log_probs = token_log_probs[:, min_prompt_len-1:] # get rid of prompt
            completion_log_probs_ref = token_log_probs_ref[:, min_prompt_len-1:]

            # align shapes: old_probs is completion-only, completion_log_probs may be longer due to variable prompt lengths
            seq_len = min(completion_log_probs.size(1), old_probs.size(1))
            completion_log_probs = completion_log_probs[:, :seq_len]
            completion_log_probs_ref = completion_log_probs_ref[:, :seq_len]
            old_probs = old_probs[:, :seq_len]

            # build mask from actual completion lengths (not EOS search, since pad_token == eos_token)
            comp_lens = torch.tensor(completion_lengths, device=device) # (B*n,)
            seq_indices = torch.arange(seq_len, device=device).unsqueeze(0) # (1, seq_len), crazy broadcast look at next line
            mask = (seq_indices < comp_lens.unsqueeze(1)).float() # (B*n, seq_len), 1 for real tokens, 0 for padding
            # tokens *= mask why is this wrong? Exercise for the reader

            log_ratio = completion_log_probs_ref - completion_log_probs # KL divergence
            KL = torch.exp(log_ratio) - log_ratio - 1 # schulman approximation, always non negative as opposed to the log_ratio

            ratio = torch.exp(completion_log_probs - old_probs)
            clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
            # clip fraction: how often the ratio was clipped — if too high, policy is changing too fast
            clip_frac = ((ratio - 1).abs() > eps).float().mean().item()
            per_token_loss = -torch.min(adv * ratio, adv * clipped) + beta * KL
            loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean() # average per token across sequences

        loss = loss / grad_accum_steps # need to scale it down --> (a+b+c)/n + (d+e+f)/n = (a+b+c+d+e+f)/n but we want (a+b+c+d+e+f)/(2*n) look at the loss --> grad accumulation step
        loss.backward()

        # track metrics
        history["loss"].append(loss.item() * grad_accum_steps) # unscaled loss
        history["reward"].append(torch.tensor(raw_scores).mean().item())
        history["kl"].append(KL.mean().item())
        history["clip_frac"].append(clip_frac)

        if (i+1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            # sync updated weights to vLLM — merge LoRA into base weights temporarily
            # merge_and_unload() is destructive (can't train after), so use merge/unmerge instead
            model.merge_adapter() # folds lora_A @ lora_B into base weights
            llm.llm_engine.model_executor.driver_worker.model_runner.model.load_weights(model.state_dict()) # merge LoRA into base
            model.unmerge_adapter() # restores base weights + separate LoRA for continued training

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
