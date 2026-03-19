import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import re
from reward.programmatic import compute_reward_code, render_tsx_to_image
from peft import set_peft_model_state_dict, get_peft_model_state_dict
from copy import deepcopy
from transformers import get_cosine_schedule_with_warmup
from reward.round_robin import round_robin_scoring
from training.sft import SYSTEM_PROMPT, _user_message
from concurrent.futures import ThreadPoolExecutor


def selective_log_softmax(logits, targets):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def _render_candidate(text):
    code_match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if code_match:
        tsx = code_match.group(1)
        try:
            return (render_tsx_to_image(tsx), tsx)
        except Exception:
            return (None, tsx)
    return (None, text)


def run_grpo(model, processor, screenshot_paths, ref_tsx_list=None, model_name="zai-org/GLM-4.1V-9B-Thinking",
             training_steps=1000, lr=1e-5, n=5, batch_size=4, beta=0.05, eps=0.2,
             save_dir="/shared/advey", device=torch.device("cuda:0"), num_epochs=4,
             use_vlm_reward=True):

    sft_state = deepcopy(model.peft_config["default"])
    model.add_adapter("reference", sft_state)
    ref_weights = {k: v.clone() for k, v in get_peft_model_state_dict(model, adapter_name="default").items()} # get the sft weights
    set_peft_model_state_dict(model, ref_weights, adapter_name="reference") # clone into reference that wont be touched by optimizer

    # freeze reference adapter so optimizer doesn't update it
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad = False
    model.set_adapter("default")  # make sure default is active

    tokenizer = processor.tokenizer
    PAD_TOKEN_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=training_steps * num_epochs) # optimizer scheduler
    history = {"loss": [], "reward": [], "kl": [], "clip_frac": []}

    for i in range(training_steps):
        completion_scores = []
        raw_scores = []
        generations = []

        idxs = [(i * batch_size + b) % len(screenshot_paths) for b in range(batch_size)] # indexes in the batch

        prompt_lengths = []
        completion_lengths = [] # track actual completion lengths since pad_token == eos_token
        model.eval() # get rid of the dropout noise added by lora

        for b in range(batch_size):
            # build VLM prompt with reference image + dimensions
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user", "content": _user_message(screenshot_paths[idxs[b]])},
            ]
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(device)
            prompt_len = inputs["input_ids"].shape[1]
            prompt_lengths.extend([prompt_len] * n)

            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                gen_out = model.generate(
                    **inputs, output_logits=True, return_dict_in_generate=True,
                    temperature=0.7, do_sample=True, max_new_tokens=2048,
                    pad_token_id=PAD_TOKEN_ID, num_return_sequences=n
                )

            # extract completions and render in parallel
            texts = []
            for j in range(n):
                # reconstruct full token sequence (prompt + completion) for HF model forward pass
                full_ids = gen_out.sequences[j]
                completion_ids = full_ids[prompt_len:]
                completion_lengths.append(len(completion_ids))
                text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                texts.append(text)
                generations.append(full_ids.clone())

            # parallel Playwright rendering — each candidate rendered concurrently
            with ThreadPoolExecutor(max_workers=n) as pool:
                candidate_images = list(pool.map(_render_candidate, texts))

            # load reference image and tsx for reward
            with open(screenshot_paths[idxs[b]], "rb") as f:
                ref_image = f.read()
            ref_tsx = ref_tsx_list[idxs[b]] if ref_tsx_list is not None else None

            if use_vlm_reward:
                completion_interim = round_robin_scoring(ref_image, candidate_images, ref_tsx=ref_tsx)
            else:
                completion_interim = []
                for rendered_img, tsx in candidate_images:
                    if rendered_img is None:
                        completion_interim.append(-1.0)
                    else:
                        completion_interim.append(compute_reward_code(ref_image, tsx, rendered_image=rendered_img, ref_tsx=ref_tsx))

            raw_scores.extend(completion_interim)
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
        # build attention mask from known lengths (not token identity, avoids PAD==EOS masking real EOS tokens)
        actual_lengths = [prompt_lengths[s] + completion_lengths[s] for s in range(len(generations))]
        attention_mask = torch.zeros_like(tokens)
        for s, length in enumerate(actual_lengths):
            attention_mask[s, :length] = 1

        # compute old_probs from a forward pass, since model.generate is a different path, KV cache rounds bf16, so doing a forward pass is better
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            old_outputs = model(tokens, attention_mask=attention_mask)
            old_logits = old_outputs.logits[:, :-1]
            old_targets = tokens[:, 1:]
            old_token_log_probs = selective_log_softmax(old_logits, old_targets)
            old_probs_list = []
            for s in range(len(generations)):
                pl = prompt_lengths[s]
                old_probs_list.append(old_token_log_probs[s, pl-1:])
            old_probs = torch.nn.utils.rnn.pad_sequence(old_probs_list, batch_first=True, padding_value=0.0) # [batch, max_seq_len], this isnt about correctness its necessary for us to create the tensors of uniform length

        with torch.no_grad():
            model.set_adapter("reference")
            outputs_ref = model(tokens, attention_mask=attention_mask) # frozen SFT adapter for KL reference
            model.set_adapter("default")
                    
        model.train()
        for _ in range(num_epochs):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(tokens, attention_mask=attention_mask)

                adv = torch.tensor(completion_scores, device=device).detach().unsqueeze(1) # (B*n, 1), broadcasts with (B*n, seq_len)

                logits = outputs.logits[:, :-1] # batch, seq, vocab
                logits_ref = outputs_ref.logits[:, :-1]
                targets = tokens[:, 1:] # align logits and targets (logits predict next token)

                token_log_probs = selective_log_softmax(logits, targets)
                token_log_probs_ref = selective_log_softmax(logits_ref, targets)

                completion_log_probs = []
                completion_log_probs_ref = []

                for s in range(len(generations)):
                    pl = prompt_lengths[s]
                    completion_log_probs.append(token_log_probs[s, pl-1:]) # get rid of prompt
                    completion_log_probs_ref.append(token_log_probs_ref[s, pl-1:])

                completion_log_probs = torch.nn.utils.rnn.pad_sequence(completion_log_probs, batch_first=True, padding_value=0.0)
                completion_log_probs_ref = torch.nn.utils.rnn.pad_sequence(completion_log_probs_ref, batch_first=True, padding_value=0.0)

                # align shapes and build mask
                seq_len = min(completion_log_probs.shape[1], old_probs.shape[1])
                completion_log_probs = completion_log_probs[:, :seq_len]
                completion_log_probs_ref = completion_log_probs_ref[:, :seq_len]
                old_probs_aligned = old_probs[:, :seq_len]
                comp_lens = torch.tensor(completion_lengths, device=device)
                seq_indices = torch.arange(seq_len, device=device).unsqueeze(0) # double broadcast this line and next
                mask = (seq_indices < comp_lens.unsqueeze(1)).float() # only takes into account [completion + padding] vs [prompt + completion + padding]
                
                # Just a note, we could have done the following above not vectorized
                # for i, comp in enumerate(completion_lengths):
                #     mask[i, :comp] = 1  

                log_ratio = completion_log_probs_ref - completion_log_probs # KL divergence
                KL = torch.exp(log_ratio) - log_ratio - 1 # schulman approximation, always non negative as opposed to the log_ratio

                ratio = torch.exp(completion_log_probs - old_probs_aligned)
                clipped = torch.clamp(ratio, 1 - eps, 1 + eps)
                # clip fraction: how often the ratio was clipped — if too high, policy is changing too fast
                clip_frac = ((ratio - 1).abs() > eps).float().mean().item()
                per_token_loss = -torch.min(adv * ratio, adv * clipped) + beta * KL
                loss = ((per_token_loss * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean() # average per token across sequences
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
