import torch
import torch.nn.functional as F
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformers import get_cosine_schedule_with_warmup
from einops import rearrange


MODEL_NAME = "zai-org/GLM-4.1V-9B-Thinking"

SYSTEM_PROMPT = (
    "You are a high-fidelity UI reproduction expert. Given a screenshot of a UI widget, "
    "generate a React functional component with Tailwind CSS that visually matches it as closely as possible.\n\n"
    "## Rules\n"
    "- Output a single default-exported React component: `export default function Widget() { ... }`\n"
    "- Use Tailwind CSS utility classes for all styling\n"
    "- Colors: use exact hex values via arbitrary-value syntax (e.g. `bg-[#3B82F6]`, `text-[#1F2937]`)\n"
    "- Match layout exactly: flex/grid direction, alignment, spacing, gaps, padding, margins\n"
    "- Match typography: text size, font weight, tracking, leading\n"
    "- Match border-radius, sno thadow, opacity, gradients\n"
    "- Charts/gauges/progress: use recharts (import from 'recharts')\n"
    "- Icons: use lucide-react (import from 'lucide-react')\n"
    "- The root container must match the widget dimensions given in the user message\n"
    "- Text content must be character-perfect — copy every word and number exactly\n\n"
    "## Output format\n"
    "<think>\n"
    "## 1. Structure Analysis\n[widget type, component hierarchy]\n"
    "## 2. Layout Plan\n[flex/grid, dimensions, spacing]\n"
    "## 3. Color & Style\n[exact hex colors, borders, shadows]\n"
    "## 4. Typography\n[font sizes, weights, line-heights]\n"
    "## 5. Implementation Plan\n[component structure outline]\n"
    "</think>\n"
    "<code>[complete React component]</code>"
)


def _get_image_size(path: str) -> tuple[int, int]:
    """Read PNG width/height from IHDR chunk without extra dependencies."""
    import struct
    with open(path, "rb") as f:
        header = f.read(24)
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        return struct.unpack(">II", header[16:24])
    return (800, 600)  # fallback


def _user_message(image_path: str) -> list[dict]:
    w, h = _get_image_size(image_path)
    return [
        {"type": "image", "url": image_path},
        {"type": "text", "text": f"Widget dimensions: {w}x{h}px. Recreate this widget as a React component with Tailwind CSS."},
    ]


def selective_log_softmax(logits, targets):
    log_probs = logits.log_softmax(dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


ACCUM_STEPS = 4


def run_sft(model, processor, samples, training_steps=1000, lr=1e-4, save_dir="/shared/advey",
            plot_path="sft.png", device=torch.device("cuda:0")):

     #   - screenshot_path: path to widget screenshot
     #   - cot: target output (<think>...</think><code>...</code>)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    optimizer_steps = training_steps // ACCUM_STEPS
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=optimizer_steps)
    loss_history = []

    for i in range(training_steps):
        idx = i % len(samples) # loops over all samples indefinitely
        sample = samples[idx]

        # build VLM input: image + text prompt
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": _user_message(sample['screenshot_path'])},
            {"role": "assistant", "content": [{"type": "text", "text": sample["cot"]}]},
        ]

        inputs = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        ).to(device)

        # find where the assistant response starts
        # the target is only the assistant's tokens
        assistant_messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": _user_message(sample['screenshot_path'])},
        ]
        prompt_inputs = processor.apply_chat_template(
            assistant_messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(device)
        prompt_len = prompt_inputs["input_ids"].shape[1]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs).logits[:, prompt_len-1:-1, :]
            target_ids = inputs["input_ids"][:, prompt_len:] # shift by 1, logits --> token ids
            # einops rearrange is goated
            loss = F.cross_entropy(rearrange(logits, "B T D -> (B T) D"), rearrange(target_ids, "B T -> (B T)")) # cross entropy, already does mean and softmax internally
        loss = loss / ACCUM_STEPS
        loss.backward()

        if (i + 1) % ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        loss_history.append(loss.item() * ACCUM_STEPS)  # log unscaled loss

        if (i+1) % 10 == 0:
            print(f"SFT step {i+1}/{training_steps} | loss: {loss.item() * ACCUM_STEPS:.4f} | lr: {scheduler.get_last_lr()[0]:.6f}")

        if (i+1) % 100 == 0:
            _save_plot(loss_history, i+1, training_steps, plot_path)

        if (i+1) % 200 == 0:
            ckpt_dir = f"{save_dir}/checkpoints/sft_step_{i+1}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            print(f"SFT checkpoint saved at step {i+1}")

    _save_plot(loss_history, training_steps, training_steps, plot_path)
    return model


def _save_plot(loss_history, step, total_steps, plot_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(loss_history, alpha=0.3, label="per-step")
    if len(loss_history) > 10:
        window = min(50, len(loss_history) // 5)
        smoothed = [sum(loss_history[max(0,j-window):j+1]) / len(loss_history[max(0,j-window):j+1]) for j in range(len(loss_history))]
        ax.plot(smoothed, label=f"smoothed (window={window})")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(f"SFT Training Loss (step {step}/{total_steps})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {plot_path}")
