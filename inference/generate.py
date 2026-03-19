"""
Phase 3: Single-shot generation.

Generate HTML from a widget screenshot using the trained model.

Usage:
    python -m inference.generate --checkpoint /shared/advey/checkpoints/grpo_final --image widget.png
"""
import re
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from training.sft import SYSTEM_PROMPT


def load_model(model_name: str, checkpoint_path: str, device: str = "cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map={"": device}
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt: str, temperature: float = 0.7, max_new_tokens: int = 2048) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
    prompt_len = inputs["input_ids"].shape[1]
    text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return text


def extract_code(text: str) -> str | None:
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    return match.group(1).strip() if match else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Recreate the widget shown in the reference image.")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_name, args.checkpoint)
    full_prompt = SYSTEM_PROMPT + "\n" + args.prompt
    text = generate(model, tokenizer, full_prompt, temperature=args.temperature)

    html = extract_code(text)
    if html:
        print(html)
    else:
        print("No <code> block found in output:")
        print(text)
