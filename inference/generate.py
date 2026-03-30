"""
Phase 3: Single-shot generation with GLM-4.1V VLM.

Usage:
    python -m inference.generate --checkpoint /shared/advey/checkpoints/grpo_final --image widget.png
"""
import re
import torch
import argparse
from transformers import AutoProcessor, Glm4vForConditionalGeneration
from peft import PeftModel
from training.sft import SYSTEM_PROMPT, MODEL_NAME, _user_message


def load_model(checkpoint_path: str, model_name: str = MODEL_NAME, device: str = "cuda:0"):
    processor = AutoProcessor.from_pretrained(model_name, use_fast=True)
    base_model = Glm4vForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": device}
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    return model, processor


def generate(model, processor, image_path: str, temperature: float = 0.7, max_new_tokens: int = 8192) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": _user_message(image_path)},
    ]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=True,
        )
    prompt_len = inputs["input_ids"].shape[1]
    text = processor.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
    return text


def _unwrap_chat_template(text: str) -> str:
    """Extract raw text from chat template dict format if present."""
    # model sometimes outputs the chat template wrapper: [{'type': 'text', 'text': '...'}]
    # with escaped newlines (\\n) and quotes (\')
    if text.lstrip().startswith("[{") or text.lstrip().startswith("{'"):
        match = re.search(r"['\"]text['\"]:\s*['\"](.+)", text, re.DOTALL)
        if match:
            inner = match.group(1)
            # strip trailing wrapper
            inner = re.sub(r"['\"]?\s*\}?\]?\s*$", "", inner)
            # unescape
            inner = inner.replace("\\'", "'").replace('\\"', '"').replace("\\n", "\n").replace("\\t", "\t")
            return inner
    return text


def extract_code(text: str) -> str | None:
    text = _unwrap_chat_template(text)
    # try standard tags first
    match = re.search(r"<code>(.*?)</code>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # fallback: unclosed <code> tag (model hit token limit before closing)
    match = re.search(r"<code>(.*)", text, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if "export default" in code or "import" in code:
            return code
    # fallback: try markdown code fences
    match = re.search(r"```(?:tsx|jsx|javascript|react)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True, help="Path to widget screenshot")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, processor = load_model(args.checkpoint)
    text = generate(model, processor, args.image, temperature=args.temperature)

    tsx = extract_code(text)
    if tsx:
        print(tsx)
    else:
        print("No <code> block found in output:")
        print(text)
