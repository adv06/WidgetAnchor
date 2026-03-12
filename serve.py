import torch
from flask import Flask, request, render_template_string
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

app = Flask(__name__)

# ============================================================
# Load model
# ============================================================
model_name = "Qwen/Qwen2.5-7B"
checkpoint_path = "./checkpoints/grpo_final"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(model, checkpoint_path)
model.eval()

# ============================================================
# HTML template
# ============================================================
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Widget Generator</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
        h1 { margin-bottom: 30px; font-size: 2em; }
        .container { width: 100%; max-width: 800px; }
        textarea { width: 100%; height: 80px; padding: 12px; border-radius: 8px; border: 1px solid #333; background: #1a1a1a; color: #e0e0e0; font-size: 16px; resize: vertical; }
        button { margin-top: 12px; padding: 10px 24px; border-radius: 8px; border: none; background: #2563eb; color: white; font-size: 16px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .output { margin-top: 30px; width: 100%; }
        .output-label { font-size: 14px; color: #888; margin-bottom: 8px; }
        .preview { border: 1px solid #333; border-radius: 8px; overflow: hidden; background: white; min-height: 200px; }
        .preview iframe { width: 100%; height: 500px; border: none; }
        .code { margin-top: 20px; }
        .code pre { background: #1a1a1a; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; max-height: 400px; overflow-y: auto; }
    </style>
</head>
<body>
    <h1>Widget Generator</h1>
    <div class="container">
        <form method="POST">
            <textarea name="prompt" placeholder="Describe a widget... e.g. 'a login form with dark mode and neon accents'">{{ prompt or '' }}</textarea>
            <button type="submit">Generate</button>
        </form>
        {% if html_output %}
        <div class="output">
            <div class="output-label">Preview</div>
            <div class="preview">
                <iframe srcdoc="{{ html_output | e }}"></iframe>
            </div>
            <div class="code">
                <div class="output-label">Generated HTML</div>
                <pre>{{ html_output }}</pre>
            </div>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

# ============================================================
# Routes
# ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    html_output = None
    prompt = None

    if request.method == "POST":
        prompt = request.form["prompt"]
        full_prompt = f"Generate a single self-contained HTML file (with inline CSS) for a {prompt} widget. The widget should be centered on the page. Use only HTML and CSS, no JavaScript."

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                do_sample=True,
            )
        prompt_len = inputs["input_ids"].shape[1]
        html_output = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

    return render_template_string(TEMPLATE, html_output=html_output, prompt=prompt)


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
