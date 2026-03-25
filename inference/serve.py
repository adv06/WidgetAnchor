import os
import torch
import base64
from flask import Flask, request, render_template_string
from inference.generate import load_model, generate, extract_code
from reward.programmatic import render_tsx_to_image
from training.sft import MODEL_NAME

app = Flask(__name__)

checkpoint_path = os.environ.get("CHECKPOINT", "/shared/advey/checkpoints/grpo_final")
device = os.environ.get("DEVICE", "cuda:0")

model, processor = load_model(checkpoint_path, model_name=MODEL_NAME, device=device)

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Widget2Code — GLM-4.1V</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, sans-serif; background: #0f0f0f; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
        h1 { margin-bottom: 8px; font-size: 2em; }
        .subtitle { color: #888; margin-bottom: 30px; font-size: 14px; }
        .container { width: 100%; max-width: 900px; }
        .upload-area { border: 2px dashed #333; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: border-color 0.2s; }
        .upload-area:hover { border-color: #2563eb; }
        .upload-area input { display: none; }
        button { margin-top: 12px; padding: 10px 24px; border-radius: 8px; border: none; background: #2563eb; color: white; font-size: 16px; cursor: pointer; }
        button:hover { background: #1d4ed8; }
        .results { margin-top: 30px; display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .panel { background: #1a1a1a; border-radius: 8px; overflow: hidden; }
        .panel-header { padding: 12px 16px; border-bottom: 1px solid #333; font-size: 13px; color: #888; }
        .panel img { width: 100%; display: block; }
        .code-panel { margin-top: 20px; }
        .code-panel pre { background: #1a1a1a; padding: 16px; border-radius: 8px; overflow-x: auto; font-size: 13px; max-height: 400px; overflow-y: auto; white-space: pre-wrap; }
    </style>
</head>
<body>
    <h1>Widget2Code</h1>
    <p class="subtitle">Upload a widget screenshot → GLM-4.1V generates React+Tailwind TSX</p>
    <div class="container">
        <form method="POST" enctype="multipart/form-data">
            <div class="upload-area" onclick="this.querySelector('input').click()">
                <input type="file" name="image" accept="image/*" onchange="this.form.submit()" />
                <p>Click or drag to upload a widget screenshot</p>
            </div>
        </form>
        {% if source_b64 %}
        <div class="results">
            <div class="panel">
                <div class="panel-header">Source Screenshot</div>
                <img src="data:image/png;base64,{{ source_b64 }}" />
            </div>
            {% if render_b64 %}
            <div class="panel">
                <div class="panel-header">Generated Render</div>
                <img src="data:image/png;base64,{{ render_b64 }}" />
            </div>
            {% else %}
            <div class="panel">
                <div class="panel-header">Render Failed</div>
                <p style="padding: 16px; color: #f87171;">{{ render_error }}</p>
            </div>
            {% endif %}
        </div>
        {% if tsx %}
        <div class="code-panel">
            <div class="panel-header">Generated TSX</div>
            <pre>{{ tsx }}</pre>
        </div>
        {% endif %}
        {% endif %}
    </div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    source_b64 = None
    render_b64 = None
    render_error = None
    tsx = None

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            file.save(tmp.name)

            with open(tmp.name, "rb") as f:
                source_b64 = base64.b64encode(f.read()).decode()

            text = generate(model, processor, tmp.name)
            tsx = extract_code(text)

            if tsx:
                try:
                    rendered = render_tsx_to_image(tsx)
                    render_b64 = base64.b64encode(rendered).decode()
                except Exception as e:
                    render_error = str(e)

            os.unlink(tmp.name)

    return render_template_string(TEMPLATE, source_b64=source_b64, render_b64=render_b64,
                                  render_error=render_error, tsx=tsx)


if __name__ == "__main__":
    print("Starting server at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
