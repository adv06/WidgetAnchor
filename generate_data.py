import json
import os
import random
from dotenv import load_dotenv
from openai import OpenAI
from reward import render_html_to_image

load_dotenv()

client = OpenAI()

WIDGET_TYPES = [
    "login form", "signup form", "pricing card", "navigation bar", "footer",
    "search bar", "user profile card", "notification banner", "toggle switch",
    "dropdown menu", "modal dialog", "progress bar", "file upload button",
    "rating stars widget", "shopping cart summary", "weather widget",
    "music player controls", "chat bubble", "calendar date picker",
    "cookie consent banner", "testimonial card", "FAQ accordion",
    "social media share buttons", "breadcrumb navigation", "pagination controls",
    "tooltip", "tab navigation", "sidebar menu", "dashboard stat card",
    "email subscription form", "countdown timer", "stepper/wizard progress",
    "avatar with status indicator", "tag/chip input", "color picker",
    "image carousel placeholder", "credit card input form", "settings toggle list",
    "notification bell dropdown", "kanban column",
]

STYLE_VARIANTS = [
    "minimal and clean with lots of whitespace",
    "dark mode with neon accents",
    "glassmorphism with frosted glass effect",
    "neumorphism with soft shadows",
    "bold and colorful with gradients",
    "flat design with bright primary colors",
    "retro/vintage style",
    "professional corporate style with blue tones",
    "playful with rounded corners and pastel colors",
    "brutalist with sharp edges and monospace fonts",
]


def generate_prompt(widget_type, style):
    return f"Generate a single self-contained HTML file (with inline CSS) for a {widget_type} widget. Style: {style}. The widget should be centered on the page. Use only HTML and CSS, no JavaScript."


def generate_html_from_prompt(prompt):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert frontend developer. Return ONLY valid HTML code, no markdown, no explanation, no code fences."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
    )
    html = response.choices[0].message.content.strip()
    # strip markdown code fences if present
    if html.startswith("```"):
        html = html.split("\n", 1)[1]
    if html.endswith("```"):
        html = html.rsplit("```", 1)[0]
    return html.strip()


def generate_dataset(num_samples=200, output_dir="./data"):
    os.makedirs(f"{output_dir}/images", exist_ok=True)

    dataset = []
    for i in range(num_samples):
        widget_type = random.choice(WIDGET_TYPES)
        style = random.choice(STYLE_VARIANTS)
        prompt = generate_prompt(widget_type, style)

        print(f"[{i+1}/{num_samples}] Generating: {widget_type} ({style})")

        try:
            html_code = generate_html_from_prompt(prompt)
            image_bytes = render_html_to_image(html_code)

            image_path = f"{output_dir}/images/widget_{i:04d}.png"
            with open(image_path, "wb") as f:
                f.write(image_bytes)

            dataset.append({
                "id": i,
                "prompt": prompt,
                "html": html_code,
                "image_path": image_path,
                "widget_type": widget_type,
                "style": style,
            })
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # train/val split (80/20)
    random.shuffle(dataset)
    split_idx = int(len(dataset) * 0.8)
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]

    with open(f"{output_dir}/train.json", "w") as f:
        json.dump(train_data, f, indent=2)
    with open(f"{output_dir}/val.json", "w") as f:
        json.dump(val_data, f, indent=2)

    print(f"\nDone! {len(train_data)} train, {len(val_data)} val samples saved to {output_dir}/")
    return train_data, val_data


if __name__ == "__main__":
    generate_dataset(num_samples=200)
