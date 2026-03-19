import base64
import re
from openai import OpenAI


def compute_vlm_reward(ref_image: bytes, rendered_image: bytes, model: str = "gpt-4o") -> dict:
    """
    Decomposed VLM scoring, asks for 4 separate dimension scores instead of
    one holistic number. This is our improvement over UI2Code^N's single score.
    """
    client = OpenAI()
    img1 = base64.b64encode(ref_image).decode()
    img2 = base64.b64encode(rendered_image).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You will be given two images:\n"
                    "- Image 1: the reference widget (target design)\n"
                    "- Image 2: a code rendering generated from the reference\n\n"
                    "Score EACH dimension independently (0-100):\n"
                    "1. Layout fidelity (positions, sizes, spacing, alignment)\n"
                    "2. Color accuracy (palette match, vibrancy, dark/light mode correctness)\n"
                    "3. Typography fidelity (font sizes, weights, contrast, readability)\n"
                    "4. Overall visual similarity\n\n"
                    "Strictly output in this format:\n"
                    "\\boxed{layout: X, color: Y, typo: Z, overall: W}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img2}"}}
            ]
        }],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    scores = _parse_decomposed_scores(text)

    # weighted combination — color weighted high because style is our weakest area
    scores["total"] = (0.30 * scores["layout"] +
                       0.30 * scores["color"] +
                       0.20 * scores["typo"] +
                       0.20 * scores["overall"])
    return scores


def compute_vlm_comparison(ref_image: bytes, candidate_a: bytes, candidate_b: bytes,
                           model: str = "gpt-4o") -> tuple[dict, dict, str]:
    """
    Pairwise VLM comparison for round-robin ranking.
    Returns (scores_a, scores_b, winner) where winner is "A", "B", or "tie".
    """
    client = OpenAI()
    img_ref = base64.b64encode(ref_image).decode()
    img_a = base64.b64encode(candidate_a).decode()
    img_b = base64.b64encode(candidate_b).decode()

    response = client.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You will be given three images:\n"
                    "- Image 1: the reference widget (target design)\n"
                    "- Image 2: candidate A rendering\n"
                    "- Image 3: candidate B rendering\n\n"
                    "For each candidate, score these dimensions (0-100):\n"
                    "1. Layout fidelity\n"
                    "2. Color accuracy\n"
                    "3. Typography fidelity\n"
                    "4. Overall similarity\n\n"
                    "Then state which candidate is closer to the reference.\n\n"
                    "Output format:\n"
                    "Candidate A: layout=X, color=Y, typo=Z, overall=W\n"
                    "Candidate B: layout=X, color=Y, typo=Z, overall=W\n"
                    "\\boxed{Candidate A is better} or \\boxed{Candidate B is better} or \\boxed{Tie}"
                )},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_ref}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_a}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b}"}}
            ]
        }],
        temperature=0.0,
    )

    text = response.choices[0].message.content.strip()
    scores_a, scores_b, winner = _parse_comparison(text)
    return scores_a, scores_b, winner


def _parse_decomposed_scores(text: str) -> dict:
    """Parse '\\boxed{layout: X, color: Y, typo: Z, overall: W}' into normalized [0,1] scores."""
    defaults = {"layout": 0.0, "color": 0.0, "typo": 0.0, "overall": 0.0}

    # try to find the boxed content
    boxed = re.search(r"\\?boxed\{([^}]+)\}", text)
    if not boxed:
        # fallback: look for key: value patterns anywhere
        source = text
    else:
        source = boxed.group(1)

    for key in defaults:
        match = re.search(rf"{key}\s*[:=]\s*(\d+(?:\.\d+)?)", source, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            defaults[key] = max(0.0, min(1.0, val / 100.0))

    return defaults


def _parse_comparison(text: str) -> tuple[dict, dict, str]:
    """Parse the pairwise comparison response."""
    scores_a = {"layout": 0.0, "color": 0.0, "typo": 0.0, "overall": 0.0}
    scores_b = {"layout": 0.0, "color": 0.0, "typo": 0.0, "overall": 0.0}

    # parse Candidate A scores
    a_match = re.search(r"Candidate A.*?layout\s*=\s*(\d+).*?color\s*=\s*(\d+).*?typo\s*=\s*(\d+).*?overall\s*=\s*(\d+)", text, re.IGNORECASE | re.DOTALL)
    if a_match:
        scores_a = {
            "layout": float(a_match.group(1)) / 100,
            "color": float(a_match.group(2)) / 100,
            "typo": float(a_match.group(3)) / 100,
            "overall": float(a_match.group(4)) / 100,
        }

    # parse Candidate B scores
    b_match = re.search(r"Candidate B.*?layout\s*=\s*(\d+).*?color\s*=\s*(\d+).*?typo\s*=\s*(\d+).*?overall\s*=\s*(\d+)", text, re.IGNORECASE | re.DOTALL)
    if b_match:
        scores_b = {
            "layout": float(b_match.group(1)) / 100,
            "color": float(b_match.group(2)) / 100,
            "typo": float(b_match.group(3)) / 100,
            "overall": float(b_match.group(4)) / 100,
        }

    # determine winner
    winner = "tie"
    if re.search(r"boxed\{.*?A is better.*?\}", text, re.IGNORECASE):
        winner = "A"
    elif re.search(r"boxed\{.*?B is better.*?\}", text, re.IGNORECASE):
        winner = "B"

    return scores_a, scores_b, winner
