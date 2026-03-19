from reward.programmatic import compute_reward_code, render_html_to_image
from reward.vlm_reward import compute_vlm_reward


def compute_composite_reward(ref_image: bytes, generated_html: str,
                             model: str = "gpt-4o") -> dict:
    R_prog = compute_reward_code(ref_image, generated_html)

    rendered_image = render_html_to_image(generated_html)
    vlm_scores = compute_vlm_reward(ref_image, rendered_image, model=model)
    R_vlm = vlm_scores["total"]

    R_total = 0.4 * R_prog + 0.6 * R_vlm

    return {
        "programmatic": R_prog,
        "vlm": vlm_scores,
        "total": R_total,
    }
