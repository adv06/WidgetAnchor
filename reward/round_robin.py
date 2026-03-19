from reward.vlm_reward import compute_vlm_comparison
from reward.programmatic import compute_reward_code


def round_robin_scoring(ref_image: bytes, candidates: list[tuple[bytes | None, str]],
                        prog_threshold: float = 0.3) -> list[float]:
    n = len(candidates)
    rewards = [0.0 for i in range(n)]
    pool = []

    # programmatic pre-filter
    for i in range(n):
        rendered_img, html = candidates[i]
        if rendered_img is None:
            rewards[i] = -1.0
            continue
        r_prog = compute_reward_code(ref_image, html)
        if r_prog < prog_threshold:
            rewards[i] = 0.0
        else:
            pool.append(i)
            rewards[i] = 1.0

    # round-robin VLM comparison 
    for i in range(len(pool)):
        for j in range(i):
            score = compute_vlm_comparison(ref_image, candidates[pool[i]][0], candidates[pool[j]][0])[2]
            if score == "A":
                rewards[pool[i]] += 1
            elif score == "B":
                rewards[pool[j]] += 1
            else:
                rewards[pool[i]] += 0.5
                rewards[pool[j]] += 0.5

    return rewards
