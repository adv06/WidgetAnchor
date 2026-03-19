from reward.vlm_reward import compute_vlm_comparison
from reward.programmatic import compute_reward_code


def round_robin_scoring(ref_image: bytes, candidates: list[tuple[bytes | None, str]],
                        prog_threshold: float = 0.3, prog_weight: float = 0.3,
                        ref_tsx: str = None) -> list[float]:
    n = len(candidates)
    prog_scores = [0.0] * n
    pool = []

    # programmatic scoring + pre-filter
    for i in range(n):
        rendered_img, tsx = candidates[i]
        if rendered_img is None:
            prog_scores[i] = -1.0
            continue
        prog_scores[i] = compute_reward_code(ref_image, tsx, rendered_image=rendered_img, ref_tsx=ref_tsx)
        if prog_scores[i] >= prog_threshold:
            pool.append(i)

    # round-robin VLM comparison on survivors
    win_counts = [0.0] * n
    for i in range(len(pool)):
        for j in range(i):
            score = compute_vlm_comparison(ref_image, candidates[pool[i]][0], candidates[pool[j]][0])[2]
            if score == "A":
                win_counts[pool[i]] += 1
            elif score == "B":
                win_counts[pool[j]] += 1
            else:
                win_counts[pool[i]] += 0.5
                win_counts[pool[j]] += 0.5

    # normalize win counts to [0, 1] among survivors bro we want small numebers
    max_wins = max((win_counts[i] for i in pool), default=1.0)
    if max_wins > 0:
        for i in pool:
            win_counts[i] /= max_wins

    # blend: prog_weight * programmatic + (1 - prog_weight) * vlm_wins --> dont want to get rid of the other reward signals
    rewards = [0.0] * n
    for i in range(n):
        if prog_scores[i] < 0:
            rewards[i] = -1.0  # render failed
        elif i not in pool:
            rewards[i] = prog_weight * prog_scores[i]  # below threshold, no VLM signal
        else:
            rewards[i] = prog_weight * prog_scores[i] + (1 - prog_weight) * win_counts[i]

    return rewards
