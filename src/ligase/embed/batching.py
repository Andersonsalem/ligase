from __future__ import annotations

MAX_TOKENS_PER_BATCH = 4096


def plan_batches(lengths: list[int], budget: int = MAX_TOKENS_PER_BATCH) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: list[list[int]] = []
    current: list[int] = []
    cur_max = 0
    for i in order:
        new_max = max(cur_max, lengths[i])
        if current and new_max * (len(current) + 1) > budget:
            batches.append(current)
            current, cur_max = [i], lengths[i]
        else:
            current.append(i)
            cur_max = new_max
    if current:
        batches.append(current)
    return batches
