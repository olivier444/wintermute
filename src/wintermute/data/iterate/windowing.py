from __future__ import annotations
from typing import Sequence, Iterator

from wintermute.data.iterate.dataclasses import TrainingExample, WindowingPolicy, WindowDefinition


def iter_windows(
    toks: Sequence[int],
    policy: WindowingPolicy,
) -> Iterator[WindowDefinition]:

    n = len(toks)

    if n == 0:
        return

    window_len = policy.window_max_len
    overlap_size = policy.overlap
    assert 0 <= overlap_size < window_len

    if n <= window_len: # 'small' doc - return one record only
        yield WindowDefinition(0, n, 0)
        return

    window_start = 0
    while window_start < n:
        window_end = min(window_start + window_len, n)
        new_tok_start = window_start if window_start == 0 else window_start + overlap_size

        new_tok_len = window_end - new_tok_start
        if new_tok_len < policy.min_new_tokens:
            if policy.align_last_window_to_end: # Make one last window aligned to the end
                start2 = max(0, n - window_len) # window_len < n here
                end2 = n
                new_start2 = new_tok_start

                if start2 != window_start:
                    yield WindowDefinition(start2, end2, new_start2)

            return

        yield WindowDefinition(window_start, window_end, new_tok_start)

        if window_end == n:
            return
        
        window_start += (window_len - overlap_size)


def build_example_from_window(
    rec_id: str,
    toks: Sequence[int],
    wd: WindowDefinition
) -> TrainingExample:
    
    w_toks = list(toks[wd.start:wd.end_excluded])

    loss_mask = [1.0] * len(w_toks)
    for i in range(wd.overlap_length()):
        loss_mask[i] = 0.0
    if loss_mask:
        # The first token in a standalone window has no previous token to predict it.
        loss_mask[0] = 0.0

    return TrainingExample(
        uid = f"{rec_id}_ws{wd.start}",
        payload = {
            "input_ids": w_toks,
            "loss_mask": loss_mask,
        },
        work_units = int(sum(loss_mask)),
        metadata = {
            "segments": [
                {
                "doc_id": rec_id,
                "w_start": wd.start,
                "w_end_excluded": wd.end_excluded,
                "w_new_tok_start": wd.new_tok_start,
                }
            ],
        },
    )        
