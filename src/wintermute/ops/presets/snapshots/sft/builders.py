SFT_EVAL_MAX_CHARS_EXCLUSIVE = 10_000_000


def unitary_sft_split_sizes(total_chars: int) -> dict[str, int]:
    if total_chars < 21:
        raise ValueError("a unitary SFT datasource needs at least 21 chars for non-empty train and eval splits")

    eval_chars = min(total_chars // 21, SFT_EVAL_MAX_CHARS_EXCLUSIVE - 1)
    return {
        "train": total_chars - eval_chars,
        "eval": eval_chars,
    }
