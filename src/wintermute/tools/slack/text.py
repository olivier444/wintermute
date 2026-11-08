from __future__ import annotations

import re


SLACK_CLIENT_SIGNATURE_SUFFIX_RE = re.compile(
    r"\s*\*?Envoyé\s+avec\*?\s+\S+\s*$",
    re.IGNORECASE,
)


def strip_slack_client_signature(text: str) -> str:
    return SLACK_CLIENT_SIGNATURE_SUFFIX_RE.sub("", text).strip()


def truncate_for_slack(text: str, *, max_chars: int, max_lines: int) -> str:
    truncated = False
    compact = text
    lines = compact.splitlines()
    if len(lines) > max_lines:
        compact = "\n".join(lines[:max_lines])
        truncated = True

    suffix = "\n... [truncated]" if "\n" in compact else " ... [truncated]"
    if len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
        truncated = True

    if truncated:
        budget = max(0, max_chars - len(suffix))
        compact = compact[:budget].rstrip()
        compact = f"{compact}{suffix}" if compact else suffix.strip()

    return compact


def format_slack_code_block(text: str, *, language: str = "") -> str:
    safe_text = text.strip().replace("```", "``\u200b`")
    return f"```{language}\n{safe_text}\n```"
