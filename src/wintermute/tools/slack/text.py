from __future__ import annotations

import re


CHATGPT_SIGNATURE_LINES = {
    "*Envoyé avec* ChatGPT",
    "*Envoye avec* ChatGPT",
    "Envoyé avec ChatGPT",
    "Envoye avec ChatGPT",
    "*Sent with* ChatGPT",
    "Sent with ChatGPT",
}

CHATGPT_SIGNATURE_SUFFIX_RE = re.compile(
    r"\s*\*?\s*(?:Envoy[eé]\s+avec|Sent\s+with)\s*\*?\s+ChatGPT\s*$",
    re.IGNORECASE,
)


def strip_slack_client_signature(text: str) -> str:
    lines = _strip_slack_client_signature_suffix(text).splitlines()
    while lines and _is_slack_client_signature_line(lines[-1]):
        lines.pop()
    return "\n".join(lines).strip()


def _strip_slack_client_signature_suffix(text: str) -> str:
    return CHATGPT_SIGNATURE_SUFFIX_RE.sub("", text.rstrip())


def _is_slack_client_signature_line(line: str) -> bool:
    clean = line.strip()
    if clean in CHATGPT_SIGNATURE_LINES:
        return True

    normalized = clean.replace("*", "").strip().lower()
    return normalized in {
        "envoyé avec chatgpt",
        "envoye avec chatgpt",
        "sent with chatgpt",
    }


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
