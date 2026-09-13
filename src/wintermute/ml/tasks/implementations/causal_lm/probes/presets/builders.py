from __future__ import annotations

from wintermute.ml.tasks.implementations.causal_lm.probes import ChrF, Exact, Probe, Verifier
from wintermute.ml.tokenization.chat_format import (
    AssistantControlToken,
    AssistantPromptFormat,
)


_MODEL_SELECTED_ASSISTANT = AssistantPromptFormat(
    last_prompt_control_token=AssistantControlToken.ASSISTANT,
)


def _raw(id: str, prompt: str, verifier: Verifier | None = None) -> Probe:
    return Probe(id=id, prompt=prompt, verifier=verifier)


def _chat(id: str, prompt: str, verifier: Verifier | None = None) -> Probe:
    return Probe(
        id=id,
        prompt=prompt,
        assistant_prompt_format=_MODEL_SELECTED_ASSISTANT,
        verifier=verifier,
    )


def _exact(expected: str) -> Exact:
    return Exact(expected, embedded_score=0.7)


def _chrf(*references: str) -> ChrF:
    return ChrF(references=references)
