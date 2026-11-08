from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from wintermute.ml.inference.conversation import (
    InferenceSession,
    InferenceSettings,
    InferenceTurn,
    OutputPreview,
)
from wintermute.ml.tokenization.chat_format import (
    AssistantControlToken,
    AssistantPromptFormat,
    ConversationContext,
    ConversationTurn,
)
from wintermute.ml.tasks.implementations.causal_lm.generation import (
    build_chat_prompt_tokens,
)
from wintermute.tools.logging import console_log

if TYPE_CHECKING:
    from wintermute.ml.tasks.implementations.causal_lm.backend import CausalLmBackend


class CausalLmInferenceSession(InferenceSession):
    def __init__(
        self,
        *,
        backend: CausalLmBackend,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_turns: tuple[InferenceTurn, ...] = (),
    ) -> None:
        self._backend = backend
        self._model = model
        self._system_prompt = (system_prompt or "").strip() or None
        self._turns = [
            ConversationTurn(user=turn.user_text, assistant=turn.assistant_text)
            for turn in initial_turns
        ]

    @property
    def history_turn_count(self) -> int:
        return len(self._turns)

    def infer(self, arguments: str, settings: InferenceSettings) -> OutputPreview:
        settings.validate()
        requested_max_new_tokens = settings.max_new_tokens
        context_limit = self._backend.context_token_limit(self._model)
        prompt_budget = max(
            1,
            context_limit - min(requested_max_new_tokens, context_limit - 1),
        )
        readable, prompt_tokens = self._build_prompt_tokens(
            arguments,
            prompt_budget=prompt_budget,
            settings=settings,
        )

        console_log("infer", f"input sequence:\n\n{readable}")

        max_new_tokens = min(
            requested_max_new_tokens,
            context_limit - len(prompt_tokens),
        )

        generated_tokens = self._backend.generator.generate(
            prompt_tokens,
            self._model,
            greedy=settings.greedy,
            max_new_tokens=max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            top_k=settings.top_k,
        )
        completion = self._backend.generator.decode(
            generated_tokens,
            assistant_prompt_format=self._assistant_prompt_format(settings),
            use_chat_format=settings.use_chat_format,
        )

        rendered_completion = (
            self._backend.chat_format.render_completion(
                final=completion.final,
                task=completion.task,
                verbosity=completion.verbosity,
                format=completion.format,
                scratchpad=completion.scratchpad,
            )
            if settings.use_chat_format
            else completion.final
        )
        console_log("infer", f"reply:\n\n{rendered_completion}")

        self._turns.append(ConversationTurn(user=arguments, assistant=completion.final))
        return OutputPreview(completion)

    def reset(self) -> None:
        self._turns.clear()

    def export_turns(self) -> tuple[InferenceTurn, ...]:
        return tuple(
            InferenceTurn(user_text=turn.user, assistant_text=turn.assistant)
            for turn in self._turns
        )

    def _build_prompt_tokens(
        self,
        arguments: str,
        *,
        prompt_budget: int,
        settings: InferenceSettings,
    ) -> tuple[str, list[int]]:
        if not settings.use_chat_format:
            return self._build_raw_prompt_tokens(arguments, prompt_budget=prompt_budget)
        return build_chat_prompt_tokens(
            ConversationContext(
                current_user=arguments,
                history=tuple(self._turns),
                system=self._system_prompt,
            ),
            prompt_budget=prompt_budget,
            tokenizer=self._backend.tokenizer,
            chat_format=self._backend.chat_format,
            assistant_prompt_format=self._assistant_prompt_format(settings),
        )

    def _assistant_prompt_format(
        self,
        settings: InferenceSettings,
    ) -> AssistantPromptFormat | None:
        if not settings.use_chat_format or settings.force_final:
            return None
        return AssistantPromptFormat(
            last_prompt_control_token=AssistantControlToken.ASSISTANT,
        )

    def _build_raw_prompt_tokens(
        self,
        arguments: str,
        *,
        prompt_budget: int,
    ) -> tuple[str, list[int]]:
        prompt_tokens = self._backend.tokenizer.encode_to_ids(
            arguments,
            add_special_tokens=False,
        )
        if len(prompt_tokens) <= prompt_budget:
            return arguments, list(prompt_tokens)

        truncated_tokens = prompt_tokens[-prompt_budget:]
        truncated_text = self._backend.tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=False,
        )
        return f"**TRUNCATED **\n\n{truncated_text}", list(truncated_tokens)
