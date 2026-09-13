from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from wintermute.ml.tasks.factory import (
    InferenceSession,
    InferenceSessionState,
    InferenceSettings,
    InferenceTurnState,
    OutputPreview,
)
from wintermute.ml.tokenization.chat_format import (
    AssistantControlToken,
    AssistantPromptFormat,
    ConversationContext,
    ConversationTurn,
)
from wintermute.tools.logging import console_log

if TYPE_CHECKING:
    from wintermute.ml.tasks.implementations.causal_lm.task import CausalLmTaskWrapper


class CausalLmInferenceSession(InferenceSession):
    def __init__(
        self,
        *,
        wrapper: CausalLmTaskWrapper,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_state: InferenceSessionState | None = None,
    ) -> None:
        self._wrapper = wrapper
        self._model = model
        self._system_prompt = (system_prompt or "").strip() or None
        self._turns: list[ConversationTurn] = []
        self._use_chat_format = True
        self._settings = InferenceSettings(
            max_new_tokens=self._wrapper._requested_infer_max_tokens(),
            temperature=self._wrapper._default_infer_temperature(),
            top_p=self._wrapper._default_infer_top_p(),
            top_k=self._wrapper._default_infer_top_k(),
            greedy=False,
        )
        if initial_state is not None:
            self._turns = [
                ConversationTurn(user=turn.user_text, assistant=turn.assistant_text)
                for turn in initial_state.turns
            ]
            self._settings = initial_state.settings.clone()
            self._use_chat_format = initial_state.use_chat_format

    @property
    def settings(self) -> InferenceSettings:
        return self._settings

    @property
    def history_turn_count(self) -> int:
        return len(self._turns)

    def infer(self, arguments: str) -> OutputPreview:
        self._settings.validate()
        requested_max_new_tokens = self._settings.max_new_tokens
        context_limit = self._wrapper._context_token_limit(self._model)
        prompt_budget = max(
            1,
            context_limit - min(requested_max_new_tokens, context_limit - 1),
        )
        readable, prompt_tokens = self._build_prompt_tokens(
            arguments,
            prompt_budget=prompt_budget,
        )

        console_log("infer", f"input sequence:\n\n{readable}")

        max_new_tokens = min(
            requested_max_new_tokens,
            context_limit - len(prompt_tokens),
        )

        generated_tokens = self._wrapper.generator.generate(
            prompt_tokens,
            self._model,
            greedy=self._settings.greedy,
            max_new_tokens=max_new_tokens,
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            top_k=self._settings.top_k,
        )
        completion = self._wrapper.generator.decode(
            generated_tokens,
            assistant_prompt_format=self._assistant_prompt_format(),
            use_chat_format=self._use_chat_format,
        )

        rendered_completion = (
            self._wrapper.chat.render_completion(
                final=completion.final,
                task=completion.task,
                verbosity=completion.verbosity,
                format=completion.format,
                scratchpad=completion.scratchpad,
            )
            if self._use_chat_format
            else completion.final
        )
        console_log("infer", f"reply:\n\n{rendered_completion}")

        self._turns.append(ConversationTurn(user=arguments, assistant=completion.final))
        return OutputPreview(completion)

    def reset(self) -> None:
        self._turns.clear()

    def export_state(self) -> InferenceSessionState:
        return InferenceSessionState(
            turns=tuple(
                InferenceTurnState(user_text=turn.user, assistant_text=turn.assistant)
                for turn in self._turns
            ),
            settings=self._settings.clone(),
            use_chat_format=self._use_chat_format,
        )

    def _build_prompt_tokens(
        self,
        arguments: str,
        *,
        prompt_budget: int,
    ) -> tuple[str, list[int]]:
        if not self._use_chat_format:
            return self._build_raw_prompt_tokens(arguments, prompt_budget=prompt_budget)

        turns = self._turns
        readable = "<undef>"

        while True:
            readable, prompt_tokens = self._encode_context(
                ConversationContext(
                    current_user=arguments,
                    history=tuple(turns),
                    system=self._system_prompt,
                )
            )
            if len(prompt_tokens) <= prompt_budget:
                return (readable, list(prompt_tokens))
            if len(turns) == 0:
                trimmed_readable, trimmed_tokens = self._build_prompt_tokens_with_trimmed_user(
                    arguments,
                    prompt_budget=prompt_budget,
                )
                return (f"**TRUNCATED **\n\n{trimmed_readable}", trimmed_tokens)
            turns = turns[1:]

    def _build_prompt_tokens_with_trimmed_user(
        self,
        arguments: str,
        *,
        prompt_budget: int,
    ) -> tuple[str, list[int]]:
        base_readable, base_prompt_tokens = self._encode_context(
            ConversationContext(current_user="", system=self._system_prompt)
        )
        if len(base_prompt_tokens) > prompt_budget:
            raise ValueError(
                "[inference] system prompt and chat framing exceed the available context budget"
            )

        user_ids = self._wrapper.tokenizer.encode_to_ids(
            arguments,
            add_special_tokens=False,
        )
        best_readable = base_readable
        best_prompt_tokens = list(base_prompt_tokens)
        low = 0
        high = len(user_ids)

        while low <= high:
            keep_count = (low + high) // 2
            truncated_user_ids = user_ids[-keep_count:] if keep_count > 0 else []
            truncated_user_text = self._wrapper.tokenizer.decode(
                truncated_user_ids,
                skip_special_tokens=False,
            )
            readable, prompt_tokens = self._encode_context(
                ConversationContext(
                    current_user=truncated_user_text,
                    system=self._system_prompt,
                )
            )
            if len(prompt_tokens) <= prompt_budget:
                best_readable = readable
                best_prompt_tokens = list(prompt_tokens)
                low = keep_count + 1
            else:
                high = keep_count - 1

        return best_readable, best_prompt_tokens

    def _encode_context(self, context: ConversationContext) -> tuple[str, list[int]]:
        rendered_context = self._wrapper.chat.render_prompt(
            context,
            self._assistant_prompt_format(),
        )
        rendered = rendered_context.text
        token_ids = self._wrapper.tokenizer.encode_to_ids(
            rendered,
            add_special_tokens=False,
        )
        return (rendered, list(token_ids))

    def _assistant_prompt_format(self) -> AssistantPromptFormat | None:
        if not self._use_chat_format or self._settings.force_final:
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
        prompt_tokens = self._wrapper.tokenizer.encode_to_ids(
            arguments,
            add_special_tokens=False,
        )
        if len(prompt_tokens) <= prompt_budget:
            return arguments, list(prompt_tokens)

        truncated_tokens = prompt_tokens[-prompt_budget:]
        truncated_text = self._wrapper.tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=False,
        )
        return f"**TRUNCATED **\n\n{truncated_text}", list(truncated_tokens)
