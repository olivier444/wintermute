from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

from wintermute.data.iterate.feedback import GeneratedCompletion
from wintermute.ml.inference.conversation import InferenceSettings
from wintermute.ml.tokenization.chat_format import (
    AssistantCompletion,
    AssistantPromptFormat,
    ChatFormat,
    ConversationContext,
)
from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.tools.model import (
    evaluating,
    extract_logits_from_model_output,
    extract_past_key_values_from_model_output,
)

if TYPE_CHECKING:
    from wintermute.ml.tasks.implementations.causal_lm.backend import CausalLmBackend


def encode_chat_context(
    context: ConversationContext,
    *,
    tokenizer: TokenizerWrapper,
    chat_format: ChatFormat,
    assistant_prompt_format: AssistantPromptFormat | None,
) -> tuple[str, list[int]]:
    rendered = chat_format.render_prompt(context, assistant_prompt_format).text
    token_ids = tokenizer.encode_to_ids(rendered, add_special_tokens=False)
    return rendered, list(token_ids)


def build_chat_prompt_tokens(
    context: ConversationContext,
    *,
    prompt_budget: int,
    tokenizer: TokenizerWrapper,
    chat_format: ChatFormat,
    assistant_prompt_format: AssistantPromptFormat | None,
) -> tuple[str, list[int]]:
    """Fit a chat prompt by dropping old turns, then trimming the current user."""
    turns = context.history
    while True:
        readable, prompt_tokens = encode_chat_context(
            ConversationContext(
                current_user=context.current_user,
                history=turns,
                system=context.system,
            ),
            tokenizer=tokenizer,
            chat_format=chat_format,
            assistant_prompt_format=assistant_prompt_format,
        )
        if len(prompt_tokens) <= prompt_budget:
            return readable, prompt_tokens
        if not turns:
            break
        turns = turns[1:]

    base_readable, base_prompt_tokens = encode_chat_context(
        ConversationContext(current_user="", system=context.system),
        tokenizer=tokenizer,
        chat_format=chat_format,
        assistant_prompt_format=assistant_prompt_format,
    )
    if len(base_prompt_tokens) > prompt_budget:
        raise ValueError(
            "system prompt and chat framing exceed the available context budget"
        )

    user_ids = tokenizer.encode_to_ids(context.current_user, add_special_tokens=False)
    best_readable = base_readable
    best_prompt_tokens = base_prompt_tokens
    low = 0
    high = len(user_ids)
    while low <= high:
        keep_count = (low + high) // 2
        truncated_ids = user_ids[-keep_count:] if keep_count > 0 else []
        truncated_user = tokenizer.decode(truncated_ids, skip_special_tokens=False)
        readable, prompt_tokens = encode_chat_context(
            ConversationContext(current_user=truncated_user, system=context.system),
            tokenizer=tokenizer,
            chat_format=chat_format,
            assistant_prompt_format=assistant_prompt_format,
        )
        if len(prompt_tokens) <= prompt_budget:
            best_readable = readable
            best_prompt_tokens = prompt_tokens
            low = keep_count + 1
        else:
            high = keep_count - 1

    return f"**TRUNCATED **\n\n{best_readable}", best_prompt_tokens


class CausalLmTokenGenerator:
    def __init__(
        self,
        *,
        device: torch.device,
        bf16_enabled: bool,
        tokenizer: TokenizerWrapper,
        chat_format: ChatFormat,
    ) -> None:
        self.device = device
        self.bf16_enabled = bf16_enabled
        self.tokenizer = tokenizer
        self.chat_format = chat_format

    def generate(
        self,
        prompt_tokens: list[int],
        model: nn.Module,
        *,
        greedy: bool,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int | None = 50,
        repetition_penalty: float = 1.15,
    ) -> list[int]:
        input_ids = torch.tensor(
            prompt_tokens,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        output_tokens = (
            self._generate_greedy(model, input_ids, max_new_tokens)
            if greedy
            else self._generate_sample(
                model,
                input_ids,
                max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
            )
        )
        return output_tokens[len(prompt_tokens):]

    def decode(
        self,
        generated_tokens: list[int],
        *,
        assistant_prompt_format: AssistantPromptFormat | None = None,
        use_chat_format: bool = True,
    ) -> AssistantCompletion:
        eos_id = self.tokenizer.eos_id
        if eos_id in generated_tokens:
            generated_tokens = generated_tokens[:generated_tokens.index(eos_id)]

        decoded = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=False,
        )
        if not use_chat_format:
            return AssistantCompletion(final=decoded)
        return self.chat_format.parse_completion(decoded, assistant_prompt_format)

    def _generate_greedy(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int,
    ) -> list[int]:
        eos_id = self.tokenizer.eos_id

        with evaluating(model):
            ids = input_ids
            model_input_ids = ids
            past_key_values: object | None = None

            for _ in range(max_new_tokens):
                logits, past_key_values = self._forward_model(
                    model,
                    model_input_ids,
                    past_key_values,
                )

                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if all(int(tok) == eos_id for tok in next_id.view(-1).tolist()):
                    break
                model_input_ids = next_id if past_key_values is not None else ids

        return cast(list[int], ids[0].tolist())

    def _generate_sample(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_p: float = 0.85,
        top_k: int | None = 50,
        repetition_penalty: float = 1.15,
    ) -> list[int]:
        eos_id = self.tokenizer.eos_id

        with evaluating(model):
            ids = input_ids
            model_input_ids = ids
            past_key_values: object | None = None

            for _ in range(max_new_tokens):
                logits, past_key_values = self._forward_model(
                    model,
                    model_input_ids,
                    past_key_values,
                )
                next_logits = logits[:, -1, :]

                next_logits = self._apply_repetition_penalty(
                    next_logits,
                    ids,
                    repetition_penalty,
                )
                next_logits = next_logits / temperature

                if top_k is not None:
                    next_logits = self._top_k_filter(next_logits, top_k)

                probs = torch.softmax(next_logits, dim=-1)
                next_id = self._top_p_sample(probs, top_p)

                ids = torch.cat([ids, next_id], dim=1)
                if all(int(tok) == eos_id for tok in next_id.view(-1).tolist()):
                    break
                model_input_ids = next_id if past_key_values is not None else ids

        return cast(list[int], ids[0].tolist())

    def _forward_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        past_key_values: object | None,
    ) -> tuple[torch.Tensor, object | None]:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.bf16_enabled,
        ):
            output = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
        return (
            extract_logits_from_model_output(output),
            extract_past_key_values_from_model_output(output),
        )
    
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        generated_ids: torch.Tensor,
        repetition_penalty: float,
    ) -> torch.Tensor:
        if repetition_penalty == 1.0:
            return logits

        for batch_index in range(logits.size(0)):
            uniq = torch.unique(generated_ids[batch_index])
            selected = logits[batch_index, uniq]
            logits[batch_index, uniq] = torch.where(
                selected < 0,
                selected * repetition_penalty,
                selected / repetition_penalty,
            )

        return logits

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        vocab_size = logits.size(-1)
        if k <= 0 or k >= vocab_size:
            return logits

        values, _ = torch.topk(logits, k)
        cutoff = values[..., -1, None]
        return torch.where(
            logits < cutoff,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    @staticmethod
    def _top_p_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        remove = cumulative_probs > p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False

        filtered = torch.where(remove, torch.zeros_like(sorted_probs), sorted_probs)
        denominator = filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        filtered = filtered / denominator

        sampled_in_sorted = torch.multinomial(filtered, num_samples=1)
        return torch.gather(sorted_idx, dim=-1, index=sampled_in_sorted)


@dataclass(frozen=True)
class CausalLmGenerationTrace:
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]


class CausalLmFeedbackCandidateGenerator:
    """Sample final-answer candidates from the current policy model."""

    def __init__(
        self,
        *,
        backend: CausalLmBackend,
        model: nn.Module,
        settings: InferenceSettings | None = None,
    ) -> None:
        self.backend = backend
        self.model = model
        self.settings = settings if settings is not None else InferenceSettings()
        self.settings.validate()
        if self.settings.greedy:
            raise ValueError("online feedback generation requires sampling")

    def generate(
        self,
        contexts: Sequence[ConversationContext],
        *,
        candidates_per_prompt: int,
    ) -> Sequence[Sequence[GeneratedCompletion]]:
        if candidates_per_prompt < 2:
            raise ValueError("candidates_per_prompt must be at least two")

        context_limit = self.backend.context_token_limit(self.model)
        if context_limit < 2:
            raise ValueError("model context limit leaves no room for generation")
        requested_tokens = self.settings.max_new_tokens
        prompt_budget = max(1, context_limit - min(requested_tokens, context_limit - 1))

        results: list[list[GeneratedCompletion]] = []
        for context in contexts:
            _, prompt_tokens = build_chat_prompt_tokens(
                context,
                prompt_budget=prompt_budget,
                tokenizer=self.backend.tokenizer,
                chat_format=self.backend.chat_format,
                assistant_prompt_format=None,
            )
            max_new_tokens = min(requested_tokens, context_limit - len(prompt_tokens))
            prompt_token_ids = tuple(prompt_tokens)
            candidates: list[GeneratedCompletion] = []
            for _ in range(candidates_per_prompt):
                generated_tokens = self.backend.generator.generate(
                    prompt_tokens,
                    self.model,
                    greedy=False,
                    max_new_tokens=max_new_tokens,
                    temperature=self.settings.temperature,
                    top_p=self.settings.top_p,
                    top_k=self.settings.top_k,
                )
                candidates.append(
                    GeneratedCompletion(
                        completion=self.backend.generator.decode(
                            generated_tokens,
                            assistant_prompt_format=None,
                        ).final,
                        trace=CausalLmGenerationTrace(
                            prompt_token_ids=prompt_token_ids,
                            completion_token_ids=tuple(generated_tokens),
                        ),
                    )
                )
            results.append(candidates)
        return results
