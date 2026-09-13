from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from wintermute.ml.tokenization.chat_format import (
    AssistantCompletion,
    AssistantPromptFormat,
    ChatFormat,
)
from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.tools.model import evaluating, extract_logits_from_model_output


class CausalLmGenerator:
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

            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.bf16_enabled,
                ):
                    logits = extract_logits_from_model_output(model(input_ids=ids))

                next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                ids = torch.cat([ids, next_id], dim=1)
                if all(int(tok) == eos_id for tok in next_id.view(-1).tolist()):
                    break

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

            for _ in range(max_new_tokens):
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.bf16_enabled,
                ):
                    logits = extract_logits_from_model_output(model(input_ids=ids))
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

        return cast(list[int], ids[0].tolist())
    
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
