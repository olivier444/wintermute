from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Iterator, Sequence

from lm_eval.api.model import LM  # type: ignore[import-not-found]
import torch
import torch.nn as nn
import torch.nn.functional as F

from wintermute.ml.tokenization.chat_format import ConversationContext, ConversationTurn
from wintermute.tools.logging import console_log
from wintermute.tools.model import evaluating, extract_logits_from_model_output


class WintermuteLmEvalAdapter(LM):
    """Minimal lm-evaluation-harness interface for a Wintermute causal LM."""

    tokenizer = None

    def __init__(
        self,
        *,
        model: nn.Module,
        task_wrapper: Any,
        model_name: str,
        max_length: int,
        max_gen_toks: int = 256,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__()
        if max_length <= 1:
            raise ValueError("lm-eval requires a model context length greater than one")
        self.model = model
        self.task_wrapper = task_wrapper
        self._tokenizer = task_wrapper.tokenizer
        self._device = task_wrapper.device
        self._model_name = model_name
        self.max_length = max_length
        self.max_gen_toks = max_gen_toks
        self.batch_size = 1
        self._progress_callback = progress_callback
        self._completed_request_count = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def eot_token_id(self) -> int:
        return self._tokenizer.eos_id

    @property
    def tokenizer_name(self) -> str:
        return self._model_name

    def set_cache_hook(self, cache_hook: Any) -> None:
        self.cache_hook = cache_hook

    def loglikelihood(self, requests: Sequence[Any]) -> list[tuple[float, bool]]:
        responses: list[tuple[float, bool]] = []
        for request in self._requests_with_benchmark_logging(requests, "loglikelihood"):
            context, continuation = request.args
            context_tokens, continuation_tokens = self._encode_pair(
                str(context),
                str(continuation),
            )
            response = self._score_continuation(context_tokens, continuation_tokens)
            responses.append(response)
            self.cache_hook.add_partial("loglikelihood", request.args, response)
            self._record_completed_request()
        return responses

    def loglikelihood_rolling(self, requests: Sequence[Any]) -> list[float]:
        responses: list[float] = []
        for request in self._requests_with_benchmark_logging(
            requests,
            "loglikelihood_rolling",
        ):
            (text,) = request.args
            tokens = self._tokenizer.encode_to_ids(
                str(text),
                add_special_tokens=False,
            )
            response, _ = self._score_continuation([self.eot_token_id], tokens)
            responses.append(response)
            self.cache_hook.add_partial("loglikelihood_rolling", request.args, response)
            self._record_completed_request()
        return responses

    def generate_until(self, requests: Sequence[Any]) -> list[str]:
        responses: list[str] = []
        for request in self._requests_with_benchmark_logging(requests, "generate_until"):
            context, raw_generation_kwargs = request.args
            generation_kwargs = dict(raw_generation_kwargs)
            if generation_kwargs.get("do_sample", False):
                raise ValueError("Wintermute's lm-eval adapter only supports greedy generation")

            max_new_tokens = int(
                generation_kwargs.get("max_gen_toks", self.max_gen_toks)
            )
            response = self._generate(
                str(context),
                max_new_tokens=max_new_tokens,
                stop_sequences=self._stop_sequences(generation_kwargs.get("until")),
            )
            responses.append(response)
            self.cache_hook.add_partial("generate_until", request.args, response)
            self._record_completed_request()
        return responses

    @staticmethod
    def _requests_with_benchmark_logging(
        requests: Sequence[Any],
        request_type: str,
    ) -> Iterator[Any]:
        task_counts = Counter(
            task_name
            for request in requests
            if (task_name := getattr(request, "task_name", None))
        )
        current_task_name: str | None = None
        for request in requests:
            task_name = getattr(request, "task_name", None)
            if task_name and task_name != current_task_name:
                request_count = task_counts[task_name]
                request_label = "request" if request_count == 1 else "requests"
                console_log(
                    "eval",
                    f"starting lm-eval benchmark: {task_name} "
                    f"({request_type}, {request_count:,} {request_label})",
                )
            current_task_name = task_name
            yield request

    def _record_completed_request(self) -> None:
        self._completed_request_count += 1
        if self._progress_callback is not None:
            self._progress_callback(self._completed_request_count)

    def apply_chat_template(
        self,
        chat_history: list[dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        system: str | None = None
        turns: list[ConversationTurn] = []
        pending_user: str | None = None

        for message in chat_history:
            role = message["role"]
            content = message["content"]
            if not isinstance(content, str):
                raise TypeError("Wintermute's lm-eval chat adapter only supports text messages")
            if role == "system" and system is None and pending_user is None and not turns:
                system = content
            elif role == "user" and pending_user is None:
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                turns.append(ConversationTurn(user=pending_user, assistant=content))
                pending_user = None
            else:
                raise ValueError("lm-eval chat history must alternate user and assistant messages")

        if pending_user is not None:
            if not add_generation_prompt:
                raise ValueError(
                    "add_generation_prompt=False requires a final assistant message"
                )
            return self.task_wrapper.chat.render_prompt(
                ConversationContext(
                    current_user=pending_user,
                    history=tuple(turns),
                    system=system,
                )
            ).text

        if not turns or add_generation_prompt:
            raise ValueError("lm-eval chat history must end with a user message")

        final_turn = turns[-1]
        prompt = self.task_wrapper.chat.render_prompt(
            ConversationContext(
                current_user=final_turn.user,
                history=tuple(turns[:-1]),
                system=system,
            )
        ).text
        return prompt + self.task_wrapper.chat.render_completion(final=final_turn.assistant)

    def chat_template(self, chat_template: bool | str = False) -> str | None:
        return type(self.task_wrapper.chat).__name__ if chat_template else None

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self._model_name,
            "max_length": self.max_length,
        }

    def _encode_pair(
        self,
        context: str,
        continuation: str,
    ) -> tuple[list[int], list[int]]:
        if context == "":
            return (
                [self.eot_token_id],
                self._tokenizer.encode_to_ids(
                    continuation,
                    add_special_tokens=False,
                ),
            )

        trailing_space_count = len(context) - len(context.rstrip())
        if trailing_space_count:
            continuation = context[-trailing_space_count:] + continuation
            context = context[:-trailing_space_count]

        whole_tokens = self._tokenizer.encode_to_ids(
            context + continuation,
            add_special_tokens=False,
        )
        context_tokens = self._tokenizer.encode_to_ids(
            context,
            add_special_tokens=False,
        )
        if not context_tokens:
            return [self.eot_token_id], whole_tokens
        return context_tokens, whole_tokens[len(context_tokens):]

    def _score_continuation(
        self,
        context_tokens: list[int],
        continuation_tokens: list[int],
    ) -> tuple[float, bool]:
        if not continuation_tokens:
            return 0.0, True

        all_tokens = context_tokens + continuation_tokens
        target_index = len(context_tokens)
        total_log_likelihood = 0.0
        is_greedy = True

        with evaluating(self.model):
            while target_index < len(all_tokens):
                remaining = len(all_tokens) - target_index
                target_count = min(self.max_length, remaining)
                target_end = target_index + target_count
                if remaining <= self.max_length:
                    input_start = max(0, target_end - 1 - self.max_length)
                else:
                    input_start = target_index - 1

                input_tokens = all_tokens[input_start:target_end - 1]
                score_offset = target_index - 1 - input_start
                input_ids = torch.tensor(
                    input_tokens,
                    dtype=torch.long,
                    device=self.device,
                ).unsqueeze(0)
                targets = torch.tensor(
                    all_tokens[target_index:target_end],
                    dtype=torch.long,
                    device=self.device,
                )
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.task_wrapper.bf_16_enabled,
                ):
                    logits = extract_logits_from_model_output(self.model(input_ids=input_ids))
                scored_logits = logits[0, score_offset:score_offset + target_count, :]
                total_log_likelihood -= float(
                    F.cross_entropy(scored_logits.float(), targets, reduction="sum").item()
                )
                is_greedy = is_greedy and bool(
                    torch.equal(scored_logits.argmax(dim=-1), targets)
                )
                target_index = target_end

        return total_log_likelihood, is_greedy

    def _generate(
        self,
        context: str,
        *,
        max_new_tokens: int,
        stop_sequences: tuple[str, ...],
    ) -> str:
        if max_new_tokens <= 0:
            return ""

        max_new_tokens = min(max_new_tokens, self.max_length - 1)
        prompt_tokens = self._tokenizer.encode_to_ids(
            context,
            add_special_tokens=False,
        ) or [self.eot_token_id]
        prompt_tokens = prompt_tokens[-(self.max_length - max_new_tokens):]
        generated_tokens = self.task_wrapper.generator.generate(
            prompt_tokens,
            self.model,
            greedy=True,
            max_new_tokens=max_new_tokens,
        )
        text = self.task_wrapper.generator.decode(
            generated_tokens,
            use_chat_format=False,
        ).final
        stop_indexes = [
            index
            for stop in stop_sequences
            if stop and (index := text.find(stop)) >= 0
        ]
        return text[:min(stop_indexes)] if stop_indexes else text

    @staticmethod
    def _stop_sequences(raw_until: Any) -> tuple[str, ...]:
        if raw_until is None:
            return ()
        if isinstance(raw_until, str):
            return (raw_until,)
        if isinstance(raw_until, Sequence):
            return tuple(str(value) for value in raw_until)
        raise TypeError("lm-eval generation 'until' must be a string or list of strings")
