from __future__ import annotations

from dataclasses import dataclass

from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.ml.tokenization.chat_format import (
    AssistantControlToken,
    AssistantPromptFormat,
    WINTERMUTE_FORMAT,
    WINTERMUTE_TASK,
    WINTERMUTE_VERBOSITY,
    ChatFormat,
    ConversationContext,
    WintermuteChatFormat,
)


@dataclass(frozen=True)
class TokenizedConversation:
    """One bounded SFT turn with routing-label masks aligned to retained tokens."""

    prompt_ids: list[int]
    completion_ids: list[int]
    tag_loss_mask: list[float]
    task_label_mask: list[float]
    verbosity_label_mask: list[float]
    format_label_mask: list[float]
    final_start: int
    final_len: int
    used_scratchpad: bool
    dropped_history_turns: int
    dropped_system: bool
    rejection_reason: str | None = None

    @property
    def is_rejected(self) -> bool:
        return self.rejection_reason is not None

    @classmethod
    def from_context(
        cls,
        tokenizer: TokenizerWrapper,
        *,
        chat_format: ChatFormat,
        context: ConversationContext,
        current_assistant: str,
        current_scratchpad: str | None,
        current_task: str | None = None,
        current_verbosity: str | None = None,
        current_format: str | None = None,
        max_len: int,
    ) -> TokenizedConversation:
        rendered_context = chat_format.render_prompt(
            context,
            AssistantPromptFormat(
                last_prompt_control_token=AssistantControlToken.ASSISTANT,
            ),
        )
        current_ids = cls._encode_ids(tokenizer, rendered_context.current)

        # The current prompt and final answer are mandatory. Optional context is
        # considered only after this minimal form is known to fit.
        (
            final_only_ids,
            final_only_tag_mask,
            final_only_task_mask,
            final_only_verbosity_mask,
            final_only_format_mask,
            final_only_start,
            final_only_len,
        ) = cls._build_completion(
            tokenizer,
            chat_format=chat_format,
            assistant=current_assistant,
            task=current_task,
            verbosity=current_verbosity,
            format=current_format,
        )
        required_turn_len = len(current_ids) + len(final_only_ids)
        if required_turn_len > max_len:
            return cls(
                prompt_ids=[],
                completion_ids=[],
                tag_loss_mask=[],
                task_label_mask=[],
                verbosity_label_mask=[],
                format_label_mask=[],
                final_start=0,
                final_len=0,
                used_scratchpad=False,
                dropped_history_turns=len(context.history),
                dropped_system=bool(context.system),
                rejection_reason="required_turn_exceeds_max_example_len",
            )

        completion_ids = final_only_ids
        completion_tag_mask = final_only_tag_mask
        completion_task_mask = final_only_task_mask
        completion_verbosity_mask = final_only_verbosity_mask
        completion_format_mask = final_only_format_mask
        final_start = final_only_start
        final_len = final_only_len
        used_scratchpad = False
        if current_scratchpad:
            # A scratchpad is kept whole or removed whole; it is never truncated.
            (
                full_completion_ids,
                full_completion_tag_mask,
                full_completion_task_mask,
                full_completion_verbosity_mask,
                full_completion_format_mask,
                full_final_start,
                full_final_len,
            ) = cls._build_completion(
                tokenizer,
                chat_format=chat_format,
                assistant=current_assistant,
                task=current_task,
                verbosity=current_verbosity,
                format=current_format,
                scratchpad=current_scratchpad,
            )
            if len(current_ids) + len(full_completion_ids) <= max_len:
                completion_ids = full_completion_ids
                completion_tag_mask = full_completion_tag_mask
                completion_task_mask = full_completion_task_mask
                completion_verbosity_mask = full_completion_verbosity_mask
                completion_format_mask = full_completion_format_mask
                final_start = full_final_start
                final_len = full_final_len
                used_scratchpad = True

        consumed_len = len(current_ids) + len(completion_ids)

        # Historical turns contain only the assistant's final answer. Task labels
        # and scratchpads supervise the current turn but are not conversation state.
        history_blocks: list[tuple[list[int], list[float]]] = []
        for block in rendered_context.history:
            block_ids = cls._encode_ids(tokenizer, block)
            history_blocks.append((block_ids, [0.0] * len(block_ids)))

        # Retain the most recent complete turns that fit the remaining budget.
        kept_history_reversed: list[tuple[list[int], list[float]]] = []
        for block_ids, block_task_mask in reversed(history_blocks):
            if consumed_len + len(block_ids) > max_len:
                break
            kept_history_reversed.append((block_ids, block_task_mask))
            consumed_len += len(block_ids)

        kept_history_blocks = list(reversed(kept_history_reversed))
        prompt_parts = list(kept_history_blocks)

        # Recent history has priority over the system prompt under context pressure.
        dropped_system = False
        if rendered_context.system:
            system_ids = cls._encode_ids(tokenizer, rendered_context.system)
            if consumed_len + len(system_ids) <= max_len:
                prompt_parts.insert(0, (system_ids, [0.0] * len(system_ids)))
            else:
                dropped_system = True

        prompt_parts.append((current_ids, [0.0] * len(current_ids)))
        prompt_ids: list[int] = []
        prompt_task_mask: list[float] = []
        for part_ids, part_task_mask in prompt_parts:
            prompt_ids.extend(part_ids)
            prompt_task_mask.extend(part_task_mask)

        return cls(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            tag_loss_mask=[0.0] * len(prompt_ids) + completion_tag_mask,
            task_label_mask=prompt_task_mask + completion_task_mask,
            verbosity_label_mask=[0.0] * len(prompt_ids) + completion_verbosity_mask,
            format_label_mask=[0.0] * len(prompt_ids) + completion_format_mask,
            final_start=final_start,
            final_len=final_len,
            used_scratchpad=used_scratchpad,
            dropped_history_turns=len(history_blocks) - len(kept_history_blocks),
            dropped_system=dropped_system,
        )

    @staticmethod
    def _build_completion(
        tokenizer: TokenizerWrapper,
        *,
        chat_format: ChatFormat,
        assistant: str,
        task: str | None,
        verbosity: str | None,
        format: str | None,
        scratchpad: str | None = None,
    ) -> tuple[
        list[int],
        list[float],
        list[float],
        list[float],
        list[float],
        int,
        int,
    ]:
        content = chat_format.render_completion(
            final=assistant,
            task=task,
            verbosity=verbosity,
            format=format,
            scratchpad=scratchpad,
            include_terminator=False,
        )
        rendered = chat_format.render_completion(
            final=assistant,
            task=task,
            verbosity=verbosity,
            format=format,
            scratchpad=scratchpad,
        )
        label_spans = (
            TokenizedConversation._routing_label_spans(
                content,
                task=task,
                verbosity=verbosity,
                format=format,
            )
            if isinstance(chat_format, WintermuteChatFormat)
            else {}
        )
        completion_ids, tag_mask, label_masks = TokenizedConversation._encode_with_label_masks(
            tokenizer,
            rendered,
            label_spans,
        )
        final_char_start = len(content) - len(assistant)
        final_start = len(
            TokenizedConversation._encode_ids(tokenizer, rendered[:final_char_start])
        )
        final_end = len(TokenizedConversation._encode_ids(tokenizer, content))
        eos_id = tokenizer.eos_id
        if eos_id not in completion_ids:
            completion_ids.append(eos_id)
            tag_mask.append(0.0)
            for mask in label_masks.values():
                mask.append(0.0)
        return (
            completion_ids,
            tag_mask,
            label_masks["task"],
            label_masks["verbosity"],
            label_masks["format"],
            final_start,
            final_end - final_start,
        )

    @staticmethod
    def _routing_label_spans(
        continuation: str,
        *,
        task: str | None,
        verbosity: str | None,
        format: str | None,
    ) -> dict[str, tuple[int, int]]:
        """Locate routing values only, excluding their special-token markers."""

        spans: dict[str, tuple[int, int]] = {}
        cursor = 0
        for name, marker, label in (
            ("task", WINTERMUTE_TASK, task),
            ("verbosity", WINTERMUTE_VERBOSITY, verbosity),
            ("format", WINTERMUTE_FORMAT, format),
        ):
            if not label:
                continue
            clause = f"{marker}{label}"
            if not continuation.startswith(clause, cursor):
                return {}
            spans[name] = (cursor + len(marker), cursor + len(clause))
            cursor += len(clause)
        return spans

    @staticmethod
    def _encode_with_label_masks(
        tokenizer: TokenizerWrapper,
        text: str,
        label_spans: dict[str, tuple[int, int]],
    ) -> tuple[list[int], list[float], dict[str, list[float]]]:
        ids = TokenizedConversation._encode_ids(tokenizer, text)
        tag_mask = [0.0] * len(ids)
        label_masks = {
            name: [0.0] * len(ids)
            for name in ("task", "verbosity", "format")
        }
        for name, span in label_spans.items():
            # Prefix lengths translate the rendered character span into positions
            # in the single, full-sequence tokenization used for training.
            start = len(TokenizedConversation._encode_ids(tokenizer, text[:span[0]]))
            end = len(TokenizedConversation._encode_ids(tokenizer, text[:span[1]]))
            tag_mask[start:end] = [1.0] * (end - start)
            label_masks[name][start:end] = [1.0] * (end - start)
        return ids, tag_mask, label_masks

    @staticmethod
    def _encode_ids(tokenizer: TokenizerWrapper, text: str) -> list[int]:
        return list(tokenizer.encode_to_ids(text, add_special_tokens=False))
