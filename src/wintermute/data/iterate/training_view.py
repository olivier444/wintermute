from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import Random
from typing import Iterator, List, Optional, cast

from wintermute.data.iterate.completion_format import CompletionFormatRule
from wintermute.data.iterate.dataclasses import (
    CompositeTrainingViewConfig,
    CompositeTrainingSource,
    PackingPolicy,
    SnapshotTrainingViewConfig,
    TrainingExample,
    TrainingViewConfig,
    WindowingPolicy,
)
from wintermute.data.iterate.segment_metadata import (
    SEGMENT_COMPONENT_NAME,
    SEGMENT_COMPLETION_LEN,
    SEGMENT_DOC_ID,
    SEGMENT_FINAL_LEN,
    SEGMENT_FINAL_START,
    SEGMENT_HAD_HISTORY,
    SEGMENT_HAD_SCRATCHPAD,
    SEGMENT_HAD_SYSTEM_PROMPT,
    SEGMENT_HISTORY_TURNS_DROPPED,
    SEGMENT_KIND,
    SEGMENT_KIND_PROMPT_COMPLETION,
    SEGMENT_PROMPT_LEN,
    SEGMENT_SYSTEM_PROMPT_DROPPED,
    SEGMENT_USED_SCRATCHPAD,
    SEGMENT_WINDOW_START,
)
from wintermute.data.iterate.windowing import build_example_from_window, iter_windows
from wintermute.data.constants import (
    FLD_GENERIC_COMPLETION,
    FLD_GENERIC_PROMPT,
    FLD_GENERIC_SCRATCHPAD,
    FLD_GENERIC_SYSTEM_PROMPT,
    FLD_GENERIC_TASK,
    FLD_GENERIC_TEXT,
)
from wintermute.data.iterate.conversation_encoding import TokenizedConversation
from wintermute.data.iterate.transformed_materialized_view import TransformedMaterializedView
from wintermute.data.transform import RecordTransform, SequentialTransform
from wintermute.data.transform.builder import build_record_transform
from wintermute.data.transform.context import DEFAULT_RUNTIME_TRANSFORM_SEED
from wintermute.data.transform.implementations.normalize_single_turn_sft import NormalizeSingleTurnSftRecord
from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.data.snapshot.materialized_iterable import (
    AbstractMaterializedView,
    DataRecord,
    MaterializedView,
)
from wintermute.ml.tokenization.chat_format import (
    ChatFormat,
    ConversationContext,
    ConversationTurn,
)
from wintermute.tools.logging import console_log
from wintermute.tools.misc import batch_iter


@dataclass
class TrainingViewIterationStats:
    sft_candidates: int = 0
    sft_emitted: int = 0
    sft_rejected: int = 0
    sft_examples_with_scratchpad: int = 0
    sft_examples_with_system_prompt: int = 0
    sft_examples_with_history: int = 0
    sft_scratchpad_removed: int = 0
    sft_history_reduced: int = 0
    sft_history_turns_dropped_total: int = 0
    sft_system_prompt_dropped: int = 0
    sft_reject_required_turn_too_long: int = 0
    sft_reject_invalid_example: int = 0

    def merge(self, other: "TrainingViewIterationStats") -> None:
        self.sft_candidates += other.sft_candidates
        self.sft_emitted += other.sft_emitted
        self.sft_rejected += other.sft_rejected
        self.sft_examples_with_scratchpad += other.sft_examples_with_scratchpad
        self.sft_examples_with_system_prompt += other.sft_examples_with_system_prompt
        self.sft_examples_with_history += other.sft_examples_with_history
        self.sft_scratchpad_removed += other.sft_scratchpad_removed
        self.sft_history_reduced += other.sft_history_reduced
        self.sft_history_turns_dropped_total += other.sft_history_turns_dropped_total
        self.sft_system_prompt_dropped += other.sft_system_prompt_dropped
        self.sft_reject_required_turn_too_long += other.sft_reject_required_turn_too_long
        self.sft_reject_invalid_example += other.sft_reject_invalid_example

    def copy(self) -> "TrainingViewIterationStats":
        clone = TrainingViewIterationStats()
        clone.merge(self)
        return clone

    def has_sft_activity(self) -> bool:
        return self.sft_candidates > 0

    def describe(self) -> str:
        return (
            f"candidates={self.sft_candidates}, emitted={self.sft_emitted}, rejected={self.sft_rejected}, "
            f"with_scratchpad={self.sft_examples_with_scratchpad}, with_system_prompt={self.sft_examples_with_system_prompt}, "
            f"with_history={self.sft_examples_with_history}, scratchpad_removed={self.sft_scratchpad_removed}, "
            f"history_reduced={self.sft_history_reduced}, history_turns_dropped_total={self.sft_history_turns_dropped_total}, "
            f"system_prompt_dropped={self.sft_system_prompt_dropped}, "
            f"reject_required_turn_too_long={self.sft_reject_required_turn_too_long}, "
            f"reject_invalid_example={self.sft_reject_invalid_example}"
        )


class AbstractTrainingView(ABC):
    name: str
    max_example_len: int
    tokenizer: TokenizerWrapper
    last_iteration_stats: TrainingViewIterationStats
    completed_iteration_count: int

    @abstractmethod
    def __iter__(self) -> Iterator[TrainingExample]: ...


class TrainingView(AbstractTrainingView):

    def __init__(
            self,
            tokenizer: TokenizerWrapper,
            chat_format: ChatFormat,
            training_view_config: SnapshotTrainingViewConfig,
            windowing_policy: WindowingPolicy,
            output_root: str,
            use_task_labels: bool = False,
            tag_loss_weight: float = 1.0,
            verbosity_labels: tuple[tuple[str, int | None], ...] | None = None,
            format_labels: tuple[CompletionFormatRule, ...] | None = None,
            transform_seed: int = DEFAULT_RUNTIME_TRANSFORM_SEED,
            silent: bool = False
        ) -> None:
        self.silent = silent

        self.training_view_config = training_view_config
        self.windowing_policy = windowing_policy
        self.max_example_len = windowing_policy.window_max_len
        self.tokenizer = tokenizer
        self.chat_format = chat_format
        self.include_prompt_in_loss = training_view_config.include_prompt_in_loss
        self.use_task_labels = use_task_labels
        self.tag_loss_weight = tag_loss_weight
        self.verbosity_labels = verbosity_labels
        self.format_labels = format_labels
        materialized_view: AbstractMaterializedView = MaterializedView(
            output_root,
            self.training_view_config.materialized_config,
        )
        runtime_transform: RecordTransform = NormalizeSingleTurnSftRecord()
        transform_config = self.training_view_config.record_transform
        if transform_config is not None:
            runtime_transform = SequentialTransform(
                [build_record_transform(transform_config), runtime_transform]
            )
        self.materialized_view = TransformedMaterializedView(
            materialized_view,
            runtime_transform,
            seed=transform_seed,
        )
        self.name = self.materialized_view.name
        self.last_iteration_stats = TrainingViewIterationStats()
        self.completed_iteration_count = 0

        if not self.silent:
            console_log(
                "training-view",
                (
                    f"[{self.name}] building adaptive training view "
                    f"(max_seq_len={self.max_example_len}, eos={self.windowing_policy.add_eos_at_end_of_doc}, "
                    f"include_prompt_in_loss={self.include_prompt_in_loss}, "
                    f"record_transform={self.training_view_config.record_transform.kind if self.training_view_config.record_transform else None}, "
                    f"use_task_labels={self.use_task_labels}, "
                    f"tag_loss_weight={self.tag_loss_weight}, "
                    f"verbosity_labels={self.verbosity_labels}, "
                    f"format_labels={self.format_labels})"
                ),
            )

    def __iter__(self) -> Iterator[TrainingExample]:
        if not self.silent:
            console_log("training-view", f"[{self.name}] building iterator")
        token_count = 0
        skipped_count = 0
        stats = TrainingViewIterationStats()
        self.last_iteration_stats = stats

        try:
            for record in self.materialized_view:
                if FLD_GENERIC_TEXT in record.fields:
                    tokens = self._build_text_tokens(record)
                    token_count += len(tokens)
                    yield from self._iter_text_examples(record.record_id, tokens)
                    continue

                if f"{FLD_GENERIC_PROMPT}.0" in record.fields:
                    history_turns: List[ConversationTurn] = []
                    system_prompt = record.fields.get(FLD_GENERIC_SYSTEM_PROMPT)

                    index = 0
                    while True:
                        prompt = record.fields.get(f"{FLD_GENERIC_PROMPT}.{index}")
                        if prompt is None:
                            break

                        completion = record.fields[f"{FLD_GENERIC_COMPLETION}.{index}"]
                        task = (
                            record.fields.get(f"{FLD_GENERIC_TASK}.{index}")
                            if self.use_task_labels
                            else None
                        )
                        scratchpad = record.fields.get(f"{FLD_GENERIC_SCRATCHPAD}.{index}")

                        example = self._build_prompt_completion_example(
                            rec_id=f"{record.record_id}.{index}",
                            system_prompt=system_prompt,
                            history_turns=history_turns,
                            prompt=prompt,
                            completion=completion,
                            task=task,
                            scratchpad=scratchpad,
                            stats=stats,
                        )
                        history_turns.append(
                            ConversationTurn(
                                user=prompt,
                                assistant=completion,
                            )
                        )

                        if example is None:
                            skipped_count += 1
                        else:
                            token_count += len(self._input_ids(example))
                            yield example

                        index += 1

                    continue

                raise RuntimeError(
                    f"Materialized record '{record.record_id}' has unsupported schema: "
                    f"expected '{FLD_GENERIC_TEXT}' or '{FLD_GENERIC_PROMPT}/{FLD_GENERIC_COMPLETION}'."
                )
        finally:
            self.last_iteration_stats = stats
            self.completed_iteration_count += 1
            if not self.silent:
                msg = (
                    f"[{self.name}] closing iterator - {token_count:_} token served, {skipped_count:_} record skipped"
                )
                if stats.has_sft_activity():
                    msg = f"{msg}, sft: {stats.describe()}"
                console_log("training-view", msg)


    def _build_text_tokens(self, record: DataRecord) -> List[int]:
        rec_text = record.fields[FLD_GENERIC_TEXT]
        return self._build_text_tokens_from_string(rec_text)


    def _build_text_tokens_from_string(self, text: str) -> List[int]:
        tokens = self.tokenizer.encode_to_ids(text, add_special_tokens=False)

        if self.windowing_policy.add_eos_at_end_of_doc:
            tokens = list(tokens) + [self.tokenizer.eos_id]

        return tokens


    def _iter_text_examples(self, record_id: str, tokens: List[int]) -> Iterator[TrainingExample]:
        for wd in iter_windows(tokens, self.windowing_policy):
            yield build_example_from_window(
                rec_id=record_id,
                toks=tokens,
                wd=wd,
            )


    def _build_prompt_completion_example(
        self,
        *,
        rec_id: str,
        system_prompt: str | None,
        history_turns: List[ConversationTurn],
        prompt: str,
        completion: str,
        task: str | None,
        scratchpad: str | None,
        stats: TrainingViewIterationStats,
    ) -> Optional[TrainingExample]:
        stats.sft_candidates += 1
        if scratchpad:
            stats.sft_examples_with_scratchpad += 1
        if system_prompt:
            stats.sft_examples_with_system_prompt += 1
        if history_turns:
            stats.sft_examples_with_history += 1
        build_result = TokenizedConversation.from_context(
            tokenizer=self.tokenizer,
            chat_format=self.chat_format,
            context=ConversationContext(
                system=system_prompt,
                history=tuple(history_turns),
                current_user=prompt,
            ),
            current_assistant=completion,
            current_scratchpad=scratchpad,
            current_task=task,
            current_verbosity=self._completion_verbosity(completion),
            current_format=self._completion_format(completion),
            max_len=self.max_example_len,
        )
        if build_result.is_rejected:
            stats.sft_rejected += 1
            if build_result.rejection_reason == "required_turn_exceeds_max_example_len":
                stats.sft_reject_required_turn_too_long += 1
            return None

        if scratchpad and not build_result.used_scratchpad:
            stats.sft_scratchpad_removed += 1
        stats.sft_history_turns_dropped_total += build_result.dropped_history_turns
        if build_result.dropped_history_turns > 0:
            stats.sft_history_reduced += 1
        if build_result.dropped_system:
            stats.sft_system_prompt_dropped += 1

        prompt_ids = build_result.prompt_ids
        completion_ids = build_result.completion_ids
        input_ids = prompt_ids + completion_ids
        if len(input_ids) < 2:
            stats.sft_rejected += 1
            stats.sft_reject_invalid_example += 1
            return None

        if self.include_prompt_in_loss:
            loss_mask = [1.0] * len(input_ids)
        else:
            loss_mask = [0.0] * len(prompt_ids) + [1.0] * len(completion_ids)
        loss_mask[0] = 0.0

        work_units = int(sum(loss_mask))
        if work_units <= 0:
            stats.sft_rejected += 1
            stats.sft_reject_invalid_example += 1
            return None

        stats.sft_emitted += 1
        payload = {
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        }
        for label_name, label_mask in (
            ("task", build_result.task_label_mask),
            ("verbosity", build_result.verbosity_label_mask),
            ("format", build_result.format_label_mask),
        ):
            if any(label_mask):
                payload[f"{label_name}_label_mask"] = label_mask
        if self.tag_loss_weight != 1.0 and any(build_result.tag_loss_mask):
            payload["loss_weights"] = [
                self.tag_loss_weight if is_tag_value else 1.0
                for is_tag_value in build_result.tag_loss_mask
            ]

        return TrainingExample(
            uid=f"{rec_id}_sft",
            payload=payload,
            work_units=work_units,
            metadata={
                "segments": [
                    {
                        SEGMENT_DOC_ID: rec_id,
                        SEGMENT_KIND: SEGMENT_KIND_PROMPT_COMPLETION,
                        SEGMENT_PROMPT_LEN: len(prompt_ids),
                        SEGMENT_COMPLETION_LEN: len(completion_ids),
                        SEGMENT_FINAL_START: len(prompt_ids) + build_result.final_start,
                        SEGMENT_FINAL_LEN: build_result.final_len,
                        SEGMENT_HISTORY_TURNS_DROPPED: build_result.dropped_history_turns,
                        SEGMENT_HAD_HISTORY: len(history_turns) > 0,
                        SEGMENT_HAD_SCRATCHPAD: scratchpad is not None,
                        SEGMENT_USED_SCRATCHPAD: build_result.used_scratchpad,
                        SEGMENT_HAD_SYSTEM_PROMPT: system_prompt is not None,
                        SEGMENT_SYSTEM_PROMPT_DROPPED: build_result.dropped_system,
                    }
                ],
            },
        )

    def _completion_verbosity(self, completion: str) -> str | None:
        if self.verbosity_labels is None:
            return None

        token_count = self.tokenizer.count_non_special_tokens(completion)
        for label, max_tokens in self.verbosity_labels:
            if max_tokens is None or token_count <= max_tokens:
                return label
        return None


    def _completion_format(self, completion: str) -> str | None:
        if self.format_labels is None:
            return None

        for rule in self.format_labels:
            if rule.matches(completion):
                return rule.label
        return None


    def _input_ids(self, example: TrainingExample) -> List[int]:
        return cast(List[int], example.payload["input_ids"])


@dataclass
class _CompositeTrainingSourceState:
    source: CompositeTrainingSource
    view: AbstractTrainingView
    iterator: Iterator[TrainingExample]
    next_example: TrainingExample | None
    restart_count: int = 0
    served_examples: int = 0
    served_units: int = 0
    stats: TrainingViewIterationStats = field(default_factory=TrainingViewIterationStats)
    last_merged_iteration_count: int = 0

    @property
    def name(self) -> str:
        return self.source.display_name


class CompositeTrainingView(AbstractTrainingView):
    def __init__(
        self,
        *,
        tokenizer: TokenizerWrapper,
        chat_format: ChatFormat,
        output_root: str,
        config: CompositeTrainingViewConfig,
        windowing_policy: WindowingPolicy,
        use_task_labels: bool = False,
        tag_loss_weight: float = 1.0,
        verbosity_labels: tuple[tuple[str, int | None], ...] | None = None,
        format_labels: tuple[CompletionFormatRule, ...] | None = None,
        transform_seed: int | None = None,
        silent: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_example_len = windowing_policy.window_max_len
        self.silent = silent
        self.config = config
        self.name = config.display_name
        self.last_iteration_stats = TrainingViewIterationStats()
        self.completed_iteration_count = 0
        child_seed_rng = Random(config.seed if transform_seed is None else transform_seed)
        self.children = [
            _CompositeTrainingSourceState(
                source=source,
                view=build_base_training_view(
                    tokenizer=tokenizer,
                    chat_format=chat_format,
                    training_view_config=source.view_config,
                    windowing_policy=windowing_policy,
                    output_root=output_root,
                    use_task_labels=use_task_labels,
                    tag_loss_weight=tag_loss_weight,
                    verbosity_labels=verbosity_labels,
                    format_labels=format_labels,
                    transform_seed=child_seed_rng.getrandbits(64),
                    silent=silent,
                ),
                iterator=iter(()),
                next_example=None,
            )
            for source in config.sources
        ]

    def __iter__(self) -> Iterator[TrainingExample]:
        rng = Random(self.config.seed)
        runtime_states = [self._build_runtime_state(state) for state in self.children]

        live_stats = TrainingViewIterationStats()
        self.last_iteration_stats = live_stats

        try:
            while runtime_states:
                state = self._choose_state(runtime_states, rng)
                example = state.next_example
                assert example is not None

                state.served_examples += 1
                state.served_units += example.effective_units()
                state.next_example = self._advance_state(state)
                self._refresh_live_iteration_stats(runtime_states)
                yield self._annotate_example(example, state.name)

                if state.next_example is None:
                    console_log(
                        "training-view",
                        f"[{self.name}] source '{state.name}' exhausted after {state.served_examples:_} examples"
                        f" and {state.served_units:_} units",
                    )
                    console_log(
                        "training-view",
                        f"[{self.name}] stopping composite iterator because source '{state.name}' "
                        "has exhausted its last allowed pass",
                    )
                    break
        finally:
            for state in runtime_states:
                self._close_iterator(state.iterator)
                self._merge_child_iteration_stats(state)

            combined_stats = TrainingViewIterationStats()
            for state in runtime_states:
                combined_stats.merge(state.stats)
            self.last_iteration_stats = combined_stats
            self.completed_iteration_count += 1

            summary = ", ".join(
                (
                    f"{state.name}=examples:{state.served_examples},units:{state.served_units},"
                    f"restarts:{state.restart_count}"
                )
                for state in runtime_states
            )
            if combined_stats.has_sft_activity():
                summary = f"{summary}, sft: {combined_stats.describe()}"
            console_log("training-view", f"[{self.name}] source summary: {summary}")

    def _annotate_example(self, example: TrainingExample, component_name: str) -> TrainingExample:
        example.metadata[SEGMENT_COMPONENT_NAME] = component_name
        segments = cast(List[dict], example.metadata.get("segments", []))
        for segment in segments:
            segment.setdefault(SEGMENT_COMPONENT_NAME, component_name)
        return example

    def _build_runtime_state(self, state: _CompositeTrainingSourceState) -> _CompositeTrainingSourceState:
        iterator = iter(state.view)
        try:
            next_example = next(iterator)
        except StopIteration as exc:
            raise RuntimeError(f"Training view '{state.name}' is empty.") from exc

        return _CompositeTrainingSourceState(
            source=state.source,
            view=state.view,
            iterator=iterator,
            next_example=next_example,
        )

    def _advance_state(self, state: _CompositeTrainingSourceState) -> TrainingExample | None:
        try:
            return next(state.iterator)
        except StopIteration:
            self._merge_child_iteration_stats(state)
            if state.restart_count >= state.source.max_restarts:
                return None

            state.restart_count += 1
            console_log(
                "training-view",
                f"[{self.name}] source '{state.name}' exhausted - restarting iterator "
                f"#{state.restart_count}/{state.source.max_restarts}",
            )
            state.iterator = iter(state.view)
            try:
                return next(state.iterator)
            except StopIteration as exc:
                raise RuntimeError(f"Training view '{state.name}' is empty.") from exc

    def _choose_state(
        self,
        states: List[_CompositeTrainingSourceState],
        rng: Random,
    ) -> _CompositeTrainingSourceState:
        if len(states) == 1:
            return states[0]

        total_weight = sum(state.source.weight for state in states)
        if total_weight <= 0:
            raise RuntimeError(f"Composite training view '{self.name}' has no positive weights.")

        if self.config.mix_unit == "loss_tokens":
            total_served = sum(state.served_units for state in states)
            observed_values = {id(state): float(state.served_units) for state in states}
        else:
            total_served = sum(state.served_examples for state in states)
            observed_values = {id(state): float(state.served_examples) for state in states}

        if total_served <= 0:
            return self._choose_weighted_random_state(states, rng)

        best_states: List[_CompositeTrainingSourceState] = []
        best_deficit: float | None = None
        for state in states:
            target_share = state.source.weight / total_weight
            observed_share = observed_values[id(state)] / float(total_served)
            deficit = target_share - observed_share

            if best_deficit is None or deficit > best_deficit + 1e-12:
                best_deficit = deficit
                best_states = [state]
            elif abs(deficit - best_deficit) <= 1e-12:
                best_states.append(state)

        if len(best_states) == 1:
            return best_states[0]

        return best_states[rng.randrange(len(best_states))]

    def _choose_weighted_random_state(
        self,
        states: List[_CompositeTrainingSourceState],
        rng: Random,
    ) -> _CompositeTrainingSourceState:
        total_weight = sum(state.source.weight for state in states)
        target = rng.random() * total_weight
        cumulative = 0.0
        for state in states:
            cumulative += state.source.weight
            if target <= cumulative:
                return state
        return states[-1]

    def _merge_child_iteration_stats(self, state: _CompositeTrainingSourceState) -> None:
        completed_iteration_count = state.view.completed_iteration_count
        if completed_iteration_count <= state.last_merged_iteration_count:
            return
        state.stats.merge(state.view.last_iteration_stats)
        state.last_merged_iteration_count = completed_iteration_count

    def _refresh_live_iteration_stats(self, states: List[_CompositeTrainingSourceState]) -> None:
        combined = TrainingViewIterationStats()
        for state in states:
            combined.merge(state.stats)
            if state.view.completed_iteration_count == state.last_merged_iteration_count:
                combined.merge(state.view.last_iteration_stats)
        self.last_iteration_stats = combined

    def _close_iterator(self, iterator: Iterator[TrainingExample]) -> None:
        close = getattr(iterator, "close", None)
        if callable(close):
            close()


def build_base_training_view(
    *,
    tokenizer: TokenizerWrapper,
    chat_format: ChatFormat,
    training_view_config: TrainingViewConfig,
    windowing_policy: WindowingPolicy,
    output_root: str,
    use_task_labels: bool = False,
    tag_loss_weight: float = 1.0,
    verbosity_labels: tuple[tuple[str, int | None], ...] | None = None,
    format_labels: tuple[CompletionFormatRule, ...] | None = None,
    transform_seed: int | None = None,
    silent: bool = False,
) -> AbstractTrainingView:
    if isinstance(training_view_config, SnapshotTrainingViewConfig):
        return TrainingView(
            tokenizer=tokenizer,
            chat_format=chat_format,
            training_view_config=training_view_config,
            windowing_policy=windowing_policy,
            output_root=output_root,
            use_task_labels=use_task_labels,
            tag_loss_weight=tag_loss_weight,
            verbosity_labels=verbosity_labels,
            format_labels=format_labels,
            transform_seed=(
                DEFAULT_RUNTIME_TRANSFORM_SEED
                if transform_seed is None
                else transform_seed
            ),
            silent=silent,
        )

    if isinstance(training_view_config, CompositeTrainingViewConfig):
        return CompositeTrainingView(
            tokenizer=tokenizer,
            chat_format=chat_format,
            output_root=output_root,
            config=training_view_config,
            windowing_policy=windowing_policy,
            use_task_labels=use_task_labels,
            tag_loss_weight=tag_loss_weight,
            verbosity_labels=verbosity_labels,
            format_labels=format_labels,
            transform_seed=transform_seed,
            silent=silent,
        )

    raise TypeError(f"Unsupported training view config type: {type(training_view_config)!r}")


class PackedTrainingView(AbstractTrainingView):

    def __init__(
            self,
            wrapped: AbstractTrainingView,
            config: PackingPolicy,
            silent: bool = False
        ) -> None:

        self.silent = silent
        self.name = f"wrp.{wrapped.name}"

        if not self.silent:
            console_log("training-view", f"[{self.name}] building packed training view over {wrapped.name}")

        self.wrapped = wrapped
        self.config = config
        self.target_len = wrapped.max_example_len
        self.threshold = self.config.target_fill_ratio * self.target_len
        self.tokenizer = wrapped.tokenizer
        self.max_example_len = wrapped.max_example_len
        self.last_iteration_stats = TrainingViewIterationStats()
        self.completed_iteration_count = 0

        self.separator = self.wrapped.tokenizer.eos_id


    def __iter__(self) -> Iterator[TrainingExample]:
        if not self.silent:
            console_log("training-view", f"[{self.name}] building iterator")
        try:
            for batch in batch_iter(self.wrapped, self.config.example_buffer_size):
                self._pack_batch(batch)
                self.last_iteration_stats = self.wrapped.last_iteration_stats.copy()

                for example in batch:
                    if example is not None:
                        yield example
        finally:
            self.last_iteration_stats = self.wrapped.last_iteration_stats.copy()
            self.completed_iteration_count += 1
            if not self.silent:
                msg = f"[{self.name}] closing iterator"
                if self.last_iteration_stats.has_sft_activity():
                    msg = f"{msg}, sft: {self.last_iteration_stats.describe()}"
                console_log("training-view", msg)


    def _pack_batch(self, batch: List[Optional[TrainingExample]]) -> None:

        for i, example in enumerate(batch):
            if not example:
                continue

            example_input_ids = self._input_ids(example)
            if len(example_input_ids) < self.threshold and self._ends_at_document_boundary(example):
                docs = 1
                for offset_i, candidate in enumerate(batch[i+1:]):
                    if not candidate:
                        continue
                    if not self._starts_at_document_boundary(candidate):
                        continue

                    candidate_input_ids = self._input_ids(candidate)
                    s = None if example_input_ids[-1] == self.separator else self.separator
                    if len(candidate_input_ids) <= self.target_len - len(example_input_ids) - (0 if s is None else 1):
                        self._append_example(example, candidate, s)
                        example_input_ids = self._input_ids(example)
                        docs += 1
                        batch[i + 1 + offset_i] = None

                        if docs >= self.config.max_docs_per_pack or len(example_input_ids) >= self.threshold:
                            break


    def _append_example(self, target: TrainingExample, other: TrainingExample, separator: int | None) -> None:
        target_input_ids = self._input_ids(target)
        target_loss_mask = self._loss_mask(target)
        other_input_ids = self._input_ids(other)
        other_loss_mask = self._loss_mask(other)
        has_loss_weights = "loss_weights" in target.payload or "loss_weights" in other.payload
        target_loss_weights = self._loss_weights(target) if has_loss_weights else []
        other_loss_weights = self._loss_weights(other) if has_loss_weights else []
        label_mask_keys = tuple(
            key
            for key in ("task_label_mask", "verbosity_label_mask", "format_label_mask")
            if key in target.payload or key in other.payload
        )
        target_label_masks = {
            key: self._label_mask(target, key)
            for key in label_mask_keys
        }
        other_label_masks = {
            key: self._label_mask(other, key)
            for key in label_mask_keys
        }

        if separator is not None:
            target_input_ids.append(separator)
            target_loss_mask.append(1.0)
            if has_loss_weights:
                target_loss_weights.append(1.0)
            for mask in target_label_masks.values():
                mask.append(0.0)

        target_input_ids.extend(other_input_ids)
        target_loss_mask.extend(other_loss_mask)
        if has_loss_weights:
            target_loss_weights.extend(other_loss_weights)
            target.payload["loss_weights"] = target_loss_weights
        for key, mask in target_label_masks.items():
            mask.extend(other_label_masks[key])
            target.payload[key] = mask
        target.work_units = int(sum(target_loss_mask))

        target_segments = cast(List[dict], target.metadata.setdefault("segments", []))
        other_segments = cast(List[dict], other.metadata.get("segments", []))
        target_segments.extend(other_segments)


    def _input_ids(self, example: TrainingExample) -> List[int]:
        return cast(List[int], example.payload["input_ids"])


    def _loss_mask(self, example: TrainingExample) -> List[float]:
        return cast(List[float], example.payload["loss_mask"])

    def _loss_weights(self, example: TrainingExample) -> List[float]:
        weights = example.payload.get("loss_weights")
        if weights is not None:
            return list(cast(List[float], weights))
        return [1.0] * len(self._input_ids(example))

    def _label_mask(self, example: TrainingExample, key: str) -> List[float]:
        mask = example.payload.get(key)
        if mask is not None:
            return list(cast(List[float], mask))
        return [0.0] * len(self._input_ids(example))

    def _ends_at_document_boundary(self, example: TrainingExample) -> bool:
        input_ids = self._input_ids(example)
        return bool(input_ids) and input_ids[-1] == self.separator

    def _starts_at_document_boundary(self, example: TrainingExample) -> bool:
        segments = cast(List[dict], example.metadata.get("segments", []))
        if not segments:
            return False

        first = segments[0]
        if SEGMENT_WINDOW_START not in first:
            return True
        return int(first[SEGMENT_WINDOW_START]) == 0
