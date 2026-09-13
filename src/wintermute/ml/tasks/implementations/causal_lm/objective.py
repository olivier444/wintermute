from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Mapping, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from wintermute.data.iterate.dataclasses import TrainingExample
from wintermute.data.iterate.segment_metadata import (
    SEGMENT_COMPLETION_LEN,
    SEGMENT_FINAL_LEN,
    SEGMENT_FINAL_START,
    SEGMENT_PROMPT_LEN,
    SEGMENT_WINDOW_END_EXCLUDED,
    SEGMENT_WINDOW_START,
)
from wintermute.data.iterate.training_view import (
    AbstractTrainingView,
    TrainingViewIterationStats,
)
from wintermute.ml.tasks.implementations.causal_lm.config import (
    CompletionPositionWeighting,
    CompletionPrefixMasking,
)
from wintermute.ml.tokenization.wrapper import TOK_MSK, TokenizerWrapper
from wintermute.tools.model import extract_logits_from_model_output


@dataclass(frozen=True)
class _EvalBatchStats:
    objective_loss_sum: float
    objective_weight: float
    unweighted_loss_sum: float
    tokens: int
    bytes_count: int
    top10_hit_count: int
    top10_valid_count: int
    label_exact_counts: dict[str, tuple[int, int]]


@dataclass
class _PositionBpbStats:
    loss_sum: float = 0.0
    bytes_count: int = 0


class _CompletionPositionWeightedView(AbstractTrainingView):
    def __init__(
        self,
        wrapped: AbstractTrainingView,
        objective: CausalLmObjective,
    ) -> None:
        self.wrapped = wrapped
        self.objective = objective
        self.name = wrapped.name
        self.max_example_len = wrapped.max_example_len
        self.tokenizer = wrapped.tokenizer

    @property
    def last_iteration_stats(self) -> TrainingViewIterationStats:
        return self.wrapped.last_iteration_stats

    @property
    def completed_iteration_count(self) -> int:
        return self.wrapped.completed_iteration_count

    def __iter__(self) -> Iterator[TrainingExample]:
        return iter(self.objective._apply_completion_position_weighting(self.wrapped))


class CausalLmObjective:
    def __init__(
        self,
        *,
        device: torch.device,
        bf16_enabled: bool,
        tokenizer: TokenizerWrapper,
        eval_position_boundaries: tuple[int, ...],
        completion_position_weighting: CompletionPositionWeighting | None,
        completion_prefix_masking: CompletionPrefixMasking | None,
    ) -> None:
        self.device = device
        self.bf16_enabled = bf16_enabled
        self.tokenizer = tokenizer
        self.eval_position_boundaries = eval_position_boundaries
        self.completion_position_weighting = completion_position_weighting
        self.completion_prefix_masking = completion_prefix_masking
        self.mask_token_id = self._resolve_mask_token_id(completion_prefix_masking)

    def wrap_training_view(
        self,
        view: AbstractTrainingView,
    ) -> AbstractTrainingView:
        if self.completion_position_weighting is None:
            return view
        return _CompletionPositionWeightedView(view, self)

    def compute_loss(
        self,
        model: nn.Module,
        examples: list[TrainingExample],
    ) -> torch.Tensor:
        return self._compute_loss_impl(
            model,
            examples,
            apply_completion_prefix_masking=True,
        )[0]

    def compute_eval_metrics(
        self,
        model: nn.Module,
        batches: Iterable[list[TrainingExample]],
    ) -> dict[str, float]:
        batch_stats: list[_EvalBatchStats] = []
        position_stats = {
            self._position_metric_name(start, end): _PositionBpbStats()
            for start, end, _ in self._position_bands(
                [],
                self.eval_position_boundaries,
            )
        }
        minimum_position_count = (
            self.eval_position_boundaries[-1] + 1
            if self.eval_position_boundaries
            else 1
        )

        for batch in batches:
            (
                loss_t,
                inputs_t,
                mask_t,
                weights_t,
                logits_t,
                token_losses_t,
            ) = self._compute_loss_impl(model, batch)
            target_mask = mask_t[:, 1:]
            objective_weight = float(target_mask.sum().item())
            if weights_t is not None:
                objective_weight = float(
                    (target_mask * weights_t[:, 1:]).sum().item()
                )
            batch_tokens = int(target_mask.sum().item())
            unweighted_loss_t = self._masked_mean(token_losses_t, target_mask)
            label_exact_counts: dict[str, tuple[int, int]] = {}
            for label_name in ("task", "verbosity", "format"):
                mask_key = f"{label_name}_label_mask"
                if any(mask_key in example.payload for example in batch):
                    label_mask = self._compute_label_mask(
                        batch,
                        inputs_t.shape[1],
                        mask_key,
                    ).to(self.device)
                    label_exact_counts[label_name] = self._label_exact_count(
                        logits_t,
                        inputs_t,
                        target_mask,
                        label_mask[:, 1:],
                    )

            sub_inputs_t = inputs_t[mask_t.bool()]
            sub_token_ids = cast(list[int], sub_inputs_t.detach().cpu().tolist())
            sub_str = self.tokenizer.decode(
                sub_token_ids,
                skip_special_tokens=True,
            )
            top10_hits, top10_valid = self._top_k_count(
                logits_t,
                inputs_t,
                mask_t[:, 1:],
                10,
            )

            token_losses_cpu = token_losses_t.detach().cpu()
            for example_index, example in enumerate(batch):
                input_ids = self._example_input_ids(example)
                for positions in self._eval_position_sequences(example):
                    if len(positions) < minimum_position_count:
                        continue
                    for start, end, band_positions in self._position_bands(
                        positions,
                        self.eval_position_boundaries,
                    ):
                        name = self._position_metric_name(start, end)
                        target_indices = [position - 1 for position in band_positions]
                        position_stats[name].loss_sum += float(
                            token_losses_cpu[
                                example_index,
                                target_indices,
                            ].sum().item()
                        )
                        decoded = self.tokenizer.decode(
                            [input_ids[position] for position in band_positions],
                            skip_special_tokens=True,
                        )
                        position_stats[name].bytes_count += len(decoded.encode("utf-8"))

            batch_stats.append(
                _EvalBatchStats(
                    objective_loss_sum=float(loss_t.item()) * objective_weight,
                    objective_weight=objective_weight,
                    unweighted_loss_sum=float(unweighted_loss_t.item()) * batch_tokens,
                    tokens=batch_tokens,
                    bytes_count=len(sub_str.encode("utf-8")),
                    top10_hit_count=top10_hits,
                    top10_valid_count=top10_valid,
                    label_exact_counts=label_exact_counts,
                )
            )

        total_objective_loss = sum(item.objective_loss_sum for item in batch_stats)
        total_objective_weight = sum(item.objective_weight for item in batch_stats)
        total_unweighted_loss = sum(item.unweighted_loss_sum for item in batch_stats)
        tokens_count = sum(item.tokens for item in batch_stats)
        bytes_count = sum(item.bytes_count for item in batch_stats)
        top10_hit_count = sum(item.top10_hit_count for item in batch_stats)
        top10_valid_count = sum(item.top10_valid_count for item in batch_stats)

        shard_losses = self._compute_eval_shard_losses(batch_stats, shard_count=10)
        objective_loss = total_objective_loss / max(1.0, total_objective_weight)
        loss_per_token = total_unweighted_loss / max(1, tokens_count)
        loss_per_byte = loss_per_token * tokens_count / max(1, bytes_count)
        bpb = -math.log2(math.exp(-loss_per_byte))

        metrics = {
            "loss": objective_loss,
            "unweighted_loss": loss_per_token,
            "loss_sem": self._compute_standard_error(shard_losses),
            "bpb": bpb,
            "top10_acc": top10_hit_count / max(1, top10_valid_count),
        }
        for name, stats in position_stats.items():
            if stats.bytes_count > 0:
                position_loss_per_byte = stats.loss_sum / stats.bytes_count
                metrics[name] = -math.log2(math.exp(-position_loss_per_byte))
        for label_name in ("task", "verbosity", "format"):
            hit_count = sum(
                item.label_exact_counts.get(label_name, (0, 0))[0]
                for item in batch_stats
            )
            label_count = sum(
                item.label_exact_counts.get(label_name, (0, 0))[1]
                for item in batch_stats
            )
            if label_count > 0:
                metrics[f"{label_name}_label_acc"] = hit_count / label_count
                metrics[f"{label_name}_label_count"] = float(label_count)
        return metrics

    def _resolve_mask_token_id(
        self,
        masking: CompletionPrefixMasking | None,
    ) -> int | None:
        if masking is None:
            return None
        return cast(int, self.tokenizer.token_to_id(TOK_MSK))

    def _apply_completion_position_weighting(
        self,
        examples: Iterable[TrainingExample],
    ) -> Iterable[TrainingExample]:
        weighting = self.completion_position_weighting
        if weighting is None:
            return examples

        def weighted_examples() -> Iterator[TrainingExample]:
            for example in examples:
                completion_positions = list(
                    self._completion_position_sequences(example)
                )
                if completion_positions:
                    weights = list(self._example_loss_weights(example))
                    for positions in completion_positions:
                        for band_index, (_, _, band_positions) in enumerate(
                            self._position_bands(positions, weighting.boundaries)
                        ):
                            weight = weighting.weights[band_index]
                            for token_index in band_positions:
                                weights[token_index] *= weight
                    example.payload["loss_weights"] = weights
                yield example

        return weighted_examples()

    def _completion_position_sequences(
        self,
        example: TrainingExample,
    ) -> Iterable[list[int]]:
        loss_mask = self._example_loss_mask(example)
        for offset, _, segment in self._iter_segments(example):
            positions = self._completion_positions(offset, segment, loss_mask)
            if positions is not None:
                yield positions

    def _eval_position_sequences(
        self,
        example: TrainingExample,
    ) -> Iterable[list[int]]:
        loss_mask = self._example_loss_mask(example)

        for offset, length, segment in self._iter_segments(example):
            completion_positions = self._completion_positions(
                offset,
                segment,
                loss_mask,
            )
            if completion_positions is not None:
                yield completion_positions
                continue
            yield [
                index
                for index in range(offset, offset + length)
                if index > 0 and loss_mask[index]
            ]

    @staticmethod
    def _completion_positions(
        offset: int,
        segment: Mapping[str, Any],
        loss_mask: list[float],
    ) -> list[int] | None:
        if SEGMENT_FINAL_START not in segment:
            return None
        start = offset + int(segment[SEGMENT_FINAL_START])
        end = start + int(segment[SEGMENT_FINAL_LEN])
        return [index for index in range(start, end) if loss_mask[index]]

    @staticmethod
    def _iter_segments(
        example: TrainingExample,
    ) -> Iterable[tuple[int, int, Mapping[str, Any]]]:
        offset = 0
        segments = cast(
            list[Mapping[str, Any]],
            example.metadata.get("segments", []),
        )
        for segment in segments:
            if SEGMENT_PROMPT_LEN in segment:
                length = int(segment[SEGMENT_PROMPT_LEN]) + int(
                    segment[SEGMENT_COMPLETION_LEN]
                )
            else:
                length = int(segment[SEGMENT_WINDOW_END_EXCLUDED]) - int(
                    segment[SEGMENT_WINDOW_START]
                )
            yield offset, length, segment
            offset += length

    @staticmethod
    def _position_bands(
        positions: list[int],
        boundaries: tuple[int, ...],
    ) -> Iterable[tuple[int, int | None, list[int]]]:
        start = 0
        for boundary in boundaries:
            yield start + 1, boundary, positions[start:boundary]
            start = boundary
        yield start + 1, None, positions[start:]

    @staticmethod
    def _position_metric_name(start: int, end: int | None) -> str:
        suffix = f"{start}_{end}" if end is not None else f"{start}_plus"
        return f"bpb_pos_{suffix}"

    def _compute_loss_impl(
        self,
        model: nn.Module,
        examples: list[TrainingExample],
        *,
        apply_completion_prefix_masking: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        input_ids = self._extract_input_tensor(examples).to(self.device)
        model_input_ids = input_ids
        if (
            apply_completion_prefix_masking
            and self.completion_prefix_masking is not None
        ):
            model_input_ids = self._apply_completion_prefix_masking(
                input_ids,
                examples,
                self.completion_prefix_masking,
            )
        logits = self._invoke_model(model, model_input_ids)

        loss_mask = self._compute_loss_mask(examples, input_ids.shape[1]).to(
            self.device
        )
        target_mask = loss_mask[:, 1:]
        loss_weights = None
        effective_target_mask = target_mask
        if any("loss_weights" in example.payload for example in examples):
            loss_weights = self._compute_loss_weights(
                examples,
                input_ids.shape[1],
            ).to(self.device)
            effective_target_mask = target_mask * loss_weights[:, 1:]

        token_losses = self._compute_token_losses(logits, input_ids)
        loss = self._masked_mean(token_losses, effective_target_mask)
        return loss, input_ids, loss_mask, loss_weights, logits, token_losses

    def _apply_completion_prefix_masking(
        self,
        input_ids: torch.Tensor,
        examples: list[TrainingExample],
        masking: CompletionPrefixMasking,
    ) -> torch.Tensor:
        if masking.probability == 0:
            return input_ids

        masked_input_ids = input_ids.clone()
        for row_index, example in enumerate(examples):
            for positions in self._completion_position_sequences(example):
                max_tokens = min(masking.max_tokens, len(positions))
                if max_tokens == 0:
                    continue
                if masking.probability < 1 and float(
                    torch.rand((), device=input_ids.device).item()
                ) >= masking.probability:
                    continue

                token_count = int(
                    torch.randint(
                        1,
                        max_tokens + 1,
                        (),
                        device=input_ids.device,
                    ).item()
                )
                masked_input_ids[row_index, positions[:token_count]] = (
                    cast(int, self.mask_token_id)
                )

        return masked_input_ids

    def _top_k_count(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
        k: int,
    ) -> tuple[int, int]:
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:]

        topk_indices = torch.topk(logits, k=k, dim=-1).indices
        hits = (topk_indices == targets.unsqueeze(-1)).any(dim=-1)

        valid_mask = (targets != self.tokenizer.pad_id) & target_mask.bool()
        hits = hits & valid_mask

        return int(hits.sum().item()), int(valid_mask.sum().item())

    @staticmethod
    def _label_exact_count(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor,
        label_mask: torch.Tensor,
    ) -> tuple[int, int]:
        predictions = logits[:, :-1, :].argmax(dim=-1)
        targets = input_ids[:, 1:]
        correct_rows = (predictions == targets).detach().cpu().tolist()
        label_rows = (label_mask.bool() & target_mask.bool()).detach().cpu().tolist()

        hit_count = 0
        label_count = 0
        for correct_row, label_row in zip(correct_rows, label_rows):
            label_is_correct = True
            inside_label = False
            for is_correct, is_label in zip(correct_row, label_row):
                if is_label:
                    inside_label = True
                    label_is_correct = label_is_correct and bool(is_correct)
                elif inside_label:
                    label_count += 1
                    hit_count += int(label_is_correct)
                    inside_label = False
                    label_is_correct = True
            if inside_label:
                label_count += 1
                hit_count += int(label_is_correct)

        return hit_count, label_count

    @staticmethod
    def _compute_eval_shard_losses(
        batch_stats: list[_EvalBatchStats],
        *,
        shard_count: int,
    ) -> list[float]:
        if shard_count <= 0:
            raise ValueError("shard_count must be > 0")
        if not batch_stats:
            return []

        used_shard_count = min(shard_count, len(batch_stats))
        shard_losses: list[float] = []

        for shard_index in range(used_shard_count):
            start = (shard_index * len(batch_stats)) // used_shard_count
            end = ((shard_index + 1) * len(batch_stats)) // used_shard_count
            shard = batch_stats[start:end]
            shard_weight = sum(item.objective_weight for item in shard)
            shard_loss = sum(item.objective_loss_sum for item in shard) / max(
                1.0,
                shard_weight,
            )
            shard_losses.append(shard_loss)

        return shard_losses

    @staticmethod
    def _compute_standard_error(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (
            len(values) - 1
        )
        return math.sqrt(variance / len(values))

    def _invoke_model(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.bf16_enabled,
        ):
            return extract_logits_from_model_output(model(input_ids=input_ids))

    def _compute_token_losses(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits[:, :-1, :].contiguous()
        targets = input_ids[:, 1:].clone()
        targets[targets == self.tokenizer.pad_id] = -100

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view_as(targets)

    @staticmethod
    def _masked_mean(
        loss: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        loss = loss * target_mask
        denominator = torch.clamp_min(target_mask.sum(), 1)
        return loss.sum() / denominator

    def _extract_input_tensor(
        self,
        examples: list[TrainingExample],
    ) -> torch.Tensor:
        max_len = max(len(self._example_input_ids(example)) for example in examples)
        input_ids: list[list[int]] = []
        for example in examples:
            example_input_ids = self._example_input_ids(example)
            pad_len = max_len - len(example_input_ids)
            input_ids.append(
                example_input_ids + [self.tokenizer.pad_id] * pad_len
            )
        return torch.tensor(input_ids, dtype=torch.long)

    def _compute_loss_mask(
        self,
        examples: list[TrainingExample],
        max_len: int,
    ) -> torch.Tensor:
        loss_mask: list[list[float]] = []
        for example in examples:
            example_loss_mask = self._example_loss_mask(example)
            pad_len = max_len - len(example_loss_mask)
            loss_mask.append(
                [float(value) for value in example_loss_mask] + [0.0] * pad_len
            )
        return torch.tensor(loss_mask, dtype=torch.float32)

    def _compute_loss_weights(
        self,
        examples: list[TrainingExample],
        max_len: int,
    ) -> torch.Tensor:
        loss_weights: list[list[float]] = []
        for example in examples:
            example_loss_weights = self._example_loss_weights(example)
            pad_len = max_len - len(example_loss_weights)
            loss_weights.append(example_loss_weights + [1.0] * pad_len)
        return torch.tensor(loss_weights, dtype=torch.float32)

    def _compute_label_mask(
        self,
        examples: list[TrainingExample],
        max_len: int,
        key: str,
    ) -> torch.Tensor:
        label_masks: list[list[float]] = []
        for example in examples:
            label_mask = self._example_label_mask(example, key)
            pad_len = max_len - len(label_mask)
            label_masks.append(label_mask + [0.0] * pad_len)
        return torch.tensor(label_masks, dtype=torch.float32)

    @staticmethod
    def _example_input_ids(example: TrainingExample) -> list[int]:
        return cast(list[int], example.payload["input_ids"])

    @staticmethod
    def _example_loss_mask(example: TrainingExample) -> list[float]:
        return cast(list[float], example.payload["loss_mask"])

    def _example_loss_weights(self, example: TrainingExample) -> list[float]:
        weights = example.payload.get("loss_weights")
        if weights is not None:
            return cast(list[float], weights)
        return [1.0] * len(self._example_input_ids(example))

    def _example_label_mask(
        self,
        example: TrainingExample,
        key: str,
    ) -> list[float]:
        mask = example.payload.get(key)
        if mask is not None:
            return cast(list[float], mask)
        return [0.0] * len(self._example_input_ids(example))
