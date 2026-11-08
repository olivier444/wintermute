from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from contextlib import nullcontext
from typing import ContextManager, Protocol

import torch

from wintermute.data.iterate.feedback import (
    FeedbackExample,
    FeedbackPrompt,
    FeedbackTrainingBatch,
    GeneratedCompletion,
    ScoredCompletion,
    ScoreKind,
)
from wintermute.ml.tokenization.chat_format import ConversationContext
from wintermute.tools.model import iter_batches


class FeedbackCandidateGenerator(Protocol):
    def generate(
        self,
        contexts: Sequence[ConversationContext],
        *,
        candidates_per_prompt: int,
    ) -> Sequence[Sequence[GeneratedCompletion]]: ...


class FeedbackEvaluator(Protocol):
    score_kind: ScoreKind

    def score(
        self,
        prompts: Sequence[FeedbackPrompt],
        completions: Sequence[Sequence[str]],
    ) -> Sequence[Sequence[float]]: ...


class OnlineFeedbackBatchProducer:
    """Generate and score candidates, keeping batching outside the view."""

    def __init__(
        self,
        *,
        prompt_batch_size: int,
        candidates_per_prompt: int,
        generator: FeedbackCandidateGenerator,
        evaluator: FeedbackEvaluator,
        evaluation_phase: Callable[[], ContextManager[None]] | None = None,
    ) -> None:
        if prompt_batch_size <= 0:
            raise ValueError("prompt_batch_size must be positive")
        if candidates_per_prompt < 2:
            raise ValueError("candidates_per_prompt must be at least two")

        self.prompt_batch_size = prompt_batch_size
        self.candidates_per_prompt = candidates_per_prompt
        self.generator = generator
        self.evaluator = evaluator
        self.evaluation_phase = evaluation_phase

    def iter_batches(
        self,
        source: Iterable[FeedbackPrompt],
    ) -> Iterator[FeedbackTrainingBatch]:
        for prompts in iter_batches(source, self.prompt_batch_size):
            # The generator never receives the current reference completion.
            contexts = tuple(prompt.context for prompt in prompts)
            with torch.inference_mode():
                generated = self.generator.generate(
                    contexts,
                    candidates_per_prompt=self.candidates_per_prompt,
                )
            if len(generated) != len(prompts):
                raise ValueError("generator returned the wrong number of prompt groups")
            candidates = tuple(tuple(group) for group in generated)
            if any(len(group) != self.candidates_per_prompt for group in candidates):
                raise ValueError("generator returned the wrong number of candidates")
            completions = tuple(
                tuple(candidate.completion for candidate in group)
                for group in candidates
            )

            phase = self.evaluation_phase() if self.evaluation_phase else nullcontext()
            with phase:
                with torch.inference_mode():
                    scores = self.evaluator.score(prompts, completions)
                if len(scores) != len(prompts):
                    raise ValueError("evaluator returned the wrong number of prompt groups")
                if any(len(group) != self.candidates_per_prompt for group in scores):
                    raise ValueError("evaluator returned the wrong number of scores")

                batch = FeedbackTrainingBatch(
                    examples=tuple(
                        FeedbackExample(
                            prompt=prompt,
                            score_kind=self.evaluator.score_kind,
                            completions=tuple(
                                ScoredCompletion(
                                    completion=candidate.completion,
                                    score=float(score),
                                    trace=candidate.trace,
                                )
                                for candidate, score in zip(group, score_group)
                            ),
                        )
                        for prompt, group, score_group in zip(prompts, candidates, scores)
                    )
                )
            yield batch
