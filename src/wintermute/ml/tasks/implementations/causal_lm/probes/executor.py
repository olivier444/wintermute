from __future__ import annotations

import torch.nn as nn

from wintermute.ml.tasks.implementations.causal_lm.generation import CausalLmGenerator
from wintermute.ml.tasks.implementations.causal_lm.probes.base import (
    ProbeReport,
    ProbeResult,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.evaluation import (
    aggregate_probe_metrics,
    evaluate_probe,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.suite import (
    ProbeExecutionPlan,
)
from wintermute.ml.tokenization.chat_format import ChatFormat, ConversationContext
from wintermute.ml.tokenization.wrapper import TokenizerWrapper


class CausalLmProbeExecutor:
    def __init__(
        self,
        *,
        tokenizer: TokenizerWrapper,
        chat_format: ChatFormat,
        generator: CausalLmGenerator,
    ) -> None:
        self.tokenizer = tokenizer
        self.chat_format = chat_format
        self.generator = generator

    def run(
        self,
        model: nn.Module,
        plan: ProbeExecutionPlan,
    ) -> ProbeReport:
        results = []
        suite = plan.suite

        for probe in suite.probes:
            use_chat_format = probe.assistant_prompt_format is not None
            rendered_prompt = (
                self.chat_format.render_prompt(
                    ConversationContext(current_user=probe.prompt),
                    probe.assistant_prompt_format,
                ).text
                if use_chat_format
                else probe.prompt
            )
            prompt_tokens = list(
                self.tokenizer.encode_to_ids(
                    rendered_prompt,
                    add_special_tokens=False,
                )
            )
            observable_candidates = ()
            metric_values: dict[str, float] = {}
            for sampling_group in plan.sampling_groups:
                if (
                    probe.verifier is None
                    and sampling_group.observable_sample_count == 0
                ):
                    continue

                sampling = sampling_group.sampling
                sample_count = (
                    sampling.sample_count
                    if probe.verifier is not None
                    else sampling_group.observable_sample_count
                )
                completions = tuple(
                    self.generator.decode(
                        self.generator.generate(
                            prompt_tokens,
                            model,
                            greedy=sampling.greedy,
                            max_new_tokens=probe.max_new_tokens,
                            temperature=sampling.temperature,
                            top_p=sampling.top_p,
                            top_k=sampling.top_k,
                            repetition_penalty=1.0,
                        ),
                        assistant_prompt_format=probe.assistant_prompt_format,
                        use_chat_format=use_chat_format,
                    )
                    for _ in range(sample_count)
                )
                group_result = evaluate_probe(
                    probe,
                    completions,
                    sampling_group.metrics,
                )
                metric_values.update(group_result.metrics)
                if sampling_group.observable_sample_count > 0:
                    observable_candidates = group_result.candidates[
                        :sampling_group.observable_sample_count
                    ]

            results.append(
                ProbeResult(
                    probe_id=probe.id,
                    candidates=observable_candidates,
                    metrics=metric_values,
                )
            )

        probe_results = tuple(results)
        return ProbeReport(
            suite_name=suite.name,
            results=probe_results,
            metrics=aggregate_probe_metrics(probe_results),
        )
