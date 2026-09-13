from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, cast

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from wintermute.data.constants import DIR_EXTERNAL_MODELS, DIR_HUGGING_FACE_MODELS
from wintermute.data.iterate.dataclasses import TrainingExample, TrainingViewConfig
from wintermute.data.iterate.training_view import (
    PackedTrainingView,
    build_base_training_view,
)
from wintermute.ml.runs import Run
from wintermute.ml.tasks.baseline import BaselineSuite
from wintermute.ml.tasks.factory import (
    InferenceSession,
    InferenceSessionState,
    TaskWrapper,
)
from wintermute.ml.tasks.implementations.causal_lm.baseline import CausalLmBaselineSuite
from wintermute.ml.tasks.implementations.causal_lm.config import CausalLmTaskConfig
from wintermute.ml.tasks.implementations.causal_lm.generation import CausalLmGenerator
from wintermute.ml.tasks.implementations.causal_lm.objective import CausalLmObjective
from wintermute.ml.tasks.implementations.causal_lm.probes.executor import (
    CausalLmProbeExecutor,
)
from wintermute.ml.tasks.implementations.causal_lm.probes.monitor import (
    CausalLmProbeMonitor,
)
from wintermute.ml.tasks.implementations.causal_lm.session import CausalLmInferenceSession
from wintermute.ml.tokenization.chat_format import (
    ChatFormat,
    PlainTextChatFormat,
    WintermuteChatFormat,
    build_hugging_face_chat_format,
)
from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.ml.training.logger import Logger
from wintermute.ml.training.monitor import TrainingMonitor
from wintermute.tools.files import get_tokenizer_file
from wintermute.tools.logging import console_log


class CausalLmTaskWrapper(TaskWrapper):
    _DEFAULT_INFER_MAX_NEW_TOKENS = 256
    _DEFAULT_INFER_TEMPERATURE = 0.7
    _DEFAULT_INFER_TOP_P = 0.85
    _DEFAULT_INFER_TOP_K: int | None = 50

    def __init__(
            self,
            *,
            device: torch.device,
            bf_16_enabled: bool,
            params: Dict[str, Any],
            root_path: str,
        ):
        super().__init__(device)
        cfg = CausalLmTaskConfig.from_dict(params)

        self.params = dict(params)
        self.root_path = root_path
        self.tokenizer, self.chat = self._load_tokenizer_and_chat_format(
            cfg,
            root_path,
        )
        self.packing_policy = cfg.packing_policy
        self.windowing_policy = cfg.windowing_policy
        self.use_task_labels = cfg.use_task_labels
        self.tag_loss_weight = cfg.tag_loss_weight
        self.verbosity_labels = cfg.verbosity_labels
        self.format_labels = cfg.format_labels
        self.bf_16_enabled = bf_16_enabled
        self.generator = CausalLmGenerator(
            device=device,
            bf16_enabled=bf_16_enabled,
            tokenizer=self.tokenizer,
            chat_format=self.chat,
        )
        self.objective = CausalLmObjective(
            device=device,
            bf16_enabled=bf_16_enabled,
            tokenizer=self.tokenizer,
            eval_position_boundaries=cfg.eval_position_boundaries,
            completion_position_weighting=cfg.completion_position_weighting,
            completion_prefix_masking=cfg.completion_prefix_masking,
        )


    def _load_tokenizer_and_chat_format(
        self,
        cfg: CausalLmTaskConfig,
        root_path: str,
    ) -> tuple[TokenizerWrapper, ChatFormat]:
        if cfg.tokenizer_id is not None:
            tokenizer_file = get_tokenizer_file(root_path, cfg.tokenizer_id)
            return (
                TokenizerWrapper.from_file(tokenizer_file),
                WintermuteChatFormat(assistant_format=cfg.assistant_format),
            )

        if cfg.tokenizer_file is not None:
            return (
                TokenizerWrapper.from_file(cfg.tokenizer_file),
                WintermuteChatFormat(assistant_format=cfg.assistant_format),
            )

        assert cfg.hf_tokenizer_name is not None
        cache_dir = str(Path(root_path) / DIR_EXTERNAL_MODELS / DIR_HUGGING_FACE_MODELS)
        hf_tokenizer = AutoTokenizer.from_pretrained(cfg.hf_tokenizer_name, cache_dir=cache_dir)
        chat_format = build_hugging_face_chat_format(hf_tokenizer)
        if isinstance(chat_format, PlainTextChatFormat):
            console_log(
                "chat-format",
                f"{cfg.hf_tokenizer_name} has no chat template; using plain-text completion fallback",
            )
        return (
            TokenizerWrapper.from_external(hf_tokenizer),
            chat_format,
        )

    def start_inference_session(
        self,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_state: InferenceSessionState | None = None,
    ) -> InferenceSession:
        return CausalLmInferenceSession(
            wrapper=self,
            model=model,
            system_prompt=system_prompt,
            initial_state=initial_state,
        )


    def build_training_view(
            self,
            output_root: str,
            training_view_config: TrainingViewConfig,
            silent: bool = False
        ) -> Iterable[TrainingExample]:
        view = build_base_training_view(
            tokenizer=self.tokenizer,
            chat_format=self.chat,
            training_view_config=training_view_config,
            windowing_policy=self.windowing_policy,
            output_root=output_root,
            use_task_labels=self.use_task_labels,
            tag_loss_weight=self.tag_loss_weight,
            verbosity_labels=self.verbosity_labels,
            format_labels=self.format_labels,
            silent=silent,
        )

        if self.packing_policy.enabled:
            view = PackedTrainingView(view, self.packing_policy, silent)

        return self.objective.wrap_training_view(view)


    def build_baseline_suite(self, baseline_config: Mapping[str, Any]) -> BaselineSuite:
        return CausalLmBaselineSuite(
            wrapper_builder=self._build_hf_baseline_wrapper,
            run_wrapper_builder=self._build_run_baseline_wrapper,
            baseline_config=baseline_config,
            output_root=Path(self.root_path),
        )


    def build_training_monitor(
        self,
        *,
        model: nn.Module,
        logger: Logger,
        configs: Iterable[Mapping[str, Any]],
    ) -> TrainingMonitor:
        return CausalLmProbeMonitor.from_configs(
            executor=CausalLmProbeExecutor(
                tokenizer=self.tokenizer,
                chat_format=self.chat,
                generator=self.generator,
            ),
            model=model,
            logger=logger,
            configs=configs,
        )


    def export_assets(self, output_dir: str | Path) -> None:
        self.tokenizer.save((Path(output_dir) / "tokenizer.json").as_posix())


    def _context_token_limit(self, model: nn.Module) -> int:
        limit = int(self.windowing_policy.window_max_len)
        model_limit = getattr(model, "max_seq_len", None)
        if isinstance(model_limit, int) and model_limit > 0:
            limit = min(limit, model_limit)
        else:
            model_config = getattr(model, "config", None)
            for attr_name in ("max_position_embeddings", "n_positions"):
                config_limit = getattr(model_config, attr_name, None)
                if isinstance(config_limit, int) and config_limit > 0:
                    limit = min(limit, config_limit)
                    break
        return max(1, limit)


    def _requested_infer_max_tokens(self) -> int:
        return self._DEFAULT_INFER_MAX_NEW_TOKENS

    def _default_infer_temperature(self) -> float:
        return self._DEFAULT_INFER_TEMPERATURE

    def _default_infer_top_p(self) -> float:
        return self._DEFAULT_INFER_TOP_P

    def _default_infer_top_k(self) -> int | None:
        return self._DEFAULT_INFER_TOP_K


    def _build_hf_baseline_wrapper(self, model_name: str) -> TaskWrapper:
        task_params = dict(self.params)
        task_params.pop("tokenizer_id", None)
        task_params.pop("tokenizer_file", None)
        task_params.pop("completion_prefix_masking", None)
        task_params["hf_tokenizer_name"] = model_name

        wrapper_cls = cast(Any, type(self))
        return wrapper_cls(
            device=self.device,
            bf_16_enabled=self.bf_16_enabled,
            params=task_params,
            root_path=self.root_path,
        )


    def _build_run_baseline_wrapper(self, run: Run) -> TaskWrapper:
        task_params = dict(run.config.task_config.params)
        tokenizer_file = run.model_path / "tokenizer.json"
        if tokenizer_file.is_file():
            task_params.pop("tokenizer_id", None)
            task_params.pop("hf_tokenizer_name", None)
            task_params["tokenizer_file"] = tokenizer_file.as_posix()

        wrapper_cls = cast(Any, type(self))
        return wrapper_cls(
            device=self.device,
            bf_16_enabled=self.bf_16_enabled,
            params=task_params,
            root_path=self.root_path,
        )


    def compute_loss(
        self,
        model: nn.Module,
        examples: List[TrainingExample],
    ) -> torch.Tensor:
        return self.objective.compute_loss(model, examples)

    def compute_eval_metrics(
        self,
        model: nn.Module,
        batches: Iterable[List[TrainingExample]],
    ) -> Dict[str, float]:
        return self.objective.compute_eval_metrics(model, batches)
