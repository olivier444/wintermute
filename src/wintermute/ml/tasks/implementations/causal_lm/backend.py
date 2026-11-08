from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from wintermute.data.constants import DIR_EXTERNAL_MODELS, DIR_HUGGING_FACE_MODELS
from wintermute.ml.inference.conversation import (
    InferenceSession,
    InferenceTurn,
)
from wintermute.ml.tasks.implementations.causal_lm.config import CausalLmTaskConfig
from wintermute.ml.tasks.implementations.causal_lm.generation import CausalLmTokenGenerator
from wintermute.ml.tokenization.chat_format import (
    ChatFormat,
    PlainTextChatFormat,
    WintermuteChatFormat,
    build_hugging_face_chat_format,
)
from wintermute.ml.tokenization.wrapper import TokenizerWrapper
from wintermute.tools.files import get_tokenizer_file
from wintermute.tools.logging import console_log


class CausalLmBackend:
    """Shared causal-LM services independent of a model's ownership."""

    def __init__(
        self,
        *,
        device: torch.device,
        bf16_enabled: bool,
        tokenizer: TokenizerWrapper,
        chat_format: ChatFormat,
        window_max_len: int,
    ) -> None:
        self.device = device
        self.bf16_enabled = bf16_enabled
        self.tokenizer = tokenizer
        self.chat_format = chat_format
        self.window_max_len = window_max_len
        self.generator = CausalLmTokenGenerator(
            device=device,
            bf16_enabled=bf16_enabled,
            tokenizer=tokenizer,
            chat_format=chat_format,
        )

    @classmethod
    def from_config(
        cls,
        *,
        config: CausalLmTaskConfig,
        root_path: str | Path,
        device: torch.device,
        bf16_enabled: bool,
    ) -> "CausalLmBackend":
        tokenizer, chat_format = cls._load_tokenizer_and_chat_format(
            config,
            Path(root_path),
        )
        return cls(
            device=device,
            bf16_enabled=bf16_enabled,
            tokenizer=tokenizer,
            chat_format=chat_format,
            window_max_len=config.windowing_policy.window_max_len,
        )

    @staticmethod
    def _load_tokenizer_and_chat_format(
        config: CausalLmTaskConfig,
        root_path: Path,
    ) -> tuple[TokenizerWrapper, ChatFormat]:
        if config.tokenizer_id is not None:
            tokenizer_file = get_tokenizer_file(
                root_path.as_posix(),
                config.tokenizer_id,
            )
            return (
                TokenizerWrapper.from_file(tokenizer_file),
                WintermuteChatFormat(assistant_format=config.assistant_format),
            )

        if config.tokenizer_file is not None:
            return (
                TokenizerWrapper.from_file(config.tokenizer_file),
                WintermuteChatFormat(assistant_format=config.assistant_format),
            )

        assert config.hf_tokenizer_name is not None
        cache_dir = root_path / DIR_EXTERNAL_MODELS / DIR_HUGGING_FACE_MODELS
        hf_tokenizer = AutoTokenizer.from_pretrained(
            config.hf_tokenizer_name,
            cache_dir=cache_dir.as_posix(),
        )
        chat_format = build_hugging_face_chat_format(hf_tokenizer)
        if isinstance(chat_format, PlainTextChatFormat):
            console_log(
                "chat-format",
                f"{config.hf_tokenizer_name} has no chat template; "
                "using plain-text completion fallback",
            )
        return TokenizerWrapper.from_external(hf_tokenizer), chat_format

    def context_token_limit(self, model: nn.Module) -> int:
        limit = int(self.window_max_len)
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

    def start_inference_session(
        self,
        model: nn.Module,
        system_prompt: str | None = None,
        initial_turns: tuple[InferenceTurn, ...] = (),
    ) -> InferenceSession:
        from wintermute.ml.tasks.implementations.causal_lm.session import (
            CausalLmInferenceSession,
        )

        return CausalLmInferenceSession(
            backend=self,
            model=model,
            system_prompt=system_prompt,
            initial_turns=initial_turns,
        )

    def export_assets(self, output_dir: str | Path) -> None:
        self.tokenizer.save((Path(output_dir) / "tokenizer.json").as_posix())
