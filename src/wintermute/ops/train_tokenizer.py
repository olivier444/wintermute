# -*- coding: utf-8 -*-

from wintermute.ml.tokenization.config import TokenizeConfig
from wintermute.ml.tokenization.train import train_and_save_tokenizer
from wintermute.ops.presets import TrainTokenizerPreset


def run_train_tokenizer(preset: TrainTokenizerPreset, output_root: str) -> None:
    cfg = TokenizeConfig(
        tokenizer_id=preset.tokenizer_id,
        output_root=output_root,
        training_snapshot_id=preset.snapshot_id,
        training_snapshot_split=preset.split,
        vocab_size=preset.vocab_size,
        normalization=preset.normalization,
    )
    
    train_and_save_tokenizer(cfg)
