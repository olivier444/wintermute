from __future__ import annotations

from ..model import TrainTokenizerPreset

SMOKE_TRAIN_TOKENIZER_PRESETS = [
    TrainTokenizerPreset(
        name="smoke",
        tokenizer_id="en-bpe-12",
        snapshot_id="tns-smoke",
        split="train",
        vocab_size=12000,
        normalization="NFKC",
    ),
]
