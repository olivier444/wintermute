from __future__ import annotations

from ..model import TrainTokenizerPreset

DEFAULT_TRAIN_TOKENIZER_PRESETS = [
    TrainTokenizerPreset(
        name="default",
        tokenizer_id="enfr-bpe-sd-nfc-48",
        snapshot_id="mix-tok",
        split="big",
        vocab_size=48000,
        normalization="NFC",
    ),
]
