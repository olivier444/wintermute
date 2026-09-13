from __future__ import annotations

from collections.abc import Iterator

from tokenizers import Regex, Tokenizer as HFTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC, NFKC
from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence as PreTokenizerSequence, Split
from tokenizers.trainers import BpeTrainer

from wintermute.data.constants import FLD_GENERIC_UID
from wintermute.ml.tokenization.config import TokenizeConfig
from wintermute.ml.tokenization.wrapper import (
    TOK_EOS,
    TOK_PAD,
    TOK_UNK,
    WINTERMUTE_SPECIAL_TOKENS,
    TokenizerWrapper,
)
from wintermute.tools.files import jsonl_dir_iterator


def train_and_save_tokenizer(config: TokenizeConfig) -> None:
    iterator = _tokenizer_training_text_iterator(config.get_training_data_path())
    wrapper = train_tokenizer(
        iterator,
        vocab_size=config.vocab_size,
        normalization=config.normalization,
    )
    wrapper.save(config.get_tokenizer_file(ensure_dir_exist=True))


def _tokenizer_training_text_iterator(jsonl_directory: str) -> Iterator[str]:
    for obj in jsonl_dir_iterator(jsonl_directory):
        for field, value in obj.items():
            if field != FLD_GENERIC_UID and isinstance(value, str):
                yield value


def train_tokenizer(
    text_iterator: Iterator[str],
    *,
    vocab_size: int,
    normalization: str = "NFKC",
) -> TokenizerWrapper:
    tokenizer = HFTokenizer(BPE(unk_token=TOK_UNK))
    tokenizer.normalizer = _build_training_normalizer(normalization)
    tokenizer.pre_tokenizer = _build_pre_tokenizer()

    tokenizer.train_from_iterator(
        (TokenizerWrapper.normalize_text(text) for text in text_iterator),
        trainer=BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=3,
            special_tokens=WINTERMUTE_SPECIAL_TOKENS,
        ),
    )
    tokenizer.decoder = ByteLevelDecoder()

    return TokenizerWrapper(
        tokenizer,
        eos_token=TOK_EOS,
        pad_token=TOK_PAD,
    )


def _build_training_normalizer(normalization: str):
    normalized = normalization.strip().upper()
    if normalized == "NFKC":
        return NFKC()
    if normalized == "NFC":
        return NFC()
    raise ValueError(f"unsupported tokenizer normalization: {normalization!r}")


def _build_pre_tokenizer() -> PreTokenizerSequence:
    multi_ops = (
        r"==|!=|<=|>=|->|=>|::|//|/\*|\*/|\*\*|"
        r"\+=|-=|\*=|/=|%=|&&|\|\||<<|>>|:="
    )
    single_ops = r"[+\-*/=<>%^|&~!?:;,.\(\)\[\]\{\}\"'`@#$]"
    operator_pattern = Regex(f"({multi_ops}|{single_ops})")

    return PreTokenizerSequence(
        [
            ByteLevel(add_prefix_space=True),
            Split(operator_pattern, behavior="isolated"),
            Digits(individual_digits=True),
        ]
    )
