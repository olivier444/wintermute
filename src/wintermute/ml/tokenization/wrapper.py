from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from tokenizers import Encoding, Tokenizer as HFTokenizer
from transformers import PreTrainedTokenizerBase

from wintermute.ml.tokenization.chat_format import (
    WINTERMUTE_ASSISTANT,
    WINTERMUTE_EOT,
    WINTERMUTE_FINAL,
    WINTERMUTE_FORMAT,
    WINTERMUTE_SCRATCHPAD,
    WINTERMUTE_SYSTEM,
    WINTERMUTE_TASK,
    WINTERMUTE_USER,
    WINTERMUTE_VERBOSITY,
)
from wintermute.tools.logging import console_log

TOK_EOS = "[:eos:]"
TOK_MSK = "[:msk:]"
TOK_PAD = "[:pad:]"
TOK_UNK = "[:unk:]"

WINTERMUTE_SPECIAL_TOKENS = [
    TOK_PAD,
    TOK_MSK,
    TOK_EOS,
    TOK_UNK,
    WINTERMUTE_SYSTEM,
    WINTERMUTE_USER,
    WINTERMUTE_ASSISTANT,
    WINTERMUTE_EOT,
    "[:start_meta:]",
    "[:end_meta:]",
    WINTERMUTE_SCRATCHPAD,
    WINTERMUTE_FINAL,
    WINTERMUTE_TASK,
    WINTERMUTE_VERBOSITY,
    WINTERMUTE_FORMAT,
    "[:rsv6:]",
]

DROP_INVISIBLE_CHARS = {
    "\ufeff",  # BOM / zero width no-break space
    "\u200b",  # zero width space
    "\u2060",  # word joiner
    "\u2061",  # function application
    "\u2062",  # invisible times
    "\u2063",  # invisible separator
    "\u2064",  # invisible plus
    "\u00ad",  # soft hyphen
    "\u034f",  # combining grapheme joiner
}

SPACE_NORMALIZE_CHARS = {
    "\u00a0",  # no-break space
    "\u202f",  # narrow no-break space
    "\u2007",  # figure space
}

BIDI_CONTROL_CHARS = {
    "\u061c",  # Arabic letter mark
    "\u200e",  # left-to-right mark
    "\u200f",  # right-to-left mark
    "\u202a", "\u202b", "\u202c", "\u202d", "\u202e",
    "\u2066", "\u2067", "\u2068", "\u2069",
}


@dataclass(frozen=True)
class VocabItem:
    token_id: int
    raw_token: str
    decoded_token: str


class TokenizerWrapper:
    def __init__(
        self,
        tokenizer: HFTokenizer,
        *,
        eos_token: str,
        pad_token: str,
    ) -> None:
        self._tokenizer = tokenizer
        self.eos_id = self._required_token_id(eos_token)
        self.pad_id = self._required_token_id(pad_token)
        self._special_token_ids = frozenset(
            token_id
            for token_id, token in tokenizer.get_added_tokens_decoder().items()
            if token.special
        )

    @classmethod
    def from_external(cls, tokenizer: PreTrainedTokenizerBase) -> TokenizerWrapper:
        if tokenizer.eos_token is None:
            raise ValueError("[tokenizer] external tokenizer has no EOS token")

        eos = cast(str, tokenizer.eos_token)
        pad = cast(str, tokenizer.pad_token or eos)

        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is None:
            raise ValueError("[tokenizer] unsupported external tokenizer: no backend tokenizer found")

        return cls(
            backend,
            eos_token=eos,
            pad_token=pad,
        )

    @classmethod
    def from_file(cls, path: str) -> TokenizerWrapper:
        console_log("tokenizer", f"loading tokenizer from {path}")
        tokenizer = HFTokenizer.from_file(path)

        return cls(
            tokenizer,
            eos_token=TOK_EOS,
            pad_token=TOK_PAD,
        )

    def token_to_id(self, token: str) -> int | None:
        return self._tokenizer.token_to_id(token)

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.replace("\r\n", "\n").replace("\n\r", "\n").replace("\r", "\n")
        out: list[str] = []

        for ch in text:
            if ch in DROP_INVISIBLE_CHARS:
                continue

            if ch in SPACE_NORMALIZE_CHARS:
                out.append(" ")
                continue

            if ch in BIDI_CONTROL_CHARS:
                continue

            out.append(ch)

        return "".join(out)

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = True) -> str:
        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def encode(self, text: str, *, add_special_tokens: bool = True) -> Encoding:
        return self._tokenizer.encode(
            self.normalize_text(text),
            add_special_tokens=add_special_tokens,
        )

    def encode_to_ids(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return self.encode(text, add_special_tokens=add_special_tokens).ids

    def count_non_special_tokens(self, text: str) -> int:
        return sum(
            token_id not in self._special_token_ids
            for token_id in self.encode_to_ids(text, add_special_tokens=False)
        )

    def get_vocab_items(self) -> list[VocabItem]:
        vocab = self._tokenizer.get_vocab()
        return [
            VocabItem(
                token_id=token_id,
                raw_token=raw_token,
                decoded_token=self.decode([token_id], skip_special_tokens=False),
            )
            for raw_token, token_id in sorted(vocab.items(), key=lambda item: item[1])
        ]

    def _required_token_id(self, token: str) -> int:
        tok_id = self._tokenizer.token_to_id(token)
        if tok_id is None:
            raise ValueError(f"[tokenizer] special token {token} not found in vocabulary")
        return tok_id

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.get_vocab_size()

    def save(self, path: str) -> None:
        self._tokenizer.save(path)
