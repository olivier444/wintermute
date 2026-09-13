from __future__ import annotations

import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List

from wintermute.data.constants import FLD_GENERIC_TEXT
from wintermute.ml.tokenization.wrapper import WINTERMUTE_SPECIAL_TOKENS, TokenizerWrapper, VocabItem
from wintermute.ops.presets import TrainTokenizerPreset, list_train_tokenizer_presets
from wintermute.tools.files import get_materialized_dir, get_tokenizer_file, text_iterator

_SINGLE_DIGIT_RE = re.compile(r"[0-9]")
_PURE_MULTI_DIGIT_RE = re.compile(r"[0-9]{2,}")
_MULTI_DIGIT_RUN_RE = re.compile(r"[0-9]{2,}")
_ANY_DIGIT_RE = re.compile(r"[0-9]")
_NON_DIGIT_RE = re.compile(r"[^0-9]")
_URL_DOMAIN_RE = re.compile(
    r"(?:^|[^a-z0-9])(?:[a-z0-9-]+\.)+(?:com|org|net|io|fr|co|ai|gov|edu)(?:/|$)",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}", re.IGNORECASE)
_HEX_PREFIX_RE = re.compile(r"0x[0-9a-f]{2,}", re.IGNORECASE)
_HEX_FULL_RE = re.compile(r"[0-9a-f]{8,}", re.IGNORECASE)
_HEX_RUN_RE = re.compile(r"[0-9a-f]{12,}", re.IGNORECASE)
_INTERNAL_WHITESPACE_RE = re.compile(r"\S\s+\S")
_URL_LITERAL_PARTS = {
    "http",
    "https",
    "http://",
    "https://",
    "www",
    "www.",
    "://",
    "//",
    ".com",
    ".org",
    ".net",
    ".io",
    ".fr",
    ".co",
    ".ai",
    ".gov",
    ".edu",
}
_OPERATOR_CHARS = frozenset("+-*/=%<>!&|^~")
_VERY_LONG_CHAR_THRESHOLD = 24
_VERY_LONG_BYTE_THRESHOLD = 32
_LENGTH_BUCKET_LIMITS = (1, 2, 4, 8, 16, 32)
_DEFAULT_USAGE_SAMPLE_RECORD_LIMIT = 10_000
_DEFAULT_USAGE_SAMPLE_CHAR_LIMIT = 2_000_000

_DEFAULT_BEHAVIOR_SAMPLES = [
    "2024",
    "1234567890",
    "12+34=46",
    "-12.5%",
    "v2.0.1",
    "id123",
    "x<=100",
    "2024-01-03",
]

_SUSPICIOUS_CHARS = {
    "﻿": "BOM / zero width no-break space",
    "​": "zero width space",
    "⁠": "word joiner",
    "­": "soft hyphen",
    "͏": "combining grapheme joiner",
    "؜": "arabic letter mark",
    "‎": "left-to-right mark",
    "‏": "right-to-left mark",
    " ": "no-break space",
    " ": "narrow no-break space",
    " ": "figure space",
}
_BIDI_CONTROL_RANGES = (
    range(0x202A, 0x202F),
    range(0x2066, 0x206A),
)


@dataclass(frozen=True)
class SuspiciousCharTokenAudit:
    token_id: int
    raw_token: str
    decoded_token: str
    suspicious_chars: List[str]


@dataclass(frozen=True)
class SuspiciousCharCount:
    description: str
    count: int


@dataclass(frozen=True)
class SpecialTokenEncodingAudit:
    token: str
    expected_id: int | None
    encoded_ids: List[int]
    encoded_tokens: List[str]
    is_single_token_encoding: bool


@dataclass(frozen=True)
class EncodedSample:
    text: str
    tokens: List[str]
    ids: List[int]


@dataclass(frozen=True)
class LengthBucket:
    label: str
    count: int


@dataclass(frozen=True)
class LengthDistribution:
    unit: str
    avg: float
    p50: int
    p95: int
    max_value: int
    buckets: List[LengthBucket]


@dataclass(frozen=True)
class SampleUsageAudit:
    source_path: str | None
    sampled_records: int
    sampled_chars: int
    sampled_tokens: int
    is_truncated: bool
    non_special_total: int
    unused_non_special_items: List[VocabItem]
    unavailable_reason: str | None = None


def run_audit_tokenizer(
    output_root: str,
    tokenizer_name: str,
    example_limit: int = 10,
    *,
    strict_digitwise: bool = False,
) -> None:
    tokenizer_id = _resolve_tokenizer_id(tokenizer_name)
    tokenizer_path = get_tokenizer_file(output_root, tokenizer_id)
    wrapper = TokenizerWrapper.from_file(tokenizer_path)
    vocab_items = wrapper.get_vocab_items()
    preset = _find_tokenizer_preset(tokenizer_id)

    single_digit_items = [item for item in vocab_items if _SINGLE_DIGIT_RE.fullmatch(item.decoded_token)]
    any_digit_items = [item for item in vocab_items if _ANY_DIGIT_RE.search(item.decoded_token)]
    pure_multi_digit_items = [item for item in vocab_items if _PURE_MULTI_DIGIT_RE.fullmatch(item.decoded_token)]
    contains_multi_digit_items = [item for item in vocab_items if _MULTI_DIGIT_RUN_RE.search(item.decoded_token)]
    pure_multi_digit_ids = {item.token_id for item in pure_multi_digit_items}
    mixed_multi_digit_items = [
        item
        for item in contains_multi_digit_items
        if item.token_id not in pure_multi_digit_ids
    ]
    digit_and_non_digit_items = [item for item in vocab_items if _contains_digit_and_non_digit(item.decoded_token)]
    digit_and_operator_items = [item for item in vocab_items if _contains_digit_and_operator(item.decoded_token)]
    operator_and_alnum_items = [item for item in vocab_items if _contains_operator_and_alnum(item.decoded_token)]
    non_single_digit_items = _find_non_single_digit_items(vocab_items)
    url_like_items = [item for item in vocab_items if _looks_like_url(item.decoded_token)]
    email_like_items = [item for item in vocab_items if _looks_like_email(item.decoded_token)]
    path_like_items = [item for item in vocab_items if _looks_like_path(item.decoded_token)]
    hex_like_items = [item for item in vocab_items if _looks_like_hex(item.decoded_token)]
    internal_whitespace_items = [item for item in vocab_items if _contains_internal_whitespace(item.decoded_token)]
    very_long_items = [item for item in vocab_items if _is_very_long_token(item.decoded_token)]
    char_length_distribution = _build_length_distribution(
        [_token_char_length(item.decoded_token) for item in vocab_items],
        unit="chars",
    )
    byte_length_distribution = _build_length_distribution(
        [_token_byte_length(item.decoded_token) for item in vocab_items],
        unit="bytes",
    )
    suspicious_char_audits = _audit_suspicious_vocab_chars(vocab_items)
    suspicious_char_counts = _build_suspicious_char_counts(suspicious_char_audits)
    suspicious_char_examples = _group_suspicious_char_examples(suspicious_char_audits)
    special_token_audits = [_audit_special_token_encoding(wrapper, token) for token in WINTERMUTE_SPECIAL_TOKENS]
    sample_usage_audit = _audit_sample_vocab_usage(
        output_root=output_root,
        wrapper=wrapper,
        tokenizer_id=tokenizer_id,
        preset=preset,
        vocab_items=vocab_items,
    )
    encoded_samples = _encode_samples(wrapper, _DEFAULT_BEHAVIOR_SAMPLES)

    print(
        _render_report(
            tokenizer_id=tokenizer_id,
            tokenizer_path=tokenizer_path,
            vocab_items=vocab_items,
            preset=preset,
            single_digit_items=single_digit_items,
            any_digit_items=any_digit_items,
            pure_multi_digit_items=pure_multi_digit_items,
            contains_multi_digit_items=contains_multi_digit_items,
            mixed_multi_digit_items=mixed_multi_digit_items,
            digit_and_non_digit_items=digit_and_non_digit_items,
            digit_and_operator_items=digit_and_operator_items,
            operator_and_alnum_items=operator_and_alnum_items,
            non_single_digit_items=non_single_digit_items,
            url_like_items=url_like_items,
            email_like_items=email_like_items,
            path_like_items=path_like_items,
            hex_like_items=hex_like_items,
            internal_whitespace_items=internal_whitespace_items,
            very_long_items=very_long_items,
            char_length_distribution=char_length_distribution,
            byte_length_distribution=byte_length_distribution,
            suspicious_char_audits=suspicious_char_audits,
            suspicious_char_counts=suspicious_char_counts,
            suspicious_char_examples=suspicious_char_examples,
            special_token_audits=special_token_audits,
            sample_usage_audit=sample_usage_audit,
            encoded_samples=encoded_samples,
            example_limit=example_limit,
        )
    )

    if strict_digitwise and non_single_digit_items:
        raise SystemExit(
            "[tokenizer audit] strict digit-wise check failed: "
            f"{len(non_single_digit_items):_} non-special vocab item(s) contain a digit "
            "outside pure single-digit tokens"
        )


def _resolve_tokenizer_id(tokenizer_name: str) -> str:
    normalized = tokenizer_name.strip()
    if not normalized:
        raise ValueError("tokenizer name is required")

    lowered = normalized.lower()
    for preset in list_train_tokenizer_presets().values():
        if preset.name.lower() == lowered or preset.tokenizer_id.lower() == lowered:
            return preset.tokenizer_id

    return normalized


def _find_tokenizer_preset(tokenizer_id: str) -> TrainTokenizerPreset | None:
    lowered = tokenizer_id.lower()
    for preset in list_train_tokenizer_presets().values():
        if preset.tokenizer_id.lower() == lowered:
            return preset
    return None


def _find_non_single_digit_items(vocab_items: List[VocabItem]) -> List[VocabItem]:
    special_tokens = set(WINTERMUTE_SPECIAL_TOKENS)
    return [
        item
        for item in vocab_items
        if item.decoded_token not in special_tokens
        and not _SINGLE_DIGIT_RE.fullmatch(item.decoded_token)
        and _ANY_DIGIT_RE.search(item.decoded_token)
    ]


def _contains_digit_and_non_digit(token: str) -> bool:
    return _ANY_DIGIT_RE.search(token) is not None and _NON_DIGIT_RE.search(token) is not None


def _has_operator(token: str) -> bool:
    return any(ch in _OPERATOR_CHARS for ch in token)


def _contains_digit_and_operator(token: str) -> bool:
    return _ANY_DIGIT_RE.search(token) is not None and _has_operator(token)


def _contains_operator_and_alnum(token: str) -> bool:
    return _has_operator(token) and any(ch.isalnum() for ch in token)


def _looks_like_url(token: str) -> bool:
    lowered = token.strip().lower()
    if not lowered:
        return False
    if lowered in _URL_LITERAL_PARTS:
        return True
    if "http://" in lowered or "https://" in lowered or "://" in lowered or "www." in lowered:
        return True
    return _URL_DOMAIN_RE.search(lowered) is not None


def _looks_like_email(token: str) -> bool:
    lowered = token.strip().lower()
    if not lowered:
        return False
    return _EMAIL_RE.search(lowered) is not None


def _looks_like_path(token: str) -> bool:
    stripped = token.strip()
    if not stripped:
        return False
    if stripped.startswith(("/", "./", "../", "~/")):
        return any(ch.isalnum() for ch in stripped)
    if re.match(r"^[A-Za-z]:\\", stripped) is not None:
        return any(ch.isalnum() for ch in stripped)
    if "/" in stripped or "\\" in stripped:
        return any(ch.isalnum() for ch in stripped)
    return False

def _looks_like_hex(token: str) -> bool:
    lowered = token.strip().lower()
    if not lowered:
        return False
    if _HEX_PREFIX_RE.fullmatch(lowered) is not None:
        return True
    if _HEX_FULL_RE.fullmatch(lowered) is not None:
        return True
    if _HEX_PREFIX_RE.search(lowered) is not None:
        return True
    return _HEX_RUN_RE.search(lowered) is not None


def _contains_internal_whitespace(token: str) -> bool:
    return _INTERNAL_WHITESPACE_RE.search(token) is not None


def _token_char_length(token: str) -> int:
    return len(token)


def _token_byte_length(token: str) -> int:
    return len(token.encode("utf-8"))


def _is_very_long_token(token: str) -> bool:
    return (
        _token_char_length(token) >= _VERY_LONG_CHAR_THRESHOLD
        or _token_byte_length(token) >= _VERY_LONG_BYTE_THRESHOLD
    )


def _build_length_distribution(lengths: List[int], *, unit: str) -> LengthDistribution:
    if not lengths:
        return LengthDistribution(unit=unit, avg=0.0, p50=0, p95=0, max_value=0, buckets=[])

    sorted_lengths = sorted(lengths)
    total = len(sorted_lengths)
    avg = sum(sorted_lengths) / total
    p50 = sorted_lengths[(total - 1) * 50 // 100]
    p95 = sorted_lengths[(total - 1) * 95 // 100]
    buckets: List[LengthBucket] = []
    lower = 0
    for upper in _LENGTH_BUCKET_LIMITS:
        count = sum(1 for length in sorted_lengths if lower < length <= upper)
        label = str(upper) if lower + 1 == upper else f"{lower + 1}-{upper}"
        buckets.append(LengthBucket(label=label, count=count))
        lower = upper
    buckets.append(
        LengthBucket(
            label=f"{_LENGTH_BUCKET_LIMITS[-1] + 1}+",
            count=sum(1 for length in sorted_lengths if length > _LENGTH_BUCKET_LIMITS[-1]),
        )
    )
    return LengthDistribution(
        unit=unit,
        avg=avg,
        p50=p50,
        p95=p95,
        max_value=sorted_lengths[-1],
        buckets=buckets,
    )


def _is_bidi_control(ch: str) -> bool:
    codepoint = ord(ch)
    return any(codepoint in r for r in _BIDI_CONTROL_RANGES)


def _describe_char(ch: str) -> str:
    codepoint = f"U+{ord(ch):04X}"
    name = unicodedata.name(ch, "UNKNOWN")
    category = unicodedata.category(ch)
    extra = _SUSPICIOUS_CHARS.get(ch)
    if extra:
        return f"{codepoint} {name} {category} ({extra})"
    return f"{codepoint} {name} {category}"


def _is_suspicious_char(ch: str) -> bool:
    category = unicodedata.category(ch)
    return ch in _SUSPICIOUS_CHARS or _is_bidi_control(ch) or category in {"Cc", "Cf"}


def _audit_sample_vocab_usage(
    *,
    output_root: str,
    wrapper: TokenizerWrapper,
    tokenizer_id: str,
    preset: TrainTokenizerPreset | None,
    vocab_items: List[VocabItem],
    sample_record_limit: int = _DEFAULT_USAGE_SAMPLE_RECORD_LIMIT,
    sample_char_limit: int = _DEFAULT_USAGE_SAMPLE_CHAR_LIMIT,
) -> SampleUsageAudit:
    special_tokens = set(WINTERMUTE_SPECIAL_TOKENS)
    non_special_items = [item for item in vocab_items if item.decoded_token not in special_tokens]
    if preset is None:
        return SampleUsageAudit(
            source_path=None,
            sampled_records=0,
            sampled_chars=0,
            sampled_tokens=0,
            is_truncated=False,
            non_special_total=len(non_special_items),
            unused_non_special_items=[],
            unavailable_reason=f"no matching train-tokenizer preset found for {tokenizer_id}",
        )

    source_path = get_materialized_dir(output_root, preset.snapshot_id, preset.split)
    if not os.path.exists(source_path):
        return SampleUsageAudit(
            source_path=source_path,
            sampled_records=0,
            sampled_chars=0,
            sampled_tokens=0,
            is_truncated=False,
            non_special_total=len(non_special_items),
            unused_non_special_items=[],
            unavailable_reason="training materialized split not found on disk",
        )

    used_token_ids: set[int] = set()
    sampled_records = 0
    sampled_chars = 0
    sampled_tokens = 0
    is_truncated = False

    try:
        for text in text_iterator(source_path, FLD_GENERIC_TEXT):
            if sampled_records >= sample_record_limit or sampled_chars >= sample_char_limit:
                is_truncated = True
                break
            sampled_records += 1
            sampled_chars += len(text)
            ids = wrapper.encode_to_ids(text, add_special_tokens=False)
            sampled_tokens += len(ids)
            used_token_ids.update(ids)
    except Exception as exc:
        return SampleUsageAudit(
            source_path=source_path,
            sampled_records=sampled_records,
            sampled_chars=sampled_chars,
            sampled_tokens=sampled_tokens,
            is_truncated=is_truncated,
            non_special_total=len(non_special_items),
            unused_non_special_items=[],
            unavailable_reason=f"failed to scan sample corpus: {exc}",
        )

    return SampleUsageAudit(
        source_path=source_path,
        sampled_records=sampled_records,
        sampled_chars=sampled_chars,
        sampled_tokens=sampled_tokens,
        is_truncated=is_truncated,
        non_special_total=len(non_special_items),
        unused_non_special_items=[item for item in non_special_items if item.token_id not in used_token_ids],
        unavailable_reason=None if sampled_records > 0 else "sample corpus was empty",
    )


def _audit_suspicious_vocab_chars(vocab_items: List[VocabItem]) -> List[SuspiciousCharTokenAudit]:
    audits: List[SuspiciousCharTokenAudit] = []
    for item in vocab_items:
        suspicious = [_describe_char(ch) for ch in item.decoded_token if _is_suspicious_char(ch)]
        if suspicious:
            audits.append(
                SuspiciousCharTokenAudit(
                    token_id=item.token_id,
                    raw_token=item.raw_token,
                    decoded_token=item.decoded_token,
                    suspicious_chars=suspicious,
                )
            )
    return audits


def _build_suspicious_char_counts(audits: List[SuspiciousCharTokenAudit]) -> List[SuspiciousCharCount]:
    counts: Counter[str] = Counter()
    for audit in audits:
        counts.update(audit.suspicious_chars)
    return [
        SuspiciousCharCount(description=description, count=count)
        for description, count in counts.most_common()
    ]


def _group_suspicious_char_examples(
    audits: List[SuspiciousCharTokenAudit],
) -> DefaultDict[str, List[SuspiciousCharTokenAudit]]:
    grouped: DefaultDict[str, List[SuspiciousCharTokenAudit]] = defaultdict(list)
    for audit in audits:
        for description in set(audit.suspicious_chars):
            grouped[description].append(audit)
    return grouped


def _audit_special_token_encoding(wrapper: TokenizerWrapper, token: str) -> SpecialTokenEncodingAudit:
    expected_id = wrapper.token_to_id(token)
    encoding = wrapper.encode(token)
    encoded_ids = list(encoding.ids)
    encoded_tokens = [str(piece) for piece in encoding.tokens]
    is_single_token_encoding = (
        expected_id is not None
        and len(encoded_ids) == 1
        and encoded_ids[0] == expected_id
        and len(encoded_tokens) == 1
        and encoded_tokens[0] == token
    )
    return SpecialTokenEncodingAudit(
        token=token,
        expected_id=expected_id,
        encoded_ids=encoded_ids,
        encoded_tokens=encoded_tokens,
        is_single_token_encoding=is_single_token_encoding,
    )


def _encode_samples(wrapper: TokenizerWrapper, samples: List[str]) -> List[EncodedSample]:
    encoded_samples: List[EncodedSample] = []
    for sample in samples:
        encoding = wrapper.encode(sample, add_special_tokens=False)
        encoded_samples.append(
            EncodedSample(
                text=sample,
                tokens=list(encoding.tokens),
                ids=list(encoding.ids),
            )
        )
    return encoded_samples


def _render_report(
    *,
    tokenizer_id: str,
    tokenizer_path: str,
    vocab_items: List[VocabItem],
    preset: TrainTokenizerPreset | None,
    single_digit_items: List[VocabItem],
    any_digit_items: List[VocabItem],
    pure_multi_digit_items: List[VocabItem],
    contains_multi_digit_items: List[VocabItem],
    mixed_multi_digit_items: List[VocabItem],
    digit_and_non_digit_items: List[VocabItem],
    digit_and_operator_items: List[VocabItem],
    operator_and_alnum_items: List[VocabItem],
    non_single_digit_items: List[VocabItem],
    url_like_items: List[VocabItem],
    email_like_items: List[VocabItem],
    path_like_items: List[VocabItem],
    hex_like_items: List[VocabItem],
    internal_whitespace_items: List[VocabItem],
    very_long_items: List[VocabItem],
    char_length_distribution: LengthDistribution,
    byte_length_distribution: LengthDistribution,
    suspicious_char_audits: List[SuspiciousCharTokenAudit],
    suspicious_char_counts: List[SuspiciousCharCount],
    suspicious_char_examples: DefaultDict[str, List[SuspiciousCharTokenAudit]],
    special_token_audits: List[SpecialTokenEncodingAudit],
    sample_usage_audit: SampleUsageAudit,
    encoded_samples: List[EncodedSample],
    example_limit: int,
) -> str:
    total = len(vocab_items)
    passed_special_tokens = sum(1 for audit in special_token_audits if audit.is_single_token_encoding)
    preset_label = (
        f"{preset.name} -> {preset.snapshot_id}/{preset.split}"
        if preset is not None
        else "(no matching train-tokenizer preset)"
    )
    lines = [
        f"Tokenizer audit: {tokenizer_id}",
        f"Tokenizer path: {tokenizer_path}",
        f"Training preset linkage: {preset_label}",
        f"Vocab size: {_fmt_count(total)}",
        f"Reserved special tokens with single-token encoding: {passed_special_tokens}/{len(special_token_audits)}",
        f"Pure single-digit items: {_fmt_metric(len(single_digit_items), total)}",
        f"Items containing any digit: {_fmt_metric(len(any_digit_items), total)}",
        f"Pure digit-assembly items: {_fmt_metric(len(pure_multi_digit_items), total)}",
        f"Items containing a multi-digit run: {_fmt_metric(len(contains_multi_digit_items), total)}",
        f"Items containing a multi-digit run mixed with non-digits: {_fmt_metric(len(mixed_multi_digit_items), total)}",
        f"Items containing digit + non-digit: {_fmt_metric(len(digit_and_non_digit_items), total)}",
        f"Items containing digit + operator: {_fmt_metric(len(digit_and_operator_items), total)}",
        f"Items containing operator + alnum: {_fmt_metric(len(operator_and_alnum_items), total)}",
        f"Non-special items containing any digit except pure single digits: {_fmt_metric(len(non_single_digit_items), total)}",
        f"URL-like items: {_fmt_metric(len(url_like_items), total)}",
        f"Email-like items: {_fmt_metric(len(email_like_items), total)}",
        f"Path-like items: {_fmt_metric(len(path_like_items), total)}",
        f"Hex-like items: {_fmt_metric(len(hex_like_items), total)}",
        f"Items containing internal whitespace: {_fmt_metric(len(internal_whitespace_items), total)}",
        f"Very long items: {_fmt_metric(len(very_long_items), total)}",
        f"Tokens with invisible/control Unicode chars (Cf/Cc): {_fmt_metric(len(suspicious_char_audits), total)}",
        f"Distinct suspicious Unicode chars: {_fmt_count(len(suspicious_char_counts))}",
        _render_sample_usage_summary(sample_usage_audit),
        "",
        "Length distribution:",
        _render_length_distribution(char_length_distribution),
        _render_length_distribution(byte_length_distribution),
        "",
        "Behavior samples:",
        _render_encoded_samples(encoded_samples),
        "",
        "Definitions:",
        "- pure single-digit item: decoded token matches exactly [0-9]",
        "- item containing any digit: decoded token contains [0-9] anywhere",
        "- pure digit-assembly item: decoded token matches exactly [0-9]{2,}",
        "- contains a multi-digit run: decoded token contains [0-9]{2,} anywhere",
        "- digit + non-digit item: token contains at least one digit and at least one non-digit",
        "- digit + operator item: token contains at least one digit and one operator from +-*/=%<>!&|^~",
        "- operator + alnum item: token contains at least one operator and at least one letter or digit",
        "- non-special item containing any digit except pure single digits: useful strict digit-wise invariant",
        "- url/email/path-like items: decoded token matches common web/code fragments (heuristic)",
        "- hex-like item: decoded token matches `0x...` or long hexadecimal runs (heuristic)",
        "- internal whitespace item: token contains whitespace between two non-whitespace spans",
        f"- very long item: decoded token length is >= {_VERY_LONG_CHAR_THRESHOLD} chars or >= {_VERY_LONG_BYTE_THRESHOLD} UTF-8 bytes",
        "- suspicious Unicode char: decoded token contains format/control chars, bidi controls, or known invisible spacing/joiner chars",
        "",
        "Note: changing tokenizer training code is not retroactive; existing tokenizer.json files keep their saved vocab and pre-tokenizer.",
        "",
        _render_suspicious_char_counts(suspicious_char_counts),
        "",
        _render_special_token_audits(special_token_audits),
    ]

    if example_limit > 0:
        lines.extend(
            [
                "",
                _render_examples("Examples: pure single-digit items", single_digit_items, example_limit),
                "",
                _render_examples("Examples: items containing any digit", any_digit_items, example_limit),
                "",
                _render_examples("Examples: pure digit-assembly items", pure_multi_digit_items, example_limit),
                "",
                _render_examples("Examples: items containing a multi-digit run", contains_multi_digit_items, example_limit),
                "",
                _render_examples("Examples: items containing digit + non-digit", digit_and_non_digit_items, example_limit),
                "",
                _render_examples("Examples: items containing digit + operator", digit_and_operator_items, example_limit),
                "",
                _render_examples("Examples: items containing operator + alnum", operator_and_alnum_items, example_limit),
                "",
                _render_examples(
                    "Examples: non-special items containing any digit except pure single digits",
                    non_single_digit_items,
                    example_limit,
                ),
                "",
                _render_examples("Examples: URL-like items", url_like_items, example_limit),
                "",
                _render_examples("Examples: email-like items", email_like_items, example_limit),
                "",
                _render_examples("Examples: path-like items", path_like_items, example_limit),
                "",
                _render_examples("Examples: hex-like items", hex_like_items, example_limit),
                "",
                _render_examples("Examples: items containing internal whitespace", internal_whitespace_items, example_limit),
                "",
                _render_examples("Examples: very long items", very_long_items, example_limit),
                "",
                _render_unused_sample_examples(sample_usage_audit, example_limit),
                "",
                _render_suspicious_char_examples(
                    suspicious_char_audits=suspicious_char_audits,
                    suspicious_char_examples=suspicious_char_examples,
                    example_limit=example_limit,
                ),
            ]
        )

    return "\n".join(lines)


def _render_encoded_samples(samples: List[EncodedSample]) -> str:
    if not samples:
        return "(none)"

    lines: List[str] = []
    for sample in samples:
        tokens = " ".join(repr(token) for token in sample.tokens)
        lines.append(f"- {sample.text!r}: {tokens}")
    return "\n".join(lines)


def _render_length_distribution(distribution: LengthDistribution) -> str:
    total = sum(item.count for item in distribution.buckets)
    bucket_text = ", ".join(
        f"{bucket.label}={_fmt_count(bucket.count)} ({_fmt_pct(bucket.count, total)})"
        for bucket in distribution.buckets
    )
    return (
        f"- {distribution.unit}: avg={distribution.avg:.2f}, p50={distribution.p50}, "
        f"p95={distribution.p95}, max={distribution.max_value}; buckets: {bucket_text}"
    )


def _render_sample_usage_summary(audit: SampleUsageAudit) -> str:
    if audit.unavailable_reason is not None:
        location = f" at {audit.source_path}" if audit.source_path else ""
        return f"Sample-corpus usage audit: unavailable{location} ({audit.unavailable_reason})"

    unused = len(audit.unused_non_special_items)
    truncation = " (truncated sample)" if audit.is_truncated else ""
    return (
        "Non-special items never seen on sample corpus: "
        f"{_fmt_count(unused)}/{_fmt_count(audit.non_special_total)} "
        f"({_fmt_pct(unused, audit.non_special_total)}), "
        f"sampled_records={_fmt_count(audit.sampled_records)}, "
        f"sampled_chars={_fmt_count(audit.sampled_chars)}, "
        f"sampled_tokens={_fmt_count(audit.sampled_tokens)}, "
        f"source={audit.source_path}{truncation}"
    )


def _render_unused_sample_examples(audit: SampleUsageAudit, limit: int) -> str:
    title = "Examples: non-special items never seen on sample corpus"
    if audit.unavailable_reason is not None:
        return f"{title}\n(unavailable: {audit.unavailable_reason})"
    return _render_examples(title, audit.unused_non_special_items, limit)


def _render_special_token_audits(audits: List[SpecialTokenEncodingAudit]) -> str:
    lines = ["Special token encoding checks"]
    for audit in audits:
        status = "PASS" if audit.is_single_token_encoding else "FAIL"
        lines.append(
            f"- {status} {audit.token}: expected_id={audit.expected_id}, encoded_ids={audit.encoded_ids}, encoded_tokens={audit.encoded_tokens}"
        )
    return "\n".join(lines)


def _render_suspicious_char_counts(counts: List[SuspiciousCharCount]) -> str:
    lines = ["Suspicious Unicode char counts"]
    if not counts:
        lines.append("(none)")
        return "\n".join(lines)

    for item in counts:
        lines.append(f"- {item.count:_}: {item.description}")
    return "\n".join(lines)


def _render_suspicious_char_examples(
    *,
    suspicious_char_audits: List[SuspiciousCharTokenAudit],
    suspicious_char_examples: DefaultDict[str, List[SuspiciousCharTokenAudit]],
    example_limit: int,
) -> str:
    lines = ["Examples: tokens with suspicious Unicode chars"]
    if not suspicious_char_audits:
        lines.append("(none)")
        return "\n".join(lines)

    for audit in suspicious_char_audits[:example_limit]:
        lines.append(
            f"- id={audit.token_id}: raw={audit.raw_token!r}, decoded={audit.decoded_token!r}, suspicious={audit.suspicious_chars}"
        )

    remaining = len(suspicious_char_audits) - min(len(suspicious_char_audits), example_limit)
    if remaining > 0:
        lines.append(f"- ... {remaining:_} more")

    lines.append("")
    lines.append("Examples by suspicious char")
    for description in sorted(suspicious_char_examples):
        lines.append(f"- {description}")
        for audit in suspicious_char_examples[description][:example_limit]:
            lines.append(
                f"  id={audit.token_id}: raw={audit.raw_token!r}, decoded={audit.decoded_token!r}"
            )
        extra = len(suspicious_char_examples[description]) - min(len(suspicious_char_examples[description]), example_limit)
        if extra > 0:
            lines.append(f"  ... {extra:_} more")
    return "\n".join(lines)


def _render_examples(title: str, items: List[VocabItem], limit: int) -> str:
    lines = [title]
    if not items:
        lines.append("(none)")
        return "\n".join(lines)

    for item in items[:limit]:
        lines.append(f"- id={item.token_id}: raw={item.raw_token!r}, decoded={item.decoded_token!r}")

    remaining = len(items) - min(len(items), limit)
    if remaining > 0:
        lines.append(f"- ... {remaining:_} more")
    return "\n".join(lines)


def _fmt_metric(count: int, total: int) -> str:
    return f"{_fmt_count(count)} ({_fmt_pct(count, total)})"


def _fmt_count(value: int) -> str:
    return f"{value:_}"


def _fmt_pct(count: int, total: int) -> str:
    if total <= 0:
        return "n/a"
    return f"{100.0 * count / total:.2f}%"
