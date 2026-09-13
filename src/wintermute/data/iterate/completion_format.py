from __future__ import annotations

import ast
import re
from dataclasses import dataclass


PYTHON_FORMAT_MATCHER = "python"
WORD_FORMAT_MATCHER = "word"
SUPPORTED_FORMAT_MATCHERS = frozenset({PYTHON_FORMAT_MATCHER, WORD_FORMAT_MATCHER})

_PYTHON_FENCE_PATTERN = re.compile(
    r"\s*```(?:python|py)[ \t]*\r?\n(?P<code>.*?)\r?\n```[ \t]*\s*",
    re.IGNORECASE | re.DOTALL,
)
_PYTHON_SHAPE_PATTERN = re.compile(
    r"(?m)^[ \t]*(?:"
    r"@|"
    r"(?:async[ \t]+)?(?:def|for|with)[ \t]+|"
    r"class[ \t]+|"
    r"import[ \t]+|"
    r"from[ \t]+\S+[ \t]+import[ \t]+|"
    r"(?:while|if|try|match|assert|del)\b|"
    r"[^\n#]*(?:\+=|-=|\*=|/=|//=|%=|(?<![=!<>:])=(?!=))"
    r")"
)
_SUBSTANTIAL_PYTHON_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AugAssign,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.Assert,
    ast.Delete,
    ast.Match,
)
_PYTHON_PARSE_ERRORS = (SyntaxError, ValueError, MemoryError, RecursionError)


@dataclass(frozen=True)
class CompletionFormatRule:
    label: str
    pattern: str | None = None
    matcher: str | None = None

    def matches(self, completion: str) -> bool:
        if self.pattern is not None:
            return re.fullmatch(self.pattern, completion) is not None
        if self.matcher == PYTHON_FORMAT_MATCHER:
            return matches_python_completion(completion)
        if self.matcher == WORD_FORMAT_MATCHER:
            return matches_word_completion(completion)
        return self.matcher is None


def matches_python_completion(completion: str) -> bool:
    fenced_match = _PYTHON_FENCE_PATTERN.fullmatch(completion)
    if fenced_match is not None:
        tree = _parse_python(fenced_match.group("code").strip())
        return tree is not None and bool(tree.body)

    candidate = completion.strip()
    if not candidate or "```" in candidate:
        return False
    if _PYTHON_SHAPE_PATTERN.search(candidate) is None:
        return False

    tree = _parse_python(candidate)
    if tree is None:
        return False
    if any(isinstance(node, _SUBSTANTIAL_PYTHON_NODES) for node in ast.walk(tree)):
        return True
    return any(
        isinstance(node, ast.AnnAssign) and node.value is not None
        for node in ast.walk(tree)
    )


def matches_word_completion(completion: str) -> bool:
    return completion.strip().isalpha()


def _parse_python(code: str) -> ast.Module | None:
    if not code:
        return None
    try:
        return ast.parse(code, mode="exec")
    except _PYTHON_PARSE_ERRORS:
        return None
