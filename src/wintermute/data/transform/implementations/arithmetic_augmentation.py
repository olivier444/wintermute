from __future__ import annotations

import re
from random import Random
from typing import Any

from wintermute.data.record import DataRecord
from wintermute.data.transform.base import RecordTransform
from wintermute.data.transform.context import TransformContext
from wintermute.data.transform.implementations.helpers import ensure_allowed_params


_COMMON_PARAMS = {
    "variants_per_record",
    "keep_original",
    "relative_delta",
    "absolute_delta",
    "max_attempts_per_variant",
    "min_value",
}


class ArithmeticTemplateAugmentationRecordTransform(RecordTransform):
    """Generate valid numeric variants of a structured arithmetic template.

    It resamples the ``Numbers`` values, reevaluates the prefix ``Equation``,
    and writes the matching ``Answer``.  Use it for datasets whose arithmetic
    structure is explicit in fields rather than embedded in natural language.
    """

    KIND = "arithmetic_template_augmentation"
    ALLOWED_PARAMS = _COMMON_PARAMS | {"numbers_field", "equation_field", "answer_field"}

    def __init__(self, params: dict[str, Any]) -> None:
        ensure_allowed_params(params, self.ALLOWED_PARAMS, kind=self.KIND)
        self.numbers_field = str(params.get("numbers_field", "Numbers"))
        self.equation_field = str(params.get("equation_field", "Equation"))
        self.answer_field = str(params.get("answer_field", "Answer"))
        self.variants_per_record = max(int(params.get("variants_per_record", 1)), 0)
        self.keep_original = bool(params.get("keep_original", True))
        self.min_value = int(params.get("min_value", 1))
        self.max_attempts_per_variant = max(int(params.get("max_attempts_per_variant", 64)), 1)
        self.relative_delta = max(float(params.get("relative_delta", 0.4)), 0.0)
        self.absolute_delta = max(int(params.get("absolute_delta", 5)), 0)

    def transform(self, record: DataRecord, *, context: TransformContext) -> list[DataRecord]:
        outputs = [record] if self.keep_original else []
        for variant_index, fields in enumerate(
            self._build_variants(record.fields, context.rng),
            start=1,
        ):
            outputs.append(
                DataRecord(
                    record_id=f"{record.record_id}.arithmetic-{context.iteration}-{variant_index}",
                    fields=fields,
                )
            )
        return outputs

    def _build_variants(self, item: dict[str, Any], rng: Random) -> list[dict[str, Any]]:
        if self.variants_per_record <= 0:
            return []

        numbers = item.get(self.numbers_field)
        equation = item.get(self.equation_field)
        if not isinstance(numbers, list) or not isinstance(equation, list):
            return []

        original_numbers = self._coerce_numbers(numbers)
        if original_numbers is None:
            return []

        variants: list[dict[str, Any]] = []
        seen_number_sets: set[tuple[float, ...]] = {tuple(original_numbers)}
        for variant_index in range(self.variants_per_record):
            sampled = self._sample_valid_numbers(original_numbers, equation, seen_number_sets, rng)
            if sampled is None:
                continue

            seen_number_sets.add(tuple(sampled))
            answer = self._evaluate_equation(equation, sampled)
            if answer is None:
                continue

            variant = dict(item)
            variant[self.numbers_field] = [float(value) for value in sampled]
            variant[self.answer_field] = float(answer)
            variant["augmentation_kind"] = "arithmetic_template"
            variant["augmentation_variant"] = variant_index + 1
            variants.append(variant)

        return variants

    def _coerce_numbers(self, raw_numbers: list[Any]) -> list[float] | None:
        result: list[float] = []
        for value in raw_numbers:
            if not isinstance(value, (int, float)):
                return None
            numeric = float(value)
            if numeric <= 0 or not numeric.is_integer():
                return None
            result.append(numeric)
        return result

    def _sample_valid_numbers(
        self,
        original_numbers: list[float],
        equation: list[Any],
        seen_number_sets: set[tuple[float, ...]],
        rng: Random,
    ) -> list[float] | None:
        for _ in range(self.max_attempts_per_variant):
            sampled = [self._sample_number(value, rng) for value in original_numbers]
            if tuple(sampled) in seen_number_sets:
                continue
            if self._evaluate_equation(equation, sampled) is not None:
                return sampled
        return None

    def _sample_number(self, value: float, rng: Random) -> float:
        base = int(value)
        delta = max(int(round(base * self.relative_delta)), self.absolute_delta)
        lower = max(self.min_value, base - delta)
        upper = max(lower, base + delta)
        return float(rng.randint(lower, upper))

    def _evaluate_equation(self, equation: list[Any], numbers: list[float]) -> float | None:
        env = {f"number{index}": value for index, value in enumerate(numbers)}

        def _eval_at(index: int) -> tuple[float | None, int]:
            if index >= len(equation):
                return None, index

            token = equation[index]
            if isinstance(token, str) and token in {"+", "-", "*", "/"}:
                left_value, next_index = _eval_at(index + 1)
                if left_value is None:
                    return None, next_index
                right_value, next_index = _eval_at(next_index)
                if right_value is None:
                    return None, next_index

                if token == "+":
                    return left_value + right_value, next_index
                if token == "-":
                    result = left_value - right_value
                    return (result, next_index) if result >= self.min_value else (None, next_index)
                if token == "*":
                    return left_value * right_value, next_index
                if right_value == 0:
                    return None, next_index
                result = left_value / right_value
                if not result.is_integer() or result < self.min_value:
                    return None, next_index
                return result, next_index

            if isinstance(token, str) and token in env:
                return env[token], index + 1
            if isinstance(token, (int, float)):
                return float(token), index + 1
            return None, index + 1

        result, next_index = _eval_at(0)
        if result is None or next_index != len(equation) or result <= 0 or not result.is_integer():
            return None
        return float(result)


class EquationAnswerArithmeticAugmentationRecordTransform(RecordTransform):
    """Augment word problems by changing numbers in both question and answer.

    It accepts answers formatted as an infix equation followed by ``####`` and
    emits only variants whose recalculated result remains a positive integer.
    Use it when the question text and answer expose the same arithmetic facts.
    """

    KIND = "equation_answer_arithmetic_augmentation"
    ALLOWED_PARAMS = _COMMON_PARAMS | {"question_field", "answer_field"}
    _ANSWER_PATTERN = re.compile(
        r"^\s*(?P<expr>.+?)\s*=\s*(?P<result>[-+]?\d+(?:\.\d+)?)\s*####\s*(?P<final>[-+]?\d+(?:\.\d+)?)\s*$"
    )
    _QUESTION_NUMBER_PATTERN = re.compile(r"\d+")

    def __init__(self, params: dict[str, Any]) -> None:
        ensure_allowed_params(params, self.ALLOWED_PARAMS, kind=self.KIND)
        self.question_field = str(params.get("question_field", "question"))
        self.answer_field = str(params.get("answer_field", "answer"))
        self.variants_per_record = max(int(params.get("variants_per_record", 1)), 0)
        self.keep_original = bool(params.get("keep_original", True))
        self.min_value = int(params.get("min_value", 1))
        self.max_attempts_per_variant = max(int(params.get("max_attempts_per_variant", 128)), 1)
        self.relative_delta = max(float(params.get("relative_delta", 0.4)), 0.0)
        self.absolute_delta = max(int(params.get("absolute_delta", 5)), 0)

    def transform(self, record: DataRecord, *, context: TransformContext) -> list[DataRecord]:
        outputs = [record] if self.keep_original else []
        for variant_index, fields in enumerate(
            self._build_variants(record.fields, context.rng),
            start=1,
        ):
            outputs.append(
                DataRecord(
                    record_id=f"{record.record_id}.equation-arithmetic-{context.iteration}-{variant_index}",
                    fields=fields,
                )
            )
        return outputs

    def _build_variants(self, item: dict[str, Any], rng: Random) -> list[dict[str, Any]]:
        if self.variants_per_record <= 0:
            return []

        question = item.get(self.question_field)
        answer = item.get(self.answer_field)
        if not isinstance(question, str) or not isinstance(answer, str):
            return []

        parsed = self._parse_item(question, answer)
        if parsed is None:
            return []
        expr_tokens, original_numbers, replacement_spans = parsed

        variants: list[dict[str, Any]] = []
        seen_number_sets: set[tuple[int, ...]] = {tuple(original_numbers)}
        for variant_index in range(self.variants_per_record):
            sampled = self._sample_valid_numbers(original_numbers, expr_tokens, seen_number_sets, rng)
            if sampled is None:
                continue

            seen_number_sets.add(tuple(sampled))
            result = self._evaluate_infix_expression(self._render_expr_tokens(expr_tokens, sampled))
            if result is None:
                continue

            variant = dict(item)
            variant[self.question_field] = self._rewrite_question(question, replacement_spans, sampled)
            variant[self.answer_field] = self._build_answer_text(expr_tokens, sampled, result)
            variant["augmentation_kind"] = "equation_answer_arithmetic"
            variant["augmentation_variant"] = variant_index + 1
            variants.append(variant)
        return variants

    def _parse_item(
        self,
        question: str,
        answer: str,
    ) -> tuple[list[str], list[int], list[tuple[int, int]]] | None:
        match = self._ANSWER_PATTERN.match(answer)
        if match is None:
            return None

        tokens = match.group("expr").strip().split()
        if not self._is_safe_expression(tokens):
            return None
        original_numbers = [int(tokens[index]) for index in range(0, len(tokens), 2)]
        if len(set(original_numbers)) != len(original_numbers):
            return None

        result = self._evaluate_infix_expression(match.group("expr").strip())
        if result is None:
            return None
        try:
            answer_result = float(match.group("final"))
        except ValueError:
            return None
        if float(result) != answer_result:
            return None

        spans_by_value = self._locate_question_number_spans(question)
        replacement_spans: list[tuple[int, int]] = []
        for number in original_numbers:
            spans = spans_by_value.get(str(number), [])
            if len(spans) != 1:
                return None
            replacement_spans.append(spans[0])
        return tokens, original_numbers, replacement_spans

    def _locate_question_number_spans(self, question: str) -> dict[str, list[tuple[int, int]]]:
        spans_by_value: dict[str, list[tuple[int, int]]] = {}
        for match in self._QUESTION_NUMBER_PATTERN.finditer(question):
            spans_by_value.setdefault(match.group(0), []).append(match.span())
        return spans_by_value

    def _is_safe_expression(self, tokens: list[str]) -> bool:
        if len(tokens) < 3 or len(tokens) % 2 == 0:
            return False
        for index, token in enumerate(tokens):
            if index % 2 == 0:
                if re.fullmatch(r"\d+", token) is None:
                    return False
            elif token not in {"+", "-", "*", "/"}:
                return False
        return True

    def _sample_valid_numbers(
        self,
        original_numbers: list[int],
        expr_tokens: list[str],
        seen_number_sets: set[tuple[int, ...]],
        rng: Random,
    ) -> list[int] | None:
        for _ in range(self.max_attempts_per_variant):
            sampled = [self._sample_number(value, rng) for value in original_numbers]
            if tuple(sampled) in seen_number_sets:
                continue
            if self._evaluate_infix_expression(self._render_expr_tokens(expr_tokens, sampled)) is not None:
                return sampled
        return None

    def _sample_number(self, value: int, rng: Random) -> int:
        delta = max(int(round(value * self.relative_delta)), self.absolute_delta)
        lower = max(self.min_value, value - delta)
        upper = max(lower, value + delta)
        return rng.randint(lower, upper)

    def _render_expr_tokens(self, expr_tokens: list[str], numbers: list[int]) -> str:
        rendered: list[str] = []
        number_index = 0
        for token in expr_tokens:
            if token in {"+", "-", "*", "/"}:
                rendered.append(token)
            else:
                rendered.append(str(numbers[number_index]))
                number_index += 1
        return " ".join(rendered)

    def _evaluate_infix_expression(self, expr_text: str) -> int | None:
        tokens = expr_text.split()
        if not self._is_safe_expression(tokens):
            return None
        values = [int(tokens[index]) for index in range(0, len(tokens), 2)]
        ops = [tokens[index] for index in range(1, len(tokens), 2)]

        collapsed_values: list[int] = [values[0]]
        collapsed_ops: list[str] = []
        for op, value in zip(ops, values[1:]):
            if op == "*":
                collapsed_values[-1] *= value
            elif op == "/":
                if value == 0 or collapsed_values[-1] % value != 0:
                    return None
                collapsed_values[-1] //= value
                if collapsed_values[-1] < self.min_value:
                    return None
            else:
                collapsed_ops.append(op)
                collapsed_values.append(value)

        result = collapsed_values[0]
        for op, value in zip(collapsed_ops, collapsed_values[1:]):
            if op == "+":
                result += value
            elif op == "-":
                result -= value
                if result < self.min_value:
                    return None
            else:
                return None
        return result if result >= self.min_value else None

    def _rewrite_question(
        self,
        question: str,
        replacement_spans: list[tuple[int, int]],
        sampled_numbers: list[int],
    ) -> str:
        chunks: list[str] = []
        cursor = 0
        for (start, end), sampled in sorted(
            zip(replacement_spans, sampled_numbers),
            key=lambda item: item[0][0],
        ):
            chunks.append(question[cursor:start])
            chunks.append(str(sampled))
            cursor = end
        chunks.append(question[cursor:])
        return "".join(chunks)

    def _build_answer_text(self, expr_tokens: list[str], sampled_numbers: list[int], result: int) -> str:
        expr_text = self._render_expr_tokens(expr_tokens, sampled_numbers)
        return f"{expr_text} = {result} #### {result}"
