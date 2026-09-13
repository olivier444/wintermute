from __future__ import annotations

import ast
import math
import operator
from dataclasses import dataclass
from typing import Mapping, TypeAlias, cast

from wintermute.ml.tasks.implementations.causal_lm.probes.verifiers.base import Verifier, extract_fenced_code


PythonScalar: TypeAlias = None | bool | int | float | str
PythonValue: TypeAlias = (
    PythonScalar
    | list["PythonValue"]
    | tuple["PythonValue", ...]
    | dict["PythonValue", "PythonValue"]
)


@dataclass(frozen=True)
class PythonExpression(Verifier):
    cases: tuple[tuple[Mapping[str, PythonValue], PythonValue], ...]
    allow_comments: bool = True

    def __post_init__(self) -> None:
        if not self.cases:
            raise ValueError("PythonExpression requires at least one test case")
        for variables, expected in self.cases:
            if any(not name.isidentifier() for name in variables):
                raise ValueError("PythonExpression variable names must be identifiers")
            for value in (*variables.values(), expected):
                _validate_python_value(value)

    def score(self, completion: str) -> float:
        source = completion.strip()
        tree = _parse_python_expression(source)
        if tree is None and self.allow_comments:
            tree = _parse_single_return_function(source)
            if tree is None:
                fenced_source = extract_fenced_code(completion, language="python")
                if fenced_source is not None:
                    fenced_source = fenced_source.strip()
                    tree = _parse_python_expression(fenced_source)
                    if tree is None:
                        tree = _parse_single_return_function(fenced_source)
        if tree is None:
            return 0.0
        if sum(1 for _ in ast.walk(tree)) > _PYTHON_MAX_AST_NODES:
            return 0.0

        passed = 0
        for variables, expected in self.cases:
            try:
                actual = _PythonExpressionInterpreter(variables).evaluate(tree)
            except (
                _PythonExpressionError,
                ArithmeticError,
                IndexError,
                KeyError,
                TypeError,
                ValueError,
            ):
                continue
            if _python_values_equal(actual, expected):
                passed += 1
        return passed / len(self.cases)


def _parse_python_expression(source: str) -> ast.Expression | None:
    if not source or len(source) > _PYTHON_MAX_SOURCE_CHARS:
        return None
    try:
        return ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError):
        return None


def _parse_single_return_function(source: str) -> ast.Expression | None:
    if not source or len(source) > _PYTHON_MAX_SOURCE_CHARS:
        return None
    try:
        module = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError):
        return None
    if len(module.body) != 1 or not isinstance(module.body[0], ast.FunctionDef):
        return None
    function = module.body[0]
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        return None
    if function.body[0].value is None:
        return None
    return ast.Expression(body=function.body[0].value)


_PYTHON_MAX_SOURCE_CHARS = 10_000
_PYTHON_MAX_AST_NODES = 256
_PYTHON_MAX_EVALUATION_STEPS = 10_000
_PYTHON_MAX_COLLECTION_ITEMS = 1_000
_PYTHON_MAX_STRING_CHARS = 10_000
_PYTHON_MAX_ABS_NUMBER = 1_000_000_000


class _PythonExpressionError(ValueError):
    pass


def _validate_python_value(value: object) -> PythonValue:
    if value is None or type(value) is bool:
        return cast(PythonValue, value)
    if type(value) is int:
        if abs(value) > _PYTHON_MAX_ABS_NUMBER:
            raise ValueError("PythonExpression integers exceed the supported range")
        return value
    if type(value) is float:
        if not math.isfinite(value) or abs(value) > _PYTHON_MAX_ABS_NUMBER:
            raise ValueError("PythonExpression floats must be finite and bounded")
        return value
    if type(value) is str:
        if len(value) > _PYTHON_MAX_STRING_CHARS:
            raise ValueError("PythonExpression strings exceed the supported length")
        return value
    if type(value) in (list, tuple):
        sequence = cast(list[object] | tuple[object, ...], value)
        if len(sequence) > _PYTHON_MAX_COLLECTION_ITEMS:
            raise ValueError("PythonExpression collections exceed the supported length")
        for item in sequence:
            _validate_python_value(item)
        return cast(PythonValue, value)
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if len(mapping) > _PYTHON_MAX_COLLECTION_ITEMS:
            raise ValueError("PythonExpression mappings exceed the supported length")
        for key, item in mapping.items():
            _validate_python_value(key)
            _validate_python_value(item)
        return cast(PythonValue, value)
    raise ValueError(f"PythonExpression does not support values of type {type(value).__name__}")


def _python_values_equal(actual: PythonValue, expected: PythonValue) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(actual) in (list, tuple):
        actual_sequence = cast(list[PythonValue] | tuple[PythonValue, ...], actual)
        expected_sequence = cast(list[PythonValue] | tuple[PythonValue, ...], expected)
        return len(actual_sequence) == len(expected_sequence) and all(
            _python_values_equal(actual_item, expected_item)
            for actual_item, expected_item in zip(actual_sequence, expected_sequence)
        )
    if type(actual) is dict:
        actual_mapping = cast(dict[PythonValue, PythonValue], actual)
        expected_mapping = cast(dict[PythonValue, PythonValue], expected)
        if len(actual_mapping) != len(expected_mapping):
            return False
        unmatched_items = list(actual_mapping.items())
        for expected_key, expected_value in expected_mapping.items():
            for index, (actual_key, actual_value) in enumerate(unmatched_items):
                if _python_values_equal(actual_key, expected_key):
                    if not _python_values_equal(actual_value, expected_value):
                        return False
                    unmatched_items.pop(index)
                    break
            else:
                return False
        return True
    return actual == expected


class _PythonExpressionInterpreter:
    def __init__(self, variables: Mapping[str, PythonValue]) -> None:
        self.variables = dict(variables)
        self.remaining_steps = _PYTHON_MAX_EVALUATION_STEPS

    def evaluate(self, tree: ast.Expression) -> PythonValue:
        value = self._evaluate(tree.body, self.variables)
        return _validate_python_value(value)

    def _evaluate(self, node: ast.expr, variables: Mapping[str, PythonValue]) -> PythonValue:
        self._consume_step()
        if isinstance(node, ast.Constant):
            return _validate_python_value(node.value)
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise _PythonExpressionError(f"Unknown variable: {node.id}")
            return variables[node.id]
        if isinstance(node, ast.List):
            return self._bounded([self._evaluate(item, variables) for item in node.elts])
        if isinstance(node, ast.Tuple):
            return self._bounded(tuple(self._evaluate(item, variables) for item in node.elts))
        if isinstance(node, ast.Dict):
            if any(key is None for key in node.keys):
                raise _PythonExpressionError("Dictionary unpacking is not supported")
            return self._bounded(
                {
                    self._evaluate(key, variables): self._evaluate(value, variables)
                    for key, value in zip(node.keys, node.values)
                    if key is not None
                }
            )
        if isinstance(node, ast.UnaryOp):
            return self._evaluate_unary(node, variables)
        if isinstance(node, ast.BinOp):
            return self._evaluate_binary(node, variables)
        if isinstance(node, ast.BoolOp):
            return self._evaluate_boolean(node, variables)
        if isinstance(node, ast.Compare):
            return self._evaluate_comparison(node, variables)
        if isinstance(node, ast.IfExp):
            branch = node.body if self._evaluate(node.test, variables) else node.orelse
            return self._evaluate(branch, variables)
        if isinstance(node, ast.Subscript):
            return self._evaluate_subscript(node, variables)
        if isinstance(node, ast.Call):
            return self._evaluate_call(node, variables)
        if isinstance(node, ast.ListComp):
            return self._bounded(
                [self._evaluate(node.elt, scope) for scope in self._comprehension_scopes(node.generators, variables)]
            )
        if isinstance(node, ast.DictComp):
            return self._bounded(
                {
                    self._evaluate(node.key, scope): self._evaluate(node.value, scope)
                    for scope in self._comprehension_scopes(node.generators, variables)
                }
            )
        if isinstance(node, ast.GeneratorExp):
            return self._bounded(
                tuple(
                    self._evaluate(node.elt, scope)
                    for scope in self._comprehension_scopes(node.generators, variables)
                )
            )
        raise _PythonExpressionError(f"Unsupported syntax: {type(node).__name__}")

    def _evaluate_unary(
        self,
        node: ast.UnaryOp,
        variables: Mapping[str, PythonValue],
    ) -> PythonValue:
        operand = self._evaluate(node.operand, variables)
        if isinstance(node.op, ast.Not):
            return not operand
        if type(operand) not in (int, float):
            raise _PythonExpressionError("Unary arithmetic requires a number")
        number = cast(int | float, operand)
        if isinstance(node.op, ast.UAdd):
            return self._bounded(operator.pos(number))
        if isinstance(node.op, ast.USub):
            return self._bounded(operator.neg(number))
        raise _PythonExpressionError(f"Unsupported unary operator: {type(node.op).__name__}")

    def _evaluate_binary(
        self,
        node: ast.BinOp,
        variables: Mapping[str, PythonValue],
    ) -> PythonValue:
        left = self._evaluate(node.left, variables)
        right = self._evaluate(node.right, variables)
        if type(left) not in (int, float) or type(right) not in (int, float):
            raise _PythonExpressionError("Binary arithmetic requires numbers")
        left_number = cast(int | float, left)
        right_number = cast(int | float, right)
        if isinstance(node.op, ast.Add):
            result = operator.add(left_number, right_number)
        elif isinstance(node.op, ast.Sub):
            result = operator.sub(left_number, right_number)
        elif isinstance(node.op, ast.Mult):
            result = operator.mul(left_number, right_number)
        elif isinstance(node.op, ast.Div):
            result = operator.truediv(left_number, right_number)
        elif isinstance(node.op, ast.FloorDiv):
            result = operator.floordiv(left_number, right_number)
        elif isinstance(node.op, ast.Mod):
            result = operator.mod(left_number, right_number)
        else:
            raise _PythonExpressionError(f"Unsupported binary operator: {type(node.op).__name__}")
        return self._bounded(result)

    def _evaluate_boolean(
        self,
        node: ast.BoolOp,
        variables: Mapping[str, PythonValue],
    ) -> PythonValue:
        value = self._evaluate(node.values[0], variables)
        for operand in node.values[1:]:
            if isinstance(node.op, ast.And):
                if not value:
                    return value
            elif isinstance(node.op, ast.Or):
                if value:
                    return value
            else:
                raise _PythonExpressionError(f"Unsupported boolean operator: {type(node.op).__name__}")
            value = self._evaluate(operand, variables)
        return value

    def _evaluate_comparison(
        self,
        node: ast.Compare,
        variables: Mapping[str, PythonValue],
    ) -> bool:
        left = self._evaluate(node.left, variables)
        for operator, comparator in zip(node.ops, node.comparators):
            right = self._evaluate(comparator, variables)
            if not self._compare(operator, left, right):
                return False
            left = right
        return True

    def _compare(self, comparison: ast.cmpop, left: PythonValue, right: PythonValue) -> bool:
        if isinstance(comparison, ast.Eq):
            return left == right
        if isinstance(comparison, ast.NotEq):
            return left != right
        if isinstance(comparison, ast.Lt):
            return self._ordered_compare(comparison, left, right)
        if isinstance(comparison, ast.LtE):
            return self._ordered_compare(comparison, left, right)
        if isinstance(comparison, ast.Gt):
            return self._ordered_compare(comparison, left, right)
        if isinstance(comparison, ast.GtE):
            return self._ordered_compare(comparison, left, right)
        if isinstance(comparison, ast.In):
            return operator.contains(self._as_builtin_iterable(right), left)
        if isinstance(comparison, ast.NotIn):
            return not operator.contains(self._as_builtin_iterable(right), left)
        if isinstance(comparison, ast.Is):
            return left is right
        if isinstance(comparison, ast.IsNot):
            return left is not right
        raise _PythonExpressionError(f"Unsupported comparison: {type(comparison).__name__}")

    def _evaluate_subscript(
        self,
        node: ast.Subscript,
        variables: Mapping[str, PythonValue],
    ) -> PythonValue:
        value = self._evaluate(node.value, variables)
        if type(value) not in (str, list, tuple, dict):
            raise _PythonExpressionError("Subscripting is limited to built-in containers")
        index = (
            self._evaluate_slice(node.slice, variables)
            if isinstance(node.slice, ast.Slice)
            else self._evaluate(node.slice, variables)
        )
        if type(value) is str:
            if type(index) is not int and not isinstance(index, slice):
                raise _PythonExpressionError("String indices must be integers or slices")
            return self._bounded(cast(str, value)[index])
        if type(value) is list:
            if type(index) is not int and not isinstance(index, slice):
                raise _PythonExpressionError("List indices must be integers or slices")
            return self._bounded(cast(list[PythonValue], value)[index])
        if type(value) is tuple:
            if type(index) is not int and not isinstance(index, slice):
                raise _PythonExpressionError("Tuple indices must be integers or slices")
            return self._bounded(cast(tuple[PythonValue, ...], value)[index])
        if isinstance(index, slice):
            raise _PythonExpressionError("Mappings do not support slice indices")
        return self._bounded(cast(dict[PythonValue, PythonValue], value)[index])

    def _evaluate_slice(
        self,
        node: ast.Slice,
        variables: Mapping[str, PythonValue],
    ) -> slice:
        return slice(
            None if node.lower is None else self._evaluate(node.lower, variables),
            None if node.upper is None else self._evaluate(node.upper, variables),
            None if node.step is None else self._evaluate(node.step, variables),
        )

    def _evaluate_call(
        self,
        node: ast.Call,
        variables: Mapping[str, PythonValue],
    ) -> PythonValue:
        if node.keywords:
            raise _PythonExpressionError("Keyword arguments are not supported")
        arguments = [self._evaluate(argument, variables) for argument in node.args]
        if isinstance(node.func, ast.Name):
            if node.func.id == "len" and len(arguments) == 1:
                return self._bounded(len(self._as_builtin_iterable(arguments[0])))
            if node.func.id == "any" and len(arguments) == 1:
                return any(self._as_builtin_iterable(arguments[0]))
            if node.func.id == "all" and len(arguments) == 1:
                return all(self._as_builtin_iterable(arguments[0]))
            raise _PythonExpressionError(f"Unsupported function: {node.func.id}")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            receiver = self._evaluate(node.func.value, variables)
            if type(receiver) is dict and len(arguments) in (1, 2):
                mapping = cast(dict[PythonValue, PythonValue], receiver)
                return self._bounded(mapping.get(*arguments))
        raise _PythonExpressionError("Unsupported method call")

    def _comprehension_scopes(
        self,
        generators: list[ast.comprehension],
        variables: Mapping[str, PythonValue],
    ) -> list[dict[str, PythonValue]]:
        scopes = [dict(variables)]
        for generator in generators:
            if generator.is_async:
                raise _PythonExpressionError("Async comprehensions are not supported")
            nested_scopes: list[dict[str, PythonValue]] = []
            for scope in scopes:
                iterable = self._as_builtin_iterable(self._evaluate(generator.iter, scope))
                for item in iterable:
                    self._consume_step()
                    nested_scope = dict(scope)
                    self._bind_target(generator.target, item, nested_scope)
                    if all(self._evaluate(condition, nested_scope) for condition in generator.ifs):
                        nested_scopes.append(nested_scope)
                        if len(nested_scopes) > _PYTHON_MAX_COLLECTION_ITEMS:
                            raise _PythonExpressionError("Comprehension result is too large")
            scopes = nested_scopes
        return scopes

    def _bind_target(
        self,
        target: ast.expr,
        value: PythonValue,
        variables: dict[str, PythonValue],
    ) -> None:
        if isinstance(target, ast.Name):
            variables[target.id] = value
            return
        if isinstance(target, (ast.Tuple, ast.List)) and type(value) in (tuple, list):
            sequence = cast(list[PythonValue] | tuple[PythonValue, ...], value)
            if len(target.elts) != len(sequence):
                raise _PythonExpressionError("Cannot unpack comprehension value")
            for nested_target, nested_value in zip(target.elts, sequence):
                self._bind_target(nested_target, nested_value, variables)
            return
        raise _PythonExpressionError("Unsupported comprehension target")

    def _as_builtin_iterable(
        self,
        value: PythonValue,
    ) -> str | list[PythonValue] | tuple[PythonValue, ...] | dict[PythonValue, PythonValue]:
        if type(value) not in (str, list, tuple, dict):
            raise _PythonExpressionError("A built-in iterable is required")
        return cast(
            str | list[PythonValue] | tuple[PythonValue, ...] | dict[PythonValue, PythonValue],
            value,
        )

    def _ordered_compare(
        self,
        comparison: ast.cmpop,
        left: PythonValue,
        right: PythonValue,
    ) -> bool:
        if type(left) in (int, float) and type(right) in (int, float):
            comparable_left = cast(int | float, left)
            comparable_right = cast(int | float, right)
            if isinstance(comparison, ast.Lt):
                return comparable_left < comparable_right
            if isinstance(comparison, ast.LtE):
                return comparable_left <= comparable_right
            if isinstance(comparison, ast.Gt):
                return comparable_left > comparable_right
            if isinstance(comparison, ast.GtE):
                return comparable_left >= comparable_right
        if type(left) is str and type(right) is str:
            if isinstance(comparison, ast.Lt):
                return left < right
            if isinstance(comparison, ast.LtE):
                return left <= right
            if isinstance(comparison, ast.Gt):
                return left > right
            if isinstance(comparison, ast.GtE):
                return left >= right
        raise _PythonExpressionError("Ordered comparisons require compatible scalars")

    def _bounded(self, value: object) -> PythonValue:
        try:
            return _validate_python_value(value)
        except ValueError as error:
            raise _PythonExpressionError(str(error)) from error

    def _consume_step(self) -> None:
        self.remaining_steps -= 1
        if self.remaining_steps < 0:
            raise _PythonExpressionError("PythonExpression evaluation budget exceeded")
