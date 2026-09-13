from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence


_MIN_POINTS = 20
_MIN_TOKEN_RATIO = 2.0
_LOCAL_POINT_COUNT = 15
_PROFILE_GRID_SIZE = 64
_PROFILE_ITERATIONS = 64


@dataclass(frozen=True)
class ScalingLawFit:
    """A log-space fit of ``L(N) = l_inf + a * N**(-alpha)``."""

    l_inf: float
    a: float
    alpha: float
    fit_r2: float
    predicted_loss_x2_tokens: float
    alpha_local: float | None

    def metrics(self) -> dict[str, float]:
        result = {
            "scaling/l_inf": self.l_inf,
            "scaling/alpha": self.alpha,
            "scaling/a": self.a,
            "scaling/fit_r2": self.fit_r2,
            "scaling/predicted_loss_x2_tokens": self.predicted_loss_x2_tokens,
        }
        if self.alpha_local is not None:
            result["scaling/alpha_local"] = self.alpha_local
        return result


@dataclass(frozen=True)
class _ProfileResult:
    l_inf: float
    log_a: float
    alpha: float
    sse: float
    r2: float


class ScalingLawTracker:
    """Keeps evaluation history and safely produces optional scaling diagnostics."""

    def __init__(self) -> None:
        self.history: list[tuple[float, float]] = []

    def record(
        self,
        *,
        cumulative_tokens: float,
        eval_loss: float,
    ) -> ScalingLawFit | None:
        try:
            point = (float(cumulative_tokens), float(eval_loss))
            if not _is_valid_point(point):
                return None
            self.history.append(point)
            return fit_scaling_law(self.history)
        except Exception:
            # Diagnostics must remain observational and never interrupt training.
            return None


def fit_scaling_law(
    history: Sequence[tuple[float, float]],
    *,
    min_points: int = _MIN_POINTS,
    min_token_ratio: float = _MIN_TOKEN_RATIO,
) -> ScalingLawFit | None:
    """Profile-fit a power-law loss curve, returning ``None`` when unsuitable.

    For each candidate irreducible loss, the remaining two parameters are found
    by ordinary least squares in log space.  The sole numerical search is the
    bounded one-dimensional profile over ``l_inf``.
    """

    try:
        points = [(float(tokens), float(loss)) for tokens, loss in history]
        if len(points) < min_points or any(not _is_valid_point(point) for point in points):
            return None

        min_tokens = min(tokens for tokens, _ in points)
        max_tokens = max(tokens for tokens, _ in points)
        if max_tokens / min_tokens < min_token_ratio:
            return None

        min_loss = min(loss for _, loss in points)
        lower = min_loss * 1e-12
        upper = math.nextafter(min_loss, 0.0)
        if not 0.0 < lower < upper:
            return None

        best = _minimize_profile(points, lower=lower, upper=upper)
        if best is None:
            return None

        a = math.exp(best.log_a)
        current_tokens = points[-1][0]
        predicted_loss = best.l_inf + a * (2.0 * current_tokens) ** (-best.alpha)
        if not all(math.isfinite(value) for value in (a, predicted_loss, best.r2)):
            return None

        local = _fit_profile_at_l_inf(points[-_LOCAL_POINT_COUNT:], best.l_inf)
        return ScalingLawFit(
            l_inf=best.l_inf,
            a=a,
            alpha=best.alpha,
            fit_r2=best.r2,
            predicted_loss_x2_tokens=predicted_loss,
            # The regression slope is -alpha, so expose the positive exponent.
            alpha_local=local.alpha if local is not None else None,
        )
    except (ArithmeticError, OverflowError, TypeError, ValueError):
        return None


def _is_valid_point(point: tuple[float, float]) -> bool:
    tokens, loss = point
    return tokens > 0.0 and loss > 0.0 and math.isfinite(tokens) and math.isfinite(loss)


def _minimize_profile(
    points: list[tuple[float, float]],
    *,
    lower: float,
    upper: float,
) -> _ProfileResult | None:
    grid = [lower + (upper - lower) * index / _PROFILE_GRID_SIZE for index in range(_PROFILE_GRID_SIZE + 1)]
    evaluated = [_fit_profile_at_l_inf(points, l_inf) for l_inf in grid]
    valid = [(index, result) for index, result in enumerate(evaluated) if result is not None]
    if not valid:
        return None

    best_index, best = min(valid, key=lambda item: item[1].sse)
    left = grid[max(0, best_index - 1)]
    right = grid[min(_PROFILE_GRID_SIZE, best_index + 1)]
    candidates = [result for _, result in valid]

    # Golden-section refinement of the best finite interval found by the scan.
    golden = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = right - golden * (right - left)
    x2 = left + golden * (right - left)
    f1 = _fit_profile_at_l_inf(points, x1)
    f2 = _fit_profile_at_l_inf(points, x2)
    for _ in range(_PROFILE_ITERATIONS):
        score1 = f1.sse if f1 is not None else math.inf
        score2 = f2.sse if f2 is not None else math.inf
        if score1 <= score2:
            right, x2, f2 = x2, x1, f1
            x1 = right - golden * (right - left)
            f1 = _fit_profile_at_l_inf(points, x1)
        else:
            left, x1, f1 = x1, x2, f2
            x2 = left + golden * (right - left)
            f2 = _fit_profile_at_l_inf(points, x2)
    candidates.extend(result for result in (f1, f2) if result is not None)
    return min(candidates, key=lambda result: result.sse)


def _fit_profile_at_l_inf(
    points: Sequence[tuple[float, float]],
    l_inf: float,
) -> _ProfileResult | None:
    if not math.isfinite(l_inf) or l_inf <= 0.0:
        return None
    try:
        xs = [math.log(tokens) for tokens, _ in points]
        ys = [math.log(loss - l_inf) for _, loss in points]
    except (ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        return None

    x_mean = math.fsum(xs) / len(xs)
    y_mean = math.fsum(ys) / len(ys)
    sum_xx = math.fsum((x - x_mean) ** 2 for x in xs)
    if sum_xx <= 0.0 or not math.isfinite(sum_xx):
        return None
    slope = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / sum_xx
    alpha = -slope
    if alpha <= 0.0 or not math.isfinite(alpha):
        return None

    log_a = y_mean - slope * x_mean
    residuals = [y - (log_a - alpha * x) for x, y in zip(xs, ys)]
    sse = math.fsum(residual * residual for residual in residuals)
    total = math.fsum((y - y_mean) ** 2 for y in ys)
    if not math.isfinite(sse) or total <= 0.0 or not math.isfinite(total):
        return None
    r2 = 1.0 - sse / total
    if not math.isfinite(r2):
        return None
    return _ProfileResult(
        l_inf=l_inf,
        log_a=log_a,
        alpha=alpha,
        sse=sse,
        r2=r2,
    )
