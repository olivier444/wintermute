from __future__ import annotations

from pathlib import Path

from wintermute.ml.evaluation.lm_eval.runner import LmEvalExecution, LmEvalResult
from wintermute.ml.evaluation.lm_eval.targets import LmEvalTarget


def run_eval(
    output_root: str | Path,
    target: LmEvalTarget,
    lm_eval_config_path: str | Path,
) -> LmEvalResult:
    """Evaluate one configured target with lm-evaluation-harness."""
    return LmEvalExecution.from_yaml(output_root, lm_eval_config_path).run(target)
