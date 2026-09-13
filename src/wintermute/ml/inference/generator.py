from __future__ import annotations

import torch.nn as nn

from wintermute.tools.model import evaluating
from wintermute.ml.tasks.factory import InferenceSession, OutputPreview, TaskWrapper


class Generator:
    def __init__(
        self,
        task_wrapper: TaskWrapper,
        model: nn.Module,
        system_prompt: str | None = None,
    ) -> None:
        self.model = model
        self.task_wrapper = task_wrapper
        self.model.to(self.task_wrapper.device)
        self.inference_session: InferenceSession = self.task_wrapper.start_inference_session(
            self.model,
            system_prompt=system_prompt,
        )

    def generate(self, arguments: str) -> OutputPreview:
        with evaluating(self.model):
            return self.inference_session.infer(arguments)
