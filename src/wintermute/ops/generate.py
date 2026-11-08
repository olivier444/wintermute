# -*- coding: utf-8 -*-

from pathlib import Path

from wintermute.ml.inference.components_loader import InferenceComponentsLoader
from wintermute.ml.inference.service import InferenceConversationService


def generate(output_root: str, run_id: str) -> None:
    loader = InferenceComponentsLoader.open(Path(output_root), run_id)
    task_wrapper, model = loader.load()

    with InferenceConversationService(
        task_wrapper=task_wrapper,
        model=model,
    ) as service:
        while True:
            arguments = input("[> ")
            result = service.handle_message(
                conversation_id="interactive",
                text=arguments,
            )
            print(result.reply_text)
            print("")
