# -*- coding: utf-8 -*-

from pathlib import Path

from wintermute.ml.inference.workspace import InferenceWorkspace


def generate(output_root: str, run_id: str) -> None:
    with InferenceWorkspace.open(Path(output_root), run_id) as workspace:
        generator = workspace.create_generator(system_prompt="")

        while True:
            arguments = input("[> ")
            reply = generator.generate(arguments).as_text()
            print(f"{reply}")
            print("")
