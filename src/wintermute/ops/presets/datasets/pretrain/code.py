from __future__ import annotations

from ..builders import the_stack_dataset

CODE_PRETRAIN_DATASET_PRESETS = [
    # Dataset: The Stack Deduplicated (`bigcode/the-stack-dedup`; provider: BigCode).
    # Summary: Permissively licensed source-code repositories deduplicated at repository level.
    # Natural format: classic NTP (source code and Markdown).
    # Orientation: code, shell, SQL, JavaScript, documentation.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large language subset of the 3 TB global corpus; this preset samples 50%.
    the_stack_dataset(
        uid="sto2",
        name="the-stack-dedup-other2",
        languages=["shell", "sql", "javascript", "markdown"],
        data_files=[
            "data/shell/*.parquet",
            "data/sql/*.parquet",
            "data/markdown/*.parquet",
            "data/javascript/*.parquet",
        ],
        output_max_shards=100,
    ),
    # Dataset: The Stack Deduplicated (`bigcode/the-stack-dedup`; provider: BigCode).
    # Summary: Permissively licensed source-code repositories deduplicated at repository level.
    # Natural format: classic NTP (Python source code).
    # Orientation: code, Python.
    # Recommended SFT task spec: not applicable; pretraining-only source.
    # Volume: large language subset of the 3 TB global corpus; this preset samples 50%.
    the_stack_dataset(
        uid="stp2",
        name="the-stack-dedup-python2",
        languages=["python"],
        data_files=["data/python/*.parquet"],
        output_max_shards=200,
    ),
]
