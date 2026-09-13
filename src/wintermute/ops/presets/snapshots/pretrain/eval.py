from __future__ import annotations

from ..builders import text_src

from ...model import DedupConfig, SnapshotPreset

EVAL_PRETRAIN_SNAPSHOT_PRESETS = [
    # Snapshot: French Text Evaluation (`french-txt`).
    # Purpose: Hold-out evaluation of French pretraining text.
    # Output format: NTP text.
    # Composition: FineWeb2-HQ French and French Wikipedia.
    # Budget: 10M eval chars.
    # Selection: Deduplicated weighted mixture; no training split.
    SnapshotPreset(
        name="french-txt",
        snapshot_id="french-txt",
        split_sizes_chars={
            "eval": int(1e7),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2027,
        source_configs={
            # This evaluation was materialized without excluding Mix 40B Text; it should have been.
            "fw2-fr": text_src(  # epfml/FineWeb2-HQ (high-quality French web)
                5.0,
                oversampling=1,
                # excluded_snapshot_ids=["mix40b-txt"],
            ),
            "wk-fr": text_src(  # wikimedia/wikipedia (French encyclopedia)
                1.2,
                oversampling=1,
                # excluded_snapshot_ids=["mix40b-txt"],
            ),
        },
    ),    
    # Snapshot: Code Text Evaluation (`code-txt`).
    # Purpose: Hold-out evaluation of code modeling.
    # Output format: NTP text.
    # Composition: Deduplicated Python and other code from The Stack.
    # Budget: 10M eval chars.
    # Selection: Deduplicated weighted mixture; source records must contain at least 200 characters.
    SnapshotPreset(
        name="code-txt",
        snapshot_id="code-txt",
        split_sizes_chars={
            "eval": int(1e7)
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2027,
        source_configs={
            # This evaluation was materialized without excluding Mix 40B Text; it should have been.
            "stp2": text_src(  # bigcode/the-stack-dedup (Python code)
                3.0,
                field="content",
                oversampling=1,
                min_chars=200,
                # excluded_snapshot_ids=["mix40b-txt"],
            ),
            "sto2": text_src(  # bigcode/the-stack-dedup (multi-language code)
                1.0,
                field="content",
                oversampling=1,
                min_chars=200,
                # excluded_snapshot_ids=["mix40b-txt"],
            ),
        },
    ),
    # Snapshot: DCLM Text Evaluation (`dclm-eval-txt`).
    # Purpose: Hold-out evaluation on the DCLM distribution used by continued pretraining.
    # Output format: NTP text.
    # Composition: DCLM Baseline English web only.
    # Budget: 10M eval chars.
    # Selection: Deduplicated and explicitly excludes records already used by the DCLM training snapshot.
    SnapshotPreset(
        name="dclm-eval-txt",
        snapshot_id="dclm-eval-txt",
        split_sizes_chars={
            "eval": int(1e7),
        },
        dedup=DedupConfig(),
        excluded_languages=[],
        seed=2741,
        source_configs={
            "dclm": text_src(
                1.0,
                oversampling=1,
                excluded_snapshot_ids=["dclm-baseline-20b"],
            ),
        },
    ),
]
