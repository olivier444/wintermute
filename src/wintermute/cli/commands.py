# -*- coding: utf-8 -*-

import argparse
from wintermute.ops.analyze import run_analyze_training_config
from wintermute.ops.audit_tokenizer import run_audit_tokenizer
from wintermute.ops.create_index import run_create_index_group
from wintermute.ops.create_snapshot import run_create_snapshot
from wintermute.ml.evaluation.lm_eval import LmEvalTarget
from wintermute.ops.evaluate import run_eval
from wintermute.ops.inference_endpoint import start_inference_endpoint as run_start_inference_endpoint
from wintermute.ops.get_raw import run_get_mraw
from wintermute.ops.train_tokenizer import run_train_tokenizer
from wintermute.ops.train_model import run_fork, run_new, run_resume
from wintermute.ops.generate import generate as run_generate
from wintermute.ops.materialized_stats import ESTIMATED_CHARS_PER_TOKEN as MATERIALIZED_STATS_ESTIMATED_CHARS_PER_TOKEN
from wintermute.ops.materialized_stats import run_materialized_stats
from wintermute.ops.write_signatures import run_write_signatures
from wintermute.ops.presets import (
    get_dataset_preset,
    get_dataset_presets,    
    get_snapshot_presets,
    get_train_tokenizer_preset,
    list_dataset_presets,
)


def cmd_get_raw(args: argparse.Namespace) -> None:
    if not args.preset:
        raise SystemExit("preset is required")
    if not args.output_root:
        raise SystemExit("output_root is required")
    presets = get_dataset_presets(args.preset)
    run_get_mraw(presets, args.output_root)


def cmd_write_signatures(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if args.preset:
        presets = [get_dataset_preset(args.preset)]
    else:
        presets = list(list_dataset_presets().values())
    for preset in presets:
        run_write_signatures(preset, args.output_root)


def cmd_create_index(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if args.preset:
        presets = [get_dataset_preset(args.preset)]
    else:
        presets = list(list_dataset_presets().values())
    run_create_index_group(presets, args.output_root)


def cmd_create_snapshot(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.preset:
        raise SystemExit("snapshot preset is required")
    for preset in get_snapshot_presets(args.preset):
        run_create_snapshot(preset, args.output_root)


def cmd_materialized_stats(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.snapshot_id:
        raise SystemExit("snapshot_id is required")
    if args.chars_per_token <= 0:
        raise SystemExit("chars_per_token must be > 0")
    if args.top_sources == 0:
        raise SystemExit("top_sources cannot be 0; use a negative value to show all rows")

    run_materialized_stats(
        output_root=args.output_root,
        snapshot_id=args.snapshot_id,
        split=args.split,
        top_sources=args.top_sources,
        chars_per_token=args.chars_per_token,
    )


def cmd_analyze(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.training_config:
        raise SystemExit("training_config is required")
    if args.top_sources == 0:
        raise SystemExit("top_sources cannot be 0; use a negative value to show all rows")

    run_analyze_training_config(
        output_root=args.output_root,
        training_config_path=args.training_config,
        top_sources=args.top_sources,
    )


def cmd_audit_tokenizer(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit('output_root is required')
    if not args.tokenizer:
        raise SystemExit('tokenizer is required')
    if args.example_limit < 0:
        raise SystemExit('example_limit must be >= 0')

    run_audit_tokenizer(
        output_root=args.output_root,
        tokenizer_name=args.tokenizer,
        example_limit=args.example_limit,
    )


def cmd_train_tokenizer(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.preset:
        raise SystemExit("train tokenizer preset is required")
    preset = get_train_tokenizer_preset(args.preset)
    run_train_tokenizer(preset, args.output_root)


def cmd_run_new(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.training_config:
        raise SystemExit("training_config is required")
    run_id = run_new(args.output_root, args.training_config)
    print(run_id)


def cmd_resume(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.run_id:
        raise SystemExit("run_id is required")
    run_resume(args.output_root, args.run_id, args.checkpoint)


def cmd_fork(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.run_id:
        raise SystemExit("run_id is required")
    if not args.to:
        raise SystemExit("to is required")
    run_id = run_fork(args.output_root, args.run_id, args.to, args.checkpoint)
    print(run_id)


def cmd_eval(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    run_ids = _normalize_run_ids(args.run_id)
    hf_models = _normalize_run_ids(args.hf_model)
    if not run_ids and not hf_models:
        raise SystemExit("at least one of run_id or hf_model is required")
    if not args.lm_eval_config:
        raise SystemExit("lm_eval_config is required")
    for run_id in run_ids:
        evaluation = run_eval(
            args.output_root,
            LmEvalTarget.from_run(run_id, args.checkpoint),
            args.lm_eval_config,
        )
        print(evaluation.results_path.as_posix())
    for model_name in hf_models:
        evaluation = run_eval(
            args.output_root,
            LmEvalTarget.from_huggingface(model_name),
            args.lm_eval_config,
        )
        print(evaluation.results_path.as_posix())


def _normalize_run_ids(raw_run_ids: object) -> list[str]:
    if raw_run_ids is None:
        return []
    values = raw_run_ids if isinstance(raw_run_ids, list) else [raw_run_ids]
    return [
        run_id
        for value in values
        for run_id in (part.strip() for part in str(value).split(","))
        if run_id
    ]


def cmd_generate(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.run_id:
        raise SystemExit("run_id is required")
    run_generate(args.output_root, args.run_id)


def cmd_start_inference_endpoint(args: argparse.Namespace) -> None:
    if not args.output_root:
        raise SystemExit("output_root is required")
    if not args.endpoint_config:
        raise SystemExit("endpoint_config is required")
    run_start_inference_endpoint(args.output_root, args.endpoint_config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wintermute", description="Wintermute CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_snap = sub.add_parser("get_raw", help="download/stream HF dataset")
    p_snap.add_argument(
        "--preset",
        required=True,
        help="preset name (wikipedia, opensubtitles, fineweb-edu, the-stack-dedup, pg19, scientific-papers-arxiv) or comma separated list of preset names",
    )
    p_snap.add_argument("--output-root", required=True, help="base output root")
    p_snap.set_defaults(func=cmd_get_raw)

    p_sig = sub.add_parser(
        "write-signatures",
        aliases=["write_signatures"],
        help="write minhash signatures parquet from raw data files",
    )
    p_sig.add_argument("--preset", required=False, help="preset name (default: all presets)")
    p_sig.add_argument("--output-root", required=True, help="base output root")
    p_sig.set_defaults(func=cmd_write_signatures)

    p_index = sub.add_parser(
        "create-index",
        aliases=["create_index"],
        help="create index parquet tables from raw data files + signatures",
    )
    p_index.add_argument("--preset", required=False, help="preset name (default: all presets)")
    p_index.add_argument("--output-root", required=True, help="base output root")
    p_index.set_defaults(func=cmd_create_index)

    p_snapshot = sub.add_parser(
        "create-snapshot",
        aliases=["create_snapshot"],
        help="create snapshot tables",
    )
    p_snapshot.add_argument(
        "--preset",
        required=True,
        help="snapshot preset name or comma-separated list of snapshot preset names",
    )
    p_snapshot.add_argument("--output-root", required=True, help="base output root")
    p_snapshot.set_defaults(func=cmd_create_snapshot)

    p_materialized_stats = sub.add_parser(
        "materialized-stats",
        aliases=["materialized_stats"],
        help="describe a snapshot-backed materialized dataset from index data without scanning materialized jsonl files",
    )
    p_materialized_stats.add_argument("--output-root", required=True, help="base output root")
    p_materialized_stats.add_argument("--snapshot-id", required=True, help="snapshot id / materialized dataset id")
    p_materialized_stats.add_argument("--split", help="optional split filter (default: all splits)")
    p_materialized_stats.add_argument(
        "--top-sources",
        type=int,
        default=20,
        help="max number of source rows (per split when several splits are shown); use a negative value for all rows",
    )
    p_materialized_stats.add_argument(
        "--chars-per-token",
        type=float,
        default=MATERIALIZED_STATS_ESTIMATED_CHARS_PER_TOKEN,
        help="rough token estimate heuristic (estimated_tokens ~= chars / value)",
    )
    p_materialized_stats.set_defaults(func=cmd_materialized_stats)

    p_analyze = sub.add_parser(
        "analyze",
        help="analyze a training config with exact tokenizer counts on its training-view preset",
    )
    p_analyze.add_argument("--output-root", required=True, help="base output root")
    p_analyze.add_argument(
        "--training-config",
        required=True,
        help=(
            "path to a training config YAML like the example run descriptions; "
            "if the top-level mapping contains 'configuration', that nested payload is used"
        ),
    )
    p_analyze.add_argument(
        "--top-sources",
        type=int,
        default=-1,
        help="max number of source rows to show; default is all rows, use a negative value for all rows",
    )
    p_analyze.set_defaults(func=cmd_analyze)

    p_audit_tok = sub.add_parser(
        'audit-tokenizer',
        aliases=['audit_tokenizer'],
        help='audit a trained tokenizer vocabulary by tokenizer name or preset name',
    )
    p_audit_tok.add_argument('--output-root', required=True, help='base output root')
    p_audit_tok.add_argument(
        '--tokenizer',
        required=True,
        help='tokenizer preset name or tokenizer id',
    )
    p_audit_tok.add_argument(
        '--example-limit',
        type=int,
        default=10,
        help='max number of example tokens to print per category (0 disables examples)',
    )
    p_audit_tok.set_defaults(func=cmd_audit_tokenizer)

    p_train_tok = sub.add_parser(
        "train-tokenizer",
        aliases=["train_tokenizer"],
        help="train tokenizer from a materialized snapshot split",
    )
    p_train_tok.add_argument("--preset", required=True, help="train tokenizer preset name")
    p_train_tok.add_argument("--output-root", required=True, help="base output root")
    p_train_tok.set_defaults(func=cmd_train_tokenizer)

    p_run_new = sub.add_parser(
        "run-new",
        aliases=["run_new"],
        help="create a new run from a training config and launch fit",
    )
    p_run_new.add_argument("--output-root", required=True, help="base output root")
    p_run_new.add_argument(
        "--training-config",
        required=True,
        help="path to the training YAML (must contain 'configuration'; run_id will be regenerated)",
    )
    p_run_new.set_defaults(func=cmd_run_new)

    p_resume = sub.add_parser(
        "resume",
        help="resume fit from an existing run checkpoint",
    )
    p_resume.add_argument("--output-root", required=True, help="base output root")
    p_resume.add_argument("--run-id", required=True, help="run identifier")
    p_resume.add_argument(
        "--checkpoint",
        default="last",
        help="checkpoint selector: last (default), best, or a named checkpoint",
    )
    p_resume.set_defaults(func=cmd_resume)

    p_fork = sub.add_parser(
        "fork",
        help="clone a run into a new run id without starting training",
    )
    p_fork.add_argument("--output-root", required=True, help="base output root")
    p_fork.add_argument("--run-id", required=True, help="source run identifier")
    p_fork.add_argument("--to", required=True, help="destination run identifier")
    p_fork.add_argument(
        "--checkpoint",
        default="last",
        help="checkpoint selector: last (default), best, or a named checkpoint",
    )
    p_fork.set_defaults(func=cmd_fork)

    p_eval = sub.add_parser(
        "eval",
        help="evaluate Wintermute runs or Hugging Face models with lm-evaluation-harness",
    )
    p_eval.add_argument("--output-root", required=True, help="base output root")
    p_eval.add_argument(
        "--run-id",
        action="append",
        help="run identifier; repeat the option or use comma-separated identifiers",
    )
    p_eval.add_argument(
        "--hf-model",
        action="append",
        help="Hugging Face causal model id; repeat the option or use comma-separated identifiers",
    )
    p_eval.add_argument(
        "--checkpoint",
        default="last",
        help=(
            "checkpoint selector: last (default), best, a named checkpoint, "
            "or a checkpoint id such as step_0_100"
        ),
    )
    p_eval.add_argument(
        "--lm-eval-config",
        required=True,
        help="path to an lm-evaluation-harness YAML config",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_generate = sub.add_parser(
        "generate",
        help="interactive text generation from an existing run id",
    )
    p_generate.add_argument("--output-root", required=True, help="base output root")
    p_generate.add_argument("--run-id", required=True, help="run identifier")
    p_generate.set_defaults(func=cmd_generate)

    p_endpoint = sub.add_parser(
        "start-inference-endpoint",
        aliases=["start_inference_endpoint"],
        help="start a Slack-backed inference endpoint with one inference session per Slack thread",
    )
    p_endpoint.add_argument("--output-root", required=True, help="base output root")
    p_endpoint.add_argument(
        "--endpoint-config",
        required=True,
        help="path to the inference endpoint YAML config",
    )
    p_endpoint.set_defaults(func=cmd_start_inference_endpoint)

    return parser
