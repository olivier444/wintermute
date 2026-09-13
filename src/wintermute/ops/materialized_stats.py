# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import duckdb
from tabulate import tabulate

from wintermute.data.constants import *
from wintermute.tools.files import get_index_dir, get_materialized_dir, json_save
from wintermute.tools.logging import console_log
from wintermute.tools.misc import utc_now
from wintermute.tools.tables import build_table_dir_path, build_table_file_path, build_table_sql, create_duckdb_connection


ESTIMATED_CHARS_PER_TOKEN = 4.0


def _fmt_int(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{int(v):_}"


def _fmt_float(v: Any, digits: int = 2) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.{digits}f}"


def _fmt_pct(v: Any) -> str:
    if v is None:
        return "n/a"
    return f"{float(v):.2f}%"


def _fetch_all(con: duckdb.DuckDBPyConnection, sql: str) -> List[Dict[str, Any]]:
    res = con.execute(sql)
    cols = [c[0] for c in res.description]
    rows = res.fetchall()
    return [dict(zip(cols, row)) for row in rows]


def _print_table(rows: List[Dict[str, Any]]) -> None:
    print(_render_table(rows))


def _render_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "(no rows)"
    return str(tabulate(rows, headers="keys", tablefmt="github"))


def _write_materialized_stats_artifacts(
    *,
    output_dir: Path,
    file_stem: str,
    json_payload: Dict[str, Any],
    text_report: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{file_stem}.json"
    txt_path = output_dir / f"{file_stem}.txt"
    json_save(json_payload, output_dir.as_posix(), json_path.name)
    txt_path.write_text(text_report, encoding="utf-8")
    return (json_path, txt_path)


def _load_snapshot_manifest(index_path: str, snapshot_id: str) -> Dict[str, Any]:
    manifest_path = Path(build_table_dir_path(TBL_SNAPSHOT, index_path)) / f"{snapshot_id}.manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _safe_sql_string(value: str) -> str:
    return value.replace("'", "''")


def _split_order(manifest: Dict[str, Any], split_filter: str | None, rows: List[Dict[str, Any]]) -> List[str]:
    if split_filter:
        return [split_filter]

    manifest_order = manifest.get("split_order_used")
    if isinstance(manifest_order, list) and manifest_order:
        return [str(item) for item in manifest_order]

    discovered = sorted(
        {
            str(row["scope"])
            for row in rows
            if row.get("scope") not in (None, "ALL")
        }
    )
    return discovered


def _scope_target_chars(manifest: Dict[str, Any], selected_splits: List[str], scope: str) -> int | None:
    split_sizes = manifest.get("config", {}).get("split_sizes_chars", {})
    if not isinstance(split_sizes, dict):
        return None

    if scope == "ALL":
        total = 0
        found = False
        for split in selected_splits:
            value = split_sizes.get(split)
            if value is None:
                continue
            total += int(value)
            found = True
        return total if found else None

    value = split_sizes.get(scope)
    if value is None:
        return None
    return int(value)


def _source_budget_chars(
    manifest: Dict[str, Any],
    src_id: str,
    selected_splits: List[str],
    scope: str | None = None,
) -> int | None:
    budgets = manifest.get("budgets_chars_by_source", {})
    if not isinstance(budgets, dict):
        return None

    src_budget = budgets.get(src_id)
    if not isinstance(src_budget, dict):
        return None

    if scope is not None:
        value = src_budget.get(scope)
        if value is None:
            return None
        return int(value)

    total = 0
    found = False
    for split in selected_splits:
        value = src_budget.get(split)
        if value is None:
            continue
        total += int(value)
        found = True
    return total if found else None


def _source_config_value(manifest: Dict[str, Any], src_id: str, key: str) -> Any:
    source_configs = manifest.get("config", {}).get("source_configs", {})
    if not isinstance(source_configs, dict):
        return None
    cfg = source_configs.get(src_id)
    if not isinstance(cfg, dict):
        return None
    return cfg.get(key)


def _source_record_transform_kind(manifest: Dict[str, Any], src_id: str) -> str:
    source_configs = manifest.get("config", {}).get("source_configs", {})
    if not isinstance(source_configs, dict) or src_id not in source_configs:
        return "n/a"
    value = _source_config_value(manifest, src_id, "record_transform")
    if not isinstance(value, dict):
        return "raw"
    kind = str(value.get("kind", "raw"))
    children = value.get("children")
    if not isinstance(children, list) or not children:
        return kind
    child_kinds = [
        str(child.get("kind", "raw")) if isinstance(child, dict) else str(child)
        for child in children
    ]
    return f"{kind}({'+'.join(child_kinds)})"


def _format_overview_rows(
    rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    selected_splits: List[str],
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for row in rows:
        target_chars = _scope_target_chars(manifest, selected_splits, str(row["scope"]))
        chars = int(row["chars"])
        examples = int(row["examples"])
        formatted.append(
            {
                "scope": row["scope"],
                "examples": _fmt_int(examples),
                "uniq_examples": _fmt_int(row["unique_examples"]),
                "os_extra_examples": _fmt_int(row["oversampling_extra_examples"]),
                "chars": _fmt_int(chars),
                "uniq_chars": _fmt_int(row["unique_chars"]),
                "os_extra_chars": _fmt_int(row["oversampling_extra_chars"]),
                "est_tokens": _fmt_int(row["estimated_tokens"]),
                "avg_chars": _fmt_float(row["avg_chars"], digits=1),
                "srcs": _fmt_int(row["source_count"]),
                "target_chars": _fmt_int(target_chars),
                "fill": _fmt_pct(100.0 * chars / target_chars if target_chars else None),
            }
        )
    return formatted


def _format_source_rows(
    rows: List[Dict[str, Any]],
    manifest: Dict[str, Any],
    scope_total_chars: int,
    selected_splits: List[str],
    scope_split: str | None = None,
) -> List[Dict[str, Any]]:
    formatted: List[Dict[str, Any]] = []
    for row in rows:
        chars = int(row["chars"])
        target_chars = _source_budget_chars(
            manifest=manifest,
            src_id=str(row["src_id"]),
            selected_splits=selected_splits,
            scope=scope_split,
        )
        formatted.append(
            {
                "src_id": row["src_id"],
                "src_label": row["src_label"],
                "examples": _fmt_int(row["examples"]),
                "uniq_examples": _fmt_int(row["unique_examples"]),
                "os_extra_examples": _fmt_int(row["oversampling_extra_examples"]),
                "chars": _fmt_int(chars),
                "os_extra_chars": _fmt_int(row["oversampling_extra_chars"]),
                "est_tokens": _fmt_int(row["estimated_tokens"]),
                "avg_chars": _fmt_float(row["avg_chars"], digits=1),
                "char_share": _fmt_pct(100.0 * chars / scope_total_chars if scope_total_chars else None),
                "target_chars": _fmt_int(target_chars),
                "fill": _fmt_pct(100.0 * chars / target_chars if target_chars else None),
                "cfg_os": _fmt_int(_source_config_value(manifest, str(row["src_id"]), "oversampling")),
                "transforms": _source_record_transform_kind(manifest, str(row["src_id"])),
            }
        )
    return formatted


def run_materialized_stats(
    output_root: str,
    snapshot_id: str,
    split: str | None = None,
    top_sources: int = 20,
    chars_per_token: float = ESTIMATED_CHARS_PER_TOKEN,
) -> None:
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be > 0")
    if top_sources == 0:
        raise ValueError("top_sources cannot be 0; use a negative value to show all rows")

    index_path = get_index_dir(output_root)
    members_path = Path(build_table_file_path(TBL_SNAPSHOT_MEMBERS, index_path, snapshot_id))
    if not members_path.exists():
        materialized_dir = Path(get_materialized_dir(output_root, snapshot_id))
        if materialized_dir.exists():
            raise FileNotFoundError(
                f"snapshot_members index table not found for '{snapshot_id}'. "
                "This index-based command only supports snapshot-backed materialized datasets."
            )
        raise FileNotFoundError(f"snapshot_members index table not found for '{snapshot_id}' at {members_path}")

    manifest = _load_snapshot_manifest(index_path, snapshot_id)

    record_table = build_table_sql(TBL_RECORD, index_path)
    file_table = build_table_sql(TBL_FILE, index_path)
    source_table = build_table_sql(TBL_SOURCE, index_path)
    members_table = build_table_sql(TBL_SNAPSHOT_MEMBERS, index_path, snapshot_id)

    split_where = ""
    if split:
        split_where = f"WHERE m.{FLD_MEMBERS_SPLIT} = '{_safe_sql_string(split)}'"

    con = create_duckdb_connection(temp_directory_root=index_path)

    try:
        report_sections: List[str] = []
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW materialized_member_groups AS
            WITH members AS (
                SELECT
                    m.{FLD_MEMBERS_SPLIT} AS split,
                    m.{FLD_MEMBERS_REC_ID} AS rec_id,
                    s.{FLD_SOURCE_ID} AS src_id,
                    s.{FLD_SOURCE_LABEL} AS src_label,
                    COALESCE(r.{FLD_RECORD_LEN}, 0) AS rec_len
                FROM {members_table} m
                INNER JOIN {record_table} r ON r.{FLD_RECORD_ID} = m.{FLD_MEMBERS_REC_ID}
                INNER JOIN {file_table} f ON f.{FLD_FILE_ID} = r.{FLD_RECORD_FILE_ID}
                INNER JOIN {source_table} s ON s.{FLD_SOURCE_ID} = f.{FLD_FILE_SOURCE_ID}
                {split_where}
            )
            SELECT
                split,
                src_id,
                src_label,
                rec_id,
                MAX(rec_len) AS rec_len,
                COUNT(*) AS copies
            FROM members
            GROUP BY split, src_id, src_label, rec_id;
            """
        )

        console_log("materialized-stats", f"snapshot: {snapshot_id}")
        console_log("materialized-stats", f"output_root: {output_root}")
        console_log("materialized-stats", f"scope: {split if split else 'all splits'}")
        console_log(
            "materialized-stats",
            (
                f"estimated_tokens ~= chars / {chars_per_token:.2f}; "
                "chars are approximated from indexed rec_len before optional record transformation."
            ),
        )

        dedup_stats = manifest.get("dedup_stats")
        if dedup_stats is not None:
            candidate_records = int(dedup_stats["candidate_records"])
            candidate_chars = int(dedup_stats["candidate_chars"])
            rejected_records = int(dedup_stats["rejected_records"])
            rejected_chars = int(dedup_stats["rejected_chars"])
            dedup_rows = [
                {
                    "candidates": _fmt_int(candidate_records),
                    "kept": _fmt_int(dedup_stats["kept_records"]),
                    "rejected": _fmt_int(rejected_records),
                    "rejected_rate": _fmt_pct(
                        100.0 * rejected_records / candidate_records if candidate_records else 0.0
                    ),
                    "candidate_chars": _fmt_int(candidate_chars),
                    "rejected_chars": _fmt_int(rejected_chars),
                    "rejected_char_rate": _fmt_pct(
                        100.0 * rejected_chars / candidate_chars if candidate_chars else 0.0
                    ),
                    "reference": _fmt_int(dedup_stats["by_reason"]["reference"]["records"]),
                    "within_snapshot": _fmt_int(
                        dedup_stats["by_reason"]["within_snapshot"]["records"]
                    ),
                }
            ]
            print("\n=== Snapshot Deduplication ===")
            _print_table(dedup_rows)
            report_sections.append(
                "=== Snapshot Deduplication ===\n" + _render_table(dedup_rows)
            )

        overview_rows = _fetch_all(
            con,
            f"""
            SELECT
                'ALL' AS scope,
                COALESCE(SUM(copies), 0) AS examples,
                COUNT(*) AS unique_examples,
                COALESCE(SUM(copies), 0) - COUNT(*) AS oversampling_extra_examples,
                COALESCE(SUM(copies * rec_len), 0) AS chars,
                COALESCE(SUM(rec_len), 0) AS unique_chars,
                COALESCE(SUM((copies - 1) * rec_len), 0) AS oversampling_extra_chars,
                CAST(ROUND(COALESCE(SUM(copies * rec_len), 0) / {chars_per_token}) AS BIGINT) AS estimated_tokens,
                COALESCE(SUM(copies * rec_len), 0) / NULLIF(COALESCE(SUM(copies), 0), 0) AS avg_chars,
                COUNT(DISTINCT src_id) AS source_count
            FROM materialized_member_groups

            UNION ALL

            SELECT
                split AS scope,
                COALESCE(SUM(copies), 0) AS examples,
                COUNT(*) AS unique_examples,
                COALESCE(SUM(copies), 0) - COUNT(*) AS oversampling_extra_examples,
                COALESCE(SUM(copies * rec_len), 0) AS chars,
                COALESCE(SUM(rec_len), 0) AS unique_chars,
                COALESCE(SUM((copies - 1) * rec_len), 0) AS oversampling_extra_chars,
                CAST(ROUND(COALESCE(SUM(copies * rec_len), 0) / {chars_per_token}) AS BIGINT) AS estimated_tokens,
                COALESCE(SUM(copies * rec_len), 0) / NULLIF(COALESCE(SUM(copies), 0), 0) AS avg_chars,
                COUNT(DISTINCT src_id) AS source_count
            FROM materialized_member_groups
            GROUP BY split;
            """,
        )

        if not overview_rows:
            raise RuntimeError(f"No snapshot members found for snapshot '{snapshot_id}'.")

        selected_splits = _split_order(manifest, split, overview_rows)
        if split and not any(row["scope"] == split for row in overview_rows):
            raise RuntimeError(f"No snapshot members found for snapshot '{snapshot_id}' and split '{split}'.")

        split_rank = {name: index for index, name in enumerate(selected_splits)}
        overview_rows.sort(
            key=lambda row: (
                0 if row["scope"] == "ALL" else 1,
                split_rank.get(str(row["scope"]), len(split_rank)),
                str(row["scope"]),
            )
        )

        formatted_overview_rows = _format_overview_rows(overview_rows, manifest, selected_splits)
        print("\n=== Overview ===")
        _print_table(formatted_overview_rows)
        report_sections.append("=== Overview ===\n" + _render_table(formatted_overview_rows))

        source_limit_sql = "" if top_sources < 0 else f"LIMIT {int(top_sources)}"
        scope_source_rows = _fetch_all(
            con,
            f"""
            SELECT
                src_id,
                src_label,
                COALESCE(SUM(copies), 0) AS examples,
                COUNT(*) AS unique_examples,
                COALESCE(SUM(copies), 0) - COUNT(*) AS oversampling_extra_examples,
                COALESCE(SUM(copies * rec_len), 0) AS chars,
                COALESCE(SUM((copies - 1) * rec_len), 0) AS oversampling_extra_chars,
                CAST(ROUND(COALESCE(SUM(copies * rec_len), 0) / {chars_per_token}) AS BIGINT) AS estimated_tokens,
                COALESCE(SUM(copies * rec_len), 0) / NULLIF(COALESCE(SUM(copies), 0), 0) AS avg_chars
            FROM materialized_member_groups
            GROUP BY src_id, src_label
            ORDER BY chars DESC, examples DESC, src_id ASC
            {source_limit_sql};
            """,
        )

        scope_all_row = next(row for row in overview_rows if row["scope"] == "ALL")
        formatted_scope_source_rows = _format_source_rows(
            rows=scope_source_rows,
            manifest=manifest,
            scope_total_chars=int(scope_all_row["chars"]),
            selected_splits=selected_splits,
        )
        print(
            f"\n=== Per Source ({'split ' + split if split else 'selected scope'}, "
            f"top {top_sources if top_sources > 0 else 'all'} by chars) ==="
        )
        _print_table(formatted_scope_source_rows)
        report_sections.append(
            (
                f"=== Per Source ({'split ' + split if split else 'selected scope'}, "
                f"top {top_sources if top_sources > 0 else 'all'} by chars) ===\n"
                f"{_render_table(formatted_scope_source_rows)}"
            )
        )

        per_split_scope_rows: Dict[str, List[Dict[str, Any]]] = {}
        if split is None and len(selected_splits) > 1:
            per_split_rows = _fetch_all(
                con,
                f"""
                WITH aggregated AS (
                    SELECT
                        split,
                        src_id,
                        src_label,
                        COALESCE(SUM(copies), 0) AS examples,
                        COUNT(*) AS unique_examples,
                        COALESCE(SUM(copies), 0) - COUNT(*) AS oversampling_extra_examples,
                        COALESCE(SUM(copies * rec_len), 0) AS chars,
                        COALESCE(SUM((copies - 1) * rec_len), 0) AS oversampling_extra_chars,
                        CAST(ROUND(COALESCE(SUM(copies * rec_len), 0) / {chars_per_token}) AS BIGINT) AS estimated_tokens,
                        COALESCE(SUM(copies * rec_len), 0) / NULLIF(COALESCE(SUM(copies), 0), 0) AS avg_chars
                    FROM materialized_member_groups
                    GROUP BY split, src_id, src_label
                ),
                ranked AS (
                    SELECT
                        *,
                        ROW_NUMBER() OVER (
                            PARTITION BY split
                            ORDER BY chars DESC, examples DESC, src_id ASC
                        ) AS rn
                    FROM aggregated
                )
                SELECT
                    split,
                    src_id,
                    src_label,
                    examples,
                    unique_examples,
                    oversampling_extra_examples,
                    chars,
                    oversampling_extra_chars,
                    estimated_tokens,
                    avg_chars
                FROM ranked
                {"" if top_sources < 0 else f"WHERE rn <= {int(top_sources)}"}
                ORDER BY split ASC, chars DESC, examples DESC, src_id ASC;
                """,
            )

            split_totals = {
                str(row["scope"]): int(row["chars"])
                for row in overview_rows
                if row["scope"] != "ALL"
            }
            print(
                f"\n=== Per Source By Split (top {top_sources if top_sources > 0 else 'all'} by chars per split) ==="
            )
            per_split_report_sections: List[str] = []
            for split_name in selected_splits:
                split_rows = [row for row in per_split_rows if row["split"] == split_name]
                if not split_rows:
                    continue
                per_split_scope_rows[split_name] = split_rows
                formatted_split_rows = _format_source_rows(
                    rows=split_rows,
                    manifest=manifest,
                    scope_total_chars=split_totals.get(split_name, 0),
                    selected_splits=[split_name],
                    scope_split=split_name,
                )
                print(f"\n-- {split_name} --")
                _print_table(formatted_split_rows)
                per_split_report_sections.append(f"-- {split_name} --\n{_render_table(formatted_split_rows)}")

            if per_split_report_sections:
                report_sections.append(
                    (
                        f"=== Per Source By Split (top {top_sources if top_sources > 0 else 'all'} by chars per split) ===\n\n"
                        + "\n\n".join(per_split_report_sections)
                    )
                )

        generated_at = utc_now()
        timestamp = generated_at.strftime("%Y%m%d_%H%M%S")
        scope_name = split or "all"
        file_stem = f"materialized_stats_{snapshot_id}_{scope_name}_{timestamp}"
        materialized_dir = Path(get_materialized_dir(output_root, snapshot_id, split))
        text_report = "\n\n".join(report_sections) + "\n"
        json_payload = {
            "snapshot_id": snapshot_id,
            "split": split,
            "output_root": output_root,
            "generated_at_utc": generated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "top_sources": top_sources,
            "chars_per_token": chars_per_token,
            "selected_splits": selected_splits,
            "overview_rows": overview_rows,
            "scope_source_rows": scope_source_rows,
            "per_split_source_rows": per_split_scope_rows,
        }
        json_path, txt_path = _write_materialized_stats_artifacts(
            output_dir=materialized_dir,
            file_stem=file_stem,
            json_payload=json_payload,
            text_report=text_report,
        )
        console_log("materialized-stats", f"saved materialized stats: {json_path}")
        console_log("materialized-stats", f"saved materialized stats: {txt_path}")

    finally:
        con.close()
