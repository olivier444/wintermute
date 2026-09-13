from __future__ import annotations

from wintermute.data.snapshot.config import SnapshotConfig
from wintermute.data.snapshot.dedup import build_dedup_rejections
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
from wintermute.tools.logging import console_log
from wintermute.data.constants import *
from wintermute.tools.tables import build_table_file_path, build_table_dir_path, create_table, build_table_sql, create_duckdb_connection
import pyarrow as pa
import os
import math
from duckdb import DuckDBPyConnection
from tabulate import tabulate


def create_snapshot(config: SnapshotConfig) -> None:
    _build_snapshot_def(config)
    _build_snapshot_members(config)


def _build_snapshot_def(config: SnapshotConfig) -> None:
    content = {
        FLD_SNAPSHOT_ID: ([config.snapshot_id], pa.string())
    }
    create_table(config.get_index_path(), config.snapshot_id, TBL_SNAPSHOT, content)


def _compute_quotas(config:SnapshotConfig) -> Dict[str, Dict[str, int]]:
    """
    result: src_id => dict(split => num_chars)
    """
    split_sizes_chars = {k: int(v) for k, v in config.split_sizes_chars.items()}
    total_weight = sum(sc.weight for sc in config.source_configs.values())

    quotas: Dict[str, Dict[str, int]] = {}
    for src_id, sc in config.source_configs.items():
        quotas[src_id] = {}
        for split, total_chars in split_sizes_chars.items():
            quotas[src_id][split] = int(int(total_chars) * sc.weight / total_weight)

    return quotas


def _build_base_eligible_view(config: SnapshotConfig, con: DuckDBPyConnection) -> None:
    record_table = build_table_sql(TBL_RECORD, config.get_index_path())
    file_table = build_table_sql(TBL_FILE, config.get_index_path())
    excluded_join_sql = ""

    where = [f"r.{FLD_RECORD_LEN} IS NOT NULL", f"r.{FLD_RECORD_LEN} > 0"]

    if config.excluded_languages and len(config.excluded_languages) > 0:
        where.append(f"r.{FLD_RECORD_LAN} NOT IN ({", ".join(f"'{l}'" for l in config.excluded_languages)})")

    _build_excluded_records_view(config, con)
    if _has_excluded_snapshot_ids(config):
        where.append(f"x.{FLD_MEMBERS_REC_ID} IS NULL")
        excluded_join_sql = (
            f"LEFT JOIN excluded_records x "
            f"ON x.{FLD_FILE_SOURCE_ID} = f.{FLD_FILE_SOURCE_ID} "
            f"AND x.{FLD_MEMBERS_REC_ID} = r.{FLD_RECORD_ID}"
        )

    where_clause = " AND ".join(where)

    sql = f"""
    CREATE OR REPLACE TEMP VIEW base_eligible AS
    SELECT
        r.{FLD_RECORD_ID},
        r.{FLD_RECORD_LEN},
        r.{FLD_RECORD_LAN},
        f.{FLD_FILE_SOURCE_ID}
    FROM {record_table} AS r
    INNER JOIN {file_table} f ON f.{FLD_FILE_ID} = r.{FLD_RECORD_FILE_ID}
    {excluded_join_sql}
    WHERE {where_clause}
        AND f.{FLD_FILE_SOURCE_ID} IN ({", ".join([f"'{sid}'" for sid in list(config.source_configs.keys())])});
    """
    
    con.execute(sql)


def _build_eligible_view(con: DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW eligible AS
        SELECT e.*
        FROM base_eligible e
        LEFT JOIN dedup_rejections d USING ({FLD_RECORD_ID})
        WHERE d.{FLD_RECORD_ID} IS NULL
        """
    )


def _has_excluded_snapshot_ids(config: SnapshotConfig) -> bool:
    return any(len(sc.excluded_snapshot_ids) > 0 for sc in config.source_configs.values())


def _build_excluded_records_view(config: SnapshotConfig, con: DuckDBPyConnection) -> None:
    if not _has_excluded_snapshot_ids(config):
        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW excluded_records AS
            SELECT
                CAST(NULL AS VARCHAR) AS {FLD_FILE_SOURCE_ID},
                CAST(NULL AS VARCHAR) AS {FLD_MEMBERS_REC_ID}
            WHERE FALSE
            """
        )
        return

    excluded_sources_sql = _build_excluded_snapshot_sources_sql(config)
    excluded_members_sql = _build_excluded_snapshot_members_sql(config)
    record_table = build_table_sql(TBL_RECORD, config.get_index_path())
    file_table = build_table_sql(TBL_FILE, config.get_index_path())

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW excluded_snapshot_sources AS
        {excluded_sources_sql}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW excluded_snapshot_members AS
        {excluded_members_sql}
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW excluded_records AS
        SELECT DISTINCT
            s.{FLD_FILE_SOURCE_ID},
            m.{FLD_MEMBERS_REC_ID}
        FROM excluded_snapshot_sources s
        INNER JOIN excluded_snapshot_members m
            ON m.{FLD_MEMBERS_SNAPSHOT_ID} = s.{FLD_MEMBERS_SNAPSHOT_ID}
        INNER JOIN {record_table} r
            ON r.{FLD_RECORD_ID} = m.{FLD_MEMBERS_REC_ID}
        INNER JOIN {file_table} f
            ON f.{FLD_FILE_ID} = r.{FLD_RECORD_FILE_ID}
            AND f.{FLD_FILE_SOURCE_ID} = s.{FLD_FILE_SOURCE_ID}
        """
    )


def _build_excluded_snapshot_sources_sql(config: SnapshotConfig) -> str:
    rows_sql: List[str] = []
    for src_id, sc in config.source_configs.items():
        for snapshot_id in sc.excluded_snapshot_ids:
            rows_sql.append(f"SELECT '{src_id}' AS {FLD_FILE_SOURCE_ID}, '{snapshot_id}' AS {FLD_MEMBERS_SNAPSHOT_ID}")

    if len(rows_sql) == 0:
        raise ValueError("Excluded snapshot source SQL requested without any excluded snapshot ids.")

    return "\nUNION ALL\n".join(rows_sql)


def _build_excluded_snapshot_members_sql(config: SnapshotConfig) -> str:
    parts: List[str] = []
    for snapshot_id in sorted(_list_unique_excluded_snapshot_ids(config)):
        members_path = build_table_file_path(TBL_SNAPSHOT_MEMBERS, config.get_index_path(), snapshot_id)
        if not Path(members_path).exists():
            raise FileNotFoundError(f"Excluded snapshot members file not found: snapshot_id='{snapshot_id}', path='{members_path}'")

        parts.append(
            f"""
            SELECT
                '{snapshot_id}' AS {FLD_MEMBERS_SNAPSHOT_ID},
                m.{FLD_MEMBERS_REC_ID}
            FROM read_parquet('{members_path}', union_by_name=1) m
            """
        )

    if len(parts) == 0:
        raise ValueError("Excluded snapshot members SQL requested without any excluded snapshot ids.")

    return "\nUNION ALL\n".join(parts)


def _list_unique_excluded_snapshot_ids(config: SnapshotConfig) -> List[str]:
    snapshot_ids = {
        snapshot_id
        for sc in config.source_configs.values()
        for snapshot_id in sc.excluded_snapshot_ids
    }
    if config.snapshot_id in snapshot_ids:
        raise ValueError(f"Snapshot '{config.snapshot_id}' cannot exclude itself.")
    return list(snapshot_ids)
    

def _build_quota_table(
        config: SnapshotConfig, 
        splits: List[str], 
        quotas: Dict[str, Dict[str, int]], 
        con: DuckDBPyConnection
    ):
    """
    quotas: 
    src_id | bud_train | ... | bud_test | min_len | max_len
    """
    rows_sql = []
    for src_id, sc in config.source_configs.items():
        for oversampling_index in range(sc.oversampling):
            buds = [quotas[src_id].get(s, 0) / sc.oversampling for s in splits]
            buds_sql = ", ".join(f"{int(b)}" for b in buds)
            rows_sql.append(
                f"('{src_id}', {buds_sql}, {int(sc.min_record_size_char)}, {int(sc.max_record_size_char)}, {oversampling_index})"
            )
    values_sql = ",\n    ".join(rows_sql)

    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW quotas AS
    SELECT * FROM (
        VALUES
        {values_sql}
    ) AS t(
        {FLD_FILE_SOURCE_ID},
        {", ".join([f"bud_{s}" for s in splits])},
        min_len,
        max_len,
        oversampling_index
    );
    """)


def _write_manifest(        
        config: SnapshotConfig, 
        splits: List[str], 
        quotas: Dict[str, Dict[str, int]],
        members_path: str,
        dedup_stats: Dict[str, Any] | None,
    ):

    manifest = {
        "snapshot_id": config.snapshot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "split_order_used": splits,
        "budgets_chars_by_source": quotas,  # dict[src_id][split] -> chars
        "dedup_stats": dedup_stats,
        "outputs": {"members_parquet": str(members_path)},
    }
    
    manifest_path = Path(
        os.path.join(
            build_table_dir_path(TBL_SNAPSHOT, config.get_index_path()), 
            f"{config.snapshot_id}.manifest.json"
        )
    )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _measure_datasets(con: DuckDBPyConnection) -> Dict[str, int]:
    res = con.execute(f"""
        SELECT
            e.{FLD_FILE_SOURCE_ID},
            SUM(e.{FLD_RECORD_LEN})
        FROM eligible e
        INNER JOIN quotas q ON e.{FLD_FILE_SOURCE_ID} = q.{FLD_FILE_SOURCE_ID} AND q.oversampling_index = 0
        WHERE e.{FLD_RECORD_LEN} BETWEEN q.min_len AND q.max_len
        GROUP BY e.{FLD_FILE_SOURCE_ID}
    """)

    return {row[0]: row[1] for row in res.fetchall()}


# dataset_sizes dict[src] => size
# quota: dict[src][split_id] => quota
def _check_quotas(config: SnapshotConfig, dataset_sizes: Dict[str, int], quotas: Dict[str, Dict[str, int]]):
    console_log("snapshot", "checking quotas")

    errors: List[str] = []
    rows: List[Any] = []

    tot_requested = 0
    tot_available = 0
    for src_id, src_quotas in quotas.items():
        sc = config.source_configs[src_id]
        available = int(dataset_sizes.get(src_id, -1))
        tot_available += available
        requested = int(sum(int(v) for v in src_quotas.values()))
        tot_requested += requested
        required_oversampling = math.inf if available <= 0 else (requested / available)

        warn = "" if required_oversampling <= sc.oversampling else "/!\\  "

        rows.append({
            "source":src_id, 
            "weight": sc.weight,
            "cfg oversampling":sc.oversampling,
            " ":warn,            
            "req oversampling":f"{required_oversampling:.2f}",            
            "chars requested": f"{requested:_}",
            "raw chars available": f"{available:_}",
            "~ tok requested": f"{int(requested/4):_}",               
            "~ raw tok available": f"{int(available/4):_}",            
        })

        if requested > available * sc.oversampling:
            errors.append(f"{src_id}: {required_oversampling:.2f} oversampling required")

    tot_row = {
        "source": "** TOTAL **",
        "weight": sum(r["weight"] for r in rows),
        "chars requested": f"{tot_requested:_}",
        "raw chars available": f"{tot_available:_}",
        "~ tok requested": f"{int(tot_requested/4):_}",
        "~ raw tok available": f"{int(tot_available/4):_}",        
    }

    print(tabulate(rows + [tot_row], headers="keys", tablefmt="github"))
    if errors:
        raise RuntimeError(f"Snapshot inconsistencies in {", ".join(errors)}")

    return


def _build_snapshot_members(config: SnapshotConfig) -> None:
    members_path = build_table_file_path(TBL_SNAPSHOT_MEMBERS, config.get_index_path(), config.snapshot_id, ensure_dir_exist=True)
    console_log("_build_snapshot_members", f"storing snapshot members in {members_path}")

    con = create_duckdb_connection(temp_directory_root=config.get_index_path())

    try:
        quotas = _compute_quotas(config)
        splits = list(config.split_sizes_chars.keys())
        _build_base_eligible_view(config, con)
        _build_quota_table(config, splits, quotas, con)
        dedup_stats = build_dedup_rejections(config, con)
        _build_eligible_view(con)
        _check_quotas(config, _measure_datasets(con), quotas)

        # build cumulated bounds
        #   test = bud_test
        #   valid = bud_test + bud_valid
        #   train = bud_test + bud_valid + bud_train

        cum_bounds = []
        running = "0::UBIGINT"
        for split in splits:
            running = f"({running} + q.bud_{split})"
            cum_bounds.append((split, running))

        # CASE expression
        case_lines = []
        for split, bound_expr in cum_bounds:
            case_lines.append(f"WHEN c.cum_chars <= {bound_expr} THEN '{split}'")
        case_sql = "\n                ".join(case_lines)

        con.execute(f"""
        COPY (
            WITH filtered AS (
                SELECT
                    e.{FLD_RECORD_ID},
                    e.{FLD_FILE_SOURCE_ID},
                    q.oversampling_index,
                    e.{FLD_RECORD_LEN},
                    (hash(e.{FLD_RECORD_ID} || '::sample_{config.seed}')  & 9223372036854775807) as sample_key,
                    (hash(e.{FLD_RECORD_ID} || '::shuffle_{config.seed}::osi_' || q.oversampling_index::VARCHAR) & 9223372036854775807) as {FLD_MEMBERS_HASH}
                FROM eligible e
                INNER JOIN quotas q ON e.{FLD_FILE_SOURCE_ID} = q.{FLD_FILE_SOURCE_ID}
                WHERE e.{FLD_RECORD_LEN} BETWEEN q.min_len AND q.max_len
            ),
            cum AS (
                SELECT
                    f.*,
                    sum({FLD_RECORD_LEN}) OVER (
                        PARTITION BY {FLD_FILE_SOURCE_ID}, oversampling_index
                        ORDER BY sample_key
                    )::UBIGINT AS cum_chars
                FROM filtered f            
            ),
            labeled AS (
                SELECT
                    c.{FLD_RECORD_ID},
                    c.{FLD_MEMBERS_HASH},
                    c.oversampling_index as {FLD_MEMBERS_OVERSAMPLING},
                    CASE
                        {case_sql}
                        ELSE NULL
                    END AS {FLD_MEMBERS_SPLIT}
                FROM cum c
                JOIN quotas q
                    ON c.{FLD_FILE_SOURCE_ID} = q.{FLD_FILE_SOURCE_ID}
                    AND c.oversampling_index = q.oversampling_index
            )

            SELECT '{config.snapshot_id}' as {FLD_MEMBERS_SNAPSHOT_ID}, {FLD_MEMBERS_SPLIT}, {FLD_MEMBERS_REC_ID}, {FLD_MEMBERS_OVERSAMPLING}, {FLD_MEMBERS_HASH}
            FROM labeled
            WHERE {FLD_MEMBERS_SPLIT} IS NOT NULL
            ORDER BY {FLD_MEMBERS_HASH}
        )
        TO '{members_path}'
        (FORMAT PARQUET);
        """)

        _write_manifest(config, splits, quotas, members_path, dedup_stats)

    finally:
        con.close()
