from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pyarrow as pa
from duckdb import DuckDBPyConnection

from wintermute.data.constants import (
    FLD_BAND_HASH,
    FLD_BAND_ID,
    FLD_BAND_RECORD_ID,
    FLD_FILE_ID,
    FLD_FILE_SOURCE_ID,
    FLD_RECORD_FILE_ID,
    FLD_RECORD_ID,
    FLD_RECORD_LEN,
    FLD_SIGNATURE_RECORD_ID,
    FLD_SIGNATURE_VECTOR,
    TBL_BAND,
    TBL_FILE,
    TBL_RECORD,
    TBL_SIGNATURE,
)
from wintermute.data.snapshot.config import SnapshotConfig
from wintermute.tools.logging import console_log
from wintermute.tools.tables import build_table_dir_path, build_table_sql
from wintermute.tools.union_join import UnionJoin


NUM_HASH = 128
SIMILARITY_THRESHOLD = 0.75
MAX_BAND_BUCKET_SIZE = 500
REFERENCE_REASON = "reference"
WITHIN_SNAPSHOT_REASON = "within_snapshot"


def build_dedup_rejections(
    config: SnapshotConfig,
    con: DuckDBPyConnection,
) -> dict[str, Any] | None:
    if config.dedup is None:
        _register_rejections(con, {})
        return None

    band_dir = build_table_dir_path(TBL_BAND, config.get_index_path())
    record_table = build_table_sql(TBL_RECORD, config.get_index_path())
    file_table = build_table_sql(TBL_FILE, config.get_index_path())
    signature_dir = build_table_dir_path(TBL_SIGNATURE, config.get_index_path())
    scoped_source_ids = sorted(
        set(config.source_configs) | set(config.dedup.reference_source_ids)
    )
    band_globs = ", ".join(
        f"'{band_dir}/{source_id}/*/*.parquet'"
        for source_id in scoped_source_ids
    )
    signature_files = ", ".join(
        f"'{signature_dir}/{source_id}.parquet'"
        for source_id in scoped_source_ids
    )
    signature_table = f"read_parquet([{signature_files}], union_by_name=1)"
    reference_source_ids = ", ".join(
        f"'{source_id}'" for source_id in config.dedup.reference_source_ids
    ) or "NULL"

    console_log(
        "dedup",
        (
            f"computing snapshot-scoped edges; references="
            f"{config.dedup.reference_source_ids or '(none)'}"
        ),
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW dedup_candidates AS
        SELECT DISTINCT
            e.{FLD_RECORD_ID},
            e.{FLD_FILE_SOURCE_ID},
            e.{FLD_RECORD_LEN}
        FROM base_eligible e
        INNER JOIN quotas q
            ON q.{FLD_FILE_SOURCE_ID} = e.{FLD_FILE_SOURCE_ID}
            AND q.oversampling_index = 0
        WHERE e.{FLD_RECORD_LEN} BETWEEN q.min_len AND q.max_len
        """
    )

    sql = f"""
    WITH
    reference_records AS (
        SELECT r.{FLD_RECORD_ID}
        FROM {record_table} r
        INNER JOIN {file_table} f
            ON f.{FLD_FILE_ID} = r.{FLD_RECORD_FILE_ID}
        WHERE f.{FLD_FILE_SOURCE_ID} IN ({reference_source_ids})
    ),
    bands AS (
        SELECT
            {FLD_BAND_RECORD_ID},
            {FLD_BAND_ID}::INTEGER AS {FLD_BAND_ID},
            {FLD_BAND_HASH}::UBIGINT AS {FLD_BAND_HASH}
        FROM read_parquet([{band_globs}], hive_partitioning=1)
    ),
    candidate_bands AS (
        SELECT b.*
        FROM bands b
        INNER JOIN dedup_candidates c USING ({FLD_RECORD_ID})
    ),
    reference_bands AS (
        SELECT b.*
        FROM bands b
        INNER JOIN reference_records r USING ({FLD_RECORD_ID})
    ),
    scoped_bands AS (
        SELECT * FROM candidate_bands
        UNION ALL
        SELECT * FROM reference_bands
    ),
    good_buckets AS (
        SELECT {FLD_BAND_ID}, {FLD_BAND_HASH}
        FROM scoped_bands
        GROUP BY 1, 2
        HAVING COUNT(*) BETWEEN 2 AND {MAX_BAND_BUCKET_SIZE}
    ),
    candidate_pairs AS (
        SELECT
            a.{FLD_RECORD_ID} AS uid1,
            b.{FLD_RECORD_ID} AS uid2,
            FALSE AS has_reference
        FROM good_buckets g
        INNER JOIN candidate_bands a USING ({FLD_BAND_ID}, {FLD_BAND_HASH})
        INNER JOIN candidate_bands b
            ON b.{FLD_BAND_ID} = g.{FLD_BAND_ID}
            AND b.{FLD_BAND_HASH} = g.{FLD_BAND_HASH}
            AND b.{FLD_RECORD_ID} > a.{FLD_RECORD_ID}
    ),
    reference_pairs AS (
        SELECT
            c.{FLD_RECORD_ID} AS uid1,
            r.{FLD_RECORD_ID} AS uid2,
            TRUE AS has_reference
        FROM good_buckets g
        INNER JOIN candidate_bands c USING ({FLD_BAND_ID}, {FLD_BAND_HASH})
        INNER JOIN reference_bands r
            ON r.{FLD_BAND_ID} = g.{FLD_BAND_ID}
            AND r.{FLD_BAND_HASH} = g.{FLD_BAND_HASH}
    ),
    pairs AS (
        SELECT DISTINCT uid1, uid2, has_reference
        FROM (
            SELECT * FROM candidate_pairs
            UNION ALL
            SELECT * FROM reference_pairs
        )
    ),
    scored AS (
        SELECT
            p.uid1,
            p.uid2,
            p.has_reference,
            (
                SELECT avg(
                    CASE
                        WHEN s1.{FLD_SIGNATURE_VECTOR}[i] = s2.{FLD_SIGNATURE_VECTOR}[i]
                        THEN 1.0
                        ELSE 0.0
                    END
                )
                FROM range(1, {NUM_HASH + 1}) hashes(i)
            ) AS similarity
        FROM pairs p
        INNER JOIN {signature_table} s1
            ON s1.{FLD_SIGNATURE_RECORD_ID} = p.uid1
        INNER JOIN {signature_table} s2
            ON s2.{FLD_SIGNATURE_RECORD_ID} = p.uid2
    )
    SELECT uid1, uid2, has_reference
    FROM scored
    WHERE similarity >= {SIMILARITY_THRESHOLD}
    """

    reader = con.execute(sql).fetch_record_batch(rows_per_batch=200_000)
    edges = (
        (str(uid1), str(uid2), bool(has_reference))
        for batch in reader
        for uid1, uid2, has_reference in zip(
            batch.column(0).to_pylist(),
            batch.column(1).to_pylist(),
            batch.column(2).to_pylist(),
        )
    )
    rejections = _resolve_rejections(edges)
    _register_rejections(con, rejections)
    stats = _build_stats(con)
    _log_stats(stats)
    return stats


def _resolve_rejections(edges: Iterable[tuple[str, str, bool]]) -> dict[str, str]:
    registry = UnionJoin()
    candidates: set[str] = set()
    references: set[str] = set()

    for uid1, uid2, has_reference in edges:
        registry.link(uid1, uid2)
        candidates.add(uid1)
        if has_reference:
            references.add(uid2)
        else:
            candidates.add(uid2)

    reference_roots = {registry.find_root(uid) for uid in references}
    winners: dict[str, str] = {}
    for uid in candidates:
        root = registry.find_root(uid)
        winner = winners.get(root)
        if winner is None or uid < winner:
            winners[root] = uid

    rejections: dict[str, str] = {}
    for uid in candidates:
        root = registry.find_root(uid)
        if root in reference_roots:
            rejections[uid] = REFERENCE_REASON
        elif uid != winners[root]:
            rejections[uid] = WITHIN_SNAPSHOT_REASON
    return rejections


def _register_rejections(con: DuckDBPyConnection, rejections: dict[str, str]) -> None:
    con.register(
        "dedup_rejections",
        pa.table(
            {
                FLD_RECORD_ID: pa.array(list(rejections), type=pa.string()),
                "reason": pa.array(list(rejections.values()), type=pa.string()),
            }
        ),
    )


def _build_stats(con: DuckDBPyConnection) -> dict[str, Any]:
    rows = con.execute(
        f"""
        SELECT
            c.{FLD_FILE_SOURCE_ID},
            d.reason,
            COUNT(*) AS records,
            SUM(c.{FLD_RECORD_LEN}) AS chars
        FROM dedup_candidates c
        LEFT JOIN dedup_rejections d USING ({FLD_RECORD_ID})
        GROUP BY c.{FLD_FILE_SOURCE_ID}, d.reason
        ORDER BY c.{FLD_FILE_SOURCE_ID}, d.reason
        """
    ).fetchall()

    totals = {
        "candidate_records": 0,
        "candidate_chars": 0,
        "kept_records": 0,
        "kept_chars": 0,
        "rejected_records": 0,
        "rejected_chars": 0,
    }
    by_reason: dict[str, dict[str, int]] = {
        REFERENCE_REASON: {"records": 0, "chars": 0},
        WITHIN_SNAPSHOT_REASON: {"records": 0, "chars": 0},
    }
    by_source: dict[str, dict[str, int]] = {}

    for source_id, reason, record_count, char_count in rows:
        records = int(record_count)
        chars = int(char_count)
        source = by_source.setdefault(
            str(source_id),
            {
                "candidate_records": 0,
                "candidate_chars": 0,
                "kept_records": 0,
                "kept_chars": 0,
                "rejected_records": 0,
                "rejected_chars": 0,
            },
        )
        totals["candidate_records"] += records
        totals["candidate_chars"] += chars
        source["candidate_records"] += records
        source["candidate_chars"] += chars

        if reason is None:
            totals["kept_records"] += records
            totals["kept_chars"] += chars
            source["kept_records"] += records
            source["kept_chars"] += chars
        else:
            totals["rejected_records"] += records
            totals["rejected_chars"] += chars
            source["rejected_records"] += records
            source["rejected_chars"] += chars
            by_reason[str(reason)]["records"] += records
            by_reason[str(reason)]["chars"] += chars

    return {**totals, "by_reason": by_reason, "by_source": by_source}


def _log_stats(stats: dict[str, Any]) -> None:
    candidate_records = int(stats["candidate_records"])
    candidate_chars = int(stats["candidate_chars"])
    rejected_records = int(stats["rejected_records"])
    rejected_chars = int(stats["rejected_chars"])
    by_reason = stats["by_reason"]

    console_log(
        "dedup",
        f"candidates: {candidate_records:_} records, {candidate_chars:_} chars",
    )
    console_log(
        "dedup",
        (
            f"rejected: {rejected_records:_} records "
            f"({_percentage(rejected_records, candidate_records):.2f}%), "
            f"{rejected_chars:_} chars "
            f"({_percentage(rejected_chars, candidate_chars):.2f}%)"
        ),
    )
    console_log(
        "dedup",
        (
            f"reasons: reference={by_reason[REFERENCE_REASON]['records']:_}, "
            f"within_snapshot={by_reason[WITHIN_SNAPSHOT_REASON]['records']:_}; "
            f"kept={stats['kept_records']:_} records, {stats['kept_chars']:_} chars"
        ),
    )


def _percentage(part: int, total: int) -> float:
    return 0.0 if total == 0 else 100.0 * part / total
