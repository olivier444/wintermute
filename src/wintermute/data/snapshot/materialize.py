from __future__ import annotations

from random import Random
from wintermute.data.snapshot.config import SnapshotConfig
from wintermute.data.record import DataRecord
from wintermute.data.transform import TransformContext
from wintermute.data.transform.builder import build_record_transform
from typing import Dict, Tuple, List, Any, Set, Optional, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from wintermute.data.constants import *
from wintermute.tools.files import iter_jsonl_paths, get_raw_dir, get_materialized_dir, load_jsonl_file, JSonLShardWriter
from wintermute.tools.logging import console_log
from wintermute.tools.tables import execute_sql_as_dict, build_table_sql
from wintermute.tools.misc import batch_sequence, utc_now
from wintermute.data.text_fields import load_source_rows
import psutil


def materialize(
        config: SnapshotConfig
    ) -> None:
    materialized_dir = Path(get_materialized_dir(config.output_root, config.snapshot_id))

    if materialized_dir.exists():
        console_log("materialize", f"deleting existing materialized dir {materialized_dir}")
        shutil.rmtree(materialized_dir)

    sources = load_source_rows(config.get_index_path(), list(config.source_configs.keys()))
    source_transforms = {
        source_id: (
            build_record_transform(transform_config)
            if (transform_config := config.source_configs[source_id].record_transform) is not None
            else None
        )
        for source_id in sources.keys()
    }
    transform_context = TransformContext(rng=Random(config.seed), iteration=0)
    all_snapshot_records = _get_snapshot_records(config)
    nr_of_batches, record_locations = _dump_batches(all_snapshot_records, config, sources)
    missing_records_message = _missing_raw_records_message(all_snapshot_records, record_locations)
    if missing_records_message is not None:
        shutil.rmtree(materialized_dir / "batches", ignore_errors=True)
        raise RuntimeError(
            f"Snapshot '{config.snapshot_id}' cannot be materialized: {missing_records_message}"
        )

    current_file_by_batch: List[Optional[str]] = [None] * nr_of_batches
    current_data_by_batch: List[Optional[Dict[str, Dict[str, Any]]]] = [None] * nr_of_batches

    writers: Dict[str, JSonLShardWriter] = { 
        split: JSonLShardWriter((materialized_dir / split).as_posix(), config.max_bytes, 1, 4) 
        for split in config.split_sizes_chars.keys()
    }

    processed_uids: Dict[str, str] = {}

    for rec in all_snapshot_records:
        raw_rec_id = str(rec[FLD_MEMBERS_REC_ID])
        oversampled_rec_id = f"{rec[FLD_MEMBERS_REC_ID]}.{rec[FLD_MEMBERS_OVERSAMPLING]}"
        split = str(rec[FLD_MEMBERS_SPLIT])
        source_id = str(rec[FLD_FILE_SOURCE_ID])

        existing = processed_uids.get(raw_rec_id)
        if existing is not None and existing != split: # we allow oversampling in the same split
            raise RuntimeError(f"DUPLICATE REC_ID: {raw_rec_id} (from split '{existing}') is being used a second time in {split}")
        else:
            processed_uids[raw_rec_id] = split
        
        b, f = record_locations[oversampled_rec_id]
        if f != current_file_by_batch[b]:
            container: Dict[str, Dict[str, Any]] = {}
            load_jsonl_file(f, FLD_RECORD_ID, None, container)

            current_file_by_batch[b] = f
            current_data_by_batch[b] = container

        batch_records = current_data_by_batch[b]
        if batch_records is None:
            raise RuntimeError(f"Invalid materialize state: batch {b} was not loaded")

        raw_record = batch_records[oversampled_rec_id]
        input_record = DataRecord(
            record_id=oversampled_rec_id,
            fields={key: value for key, value in raw_record.items() if key != FLD_RECORD_ID},
        )
        transform = source_transforms[source_id]
        output_records = (
            transform.transform(input_record, context=transform_context)
            if transform is not None
            else (input_record,)
        )
        for output_record in output_records:
            writers[split].write({**output_record.fields, FLD_RECORD_ID: output_record.record_id})

    for writer in writers.values():
        writer.close()

    _write_materialized_manifest(config, materialized_dir)
    _clean_batch_files(set([f for (_, f) in record_locations.values()]))
    return


def _missing_raw_records_message(
        snapshot_records: Sequence[Dict[str, Any]],
        record_locations: Dict[str, Tuple[int, str]],
    ) -> Optional[str]:

    missing_by_source: Dict[str, List[str]] = {}
    for rec in snapshot_records:
        oversampled_rec_id = f"{rec[FLD_MEMBERS_REC_ID]}.{rec[FLD_MEMBERS_OVERSAMPLING]}"
        if oversampled_rec_id not in record_locations:
            source_id = str(rec[FLD_FILE_SOURCE_ID])
            missing_by_source.setdefault(source_id, []).append(oversampled_rec_id)

    if not missing_by_source:
        return None

    total_missing = sum(len(record_ids) for record_ids in missing_by_source.values())
    source_details = []
    for source_id, record_ids in sorted(missing_by_source.items()):
        examples = ", ".join(record_ids[:3])
        if len(record_ids) > 3:
            examples += ", …"
        source_details.append(f"{source_id}: {len(record_ids):_} ({examples})")

    return (
        f"{total_missing:_} raw record(s) selected by the snapshot are absent from the current raw shards. "
        "The raw data was removed or changed after indexing; rebuild the affected source index and create the snapshot again. "
        f"Missing by source: {'; '.join(source_details)}."
    )


def _write_materialized_manifest(config: SnapshotConfig, materialized_dir: Path) -> None:
    materialized_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = materialized_dir / "manifest.json"
    manifest = {
        "snapshot_id": config.snapshot_id,
        "created_at_utc": utc_now().isoformat(),
        "config": asdict(config),
        "outputs": {
            "materialized_dir": materialized_dir.as_posix(),
            "splits": {
                split: (materialized_dir / split).as_posix()
                for split in config.split_sizes_chars.keys()
            },
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _clean_batch_files(files: Set[str]) -> None:
    for f in files:
        console_log("materialize", f"deleting {f}")
        os.remove(f)

    subdirs = set([os.path.dirname(f) for f in files])
    for sd in subdirs:
        try:
            console_log("materialize", f"deleting {sd}")
            Path(sd).rmdir() # don't recurse here to avoid catastrophic errors
        except Exception as e:
            console_log("materialize", f"... error: [{e}]")

    dirs = [d for d in set([os.path.dirname(sd) for sd in subdirs])]
    if len(dirs) == 1:
        try:
            Path(dirs[0]).rmdir()
        except Exception as e:
            pass 

    return


def _dump_batches(
        all_snapshot_items: List[Dict[str, Any]], 
        config: SnapshotConfig,
        sources: Dict[str, Dict[str, Any]],
    ) -> Tuple[int, Dict[str, Tuple[int, str]]]: # nr_of_batches + map: record_id.oversampling => (batch_nr, file_path)

    materialized_dir = Path(get_materialized_dir(config.output_root, config.snapshot_id))

    all_input_files = _list_all_files(config.output_root, sources)

    total_ram = psutil.virtual_memory().total
    batch_size = int(total_ram / config.max_bytes / 10.)
    if batch_size == 0:
        raise RuntimeError(f"Invalid batch size: {batch_size} (total_ram={total_ram} and max_bytes={config.max_bytes})")

    console_log("materialize", f"RAM: {total_ram:_} bytes, batch_size: {batch_size:_} files")

    locations: Dict[str, Tuple[int, str]] = {}

    last_b = -1
    for b, batch in enumerate(batch_sequence(all_input_files, batch_size)):
        last_b = b
        batch_source_container = _load_fully(batch)

        subdir = materialized_dir / "batches" / f"batch_{b}"
        subdir.mkdir(parents=True, exist_ok=True)

        writer = JSonLShardWriter(subdir.as_posix(), config.max_bytes, 1, 4)

        for elected_item in all_snapshot_items:
            elected_source_rec_id = elected_item[FLD_MEMBERS_REC_ID]
            elected_write_rec_id = f"{elected_source_rec_id}.{elected_item[FLD_MEMBERS_OVERSAMPLING]}"
            elected_data = batch_source_container.get(elected_source_rec_id)
            if elected_data is not None:
                data_to_write = dict(elected_data)
                data_to_write[FLD_RECORD_ID] = elected_write_rec_id
                writer.write(data_to_write)
                locations[elected_write_rec_id] = (b, writer.data_path)

        writer.close()
    return (last_b + 1, locations)


def _get_snapshot_records(
        config: SnapshotConfig
    ) -> List[Dict[str, Any]]:
    
    members_table = build_table_sql(TBL_SNAPSHOT_MEMBERS, config.get_index_path(), config.snapshot_id)
    record_table = build_table_sql(TBL_RECORD, config.get_index_path())
    file_table = build_table_sql(TBL_FILE, config.get_index_path())

    sql = f"""
        SELECT
            m.{FLD_MEMBERS_REC_ID},
            m.{FLD_MEMBERS_OVERSAMPLING},
            m.{FLD_MEMBERS_SPLIT},
            f.{FLD_FILE_SOURCE_ID}
        FROM {members_table} m
        INNER JOIN {record_table} r ON r.{FLD_RECORD_ID} = m.{FLD_MEMBERS_REC_ID}
        INNER JOIN {file_table} f ON f.{FLD_FILE_ID} = r.{FLD_RECORD_FILE_ID}
        ORDER BY m.{FLD_MEMBERS_HASH} ASC
        """
    
    return execute_sql_as_dict(sql)


def _load_fully(
        input_files: Sequence[Tuple[str, str]],
    ) -> Dict[str, Dict[str, Any]]:

    container: Dict[str, Dict[str, Any]] = {}
    for input_path, _ in input_files:
        load_jsonl_file(input_path, FLD_GENERIC_UID, None, container)

    console_log("materialize", f"{len(container):_} records fully loaded in memory.")
    return container


def _list_all_files(
        root_dir: str, 
        sources: Dict[str, Dict[str, Any]]
    ) -> List[Tuple[str, str]]:

    result: List[Tuple[str, str]] = []
    for source in sources.values():
        src_id = source[FLD_SOURCE_ID]
        raw_dir = get_raw_dir(root_dir, source[FLD_SOURCE_LABEL])
        result.extend([(f, src_id) for f in iter_jsonl_paths(raw_dir)])

    return result
