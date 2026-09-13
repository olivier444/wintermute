from wintermute.data.index.config import IndexConfig
from typing import List, Tuple, Optional
import wintermute.data.constants as constants
from wintermute.tools.files import iter_jsonl_paths
import pyarrow as pa
from wintermute.data.constants import *
import os
from dataclasses import dataclass
import shutil
from wintermute.tools.tables import build_table_file_path, create_table, execute_sql
from wintermute.tools.files import file_nr_from_path, jsonl_file_iterator
from wintermute.tools.arrow import rename_columns_inplace
from wintermute.tools.logging import console_log
from wintermute.data.text_fields import build_index_text
import json

@dataclass
class RawRecord:
    uid: str
    lan: Optional[str]
    text_len: int
    text_snippet: str


def create_index(config: IndexConfig):
    records = _extract_raw_data(config)

    _create_datasource_table(config)
    _create_datafile_table(config, records)
    _create_record_table(config, records)
    _create_signature_table(config)
    _create_hash_bands(config)


def _create_hash_bands(config:IndexConfig) -> None:
    signature_file = build_table_file_path(TBL_SIGNATURE, config.output_root_path, config.source_id)
    band_dir  = build_table_file_path(TBL_BAND, config.output_root_path, config.source_id, as_dir=True, ensure_dir_exist=False)
    r = config.band_size
    k = config.num_hash
    if k%r != 0:
        raise Exception(f"Invalid band size ... r={r}, k={k}")
    b = int(k / r)

    console_log("index", f"creating band hashs from {signature_file}")
    console_log("index", f"r={r}, k={k}, b={b}")

    # DuckDB refuses writing partitioned parquet into an existing non-empty directory.
    # Recreate the per-source band directory so index builds can be rerun safely.
    if os.path.isdir(band_dir):
        shutil.rmtree(band_dir)
    os.makedirs(band_dir, exist_ok=True)

    sql = f"""
    COPY (
    SELECT
        {FLD_SIGNATURE_RECORD_ID} AS {FLD_BAND_RECORD_ID},
        band_id::SMALLINT AS {FLD_BAND_ID},
        hash(list_slice({FLD_SIGNATURE_VECTOR}, band_id * {r} + 1, (band_id + 1) * {r}))::UBIGINT AS {FLD_BAND_HASH}
    FROM read_parquet('{signature_file}')
    CROSS JOIN range({b}) bands(band_id)
    )
    TO '{band_dir}'
    (FORMAT PARQUET, PARTITION_BY ({FLD_BAND_ID}));
    """

    execute_sql(sql)

    console_log("index", f"bands created in {band_dir}")


def _create_record_table(config: IndexConfig, records: List[Tuple[str, List[RawRecord]]]) -> None:
    ids = []
    lans = []
    lengths = []
    file_ids = []
    snippets = []

    for file_data in records:
        file_path = file_data[0]
        for record in file_data[1]:
            ids.append(record.uid)
            lans.append(record.lan)
            lengths.append(record.text_len)
            file_ids.append(_file_id_from_path(config, file_path))
            snippets.append(record.text_snippet)

    content = {
        FLD_RECORD_ID: (ids, pa.string()),
        FLD_RECORD_FILE_ID: (file_ids, pa.string()),
        FLD_RECORD_LAN: (lans, pa.string()),
        FLD_RECORD_LEN: (lengths, pa.int64()),
        FLD_RECORD_SNIPPET: (snippets, pa.string()),
    }

    create_table(config.output_root_path, config.source_id, TBL_RECORD, content)


def _create_signature_table(config: IndexConfig) -> None:
    source_file_path = config.sig_path
    dest_file_path = build_table_file_path(TBL_SIGNATURE, config.output_root_path, config.source_id, ensure_dir_exist=True)
    shutil.copy(source_file_path, dest_file_path)
    rename_columns_inplace(dest_file_path, {FLD_GENERIC_UID: FLD_SIGNATURE_RECORD_ID, FLD_GENERIC_SIGNATURE: FLD_SIGNATURE_VECTOR})


def _create_datafile_table(config: IndexConfig, records: List[Tuple[str, List[RawRecord]]]) -> None:
    files = [file for (file, _) in records]

    content = {
        FLD_FILE_ID: ([_file_id_from_path(config, p) for p in files], pa.string()),
        FLD_FILE_SOURCE_ID: ([config.source_id] * len(files), pa.string()),
        FLD_FILE_NAME: ([os.path.basename(p) for p in files], pa.string()),
        FLD_FILE_PATH: (files, pa.string()),
    }

    create_table(config.output_root_path, config.source_id, TBL_FILE, content)


def _file_id_from_path(config: IndexConfig, p: str) -> str:
    return f"{config.source_id}_{file_nr_from_path(p)}"


def _create_datasource_table(config: IndexConfig) -> None:
    content = {
        FLD_SOURCE_ID: ([config.source_id], pa.string()),
        FLD_SOURCE_LABEL: ([config.source_label], pa.string()),
        FLD_SOURCE_DATASET: ([config.dataset_name], pa.string()),
        FLD_SOURCE_CONFIG: ([config.config_name], pa.string()),
        FLD_SOURCE_REVISION: ([config.revision], pa.string()),
        FLD_SOURCE_SPLIT: ([config.split], pa.string()),
        FLD_SOURCE_TXT_FIELDS: ([json.dumps(config.text_fields, ensure_ascii=False)], pa.string()),
        FLD_SOURCE_LAN_FIELD: ([config.lan_field], pa.string()),
    }

    create_table(config.output_root_path, config.source_id, TBL_SOURCE, content)


def _extract_raw_data(config: IndexConfig) -> List[Tuple[str, List[RawRecord]]]:
    input_paths = list(iter_jsonl_paths(config.raw_jsonl_path))
    res: List[Tuple[str, List[RawRecord]]] = []
    
    for input_path in input_paths:
        res.append((input_path, _extract_raw_data_from_path(config, input_path)))

    return res


def _extract_raw_data_from_path(config: IndexConfig, input_path: str) -> List[RawRecord]:
    console_log("index", f"reading [{input_path}]")
    rows: List[RawRecord] = []

    for obj in jsonl_file_iterator(input_path):
        r_uid = obj[constants.FLD_GENERIC_UID]
        text = build_index_text(obj, config.text_fields)

        lang = None if config.lan_field is None else obj[config.lan_field]
        rows.append(RawRecord(r_uid, lang, len(text), text[:100]))
    return rows
