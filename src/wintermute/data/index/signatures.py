from wintermute.data.index.config import SignatureConfig
from typing import List, Tuple
import xxhash
import re
import os
import shutil
from pathlib import Path
import unicodedata
import wintermute.data.constants as constants
from wintermute.tools.files import iter_jsonl_paths, jsonl_file_iterator
import pyarrow as pa
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from wintermute.tools.logging import console_log
from wintermute.tools.misc import compute_max_threads
from wintermute.data.text_fields import build_index_text


def write_signatures(config: SignatureConfig):
    temp_dir = _temp_dir_path(config)
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    _extract_signatures(config)
    _consolidate(config)
    shutil.rmtree(temp_dir)


def _consolidate(config: SignatureConfig) -> None:
    output_dir = _temp_dir_path(config)
    tmp_path = output_dir / "all.parquet.tmp"

    shard_paths = sorted(p for p in output_dir.glob("*.parquet"))
    if not shard_paths:
        raise ValueError(f"no shard parquet files found in {output_dir}")

    writer = None
    try:
        for shard_path in shard_paths:
            table = pq.read_table(shard_path)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    os.replace(tmp_path, Path(config.output_file_path))
    console_log("consolidate", "-> done")


def _write_file(path: Path, rows: List[Tuple[str, List[int]]]):
    uids = [uid for uid, _ in rows]
    sigs = [sig for _, sig in rows]

    table = pa.Table.from_arrays(
        [
            pa.array(uids, type=pa.string()),
            pa.array(sigs, type=pa.list_(pa.uint64())),
        ],
        names=[constants.FLD_GENERIC_UID, constants.FLD_GENERIC_SIGNATURE],
    )

    console_log("write_signatures", f"writing to {path}")
    pq.write_table(table, path, compression="zstd")


def _extract_signatures(config: SignatureConfig) -> None:
    input_paths = list(iter_jsonl_paths(config.raw_jsonl_path))
    
    max_threads =  compute_max_threads()
    with ProcessPoolExecutor(max_workers=max_threads) as ex:
        futures = [
            ex.submit(_extract_signatures_from_path, config, p)
            for p in input_paths
        ]

        for fut in as_completed(futures):
            fut.result()


def _temp_dir_path(config: SignatureConfig) -> Path:
    out_dir, _ = os.path.splitext(config.output_file_path)
    return Path(out_dir)


def _extract_signatures_from_path(config: SignatureConfig, input_path: str) -> None:
    name = os.path.basename(input_path).replace(".jsonl.gz", "").replace(".jsonl", "")
    out_path = _temp_dir_path(config) / f"{name}.parquet"

    rows: List[Tuple[str, List[int]]] = []
    cnt = 0
    for obj in jsonl_file_iterator(input_path):
        if cnt%1000 == 0:
            console_log("write_signatures", f"... {input_path} line {cnt}")
        
        cnt += 1
        r_uid = obj[constants.FLD_GENERIC_UID]
        text = build_index_text(obj, config.text_fields)
        ngrams = _extract_ngrams(text, config.ngram_size)
        sig = _minhash_splitmix(ngrams, config.num_hash)

        rows.append((r_uid, sig))

    _write_file(out_path, rows)


def _extract_ngrams(text: str, ngram_size: int) -> List[str]:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    tokens = re.findall(r"\w+", text)

    if len(tokens) < ngram_size:
        return []

    return [
        " ".join(tokens[i:i + ngram_size])
        for i in range(len(tokens) - ngram_size + 1)
    ]


MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & MASK64
    return (z ^ (z >> 31)) & MASK64


def _minhash_splitmix(ngrams: List[str], num_hash: int = 128) -> List[int]:
    base_hashes = [xxhash.xxh3_64(ng).intdigest() & MASK64 for ng in ngrams]

    sig = [MASK64] * num_hash
    for seed in range(num_hash):
        salt = (seed * 0x9E3779B97F4A7C15) & MASK64

        hmin = MASK64
        for h in base_hashes:
            mixed = _splitmix64(h ^ salt)
            if mixed < hmin:
                hmin = mixed

        sig[seed] = hmin

    return sig
