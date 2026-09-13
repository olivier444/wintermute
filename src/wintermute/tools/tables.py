from pathlib import Path
from wintermute.data.constants import *
import os
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Any, Dict, List, Tuple, Optional
from wintermute.tools.logging import console_log
from wintermute.tools.misc import compute_max_threads, compute_max_memory_gb
import duckdb


def _escape_sql_string(value: str) -> str:
    return value.replace("'", "''")


def _build_duckdb_temp_directory(root_path: str) -> str:
    temp_dir = os.path.join(root_path, "_duckdb_tmp")
    Path(temp_dir).mkdir(parents=True, exist_ok=True)
    return temp_dir


def create_duckdb_connection(temp_directory_root: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"PRAGMA threads = {compute_max_threads()};")
    con.execute(f"PRAGMA memory_limit = '{compute_max_memory_gb()}GB';")
    con.execute("SET preserve_insertion_order = false;")

    if temp_directory_root:
        temp_dir = _build_duckdb_temp_directory(temp_directory_root)
        con.execute(f"SET temp_directory = '{_escape_sql_string(temp_dir)}';")

    return con


def build_table_dir_path(table: str, index_root_path: str, ensure_dir_exist: bool = False) -> str:
    dir = os.path.join(index_root_path, table)

    if ensure_dir_exist:
        dp = Path(dir)
        dp.mkdir(parents=True, exist_ok=True)

    return dir


def build_table_sql(table: str, index_root_path: str, file_name: Optional[str] = None) -> str:
    data_dir = build_table_dir_path(table, index_root_path, ensure_dir_exist=False)
    return f"read_parquet('{data_dir}/{file_name or "*"}.parquet', union_by_name=1)"


def build_table_file_path(
        table: str, 
        index_root_path: str,
        file_name: str,
        ensure_dir_exist: bool = False, 
        as_dir: bool = False
    ) -> str:

    dir = build_table_dir_path(table, index_root_path, ensure_dir_exist)
    ext = "" if as_dir else ".parquet"
    result = os.path.join(dir, f"{file_name}{ext}")

    if ensure_dir_exist and as_dir:
        dp = Path(result)
        dp.mkdir(parents=True, exist_ok=True)

    return result


def execute_sql(sql: str, *, temp_directory_root: Optional[str] = None):
    con = create_duckdb_connection(temp_directory_root=temp_directory_root)
    con.execute(sql)
    con.close()


def execute_sql_as_numpy(sql: str, *, temp_directory_root: Optional[str] = None):
    con = create_duckdb_connection(temp_directory_root=temp_directory_root)
    tbl = con.sql(sql).fetch_arrow_table() 

    result = (
        tbl.num_rows, 
        {col:tbl[col].to_numpy() for col in tbl.column_names}
    )
    
    con.close()
    return result


def execute_sql_as_dict(sql: str, *, temp_directory_root: Optional[str] = None) -> List[Dict[str, Any]]:
    con = create_duckdb_connection(temp_directory_root=temp_directory_root)
    res = con.execute(sql)

    cols = [c[0] for c in res.description]
    rows = res.fetchall()

    result = [dict(zip(cols, row)) for row in rows]
    con.close()
    return result


def create_table(
        index_root_path: str, 
        file_name: str,
        table:str, 
        content: Dict[str, Tuple[List[Any], Any]]
    ) -> None:

    file_path = build_table_file_path(table, index_root_path, file_name, ensure_dir_exist=True)

    data = []
    names = []

    rows = 0
    for (name, desc) in content.items():
        names.append(name)

        list = desc[0]
        rows = len(list)
        type = desc[1]
        data.append(pa.array(list, type=type))

    table = pa.Table.from_arrays(data, names=names)

    console_log("index", f"writing {file_path}")
    console_log("index", f"columns: {', '.join(names)}")
    console_log("index", f"rows: {rows:_}")
    pq.write_table(table, file_path, compression="zstd")
