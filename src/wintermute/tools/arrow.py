from __future__ import annotations

from pathlib import Path
import pyarrow.parquet as pq

from wintermute.tools.logging import console_log

def rename_columns_inplace(
    path: str | Path,
    rename_map: dict[str, str],
) -> None:
    """
    Read a Parquet file, print existing columns,
    rename multiple columns in-place using a dict,
    rewrite the same file, then print the final columns.

    rename_map works like pandas: {"old": "new", "old2": "new2"}
    """

    path = Path(path)

    # Read input table
    table = pq.read_table(path)
    console_log("before", str(path))
    print("Columns:", table.column_names)

    # Validate rename map
    for old, new in rename_map.items():
        if old not in table.column_names:
            raise ValueError(f"Column '{old}' not found in {path}")
        if new in table.column_names and new != old:
            raise ValueError(f"Target column '{new}' already exists")

    # Build new column list
    new_cols = [
        rename_map.get(col, col)  # replace only if in dict
        for col in table.column_names
    ]

    # Apply rename
    table2 = table.rename_columns(new_cols)

    # Overwrite file in-place
    pq.write_table(table2, path)

    # Confirm
    table_out = pq.read_table(path)
    console_log("after ", str(path))
    print("Columns:", table_out.column_names)
