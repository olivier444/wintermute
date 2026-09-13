from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any

from wintermute.data.constants import FLD_SOURCE_ID, TBL_SOURCE


def _text_values(
    record: Mapping[str, Any],
    text_fields: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(
        value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in (record[field] for field in text_fields)
    )


def build_index_text(record: Mapping[str, Any], text_fields: tuple[str, ...]) -> str:
    return "\n\n".join(_text_values(record, text_fields))


def build_display_text(record: Mapping[str, Any], text_fields: tuple[str, ...]) -> str:
    values = _text_values(record, text_fields)
    if len(text_fields) == 1:
        return values[0]
    return "\n\n".join(
        f"[{field}]\n{value}"
        for field, value in zip(text_fields, values)
    )


def load_source_rows(
    index_path: str,
    source_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, Any]]:
    from wintermute.tools.tables import build_table_sql, execute_sql_as_dict

    source_table = build_table_sql(TBL_SOURCE, index_path)
    rows = {
        str(item[FLD_SOURCE_ID]): item
        for item in execute_sql_as_dict(f"SELECT * FROM {source_table}")
    }
    if source_ids is None:
        return rows

    return {source_id: rows[source_id] for source_id in source_ids}
