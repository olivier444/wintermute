from __future__ import annotations

from typing import Any, Dict, List, Set

from wintermute.data.get_raw.preprocessors.base import AbstractPreProcessor, Item, ItemsIterable
from wintermute.tools.logging import console_log


class TreePreProcessor(AbstractPreProcessor):
    MESSAGES_FIELD = "messages"

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        params = dict(params or {})
        self.id_column = params.get("id_column", "message_id")
        self.parent_column = params.get("parent_column", "parent_id")
        self.role_column = params.get("role_column", "role")
        self.text_column = params.get("text_column", "text")
        self.null_value = params.get("null_value", "null")
        self.lang_field = params.get("language_field")

        self.languages: Set[str] | None = None
        languages = params.get("language")
        if languages is not None:
            self.languages = set(languages)

    def preprocess(self, items: ItemsIterable) -> ItemsIterable:
        def _accept(item: Item) -> bool:
            if self.lang_field is None or self.languages is None:
                return True

            item_lang = item[self.lang_field]
            return isinstance(item_lang, str) and item_lang.lower() in self.languages

        items = [item for item in items if _accept(item)]
        indexed: Dict[str, Item] = {item[self.id_column]: item for item in items}
        parents: Set[str] = {item[self.parent_column] for item in items}
        leaves: List[Item] = [item for item in items if item[self.id_column] not in parents]

        def _insert(item_list: List[Item], item: Item) -> None:
            item_list.insert(0, {
                self.role_column: item[self.role_column],
                self.text_column: item[self.text_column],
            })

        result = []
        for leaf in leaves:
            item = leaf
            thread: List[Item] = []

            _insert(thread, item)
            while item[self.parent_column] is not None and item[self.parent_column] != self.null_value:
                parent = indexed.get(item[self.parent_column])
                if parent is None:
                    break
                item = parent
                _insert(thread, item)

            if len(thread) > 1:
                thread_record = {self.MESSAGES_FIELD: thread}
                if self.lang_field is not None:
                    thread_record[self.lang_field] = leaf[self.lang_field]
                result.append(thread_record)

        console_log("tree-preproc", f"{len(items)} messages turned into {len(result)} conversations")
        return result
