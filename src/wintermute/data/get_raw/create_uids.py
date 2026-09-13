# -*- coding: utf-8 -*-

import json
import os

from wintermute.tools.files import iter_jsonl_paths, file_nr_from_path, jsonl_file_iterator, open_write
from wintermute.tools.logging import console_log

import wintermute.data.constants as constants

def create_uids(raw_dir: str, source_uid: str) -> None:
    input_paths = iter_jsonl_paths(raw_dir)
    total_records = 0
    for input_path in input_paths:
        file_nr = file_nr_from_path(input_path)
        tmp_path = f"{input_path}.tmp.gz"
        written = 0
        with open_write(tmp_path) as dst:
            record_nr = 0
            for obj in jsonl_file_iterator(input_path):
                record_nr += 1
                r_uid = f"{source_uid}_{file_nr}_{record_nr}"
                obj = {constants.FLD_GENERIC_UID: r_uid, **obj}
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

        os.replace(tmp_path, input_path)
        total_records += written
        console_log("create_uids", f"updated {written:_} records in {input_path}")
    console_log("create_uids", f"done; total records updated: {total_records:_}")
