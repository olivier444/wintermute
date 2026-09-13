from __future__ import annotations

from datetime import datetime


def console_log(sender: str, message: str) -> None:
    date_time = datetime.now().strftime("%Y%m%d %H:%M:%S")
    print(f"[{date_time}] [{sender}] {message}")
