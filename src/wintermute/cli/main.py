from __future__ import annotations

from collections.abc import Sequence

from wintermute.cli.commands import build_parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    args.func(args)
    return 0
