"""Combine multiple RINEX navigation files into a single JSON output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from rinex_loader import merge_nav_datasets
from rinex_to_json import dataset_to_json


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple RINEX navigation files into a single JSON export.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input files or directories containing navigation RINEX files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to write the combined JSON output.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the output JSON.",
    )
    parser.add_argument(
        "--per-source",
        action="store_true",
        help="Preserve each input file on a separate `source` dimension instead of collapsing duplicate epochs.",
    )
    return parser.parse_args(argv)


def expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    for raw in inputs:
        path = Path(raw).expanduser()
        if path.is_dir():
            paths.extend(sorted(child for child in path.iterdir() if child.is_file()))
        else:
            paths.append(path)
    unique_paths: List[Path] = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if resolved.is_file():
            unique_paths.append(resolved)
            seen.add(resolved)
    return unique_paths


def combine_to_json(paths: List[Path], *, per_source: bool) -> dict:
    dataset = merge_nav_datasets(paths, preserve_sources=per_source)
    primary_source = paths[0] if paths else Path(".")
    return dataset_to_json(dataset, primary_source)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    paths = expand_inputs(args.inputs)
    if not paths:
        raise SystemExit("No valid RINEX navigation files found in the provided inputs.")

    json_obj = combine_to_json(paths, per_source=args.per_source)
    indent = 2 if args.pretty else None
    output_path = Path(args.output).expanduser().resolve()
    output_path.write_text(json.dumps(json_obj, indent=indent, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote combined navigation JSON to {output_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


