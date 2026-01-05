#!/usr/bin/env python3
"""Combine multiple RINEX navigation files into a single JSON dataset.

This script merges navigation datasets, keeping the most recent message when
duplicate epochs are present, and outputs a single JSON file in the same format
as rinex_to_json.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable

from rinex_loader import merge_nav_datasets, _extract_quarter_hour_offset
from rinex_to_json import (
    _build_records,
    _clean_attrs,
    _coordinates_to_json,
    _data_array_to_json,
)


def find_rinex_files(paths: Iterable[Path]) -> list[Path]:
    """Find all RINEX navigation files from the given paths (files or directories)."""
    rinex_files: list[Path] = []
    for path in paths:
        resolved = Path(path).expanduser().resolve()
        if resolved.is_file():
            rinex_files.append(resolved)
        elif resolved.is_dir():
            # Find all .n and .25n files in directory
            rinex_files.extend(resolved.glob("*.n"))
            rinex_files.extend(resolved.glob("*.25n"))
            rinex_files.extend(resolved.glob("*.nav"))
        else:
            raise FileNotFoundError(f"Path does not exist: {resolved}")
    return sorted(set(rinex_files))  # Remove duplicates and sort


def combined_dataset_to_json(
    dataset, sources: list[Path], preserve_sources: bool = False
) -> Dict[str, Any]:
    """Convert a combined xarray.Dataset to JSON format.
    
    This is similar to dataset_to_json but handles combined datasets.
    """
    # Build the main JSON structure
    result: Dict[str, Any] = {
        "source": "combined_nav" if not preserve_sources else None,
        "rinex_type": dataset.attrs.get("rinextype", "nav"),
        "dimensions": {str(dim): int(size) for dim, size in dataset.sizes.items()},
        "coordinates": _coordinates_to_json(dataset.coords),
        "data_variables": {
            str(name): _data_array_to_json(data_array)
            for name, data_array in dataset.data_vars.items()
        },
        "attributes": _clean_attrs(dataset.attrs),
    }
    
    # Build indexed records
    indexed_records = _build_records(dataset, file_header_time=None)
    if indexed_records:
        result["indexed_records"] = indexed_records
    
    # Store source file paths in attributes
    result["attributes"]["sources"] = [str(path) for path in sources]
    
    return result


def cli(argv: Iterable[str] | None = None) -> int:
    """Command-line interface for combining navigation files."""
    parser = argparse.ArgumentParser(
        description=(
            "Combine multiple RINEX navigation files into a single JSON dataset. "
            "When duplicate epochs are present, the most recent message is kept."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to RINEX navigation files or directories containing .n/.25n files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--per-source",
        action="store_true",
        help=(
            "Retain each file on its own 'source' dimension. "
            "Useful for high-rate captures where multiple uploads share the same epoch."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON with indentation",
    )
    args = parser.parse_args(argv)
    
    # Find all RINEX files
    input_paths = [Path(p) for p in args.inputs]
    rinex_files = find_rinex_files(input_paths)
    
    if not rinex_files:
        print("Error: No RINEX navigation files found in the provided paths.")
        return 1
    
    # Auto-detect highrate data (files with quarter-hour offsets)
    # Highrate data should use --per-source to keep each file as its own entry
    auto_per_source = args.per_source
    if not auto_per_source:
        # Check if any files have quarter-hour offsets (highrate pattern)
        has_highrate_pattern = any(
            _extract_quarter_hour_offset(f.stem) is not None for f in rinex_files
        )
        if has_highrate_pattern:
            auto_per_source = True
            print("Detected highrate data pattern - using --per-source to keep each file as separate entry")
    
    print(f"Found {len(rinex_files)} RINEX navigation file(s)")
    if len(rinex_files) <= 10:
        for f in rinex_files:
            print(f"  - {f}")
    else:
        for f in rinex_files[:5]:
            print(f"  - {f}")
        print(f"  ... and {len(rinex_files) - 5} more")
    
    # Merge the datasets
    print("\nCombining navigation files...")
    try:
        combined_dataset = merge_nav_datasets(
            rinex_files,
            preserve_sources=auto_per_source,
        )
    except Exception as e:
        print(f"Error combining files: {e}")
        return 1
    
    # Convert to JSON
    print("Converting to JSON format...")
    json_obj = combined_dataset_to_json(
        combined_dataset,
        sources=rinex_files,
        preserve_sources=auto_per_source,
    )
    
    # Write output
    output_path = Path(args.output).expanduser().resolve()
    indent = 2 if args.pretty else None
    json_str = json.dumps(json_obj, indent=indent, ensure_ascii=False, allow_nan=False)
    output_path.write_text(json_str, encoding="utf-8")
    
    print(f"\nSuccessfully combined {len(rinex_files)} file(s) into: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())

