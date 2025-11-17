"""Utilities for converting RINEX observation and navigation files into JSON.

This module relies on GeoRinex to parse the RINEX contents into xarray
datasets and then serialises those datasets into JSON-friendly dictionaries.
It is designed to capture both the metadata/header information and the
time-indexed measurement data for downstream GNSS spoofing detection tooling.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Optional

import numpy as np
import xarray as xr

from rinex_loader import load_rinex_dataset
try:  # pandas is an optional dependency but normally present with GeoRinex
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - pandas ships with GeoRinex
    pd = None  # type: ignore


def _convert_scalar(value: Any) -> Any:
    """Convert a scalar value into a JSON-serialisable representation."""
    if value is None:
        return None

    if isinstance(value, (bool, str, int)):
        return value

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (np.generic,)):
        return _convert_scalar(value.item())

    if isinstance(value, (np.datetime64,)):
        return np.datetime_as_string(value, unit="ns")

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, (np.timedelta64,)):
        seconds = value / np.timedelta64(1, "ns")
        return seconds / 1e9

    if isinstance(value, timedelta):
        return value.total_seconds()

    if pd is not None and isinstance(value, pd.Timestamp):
        return value.isoformat()

    if pd is not None and isinstance(value, pd.Timedelta):
        return value.total_seconds()

    if isinstance(value, bytes):
        return value.decode("utf-8")

    return value


def _convert(value: Any) -> Any:
    """Recursively convert arrays, mappings, and scalars into JSON-friendly types."""
    if isinstance(value, MutableMapping):
        return {str(k): _convert(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_convert(v) for v in value]

    if isinstance(value, np.ndarray):
        return [_convert(v) for v in value.tolist()]

    return _convert_scalar(value)


def _clean_attrs(attrs: MutableMapping[str, Any]) -> Dict[str, Any]:
    return {str(k): _convert(v) for k, v in attrs.items()}


def _data_array_to_json(var: xr.DataArray) -> Dict[str, Any]:
    return {
        "dimensions": list(var.dims),
        "shape": list(var.shape),
        "attributes": _clean_attrs(var.attrs),
        "data": _convert(var.values.tolist()),
    }


def _coordinates_to_json(coords: MutableMapping[str, xr.DataArray]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for name, coord in coords.items():
        serialized[str(name)] = {
            "dimensions": list(coord.dims),
            "attributes": _clean_attrs(coord.attrs),
            "data": _convert(coord.values.tolist()),
        }
    return serialized


def _row_values_to_dict(row: "pd.Series", data_columns: Iterable[str]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for column in data_columns:
        value = row[column]
        if pd is not None and pd.isna(value):
            continue
        converted = _convert_scalar(value)
        if converted is not None:
            values[column] = converted
    return values


def _build_records(dataset: xr.Dataset, file_header_time: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Build flattened and indexed records for efficient querying.
    
    For navigation files, if file_header_time is provided, it will be used as the 
    primary timestamp for indexing (capture time) instead of TOC (broadcast time).
    """
    if pd is None or not dataset.data_vars:
        return None

    df = dataset.to_dataframe().reset_index()
    if df.empty:
        return None

    data_columns = list(dataset.data_vars)
    dims = list(dataset.dims)
    preferred_dim_order = ["time", "epoch", "datetime", "sv", "sat", "satellite", "prn"]
    ordered_dims = []
    for candidate in preferred_dim_order:
        if candidate in dims:
            ordered_dims.append(candidate)
    for dim in dims:
        if dim not in ordered_dims:
            ordered_dims.append(dim)
    dims = ordered_dims

    time_dim: Optional[str] = None
    satellite_dim: Optional[str] = None

    for dim in dims:
        dim_lower = dim.lower()
        if time_dim is None and any(
            keyword in dim_lower for keyword in ("time", "epoch", "datetime", "tow")
        ):
            time_dim = dim
        if satellite_dim is None and dim_lower in {"sv", "sat", "satellite", "prn"}:
            satellite_dim = dim

    if time_dim is None and dims:
        time_dim = dims[0]
    if satellite_dim is None:
        for candidate in ("sv", "sat", "satellite", "prn"):
            if candidate in df.columns:
                satellite_dim = candidate
                break

    flat_records: list[Dict[str, Any]] = []
    by_time: Dict[str, Dict[str, Any]] = {}
    by_constellation: Dict[str, Dict[str, Any]] = {}

    def ensure_satellite(container: Dict[str, Any], key: str) -> Dict[str, Any]:
        entry = container.get(key)
        if entry is None:
            entry = {"values": {}, "measurements": []}
            container[key] = entry
        return entry

    def store_measurement(
        entry: Dict[str, Any], values: Dict[str, Any], indices: Dict[str, Any]
    ) -> None:
        clean_indices = {k: v for k, v in indices.items() if v is not None}
        if clean_indices:
            entry.setdefault("measurements", []).append(
                {
                    "indices": clean_indices,
                    "values": values,
                }
            )
        else:
            entry.setdefault("values", {}).update(values)

    for _, row in df.iterrows():
        record: Dict[str, Any] = {}
        for dim in dims:
            record[dim] = _convert_scalar(row[dim])
        values = _row_values_to_dict(row, data_columns)
        if not values:
            continue
        record["values"] = values
        flat_records.append(record)

        epoch_key = _convert_scalar(row[time_dim]) if time_dim else None
        satellite_key = _convert_scalar(row[satellite_dim]) if satellite_dim else None

        # For navigation files with header time, use that as the primary timestamp
        # Store the TOC (broadcast time) in the values
        if file_header_time and epoch_key is not None:
            # Store the original TOC in the values
            if isinstance(epoch_key, (datetime, date)):
                values["TOC"] = epoch_key.isoformat()
            else:
                values["TOC"] = str(epoch_key)
            # Use file header time as the primary indexing key
            epoch_str = file_header_time
        else:
            # Standard behavior: use the epoch from the data
            if isinstance(epoch_key, (datetime, date)):
                epoch_key = epoch_key.isoformat()
            epoch_str = str(epoch_key) if epoch_key is not None else "unknown_epoch"
        satellite_str = str(satellite_key) if satellite_key is not None else "unknown_sv"

        const_prefix = "UNKNOWN"
        if isinstance(satellite_key, str) and satellite_key:
            prefix = "".join(ch for ch in satellite_key if not ch.isdigit())
            if prefix:
                const_prefix = prefix
            elif satellite_key[0].isalpha():
                const_prefix = satellite_key[0]

        extra_dims = {
            dim: record[dim]
            for dim in dims
            if dim not in {time_dim, satellite_dim}
        }

        epoch_entry = by_time.setdefault(
            epoch_str,
            {"satellites": {}, "constellations": {}},
        )
        sat_entry = ensure_satellite(epoch_entry["satellites"], satellite_str)
        store_measurement(sat_entry, values, extra_dims)

        constellation_entry = epoch_entry["constellations"].setdefault(
            const_prefix,
            {"satellites": {}},
        )
        constellation_sat = ensure_satellite(constellation_entry["satellites"], satellite_str)
        store_measurement(constellation_sat, values, extra_dims)

        constellation_bucket = by_constellation.setdefault(
            const_prefix,
            {},
        )
        epoch_bucket = constellation_bucket.setdefault(
            epoch_str,
            {"satellites": {}},
        )
        constellation_epoch_sat = ensure_satellite(epoch_bucket["satellites"], satellite_str)
        store_measurement(constellation_epoch_sat, values, extra_dims)

    if not flat_records and not by_time:
        return None

    return {
        "dimensions": dims,
        "time_dimension": time_dim,
        "satellite_dimension": satellite_dim,
        "flat": flat_records,
        "by_time": by_time,
        "by_constellation": by_constellation,
    }


def dataset_to_json(dataset: xr.Dataset, source: Path) -> Dict[str, Any]:
    """Convert an xarray.Dataset to a JSON-friendly dictionary."""
    # Extract file header timestamp for navigation files
    file_header_time = None
    rinex_type = dataset.attrs.get("rinextype")
    if rinex_type == "nav":
        from rinex_loader import _extract_file_time_from_header
        header_tuple = _extract_file_time_from_header(source)
        if header_tuple:
            year, month, day, hour, minute, second = header_tuple
            file_header_time = datetime(year, month, day, hour, minute, second).isoformat()
    
    return {
        "source": str(source),
        "rinex_type": rinex_type,
        "file_header_time": file_header_time,
        "dimensions": {str(dim): int(size) for dim, size in dataset.sizes.items()},
        "coordinates": _coordinates_to_json(dataset.coords),
        "data_variables": {
            str(name): _data_array_to_json(data_array)
            for name, data_array in dataset.data_vars.items()
        },
        "attributes": _clean_attrs(dataset.attrs),
        "indexed_records": _build_records(dataset, file_header_time=file_header_time),
    }


def parse_rinex(path: Path) -> Dict[str, Any]:
    """Load a RINEX file and convert its contents into a JSON-ready dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"RINEX file does not exist: {path}")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="georinex")
        warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
        dataset = load_rinex_dataset(path)

    return dataset_to_json(dataset, path)


def parse_multiple(paths: Iterable[Path]) -> Dict[str, Any]:
    """Parse multiple RINEX files and return a dictionary keyed by filename."""
    result: Dict[str, Any] = {}
    for path in paths:
        parsed = parse_rinex(path)
        result[path.name] = parsed
    return result


def cli(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert RINEX observation/navigation files into JSON.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Path(s) to RINEX files (.obs/.o, .nav/.n, etc.)",
    )
    parser.add_argument(
        "-o",
        "--output",
        help=(
            "Optional output JSON file. When multiple inputs are supplied, the JSON will map "
            "filenames to their parsed content."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON with indentation.",
    )
    args = parser.parse_args(argv)

    paths = [Path(p).expanduser().resolve() for p in args.inputs]
    json_obj: Dict[str, Any]
    if len(paths) == 1:
        json_obj = parse_rinex(paths[0])
    else:
        json_obj = parse_multiple(paths)

    indent = 2 if args.pretty else None
    json_str = json.dumps(json_obj, indent=indent, ensure_ascii=False, allow_nan=False)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.write_text(json_str, encoding="utf-8")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(cli())


