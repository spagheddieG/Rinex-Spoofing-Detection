"""Helpers for loading RINEX datasets with GeoRinex plus local fallbacks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List

import georinex as gr
import numpy as np
import xarray as xr


def load_rinex_dataset(path: Path) -> xr.Dataset:
    """Load a RINEX file, retrying with duplicate-friendly logic when needed."""
    dataset = gr.load(path)
    if _dataset_has_data(dataset):
        return dataset

    version = dataset.attrs.get("version")
    rinex_type = dataset.attrs.get("rinextype")
    if rinex_type == "nav" and version and float(version) < 3.0:
        logging.warning(
            "GeoRinex produced an all-NaN dataset; retrying with duplicate-tolerant parser."
        )
        return _load_nav2_allow_duplicates(path)

    return dataset


def merge_nav_datasets(
    paths: Iterable[Path],
    *,
    preserve_sources: bool = False,
) -> xr.Dataset:
    """Merge multiple navigation datasets along the time dimension."""
    resolved_paths: List[Path] = [Path(p).expanduser().resolve() for p in paths]
    if not resolved_paths:
        raise ValueError("No RINEX files provided for merging.")

    datasets = [load_rinex_dataset(path).sortby("time") for path in resolved_paths]

    if preserve_sources:
        datasets_with_source = []
        for path, dataset in zip(resolved_paths, datasets, strict=True):
            label = path.stem
            # Extract quarter-hour offset from filename (e.g., x00=0, x15=15, x30=30, x45=45)
            offset_minutes = _extract_quarter_hour_offset(label)
            
            # Try to get file creation time from header
            header_time = _extract_file_time_from_header(path)
            
            # Use header time when available, even if no offset in filename
            # This ensures highrate files with 15-minute intervals use header time
            # instead of broadcast time (TOC)
            if len(dataset.time) > 0:
                from datetime import datetime, timedelta
                import pandas as pd
                
                if header_time:
                    # Use header time to determine the date and hour
                    year, month, day, hour, minute, second = header_time
                    file_datetime = datetime(year, month, day, hour, minute, second)
                    
                    if offset_minutes is not None:
                        # Handle hour rollover: if header minute is small (< 10) but filename says 45,
                        # it likely belongs to the previous hour
                        if minute < 10 and offset_minutes == 45:
                            # Roll back one hour
                            dataset_hour = (
                                (file_datetime - timedelta(hours=1))
                                .replace(minute=0, second=0, microsecond=0)
                            )
                        else:
                            # Round to the hour, then add quarter-hour offset from filename
                            dataset_hour = file_datetime.replace(minute=0, second=0, microsecond=0)
                        
                        # Add the quarter-hour offset
                        offset_timedelta = timedelta(minutes=offset_minutes)
                        normalized_base = dataset_hour + offset_timedelta
                    else:
                        # No offset in filename - use header time directly (rounded to nearest minute)
                        normalized_base = file_datetime.replace(second=0, microsecond=0)
                    
                    # For high-rate data, normalize ALL timestamps to the normalized base time
                    # This groups all measurements from the same source file together
                    new_times = pd.to_datetime([normalized_base] * len(dataset.time))
                    dataset = dataset.assign_coords(time=new_times.values)
                    
                    # Deduplicate within this dataset: keep latest per (time, sv)
                    # This handles cases where a file has multiple navigation messages
                    df = dataset.to_dataframe().reset_index()
                    value_columns = [col for col in df.columns if col not in {"time", "sv"}]
                    if value_columns:
                        df = df.dropna(how="all", subset=value_columns)
                    df = df.drop_duplicates(subset=["time", "sv"], keep="last")
                    dataset = df.set_index(["time", "sv"]).to_xarray()
            
            expanded = dataset.expand_dims({"source": [label]}).assign_coords(source=[label])
            datasets_with_source.append(expanded)

        combined = xr.concat(
            datasets_with_source,
            dim="source",
            join="outer",
            data_vars="all",
            coords="all",
        ).sortby("time")
        combined = _transpose_preferred(combined, preferred=("time", "source", "sv"))
    else:
        combined = xr.concat(datasets, dim="time", join="outer", data_vars="all").sortby("time")

        df = combined.to_dataframe().reset_index()
        value_columns = [column for column in df.columns if column not in {"time", "sv"}]
        if value_columns:
            df = df.dropna(how="all", subset=value_columns)
        df = df.drop_duplicates(subset=["time", "sv"], keep="last")
        combined = df.set_index(["time", "sv"]).to_xarray().sortby("time")

    combined.attrs.update(datasets[-1].attrs)
    combined.attrs["filename"] = "combined_nav"
    combined.attrs["sources"] = [str(path) for path in resolved_paths]
    combined.attrs["rinextype"] = "nav"

    for var in combined.data_vars:
        for dataset in reversed(datasets):
            if var in dataset.data_vars:
                combined[var].attrs = dataset[var].attrs
                break

    return combined


# internal helpers ----------------------------------------------------------------------


def _extract_file_time_from_header(path: Path) -> tuple[int, int, int, int, int, int] | None:
    """Extract file creation date/time from RINEX header (PGM / RUN BY / DATE line).
    
    Returns (year, month, day, hour, minute, second) or None if not found.
    """
    try:
        from georinex.nav2 import opener
        with opener(path) as fh:
            for line in fh:
                # Look for PGM / RUN BY / DATE line: format is typically
                # "program                    YYYYMMDD HH:MM:SSUTC..."
                if "PGM / RUN BY / DATE" in line or "RUN BY / DATE" in line:
                    # Try to extract date/time: YYYYMMDD HH:MM:SS
                    import re
                    # Match pattern like "20251110 00:16:17" or "2025 11 10 00:16:17"
                    match = re.search(
                        r"(\d{4})\s*(\d{2})\s*(\d{2})\s+(\d{2}):(\d{2}):(\d{2})", line
                    )
                    if match:
                        year, month, day, hour, minute, second = map(int, match.groups())
                        return (year, month, day, hour, minute, second)
                    # Try alternative format: YYYYMMDD HHMMSS
                    match = re.search(r"(\d{4})(\d{2})(\d{2})\s+(\d{2})(\d{2})(\d{2})", line)
                    if match:
                        year, month, day, hour, minute, second = map(int, match.groups())
                        return (year, month, day, hour, minute, second)
    except Exception:
        pass
    return None


def _extract_quarter_hour_offset(filename: str) -> int | None:
    """Extract quarter-hour offset from filename suffix (e.g., x00=0, x15=15, y30=30, z45=45).
    
    The letter prefix may change by hour, but the two-digit suffix always represents minutes.
    """
    import re
    # Match pattern: letter followed by two digits at the end (e.g., x00, y15, z30, a45)
    match = re.search(r"[a-z](\d{2})$", filename, re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _dataset_has_data(dataset: xr.Dataset) -> bool:
    for data_array in dataset.data_vars.values():
        values = getattr(data_array, "values", None)
        if values is None:
            continue
        if np.any(np.isfinite(values)):
            return True
    return False


def _load_nav2_allow_duplicates(path: Path) -> xr.Dataset:
    """Adaptation of GeoRinex nav2 reader that keeps the last duplicate per epoch."""
    from georinex.nav2 import (
        Nl,
        STARTCOL2,
        _timenav,
        navheader2,
        opener,
        rinex_string_to_float,
    )

    Lf = 19

    svs: list[str] = []
    times: list[Any] = []
    raws: list[str] = []

    with opener(path) as fh:
        header = navheader2(fh)
        filetype = header["filetype"]

        if filetype == "N":
            svtype = "G"
            fields = [
                "SVclockBias",
                "SVclockDrift",
                "SVclockDriftRate",
                "IODE",
                "Crs",
                "DeltaN",
                "M0",
                "Cuc",
                "Eccentricity",
                "Cus",
                "sqrtA",
                "Toe",
                "Cic",
                "Omega0",
                "Cis",
                "Io",
                "Crc",
                "omega",
                "OmegaDot",
                "IDOT",
                "CodesL2",
                "GPSWeek",
                "L2Pflag",
                "SVacc",
                "health",
                "TGD",
                "IODC",
                "TransTime",
                "FitIntvl",
            ]
        elif filetype == "G":
            svtype = "R"
            fields = [
                "SVclockBias",
                "SVrelFreqBias",
                "MessageFrameTime",
                "X",
                "dX",
                "dX2",
                "health",
                "Y",
                "dY",
                "dY2",
                "FreqNum",
                "Z",
                "dZ",
                "dZ2",
                "AgeOpInfo",
            ]
        elif filetype == "E":
            svtype = "E"
            fields = [
                "SVclockBias",
                "SVclockDrift",
                "SVclockDriftRate",
                "IODnav",
                "Crs",
                "DeltaN",
                "M0",
                "Cuc",
                "Eccentricity",
                "Cus",
                "sqrtA",
                "Toe",
                "Cic",
                "Omega0",
                "Cis",
                "Io",
                "Crc",
                "omega",
                "OmegaDot",
                "IDOT",
                "DataSrc",
                "GALWeek",
                "SISA",
                "health",
                "BGDe5a",
                "BGDe5b",
                "TransTime",
            ]
        else:
            raise NotImplementedError(
                f"RINEX 2 NAV file type {filetype!r} not supported for duplicate fallback."
            )

        for line in fh:
            try:
                toc = _timenav(line)
            except ValueError:
                continue

            svs.append(f"{svtype}{line[:2]}")
            times.append(toc)

            raw = line[22:79]
            for _ in range(Nl[header["systems"]]):
                raw += fh.readline()[STARTCOL2:79]
            raws.append(raw.replace("D", "E").replace("\n", ""))

    svs = [sv.replace(" ", "0") for sv in svs]
    unique_sv = sorted(set(svs))

    atimes = np.asarray(times)
    times_unique = np.unique(atimes)
    times_coord = np.array([np.datetime64(t, "ns") for t in times_unique], dtype="datetime64[ns]")
    data = np.full((len(fields), times_unique.size, len(unique_sv)), np.nan, dtype=float)

    for j, sv in enumerate(unique_sv):
        indices = [idx for idx, val in enumerate(svs) if val == sv]

        sv_times = atimes[indices]
        if np.unique(sv_times).size != sv_times.size:
            logging.warning(
                "duplicate times detected for %s; keeping the last occurrence per epoch", sv
            )

        for idx in indices:
            time_idx = np.nonzero(times_unique == times[idx])[0][0]
            raw = raws[idx]
            width = min(len(fields), len(raw) // Lf)
            values = [float(raw[k * Lf : (k + 1) * Lf]) for k in range(width)]
            data[:width, time_idx, j] = values

    dataset = xr.Dataset(coords={"time": times_coord, "sv": unique_sv})
    for i, name in enumerate(fields):
        if name is None:
            continue
        dataset[name] = (("time", "sv"), data[i, :, :])

    if svtype == "R":
        for name in {"X", "Y", "Z", "dX", "dY", "dZ", "dX2", "dY2", "dZ2"} & set(dataset.data_vars):
            dataset[name] *= 1e3

    dataset.attrs["version"] = header["version"]
    dataset.attrs["svtype"] = [svtype]
    dataset.attrs["rinextype"] = "nav"
    dataset.attrs["filename"] = Path(path).name

    if "ION ALPHA" in header and "ION BETA" in header:
        alpha = header["ION ALPHA"]
        beta = header["ION BETA"]
        alpha_vals = [rinex_string_to_float(alpha[2 + i * 12 : 2 + (i + 1) * 12]) for i in range(4)]
        beta_vals = [rinex_string_to_float(beta[2 + i * 12 : 2 + (i + 1) * 12]) for i in range(4)]
        dataset.attrs["ionospheric_corr_GPS"] = np.hstack((alpha_vals, beta_vals))

    return dataset


def _transpose_preferred(dataset: xr.Dataset, preferred: Iterable[str]) -> xr.Dataset:
    ordering: List[str] = [dim for dim in preferred if dim in dataset.dims]
    ordering.extend(dim for dim in dataset.dims if dim not in ordering)
    return dataset.transpose(*ordering, missing_dims="ignore")


