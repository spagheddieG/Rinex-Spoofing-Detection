"""Utility helpers for GNSS navigation spoofing detection workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EpochRecord:
    """Represents the navigation data broadcast for a satellite at a given epoch."""

    epoch: datetime
    values: Dict[str, Any]

    @property
    def iode(self) -> Optional[int]:
        value = self.values.get("IODE")
        return _to_int(value)

    @property
    def iodc(self) -> Optional[int]:
        value = self.values.get("IODC")
        return _to_int(value)


@dataclass(frozen=True)
class Finding:
    """Finding produced by spoofing detection rule."""

    satellite: str
    epoch: datetime
    code: str
    description: str
    details: Dict[str, Any]
    discovered_at: datetime


def load_nav_json(path: Path) -> Dict[str, Any]:
    """Load a navigation JSON file produced by ``rinex_to_json.py``."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_satellite_timeseries(indexed_records: Dict[str, Any]) -> Dict[str, List[EpochRecord]]:
    """Build per-satellite time series from the ``indexed_records`` structure."""
    by_time = indexed_records.get("by_time")
    if not isinstance(by_time, dict):
        raise ValueError("indexed_records missing 'by_time' dictionary")

    series: Dict[str, List[EpochRecord]] = {}
    for epoch_str, entry in sorted(by_time.items()):
        epoch_dt = _parse_epoch(epoch_str)
        satellites = entry.get("satellites", {})
        if not isinstance(satellites, dict):
            continue
        for sat, payload in satellites.items():
            # Try to get values from measurements array first (newer format)
            measurements = payload.get("measurements", [])
            if measurements and isinstance(measurements, list) and len(measurements) > 0:
                # Use the first measurement's values
                values = measurements[0].get("values")
                if not isinstance(values, dict):
                    continue
            else:
                # Fall back to direct values key (older format)
                values = payload.get("values")
                if not isinstance(values, dict):
                    continue
            record = EpochRecord(epoch=epoch_dt, values=values)
            series.setdefault(sat, []).append(record)

    # ensure per-satellite records sorted by epoch
    for sat in series:
        series[sat].sort(key=lambda record: record.epoch)

    return series


def detect_parameter_change_without_iode_change(
    records: Iterable[EpochRecord],
    satellite: str,
    tolerance: float = 0.0,
) -> List[Finding]:
    """Flag epochs where navigation values changed but IODE/IODC remained the same."""
    findings: List[Finding] = []
    prev: Optional[EpochRecord] = None
    for record in records:
        if prev is None:
            prev = record
            continue

        same_iode = prev.iode == record.iode
        same_iodc = prev.iodc == record.iodc
        if same_iode and same_iodc:
            changed_fields = _diff_values(prev.values, record.values, tolerance, ignore={"IODE", "IODC"})
            if changed_fields:
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="param_change_without_iode",
                        description="Navigation parameters changed without IODE/IODC update.",
                        details={"fields": changed_fields, "previous_epoch": prev.epoch.isoformat()},
                        discovered_at=record.epoch,
                    )
                )

        prev = record
    return findings


def detect_stale_data(
    records: Iterable[EpochRecord],
    satellite: str,
    max_interval: timedelta = timedelta(hours=2),
    by_time: Optional[Dict[str, Any]] = None,
) -> List[Finding]:
    """Flag long gaps where a satellite keeps broadcasting identical data.
    
    If by_time is provided, checks for missing entries between epochs to distinguish
    between truly stale data (repeated broadcasts) vs. gaps (no broadcasts).
    """
    findings: List[Finding] = []
    prev: Optional[EpochRecord] = None
    
    for record in records:
        if prev is None:
            prev = record
            continue

        delta = record.epoch - prev.epoch
        if delta > max_interval:
            same_values = not _diff_values(prev.values, record.values, 0.0, ignore=set())
            if same_values:
                # Check if there are missing entries between prev and current
                has_gaps = False
                if by_time:
                    # Check if there are epochs between prev and current where satellite doesn't appear
                    # This indicates a gap (satellite didn't broadcast) rather than stale data
                    for epoch_str, entry in by_time.items():
                        epoch_dt = _parse_epoch(epoch_str)
                        if prev.epoch < epoch_dt < record.epoch:
                            # Check if satellite appears at this intermediate epoch
                            satellites = entry.get("satellites", {})
                            if satellite not in satellites:
                                # This epoch exists but satellite doesn't appear - it's a gap
                                has_gaps = True
                                break
                
                if not has_gaps:
                    # No gaps detected - this is truly stale data
                    findings.append(
                        Finding(
                            satellite=satellite,
                            epoch=record.epoch,
                            code="stale_data",
                            description=f"Data unchanged for {delta.total_seconds()/3600:.1f} hours.",
                            details={
                                "elapsed_seconds": delta.total_seconds(),
                                "previous_epoch": prev.epoch.isoformat(),
                            },
                            discovered_at=record.epoch,
                        )
                    )
                # If has_gaps is True, we skip flagging it as stale since the satellite
                # simply didn't broadcast during that period

        prev = record
    return findings


def detect_unexpected_iod_changes(
    records: Iterable[EpochRecord],
    satellite: str,
) -> List[Finding]:
    """Warn when IODE/IODC regress while other navigation parameters change."""
    findings: List[Finding] = []
    prev: Optional[EpochRecord] = None
    for record in records:
        if prev is None:
            prev = record
            continue

        findings.extend(
            _detect_regression(
                previous=prev,
                current=record,
                satellite=satellite,
            )
        )

        prev = record

    return findings


# helper functions

def _parse_epoch(value: str) -> datetime:
    """Parse ISO-formatted epoch strings emitted by the JSON exporter."""
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        cleaned = value.replace("Z", "")
        return datetime.fromisoformat(cleaned)


def _diff_values(
    previous: Dict[str, Any],
    current: Dict[str, Any],
    tolerance: float,
    ignore: Iterable[str],
) -> Dict[str, Tuple[Any, Any]]:
    """Return fields that differ between two navigation snapshots."""
    ignore_set = set(ignore)
    differences: Dict[str, Tuple[Any, Any]] = {}
    keys = set(previous) | set(current)
    for key in keys:
        if key in ignore_set:
            continue
        a = previous.get(key)
        b = current.get(key)
        if _equal_with_tolerance(a, b, tolerance):
            continue
        differences[key] = (a, b)
    return differences


def _equal_with_tolerance(a: Any, b: Any, tolerance: float) -> bool:
    """Compare scalars with optional numeric tolerance."""
    if a == b:
        return True
    try:
        af = float(a)
        bf = float(b)
    except (TypeError, ValueError):
        return False
    return abs(af - bf) <= tolerance


def _to_int(value: Any) -> Optional[int]:
    """Safely convert navigation value to integer, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iod_jump_warning(previous: Optional[int], current: Optional[int], field: str) -> Optional[str]:
    """Return warning when an IOD value wraps backwards."""
    if previous is None or current is None:
        return None
    if current == previous:
        return None

    diff = (current - previous) % 1024

    if diff == 0:
        return None

    if diff == 1:
        return None

    if diff > 1 and diff < 1023:
        return None

    return f"{field} wrapped backwards from {previous} to {current}."


def _detect_regression(previous: EpochRecord, current: EpochRecord, satellite: str) -> List[Finding]:
    """Create findings when parameters change while IODE/IODC regress."""
    findings: List[Finding] = []

    prev_iode = previous.iode
    prev_iodc = previous.iodc
    curr_iode = current.iode
    curr_iodc = current.iodc

    same_values = not _diff_values(previous.values, current.values, 0.0, ignore=set())

    iodc_warning = _iod_jump_warning(prev_iodc, curr_iodc, "IODC")
    iode_warning = _iod_jump_warning(prev_iode, curr_iode, "IODE")

    if same_values:
        return findings

    warnings = {}
    if iodc_warning:
        warnings["IODC"] = iodc_warning
    if iode_warning:
        warnings["IODE"] = iode_warning

    if not warnings:
        return findings

    findings.append(
        Finding(
            satellite=satellite,
            epoch=current.epoch,
            code="iod_regression",
            description="Navigation parameters changed while IODE/IODC regressed.",
            details={
                "previous": {"IODE": prev_iode, "IODC": prev_iodc},
                "current": {"IODE": curr_iode, "IODC": curr_iodc},
                "warnings": warnings,
            },
            discovered_at=current.epoch,
        )
    )
    return findings


