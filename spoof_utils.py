"""Utility helpers for GNSS navigation spoofing detection workflow."""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class EpochRecord:
    """Represents the navigation data broadcast for a satellite at a given epoch."""

    epoch: datetime
    values: Dict[str, Any]
    source: Optional[str] = None

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
            # Handle multi-dimensional data (e.g., with source dimension)
            measurements = payload.get("measurements", [])
            if measurements and isinstance(measurements, list):
                # Multi-source data - process each measurement
                for measurement in measurements:
                    indices = measurement.get("indices", {})
                    source = indices.get("source")
                    values = measurement.get("values", {})
                    if isinstance(values, dict):
                        record = EpochRecord(epoch=epoch_dt, values=values, source=source)
                        series.setdefault(sat, []).append(record)
            else:
                # Single source data - fall back to direct values key
                values = payload.get("values")
                if not isinstance(values, dict):
                    continue
                record = EpochRecord(epoch=epoch_dt, values=values)
                series.setdefault(sat, []).append(record)

    # ensure per-satellite records sorted by epoch
    for sat in series:
        series[sat].sort(key=lambda record: record.epoch)

    return series


def extract_satellite_timeseries_multisource(data: Dict[str, Any]) -> Dict[str, List[EpochRecord]]:
    """Build per-satellite time series from multi-source JSON structure (like tst.json)."""
    series: Dict[str, List[EpochRecord]] = {}

    # Check if this is multi-source format
    top_keys = list(data.keys())
    if not top_keys or not any(key.endswith(('.25n', '.25o', '.n', '.o')) for key in top_keys[:5]):
        # Not multi-source, fall back to regular extraction
        if 'indexed_records' in data:
            return extract_satellite_timeseries(data['indexed_records'])
        else:
            return {}

    # Process each source file
    for source_filename, source_data in data.items():
        if not isinstance(source_data, dict) or 'indexed_records' not in source_data:
            continue

        source_indexed = source_data['indexed_records']
        source_by_time = source_indexed.get('by_time', {})

        for epoch_str, entry in source_by_time.items():
            epoch_dt = _parse_epoch(epoch_str)
            satellites = entry.get("satellites", {})
            if not isinstance(satellites, dict):
                continue

            for sat, payload in satellites.items():
                values = payload.get("values")
                if not isinstance(values, dict):
                    continue

                record = EpochRecord(epoch=epoch_dt, values=values, source=source_filename)
                series.setdefault(sat, []).append(record)

    # Ensure per-satellite records sorted by epoch
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
            changed_fields = _diff_values(
                prev.values, record.values, tolerance, ignore={"IODE", "IODC"}
            )
            if changed_fields:
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="param_change_without_iode",
                        description="Navigation parameters changed without IODE/IODC update.",
                        details={
                            "fields": changed_fields,
                            "previous_epoch": prev.epoch.isoformat(),
                        },
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
                    # Check if there are epochs between prev and current where
                    # satellite doesn't appear
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
                            description=(
                                f"Data unchanged for {delta.total_seconds()/3600:.1f} hours."
                            ),
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


def _detect_regression(
    previous: EpochRecord, current: EpochRecord, satellite: str
) -> List[Finding]:
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


def detect_redundancy_inconsistencies(
    records: Iterable[EpochRecord],
    satellite: str,
    tolerance: float = 0.0,
) -> List[Finding]:
    """Detect spoofing by checking for parameter inconsistencies between multiple sources claiming the same broadcast timestamp."""
    findings: List[Finding] = []

    # Group records by epoch to find multiple sources per timestamp
    epoch_groups: Dict[datetime, List[EpochRecord]] = {}
    for record in records:
        epoch_groups.setdefault(record.epoch, []).append(record)

    # Check each epoch that has multiple sources
    for epoch, epoch_records in epoch_groups.items():
        if len(epoch_records) < 2:
            continue  # Need at least 2 sources to check consistency

        # Group by source if available, otherwise treat all as separate measurements
        source_groups: Dict[Optional[str], List[EpochRecord]] = {}
        for record in epoch_records:
            source = record.source
            source_groups.setdefault(source, []).append(record)

        # If we have multiple sources for the same epoch, check consistency
        if len(source_groups) > 1:
            # Compare parameters across sources
            reference_record = epoch_records[0]
            reference_values = reference_record.values

            key_params = ['SVclockBias', 'SVclockDrift', 'SVclockDriftRate', 'IODE', 'IODC', 'Crs', 'Crc']

            for source, source_records in source_groups.items():
                if source == reference_record.source:
                    continue  # Skip comparing reference to itself

                for source_record in source_records:
                    inconsistent_params = []
                    for param in key_params:
                        ref_value = reference_values.get(param)
                        src_value = source_record.values.get(param)

                        # Skip if either value is None
                        if ref_value is None or src_value is None:
                            continue

                        # Check for exact equality (or within tolerance for floats)
                        try:
                            ref_float = float(ref_value)
                            src_float = float(src_value)
                            if abs(ref_float - src_float) > tolerance:
                                inconsistent_params.append(param)
                        except (ValueError, TypeError):
                            # For non-numeric values, check exact equality
                            if ref_value != src_value:
                                inconsistent_params.append(param)

                    if inconsistent_params:
                        findings.append(
                            Finding(
                                satellite=satellite,
                                epoch=epoch,
                                code="REDUNDANCY_INCONSISTENCY",
                                description=(
                                    f"Parameter inconsistency between sources at same broadcast timestamp. "
                                    f"Inconsistent: {', '.join(inconsistent_params)}. "
                                    f"Sources: {reference_record.source or 'unknown'} vs {source or 'unknown'}"
                                ),
                                details={
                                    "inconsistent_parameters": inconsistent_params,
                                    "reference_source": reference_record.source,
                                    "comparison_source": source,
                                    "reference_values": {p: reference_values.get(p) for p in inconsistent_params},
                                    "comparison_values": {p: source_record.values.get(p) for p in inconsistent_params},
                                    "all_sources": list(source_groups.keys()),
                                    "total_sources": len(source_groups),
                                },
                                discovered_at=datetime.now(),
                            )
                        )

    return findings


def detect_ephemeris_age_anomalies(
    records: Iterable[EpochRecord],
    satellite: str,
    max_age_hours: float = 4.0,
) -> List[Finding]:
    """Detect stale or inconsistent Toe (Time of Ephemeris) values."""
    findings: List[Finding] = []
    
    max_age_seconds = max_age_hours * 3600.0
    prev_toe: Optional[float] = None
    
    for record in records:
        toe_value = _parse_toe(record.values.get("Toe"))
        if toe_value is None:
            continue
        
        # Check if Toe is too old relative to capture time
        # Toe is in seconds-of-week (SOW), not absolute GPS seconds
        # We need to convert it using GPSWeek: absolute_gps_seconds = (GPSWeek * 604800) + Toe
        gps_week = _to_int(record.values.get("GPSWeek"))
        if gps_week is None:
            # If GPSWeek is missing, we can't calculate absolute time, skip this check
            continue
        
        # Convert Toe from seconds-of-week to absolute GPS seconds
        seconds_per_week = 604800  # 7 * 24 * 3600
        toe_absolute_gps_seconds = (gps_week * seconds_per_week) + toe_value
        
        # Calculate capture time in absolute GPS seconds
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        capture_gps_seconds = (record.epoch - gps_epoch).total_seconds()
        
        # Calculate age (how old the ephemeris is)
        toe_age_seconds = abs(capture_gps_seconds - toe_absolute_gps_seconds)
        
        if toe_age_seconds > max_age_seconds:
            findings.append(
                Finding(
                    satellite=satellite,
                    epoch=record.epoch,
                    code="ephemeris_age_anomaly",
                    description=(
                        f"Ephemeris age {toe_age_seconds/3600:.2f} hours exceeds "
                        f"maximum {max_age_hours:.1f} hours"
                    ),
                    details={
                        "toe_value": toe_value,
                        "toe_absolute_gps_seconds": toe_absolute_gps_seconds,
                        "gps_week": gps_week,
                        "toe_age_hours": toe_age_seconds / 3600.0,
                        "max_age_hours": max_age_hours,
                        "capture_gps_seconds": capture_gps_seconds,
                    },
                    discovered_at=record.epoch,
                )
            )
        
        # Check if Toe regresses (goes backwards)
        if prev_toe is not None:
            # Allow small backward change due to wrap-around or minor errors
            if toe_value < prev_toe - 3600:  # More than 1 hour backward
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="toe_regression",
                        description=(
                            f"Toe regressed from {prev_toe:.1f} to {toe_value:.1f} "
                            f"({prev_toe - toe_value:.1f} seconds backward)"
                        ),
                        details={
                            "previous_toe": prev_toe,
                            "current_toe": toe_value,
                            "regression_seconds": prev_toe - toe_value,
                        },
                        discovered_at=record.epoch,
                    )
                )
        
        prev_toe = toe_value
    
    return findings


def detect_replay_patterns(
    records: Iterable[EpochRecord],
    satellite: str,
    sequence_length: int = 4,
    parameters: Optional[List[str]] = None,
) -> List[Finding]:
    """Detect repeated sequences of parameter values suggesting replay attacks."""
    findings: List[Finding] = []
    
    if parameters is None:
        parameters = ["SVclockBias", "IODE", "IODC", "Crs", "Crc"]
    
    # Convert to list and sort by epoch
    records_list = list(records)
    records_list.sort(key=lambda r: r.epoch)
    
    if len(records_list) < sequence_length * 2:
        return findings  # Need at least 2 sequences to detect repetition
    
    # Extract sequences for each parameter
    for param in parameters:
        sequences: Dict[str, List[int]] = {}  # hash -> list of starting indices
        
        for i in range(len(records_list) - sequence_length + 1):
            sequence_hash = _extract_sequence_hash(
                records_list[i:i+sequence_length], param, sequence_length
            )
            if sequence_hash is None:
                continue
            
            if sequence_hash not in sequences:
                sequences[sequence_hash] = []
            sequences[sequence_hash].append(i)
        
        # Find repeated sequences (filter out overlapping sequences)
        for seq_hash, indices in sequences.items():
            if len(indices) < 2:
                continue
            
            # Filter out overlapping sequences - only consider sequences that are
            # separated by at least sequence_length positions (non-overlapping)
            non_overlapping_indices = []
            for idx in sorted(indices):
                # Check if this index is far enough from the last non-overlapping one
                if not non_overlapping_indices or idx >= non_overlapping_indices[-1] + sequence_length:
                    non_overlapping_indices.append(idx)
            
            # Only flag if we have at least 2 non-overlapping occurrences
            if len(non_overlapping_indices) >= 2:
                # Calculate time gaps between occurrences to ensure they're separated in time
                time_gaps = []
                for i in range(len(non_overlapping_indices) - 1):
                    idx1 = non_overlapping_indices[i]
                    idx2 = non_overlapping_indices[i + 1]
                    # Time gap is from end of first sequence to start of second
                    time_gap = (records_list[idx2].epoch - records_list[idx1 + sequence_length - 1].epoch).total_seconds()
                    time_gaps.append(time_gap)
                
                # Only flag if sequences are separated by at least 1 hour (3600 seconds)
                # This ensures we're detecting actual replays, not just overlapping windows
                min_gap_seconds = 3600.0
                if any(gap >= min_gap_seconds for gap in time_gaps):
                    findings.append(
                        Finding(
                            satellite=satellite,
                            epoch=records_list[non_overlapping_indices[-1]].epoch,
                            code="replay_pattern",
                            description=(
                                f"Repeated sequence of {param} values detected "
                                f"({len(non_overlapping_indices)} non-overlapping occurrences)"
                            ),
                            details={
                                "parameter": param,
                                "sequence_length": sequence_length,
                                "occurrence_count": len(non_overlapping_indices),
                                "occurrence_indices": non_overlapping_indices,
                                "sequence_hash": seq_hash,
                                "time_gaps_seconds": time_gaps,
                            },
                            discovered_at=records_list[non_overlapping_indices[-1]].epoch,
                        )
                    )
    
    return findings


def detect_temporal_source_inconsistencies(
    records: Iterable[EpochRecord],
    satellite: str,
    time_window: timedelta = timedelta(minutes=30),
    tolerance: float = 1e-9,
) -> List[Finding]:
    """Detect parameter inconsistencies between sources within time windows."""
    findings: List[Finding] = []
    
    records_list = list(records)
    if len(records_list) < 2:
        return findings
    
    # Group records by time window
    window_groups = _group_by_time_window(records_list, time_window)
    
    key_params = ["SVclockBias", "IODE", "IODC", "Crs", "Crc"]
    
    for window_start, window_records in window_groups.items():
        if len(window_records) < 2:
            continue
        
        # Group by source
        source_groups: Dict[Optional[str], List[EpochRecord]] = {}
        for record in window_records:
            source_groups.setdefault(record.source, []).append(record)
        
        if len(source_groups) < 2:
            continue  # Need multiple sources
        
        # Compare parameters across sources in this window
        sources_list = list(source_groups.keys())
        reference_source = sources_list[0]
        reference_records = source_groups[reference_source]
        
        if not reference_records:
            continue
        
        reference_record = reference_records[0]
        
        for other_source in sources_list[1:]:
            other_records = source_groups.get(other_source, [])
            if not other_records:
                continue
            
            for other_record in other_records:
                inconsistent_params = []
                for param in key_params:
                    ref_value = _to_float(reference_record.values.get(param))
                    other_value = _to_float(other_record.values.get(param))
                    
                    if ref_value is None or other_value is None:
                        continue
                    
                    if abs(ref_value - other_value) > tolerance:
                        inconsistent_params.append(param)
                
                if inconsistent_params:
                    findings.append(
                        Finding(
                            satellite=satellite,
                            epoch=other_record.epoch,
                            code="temporal_source_inconsistency",
                            description=(
                                f"Inconsistent parameters between sources in time window: "
                                f"{', '.join(inconsistent_params)}"
                            ),
                            details={
                                "inconsistent_parameters": inconsistent_params,
                                "reference_source": reference_source,
                                "comparison_source": other_source,
                                "time_window_start": window_start.isoformat(),
                                "time_window_size_minutes": time_window.total_seconds() / 60.0,
                            },
                            discovered_at=other_record.epoch,
                        )
                    )
    
    return findings


def detect_parameter_velocity_anomalies(
    records: Iterable[EpochRecord],
    satellite: str,
    parameter: str = "SVclockBias",
    max_acceleration: float = 1e-12,
) -> List[Finding]:
    """Detect anomalous acceleration (rate of change of velocity) in parameters."""
    findings: List[Finding] = []
    
    prev: Optional[EpochRecord] = None
    prev_velocity: Optional[float] = None
    
    for record in records:
        if prev is None:
            prev = record
            continue
        
        velocity = _calculate_parameter_velocity(prev, record, parameter)
        if velocity is None:
            prev = record
            continue
        
        if prev_velocity is not None:
            time_delta = (record.epoch - prev.epoch).total_seconds()
            if time_delta > 0:
                acceleration = (velocity - prev_velocity) / time_delta
                abs_acceleration = abs(acceleration)
                
                if abs_acceleration > max_acceleration:
                    findings.append(
                        Finding(
                            satellite=satellite,
                            epoch=record.epoch,
                            code="parameter_acceleration_anomaly",
                            description=(
                                f"Parameter {parameter} acceleration {abs_acceleration:.6e} "
                                f"exceeds maximum {max_acceleration:.6e}"
                            ),
                            details={
                                "parameter": parameter,
                                "acceleration": acceleration,
                                "max_acceleration": max_acceleration,
                                "velocity": velocity,
                                "previous_velocity": prev_velocity,
                            },
                            discovered_at=record.epoch,
                        )
                    )
        
        prev_velocity = velocity
        prev = record
    
    return findings


def detect_cross_satellite_correlations(
    all_timeseries: Dict[str, List[EpochRecord]],
    min_correlation: float = 0.9,
    parameters: Optional[List[str]] = None,
) -> List[Finding]:
    """Detect suspicious correlations between satellites (possible coordinated spoofing)."""
    findings: List[Finding] = []
    
    # This is a simplified version - full correlation analysis would be more complex
    # For now, we'll check if multiple satellites show anomalies at the same time
    
    if parameters is None:
        parameters = ["SVclockBias"]
    
    if len(all_timeseries) < 2:
        return findings
    
    # Group records by epoch to find simultaneous anomalies
    epoch_groups: Dict[datetime, List[Tuple[str, EpochRecord]]] = {}
    for sat, records in all_timeseries.items():
        for record in records:
            epoch_groups.setdefault(record.epoch, []).append((sat, record))
    
    # Check each epoch for multiple satellites
    for epoch, sat_records in epoch_groups.items():
        if len(sat_records) < 2:
            continue
        
        # For each parameter, check if multiple satellites have similar anomalous values
        for param in parameters:
            values: List[Tuple[str, float]] = []
            for sat, record in sat_records:
                value = _to_float(record.values.get(param))
                if value is not None:
                    values.append((sat, value))
            
            if len(values) < 2:
                continue
            
            # Calculate variance - low variance with many satellites suggests correlation
            value_list = [v for _, v in values]
            if len(value_list) >= 2:
                stats = _calculate_statistics(value_list)
                std = stats["std"]
                mean = stats["mean"]
                
                # Low relative standard deviation with multiple satellites
                if mean != 0 and len(values) >= 3:
                    relative_std = abs(std / mean) if mean != 0 else 0.0
                    if relative_std < 0.01:  # Very low variance
                        findings.append(
                            Finding(
                                satellite="MULTIPLE",
                                epoch=epoch,
                                code="cross_satellite_correlation",
                                description=(
                                    f"Suspicious correlation in {param} across "
                                    f"{len(values)} satellites (relative std: {relative_std:.6f})"
                                ),
                                details={
                                    "parameter": param,
                                    "satellite_count": len(values),
                                    "relative_std": relative_std,
                                    "satellites": [sat for sat, _ in values],
                                },
                                discovered_at=epoch,
                            )
                        )
    
    return findings


def detect_transmission_time_anomalies(
    records: Iterable[EpochRecord],
    satellite: str,
    max_age_hours: float = 4.0,
) -> List[Finding]:
    """Detect stale or inconsistent TransTime (Transmission Time) values."""
    findings: List[Finding] = []
    
    max_age_seconds = max_age_hours * 3600.0
    prev_transtime: Optional[float] = None
    
    for record in records:
        transtime_value = _parse_transtime(record.values.get("TransTime"))
        if transtime_value is None:
            continue
        
        # Check if TransTime is too old relative to capture time
        # TransTime is in seconds-of-week (SOW), not absolute GPS seconds
        # We need to convert it using GPSWeek: absolute_gps_seconds = (GPSWeek * 604800) + TransTime
        gps_week = _to_int(record.values.get("GPSWeek"))
        if gps_week is None:
            # If GPSWeek is missing, we can't calculate absolute time, skip this check
            continue
        
        # Convert TransTime from seconds-of-week to absolute GPS seconds
        seconds_per_week = 604800  # 7 * 24 * 3600
        transtime_absolute_gps_seconds = (gps_week * seconds_per_week) + transtime_value
        
        # Calculate capture time in absolute GPS seconds
        gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
        capture_gps_seconds = (record.epoch - gps_epoch).total_seconds()
        
        # Calculate age (how old the transmission time is)
        transtime_age_seconds = abs(capture_gps_seconds - transtime_absolute_gps_seconds)
        
        if transtime_age_seconds > max_age_seconds:
            findings.append(
                Finding(
                    satellite=satellite,
                    epoch=record.epoch,
                    code="transmission_time_anomaly",
                    description=(
                        f"TransTime age {transtime_age_seconds/3600:.2f} hours exceeds "
                        f"maximum {max_age_hours:.1f} hours"
                    ),
                    details={
                        "transtime_value": transtime_value,
                        "transtime_absolute_gps_seconds": transtime_absolute_gps_seconds,
                        "gps_week": gps_week,
                        "transtime_age_hours": transtime_age_seconds / 3600.0,
                        "max_age_hours": max_age_hours,
                        "capture_gps_seconds": capture_gps_seconds,
                    },
                    discovered_at=record.epoch,
                )
            )
        
        # Check for regression
        if prev_transtime is not None:
            if transtime_value < prev_transtime - 3600:  # More than 1 hour backward
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="transtime_regression",
                        description=(
                            f"TransTime regressed from {prev_transtime:.1f} to "
                            f"{transtime_value:.1f}"
                        ),
                        details={
                            "previous_transtime": prev_transtime,
                            "current_transtime": transtime_value,
                            "regression_seconds": prev_transtime - transtime_value,
                        },
                        discovered_at=record.epoch,
                    )
                )
        
        prev_transtime = transtime_value
    
    return findings


def detect_physics_violations(
    records: Iterable[EpochRecord],
    satellite: str,
    tolerance: float = 1e-6,
) -> List[Finding]:
    """Detect violations of physical constraints in orbital parameters."""
    findings: List[Finding] = []
    
    for record in records:
        # Check eccentricity bounds (should be 0 <= e < 1)
        eccentricity = _to_float(record.values.get("Eccentricity"))
        if eccentricity is not None:
            if eccentricity < 0 or eccentricity >= 1:
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="eccentricity_violation",
                        description=(
                            f"Eccentricity {eccentricity:.6e} outside valid range [0, 1)"
                        ),
                        details={
                            "eccentricity": eccentricity,
                            "valid_range": "[0, 1)",
                        },
                        discovered_at=record.epoch,
                    )
                )
        
        # Check sqrtA (should be positive)
        sqrtA = _to_float(record.values.get("sqrtA"))
        if sqrtA is not None:
            if sqrtA <= 0:
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="sqrtA_violation",
                        description=f"sqrtA {sqrtA:.6e} is not positive",
                        details={"sqrtA": sqrtA},
                        discovered_at=record.epoch,
                    )
                )
        
        # Check inclination bounds (typically 0 < Io < π)
        Io = _to_float(record.values.get("Io"))
        if Io is not None:
            if Io < 0 or Io > 3.141592653589793 * 2:  # 2π
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="inclination_violation",
                        description=(
                            f"Inclination {Io:.6e} outside reasonable range "
                            f"[0, 2π]"
                        ),
                        details={"Io": Io, "valid_range": "[0, 2π]"},
                        discovered_at=record.epoch,
                    )
                )
        
        # Check IODE and IODC consistency (should match in most cases)
        iode = record.iode
        iodc = record.iodc
        if iode is not None and iodc is not None:
            # IODE and IODC should match (IODC has more bits, but lower 8 bits should match IODE)
            if (iodc & 0xFF) != (iode & 0xFF):
                findings.append(
                    Finding(
                        satellite=satellite,
                        epoch=record.epoch,
                        code="iode_iodc_inconsistency",
                        description=(
                            f"IODE {iode} and IODC {iodc} lower 8 bits don't match "
                            f"({iodc & 0xFF} vs {iode & 0xFF})"
                        ),
                        details={
                            "IODE": iode,
                            "IODC": iodc,
                            "IODE_lower_8": iode & 0xFF,
                            "IODC_lower_8": iodc & 0xFF,
                        },
                        discovered_at=record.epoch,
                    )
                )
    
    return findings


def _calculate_parameter_velocity(
    prev_record: EpochRecord,
    current_record: EpochRecord,
    parameter: str,
) -> Optional[float]:
    """Calculate the rate of change (velocity) of a parameter between two records."""
    prev_value = prev_record.values.get(parameter)
    curr_value = current_record.values.get(parameter)
    
    if prev_value is None or curr_value is None:
        return None
    
    try:
        prev_float = float(prev_value)
        curr_float = float(curr_value)
        time_delta = (current_record.epoch - prev_record.epoch).total_seconds()
        
        if time_delta <= 0:
            return None
        
        velocity = (curr_float - prev_float) / time_delta
        return velocity
    except (TypeError, ValueError):
        return None


def _calculate_statistics(
    values: List[float],
) -> Dict[str, float]:
    """Calculate mean and standard deviation of a list of values."""
    if not values or len(values) < 2:
        return {"mean": 0.0, "std": 0.0}
    
    try:
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        return {"mean": mean, "std": std}
    except (statistics.StatisticsError, ValueError):
        return {"mean": 0.0, "std": 0.0}


def _parse_toe(value: Any) -> Optional[float]:
    """Extract and parse Toe (Time of Ephemeris) value, returning seconds since GPS epoch start."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_transtime(value: Any) -> Optional[float]:
    """Extract and parse TransTime (Transmission Time) value, returning seconds since GPS epoch start."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_by_time_window(
    records: List[EpochRecord],
    window_size: timedelta,
) -> Dict[datetime, List[EpochRecord]]:
    """Group records by time windows of specified size."""
    if not records:
        return {}
    
    groups: Dict[datetime, List[EpochRecord]] = {}
    
    # Sort records by epoch
    sorted_records = sorted(records, key=lambda r: r.epoch)
    
    for record in sorted_records:
        # Calculate window start (truncate to window_size boundary)
        epoch_seconds = record.epoch.timestamp()
        window_seconds = window_size.total_seconds()
        window_start_seconds = int(epoch_seconds / window_seconds) * window_seconds
        window_start = datetime.fromtimestamp(window_start_seconds)
        
        groups.setdefault(window_start, []).append(record)
    
    return groups


def _extract_sequence_hash(
    records: List[EpochRecord],
    parameter: str,
    sequence_length: int,
) -> Optional[str]:
    """Extract a hash of parameter values from a sequence of records."""
    if len(records) < sequence_length:
        return None
    
    sequence_values = []
    for record in records[:sequence_length]:
        value = record.values.get(parameter)
        if value is None:
            return None
        try:
            sequence_values.append(str(float(value)))
        except (TypeError, ValueError):
            return None
    
    # Create hash from sequence
    sequence_str = ",".join(sequence_values)
    return hashlib.md5(sequence_str.encode()).hexdigest()


def _to_float(value: Any) -> Optional[float]:
    """Safely convert navigation value to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

