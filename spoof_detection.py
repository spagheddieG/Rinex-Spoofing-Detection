"""Command line interface for GNSS navigation spoofing detection."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List

from spoof_utils import (
    EpochRecord,
    Finding,
    detect_parameter_change_without_iode_change,
    detect_stale_data,
    detect_unexpected_iod_changes,
    detect_redundancy_inconsistencies,
    extract_satellite_timeseries,
    extract_satellite_timeseries_multisource,
    load_nav_json,
)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Define CLI arguments for the spoofing detector."""
    parser = argparse.ArgumentParser(
        description="Analyze RINEX navigation JSON for spoofing indicators.",
    )
    parser.add_argument(
        "json_path",
        type=Path,
        help="Navigation JSON file created with rinex_to_json.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write findings as JSON.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="Numeric tolerance when comparing parameter changes (default: exact).",
    )
    parser.add_argument(
        "--max-interval-hours",
        type=float,
        default=2.0,
        help=(
            "Maximum expected interval between updates before flagging stale data (default: 2 hours)."
        ),
    )
    parser.add_argument(
        "--ignore-satellites",
        nargs="+",
        help="Satellite identifiers to skip (e.g. G01 G05).",
    )
    return parser.parse_args(argv)


def run_detection(args: argparse.Namespace) -> List[Finding]:
    """Run the configured spoofing checks and return findings."""
    data = load_nav_json(args.json_path)

    # Extract timeseries, handling both single-source and multi-source files
    timeseries = extract_satellite_timeseries_multisource(data)

    findings: List[Finding] = []
    ignore_set = {sat.upper() for sat in (args.ignore_satellites or [])}

    max_interval = timedelta(hours=args.max_interval_hours)

    for satellite, records in sorted(timeseries.items()):
        if satellite.upper() in ignore_set:
            continue

        findings.extend(
            detect_parameter_change_without_iode_change(
                records, satellite=satellite, tolerance=args.tolerance
            )
        )
        findings.extend(
            detect_stale_data(
                records,
                satellite=satellite,
                max_interval=max_interval,
                by_time={},  # Not used in multi-source mode
            )
        )
        findings.extend(detect_unexpected_iod_changes(records, satellite=satellite))
        findings.extend(
            detect_redundancy_inconsistencies(
                records, satellite=satellite, tolerance=args.tolerance
            )
        )

    return findings


def print_summary(findings: List[Finding]) -> None:
    """Print a human-readable summary of all findings."""
    if not findings:
        print("No spoofing indicators detected.")
        return

    print(f"Detected {len(findings)} potential spoofing indicators:")
    for finding in findings:
        epoch_str = finding.epoch.isoformat(sep=" ")
        discovered_str = finding.discovered_at.isoformat(sep=" ")
        print(
            f"- [{finding.code}] {finding.satellite} @ {epoch_str} "
            f"(discovered at {discovered_str}): {finding.description}"
        )


def write_output(path: Path, findings: List[Finding]) -> None:
    """Write findings to disk as JSON."""
    serialized = [asdict(finding) for finding in findings]
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with path.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=False, default=str)
    print(f"Wrote findings to {path}")


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entrypoint; parse arguments, run checks, and report results."""
    args = parse_args(argv)
    findings = run_detection(args)
    print_summary(findings)
    if args.output:
        write_output(args.output, findings)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


