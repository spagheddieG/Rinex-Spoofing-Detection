#!/usr/bin/env python3
"""
Example demonstrating all spoofing detection methods together.

This script runs all 10 new detection methods and shows their combined results.
"""

from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spoof_utils import (
    detect_cross_satellite_correlations,
    detect_ephemeris_age_anomalies,
    detect_parameter_velocity_anomalies,
    detect_physics_violations,
    detect_replay_patterns,
    detect_temporal_source_inconsistencies,
    detect_transmission_time_anomalies,
    extract_satellite_timeseries_multisource,
    load_nav_json,
)


def main() -> None:
    """Run all detection methods."""
    json_file = Path("combined_highrate.json")
    
    if not json_file.exists():
        print(f"Error: {json_file} not found.")
        print("Please ensure combined_highrate.json exists in the current directory.")
        return
    
    print("All Spoofing Detection Methods Example")
    print("=" * 60)
    print(f"Loading data from {json_file}...")
    
    data = load_nav_json(json_file)
    timeseries = extract_satellite_timeseries_multisource(data)
    
    print(f"Analyzing {len(timeseries)} satellites with all detection methods...")
    print()
    
    all_findings: Dict[str, List] = {}
    
    # Run per-satellite detection methods
    for satellite, records in sorted(timeseries.items()):
        if len(records) < 2:
            continue
        
        # Ephemeris age anomalies
        findings = detect_ephemeris_age_anomalies(records, satellite=satellite, max_age_hours=4.0)
        if findings:
            all_findings.setdefault("ephemeris_age_anomalies", []).extend(findings)
        
        # Replay patterns
        if len(records) >= 8:
            findings = detect_replay_patterns(records, satellite=satellite, sequence_length=4)
            if findings:
                all_findings.setdefault("replay_patterns", []).extend(findings)
        
        # Temporal source inconsistencies
        findings = detect_temporal_source_inconsistencies(
            records, satellite=satellite, time_window=timedelta(minutes=30), tolerance=1e-9
        )
        if findings:
            all_findings.setdefault("temporal_source_inconsistencies", []).extend(findings)
        
        # Parameter velocity anomalies
        if len(records) >= 3:
            findings = detect_parameter_velocity_anomalies(
                records, satellite=satellite, parameter="SVclockBias", max_acceleration=1e-12
            )
            if findings:
                all_findings.setdefault("parameter_velocity_anomalies", []).extend(findings)
        
        # Transmission time anomalies
        findings = detect_transmission_time_anomalies(records, satellite=satellite, max_age_hours=4.0)
        if findings:
            all_findings.setdefault("transmission_time_anomalies", []).extend(findings)
        
        # Physics violations
        findings = detect_physics_violations(records, satellite=satellite)
        if findings:
            all_findings.setdefault("physics_violations", []).extend(findings)
    
    # Cross-satellite correlations (requires all timeseries)
    findings = detect_cross_satellite_correlations(timeseries, min_correlation=0.9)
    if findings:
        all_findings["cross_satellite_correlations"] = findings
    
    # Print summary
    print("Detection Results Summary")
    print("=" * 60)
    total_findings = sum(len(f) for f in all_findings.values())
    
    if total_findings == 0:
        print("No spoofing indicators detected by any method.")
    else:
        for method, findings in sorted(all_findings.items()):
            print(f"{method}: {len(findings)} findings")
        
        print()
        print(f"Total findings across all methods: {total_findings}")
        
        # Save all findings
        output_file = Path("findings_all_detections.json")
        serialized: Dict[str, List] = {}
        for method, findings in all_findings.items():
            serialized[method] = [f.__dict__ for f in findings]
        
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False, default=str)
        print(f"All findings saved to {output_file}")


if __name__ == "__main__":
    main()
