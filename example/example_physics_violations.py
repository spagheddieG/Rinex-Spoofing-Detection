#!/usr/bin/env python3
"""
Example demonstrating physics violations detection.

This script shows how to detect violations of physical constraints in orbital
parameters, which may indicate spoofing or data corruption.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spoof_utils import detect_physics_violations, extract_satellite_timeseries_multisource, load_nav_json


def main() -> None:
    """Run physics violations detection example."""
    json_file = Path("combined_highrate.json")
    
    if not json_file.exists():
        print(f"Error: {json_file} not found.")
        print("Please ensure combined_highrate.json exists in the current directory.")
        return
    
    print("Physics Violations Detection Example")
    print("=" * 60)
    print(f"Loading data from {json_file}...")
    
    data = load_nav_json(json_file)
    timeseries = extract_satellite_timeseries_multisource(data)
    
    print(f"Analyzing {len(timeseries)} satellites for physics violations...")
    print("Checking: eccentricity bounds, sqrtA positivity, inclination bounds, IODE/IODC consistency")
    print()
    
    all_findings: List = []
    
    for satellite, records in sorted(timeseries.items()):
        findings = detect_physics_violations(records, satellite=satellite)
        
        if findings:
            all_findings.extend(findings)
            print(f"Satellite {satellite}: Found {len(findings)} physics violations")
            for finding in findings[:3]:  # Show first 3
                print(f"  - {finding.code} @ {finding.epoch.isoformat()}")
                print(f"    {finding.description}")
    
    if not all_findings:
        print("No physics violations detected.")
    else:
        print()
        print(f"Total findings: {len(all_findings)}")
        
        # Save findings
        output_file = Path("findings_physics_violations.json")
        serialized = [f.__dict__ for f in all_findings]
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False, default=str)
        print(f"Findings saved to {output_file}")


if __name__ == "__main__":
    main()
