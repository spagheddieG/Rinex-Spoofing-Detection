#!/usr/bin/env python3
"""
Example demonstrating replay patterns detection.

This script shows how to detect repeated sequences of parameter values,
which may indicate replay attacks or data manipulation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spoof_utils import detect_replay_patterns, extract_satellite_timeseries_multisource, load_nav_json


def main() -> None:
    """Run replay patterns detection example."""
    json_file = Path("combined_highrate.json")
    
    if not json_file.exists():
        print(f"Error: {json_file} not found.")
        print("Please ensure combined_highrate.json exists in the current directory.")
        return
    
    print("Replay Patterns Detection Example")
    print("=" * 60)
    print(f"Loading data from {json_file}...")
    
    data = load_nav_json(json_file)
    timeseries = extract_satellite_timeseries_multisource(data)
    
    print(f"Analyzing {len(timeseries)} satellites for repeated sequences...")
    print("Sequence length: 4 (1 hour at 15-minute intervals)")
    print()
    
    all_findings: List = []
    
    for satellite, records in sorted(timeseries.items()):
        if len(records) < 8:  # Need at least 8 records for 2 sequences
            continue
        
        findings = detect_replay_patterns(records, satellite=satellite, sequence_length=4)
        
        if findings:
            all_findings.extend(findings)
            print(f"Satellite {satellite}: Found {len(findings)} replay patterns")
            for finding in findings[:3]:  # Show first 3
                print(f"  - {finding.code} @ {finding.epoch.isoformat()}")
                print(f"    Parameter: {finding.details.get('parameter')}")
                print(f"    Occurrences: {finding.details.get('occurrence_count')}")
    
    if not all_findings:
        print("No replay patterns detected.")
    else:
        print()
        print(f"Total findings: {len(all_findings)}")
        
        # Save findings
        output_file = Path("findings_replay_patterns.json")
        serialized = [f.__dict__ for f in all_findings]
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False, default=str)
        print(f"Findings saved to {output_file}")


if __name__ == "__main__":
    main()
