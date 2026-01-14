#!/usr/bin/env python3
"""
Example demonstrating cross-satellite correlations detection.

This script shows how to detect suspicious correlations between satellites,
which may indicate coordinated spoofing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from spoof_utils import detect_cross_satellite_correlations, extract_satellite_timeseries_multisource, load_nav_json


def main() -> None:
    """Run cross-satellite correlations detection example."""
    json_file = Path("combined_highrate.json")
    
    if not json_file.exists():
        print(f"Error: {json_file} not found.")
        print("Please ensure combined_highrate.json exists in the current directory.")
        return
    
    print("Cross-Satellite Correlations Detection Example")
    print("=" * 60)
    print(f"Loading data from {json_file}...")
    
    data = load_nav_json(json_file)
    timeseries = extract_satellite_timeseries_multisource(data)
    
    print(f"Analyzing correlations across {len(timeseries)} satellites...")
    print()
    
    findings = detect_cross_satellite_correlations(timeseries, min_correlation=0.9)
    
    if not findings:
        print("No cross-satellite correlations detected.")
    else:
        print(f"Found {len(findings)} cross-satellite correlation findings:")
        for finding in findings[:5]:  # Show first 5
            print(f"  - {finding.code} @ {finding.epoch.isoformat()}")
            print(f"    {finding.description}")
            print(f"    Satellites: {finding.details.get('satellites', [])}")
        
        print()
        print(f"Total findings: {len(findings)}")
        
        # Save findings
        output_file = Path("findings_cross_satellite_correlations.json")
        serialized = [f.__dict__ for f in findings]
        with output_file.open("w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False, default=str)
        print(f"Findings saved to {output_file}")


if __name__ == "__main__":
    main()
