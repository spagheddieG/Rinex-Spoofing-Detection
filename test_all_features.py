#!/usr/bin/env python3
"""
Comprehensive test script for all new spoofing detection features.
This script tests all functionality and generates a test report.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

from spoof_utils import (
    EpochRecord,
    detect_cross_satellite_correlations,
    detect_ephemeris_age_anomalies,
    detect_parameter_velocity_anomalies,
    detect_physics_violations,
    detect_replay_patterns,
    detect_temporal_source_inconsistencies,
    detect_transmission_time_anomalies,
)


class TestReport:
    """Test report generator."""
    
    def __init__(self):
        self.results: List[Dict] = []
        self.passed = 0
        self.failed = 0
    
    def add_test(self, name: str, passed: bool, details: str = ""):
        """Add a test result."""
        self.results.append({
            "name": name,
            "passed": passed,
            "details": details,
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1
    
    def generate_report(self) -> str:
        """Generate test report."""
        report = []
        report.append("=" * 80)
        report.append("SPOOFING DETECTION FEATURES - TEST REPORT")
        report.append("=" * 80)
        report.append("")
        report.append(f"Total Tests: {len(self.results)}")
        report.append(f"Passed: {self.passed}")
        report.append(f"Failed: {self.failed}")
        report.append("")
        report.append("-" * 80)
        
        for result in self.results:
            status = "PASS" if result["passed"] else "FAIL"
            report.append(f"[{status}] {result['name']}")
            if result["details"]:
                report.append(f"      {result['details']}")
        
        report.append("")
        report.append("=" * 80)
        return "\n".join(report)


def create_test_records() -> Dict[str, List[EpochRecord]]:
    """Create test records for various scenarios."""
    base_epoch = datetime(2025, 11, 10, 0, 0, 0)
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    
    records = {}
    
    # Normal sequence
    normal_records = []
    for i in range(10):
        epoch = base_epoch + timedelta(minutes=15 * i)
        gps_seconds = (epoch - gps_epoch).total_seconds()
        normal_records.append(EpochRecord(
            epoch=epoch,
            values={
                "SVclockBias": -0.0001 + i * 1e-8,
                "IODE": 10.0,
                "IODC": 10.0,
                "Toe": gps_seconds,
                "TransTime": gps_seconds,
                "Eccentricity": 0.01,
                "sqrtA": 5153.0,
                "Io": 0.9,
            },
            source=f"source_{i}",
        ))
    records["normal"] = normal_records
    
    # Rapid change
    rapid_records = [
        EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source1"),
        EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": 100.0}, source="source2"),
    ]
    records["rapid_change"] = rapid_records
    
    # Stale data
    stale_records = []
    base_values = {"SVclockBias": -0.0001, "IODE": 10.0}
    for i in range(5):
        epoch = base_epoch + timedelta(minutes=15 * i)
        stale_records.append(EpochRecord(epoch=epoch, values=base_values.copy(), source="source"))
    records["stale"] = stale_records
    
    # Replay pattern
    replay_records = []
    patterns = [
        {"SVclockBias": -0.0001, "IODE": 10.0},
        {"SVclockBias": -0.0002, "IODE": 11.0},
        {"SVclockBias": -0.0003, "IODE": 12.0},
        {"SVclockBias": -0.0004, "IODE": 13.0},
    ]
    for repeat in range(2):
        for i, pattern in enumerate(patterns):
            epoch = base_epoch + timedelta(minutes=15 * (repeat * 4 + i))
            replay_records.append(EpochRecord(epoch=epoch, values=pattern.copy(), source=f"source_{repeat}"))
    records["replay"] = replay_records
    
    # Outlier
    outlier_records = []
    base_value = -0.0001
    for i in range(9):
        epoch = base_epoch + timedelta(minutes=15 * i)
        outlier_records.append(EpochRecord(epoch=epoch, values={"SVclockBias": base_value + i * 1e-8}, source="source"))
    outlier_records.append(EpochRecord(epoch=base_epoch + timedelta(minutes=15 * 9), values={"SVclockBias": base_value + 1.0}, source="source"))
    records["outlier"] = outlier_records
    
    # Physics violation
    physics_violation_records = [
        EpochRecord(epoch=base_epoch, values={"Eccentricity": 1.5, "sqrtA": -1.0, "Io": 10.0, "IODE": 10.0, "IODC": 266.0}, source="source"),
    ]
    records["physics_violation"] = physics_violation_records
    
    # Old ephemeris
    old_ephemeris_records = []
    epoch = base_epoch
    gps_seconds = (epoch - gps_epoch).total_seconds()
    old_toe = gps_seconds - 10 * 3600  # 10 hours ago
    old_ephemeris_records.append(EpochRecord(epoch=epoch, values={"Toe": old_toe}, source="source"))
    records["old_ephemeris"] = old_ephemeris_records
    
    return records


def test_ephemeris_age_anomalies(report: TestReport, test_records: Dict[str, List[EpochRecord]]):
    """Test ephemeris age anomalies detection."""
    print("Testing ephemeris age anomalies detection...")
    
    # Test normal ephemeris
    findings = detect_ephemeris_age_anomalies(test_records["normal"], satellite="G01", max_age_hours=4.0)
    report.add_test(
        "Ephemeris Age - Normal ephemeris",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test old ephemeris
    findings = detect_ephemeris_age_anomalies(test_records["old_ephemeris"], satellite="G01", max_age_hours=4.0)
    report.add_test(
        "Ephemeris Age - Old ephemeris detected",
        len(findings) > 0,
        f"Found {len(findings)} findings (expected > 0)"
    )
    
    # Test empty records
    findings = detect_ephemeris_age_anomalies([], satellite="G01")
    report.add_test(
        "Ephemeris Age - Empty records",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )


def test_replay_patterns(report: TestReport, test_records: Dict[str, List[EpochRecord]]):
    """Test replay patterns detection."""
    print("Testing replay patterns detection...")
    
    # Test normal sequence
    findings = detect_replay_patterns(test_records["normal"], satellite="G01", sequence_length=4)
    report.add_test(
        "Replay Patterns - Normal sequence",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test replay pattern
    findings = detect_replay_patterns(test_records["replay"], satellite="G01", sequence_length=4)
    report.add_test(
        "Replay Patterns - Replay pattern detected",
        len(findings) > 0,
        f"Found {len(findings)} findings (expected > 0)"
    )
    
    # Test short sequence
    short_records = test_records["normal"][:2]
    findings = detect_replay_patterns(short_records, satellite="G01", sequence_length=4)
    report.add_test(
        "Replay Patterns - Short sequence",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )


def test_temporal_source_inconsistencies(report: TestReport, test_records: Dict[str, List[EpochRecord]]):
    """Test temporal source inconsistencies detection."""
    print("Testing temporal source inconsistencies detection...")
    
    # Test consistent sources
    consistent_records = []
    base_epoch = datetime(2025, 11, 10, 0, 0, 0)
    for i in range(3):
        epoch = base_epoch + timedelta(minutes=15 * i)
        values = {"SVclockBias": -0.0001, "IODE": 10.0}
        consistent_records.append(EpochRecord(epoch=epoch, values=values, source="source1"))
        consistent_records.append(EpochRecord(epoch=epoch, values=values, source="source2"))
    
    findings = detect_temporal_source_inconsistencies(consistent_records, satellite="G01", time_window=timedelta(minutes=30))
    report.add_test(
        "Temporal Source Inconsistencies - Consistent sources",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test inconsistent sources
    inconsistent_records = []
    for i in range(3):
        epoch = base_epoch + timedelta(minutes=15 * i)
        inconsistent_records.append(EpochRecord(epoch=epoch, values={"SVclockBias": -0.0001, "IODE": 10.0}, source="source1"))
        inconsistent_records.append(EpochRecord(epoch=epoch, values={"SVclockBias": -0.0002, "IODE": 11.0}, source="source2"))
    
    findings = detect_temporal_source_inconsistencies(inconsistent_records, satellite="G01", time_window=timedelta(minutes=30), tolerance=1e-10)
    report.add_test(
        "Temporal Source Inconsistencies - Inconsistent sources detected",
        len(findings) > 0,
        f"Found {len(findings)} findings (expected > 0)"
    )


def test_parameter_velocity_anomalies(report: TestReport, test_records: Dict[str, List[EpochRecord]]):
    """Test parameter velocity anomalies detection."""
    print("Testing parameter velocity anomalies detection...")
    
    # Test normal velocity
    findings = detect_parameter_velocity_anomalies(test_records["normal"], satellite="G01", parameter="SVclockBias", max_acceleration=1e-6)
    report.add_test(
        "Parameter Velocity Anomalies - Normal velocity",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test acceleration anomaly
    acceleration_records = [
        EpochRecord(epoch=datetime(2025, 11, 10, 0, 0, 0), values={"SVclockBias": -0.0001}, source="source"),
        EpochRecord(epoch=datetime(2025, 11, 10, 0, 15, 0), values={"SVclockBias": -0.0001 + 1e-8}, source="source"),
        EpochRecord(epoch=datetime(2025, 11, 10, 0, 30, 0), values={"SVclockBias": -0.0001 + 1.0}, source="source"),
    ]
    findings = detect_parameter_velocity_anomalies(acceleration_records, satellite="G01", parameter="SVclockBias", max_acceleration=1e-12)
    report.add_test(
        "Parameter Velocity Anomalies - Acceleration anomaly detected",
        len(findings) > 0,
        f"Found {len(findings)} findings (expected > 0)"
    )


def test_cross_satellite_correlations(report: TestReport):
    """Test cross-satellite correlations detection."""
    print("Testing cross-satellite correlations detection...")
    
    # Test normal variation
    base_epoch = datetime(2025, 11, 10, 0, 0, 0)
    timeseries = {
        "G01": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0002}, source="source"),
        ],
        "G02": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0003}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0004}, source="source"),
        ],
    }
    findings = detect_cross_satellite_correlations(timeseries, min_correlation=0.9)
    report.add_test(
        "Cross-Satellite Correlations - Normal variation",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test empty timeseries
    findings = detect_cross_satellite_correlations({})
    report.add_test(
        "Cross-Satellite Correlations - Empty timeseries",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )


def test_transmission_time_anomalies(report: TestReport, test_records: Dict[str, List[EpochRecord]]):
    """Test transmission time anomalies detection."""
    print("Testing transmission time anomalies detection...")
    
    # Test normal transmission time
    findings = detect_transmission_time_anomalies(test_records["normal"], satellite="G01", max_age_hours=4.0)
    report.add_test(
        "Transmission Time Anomalies - Normal transmission time",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test empty records
    findings = detect_transmission_time_anomalies([], satellite="G01")
    report.add_test(
        "Transmission Time Anomalies - Empty records",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )


def test_physics_violations(report: TestReport, test_records: Dict[str, List[EpochRecord]]):
    """Test physics violations detection."""
    print("Testing physics violations detection...")
    
    # Test valid parameters
    findings = detect_physics_violations(test_records["normal"], satellite="G01")
    report.add_test(
        "Physics Violations - Valid parameters",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )
    
    # Test physics violations
    findings = detect_physics_violations(test_records["physics_violation"], satellite="G01")
    report.add_test(
        "Physics Violations - Violations detected",
        len(findings) > 0,
        f"Found {len(findings)} findings (expected > 0)"
    )
    
    # Test empty records
    findings = detect_physics_violations([], satellite="G01")
    report.add_test(
        "Physics Violations - Empty records",
        len(findings) == 0,
        f"Found {len(findings)} findings (expected 0)"
    )


def test_integration(report: TestReport):
    """Test integration with spoof_detection.py."""
    print("Testing integration...")
    
    try:
        from spoof_detection import parse_args, run_detection
        
        # Test parse_args
        args = parse_args(["test.json"])
        report.add_test(
            "Integration - Parse args",
            args.json_path == Path("test.json"),
            f"Args parsed correctly: {args.json_path}"
        )
        
        # Test that new CLI arguments exist
        has_new_args = (
            hasattr(args, "max_ephemeris_age_hours") and
            hasattr(args, "replay_sequence_length") and
            hasattr(args, "enable_cross_satellite_checks")
        )
        report.add_test(
            "Integration - New CLI arguments",
            has_new_args,
            f"New arguments present: {has_new_args}"
        )
        
    except Exception as e:
        report.add_test(
            "Integration - Import and basic functionality",
            False,
            f"Error: {str(e)}"
        )


def main():
    """Run all tests and generate report."""
    print("Starting comprehensive test suite...")
    print()
    
    report = TestReport()
    test_records = create_test_records()
    
    # Run all test suites
    test_ephemeris_age_anomalies(report, test_records)
    test_replay_patterns(report, test_records)
    test_temporal_source_inconsistencies(report, test_records)
    test_parameter_velocity_anomalies(report, test_records)
    test_cross_satellite_correlations(report)
    test_transmission_time_anomalies(report, test_records)
    test_physics_violations(report, test_records)
    test_integration(report)
    
    # Generate and save report
    report_text = report.generate_report()
    print()
    print(report_text)
    
    # Save to file
    report_file = Path("TEST_REPORT.txt")
    report_file.write_text(report_text)
    print(f"\nTest report saved to {report_file}")
    
    # Return exit code
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    exit(main())
