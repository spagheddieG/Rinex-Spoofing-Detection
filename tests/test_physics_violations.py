"""Tests for physics violations detection."""

from datetime import timedelta

import pytest

from spoof_utils import EpochRecord, detect_physics_violations
from tests.conftest import base_epoch, sample_epoch_record


def test_no_findings_for_valid_parameters(sample_epoch_record):
    """Test that valid parameters don't trigger findings."""
    findings = detect_physics_violations([sample_epoch_record], satellite="G01")
    assert len(findings) == 0


def test_finds_eccentricity_violation(base_epoch):
    """Test that eccentricity violations are detected."""
    records = [
        EpochRecord(epoch=base_epoch, values={"Eccentricity": 1.5}, source="source"),  # Invalid (> 1)
    ]
    findings = detect_physics_violations(records, satellite="G01")
    assert len(findings) > 0
    assert any(f.code == "eccentricity_violation" for f in findings)


def test_finds_sqrtA_violation(base_epoch):
    """Test that sqrtA violations are detected."""
    records = [
        EpochRecord(epoch=base_epoch, values={"sqrtA": -1.0}, source="source"),  # Invalid (negative)
    ]
    findings = detect_physics_violations(records, satellite="G01")
    assert len(findings) > 0
    assert any(f.code == "sqrtA_violation" for f in findings)


def test_finds_inclination_violation(base_epoch):
    """Test that inclination violations are detected."""
    records = [
        EpochRecord(epoch=base_epoch, values={"Io": 10.0}, source="source"),  # Invalid (> 2π)
    ]
    findings = detect_physics_violations(records, satellite="G01")
    assert len(findings) > 0
    assert any(f.code == "inclination_violation" for f in findings)


def test_finds_iode_iodc_inconsistency(base_epoch):
    """Test that IODE/IODC inconsistencies are detected."""
    records = [
        EpochRecord(epoch=base_epoch, values={"IODE": 10.0, "IODC": 266.0}, source="source"),  # Lower 8 bits don't match
    ]
    findings = detect_physics_violations(records, satellite="G01")
    assert len(findings) > 0
    assert any(f.code == "iode_iodc_inconsistency" for f in findings)


def test_empty_records():
    """Test that empty record list returns no findings."""
    findings = detect_physics_violations([], satellite="G01")
    assert len(findings) == 0


def test_finding_structure(base_epoch):
    """Test that findings have correct structure."""
    records = [
        EpochRecord(epoch=base_epoch, values={"Eccentricity": 1.5}, source="source"),
    ]
    findings = detect_physics_violations(records, satellite="G01")
    if findings:
        finding = findings[0]
        assert finding.code in (
            "eccentricity_violation",
            "sqrtA_violation",
            "inclination_violation",
            "iode_iodc_inconsistency",
        )
        assert finding.satellite == "G01"
