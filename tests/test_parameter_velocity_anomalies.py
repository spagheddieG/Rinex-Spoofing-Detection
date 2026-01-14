"""Tests for parameter velocity anomalies detection."""

from datetime import timedelta

import pytest

from spoof_utils import EpochRecord, detect_parameter_velocity_anomalies
from tests.conftest import base_epoch, sample_records_sequence


def test_no_findings_for_normal_velocity(sample_records_sequence):
    """Test that normal velocity changes don't trigger findings."""
    findings = detect_parameter_velocity_anomalies(
        sample_records_sequence, satellite="G01", parameter="SVclockBias", max_acceleration=1e-6
    )
    assert len(findings) == 0


def test_finds_acceleration_anomalies(base_epoch):
    """Test that acceleration anomalies are detected."""
    records = []
    base_value = -0.0001
    
    # First record
    records.append(EpochRecord(epoch=base_epoch, values={"SVclockBias": base_value}, source="source"))
    
    # Second record - small change
    records.append(
        EpochRecord(
            epoch=base_epoch + timedelta(minutes=15),
            values={"SVclockBias": base_value + 1e-8},
            source="source",
        )
    )
    
    # Third record - huge acceleration
    records.append(
        EpochRecord(
            epoch=base_epoch + timedelta(minutes=30),
            values={"SVclockBias": base_value + 1.0},  # Huge jump
            source="source",
        )
    )
    
    findings = detect_parameter_velocity_anomalies(records, satellite="G01", parameter="SVclockBias", max_acceleration=1e-12)
    assert len(findings) > 0
    assert all(f.code == "parameter_acceleration_anomaly" for f in findings)


def test_empty_records():
    """Test that empty record list returns no findings."""
    findings = detect_parameter_velocity_anomalies([], satellite="G01")
    assert len(findings) == 0


def test_single_record(base_epoch):
    """Test that single record returns no findings."""
    record = EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source")
    findings = detect_parameter_velocity_anomalies([record], satellite="G01")
    assert len(findings) == 0


def test_finding_structure(base_epoch):
    """Test that findings have correct structure."""
    records = []
    base_value = -0.0001
    records.append(EpochRecord(epoch=base_epoch, values={"SVclockBias": base_value}, source="source"))
    records.append(
        EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": base_value + 1e-8}, source="source")
    )
    records.append(
        EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": base_value + 1.0}, source="source")
    )
    
    findings = detect_parameter_velocity_anomalies(records, satellite="G01", parameter="SVclockBias", max_acceleration=1e-12)
    if findings:
        finding = findings[0]
        assert finding.code == "parameter_acceleration_anomaly"
        assert finding.satellite == "G01"
        assert "parameter" in finding.details
        assert "acceleration" in finding.details
        assert "max_acceleration" in finding.details
