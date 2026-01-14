"""Tests for temporal source inconsistencies detection."""

from datetime import timedelta

import pytest

from spoof_utils import EpochRecord, detect_temporal_source_inconsistencies
from tests.conftest import base_epoch, sample_records_sequence


def test_no_findings_for_consistent_sources(base_epoch):
    """Test that consistent sources don't trigger findings."""
    records = []
    for i in range(3):
        epoch = base_epoch + timedelta(minutes=15 * i)
        values = {"SVclockBias": -0.0001, "IODE": 10.0, "IODC": 10.0}
        records.append(EpochRecord(epoch=epoch, values=values, source="source1"))
        records.append(EpochRecord(epoch=epoch, values=values, source="source2"))
    
    findings = detect_temporal_source_inconsistencies(records, satellite="G01")
    assert len(findings) == 0


def test_finds_inconsistent_sources(base_epoch):
    """Test that inconsistent sources are detected."""
    records = []
    for i in range(3):
        epoch = base_epoch + timedelta(minutes=15 * i)
        values1 = {"SVclockBias": -0.0001, "IODE": 10.0, "IODC": 10.0}
        values2 = {"SVclockBias": -0.0002, "IODE": 11.0, "IODC": 11.0}  # Different values
        records.append(EpochRecord(epoch=epoch, values=values1, source="source1"))
        records.append(EpochRecord(epoch=epoch, values=values2, source="source2"))
    
    findings = detect_temporal_source_inconsistencies(records, satellite="G01", tolerance=1e-10)
    assert len(findings) > 0
    assert all(f.code == "temporal_source_inconsistency" for f in findings)


def test_empty_records():
    """Test that empty record list returns no findings."""
    findings = detect_temporal_source_inconsistencies([], satellite="G01")
    assert len(findings) == 0


def test_single_record(base_epoch):
    """Test that single record returns no findings."""
    record = EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source")
    findings = detect_temporal_source_inconsistencies([record], satellite="G01")
    assert len(findings) == 0


def test_finding_structure(base_epoch):
    """Test that findings have correct structure."""
    records = []
    for i in range(2):
        epoch = base_epoch + timedelta(minutes=15 * i)
        values1 = {"SVclockBias": -0.0001, "IODE": 10.0}
        values2 = {"SVclockBias": -0.0002, "IODE": 11.0}
        records.append(EpochRecord(epoch=epoch, values=values1, source="source1"))
        records.append(EpochRecord(epoch=epoch, values=values2, source="source2"))
    
    findings = detect_temporal_source_inconsistencies(records, satellite="G01", tolerance=1e-10)
    if findings:
        finding = findings[0]
        assert finding.code == "temporal_source_inconsistency"
        assert finding.satellite == "G01"
        assert "inconsistent_parameters" in finding.details
        assert "reference_source" in finding.details
        assert "comparison_source" in finding.details
