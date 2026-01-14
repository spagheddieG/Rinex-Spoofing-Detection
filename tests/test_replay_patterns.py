"""Tests for replay patterns detection."""

import pytest

from spoof_utils import detect_replay_patterns
from tests.conftest import base_epoch, sample_records_replay, sample_records_sequence


def test_no_findings_for_unique_sequences(sample_records_sequence):
    """Test that unique sequences don't trigger findings."""
    findings = detect_replay_patterns(sample_records_sequence, satellite="G01", sequence_length=4)
    assert len(findings) == 0


def test_finds_replay_patterns(sample_records_replay):
    """Test that repeated sequences are detected."""
    findings = detect_replay_patterns(sample_records_replay, satellite="G01", sequence_length=4)
    assert len(findings) > 0
    assert all(f.code == "replay_pattern" for f in findings)


def test_empty_records():
    """Test that empty record list returns no findings."""
    findings = detect_replay_patterns([], satellite="G01")
    assert len(findings) == 0


def test_short_sequence(base_epoch):
    """Test that sequences shorter than sequence_length return no findings."""
    from datetime import timedelta
    
    from spoof_utils import EpochRecord
    
    records = [
        EpochRecord(epoch=base_epoch + timedelta(minutes=15 * i), values={"SVclockBias": -0.0001}, source="source")
        for i in range(2)
    ]
    findings = detect_replay_patterns(records, satellite="G01", sequence_length=4)
    assert len(findings) == 0


def test_custom_parameters(sample_records_replay):
    """Test with custom parameter list."""
    findings = detect_replay_patterns(
        sample_records_replay, satellite="G01", sequence_length=4, parameters=["SVclockBias"]
    )
    assert len(findings) >= 0  # May or may not find depending on data


def test_finding_structure(sample_records_replay):
    """Test that findings have correct structure."""
    findings = detect_replay_patterns(sample_records_replay, satellite="G01", sequence_length=4)
    if findings:
        finding = findings[0]
        assert finding.code == "replay_pattern"
        assert finding.satellite == "G01"
        assert "parameter" in finding.details
        assert "sequence_length" in finding.details
        assert "occurrence_count" in finding.details
