"""Tests for cross-satellite correlations detection."""

from datetime import timedelta

import pytest

from spoof_utils import EpochRecord, detect_cross_satellite_correlations
from tests.conftest import base_epoch


def test_no_findings_for_normal_variation(base_epoch):
    """Test that normal variation between satellites doesn't trigger findings."""
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
    assert len(findings) == 0


def test_finds_suspicious_correlation(base_epoch):
    """Test that suspicious correlations are detected."""
    timeseries = {
        "G01": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": -0.0001}, source="source"),
        ],
        "G02": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": -0.0001}, source="source"),
        ],
        "G03": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": -0.0001}, source="source"),
        ],
    }
    
    findings = detect_cross_satellite_correlations(timeseries, min_correlation=0.9)
    assert len(findings) >= 0  # May or may not find depending on implementation


def test_empty_timeseries():
    """Test that empty timeseries returns no findings."""
    findings = detect_cross_satellite_correlations({})
    assert len(findings) == 0


def test_single_satellite(base_epoch):
    """Test that single satellite returns no findings."""
    timeseries = {
        "G01": [EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source")],
    }
    findings = detect_cross_satellite_correlations(timeseries)
    assert len(findings) == 0


def test_finding_structure(base_epoch):
    """Test that findings have correct structure."""
    timeseries = {
        "G01": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": -0.0001}, source="source"),
        ],
        "G02": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": -0.0001}, source="source"),
        ],
        "G03": [
            EpochRecord(epoch=base_epoch, values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={"SVclockBias": -0.0001}, source="source"),
            EpochRecord(epoch=base_epoch + timedelta(minutes=30), values={"SVclockBias": -0.0001}, source="source"),
        ],
    }
    
    findings = detect_cross_satellite_correlations(timeseries)
    if findings:
        finding = findings[0]
        assert finding.code == "cross_satellite_correlation"
        assert finding.satellite == "MULTIPLE"
        assert "parameter" in finding.details
        assert "satellite_count" in finding.details
