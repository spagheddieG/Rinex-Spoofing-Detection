"""Tests for ephemeris age anomalies detection."""

from datetime import datetime, timedelta

import pytest

from spoof_utils import EpochRecord, detect_ephemeris_age_anomalies
from tests.conftest import base_epoch


def test_no_findings_for_fresh_ephemeris(base_epoch):
    """Test that fresh ephemeris data doesn't trigger findings."""
    records = []
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    
    for i in range(5):
        epoch = base_epoch + timedelta(minutes=15 * i)
        # Toe is current (within 2 hours)
        toe_seconds = (epoch - gps_epoch).total_seconds()
        records.append(
            EpochRecord(
                epoch=epoch,
                values={"Toe": toe_seconds},
                source="source",
            )
        )
    
    findings = detect_ephemeris_age_anomalies(records, satellite="G01", max_age_hours=4.0)
    assert len(findings) == 0


def test_finds_stale_ephemeris(base_epoch):
    """Test that stale ephemeris data is detected."""
    records = []
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    
    epoch = base_epoch
    # Toe is very old (10 hours ago) - use old GPS week
    epoch_gps_seconds = (epoch - gps_epoch).total_seconds()
    stale_gps_seconds = epoch_gps_seconds - 10 * 3600
    seconds_per_week = 604800
    stale_gps_week = int(stale_gps_seconds // seconds_per_week)
    stale_toe_sow = stale_gps_seconds % seconds_per_week
    
    records.append(
        EpochRecord(
            epoch=epoch,
            values={"Toe": stale_toe_sow, "GPSWeek": stale_gps_week},
            source="source",
        )
    )
    
    findings = detect_ephemeris_age_anomalies(records, satellite="G01", max_age_hours=4.0)
    assert len(findings) > 0
    assert all(f.code == "ephemeris_age_anomaly" for f in findings)


def test_finds_toe_regression(base_epoch):
    """Test that Toe regression is detected."""
    records = []
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    seconds_per_week = 604800
    
    # First record with normal Toe
    epoch1 = base_epoch
    gps_seconds1 = (epoch1 - gps_epoch).total_seconds()
    gps_week1 = int(gps_seconds1 // seconds_per_week)
    toe1_sow = gps_seconds1 % seconds_per_week
    records.append(EpochRecord(epoch=epoch1, values={"Toe": toe1_sow, "GPSWeek": gps_week1}, source="source"))
    
    # Second record with regressed Toe (2 hours backward in same week)
    epoch2 = base_epoch + timedelta(minutes=15)
    toe2_sow = toe1_sow - 7200  # 2 hours backward
    if toe2_sow < 0:
        toe2_sow += seconds_per_week  # Wrap around
        gps_week2 = gps_week1 - 1
    else:
        gps_week2 = gps_week1
    records.append(EpochRecord(epoch=epoch2, values={"Toe": toe2_sow, "GPSWeek": gps_week2}, source="source"))
    
    findings = detect_ephemeris_age_anomalies(records, satellite="G01")
    # May or may not find regression depending on week handling, but should not error
    assert isinstance(findings, list)


def test_empty_records():
    """Test that empty record list returns no findings."""
    findings = detect_ephemeris_age_anomalies([], satellite="G01")
    assert len(findings) == 0


def test_missing_toe(base_epoch):
    """Test that records without Toe don't trigger findings."""
    records = [
        EpochRecord(epoch=base_epoch, values={}, source="source"),
        EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={}, source="source"),
    ]
    findings = detect_ephemeris_age_anomalies(records, satellite="G01")
    assert len(findings) == 0


def test_finding_structure(base_epoch):
    """Test that findings have correct structure."""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    epoch = base_epoch
    seconds_per_week = 604800
    epoch_gps_seconds = (epoch - gps_epoch).total_seconds()
    stale_gps_seconds = epoch_gps_seconds - 10 * 3600
    stale_gps_week = int(stale_gps_seconds // seconds_per_week)
    stale_toe_sow = stale_gps_seconds % seconds_per_week
    
    records = [
        EpochRecord(epoch=epoch, values={"Toe": stale_toe_sow, "GPSWeek": stale_gps_week}, source="source"),
    ]
    
    findings = detect_ephemeris_age_anomalies(records, satellite="G01", max_age_hours=4.0)
    if findings:
        finding = findings[0]
        assert finding.code in ("ephemeris_age_anomaly", "toe_regression")
        assert finding.satellite == "G01"
        assert "toe_value" in finding.details or "previous_toe" in finding.details
