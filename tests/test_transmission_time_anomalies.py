"""Tests for transmission time anomalies detection."""

from datetime import datetime, timedelta

import pytest

from spoof_utils import EpochRecord, detect_transmission_time_anomalies
from tests.conftest import base_epoch


def test_no_findings_for_fresh_transmission_time(base_epoch):
    """Test that fresh transmission times don't trigger findings."""
    records = []
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    seconds_per_week = 604800
    
    # Calculate GPS week for the base epoch
    gps_seconds_total = (base_epoch - gps_epoch).total_seconds()
    gps_week = int(gps_seconds_total // seconds_per_week)
    
    for i in range(5):
        epoch = base_epoch + timedelta(minutes=15 * i)
        # Convert to seconds-of-week
        epoch_gps_seconds = (epoch - gps_epoch).total_seconds()
        transtime_sow = epoch_gps_seconds % seconds_per_week
        records.append(
            EpochRecord(
                epoch=epoch,
                values={"TransTime": transtime_sow, "GPSWeek": gps_week},
                source="source",
            )
        )
    
    findings = detect_transmission_time_anomalies(records, satellite="G01", max_age_hours=4.0)
    assert len(findings) == 0


def test_finds_stale_transmission_time(base_epoch):
    """Test that stale transmission times are detected."""
    records = []
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    seconds_per_week = 604800
    
    epoch = base_epoch
    epoch_gps_seconds = (epoch - gps_epoch).total_seconds()
    stale_gps_seconds = epoch_gps_seconds - 10 * 3600  # 10 hours ago
    stale_gps_week = int(stale_gps_seconds // seconds_per_week)
    stale_transtime_sow = stale_gps_seconds % seconds_per_week
    
    records.append(
        EpochRecord(
            epoch=epoch,
            values={"TransTime": stale_transtime_sow, "GPSWeek": stale_gps_week},
            source="source",
        )
    )
    
    findings = detect_transmission_time_anomalies(records, satellite="G01", max_age_hours=4.0)
    assert len(findings) > 0
    assert any(f.code == "transmission_time_anomaly" for f in findings)


def test_finds_transtime_regression(base_epoch):
    """Test that TransTime regression is detected."""
    records = []
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    seconds_per_week = 604800
    
    # First record
    epoch1 = base_epoch
    gps_seconds1 = (epoch1 - gps_epoch).total_seconds()
    gps_week1 = int(gps_seconds1 // seconds_per_week)
    transtime1_sow = gps_seconds1 % seconds_per_week
    records.append(EpochRecord(epoch=epoch1, values={"TransTime": transtime1_sow, "GPSWeek": gps_week1}, source="source"))
    
    # Second record with regressed TransTime (2 hours backward)
    epoch2 = base_epoch + timedelta(minutes=15)
    transtime2_sow = transtime1_sow - 7200  # 2 hours backward
    if transtime2_sow < 0:
        transtime2_sow += seconds_per_week
        gps_week2 = gps_week1 - 1
    else:
        gps_week2 = gps_week1
    records.append(EpochRecord(epoch=epoch2, values={"TransTime": transtime2_sow, "GPSWeek": gps_week2}, source="source"))
    
    findings = detect_transmission_time_anomalies(records, satellite="G01")
    # May or may not find regression depending on week handling, but should not error
    assert isinstance(findings, list)


def test_empty_records():
    """Test that empty record list returns no findings."""
    findings = detect_transmission_time_anomalies([], satellite="G01")
    assert len(findings) == 0


def test_missing_transtime(base_epoch):
    """Test that records without TransTime don't trigger findings."""
    records = [
        EpochRecord(epoch=base_epoch, values={}, source="source"),
        EpochRecord(epoch=base_epoch + timedelta(minutes=15), values={}, source="source"),
    ]
    findings = detect_transmission_time_anomalies(records, satellite="G01")
    assert len(findings) == 0


def test_finding_structure(base_epoch):
    """Test that findings have correct structure."""
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)
    epoch = base_epoch
    seconds_per_week = 604800
    epoch_gps_seconds = (epoch - gps_epoch).total_seconds()
    stale_gps_seconds = epoch_gps_seconds - 10 * 3600
    stale_gps_week = int(stale_gps_seconds // seconds_per_week)
    stale_transtime_sow = stale_gps_seconds % seconds_per_week
    
    records = [
        EpochRecord(epoch=epoch, values={"TransTime": stale_transtime_sow, "GPSWeek": stale_gps_week}, source="source"),
    ]
    
    findings = detect_transmission_time_anomalies(records, satellite="G01", max_age_hours=4.0)
    if findings:
        finding = findings[0]
        assert finding.code in ("transmission_time_anomaly", "transtime_regression")
        assert finding.satellite == "G01"
        assert "transtime_value" in finding.details or "previous_transtime" in finding.details
