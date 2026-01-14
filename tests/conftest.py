"""Shared fixtures for spoofing detection tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

import pytest

from spoof_utils import EpochRecord


@pytest.fixture
def base_epoch() -> datetime:
    """Base epoch for test data."""
    return datetime(2025, 11, 10, 0, 0, 0)


@pytest.fixture
def sample_epoch_record(base_epoch: datetime) -> EpochRecord:
    """Sample epoch record with standard navigation parameters."""
    return EpochRecord(
        epoch=base_epoch,
        values={
            "SVclockBias": -0.0002233255654573,
            "SVclockDrift": -7.958078640513e-13,
            "SVclockDriftRate": 0.0,
            "IODE": 46.0,
            "IODC": 46.0,
            "Crs": -80.90625,
            "DeltaN": 4.136243719725e-09,
            "M0": 1.796562856363,
            "Cuc": -4.347413778305e-06,
            "Eccentricity": 0.005502786720172,
            "Cus": 3.525987267494e-06,
            "sqrtA": 5153.64276123,
            "Toe": 86400.0,
            "Cic": 3.911554813385e-08,
            "Omega0": -0.2140837155261,
            "Cis": -1.620501279831e-07,
            "Io": 0.9788563651595,
            "Crc": 322.21875,
            "omega": 1.438442713713,
            "OmegaDot": -8.064621638148e-09,
            "IDOT": -1.464346710204e-10,
            "CodesL2": 1.0,
            "GPSWeek": 2392.0,
            "L2Pflag": 0.0,
            "SVacc": 2.0,
            "health": 0.0,
            "TGD": -1.071020960808e-08,
            "TransTime": 79218.0,
            "FitIntvl": 4.0,
        },
        source="test_source",
    )


@pytest.fixture
def sample_records_sequence(base_epoch: datetime) -> List[EpochRecord]:
    """Sequence of records with gradually changing parameters."""
    records: List[EpochRecord] = []
    
    base_values = {
        "SVclockBias": -0.0002233255654573,
        "SVclockDrift": -7.958078640513e-13,
        "SVclockDriftRate": 0.0,
        "IODE": 46.0,
        "IODC": 46.0,
        "Crs": -80.90625,
        "DeltaN": 4.136243719725e-09,
        "M0": 1.796562856363,
        "Cuc": -4.347413778305e-06,
        "Eccentricity": 0.005502786720172,
        "Cus": 3.525987267494e-06,
        "sqrtA": 5153.64276123,
        "Toe": 86400.0,
        "Cic": 3.911554813385e-08,
        "Omega0": -0.2140837155261,
        "Cis": -1.620501279831e-07,
        "Io": 0.9788563651595,
        "Crc": 322.21875,
        "omega": 1.438442713713,
        "OmegaDot": -8.064621638148e-09,
        "IDOT": -1.464346710204e-10,
        "CodesL2": 1.0,
        "GPSWeek": 2392.0,
        "L2Pflag": 0.0,
        "SVacc": 2.0,
        "health": 0.0,
        "TGD": -1.071020960808e-08,
        "TransTime": 79218.0,
        "FitIntvl": 4.0,
    }
    
    for i in range(10):
        epoch = base_epoch + timedelta(minutes=15 * i)
        values = base_values.copy()
        # Gradually change some parameters
        values["SVclockBias"] = base_values["SVclockBias"] + i * 1e-7
        values["Toe"] = base_values["Toe"] + i * 900.0  # 15 minutes = 900 seconds
        values["TransTime"] = base_values["TransTime"] + i * 900.0
        
        records.append(EpochRecord(epoch=epoch, values=values, source=f"source_{i}"))
    
    return records


@pytest.fixture
def sample_records_rapid_change(base_epoch: datetime) -> List[EpochRecord]:
    """Sequence of records with rapid parameter changes."""
    records: List[EpochRecord] = []
    
    base_values = {
        "SVclockBias": -0.0002233255654573,
        "IODE": 46.0,
        "IODC": 46.0,
        "Crs": -80.90625,
    }
    
    # First record
    values1 = base_values.copy()
    records.append(EpochRecord(epoch=base_epoch, values=values1, source="source1"))
    
    # Second record with rapid change (15 minutes later)
    values2 = base_values.copy()
    values2["SVclockBias"] = base_values["SVclockBias"] + 100.0  # Huge change
    records.append(EpochRecord(epoch=base_epoch + timedelta(minutes=15), values=values2, source="source2"))
    
    return records


@pytest.fixture
def sample_records_replay(base_epoch: datetime) -> List[EpochRecord]:
    """Sequence of records with repeated patterns (replay simulation)."""
    records: List[EpochRecord] = []
    
    pattern1 = {"SVclockBias": -0.0001, "IODE": 10.0, "IODC": 10.0, "Crs": 100.0}
    pattern2 = {"SVclockBias": -0.0002, "IODE": 11.0, "IODC": 11.0, "Crs": 101.0}
    pattern3 = {"SVclockBias": -0.0003, "IODE": 12.0, "IODC": 12.0, "Crs": 102.0}
    pattern4 = {"SVclockBias": -0.0004, "IODE": 13.0, "IODC": 13.0, "Crs": 103.0}
    
    patterns = [pattern1, pattern2, pattern3, pattern4]
    
    # First occurrence of the pattern (4 records, 1 hour total)
    for i, pattern in enumerate(patterns):
        epoch = base_epoch + timedelta(minutes=15 * i)
        records.append(EpochRecord(epoch=epoch, values=pattern.copy(), source="source_1"))
    
    # Gap of 2 hours (ensures at least 1 hour gap from end of first sequence)
    gap_hours = 2
    
    # Repeat the pattern (replay attack simulation)
    for i, pattern in enumerate(patterns):
        epoch = base_epoch + timedelta(hours=gap_hours, minutes=15 * i)
        records.append(EpochRecord(epoch=epoch, values=pattern.copy(), source="source_2"))
    
    return records


@pytest.fixture
def sample_records_stale(base_epoch: datetime) -> List[EpochRecord]:
    """Sequence of records with stale (unchanged) data."""
    records: List[EpochRecord] = []
    
    base_values = {
        "SVclockBias": -0.0002233255654573,
        "IODE": 46.0,
        "IODC": 46.0,
    }
    
    # Same values for multiple intervals
    for i in range(5):
        epoch = base_epoch + timedelta(minutes=15 * i)
        records.append(EpochRecord(epoch=epoch, values=base_values.copy(), source="source"))
    
    return records


@pytest.fixture
def sample_records_outlier(base_epoch: datetime) -> List[EpochRecord]:
    """Sequence of records with one outlier value."""
    records: List[EpochRecord] = []
    
    base_value = -0.0002233255654573
    
    # Normal values
    for i in range(9):
        epoch = base_epoch + timedelta(minutes=15 * i)
        values = {"SVclockBias": base_value + i * 1e-8, "IODE": 46.0}
        records.append(EpochRecord(epoch=epoch, values=values, source="source"))
    
    # Outlier value (very different)
    epoch = base_epoch + timedelta(minutes=15 * 9)
    values = {"SVclockBias": base_value + 1.0, "IODE": 46.0}  # Huge outlier
    records.append(EpochRecord(epoch=epoch, values=values, source="source"))
    
    return records
