"""Integration tests for spoof_detection.py."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spoof_detection import parse_args, run_detection
from spoof_utils import Finding, load_nav_json


def test_parse_args_defaults():
    """Test that parse_args returns expected defaults."""
    args = parse_args(["test.json"])
    assert args.json_path == Path("test.json")
    assert args.tolerance == 0.0
    assert args.max_interval_hours == 2.0
    assert args.max_ephemeris_age_hours == 4.0
    assert args.replay_sequence_length == 4
    assert args.enable_cross_satellite_checks is False
    assert args.disable_new_detections is False


def test_parse_args_with_options():
    """Test parse_args with various options."""
    args = parse_args(
        [
            "test.json",
            "--tolerance",
            "1e-9",
            "--max-interval-hours",
            "3.0",
            "--max-ephemeris-age-hours",
            "5.0",
            "--replay-sequence-length",
            "6",
            "--enable-cross-satellite-checks",
            "--ignore-satellites",
            "G01",
            "G02",
        ]
    )
    assert args.tolerance == 1e-9
    assert args.max_interval_hours == 3.0
    assert args.max_ephemeris_age_hours == 5.0
    assert args.replay_sequence_length == 6
    assert args.enable_cross_satellite_checks is True
    assert args.ignore_satellites == ["G01", "G02"]


def test_parse_args_disable_new_detections():
    """Test parse_args with disable-new-detections flag."""
    args = parse_args(["test.json", "--disable-new-detections"])
    assert args.disable_new_detections is True


@patch("spoof_detection.load_nav_json")
@patch("spoof_detection.extract_satellite_timeseries_multisource")
def test_run_detection_basic(mock_extract, mock_load):
    """Test run_detection with basic setup."""
    from datetime import datetime
    from spoof_utils import EpochRecord
    
    # Mock data
    mock_load.return_value = {"test": "data"}
    mock_extract.return_value = {
        "G01": [
            EpochRecord(epoch=datetime(2025, 1, 1), values={"IODE": 10.0, "IODC": 10.0}, source="source"),
        ]
    }
    
    args = parse_args(["test.json", "--disable-new-detections"])
    findings = run_detection(args)
    
    # Should run without errors
    assert isinstance(findings, list)


@patch("spoof_detection.load_nav_json")
def test_run_detection_file_not_found(mock_load):
    """Test run_detection handles file not found."""
    mock_load.side_effect = FileNotFoundError("File not found")
    
    args = parse_args(["nonexistent.json"])
    
    with pytest.raises(FileNotFoundError):
        run_detection(args)


def test_finding_structure():
    """Test that Finding dataclass has expected structure."""
    from datetime import datetime
    
    finding = Finding(
        satellite="G01",
        epoch=datetime(2025, 1, 1),
        code="test_code",
        description="Test description",
        details={"key": "value"},
        discovered_at=datetime(2025, 1, 1),
    )
    
    assert finding.satellite == "G01"
    assert finding.code == "test_code"
    assert finding.description == "Test description"
    assert finding.details == {"key": "value"}
