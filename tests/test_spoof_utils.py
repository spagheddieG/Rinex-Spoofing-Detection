import unittest
from datetime import datetime, timedelta
from rinex_spoofing.spoof_utils import (
    EpochRecord,
    detect_parameter_change_without_iode_change,
    detect_stale_data,
    detect_unexpected_iod_changes,
)

class TestSpoofUtils(unittest.TestCase):
    def test_detect_parameter_change_without_iode_change(self):
        # Case 1: No change
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 1, "IODC": 1, "Crs": 10.0}),
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 15), values={"IODE": 1, "IODC": 1, "Crs": 10.0}),
        ]
        findings = detect_parameter_change_without_iode_change(records, "G01")
        self.assertEqual(len(findings), 0)

        # Case 2: Change with IODE update (normal)
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 1, "IODC": 1, "Crs": 10.0}),
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 15), values={"IODE": 2, "IODC": 2, "Crs": 20.0}),
        ]
        findings = detect_parameter_change_without_iode_change(records, "G01")
        self.assertEqual(len(findings), 0)

        # Case 3: Change WITHOUT IODE update (spoofing indicator)
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 1, "IODC": 1, "Crs": 10.0}),
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 15), values={"IODE": 1, "IODC": 1, "Crs": 20.0}),
        ]
        findings = detect_parameter_change_without_iode_change(records, "G01")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "param_change_without_iode")
        self.assertIn("Crs", findings[0].details["fields"])

    def test_detect_stale_data(self):
        # Case 1: Normal updates
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 1}),
            EpochRecord(epoch=datetime(2023, 1, 1, 1, 0), values={"IODE": 2}),
        ]
        findings = detect_stale_data(records, "G01", max_interval=timedelta(hours=2))
        self.assertEqual(len(findings), 0)

        # Case 2: Stale data (same values for long time)
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 1}),
            EpochRecord(epoch=datetime(2023, 1, 1, 3, 0), values={"IODE": 1}),
        ]
        findings = detect_stale_data(records, "G01", max_interval=timedelta(hours=2))
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "stale_data")

    def test_detect_unexpected_iod_changes(self):
        # Case 1: Normal increment
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 10, "IODC": 10}),
            EpochRecord(epoch=datetime(2023, 1, 1, 1, 0), values={"IODE": 11, "IODC": 11}),
        ]
        findings = detect_unexpected_iod_changes(records, "G01")
        self.assertEqual(len(findings), 0)

        # Case 2: Regression (backwards jump)
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 10, "IODC": 10}),
            EpochRecord(epoch=datetime(2023, 1, 1, 1, 0), values={"IODE": 9, "IODC": 9}),
        ]
        findings = detect_unexpected_iod_changes(records, "G01")
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].code, "iod_regression")

        # Case 3: Wrap around (valid)
        # Assuming 1024 wrap around for IODC/IODE? The code checks for small negative diffs.
        # If prev=1023, curr=0 -> diff = -1023. (0-1023)%1024 = 1. Valid.
        records = [
            EpochRecord(epoch=datetime(2023, 1, 1, 0, 0), values={"IODE": 1023, "IODC": 1023}),
            EpochRecord(epoch=datetime(2023, 1, 1, 1, 0), values={"IODE": 0, "IODC": 0}),
        ]
        findings = detect_unexpected_iod_changes(records, "G01")
        self.assertEqual(len(findings), 0)

if __name__ == "__main__":
    unittest.main()
