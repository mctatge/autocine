"""Unit tests for permission report shaping logic."""

import unittest

from autocine import permissions


class PermissionReport(unittest.TestCase):
    def test_build_report_all_granted(self):
        checks = {
            "screen_recording": {"label": "Screen Recording", "state": "granted", "detail": "ok"},
            "input_monitoring": {"label": "Input Monitoring", "state": "granted", "detail": "ok"},
            "accessibility": {"label": "Accessibility", "state": "granted", "detail": "ok"},
        }
        rep = permissions.build_report(checks, "darwin")
        self.assertTrue(rep["all_required_granted"])
        self.assertTrue(rep["can_attempt_record"])
        self.assertEqual(rep["missing_required"], [])
        self.assertEqual(rep["unknown_required"], [])

    def test_build_report_missing_blocks_record(self):
        checks = {
            "screen_recording": {"label": "Screen Recording", "state": "missing", "detail": "grant it"},
            "input_monitoring": {"label": "Input Monitoring", "state": "granted", "detail": "ok"},
            "accessibility": {"label": "Accessibility", "state": "granted", "detail": "ok"},
        }
        rep = permissions.build_report(checks, "darwin")
        self.assertFalse(rep["all_required_granted"])
        self.assertFalse(rep["can_attempt_record"])
        self.assertEqual(rep["missing_required"], ["screen_recording"])
        self.assertIn("blocked", rep["summary"])

    def test_build_report_unknown_allows_attempt(self):
        checks = {
            "screen_recording": {"label": "Screen Recording", "state": "unknown", "detail": "n/a"},
            "input_monitoring": {"label": "Input Monitoring", "state": "granted", "detail": "ok"},
            "accessibility": {"label": "Accessibility", "state": "granted", "detail": "ok"},
        }
        rep = permissions.build_report(checks, "darwin")
        self.assertFalse(rep["all_required_granted"])
        self.assertTrue(rep["can_attempt_record"])
        self.assertEqual(rep["missing_required"], [])
        self.assertEqual(rep["unknown_required"], ["screen_recording"])


if __name__ == "__main__":
    unittest.main()
