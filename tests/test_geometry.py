"""Unit tests for geometry.load_events, including the key-activity stream."""

import json
import os
import tempfile
import unittest

import numpy as np

from autocine import geometry


def _write_events(lines):
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for e in lines:
            f.write(json.dumps(e) + "\n")
    return path


class LoadEvents(unittest.TestCase):
    def test_key_lines_parsed_into_sorted_keys_t(self):
        path = _write_events([
            {"t": 1.0, "type": "move", "x": 10, "y": 20},
            {"t": 2.3, "type": "key", "x": 10.0, "y": 20.0},
            {"t": 1.9, "type": "key", "x": 10.0, "y": 20.0},
            {"t": 2.5, "type": "down", "x": 30, "y": 40},
        ])
        try:
            ev = geometry.load_events(path)
        finally:
            os.remove(path)
        np.testing.assert_allclose(ev["keys_t"], [1.9, 2.3])
        self.assertEqual(ev["clicks_t"].size, 1)
        # key lines are NOT position samples: move track has move + click only
        self.assertEqual(ev["moves_t"].size, 2)

    def test_key_lines_without_xy_are_tolerated(self):
        # Hand-written synthetic sessions may omit the x/y padding.
        path = _write_events([
            {"t": 1.0, "type": "key"},
            {"t": 1.1, "type": "key"},
        ])
        try:
            ev = geometry.load_events(path)
        finally:
            os.remove(path)
        np.testing.assert_allclose(ev["keys_t"], [1.0, 1.1])

    def test_up_lines_parsed_sorted_and_kept_out_of_move_track(self):
        path = _write_events([
            {"t": 1.0, "type": "down", "x": 10, "y": 20},
            {"t": 3.5, "type": "up", "x": 200, "y": 90},
            {"t": 0.9, "type": "move", "x": 5, "y": 5},
        ])
        try:
            ev = geometry.load_events(path)
        finally:
            os.remove(path)
        np.testing.assert_allclose(ev["ups_t"], [3.5])
        # up is not a move sample (keeps drag_hold-off renders bit-exact)
        self.assertEqual(ev["moves_t"].size, 2)   # the move + the click

    def test_scroll_lines_parsed_sorted_and_kept_out_of_move_track(self):
        path = _write_events([
            {"t": 1.0, "type": "move", "x": 10, "y": 20},
            {"t": 3.1, "type": "scroll", "x": 500, "y": 400},
            {"t": 2.4, "type": "scroll", "x": 510, "y": 410},
            {"t": 2.5, "type": "down", "x": 30, "y": 40},
        ])
        try:
            ev = geometry.load_events(path)
        finally:
            os.remove(path)
        # sorted by time, x/y staying aligned with their t
        np.testing.assert_allclose(ev["scrolls_t"], [2.4, 3.1])
        np.testing.assert_allclose(ev["scrolls_x"], [510.0, 500.0])
        np.testing.assert_allclose(ev["scrolls_y"], [410.0, 400.0])
        # scroll lines are NOT move samples (keeps scroll_zoom-off renders
        # and every pre-scroll session bit-exact): move + click only
        self.assertEqual(ev["moves_t"].size, 2)

    def test_old_session_without_keys_yields_empty_array(self):
        path = _write_events([
            {"t": 1.0, "type": "move", "x": 10, "y": 20},
            {"t": 2.0, "type": "down", "x": 30, "y": 40},
        ])
        try:
            ev = geometry.load_events(path)
        finally:
            os.remove(path)
        self.assertEqual(ev["keys_t"].size, 0)
        self.assertEqual(ev["scrolls_t"].size, 0)

    def test_missing_file_yields_empty_arrays(self):
        ev = geometry.load_events("/nonexistent/events.jsonl")
        self.assertEqual(ev["keys_t"].size, 0)
        self.assertEqual(ev["clicks_t"].size, 0)
        self.assertEqual(ev["scrolls_t"].size, 0)


if __name__ == "__main__":
    unittest.main()
