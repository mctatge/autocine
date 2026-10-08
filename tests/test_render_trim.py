"""Unit tests for render trim-frame bounds."""

import unittest

from autocine import render


class TrimBounds(unittest.TestCase):
    def test_full_clip_when_trim_end_none(self):
        start, end = render._trim_frame_bounds(300, 60.0, 0.0, None)
        self.assertEqual(start, 0)
        self.assertEqual(end, 300)

    def test_fractional_seconds_map_to_frame_bounds(self):
        start, end = render._trim_frame_bounds(300, 60.0, 1.2, 2.3)
        self.assertEqual(start, 72)   # floor(1.2 * 60)
        self.assertEqual(end, 138)    # ceil(2.3 * 60)

    def test_trim_start_beyond_clip_clamps_to_last_frame(self):
        start, end = render._trim_frame_bounds(120, 30.0, 9.0, None)
        self.assertEqual(start, 119)
        self.assertEqual(end, 120)

    def test_trim_end_before_start_keeps_one_frame(self):
        start, end = render._trim_frame_bounds(200, 50.0, 3.0, 2.0)
        self.assertEqual(start, 150)
        self.assertEqual(end, 151)


if __name__ == "__main__":
    unittest.main()
