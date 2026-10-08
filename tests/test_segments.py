"""Unit tests for autocine/segments.py -- the pure segmented-take math.

No ffmpeg, no video, no permissions. The load-bearing piece is `SegmentClock`:
it is the ONE shared event-time -> output-time map that `render`, `beats`, and
`describe_session` must agree on (the trailing-cluster contract). These tests
pin the map, the clamp-not-drop / shape-preserving guarantee, and the
fail-safe discriminator that keeps the off switch bit-exact.

Design: docs/architecture.md.
"""

import unittest

import numpy as np

from autocine import segments


class IsSegmentedMeta(unittest.TestCase):
    def test_two_entries_with_files_is_segmented(self):
        meta = {"capture_segments": [{"file": "seg_0.mov"},
                                     {"file": "seg_1.mov"}]}
        self.assertTrue(segments.is_segmented_meta(meta))

    def test_single_entry_is_not_segmented(self):
        # A no-pause take never writes the key at all; even a stray 1-entry
        # list must not trip the branch.
        self.assertFalse(segments.is_segmented_meta(
            {"capture_segments": [{"file": "seg_0.mov"}]}))

    def test_missing_file_is_not_segmented(self):
        self.assertFalse(segments.is_segmented_meta(
            {"capture_segments": [{"file": "seg_0.mov"}, {"index": 1}]}))

    def test_absent_key_is_not_segmented(self):
        self.assertFalse(segments.is_segmented_meta({"raw": "raw.mov"}))
        self.assertFalse(segments.is_segmented_meta({}))
        self.assertFalse(segments.is_segmented_meta(None))

    def test_multi_native_is_never_segmented(self):
        # capture_channels present -> multi-native owns the shape, even if a
        # capture_segments list were somehow also present.
        meta = {"capture_channels": [{"file": "raw_0.mov"}],
                "capture_segments": [{"file": "seg_0.mov"},
                                     {"file": "seg_1.mov"}]}
        self.assertFalse(segments.is_segmented_meta(meta))


class SegmentClockMap(unittest.TestCase):
    def _clock(self):
        # Two segments, fps 10 so a frame is 0.1s. Segment 0: t0=100.0,
        # 20 frames -> D0 = 2.0s (content 100.0..102.0). A 5s paused gap.
        # Segment 1: t0=107.0, 30 frames -> D1 = 3.0s (content 107.0..110.0).
        # S0=0, S1=2.0, total=5.0.
        meta = {"capture_segments": [{"t0_monotonic": 100.0},
                                     {"t0_monotonic": 107.0}]}
        return segments.SegmentClock.from_meta(meta, [20, 30], fps=10)

    def test_starts_and_total(self):
        c = self._clock()
        self.assertEqual(c.starts, [0.0, 2.0])
        self.assertAlmostEqual(c.total, 5.0)
        self.assertAlmostEqual(c.duration(), 5.0)

    def test_in_segment_zero(self):
        c = self._clock()
        # t=101.0 is 1.0s into seg 0 -> output 1.0.
        self.assertAlmostEqual(c.media([101.0])[0], 1.0)

    def test_in_segment_one_deletes_the_gap(self):
        c = self._clock()
        # t=108.5 is 1.5s into seg 1 -> S1 + 1.5 = 3.5. The 5s paused gap and
        # seg1's t0 (107.0) do NOT appear -- gap deleted.
        self.assertAlmostEqual(c.media([108.5])[0], 3.5)

    def test_seam_is_continuous(self):
        c = self._clock()
        # Last content instant of seg0 (~102.0-) maps to ~2.0; first of seg1
        # (107.0) maps to exactly S1=2.0. Contiguous across the deleted gap.
        self.assertAlmostEqual(c.media([107.0])[0], 2.0)

    def test_before_the_take_clamps_to_zero(self):
        c = self._clock()
        # A geometry sample at t=99.5 (state the take opened in) clamps to 0.
        self.assertAlmostEqual(c.media([99.5])[0], 0.0)

    def test_in_gap_clamps_to_the_seam(self):
        c = self._clock()
        # t=104.0 is inside the deleted 102.0..107.0 gap -> clamps to seam 2.0.
        self.assertAlmostEqual(c.media([104.0])[0], 2.0)

    def test_subframe_past_segment_end_clamps_not_drops(self):
        c = self._clock()
        # A click 102.05 -- just past seg0's content end (102.0), before seg1
        # -> clamps to the seam 2.0 (belongs to seg0's tail, not the gap).
        self.assertAlmostEqual(c.media([102.05])[0], 2.0)

    def test_past_the_end_clamps_to_total(self):
        c = self._clock()
        self.assertAlmostEqual(c.media([200.0])[0], 5.0)

    def test_shape_preserving_never_drops(self):
        c = self._clock()
        # A mix of in-segment, in-gap, before, and after -- output length MUST
        # equal input length (the clamp-not-drop invariant that keeps the
        # parallel x/y arrays index-aligned).
        ts = [99.0, 101.0, 104.0, 108.0, 500.0]
        out = c.media(ts)
        self.assertEqual(out.shape[0], len(ts))
        self.assertFalse(np.any(np.isnan(out)))
        # Monotonic non-decreasing since the inputs are ascending.
        self.assertTrue(np.all(np.diff(out) >= -1e-9))

    def test_empty_input(self):
        c = self._clock()
        self.assertEqual(c.media([]).size, 0)

    def test_ascending_clicks_across_seam_stay_ordered(self):
        c = self._clock()
        # Clicks straddling the pause become CONTIGUOUS in output time -- the
        # whole point (a cluster spanning the gap clusters as one).
        out = c.media([101.9, 107.1])  # ~1.9 in seg0, 0.1 into seg1 -> 2.1
        self.assertAlmostEqual(out[0], 1.9)
        self.assertAlmostEqual(out[1], 2.1)
        self.assertLess(out[1] - out[0], 0.5)  # contiguous, gap gone

    def test_from_meta_length_mismatch_raises(self):
        meta = {"capture_segments": [{"t0_monotonic": 1.0}]}
        with self.assertRaises(ValueError):
            segments.SegmentClock.from_meta(meta, [10, 20], fps=10)


class SegmentClockIdentityBaseline(unittest.TestCase):
    """A single-segment clock must reduce to plain `arr - t0` -- the guarantee
    that lets beats/describe route through the same mapper without changing the
    non-segmented (identity) numbers when they ever hold a 1-entry clock."""

    def test_single_segment_is_plain_subtraction(self):
        c = segments.SegmentClock([50.0], [10.0])
        for t in (50.0, 55.0, 59.999):
            self.assertAlmostEqual(c.media([t])[0], t - 50.0)


if __name__ == "__main__":
    unittest.main()
