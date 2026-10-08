"""Pure math tests for SCENE takes (docs/architecture.md): the
`capture_scenes` discriminator, the per-scene channel alignment, the
origin-adjusted scene clock, event OWNERSHIP at seams, and the scene-scoped
events view. No ffmpeg, no permissions, no files.
"""

import unittest

import numpy as np

from autocine import segments


def _chan(file="raw_0.mov", t0=100.0):
    return {"file": file, "t0_monotonic": t0, "mode": "window_native"}


def _scene_meta(scenes):
    return {"fps": 60, "capture_scenes": scenes}


def _two_scene_meta(t0a=100.0, t0b=110.0, extra_a=None):
    a = {"index": 0, "t0_monotonic": t0a,
         "channels": [_chan("scene_0_raw_0.mov", t0a)]}
    if extra_a:
        a.update(extra_a)
    b = {"index": 1, "t0_monotonic": t0b,
         "channels": [_chan("scene_1_raw_0.mov", t0b),
                      _chan("scene_1_raw_1.mov", t0b + 0.05)]}
    return _scene_meta([a, b])


class IsSceneMeta(unittest.TestCase):
    def test_two_fleet_scenes_is_a_scene_take(self):
        self.assertTrue(segments.is_scene_meta(_two_scene_meta()))

    def test_one_scene_is_not(self):
        # A one-scene take collapses to its legacy shape at finalize; a
        # 1-entry manifest on disk is malformed, mirroring is_segmented_meta.
        meta = _scene_meta([{"index": 0, "channels": [_chan()]}])
        self.assertFalse(segments.is_scene_meta(meta))

    def test_sibling_manifests_are_rejected(self):
        meta = _two_scene_meta()
        meta["capture_channels"] = [_chan()]
        self.assertFalse(segments.is_scene_meta(meta))
        meta = _two_scene_meta()
        meta["capture_segments"] = [{"file": "seg_0.mov"}]
        self.assertFalse(segments.is_scene_meta(meta))

    def test_file_xor_channels(self):
        meta = _two_scene_meta()
        meta["capture_scenes"][0]["file"] = "scene_0.mov"   # both -> malformed
        self.assertFalse(segments.is_scene_meta(meta))
        meta = _scene_meta([{"index": 0}, {"index": 1, "channels": [_chan()]}])
        self.assertFalse(segments.is_scene_meta(meta))      # neither

    def test_whole_screen_scene_entries_are_valid(self):
        # S2 forward-compat: a `file` scene beside a fleet scene parses.
        meta = _scene_meta([
            {"index": 0, "t0_monotonic": 1.0, "file": "scene_0.mov"},
            {"index": 1, "t0_monotonic": 9.0, "channels": [_chan()]}])
        self.assertTrue(segments.is_scene_meta(meta))

    def test_channel_count_bounds(self):
        five = [_chan("c{}.mov".format(i)) for i in range(5)]
        meta = _scene_meta([{"index": 0, "channels": five},
                            {"index": 1, "channels": [_chan()]}])
        self.assertFalse(segments.is_scene_meta(meta))

    def test_channel_without_file_is_rejected(self):
        meta = _two_scene_meta()
        del meta["capture_scenes"][1]["channels"][1]["file"]
        self.assertFalse(segments.is_scene_meta(meta))

    def test_not_a_dict_and_missing_key_fail_safe(self):
        self.assertFalse(segments.is_scene_meta(None))
        self.assertFalse(segments.is_scene_meta({}))
        self.assertFalse(segments.is_scene_meta({"capture_scenes": "x"}))


class SceneChannelAlignment(unittest.TestCase):
    def test_matches_the_multi_native_math(self):
        # ch0 at t=100.0, ch1 starts 3 frames later, ch2 one frame EARLIER.
        fps = 60.0
        t0s = [100.0, 100.05, 100.0 - 1 / 60.0]
        counts = [600, 600, 600]
        origin, n_out, offsets = segments.scene_channel_alignment(
            t0s, counts, fps)
        self.assertEqual(offsets, [0, 3, -1])
        self.assertEqual(origin, 3)
        # ch0 contributes 600-3, ch1 600-0, ch2 600-4 -> min is 596.
        self.assertEqual(n_out, 596)

    def test_single_channel_is_identity(self):
        origin, n_out, offsets = segments.scene_channel_alignment(
            [50.0], [240], 60.0)
        self.assertEqual((origin, n_out, offsets), (0, 240, [0]))

    def test_null_channel_t0_raises(self):
        # Inheriting the reader-side `or session_t0` fallback would anchor the
        # channel to the TAKE origin and silently misalign the scene.
        with self.assertRaises(ValueError):
            segments.scene_channel_alignment([100.0, None], [10, 10], 60.0)


class SceneClockEntries(unittest.TestCase):
    def test_origin_adjusted_t0(self):
        # Scene 1's channel 0 starts at 110.0 but channel 1 starts 3 frames
        # later -> composite frame 0 is at 110.0 + 3/60.
        meta = _two_scene_meta(t0a=100.0, t0b=110.0)
        meta["capture_scenes"][1]["channels"][1]["t0_monotonic"] = 110.05
        t0s, durs, aligns, warns = segments.scene_clock_entries(
            meta["capture_scenes"], [[300], [300, 300]], 60.0)
        self.assertAlmostEqual(t0s[0], 100.0)
        self.assertAlmostEqual(t0s[1], 110.0 + 3 / 60.0)
        self.assertAlmostEqual(durs[0], 5.0)
        self.assertAlmostEqual(durs[1], 297 / 60.0)
        self.assertEqual(warns, [])

    def test_overlap_is_clipped_with_a_warning(self):
        meta = _two_scene_meta(t0a=100.0, t0b=104.0)
        t0s, durs, aligns, warns = segments.scene_clock_entries(
            meta["capture_scenes"], [[600], [60, 60]], 60.0)   # 10s vs 4s span
        # The clip lands ON THE FRAME GRID: durs must equal n_out/fps (the
        # frame stream the render emits and the beats rebuild derive from),
        # never the raw span -- or the three surfaces disagree sub-frame
        # exactly where the warning fires.
        span = t0s[1] - t0s[0]
        n_out = aligns[0][1]
        self.assertAlmostEqual(durs[0], n_out / 60.0)
        self.assertLessEqual(durs[0], span + 1e-9)
        self.assertTrue(any("overlaps" in w for w in warns))

    def test_float_noise_never_triggers_a_spurious_clip(self):
        # Scene 0's duration lands within float noise of the exact span; the
        # epsilon must keep the overlap guard quiet (a spurious clip drops
        # the scene's last frame and prints an alarming CFR warning).
        meta = _two_scene_meta(t0a=100.0, t0b=100.0 + 599.9999999 / 60.0)
        t0s, durs, aligns, warns = segments.scene_clock_entries(
            meta["capture_scenes"], [[600], [60, 60]], 60.0)
        self.assertEqual(aligns[0][1], 600)
        self.assertFalse(any("overlaps" in w for w in warns))

    def test_whole_screen_scene_raises_a_named_error_in_s1(self):
        # is_scene_meta accepts `file` scenes (the S2 forward shape), but no
        # S1 reader can produce a count for one -- the failure must be the
        # promised loud, NAMED error, never a bare IndexError.
        meta = _scene_meta([
            {"index": 0, "t0_monotonic": 1.0, "file": "scene_0.mov"},
            {"index": 1, "t0_monotonic": 9.0, "channels": [_chan()]}])
        with self.assertRaises(RuntimeError) as ctx:
            segments.scene_clock_entries(
                meta["capture_scenes"], [[], [300]], 60.0)
        self.assertIn("S2", str(ctx.exception))

    def test_shortfall_warning_from_wall_end(self):
        meta = _two_scene_meta(
            t0a=100.0, t0b=110.0,
            extra_a={"wall_end_monotonic": 106.0})   # decoded ends at 105.0
        _t0s, _durs, _al, warns = segments.scene_clock_entries(
            meta["capture_scenes"], [[300], [300, 300]], 60.0)
        self.assertTrue(any("clamp to the seam" in w for w in warns))

    def test_from_scene_meta_builds_the_same_clock(self):
        meta = _two_scene_meta()
        clock, aligns, warns = segments.SegmentClock.from_scene_meta(
            meta, [[300], [300, 297]], 60.0)
        self.assertEqual(len(clock.t0s), 2)
        self.assertAlmostEqual(clock.durs[0], 5.0)


class SceneOwnership(unittest.TestCase):
    def _clock(self):
        # scene 0: [100, 104), scene 1: [110, 115)
        return segments.SegmentClock([100.0, 110.0], [4.0, 5.0])

    def test_in_scene_events(self):
        clock = self._clock()
        own = clock.owner([101.0, 111.0])
        self.assertEqual(list(own), [0, 1])

    def test_gap_event_belongs_to_the_earlier_scene(self):
        # A click at 105.5 (after scene 0's content, before scene 1) clamps
        # to the seam in media(); ownership must follow the clamp target.
        clock = self._clock()
        self.assertEqual(list(clock.owner([105.5])), [0])
        self.assertAlmostEqual(float(clock.media([105.5])[0]), 4.0)

    def test_before_zero_belongs_to_scene_zero(self):
        clock = self._clock()
        self.assertEqual(list(clock.owner([99.0])), [0])

    def test_final_tail_belongs_to_the_final_scene(self):
        # media() clamps it to total; value-containment in [S_1, S_1+D_1)
        # would exclude it from EVERY scene -- the review's dropped-click bug.
        clock = self._clock()
        self.assertEqual(list(clock.owner([116.0])), [1])
        self.assertAlmostEqual(float(clock.media([116.0])[0]), 9.0)

    def test_owner_is_shape_preserving(self):
        clock = self._clock()
        self.assertEqual(clock.owner([]).shape, (0,))
        self.assertEqual(clock.owner([101, 105.5, 116]).shape, (3,))

    def test_scene_local_rebases_and_clamps(self):
        clock = self._clock()
        out = clock.scene_local(0, [100.0, 102.0, 105.5])
        self.assertAlmostEqual(float(out[0]), 0.0)
        self.assertAlmostEqual(float(out[1]), 2.0)
        # tail event clamps INSIDE the scene, never onto the next scene
        self.assertLess(float(out[2]), 4.0)
        self.assertGreater(float(out[2]), 3.99)


class SceneEventsView(unittest.TestCase):
    def _ev(self):
        return {
            "clicks_t": np.array([101.0, 105.5, 111.0]),
            "clicks_x": np.array([10.0, 20.0, 30.0]),
            "clicks_y": np.array([1.0, 2.0, 3.0]),
            "moves_t": np.array([100.5, 112.0]),
            "moves_x": np.array([5.0, 6.0]),
            "moves_y": np.array([50.0, 60.0]),
            "scrolls_t": np.array([]), "scrolls_x": np.array([]),
            "scrolls_y": np.array([]),
            "ups_t": np.array([101.1, 111.1]),
            "keys_t": np.array([110.5]),
            "windows_t": np.array([99.0, 101.0, 109.5, 111.5]),
            "windows_rect": np.array([[0, 0, 10, 10], [1, 0, 10, 10],
                                      [2, 0, 10, 10], [3, 0, 10, 10]],
                                     dtype=float),
            "windows_id": np.array([7, 7, 7, 7]),
            "windows_z": np.array([0, 0, 1, 0]),
        }

    def _clock(self):
        return segments.SegmentClock([100.0, 110.0], [4.0, 5.0])

    def test_point_events_subset_by_ownership_zip_aligned(self):
        view = segments.scene_events_view(self._clock(), 0, self._ev())
        # Scene 0 owns the in-scene click AND the seam-clamped one; scene 1's
        # click must NOT appear (the phantom-cluster rule).
        self.assertEqual(list(view["clicks_t"]), [101.0, 105.5])
        self.assertEqual(list(view["clicks_x"]), [10.0, 20.0])
        self.assertEqual(list(view["clicks_y"]), [1.0, 2.0])
        view1 = segments.scene_events_view(self._clock(), 1, self._ev())
        self.assertEqual(list(view1["clicks_t"]), [111.0])
        self.assertEqual(list(view1["clicks_x"]), [30.0])
        self.assertEqual(list(view1["keys_t"]), [110.5])

    def test_geometry_containment_plus_prior_anchor(self):
        view = segments.scene_events_view(self._clock(), 1, self._ev())
        # In-scene sample at 111.5, plus the nearest PRIOR sample (109.5) as
        # the left anchor for the resampler; scene-0 samples stay out.
        self.assertEqual(list(view["windows_t"]), [109.5, 111.5])
        self.assertEqual(list(view["windows_id"]), [7, 7])
        self.assertEqual(list(view["windows_z"]), [1, 0])
        self.assertEqual(view["windows_rect"].shape, (2, 4))

    def test_empty_events_stay_empty(self):
        empty = {k: np.array([]) for k in self._ev()}
        empty["windows_rect"] = np.zeros((0, 4))
        view = segments.scene_events_view(self._clock(), 0, empty)
        self.assertEqual(view["clicks_t"].size, 0)
        self.assertEqual(view["windows_t"].size, 0)
        self.assertEqual(view["windows_rect"].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
