"""Unit tests for the click-driven auto-zoom camera.

The legacy multi-layer planner (typing/scroll/drag/focused-vs-smooth,
regrip bridges, suppression merging) is gone; these tests exercise the
new surface: one concept (zoomRanges), auto-cluster clicks into
proposals, spring-driven follow-click-groups, snap-to-edges, and the
CLI-vs-authoritative caller distinction.
"""

import unittest

import numpy as np

from autocine import camera


def _P(max_zoom=2.0, **over):
    return camera.build_params(max_zoom, over)


def _frames(dur_s, fps=60.0):
    n = int(round(dur_s * fps))
    return np.arange(n) / float(fps)


def _empty():
    return np.array([])


class ClickClustering(unittest.TestCase):
    def test_gapped_clicks_split_into_separate_clusters(self):
        clicks = [(1.0, 100, 100), (2.0, 110, 110), (10.0, 200, 200)]
        cl = camera.cluster_clicks(clicks, chain_gap=4.0)
        self.assertEqual(len(cl), 2)
        self.assertEqual(len(cl[0]), 2)
        self.assertEqual(len(cl[1]), 1)

    def test_close_clicks_join(self):
        clicks = [(1.0, 100, 100), (2.0, 110, 110), (5.5, 120, 120)]
        cl = camera.cluster_clicks(clicks, chain_gap=4.0)
        self.assertEqual(len(cl), 1)
        self.assertEqual(len(cl[0]), 3)


class ProposedRanges(unittest.TestCase):
    def test_pre_roll_leads_first_click(self):
        P = _P()
        r = camera.cluster_to_range([(10.0, 100, 100)], P, T_end=30.0)
        self.assertAlmostEqual(r["startTime"], 10.0 - P.pre_roll)

    def test_tail_follows_last_click(self):
        P = _P()
        r = camera.cluster_to_range(
            [(10.0, 100, 100), (12.0, 110, 110)], P, T_end=30.0)
        self.assertAlmostEqual(r["endTime"], 12.0 + P.tail)

    def test_trailing_lone_click_dropped_when_no_room(self):
        # Matches the reference behavior on the reference session's 9th
        # click, which lands PAST T_end. min_room=0.5s: a LONE click
        # within 0.5s of clip end (or beyond it) has no room for a
        # zoom-out arc.
        P = _P()
        self.assertIsNone(
            camera.cluster_to_range([(30.2, 100, 100)], P, T_end=30.0))
        self.assertIsNone(
            camera.cluster_to_range([(29.9, 100, 100)], P, T_end=30.0))

    def test_trailing_multiclick_cluster_holds_to_end(self):
        # A whole multi-click cluster must NOT be discarded just because
        # its LAST click lands near the clip end -- the user clicked right
        # up until they hit stop. Zoom and hold the level to T_end (no
        # zoom-out arc). Regression: the old guard nuked the entire span,
        # so most real recordings barely zoomed at all.
        P = _P()
        r = camera.cluster_to_range(
            [(24.0, 100, 100), (25.0, 110, 110), (29.9, 120, 120)],
            P, T_end=30.0)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r["endTime"], 30.0)   # held to clip end
        self.assertTrue(r["holdOut"])
        self.assertEqual(len(r["clicks"]), 3)

    def test_multiclick_cluster_with_room_still_zooms_out(self):
        # With room after the last click, nothing changes: normal tail +
        # zoom-out arc, not a hold-to-end.
        P = _P()
        r = camera.cluster_to_range(
            [(10.0, 100, 100), (12.0, 110, 110)], P, T_end=30.0)
        self.assertAlmostEqual(r["endTime"], 12.0 + P.tail)
        self.assertFalse(r["holdOut"])


class BuildPathBasics(unittest.TestCase):
    W, H = 1440, 900

    def test_empty_clicks_flat_path(self):
        ft = _frames(2.0)
        out = camera.build_path(ft, self.W, self.H,
                                _empty(), _empty(), _empty(),
                                _empty(), _empty(), _empty())
        # Zoom stays at 1.0 throughout; center stays at source center.
        self.assertTrue(np.allclose(out[:, 2], 1.0))
        self.assertTrue(np.allclose(out[:, 0], self.W / 2.0))
        self.assertTrue(np.allclose(out[:, 1], self.H / 2.0))

    def test_click_produces_zoom_range(self):
        ft = _frames(6.0)
        clicks_t = np.array([3.0])
        clicks_x = np.array([800.0])
        clicks_y = np.array([500.0])
        out = camera.build_path(ft, self.W, self.H,
                                clicks_t, clicks_x, clicks_y,
                                _empty(), _empty(), _empty())
        # Peak zoom near the click instant.
        idx3 = int(3.0 * 60)
        self.assertGreater(out[idx3, 2], 1.5)
        # Settled by clip end.
        self.assertLess(out[-1, 2], 1.15)

    def test_camera_starts_at_source_center(self):
        # Camera model: cursor is decorative; camera begins centered.
        ft = _frames(1.0)
        out = camera.build_path(ft, self.W, self.H,
                                _empty(), _empty(), _empty(),
                                _empty(), _empty(), _empty())
        self.assertEqual(out[0, 0], self.W / 2.0)
        self.assertEqual(out[0, 1], self.H / 2.0)


class ManualZoomsWinOverAutoCluster(unittest.TestCase):
    W, H = 1440, 900

    def test_none_manual_zooms_triggers_auto_cluster(self):
        # CLI-style caller: manual_zooms=None means "auto-cluster please".
        ft = _frames(6.0)
        ct = np.array([3.0])
        cx = np.array([800.0])
        cy = np.array([500.0])
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty(),
                                manual_zooms=None)
        self.assertGreater(out[int(3.0 * 60), 2], 1.5)

    def test_empty_manual_zooms_suppresses_auto_cluster(self):
        # studio_app-style caller: an explicit [] means "no zooms" even
        # if clicks exist -- the caller has already materialized the doc
        # and is now the authoritative source.
        ft = _frames(6.0)
        ct = np.array([3.0])
        cx = np.array([800.0])
        cy = np.array([500.0])
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty(),
                                manual_zooms=[])
        self.assertTrue(np.allclose(out[:, 2], 1.0))

    def test_manual_follow_range_targets_clicks(self):
        # A manual range with x=None acts as follow-click-groups: the
        # runtime pulls in-range clicks from the click stream.
        ft = _frames(8.0)
        ct = np.array([4.0])
        cx = np.array([1200.0])
        cy = np.array([700.0])
        manual = [{"start": 1.0, "end": 6.0, "x": None, "y": None,
                   "level": 2.0}]
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty(),
                                manual_zooms=manual)
        idx = int(4.5 * 60)
        self.assertGreater(out[idx, 2], 1.5)
        self.assertGreater(out[idx, 0], self.W / 2.0 + 50)

    def test_manual_pinned_range_locks_target(self):
        # A manual range with explicit x/y is a "fixed" pin.
        ft = _frames(5.0)
        manual = [{"start": 1.0, "end": 4.0, "x": 300.0, "y": 200.0,
                   "level": 2.0}]
        out = camera.build_path(ft, self.W, self.H,
                                _empty(), _empty(), _empty(),
                                _empty(), _empty(), _empty(),
                                manual_zooms=manual)
        idx = int(2.5 * 60)
        # Snap-to-edges may clamp the target, but it's on the top-left side.
        self.assertLess(out[idx, 0], self.W / 2.0)
        self.assertLess(out[idx, 1], self.H / 2.0)


class OverviewFraming(unittest.TestCase):
    """Clicks spread wider than a max_zoom window must NOT be chased --
    that whip-pans across the screen (motion sickness). They become a
    still, lower-zoom overview locked on the activity bbox center."""
    W, H = 1440, 900

    def _cluster_range(self, clicks, P, win_w=None, win_h=None):
        ranges = camera.plan_zoom(clicks, self.W, self.H, 30.0, P,
                                  win_w=win_w or self.W, win_h=win_h or self.H)
        # the cluster range (skip any settle-only artifacts)
        return next(r for r in ranges if len(r.get("clicks", [])) >= 1)

    def test_scattered_cluster_becomes_still_overview(self):
        P = _P()
        # bbox 700x600 -- wider than the 2x window (720x450) times fill.
        clicks = [(2.0, 100, 100), (2.5, 800, 700),
                  (3.0, 150, 650), (3.5, 780, 120)]
        r = self._cluster_range(clicks, P)
        self.assertEqual(r["type"], "fixed")
        self.assertTrue(r.get("overview"))
        self.assertLess(r["zoom"], 2.0)             # zoomed OUT to fit
        self.assertGreaterEqual(r["zoom"], P.overview_zoom_min - 1e-9)
        self.assertAlmostEqual(r["x"], (100 + 800) / 2.0)  # locked on bbox
        self.assertAlmostEqual(r["y"], (100 + 700) / 2.0)

    def test_tight_cluster_keeps_full_zoom_follow(self):
        P = _P()
        clicks = [(2.0, 700, 450), (2.5, 720, 460), (3.0, 710, 440)]
        r = self._cluster_range(clicks, P)
        self.assertEqual(r["type"], "follow-click-groups")
        self.assertAlmostEqual(r["zoom"], 2.0)
        self.assertFalse(r.get("overview"))

    def test_overview_off_is_bit_exact_full_zoom(self):
        P = _P(overview=False)
        clicks = [(2.0, 100, 100), (2.5, 800, 700), (3.0, 150, 650)]
        r = self._cluster_range(clicks, P)
        self.assertEqual(r["type"], "follow-click-groups")
        self.assertAlmostEqual(r["zoom"], 2.0)

    def test_overview_holds_center_still(self):
        # End to end: the mid-hold of a scattered cluster is rock-still.
        ft = _frames(10.0)
        ct = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
        cx = np.array([100., 800., 150., 780., 120.])
        cy = np.array([100., 700., 650., 120., 680.])
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty())
        lo, hi = int(4.0 * 60), int(6.0 * 60)
        self.assertLess(np.ptp(out[lo:hi, 0]), 25.0)
        self.assertLess(np.ptp(out[lo:hi, 1]), 25.0)
        self.assertLess(out[int(5.0 * 60), 2], 2.0)   # reduced zoom

    def test_overview_applies_to_manual_follow_too(self):
        # A materialized/manual follow zoom over spread clicks also
        # overviews (the web/MCP path renders auto-zooms this way).
        P = _P()
        manual = camera._normalize_manual(
            [{"start": 1.0, "end": 8.0, "x": None, "y": None, "level": 2.0}], P)
        # inject spread clicks so the range pools a wide bbox
        clicks = [(2.0, 100, 100), (3.0, 800, 700), (4.0, 150, 650)]
        ranges = camera.plan_zoom(clicks, self.W, self.H, 30.0, P,
                                  manual=[{"start": 1.0, "end": 8.0,
                                           "x": None, "y": None, "level": 2.0}],
                                  auto_cluster=False,
                                  win_w=self.W, win_h=self.H)
        r = ranges[0]
        self.assertEqual(r["type"], "fixed")
        self.assertLess(r["zoom"], 2.0)


class SnapToEdges(unittest.TestCase):
    W, H = 1440, 900

    def test_click_near_left_edge_snaps_camera_to_left(self):
        ft = _frames(6.0)
        ct = np.array([3.0])
        cx = np.array([50.0])
        cy = np.array([450.0])
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty())
        # Camera window at z=2 is 720 px wide; half-width=360, so
        # lo_x=360 is the leftmost the center can sit.
        idx = int(3.2 * 60)
        self.assertAlmostEqual(out[idx, 0], 360.0, delta=15.0)


class Spring(unittest.TestCase):
    def test_settles_to_target(self):
        x, v = 0.0, 0.0
        for _ in range(120):
            x, v = camera._damped_step(x, v, 1.0, 9.43, 0.94, 1.0 / 60)
        self.assertAlmostEqual(x, 1.0, places=3)

    def test_fps_independent_endpoint(self):
        def run(dt, n):
            x, v = 0.0, 0.0
            for _ in range(n):
                x, v = camera._damped_step(x, v, 1.0, 9.43, 0.94, dt)
            return x
        a = run(1.0 / 60, 60)
        b = run(1.0 / 120, 120)
        self.assertAlmostEqual(a, b, places=6)


class AlwaysZoomed(unittest.TestCase):
    W, H = 1440, 900

    def test_hold_last_zoom_after_final_range(self):
        ft = _frames(6.0)
        ct = np.array([2.0])
        cx = np.array([720.0])
        cy = np.array([450.0])
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty(),
                                params={"always_zoomed": True})
        # By clip end the zoom has NOT decayed back to 1.
        self.assertGreater(out[-1, 2], 1.5)


class Suppression(unittest.TestCase):
    W, H = 1440, 900

    def test_suppression_drops_click(self):
        ft = _frames(6.0)
        ct = np.array([3.0])
        cx = np.array([720.0])
        cy = np.array([450.0])
        out = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                _empty(), _empty(), _empty(),
                                suppressed_ranges=[{"start": 2.5, "end": 3.5}])
        self.assertTrue(np.allclose(out[:, 2], 1.0))


class LegacyKwargsAreAcceptedAndIgnored(unittest.TestCase):
    """The new model doesn't consume typing/scroll/anchor signals.
    Accepting them silently keeps render.py's existing call site working."""

    W, H = 1440, 900

    def test_extra_kwargs_do_not_change_output(self):
        ft = _frames(3.0)
        ct = np.array([1.5])
        cx = np.array([720.0])
        cy = np.array([450.0])
        base = camera.build_path(ft, self.W, self.H, ct, cx, cy,
                                 _empty(), _empty(), _empty())
        with_extras = camera.build_path(
            ft, self.W, self.H, ct, cx, cy,
            _empty(), _empty(), _empty(),
            keys_t=np.array([0.5, 0.6, 0.7, 0.8]),
            ups_t=np.array([1.55]),
            scrolls_t=np.array([2.0, 2.1, 2.2]),
            scrolls_x=np.array([100.0, 110.0, 120.0]),
            scrolls_y=np.array([200.0, 210.0, 220.0]),
            typing_anchors=[{"start": 0.5, "end": 0.9,
                             "x": 300, "y": 400}])
        self.assertTrue(np.array_equal(base, with_extras))

    def test_typing_bursts_shim_returns_empty(self):
        self.assertEqual(
            camera.typing_bursts(np.array([1.0, 1.1, 1.2, 1.3])), [])


if __name__ == "__main__":
    unittest.main()
