"""Unit tests for the event-derived beat sheet (autocine/beats.py).

Pure functions over synthetic event arrays -- no ffmpeg, no video, no macOS
permissions, so the whole file runs anywhere the suite does.

The load-bearing test in here is `ClusterContractTests`: `beats` is the THIRD
consumer of the click-clustering contract (`camera` plans with it,
`edits.auto_zoom_proposals` materializes it into edits.json, and a beat
reports it), and docs/architecture.md already flags that seam as one this project has
been bitten by. Adding a consumer without pinning it against the others is
how the copies drift.
"""

import unittest

import numpy as np

from autocine import beats
from autocine import camera
from autocine import edits as ed


def _ev(**kw):
    """A `geometry.load_events`-shaped dict, empty except for what's given."""
    base = {}
    for k in ("clicks_t", "clicks_x", "clicks_y", "moves_t", "moves_x",
              "moves_y", "ups_t", "keys_t", "scrolls_t", "scrolls_x",
              "scrolls_y", "windows_t"):
        base[k] = np.array([], dtype=float)
    base["windows_rect"] = np.zeros((0, 4), dtype=float)
    base["windows_id"] = np.array([], dtype=int)
    base["windows_z"] = np.array([], dtype=int)
    for k, v in kw.items():
        base[k] = np.asarray(v, dtype=float) if k != "windows_rect" else \
            np.asarray(v, dtype=float).reshape(-1, 4)
    for k in ("windows_id", "windows_z"):
        if k in kw:
            base[k] = np.asarray(kw[k], dtype=int)
    return base


def _clicks(times, x=100.0, y=100.0):
    t = list(times)
    return {"clicks_t": np.asarray(t, dtype=float),
            "clicks_x": np.full(len(t), x, dtype=float),
            "clicks_y": np.full(len(t), y, dtype=float)}


def _track(samples, period=1.0):
    """samples: list of (t, window_id, x, y, w, h, z) -> ev window arrays."""
    return {
        "windows_t": np.asarray([s[0] for s in samples], dtype=float),
        "windows_id": np.asarray([s[1] for s in samples], dtype=int),
        "windows_rect": np.asarray([[s[2], s[3], s[4], s[5]]
                                    for s in samples], dtype=float),
        "windows_z": np.asarray([s[6] for s in samples], dtype=int),
    }


def _heartbeat(window_id, t0, t1, rect, z, period=1.0):
    """Geometry samples for one window over [t0, t1] at the real cadence."""
    out = []
    t = t0
    while t <= t1 + 1e-9:
        out.append((t, window_id, rect[0], rect[1], rect[2], rect[3], z))
        t += period
    return out


def _kinds(sheet, kind):
    return [b for b in sheet["beats"] if b["kind"] == kind]


class ClusterContractTests(unittest.TestCase):
    """beats/camera/edits must agree on where a zoom starts and ends."""

    CASES = [
        ([1.0], 30.0),                               # lone click, room to spare
        ([1.0, 2.0, 3.0], 30.0),                     # one cluster
        ([1.0, 9.0, 20.0], 30.0),                    # three clusters
        ([1.0, 29.9], 30.0),                         # lone TRAILING click
        ([1.0, 29.5, 29.9], 30.0),                   # trailing MULTI-click
        ([0.1], 30.0),                               # start clamps at 0
        ([5.0, 5.2, 5.4, 12.0, 12.1], 30.0),
    ]

    def test_beats_zoom_spans_match_auto_zoom_proposals(self):
        for times, dur in self.CASES:
            sheet = beats.beat_sheet(_ev(**_clicks(times)), dur)
            mine = [(round(b["zoom_proposal"]["start"], 6),
                     round(b["zoom_proposal"]["end"], 6))
                    for b in _kinds(sheet, "clicks") if "zoom_proposal" in b]
            theirs = [(round(p["start"], 6), round(p["end"], 6))
                      for p in ed.auto_zoom_proposals(times, dur)]
            self.assertEqual(mine, theirs,
                             "drifted from edits.auto_zoom_proposals on %r"
                             % (times,))

    def test_beats_zoom_spans_match_the_planner(self):
        for times, dur in self.CASES:
            P = camera.build_params(2.0, None)
            clicks = [(t, 100.0, 100.0) for t in times]
            planned = []
            for cl in camera.cluster_clicks(clicks, P.chain_gap):
                r = camera.cluster_to_range(cl, P, dur)
                if r is not None:
                    planned.append((round(r["startTime"], 6),
                                    round(r["endTime"], 6)))
            sheet = beats.beat_sheet(_ev(**_clicks(times)), dur)
            mine = [(round(b["zoom_proposal"]["start"], 6),
                     round(b["zoom_proposal"]["end"], 6))
                    for b in _kinds(sheet, "clicks") if "zoom_proposal" in b]
            self.assertEqual(mine, planned)

    def test_lone_trailing_click_is_a_beat_with_no_zoom(self):
        """The click HAPPENED even though the camera won't zoom on it.

        Dropping the beat too would tell an agent nothing occurred there.
        """
        sheet = beats.beat_sheet(_ev(**_clicks([1.0, 29.9])), 30.0)
        cl = _kinds(sheet, "clicks")
        self.assertEqual(len(cl), 2)
        self.assertIn("zoom_proposal", cl[0])
        self.assertNotIn("zoom_proposal", cl[1])

    def test_trailing_multi_click_holds_out(self):
        sheet = beats.beat_sheet(_ev(**_clicks([29.5, 29.9])), 30.0)
        cl = _kinds(sheet, "clicks")
        self.assertEqual(len(cl), 1)
        self.assertTrue(cl[0]["zoom_proposal"]["hold_out"])
        self.assertAlmostEqual(cl[0]["zoom_proposal"]["end"], 30.0, places=6)

    def test_zoom_speed_reaches_the_reported_span(self):
        """A render option that moves the camera must move the beat too."""
        ev = _ev(**_clicks([10.0]))
        normal = beats.beat_sheet(ev, 30.0, params={"zoom_speed": "normal"})
        slow = beats.beat_sheet(ev, 30.0, params={"zoom_speed": "slow"})
        self.assertAlmostEqual(
            _kinds(normal, "clicks")[0]["zoom_proposal"]["start"], 7.5,
            places=6)
        self.assertAlmostEqual(
            _kinds(slow, "clicks")[0]["zoom_proposal"]["start"], 6.25,
            places=6)


class MaterializedZoomTests(unittest.TestCase):
    """`zoom` must be what RENDERS, not what the planner would propose.

    Every surface except a bare CLI render plans zoom ranges only from the
    materialized edits doc -- `camera.build_path` sets
    `auto_cluster = manual_zooms is None` and the MCP/web resolvers always
    pass a list. So on a session whose auto-zooms were never materialized,
    reporting the proposal as the plan states the exact opposite of what a
    render will do.
    """

    def test_no_materialized_zooms_means_no_zoom_and_a_warning(self):
        sheet = beats.beat_sheet(_ev(**_clicks([5.0])), 30.0, zooms=[])
        b = _kinds(sheet, "clicks")[0]
        self.assertNotIn("zoom", b)
        self.assertIn("zoom_proposal", b)
        self.assertTrue(any("NOTHING will zoom" in n for n in sheet["notes"]))

    def test_materialized_zoom_is_reported_as_the_real_one(self):
        zooms = [{"id": "z1", "start": 2.5, "end": 5.5, "level": 2.0}]
        b = _kinds(beats.beat_sheet(_ev(**_clicks([5.0])), 30.0, zooms=zooms),
                   "clicks")[0]
        # `level` rides along because "that zoom was too aggressive" is a
        # statement about it -- an agent has to be able to read the current
        # one, and read the changed one back after `adjust_zoom`.
        self.assertEqual(b["zoom"], {"start": 2.5, "end": 5.5, "level": 2.0,
                                     "ids": ["z1"]})
        self.assertNotIn("zoom_proposal", b)

    def test_a_deleted_zoom_stops_being_reported(self):
        """remove_zoom must be visible here, or an agent re-reads its own
        deleted edit back as fact."""
        ev = _ev(**_clicks([5.0, 20.0]))
        both = [{"id": "z1", "start": 2.5, "end": 5.5, "level": 2.0},
                {"id": "z2", "start": 17.5, "end": 20.5, "level": 2.0}]
        full = _kinds(beats.beat_sheet(ev, 30.0, zooms=both), "clicks")
        self.assertTrue(all("zoom" in b for b in full))
        after = _kinds(beats.beat_sheet(ev, 30.0, zooms=both[:1]), "clicks")
        self.assertIn("zoom", after[0])
        self.assertNotIn("zoom", after[1])
        self.assertIn("zoom_proposal", after[1])

    def test_zooms_none_means_unknown_and_reports_a_proposal(self):
        """The CLI path auto-clusters, so 'no edits doc' is not 'no zooms'."""
        sheet = beats.beat_sheet(_ev(**_clicks([5.0])), 30.0, zooms=None)
        self.assertIn("zoom_proposal", _kinds(sheet, "clicks")[0])
        self.assertFalse(any("NOTHING will zoom" in n for n in sheet["notes"]))


class PresenceIntervalTests(unittest.TestCase):
    """A window that is minimised/hidden stops being clicked on.

    devices._filter_windows drops any window without kCGWindowIsOnscreen --
    minimise, Cmd-H and a Space switch all produce an interior hole in the
    track, and the window returns under the SAME id.
    """

    def _minimised_midway(self):
        s = _heartbeat(1, 0.0, 10.0, (0, 0, 1000, 800), 1)
        s += _heartbeat(1, 10.5, 25.0, (0, 0, 1000, 800), 0)
        s += _heartbeat(1, 25.5, 40.0, (0, 0, 1000, 800), 1)
        s += _heartbeat(2, 0.0, 10.0, (200, 200, 300, 300), 0)
        s += _heartbeat(2, 25.0, 40.0, (200, 200, 300, 300), 0)
        return _track(sorted(s))

    def test_click_during_the_gap_goes_to_the_visible_window(self):
        ev = _ev(**dict(self._minimised_midway(),
                        **_clicks([15.0], x=300, y=300)))
        b = _kinds(beats.beat_sheet(ev, 40.0), "clicks")[0]
        self.assertEqual(b["window_id"], 1)

    def test_scroll_during_the_gap_goes_to_the_visible_window(self):
        ev = _ev(**dict(self._minimised_midway(),
                        scrolls_t=[16.0, 16.2, 16.4],
                        scrolls_x=[300.0] * 3, scrolls_y=[300.0] * 3))
        b = _kinds(beats.beat_sheet(ev, 40.0), "scroll")[0]
        self.assertEqual(b["window_id"], 1)

    def test_the_gap_is_visible_as_close_then_open(self):
        sheet = beats.beat_sheet(_ev(**self._minimised_midway()), 40.0)
        self.assertEqual([round(b["t"]) for b in _kinds(sheet, "close")
                          if b["window_id"] == 2], [10])
        self.assertEqual([round(b["t"]) for b in _kinds(sheet, "open")
                          if b["window_id"] == 2], [25])

    def test_a_normal_heartbeat_is_not_split(self):
        """The real cadence must never look like a disappearance."""
        s = _heartbeat(1, 0.0, 40.0, (0, 0, 800, 600), 0, period=1.079)
        sheet = beats.beat_sheet(_ev(**_track(s)), 40.0)
        self.assertEqual(_kinds(sheet, "open"), [])
        self.assertEqual(_kinds(sheet, "close"), [])


class AttributionTests(unittest.TestCase):
    """Which window a click landed in, from geometry + z alone."""

    def _two_windows(self):
        # `back` spans the screen; `front` is a smaller window on top of it.
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 1000, 800), 1)
        s += _heartbeat(2, 0.0, 30.0, (200, 200, 300, 300), 0)
        return _track(sorted(s))

    def test_click_in_overlap_goes_to_the_frontmost(self):
        ev = _ev(**dict(self._two_windows(), **_clicks([5.0], x=300, y=300)))
        b = _kinds(beats.beat_sheet(ev, 30.0), "clicks")[0]
        self.assertEqual(b["window_id"], 2)

    def test_click_outside_the_front_window_goes_to_the_one_below(self):
        ev = _ev(**dict(self._two_windows(), **_clicks([5.0], x=900, y=700)))
        b = _kinds(beats.beat_sheet(ev, 30.0), "clicks")[0]
        self.assertEqual(b["window_id"], 1)

    def test_click_in_no_window_carries_no_window_id(self):
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 100, 100), 0)
        ev = _ev(**dict(_track(s), **_clicks([5.0], x=900, y=700)))
        b = _kinds(beats.beat_sheet(ev, 30.0), "clicks")[0]
        self.assertNotIn("window_id", b)

    def test_click_between_heartbeats_still_attributes(self):
        """Regression: the poller heartbeats each window about once a second,
        so a fixed sub-cadence staleness cut silently dropped every window
        for any click landing late in a heartbeat interval."""
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 1000, 800), 0, period=1.05)
        # 0.999s after the sample at t=5.25, i.e. just before the next one.
        ev = _ev(**dict(_track(s), **_clicks([6.249], x=500, y=400)))
        b = _kinds(beats.beat_sheet(ev, 30.0), "clicks")[0]
        self.assertEqual(b["window_id"], 1)

    def test_closed_window_is_not_attributed_after_it_closes(self):
        s = _heartbeat(1, 0.0, 10.0, (0, 0, 1000, 800), 0)
        s += _heartbeat(2, 0.0, 30.0, (0, 0, 1000, 800), 1)
        ev = _ev(**dict(_track(sorted(s)), **_clicks([20.0], x=500, y=400)))
        b = _kinds(beats.beat_sheet(ev, 30.0), "clicks")[0]
        self.assertEqual(b["window_id"], 2)

    def test_mixed_cluster_reports_the_breakdown(self):
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 500, 800), 0)
        s += _heartbeat(2, 0.0, 30.0, (500, 0, 500, 800), 0)
        ev = _ev(**dict(
            _track(sorted(s)),
            clicks_t=[5.0, 5.5, 6.0],
            clicks_x=[100.0, 700.0, 120.0],
            clicks_y=[100.0, 100.0, 100.0]))
        b = _kinds(beats.beat_sheet(ev, 30.0), "clicks")[0]
        self.assertEqual(b["window_id"], 1)          # 2 of 3 clicks
        self.assertEqual(b["windows"], {"1": 2, "2": 1})

    def test_no_z_track_falls_back_to_smallest_and_says_so(self):
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 1000, 800), -1)
        s += _heartbeat(2, 0.0, 30.0, (200, 200, 300, 300), -1)
        ev = _ev(**dict(_track(sorted(s)), **_clicks([5.0], x=300, y=300)))
        sheet = beats.beat_sheet(ev, 30.0)
        self.assertEqual(_kinds(sheet, "clicks")[0]["window_id"], 2)
        self.assertEqual(_kinds(sheet, "front"), [])
        self.assertTrue(any("front-to-back" in n for n in sheet["notes"]))


class WindowBeatTests(unittest.TestCase):

    def test_front_open_and_close(self):
        # window 1 sits behind until window 2 goes away at 20s, then leads.
        s = _heartbeat(1, 0.0, 20.0, (0, 0, 800, 600), 1)
        s += _heartbeat(1, 21.0, 30.0, (0, 0, 800, 600), 0)
        s += _heartbeat(2, 10.0, 20.0, (0, 0, 400, 300), 0)
        sheet = beats.beat_sheet(_ev(**_track(sorted(s))), 30.0)
        self.assertEqual([(b["window_id"], round(b["t"], 1))
                          for b in _kinds(sheet, "open")], [(2, 10.0)])
        self.assertEqual([(b["window_id"], round(b["t"], 1))
                          for b in _kinds(sheet, "close")], [(2, 20.0)])
        self.assertEqual([(b["window_id"], round(b["t"], 1))
                          for b in _kinds(sheet, "front")], [(1, 21.0)])

    def test_window_open_at_record_start_is_not_an_open_beat(self):
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 800, 600), 0)
        sheet = beats.beat_sheet(_ev(**_track(s)), 30.0)
        self.assertEqual(_kinds(sheet, "open"), [])
        self.assertEqual(_kinds(sheet, "close"), [])

    def test_initial_z_is_state_not_a_front_beat(self):
        """A window that is frontmost from its first sample never 'came' to
        the front; emitting a beat there would put a scene cut at t=0 in
        every single recording."""
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 800, 600), 0)
        self.assertEqual(_kinds(beats.beat_sheet(_ev(**_track(s)), 30.0),
                                "front"), [])

    def test_no_geometry_track_says_so_and_still_produces_beats(self):
        sheet = beats.beat_sheet(_ev(**_clicks([1.0, 2.0])), 30.0)
        self.assertTrue(any("no window geometry track" in n
                            for n in sheet["notes"]))
        self.assertEqual(len(_kinds(sheet, "clicks")), 1)
        self.assertNotIn("window_id", _kinds(sheet, "clicks")[0])


class SpanBeatTests(unittest.TestCase):

    def test_scroll_runs_split_on_a_gap(self):
        ts = [1.0, 1.2, 1.4, 1.6, 8.0, 8.2]
        ev = _ev(scrolls_t=ts, scrolls_x=[10.0] * 6, scrolls_y=[10.0] * 6)
        runs = _kinds(beats.beat_sheet(ev, 30.0), "scroll")
        self.assertEqual([(r["start"], r["end"], r["n"]) for r in runs],
                         [(1.0, 1.6, 4), (8.0, 8.2, 2)])

    def test_typing_bursts_carry_no_window_id(self):
        """Key lines record a tick and nothing else -- their x/y is padded-in
        last-cursor position. Attributing typing to a window by position would
        be a confident guess about the thing the log refuses to record."""
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 1000, 800), 0)
        ev = _ev(keys_t=[5.0, 5.1, 5.2], **_track(s))
        b = _kinds(beats.beat_sheet(ev, 30.0), "typing")
        self.assertEqual(len(b), 1)
        self.assertNotIn("window_id", b[0])

    def test_idle_spans_are_the_holes_between_activity(self):
        ev = _ev(**_clicks([1.0, 25.0]))
        idle = _kinds(beats.beat_sheet(ev, 30.0), "idle")
        self.assertTrue(idle)
        self.assertTrue(all(i["end"] - i["start"] >= beats.MIN_IDLE
                            for i in idle))
        # The hole between the two clicks is found.
        self.assertTrue(any(i["start"] > 1.0 and i["end"] < 25.0
                            for i in idle))


class UnitAndClockTests(unittest.TestCase):

    def test_bbox_is_source_pixels(self):
        ev = _ev(clicks_t=[5.0, 6.0], clicks_x=[100.0, 200.0],
                 clicks_y=[50.0, 150.0])
        b = _kinds(beats.beat_sheet(ev, 30.0, scale_x=2.0, scale_y=2.0),
                   "clicks")[0]
        self.assertEqual(b["bbox"], [200.0, 100.0, 200.0, 200.0])

    def test_bbox_subtracts_the_capture_crop_origin(self):
        ev = _ev(clicks_t=[5.0], clicks_x=[100.0], clicks_y=[50.0])
        b = _kinds(beats.beat_sheet(ev, 30.0, scale_x=2.0, scale_y=2.0,
                                    origin_x=50.0, origin_y=20.0),
                   "clicks")[0]
        self.assertEqual(b["bbox"][:2], [150.0, 80.0])

    def test_times_are_media_seconds(self):
        ev = _ev(**_clicks([1005.0, 1006.0]))
        b = _kinds(beats.beat_sheet(ev, 30.0, t0=1000.0), "clicks")[0]
        self.assertAlmostEqual(b["start"], 5.0, places=6)
        self.assertAlmostEqual(b["end"], 6.0, places=6)

    def test_events_outside_the_clip_are_dropped(self):
        ev = _ev(**_clicks([-5.0, 5.0, 999.0]))
        sheet = beats.beat_sheet(ev, 30.0)
        for b in sheet["beats"]:
            t = b.get("start", b.get("t"))
            self.assertGreaterEqual(t, 0.0)
            self.assertLessEqual(t, 30.0)

    def test_beats_are_time_ordered(self):
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 800, 600), 1)
        s += _heartbeat(2, 12.0, 30.0, (0, 0, 400, 300), 0)
        ev = _ev(**dict(_track(sorted(s)),
                        **dict(_clicks([2.0, 20.0]),
                               scrolls_t=np.asarray([8.0, 8.2]),
                               scrolls_x=np.asarray([5.0, 5.0]),
                               scrolls_y=np.asarray([5.0, 5.0]))))
        sheet = beats.beat_sheet(ev, 30.0)
        ts = [b.get("start", b.get("t")) for b in sheet["beats"]]
        self.assertEqual(ts, sorted(ts))

    def test_empty_session_is_empty_not_an_error(self):
        sheet = beats.beat_sheet(_ev(), 0.0)
        self.assertEqual(sheet["beats"], [])
        self.assertEqual(sheet["truncated"], 0)


class TruncationTests(unittest.TestCase):

    def _long_take(self):
        """400s, 12 windows, a front change and a scroll run every 30s."""
        s, scr = [], []
        for w in range(1, 13):
            s += _heartbeat(w, 0.0, 400.0, (0, 0, 100 * w, 800), 1)
        for i in range(13):
            t = 30.0 * i + 30.0
            s += [(t, 1 + (i % 12), 0, 0, 100, 800, 0)]
            scr += [t + 1.0, t + 2.0, t + 3.0]
        ev = _ev(**dict(_track(sorted(s)),
                        scrolls_t=scr,
                        scrolls_x=[10.0] * len(scr),
                        scrolls_y=[10.0] * len(scr)))
        return ev

    def test_cap_reports_what_it_dropped(self):
        """No silent caps: a long take must not read as a quiet one."""
        times = [float(i) * 10.0 for i in range(60)]   # 60 lone clusters
        sheet = beats.beat_sheet(_ev(**_clicks(times)), 700.0, max_beats=10)
        self.assertEqual(len(sheet["beats"]), 10)
        self.assertGreater(sheet["truncated"], 0)
        self.assertTrue(any("omitted" in n for n in sheet["notes"]))

    def test_kept_beats_stay_time_ordered(self):
        times = [float(i) * 10.0 for i in range(60)]
        sheet = beats.beat_sheet(_ev(**_clicks(times)), 700.0, max_beats=10)
        ts = [b.get("start", b.get("t")) for b in sheet["beats"]]
        self.assertEqual(ts, sorted(ts))

    def test_under_the_cap_nothing_is_dropped(self):
        sheet = beats.beat_sheet(_ev(**_clicks([1.0, 20.0])), 30.0,
                                 max_beats=100)
        self.assertEqual(sheet["truncated"], 0)
        self.assertFalse(any("omitted" in n for n in sheet["notes"]))

    def test_truncation_covers_the_whole_take_not_just_the_start(self):
        """Ranking by span length dropped every instant beat and every
        zero-length span, and dropped them from the END: the tail of a long
        recording read as 'they stopped switching windows'."""
        ev = self._long_take()
        full = beats.beat_sheet(ev, 400.0, max_beats=0)
        capped = beats.beat_sheet(ev, 400.0, max_beats=20)
        last_full = max(b.get("start", b.get("t")) for b in full["beats"])
        last_capped = max(b.get("start", b.get("t")) for b in capped["beats"])
        # The kept beats must still reach the end of the take.
        self.assertGreater(last_capped, last_full * 0.8)

    def test_truncation_keeps_more_than_one_kind(self):
        ev = self._long_take()
        capped = beats.beat_sheet(ev, 400.0, max_beats=20)
        kinds = set(b["kind"] for b in capped["beats"])
        self.assertIn("front", kinds,
                      "window changes were dropped wholesale: %r" % (kinds,))

    def test_cap_is_actually_filled(self):
        times = [float(i) * 10.0 for i in range(60)]
        sheet = beats.beat_sheet(_ev(**_clicks(times)), 700.0, max_beats=25)
        self.assertEqual(len(sheet["beats"]), 25)

    def test_the_note_does_not_promise_an_option_that_exists_nowhere(self):
        """The first version told the caller to 'ask for a narrower span'.
        There is no span argument on beat_sheet or on the MCP tool."""
        times = [float(i) * 10.0 for i in range(60)]
        sheet = beats.beat_sheet(_ev(**_clicks(times)), 700.0, max_beats=10)
        note = " ".join(sheet["notes"])
        self.assertIn("max_beats", note)
        self.assertNotIn("narrower span", note)


class ScrollRateTests(unittest.TestCase):

    def test_rate_counts_intervals_not_ticks(self):
        """n ticks span n-1 intervals; using n inflates a short run up to 2x."""
        ev = _ev(scrolls_t=[1.0, 1.1], scrolls_x=[5.0] * 2,
                 scrolls_y=[5.0] * 2)
        b = _kinds(beats.beat_sheet(ev, 30.0), "scroll")[0]
        self.assertAlmostEqual(b["rate"], 10.0, places=3)

    def test_single_tick_run_reports_no_rate(self):
        ev = _ev(scrolls_t=[1.0], scrolls_x=[5.0], scrolls_y=[5.0])
        b = _kinds(beats.beat_sheet(ev, 30.0), "scroll")[0]
        self.assertNotIn("rate", b)
        self.assertEqual(b["n"], 1)


class ToSrcTests(unittest.TestCase):

    def test_bbox_uses_the_supplied_mapper(self):
        """On a tracked --capture-window session the crop origin MOVES, so a
        static origin puts the bbox off by the window's displacement."""
        ev = _ev(clicks_t=[5.0], clicks_x=[100.0], clicks_y=[50.0])

        def to_src(t, ax, ay):
            return ax * 2.0 - 140.0, ay * 2.0 - 40.0

        b = _kinds(beats.beat_sheet(ev, 30.0, scale_x=2.0, scale_y=2.0,
                                    origin_x=0.0, origin_y=0.0,
                                    to_src=to_src), "clicks")[0]
        self.assertEqual(b["bbox"][:2], [60.0, 60.0])

    def test_no_mapper_falls_back_to_the_static_origin(self):
        ev = _ev(clicks_t=[5.0], clicks_x=[100.0], clicks_y=[50.0])
        b = _kinds(beats.beat_sheet(ev, 30.0, scale_x=2.0, scale_y=2.0,
                                    origin_x=140.0, origin_y=40.0),
                   "clicks")[0]
        self.assertEqual(b["bbox"][:2], [60.0, 60.0])


class PrivacyTests(unittest.TestCase):

    def test_no_beat_ever_carries_a_name_or_title(self):
        """The event log deliberately records no app name or window title;
        this read model must not invent a place to put one.

        Checks VALUES as well as keys, and recurses: a string anywhere in the
        payload that isn't a beat kind, a note, or a zoom id is the shape a
        leak would take. (The first version compared top-level keys against
        five exact strings, which could not have caught one.)
        """
        s = _heartbeat(1, 0.0, 30.0, (0, 0, 800, 600), 0)
        s += _heartbeat(2, 10.0, 20.0, (0, 0, 400, 300), 0)
        ev = _ev(**dict(_track(sorted(s)), **_clicks([5.0, 15.0])))
        sheet = beats.beat_sheet(
            ev, 30.0, zooms=[{"id": "zoom-1", "start": 2.5, "end": 5.5}])
        allowed_kinds = set(beats._KINDS_BY_PRIORITY)
        banned_substrings = ("app", "title", "name", "bundle", "owner", "path")

        def _walk(node, path):
            if isinstance(node, dict):
                for k, v in node.items():
                    self.assertFalse(
                        any(b in str(k).lower() for b in banned_substrings),
                        "suspicious key %r at %s" % (k, path))
                    _walk(v, path + "." + str(k))
            elif isinstance(node, (list, tuple)):
                for i, v in enumerate(node):
                    _walk(v, "%s[%d]" % (path, i))
            elif isinstance(node, str):
                self.assertTrue(
                    node in allowed_kinds or node.startswith("zoom-"),
                    "unexplained string %r at %s -- every string in a beat "
                    "must be a kind or a zoom id, or identity has leaked in"
                    % (node, path))

        for b in sheet["beats"]:
            _walk(b, "beat")


class CutsStayOutOfTheClusterContract(unittest.TestCase):
    """THE RULING for cuts (ripple delete) vs the trailing-cluster contract:
    every read model stays on the SOURCE clock. Cuts are never folded into
    the duration or the click times handed to any of the three hand-kept
    contract holders -- the camera warps events itself at render time.

    Pinned at the signature level: the day one of the three grows a `cuts`
    parameter, this fails and forces the change to be made to ALL of them
    together (the extend-not-parameterize discipline both prior
    time-features followed), instead of one copy silently drifting.
    """

    def test_no_contract_holder_takes_cuts(self):
        import inspect
        for fn in (camera.cluster_to_range, camera.cluster_clicks,
                   ed.auto_zoom_proposals, beats.beat_sheet):
            params = inspect.signature(fn).parameters
            self.assertNotIn(
                "cuts", params,
                "%s grew a 'cuts' parameter -- the trailing-cluster "
                "contract must be extended across camera/edits/beats "
                "together, and this pin updated deliberately"
                % getattr(fn, "__qualname__", fn))


if __name__ == "__main__":
    unittest.main()
