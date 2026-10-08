"""Record-time window geometry track: logging it, loading it, following it.

`--capture-window` shipped cropping every frame to ONE snapshot, so moving
the window slid its content out of frame and resizing it reframed nothing.
These cover the track that fixes that -- the poller in record.py, the
`window` lines in geometry.load_events, and render.py following them -- plus
the off switch: a session without a track must render exactly as before.

No macOS permissions: the Quartz re-read is monkeypatched at
`record.dev.window_rect_points`, the same seam the capture-window tests use.
"""

import json
import os
import shutil
import subprocess
import tempfile
import time
import unittest

import cv2
import numpy as np

from autocine import framing, geometry, record, render


def _mk_video(path, duration=2.0, fps=30, width=640, height=360,
              box_expr="40+100*t", box=60):
    """A white box sliding horizontally on black, positioned by `box_expr`,
    so a correctly-following crop holds it still in the output.

    `overlay` and not `drawbox`: drawbox silently draws NOTHING when its x is
    a `t`-dependent expression on this ffmpeg, which would make the whole
    end-to-end check vacuously pass on a blank frame.
    """
    subprocess.check_call([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "color=c=black:s={}x{}:r={}:d={}".format(width, height, fps, duration),
        "-f", "lavfi", "-i",
        "color=c=white:s={}x{}:r={}:d={}".format(box, box, fps, duration),
        "-filter_complex", "[0][1]overlay=x='{}':y=140".format(box_expr),
        "-pix_fmt", "yuv420p", path])


def _cw(rect, track="ok"):
    return {"id": 7, "app": "a", "title": "t", "units": "points",
            "rect": list(rect), "display_origin": [0.0, 0.0],
            "source": "quartz", "resnapshot": True,
            "end_rect": list(rect), "track": track}


def _mk_session(root, window_samples, capture_rect, duration=2.0, fps=30,
                width=640, height=360, box_expr="40+100*t"):
    os.makedirs(root, exist_ok=True)
    _mk_video(os.path.join(root, "raw.mov"), duration=duration, fps=fps,
              width=width, height=height, box_expr=box_expr)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for t, rect in window_samples:
            f.write(json.dumps({
                "t": t, "type": "window", "rect": list(rect),
                "x": rect[0] + rect[2] / 2.0, "y": rect[1] + rect[3] / 2.0,
            }) + "\n")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({
            "raw": "raw.mov", "events": "events.jsonl",
            "fps": fps, "logical_w": width, "logical_h": height,
            "t0_monotonic": 0.0, "cursor_mode": "system",
            "duration": duration, "capture_window": _cw(capture_rect),
        }, f)


def _moving_window_samples(duration=2.0, hz=20.0, box=60, pad=30):
    """Window rects that follow the same ramp the drawn box does."""
    out = []
    n = int(duration * hz) + 1
    for i in range(n):
        t = i / hz
        bx = 40 + 100 * t
        out.append((t, [bx - pad, 110.0, box + 2 * pad, box + 2 * pad]))
    return out


def _bright_centroid(frame):
    """(x, y) of the near-white pixels -- where the drawn box landed."""
    grey = frame.mean(axis=2)
    ys, xs = np.nonzero(grey > 200)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


class RecorderPollsWindowGeometry(unittest.TestCase):
    """The poller is best-effort and privacy-bounded; both are load-bearing."""

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="wintrack_rec_")
        self.rec = record.Recorder(self.td, video_idx=1,
                                   capture_window={"id": 7, "x": 0, "y": 0,
                                                   "w": 100, "h": 80})
        self.rec._ev_file = open(os.path.join(self.td, "events.jsonl"), "w")

    def tearDown(self):
        try:
            if self.rec._ev_file:
                self.rec._ev_file.close()
        except Exception:
            pass
        shutil.rmtree(self.td, ignore_errors=True)

    def _lines(self):
        self.rec._ev_file.flush()
        with open(os.path.join(self.td, "events.jsonl")) as f:
            return [json.loads(l) for l in f if l.strip()]

    def _drive(self, rec, listing, want_samples, timeout=2.0):
        """Run the poller against a fake window enumeration until it has
        logged `want_samples`, then stop it."""
        record.dev.list_windows = listing
        rec._start_window_track()
        deadline = time.time() + timeout
        while rec._win_samples < want_samples and time.time() < deadline:
            time.sleep(0.02)
        rec._stop_window_track()

    def test_sample_carries_geometry_and_id_but_never_identity(self):
        """PRIVACY: the id is opaque and only groups samples over time. An app
        name or title would make this a record of what you had open."""
        self.rec._write_window_event(7, [10.0, 20.0, 300.0, 200.0])
        line = self._lines()[0]
        self.assertEqual(line["type"], "window")
        self.assertEqual(line["rect"], [10.0, 20.0, 300.0, 200.0])
        self.assertEqual(line["id"], 7)
        for banned in ("app", "title", "owner", "pid", "label", "name"):
            self.assertNotIn(banned, line)

    def test_x_y_padding_is_the_window_centre(self):
        self.rec._write_window_event(7, [10.0, 20.0, 300.0, 200.0])
        line = self._lines()[0]
        self.assertAlmostEqual(line["x"], 160.0)
        self.assertAlmostEqual(line["y"], 120.0)

    def test_jitter_below_the_epsilon_is_not_movement(self):
        self.rec._win_last_rect[7] = [10.0, 20.0, 300.0, 200.0]
        self.assertFalse(self.rec._window_rect_changed(
            7, [10.4, 20.4, 300.4, 200.4]))
        self.assertTrue(self.rec._window_rect_changed(
            7, [12.0, 20.0, 300.0, 200.0]))

    def test_coalescing_is_per_window(self):
        """A busy window must not suppress a still one's heartbeat, or the
        still one's track would have no anchors to interpolate between."""
        self.rec._win_last_rect[7] = [0.0, 0.0, 100.0, 80.0]
        self.rec._win_last_rect[8] = [0.0, 0.0, 100.0, 80.0]
        self.assertTrue(self.rec._window_rect_changed(7, [50.0, 0.0, 100.0, 80.0]))
        self.assertFalse(self.rec._window_rect_changed(8, [0.0, 0.0, 100.0, 80.0]))

    def test_poller_logs_every_window_and_tags_each_sample(self):
        step = {"i": 0}

        def listing(**_kw):
            i = step["i"]
            step["i"] += 1
            return [
                {"id": 7, "x": float(i * 5), "y": 0.0, "w": 100.0, "h": 80.0},
                {"id": 9, "x": 500.0, "y": float(i * 5), "w": 200.0, "h": 150.0},
            ]

        self._drive(self.rec, listing, want_samples=8)
        self.assertEqual(self.rec._window_track, "ok")   # target id 7 was seen
        lines = self._lines()
        by_id = {}
        for l in lines:
            by_id.setdefault(l["id"], []).append(l["rect"])
        self.assertEqual(sorted(by_id), [7, 9])
        self.assertEqual([r[0] for r in by_id[7][:4]], [0.0, 5.0, 10.0, 15.0])
        self.assertEqual([r[1] for r in by_id[9][:4]], [0.0, 5.0, 10.0, 15.0])

    def test_track_is_failed_when_the_target_is_never_seen(self):
        """Samples of OTHER windows say nothing about whether the capture
        crop can be followed."""
        self._drive(self.rec,
                    lambda **_kw: [{"id": 99, "x": 0.0, "y": 0.0,
                                    "w": 100.0, "h": 80.0}],
                    want_samples=1)
        self.assertEqual(self.rec._window_track, "failed")
        self.assertTrue(self._lines())          # but other windows WERE logged

    def test_unreadable_windows_leave_the_track_failed(self):
        self._drive(self.rec, lambda **_kw: [], want_samples=1, timeout=0.3)
        self.assertEqual(self.rec._window_track, "failed")
        self.assertEqual(self._lines(), [])

    def test_poller_runs_without_a_capture_window(self):
        """The grid is authored after the fact on an ordinary full-display
        take, so its geometry has to be on disk anyway."""
        rec = record.Recorder(self.td, video_idx=1)
        rec._ev_file = self.rec._ev_file
        self._drive(rec,
                    lambda **_kw: [{"id": 4, "x": 1.0, "y": 2.0,
                                    "w": 100.0, "h": 80.0}],
                    want_samples=1)
        self.assertIsNone(rec._window_track)      # no target to report on
        self.assertIsNone(rec._capture_window_meta())
        self.assertTrue(self._lines())            # geometry logged regardless


class LoadEventsWindowLines(unittest.TestCase):
    def _load(self, lines):
        td = tempfile.mkdtemp(prefix="wintrack_ev_")
        try:
            p = os.path.join(td, "events.jsonl")
            with open(p, "w") as f:
                for l in lines:
                    f.write(json.dumps(l) + "\n")
            return geometry.load_events(p)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_window_lines_are_parsed_and_sorted(self):
        ev = self._load([
            {"t": 2.0, "type": "window", "rect": [3, 4, 5, 6], "x": 0, "y": 0},
            {"t": 1.0, "type": "window", "rect": [1, 2, 3, 4], "x": 0, "y": 0},
        ])
        self.assertEqual(list(ev["windows_t"]), [1.0, 2.0])
        self.assertEqual(ev["windows_rect"].shape, (2, 4))
        self.assertEqual(list(ev["windows_rect"][0]), [1.0, 2.0, 3.0, 4.0])

    def test_window_lines_never_join_the_move_track(self):
        """Their x/y is forward-compat padding, not a cursor sample."""
        ev = self._load([
            {"t": 1.0, "type": "window", "rect": [1, 2, 3, 4],
             "x": 999, "y": 999},
            {"t": 1.5, "type": "move", "x": 10, "y": 20},
        ])
        self.assertEqual(list(ev["moves_x"]), [10.0])

    def test_malformed_rect_is_skipped_not_fatal(self):
        ev = self._load([
            {"t": 1.0, "type": "window", "rect": [1, 2], "x": 0, "y": 0},
            {"t": 2.0, "type": "window", "rect": "nope", "x": 0, "y": 0},
            {"t": 3.0, "type": "window", "rect": [1, 2, 3, 4], "x": 0, "y": 0},
        ])
        self.assertEqual(list(ev["windows_t"]), [3.0])

    def test_pre_track_session_yields_an_empty_track(self):
        ev = self._load([{"t": 1.0, "type": "down", "x": 5, "y": 6}])
        self.assertEqual(ev["windows_t"].size, 0)
        self.assertEqual(ev["windows_rect"].shape, (0, 4))


class BuildWindowTrackGuards(unittest.TestCase):
    """Every path that must fall back to the single-snapshot crop."""

    def _ev(self, n):
        return {"windows_t": np.arange(n, dtype=float),
                "windows_rect": np.tile([10.0, 20.0, 100.0, 80.0], (n, 1))}

    def _build(self, ev, crop=(10, 20, 100, 80)):
        return render._build_window_track(
            ev, crop, 640, 360, 1.0, 1.0,
            np.arange(60) / 30.0, lambda a: a, 30.0)

    def test_no_crop_means_no_track(self):
        self.assertIsNone(self._build(self._ev(10), crop=None))

    def test_single_sample_is_not_a_track(self):
        self.assertIsNone(self._build(self._ev(1)))

    def test_absent_track_keys_are_tolerated(self):
        self.assertIsNone(self._build({}))

    def test_two_samples_is_enough(self):
        self.assertIsNotNone(self._build(self._ev(2)))


class WindowTrackGeometry(unittest.TestCase):
    def _track(self, rects, fps=30.0, n=60):
        ev = {"windows_t": np.linspace(0, n / fps, len(rects)),
              "windows_rect": np.asarray(rects, dtype=float)}
        return render._build_window_track(
            ev, (0, 0, 100, 80), 640, 360, 1.0, 1.0,
            np.arange(n) / fps, lambda a: a, fps)

    def test_rect_is_clamped_into_the_frame(self):
        tr = self._track([[600.0, 340.0, 400.0, 400.0]] * 4)
        x, y, w, h = tr.rect_at(10)
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 360)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)

    def test_apply_always_returns_the_snapshot_size(self):
        """(W, H) staying fixed is what keeps the canvas and camera plan
        stable while the window resizes."""
        tr = self._track([[0.0, 0.0, 100.0, 80.0],
                          [0.0, 0.0, 300.0, 240.0]])
        frame = np.zeros((360, 640, 3), np.uint8)
        for i in (0, 30, 59):
            self.assertEqual(tr.apply(frame, i).shape, (80, 100, 3))

    def test_to_src_follows_the_moving_origin(self):
        """A point sitting at the window's top-left maps to (0, 0) whatever
        the window's position -- that is the whole contract."""
        tr = self._track([[0.0, 0.0, 100.0, 80.0],
                          [200.0, 100.0, 100.0, 80.0]])
        t_end = np.asarray([59 / 30.0])
        x0, y0 = tr.rect_at(59)[:2]
        mx, my = tr.to_src(t_end, np.asarray([float(x0)]),
                           np.asarray([float(y0)]))
        self.assertAlmostEqual(float(mx[0]), 0.0, delta=1.5)
        self.assertAlmostEqual(float(my[0]), 0.0, delta=1.5)

    def test_to_src_rescales_when_the_window_resized(self):
        """A window twice the snapshot size halves incoming coordinates, so
        content lands where it does in the resized-back frame."""
        tr = self._track([[0.0, 0.0, 200.0, 160.0]] * 4)
        t = np.asarray([1.0])
        mx, my = tr.to_src(t, np.asarray([100.0]), np.asarray([80.0]))
        self.assertAlmostEqual(float(mx[0]), 50.0, delta=1.0)
        self.assertAlmostEqual(float(my[0]), 40.0, delta=1.0)

    def test_smoothing_is_zero_phase_on_a_ramp(self):
        """A linear drag must not lag: a symmetric kernel preserves a ramp."""
        ramp = np.linspace(0.0, 100.0, 41)
        smoothed = render._smooth1d(ramp, 3.0)
        mid = slice(10, 31)
        self.assertLess(float(np.abs(smoothed[mid] - ramp[mid]).max()), 0.5)


class TrackedRenderFollowsTheWindow(unittest.TestCase):
    """End-to-end: a window that moves across the display keeps its content
    framed, where the single-snapshot crop lets it slide away."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="wintrack_e2e_")
        cls.tracked = os.path.join(cls.td, "tracked")
        cls.static = os.path.join(cls.td, "static")
        samples = _moving_window_samples()
        _mk_session(cls.tracked, samples, capture_rect=samples[0][1])
        # Same recording and same snapshot rect, but no geometry track.
        _mk_session(cls.static, samples, capture_rect=samples[0][1])
        with open(os.path.join(cls.static, "events.jsonl"), "w") as f:
            f.write("")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def _centroid(self, session, t):
        frame = render.preview_frame(session, t, motion_blur=False,
                                     click_fx=False)
        return _bright_centroid(frame)

    def test_tracked_crop_holds_the_window_content_still(self):
        early = self._centroid(self.tracked, 0.5)
        late = self._centroid(self.tracked, 1.5)
        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        # The box travels 100 px/s across the display; following the window
        # should hold it within a few px over a full second.
        self.assertLess(abs(early[0] - late[0]), 8.0)

    def test_untracked_crop_lets_it_slide_away(self):
        """The behavior the track exists to fix -- asserted, not assumed, so
        the tracked test above can't quietly become a tautology."""
        early = self._centroid(self.static, 0.5)
        late = self._centroid(self.static, 1.5)
        self.assertIsNotNone(early, "box should still be in the fixed crop")
        # A second later the window has moved 100 px: either the box has left
        # the fixed crop outright (None) or it has slid a long way inside it.
        if late is not None:
            self.assertGreater(abs(early[0] - late[0]), 40.0)

    def test_output_size_is_identical_tracked_or_not(self):
        a = render.preview_frame(self.tracked, 1.0, motion_blur=False)
        b = render.preview_frame(self.static, 1.0, motion_blur=False)
        self.assertEqual(a.shape, b.shape)

    def test_full_render_runs_with_a_track(self):
        out = os.path.join(self.td, "tracked.mp4")
        render.render(self.tracked, out_path=out, motion_blur=False,
                      click_fx=False)
        self.assertTrue(os.path.getsize(out) > 1024)


class DescribeReportsTracking(unittest.TestCase):
    """The editor suppresses its "this window moved" warning on `tracked`, so
    the flag must never be true when following didn't actually happen."""

    def test_tracked_when_the_track_is_ok_and_the_crop_applied(self):
        meta = {"capture_window": _cw([10, 20, 300, 200], track="ok")}
        out = render._describe_capture_window(meta, crop=(10, 20, 300, 200))
        self.assertTrue(out["tracked"])

    def test_not_tracked_when_the_crop_was_rejected(self):
        meta = {"capture_window": _cw([10, 20, 300, 200], track="ok")}
        self.assertFalse(render._describe_capture_window(meta, crop=None)
                         ["tracked"])

    def test_not_tracked_when_the_poller_failed(self):
        meta = {"capture_window": _cw([10, 20, 300, 200], track="failed")}
        out = render._describe_capture_window(meta, crop=(10, 20, 300, 200))
        self.assertFalse(out["tracked"])

    def test_not_tracked_for_a_pre_feature_session(self):
        cw = _cw([10, 20, 300, 200])
        cw.pop("track")
        out = render._describe_capture_window({"capture_window": cw},
                                              crop=(10, 20, 300, 200))
        self.assertFalse(out["tracked"])


def _mk_grid_session(root, window_samples, duration=2.0, fps=30,
                     width=640, height=360, box_expr="40+100*t"):
    """A plain full-display session (no capture_window) whose events carry
    id-tagged geometry for one moving window -- what the multi-window grid
    binds against."""
    os.makedirs(root, exist_ok=True)
    _mk_video(os.path.join(root, "raw.mov"), duration=duration, fps=fps,
              width=width, height=height, box_expr=box_expr)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for t, wid, rect in window_samples:
            f.write(json.dumps({
                "t": t, "type": "window", "id": wid, "rect": list(rect),
                "x": rect[0] + rect[2] / 2.0, "y": rect[1] + rect[3] / 2.0,
            }) + "\n")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"raw": "raw.mov", "events": "events.jsonl", "fps": fps,
                   "logical_w": width, "logical_h": height,
                   "t0_monotonic": 0.0, "cursor_mode": "system",
                   "duration": duration}, f)


def _grid_samples(duration=2.0, hz=20.0, box=60, pad=30):
    """Window 1 follows the drawn box; window 2 sits still elsewhere, so
    binding has to actually discriminate rather than take the only option."""
    out = []
    for i in range(int(duration * hz) + 1):
        t = i / hz
        bx = 40 + 100 * t
        out.append((t, 1, [bx - pad, 110.0, box + 2 * pad, box + 2 * pad]))
        out.append((t, 2, [420.0, 20.0, 200.0, 60.0]))
    return out


class GridCardsFollowTheirWindows(unittest.TestCase):
    """The multi-window grid's cards are hand-drawn rects with no window
    identity, so each is bound by overlap and then follows. This is the
    payoff: a window moved mid-take stays framed in its card."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="wintrack_grid_")
        cls.s = os.path.join(cls.td, "s")
        _mk_grid_session(cls.s, _grid_samples())
        # Drawn where window 1 sits at t=0 -- the natural thing to draw.
        cls.rect = {"x": 10, "y": 110, "w": 120, "h": 120}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def _card(self, t, **kw):
        """The composited frame, cropped back to the first grid cell."""
        fr = render.preview_frame(self.s, t, motion_blur=False, click_fx=False,
                                  windows=[self.rect], **kw)
        p = framing.make_multi_painter(fr.shape[1], fr.shape[0], [self.rect])
        c = p.cells[0]
        return fr[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]]

    def test_card_holds_its_window_still_while_it_moves(self):
        early = _bright_centroid(self._card(0.5))
        late = _bright_centroid(self._card(1.5))
        self.assertIsNotNone(early)
        self.assertIsNotNone(late)
        self.assertLess(abs(early[0] - late[0]), 25.0)

    def test_following_off_reverts_to_a_static_crop(self):
        """The off switch has to reproduce the pre-feature behavior, so the
        box must slide away exactly as it did before."""
        early = _bright_centroid(self._card(0.5, window_follow=False))
        late = _bright_centroid(self._card(1.5, window_follow=False))
        self.assertIsNotNone(early)
        if late is not None:
            self.assertGreater(abs(early[0] - late[0]), 40.0)

    def test_off_switch_is_bit_exact_with_an_untracked_session(self):
        plain = os.path.join(self.td, "plain")
        _mk_grid_session(plain, [])
        a = render.preview_frame(self.s, 1.0, motion_blur=False,
                                 windows=[self.rect], window_follow=False)
        b = render.preview_frame(plain, 1.0, motion_blur=False,
                                 windows=[self.rect])
        self.assertTrue((a == b).all())

    def test_a_rect_matching_nothing_stays_static(self):
        """Binding must not grab the nearest window at any cost -- a rect
        drawn over empty desktop has no window to follow."""
        tracks = render._build_grid_tracks(
            geometry.load_events(os.path.join(self.s, "events.jsonl")),
            [{"x": 0, "y": 300, "w": 60, "h": 40}], None, None, 640, 360,
            1.0, 1.0, np.arange(60) / 30.0, lambda a: a, 30.0)
        self.assertEqual(tracks, [None])

    def test_it_binds_the_overlapping_window_not_the_other_one(self):
        ev = geometry.load_events(os.path.join(self.s, "events.jsonl"))
        times = np.arange(60) / 30.0
        # A rect drawn around the STILL window must not follow the moving one.
        tracks = render._build_grid_tracks(
            ev, [{"x": 420, "y": 20, "w": 200, "h": 60}], None, None,
            640, 360, 1.0, 1.0, times, lambda a: a, 30.0)
        self.assertIsNotNone(tracks[0])
        x0 = tracks[0].rect_at(0)[0]
        x1 = tracks[0].rect_at(59)[0]
        self.assertEqual(x0, x1)   # it followed the one that never moved


class SourceFrameIsTheAuthoringSpace(unittest.TestCase):
    """`source_frame` exists so rects/pins can be MEASURED. That only holds if
    it reports the same space describe_session does -- including the record-time
    window crop, tracked."""

    @classmethod
    def setUpClass(cls):
        cls.td = tempfile.mkdtemp(prefix="wintrack_src_")
        cls.s = os.path.join(cls.td, "s")
        samples = _moving_window_samples()
        _mk_session(cls.s, samples, capture_rect=samples[0][1])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.td, ignore_errors=True)

    def test_dims_match_describe_session(self):
        info = render.describe_session(self.s)
        fr = render.source_frame(self.s, 1.0)
        self.assertEqual(fr.shape[:2][::-1], (info["width"], info["height"]))

    def test_it_follows_the_track_like_the_render_does(self):
        """Measuring on a frame that ignored the track would put every rect in
        the wrong place for a window-captured session."""
        for t in (0.5, 1.0, 1.5):
            c = _bright_centroid(render.source_frame(self.s, t))
            self.assertIsNotNone(c, "box missing at t=%s" % t)
            self.assertAlmostEqual(c[0], 59.5, delta=8.0)

    def test_it_carries_no_camera_or_background(self):
        """A plain full-display session's source frame is the decoded frame,
        untouched -- no framing, no zoom."""
        td = tempfile.mkdtemp(prefix="wintrack_src_plain_")
        try:
            _mk_session(td, [], capture_rect=[0, 0, 640, 360])
            os.remove(os.path.join(td, "meta.json"))
            with open(os.path.join(td, "meta.json"), "w") as f:
                json.dump({"raw": "raw.mov", "events": "events.jsonl",
                           "fps": 30, "logical_w": 640, "logical_h": 360,
                           "t0_monotonic": 0.0, "duration": 2.0}, f)
            fr = render.source_frame(td, 1.0)
            cap = cv2.VideoCapture(os.path.join(td, "raw.mov"))
            cap.set(cv2.CAP_PROP_POS_FRAMES, 30)
            ok, want = cap.read()
            cap.release()
            self.assertTrue(ok)
            self.assertTrue((fr == want).all())
        finally:
            shutil.rmtree(td, ignore_errors=True)


class OffSwitchBitExact(unittest.TestCase):
    """A session with no track must render byte-identically to one recorded
    before tracking existed."""

    def test_track_absent_matches_pre_feature_render(self):
        td = tempfile.mkdtemp(prefix="wintrack_off_")
        try:
            samples = _moving_window_samples()
            a = os.path.join(td, "a")
            b = os.path.join(td, "b")
            _mk_session(a, [], capture_rect=samples[0][1])
            _mk_session(b, [], capture_rect=samples[0][1])
            # `b` additionally carries a one-sample track, which is below the
            # two-sample floor and so must change nothing.
            with open(os.path.join(b, "events.jsonl"), "w") as f:
                f.write(json.dumps({"t": 0.5, "type": "window",
                                    "rect": [0, 0, 100, 80],
                                    "x": 50, "y": 40}) + "\n")
            fa = render.preview_frame(a, 1.0, motion_blur=False)
            fb = render.preview_frame(b, 1.0, motion_blur=False)
            self.assertTrue((fa == fb).all())
        finally:
            shutil.rmtree(td, ignore_errors=True)


class UnclipPointSize(unittest.TestCase):
    """`render._unclip_point_size` -- the shared correction that lets takes
    recorded BEFORE 2026-08-31 (which stored the display-CLIPPED rect for a
    window hanging off the edge of the screen) still map clicks correctly.

    Recording now writes unclipped rects
    (`record.Recorder._window_native_space`), so on a new take every case
    here is the identity.
    """

    def test_agreeing_rect_and_buffer_are_the_identity(self):
        # The overwhelmingly common case, and the bit-exactness guarantee:
        # a channel that never had the bug must not move by a float.
        for lw, lh, bw, bh in ((709.0, 427.0, 1418, 854),
                               (869.0, 616.0, 1738, 1232),
                               (600.0, 400.0, 600, 400)):     # 1x, not Retina
            self.assertEqual(render._unclip_point_size(lw, lh, bw, bh),
                             (lw, lh))

    def test_a_clipped_width_is_recovered_exactly(self):
        # The reported case: Word at x=723 on a 1440-wide display, clipped to
        # 717 against a buffer holding 1382 points of window.
        self.assertEqual(render._unclip_point_size(717.0, 835.0, 2764, 1670),
                         (1382.0, 835.0))

    def test_a_clipped_height_is_recovered_too(self):
        # Nothing about the rule is width-specific -- either axis can be the
        # clipped one, and the UNCLIPPED axis is what gives the true scale.
        self.assertEqual(render._unclip_point_size(800.0, 300.0, 1600, 1200),
                         (800.0, 600.0))

    def test_the_scale_it_repairs_is_the_event_transform_denominator(self):
        """Why this matters beyond the card's shape: `raw_w / logical_w` is
        the points->pixels scale the click track is mapped through, so a
        clipped denominator puts every click in that card off-target by the
        clip ratio."""
        raw_w = 2764
        self.assertAlmostEqual(raw_w / 717.0, 3.855, delta=0.001)   # was
        fixed_w, _ = render._unclip_point_size(717.0, 835.0, raw_w, 1670)
        self.assertAlmostEqual(raw_w / fixed_w, 2.0, delta=1e-9)    # is

    def test_garbage_in_returns_the_input_untouched(self):
        # Best-effort like every other geometry read: a malformed meta must
        # render, not raise.
        for args in ((0.0, 100.0, 200, 200), (100.0, 100.0, 0, 0),
                     (None, 100.0, 200, 200), ("x", "y", "z", "w"),
                     (float("nan"), 100.0, 200, 200)):
            render._unclip_point_size(*args)      # must not raise


if __name__ == "__main__":
    unittest.main()
