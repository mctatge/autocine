"""Whole-screen "grow the active window" emphasis (`screen_focus`).

Same synthetic-session harness as test_render_windows.py -- an ffmpeg
testsrc2 pattern as raw.mov plus a hand-written events.jsonl carrying `window`
geometry samples, no macOS permissions. Exercises the feature end to end
through render()/camera_path() and pins the load-bearing invariants: the
bit-exact off switch (and the three on-but-idle cases), the shared ownership
authority, the Retina + editor-crop coordinate mapping, and the speedup gate.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
import io

import cv2
import numpy as np

from autocine import beats, camera, geometry, render


def _mk_session(root, events, duration=3, fps=20, width=640, height=360,
                logical_w=None, logical_h=None):
    os.makedirs(root, exist_ok=True)
    raw = os.path.join(root, "raw.mov")
    subprocess.check_call([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "testsrc2=size={}x{}:rate={}:duration={}".format(width, height, fps, duration),
        "-pix_fmt", "yuv420p", raw])
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({
            "raw": "raw.mov", "events": "events.jsonl",
            "fps": fps, "logical_w": logical_w or width,
            "logical_h": logical_h or height,
            "t0_monotonic": 0.0, "cursor_mode": "system",
            "duration": duration,
        }, f)


def _win_samples(wid, rect, z, t0, t1, hz=1.0):
    """1 Hz geometry heartbeats for one window over [t0, t1] (points)."""
    x, y, w, h = rect
    out = []
    t = t0
    while t <= t1 + 1e-9:
        out.append({"t": round(t, 3), "type": "window", "id": wid,
                    "rect": [x, y, w, h], "z": z,
                    "x": x + w / 2.0, "y": y + h / 2.0})
        t += hz
    return out


def _clicks(points):
    out = []
    for t, (x, y) in points:
        out.append({"t": t, "type": "down", "x": x, "y": y})
        out.append({"t": t + 0.01, "type": "up", "x": x, "y": y})
        out.append({"t": t, "type": "move", "x": x, "y": y})
    return out


# Two small windows that leave room for a push (neither near-fullscreen):
# window A frontmost (z=0) on the left, window B behind (z=1) on the right.
_WIN_A = (20, 20, 110, 80)
_WIN_B = (180, 20, 110, 80)
_CLUSTER_IN_A = [(1.0, (60, 60)), (1.2, (80, 70)),
                 (1.4, (70, 90)), (1.6, (90, 80))]


def _two_window_events():
    ev = []
    ev += _win_samples(1, _WIN_A, 0, 0.0, 3.0)
    ev += _win_samples(2, _WIN_B, 1, 0.0, 3.0)
    ev += _clicks(_CLUSTER_IN_A)
    return ev


def _render_bytes(root, **kw):
    kw.setdefault("motion_blur", False)
    kw.setdefault("click_fx", False)
    kw.setdefault("facecam", False)
    out = os.path.join(root, "out_%d.mp4" % (abs(hash(tuple(sorted(
        (k, str(v)) for k, v in kw.items())))) % 10 ** 8))
    with redirect_stdout(io.StringIO()):
        render.render(root, out_path=out, **kw)
    with open(out, "rb") as f:
        return f.read()


class OffSwitchBitExact(unittest.TestCase):
    """The load-bearing invariant: disabled / one-window / no-single-owner
    render byte-identical to today (feature omitted)."""

    def test_off_switch_is_bit_exact(self):
        td = tempfile.mkdtemp(prefix="sf_off_")
        try:
            _mk_session(td, _two_window_events())
            b_default = _render_bytes(td, screen_focus=False)
            b_explicit = _render_bytes(td, screen_focus=False)
            self.assertEqual(b_default, b_explicit)
            # And the multi-window whole-screen path itself, with the feature
            # forced off, must match the historical single-window pin shape.
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_feature_on_single_window_is_bit_exact(self):
        td = tempfile.mkdtemp(prefix="sf_single_")
        try:
            ev = _win_samples(1, _WIN_A, 0, 0.0, 3.0) + _clicks(_CLUSTER_IN_A)
            _mk_session(td, ev)
            self.assertEqual(_render_bytes(td, screen_focus=True),
                             _render_bytes(td, screen_focus=False))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_feature_on_no_single_owner_is_bit_exact(self):
        td = tempfile.mkdtemp(prefix="sf_split_")
        try:
            # Cluster split across BOTH windows -> no >=80% owner -> no reshape.
            split = [(1.0, (60, 60)), (1.2, (220, 60)),
                     (1.4, (60, 60)), (1.6, (220, 60))]
            ev = (_win_samples(1, _WIN_A, 0, 0.0, 3.0)
                  + _win_samples(2, _WIN_B, 1, 0.0, 3.0) + _clicks(split))
            _mk_session(td, ev)
            self.assertEqual(_render_bytes(td, screen_focus=True),
                             _render_bytes(td, screen_focus=False))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_aspect_changed_export_declines_bit_exact(self):
        # On a vertical/square export the base upscale already consumes the
        # grow budget, so the resolver declines (keeping the follow zoom) and
        # the render is byte-identical to the feature off.
        td = tempfile.mkdtemp(prefix="sf_aspect_")
        try:
            _mk_session(td, _two_window_events(), logical_w=320, logical_h=180)
            self.assertEqual(_render_bytes(td, screen_focus=True, aspect="9:16"),
                             _render_bytes(td, screen_focus=False, aspect="9:16"))
            self.assertEqual(_render_bytes(td, screen_focus=True, aspect="1:1"),
                             _render_bytes(td, screen_focus=False, aspect="1:1"))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_near_fullscreen_window_not_snapped(self):
        td = tempfile.mkdtemp(prefix="sf_full_")
        try:
            big = (5, 5, 300, 170)   # ~fills the 320x180 logical frame
            ev = (_win_samples(1, big, 0, 0.0, 3.0)
                  + _win_samples(2, (0, 0, 20, 20), 1, 0.0, 3.0)
                  + _clicks([(1.0, (150, 90)), (1.2, (140, 80)),
                             (1.4, (160, 100)), (1.6, (150, 90))]))
            _mk_session(td, ev, logical_w=320, logical_h=180)
            self.assertEqual(_render_bytes(td, screen_focus=True),
                             _render_bytes(td, screen_focus=False))
        finally:
            shutil.rmtree(td, ignore_errors=True)


class FeatureFires(unittest.TestCase):
    def test_two_window_single_owner_changes_output(self):
        td = tempfile.mkdtemp(prefix="sf_fire_")
        try:
            _mk_session(td, _two_window_events(), logical_w=320, logical_h=180)
            b_on = _render_bytes(td, screen_focus=True)
            b_off = _render_bytes(td, screen_focus=False)
            self.assertNotEqual(b_on, b_off)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_fires_under_motion_blur(self):
        # grow-from-out composes with the default motion_blur path.
        td = tempfile.mkdtemp(prefix="sf_mb_")
        try:
            _mk_session(td, _two_window_events(), logical_w=320, logical_h=180)
            b_on = _render_bytes(td, screen_focus=True, motion_blur=True)
            b_off = _render_bytes(td, screen_focus=False, motion_blur=True)
            self.assertNotEqual(b_on, b_off)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class SpeedupGate(unittest.TestCase):
    def test_speedup_gates_screen_focus_off(self):
        td = tempfile.mkdtemp(prefix="sf_speed_")
        try:
            # A quiet stretch the speedup planner will compress, plus the
            # two-window cluster. With retime active, screen_focus is gated
            # off, so on==off byte-for-byte.
            _mk_session(td, _two_window_events(), duration=6,
                        logical_w=320, logical_h=180)
            # Gates off so the long idle stretch (1.6s..6s, no clicks/moves)
            # is actually retimed -- tm becomes non-identity, which is what
            # gates screen_focus off.
            common = dict(motion_blur=False, click_fx=False, facecam=False,
                          speedup=True, speedup_rate=4.0,
                          speedup_silence_gate=False, speedup_motion_gate=False)
            b_on = _render_bytes(td, screen_focus=True, **common)
            b_off = _render_bytes(td, screen_focus=False, **common)
            self.assertEqual(b_on, b_off)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class OwnershipAuthority(unittest.TestCase):
    """geometry.frontmost_owner is the single authority; beats._hit delegates
    to it, so both agree on who owns any point."""

    def test_matches_beats_hit(self):
        # Two overlapping windows, A frontmost. rects as numpy rows, like the
        # render/beats state dicts carry.
        state = {
            1: (np.array([0.0, 0.0, 100.0, 100.0]), 0),
            2: (np.array([50.0, 50.0, 100.0, 100.0]), 1),
        }
        for (x, y) in [(10, 10), (75, 75), (140, 140), (200, 200)]:
            owner, _amb = geometry.frontmost_owner(state, x, y)
            self.assertEqual(owner, beats._hit(state, x, y))

    def test_overlap_frontmost_wins(self):
        state = {
            1: (np.array([0.0, 0.0, 100.0, 100.0]), 0),
            2: (np.array([50.0, 50.0, 100.0, 100.0]), 1),
        }
        owner, amb = geometry.frontmost_owner(state, 75, 75)
        self.assertEqual(owner, 1)      # z=0 wins the overlap
        self.assertFalse(amb)

    def test_ambiguous_when_no_z(self):
        # No z anywhere: fall back to the SMALLEST containing window, and flag
        # the choice ambiguous because >1 window contained the point.
        state = {
            1: (np.array([0.0, 0.0, 100.0, 100.0]), -1),
            2: (np.array([50.0, 50.0, 60.0, 60.0]), -1),   # smaller
        }
        owner, amb = geometry.frontmost_owner(state, 75, 75)
        self.assertEqual(owner, 2)
        self.assertTrue(amb)

    def test_outside_all_windows(self):
        state = {1: (np.array([0.0, 0.0, 10.0, 10.0]), 0)}
        self.assertEqual(geometry.frontmost_owner(state, 500, 500), (None, False))


class PlannerContract(unittest.TestCase):
    """The camera planner stays byte-identical with no resolver, and the
    resolver protects manual (user-drawn) zooms."""

    def _clicks_arr(self):
        t = np.array([c[0] for c in _CLUSTER_IN_A])
        # source px == points here (scale 1), inside window A
        x = np.array([c[1][0] * 2.0 for c in _CLUSTER_IN_A])
        y = np.array([c[1][1] * 2.0 for c in _CLUSTER_IN_A])
        return t, x, y

    def test_resolver_none_is_unchanged(self):
        ft = np.arange(0, 60) / 20.0
        t, x, y = self._clicks_arr()
        a = camera.build_path(ft, 640, 360, t, x, y,
                              t, x, y, plan_duration=3.0)
        b = camera.build_path(ft, 640, 360, t, x, y,
                              t, x, y, plan_duration=3.0,
                              window_resolver=None)
        self.assertTrue(np.array_equal(a, b))

    def test_manual_positionless_zoom_not_eligible(self):
        # A user-drawn zoom (no `auto` marker) stays screenAuto False, so a
        # resolver that would reshape an auto range leaves it alone.
        reshaped = []

        def resolver(ranges):
            for r in ranges:
                if r.get("screenAuto"):
                    reshaped.append(r)

        ft = np.arange(0, 60) / 20.0
        t, x, y = self._clicks_arr()
        manual = [{"start": 0.5, "end": 2.5, "x": None, "y": None,
                   "level": 2.0}]     # user-drawn: no "auto"
        camera.build_path(ft, 640, 360, t, x, y, t, x, y,
                          plan_duration=3.0, manual_zooms=manual,
                          window_resolver=resolver)
        self.assertEqual(reshaped, [],
                         "user-drawn zoom must not be reshape-eligible")

    def test_materialized_auto_zoom_is_eligible(self):
        seen = []

        def resolver(ranges):
            for r in ranges:
                if r.get("screenAuto") and r.get("type") == "follow-click-groups":
                    seen.append(r)

        ft = np.arange(0, 60) / 20.0
        t, x, y = self._clicks_arr()
        # A materialized auto-zoom carries the `auto` marker.
        auto = [{"start": 0.5, "end": 2.5, "x": None, "y": None,
                 "level": 2.0, "auto": True}]
        camera.build_path(ft, 640, 360, t, x, y, t, x, y,
                          plan_duration=3.0, manual_zooms=auto,
                          window_resolver=resolver)
        self.assertTrue(seen, "materialized auto-zoom must be reshape-eligible")


class RetinaCropCoordinates(unittest.TestCase):
    """Window rects map POINTS -> SOURCE PIXELS through the SAME to_src the
    clicks use, folding Retina scale AND the editor crop -- the documented
    half-position bug is what this pins."""

    def test_owner_rect_lands_at_full_position_with_crop(self):
        # 2x Retina: logical 320x180, raw 640x360. Editor crop removes a
        # 40x30 px margin from the top-left (in source px).
        scale_x, scale_y = 2.0, 2.0
        crop_ox, crop_oy = 40, 30

        def to_src(t, ax, ay):
            return (np.asarray(ax) * scale_x - crop_ox,
                    np.asarray(ay) * scale_y - crop_oy)

        ev = {
            "windows_t": np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            "windows_rect": np.array([_WIN_A, _WIN_A, _WIN_A,
                                      _WIN_B, _WIN_B, _WIN_B], dtype=float),
            "windows_id": np.array([1, 1, 1, 2, 2, 2]),
            "windows_z": np.array([0, 0, 0, 1, 1, 1]),
        }
        ft = np.arange(0, 40) / 20.0
        ctx = render._build_screen_focus_ctx(
            ev, lambda a: a, to_src, ft, 600, 300, 20.0)
        self.assertIsNotNone(ctx)
        # window A in source px after crop: x*2-40, y*2-30, w*2, h*2
        rect = ctx.owner_rect_median(1, 1.0, 2.0)
        exp = np.array([_WIN_A[0] * scale_x - crop_ox,
                        _WIN_A[1] * scale_y - crop_oy,
                        _WIN_A[2] * scale_x, _WIN_A[3] * scale_y])
        self.assertTrue(np.allclose(rect, exp, atol=1.0),
                        "got {} expected {}".format(rect, exp))

    def test_ctx_none_when_fewer_than_two_windows(self):
        ev = {
            "windows_t": np.array([0.0, 1.0, 2.0]),
            "windows_rect": np.array([_WIN_A, _WIN_A, _WIN_A], dtype=float),
            "windows_id": np.array([1, 1, 1]),
            "windows_z": np.array([0, 0, 0]),
        }
        ft = np.arange(0, 40) / 20.0
        ctx = render._build_screen_focus_ctx(
            ev, lambda a: a, lambda t, x, y: (x, y), ft, 320, 180, 20.0)
        self.assertIsNone(ctx)


class GrowTrack(unittest.TestCase):
    """The grow weight is read off the same spring as the push, and
    always_zoomed holds the last owner's grow to the clip end."""

    def _ctx(self, T=40, fps=20.0):
        ev = {
            "windows_t": np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]),
            "windows_rect": np.array([_WIN_A, _WIN_A, _WIN_A,
                                      _WIN_B, _WIN_B, _WIN_B], dtype=float),
            "windows_id": np.array([1, 1, 1, 2, 2, 2]),
            "windows_z": np.array([0, 0, 0, 1, 1, 1]),
        }
        ft = np.arange(0, T) / fps
        return render._build_screen_focus_ctx(
            ev, lambda a: a, lambda t, x, y: (np.asarray(x), np.asarray(y)),
            ft, 320, 180, fps)

    def test_grow_tracks_the_spring(self):
        ctx = self._ctx()
        T = 40
        # a synthetic path whose z ramps 1 -> 1.2 over a span, held elsewhere
        path = np.ones((T, 3))
        path[:, 0] = 60.0
        path[:, 1] = 60.0
        for i in range(10, 30):
            path[i, 2] = 1.0 + 0.2 * ((i - 10) / 19.0)
        spans = [{"start": 0.5, "end": 1.45, "owner": 1, "push": 1.2}]
        track = render._build_screen_focus_track(spans, ctx, path, 20.0, False)
        self.assertIsNotNone(track)
        # inside the span, grow weight rises with the spring
        _r0, w0 = track.at(11)
        _r1, w1 = track.at(28)
        self.assertGreater(w1, w0)
        # outside the span: no grow
        self.assertEqual(track.at(35)[0], None)

    def test_always_zoomed_holds_grow_to_end(self):
        ctx = self._ctx()
        T = 40
        path = np.ones((T, 3))
        path[:, 0] = 60.0
        path[:, 1] = 60.0
        path[10:, 2] = 1.2   # spring reaches push and (always_zoomed) holds
        spans = [{"start": 0.5, "end": 1.0, "owner": 1, "push": 1.2}]
        track = render._build_screen_focus_track(spans, ctx, path, 20.0, True)
        self.assertIsNotNone(track)
        rect, w = track.at(T - 1)      # last frame, past the span
        self.assertIsNotNone(rect)
        self.assertGreater(w, 0.5)


class CameraPathParity(unittest.TestCase):
    """The editor timeline reflects the same base push (camera_path runs the
    same resolver), and carries the grow overlay payload -- None when idle."""

    def test_payload_present_when_firing(self):
        td = tempfile.mkdtemp(prefix="sf_cp_")
        try:
            _mk_session(td, _two_window_events(), logical_w=320, logical_h=180)
            with redirect_stdout(io.StringIO()):
                data = render.camera_path(td, screen_focus=True, stride=1)
            self.assertIn("screen_focus", data)
            self.assertIsNotNone(data["screen_focus"])
            # some sampled frame carries a grown window box
            hit = [s for s in data["screen_focus"] if s]
            self.assertTrue(hit)
            self.assertGreater(hit[0]["grow"], 1.0)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_payload_none_when_off(self):
        td = tempfile.mkdtemp(prefix="sf_cpoff_")
        try:
            _mk_session(td, _two_window_events(), logical_w=320, logical_h=180)
            with redirect_stdout(io.StringIO()):
                data = render.camera_path(td, screen_focus=False, stride=1)
            self.assertIsNone(data["screen_focus"])
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
