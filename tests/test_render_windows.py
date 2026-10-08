"""End-to-end render checks for multi-window crop compositing.

Same synthetic-session harness as test_render_speedup.py -- an ffmpeg
testsrc2 pattern as raw.mov, no macOS permissions, no real recording --
so the windows-mode path is exercised through render()/preview_frame()
including cv2 decode, MultiFramePainter compositing, and ffmpeg encode.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout

import cv2
import numpy as np

from autocine import camera, framing, render
from tests import arrangement_claims as claims


# A real `MultiFramePainter`, cached. The arrangement sweeps below want the
# same handful over and over, and each one bakes its drop shadows in its
# constructor: measured, ~1.8s on a 2880x1800 canvas (a full-canvas
# GaussianBlur at sigma ~58px) against 14ms at 640x360. Going through real
# painters rather than re-deriving the placement pass is the point -- that
# copy already exists twice in the tree (`test_framing._arrange`,
# `arrangement_claims.placements`) and a third would be a third thing to
# drift.
_PAINTERS = {}


def _painter(W, H, rects, layout):
    key = (int(W), int(H), layout,
           tuple(tuple(sorted(r.items())) for r in rects))
    if key not in _PAINTERS:
        _PAINTERS[key] = framing.make_multi_painter(W, H, rects, layout=layout)
    return _PAINTERS[key]


def _mk_session(root, events, duration=5, fps=30, width=640, height=360,
                cursor_mode="system"):
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
            "fps": fps, "logical_w": width, "logical_h": height,
            "t0_monotonic": 0.0, "cursor_mode": cursor_mode,
            "duration": duration,
        }, f)


def _probe_frames_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=nb_frames:format=duration",
        "-of", "csv=p=0", path], text=True).strip().splitlines()
    frames = int(out[0]) if out and out[0] else 0
    dur = float(out[1]) if len(out) > 1 and out[1] else 0.0
    return frames, dur


def _probe_dims(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", path], text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)


class OffSwitchBitExact(unittest.TestCase):
    """`windows` empty/absent must be byte-identical to today's output --
    a checked-in golden isn't available, so this is a within-one-run
    comparison (None vs [] vs the omitted-kwarg default all take the
    identical `use_multi = bool(windows)` False branch)."""

    def test_none_and_empty_windows_produce_identical_bytes(self):
        td = tempfile.mkdtemp(prefix="windows_off_")
        try:
            events = [{"t": 1.0, "type": "down", "x": 100, "y": 100}]
            _mk_session(td, events, duration=3)
            out_none = os.path.join(td, "none.mp4")
            out_empty = os.path.join(td, "empty.mp4")
            render.render(td, out_path=out_none, motion_blur=False,
                         click_fx=False, windows=None)
            render.render(td, out_path=out_empty, motion_blur=False,
                         click_fx=False, windows=[])
            with open(out_none, "rb") as f:
                b_none = f.read()
            with open(out_empty, "rb") as f:
                b_empty = f.read()
            self.assertEqual(b_none, b_empty)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_omitted_kwarg_matches_explicit_none(self):
        td = tempfile.mkdtemp(prefix="windows_default_")
        try:
            events = [{"t": 1.0, "type": "down", "x": 100, "y": 100}]
            _mk_session(td, events, duration=3)
            out_default = os.path.join(td, "default.mp4")
            out_explicit = os.path.join(td, "explicit.mp4")
            render.render(td, out_path=out_default, motion_blur=False,
                         click_fx=False)
            render.render(td, out_path=out_explicit, motion_blur=False,
                         click_fx=False, windows=None)
            with open(out_default, "rb") as f:
                b_default = f.read()
            with open(out_explicit, "rb") as f:
                b_explicit = f.read()
            self.assertEqual(b_default, b_explicit)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class MultiWindowRenderGeometry(unittest.TestCase):
    def test_two_window_render_produces_expected_canvas(self):
        td = tempfile.mkdtemp(prefix="windows_2_")
        try:
            _mk_session(td, [], duration=2, fps=30, width=640, height=360)
            out = os.path.join(td, "two.mp4")
            windows = [
                {"x": 0, "y": 0, "w": 300, "h": 200},
                {"x": 300, "y": 100, "w": 300, "h": 200},
            ]
            render.render(td, out_path=out, motion_blur=False,
                         click_fx=False, windows=windows)
            w, h = _probe_dims(out)
            frames, _dur = _probe_frames_duration(out)
            self.assertEqual((w, h), (640, 360))  # canvas == source (aspect="auto")
            self.assertEqual(frames, 60)  # 2s @ 30fps
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_four_window_render_does_not_crash(self):
        td = tempfile.mkdtemp(prefix="windows_4_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=640, height=360)
            out = os.path.join(td, "four.mp4")
            windows = [
                {"x": 0, "y": 0, "w": 200, "h": 150},
                {"x": 200, "y": 0, "w": 200, "h": 150},
                {"x": 0, "y": 150, "w": 200, "h": 150},
                {"x": 200, "y": 150, "w": 200, "h": 150},
            ]
            render.render(td, out_path=out, motion_blur=False,
                         click_fx=False, windows=windows)
            self.assertTrue(os.path.getsize(out) > 1024)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_out_of_bounds_rect_does_not_crash(self):
        """edits.py deliberately does not clamp window rects spatially --
        render.py's _clamp_window_rect is what must catch this."""
        td = tempfile.mkdtemp(prefix="windows_oob_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=640, height=360)
            out = os.path.join(td, "oob.mp4")
            windows = [{"x": 9999, "y": -500, "w": 50, "h": 50}]
            render.render(td, out_path=out, motion_blur=False,
                         click_fx=False, windows=windows)
            self.assertTrue(os.path.getsize(out) > 1024)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class CompositionWithOtherFeatures(unittest.TestCase):
    def test_trim_still_applies_in_windows_mode(self):
        td = tempfile.mkdtemp(prefix="windows_trim_")
        try:
            _mk_session(td, [], duration=5, fps=30, width=640, height=360)
            out = os.path.join(td, "trim.mp4")
            windows = [{"x": 0, "y": 0, "w": 300, "h": 200}]
            render.render(td, out_path=out, motion_blur=False,
                         click_fx=False, windows=windows,
                         trim_start=1.0, trim_end=3.0)
            _frames, dur = _probe_frames_duration(out)
            self.assertAlmostEqual(dur, 2.0, delta=0.15)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_speedup_still_applies_in_windows_mode(self):
        td = tempfile.mkdtemp(prefix="windows_speedup_")
        try:
            events = [
                {"t": 1.0, "type": "down", "x": 100, "y": 100},
                {"t": 18.0, "type": "down", "x": 200, "y": 200},
            ]
            _mk_session(td, events, duration=20, fps=60, width=640, height=360)
            out = os.path.join(td, "sped.mp4")
            windows = [{"x": 0, "y": 0, "w": 300, "h": 200}]
            render.render(td, out_path=out, motion_blur=False,
                         click_fx=False, windows=windows,
                         speedup=True, speedup_rate=6.0,
                         speedup_silence_gate=True,
                         speedup_motion_gate=False)
            _frames, dur = _probe_frames_duration(out)
            self.assertLess(dur, 18.0)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_gif_export_works_in_windows_mode(self):
        td = tempfile.mkdtemp(prefix="windows_gif_")
        try:
            _mk_session(td, [], duration=1, fps=15, width=320, height=180)
            out = os.path.join(td, "out.mp4")
            windows = [{"x": 0, "y": 0, "w": 200, "h": 100}]
            render.render(td, out_path=out, motion_blur=False,
                         click_fx=False, windows=windows, make_gif=True)
            gif_path = os.path.splitext(out)[0] + ".gif"
            self.assertTrue(os.path.isfile(gif_path))
            self.assertGreater(os.path.getsize(gif_path), 128)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class ForcedOffOptionsWarn(unittest.TestCase):
    """Camera-driven effects are structurally inert in windows mode (no
    per-frame camera path to key off of) -- a caller who explicitly asked
    for them must see an explained no-op, not silence."""

    def test_ignored_options_are_named_in_a_warning(self):
        td = tempfile.mkdtemp(prefix="windows_warn_")
        try:
            events = [{"t": 0.5, "type": "down", "x": 100, "y": 100}]
            _mk_session(td, events, duration=1, fps=30, width=320, height=180)
            out = os.path.join(td, "warn.mp4")
            windows = [{"x": 0, "y": 0, "w": 200, "h": 100}]
            buf = io.StringIO()
            with redirect_stdout(buf):
                render.render(td, out_path=out, windows=windows,
                             motion_blur=True, click_fx=True,
                             spotlight=True, cursor_fx=True)
            printed = buf.getvalue()
            for name in ("motion_blur", "click_fx", "spotlight"):
                self.assertIn(name, printed)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_cursor_fx_is_not_named_because_it_is_honored(self):
        """cursor_fx used to be on the ignored list. It is positional, not
        camera-driven, so windows mode now maps it through each card's own
        crop-and-scale -- naming it here again would be a regression that
        tells users a working effect is dead."""
        td = tempfile.mkdtemp(prefix="windows_warn_cursor_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=320, height=180,
                        cursor_mode="synthetic")
            out = os.path.join(td, "warn.mp4")
            windows = [{"x": 0, "y": 0, "w": 200, "h": 100}]
            buf = io.StringIO()
            with redirect_stdout(buf):
                render.render(td, out_path=out, windows=windows,
                             motion_blur=False, click_fx=False,
                             spotlight=False, cursor_fx=True)
            self.assertNotIn("ignored in multi-window mode", buf.getvalue())
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_no_warning_when_nothing_was_requested(self):
        td = tempfile.mkdtemp(prefix="windows_nowarn_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=320, height=180)
            out = os.path.join(td, "nowarn.mp4")
            windows = [{"x": 0, "y": 0, "w": 200, "h": 100}]
            buf = io.StringIO()
            with redirect_stdout(buf):
                render.render(td, out_path=out, windows=windows,
                             motion_blur=False, click_fx=False,
                             spotlight=False, cursor_fx=False)
            self.assertNotIn("ignored in multi-window mode", buf.getvalue())
        finally:
            shutil.rmtree(td, ignore_errors=True)


class PreviewFrameWindowsMode(unittest.TestCase):
    def test_preview_frame_returns_composited_canvas(self):
        td = tempfile.mkdtemp(prefix="windows_preview_")
        try:
            _mk_session(td, [], duration=2, fps=30, width=640, height=360)
            windows = [
                {"x": 0, "y": 0, "w": 300, "h": 200},
                {"x": 300, "y": 100, "w": 300, "h": 200},
            ]
            out = render.preview_frame(td, 1.0, windows=windows)
            self.assertEqual(out.shape, (360, 640, 3))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_preview_frame_windows_off_matches_clean_style(self):
        td = tempfile.mkdtemp(prefix="windows_preview_off_")
        try:
            _mk_session(td, [], duration=2, fps=30, width=640, height=360)
            a = render.preview_frame(td, 1.0, windows=None, style="clean")
            b = render.preview_frame(td, 1.0, windows=[], style="clean")
            self.assertTrue((a == b).all())
        finally:
            shutil.rmtree(td, ignore_errors=True)


class MultiWindowLayoutForLivePreview(unittest.TestCase):
    """`render.multi_window_layout` is what the editor redraws the composite
    from during playback. Its whole reason to exist is that the browser must
    NOT own a second copy of the grid math, so these pin that it stays a
    faithful description of what `MultiFramePainter.paint` actually does."""

    WINDOWS = [
        {"x": 0, "y": 0, "w": 300, "h": 200},
        {"x": 300, "y": 100, "w": 300, "h": 200},
    ]

    def test_canvas_and_cells_match_the_composited_frame(self):
        td = tempfile.mkdtemp(prefix="windows_layout_")
        try:
            _mk_session(td, [], duration=2, fps=30, width=640, height=360)
            layout = render.multi_window_layout(td, self.WINDOWS)
            frame = render.preview_frame(td, 1.0, windows=self.WINDOWS)
            self.assertEqual(layout["canvas"], (640, 360))
            # canvas is (w, h); ndarray shape is (h, w, 3)
            self.assertEqual(frame.shape[:2][::-1], layout["canvas"])
            self.assertEqual(len(layout["cells"]), len(self.WINDOWS))
            self.assertEqual(layout["plate"].shape, frame.shape)
            for cell in layout["cells"]:
                self.assertTrue(0 <= cell["x"] and 0 <= cell["y"])
                self.assertLessEqual(cell["x"] + cell["w"], 640)
                self.assertLessEqual(cell["y"] + cell["h"], 360)
                self.assertGreater(cell["radius"], 0)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_src_rects_are_the_crops_render_takes(self):
        td = tempfile.mkdtemp(prefix="windows_layout_src_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=640, height=360)
            layout = render.multi_window_layout(td, self.WINDOWS)
            got = [tuple(c["src"]) for c in layout["cells"]]
            want = [render._clamp_window_rect(w, 640, 360) for w in self.WINDOWS]
            self.assertEqual(got, want)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_out_of_bounds_rect_is_clamped_into_the_frame(self):
        td = tempfile.mkdtemp(prefix="windows_layout_oob_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=640, height=360)
            layout = render.multi_window_layout(
                td, [{"x": 600, "y": 340, "w": 900, "h": 900}])
            sx, sy, sw, sh = layout["cells"][0]["src"]
            self.assertLessEqual(sx + sw, 640)
            self.assertLessEqual(sy + sh, 360)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_src_rects_carry_the_capture_window_offset(self):
        """Window rects live in WINDOW space, but the browser samples the RAW
        file -- so a window-captured session's src must be shifted back by the
        capture crop, or every cell would draw the wrong region."""
        td = tempfile.mkdtemp(prefix="windows_layout_cw_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=640, height=360)
            with open(os.path.join(td, "meta.json")) as f:
                meta = json.load(f)
            meta["capture_window"] = {
                "id": 1, "app": "a", "title": "t", "units": "points",
                "rect": [100, 60, 400, 240], "display_origin": [0.0, 0.0],
                "source": "quartz", "resnapshot": True,
                "end_rect": [100, 60, 400, 240],
            }
            with open(os.path.join(td, "meta.json"), "w") as f:
                json.dump(meta, f)
            crop = render._capture_crop_px(meta, 640, 360)
            self.assertIsNotNone(crop)      # guard: the fixture must crop
            crop_x, crop_y, _cw, _ch = crop
            layout = render.multi_window_layout(td, [{"x": 10, "y": 20,
                                                     "w": 100, "h": 80}])
            sx, sy, _sw, _sh = layout["cells"][0]["src"]
            self.assertEqual((sx, sy), (10 + crop_x, 20 + crop_y))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_plate_plus_cells_reproduces_paint(self):
        """The exact composite editor.js builds -- plate, then each crop
        resized into its cell through a rounded clip -- must land on
        `paint()`'s output. Only the plate's uint8 quantization may differ."""
        rects = [{"x": 0, "y": 0, "w": 300, "h": 200},
                 {"x": 300, "y": 100, "w": 300, "h": 200},
                 {"x": 100, "y": 50, "w": 200, "h": 300}]
        painter = framing.make_multi_painter(640, 360, rects)
        crops = [
            np.full((r["h"], r["w"], 3), 40 * (i + 1), np.uint8)
            for i, r in enumerate(rects)
        ]
        want = painter.paint(crops)

        got = painter.base_plate().astype(np.float32)
        for cell, crop in zip(painter.cells, crops):
            x, y, w, h = cell["x"], cell["y"], cell["w"], cell["h"]
            rec = cv2.resize(crop, (w, h),
                             interpolation=cv2.INTER_AREA).astype(np.float32)
            mask = framing._rounded_mask(w, h, cell["radius"])
            m3 = (mask.astype(np.float32) / 255.0)[:, :, None]
            roi = got[y:y + h, x:x + w]
            got[y:y + h, x:x + w] = rec * m3 + roi * (1.0 - m3)
        got = got.astype(np.uint8)

        self.assertEqual(got.shape, want.shape)
        self.assertLessEqual(
            int(np.abs(got.astype(np.int16) - want.astype(np.int16)).max()), 1)

    def test_empty_windows_is_rejected(self):
        td = tempfile.mkdtemp(prefix="windows_layout_empty_")
        try:
            _mk_session(td, [], duration=1, fps=30, width=640, height=360)
            with self.assertRaises(ValueError):
                render.multi_window_layout(td, [])
        finally:
            shutil.rmtree(td, ignore_errors=True)


class SyntheticCursorInWindowsMode(unittest.TestCase):
    """The synthetic cursor survives the multi-window compositor.

    It is the one 'camera-driven' effect that never actually needed a camera:
    each card is a known crop-and-scale, so a source position maps into it by
    that card's own transform. These pin WHERE it lands, that it stays inside
    the card showing it, and that both off switches are still bit-exact.
    """

    W, H = 640, 360
    # Two non-overlapping cards. A holds the top-left quadrant, B the
    # bottom-right one, so a point can be inside exactly one of them.
    CARD_A = {"x": 0, "y": 0, "w": 300, "h": 180}
    CARD_B = {"x": 320, "y": 180, "w": 300, "h": 180}

    def _sweep(self, x0, x1, y, duration=2.0, hz=20.0):
        """A moving cursor. A STATIONARY one is not usable here: CursorFX
        fades out after `idle_after` of no movement, so a still cursor would
        make these tests pass or fail on the alpha ramp instead of on the
        geometry they are about."""
        n = int(duration * hz)
        return [{"t": i / hz, "type": "move",
                 "x": x0 + (x1 - x0) * (i / float(n - 1)), "y": y}
                for i in range(n)]

    def _session(self, td, events, cursor_mode="synthetic"):
        _mk_session(td, events, duration=2, fps=30,
                    width=self.W, height=self.H, cursor_mode=cursor_mode)

    def _frame(self, td, windows, cursor_fx, t=1.5, **kw):
        return render.preview_frame(td, t_sec=t, windows=windows,
                                    cursor_fx=cursor_fx, motion_blur=False,
                                    click_fx=False, window_follow=False, **kw)

    def _cells(self, windows):
        painter = framing.make_multi_painter(self.W, self.H, windows)
        return painter.cells

    def test_cursor_is_drawn_into_the_card_that_contains_it(self):
        td = tempfile.mkdtemp(prefix="windows_cursor_in_")
        try:
            # Sweep across the middle of card B and nowhere near card A.
            self._session(td, self._sweep(400, 560, 300))
            windows = [self.CARD_A, self.CARD_B]
            off = self._frame(td, windows, False)
            on = self._frame(td, windows, True)
            diff = np.abs(on.astype(int) - off.astype(int)).sum(axis=2)
            ys, xs = np.where(diff > 0)
            self.assertGreater(len(ys), 0, "cursor was not drawn at all")

            cell_a, cell_b = self._cells(windows)
            # Everything drawn is inside card B's destination box ...
            self.assertGreaterEqual(int(xs.min()), cell_b["x"])
            self.assertLess(int(xs.max()), cell_b["x"] + cell_b["w"])
            self.assertGreaterEqual(int(ys.min()), cell_b["y"])
            self.assertLess(int(ys.max()), cell_b["y"] + cell_b["h"])
            # ... and nothing leaked into card A.
            self.assertEqual(
                0, int((diff[cell_a["y"]:cell_a["y"] + cell_a["h"],
                             cell_a["x"]:cell_a["x"] + cell_a["w"]] > 0).sum()))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_cursor_lands_where_the_card_transform_says_it_should(self):
        """Position, not just presence: the drawn glyph's tip must sit where
        the card's own crop-and-scale puts the smoothed source position."""
        from autocine import effects, geometry

        td = tempfile.mkdtemp(prefix="windows_cursor_at_")
        try:
            self._session(td, self._sweep(400, 560, 300))
            windows = [self.CARD_A, self.CARD_B]
            off = self._frame(td, windows, False)
            on = self._frame(td, windows, True)
            diff = np.abs(on.astype(int) - off.astype(int)).sum(axis=2)
            ys, xs = np.where(diff > 0)

            ev = geometry.load_events(os.path.join(td, "events.jsonl"))
            idx = int(round(1.5 * 30))
            cfx = effects.CursorFX(self.W, self.H,
                                   np.arange(idx + 1) / 30.0,
                                   ev["moves_t"], ev["moves_x"], ev["moves_y"],
                                   clicks_t=ev["clicks_t"])
            cell = self._cells(windows)[1]
            b = self.CARD_B
            want_x = cell["x"] + (cfx.sx[idx] - b["x"]) * cell["w"] / float(b["w"])
            want_y = cell["y"] + (cfx.sy[idx] - b["y"]) * cell["h"] / float(b["h"])
            # The glyph hangs down-right of its tip, so the drawn bounding box
            # STARTS at the hotspot (modulo the shadow/outline feather).
            self.assertLess(abs(int(xs.min()) - want_x), 8)
            self.assertLess(abs(int(ys.min()) - want_y), 8)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_cursor_outside_every_card_draws_nothing(self):
        """The cursor is over desktop that no card is showing -- it must not
        be clamped onto the nearest card's edge."""
        td = tempfile.mkdtemp(prefix="windows_cursor_out_")
        try:
            # y=300 is below card A; x=100 is left of card B. In neither.
            self._session(td, self._sweep(60, 140, 300))
            windows = [self.CARD_A, self.CARD_B]
            off = self._frame(td, windows, False)
            on = self._frame(td, windows, True)
            self.assertTrue(np.array_equal(on, off))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_overlapping_cards_each_show_the_cursor(self):
        """Overlapping cards duplicate the same source pixels into the
        composite, so the cursor over that region belongs in both copies --
        drawing it once would leave one copy disagreeing with the other."""
        td = tempfile.mkdtemp(prefix="windows_cursor_dup_")
        try:
            self._session(td, self._sweep(200, 260, 150))
            # Both rects contain the sweep.
            windows = [{"x": 100, "y": 100, "w": 300, "h": 180},
                       {"x": 120, "y": 110, "w": 300, "h": 180}]
            off = self._frame(td, windows, False)
            on = self._frame(td, windows, True)
            diff = np.abs(on.astype(int) - off.astype(int)).sum(axis=2)
            cells = self._cells(windows)
            for i, cell in enumerate(cells):
                sub = diff[cell["y"]:cell["y"] + cell["h"],
                           cell["x"]:cell["x"] + cell["w"]]
                self.assertGreater(int((sub > 0).sum()), 0,
                                   "card {} did not get the cursor".format(i))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_off_switch_is_bit_exact(self):
        td = tempfile.mkdtemp(prefix="windows_cursor_off_")
        try:
            self._session(td, self._sweep(400, 560, 300))
            windows = [self.CARD_A, self.CARD_B]
            a = self._frame(td, windows, False)
            b = self._frame(td, windows, False)
            self.assertTrue(np.array_equal(a, b))
            # And the drawn version really is different, so the comparison
            # above is not passing for want of any cursor at all.
            self.assertFalse(np.array_equal(a, self._frame(td, windows, True)))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_system_cursor_mode_still_suppresses_it(self):
        """A session recorded WITHOUT --cursor synthetic already has the real
        cursor baked into its pixels; drawing a second one would double up.
        That gate predates windows mode and must survive it."""
        td = tempfile.mkdtemp(prefix="windows_cursor_sys_")
        try:
            self._session(td, self._sweep(400, 560, 300),
                          cursor_mode="system")
            windows = [self.CARD_A, self.CARD_B]
            off = self._frame(td, windows, False)
            on = self._frame(td, windows, True)
            self.assertTrue(np.array_equal(on, off))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_but_erasing_the_recorded_one_lets_it_in(self):
        """The gate above is not "was this recorded synthetic", it is "is the
        frame free of a pointer" (`render._cursor_fx_draws`) -- and the cursor
        eraser is the second way to make it so. One helper serves all four
        render paths, so the compositor inherits the retrofit rather than
        disagreeing with the single-view export about one edits.json."""
        td = tempfile.mkdtemp(prefix="windows_cursor_retro_")
        try:
            self._session(td, self._sweep(400, 560, 300),
                          cursor_mode="system")
            windows = [self.CARD_A, self.CARD_B]
            # Both erased, so the ONLY thing that can differ is the cursor.
            off = self._frame(td, windows, False, cursor_erase=True)
            on = self._frame(td, windows, True, cursor_erase=True)
            self.assertFalse(np.array_equal(on, off))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_full_render_draws_it_too(self):
        """preview_frame and render() are separate code paths; the paused
        still agreeing with the export is the whole point."""
        td = tempfile.mkdtemp(prefix="windows_cursor_e2e_")
        try:
            self._session(td, self._sweep(400, 560, 300))
            windows = [self.CARD_A, self.CARD_B]
            outs = {}
            for flag in (False, True):
                out = os.path.join(td, "cursor_{}.mp4".format(flag))
                render.render(td, out_path=out, windows=windows,
                             cursor_fx=flag, motion_blur=False,
                             click_fx=False, window_follow=False)
                cap = cv2.VideoCapture(out)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 45)
                ok, fr = cap.read()
                cap.release()
                self.assertTrue(ok)
                outs[flag] = fr
            self.assertFalse(np.array_equal(outs[False], outs[True]))
            diff = np.abs(outs[True].astype(int)
                          - outs[False].astype(int)).sum(axis=2)
            # These are two separate H.264 encodes, so an untouched region
            # still differs by a little codec noise -- "card A is byte-equal"
            # is not assertable here (preview_frame's ndarray test above is
            # where that belongs). What IS robust: the strongest difference in
            # the whole frame has to be where the cursor was drawn.
            cell_a, cell_b = self._cells(windows)
            peak_y, peak_x = np.unravel_index(int(diff.argmax()), diff.shape)
            self.assertTrue(
                cell_b["x"] <= peak_x < cell_b["x"] + cell_b["w"]
                and cell_b["y"] <= peak_y < cell_b["y"] + cell_b["h"],
                "peak change at ({}, {}) is outside card B".format(peak_x, peak_y))
            in_a = diff[cell_a["y"]:cell_a["y"] + cell_a["h"],
                        cell_a["x"]:cell_a["x"] + cell_a["w"]]
            in_b = diff[cell_b["y"]:cell_b["y"] + cell_b["h"],
                        cell_b["x"]:cell_b["x"] + cell_b["w"]]
            self.assertGreater(int(in_b.max()), 4 * int(in_a.max()) + 1)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class CursorTrackForLivePreview(unittest.TestCase):
    """`multi_window_cursor_track` is what stops the cursor blinking out on
    play: the browser cannot run CursorFX, so it replays this instead. It has
    to agree with what the server itself composites, and stay absent whenever
    nothing would be drawn."""

    W, H = 640, 360
    CARDS = [{"x": 0, "y": 0, "w": 300, "h": 180},
             {"x": 320, "y": 180, "w": 300, "h": 180}]

    def _sweep(self, x0, x1, y, duration=2.0, hz=20.0):
        n = int(duration * hz)
        return [{"t": i / hz, "type": "move",
                 "x": x0 + (x1 - x0) * (i / float(n - 1)), "y": y}
                for i in range(n)]

    def test_track_matches_what_the_server_composites(self):
        td = tempfile.mkdtemp(prefix="windows_track_")
        try:
            _mk_session(td, self._sweep(400, 560, 300), duration=2, fps=30,
                        width=self.W, height=self.H, cursor_mode="synthetic")
            track = render.multi_window_cursor_track(td, cursor_fx=True,
                                                     stride=1)
            self.assertIsNotNone(track)
            # Same numbers CursorFX hands _draw_multi_cursor for that frame.
            from autocine import effects, geometry
            ev = geometry.load_events(os.path.join(td, "events.jsonl"))
            idx = int(round(1.5 * 30))
            cfx = effects.CursorFX(self.W, self.H, np.arange(idx + 1) / 30.0,
                                   ev["moves_t"], ev["moves_x"], ev["moves_y"],
                                   clicks_t=ev["clicks_t"])
            self.assertAlmostEqual(track["x"][idx], float(cfx.sx[idx]), places=1)
            self.assertAlmostEqual(track["y"][idx], float(cfx.sy[idx]), places=1)
            self.assertAlmostEqual(track["size"], float(cfx.size), places=4)
            self.assertEqual(len(track["shape"]), len(effects._CURSOR_SHAPE))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_stride_thins_without_moving_the_clock(self):
        td = tempfile.mkdtemp(prefix="windows_track_stride_")
        try:
            _mk_session(td, self._sweep(400, 560, 300), duration=2, fps=30,
                        width=self.W, height=self.H, cursor_mode="synthetic")
            one = render.multi_window_cursor_track(td, cursor_fx=True, stride=1)
            two = render.multi_window_cursor_track(td, cursor_fx=True, stride=2)
            self.assertEqual(two["stride"], 2)
            self.assertEqual(two["fps"], one["fps"])
            self.assertEqual(len(two["x"]), (len(one["x"]) + 1) // 2)
            # index i of the strided track is frame i*stride of the dense one,
            # which is exactly what the JS `round(t * fps / stride)` assumes.
            for i in range(len(two["x"])):
                self.assertAlmostEqual(two["x"][i], one["x"][i * 2], places=2)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_absent_when_nothing_would_be_drawn(self):
        td = tempfile.mkdtemp(prefix="windows_track_none_")
        try:
            _mk_session(td, self._sweep(400, 560, 300), duration=2, fps=30,
                        width=self.W, height=self.H, cursor_mode="synthetic")
            self.assertIsNone(
                render.multi_window_cursor_track(td, cursor_fx=False),
                "effect off must not ship a track")
        finally:
            shutil.rmtree(td, ignore_errors=True)

        td = tempfile.mkdtemp(prefix="windows_track_sys_")
        try:
            _mk_session(td, self._sweep(400, 560, 300), duration=2, fps=30,
                        width=self.W, height=self.H, cursor_mode="system")
            self.assertIsNone(
                render.multi_window_cursor_track(td, cursor_fx=True),
                "a baked-in system cursor must not get a second one")
        finally:
            shutil.rmtree(td, ignore_errors=True)

        td = tempfile.mkdtemp(prefix="windows_track_nomoves_")
        try:
            _mk_session(td, [], duration=2, fps=30, width=self.W,
                        height=self.H, cursor_mode="synthetic")
            self.assertIsNone(
                render.multi_window_cursor_track(td, cursor_fx=True),
                "no move track means nothing to smooth")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class PerCardAutoZoom(unittest.TestCase):
    """`render.window_zoom`: each card gets its own camera, and only the card
    with the most recent activity is ever the one driving."""

    W, H = 640, 360
    CARD_A = {"x": 0, "y": 0, "w": 300, "h": 180}
    CARD_B = {"x": 320, "y": 180, "w": 300, "h": 180}

    def _clicks(self, spec, times, at=(0.5, 0.5)):
        """Clicks landing inside `spec`, at a fraction of the card."""
        x = spec["x"] + spec["w"] * at[0]
        y = spec["y"] + spec["h"] * at[1]
        return [{"t": t, "type": "down", "x": x, "y": y} for t in times]

    def _session(self, td, events, duration=20):
        _mk_session(td, sorted(events, key=lambda e: e["t"]),
                    duration=duration, fps=30, width=self.W, height=self.H)

    def _paths(self, td, windows, enabled=True, duration=20, fps=30):
        from autocine import geometry
        ev = geometry.load_events(os.path.join(td, "events.jsonl"))
        painter = framing.make_multi_painter(self.W, self.H, windows)
        n = int(duration * fps)
        return render._build_card_cameras(
            windows, painter.cells, None,
            ev["clicks_t"], ev["clicks_x"], ev["clicks_y"],
            np.arange(n) / float(fps), self.W, self.H, 2.0, None, [],
            n / float(fps), fps, enabled=enabled)

    def test_each_card_zooms_on_its_own_clicks(self):
        td = tempfile.mkdtemp(prefix="windows_zoom_own_")
        try:
            self._session(td, self._clicks(self.CARD_A, [5.0, 5.6, 6.2]))
            windows = [self.CARD_A, self.CARD_B]
            paths = self._paths(td, windows)
            self.assertIsNotNone(paths[0], "card A had the clicks, must zoom")
            self.assertIsNone(paths[1], "card B had none, must not zoom")
            self.assertGreater(float(paths[0][:, 2].max()), 1.5)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_only_one_card_drives_at_a_time(self):
        """The headline guarantee. Two cards clicked in alternating bursts:
        no instant may have two cards with an ACTIVE range, and a card with
        no active range may only ease out -- never climb."""
        td = tempfile.mkdtemp(prefix="windows_zoom_one_")
        try:
            events = (self._clicks(self.CARD_A, [3.0, 3.5, 4.0])
                      + self._clicks(self.CARD_B, [9.0, 9.5, 10.0])
                      + self._clicks(self.CARD_A, [15.0, 15.5, 16.0]))
            self._session(td, events)
            windows = [self.CARD_A, self.CARD_B]
            paths = self._paths(td, windows)
            self.assertTrue(all(p is not None for p in paths),
                            "both cards were clicked, both should zoom")
            z = np.stack([p[:, 2] for p in paths])
            # Never two cards climbing at once: at every frame at most one
            # card's zoom is increasing.
            rising = (np.diff(z, axis=1) > 1e-6).sum(axis=0)
            self.assertEqual(0, int((rising > 1).sum()),
                             "two cards zoomed in simultaneously")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_a_card_yields_when_the_other_is_clicked(self):
        """Card A holds, then B is clicked: A must be settling by the time B
        is at its peak, not still holding its own zoom."""
        td = tempfile.mkdtemp(prefix="windows_zoom_yield_")
        try:
            events = (self._clicks(self.CARD_A, [3.0, 3.5, 4.0])
                      + self._clicks(self.CARD_B, [9.0, 9.5, 10.0]))
            self._session(td, events)
            paths = self._paths(td, [self.CARD_A, self.CARD_B])
            za, zb = paths[0][:, 2], paths[1][:, 2]
            peak_b = int(np.argmax(zb))
            self.assertGreater(float(zb[peak_b]), 1.5)
            self.assertLess(float(za[peak_b]), 1.05,
                            "card A was still zoomed while B was at its peak")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_alternating_attention_lets_both_cards_zoom(self):
        """The canonical two-window workflow. `chain_gap` is 4.0s, so
        clustering each card in ISOLATION chains its clicks into one
        take-length range; two of those cover each other end to end and
        arbitration can then only silence one card for the whole recording.
        Clustering across cards and arbitrating BEFORE the merge is what
        keeps both alive. Pins the exact regression that shipped once."""
        from autocine import camera
        cards = [{"w": 300.0, "h": 180.0, "clicks": []},
                 {"w": 300.0, "h": 180.0, "clicks": []}]
        # Attention swaps every 3s -- longer than `pre_roll`, so each swap has
        # room for a real zoom arc. 10 clicks per card over 60s.
        for i in range(10):
            cards[0]["clicks"].append((5.0 + 6.0 * i, 150.0, 90.0))
            cards[1]["clicks"].append((8.0 + 6.0 * i, 150.0, 90.0))
        ft = np.arange(int(70 * 30)) / 30.0
        paths = camera.build_card_paths(ft, cards, max_zoom=2.0,
                                        plan_duration=70.0)
        for i, p in enumerate(paths):
            self.assertGreater(float(p[:, 2].max()), 1.5,
                               "card {} was clicked 10 times and never "
                               "zoomed".format(i))
        z = np.stack([p[:, 2] for p in paths])
        self.assertEqual(0, int(((np.diff(z, axis=1) > 1e-6).sum(axis=0) > 1).sum()),
                         "two cards zoomed in at once")

    def test_attention_faster_than_the_pre_roll_declines_to_zoom(self):
        """The other half of the contract, and it is a FEATURE. When the user
        swaps windows faster than `pre_roll` (2.5s), no zoom arc can complete
        before it is preempted -- following that would flip the framing every
        1.5s, which is the whip-pan motion sickness the whole model exists to
        prevent. So the composition stays steady instead. The one thing that
        must never happen is both cards zoomed together."""
        from autocine import camera
        cards = [{"w": 300.0, "h": 180.0, "clicks": []},
                 {"w": 300.0, "h": 180.0, "clicks": []}]
        for i in range(20):
            cards[i % 2]["clicks"].append((5.0 + 1.5 * i, 150.0, 90.0))
        ft = np.arange(int(45 * 30)) / 30.0
        paths = camera.build_card_paths(ft, cards, max_zoom=2.0,
                                        plan_duration=45.0)
        z = np.stack([p[:, 2] for p in paths])
        zoomed_frames = int((z > 1.05).any(axis=0).sum())
        self.assertLess(zoomed_frames, len(ft) * 0.25,
                        "rapid window-swapping should mostly hold the steady "
                        "composition, not chase it")
        self.assertEqual(0, int(((z > 1.05).sum(axis=0) > 1).sum()),
                         "two cards zoomed at once")

    def test_always_zoomed_does_not_defeat_arbitration(self):
        """`always_zoomed` means 'hold the last zoom to the end'. Honoured per
        card it makes EVERY card that ever zoomed hold forever, so they all end
        up zoomed together -- the one thing this mode must not do."""
        from autocine import camera
        cards = [{"w": 300.0, "h": 180.0,
                  "clicks": [(3.0, 150.0, 90.0), (3.5, 150.0, 90.0)]},
                 {"w": 300.0, "h": 180.0,
                  "clicks": [(12.0, 150.0, 90.0), (12.5, 150.0, 90.0)]}]
        ft = np.arange(int(20 * 30)) / 30.0
        paths = camera.build_card_paths(ft, cards, max_zoom=2.0,
                                        params={"always_zoomed": True},
                                        plan_duration=20.0)
        z = np.stack([p[:, 2] for p in paths])
        self.assertEqual(0, int(((z > 1.05).sum(axis=0) > 1).sum()),
                         "always_zoomed left both cards held at zoom")

    def test_zoom_1_framing_matches_the_plain_composite(self):
        """A card's zoom-1.0 window must be the WHOLE card rect. If it were a
        cell-aspect sub-box (_contain_fit), turning zoom on would centre-crop
        and silently drop content whenever a cell's aspect differs from its
        card's -- clamped rects, hand-dragged cards, a changed --aspect."""
        # Deliberately overhanging, so _clamp_window_rect changes the aspect
        # and the cell no longer matches the card.
        spec = {"x": 460, "y": 0, "w": 300, "h": 180}
        painter = framing.make_multi_painter(640, 360, [spec, self.CARD_B])
        cell = painter.cells[0]
        flat = np.zeros((5, 3))
        flat[:, 2] = 1.0
        _dx, _dy, dw, dh = render._clamp_window_rect(spec, 640, 360)
        flat[:, 0], flat[:, 1] = dw / 2.0, dh / 2.0
        with_cam = render._card_to_cell(spec, cell, flat, 0, 640, 360)
        without = render._card_to_cell(spec, cell, None, 0, 640, 360)
        for a, b in zip(with_cam, without):
            self.assertAlmostEqual(a, b, places=6,
                                   msg="z=1.0 framing differs from the plain "
                                       "composite ({} vs {})".format(with_cam,
                                                                     without))

    def test_off_switch_is_bit_exact(self):
        td = tempfile.mkdtemp(prefix="windows_zoom_off_")
        try:
            self._session(td, self._clicks(self.CARD_A, [5.0, 5.6, 6.2]))
            windows = [self.CARD_A, self.CARD_B]
            self.assertEqual([None, None],
                             self._paths(td, windows, enabled=False))
            base = render.preview_frame(td, t_sec=6.0, windows=windows,
                                        motion_blur=False, click_fx=False,
                                        window_zoom=False)
            again = render.preview_frame(td, t_sec=6.0, windows=windows,
                                         motion_blur=False, click_fx=False)
            self.assertTrue(np.array_equal(base, again),
                            "window_zoom defaults must reproduce today's frame")
            zoomed = render.preview_frame(td, t_sec=6.0, windows=windows,
                                          motion_blur=False, click_fx=False,
                                          window_zoom=True)
            self.assertFalse(np.array_equal(base, zoomed))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_a_still_card_is_not_resampled(self):
        """A card the camera never picked must go through the untouched
        `paint()` resize, not a zoom-1.0 warp -- otherwise switching the
        feature on quietly re-filters every other card in the grid."""
        td = tempfile.mkdtemp(prefix="windows_zoom_still_")
        try:
            self._session(td, self._clicks(self.CARD_A, [5.0, 5.6, 6.2]))
            windows = [self.CARD_A, self.CARD_B]
            painter = framing.make_multi_painter(self.W, self.H, windows)
            cell_b = painter.cells[1]
            off = render.preview_frame(td, t_sec=6.0, windows=windows,
                                       motion_blur=False, click_fx=False)
            on = render.preview_frame(td, t_sec=6.0, windows=windows,
                                      motion_blur=False, click_fx=False,
                                      window_zoom=True)
            box = (slice(cell_b["y"], cell_b["y"] + cell_b["h"]),
                   slice(cell_b["x"], cell_b["x"] + cell_b["w"]))
            self.assertTrue(np.array_equal(off[box], on[box]),
                            "the non-zooming card's pixels changed")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_full_render_applies_it(self):
        td = tempfile.mkdtemp(prefix="windows_zoom_e2e_")
        try:
            self._session(td, self._clicks(self.CARD_A, [2.0, 2.5, 3.0]),
                          duration=6)
            windows = [self.CARD_A, self.CARD_B]
            frames = {}
            for flag in (False, True):
                out = os.path.join(td, "wz_{}.mp4".format(flag))
                render.render(td, out_path=out, windows=windows,
                             window_zoom=flag, motion_blur=False,
                             click_fx=False)
                cap = cv2.VideoCapture(out)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 90)   # t = 3.0s at 30fps
                ok, fr = cap.read()
                cap.release()
                self.assertTrue(ok)
                frames[flag] = fr
            self.assertFalse(np.array_equal(frames[False], frames[True]))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_live_preview_track_matches_the_render(self):
        """The browser composites window_zoom by sampling a smaller source
        rect, so its (cx, cy, z) has to be the same numbers render uses."""
        td = tempfile.mkdtemp(prefix="windows_zoom_track_")
        try:
            self._session(td, self._clicks(self.CARD_A, [5.0, 5.6, 6.2]))
            windows = [self.CARD_A, self.CARD_B]
            painter = framing.make_multi_painter(self.W, self.H, windows)
            served = render.multi_window_card_paths(
                td, windows, painter.cells, window_zoom=True, stride=1)
            self.assertIsNotNone(served)
            self.assertIsNone(served["cards"][1], "still card must stay null")
            local = self._paths(td, windows)[0]
            idx = int(round(6.0 * 30))
            self.assertAlmostEqual(served["cards"][0]["z"][idx],
                                   float(local[idx, 2]), places=3)
            self.assertAlmostEqual(served["cards"][0]["cx"][idx],
                                   float(local[idx, 0]), places=1)
            self.assertIsNone(
                render.multi_window_card_paths(td, windows, painter.cells,
                                               window_zoom=False),
                "off must not ship a track")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class WindowFocus(unittest.TestCase):
    """`render.window_focus`: the card you are working in GROWS IN PLACE and
    overlaps its neighbours, which do not move at all, and it settles back
    when attention moves.

    Distinct from `window_zoom` (which zooms the footage INSIDE a card while
    its cell stays put). Tested on the placements rather than the pixels,
    because the placements are the thing the export and the editor's browser
    compositor both consume -- one layout, two renderers.
    """

    W, H = 640, 360
    CARD_A = {"x": 0, "y": 0, "w": 300, "h": 180}
    CARD_B = {"x": 320, "y": 180, "w": 300, "h": 180}
    CARD_C = {"x": 0, "y": 180, "w": 300, "h": 180}

    def _clicks(self, spec, times):
        x = spec["x"] + spec["w"] * 0.5
        y = spec["y"] + spec["h"] * 0.5
        return [{"t": t, "type": "down", "x": x, "y": y} for t in times]

    def _session(self, td, events, duration=26):
        _mk_session(td, sorted(events, key=lambda e: e["t"]),
                    duration=duration, fps=30, width=self.W, height=self.H)

    def _emphasis(self, td, windows, duration=26, fps=30, manual=None,
                  enabled=True):
        from autocine import geometry
        ev = geometry.load_events(os.path.join(td, "events.jsonl"))
        n = int(duration * fps)
        return render._build_focus_emphasis(
            windows, None, ev["clicks_t"], ev["clicks_x"], ev["clicks_y"],
            np.arange(n) / float(fps), self.W, self.H, 2.0, None, [],
            n / float(fps), fps, manual=manual, enabled=enabled)

    def _layout(self, td, windows, **kw):
        e = self._emphasis(td, windows, **kw)
        painter = framing.make_multi_painter(self.W, self.H, windows)
        if e is None:
            return None, painter
        return render._FocusLayout(painter, e), painter

    @staticmethod
    def _area(cell):
        return float(cell["w"]) * float(cell["h"])

    def test_off_switch_is_bit_exact(self):
        """The invariant every toggle in this repo carries: off reproduces
        the pre-feature frame byte for byte."""
        td = tempfile.mkdtemp(prefix="focus_off_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            windows = [self.CARD_A, self.CARD_B]
            with redirect_stdout(io.StringIO()):
                base = render.preview_frame(td, 5.0, windows=windows)
                off = render.preview_frame(td, 5.0, windows=windows,
                                           window_focus=False)
            self.assertTrue(np.array_equal(base, off))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_an_unemphasized_frame_is_also_bit_exact(self):
        """Stronger than the off switch, and the reason `paint_at` falls
        through to `paint()`: with focus ON, a frame where nothing is the
        subject must still be the plain grid, byte for byte. Otherwise
        turning the feature on quietly re-filters the whole recording."""
        td = tempfile.mkdtemp(prefix="focus_idle_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            windows = [self.CARD_A, self.CARD_B]
            with redirect_stdout(io.StringIO()):
                plain = render.preview_frame(td, 20.0, windows=windows)
                on = render.preview_frame(td, 20.0, windows=windows,
                                          window_focus=True)
            self.assertTrue(np.array_equal(plain, on),
                            "a frame with no subject must not be touched")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_no_clicks_means_no_emphasis(self):
        td = tempfile.mkdtemp(prefix="focus_none_")
        try:
            self._session(td, [])
            self.assertIsNone(self._emphasis(td, [self.CARD_A, self.CARD_B]))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_the_clicked_card_grows_and_the_others_hold(self):
        """Stage 1, and the whole shape of the feature: ONE card changes.
        What the eye should track is a card getting bigger, not the
        composition rearranging itself -- an earlier cut shrank the
        neighbours into a side strip and it read as a re-layout instead."""
        td = tempfile.mkdtemp(prefix="focus_grow_")
        try:
            self._session(td, self._clicks(self.CARD_A, [6.0]))
            windows = [self.CARD_A, self.CARD_B, self.CARD_C]
            layout, painter = self._layout(td, windows)
            self.assertIsNotNone(layout)
            base = painter.base_placements
            cells, _order = layout.cells_at(int(6.0 * 30))
            self.assertGreater(self._area(cells[0]),
                               base[0][2] * base[0][3] * 1.3,
                               "the clicked card should have grown")
            for i in (1, 2):
                self.assertEqual(
                    (cells[i]["x"], cells[i]["y"], cells[i]["w"], cells[i]["h"]),
                    base[i], "card {} moved; the others must hold".format(i))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_stage_1_grows_the_card_without_zooming_the_screen(self):
        """The first rung. Only the layout changes -- no camera yet, which
        is what leaves the second click something bigger to do."""
        td = tempfile.mkdtemp(prefix="focus_s1_")
        try:
            self._session(td, self._clicks(self.CARD_A, [6.0]))
            windows = [self.CARD_A, self.CARD_B]
            layout, painter = self._layout(td, windows)
            self.assertIsNotNone(layout)
            i = int(6.0 * 30)
            cells, _o = layout.cells_at(i)
            self.assertGreater(self._area(cells[0]),
                               painter.base_placements[0][2]
                               * painter.base_placements[0][3] * 1.3)
            self.assertIsNone(layout.camera_at(i),
                              "one click must not zoom the screen")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_stage_2_zooms_the_whole_screen_in(self):
        """The second rung, and the one two rebuilds missed: after the card
        has grown, the WHOLE composition pushes in on it. Both mechanisms,
        in sequence -- not one or the other."""
        td = tempfile.mkdtemp(prefix="focus_s2_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            # THREE cards, like a real take: with only two, each cell is
            # already ~1:1 with its source rect and the upscale cap below
            # correctly leaves the camera nothing to do.
            windows = [self.CARD_A, self.CARD_B, self.CARD_C]
            layout, _p = self._layout(td, windows)
            self.assertIsNotNone(layout)
            cams = [layout.camera_at(i) for i in range(26 * 30)]
            zoomed = [c for c in cams if c is not None]
            self.assertTrue(zoomed, "stage 2 never zoomed the screen")
            peak = max(c[2] for c in zoomed)
            self.assertGreater(peak, 1.15, "the push-in should be visible")
            # It aims at the grown card, not the canvas centre.
            best = max(zoomed, key=lambda c: c[2])
            grown = layout.targets[0][0]
            self.assertLess(abs(best[0] - (grown[0] + grown[2] / 2.0)),
                            self.W * 0.1)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_the_screen_zoom_never_outruns_the_recorded_pixels(self):
        """The grown cell is already a real magnification of the source, so
        the camera on top of it is capped -- past `_FOCUS_MAX_UPSCALE` it is
        inventing detail the capture never had."""
        windows = [self.CARD_A, self.CARD_B, self.CARD_C]
        painter = framing.make_multi_painter(self.W, self.H, windows)
        e = np.zeros((4, 3))
        e[:, 0] = 1.0
        layout = render._FocusLayout(painter, e)
        cam = layout.camera_at(0)
        self.assertIsNotNone(cam)
        grown_w = layout.targets[0][0][2]
        src_w = painter.source_sizes[0][0]
        total = (grown_w * cam[2]) / float(src_w)
        self.assertLessEqual(total, render._FOCUS_MAX_UPSCALE + 1e-6)

    def test_the_second_click_escalates(self):
        """The headline behaviour: click once and it leans, click again and
        that card becomes the clear subject."""
        td = tempfile.mkdtemp(prefix="focus_esc_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            windows = [self.CARD_A, self.CARD_B]
            e = self._emphasis(td, windows)
            self.assertIsNotNone(e)
            lean = float(e[int(4.5 * 30), 0])
            full = float(e[:, 0].max())
            self.assertGreater(lean, 0.3)
            self.assertLess(lean, 0.7, "one click must not go all the way")
            self.assertGreater(full, 0.95, "the second click goes all the way")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_a_burst_at_the_very_end_declines_to_escalate(self):
        """Stage 2 still declines when it genuinely cannot play. `focus_hold`
        buys a two-click burst room to escalate, but it is clamped to the
        clip end -- a burst in the last second has nowhere to put the move
        and would lurch halfway in and stop. Same "no room, don't start it"
        posture `cluster_to_range` takes on a lone trailing click."""
        td = tempfile.mkdtemp(prefix="focus_room_")
        try:
            self._session(td, self._clicks(self.CARD_A, [24.6, 25.4]),
                          duration=26)
            e = self._emphasis(td, [self.CARD_A, self.CARD_B])
            self.assertIsNotNone(e)
            self.assertLess(float(e[:, 0].max()), 0.7,
                            "no room for stage 2, so it must stay at stage 1")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_only_one_card_is_ever_the_subject(self):
        """The guarantee `_arbitrate_card_ranges` buys. Two cards may briefly
        both be non-zero while one hands over to the other, but they must
        never both be committed at once."""
        td = tempfile.mkdtemp(prefix="focus_one_")
        try:
            events = (self._clicks(self.CARD_A, [3.0, 4.0, 5.0])
                      + self._clicks(self.CARD_B, [14.0, 15.0, 16.0]))
            self._session(td, events)
            e = self._emphasis(td, [self.CARD_A, self.CARD_B])
            self.assertIsNotNone(e)
            both = ((e > 0.5).sum(axis=1) > 1).sum()
            self.assertEqual(0, int(both),
                             "two cards were the subject at the same time")
            self.assertGreater(float(e[:, 0].max()), 0.9)
            self.assertGreater(float(e[:, 1].max()), 0.9)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_only_the_subject_ever_moves_and_nothing_leaves_the_canvas(self):
        """Swept over EVERY frame, because the failures this has actually
        shipped were only visible mid-animation and at a RESTING emphasis
        (stage 1 holds for seconds), never at the endpoints.

        Two invariants: a card that is not the subject is bit-identical to
        its base placement, and no card is ever pushed off the canvas by the
        growth (a card near an edge has to grow inward instead).
        """
        td = tempfile.mkdtemp(prefix="focus_vis_")
        try:
            events = (self._clicks(self.CARD_A, [4.0, 5.5, 7.0])
                      + self._clicks(self.CARD_B, [16.0, 17.5, 19.0]))
            self._session(td, events)
            windows = [self.CARD_A, self.CARD_B, self.CARD_C]
            layout, painter = self._layout(td, windows)
            self.assertIsNotNone(layout)
            base = painter.base_placements
            e = self._emphasis(td, windows)
            for i in range(0, 26 * 30, 3):
                cells, _order = layout.cells_at(i)
                for j, c in enumerate(cells):
                    self.assertGreaterEqual(c["x"], 0)
                    self.assertGreaterEqual(c["y"], 0)
                    self.assertLessEqual(c["x"] + c["w"], self.W)
                    self.assertLessEqual(c["y"] + c["h"], self.H)
                    if e[i, j] <= 1e-4:
                        self.assertEqual(
                            (c["x"], c["y"], c["w"], c["h"]), base[j],
                            "frame {}: card {} moved without being the "
                            "subject".format(i, j))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_a_card_at_the_canvas_edge_grows_inward(self):
        """Growth is about the card's own centre, so a card near an edge
        would run off it. It is clamped into the padding instead."""
        windows = [self.CARD_A, self.CARD_B, self.CARD_C]
        painter = framing.make_multi_painter(self.W, self.H, windows)
        for hero in range(3):
            for (x, y, w, h) in painter.focus_targets(hero):
                self.assertGreaterEqual(x, 0)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(x + w, self.W)
                self.assertLessEqual(y + h, self.H)

    def test_the_subject_is_painted_over_its_neighbours(self):
        """The overlap only reads as "in front" if the subject is drawn
        last; painted first it would look like a hole in the composition."""
        windows = [self.CARD_A, self.CARD_B, self.CARD_C]
        painter = framing.make_multi_painter(self.W, self.H, windows)
        _cells, order = framing.blend_placements(
            painter.base_placements,
            [painter.focus_targets(i) for i in range(3)], [0.0, 1.0, 0.0])
        self.assertEqual(1, order[-1])

    def test_every_card_keeps_its_aspect(self):
        """Focusing a window is meant to show MORE of it. `_fit_aspect` in
        every box means no card is ever cropped, at any emphasis."""
        windows = [self.CARD_A, self.CARD_B, self.CARD_C]
        painter = framing.make_multi_painter(self.W, self.H, windows)
        want = self.CARD_A["w"] / float(self.CARD_A["h"])
        for hero in range(3):
            for (_x, _y, w, h) in painter.focus_targets(hero):
                self.assertAlmostEqual(want, w / float(h), delta=0.05)

    def test_materialized_ranges_are_authoritative(self):
        """`edits.focus` is what the editor lets the user retime and delete,
        so a list -- even an empty one -- must suppress auto-planning. Same
        None-vs-[] contract `camera.build_path` uses for manual zooms; get it
        wrong and a deleted span comes back on the next render."""
        td = tempfile.mkdtemp(prefix="focus_manual_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            windows = [self.CARD_A, self.CARD_B]
            self.assertIsNotNone(self._emphasis(td, windows))
            self.assertIsNone(self._emphasis(td, windows, manual=[]),
                              "[] means the user deleted every span")
            e = self._emphasis(td, windows, manual=[
                {"start": 10.0, "end": 15.0, "card": 1, "level": "full"}])
            self.assertIsNotNone(e)
            self.assertLess(float(e[int(5.0 * 30), 0]), 0.02,
                            "the auto plan must not survive alongside it")
            self.assertGreater(float(e[int(13.0 * 30), 1]), 0.9)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_export_and_browser_get_the_same_cells(self):
        """The parity that justifies shipping placements instead of the
        emphasis track: `framing` stays the only place a layout is decided,
        so the paused server render and the playing browser canvas cannot
        disagree about where a card is."""
        td = tempfile.mkdtemp(prefix="focus_parity_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            windows = [self.CARD_A, self.CARD_B]
            painter = framing.make_multi_painter(self.W, self.H, windows)
            with redirect_stdout(io.StringIO()):
                shipped = render.multi_window_focus_cells(
                    td, windows, painter, window_focus=True, stride=1)
            self.assertIsNotNone(shipped)
            layout, _p = self._layout(td, windows)
            checked = 0
            for j, frame in enumerate(shipped["frames"]):
                cells, order = layout.cells_at(j)
                self.assertEqual(list(order), frame["o"])
                for c, got in zip(cells, frame["c"]):
                    self.assertEqual(
                        [c["x"], c["y"], c["w"], c["h"], c["radius"]], got)
                # Stage 2's camera has to travel too, or the browser would
                # show the grown card while the export showed it zoomed.
                cam = layout.camera_at(j)
                if cam is None:
                    self.assertIsNone(frame["z"])
                else:
                    self.assertIsNotNone(frame["z"])
                    for a, b in zip(cam, frame["z"]):
                        self.assertLess(abs(float(a) - float(b)), 0.1)
                checked += 1
            self.assertGreater(checked, 100)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_it_composes_with_per_card_zoom(self):
        """`window_zoom` zooms inside a card, `window_focus` re-weights which
        card the composition is about. Orthogonal, so both on must render."""
        td = tempfile.mkdtemp(prefix="focus_both_")
        try:
            self._session(td, self._clicks(self.CARD_A, [4.0, 5.5, 7.0]))
            windows = [self.CARD_A, self.CARD_B]
            with redirect_stdout(io.StringIO()):
                both = render.preview_frame(td, 6.5, windows=windows,
                                            window_focus=True,
                                            window_zoom=True)
                plain = render.preview_frame(td, 6.5, windows=windows)
            self.assertEqual(both.shape, plain.shape)
            self.assertFalse(np.array_equal(both, plain))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_renders_end_to_end(self):
        td = tempfile.mkdtemp(prefix="focus_e2e_")
        try:
            events = (self._clicks(self.CARD_A, [3.0, 4.0, 5.0])
                      + self._clicks(self.CARD_B, [14.0, 15.0, 16.0]))
            self._session(td, events, duration=20)
            out = os.path.join(td, "focus.mp4")
            with redirect_stdout(io.StringIO()):
                render.render(td, out, windows=[self.CARD_A, self.CARD_B],
                              window_focus=True, motion_blur=False,
                              click_fx=False)
            self.assertTrue(os.path.getsize(out) > 0)
            frames, _dur = _probe_frames_duration(out)
            self.assertEqual(600, frames)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class RetinaSourceSpaceScale(unittest.TestCase):
    """`_SourceSpace`-backed helpers must track windows in the SAME space the
    real render path does.

    The geometry samples in events.jsonl are in POINTS; `_build_grid_tracks`
    scales them to recorded pixels with `scale_x`/`scale_y`. Three helpers
    hardcoded `1.0, 1.0`, so on a Retina take (logical 1440x900, recorded
    2880x1800) every tracked window sat at HALF its true position -- cards
    followed their windows into the top-left quadrant and the clicks, which
    are scaled correctly, landed in none of them.

    Measured on a real 3-card take before the fix: 10 of 11 clicks
    unattributed, so the materialized focus plan was a single span on one
    card and never escalated. It silently affected the editor's live per-card
    zoom and synthetic cursor too -- both go through the same helpers.

    A same-resolution fixture cannot catch this (scale is 1.0 either way), so
    this one is deliberately Retina-shaped.
    """

    W, H = 1280, 720           # recorded pixels
    LOGICAL_W, LOGICAL_H = 640, 360   # points -- scale is 2.0

    def _session(self, td):
        """A window that sits in the RIGHT half in points, with clicks in it."""
        rect_pts = [320, 0, 320, 360]          # right half, in points
        events = []
        for t in np.arange(0.0, 12.0, 0.5):    # a geometry track for it
            events.append({"t": float(t), "type": "window", "id": 7,
                           "rect": list(rect_pts),
                           "x": 480.0, "y": 180.0})
        for t in (3.0, 4.5, 6.0):              # clicks inside it, in points
            events.append({"t": t, "type": "down", "x": 480.0, "y": 180.0})
        _mk_session(td, sorted(events, key=lambda e: e["t"]), duration=12,
                    fps=30, width=self.W, height=self.H)
        # _mk_session writes logical == pixels; make it Retina.
        path = os.path.join(td, "meta.json")
        with open(path) as f:
            meta = json.load(f)
        meta["logical_w"] = self.LOGICAL_W
        meta["logical_h"] = self.LOGICAL_H
        with open(path, "w") as f:
            json.dump(meta, f)

    def test_the_helper_path_attributes_clicks_like_the_render_path(self):
        td = tempfile.mkdtemp(prefix="focus_retina_")
        try:
            self._session(td)
            # The card, in SOURCE pixels: the right half of 1280x720.
            windows = [{"x": 640, "y": 0, "w": 640, "h": 720, "window_id": 7},
                       {"x": 0, "y": 0, "w": 640, "h": 720}]
            space = render._SourceSpace(td)
            self.assertTrue(space.ok)
            self.assertEqual(2.0, space.scale_x)
            self.assertEqual(2.0, space.scale_y)
            with redirect_stdout(io.StringIO()):
                plan = render.multi_window_focus_ranges(td, windows)
            self.assertTrue(plan, "no plan at all -- clicks went nowhere")
            self.assertTrue(all(r["card"] == 0 for r in plan),
                            "clicks landed in the wrong card: {}".format(plan))
        finally:
            shutil.rmtree(td, ignore_errors=True)


class FocusEscalationHasRoom(unittest.TestCase):
    """`focus_hold`: a cluster that earns stage 2 must be able to reach it.

    `tail` is 0.5s and `focus_full_min` is 1.0s, so before this a cluster
    whose second click was also its LAST could never escalate -- and "click
    twice and move on" is the common case, not an edge one. Measured on a
    real take: 6 spans planned, every one of them stage 1.
    """

    W, H = 640, 360
    A = {"x": 0, "y": 0, "w": 300, "h": 180}
    B = {"x": 320, "y": 180, "w": 300, "h": 180}

    def test_two_clicks_then_nothing_still_escalates(self):
        cards = [{"click_times": [5.0, 7.0]}, {"click_times": []}]
        plan = camera.plan_focus_ranges(cards, plan_duration=20.0)
        self.assertTrue(any(r["level"] == "full" for r in plan),
                        "a two-click burst must reach stage 2: {}".format(plan))

    def test_a_lone_click_still_does_not_escalate(self):
        cards = [{"click_times": [5.0]}, {"click_times": []}]
        plan = camera.plan_focus_ranges(cards, plan_duration=20.0)
        self.assertTrue(plan)
        self.assertFalse(any(r["level"] == "full" for r in plan))

    def test_the_hold_never_outlives_the_next_window(self):
        """The extension is applied BEFORE arbitration, so a later card
        claiming the screen still truncates it -- it can never hold a window
        past the point attention left it."""
        cards = [{"click_times": [5.0, 6.0]}, {"click_times": [8.0, 9.0]}]
        plan = camera.plan_focus_ranges(cards, plan_duration=20.0)
        spans = sorted((r["start"], r["end"], r["card"]) for r in plan)
        for i in range(len(spans) - 1):
            self.assertLessEqual(spans[i][1], spans[i + 1][0] + 1e-6,
                                 "spans overlap: {}".format(spans))


class FocusLayoutGeometry(unittest.TestCase):
    """`framing.focus_placements` / `blend_placements` on their own."""

    W, H = 1280, 720
    RECTS = [{"x": 0, "y": 0, "w": 400, "h": 300},
             {"x": 400, "y": 0, "w": 400, "h": 300}]

    def _painter(self, rects=None):
        return framing.MultiFramePainter(self.W, self.H, rects or self.RECTS)

    def test_focusing_a_lone_card_is_a_no_op(self):
        """With one card there is nothing to re-weight -- it already has the
        whole canvas -- so the focused layout IS the base layout and the
        animation never moves anything."""
        p = framing.MultiFramePainter(self.W, self.H, [self.RECTS[0]])
        self.assertEqual(list(p.focus_targets(0)), list(p.base_placements))

    def test_blend_at_zero_is_the_base_layout(self):
        p = self._painter()
        cells, order = framing.blend_placements(
            p.base_placements, [p.focus_targets(i) for i in range(2)],
            [0.0, 0.0])
        self.assertEqual(list(cells), list(p.base_placements))
        self.assertEqual(sorted(order), list(range(2)))

    def test_blend_at_one_is_the_focused_layout(self):
        p = self._painter()
        targets = [p.focus_targets(i) for i in range(2)]
        cells, order = framing.blend_placements(p.base_placements, targets,
                                                [1.0, 0.0])
        self.assertEqual(list(cells), list(targets[0]))
        self.assertEqual(order[-1], 0, "the subject must be painted last")

    def test_overlapping_weights_are_normalized(self):
        """Both springs are briefly non-zero while one card hands over to the
        next. Un-normalized that extrapolates past BOTH layouts and throws
        cards off the canvas."""
        p = self._painter()
        targets = [p.focus_targets(i) for i in range(2)]
        cells, _o = framing.blend_placements(p.base_placements, targets,
                                             [0.9, 0.9])
        for (x, y, w, h) in cells:
            self.assertGreaterEqual(x, 0)
            self.assertGreaterEqual(y, 0)
            self.assertLessEqual(x + w, self.W)
            self.assertLessEqual(y + h, self.H)

    def test_paint_at_the_base_layout_is_the_baked_paint(self):
        """The fallthrough that keeps an un-emphasized frame free."""
        p = self._painter()
        crops = [np.full((300, 400, 3), v, np.uint8) for v in (40, 200)]
        self.assertTrue(np.array_equal(p.paint(crops),
                                       p.paint_at(crops, p.base_placements)))

    def test_the_lowres_shadow_matches_the_baked_one(self):
        """`_draw_shadow` computes each card's shadow on a 1/8 grid (a
        full-canvas blur is 407ms). The blur is so wide that this is nearly
        lossless -- pin it, because a visibly different backdrop between a
        still frame and a moving one would read as a flicker.

        Compared at a ONE PIXEL nudge off the base layout, which is the
        smallest thing that defeats `paint_at`'s exact fallthrough: the two
        frames should be indistinguishable to the eye."""
        p = self._painter()
        crops = [np.full((300, 400, 3), v, np.uint8) for v in (40, 200)]
        baked = p.paint(crops)
        nudged = [(x + 1, y, w, h) for (x, y, w, h) in p.base_placements]
        live = p.paint_at(crops, nudged, order=[0, 1])
        # Ignore the 1px shift itself: compare only where both agree on what
        # is card and what is backdrop, i.e. away from the card edges.
        d = np.abs(baked.astype(np.int16) - live.astype(np.int16))
        for (x, y, w, h) in p.base_placements:
            d[max(0, y - 2):y + h + 2, max(0, x - 2):x + w + 2] = 0
        self.assertLess(float(d.mean()), 2.0)
        self.assertLess(int(d.max()), 14)


class LiftedCardShadow(unittest.TestCase):
    """A grown card has to read as being IN FRONT of the one it covers.

    The resting shadow cannot do that. It is wide on purpose (sigma = 0.02*W,
    58px on a 2880 canvas), which separates a card from the backdrop
    beautifully and separates it from a NEIGHBOUR not at all: measured on the
    real three windows, with card 0 fully grown over card 1, the neighbour
    pixel just outside the hero's edge was 198.4 against the hero's own 200 --
    1.6 levels, i.e. no boundary. Deepening that layer does not fix it either;
    58px of ramp is 58px of ramp however dark it gets.

    So `_shadow_layers` adds a tight contact layer once a card lifts, and
    everything is scaled by `lift_fraction` so a card at rest keeps exactly
    the shadow it had.
    """

    W, H = 2880, 1800

    def _painter(self):
        return _painter(self.W, self.H, _REAL_WINDOWS, "grid")

    @staticmethod
    def _crops(lumas):
        return [np.full((int(r["h"]), int(r["w"]), 3), v, np.uint8)
                for r, v in zip(_REAL_WINDOWS, lumas)]

    def _grown(self, p, hero):
        """`(image, hero cell)` with `hero` fully grown over its neighbours."""
        base = p.base_placements
        targets = [p.focus_targets(i) for i in range(len(base))]
        cells, order = framing.blend_placements(
            base, targets, [1.0 if i == hero else 0.0 for i in range(len(base))])
        return p.paint_at(self._crops((200, 235, 235)), cells,
                          order=order), cells[hero]

    def test_the_grown_card_is_separated_from_the_neighbour_it_covers(self):
        """The headline. Card 0 grown over card 1, both bright: the neighbour
        has to be visibly darkened right where the hero's edge crosses it,
        against the same neighbour further away."""
        p = self._painter()
        img, (hx, hy, hw, hh) = self._grown(p, 0)
        bx, by, bw, bh = p.base_placements[1]
        self.assertGreater(hx + hw, bx, "the fixture must actually overlap")
        # A band of the neighbour just outside the hero's right edge, and the
        # same neighbour far enough right to be past every shadow layer.
        near = img[max(hy, by) + 20:min(hy + hh, by + bh) - 20,
                   hx + hw + 4:hx + hw + 24]
        far = img[max(hy, by) + 20:min(hy + hh, by + bh) - 20,
                  bx + bw - 120:bx + bw - 20]
        self.assertGreater(float(far.mean()) - float(near.mean()), 40.0,
                           "the hero casts no shadow on the card it covers: "
                           "near {:.1f} vs far {:.1f}".format(float(near.mean()),
                                                              float(far.mean())))
        # And the boundary itself: the hero's own pixels against the neighbour
        # immediately outside them.
        inside = float(img[hy + 20:hy + hh - 20, hx + 20:hx + hw - 20].mean())
        self.assertGreater(inside - float(near.mean()), 30.0,
                           "no boundary between the hero and its neighbour")

    def test_a_card_at_rest_draws_exactly_the_resting_shadow(self):
        """Bit-exactness at lift 0, by construction: one layer, at the sigma,
        offset and alpha the baked backdrop uses. Everything the lift adds is
        gated on this being the only layer for an un-grown card -- which is
        also why `test_the_lowres_shadow_matches_the_baked_one` (a ONE PIXEL
        nudge, no growth) still compares against the baked plate."""
        p = self._painter()
        self.assertEqual(
            [(max(1.0, 0.02 * self.W), int(0.012 * self.H),
              framing._SHADOW_ALPHA, framing._SHADOW_DOWNSCALE)],
            p._shadow_layers(0.0))
        self.assertEqual(2, len(p._shadow_layers(1.0)),
                         "a lifted card needs the contact layer")

    def test_only_growth_lifts_a_card(self):
        """A card that shrinks (every demoted one, mid-hand-over) or merely
        MOVES keeps the resting shadow -- the lift is about being off the
        plane, not about the layout being in motion. The moving case is what
        `test_the_lowres_shadow_matches_the_baked_one` rests on: it nudges the
        cells 1px sideways, so their widths, and therefore their shadows, are
        untouched and the comparison against the baked plate stays fair."""
        p = self._painter()
        for j, (_x, _y, w, _h) in enumerate(p.base_placements):
            self.assertEqual(0.0, p.lift_of(j, w), "card {} at rest".format(j))
            self.assertEqual(0.0, p.lift_of(j, w - 60), "card {} demoted".format(j))
            self.assertAlmostEqual(1.0, p.lift_of(j, p.focus_widths[j]), places=6)
            # Growth is continuous, so a hair of it is a hair of shadow --
            # below the one level of 255 `_SHADOW_MIN_ALPHA` skips at.
            self.assertLess(p.lift_of(j, w + 1) * framing._CONTACT_ALPHA,
                            framing._SHADOW_MIN_ALPHA)

    def test_a_card_with_no_room_to_grow_never_lifts(self):
        """`focus_placements` returns the base layout untouched when the card
        already spans the padded canvas (docs/architecture.md's measured no-op stage 1).
        Nothing about that card moves, so nothing about its shadow should
        either -- and the fraction must not divide by zero deciding that."""
        self.assertEqual(0.0, framing.lift_fraction(800, 800, 800))
        self.assertEqual(0.0, framing.lift_fraction(800, 800, 900))
        feature = _painter(self.W, self.H, _REAL_WINDOWS, "feature")
        self.assertEqual(feature.focus_widths[0], feature.base_placements[0][2],
                         "the featured card has nowhere to grow -- fixture")
        self.assertEqual(0.0, feature.lift_of(0, feature.base_placements[0][2]))

    def test_the_lift_is_capped_at_the_focused_size(self):
        """Weights are normalized before the blend, so a cell should never
        exceed its target -- but the shadow is a look, not a geometry, and
        extrapolating it on a rounded pixel would be a visible pop."""
        p = self._painter()
        self.assertEqual(1.0, p.lift_of(0, p.focus_widths[0] * 3))


# ---- the two cameras under a non-grid `window_layout` -----------------------
#
# The three windows the multi-window feature was built for, at the origins
# `test_framing.InkCoverage` pins -- one tall editor and two wide panes. Shared
# through `arrangement_claims` so the whole tree measures ONE window set.
_REAL_WINDOWS = claims.REAL_WINDOWS

# That tall window's SHAPE twice, side by side on the desktop. Two tall
# windows are what make a `row` bind on the HEIGHT axis instead of the width --
# the case docs/architecture.md singles out, and the one where a row leaves no card any
# stage 1 at all.
_TWO_TALL = [dict(claims.REAL_WINDOWS[0]),
             {"x": 1500, "y": 0, "w": 1418, "h": 1718}]


def _stage_one(painter, hero):
    """Stage 1's growth factor for `hero`, and whether its cell moved at all.

    `focus_placements` returns the BASE layout untouched when the scale it
    computes is <= 1.0, so "did the placement change" and "what was the scale"
    are two different questions and both matter: the first is what the viewer
    sees, the second is why.
    """
    base = painter.base_placements
    grown = painter.focus_targets(hero)
    return (grown[hero][2] / float(base[hero][2]),
            tuple(grown[hero]) != tuple(base[hero]))


class FocusStageOneUnderEveryArrangement(unittest.TestCase):
    """How much room `window_layout` leaves the focus ladder's FIRST rung.

    `framing.focus_placements` grows the subject by `min(_FOCUS_GROW,
    usable_w/cell_w, usable_h/cell_h)` and returns the base layout untouched
    when that is <= 1.0 -- so a card whose cell already spans the padded canvas
    on either axis has nowhere to grow, the click that should have grown it
    does NOTHING VISIBLE, and nothing anywhere reports it. Filling an axis is
    precisely what the named arrangements do, which is why they turn an edge
    case into a systematic one.

    `docs/architecture.md` ships this as a measured table. A doc asserting
    measured behaviour with nothing pinning it is how the next refactor makes
    the doc a lie, so these are that table -- verified against the shipped code
    before they were written down, on the same three windows.

    On HALF-SIZE canvases where the table sweeps five layouts (a 2880x1800
    painter is ~1.8s of shadow bake, and the full five-layout sweep on it took
    8.9s): measured, 1440x900 reproduces 2880x1800's whole table to within
    0.002, because the arrangement geometry is scale-invariant apart from
    rounding. `test_the_documented_canvases_agree_with_the_half_size_ones`
    keeps that shortcut honest on the canvases docs/architecture.md actually names.
    """

    # (layout, W, H) -> stage 1's growth factor per card. 1.00 means the card
    # CANNOT grow and the first click does nothing to the layout at all.
    STAGE_ONE = {
        ("grid", 1440, 900): (1.40, 1.40, 1.40),
        ("desktop", 1440, 900): (1.02, 1.40, 1.40),
        ("feature", 1440, 900): (1.00, 1.40, 1.40),
        ("row", 1440, 900): (1.40, 1.40, 1.40),
        ("column", 1440, 900): (1.40, 1.40, 1.40),
        ("grid", 1080, 1920): (1.40, 1.01, 1.01),
        ("desktop", 1080, 1920): (1.40, 1.40, 1.40),
        ("feature", 1080, 1920): (1.00, 1.40, 1.40),
        ("row", 1080, 1920): (1.40, 1.40, 1.40),
        ("column", 1080, 1920): (1.34, 1.34, 1.34),
    }

    def test_the_measured_growth_table(self):
        for (layout, W, H), want in sorted(self.STAGE_ONE.items()):
            painter = _painter(W, H, _REAL_WINDOWS, layout)
            for hero, want_scale in enumerate(want):
                scale, moved = _stage_one(painter, hero)
                self.assertAlmostEqual(
                    scale, want_scale, delta=0.01,
                    msg="{} on {}x{}: card {} grows {:.3f}x, documented "
                        "{:.2f}x".format(layout, W, H, hero, scale, want_scale))
                # A 1.00 entry is not "grows by nothing" -- it is the early
                # return, and the placement has to come back byte-identical or
                # the card would jitter by a rounded pixel for free.
                self.assertEqual(moved, want_scale > 1.0,
                                 "{} on {}x{}: card {}".format(layout, W, H,
                                                               hero))

    def test_the_documented_canvases_agree_with_the_half_size_ones(self):
        """The two rows docs/architecture.md states on the full-size canvases, measured
        there rather than on the half-size stand-ins: focusing the FEATURED
        window has no stage 1 on a 2880x1800 export, and a `row` of two tall
        windows gets essentially none either.

        "Essentially" is load-bearing and was an exact 1.0 until the margin
        moved. The no-op is a KNIFE EDGE -- `focus_placements` early-returns
        only when the cell already spans the padded canvas -- so which side of
        it a given arrangement lands on is a function of PAD_FRAC, not of the
        windows. `row` gives every card unit height, which at the historical
        0.055 filled the padded height exactly (1.0, early return); at 0.03 it
        leaves 1.4% of slack, so the card grows by an amount no viewer can see
        and the placement nudges by a rounded pixel instead of coming back
        untouched. The claim worth pinning is "a row of tall windows gets no
        USEFUL first rung", so that is what is asserted; the exact-1.0 case is
        still pinned where it survives (feature, card 0) and swept in full by
        `test_the_no_op_is_exactly_the_cell_already_spanning_the_padding`.
        """
        painter = _painter(2880, 1800, _REAL_WINDOWS, "feature")
        scale, moved = _stage_one(painter, 0)
        self.assertAlmostEqual(scale, 1.0, delta=1e-9)
        self.assertFalse(moved, "the featured card should have nowhere to grow")
        for hero in (1, 2):
            self.assertAlmostEqual(_stage_one(painter, hero)[0], 1.4,
                                   delta=0.01)

        rows = _painter(2880, 1800, _TWO_TALL, "row")
        for hero in (0, 1):
            scale, _moved = _stage_one(rows, hero)
            self.assertLess(scale, 1.05,
                            "card {} of a row of tall windows grew "
                            "visibly after all".format(hero))

    def test_the_same_windows_keep_their_ladder_under_column(self):
        """Not a property of the WINDOWS -- a property of the arrangement. The
        two tall windows that get no useful stage 1 from `row` (which gives
        every card unit height, so a row narrow enough for the canvas makes
        every card essentially the padded height) grow the full 1.4x from
        `column`, which is the same statement with the axes swapped.

        The `row` side is asserted as a MAGNITUDE rather than as the early
        return it used to be: see
        `test_the_documented_canvases_agree_with_the_half_size_ones` for why
        that knife edge belongs to PAD_FRAC and not to these windows. 40x
        apart is the point, and that is what survives a margin change."""
        column = _painter(1440, 900, _TWO_TALL, "column")
        for hero in (0, 1):
            scale, moved = _stage_one(column, hero)
            self.assertAlmostEqual(scale, 1.4, delta=0.01)
            self.assertTrue(moved)
        row = _painter(1440, 900, _TWO_TALL, "row")
        for hero in (0, 1):
            self.assertLess(_stage_one(row, hero)[0], 1.05)

    def test_the_no_op_is_exactly_the_cell_already_spanning_the_padding(self):
        """The rule behind the table, swept rather than tabulated: stage 1 is a
        no-op for a card if and only if its cell already spans the padded
        canvas on an axis. Includes an unrecognized layout name (which grids)
        and 1-4 windows, because the arrangements are only where this becomes
        SYSTEMATIC -- `grid` and `desktop` reach it too.

        The second half is the invariant that makes stage 1 readable at all:
        whoever the subject is, no OTHER card moves.
        """
        sets = {1: _REAL_WINDOWS[:1], 2: _REAL_WINDOWS[:2], 3: _REAL_WINDOWS,
                4: _REAL_WINDOWS + [{"x": 200, "y": 1200, "w": 900, "h": 940}]}
        for W, H in ((640, 360), (360, 640), (480, 480)):
            for n in (1, 2, 3, 4):
                for layout in ("grid", "desktop", "feature", "row", "column",
                               "spiral"):
                    painter = _painter(W, H, sets[n], layout)
                    base = painter.base_placements
                    pad = painter._pad          # the painter's, never a copy
                    for hero in range(n):
                        grown = painter.focus_targets(hero)
                        spans = (base[hero][2] >= max(1, W - 2 * pad)
                                 or base[hero][3] >= max(1, H - 2 * pad))
                        self.assertEqual(
                            tuple(grown[hero]) == tuple(base[hero]), spans,
                            "{} n={} on {}x{}: card {} at {} vs padding "
                            "{}x{}".format(layout, n, W, H, hero, base[hero],
                                           W - 2 * pad, H - 2 * pad))
                        for j in range(n):
                            if j != hero:
                                self.assertEqual(
                                    tuple(grown[j]), tuple(base[j]),
                                    "{} n={} on {}x{}: card {} moved while "
                                    "{} was the subject".format(
                                        layout, n, W, H, j, hero))

    # (layout, W, H) -> stage 2's screen zoom per card, from `_FocusLayout`.
    STAGE_TWO = {
        ("feature", 2880, 1800): (1.05, 1.01, 1.01),
        ("grid", 1080, 1920): (1.47, 1.01, 1.01),
        ("feature", 1080, 1920): (1.01, 1.46, 1.47),
        ("column", 1080, 1920): (1.01, 1.01, 1.01),
    }

    def _layout(self, layout, W, H):
        painter = _painter(W, H, _REAL_WINDOWS, layout)
        # A constant emphasis; only `_FocusLayout`'s per-card destinations are
        # under test here, and those are computed once in its constructor.
        return render._FocusLayout(painter, np.zeros((4, len(_REAL_WINDOWS))))

    def test_stage_2_still_runs_where_stage_1_cannot(self):
        """Nothing else about the feature breaks when stage 1 has no room --
        the plan, the arbitration and the second rung are all unaffected. What
        is lost is one rung of a two-rung ladder, and it costs the ladder its
        WHOLE first half: 1.01x is not a zoom anyone will notice, and it is
        all that is left when the cell already spans the padded canvas.

        That residue SHRANK with the margin (it was 1.07x at the historical
        `pad_frac=0.055`), and the reason is worth stating rather than just
        re-tabulating: stage 2 has nothing to close on such a card except the
        padding itself, so a smaller margin makes the ladder's already-weak
        case weaker. It buys bigger cards at rest -- which is what the change
        was for -- and pays for it here.
        """
        for (layout, W, H), want in sorted(self.STAGE_TWO.items()):
            lay = self._layout(layout, W, H)
            for card, want_z in enumerate(want):
                self.assertAlmostEqual(
                    lay.cam[card][2], want_z, delta=0.01,
                    msg="{} on {}x{}: card {} pushes in {:.3f}x, documented "
                        "{:.2f}x".format(layout, W, H, card,
                                         lay.cam[card][2], want_z))
                self.assertGreater(lay.cam[card][2], 1.0,
                                   "stage 2 declined to run at all")

    def test_a_no_op_stage_1_leaves_only_the_padding_for_stage_2(self):
        """Stated as its own assertion because it is the honest cost: on a card
        that cannot grow, the camera has nothing left to close but the padding,
        so BOTH rungs are quiet. Measured against the same layout's cards that
        do grow, on one canvas, so it cannot pass by everything being small."""
        lay = self._layout("feature", 1080, 1920)
        painter = _painter(1080, 1920, _REAL_WINDOWS, "feature")
        self.assertFalse(_stage_one(painter, 0)[1])
        self.assertLess(lay.cam[0][2], 1.1)
        for card in (1, 2):
            self.assertTrue(_stage_one(painter, card)[1])
            self.assertGreater(lay.cam[card][2], 1.4)


class MultiWindowCamerasUnderANonGridLayout(unittest.TestCase):
    """`window_zoom` and `window_focus` through the REAL render path with
    `window_layout` set to something other than the grid.

    Both features key off cell geometry and every existing test of them runs on
    the default grid, so the arrangements were shipped without one frame of
    either being rendered under them.

    The windows are the three real ones at 1/6 scale: the harness records a
    640x360 testsrc2 pattern (no permissions, no real capture), so the rects
    have to fit inside it. The aspects are what decide every arrangement, and
    at this size they still reproduce the case that matters -- `feature` gives
    its hero no stage 1 here exactly as it does at 2880x1800.
    """

    W, H = 640, 360
    TALL = {"x": 0, "y": 0, "w": 236, "h": 286}
    WIDE = {"x": 250, "y": 0, "w": 236, "h": 142}
    WIDE2 = {"x": 250, "y": 150, "w": 236, "h": 142}
    LAYOUTS = ("grid", "desktop", "feature", "row", "column")
    DURATION = 12
    FPS = 30
    # Clicks in the TALL card. Three of them, so the ladder reaches stage 2;
    # the emphasis peaks at t=6.4s and is back to zero well before t=11.5.
    CLICKS = (4.0, 5.5, 7.0)
    PEAK, IDLE = 6.4, 11.5

    def _windows(self):
        return [self.TALL, self.WIDE, self.WIDE2]

    def _session(self, td):
        events = [{"t": t, "type": "down",
                   "x": self.TALL["x"] + self.TALL["w"] * 0.5,
                   "y": self.TALL["y"] + self.TALL["h"] * 0.5}
                  for t in self.CLICKS]
        _mk_session(td, events, duration=self.DURATION, fps=self.FPS,
                    width=self.W, height=self.H)

    def _emphasis(self, td, windows):
        from autocine import geometry
        ev = geometry.load_events(os.path.join(td, "events.jsonl"))
        n = self.DURATION * self.FPS
        return render._build_focus_emphasis(
            windows, None, ev["clicks_t"], ev["clicks_x"], ev["clicks_y"],
            np.arange(n) / float(self.FPS), self.W, self.H, 2.0, None, [],
            n / float(self.FPS), self.FPS, enabled=True)

    def _card_paths(self, td, windows, layout):
        from autocine import geometry
        ev = geometry.load_events(os.path.join(td, "events.jsonl"))
        n = self.DURATION * self.FPS
        painter = _painter(self.W, self.H, windows, layout)
        return render._build_card_cameras(
            windows, painter.cells, None,
            ev["clicks_t"], ev["clicks_x"], ev["clicks_y"],
            np.arange(n) / float(self.FPS), self.W, self.H, 2.0, None, [],
            n / float(self.FPS), self.FPS, enabled=True)

    def test_feature_grows_nothing_and_the_rest_still_runs(self):
        """The behaviour docs/architecture.md documents, on the render path rather than in
        the geometry: under `feature` the subject's cell is byte-identical to
        its base placement at EVERY frame of the take -- the first click is
        invisible -- while the emphasis plan and stage 2 run exactly as they do
        under the grid.
        """
        td = tempfile.mkdtemp(prefix="focus_layout_")
        try:
            self._session(td)
            windows = self._windows()
            e = self._emphasis(td, windows)
            self.assertIsNotNone(e)
            self.assertGreater(float(e[:, 0].max()), 0.95,
                               "the plan must reach stage 2 to make this a "
                               "test of the layout and not of the plan")
            moved, zoomed = {}, {}
            for layout in self.LAYOUTS:
                lay = render._FocusLayout(_painter(self.W, self.H, windows,
                                                   layout), e)
                base = lay.base
                moved[layout] = sum(
                    1 for i in range(len(e))
                    if (lambda c: (c["x"], c["y"], c["w"], c["h"]))(
                        lay.cells_at(i)[0][0]) != base[0])
                zoomed[layout] = sum(1 for i in range(len(e))
                                     if lay.camera_at(i) is not None)
            self.assertEqual(0, moved["feature"],
                             "the featured card grew after all")
            for layout in ("grid", "desktop", "row", "column"):
                self.assertGreater(moved[layout], 100,
                                   "{} lost stage 1 too".format(layout))
            # Stage 2 is the same plan under every arrangement -- `camera`
            # knows no geometry -- so it runs for the same frames everywhere.
            self.assertEqual(1, len(set(zoomed.values())), str(zoomed))
            self.assertGreater(zoomed["feature"], 0,
                               "stage 2 must still run where stage 1 cannot")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_an_unemphasized_frame_is_bit_exact_under_every_arrangement(self):
        """`paint_at`'s fallthrough is what keeps window focus free on the
        stretches it does nothing -- and it compares against `_placements`,
        which is whatever the arrangement produced. Pinned per layout because
        an arrangement that returned a different tuple TYPE would defeat it
        silently."""
        td = tempfile.mkdtemp(prefix="focus_layout_idle_")
        try:
            self._session(td)
            windows = self._windows()
            for layout in self.LAYOUTS:
                with redirect_stdout(io.StringIO()):
                    plain = render.preview_frame(td, self.IDLE, windows=windows,
                                                 window_layout=layout)
                    on = render.preview_frame(td, self.IDLE, windows=windows,
                                              window_layout=layout,
                                              window_focus=True)
                self.assertTrue(np.array_equal(plain, on),
                                "{}: an unemphasized frame was touched".format(
                                    layout))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_focus_changes_the_peak_frame_under_every_arrangement(self):
        """The other half of the one above: bit-exact when idle is worthless if
        the feature never does anything. Under `feature` the only thing left to
        see is stage 2, so this is also the pin that a no-op stage 1 does not
        make the whole feature a no-op."""
        td = tempfile.mkdtemp(prefix="focus_layout_peak_")
        try:
            self._session(td)
            windows = self._windows()
            for layout in self.LAYOUTS:
                with redirect_stdout(io.StringIO()):
                    plain = render.preview_frame(td, self.PEAK, windows=windows,
                                                 window_layout=layout)
                    on = render.preview_frame(td, self.PEAK, windows=windows,
                                              window_layout=layout,
                                              window_focus=True)
                self.assertFalse(np.array_equal(plain, on),
                                 "{}: window focus did nothing at its "
                                 "peak".format(layout))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_the_editor_gets_the_arrangements_cells_not_the_grids(self):
        """`multi_window_focus_cells` is the browser's copy of the moving
        layout and takes the painter, so it is the one focus payload that IS
        layout-dependent. Export and browser have to agree per arrangement or
        the live preview shows a grid while the export shows a feature."""
        td = tempfile.mkdtemp(prefix="focus_layout_parity_")
        try:
            self._session(td)
            windows = self._windows()
            lifted = {}
            for layout in ("feature", "column"):
                painter = _painter(self.W, self.H, windows, layout)
                with redirect_stdout(io.StringIO()):
                    shipped = render.multi_window_focus_cells(
                        td, windows, painter, window_focus=True, stride=1)
                self.assertIsNotNone(shipped)
                lay = render._FocusLayout(painter, self._emphasis(td, windows))
                lifted[layout] = 0
                for j, frame in enumerate(shipped["frames"]):
                    cells, order = lay.cells_at(j)
                    self.assertEqual(list(order), frame["o"])
                    for c, got in zip(cells, frame["c"]):
                        self.assertEqual(
                            [c["x"], c["y"], c["w"], c["h"], c["radius"]], got,
                            "{} frame {}".format(layout, j))
                    # The drop shadow is scaled by how far each card has
                    # grown, so the browser is sent that too -- computed by
                    # the painter here, never re-derived in JS. Absent means
                    # all-zero, which is the resting shadow.
                    want = [round(painter.lift_of(i, c["w"]), 3)
                            for i, c in enumerate(cells)]
                    self.assertEqual(want if any(want) else None,
                                     frame.get("l"),
                                     "{} frame {}".format(layout, j))
                    lifted[layout] += 1 if any(want) else 0
            # `column` grows every card 1.30x, so a run with no lift in it at
            # all would mean the parity above compared nothing but zeros.
            # `feature` is allowed none: this session's subject is the hero,
            # and the featured card has nowhere to grow (docs/architecture.md's table).
            self.assertGreater(lifted["column"], 0, str(lifted))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_the_per_card_cameras_do_not_depend_on_the_arrangement(self):
        """`window_zoom` plans in each card's own SOURCE space -- `cards` is
        built from `_clamp_window_rect`, and `_build_card_cameras` never reads
        the `cells` it is handed. So the plan is the same under every
        arrangement and only `_card_to_cell` differs. Pinned because it is easy
        to "fix" by planning against the cell, which would make a card's zoom
        depend on where the layout happened to put it."""
        td = tempfile.mkdtemp(prefix="windows_zoom_layout_")
        try:
            self._session(td)
            windows = self._windows()
            want = self._card_paths(td, windows, "grid")
            self.assertIsNotNone(want[0], "the clicked card must zoom")
            self.assertEqual([None, None], list(want[1:]))
            for layout in self.LAYOUTS[1:]:
                got = self._card_paths(td, windows, layout)
                self.assertTrue(
                    np.array_equal(got[0], want[0]),
                    "{} planned a different camera".format(layout))
                self.assertEqual([None, None], list(got[1:]), layout)
                # And the editor's served track with it, since it is the same
                # planner behind `multi_window_card_paths`.
                served = render.multi_window_card_paths(
                    td, windows, _painter(self.W, self.H, windows,
                                          layout).cells,
                    window_zoom=True, stride=1)
                self.assertIsNotNone(served)
                idx = int(round(self.PEAK * self.FPS))
                self.assertAlmostEqual(served["cards"][0]["z"][idx],
                                       float(want[0][idx, 2]), places=3)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_window_zoom_moves_only_the_clicked_card(self):
        """The two guarantees `window_zoom` carries on the grid, re-checked on
        each arrangement's own cells: the clicked card's pixels change, and a
        card the camera never picked goes through the untouched `paint()`
        resize rather than a zoom-1.0 warp."""
        td = tempfile.mkdtemp(prefix="windows_zoom_layout_px_")
        try:
            self._session(td)
            windows = self._windows()
            for layout in self.LAYOUTS:
                painter = _painter(self.W, self.H, windows, layout)
                with redirect_stdout(io.StringIO()):
                    off = render.preview_frame(td, self.PEAK, windows=windows,
                                               window_layout=layout)
                    on = render.preview_frame(td, self.PEAK, windows=windows,
                                              window_layout=layout,
                                              window_zoom=True)
                for i, cell in enumerate(painter.cells):
                    box = (slice(cell["y"], cell["y"] + cell["h"]),
                           slice(cell["x"], cell["x"] + cell["w"]))
                    same = np.array_equal(off[box], on[box])
                    self.assertEqual(
                        same, i != 0,
                        "{}: card {} {} changed".format(
                            layout, i, "should not have" if i else "should have"))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_zoom_1_framing_still_matches_the_plain_composite(self):
        """A card's zoom-1.0 window is the WHOLE card rect, never a cell-aspect
        sub-box -- otherwise turning `window_zoom` on centre-crops whenever a
        cell's aspect differs from its card's. The arrangements make that
        routine: a clamped (overhanging) rect keeps the painter's raw 5:3 while
        the crop render takes is 1:1, and each layout hands it a differently
        shaped cell (379x228 under feature, 154x92 under column)."""
        spec = {"x": 460, "y": 0, "w": 300, "h": 180}
        rects = [spec, self.WIDE, self.WIDE2]
        flat = np.zeros((5, 3))
        flat[:, 2] = 1.0
        _dx, _dy, dw, dh = render._clamp_window_rect(spec, self.W, self.H)
        flat[:, 0], flat[:, 1] = dw / 2.0, dh / 2.0
        for layout in self.LAYOUTS:
            cell = _painter(self.W, self.H, rects, layout).cells[0]
            with_cam = render._card_to_cell(spec, cell, flat, 0, self.W, self.H)
            without = render._card_to_cell(spec, cell, None, 0, self.W, self.H)
            for a, b in zip(with_cam, without):
                self.assertAlmostEqual(
                    a, b, places=6,
                    msg="{}: z=1.0 framing differs from the plain composite "
                        "({} vs {})".format(layout, with_cam, without))

    def test_both_cameras_render_end_to_end_under_a_non_grid_layout(self):
        td = tempfile.mkdtemp(prefix="windows_layout_e2e_")
        try:
            self._session(td)
            out = os.path.join(td, "feature.mp4")
            with redirect_stdout(io.StringIO()):
                render.render(td, out, windows=self._windows(),
                              window_layout="feature", window_zoom=True,
                              window_focus=True, motion_blur=False,
                              click_fx=False)
            frames, _dur = _probe_frames_duration(out)
            self.assertEqual(self.DURATION * self.FPS, frames)
            self.assertEqual(_probe_dims(out), (self.W, self.H))
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
