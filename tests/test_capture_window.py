"""Record-time window capture -> render-time crop.

avfoundation cannot target a window, so `record --window` captures the whole
display and snapshots the window's rect (POINTS, global top-left origin) into
`meta["capture_window"]`. render.py crops to it right after the points->pixels
scale is derived from the real file, rebinding (W, H) to the window size so the
camera, framing and every effect move into window space for free.

Two halves:
  * pure unit tests of `_capture_crop_px` / `_apply_capture_crop` (the
    coordinate formula, the even-dimension pin, every fail-safe path), and
  * end-to-end render/preview/camera_path/describe checks on the same
    synthetic testsrc2 session harness used by test_render_windows.py --
    no macOS permissions, no real recording.
"""

import json
import math
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import render


def _mk_session(root, events, duration=5, fps=30, width=640, height=360,
                logical_w=None, logical_h=None, capture_window=None):
    """Synthetic session; `capture_window` is written verbatim when not None."""
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
    _write_meta(root, fps=fps, duration=duration,
                logical_w=logical_w if logical_w is not None else width,
                logical_h=logical_h if logical_h is not None else height,
                capture_window=capture_window)


def _write_meta(root, fps, duration, logical_w, logical_h, capture_window=None):
    meta = {
        "raw": "raw.mov", "events": "events.jsonl",
        "fps": fps, "logical_w": logical_w, "logical_h": logical_h,
        "t0_monotonic": 0.0, "cursor_mode": "system",
        "duration": duration,
    }
    if capture_window is not None:
        meta["capture_window"] = capture_window
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)


def _cw(rect, display_origin=(0.0, 0.0), units="points", **extra):
    d = {"id": 9856, "app": "Google Chrome", "title": "t",
         "units": units, "rect": list(rect),
         "display_origin": list(display_origin),
         "source": "quartz", "resnapshot": True,
         "end_rect": list(rect)}
    d.update(extra)
    return d


def _meta(rect, logical_w, logical_h, **kw):
    return {"logical_w": logical_w, "logical_h": logical_h,
            "capture_window": _cw(rect, **kw)}


def _read_frames(path):
    """Every decoded frame of a rendered file, for bit-exactness compares."""
    import cv2
    cap = cv2.VideoCapture(path)
    out = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            out.append(frame)
    finally:
        cap.release()
    return out


def _probe_dims(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0", path], text=True).strip()
    w, h = out.split("x")
    return int(w), int(h)


# -- the coordinate formula -------------------------------------------------

class CropGeometry(unittest.TestCase):
    def test_retina_2x_is_exact(self):
        """logical 1440x900 recorded at 2880x1800 -> a clean 2x doubling."""
        meta = _meta([100, 50, 800, 600], 1440, 900)
        self.assertEqual(render._capture_crop_px(meta, 2880, 1800),
                         (200, 100, 1600, 1200))

    def test_fractional_scaling_integerizes_outward(self):
        """1512 logical points recorded at 2880px (1.904.. per point).

        Origins floor and far edges ceil so a fractional rect never shaves a
        fringe off the window; the inward even-ize may then hand back at most
        one pixel on one edge (here: x, 1714 vs 1714.28).
        """
        meta = _meta([100, 50, 800, 600], 1512, 945)
        crop = render._capture_crop_px(meta, 2880, 1800)
        self.assertEqual(crop, (190, 95, 1524, 1144))

        scale_x = 2880 / 1512.0
        scale_y = 1800 / 945.0
        x_px, y_px = 100 * scale_x, 50 * scale_y
        w_px, h_px = 800 * scale_x, 600 * scale_y
        x, y, w, h = crop
        self.assertEqual(x, int(math.floor(x_px)))
        self.assertEqual(y, int(math.floor(y_px)))
        # Outward on the far edges, modulo <=1px given back by the even-ize.
        self.assertGreaterEqual(x + w, int(math.ceil(x_px + w_px)) - 1)
        self.assertGreaterEqual(y + h, int(math.ceil(y_px + h_px)) - 1)

    def test_display_origin_is_subtracted(self):
        """A window on a display whose global origin isn't (0, 0) still maps
        to file-local pixels."""
        meta = _meta([1540, 100, 400, 300], 1440, 900,
                     display_origin=(1440.0, 0.0))
        self.assertEqual(render._capture_crop_px(meta, 1440, 900),
                         (100, 100, 400, 300))

    def test_negative_origin_clamps_to_frame(self):
        meta = _meta([-50, -30, 400, 300], 1440, 900)
        self.assertEqual(render._capture_crop_px(meta, 1440, 900),
                         (0, 0, 350, 270))

    def test_rect_past_right_and_bottom_edges_clamps(self):
        meta = _meta([1200, 700, 900, 900], 1440, 900)
        x, y, w, h = render._capture_crop_px(meta, 1440, 900)
        self.assertEqual((x, y), (1200, 700))
        self.assertLessEqual(x + w, 1440)
        self.assertLessEqual(y + h, 900)
        self.assertEqual((w % 2, h % 2), (0, 0))


class EvenDimensionPin(unittest.TestCase):
    """THE -video_size / libx264 pin.

    render._encode_cmd passes '-video_size {out_w}x{out_h}' into
    'libx264 -pix_fmt yuv420p', and framing.resolve_aspect_canvas passes the
    source dims straight through for aspect="auto" WITHOUT evening them. An
    odd crop would therefore break every export, so the crop must come out
    even -- and, because framing._even() rounds UP (which would overrun the
    frame), the even-ize here has to go INWARD.
    """

    def test_odd_pixel_size_is_evened_inward(self):
        meta = _meta([0, 0, 1439, 899], 1440, 900)
        self.assertEqual(render._capture_crop_px(meta, 1440, 900),
                         (0, 0, 1438, 898))

    def test_never_even_up_past_the_frame(self):
        for rect in ([1, 1, 1500, 1000], [0, 0, 1440, 900], [3, 5, 1437, 895],
                     [1438, 898, 500, 500], [0.5, 0.5, 1439.5, 899.5],
                     [-0.5, -0.5, 1441, 901], [719.5, 449.5, 720.5, 450.5]):
            meta = _meta(rect, 1440, 900)
            crop = render._capture_crop_px(meta, 1440, 900)
            self.assertIsNotNone(crop, rect)
            x, y, w, h = crop
            self.assertEqual(w % 2, 0, rect)
            self.assertEqual(h % 2, 0, rect)
            self.assertGreaterEqual(x, 0, rect)
            self.assertGreaterEqual(y, 0, rect)
            self.assertLessEqual(x + w, 1440, rect)
            self.assertLessEqual(y + h, 900, rect)
            self.assertGreaterEqual(w, 2, rect)
            self.assertGreaterEqual(h, 2, rect)


class CropFailsSafe(unittest.TestCase):
    """Anything unexpected must return None -- i.e. today's full-frame
    render, bit-exact -- never a half-trusted crop."""

    def test_key_absent(self):
        self.assertIsNone(render._capture_crop_px(
            {"logical_w": 1440, "logical_h": 900}, 1440, 900))

    def test_not_a_dict(self):
        for bad in ("nope", [100, 50, 800, 600], 7, None):
            meta = {"logical_w": 1440, "logical_h": 900, "capture_window": bad}
            self.assertIsNone(render._capture_crop_px(meta, 1440, 900), bad)

    def test_units_not_points(self):
        for units in ("pixels", "", None, "Points"):
            meta = _meta([100, 50, 800, 600], 1440, 900, units=units)
            self.assertIsNone(render._capture_crop_px(meta, 1440, 900), units)

    def test_malformed_rect(self):
        for rect in ([100, 50, 800], [100, 50, 800, 600, 1], "1,2,3,4",
                     [None, 50, 800, 600], ["a", "b", "c", "d"]):
            meta = _meta(rect, 1440, 900)
            self.assertIsNone(render._capture_crop_px(meta, 1440, 900), rect)
        meta = {"logical_w": 1440, "logical_h": 900,
                "capture_window": {"units": "points"}}
        self.assertIsNone(render._capture_crop_px(meta, 1440, 900))

    def test_malformed_display_origin(self):
        for origin in ((0.0, 0.0, 0.0), "0,0", (float("nan"), 0.0)):
            meta = _meta([100, 50, 800, 600], 1440, 900, display_origin=origin)
            self.assertIsNone(render._capture_crop_px(meta, 1440, 900), origin)

    def test_non_finite_numbers(self):
        for rect in ([float("nan"), 50, 800, 600],
                     [100, 50, float("inf"), 600],
                     [100, float("-inf"), 800, 600]):
            meta = _meta(rect, 1440, 900)
            self.assertIsNone(render._capture_crop_px(meta, 1440, 900), rect)

    def test_below_min_dim(self):
        meta = _meta([100, 50, 4, 300], 1440, 900)
        self.assertIsNone(render._capture_crop_px(meta, 1440, 900))
        meta = _meta([100, 50, 300, 4], 1440, 900)
        self.assertIsNone(render._capture_crop_px(meta, 1440, 900))
        # ... but a window just over the floor is honored.
        meta = _meta([100, 50, 20, 20], 1440, 900)
        self.assertIsNotNone(render._capture_crop_px(meta, 1440, 900))

    def test_no_overlap_with_the_frame(self):
        """A rect entirely off-frame is what a SECONDARY-display capture
        looks like (record.py stores the MAIN display's logical size
        regardless of --display). Multi-display is out of scope for v1, so
        this must fail safe to the full frame rather than crop garbage."""
        for rect in ([5000, 0, 400, 300], [-1000, 0, 400, 300],
                     [0, 5000, 400, 300], [0, -1000, 400, 300]):
            meta = _meta(rect, 1440, 900)
            self.assertIsNone(render._capture_crop_px(meta, 1440, 900), rect)

    def test_degenerate_frame(self):
        meta = _meta([0, 0, 400, 300], 1440, 900)
        self.assertIsNone(render._capture_crop_px(meta, 1, 900))
        self.assertIsNone(render._capture_crop_px(meta, 1440, 0))


class ApplyCropIdentity(unittest.TestCase):
    def test_none_returns_the_same_object(self):
        """The off path must be LITERAL, not a copy that happens to compare
        equal -- that's what makes 'no capture_window renders byte-identically
        to today' true by construction."""
        fr = np.zeros((10, 20, 3), np.uint8)
        self.assertIs(render._apply_capture_crop(fr, None), fr)

    def test_crop_returns_a_view_not_a_copy(self):
        fr = np.zeros((10, 20, 3), np.uint8)
        sub = render._apply_capture_crop(fr, (4, 2, 6, 4))
        self.assertEqual(sub.shape, (4, 6, 3))
        self.assertIsNotNone(sub.base)
        sub[0, 0, 0] = 200
        self.assertEqual(fr[2, 4, 0], 200)


class TypingAnchorOffset(unittest.TestCase):
    """camera.build_path ignores typing_anchors today, so this is inert --
    but they're the one coordinate stream that doesn't ride the event scale
    path, and vision's LRU hands back a SHARED list that must not be mutated."""

    def test_anchors_translate_and_are_copied(self):
        anchors = [{"start": 1.0, "end": 2.0, "x": 300.0, "y": 200.0}, None]
        out = render._offset_typing_anchors(anchors, (100, 50, 320, 180))
        self.assertEqual(out[0]["x"], 200.0)
        self.assertEqual(out[0]["y"], 150.0)
        self.assertEqual(out[0]["start"], 1.0)
        self.assertIsNone(out[1])
        self.assertEqual(anchors[0]["x"], 300.0)   # cache untouched

    def test_no_crop_passes_through(self):
        anchors = [{"start": 1.0, "end": 2.0, "x": 300.0, "y": 200.0}]
        self.assertIs(render._offset_typing_anchors(anchors, None), anchors)
        self.assertIsNone(render._offset_typing_anchors(None, (1, 2, 3, 4)))


class CaptureWindowMoved(unittest.TestCase):
    def test_identical_end_rect_is_not_moved(self):
        self.assertFalse(render._capture_window_moved(_cw([10, 20, 300, 200])))

    def test_small_jitter_is_not_moved(self):
        cw = _cw([10, 20, 300, 200], end_rect=[12, 21, 301, 200])
        self.assertFalse(render._capture_window_moved(cw))

    def test_shift_beyond_tolerance_is_moved(self):
        cw = _cw([10, 20, 300, 200], end_rect=[40, 20, 300, 200])
        self.assertTrue(render._capture_window_moved(cw))

    def test_null_end_rect_means_the_window_closed(self):
        self.assertTrue(render._capture_window_moved(
            _cw([10, 20, 300, 200], end_rect=None)))

    def test_garbage_end_rect_is_moved(self):
        for end in ("x", [1, 2, 3], [1, 2, 3, float("nan")]):
            cw = _cw([10, 20, 300, 200], end_rect=end)
            self.assertTrue(render._capture_window_moved(cw), end)


# -- end to end -------------------------------------------------------------

class OffSwitchBitExact(unittest.TestCase):
    """A session with no `capture_window` must render byte-identically to
    today. Same within-one-run comparison as test_render_windows: absent vs
    present-but-unresolvable both take the `crop is None` branch, whose
    coordinate expression is literally the pre-feature `ax * scale_x`."""

    def test_absent_and_unresolvable_produce_identical_bytes(self):
        td = tempfile.mkdtemp(prefix="capwin_off_")
        try:
            events = [{"t": 1.0, "type": "down", "x": 100, "y": 100}]
            _mk_session(td, events, duration=3)
            out_absent = os.path.join(td, "absent.mp4")
            render.render(td, out_path=out_absent, motion_blur=False,
                          click_fx=False)
            # Same session, but with a capture_window the crop must reject.
            _write_meta(td, fps=30, duration=3, logical_w=640, logical_h=360,
                        capture_window=_cw([100, 50, 300, 200], units="pixels"))
            out_bad = os.path.join(td, "bad.mp4")
            render.render(td, out_path=out_bad, motion_blur=False,
                          click_fx=False)
            with open(out_absent, "rb") as f:
                b_absent = f.read()
            with open(out_bad, "rb") as f:
                b_bad = f.read()
            self.assertEqual(b_absent, b_bad)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_preview_frame_off_path_is_identical(self):
        td = tempfile.mkdtemp(prefix="capwin_off_prev_")
        try:
            events = [{"t": 1.0, "type": "down", "x": 100, "y": 100}]
            _mk_session(td, events, duration=2)
            a = render.preview_frame(td, 1.0)
            _write_meta(td, fps=30, duration=2, logical_w=640, logical_h=360,
                        capture_window=_cw([100, 50, 300, 200], units="pixels"))
            b = render.preview_frame(td, 1.0)
            self.assertTrue((a == b).all())
        finally:
            shutil.rmtree(td, ignore_errors=True)


class CroppedRender(unittest.TestCase):
    def test_output_dims_match_the_crop_and_bytes_differ(self):
        td = tempfile.mkdtemp(prefix="capwin_render_")
        try:
            events = [{"t": 1.0, "type": "down", "x": 200, "y": 120}]
            _mk_session(td, events, duration=2)
            out_full = os.path.join(td, "full.mp4")
            render.render(td, out_path=out_full, motion_blur=False,
                          click_fx=False)
            _write_meta(td, fps=30, duration=2, logical_w=640, logical_h=360,
                        capture_window=_cw([160, 90, 320, 180]))
            out_crop = os.path.join(td, "crop.mp4")
            render.render(td, out_path=out_crop, motion_blur=False,
                          click_fx=False)
            self.assertEqual(_probe_dims(out_full), (640, 360))
            self.assertEqual(_probe_dims(out_crop), (320, 180))
            with open(out_full, "rb") as f:
                b_full = f.read()
            with open(out_crop, "rb") as f:
                b_crop = f.read()
            self.assertNotEqual(b_full, b_crop)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_odd_rect_still_encodes(self):
        """Regression for the -video_size pin: a rect whose pixel size is odd
        must not reach libx264 odd."""
        td = tempfile.mkdtemp(prefix="capwin_odd_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([0, 0, 321, 181]))
            out = os.path.join(td, "odd.mp4")
            render.render(td, out_path=out, motion_blur=False, click_fx=False)
            self.assertEqual(_probe_dims(out), (320, 180))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_cropped_vertical_aspect_encodes(self):
        td = tempfile.mkdtemp(prefix="capwin_aspect_")
        try:
            events = [{"t": 0.5, "type": "down", "x": 200, "y": 120}]
            _mk_session(td, events, duration=1,
                        capture_window=_cw([160, 90, 321, 181]))
            out = os.path.join(td, "vert.mp4")
            render.render(td, out_path=out, aspect="9:16", motion_blur=False,
                          click_fx=False)
            w, h = _probe_dims(out)
            self.assertEqual((w % 2, h % 2), (0, 0))
            self.assertLess(w, h)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_headline_effects_survive_the_crop(self):
        """The whole reason the edits.windows compositor route was rejected:
        it force-disables click FX / spotlight / cursor FX / motion blur.
        A window-captured session must keep the product's headline effect."""
        td = tempfile.mkdtemp(prefix="capwin_fx_")
        try:
            events = [
                {"t": 1.0, "type": "down", "x": 200, "y": 120},
                {"t": 1.1, "type": "up", "x": 200, "y": 120},
                {"t": 1.5, "type": "down", "x": 210, "y": 125},
                {"t": 1.6, "type": "up", "x": 210, "y": 125},
            ]
            _mk_session(td, events, duration=3,
                        capture_window=_cw([160, 90, 320, 180]))
            out = render.preview_frame(td, 1.6, motion_blur=True,
                                       click_fx=True, spotlight=True)
            self.assertEqual(out.shape, (180, 320, 3))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_edits_windows_compose_inside_the_crop(self):
        """The source crop applies FIRST; multi-window rects are then
        interpreted (and clamped) inside the cropped frame."""
        td = tempfile.mkdtemp(prefix="capwin_multi_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([160, 90, 320, 180]))
            out = os.path.join(td, "multi.mp4")
            windows = [{"x": 0, "y": 0, "w": 900, "h": 900}]   # bigger than crop
            render.render(td, out_path=out, motion_blur=False,
                          click_fx=False, windows=windows)
            self.assertEqual(_probe_dims(out), (320, 180))
        finally:
            shutil.rmtree(td, ignore_errors=True)


class CroppedCameraPath(unittest.TestCase):
    def test_payload_geometry_and_in_bounds_path(self):
        td = tempfile.mkdtemp(prefix="capwin_campath_")
        try:
            events = [
                {"t": 1.0, "type": "down", "x": 200, "y": 120},
                {"t": 1.5, "type": "down", "x": 205, "y": 125},
            ]
            _mk_session(td, events, duration=4,
                        capture_window=_cw([160, 90, 320, 180]))
            p = render.camera_path(td, stride=2)
            self.assertEqual(p["source"], [320, 180])
            self.assertEqual(p["raw_source"], [640, 360])
            self.assertEqual(p["crop"], [160, 90, 320, 180])
            self.assertTrue(p["cx"])
            for cx, cy in zip(p["cx"], p["cy"]):
                self.assertGreaterEqual(cx, 0.0)
                self.assertLessEqual(cx, 320.0)
                self.assertGreaterEqual(cy, 0.0)
                self.assertLessEqual(cy, 180.0)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_crop_is_null_without_a_capture_window(self):
        td = tempfile.mkdtemp(prefix="capwin_campath_off_")
        try:
            _mk_session(td, [], duration=2)
            p = render.camera_path(td, stride=4)
            self.assertIsNone(p["crop"])
            self.assertEqual(p["source"], [640, 360])
            self.assertEqual(p["raw_source"], [640, 360])
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_click_zooms_toward_the_translated_coordinate(self):
        """Sign guard on to_src(): a click at point (180, 100) sits at raw px
        (180, 100) and therefore at (20, 10) INSIDE the crop -- top-left. Had
        the crop offset been added instead of subtracted it would land at
        (340, 190), outside the 320x180 window, and the camera would clamp to
        the opposite corner."""
        td = tempfile.mkdtemp(prefix="capwin_sign_")
        try:
            events = [
                {"t": 1.0, "type": "down", "x": 180, "y": 100},
                {"t": 1.4, "type": "down", "x": 182, "y": 102},
                {"t": 1.8, "type": "down", "x": 181, "y": 101},
            ]
            _mk_session(td, events, duration=5,
                        capture_window=_cw([160, 90, 320, 180]))
            p = render.camera_path(td, stride=1)
            z = np.asarray(p["z"], dtype=float)
            self.assertGreater(float(z.max()), 1.05)   # it did zoom
            k = int(np.argmax(z))
            self.assertLess(p["cx"][k], 160.0)         # left half of the crop
            self.assertLess(p["cy"][k], 90.0)          # top half of the crop
        finally:
            shutil.rmtree(td, ignore_errors=True)


class DescribeSessionCrop(unittest.TestCase):
    def test_reports_cropped_dims_and_raw_dims(self):
        td = tempfile.mkdtemp(prefix="capwin_desc_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([160, 90, 320, 180]))
            d = render.describe_session(td)
            self.assertEqual((d["width"], d["height"]), (320, 180))
            self.assertEqual((d["raw_width"], d["raw_height"]), (640, 360))
            self.assertEqual(d["capture_window"]["rect"], [160, 90, 320, 180])
            self.assertFalse(d["capture_window"]["moved"])
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_moved_window_is_flagged(self):
        td = tempfile.mkdtemp(prefix="capwin_desc_moved_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([160, 90, 320, 180],
                                           end_rect=[260, 90, 320, 180]))
            self.assertTrue(render.describe_session(td)["capture_window"]["moved"])
            _write_meta(td, fps=30, duration=1, logical_w=640, logical_h=360,
                        capture_window=_cw([160, 90, 320, 180], end_rect=None))
            self.assertTrue(render.describe_session(td)["capture_window"]["moved"])
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_plain_session_is_unchanged(self):
        td = tempfile.mkdtemp(prefix="capwin_desc_off_")
        try:
            _mk_session(td, [], duration=1)
            d = render.describe_session(td)
            self.assertEqual((d["width"], d["height"]), (640, 360))
            self.assertEqual((d["raw_width"], d["raw_height"]), (640, 360))
            self.assertIsNone(d["capture_window"])
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_applied_flag_distinguishes_a_rejected_rect(self):
        """`capture_window` non-null does NOT mean the crop happened. The
        editor needs the difference stated, not inferred: a rejected rect
        renders full-frame and must not show a 'Window' chip."""
        td = tempfile.mkdtemp(prefix="capwin_applied_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([160, 90, 320, 180]))
            d = render.describe_session(td)
            self.assertTrue(d["capture_window"]["applied"])
            self.assertEqual((d["width"], d["height"]), (320, 180))

            # Rejected: wrong units -> full frame, block still reported.
            _write_meta(td, fps=30, duration=1, logical_w=640, logical_h=360,
                        capture_window=_cw([160, 90, 320, 180], units="pixels"))
            d = render.describe_session(td)
            self.assertIsNotNone(d["capture_window"])
            self.assertFalse(d["capture_window"]["applied"])
            self.assertEqual((d["width"], d["height"]), (640, 360))

            # Rejected: rect on a secondary display (no overlap with the frame).
            _write_meta(td, fps=30, duration=1, logical_w=640, logical_h=360,
                        capture_window=_cw([900, 50, 320, 180]))
            d = render.describe_session(td)
            self.assertFalse(d["capture_window"]["applied"])
            self.assertEqual((d["width"], d["height"]), (640, 360))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_applied_is_true_for_a_window_covering_the_display(self):
        """The exact case a raw-vs-cropped dimension compare gets wrong: the
        crop is a no-op in size, but it DID resolve."""
        td = tempfile.mkdtemp(prefix="capwin_full_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([0, 0, 640, 360]))
            d = render.describe_session(td)
            self.assertEqual((d["width"], d["height"]), (640, 360))
            self.assertEqual((d["raw_width"], d["raw_height"]), (640, 360))
            self.assertTrue(d["capture_window"]["applied"])
        finally:
            shutil.rmtree(td, ignore_errors=True)


def _native_cw(rect=(695.0, 34.0, 745.0, 459.0), logical_w=745.0,
               logical_h=459.0, track="ok", **kw):
    """A window-native capture_window block (raw.mov IS the window)."""
    cw = _cw(list(rect), mode="window_native", id=4242, track=track,
             buffer_w=int(logical_w * 2), buffer_h=int(logical_h * 2), **kw)
    cw["logical_w"], cw["logical_h"] = logical_w, logical_h
    return cw


def _native_meta(**kw):
    # Top-level logical_w/h stay the DISPLAY; the window's own size lives on the
    # capture_window block.
    return {"logical_w": 1440.0, "logical_h": 900.0,
            "capture_window": _native_cw(**kw)}


class WindowNativeCropBypass(unittest.TestCase):
    """render must NOT crop a window-native take -- raw.mov already IS the
    window, so `_capture_crop_px` returns None and (W,H) stays the buffer."""

    def test_native_meta_is_never_cropped(self):
        # Same rect, cropped when it's a display-crop take, NOT when native.
        self.assertIsNotNone(render._capture_crop_px(
            _meta([695, 34, 745, 459], 1440, 900), 2880, 1800))
        self.assertIsNone(render._capture_crop_px(_native_meta(), 1490, 918))

    def test_display_crop_take_is_unaffected(self):
        # The guard keys on mode, so an ordinary capture_window still crops.
        crop = render._capture_crop_px(_meta([0, 0, 320, 180], 640, 360),
                                       640, 360)
        self.assertEqual(crop, (0, 0, 320, 180))


class NativeWindowTrackTransform(unittest.TestCase):
    """The event->buffer mapping: (p - origin(t)) * scale * fit(t), top-left
    anchored, clamped. scale=2.0 (Retina), buffer 1490x918, window 745x459 pt."""

    def _track(self, rects_pt, times=(0.0, 1.0), scale=2.0):
        return render._NativeWindowTrack(rects_pt, 1490, 918,
                                         np.asarray(times, float), scale)

    def test_apply_is_identity(self):
        tr = self._track([[0, 0, 745, 459], [0, 0, 745, 459]])
        frame = np.zeros((4, 4, 3), np.uint8)
        self.assertIs(tr.apply(frame, 0), frame)

    def test_corners_and_centre_at_the_start_size(self):
        # Static window at global origin (695, 34). scale 2.0, no resize.
        tr = self._track([[695, 34, 745, 459], [695, 34, 745, 459]])
        t = np.array([0.5])
        # top-left of the window -> (0, 0)
        x, y = tr.to_src(t, np.array([695.0]), np.array([34.0]))
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y[0], 0.0)
        # bottom-right corner -> the buffer corner
        x, y = tr.to_src(t, np.array([695.0 + 745.0]), np.array([34.0 + 459.0]))
        self.assertAlmostEqual(x[0], 1490.0)
        self.assertAlmostEqual(y[0], 918.0)
        # centre -> buffer centre
        x, y = tr.to_src(t, np.array([695.0 + 372.5]), np.array([34.0 + 229.5]))
        self.assertAlmostEqual(x[0], 745.0)
        self.assertAlmostEqual(y[0], 459.0)

    def test_out_of_window_clicks_clamp_and_are_not_dropped(self):
        tr = self._track([[695, 34, 745, 459], [695, 34, 745, 459]])
        t = np.array([0.0, 0.0])
        # one far left/up, one far right/down -> both survive, clamped in-bounds
        x, y = tr.to_src(t, np.array([-500.0, 5000.0]), np.array([-500.0, 5000.0]))
        self.assertEqual(len(x), 2)                 # count preserved (time set!)
        self.assertEqual((x[0], y[0]), (0.0, 0.0))
        self.assertEqual((x[1], y[1]), (1490.0, 918.0))

    def test_origin_follows_a_moving_window(self):
        # Window drags from (0,0) to (100,50) over one second.
        tr = self._track([[0, 0, 745, 459], [100, 50, 745, 459]])
        # a click that tracks the window's top-left stays at (0,0) throughout
        x0, y0 = tr.to_src(np.array([0.0]), np.array([0.0]), np.array([0.0]))
        x1, y1 = tr.to_src(np.array([1.0]), np.array([100.0]), np.array([50.0]))
        xm, ym = tr.to_src(np.array([0.5]), np.array([50.0]), np.array([25.0]))
        for v in (x0[0], y0[0], x1[0], y1[0], xm[0], ym[0]):
            self.assertAlmostEqual(v, 0.0, places=3)

    def test_letterbox_fit_after_an_aspect_preserving_resize(self):
        # Window grows to 2x its point size; SCK fits it into the SAME buffer,
        # so the fit factor halves and the (now larger) window still spans the
        # buffer corner-to-corner.
        tr = self._track([[0, 0, 745, 459], [0, 0, 1490, 918]])
        t = np.array([1.0])
        x, y = tr.to_src(t, np.array([1490.0]), np.array([918.0]))   # bottom-right
        self.assertAlmostEqual(x[0], 1490.0, places=2)
        self.assertAlmostEqual(y[0], 918.0, places=2)
        xc, yc = tr.to_src(t, np.array([745.0]), np.array([459.0]))  # centre
        self.assertAlmostEqual(xc[0], 745.0, places=2)
        self.assertAlmostEqual(yc[0], 459.0, places=2)


class NativeTrackBuild(unittest.TestCase):
    """`_build_window_track` returns a native mapper for a native take even
    though `crop` is None, and falls back to the static rect without a track."""

    def _build(self, ev, meta):
        return render._build_window_track(
            ev, None, 1490, 918, 1.0, 1.0, np.array([0.0, 1.0]),
            lambda a: a, 60, window_id=4242, meta=meta)

    def test_static_fallback_without_a_geometry_track(self):
        tr = self._build({}, _native_meta())
        self.assertIsInstance(tr, render._NativeWindowTrack)
        # maps off the meta rect origin (695, 34)
        x, y = tr.to_src(np.array([0.0]), np.array([695.0]), np.array([34.0]))
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y[0], 0.0)

    def test_a_normal_session_still_builds_nothing(self):
        # No capture_window at all -> None, exactly as before (off switch).
        self.assertIsNone(self._build({}, {"logical_w": 1440, "logical_h": 900}))


class NativeCameraPath(unittest.TestCase):
    """End-to-end through render.camera_path: a native take is NOT cropped
    (raw IS the window), and a click at a global point zooms toward its
    WINDOW-LOCAL position -- the whole point of the coordinate transform."""

    def _session(self, td, events):
        # 640x360 synthetic "window buffer"; window is 320x180 pt at global
        # (100, 50) -> scale 2.0. Top-level logical is the (unrelated) display.
        _mk_session(td, events, duration=5, width=640, height=360,
                    logical_w=1440, logical_h=900,
                    capture_window=_native_cw(rect=(100.0, 50.0, 320.0, 180.0),
                                              logical_w=320.0, logical_h=180.0))

    def test_native_take_is_not_cropped(self):
        td = tempfile.mkdtemp(prefix="capwin_native_nocrop_")
        try:
            self._session(td, [])
            p = render.camera_path(td, stride=4)
            self.assertIsNone(p["crop"])                  # no display crop
            self.assertEqual(p["source"], [640, 360])     # source == the buffer
            self.assertEqual(p["raw_source"], [640, 360])
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_click_zooms_toward_the_window_local_coordinate(self):
        # Clicks near the window's TOP-LEFT: global (120, 60) -> local (20, 10)
        # pt -> buffer (40, 20). The camera must aim into the top-left quadrant.
        # A wrong-sign or display-scaled transform would land elsewhere.
        td = tempfile.mkdtemp(prefix="capwin_native_sign_")
        try:
            self._session(td, [
                {"t": 1.0, "type": "down", "x": 120, "y": 60},
                {"t": 1.4, "type": "down", "x": 122, "y": 62},
                {"t": 1.8, "type": "down", "x": 121, "y": 61},
            ])
            p = render.camera_path(td, stride=1)
            z = np.asarray(p["z"], dtype=float)
            self.assertGreater(float(z.max()), 1.05)      # it zoomed
            k = int(np.argmax(z))
            self.assertLess(p["cx"][k], 320.0)            # left half of buffer
            self.assertLess(p["cy"][k], 180.0)            # top half of buffer
            # and everything stays inside the window buffer
            for cx, cy in zip(p["cx"], p["cy"]):
                self.assertTrue(0.0 <= cx <= 640.0 and 0.0 <= cy <= 360.0)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class EraserScaleForNative(unittest.TestCase):
    """The cursor eraser sizes its boxes in POINTS*scale. A native take must
    use the WINDOW backing scale, not the display-derived scale_x/scale_y."""

    def test_native_track_uses_the_window_backing_scale(self):
        tr = render._NativeWindowTrack([[0, 0, 320, 180], [0, 0, 320, 180]],
                                       640, 360, np.array([0.0, 1.0]), 2.0)
        self.assertEqual(render._erase_scale(tr, 0.95, 0.95), (2.0, 2.0))

    def test_non_native_keeps_the_display_scale(self):
        self.assertEqual(render._erase_scale(None, 0.95, 0.9), (0.95, 0.9))
        wt = render._WindowTrack([[0, 0, 320, 180], [0, 0, 320, 180]],
                                 (0, 0, 320, 180), 640, 360,
                                 np.array([0.0, 1.0]), 2.0, 2.0)
        self.assertEqual(render._erase_scale(wt, 0.95, 0.9), (0.95, 0.9))


class NativeTrackRectFallback(unittest.TestCase):
    """A native take with absent/malformed logical_w/h stays mapped in WINDOW
    space via the rect's point size -- never escaping to the display mapper."""

    def test_missing_logical_falls_back_to_the_rect_size(self):
        cw = _native_cw(rect=(100.0, 50.0, 320.0, 180.0))
        del cw["logical_w"]
        del cw["logical_h"]
        meta = {"logical_w": 1440.0, "logical_h": 900.0, "capture_window": cw}
        tr = render._build_window_track({}, None, 640, 360, 1.0, 1.0,
                                        np.array([0.0, 1.0]), lambda a: a, 60,
                                        window_id=4242, meta=meta)
        self.assertIsInstance(tr, render._NativeWindowTrack)
        self.assertAlmostEqual(tr.scale, 2.0)          # 640 / rect_w(320)
        x, y = tr.to_src(np.array([0.0]), np.array([100.0]), np.array([50.0]))
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y[0], 0.0)
        x, y = tr.to_src(np.array([0.0]), np.array([420.0]), np.array([230.0]))
        self.assertAlmostEqual(x[0], 640.0)            # bottom-right -> corner
        self.assertAlmostEqual(y[0], 360.0)


class DescribeNativeCapture(unittest.TestCase):
    def test_native_is_applied_and_tracked_without_a_crop(self):
        d = render._describe_capture_window(_native_meta(), crop=None)
        self.assertEqual(d["mode"], "window_native")
        self.assertTrue(d["applied"])          # honored, though crop is None
        self.assertTrue(d["tracked"])          # track == "ok"

    def test_native_untracked_is_applied_but_not_tracked(self):
        d = render._describe_capture_window(_native_meta(track="failed"),
                                            crop=None)
        self.assertTrue(d["applied"])
        self.assertFalse(d["tracked"])


class NativeContainFit(unittest.TestCase):
    """`render._contain_fit` is where the camera crops a window buffer down to
    the output aspect. For a native take that source is any window aspect --
    wide, tall, or square -- so the identity property (auto aspect == full
    buffer) and the ratio math on unusual shapes both need pins.

    The camera hands `_contain_fit`'s result on to `_warp`, which needs
    `win_w:win_h == out_w:out_h` exactly (its docstring notes `sx/sy come out
    equal in practice`); breaking that here would leak a non-square-pixel
    scale into every warped frame."""

    def test_auto_aspect_returns_the_full_buffer_on_any_shape(self):
        # framing.resolve_aspect_canvas returns (W, H) for auto, so
        # _contain_fit(W, H, W, H) is the identity: zero letterbox.
        for W, H in ((1440, 264), (720, 1750), (1024, 1024), (640, 480)):
            w, h = render._contain_fit(W, H, W, H)
            self.assertAlmostEqual(w, W, places=3,
                                   msg="{}x{}: w drifted".format(W, H))
            self.assertAlmostEqual(h, H, places=3,
                                   msg="{}x{}: h drifted".format(W, H))

    def test_9_16_on_a_wide_narrow_source_is_bounded_by_the_height(self):
        # Very wide + very short: the biggest 9:16 rect is height-limited.
        # A per-axis bug (independent x/y scale) would extend the rect past
        # the source width, which _warp would then read as replicated border.
        w, h = render._contain_fit(9, 16, 1440, 264)
        self.assertAlmostEqual(h, 264.0, places=3)
        self.assertAlmostEqual(w, 264.0 * 9.0 / 16.0, places=3)

    def test_16_9_on_a_tall_narrow_source_is_bounded_by_the_width(self):
        w, h = render._contain_fit(16, 9, 720, 1750)
        self.assertAlmostEqual(w, 720.0, places=3)
        self.assertAlmostEqual(h, 720.0 * 9.0 / 16.0, places=3)

    def test_output_ratio_survives_every_arbitrary_source(self):
        # The result is aspect-locked to (out_w, out_h) by construction; drift
        # would corrupt every warped frame's pixel-square-ness.
        for out_w, out_h in ((9, 16), (16, 9), (1, 1), (4, 5), (21, 9)):
            for W, H in ((1440, 264), (720, 1750), (1024, 1024)):
                w, h = render._contain_fit(out_w, out_h, W, H)
                self.assertLessEqual(w, W + 1e-6,
                                     "out {}:{} src {}x{}: {}x{} escapes W"
                                     .format(out_w, out_h, W, H, w, h))
                self.assertLessEqual(h, H + 1e-6,
                                     "out {}:{} src {}x{}: {}x{} escapes H"
                                     .format(out_w, out_h, W, H, w, h))
                self.assertAlmostEqual(w / float(h),
                                       out_w / float(out_h),
                                       places=6,
                                       msg="out {}:{} src {}x{}".format(
                                           out_w, out_h, W, H))


class NativeArbitraryAspectRender(unittest.TestCase):
    """End-to-end: a native take's raw.mov IS the window's own backing buffer,
    which is any aspect the window happens to be. render must produce a valid
    output for clean + framed + --aspect + GIF on those shapes -- the doc's
    own P2 gate is 'pin one test on a non-screen-aspect source' (the plural
    below is deliberate; the extreme aspects fail differently).
    """

    # Real recorder shapes: the bar (wide-narrow), a code editor down the
    # left of a desktop (tall-narrow), and a Calculator-like utility window
    # (square-ish). Small enough that six ffmpeg-backed synthetic sessions
    # stay under a couple of seconds total.
    SHAPES = ((640, 120), (120, 640), (320, 320))

    def _native_session(self, td, width, height, events=None):
        # The native mapper's scale is buffer_px / rect_pt. Pin the rect at
        # half the buffer's point size so scale == 2 (Retina), which is what
        # every native take on this hardware actually measures.
        rect = (0.0, 0.0, width / 2.0, height / 2.0)
        _mk_session(td, events or [], duration=1, width=width, height=height,
                    logical_w=1440, logical_h=900,
                    capture_window=_native_cw(rect=rect,
                                              logical_w=width / 2.0,
                                              logical_h=height / 2.0))

    def test_clean_output_matches_the_window_buffer(self):
        # No --aspect: output IS the buffer. `_capture_crop_px` returns None
        # for native (see `WindowNativeCropBypass`), so W/H stay the raw
        # decoded dims of raw.mov.
        for width, height in self.SHAPES:
            td = tempfile.mkdtemp(prefix="capwin_native_clean_")
            try:
                self._native_session(td, width, height)
                out = os.path.join(td, "out.mp4")
                render.render(td, out_path=out, motion_blur=False,
                              click_fx=False)
                self.assertEqual(_probe_dims(out), (width, height),
                                 "clean {}x{}".format(width, height))
            finally:
                shutil.rmtree(td, ignore_errors=True)

    def test_framed_output_matches_the_window_buffer(self):
        # `--style framed` on auto aspect keeps (out_w, out_h) == (W, H), so
        # a wide-narrow native buffer renders as a wide-narrow framed take.
        # The framed painter's inner rect is aspect-matched to the CANVAS
        # (proven per-axis in test_framing.ArbitraryAspectCanvas) which is
        # what keeps the recording sized right for the frame around it.
        for width, height in self.SHAPES:
            td = tempfile.mkdtemp(prefix="capwin_native_framed_")
            try:
                self._native_session(td, width, height)
                out = os.path.join(td, "out.mp4")
                render.render(td, out_path=out, style="framed",
                              motion_blur=False, click_fx=False)
                self.assertEqual(_probe_dims(out), (width, height),
                                 "framed {}x{}".format(width, height))
            finally:
                shutil.rmtree(td, ignore_errors=True)

    def test_9_16_export_produces_a_9_16_canvas_on_any_source(self):
        # The source aspect has no bearing on --aspect: the requested aspect
        # wins, and `_contain_fit` crops the buffer down to it (see
        # `NativeContainFit`). Even a wide-narrow window (whose 9:16 crop is
        # a thin sliver of the middle) must produce an even-sized 9:16 canvas
        # rather than fail at libx264's odd-dimension guard.
        for width, height in self.SHAPES:
            td = tempfile.mkdtemp(prefix="capwin_native_916_")
            try:
                self._native_session(td, width, height,
                                     events=[{"t": 0.5, "type": "down",
                                              "x": 10, "y": 10}])
                out = os.path.join(td, "out.mp4")
                render.render(td, out_path=out, aspect="9:16",
                              motion_blur=False, click_fx=False)
                w, h = _probe_dims(out)
                self.assertEqual((w % 2, h % 2), (0, 0),
                                 "9:16 {}x{}: canvas {}x{}".format(
                                     width, height, w, h))
                self.assertLess(w, h,
                                "9:16 {}x{}: canvas {}x{} isn't portrait"
                                .format(width, height, w, h))
                self.assertAlmostEqual(w / float(h), 9 / 16.0, delta=0.02)
            finally:
                shutil.rmtree(td, ignore_errors=True)

    def test_gif_export_survives_arbitrary_source_aspect(self):
        # `_make_gif`'s vf is `scale={width}:-1:flags=lanczos`, so height is
        # auto-computed from the source aspect. Pinned once per shape: a
        # wide-narrow input must not silently swap sides or fail the
        # palettegen/paletteuse pipeline.
        for width, height in self.SHAPES:
            td = tempfile.mkdtemp(prefix="capwin_native_gif_")
            try:
                self._native_session(td, width, height)
                out = os.path.join(td, "out.mp4")
                render.render(td, out_path=out, make_gif=True,
                              gif_fps=15, gif_width=200,
                              motion_blur=False, click_fx=False)
                gif = os.path.splitext(out)[0] + ".gif"
                self.assertTrue(os.path.isfile(gif),
                                "gif {}x{}: missing output".format(width, height))
                gw, gh = _probe_dims(gif)
                self.assertEqual(gw, 200,
                                 "gif {}x{}: width {} != 200".format(
                                     width, height, gw))
                # ffmpeg's `-1` rounds to a multiple of 2 (yuv420p pin),
                # and palettegen/paletteuse may pad by one -- allow ±2 slack.
                want_gh = int(round(200 * height / float(width)))
                self.assertLessEqual(
                    abs(gh - want_gh), 2,
                    "gif {}x{}: height {} strays from want {}".format(
                        width, height, gh, want_gh))
            finally:
                shutil.rmtree(td, ignore_errors=True)


def _mk_multi_native_session(root, channels_spec, duration=1, fps=30):
    """Build a synthetic multi-window native session: N raw_i.mov files via
    testsrc2 + a manifest meta.json. Everything permissions-free (ffmpeg
    lavfi), so `_render_multi_native` can be exercised end-to-end without a
    real SCK capture.

    `channels_spec` is a list of dicts, each with `file`, `width`, `height`,
    `rect` (points), `t0_monotonic`, and optional `id`/`app`.
    """
    import subprocess as _sp
    os.makedirs(root, exist_ok=True)
    for i, spec in enumerate(channels_spec):
        path = os.path.join(root, spec["file"])
        _sp.check_call([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "testsrc2=size={}x{}:rate={}:duration={}".format(
                spec["width"], spec["height"], fps, duration),
            "-pix_fmt", "yuv420p", path])
    channels_meta = []
    for i, spec in enumerate(channels_spec):
        r = spec["rect"]
        channels_meta.append({
            "role": "screen_window",
            "file": spec["file"],
            "mode": "window_native",
            "id": int(spec.get("id", 1000 + i)),
            "app": spec.get("app", "App{}".format(i)),
            "title": spec.get("title", "t{}".format(i)),
            "units": "points",
            "rect": [float(r[0]), float(r[1]), float(r[2]), float(r[3])],
            "display_origin": [0.0, 0.0],
            "source": "quartz",
            "resnapshot": True,
            "end_rect": [float(r[0]), float(r[1]), float(r[2]), float(r[3])],
            "track": "ok",
            "logical_w": float(r[2]),
            "logical_h": float(r[3]),
            "buffer_w": int(spec["width"]),
            "buffer_h": int(spec["height"]),
            "t0_monotonic": float(spec.get("t0_monotonic", 0.0)),
        })
    meta = {
        "fps": fps,
        "logical_w": 1440.0, "logical_h": 900.0, "geom_source": "quartz",
        "t0_monotonic": float(channels_meta[0]["t0_monotonic"]),
        "video_index": 1, "mic_index": None,
        "events": "events.jsonl", "cursor_mode": "system",
        "key_capture": "activity",
        "face": None, "face_index": None, "face_fps": None,
        "face_t0_monotonic": None, "face_capture": None,
        "capture_backend": "sck",
        "capture_channels": channels_meta,
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        pass


class IsMultiNativeMeta(unittest.TestCase):
    """Guard that selects the multi-native render branch. Pinned because a
    consumer that saw both `capture_channels` and `capture_window` would
    have to guess which is authoritative -- and `render.render()` picks its
    branch off this."""

    def _ch(self, mode="window_native"):
        return {"role": "screen_window", "file": "raw_0.mov", "mode": mode}

    def test_two_or_more_native_channels_passes(self):
        self.assertTrue(render._is_multi_native_meta({
            "capture_channels": [self._ch(), self._ch()]}))
        self.assertTrue(render._is_multi_native_meta({
            "capture_channels": [self._ch(), self._ch(), self._ch()]}))

    def test_zero_or_one_channel_fails(self):
        # One channel is a degenerate multi-native that should fall through
        # to the single-window path (which the same session's `capture_window`
        # block, if present, would carry).
        self.assertFalse(render._is_multi_native_meta(
            {"capture_channels": []}))
        self.assertFalse(render._is_multi_native_meta(
            {"capture_channels": [self._ch()]}))

    def test_missing_or_wrong_type_fails(self):
        self.assertFalse(render._is_multi_native_meta({}))
        self.assertFalse(render._is_multi_native_meta({"capture_channels": None}))
        self.assertFalse(render._is_multi_native_meta("not a dict"))

    def test_a_channel_missing_native_mode_fails(self):
        # A future channel role (a "microphone" channel, say) that isn't
        # window_native must NOT flip the branch just because it's in the
        # list -- the multi-native renderer only knows how to composite
        # window-native video, so any non-native channel means this isn't
        # the right path.
        self.assertFalse(render._is_multi_native_meta({
            "capture_channels": [self._ch(), self._ch(mode="microphone")]}))


class RenderDispatchesOnCaptureChannels(unittest.TestCase):
    """render.render() branches at the TOP off `_is_multi_native_meta`. The
    single-file code below is provably untouched only if that dispatch runs
    before any single-file setup, which is what this test pins."""

    def test_multi_native_session_never_opens_a_top_level_raw(self):
        # A multi-native session has NO top-level raw.mov. If the dispatch
        # were missing, render() would fall through and try to open
        # meta.get("raw", "raw.mov") -- which doesn't exist -- and raise
        # `cannot open recording: .../raw.mov`. The successful render below
        # proves the dispatch fires before that path runs.
        td = tempfile.mkdtemp(prefix="p3_2_dispatch_")
        try:
            _mk_multi_native_session(td, [
                {"file": "raw_0.mov", "width": 320, "height": 240,
                 "rect": (0, 0, 160, 120), "t0_monotonic": 0.0},
                {"file": "raw_1.mov", "width": 320, "height": 240,
                 "rect": (200, 0, 160, 120), "t0_monotonic": 0.0},
            ])
            out = os.path.join(td, "out.mp4")
            render.render(td, out_path=out)
            self.assertTrue(os.path.isfile(out))
            self.assertFalse(os.path.isfile(os.path.join(td, "raw.mov")))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_single_window_render_never_reaches_multi_native(self):
        # Off-switch: a single-window native session with meta.capture_window
        # (no capture_channels) still routes to the ORIGINAL single-file
        # render below. Proven by rendering the same synthetic testsrc2
        # session as the other single-window tests -- if the dispatch
        # spuriously fired on it, the test would fail because there'd be no
        # `raw_i.mov` files to open.
        td = tempfile.mkdtemp(prefix="p3_2_off_")
        try:
            _mk_session(td, [], duration=1,
                        capture_window=_cw([0, 0, 320, 180]))
            out = os.path.join(td, "single.mp4")
            render.render(td, out_path=out, motion_blur=False, click_fx=False)
            self.assertEqual(_probe_dims(out), (320, 180))
        finally:
            shutil.rmtree(td, ignore_errors=True)


class MultiNativeRenderProducesComposite(unittest.TestCase):
    """End-to-end: N synthetic testsrc2 files + a manifest -> an output.mp4
    at the default canvas (1920x1080), painted by MultiFramePainter with N
    cards. The composite's ONE non-negotiable property: it renders without
    raising, at the right dims, for the shipping default settings."""

    def _sess(self):
        return [
            {"file": "raw_0.mov", "width": 320, "height": 240,
             "rect": (0, 0, 160, 120), "t0_monotonic": 0.0},
            {"file": "raw_1.mov", "width": 480, "height": 320,
             "rect": (200, 0, 240, 160), "t0_monotonic": 0.05},
        ]

    def test_default_output_is_the_multi_native_canvas(self):
        td = tempfile.mkdtemp(prefix="p3_2_canvas_")
        try:
            _mk_multi_native_session(td, self._sess())
            out = os.path.join(td, "composite.mp4")
            render.render(td, out_path=out)
            self.assertEqual(_probe_dims(out), (1920, 1080))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_aspect_flag_overrides_the_default(self):
        # --aspect 9:16 must reshape the output the same way it does for
        # every other render path. A rewrite that hard-coded 1920x1080
        # would swallow the flag silently.
        td = tempfile.mkdtemp(prefix="p3_2_aspect_")
        try:
            _mk_multi_native_session(td, self._sess())
            out = os.path.join(td, "vert.mp4")
            render.render(td, out_path=out, aspect="9:16")
            w, h = _probe_dims(out)
            self.assertEqual((w % 2, h % 2), (0, 0))
            self.assertLess(w, h)   # portrait
            self.assertAlmostEqual(w / float(h), 9 / 16.0, delta=0.02)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_layout_choice_reaches_the_painter(self):
        # `window_layout="feature"` picks the feature arrangement inside
        # MultiFramePainter. A rewrite that dropped the kwarg (or hard-coded
        # "grid") would still produce a valid mp4, but a differently-laid-out
        # one -- so pin the layout name gets through by rendering both grid
        # and feature and asserting the byte sequences differ.
        td = tempfile.mkdtemp(prefix="p3_2_layout_")
        try:
            _mk_multi_native_session(td, self._sess())
            grid = os.path.join(td, "grid.mp4")
            feat = os.path.join(td, "feat.mp4")
            render.render(td, out_path=grid, window_layout="grid")
            render.render(td, out_path=feat, window_layout="feature")
            with open(grid, "rb") as f:
                gb = f.read()
            with open(feat, "rb") as f:
                fb = f.read()
            self.assertNotEqual(gb, fb,
                                "grid and feature produced identical bytes; "
                                "window_layout is not reaching the painter")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class NativeTrackForChannel(unittest.TestCase):
    """`_native_track_for_channel` builds a `_NativeWindowTrack` off ONE
    manifest channel dict -- same math as the single-window `_native_track`,
    so a bug that quietly broke either would need this pin to survive."""

    def _ch(self, wid=100, rect=(0, 0, 320, 180), lw=320.0, lh=180.0,
            bw=640, bh=360):
        return {"role": "screen_window", "file": "raw_0.mov",
                "mode": "window_native", "id": wid,
                "logical_w": lw, "logical_h": lh,
                "buffer_w": bw, "buffer_h": bh,
                "rect": list(rect)}

    def _times(self, n=30, fps=30):
        return np.arange(n, dtype=float) / float(fps)

    def test_static_rect_track_maps_origin_to_zero(self):
        # A channel with no geometry samples for its id falls back to the
        # manifest's single rect: origin (0,0) maps to buffer (0,0), the
        # rect's bottom-right maps to (buffer_w, buffer_h). Exactly the
        # single-window native contract, per-channel.
        ch = self._ch(wid=100, rect=(50, 25, 320, 180))
        tr = render._native_track_for_channel(
            ch, {}, self._times(), lambda a: a, 30)
        self.assertIsNotNone(tr)
        x, y = tr.to_src(np.array([0.0]), np.array([50.0]), np.array([25.0]))
        self.assertAlmostEqual(x[0], 0.0)
        self.assertAlmostEqual(y[0], 0.0)
        x, y = tr.to_src(np.array([0.0]),
                         np.array([50.0 + 320.0]), np.array([25.0 + 180.0]))
        self.assertAlmostEqual(x[0], 640.0)
        self.assertAlmostEqual(y[0], 360.0)

    def test_out_of_window_clicks_clamp_not_drop(self):
        # THE hard invariant for the shared click set. A click that misses
        # window A must keep its TIME (so cluster membership stays shared
        # across camera / edits / beats), while its POSITION lands on the
        # nearest buffer edge -- which the auto-zoom planner may treat as
        # a mild pull, an acceptable trade to keep the time set intact.
        ch = self._ch(wid=100, rect=(0, 0, 320, 180))
        tr = render._native_track_for_channel(
            ch, {}, self._times(), lambda a: a, 30)
        x, y = tr.to_src(np.array([0.0, 0.0]),
                         np.array([-500.0, 5000.0]),
                         np.array([-500.0, 5000.0]))
        # Both survived (count preserved -- the time set is intact).
        self.assertEqual(len(x), 2)
        # Positions clamped into [0, buf].
        self.assertEqual((float(x[0]), float(y[0])), (0.0, 0.0))
        self.assertEqual((float(x[1]), float(y[1])), (640.0, 360.0))

    def test_missing_dims_returns_none_safely(self):
        # No buffer dims / no rect / no logical -> None, and the multi-native
        # render treats a None track as "this channel has no camera / cursor"
        # rather than raising. This is the fail-safe that keeps a partial
        # manifest from aborting a render.
        self.assertIsNone(render._native_track_for_channel(
            {"id": 100, "logical_w": 0, "logical_h": 0, "buffer_w": 0,
             "buffer_h": 0}, {}, self._times(), lambda a: a, 30))
        self.assertIsNone(render._native_track_for_channel(
            None, {}, self._times(), lambda a: a, 30))


class NativeCardCameras(unittest.TestCase):
    """`_build_native_card_cameras` + `_apply_native_card_cameras`. Per-card
    zoom paths use the shared click set projected via each channel's
    `_NativeWindowTrack.to_src` (clamped-not-dropped). Every click's TIME
    goes into every card's planner -- the invariant that keeps the trailing-
    cluster set consistent across `camera.cluster_to_range`,
    `edits.auto_zoom_proposals` and `beats`."""

    def _tr(self, rect=(0, 0, 320, 180), buf=(640, 360)):
        # A static rect track (no motion), 30 frames on 30fps.
        rects = np.tile(np.asarray(rect, dtype=float), (30, 1))
        return render._NativeWindowTrack(
            rects, buf[0], buf[1],
            np.arange(30, dtype=float) / 30.0, 2.0)

    def test_enabled_false_yields_no_cameras(self):
        # Off-switch: the flag is what selects whether a card's crop is
        # zoomed. False -> every entry is None, which keeps `paint()` on the
        # untouched resize path (measured perf property of the display-crop
        # path too, preserved here).
        chs = [{"id": 100}, {"id": 200}]
        trs = [self._tr(), self._tr()]
        paths = render._build_native_card_cameras(
            chs, trs, np.array([0.5]), np.array([160.0]), np.array([90.0]),
            np.arange(30) / 30.0, max_zoom=2.0, params={},
            suppressed_ranges=None, plan_duration=1.0, src_fps=30,
            enabled=False)
        self.assertEqual(paths, [None, None])

    def test_a_click_cluster_in_bounds_makes_that_card_zoom(self):
        # A CLUSTER of clicks inside card 0's window drives the planner past
        # its trailing-cluster threshold; a lone click doesn't cross it (same
        # bar the display-crop path uses via `camera.build_card_paths`).
        # 90 frames on a 30fps plan gives the planner room to plan its
        # zoom-in / hold / zoom-out window around the cluster.
        chs = [{"id": 100}, {"id": 200}]
        trs = [self._tr(rect=(0, 0, 320, 180)),
               self._tr(rect=(400, 0, 320, 180))]
        # 5 clicks tightly clustered around 1.0s -- inside card 0 only.
        clicks_t = np.array([1.0, 1.10, 1.20, 1.30, 1.40])
        clicks_x = np.array([160.0, 155.0, 160.0, 165.0, 160.0])
        clicks_y = np.array([90.0, 88.0, 92.0, 89.0, 90.0])
        paths = render._build_native_card_cameras(
            chs, trs, clicks_t, clicks_x, clicks_y,
            np.arange(90) / 30.0, max_zoom=2.0, params={},
            suppressed_ranges=None, plan_duration=3.0, src_fps=30,
            enabled=True)
        self.assertEqual(len(paths), 2)
        self.assertIsNotNone(paths[0], "card 0 should zoom on a click cluster")
        self.assertGreater(float(np.max(paths[0][:, 2])), 1.0)

    def test_apply_card_camera_warps_only_zooming_cards(self):
        # A None path is a still card and gets the untouched crop -- same
        # object out. A real path replaces that card's crop with a warp
        # sized to the target cell.
        crops = [np.full((360, 640, 3), 100, np.uint8),
                 np.full((360, 640, 3), 200, np.uint8)]
        trs = [self._tr(), self._tr()]
        cells = [{"x": 0, "y": 0, "w": 320, "h": 180},
                 {"x": 320, "y": 0, "w": 320, "h": 180}]
        # No paths -> unchanged crops.
        out = render._apply_native_card_cameras(
            crops, [None, None], trs, cells, 5)
        self.assertIs(out[0], crops[0])
        self.assertIs(out[1], crops[1])
        # A zoomed path for card 0 -> its crop is replaced by a cell-sized
        # warp; card 1's crop unchanged.
        path = np.tile(np.array([160.0, 90.0, 1.5]), (30, 1))
        out = render._apply_native_card_cameras(
            crops, [path, None], trs, cells, 5)
        self.assertIsNot(out[0], crops[0])
        self.assertEqual(out[0].shape, (180, 320, 3))
        self.assertIs(out[1], crops[1])


class NativeFocusAndCursor(unittest.TestCase):
    """End-to-end: --window-zoom + --window-focus + --cursor-fx render
    together on a synthetic multi-native session, and every switch is
    respected (no crash when they toggle in any combination)."""

    def _sess(self, td):
        _mk_multi_native_session(td, [
            {"file": "raw_0.mov", "width": 320, "height": 240,
             "rect": (0, 0, 160, 120), "t0_monotonic": 0.0,
             "id": 100},
            {"file": "raw_1.mov", "width": 320, "height": 240,
             "rect": (200, 0, 160, 120), "t0_monotonic": 0.0,
             "id": 200},
        ])
        # Inject a click into each window so both cards actually zoom /
        # focus. Times relative to session t0 (0.0 here).
        with open(os.path.join(td, "events.jsonl"), "w") as f:
            f.write(json.dumps({"t": 0.2, "type": "move", "x": 80, "y": 60})
                    + "\n")
            f.write(json.dumps({"t": 0.3, "type": "down", "x": 80, "y": 60})
                    + "\n")
            f.write(json.dumps({"t": 0.7, "type": "move", "x": 280, "y": 60})
                    + "\n")
            f.write(json.dumps({"t": 0.8, "type": "down", "x": 280, "y": 60})
                    + "\n")

    def test_zoom_focus_cursor_compose_without_crash(self):
        # The gate for P3.3: per-card zoom AND focus AND cursor-fx compose.
        # A regression that broke any one would fail the pipeline (they run
        # in the same loop iteration off shared state).
        td = tempfile.mkdtemp(prefix="p3_3_all_")
        try:
            self._sess(td)
            out = os.path.join(td, "all.mp4")
            render.render(td, out_path=out,
                          window_zoom=True, window_focus=True,
                          cursor_fx=True,
                          window_layout="feature",
                          background="midnight")
            self.assertEqual(_probe_dims(out), (1920, 1080))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_zoom_alone_produces_different_bytes_than_static(self):
        # Off-switch pin for the zoom flag: window_zoom=False must produce
        # the P3.2 static composite exactly (no per-card warp), so
        # window_zoom=True should differ. A rewrite that silently ignored
        # the flag would fail here.
        td = tempfile.mkdtemp(prefix="p3_3_zoom_")
        try:
            self._sess(td)
            a = os.path.join(td, "off.mp4")
            b = os.path.join(td, "on.mp4")
            render.render(td, out_path=a, window_zoom=False)
            render.render(td, out_path=b, window_zoom=True)
            with open(a, "rb") as f:
                ab = f.read()
            with open(b, "rb") as f:
                bb = f.read()
            self.assertNotEqual(
                ab, bb, "window_zoom flag has no effect on the output bytes")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class NativeFocusPlanIsEditable(unittest.TestCase):
    """`render.multi_native_focus_ranges`: the fleet take's composition camera
    as spans someone can retune ONE of (MCP `adjust_zoom`).

    The property the whole thing rests on is that materializing the plan
    changes NOTHING about the video -- same moves, now editable. That is not
    free here: a multi-native take's clicks are clamped-not-dropped, so every
    card owns every click and the plan can contain genuinely overlapping
    spans, which the two readers resolve differently (`camera._active_range`
    first-wins for an auto plan vs `camera._normalize_focus_manual`
    truncation for a materialized one). `render._first_wins_spans` is what
    keeps them equal; delete it and `test_materializing_the_plan_changes_
    nothing` fails with the emphasis on the wrong card."""

    def _sess(self, td, duration=2):
        _mk_multi_native_session(td, [
            {"file": "raw_0.mov", "width": 320, "height": 240,
             "rect": (0, 0, 160, 120), "t0_monotonic": 0.0, "id": 100},
            {"file": "raw_1.mov", "width": 320, "height": 240,
             "rect": (200, 0, 160, 120), "t0_monotonic": 0.0, "id": 200},
        ], duration=duration)
        with open(os.path.join(td, "events.jsonl"), "w") as f:
            for t, x, y in ((0.2, 80, 60), (0.5, 84, 64),
                            (1.2, 280, 60), (1.5, 284, 64)):
                f.write(json.dumps({"t": t, "type": "move",
                                    "x": x, "y": y}) + "\n")
                f.write(json.dumps({"t": t, "type": "down",
                                    "x": x, "y": y}) + "\n")

    def test_the_plan_is_editable_span_objects(self):
        td = tempfile.mkdtemp(prefix="native_focus_plan_")
        try:
            self._sess(td)
            plan = render.multi_native_focus_ranges(td)
            self.assertTrue(plan, "a 2-card take with clicks plans no moves")
            for span in plan:
                self.assertEqual(sorted(span), ["card", "end", "level",
                                                "start"])
                self.assertIn(span["level"], ("focus", "full"))
                self.assertGreater(span["end"], span["start"])
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_spans_never_overlap(self):
        """One subject at a time, so the span covering a timestamp is
        unambiguous -- which is what lets `adjust_zoom` address a move by
        time at all."""
        td = tempfile.mkdtemp(prefix="native_focus_nolap_")
        try:
            self._sess(td)
            plan = render.multi_native_focus_ranges(td)
            for a, b in zip(plan, plan[1:]):
                self.assertLessEqual(a["end"], b["start"],
                                     "overlapping focus spans: {} / {}"
                                     .format(a, b))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_materializing_the_plan_changes_nothing(self):
        """Render the same take twice -- auto-planned, then with the
        materialized plan handed back in -- and compare every decoded frame."""
        td = tempfile.mkdtemp(prefix="native_focus_same_")
        try:
            self._sess(td)
            plan = render.multi_native_focus_ranges(td)
            auto = os.path.join(td, "auto.mp4")
            fixed = os.path.join(td, "materialized.mp4")
            render.render(td, out_path=auto, window_focus=True)
            render.render(td, out_path=fixed, window_focus=True,
                          focus_ranges=[dict(p) for p in plan])
            a, b = _read_frames(auto), _read_frames(fixed)
            self.assertTrue(a and len(a) == len(b))
            for i, (fa, fb) in enumerate(zip(a, b)):
                self.assertTrue(
                    np.array_equal(fa, fb),
                    "frame {} differs: materializing the focus plan re-cut "
                    "the video".format(i))
        finally:
            shutil.rmtree(td, ignore_errors=True)


class FirstWinsSpans(unittest.TestCase):
    """`render._first_wins_spans` in isolation -- the rule is exactly
    `camera._active_range`'s: the earliest-starting span owns an instant, and
    a later one keeps only what is left over."""

    def test_a_shadowed_span_keeps_only_its_tail(self):
        got = render._first_wins_spans([
            {"start": 0.0, "end": 1.5, "card": 0, "level": "focus"},
            {"start": 0.0, "end": 2.4, "card": 1, "level": "focus"},
        ])
        self.assertEqual([(g["start"], g["end"], g["card"]) for g in got],
                         [(0.0, 1.5, 0), (1.5, 2.4, 1)])

    def test_a_span_can_come_back_in_two_pieces(self):
        got = render._first_wins_spans([
            {"start": 1.0, "end": 2.0, "card": 0, "level": "focus"},
            {"start": 0.0, "end": 3.0, "card": 1, "level": "full"},
        ])
        self.assertEqual([(g["start"], g["end"], g["card"]) for g in got],
                         [(0.0, 1.0, 1), (1.0, 2.0, 0), (2.0, 3.0, 1)])
        self.assertTrue(all(g["level"] == "full" for g in got
                            if g["card"] == 1))

    def test_disjoint_spans_are_untouched(self):
        spans = [{"start": 0.0, "end": 1.0, "card": 0, "level": "focus"},
                 {"start": 2.0, "end": 3.0, "card": 1, "level": "full"}]
        self.assertEqual(render._first_wins_spans(spans), spans)

    def test_a_fully_shadowed_span_disappears(self):
        got = render._first_wins_spans([
            {"start": 0.0, "end": 3.0, "card": 0, "level": "focus"},
            {"start": 1.0, "end": 2.0, "card": 1, "level": "full"},
        ])
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["card"], 0)


class MultiNativeManualCardLayout(unittest.TestCase):
    """`channel_layouts` — hand-placed cards on a multi-native composite
    (docs/architecture.md P1). The off-switch is bit-exact; an override moves
    the card; and a RESIZE never changes the output resolution (the
    freeze-canvas rule: the override is stamped AFTER the canvas is sized)."""

    def _sess(self):
        return [
            {"file": "raw_0.mov", "width": 320, "height": 180,
             "rect": (0, 0, 320, 180), "t0_monotonic": 0.0},
            {"file": "raw_1.mov", "width": 240, "height": 180,
             "rect": (400, 0, 240, 180), "t0_monotonic": 0.0},
        ]

    def _render(self, td, name, **kw):
        out = os.path.join(td, name)
        render.render(td, out_path=out, aspect="640x360", motion_blur=False,
                      click_fx=False, **kw)
        return out

    def test_empty_layouts_is_byte_identical_to_absent(self):
        td = tempfile.mkdtemp(prefix="mcl_off_")
        try:
            _mk_multi_native_session(td, self._sess())
            base = self._render(td, "base.mp4")
            off = self._render(td, "off.mp4", channel_layouts=[])
            with open(base, "rb") as f:
                a = f.read()
            with open(off, "rb") as f:
                b = f.read()
            self.assertEqual(a, b)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_override_moves_the_card_without_changing_resolution(self):
        td = tempfile.mkdtemp(prefix="mcl_ov_")
        try:
            _mk_multi_native_session(td, self._sess())
            base = self._render(td, "base.mp4")
            ov = self._render(td, "ov.mp4",
                              channel_layouts=[None, {"x": 0.05, "y": 0.05,
                                                      "w": 0.30, "h": 0.30}])
            with open(base, "rb") as f:
                a = f.read()
            with open(ov, "rb") as f:
                b = f.read()
            self.assertNotEqual(a, b)                       # the card moved
            self.assertEqual(_probe_dims(ov), _probe_dims(base))  # frozen canvas
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_null_slot_leaves_that_card_on_its_preset(self):
        # A [override, null] pair moves ONLY card 0; passing the same override
        # in slot 1 instead must produce a DIFFERENT frame (proves positional
        # binding, and that a null slot is a per-card no-op).
        td = tempfile.mkdtemp(prefix="mcl_null_")
        try:
            _mk_multi_native_session(td, self._sess())
            lay = {"x": 0.10, "y": 0.10, "w": 0.35, "h": 0.35}
            card0 = self._render(td, "c0.mp4", channel_layouts=[lay, None])
            card1 = self._render(td, "c1.mp4", channel_layouts=[None, lay])
            with open(card0, "rb") as f:
                a = f.read()
            with open(card1, "rb") as f:
                b = f.read()
            self.assertNotEqual(a, b)
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
