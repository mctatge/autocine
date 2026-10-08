"""The editor's crop (`edits.crop`) -- a spatial trim of the source.

A SECOND crop stage, after the record-time `capture_window` one. It is a
static rect in the space `describe_session` reports and `source_frame`
returns, sliced out of every frame before the camera runs, so `(W, H)` and
every event coordinate move into crop space and auto-zoom keeps working.

The invariant that matters most is the off switch: `crop = None` must
reproduce the pre-feature render BYTE FOR BYTE, because this feature touches
the one code path every export runs through.

Three halves:
  * schema round-trip through edits.py (normalize / merge / preset moves),
  * pure unit tests of `_user_crop_px` (the clamp, the even-izing, every
    fail-safe), and
  * end-to-end render/preview/camera_path checks on the same synthetic
    testsrc2 harness the other render tests use -- no permissions needed.
"""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from autocine import edits as ed
from autocine import render

from tests.test_capture_window import _mk_session, _cw


def _events(n=6):
    return [{"t": 0.5 + 0.5 * i, "type": "down", "x": 100 + 10 * i,
             "y": 80 + 5 * i} for i in range(n)]


def _dims(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout.strip()
    w, h = out.split(",")[:2]
    return int(w), int(h)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CropSchema(unittest.TestCase):
    """edits.py: the crop survives every doc rebuild, and junk becomes None."""

    def test_default_is_off(self):
        self.assertIsNone(ed.default_edits()["crop"])
        self.assertIsNone(ed.normalize_edits({})["crop"])

    def test_a_valid_rect_normalizes(self):
        got = ed.normalize_edits({"crop": {"x": 10, "y": 20, "w": 200,
                                           "h": 120}})["crop"]
        self.assertEqual(got, {"x": 10.0, "y": 20.0, "w": 200.0, "h": 120.0})

    def test_junk_normalizes_to_off_not_to_a_tiny_rect(self):
        """A crop that silently became 16x16 would be far worse than one that
        silently did nothing, so every malformed shape falls back to OFF."""
        for bad in ({"x": 0, "y": 0, "w": 4, "h": 100},      # under the floor
                    {"x": 0, "y": 0, "w": "abc", "h": 100},  # not a number
                    {"x": 0, "y": 0, "w": float("inf"), "h": 100},
                    {"x": 0, "y": 0},                        # no size at all
                    "not a dict", 7, [], None):
            self.assertIsNone(ed.normalize_edits({"crop": bad})["crop"], bad)

    def test_a_negative_origin_clamps_rather_than_dropping_the_crop(self):
        got = ed.normalize_edits({"crop": {"x": -5, "y": -9, "w": 200,
                                           "h": 120}})["crop"]
        self.assertEqual((got["x"], got["y"]), (0.0, 0.0))

    def test_merge_patches_and_survives_an_unrelated_patch(self):
        base = ed.default_edits()
        rect = {"x": 10, "y": 20, "w": 200, "h": 120}
        m = ed.merge_edits(base, {"crop": rect}, duration=10.0)
        self.assertEqual(m["crop"]["w"], 200.0)
        later = ed.merge_edits(m, {"render": {"zoom": 3.0}}, duration=10.0)
        self.assertEqual(later["crop"]["w"], 200.0)

    def test_null_is_a_real_patch_value_not_an_omission(self):
        """Reset sends `crop: null`. merge_edits keys off MEMBERSHIP, so this
        has to clear the crop rather than read as 'no opinion'."""
        m = ed.merge_edits(ed.default_edits(),
                           {"crop": {"x": 1, "y": 2, "w": 200, "h": 120}},
                           duration=10.0)
        self.assertIsNotNone(m["crop"])
        self.assertIsNone(ed.merge_edits(m, {"crop": None},
                                         duration=10.0)["crop"])

    def test_preset_moves_carry_the_crop(self):
        """The crop is timeline content like `trim`, not per-preset look --
        switching or duplicating a preset must not silently uncrop the
        project. Both rebuild the doc key by key, so both can drop it."""
        m = ed.merge_edits(ed.default_edits(),
                           {"crop": {"x": 1, "y": 2, "w": 200, "h": 120}},
                           duration=10.0)
        pid = m["presets"][0]["id"]
        self.assertIsNotNone(ed.set_active_preset(m, pid, duration=10.0)["crop"])
        self.assertIsNotNone(ed.duplicate_preset(m, duration=10.0)["crop"])


class UserCropPx(unittest.TestCase):
    """`render._user_crop_px`: the rect that actually reaches the slice."""

    def test_none_and_junk_are_off(self):
        for bad in (None, "x", 5, [], {}, {"w": 100}, {"w": 100, "h": "a"},
                    {"w": float("nan"), "h": 100}):
            self.assertIsNone(render._user_crop_px(bad, 640, 360), bad)

    def test_plain_rect(self):
        self.assertEqual(
            render._user_crop_px({"x": 10, "y": 20, "w": 200, "h": 120},
                                 640, 360),
            (10, 20, 200, 120))

    def test_sizes_are_evenized_inward(self):
        """`_encode_cmd` feeds these dims to libx264 -pix_fmt yuv420p, which
        rejects odd ones -- an odd crop would break every export. Inward,
        because rounding up would overrun the frame."""
        x, y, w, h = render._user_crop_px(
            {"x": 0, "y": 0, "w": 201, "h": 121}, 640, 360)
        self.assertEqual((w % 2, h % 2), (0, 0))
        self.assertLessEqual(w, 201)
        self.assertLessEqual(h, 121)

    def test_a_rect_hanging_off_the_frame_is_clamped_not_dropped(self):
        x, y, w, h = render._user_crop_px(
            {"x": 600, "y": 300, "w": 400, "h": 400}, 640, 360)
        self.assertLessEqual(x + w, 640)
        self.assertLessEqual(y + h, 360)

    def test_a_full_frame_rect_is_not_a_crop(self):
        """Identity has to resolve to None so the uncropped path keeps running
        the literal pre-feature code -- a (0, 0, W, H) slice would still be a
        different numpy object reaching the compositor."""
        self.assertIsNone(
            render._user_crop_px({"x": 0, "y": 0, "w": 640, "h": 360},
                                 640, 360))

    def test_a_rect_clamped_below_the_floor_is_off(self):
        self.assertIsNone(
            render._user_crop_px({"x": 635, "y": 0, "w": 100, "h": 100},
                                 640, 360))


class CropEndToEnd(unittest.TestCase):
    """Render/preview/camera_path on a synthetic session."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sd = os.path.join(cls.tmp, "sess")
        _mk_session(cls.sd, _events(), duration=2, fps=30,
                    width=640, height=360)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, name, **kw):
        out = os.path.join(self.tmp, name)
        render.render(self.sd, out_path=out, motion_blur=False, facecam=False,
                      **kw)
        return out

    def test_off_is_byte_identical(self):
        """THE invariant. Every toggle in this codebase reproduces the previous
        behavior exactly when disabled; this one sits on the path every export
        runs, so it is pinned on the bytes, not on the dimensions."""
        a = self._render("off_a.mp4")
        b = self._render("off_b.mp4", crop_rect=None)
        c = self._render("off_c.mp4", crop_rect={"x": 0, "y": 0, "w": 4,
                                                 "h": 4})   # rejected -> off
        self.assertEqual(_md5(a), _md5(b))
        self.assertEqual(_md5(a), _md5(c),
                         "a rejected crop must fall back to the uncropped "
                         "render, not to some clamped rect")

    def test_the_export_is_the_cropped_size(self):
        out = self._render("cropped.mp4",
                           crop_rect={"x": 40, "y": 20, "w": 320, "h": 200})
        self.assertEqual(_dims(out), (320, 200))

    def test_the_export_pixels_are_the_cropped_region(self):
        """Dimensions alone would also pass if the frame were scaled down
        rather than cut, so compare against a straight ffmpeg crop."""
        out = self._render("pix.mp4",
                           crop_rect={"x": 40, "y": 20, "w": 320, "h": 200})
        ref = os.path.join(self.tmp, "ref.png")
        got = os.path.join(self.tmp, "got.png")
        raw = os.path.join(self.sd, "raw.mov")
        subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", raw,
                               "-vf", "crop=320:200:40:20", "-frames:v", "1",
                               ref])
        subprocess.check_call(["ffmpeg", "-y", "-v", "error", "-i", out,
                               "-frames:v", "1", got])
        import numpy as np
        from PIL import Image
        a = np.asarray(Image.open(ref).convert("RGB"), dtype=float)
        b = np.asarray(Image.open(got).convert("RGB"), dtype=float)
        self.assertEqual(a.shape, b.shape)
        # Not bit-exact: the render re-encodes and the camera may be mid-zoom
        # at t=0. Mean absolute error is what separates "the same region" from
        # "a different region" here.
        self.assertLess(np.abs(a - b).mean(), 24.0)

    def test_camera_path_plans_in_crop_space(self):
        """The editor draws its camera line from this. If the plan ran in
        uncropped space the timeline would show a camera the export never
        followed."""
        plain = render.camera_path(self.sd, stride=4)
        cropped = render.camera_path(self.sd, stride=4,
                                     crop_rect={"x": 40, "y": 20,
                                                "w": 320, "h": 200})
        self.assertEqual(plain["source"], [640, 360])
        self.assertEqual(cropped["source"], [320, 200])

    def test_camera_path_reports_a_stage_crop_for_the_browser(self):
        """Live playback is a browser-side compositor, not a server render, so
        it needs the composed rect to clip its <video> to -- and `crop` alone
        cannot carry it (the editor reads that one to decide whether the
        session was window-TARGETED)."""
        p = render.camera_path(self.sd, stride=4,
                               crop_rect={"x": 40, "y": 20, "w": 320,
                                          "h": 200})
        self.assertIsNone(p["crop"])
        self.assertEqual(p["stage_crop"], [40, 20, 320, 200])
        self.assertIsNone(render.camera_path(self.sd, stride=4)["stage_crop"])

    def test_preview_frame_is_the_cropped_size(self):
        fr = render.preview_frame(self.sd, 1.0,
                                  crop_rect={"x": 40, "y": 20, "w": 320,
                                             "h": 200})
        self.assertEqual(fr.shape[1] / float(fr.shape[0]), 320 / 200.0)

    def test_source_frame_is_uncropped_unless_asked(self):
        """`source_frame` is the space rects are AUTHORED in -- the canvas the
        crop box itself is dragged on. Applying the crop there by default
        would make the rect relative to a space it defines."""
        self.assertEqual(render.source_frame(self.sd, 1.0).shape[:2],
                         (360, 640))
        self.assertEqual(
            render.source_frame(self.sd, 1.0,
                                crop_rect={"x": 40, "y": 20, "w": 320,
                                           "h": 200}).shape[:2],
            (200, 320))


class CropComposesWithCaptureWindow(unittest.TestCase):
    """Both crop stages on one session: the capture crop runs first, and the
    editor's rect is interpreted INSIDE it."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sd = os.path.join(cls.tmp, "sess")
        # 640x360 file, logical 320x180 -> 2x. A 100x80pt window at (20, 10)
        # is a 200x160px capture crop at (40, 20).
        _mk_session(cls.sd, _events(), duration=2, fps=30,
                    width=640, height=360, logical_w=320, logical_h=180,
                    capture_window=_cw((20, 10, 100, 80)))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_capture_crop_defines_the_space_the_editor_rect_lives_in(self):
        info = render.describe_session(self.sd)
        self.assertEqual((info["width"], info["height"]), (200, 160))
        out = os.path.join(self.tmp, "both.mp4")
        render.render(self.sd, out_path=out, motion_blur=False, facecam=False,
                      crop_rect={"x": 10, "y": 20, "w": 120, "h": 100})
        self.assertEqual(_dims(out), (120, 100))

    def test_stage_crop_composes_both_origins_for_the_browser(self):
        p = render.camera_path(self.sd, stride=4,
                               crop_rect={"x": 10, "y": 20, "w": 120,
                                          "h": 100})
        self.assertEqual(p["crop"], [40, 20, 200, 160])
        # 40 + 10, 20 + 20 -- the editor's rect offset by the capture origin.
        self.assertEqual(p["stage_crop"], [50, 40, 120, 100])

    def test_off_is_byte_identical_here_too(self):
        a = os.path.join(self.tmp, "cw_a.mp4")
        b = os.path.join(self.tmp, "cw_b.mp4")
        render.render(self.sd, out_path=a, motion_blur=False, facecam=False)
        render.render(self.sd, out_path=b, motion_blur=False, facecam=False,
                      crop_rect=None)
        self.assertEqual(_md5(a), _md5(b))


if __name__ == "__main__":
    unittest.main()
