"""Facecam bubble compositing (effects.FacecamOverlay).

Exercised with a synthetic face.mov + a blank output frame -- no live
camera, no macOS permissions.
"""

import os
import tempfile
import unittest

import numpy as np
import cv2

from autocine import effects


def _write_face(path, color_bgr, n=30, fps=30, size=(320, 240)):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, size)
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    frame[:] = color_bgr
    for _ in range(n):
        vw.write(frame)
    vw.release()


def _write_striped_face(path, n=30, fps=30, size=(320, 240), stripe=20):
    """A face with hard vertical black/white stripes -- unlike a flat colour,
    blurring it is measurable (a solid colour blurs to itself)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(path, fourcc, fps, size)
    frame = np.zeros((size[1], size[0], 3), np.uint8)
    cols = (np.arange(size[0]) // stripe) % 2
    frame[:, cols == 1] = 255
    for _ in range(n):
        vw.write(frame)
    vw.release()


class FacecamOverlayTest(unittest.TestCase):
    OUT_W, OUT_H = 1280, 720

    def _overlay(self, td, color=(0, 0, 255), **params):
        face = os.path.join(td, "face.mov")
        _write_face(face, color)
        p = {"shadow": False, "border_frac": 0.0}
        p.update(params)
        return effects.FacecamOverlay(
            face, self.OUT_W, self.OUT_H, t0_screen=0.0, face_t0=0.0,
            face_fps=30, params=p)

    def test_bubble_composited_at_chosen_corner(self):
        with tempfile.TemporaryDirectory() as td:
            ov = self._overlay(td, color=(0, 0, 255), position="bottom-left")
            self.assertTrue(ov.available())
            img = np.full((self.OUT_H, self.OUT_W, 3), 40, np.uint8)
            ov.draw(img, 0.5)
            D = ov.D
            cx, cy = ov.px + D // 2, ov.py + D // 2
            # bubble center is now the face color (red, BGR)
            self.assertGreater(int(img[cy, cx][2]), 150)
            self.assertLess(int(img[cy, cx][0]), 90)
            # a corner far from the bubble is untouched
            self.assertTrue((img[5, self.OUT_W - 5] == 40).all())
            ov.release()

    def test_bottom_left_places_bubble_lower_left(self):
        with tempfile.TemporaryDirectory() as td:
            ov = self._overlay(td, position="bottom-left")
            self.assertLess(ov.px, self.OUT_W // 2)
            self.assertGreater(ov.py, self.OUT_H // 2)

    def test_top_right_places_bubble_upper_right(self):
        with tempfile.TemporaryDirectory() as td:
            ov = self._overlay(td, position="top-right")
            self.assertGreater(ov.px, self.OUT_W // 2)
            self.assertLess(ov.py, self.OUT_H // 2)

    def test_size_frac_controls_diameter(self):
        with tempfile.TemporaryDirectory() as td:
            small = self._overlay(td, size_frac=0.15)
            big = self._overlay(td, size_frac=0.30)
            self.assertLess(small.D, big.D)
            self.assertAlmostEqual(big.D, int(round(0.30 * self.OUT_H)), delta=2)

    def test_blur_off_builds_no_vignette(self):
        # Off (the default) precomputes nothing, so the draw path is the exact
        # pre-feature one -- the "every off switch is bit-exact" invariant.
        with tempfile.TemporaryDirectory() as td:
            self.assertIsNone(self._overlay(td).__dict__["_vignette"])
            self.assertIsNone(self._overlay(td, blur=0.0).__dict__["_vignette"])
            self.assertIsNotNone(self._overlay(td, blur=0.6).__dict__["_vignette"])

    def _draw_striped(self, td, **params):
        # Two overlays reading ONE face file, so the decoded frame is identical
        # and any pixel difference is the blur, not codec noise.
        face = os.path.join(td, "face.mov")
        _write_striped_face(face)
        p = {"shadow": False, "border_frac": 0.0, "position": "bottom-left"}
        p.update(params)
        ov = effects.FacecamOverlay(
            face, self.OUT_W, self.OUT_H, t0_screen=0.0, face_t0=0.0,
            face_fps=30, params=p)
        img = np.full((self.OUT_H, self.OUT_W, 3), 40, np.uint8)
        ov.draw(img, 0.5)
        ov.release()
        return img, ov

    def test_blur_keeps_centre_sharp_and_softens_rim(self):
        with tempfile.TemporaryDirectory() as td_off, \
                tempfile.TemporaryDirectory() as td_on:
            off, ov = self._draw_striped(td_off)
            on, _ = self._draw_striped(td_on, blur=0.9)
            D, px, py = ov.D, ov.px, ov.py
            cx, cy = px + D // 2, py + D // 2
            # The inner core (vignette weight is exactly 0 there) is untouched.
            h = D // 8
            core_off = off[cy - h:cy + h, cx - h:cx + h]
            core_on = on[cy - h:cy + h, cx - h:cx + h]
            self.assertTrue(np.array_equal(core_off, core_on))
            # A patch out near the rim IS blurred, so it differs from sharp.
            rx = cx + int(0.78 * (D // 2))
            q = D // 12
            rim_off = off[cy - q:cy + q, rx - q:rx + q].astype(np.int32)
            rim_on = on[cy - q:cy + q, rx - q:rx + q].astype(np.int32)
            self.assertGreater(int(np.abs(rim_off - rim_on).sum()), 0)

    def test_missing_face_track_yields_no_overlay(self):
        # render._facecam_overlay returns None when meta has no face track.
        from autocine import render
        with tempfile.TemporaryDirectory() as td:
            meta = {"t0_monotonic": 0.0}  # no "face" key
            self.assertIsNone(
                render._facecam_overlay(td, meta, 1280, 720, enabled=True))

    def test_disabled_yields_no_overlay(self):
        from autocine import render
        with tempfile.TemporaryDirectory() as td:
            _write_face(os.path.join(td, "face.mov"), (0, 255, 0))
            meta = {"t0_monotonic": 0.0, "face": "face.mov",
                    "face_t0_monotonic": 0.0, "face_fps": 30}
            self.assertIsNone(
                render._facecam_overlay(td, meta, 1280, 720, enabled=False))
            self.assertIsNotNone(
                render._facecam_overlay(td, meta, 1280, 720, enabled=True))


if __name__ == "__main__":
    unittest.main()
