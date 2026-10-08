"""Unit tests for the aspect-ratio-aware camera window / warp / effect-scale
math in render.py. Pure numpy/cv2 -- no ffmpeg, no screen/input permissions.
"""

import unittest

import numpy as np

from autocine import render


class ContainFit(unittest.TestCase):
    def test_matching_aspect_is_identity(self):
        self.assertEqual(render._contain_fit(1920, 1080, 1920, 1080), (1920.0, 1080.0))

    def test_vertical_target_from_landscape_source(self):
        win_w, win_h = render._contain_fit(1080, 1920, 1920, 1080)
        self.assertAlmostEqual(win_w, 607.5)
        self.assertAlmostEqual(win_h, 1080.0)
        # the window's own aspect must match the requested output aspect
        self.assertAlmostEqual(win_w / win_h, 1080.0 / 1920.0, places=6)


class CameraWindowAndWarp(unittest.TestCase):
    def test_auto_case_matches_manual_source_sized_formula(self):
        W, H, z = 1200.0, 800.0, 2.0
        cx, cy = 700.0, 300.0
        x0, y0 = render._camera_window(cx, cy, z, W, H, W, H)
        cw, ch = W / z, H / z
        expected_x0 = min(max(cx - cw / 2.0, 0.0), W - cw)
        expected_y0 = min(max(cy - ch / 2.0, 0.0), H - ch)
        self.assertAlmostEqual(x0, expected_x0)
        self.assertAlmostEqual(y0, expected_y0)

    def test_narrow_window_produces_a_smaller_crop_than_source_sized(self):
        W, H, z = 1920.0, 1080.0, 1.0
        win_w, win_h = 607.5, 1080.0
        x0, y0 = render._camera_window(960.0, 540.0, z, win_w, win_h, W, H)
        # crop width implied is win_w/z, much narrower than the full source
        self.assertAlmostEqual(x0, 960.0 - win_w / 2.0)
        self.assertLess(win_w, W)

    def test_warp_auto_case_output_shape_matches_source(self):
        W, H = 320, 240
        frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        out = render._warp(frame, 0.0, 0.0, 1.0, W, H, W, H)
        self.assertEqual(out.shape, (H, W, 3))

    def test_warp_identity_reproduces_source_at_z1(self):
        # z=1, win == source, out == source -> should be (numerically, up to
        # cubic-interpolation edge effects) the identity transform.
        W, H = 200, 150
        frame = np.zeros((H, W, 3), np.uint8)
        frame[40:60, 80:120] = (10, 200, 30)
        out = render._warp(frame, 0.0, 0.0, 1.0, W, H, W, H)
        # a patch well inside the block, away from interpolation edges
        np.testing.assert_allclose(out[45:55, 90:110], frame[45:55, 90:110], atol=2)

    def test_warp_to_a_different_canvas_size_fills_it_exactly(self):
        W, H = 640, 480
        win_w, win_h = 240.0, 480.0    # a vertical strip of the source
        out_w, out_h = 480, 960        # upscaled vertical canvas
        frame = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
        out = render._warp(frame, 0.0, 0.0, 1.0, win_w, win_h, out_w, out_h)
        self.assertEqual(out.shape, (out_h, out_w, 3))


class ZoomToOutputScale(unittest.TestCase):
    def test_auto_case_is_identity(self):
        self.assertAlmostEqual(render._zoom_to_output_scale(2.5, 1920, 1920), 2.5)

    def test_scales_with_canvas_to_window_ratio(self):
        # out_w twice win_w -> the same camera zoom maps to twice the pixel scale
        self.assertAlmostEqual(render._zoom_to_output_scale(1.0, 500.0, 1000.0), 2.0)

    def test_floors_at_one(self):
        self.assertAlmostEqual(render._zoom_to_output_scale(0.3, 500.0, 500.0), 1.0)


if __name__ == "__main__":
    unittest.main()
