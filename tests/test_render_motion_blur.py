"""Unit tests for camera motion blur (render._motion_blur_cams/_warp_blend).

Pure math + cv2 warps on synthetic frames -- no ffmpeg, no permissions. These
pin the shutter-window contract: no blur for a slow/static camera, sample
count scaling with on-screen motion (and its hard cap), sub-samples centered
on the frame instant, zoom-change blur, and the blend being exactly the box
average of the member warps.
"""
import unittest

import cv2
import numpy as np

from autocine import render as ren


def _path(rows):
    return np.array(rows, dtype=np.float64)


class MotionBlurCams(unittest.TestCase):
    OUT_W, OUT_H = 1280, 800   # win == out == source (aspect auto, no framing)

    def _cams(self, path, i):
        return ren._motion_blur_cams(path, i, self.OUT_W, self.OUT_H,
                                     self.OUT_W, self.OUT_H)

    def test_static_camera_single_sample(self):
        path = _path([[640, 400, 1.0]] * 5)
        cams = self._cams(path, 2)
        self.assertEqual(cams, [(640.0, 400.0, 1.0)])

    def test_single_frame_path_no_blur(self):
        path = _path([[640, 400, 2.0]])
        self.assertEqual(len(self._cams(path, 0)), 1)

    def test_slow_drift_below_threshold_no_blur(self):
        # 1 px/frame at z=1 -> 0.5 px across the shutter window < _MB_MIN_PX.
        path = _path([[600 + i, 400, 1.0] for i in range(5)])
        self.assertEqual(len(self._cams(path, 2)), 1)

    def test_fast_pan_scales_sample_count(self):
        # dcx=10 px/frame at z=2 -> 20 out-px/frame; x shutter 0.5 -> 10 px
        # of motion -> ceil(10 / 2.0 px-per-sample) = 5 sub-samples.
        path = _path([[600 + 10 * i, 400, 2.0] for i in range(5)])
        self.assertEqual(len(self._cams(path, 2)), 5)

    def test_extreme_pan_caps_at_max_samples(self):
        path = _path([[100 + 200 * i, 400, 2.0] for i in range(5)])
        self.assertEqual(len(self._cams(path, 2)), ren._MB_MAX_SAMPLES)

    def test_sub_samples_centered_on_frame_instant(self):
        # Box samples must average to the nominal camera (no perceived lag)
        # and stay inside the +/- shutter/2 window.
        path = _path([[600 + 10 * i, 400, 2.0] for i in range(5)])
        cams = self._cams(path, 2)
        mean_cx = sum(c[0] for c in cams) / len(cams)
        self.assertAlmostEqual(mean_cx, 620.0, places=9)
        half_window = 10 * ren._MB_SHUTTER / 2.0
        for cx, cy, z in cams:
            self.assertLessEqual(abs(cx - 620.0), half_window + 1e-9)
            self.assertEqual(cy, 400.0)
            self.assertEqual(z, 2.0)

    def test_zoom_ramp_triggers_blur(self):
        # A pure zoom-in (no pan) must still blur: edge content is moving.
        path = _path([[640, 400, 1.4 + 0.05 * i] for i in range(5)])
        cams = self._cams(path, 2)
        self.assertGreater(len(cams), 1)
        zs = [c[2] for c in cams]
        self.assertLess(zs[0], zs[-1])   # ordered along the ramp

    def test_sub_sample_zoom_never_below_one(self):
        # Zoom starting its ramp at exactly 1.0: backward sub-samples would
        # extrapolate below 1.0 without the clamp.
        path = _path([[640, 400, 1.0], [640, 400, 1.0], [640, 400, 1.8]])
        cams = self._cams(path, 1)
        self.assertGreater(len(cams), 1)
        for _, _, z in cams:
            self.assertGreaterEqual(z, 1.0)

    def test_end_frames_use_one_sided_difference(self):
        # First/last frame must not crash and should still detect motion.
        path = _path([[600 + 50 * i, 400, 2.0] for i in range(4)])
        self.assertGreater(len(self._cams(path, 0)), 1)
        self.assertGreater(len(self._cams(path, 3)), 1)


class WarpBlend(unittest.TestCase):
    W, H = 320, 200

    def _frame(self):
        fr = np.zeros((self.H, self.W, 3), np.uint8)
        fr[:, (np.arange(self.W) // 8) % 2 == 0] = 255   # 8px vertical stripes
        return fr

    def _dims(self):
        # win == out == source: the aspect-auto case.
        return (self.W, self.H, self.W, self.H, self.W, self.H)

    def test_single_cam_identical_to_plain_warp(self):
        fr = self._frame()
        blended = ren._warp_blend(fr, [(160.0, 100.0, 1.5)], *self._dims())
        x0, y0 = ren._camera_window(160.0, 100.0, 1.5, self.W, self.H,
                                    self.W, self.H)
        crisp = ren._warp(fr, x0, y0, 1.5, self.W, self.H, self.W, self.H)
        np.testing.assert_array_equal(blended, crisp)

    def test_blend_is_box_average_of_member_warps(self):
        fr = self._frame()
        cams = [(150.0, 100.0, 1.5), (160.0, 100.0, 1.5), (170.0, 100.0, 1.5)]
        blended = ren._warp_blend(fr, cams, *self._dims())
        acc = np.zeros((self.H, self.W, 3), np.float64)
        for cx, cy, z in cams:
            x0, y0 = ren._camera_window(cx, cy, z, self.W, self.H,
                                        self.W, self.H)
            acc += ren._warp(fr, x0, y0, z, self.W, self.H, self.W, self.H,
                             interp=cv2.INTER_LINEAR)
        expected = (acc / len(cams) + 0.5).astype(np.uint8)
        # float32 vs float64 accumulation may differ by 1 at .5 boundaries
        diff = np.abs(blended.astype(int) - expected.astype(int))
        self.assertLessEqual(diff.max(), 1)

    def test_pan_blur_smears_edges_along_motion(self):
        fr = self._frame()
        cams = [(160.0 + dx, 100.0, 1.5) for dx in (-5.0, -2.5, 0.0, 2.5, 5.0)]
        blended = ren._warp_blend(fr, cams, *self._dims())
        x0, y0 = ren._camera_window(160.0, 100.0, 1.5, self.W, self.H,
                                    self.W, self.H)
        crisp = ren._warp(fr, x0, y0, 1.5, self.W, self.H, self.W, self.H)
        row_b = blended[100, :, 0].astype(int)
        row_c = crisp[100, :, 0].astype(int)
        # Hard 0<->255 stripe edges must soften substantially under a 10px
        # sweep, i.e. the max horizontal gradient drops.
        self.assertLess(np.abs(np.diff(row_b)).max(),
                        np.abs(np.diff(row_c)).max())
        self.assertLess(np.abs(np.diff(row_b)).max(), 200)

    def test_blur_preserves_mean_brightness(self):
        # A box average of warps of the same frame must not brighten/darken
        # the image (guards against accumulation/rounding bugs).
        fr = self._frame()
        cams = [(160.0 + dx, 100.0, 1.5) for dx in (-4.0, 0.0, 4.0)]
        blended = ren._warp_blend(fr, cams, *self._dims())
        x0, y0 = ren._camera_window(160.0, 100.0, 1.5, self.W, self.H,
                                    self.W, self.H)
        crisp = ren._warp(fr, x0, y0, 1.5, self.W, self.H, self.W, self.H)
        self.assertAlmostEqual(float(blended.mean()), float(crisp.mean()),
                               delta=2.0)


if __name__ == "__main__":
    unittest.main()
