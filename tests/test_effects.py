"""Unit tests for the render-time overlays (click ripples + spotlight).

Pure numpy/cv2 — no ffmpeg, no screen/input permissions. Run with:
    python3 -m unittest discover -s tests
"""
import unittest

import numpy as np

from autocine import effects


class ResolveColor(unittest.TestCase):
    def test_named(self):
        self.assertEqual(effects.resolve_color("yellow"), (60, 220, 250))

    def test_hex_is_bgr(self):
        self.assertEqual(effects.resolve_color("#ff0000"), (0, 0, 255))  # red
        self.assertEqual(effects.resolve_color("00ff00"), (0, 255, 0))   # green

    def test_tuple_passthrough(self):
        self.assertEqual(effects.resolve_color((1, 2, 3)), (1, 2, 3))

    def test_bad_falls_back(self):
        self.assertEqual(effects.resolve_color("not-a-color"),
                         effects.CLICK_DEFAULTS["color"])
        self.assertEqual(effects.resolve_color(None), (244, 244, 244))


class EaseOut(unittest.TestCase):
    def test_boundaries_and_monotonic(self):
        self.assertAlmostEqual(effects._ease_out(0.0), 0.0)
        self.assertAlmostEqual(effects._ease_out(1.0), 1.0)
        vals = [effects._ease_out(u / 10.0) for u in range(11)]
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))


class ClickFXDraw(unittest.TestCase):
    def _img(self):
        return np.zeros((80, 100, 3), np.uint8)  # H=80, W=100

    def test_pulse_marks_click_location(self):
        img = self._img()
        fx = effects.ClickFX(100, 80, [0.5], [50.0], [40.0])
        fx.draw(img, 0.5, 0.0, 0.0, 1.0)          # u=0 -> pulse at (50,40)
        self.assertTrue(img[40, 50].any(), "click center should be painted")
        self.assertFalse(img[0, 0].any(), "far corner untouched")

    def test_coordinate_mapping_with_zoom_offset(self):
        # click at screen (60,50); window top-left (10,10), zoom 2 -> (100,80)
        img = np.zeros((200, 300, 3), np.uint8)
        fx = effects.ClickFX(300, 200, [0.5], [60.0], [50.0])
        fx.draw(img, 0.5, 10.0, 10.0, 2.0)
        ys, xs = np.where(img.any(axis=2))
        self.assertTrue(len(xs) > 0)
        # centroid of painted pixels should sit at the mapped point
        self.assertAlmostEqual(xs.mean(), 100.0, delta=6)
        self.assertAlmostEqual(ys.mean(), 80.0, delta=6)

    def test_offframe_click_is_noop(self):
        img = self._img()
        fx = effects.ClickFX(100, 80, [0.5], [5000.0], [40.0])
        fx.draw(img, 0.5, 0.0, 0.0, 1.0)
        self.assertEqual(int(img.sum()), 0)

    def test_after_duration_is_noop(self):
        img = self._img()
        fx = effects.ClickFX(100, 80, [0.5], [50.0], [40.0])
        fx.draw(img, 5.0, 0.0, 0.0, 1.0)          # long after the ripple faded
        self.assertEqual(int(img.sum()), 0)

    def test_ring_grows_over_lifetime(self):
        def painted(t):
            img = self._img()
            effects.ClickFX(100, 80, [0.0], [50.0], [40.0]).draw(img, t, 0, 0, 1.0)
            return int(img.any(axis=2).sum())
        early = painted(0.10)
        late = painted(0.45)
        self.assertGreater(late, early)          # ring radius expands


class SpotlightDraw(unittest.TestCase):
    def test_center_bright_corner_dim(self):
        img = np.full((200, 320, 3), 200, np.uint8)
        effects.Spotlight(320, 200, {"dim": 0.45}).draw(img, 160, 100, 2.0)
        self.assertTrue(abs(int(img[100, 160, 0]) - 200) <= 5)   # core preserved
        self.assertTrue(abs(int(img[0, 0, 0]) - 110) <= 5)       # 200*(1-0.45)

    def test_zoom_only_gate(self):
        img = np.full((200, 320, 3), 200, np.uint8)
        effects.Spotlight(320, 200).draw(img, 160, 100, 1.0)     # z<=1.02 -> skip
        self.assertEqual(int(img.min()), 200)
        effects.Spotlight(320, 200).draw(img, 160, 100, 2.0)     # now it dims
        self.assertLess(int(img.min()), 200)


class CursorFXDraw(unittest.TestCase):
    def _img(self):
        return np.zeros((200, 300, 3), np.uint8)

    def test_maps_through_zoom_offset(self):
        frame_times = np.arange(5) / 60.0
        moves_t = np.array([0.0])
        moves_x = np.array([60.0])
        moves_y = np.array([50.0])
        fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y)
        img = self._img()
        # window top-left (10,10), zoom 2 -> tip maps to ((60-10)*2, (50-10)*2)
        fx.draw(img, len(frame_times) - 1, 10.0, 10.0, 2.0)
        ys, xs = np.where(img.any(axis=2))
        self.assertTrue(len(xs) > 0)
        self.assertAlmostEqual(xs.min(), 100.0, delta=3)
        self.assertAlmostEqual(ys.min(), 80.0, delta=3)

    def test_size_scales_with_zoom(self):
        frame_times = np.arange(5) / 60.0
        moves_t = np.array([0.0])
        moves_x = np.array([60.0])
        moves_y = np.array([40.0])
        fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y)

        def painted(z):
            img = self._img()
            fx.draw(img, 4, 0.0, 0.0, z)
            return int(img.any(axis=2).sum())

        self.assertGreater(painted(2.5), painted(1.0))

    def test_scale_param_enlarges_cursor(self):
        frame_times = np.arange(5) / 60.0
        moves_t = np.array([0.0])
        moves_x = np.array([150.0])
        moves_y = np.array([100.0])

        def painted(scale):
            fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y,
                                  params={"scale": scale})
            img = self._img()
            fx.draw(img, 4, 0.0, 0.0, 1.0)
            return int(img.any(axis=2).sum())

        self.assertGreater(painted(2.0), painted(1.0))

    def test_idle_hide_fades_out_then_noop(self):
        fps = 60.0
        frame_times = np.arange(int(3.0 * fps)) / fps
        moves_t = np.array([0.0])
        moves_x = np.array([100.0])
        moves_y = np.array([80.0])
        fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y,
                              params={"idle_after": 0.5, "fade_dur": 0.3})
        self.assertGreater(fx.alpha[0], 0.99)
        idx_late = int(2.5 * fps)   # well past idle_after + fade_dur
        self.assertEqual(fx.alpha[idx_late], 0.0)
        img = self._img()
        fx.draw(img, idx_late, 0.0, 0.0, 1.0)
        self.assertEqual(int(img.sum()), 0)

    def test_click_pulse_decays(self):
        frame_times = np.arange(60) / 60.0
        moves_t = np.array([0.0])
        moves_x = np.array([100.0])
        moves_y = np.array([80.0])
        fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y,
                              clicks_t=[0.5],
                              params={"click_pulse": 0.3, "click_pulse_dur": 0.1})
        self.assertGreater(fx.pulse[int(0.5 * 60)], 0.2)
        self.assertEqual(fx.pulse[int(0.9 * 60)], 0.0)

    def test_tilt_follows_horizontal_velocity(self):
        frame_times = np.arange(10) / 60.0
        moves_t = np.array([0.0, 10 / 60.0])
        moves_x = np.array([0.0, 1000.0])
        moves_y = np.array([50.0, 50.0])
        fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y,
                              params={"tilt_gain": 1.0, "tilt_max_deg": 20.0})
        self.assertGreater(fx.tilt[5], 0.0)

    def test_empty_frame_times_is_safe(self):
        fx = effects.CursorFX(300, 200, np.array([]), np.array([]), np.array([]), np.array([]))
        img = self._img()
        fx.draw(img, 0, 0, 0, 1.0)
        self.assertEqual(int(img.sum()), 0)

    def test_out_of_range_index_is_noop(self):
        frame_times = np.arange(5) / 60.0
        moves_t = np.array([0.0])
        moves_x = np.array([100.0])
        moves_y = np.array([80.0])
        fx = effects.CursorFX(300, 200, frame_times, moves_t, moves_x, moves_y)
        img = self._img()
        fx.draw(img, 999, 0.0, 0.0, 1.0)
        self.assertEqual(int(img.sum()), 0)


if __name__ == "__main__":
    unittest.main()
