"""Unit tests for the visual typing anchor (autocine/vision.py).

The decision core (anchor_from_frames) is a pure function over grayscale
arrays -- no video I/O needed for the interesting cases. The decoder
wrapper gets one end-to-end test against a synthetic clip with a blinking
box (ffmpeg lavfi), plus failure-path checks.
"""
import os
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock

import numpy as np

from autocine import vision

FFMPEG = shutil.which("ffmpeg")


def _frames_static(n=6, h=200, w=320, value=128):
    return [np.full((h, w), value, np.uint8) for _ in range(n)]


class AnchorFromFrames(unittest.TestCase):
    def test_static_frames_yield_none(self):
        self.assertIsNone(vision.anchor_from_frames(_frames_static()))

    def test_fewer_than_two_frames_yield_none(self):
        self.assertIsNone(vision.anchor_from_frames([]))
        self.assertIsNone(vision.anchor_from_frames(_frames_static(1)))
        self.assertIsNone(vision.anchor_from_frames(None))

    def test_blinking_box_centroid_found(self):
        frames = _frames_static(8)
        for i in range(1, 8, 2):   # box present in every other frame
            frames[i] = frames[i].copy()
            frames[i][140:156, 240:280] = 255
        a = vision.anchor_from_frames(frames)
        self.assertIsNotNone(a)
        x, y = a
        self.assertAlmostEqual(x, 260.0, delta=25.0)
        self.assertAlmostEqual(y, 148.0, delta=25.0)

    def test_global_change_rejected_as_scroll(self):
        # Every pixel changes between frames (content scroll / video).
        frames = [np.full((200, 320), 60 + 40 * (i % 2), np.uint8)
                  for i in range(6)]
        self.assertIsNone(vision.anchor_from_frames(frames))

    def test_menubar_only_flicker_rejected(self):
        frames = _frames_static(6)
        for i in range(1, 6, 2):
            frames[i] = frames[i].copy()
            frames[i][:4, 280:310] = 255   # "clock" in the excluded strip
        self.assertIsNone(vision.anchor_from_frames(frames))

    def test_compression_noise_rejected(self):
        rng = np.random.RandomState(7)
        base = rng.randint(100, 140, (200, 320)).astype(np.uint8)
        frames = [(base + rng.randint(-4, 5, base.shape)).astype(np.uint8)
                  for _ in range(6)]   # jitter below _DIFF_THRESH
        self.assertIsNone(vision.anchor_from_frames(frames))

    def test_two_equal_blinkers_are_ambiguous_not_averaged(self):
        # Review-confirmed defect: a union centroid lands BETWEEN disjoint
        # blobs (typing here, terminal cursor there) -- where nothing
        # changed. Two comparable loci must return None (click fallback).
        frames = _frames_static(8)
        for i in range(1, 8, 2):
            frames[i] = frames[i].copy()
            frames[i][140:156, 240:280] = 255   # "typing"
            frames[i][40:56, 30:70] = 255       # equal distant blinker
        self.assertIsNone(vision.anchor_from_frames(frames))

    def test_dominant_blob_beats_small_secondary_flicker(self):
        # A clearly-dominant typing region must win despite a small
        # secondary blinker -- and the anchor must sit ON the dominant
        # blob, not drift toward the flicker.
        frames = _frames_static(8)
        for i in range(1, 8, 2):
            frames[i] = frames[i].copy()
            frames[i][130:166, 200:320] = 255   # big typing region
            frames[i][40:46, 30:42] = 255       # tiny spinner
        a = vision.anchor_from_frames(frames)
        self.assertIsNotNone(a)
        x, y = a
        self.assertAlmostEqual(x, 260.0, delta=30.0)
        self.assertAlmostEqual(y, 148.0, delta=30.0)

    def test_streaming_accumulator_matches_batch_entry_point(self):
        frames = _frames_static(8)
        for i in range(1, 8, 2):
            frames[i] = frames[i].copy()
            frames[i][140:156, 240:280] = 255
        acc = vision._DiffAccumulator()
        for f in frames:
            acc.push(f)
        self.assertEqual(acc.anchor(), vision.anchor_from_frames(frames))


class DecoderWrapper(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if FFMPEG is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.dir = tempfile.mkdtemp()
        cls.clip = os.path.join(cls.dir, "blink.mov")
        # static gray screen with a box blinking at (240..280, 140..156)
        subprocess.run(
            [FFMPEG, "-y", "-v", "error", "-f", "lavfi", "-i",
             "color=c=0x606060:size=320x200:rate=30,"
             "drawbox=x=240:y=140:w=40:h=16:color=white:t=fill:"
             "enable='gte(mod(t\\,0.5)\\,0.25)'",
             "-t", "4", "-pix_fmt", "yuv420p", cls.clip],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.dir, ignore_errors=True)

    def test_finds_blinking_box_in_video(self):
        out = vision.typing_visual_anchors(self.clip, [(0.7, 1.7)])
        self.assertEqual(len(out), 1)
        self.assertIsNotNone(out[0])
        self.assertAlmostEqual(out[0]["x"], 260.0, delta=30.0)
        self.assertAlmostEqual(out[0]["y"], 148.0, delta=30.0)
        self.assertEqual(out[0]["start"], 0.7)

    def test_missing_file_degrades_to_nones(self):
        out = vision.typing_visual_anchors("/nonexistent.mov", [(0.0, 1.0)])
        self.assertEqual(out, [None])

    def test_empty_bursts_short_circuit(self):
        self.assertEqual(vision.typing_visual_anchors(self.clip, []), [])

    def test_cache_avoids_redecoding(self):
        vision._cache.clear()
        with mock.patch.object(vision, "typing_visual_anchors",
                               wraps=vision.typing_visual_anchors) as spy:
            a = vision.cached_typing_anchors(self.clip, 1.0, [(0.7, 1.7)])
            b = vision.cached_typing_anchors(self.clip, 1.0, [(0.7, 1.7)])
            self.assertEqual(spy.call_count, 1)
            self.assertEqual(a, b)
            # different events mtime -> fresh analysis
            vision.cached_typing_anchors(self.clip, 2.0, [(0.7, 1.7)])
            self.assertEqual(spy.call_count, 2)

    def test_absurdly_long_burst_is_window_capped_and_still_works(self):
        # The analysis window is clamped to _MAX_WINDOW_SEC from the burst
        # start (grab() decodes every skipped frame; unbounded windows
        # stalled the editor's first preview) -- and a span running far
        # past the clip end must neither crash nor lose the anchor.
        out = vision.typing_visual_anchors(self.clip, [(0.7, 500.0)])
        self.assertIsNotNone(out[0])
        self.assertAlmostEqual(out[0]["x"], 260.0, delta=30.0)

    def test_concurrent_misses_deduplicate_to_one_decode(self):
        import threading
        vision._cache.clear()
        calls = []
        real = vision.typing_visual_anchors

        def slow(raw_path, bursts):
            calls.append(1)
            import time as _t
            _t.sleep(0.15)
            return real(raw_path, bursts)

        with mock.patch.object(vision, "typing_visual_anchors", slow):
            results = [None] * 6
            def hit(i):
                results[i] = vision.cached_typing_anchors(
                    self.clip, 5.0, [(0.7, 1.7)])
            threads = [threading.Thread(target=hit, args=(i,))
                       for i in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(calls), 1, "exactly one thread should decode")
        for r in results:
            self.assertEqual(r, results[0])


if __name__ == "__main__":
    unittest.main()
