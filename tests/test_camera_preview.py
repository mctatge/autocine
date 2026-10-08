"""Python-owned webcam preview (autocine/camera_preview.py).

Runs without a camera and without permissions: the cv2 capture is stubbed, so
these pin the sharing/refcount/suspend contract rather than the device.
"""

import os
import sys
import threading
import time
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autocine import camera_preview as cp


class FakeCapture(object):
    """Stands in for cv2.VideoCapture. Counts opens/releases so the tests can
    assert the device is actually handed back."""

    opened = 0
    released = 0
    instances = []

    def __init__(self, ordinal, backend=None, fail=False):
        self.ordinal = ordinal
        self._fail = fail
        self._open = not fail
        FakeCapture.opened += 1
        FakeCapture.instances.append(self)

    def isOpened(self):
        return self._open

    def set(self, *a):
        return True

    def read(self):
        if not self._open:
            return False, None
        return True, object()

    def release(self):
        self._open = False
        FakeCapture.released += 1

    @classmethod
    def reset(cls):
        cls.opened = 0
        cls.released = 0
        cls.instances = []


class CameraPreviewSharing(unittest.TestCase):
    def setUp(self):
        FakeCapture.reset()
        self.mgr = cp._Manager()
        self._real_cap = cp.cv2.VideoCapture
        self._real_enc = cp.cv2.imencode
        cp.cv2.VideoCapture = lambda o, b=None: FakeCapture(o, b)
        cp.cv2.imencode = lambda ext, frame, params=None: (True, _Buf())

    def tearDown(self):
        cp.cv2.VideoCapture = self._real_cap
        cp.cv2.imencode = self._real_enc
        self.mgr.shutdown()

    def _take(self, ordinal, n=2):
        """Pull n frames from a stream, then close it."""
        gen = self.mgr.frames(ordinal)
        got = [next(gen) for _ in range(n)]
        gen.close()
        return got

    def test_frames_are_jpeg_bytes(self):
        frames = self._take(0, 2)
        self.assertEqual(len(frames), 2)
        for f in frames:
            self.assertIsInstance(f, bytes)

    def test_one_capture_is_shared_by_two_clients(self):
        a = self.mgr.frames(0)
        next(a)
        b = self.mgr.frames(0)
        next(b)
        self.assertEqual(FakeCapture.opened, 1, "should not open the device twice")
        a.close()
        b.close()

    def test_the_device_is_released_when_the_last_client_leaves(self):
        a = self.mgr.frames(0)
        next(a)
        b = self.mgr.frames(0)
        next(b)
        a.close()
        self.assertEqual(FakeCapture.released, 0, "still one client reading")
        b.close()
        deadline = time.time() + 3
        while FakeCapture.released == 0 and time.time() < deadline:
            time.sleep(0.02)
        self.assertEqual(FakeCapture.released, 1)

    def test_separate_ordinals_get_separate_captures(self):
        a = self.mgr.frames(0)
        next(a)
        b = self.mgr.frames(1)
        next(b)
        self.assertEqual(FakeCapture.opened, 2)
        self.assertEqual(sorted(c.ordinal for c in FakeCapture.instances), [0, 1])
        a.close()
        b.close()

    def test_a_camera_that_will_not_open_raises_rather_than_hanging(self):
        cp.cv2.VideoCapture = lambda o, b=None: FakeCapture(o, b, fail=True)
        with self.assertRaises(RuntimeError) as ctx:
            next(self.mgr.frames(0))
        self.assertIn("could not open camera", str(ctx.exception))

    def test_the_open_error_names_the_likely_cause(self):
        """A missing Camera grant is the overwhelmingly common reason and is
        not self-evident — the message has to say so."""
        cp.cv2.VideoCapture = lambda o, b=None: FakeCapture(o, b, fail=True)
        with self.assertRaises(RuntimeError) as ctx:
            next(self.mgr.frames(0))
        self.assertIn("Camera access", str(ctx.exception))


class RecorderHandshake(unittest.TestCase):
    """macOS gives a camera to one process at a time, so the preview has to
    let go before the recorder opens the device."""

    def setUp(self):
        FakeCapture.reset()
        self.mgr = cp._Manager()
        self._real_cap = cp.cv2.VideoCapture
        self._real_enc = cp.cv2.imencode
        cp.cv2.VideoCapture = lambda o, b=None: FakeCapture(o, b)
        cp.cv2.imencode = lambda ext, frame, params=None: (True, _Buf())

    def tearDown(self):
        cp.cv2.VideoCapture = self._real_cap
        cp.cv2.imencode = self._real_enc
        self.mgr.shutdown()

    def test_suspend_releases_the_device(self):
        gen = self.mgr.frames(0)
        next(gen)
        self.assertEqual(FakeCapture.released, 0)
        self.mgr.suspend()
        self.assertEqual(FakeCapture.released, 1)
        gen.close()

    def test_suspended_refuses_new_streams(self):
        self.mgr.suspend()
        with self.assertRaises(RuntimeError) as ctx:
            next(self.mgr.frames(0))
        self.assertIn("recorder", str(ctx.exception))

    def test_resume_allows_streaming_again(self):
        self.mgr.suspend()
        self.mgr.resume()
        gen = self.mgr.frames(0)
        self.assertIsInstance(next(gen), bytes)
        gen.close()


class CameraListing(unittest.TestCase):
    """Screens are video devices too; only real cameras belong in the picker.
    The avfoundation index and the OpenCV ordinal are different numbers and
    must not be swapped."""

    def test_screens_are_stripped_and_ordinals_are_camera_relative(self):
        devs = {"video": [(0, "FaceTime HD Camera"),
                          (1, "Capture screen 0"),
                          (2, "Logitech BRIO")]}
        self.assertEqual(cp.list_cameras(devs), [
            {"index": 0, "ordinal": 0, "name": "FaceTime HD Camera"},
            {"index": 2, "ordinal": 1, "name": "Logitech BRIO"},
        ])

    def test_a_screen_before_the_camera_shifts_index_but_not_ordinal(self):
        devs = {"video": [(0, "Capture screen 0"), (1, "FaceTime HD Camera")]}
        self.assertEqual(cp.list_cameras(devs),
                         [{"index": 1, "ordinal": 0, "name": "FaceTime HD Camera"}])

    def test_no_cameras_and_junk_input_are_survivable(self):
        self.assertEqual(cp.list_cameras({"video": [(0, "Capture screen 0")]}), [])
        self.assertEqual(cp.list_cameras({}), [])
        self.assertEqual(cp.list_cameras(None), [])


class Authorization(unittest.TestCase):
    def test_status_is_reported_as_an_int_or_none(self):
        st = cp.authorization_status()
        self.assertTrue(st is None or st in (0, 1, 2, 3), st)

    def test_requesting_when_already_answered_is_a_no_op(self):
        """Only not-determined can prompt; anything else must not try, so a
        launch never blocks on a dialog that macOS will never show."""
        st = cp.authorization_status()
        if st is None or st == cp.NOT_DETERMINED:
            self.skipTest("camera permission is unanswered on this machine")
        self.assertFalse(cp.request_access())

    def test_opencv_auth_request_is_disabled(self):
        """OpenCV prompting from a worker thread deadlocks; the app prompts."""
        self.assertEqual(os.environ.get("OPENCV_AVFOUNDATION_SKIP_AUTH"), "1")


class _Buf(object):
    def tobytes(self):
        return b"\xff\xd8\xff-fake-jpeg"


if __name__ == "__main__":
    unittest.main()
