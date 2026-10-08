"""Webcam preview frames, owned by Python and streamed as MJPEG.

**Why this exists.** The native bar renders in pywebview's WKWebView, and that
runtime exposes *no* `navigator.mediaDevices` at all — `getUserMedia` is not
merely permission-gated, the API is absent (measured; `typeof
navigator.mediaDevices === "undefined"`). So a browser-side webcam preview can
never work in the packaged app, however the permissions are set. Python holds
the camera instead and serves frames over the local HTTP server; the page just
points an `<img>` at it. The same path works in a real browser, so there is one
implementation rather than a native one and a web one.

This is also the shape a distributable app wants: the camera TCC grant belongs
to the bundle (which already needs it to record), not to an embedded web view.

**Device sharing.** macOS hands a camera to one process at a time, and the
recorder needs it. One capture is shared by all preview clients and reference
counted; `suspend()` drops it and refuses to reopen until `resume()`, which is
how the recorder takes the device without racing the preview.
"""

import os
import subprocess
import threading
import time

# OpenCV's AVFoundation backend tries to request camera authorization itself,
# which it can only do from the main thread ("can not spin main run loop from
# other thread") — and preview captures always run on a worker. Skipping its
# request makes the open succeed when the process is already authorized and
# fail cleanly when it isn't, instead of deadlocking. Must be set before the
# first VideoCapture; the app (or the packaged bundle, via
# NSCameraUsageDescription) owns the actual grant.
os.environ.setdefault("OPENCV_AVFOUNDATION_SKIP_AUTH", "1")

import cv2
import numpy as np

# Preview only — small and cheap. The recording itself never goes through here.
# The bubble is shown at up to 168pt square (336px physical on Retina), so
# 320x240 upscaled there was visibly soft; 480x360 is still cheap to encode
# at this frame rate and holds up under that upscale.
FRAME_W = 480
FRAME_H = 360
FPS = 30
JPEG_QUALITY = 78
# Stop grabbing once every client has gone away.
IDLE_GRACE_SEC = 2.0


class _FaceSink(object):
    """Writes the live camera frames to face.mov via an ffmpeg pipe.

    The facecam used to be a SECOND avfoundation capture in its own ffmpeg,
    which meant the preview had to release the camera for the whole take --
    macOS gives a camera to one process at a time. So you lost your own face
    exactly when you most wanted to see it. This takes the frames the preview
    thread is already pulling and pipes them to an encoder instead, so one
    device read feeds both.

    `t0` is stamped on the FIRST frame actually written, which is what
    `meta["face_t0_monotonic"]` means and what render lines the bubble up
    against. Best-effort throughout: a facecam failure has never been allowed
    to take down a screen recording and still isn't.
    """

    def __init__(self, path, fps, size):
        self.path = path
        self.fps = int(fps)
        self.w, self.h = int(size[0]), int(size[1])
        self.t0 = None
        self.frames = 0
        self.proc = None
        self.error = None

    def open(self):
        # rawvideo in, h264 out. -r on BOTH sides: the input rate declares
        # what the pipe carries, the output rate forces CFR so frame index ->
        # time stays exact, the same contract raw.mov is written under.
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "{}x{}".format(self.w, self.h),
               "-r", str(self.fps),
               "-i", "pipe:0",
               "-c:v", "libx264", "-preset", "ultrafast",
               "-tune", "zerolatency",
               "-crf", "20", "-pix_fmt", "yuv420p",
               "-r", str(self.fps), self.path]
        try:
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, start_new_session=True)
            return True
        except Exception as exc:
            self.error = str(exc)
            self.proc = None
            return False

    def write(self, frame):
        proc = self.proc
        if proc is None or proc.stdin is None:
            return
        # A camera that renegotiates resolution mid-stream would desync the
        # raw pipe (ffmpeg is told one size up front), so coerce instead of
        # trusting it.
        if frame.shape[1] != self.w or frame.shape[0] != self.h:
            try:
                frame = cv2.resize(frame, (self.w, self.h),
                                   interpolation=cv2.INTER_AREA)
            except Exception:
                return
        try:
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except Exception:
            # Broken pipe: the encoder died. Stop feeding it rather than
            # raising on the capture thread, which would kill the preview too.
            self.proc = None
            return
        if self.t0 is None:
            self.t0 = time.monotonic()
        self.frames += 1

    def close(self):
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=8)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


class _Camera(object):
    """One shared cv2 capture, kept warm while clients are reading it."""

    def __init__(self, ordinal):
        self.ordinal = ordinal
        self.lock = threading.Lock()
        self.clients = 0
        self.latest = None          # most recent JPEG bytes
        self.seq = 0                # bumps per frame, so readers can wait
        self.cond = threading.Condition(self.lock)
        self.thread = None
        self.stop = threading.Event()
        self.error = None
        self.sink = None            # _FaceSink while a take is recording

    def _run(self):
        cap = None
        try:
            cap = cv2.VideoCapture(self.ordinal, cv2.CAP_AVFOUNDATION)
            if not cap.isOpened():
                with self.cond:
                    # by far the most likely cause, and not self-evident
                    self.error = (
                        "could not open camera {} — grant this app Camera "
                        "access in System Settings > Privacy & Security, then "
                        "relaunch it".format(self.ordinal))
                    self.cond.notify_all()
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            # Ask for the shallowest internal capture queue. Uncapped, the
            # backend can hold a couple of frames in flight, which shows up
            # as the preview trailing slightly behind the real camera even
            # though our own read loop is right on pace. Harmless if the
            # backend ignores it.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            period = 1.0 / float(FPS)
            # cap.read() already blocks until the camera's own next frame is
            # ready (it paces itself), so sleeping the full period *on top*
            # of that on every iteration was doubling the real interval
            # between frames — the preview ran at roughly half of FPS and
            # looked choppy. Track an absolute next-tick instead: only sleep
            # the time actually left in the period, and never sleep at all
            # when read()+encode() already ate the whole budget.
            next_tick = time.monotonic() + period
            while not self.stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    time.sleep(period)
                    next_tick = time.monotonic() + period
                    continue
                try:
                    ok, buf = cv2.imencode(".jpg", frame, params)
                except Exception:
                    ok = False
                if ok:
                    with self.cond:
                        self.latest = buf.tobytes()
                        self.seq += 1
                        self.cond.notify_all()
                # Tee the SAME frame into face.mov when a take is running.
                # This is why the preview no longer goes dark while
                # recording: macOS hands a camera to one process at a time,
                # so as long as a second ffmpeg owned the device, the preview
                # had to give it up. One owner, two outputs.
                sink = self.sink
                if sink is not None:
                    sink.write(frame)
                now = time.monotonic()
                if next_tick > now:
                    time.sleep(next_tick - now)
                    next_tick += period
                else:
                    # fell behind (slow read/encode) — resync rather than
                    # bursting frames to catch up
                    next_tick = now + period
        except Exception as exc:                     # pragma: no cover - device
            with self.cond:
                self.error = str(exc)
                self.cond.notify_all()
        finally:
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
            with self.cond:
                self.latest = None
                # Bump seq alongside the None so waiters actually WAKE on
                # shutdown. `frames` blocks on `seq == seen`; leaving seq
                # unchanged here meant a notified reader re-checked, saw no
                # change, and went back to waiting until its 10s timeout
                # instead of seeing the `latest is None` end-of-stream
                # sentinel immediately below it.
                self.seq += 1
                self.cond.notify_all()

    def start(self):
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop.clear()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def shutdown(self):
        self.stop.set()
        t = self.thread
        if t is not None:
            t.join(timeout=2.0)
        self.thread = None
        with self.cond:
            self.cond.notify_all()


class _Manager(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.cams = {}
        self.suspended = False

    def _get(self, ordinal):
        cam = self.cams.get(ordinal)
        if cam is None:
            cam = _Camera(ordinal)
            self.cams[ordinal] = cam
        return cam

    def frames(self, ordinal, timeout=10.0):
        """Yield JPEG frames until the caller stops consuming.

        Raises RuntimeError when the camera can't be opened or is suspended
        for a recording, so the endpoint can answer with a real status.
        """
        with self.lock:
            if self.suspended:
                raise RuntimeError("camera is in use by the recorder")
            cam = self._get(ordinal)
            cam.clients += 1
            cam.start()
            # Subscribe from the camera's CURRENT frame, never from zero.
            # _Camera objects are cached in self.cams and REUSED across opens
            # while `seq` keeps counting, so a hardcoded 0 made a reopened
            # camera look like it had already delivered a frame: the wait loop
            # below was skipped, and it read the `latest` that _run's finally
            # had just cleared to None -- reporting end-of-stream instantly.
            # Every open after the first answered 503 "camera produced no
            # frames". Latent until the preview started being released and
            # reopened rather than held for the life of the page.
            seen = cam.seq
        try:
            deadline = time.time() + timeout
            while True:
                with cam.cond:
                    while cam.seq == seen and cam.error is None:
                        if not cam.cond.wait(timeout=1.0):
                            if time.time() > deadline and cam.latest is None:
                                raise RuntimeError(
                                    cam.error or "camera produced no frames")
                    if cam.error is not None:
                        raise RuntimeError(cam.error)
                    seen = cam.seq
                    frame = cam.latest
                if frame is None:
                    return
                deadline = time.time() + timeout
                yield frame
        finally:
            with self.lock:
                cam.clients = max(0, cam.clients - 1)
                if cam.clients == 0:
                    cam.shutdown()

    def start_recording(self, ordinal, path, fps=FPS):
        """Tee the live preview into `path` for the duration of a take.

        Returns True when the sink is armed. The point of this over the old
        second-ffmpeg facecam is that the preview KEEPS RUNNING: one device
        read now feeds both the bubble you're watching and the file. Requires
        a camera already open for preview, which is the state the bar is in
        whenever a camera is picked.
        """
        with self.lock:
            if self.suspended:
                return False
            cam = self.cams.get(int(ordinal))
        if cam is None or cam.thread is None or not cam.thread.is_alive():
            return False
        sink = _FaceSink(path, fps, (FRAME_W, FRAME_H))
        if not sink.open():
            return False
        cam.sink = sink
        return True

    def stop_recording(self, ordinal):
        """Close the face sink. Returns `(t0_monotonic, frame_count)`.

        `t0` is when the first frame actually hit the file, which is what
        meta's `face_t0_monotonic` has always meant -- the anchor render lines
        the bubble up against. `(None, 0)` when nothing was written.
        """
        with self.lock:
            cam = self.cams.get(int(ordinal))
        if cam is None:
            return None, 0
        sink, cam.sink = cam.sink, None
        if sink is None:
            return None, 0
        sink.close()
        return sink.t0, sink.frames

    def release(self):
        """Authoritatively drop every preview camera not feeding a take.

        THE CLIENT CANNOT CLOSE ITS OWN STREAM. `stopCam()` in bar.js drops
        the `<img>`'s src and the code has always assumed that ends the
        request -- but WKWebView keeps a `multipart/x-mixed-replace` load
        running after the element stops pointing at it, so the socket, the
        refcount, and the camera all outlive the UI that wanted them.
        Measured on the native bar: one preview connection survived a
        recording, the done face, and an explicit "No camera" pick, streaming
        1.6 GB the whole time with the webcam lit and nothing on screen using
        it. Since `frames()` only decrements when that loop ends, the device
        could not be released by any client action at all.

        So the release has to happen on this side. Shutting the capture down
        makes `_run` publish its end-of-stream sentinel, which ends the
        generator, which closes the socket from the server -- the reverse of
        the order that doesn't work.

        Cameras with a sink armed are SKIPPED: that sink is a take's
        `face.mov` being teed off this very capture, and dropping it would
        silently truncate the face track. Unlike `shutdown()`, this leaves
        the `_Camera` objects in place so the next `frames()` reuses them
        (see the seq/latest contract there).
        """
        with self.lock:
            cams = [c for c in self.cams.values() if c.sink is None]
        for cam in cams:
            cam.shutdown()
        return len(cams)

    def suspend(self):
        """Release every camera and refuse to reopen — for the paths that
        still hand the device to a separate capture process."""
        with self.lock:
            self.suspended = True
            cams = list(self.cams.values())
        for cam in cams:
            cam.shutdown()

    def resume(self):
        with self.lock:
            self.suspended = False

    def shutdown(self):
        with self.lock:
            cams = list(self.cams.values())
            self.cams = {}
        for cam in cams:
            cam.shutdown()


_manager = _Manager()

frames = _manager.frames
start_recording = _manager.start_recording
stop_recording = _manager.stop_recording
suspend = _manager.suspend
resume = _manager.resume
release = _manager.release
shutdown = _manager.shutdown


def is_suspended():
    return _manager.suspended


# --- macOS camera authorization ---------------------------------------
# AVFoundation's media-type constant for video. Hard-coded because pyobjc has
# no AVFoundation module here; the framework is loaded by path below.
_AV_VIDEO = "vide"
NOT_DETERMINED, RESTRICTED, DENIED, AUTHORIZED = 0, 1, 2, 3


def _av_capture_device():
    try:
        import objc
        try:
            cls = objc.lookUpClass("AVCaptureDevice")
        except Exception:
            objc.loadBundle(
                "AVFoundation", globals(),
                bundle_path="/System/Library/Frameworks/AVFoundation.framework")
            cls = objc.lookUpClass("AVCaptureDevice")
        return cls
    except Exception:
        return None


def authorization_status():
    """0 not-determined, 1 restricted, 2 denied, 3 authorized, None unknown."""
    cls = _av_capture_device()
    if cls is None:
        return None
    try:
        return int(cls.authorizationStatusForMediaType_(_AV_VIDEO))
    except Exception:
        return None


def request_access():
    """Ask macOS for camera access, showing the standard prompt.

    Only meaningful while the status is not-determined; once the user has
    answered, macOS never prompts again and the grant has to be changed in
    System Settings. Call this from the main thread at startup — OpenCV can't
    do it from a worker (see OPENCV_AVFOUNDATION_SKIP_AUTH above), which is
    exactly why the preview would otherwise fail with no prompt ever shown.
    """
    cls = _av_capture_device()
    if cls is None:
        return False
    try:
        if int(cls.authorizationStatusForMediaType_(_AV_VIDEO)) != NOT_DETERMINED:
            return False
        cls.requestAccessForMediaType_completionHandler_(
            _AV_VIDEO, lambda granted: None)
        return True
    except Exception:
        return False


def list_cameras(devices):
    """Real cameras from an avfoundation device listing, screens removed.

    `index` is the avfoundation index (what the recorder wants); `ordinal` is
    the position among cameras only, which is what OpenCV's VideoCapture takes
    — the two differ whenever a screen device sorts before a camera, so they
    must not be used interchangeably.
    """
    out = []
    ordinal = 0
    for idx, name in (devices or {}).get("video", []):
        if "capture screen" in str(name).lower():
            continue
        out.append({"index": idx, "ordinal": ordinal, "name": name})
        ordinal += 1
    return out
