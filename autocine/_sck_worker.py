"""_sck_worker.py — standalone child process: capture the screen with
ScreenCaptureKit and write a constant-rate raw.mov.

WHY THIS IS ITS OWN PROCESS, like `_key_worker.py` and for the same class of
reason. ScreenCaptureKit is driven over XPC to `com.apple.replayd`, and an
ObjC exception raised inside one of our callbacks comes back through
`NSXPCConnection _decodeAndInvokeReplyBlockWithEvent:` and calls `abort()`.
Observed on this machine 2026-07-30: rc 134, SIGABRT, from nothing worse than
a Python callback returning a value where PyObjC had declared the block
`void`. That is a typo-grade trigger for a process-ending crash, no Python
`try/except` can survive it, and the `Recorder` frequently lives inside the
long-lived `studio.py app` server process. So SCK is never imported by the
parent — only here, where dying costs one take instead of the whole app.

Every callback below therefore (a) wraps its whole body in try/except and
(b) returns None explicitly. Both are load-bearing, not style.

WHY IT LOOKS LIKE FFMPEG. The parent's process lifecycle — `Popen` with
`start_new_session=True`, SIGINT to stop, non-zero exit means failure, stderr
tailed into the error log — is code that already works and is already tested.
This process impersonates that shape so none of it needs a second version.

CONSTANT FRAME RATE is the whole difficulty. SCK does not deliver a frame
when nothing on screen changed, so arrivals are a sparse subset of the
timeline; appending them as they come produces a variable-rate file, and
`render.py`/`retime.TimeMap` turn frame index into time by dividing by fps.
`sck.plan_slots` (pure, unit-tested, no ObjC) decides which slots each
arrival occupies and where the previous frame must be repeated.

WIRE PROTOCOL, deliberately tiny — one JSON config line on stdin, then plain
lines out. Config goes over stdin rather than argv because argv is visible in
`ps` to every process on the machine.

  stdin:   <config json>\\n
           STOP\\n                  -- graceful stop (SIGINT does the same)
  stdout:  READY\\n
           SIZE <w> <h> <fps>\\n
           T0 <pts0_host_clock_seconds>\\n     -- see sck.host_pts_to_monotonic
           FILTER <applied> <missing>\\n
           STAT <slot> <appended> <dup> <dropped> <notready>\\n   ~1 Hz
           WARN <what> <detail>\\n
           DONE <frames> <media_seconds>\\n
           ERR <domain> <message>\\n
  stderr:  free text only, never payload-shaped

No output at all before READY means setup failed — the same silent-failure
shape the parent already handles for the key worker.
"""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocine import sck  # noqa: E402  (pure; imports no ObjC)

# One arriving frame is held so idle gaps can be filled with what was
# actually on screen during them. SCK hands out buffers from a pool of
# `queueDepth`; holding exactly one of eight leaves seven in flight, which is
# why the depth is raised from the default rather than left alone.
QUEUE_DEPTH = 8

# Bound on an idle gap, in frames. A machine that slept, or a stream stalled
# behind a locked screen, otherwise asks for a fill of hundreds of thousands
# of frames and turns a hiccup into an out-of-disk. 30s at 60fps.
MAX_FILL = 1800

# Frames between forced keyframes. 250 is x264's default keyint, which is
# what the ffmpeg backend has always used -- matching it is the whole point.
KEYFRAME_INTERVAL = 250

# Bits per pixel per second for the mezzanine. See the note in build().
BITS_PER_PIXEL = 0.10

# How often AVAssetWriter flushes a movie fragment (moof + mdat pair) with an
# updated header. The `moov` box lives at the top of the file from the moment
# `startWriting` completes, and each fragment's samples are indexed by its own
# `trun` inside the `moof` — so a process killed mid-take leaves a file that
# is still playable up to the last complete fragment, and cv2/ffmpeg read it
# without complaint. Without this the moov is written ONLY at `finishWriting`
# and a SIGKILL loses every frame in the file (`sck.has_moov` returns False,
# `record.start` rejects the take, moov-atom-not-found on every consumer -- the
# open kill-9 gate in docs/architecture.md and the "one missing trailer away" complaint
# in the native-window capture investigation documented in the internal
# engineering notes).
#
# 1.0s is the classic tradeoff -- ~1s of maximum tail loss on a SIGKILL,
# against the per-fragment write cost. Measured on a 60fps mezzanine bitrate,
# a 1s fragment overhead is ~1 KiB of container per second of video (<0.01%
# of the payload). Lower this to shrink the loss window; raise it to shave
# overhead. 0 (or a negative number) disables fragmenting entirely and
# restores byte-exact pre-C behavior -- required for reproducing the
# monolithic-file layout in comparisons.
MOVIE_FRAGMENT_INTERVAL_SEC = 1.0

# Window-native minimize/restore recovery (see sck.plan_restore_action and
# docs/architecture.md "Occlusion vs. minimize"). A minimized
# desktop-independent window loses its surface and the stream does not
# re-attach on restore; the worker watches the target's on-screen state and
# rebuilds its SCStream when it comes back. ONSCREEN_POLL_SEC is how often the
# main loop checks (cheap: one CGWindowList lookup for a single id). Only fires
# in window-native mode; inert for display capture and for a window that never
# minimizes.
ONSCREEN_POLL_SEC = 0.5
# Give up re-attempting a rebuild after this many consecutive failures, so a
# window that is back on-screen but not yet re-listable in SCShareableContent
# is retried briefly (a few polls) without hammering forever.
MAX_REBUILD_TRIES = 6


def emit(line):
    try:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    except Exception:
        pass


def note(text):
    try:
        sys.stderr.write(text + "\n")
        sys.stderr.flush()
    except Exception:
        pass


class Capture(object):
    """Owns the stream, the writer, and the slot bookkeeping."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.fps = int(cfg.get("fps") or 60)
        self.path = cfg["out"]
        self.exclude = [int(w) for w in (cfg.get("exclude") or [])]
        self.show_cursor = bool(cfg.get("show_cursor", True))
        self.display_id = cfg.get("display_id")
        # WINDOW-NATIVE capture: when set, build() targets this one window's
        # own backing buffer (SCContentFilter.initWithDesktopIndependentWindow_)
        # instead of the display, so occluders never bleed in. None = the
        # normal display capture. See docs/architecture.md.
        cwid = cfg.get("capture_window_id")
        self.capture_window_id = int(cwid) if cwid is not None else None
        # Overridable so a measurement run can sweep them without editing
        # code. `raw_defaults` omits the compression dict entirely, which is
        # the ONLY way to reproduce AVFoundation's unbounded behavior for
        # comparison -- a 0 here would just fall back to our constants and
        # measure the fix against itself.
        self.keyint = int(cfg.get("keyint") or KEYFRAME_INTERVAL)
        self.bpp = float(cfg.get("bpp") or BITS_PER_PIXEL)
        self.raw_defaults = bool(cfg.get("raw_defaults"))
        # Movie-fragment cadence, seconds. Absent/None -> module default (on);
        # <=0 -> disabled (byte-exact pre-C monolithic-file behavior). See the
        # comment on MOVIE_FRAGMENT_INTERVAL_SEC above.
        interval = cfg.get("movie_fragment_interval")
        if interval is None:
            interval = MOVIE_FRAGMENT_INTERVAL_SEC
        try:
            self.movie_fragment_interval = float(interval)
        except (TypeError, ValueError):
            self.movie_fragment_interval = MOVIE_FRAGMENT_INTERVAL_SEC
        # AVFoundation uniqueID string, resolved by the PARENT from the
        # avfoundation ordinal (devices.mic_unique_id). None = no mic.
        self.mic_uid = cfg.get("mic_unique_id") or None
        self.audio_in = None
        self.audio_appended = 0
        self.audio_dropped = 0
        # Mid-take exclusion changes. The stdin reader only RECORDS the
        # request; the main loop applies it. That split is deliberate:
        # resolving window ids means an async SCShareableContent fetch whose
        # completion lands on a queue we do not control, and the main loop is
        # already the thread pumping a run loop. Doing it on the reader
        # thread would mean two threads racing to drive Cocoa.
        self._want_exclude = None
        self.display = None

        self.stream = None
        self.writer = None
        self.input = None
        self.adaptor = None
        # The SCStreamOutput delegate, stored so a window-native rebuild can
        # re-add it to the fresh stream (set in main() after it is created).
        self._output = None

        # Window-native minimize/restore recovery state. `_cfg_w/h` pin the
        # rebuilt stream to the ORIGINAL buffer dims (the writer's frame size is
        # fixed at startWriting; a restored window at a new size is fitted by
        # scalesToFit -- the Option-A contract). `_win_onscreen` tracks the
        # target's on-screen state (True at start); the rest bound the retry.
        self._cfg_w = None
        self._cfg_h = None
        self._win_onscreen = True
        self._last_onscreen_check = 0.0
        self._minimized_since = None
        self._rebuild_fails = 0
        self.rebuilds = 0

        self.pts0 = None
        self.last_slot = None
        self._held = None          # the one retained pixel buffer
        self.appended = 0
        self.dup = 0
        self.dropped = 0
        self.notready = 0
        self.clamped = 0
        self.idle_filled = 0       # slots written by the wall-clock tick
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._failed = None

    # -- setup ---------------------------------------------------------
    def _pixel_size(self, display):
        """The display's real backing-store size, in pixels.

        SCStreamConfiguration defaults to the display's POINT size, which on
        a Retina panel silently halves the resolution of every recording.
        """
        import Quartz
        try:
            mode = Quartz.CGDisplayCopyDisplayMode(int(display.displayID()))
            w = int(Quartz.CGDisplayModeGetPixelWidth(mode))
            h = int(Quartz.CGDisplayModeGetPixelHeight(mode))
            if w > 0 and h > 0:
                return w, h
        except Exception as exc:
            note("pixel size lookup failed ({!r}); using point size".format(exc))
        return int(display.width()), int(display.height())

    def _window_pixel_size(self, filt, target, display):
        """Pixel size to pin the config to for a window-native capture.

        Prefer the filter's own `contentRect` x `pointPixelScale` (macOS 14+),
        which is what SCK will actually deliver (verified 2026-08-23: a 720x875
        pt window reported 720x875 @ 2.0 -> 1440x1750). Fall back to the
        window's point frame x the display backing scale. Dimensions are forced
        EVEN -- H264 4:2:0 needs them and the display path is always even too.
        """
        def even(n):
            n = int(round(n))
            return n - (n % 2) if n >= 2 else 2

        try:
            r = filt.contentRect()
            pps = float(filt.pointPixelScale())
            w = even(float(r.size.width) * pps)
            h = even(float(r.size.height) * pps)
            if w >= 2 and h >= 2:
                return w, h
        except Exception as exc:
            note("window contentRect size failed ({!r}); using frame x scale"
                 .format(exc))

        scale = 2.0
        try:
            import Quartz
            mode = Quartz.CGDisplayCopyDisplayMode(int(display.displayID()))
            px_w = float(Quartz.CGDisplayModeGetPixelWidth(mode))
            if display.width():
                scale = px_w / float(display.width())
        except Exception:
            pass
        f = target.frame()
        return (even(float(f.size.width) * scale),
                even(float(f.size.height) * scale))

    def _shareable(self, quiet=False):
        """SCShareableContent, or None. Synchronous by way of the run loop.

        `quiet=True` suppresses the `ERR content` emit (used by the rebuild
        path, where a failed re-fetch is recoverable and MUST NOT surface as a
        take-failing ERR to the parent -- the build path leaves quiet=False so
        an un-listable display/window at start still fails loudly).
        """
        import AppKit
        import ScreenCaptureKit as SCK
        box, done = {}, []

        def handler(content, error):
            try:
                box["content"] = content
                box["error"] = error
            finally:
                done.append(True)
            return None

        SCK.SCShareableContent.getShareableContentWithCompletionHandler_(handler)
        deadline = time.time() + 10.0
        while not done and time.time() < deadline:
            AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.02))
        if box.get("content") is None:
            if not quiet:
                emit("ERR content {}".format(box.get("error") or "timed out"))
            return None
        return box["content"]

    def build(self):
        import AVFoundation as AVF
        import CoreMedia
        import Foundation
        import Quartz
        import ScreenCaptureKit as SCK

        content = self._shareable()
        if content is None:
            return False

        displays = list(content.displays())
        if not displays:
            emit("ERR content no displays")
            return False
        want = int(self.display_id or Quartz.CGMainDisplayID())
        display = next((d for d in displays if int(d.displayID()) == want), None)
        if display is None:
            emit("ERR content display {} not shareable".format(want))
            return False

        self.display = display
        if self.capture_window_id is not None:
            # WINDOW-NATIVE: capture the target window's OWN backing buffer.
            # Occluders, the desktop, and our chrome are physically not in it,
            # so no exclude set is needed or meaningful here.
            target = next((w for w in content.windows()
                           if int(w.windowID()) == self.capture_window_id), None)
            if target is None:
                # Loud, not silent: a minimized / gone / non-shareable window
                # must fail the take, not record an empty buffer. Mirrors the
                # display-not-shareable check above.
                emit("ERR content window {} not shareable"
                     .format(self.capture_window_id))
                return False
            filt = (SCK.SCContentFilter.alloc()
                    .initWithDesktopIndependentWindow_(target))
            w, h = self._window_pixel_size(filt, target, display)
        else:
            wanted = set(self.exclude)
            excl = [w for w in content.windows() if int(w.windowID()) in wanted]
            missing = len(wanted) - len(excl)
            emit("FILTER {} {}".format(len(excl), missing))
            if missing:
                # Not fatal: a window that closed between the parent reading
                # ids and us building the filter is normal. It IS worth saying,
                # since the alternative reading -- our chrome silently in the
                # take -- is the bug this backend exists to fix.
                emit("WARN filter {} of {} ids not found"
                     .format(missing, len(wanted)))
            filt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, excl)
            w, h = self._pixel_size(display)

        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(w)
        cfg.setHeight_(h)
        cfg.setShowsCursor_(self.show_cursor)
        cfg.setQueueDepth_(QUEUE_DEPTH)
        cfg.setPixelFormat_(0x42475241)          # 'BGRA'
        # A CAP, not a pin: SCK still only delivers on change. The constant
        # rate comes from plan_slots, never from this.
        cfg.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self.fps))
        if self.capture_window_id is not None:
            # Window-native: if the window is resized mid-take, SCK fits it into
            # this FIXED buffer aspect-preserved and top-left anchored (measured
            # 2026-08-23). render's window-native mapper assumes exactly that,
            # so pin the policy rather than depend on the SCK default.
            try:
                cfg.setScalesToFit_(True)
            except Exception as exc:
                note("scalesToFit unavailable ({!r})".format(exc))
        if self.mic_uid:
            # macOS 15+. `capturesAudio` (SYSTEM audio) is deliberately left
            # off: measured, turning both on collapses them into a single
            # track of unclear provenance and flips the stream order. That
            # is a separate feature with its own measurement debt.
            try:
                cfg.setCaptureMicrophone_(True)
                cfg.setMicrophoneCaptureDeviceID_(self.mic_uid)
            except Exception as exc:
                emit("WARN mic config failed {!r}".format(exc))
                self.mic_uid = None
        emit("SIZE {} {} {}".format(w, h, self.fps))
        # Pin the rebuild dims to what the writer is about to be fixed at, so a
        # window-native restore rebuilds into a byte-compatible buffer.
        self._cfg_w, self._cfg_h = w, h

        url = Foundation.NSURL.fileURLWithPath_(self.path)
        try:
            os.unlink(self.path)
        except OSError:
            pass
        writer, err = AVF.AVAssetWriter.alloc().initWithURL_fileType_error_(
            url, AVF.AVFileTypeQuickTimeMovie, None)
        if writer is None:
            emit("ERR writer {}".format(err))
            return False
        # Compression is SPECIFIED, not defaulted. Left to AVFoundation's
        # defaults this wrote ~3x the bytes of the ffmpeg path (1.22 vs
        # ~0.40 MB/s), and the reason was measurable rather than mysterious:
        # 15 I-frames in 443 (one every ~0.5s) against ffmpeg's 14 in 3573
        # (one every ~4.2s). At 2880x1800 an I-frame is expensive, and a
        # screen recording is mostly still — 23% of frames in the soak were
        # literal duplicates, which cost almost nothing as P-frames and a
        # full frame as I-frames.
        #
        # KEYFRAME_INTERVAL matches x264's default keyint (250), which is
        # what the ffmpeg path has been using all along, so the two backends
        # are finally being asked for the same thing.
        #
        # raw.mov is a MEZZANINE — render.py decodes and re-encodes it, so
        # capture quality only has to survive one generation (the same
        # reasoning behind crf 20 rather than 16 on the ffmpeg side).
        # AVFoundation's H264 has no CRF equivalent, so quality is expressed
        # as an average bitrate scaled to the frame area: 0.10 bits per pixel
        # per second, i.e. ~31 Mbit/s at 2880x1800x60. That is deliberately
        # generous for a mezzanine, and still far below what the unbounded
        # default was producing on I-frames alone.
        bitrate = int(self.bpp * w * h * self.fps)
        settings = {
            AVF.AVVideoCodecKey: AVF.AVVideoCodecTypeH264,
            AVF.AVVideoWidthKey: w,
            AVF.AVVideoHeightKey: h,
        }
        if not self.raw_defaults:
            settings[AVF.AVVideoCompressionPropertiesKey] = {
                AVF.AVVideoAverageBitRateKey: bitrate,
                AVF.AVVideoMaxKeyFrameIntervalKey: self.keyint,
                AVF.AVVideoExpectedSourceFrameRateKey: self.fps,
            }
        vin = AVF.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
            AVF.AVMediaTypeVideo, settings)
        vin.setExpectsMediaDataInRealTime_(True)
        if not writer.canAddInput_(vin):
            emit("ERR writer cannot add video input")
            return False
        writer.addInput_(vin)
        if self.mic_uid:
            # Settings match the ffmpeg path's `-c:a aac -b:a 160k -ar 48000`
            # so a take sounds the same whichever backend wrote it. Inputs
            # MUST all be added before startWriting, so this cannot be
            # created lazily on the first audio buffer -- hence fixed
            # settings rather than ones read from the incoming format.
            asettings = {
                AVF.AVFormatIDKey: 1633772320,        # kAudioFormatMPEG4AAC
                AVF.AVSampleRateKey: 48000.0,
                AVF.AVNumberOfChannelsKey: 2,
                AVF.AVEncoderBitRateKey: 160000,
            }
            ain = AVF.AVAssetWriterInput.assetWriterInputWithMediaType_outputSettings_(
                AVF.AVMediaTypeAudio, asettings)
            ain.setExpectsMediaDataInRealTime_(True)
            if writer.canAddInput_(ain):
                writer.addInput_(ain)
                self.audio_in = ain
            else:
                emit("WARN mic writer rejected the audio input")
                self.mic_uid = None
        adaptor = (AVF.AVAssetWriterInputPixelBufferAdaptor
                   .assetWriterInputPixelBufferAdaptorWithAssetWriterInput_sourcePixelBufferAttributes_(
                       vin, None))
        # Movie fragments: MUST be set before startWriting (per Apple's docs;
        # the property is read once at that call and honoured for the file's
        # lifetime). With this on, AVAssetWriter writes `moov` at the top of
        # the file and appends periodic `moof`+`mdat` fragment pairs, each
        # self-indexing its own samples via `trun` -- so a kill-9 leaves the
        # file playable up to the last complete fragment (see the
        # MOVIE_FRAGMENT_INTERVAL_SEC comment for the whole story).
        #
        # Set to 0 (or negative) via `AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC=0`
        # to disable and reproduce the pre-C monolithic-file layout exactly:
        # the setter is skipped, `movieFragmentInterval` stays at
        # `kCMTimeInvalid`, and the writer defaults to a single trailing
        # `moov`. That is the byte-exact off switch this feature promises.
        #
        # Timescale 600 is the standard MP4/QuickTime container tick (evenly
        # divisible by every common frame rate we ship: 24/25/30/60/120).
        # CMTimeMakeWithSeconds picks a suitable timescale on its own, but
        # we pass one explicitly so the emitted `mvhd` timescale is stable
        # across future changes to CoreMedia's inference.
        if self.movie_fragment_interval > 0:
            interval = CoreMedia.CMTimeMakeWithSeconds(
                float(self.movie_fragment_interval), 600)
            writer.setMovieFragmentInterval_(interval)
        if not writer.startWriting():
            emit("ERR writer startWriting failed: {}".format(writer.error()))
            return False
        writer.startSessionAtSourceTime_(CoreMedia.kCMTimeZero)

        self.writer, self.input, self.adaptor = writer, vin, adaptor
        self.stream = SCK.SCStream.alloc().initWithFilter_configuration_delegate_(
            filt, cfg, None)
        return True

    def request_exclusions(self, ids):
        """Record a new exclusion set. Called from the stdin reader thread."""
        with self._lock:
            self._want_exclude = [int(i) for i in ids]

    def apply_pending_exclusions(self):
        """Swap in a new content filter if one was requested. Main loop only.

        The initial id set is resolved once, before `startCapture` — but
        windows are created DURING a take: the facecam bubble pops out, the
        picker opens. Those windows did not exist when the filter was built,
        so without this they would be recorded. `updateContentFilter` is the
        supported way to change a live stream (measured elsewhere at ~102 ms
        mid-stream), and no frames are dropped across the swap.

        Failure is non-fatal and reported: a filter we could not update is a
        take that still records, just with our chrome possibly in it, and
        stopping the recording over that would be a much worse trade.
        """
        # Inert in window-native mode: the buffer is the window's own, so there
        # is nothing to exclude, and rebuilding a display filter here would
        # REPLACE the desktop-independent one and break the capture. The parent
        # never sends EXCLUDE on this path, but guard anyway -- and BEFORE the
        # SCK import, so it stays callable (and testable) without ObjC.
        if self.capture_window_id is not None:
            return
        import ScreenCaptureKit as SCK
        with self._lock:
            want = self._want_exclude
            self._want_exclude = None
        if want is None or self.stream is None or self.display is None:
            return
        if want == self.exclude:
            return
        content = self._shareable()
        if content is None:
            emit("WARN exclusion update could not list windows")
            return
        wanted = set(want)
        excl = [w for w in content.windows() if int(w.windowID()) in wanted]
        try:
            filt = SCK.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                self.display, excl)

            def done(error):
                return None

            self.stream.updateContentFilter_completionHandler_(filt, done)
        except Exception as exc:
            emit("WARN exclusion update failed {!r}".format(exc))
            return
        self.exclude = list(want)
        emit("FILTER {} {}".format(len(excl), len(wanted) - len(excl)))

    # -- window-native minimize/restore recovery -----------------------
    def _target_onscreen(self):
        """Is the window-native target currently on-screen? -> True/False/None.

        Minimized / hidden / on another Space -> False (a minimized window
        drops out of the CGWindowList entirely -- measured). Occluded stays
        True (it is on screen, just covered), so covering never triggers a
        rebuild. None on a read hiccup, so a transient failure changes nothing.
        Cheap: one CGWindowList lookup for a single id. Never raises.
        """
        try:
            import Quartz
            wid = int(self.capture_window_id)
            infos = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionIncludingWindow, wid) or []
            for info in infos:
                if int(info.get("kCGWindowNumber", -1)) == wid:
                    return bool(info.get("kCGWindowIsOnscreen", False))
            return False                      # absent -> minimized/hidden
        except Exception as exc:
            note("onscreen check failed ({!r})".format(exc))
            return None

    def _rebuild_config(self):
        """Config for a rebuilt window-native stream, or None.

        Pinned to the ORIGINAL buffer dims (`_cfg_w/h`) so the writer's fixed
        AVVideoWidth/Height still match; a restored window at a new size is
        fitted by scalesToFit, exactly as during the take. Mirrors build()'s
        window-native config -- build() stays the source of truth; keep them in
        step.
        """
        import CoreMedia
        import ScreenCaptureKit as SCK
        if self._cfg_w is None or self._cfg_h is None:
            return None
        cfg = SCK.SCStreamConfiguration.alloc().init()
        cfg.setWidth_(self._cfg_w)
        cfg.setHeight_(self._cfg_h)
        cfg.setShowsCursor_(self.show_cursor)
        cfg.setQueueDepth_(QUEUE_DEPTH)
        cfg.setPixelFormat_(0x42475241)          # 'BGRA'
        cfg.setMinimumFrameInterval_(CoreMedia.CMTimeMake(1, self.fps))
        try:
            cfg.setScalesToFit_(True)
        except Exception as exc:
            note("rebuild scalesToFit unavailable ({!r})".format(exc))
        if self.mic_uid:
            try:
                cfg.setCaptureMicrophone_(True)
                cfg.setMicrophoneCaptureDeviceID_(self.mic_uid)
            except Exception as exc:
                note("rebuild mic config failed ({!r})".format(exc))
        return cfg

    def _rebuild_stream(self):
        """Tear down the dead stream and build a fresh one on the restored
        window's NEW surface. -> True on success. Never raises.

        The WRITER is untouched -- only the stream is replaced -- so `pts0`,
        `last_slot`, and `_held` carry over and `plan_slots` dup-fills the
        minimized gap seamlessly. The in-place `updateContentFilter_` swap does
        NOT re-bind a restored surface (measured), which is why this is a full
        stop + fresh SCStream rather than the exclusion path's filter swap.
        """
        if self.capture_window_id is None or self._stop.is_set():
            return False
        import AppKit
        import ScreenCaptureKit as SCK
        content = self._shareable(quiet=True)
        if content is None:
            return False
        target = next((w for w in content.windows()
                       if int(w.windowID()) == self.capture_window_id), None)
        if target is None:
            return False              # not yet re-listable; caller will retry
        try:
            filt = (SCK.SCContentFilter.alloc()
                    .initWithDesktopIndependentWindow_(target))
        except Exception as exc:
            note("rebuild filter failed ({!r})".format(exc))
            return False
        cfg = self._rebuild_config()
        if cfg is None:
            return False

        # Stop the dead stream first: no on_sample fires once stop completes,
        # so the writer swap below races nothing on SCK's queue.
        old = self.stream
        if old is not None:
            gone = []
            try:
                old.stopCaptureWithCompletionHandler_(
                    lambda e: gone.append(e) or None)
                deadline = time.time() + 5.0
                while (not gone and time.time() < deadline
                       and not self._stop.is_set()):
                    AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                        AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.02))
            except Exception as exc:
                note("rebuild stopCapture failed ({!r})".format(exc))

        try:
            stream = SCK.SCStream.alloc()\
                .initWithFilter_configuration_delegate_(filt, cfg, None)
            ok, err = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output, 0, None, None)
            if not ok:
                note("rebuild addStreamOutput failed ({!r})".format(err))
                return False
            if self.mic_uid and self.audio_in is not None:
                mok, merr = stream.addStreamOutput_type_sampleHandlerQueue_error_(
                    self._output, 2, None, None)
                if not mok:
                    note("rebuild mic output rejected ({!r})".format(merr))
            started = []
            stream.startCaptureWithCompletionHandler_(
                lambda e: started.append(e) or None)
            deadline = time.time() + 10.0
            while (not started and time.time() < deadline
                   and not self._stop.is_set()):
                AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.02))
            if not started or started[0] is not None:
                note("rebuild startCapture failed ({!r})"
                     .format(started[0] if started else "timeout"))
                return False
            self.stream = stream
            self.rebuilds += 1
            return True
        except Exception as exc:
            note("rebuild stream failed ({!r})".format(exc))
            return False

    def check_window_restore(self):
        """Rebuild the stream when a minimized window-native target returns.

        Runs on the main loop next to `apply_pending_exclusions`, throttled to
        `ONSCREEN_POLL_SEC`. Inert for display capture and for a window that
        never minimizes. Never raises. The frozen span while the window is
        hidden is held by `plan_slots` dup-fill exactly as it is today; this
        only ends that freeze on restore instead of letting it run forever.
        """
        if self.capture_window_id is None or self._stop.is_set():
            return
        now = time.time()
        if now - self._last_onscreen_check < ONSCREEN_POLL_SEC:
            return
        self._last_onscreen_check = now
        on = self._target_onscreen()
        action = sck.plan_restore_action(
            self._win_onscreen, on, self._rebuild_fails, MAX_REBUILD_TRIES)
        if action == "none":
            return
        wid = self.capture_window_id
        if action == "hidden":
            self._win_onscreen = False
            self._minimized_since = now
            emit("WARN window_hidden {}".format(wid))
            return
        if action == "giveup":
            self._win_onscreen = True         # stop hammering
            emit("WARN window_rebuild_gaveup {} after {} tries"
                 .format(wid, self._rebuild_fails))
            return
        # action == "rebuild": the window is back.
        hidden = (now - self._minimized_since) if self._minimized_since else 0.0
        if self._rebuild_stream():
            self._win_onscreen = True
            self._rebuild_fails = 0
            emit("WARN window_rebuild_ok {} hidden={:.1f}s".format(wid, hidden))
        else:
            self._rebuild_fails += 1
            emit("WARN window_rebuild_retry {} attempt={}"
                 .format(wid, self._rebuild_fails))

    # -- the hot path --------------------------------------------------
    def on_sample(self, sbuf):
        """One arriving frame -> zero or more appended slots.

        Runs on SCK's serial queue. Never raises: see the module docstring.
        """
        import CoreMedia
        if self._stop.is_set():
            return
        if not CoreMedia.CMSampleBufferIsValid(sbuf):
            return
        pb = CoreMedia.CMSampleBufferGetImageBuffer(sbuf)
        if pb is None:
            return
        t = CoreMedia.CMSampleBufferGetPresentationTimeStamp(sbuf)
        pts = float(t.value) / float(t.timescale or 1)

        with self._lock:
            if self.pts0 is None:
                self.pts0 = pts
                emit("T0 {!r}".format(pts))
            repeats, slot, dropped = sck.plan_slots(
                pts, self.pts0, self.fps, self.last_slot, max_fill=MAX_FILL)
            if dropped:
                self.dropped += 1
                return
            held = self._held
            # Fill the idle gap with what was actually on screen during it.
            if repeats and held is not None:
                if len(repeats) >= MAX_FILL:
                    self.clamped += 1
                    emit("WARN fill clamped at {} frames".format(MAX_FILL))
                for s in repeats:
                    if self._append(held, s):
                        self.dup += 1
            if self._append(pb, slot):
                self.last_slot = slot
                self._held = pb          # retain exactly one, released below
            else:
                self.last_slot = slot

    def fill_idle(self, lag_sec=None, now_pts=None):
        """Advance the CFR timeline to the wall clock while SCK is quiet.

        The other half of `on_sample`. SCK delivers nothing while the screen
        is still, so without this the timeline stops advancing and the take's
        idle spans -- and everything after its LAST change -- never reach the
        file (see `sck.plan_idle_fill` for the measurements). Ticked from the
        run loop during the take, and once more from `finish` with no lag to
        close the tail exactly.

        Cheap by construction: at a 20Hz tick only ~3 slots come due each
        time, so the lock is never held long enough to stall `on_sample` on
        SCK's queue. Returns the number of slots written.
        """
        if lag_sec is None:
            lag_sec = sck.IDLE_FILL_LAG_SEC
        if now_pts is None:
            now_pts = sck.now_host_pts()
        with self._lock:
            held = self._held
            if held is None:
                return 0        # nothing has arrived yet: nothing to hold
            slots = sck.plan_idle_fill(
                now_pts, self.pts0, self.fps, self.last_slot,
                lag_sec=lag_sec, max_fill=MAX_FILL)
            n = 0
            for s in slots:
                if self._stop.is_set() and lag_sec != 0.0:
                    break       # stopping: the tail pass will finish the job
                if not self._append(held, s):
                    break
                self.last_slot = s
                self.dup += 1
                self.idle_filled += 1
                n += 1
        return n

    def on_audio(self, sbuf):
        """One microphone sample buffer -> the audio track, rebased.

        Audio arrives on the same host clock as video, but the video track
        is written on a slot grid starting at zero, so a raw append would
        put the audio one machine-uptime into the future. Each buffer is
        therefore copied with its timing shifted by `pts0` — the presentation
        time of video frame 0.

        Buffers arriving BEFORE the first video frame are dropped, and that
        is deliberate. They would land at a negative time, and the
        alternative — letting the mic's genuine lead-in set a non-zero audio
        `start_time` — introduces a container property nothing downstream has
        ever seen (today's raw.mov has audio starting at exactly 0.000000,
        and `render._probe_has_audio`, retime's silencedetect and the
        waveform all predate the idea that it might not). Losing a few
        milliseconds of pre-roll silence is the cheaper side of that trade.

        KNOWN DIFFERENCE, measured: the audio track still starts LATE, not at
        zero — 0.262 s on a real take, because the microphone takes that long
        to spin up after the stream begins. This is not a desync (the audio
        really does begin there, and ffmpeg honors `start_time`), but the
        ffmpeg backend produces exactly 0.000000 because
        `aresample=async=1:first_pts=0` pads the head with silence, and we do
        not. Anything downstream that assumes audio and video share an origin
        will be off by that much. Closing the gap means synthesizing leading
        silence or a remux pass; neither is done yet.
        """
        import CoreMedia
        if self._stop.is_set() or self.audio_in is None:
            return
        with self._lock:
            pts0 = self.pts0
        if pts0 is None:
            self.audio_dropped += 1
            return
        try:
            t = CoreMedia.CMSampleBufferGetPresentationTimeStamp(sbuf)
            shifted = CoreMedia.CMTimeSubtract(
                t, CoreMedia.CMTimeMakeWithSeconds(pts0, t.timescale or 1000))
            if CoreMedia.CMTimeGetSeconds(shifted) < 0:
                self.audio_dropped += 1
                return
            timing = CoreMedia.CMSampleTimingInfo(
                CoreMedia.CMSampleBufferGetDuration(sbuf), shifted,
                CoreMedia.kCMTimeInvalid)
            ok, out = CoreMedia.CMSampleBufferCreateCopyWithNewTiming(
                None, sbuf, 1, [timing], None)
            if ok != 0 or out is None:
                self.audio_dropped += 1
                return
        except Exception as exc:
            self.audio_dropped += 1
            note("audio retiming: {!r}".format(exc))
            return
        inp = self.audio_in
        deadline = time.time() + 0.5
        while not inp.isReadyForMoreMediaData():
            if time.time() > deadline or self._stop.is_set():
                self.audio_dropped += 1
                return
            time.sleep(0.001)
        if inp.appendSampleBuffer_(out):
            self.audio_appended += 1
        else:
            self.audio_dropped += 1

    def _append(self, pb, slot):
        import CoreMedia
        inp, adaptor = self.input, self.adaptor
        if inp is None or adaptor is None:
            return False
        # The encoder pushes back when it falls behind. Spin briefly rather
        # than dropping: a dropped slot is a hole in a timeline everything
        # downstream assumes is gapless.
        deadline = time.time() + 1.0
        while not inp.isReadyForMoreMediaData():
            self.notready += 1
            if time.time() > deadline or self._stop.is_set():
                return False
            time.sleep(0.001)
        ok = adaptor.appendPixelBuffer_withPresentationTime_(
            pb, CoreMedia.CMTimeMake(int(slot), self.fps))
        if ok:
            self.appended += 1
        else:
            self._failed = str(self.writer.error())
            emit("ERR append {}".format(self._failed))
            self._stop.set()
        return bool(ok)

    # -- teardown ------------------------------------------------------
    def finish(self):
        import AppKit
        self._stop.set()
        if self.stream is not None:
            done = []

            def stopped(error):
                done.append(error)
                return None

            try:
                self.stream.stopCaptureWithCompletionHandler_(stopped)
                deadline = time.time() + 5.0
                while not done and time.time() < deadline:
                    AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                        AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.02))
            except Exception as exc:
                note("stopCapture failed: {!r}".format(exc))

        # Close the tail EXACTLY (no lag): the stream is stopped, so nothing
        # is still in flight to be displaced. Without this the movie ends at
        # the last on-screen CHANGE rather than at the stop -- a take whose
        # final seconds were still simply lost them.
        try:
            self.fill_idle(lag_sec=0.0)
        except Exception as exc:
            note("tail fill failed: {!r}".format(exc))

        with self._lock:
            self._held = None
        for inp in (self.input, self.audio_in):
            if inp is None:
                continue
            try:
                inp.markAsFinished()
            except Exception:
                pass
        if self.writer is not None:
            done = []

            def finished():
                done.append(True)
                return None

            self.writer.finishWritingWithCompletionHandler_(finished)
            deadline = time.time() + 30.0
            while not done and time.time() < deadline:
                AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
                    AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.02))
        frames = self.appended
        if self.audio_in is not None:
            emit("AUDIO {} {}".format(self.audio_appended, self.audio_dropped))
        emit("DONE {} {!r}".format(frames, frames / float(self.fps)))


def _make_output(capture):
    """The SCStreamOutput delegate, built here so ObjC types stay local."""
    import objc
    import ScreenCaptureKit  # noqa: F401  -- registers the protocol
    from Foundation import NSObject

    # objc.protocolNamed, not an attribute on the framework module: PyObjC
    # exposes formal protocols through the runtime, and SCK's module has no
    # SCStreamOutput attribute. Conforming matters -- it is what gives the
    # selector its real ObjC signature, and a mistyped one here is how a
    # callback ends up aborting the process from inside replayd's XPC reply.
    class _Output(NSObject, protocols=[objc.protocolNamed("SCStreamOutput")]):
        def stream_didOutputSampleBuffer_ofType_(self, stream, sbuf, stype):
            # EVERY line of this is inside try/except and the method returns
            # None explicitly -- an exception escaping here, or a stray
            # return value, aborts the process from inside replayd's XPC
            # reply block. See the module docstring.
            try:
                kind = int(stype)
                if kind == 0:                  # SCStreamOutputTypeScreen
                    capture.on_sample(sbuf)
                elif kind == 2:                # SCStreamOutputTypeMicrophone
                    capture.on_audio(sbuf)
            except Exception as exc:
                note("sample handler: {!r}".format(exc))
            return None

    return _Output.alloc().init()


def main():
    try:
        line = sys.stdin.readline()
        cfg = json.loads(line)
    except Exception as exc:
        emit("ERR config {!r}".format(exc))
        return 2

    cap = Capture(cfg)
    try:
        if not cap.build():
            return 2
    except Exception as exc:
        emit("ERR setup {!r}".format(exc))
        return 2

    import AppKit

    out = _make_output(cap)
    # Held so a window-native rebuild can re-add this same delegate to the
    # fresh stream (see check_window_restore).
    cap._output = out
    # nil queue: SCK provides its own serial queue for the handler. Avoids a
    # dependency on libdispatch bindings for no behavioral difference --
    # on_sample takes a lock either way, since we do not get to choose which
    # thread SCK calls us on.
    ok, err = cap.stream.addStreamOutput_type_sampleHandlerQueue_error_(
        out, 0, None, None)
    if not ok:
        emit("ERR output {}".format(err))
        return 2
    if cap.mic_uid:
        # Best-effort, same posture as the facecam: losing the mic must
        # never cost the screen recording. MIC failed/ok is reported so the
        # parent can put it in meta rather than leaving a silent take
        # looking like one where nobody spoke.
        mok, merr = cap.stream.addStreamOutput_type_sampleHandlerQueue_error_(
            out, 2, None, None)
        if not mok:
            emit("WARN mic output rejected {}".format(merr))
            cap.mic_uid = None
            cap.audio_in = None
    emit("MIC {}".format("ok" if cap.mic_uid else "failed"))

    started = []

    def on_start(error):
        started.append(error)
        return None

    cap.stream.startCaptureWithCompletionHandler_(on_start)
    deadline = time.time() + 10.0
    while not started and time.time() < deadline:
        AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.02))
    if not started:
        emit("ERR start timed out")
        return 2
    if started[0] is not None:
        emit("ERR start {}".format(started[0]))
        return 2

    emit("READY")

    def watch_stdin():
        try:
            for raw in iter(sys.stdin.readline, ""):
                line = raw.strip()
                if line == "STOP":
                    break
                if line.startswith("EXCLUDE"):
                    parts = line.split()[1:]
                    try:
                        cap.request_exclusions([int(p) for p in parts])
                    except ValueError:
                        # A malformed control line must never end a take.
                        note("bad EXCLUDE line: {!r}".format(line))
        except Exception:
            pass
        cap._stop.set()

    threading.Thread(target=watch_stdin, daemon=True).start()

    import signal
    signal.signal(signal.SIGINT, lambda *a: cap._stop.set())
    signal.signal(signal.SIGTERM, lambda *a: cap._stop.set())
    # Undo any inherited SIG_BLOCK on our stop signals. `signal.signal` sets the
    # DISPOSITION but never touches the mask, and a blocked mask is inherited
    # across fork+exec -- so a parent that blocks SIGINT/SIGTERM process-wide
    # (cli._install_signal_quit does exactly this, so the bar's sigwait thread
    # can quit the Cocoa-owned main thread) leaves those signals blocked HERE.
    # The handlers above would then never fire: the parent's stop-SIGINT stays
    # pending, this loop runs forever, and `_harvest_fleet` has to SIGKILL after
    # its 40s deadline -- losing the whole take (the moov survives via movie
    # fragments, but the recorder gated it as a failure). Every worker must own
    # its own mask; done here rather than a preexec_fn because a real exec makes
    # it unconditionally safe. See record.Recorder._harvest_fleet / _gate_fleet.
    if hasattr(signal, "pthread_sigmask"):
        try:
            signal.pthread_sigmask(
                signal.SIG_UNBLOCK, {signal.SIGINT, signal.SIGTERM})
        except (ValueError, OSError):
            pass

    last_stat = 0.0
    while not cap._stop.is_set():
        AppKit.NSRunLoop.currentRunLoop().runUntilDate_(
            AppKit.NSDate.dateWithTimeIntervalSinceNow_(0.05))
        cap.apply_pending_exclusions()
        cap.check_window_restore()
        # Keep the timeline on the wall clock even while SCK is silent. This
        # loop is the only thing still running when the screen is still.
        cap.fill_idle()
        now = time.time()
        if now - last_stat >= 1.0:
            last_stat = now
            emit("STAT {} {} {} {} {} {}".format(
                -1 if cap.last_slot is None else cap.last_slot,
                cap.appended, cap.dup, cap.dropped, cap.notready,
                cap.idle_filled))

    cap.finish()
    return 0 if cap._failed is None else 1


if __name__ == "__main__":
    sys.exit(main())
