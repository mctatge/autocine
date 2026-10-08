"""Screen recorder: ffmpeg captures the display while pynput logs the cursor.

Sync model
----------
We need pynput event timestamps (time.monotonic) aligned to ffmpeg's media
clock. ffmpeg is started with `-progress pipe:1` and `-tune zerolatency` (so the
encoder doesn't buffer frames and its reported media time tracks real time);
the first `out_time_us` it emits gives the media time of already-encoded output,
so we anchor:

    t0_monotonic = time.monotonic() - out_time_us/1e6

Events are stored raw; the renderer subtracts t0 to get media-relative time
(with an optional manual --offset nudge for any residual latency).
"""

import collections
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from typing import Optional

from . import devices as dev
from . import sck


class RecordError(RuntimeError):
    """Raised when ffmpeg fails to produce a usable recording."""


def _fmt_input(video_idx: int, mic_idx: Optional[int]) -> str:
    return "{}:{}".format(video_idx, mic_idx if mic_idx is not None else "none")


CURSOR_MODES = ("system", "synthetic")

# Key-activity coalescing bucket, in seconds. Key events are logged as bare
# activity ticks quantized to this grid -- at most one line per bucket, and
# NEVER any key identity (no keycode, character, or modifier bit): the
# 100 ms floor + coalescing also destroys the inter-key timing that
# keystroke-cadence inference attacks need. camera.DEFAULTS["typing_bucket"]
# must match (pinned by test_record.test_key_bucket_matches_camera_default).
KEY_BUCKET_SEC = 0.1

# The key-activity decoder's own process -- see _key_worker.py's docstring
# for why it isn't in-process any more. Computed relative to this file, not
# the caller's cwd, so `studio.py bar` works from any launch directory.
_KEY_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_key_worker.py")

# The ScreenCaptureKit capture child. A separate process for the same reason
# the key worker is one: SCK can abort its host from inside replayd's XPC
# reply block, and this module is imported by the long-lived server.
_SCK_WORKER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "_sck_worker.py")

# The process-death watchdog -- see _watchdog.py's docstring for why a
# crashed (not gracefully stopped) recording needs one at all.
_WATCHDOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "_watchdog.py")


def spawn_watchdog(child_pids, parent_pid=None):
    """Launch `_watchdog.py` to reap `child_pids` if this process dies
    without cleaning up after itself. Returns the Popen, or None.

    Module-level rather than a Recorder method because the SAME hazard shows
    up in two unrelated places: the recorder's ffmpeg/key-worker children,
    and the `studio app` server's detached native-bar child (which is
    spawned into its OWN session -- see `StudioState.launch_native_bar` --
    and therefore survives both a Ctrl+C to the server's process group AND
    the server dying outright, leaving a GUI window whose backend is gone).
    One implementation, one place to get the signal escalation right.

    Best-effort by contract: a watchdog that can't launch returns None and
    the caller carries on exactly as it did before watchdogs existed.
    """
    pids = [str(int(p)) for p in child_pids if p]
    if not pids:
        return None
    watched_parent = os.getpid() if parent_pid is None else int(parent_pid)
    try:
        return subprocess.Popen(
            [sys.executable, _WATCHDOG_PATH, str(watched_parent)] + pids,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        return None


def stop_watchdog(proc):
    """Tell a watchdog to stand down (normal shutdown). Safe on None/dead."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
    except Exception:
        pass

# How long to wait for the child's "READY" line before giving up on it.
# Generous: pynput's own tap creation (AXIsProcessTrusted + CGEventTapCreate)
# is normally fast, but a cold `python3` interpreter start (importing pynput,
# pyobjc's Quartz/HIServices bindings) can take real wall-clock time on a
# busy machine, and a slow-but-eventually-fine start must not be misread as
# "failed" -- that would silently and permanently disable typing-zoom for a
# take that would have worked fine.
KEY_WORKER_READY_TIMEOUT_SEC = 3.0


class _KeyWorker(object):
    """Owns the key-activity child process: launch, ready-check, read, stop.

    Deliberately dumb on this side too -- the parent's only real job is to
    turn "READY <float>\\n" / "<float>\\n" lines into calls to
    `on_tick(None, t=...)`, where `on_tick` is `Recorder._on_key` (kept as an
    injected callback so this class has no dependency on Recorder's
    internals, and so tests can swap in a plain list.append).

    The one piece of real work here is CLOCK PAIRING. `time.monotonic()`'s
    reference point is per-process on this stack -- measured 2026-07-30: a
    child spawned 0.5 s into a parent's life read 0.0058 against the parent's
    0.55 -- so the child's raw readings are NOT in our timeframe, and this
    class used to pass them straight through. Since the parent is usually the
    long-lived `studio.py bar`, that put every key tick a whole
    parent-process-age before the take started, where nothing downstream
    would ever find it. We now take our own reading of the moment READY
    arrives, subtract the child's reading of the moment it sent it, and add
    that offset to every later timestamp. Residual error is one pipe hop.

    A child that offers no clock (an older `_key_worker.py`, a test feeding a
    bare "READY") leaves the offset at 0.0 -- i.e. exactly the old
    pass-through behavior, so the unpaired path stays bit-for-bit what it was.
    """

    def __init__(self, proc, on_tick):
        self.proc = proc
        self._on_tick = on_tick
        # child clock -> our clock. Set once, on the handshake thread, before
        # READY is published to wait_ready(), so the read loop and any
        # wait_ready() caller both see the final value without locking.
        self._clock_offset = 0.0
        self._ready_q = queue.Queue(maxsize=1)
        self._handshake_thread = threading.Thread(
            target=self._read_handshake, daemon=True)
        self._reader_thread = None
        self._stop_reading = threading.Event()
        self._handshake_thread.start()

    def _read_handshake(self):
        """Block on the first line only, then hand off to _read_loop.

        Split from the main read loop so `wait_ready` can have a bounded
        timeout on just the handshake without needing to interrupt an
        in-progress readline() -- Python's blocking file reads aren't
        cancellable, so the THREAD outlives a timed-out wait_ready() call
        (it's a daemon thread either way; a worker we're about to .stop()
        makes it exit via EOF once the process is killed).
        """
        stream = self.proc.stdout
        if stream is None:
            self._ready_q.put(False)
            return
        first = stream.readline()
        parts = first.split()
        if not parts or parts[0] != "READY":
            self._ready_q.put(False)
            return
        # Pair the clocks (see the class docstring). Read ours FIRST, before
        # any other work on this thread, so the offset carries the pipe hop
        # and nothing else. A missing or malformed reading leaves the offset
        # at 0.0 rather than failing the handshake: a worker whose ticks are
        # mistimed is still worth far more than no key capture at all.
        ours = time.monotonic()
        if len(parts) > 1:
            try:
                self._clock_offset = ours - float(parts[1])
            except ValueError:
                pass
        self._ready_q.put(True)
        self._read_loop(stream)

    def _read_loop(self, stream):
        while not self._stop_reading.is_set():
            line = stream.readline()
            if not line:
                return   # child exited/pipe closed
            line = line.strip()
            if not line:
                continue
            try:
                t = float(line)
            except ValueError:
                continue   # never let a malformed line take the take down
            try:
                self._on_tick(None, t=t + self._clock_offset)
            except Exception:
                pass

    def wait_ready(self, timeout):
        try:
            return bool(self._ready_q.get(timeout=timeout))
        except queue.Empty:
            return False

    def start_reading(self):
        """No-op: the handshake thread already transitions into the read
        loop itself once READY lands. Kept as an explicit step so the call
        site in `_start_key_listener` reads as a clear two-phase sequence
        (confirm alive, THEN commit to it) even though phase two needs
        nothing further here."""

    def is_alive(self):
        return self.proc.poll() is None

    def stop(self):
        self._stop_reading.set()
        proc = self.proc
        if proc.poll() is None:
            try:
                proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._handshake_thread.join(timeout=1.0)

# Scroll rate-limit interval, in seconds: at most one scroll line per this
# span, move-style (real timestamps + the real cursor position -- scrolls
# have no keystroke-privacy concern, so no grid quantization). A trackpad
# emits scroll events at display rate; the camera only needs activity ticks.
SCROLL_MIN_INTERVAL_SEC = 0.1

# Window-geometry sampling interval, in seconds. avfoundation still captures
# the whole display, but logging window rects OVER TIME (rather than the
# single snapshot meta.json carries) is what lets render follow a window that
# is moved or resized mid-take -- both the `--capture-window` target and,
# through `edits.windows`, each card of the multi-window grid.
# 20 Hz: enumerating every on-screen window measures ~3.9 ms with the display
# list cached (see _poll_window_geometry), so this costs <8% of one core next
# to the encoder, and a hand-dragged window has no meaningful motion above
# ~2 Hz -- render smooths and interpolates between samples anyway.
# avfoundation's per-stream packet queue, in packets. ffmpeg's default is 8,
# which is far too shallow for a Retina display at 60fps: the video queue
# blocks while libx264 works, and a blocked queue DROPS AUDIO PACKETS. 1024 is
# ffmpeg's own suggested order of magnitude and costs a few MB of RAM.
THREAD_QUEUE_SIZE = 1024

WINDOW_POLL_SEC = 0.05

# Identical rects are coalesced: a window nobody touches writes one heartbeat
# line per this many seconds instead of 20/s, while interpolation still gets
# anchors at both ends of any still stretch.
WINDOW_HEARTBEAT_SEC = 1.0

# Per-edge slop (POINTS) before a new sample counts as movement worth
# logging. Quartz reports integral point rects, so this only suppresses
# genuine 1pt jitter, never real motion.
WINDOW_MOVE_EPS_PT = 1.0

# How long a window must HOLD the front before it counts as "brought up"
# (`raised_window_ids`, the seamless-join candidate rule). macOS raises a
# window for a moment on a cmd-tab sweep, a focus steal, or a sheet closing;
# without a dwell each of those flybys would burn one of the 4 fleet slots.
# Costs the user this long on top of the ~2s join, which is why it is short.
WINDOW_RAISE_DWELL_SEC = 0.75

# Head start (seconds) the screen ffmpeg gets before the webcam capture is
# launched. The two avfoundation AVCaptureSessions must not configure at the
# same instant (that makes the SCREEN input's config fail -- see
# _wait_screen_capturing / the facecam serialization invariant in
# docs/architecture.md); the screen is launched first, and this uncontested delay
# lets it grab its session before the camera touches avfoundation. Small so
# the webcam bubble fades in promptly; bump it if a --face recording ever
# loses the screen track again. The wait ends EARLY if the screen confirms a
# frame, dies, or a stop is requested -- so this is a ceiling, not a fixed sleep.
FACE_START_DELAY_SEC = 0.5


# One shared harvest deadline for stopping N native workers. The doc's
# ordering (docs/architecture.md, "Teardown ordering"):
# broadcast SIGINT/STOP to every child FIRST, then wait against ONE 40s
# budget for them all to finalize -- NOT 40s x N. A slow child cannot be
# allowed to widen every other worker's tail-loss window.
NATIVE_HARVEST_DEADLINE_SEC = 40.0


class _NativeWorker(object):
    """One SCK child in a multi-window native fleet.

    Owns the per-child state that used to be top-level Recorder attributes:
    proc + stdout/stderr threads, T0 pairing, SIZE / STAT / FILTER / DONE /
    ERR from the worker's control channel, and the raw_i.mov output path.
    Recorder holds a list of these when `_is_multi_window_native()`; the
    single-window path leaves the list empty and every state attribute on
    Recorder itself, unchanged.

    Kept intentionally passive -- no thread/subprocess ownership past
    reference-holding -- so `_start_multi_native` orchestrates the fleet
    with the same shape `start()` orchestrates one child. That is what
    lets `_read_sck_stdout` be a per-worker method rather than a global,
    without every callsite growing an `if multi:` branch.
    """

    def __init__(self, session_dir, index, window_entry):
        self.index = int(index)
        # window_entry mirrors self.capture_window's shape: a dict from
        # devices.window_rect_points, already validated by the caller.
        self.window = window_entry
        # The rect at the time this worker was created (pick time). The
        # multi-native path re-snapshots each window's rect after the
        # countdown, exactly as the single-window path does; the updated
        # rect lands in `self.entry` below (or None if it failed).
        self.entry = None
        self.resnapshot = False
        self.end_rect = None
        self.raw_path = os.path.join(session_dir,
                                     "raw_{:d}.mov".format(self.index))
        self.proc = None
        # Per-worker clock pairing. sck.host_pts_to_monotonic runs in the
        # parent from two adjacent local reads, so each worker's own
        # monotonic zero is independent -- calling it once per child T0
        # is the whole "N clock pairings" story from the doc.
        self.t0 = None
        # Latest STAT/FILTER/DONE/SIZE/ERR seen from this worker. Same
        # shape as the single-window versions on Recorder.
        self.stat = None
        self.filter = None
        self.done = None
        self.size = None
        self.error = None
        self.stderr_tail = collections.deque(maxlen=20)

    def is_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def returncode(self):
        return self.proc.returncode if self.proc is not None else None

    def has_moov(self):
        """True iff this worker's raw_i.mov landed with a moov -- the same
        finalized-take gate `start()` runs for the single-window path."""
        if not os.path.exists(self.raw_path):
            return False
        return sck.has_moov(self.raw_path)

    def read_stdout(self):
        """Per-worker analog of `Recorder._read_sck_stdout` for the fleet.

        Wire protocol is identical to the single-window case (`sck.parse_worker_line`),
        so `t0` uses the same parent-side pairing (`sck.host_pts_to_monotonic`)
        -- one call per child T0 -> N mutually consistent monotonic zeros in
        our own clock, the doc's "N clock pairings". Runs in its own thread;
        stops when the pipe closes at worker exit.
        """
        stream = self.proc.stdout
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            try:
                line = raw.decode("utf-8", "replace").strip()
            except Exception:
                continue
            kind, payload = sck.parse_worker_line(line)
            if kind == "t0":
                if self.t0 is None:
                    self.t0 = sck.host_pts_to_monotonic(payload["pts0"])
            elif kind == "size":
                self.size = payload
            elif kind == "stat":
                self.stat = payload
            elif kind == "done":
                self.done = payload
            elif kind == "filter":
                self.filter = payload
            elif kind == "err":
                self.error = "{} {}".format(payload.get("domain", ""),
                                            payload.get("detail", "")).strip()
                self.stderr_tail.append(self.error)
            elif kind == "warn":
                self.stderr_tail.append("warn: " + payload.get("detail", ""))

    def read_stderr(self):
        """Free-form stderr from the worker's Python-level warnings (its
        `note()`). Kept in a small ring so a failed take's error.log carries
        the tail, same shape as `Recorder._read_stderr`."""
        stream = self.proc.stderr
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            try:
                text = raw.decode("utf-8", "replace").rstrip()
            except Exception:
                continue
            if text:
                self.stderr_tail.append(text)


# How long resume() waits for the fresh segment's first frame (T0) before
# giving up. A whole-screen capture comes up in well under a second on both
# backends; this is a generous ceiling so a slow spin-up doesn't fail the take.
SEGMENT_T0_TIMEOUT_SEC = 8.0
# Max windows in one occlusion-free fleet -- the compositor caps cards at 4
# (edits._MAX_WINDOWS, framing layouts, the picker). A seamless join must
# refuse a 5th window rather than spawn a worker the render can't place.
_MAX_FLEET = 4
# First-frame ceiling for a JOIN worker. SEGMENT_T0_TIMEOUT_SEC (8s) is the
# resume/segment ceiling; a live join is interactive, so a picked window that
# has already vanished must fail FAST (~2s, keep recording) rather than hang
# the pill for 8s. Success latency is ~0.5-0.8s, well under this.
GROW_T0_TIMEOUT_SEC = 2.0
# Card SHRINK (docs/architecture.md Milestone 3). A captured window must be BOTH
# absent from the on-screen list AND delivery-dead for this long before its
# card exits -- longer than any Space flip or Mission Control gesture, and
# free in the output because the seam back-dates to the hide moment.
SHRINK_HIDE_DEBOUNCE_SEC = 5.0
# How long the run loop waits for a departing worker's SIGINT harvest before
# SIGKILL + moov salvage. Typical finish is <2s; the loop blocks during it
# (within the pause-40s precedent -- survivors are independent processes and
# keep writing throughout).
SHRINK_HARVEST_DEADLINE_SEC = 5.0
# Auto-REJOIN (M3.3): a departed window must be back on the pickable list
# for this long CONTINUOUSLY before its one automatic re-add fires --
# absorbs the restore animation and SCShareableContent re-listing lag so
# the grow doesn't burn its 2s T0 wait on a window mid-genie.
REJOIN_STABLE_SEC = 1.0


class _Segment(object):
    """A finalized segment of a paused/resumed take -- a passive snapshot of
    the per-segment state the active-segment instance fields held while it was
    recording (its file, its parent-clock t0 pairing, its SCK stats). Recorder
    accumulates one per pause and one for the final active segment; the list
    stays empty on a take that is never paused. See
    docs/architecture.md."""

    def __init__(self, index, file_name, t0, stat=None, filt=None, size=None):
        self.index = int(index)
        self.file = file_name          # basename in the session dir
        self.t0 = t0                   # parent-monotonic pairing for THIS file
        self.stat = stat               # SCK capture_stats, or None
        self.filter = filt             # SCK exclude applied/missing, or None
        self.size = size               # SCK SIZE, or None


class _SceneRecord(object):
    """A finalized SCENE of a paused/resumed fleet take -- the parallel
    sibling of `_Segment`: a passive snapshot of the 1-4 `_NativeWorker`s
    whose children have been harvested and moov-gated at the pause. The
    workers themselves are the per-channel state (file path, t0 pairing,
    stats, entry/end_rect), so this only adds the scene index and the
    wall-clock end the shortfall check reads
    (docs/architecture.md, "The fleet timeline shortfall")."""

    def __init__(self, index, workers, wall_end=None):
        self.index = int(index)
        self.workers = list(workers)
        self.wall_end = wall_end


class _DepartedChannel(object):
    """A fleet worker whose card EXITED mid-take (the shrink, docs/architecture.md
    Milestone 3): harvested and gated at departure, retained to finalize --
    its t0/counts/end_rect feed the manifest, where its sub-ranges end at
    the back-dated seam. `gated` False means even the moov salvage failed:
    the channel is dropped from the manifest entirely (file left on disk),
    and the survivors -- whose files span the moment continuously -- get no
    seam from it."""

    def __init__(self, worker, seam_t, gated=True, rank=None):
        self.worker = worker
        self.seam_t = float(seam_t)
        self.gated = bool(gated)
        # Display rank at departure (decision 11): a returnee inherits it so
        # an accidental minimize/restore round-trip keeps the card's SLOT in
        # the layout instead of re-entering at the right edge.
        self.rank = worker.index if rank is None else int(rank)


def _plan_join_scenes(t0s, counts, n_original, fps):
    """Split a gapfree window-JOIN fleet into K+1 scene sub-ranges. PURE
    (unit-tested; no capture state), the load-bearing core of `record.grow`.

    Since M3.1 (docs/architecture.md, Milestone 3) this is a DELEGATION WRAPPER
    over `_plan_fleet_scenes` with no exit events -- pinned byte-identical
    on every join-only shape (`test_record.PlanFleetScenes`). The full
    contract (inputs, scene spans, t0 re-anchoring, the `[]` off-switch)
    lives on `_plan_fleet_scenes`."""
    scenes, _ = _plan_fleet_scenes(t0s, counts, n_original, fps)
    return scenes


def _plan_fleet_scenes(t0s, counts, n_original, fps, exits=None):
    """Split a fleet take into scene sub-ranges from JOIN and EXIT events.
    PURE (unit-tested; no capture state). The M3.1 generalization of
    `_plan_join_scenes`: a scene seam is created by a window joining
    (`record.grow`) OR by a captured window departing (the card SHRINK,
    docs/architecture.md Milestone 3).

    A join keeps every SURVIVOR window on ONE continuous file across the seam
    (that is what makes the add gapless -- the existing workers are never torn
    down), so the `capture_scenes` manifest expresses each scene as a SUB-RANGE
    of that file rather than a per-scene file. `render._render_scenes` reads
    the `frame_start` / `frame_count` this produces. An EXIT is the symmetric
    inverse: the departing channel's sub-ranges simply END at its (back-dated)
    seam, and any frames it wrote after that -- dup-fill of a hidden window --
    go unreferenced, exactly like a demoted joiner's file today.

    Inputs (parallel lists over ALL workers, originals first then joiners in
    join order):
      t0s     -- each worker's first-frame monotonic (parent clock). A
                 joiner's t0 IS its join instant.
      counts  -- each worker's total frame count (from its finalized file).
      n_original -- how many workers were present from the start; workers at
                 index >= n_original are joiners.
      fps     -- capture fps.
      exits   -- {channel_index: exit_t (parent clock)} for departed
                 channels; None/{} = join-only, which reproduces the
                 pre-M3.1 `_plan_join_scenes` output byte-for-byte (the
                 off-switch -- that branch is the verbatim original
                 algorithm).

    Scene semantics (both paths): scene 0 keeps every original with its
    ACTUAL t0 and `frame_start` 0; every scene s>=1 re-anchors its channels'
    `t0` to the scene's start instant so `SegmentClock` sees ASCENDING,
    near-aligned scenes and `origin=max(offsets)` never re-skips consumed
    frames -- the frame_start does the file positioning instead.

    Exit-path rules (docs/architecture.md M3 decisions 2 and 7):
      - Seams = the events sorted by time, each clamped
        `max(t, prev + 1/fps)` so scene starts stay strictly ascending even
        when a back-dated exit lands beside a join.
      - Presence is COMPUTED, never forced: channel i is in scene s iff it
        has joined by the scene's start, has not exited at/before it, and
        its frame_at-clamped slice is non-empty. No `max(1, ...)` frame
        fabrication.
      - Any INTERIOR scene shorter than the M2.3 sub-floor
        (`max(2, fps // 4)` frames, wall-clock between its two seams)
        merges away: a JOIN starting it slides forward onto the next
        boundary (near-simultaneous joins coalesce); an EXIT ending it
        slides back (an exit at ~start drops the channel from the manifest
        entirely); an EXIT starting it slides forward (<=sub-floor of
        frozen dup tail, clock-exact). Events that had slid onto a removed
        seam follow it (`_retarget`). A slide that makes a channel's join
        reach its exit demotes it outright -- both its seams vanish
        (decision 7's join-then-immediately-hide case).
      - The FINAL scene applies exactly ONE shortness rule: a joiner whose
        join starts it with fewer than sub-floor frames demotes (the M2.3
        guard, generalized). It never merges for any other reason -- its
        length is only bounded by per-channel file ends, and a short FILE
        (crashed worker, salvage truncation) must never trigger merges
        that discard ANOTHER channel's real footage. A tiny trailing scene
        costs a few frames of render; discarded footage is unrecoverable.
        Scene 0 ended by a join within the sub-floor of the take start is
        KEPT for the same reason (demoting that joiner was adversarially
        measured to discard 30+ seconds).
      - Known bound (fuzz-reachable, production-unreachable): a CASCADE of
        three or more sub-floor-spaced events can strand a healthy joiner
        (every boundary it could enter at merges away) and demote it --
        reported in `demoted`, file intact, survivors unaffected. Real
        feeds cannot produce the shape: `_try_grow` serializes joins
        behind a ~0.5s+ T0 wait and shrink exits sit behind a 5 s
        debounce, so no two events land within one sub-floor of each
        other, let alone three (pinned:
        `PlanFleetScenes.test_dense_event_cascade_reports_not_corrupts`).

    Returns (scenes, demoted): scenes as
    {start_t, end_t, channels:[{index, frame_start, frame_count, t0}]} --
    [] when no seams survive (with EVERY joiner demoted then: a flat
    manifest can only express the near-equal-t0 originals) -- and demoted =
    sorted channel indices that appear in NO scene. The caller excludes
    demoted channels from a flat manifest; the user-facing departed hint is
    keyed off the caller's OWN exit records, not off this list (a demoted
    dead-file joiner never "departed"). Raises ValueError if a scene would
    have no channels at all -- the detector's last-card guard makes that
    unreachable from a real take."""
    n = len(t0s)
    joins = n - int(n_original)
    if not exits:
        # ---- join-only: the verbatim pre-M3.1 algorithm (byte-identical,
        # pinned). Every existing join take re-finalizes to the same
        # manifest through this branch. ----
        if joins < 1:
            return [], []
        join_times = [t0s[n_original + j] for j in range(joins)]
        scene_starts = [t0s[0]] + join_times      # scene s start instant
        scene_ends = join_times + [None]          # scene s end (None = EOF)

        def frame_at(i, T):
            if T is None:
                return counts[i]
            return max(0, min(counts[i], int(round((T - t0s[i]) * fps))))

        scenes = []
        for s in range(joins + 1):
            present = n_original + s               # channels 0 .. present-1
            chans = []
            for i in range(present):
                if s == 0:
                    # Actual t0s; the shared origin skip (SegmentClock)
                    # absorbs the few-frame inter-channel drift, exactly
                    # like a plain fleet.
                    t0_i, fs = t0s[i], 0
                else:
                    # Re-anchor to the seam; frame_start seeks past what
                    # scene s-1 already consumed of this continuous file.
                    t0_i = scene_starts[s]
                    fs = frame_at(i, scene_starts[s])
                fc = frame_at(i, scene_ends[s]) - fs
                fc = max(1, min(fc, counts[i] - fs))
                chans.append({"index": i, "frame_start": int(fs),
                              "frame_count": int(fc), "t0": t0_i})
            scenes.append({"start_t": scene_starts[s], "end_t": scene_ends[s],
                           "channels": chans})
        return scenes, []

    # ---- generalized path: at least one exit event ----
    step = 1.0 / float(fps)
    min_frames = max(2, int(fps // 4))            # the M2.3 sub-floor
    start_t = t0s[0]
    demoted = set()

    join_eff = {}                                  # ch -> effective join time
    exit_eff = {}                                  # ch -> effective exit time
    events = []                                    # [t, kind, ch] with a seam
    for j in range(joins):
        ch = n_original + j
        join_eff[ch] = t0s[ch]
        events.append([t0s[ch], "join", ch])
    for ch, t in exits.items():
        ch = int(ch)
        if not 0 <= ch < n:
            raise ValueError("exit for unknown channel {}".format(ch))
        exit_eff[ch] = float(t)
        events.append([float(t), "exit", ch])

    # Up-front demotions: an exit at/before the channel ever produced a frame
    # means it was never meaningfully in the take; a 0-frame FILE can appear
    # in no scene at all, and letting its seam survive would leave a spurious
    # no-op boundary (two adjacent scenes with identical content).
    for ch in list(exit_eff):
        born = join_eff.get(ch, start_t)
        if exit_eff[ch] <= born:
            demoted.add(ch)
    for ch in set(join_eff) | set(exit_eff):
        if counts[ch] <= 0:
            demoted.add(ch)
    # Exits sort ahead of joins on a tie: an exit at t owns frames UP TO t,
    # a join at t starts there.
    events = [e for e in events if e[2] not in demoted]
    events.sort(key=lambda e: (e[0], 0 if e[1] == "exit" else 1, e[2]))

    seams = []                                     # [t_clamped, kind, ch]
    prev = start_t
    for t, kind, ch in events:
        st = max(t, prev + step)
        seams.append([st, kind, ch])
        if kind == "join":
            join_eff[ch] = st
        else:
            exit_eff[ch] = st
        prev = st

    def _demote(ch):
        # A demoted channel's boundary may have FOLLOWERS -- events that
        # slid onto it trusting the boundary persists. Deleting it would
        # strand them (adversarially measured: a joiner whose target seam
        # vanished entered 10s late, or not at all). The seam survives
        # under an heir instead; joins inherit first (entrances are the
        # fragile side).
        demoted.add(ch)
        join_eff.pop(ch, None)
        exit_eff.pop(ch, None)
        kept = []
        for s in seams:
            if s[2] != ch:
                kept.append(s)
                continue
            t = s[0]
            heir = next((c for c in sorted(join_eff)
                         if join_eff[c] == t and c not in demoted), None)
            if heir is not None:
                kept.append([t, "join", heir])
                continue
            heir = next((c for c in sorted(exit_eff)
                         if exit_eff[c] == t and c not in demoted), None)
            if heir is not None:
                kept.append([t, "exit", heir])
        seams[:] = kept

    def frame_at(i, T):
        if T is None:
            return counts[i]
        return max(0, min(counts[i], int(round((T - t0s[i]) * fps))))

    def present_in(i, s_start):
        if i in demoted:
            return False
        if i >= n_original and join_eff.get(i, s_start + 1) > s_start:
            return False
        if i in exit_eff and exit_eff[i] <= s_start:
            return False
        return True

    def slice_for(i, s_idx, s_start, s_end):
        if s_idx == 0:
            t0_i, fs = t0s[i], 0
        else:
            t0_i, fs = s_start, frame_at(i, s_start)
        fc = frame_at(i, s_end) - fs
        fc = min(fc, counts[i] - fs)
        return t0_i, fs, fc

    def _retarget(old_t, new_t):
        # A removed seam's event adopted another boundary's time; any OTHER
        # event previously slid onto the removed boundary must follow it, or
        # its effective time points at a non-boundary and its channel lingers
        # past its seam (adversarially measured: cascaded merges referenced
        # dup-fill beyond the sub-floor without this).
        for d in (join_eff, exit_eff):
            for c in list(d):
                if d[c] == old_t:
                    d[c] = new_t
        for c in list(join_eff):
            if c in exit_eff and join_eff[c] >= exit_eff[c]:
                _demote(c)                    # its join reached its own exit

    # Sub-floor merge loop over INTERIOR scenes (wall-clock shortness -- both
    # bounding seams are known instants), plus the one legitimate final-scene
    # rule: a JOINER whose join starts the final scene with almost no frames
    # demotes (the M2.3 guard, generalized). The final scene is deliberately
    # NOT merged for any other shortness: its "length" is only bounded by
    # per-channel file ends, and a short FILE (a crashed worker, a salvage
    # truncation) must never trigger merges that discard ANOTHER channel's
    # real footage (adversarially measured: the earlier min-fc heuristic
    # slid healthy exits back by seconds). A tiny trailing scene renders as
    # a few extra frames; discarded footage is unrecoverable.
    # Each pass removes one seam or demotes one channel, so it terminates.
    while True:
        bounds = [start_t] + [s[0] for s in seams]
        short_k = None
        for k in range(len(bounds)):
            b_start = bounds[k]
            if k < len(seams):
                if k == 0 and seams[0][1] == "join":
                    # A join right after the take start makes scene 0
                    # sub-floor -- KEEP it. Every merge response here loses
                    # real footage (demoting the joiner discards its whole
                    # file -- adversarially measured at 30+ seconds; sliding
                    # its join back would misalign the clock), while a
                    # few-frame opening scene merely renders as a quick cut.
                    continue
                span = int(round((seams[k][0] - b_start) * fps))
                if span < min_frames:
                    short_k = k
                    break
            else:
                # Final scene: only the M2.3 joiner rule (see above).
                late_joiner = next(
                    (c for c in sorted(join_eff)
                     if c not in demoted and join_eff[c] == b_start
                     and slice_for(c, k, b_start, None)[2] < min_frames),
                    None)
                if late_joiner is not None:
                    _demote(late_joiner)
                    short_k = -1              # loop again on fresh seams
                    break
        if short_k is None or not seams:
            break
        if short_k == -1:
            continue
        # short_k is always an INTERIOR scene here (the final scene only ever
        # demotes a late joiner above), so its ending seam always exists.
        k = short_k
        s_seam = seams[k - 1] if k > 0 else None      # seam STARTING scene k
        e_seam = seams[k]                              # seam ENDING scene k
        if s_seam is not None and s_seam[1] == "join":
            # Near-simultaneous events: the join coalesces forward onto the
            # next boundary (followers come along).
            del seams[k - 1]
            _retarget(s_seam[0], e_seam[0])
        elif e_seam[1] == "exit":
            ch, old_t = e_seam[2], e_seam[0]
            new_t = s_seam[0] if s_seam is not None else start_t
            del seams[k]
            _retarget(old_t, new_t)
            if ch not in demoted and (s_seam is None
                                      or join_eff.get(ch, start_t) >= new_t):
                # Slid back to the very start (or onto its own join): the
                # channel was never meaningfully in the take.
                _demote(ch)
        elif s_seam is not None and s_seam[1] == "exit":
            # An exit starting a short scene slides forward: the departing
            # channel carries <= sub-floor of frozen dup tail, and the clock
            # stays exact (sliding the join back instead would misalign the
            # joiner's file t0 against the seam).
            del seams[k - 1]
            _retarget(s_seam[0], e_seam[0])
        else:
            # k == 0 with a join ending seam is skipped by the scan (scene 0
            # is kept), and every k > 0 scene has a join or exit s_seam --
            # there is no other seam kind.
            raise AssertionError("unreachable short-scene shape")

    if not seams:
        # No scene structure survived, so the caller writes a FLAT manifest
        # -- which can only express the originals (near-equal t0s). A joiner
        # whose seam merged away is inexpressible there (its t0 is mid-take;
        # origin=max(offsets) would truncate every survivor to it), so every
        # joiner is demoted with the scenes (adversarially caught: a
        # surviving mid-take joiner poisoned the flat fallback).
        return [], sorted(demoted | set(range(n_original, n)))

    scenes = []
    bounds = [start_t] + [s[0] for s in seams]
    seen = set()
    for k in range(len(bounds)):
        b_start = bounds[k]
        b_end = seams[k][0] if k < len(seams) else None
        chans = []
        for i in range(n):
            if not present_in(i, b_start):
                continue
            t0_i, fs, fc = slice_for(i, k, b_start, b_end)
            if fc <= 0:
                continue                       # computed, never fabricated
            chans.append({"index": i, "frame_start": int(fs),
                          "frame_count": int(fc), "t0": t0_i})
            seen.add(i)
        if not chans:
            raise ValueError(
                "scene {} has no channels -- the last-card guard should "
                "make this unreachable".format(k))
        scenes.append({"start_t": b_start, "end_t": b_end, "channels": chans})
    dropped = demoted | (set(range(n)) - seen)
    return scenes, sorted(dropped)


class Recorder:
    # crf 20, not 16. raw.mov is a MEZZANINE -- render.py decodes it and
    # re-encodes, so capture quality only has to survive one generation and the
    # difference between 16 and 20 does not survive it. It buys ~14% smaller
    # session files.
    # It is NOT an A/V-sync fix, despite looking like one. Measured on a
    # 2880x1800x60 take: crf 16 encoded 15 s of content in 7.49 s, crf 20 in
    # 7.30 s -- a 2.5% difference, with ~2x realtime headroom either way. At
    # `-preset ultrafast` the preset dominates and CRF barely moves throughput,
    # so "the encoder can't keep up" does not explain drift on this machine.
    # THREAD_QUEUE_SIZE is the real fix there: a queue only 8 packets deep
    # drops audio on a TRANSIENT stall, which needs no sustained overload at all.
    def __init__(self, session_dir, video_idx, mic_idx=None, fps=60, crf=20,
                cursor_mode="system", log_keys=True,
                face_idx=None, face_fps=30, capture_window=None,
                capture_windows=None, backend=None, exclude_windows=None,
                exclude_provider=None, window_native=False):
        self.session_dir = session_dir
        self.video_idx = video_idx
        # Which capture backend writes raw.mov. Defaults to avfoundation and
        # will keep doing so until the gate in docs/architecture.md is met
        # -- see sck.resolve_backend for why "off must be bit-exact" forces
        # that.
        self.backend = sck.resolve_backend(backend, os.environ)
        # Window ids kept OUT of the recording. Only the SCK backend can
        # honor these; avfoundation has no window channel at all (measured:
        # it ignores NSWindowSharingNone too).
        self.exclude_windows = [int(w) for w in (exclude_windows or [])]
        # Called ~2 Hz during an SCK take to re-read that list. Only the app
        # can answer it (it owns the bar), so it arrives as a callback rather
        # than something the Recorder works out for itself. None = the set is
        # fixed for the take, which is what the CLI wants.
        self._exclude_provider = exclude_provider
        self._stop_exclude_poll = threading.Event()
        self.mic_idx = mic_idx
        self.fps = fps
        self.crf = crf
        self.cursor_mode = cursor_mode if cursor_mode in CURSOR_MODES else "system"
        self.log_keys = bool(log_keys)
        # Facecam: a SECOND, best-effort avfoundation capture of the webcam to
        # face.mov. face_idx=None means no facecam. A facecam failure must
        # never abort the screen recording (same posture as key capture).
        self.face_idx = face_idx
        self.face_fps = int(face_fps)
        # When set, the facecam is NOT a second avfoundation capture: the
        # process already holding the camera for preview tees its frames into
        # face.mov instead (camera_preview.start_recording). That is what lets
        # the preview keep running through the take -- macOS gives a camera to
        # one process at a time, so as long as we opened our own, the bubble
        # had to go dark exactly when it was most wanted.
        self.face_shared_ordinal = None
        # Window capture: avfoundation CANNOT target a window, so we capture the
        # full display exactly as always and record the window's rect in
        # meta.json for render.py to CROP to. `capture_window` is a
        # devices.window_rect_points entry (POINTS, global top-left origin),
        # already validated by the caller. None = a normal full-display take,
        # and then meta.json gets no capture_window key at all -- such a session
        # renders bit-identically to before this feature existed.
        # `capture_windows` is the MULTI-window pick (2-4 entries), which is a
        # different feature riding the same substrate: there is no single rect
        # to crop to, so meta gets a `capture_windows` block instead and
        # render composites the windows onto a background. A list of ONE
        # collapses into `capture_window` above -- a single window is better
        # served by the crop path, which keeps the auto-zoom camera alive.
        # None/empty -> not one line of this runs.
        picked = [w for w in (capture_windows or []) if isinstance(w, dict)]
        if len(picked) == 1 and capture_window is None:
            capture_window, picked = picked[0], []
        self.capture_window = (capture_window
                               if isinstance(capture_window, dict) else None)
        self.capture_windows = picked if len(picked) >= 2 else []
        # Window-NATIVE capture (occlusion-free): the SCK worker captures the
        # target window's OWN backing buffer via SCContentFilter, so raw.mov IS
        # the window (occluders never bleed in) instead of a display we crop at
        # render time. `_is_window_native()` guards the SINGLE-window flavor,
        # `_is_multi_window_native()` the N-window one (P3.1) -- both require
        # the SCK backend, and are mutually exclusive at construction below.
        # Off (the default) is byte-identical to the display-crop path -- no
        # capture_window_id in the worker config, no `mode` in meta. See
        # docs/architecture.md and
        # docs/architecture.md.
        self.window_native = bool(window_native)
        # Multi-native fleet. Populated when `window_native=True` +
        # `capture_windows` has 2+ entries on the SCK backend; empty for every
        # other configuration, which is the fail-safe: the multi-native code
        # path only runs when the list is non-empty. The list is built here so
        # `start()` can dispatch on it without re-deriving the mode.
        #
        # SCENE TAKES (docs/architecture.md) widen the entry: a SINGLE
        # occlusion-free pick also rides the fleet -- as a 1-worker fleet --
        # when neither mic nor facecam was requested, because the pause/re-pick
        # machinery lives in the fleet loop and the parent gate's own opening
        # scene is one window. A never-paused 1-worker fleet down-converts at
        # finalize to the pinned single-native shape (`_finalize_fleet_single`),
        # so the on-disk contract is unchanged. With mic or facecam the take
        # stays on the single-native path (both are unsupported on the fleet;
        # losing them would be a worse regression than an unpausable take).
        self._sck_workers = []
        self._fleet_single = False
        if self.window_native and self.backend == "sck":
            fleet = list(self.capture_windows)
            if (not fleet and self.capture_window is not None
                    and self.mic_idx is None and self.face_idx is None):
                fleet = [self.capture_window]
                self._fleet_single = True
            if fleet:
                self._sck_workers = [
                    _NativeWorker(session_dir, i, w)
                    for i, w in enumerate(fleet)]
                # Multi-native lives on the manifest, not on the single-window
                # capture_window block -- keep the two mutually exclusive so a
                # meta consumer never sees both keys and has to guess. The
                # 1-worker fleet KEEPS capture_window: `_is_window_native()`
                # stays true (start() dispatches on the fleet guard FIRST),
                # which is what lets the never-paused down-conversion write
                # the pinned single-native meta through the unchanged
                # `_capture_window_meta` / window-track machinery -- and no
                # meta writer emits both keys (`_meta_dict` has no channels;
                # the manifest dicts never read capture_window).
                if not self._fleet_single:
                    self.capture_window = None
        self._cw_entry = None        # post-countdown re-snapshot, or None
        self._cw_resnapshot = False  # False -> meta rect is the pick-time one
        self._cw_end_rect = None     # geometry at stop, for the drift warning
        # Parallel to self.capture_windows, same three roles as the singles.
        self._cws_entries = []
        self._cws_resnapshot = []
        self._cws_end_rects = []
        # Filled by the un-overlap step (autocine.arrange) when it moves windows
        # out of each other's way before the take; drives the meta `arrange`
        # state and the restore in _finalize.
        self._arrange_state = "none"
        self._arrange_restore = None
        # Geometry track: "ok" (samples logged) | "failed" (poller ran but
        # never got a readable rect) | None (no window requested). Mirrors
        # key_capture's tri-state so the app can tell "window never moved"
        # apart from "we never managed to look".
        self._window_track = None
        self._win_stop = threading.Event()
        self._win_thread = None
        self._win_samples = 0
        # Per-window-id, since the poller samples every on-screen window:
        # coalescing has to be independent or one busy window would suppress
        # the heartbeat of every still one.
        self._win_last_rect = {}
        self._win_last_write = {}
        self._win_last_z = {}
        # "Brought to the front DURING the take" -- the join-candidate rule
        # (`raised_window_ids` / `_raise_tick`). `_win_front0` is whatever was
        # already frontmost on the poller's FIRST reading (never a raise);
        # `_win_front_since` is the (wid, t) the current front started at, and
        # `_win_raised` is the promoted set. The set is REBOUND, never mutated
        # in place, so a reader on the HTTP thread always sees a consistent
        # snapshot without a lock (the poller is the only writer).
        self._win_front0 = None
        self._win_front_seen = False
        self._win_front_since = None
        self._win_raised = frozenset()
        self.raw_path = os.path.join(session_dir, "raw.mov")
        self.face_path = os.path.join(session_dir, "face.mov")
        self.events_path = os.path.join(session_dir, "events.jsonl")
        self.meta_path = os.path.join(session_dir, "meta.json")
        # Written ONLY on a failed take (see _write_error_log). A failure
        # leaves no meta.json, so this is the only thing in the session dir
        # that says what went wrong.
        self.error_path = os.path.join(session_dir, "error.log")
        self.proc = None
        self.face_proc = None
        self._watchdog_proc = None
        self._face_t0 = None
        self._face_t0_lock = threading.Lock()
        self._face_stderr_tail = collections.deque(maxlen=40)
        # "ok" | "failed" | None (not requested). Mirrors key_capture.
        self._face_capture = None
        self._t0 = None
        self._t0_lock = threading.Lock()
        self._ev_lock = threading.Lock()
        self._ev_file = None
        self._last_move = 0.0
        self._last_xy = (None, None)
        self._last_key_bucket = -1
        self._last_scroll = -1e18   # so the first scroll always logs
        # "activity" (worker ran to the end of the take) | "failed" (couldn't
        # start -- the Input Monitoring trap -- OR died partway through) |
        # "disabled" (--no-key-log); absent from meta.json in sessions
        # recorded before key capture existed. The died-partway case is
        # settled at stop time by _settle_key_capture, NOT here.
        self._key_capture = None
        # CFR verdict for the file this take writes: "ok" | "suspect" |
        # "unknown", settled by _check_cfr after the file is closed. Not in
        # meta.json on the ffmpeg backend -- the normal-session key set is
        # pinned by dict equality (tests/test_record.py) and this backend has
        # never been seen to violate CFR, so it warns rather than reports.
        self._cfr = None
        self._cfr_detail = ""
        # Last STAT/FILTER seen from the SCK worker, and its first ERR. None
        # on the avfoundation path, which is what keeps these out of meta.
        self._sck_stat = None
        self._sck_filter = None
        self._sck_error = None
        # The worker's SIZE line (encoded pixel dims). Consumed for EVERY SCK
        # take, but only written into meta on the window-native path -- the
        # display-crop SCK meta key set must stay byte-identical (its buffer is
        # the display, derivable without this).
        self._sck_size = None
        # The worker's DONE line (its own "I finalized the movie" signal), or
        # None if it never reported one -- which, paired with a missing `moov`
        # box, is how a tail-lost take is told apart from a clean one.
        self._sck_done = None
        self._stderr_tail = collections.deque(maxlen=60)
        self._stop_requested = threading.Event()
        # Segmented takes (pause/resume). `_paused` gates the shared event
        # append so the deleted gap logs no input; the request Events are set
        # by pause()/resume() (called from the HTTP/bar thread) and acted on by
        # the single-path run loop. `_segments` accumulates one _Segment per
        # FINALIZED segment; it stays empty on a take that is never paused, so
        # the single-file path is byte-identical. See
        # docs/architecture.md.
        self._paused = threading.Event()
        self._pause_requested = threading.Event()
        self._resume_requested = threading.Event()
        self._ever_paused = False
        self._segments = []
        self._paused_accum = 0.0
        self._pause_started = None
        # When the active segment's capture child was spawned, in the parent's
        # monotonic clock -- the t0 fallback for a segment finalized before
        # its first-frame pairing lands (the segmented sibling of the
        # single-file path's `start_wall` fallback).
        self._seg_spawn_mono = None
        # Scene takes (docs/architecture.md): one _SceneRecord per
        # finalized fleet scene; `_scene_pending` carries the window entries a
        # `resume(scene=...)` re-pick asked for, consumed by the fleet run
        # loop. Both stay empty/None on a take that is never paused -- the
        # fleet's single-file success path then runs byte-identically.
        self._scenes = []
        self._scene_pending = None
        # Seamless window-JOIN (docs/architecture.md milestone 2): `grow(entry)` adds
        # ONE window to a LIVE fleet without pausing -- the existing workers
        # keep writing their continuous files, so this is nothing like the
        # pause seam. `_grow_requested` + `_grow_pending` are the request
        # channel (set from the HTTP/bar thread, drained by the fleet loop);
        # `_join_marks` records each accepted joiner's worker index so finalize
        # can build the sub-range `capture_scenes` manifest (`_plan_join_scenes`).
        # `_n_original` is the pre-join fleet size (the survivors). All empty on
        # a take that never grows -> the flat manifest, byte-identical.
        self._grow_requested = threading.Event()
        self._grow_pending = None
        self._join_marks = []
        self._n_original = None
        # The wid whose worker is mid-spawn inside `_try_grow` -- set before
        # the spawn, cleared once appended (or on abort). It closes the
        # spawn-not-yet-appended gap AT THE RECORDER: `captured_window_ids`
        # doesn't yet list a joiner during its ~2s T0 wait, so a same-wid
        # re-request drained in that window would otherwise spawn a duplicate.
        self._growing_wid = None
        # Card SHRINK state (docs/architecture.md Milestone 3). All empty-by-
        # default -- the structural off-switch: a take where no window ever
        # hides never enters shrink code and finalizes byte-identically.
        # `_shrink_pending` is written ONLY by the geometry-poller thread and
        # swapped out whole by the run loop (the grow mailbox discipline).
        self._shrink_requested = threading.Event()
        self._shrink_pending = None            # {wid: back-dated seam_t}
        self._shrink_tracker = {}              # sck.plan_shrink_exits state
        self._exit_marks = []                  # departed channel indexes
        self._departed = []                    # [_DepartedChannel, ...]
        # Env kill-switch for triage, read once (not a product surface).
        self._shrink_enabled = os.environ.get(
            "AUTOCINE_FLEET_SHRINK", "1") != "0"
        # Channel-index allocator (monotone) + the fleet's TRUE original
        # count. `len(self._sck_workers)` stops being either once cards can
        # leave: a rejoin must never collide with a departed channel's
        # raw_{i}.mov, and originals are index < _n_start forever.
        self._n_start = len(self._sck_workers) or None
        self._next_channel_idx = self._n_start
        # Auto-REJOIN state (M3.3). The mailbox mirrors the shrink's:
        # `_rejoin_watch`/`_rejoin_pending` are written ONLY by the geometry
        # poller; `_rejoin_attempted` and `_display_rank` ONLY by the run
        # loop (one attempt per departure; a re-departure earns a new one).
        self._rejoin_requested = threading.Event()
        self._rejoin_pending = None            # set of wids ready to re-add
        self._rejoin_watch = {}                # sck.plan_rejoin_ready state
        self._rejoin_attempted = set()
        # Layout SLOT per channel index (decision 11): originals keep their
        # index; a grown card appends; a REJOINED card inherits its departed
        # predecessor's rank. Per-scene manifest channel order sorts by this.
        self._display_rank = {w.index: w.index for w in self._sck_workers}
        # Mic-anchor freeze note (decision 6's hint): the poller stashes the
        # app name once per hide episode; the run loop notifies. One-shot:
        # `_anchor_noted` resets when the anchor is SEEN again.
        self._anchor_note_pending = None
        self._anchor_noted = False

    # ---- pynput callbacks -------------------------------------------------
    def _append(self, rec):
        """The single serialized writer to events.jsonl. ALL three event
        sources (mouse/click via `_write_event`, keys via `_on_key`, geometry
        via `_write_window_event`) route through here so the pause gate lives
        in ONE place. While PAUSED (a segmented take's deleted gap) every write
        is dropped -- the gap must contain no logged input, or a click/rect
        would map onto the seam a render can't show. When never paused
        (`_paused` unset) this is byte-identical to the old inline writes.
        """
        with self._ev_lock:
            if self._ev_file is None or self._paused.is_set():
                return False
            self._ev_file.write(json.dumps(rec) + "\n")
            return True

    def _write_event(self, etype, x, y, button=None):
        rec = {"t": time.monotonic(), "type": etype,
               "x": float(x), "y": float(y)}
        if button:
            rec["button"] = button
        self._append(rec)

    def _on_move(self, x, y):
        now = time.monotonic()
        lx, ly = self._last_xy
        moved = lx is None or (abs(x - lx) + abs(y - ly)) >= 2
        if (now - self._last_move) >= 0.008 and moved:  # ~120 Hz cap
            self._last_move = now
            self._last_xy = (x, y)
            self._write_event("move", x, y)

    def _on_click(self, x, y, button, pressed):
        self._write_event("down" if pressed else "up", x, y, str(button))

    def _on_scroll(self, x, y, dx, dy):
        # Scroll activity tick at the cursor position (macOS routes scroll to
        # the window UNDER the cursor, so x/y is where the scrolled content
        # lives -- the camera's reading anchor). Deltas are deliberately not
        # stored: the camera only needs "scrolling happened here, now".
        # Zero-delta events (trackpad gesture phase bookkeeping) are skipped.
        if not dx and not dy:
            return
        now = time.monotonic()
        if (now - self._last_scroll) >= SCROLL_MIN_INTERVAL_SEC:
            self._last_scroll = now
            self._write_event("scroll", x, y)

    def _mouse_listener_kwargs(self):
        """The pynput mouse.Listener callback wiring, extracted so a test
        can pin that every callback (in particular on_scroll) is actually
        REGISTERED -- a correct-but-unwired callback logs nothing."""
        return dict(on_move=self._on_move,
                    on_click=self._on_click,
                    on_scroll=self._on_scroll)

    def _start_key_listener(self, popen=None):
        """Best-effort keyboard listener start. Returns (worker, state).

        NOT an in-process pynput.keyboard.Listener any more. pynput's macOS
        keyboard backend decodes every press through Carbon/TIS
        (`TISCopyCurrentKeyboardInputSource` et al, reached via raw ctypes --
        see autocine/_key_worker.py's docstring for the full story), and that
        call reproduces a hard native SIGTRAP crash in this app's process --
        confirmed 6/6 across independent real recordings, including a fully
        manual one. A native trap kills the whole process; no try/except can
        stop it. So the decode now happens in a disposable CHILD PROCESS
        (_key_worker.py) that reports bare activity timestamps over a pipe --
        if IT crashes, only it dies, and this recording is unaffected. Same
        posture as the facecam capture, same reason.

        On macOS, pynput's darwin backend does NOT raise when Input
        Monitoring is denied -- the event tap fails inside the listener
        thread, which marks itself ready and silently exits. The child
        mirrors that: it prints "READY" only once its OWN wait()+is_alive()
        check passes, and exits silently (no output) on the denial path. So
        "failed" here covers both a launch failure AND a live-but-denied tap,
        exactly like the old in-process check did. `popen` is injectable for
        tests (no permissions, no real subprocess needed).
        """
        popen = popen or subprocess.Popen
        try:
            proc = popen(
                [sys.executable, _KEY_WORKER_PATH],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, start_new_session=True)
        except Exception:
            return None, "failed"
        worker = _KeyWorker(proc, self._on_key)
        if not worker.wait_ready(timeout=KEY_WORKER_READY_TIMEOUT_SEC):
            worker.stop()
            return None, "failed"
        worker.start_reading()
        return worker, "activity"

    def _settle_key_capture(self, kb_listener):
        """Downgrade `key_capture` to "failed" if the worker died mid-take.

        `_start_key_listener` can only report what was true at the READY
        handshake. The whole reason the decode lives in a child process is
        that it sits on a call with a known native-crash history -- so "it
        started" is emphatically not "it survived". Without this, a worker
        that traps one second in leaves meta.json claiming "activity" while
        events.jsonl holds zero key ticks, which is indistinguishable from a
        take where the user simply never typed. That ambiguity is exactly
        what makes a keyboard-capture regression invisible.

        MUST be called BEFORE `kb_listener.stop()` -- stop() kills the
        process, after which is_alive() is False for every take, successful
        ones included.

        Only ever downgrades "activity"; "disabled" and an already-"failed"
        start are left alone. "failed" is reused deliberately rather than
        adding a fourth state: it already means "no usable key track" and is
        already handled end to end by the app, the renderer, and the docs.
        """
        if self._key_capture != "activity" or kb_listener is None:
            return
        try:
            alive = kb_listener.is_alive()
        except Exception:
            return   # best-effort, like everything else on this path
        if not alive:
            self._key_capture = "failed"

    def _write_error_log(self, message):
        """Persist a failure explanation into the session dir.

        A failed take raises RecordError, whose message carries the ffmpeg
        stderr tail and the diagnosis -- but that only ever reached the
        console of whatever launched the recorder. Launched from the bar,
        that console is a window nobody is watching, so a failure left
        behind a session dir with events.jsonl, maybe a face.mov, and no
        clue at all about what happened. Root-causing then depends on the
        user reproducing it live. Writing it next to the take makes the
        failure diagnosable after the fact instead.

        Best-effort: a take is already failing when we get here, and a
        problem writing this file must not replace the real error.
        """
        try:
            os.makedirs(self.session_dir, exist_ok=True)
            with open(self.error_path, "w") as f:
                f.write(message.rstrip() + "\n")
        except Exception:
            pass

    def _on_key(self, _key, t=None):
        # PRIVACY INVARIANT: _key is deliberately never read (and the child
        # process that decodes it never sends it anywhere -- see
        # _key_worker.py). What lands on disk is a bare quantized activity
        # tick -- {"t", "type": "key"} plus the last observed cursor position
        # as x/y padding (the move track already logs that at up to 120 Hz,
        # so it discloses nothing new; it keeps older checkouts' load_events,
        # which destructure x/y unconditionally, from crashing on new
        # sessions). `t` is the child's time.monotonic() reading for this
        # press, ALREADY TRANSLATED into our own timeframe by _KeyWorker --
        # the two clocks do not share a reference point (see that class's
        # docstring), so an untranslated reading would land a whole
        # parent-process-age in the past. Defaults to "now" so every existing
        # caller/test (which has no child process to read a timestamp from)
        # is unaffected.
        now = time.monotonic() if t is None else t
        b = int(now / KEY_BUCKET_SEC)
        if b == self._last_key_bucket:
            return
        self._last_key_bucket = b
        lx, ly = self._last_xy
        if lx is None:
            lx = ly = 0.0
        self._append({"t": b * KEY_BUCKET_SEC, "type": "key",
                      "x": float(lx), "y": float(ly)})

    # ---- ffmpeg reader threads --------------------------------------------
    def _read_progress(self):
        stream = self.proc.stdout
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if us >= 0:
                    with self._t0_lock:
                        if self._t0 is None:
                            self._t0 = time.monotonic() - us / 1_000_000.0

    def _read_stderr(self):
        stream = self.proc.stderr
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            self._stderr_tail.append(raw.decode("utf-8", "replace"))

    # ---- facecam (second, best-effort ffmpeg) -----------------------------
    def _face_cmd(self):
        """ffmpeg command for the webcam capture. Video only (the mic rides
        the screen capture); CFR + zerolatency so it shares the screen
        track's tight monotonic sync. Extracted for unit testing."""
        return ["ffmpeg", "-y", "-hide_banner",
                "-f", "avfoundation",
                "-framerate", str(self.face_fps),
                "-thread_queue_size", str(THREAD_QUEUE_SIZE),
                "-i", _fmt_input(self.face_idx, None),
                "-c:v", "libx264", "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-crf", "20", "-pix_fmt", "yuv420p",
                "-r", str(self.face_fps),
                "-progress", "pipe:1", "-nostats", self.face_path]

    def _read_face_progress(self):
        proc = self.face_proc
        if proc is None or proc.stdout is None:
            return
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("out_time_us="):
                try:
                    us = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if us >= 0:
                    with self._face_t0_lock:
                        if self._face_t0 is None:
                            self._face_t0 = time.monotonic() - us / 1_000_000.0

    def _read_face_stderr(self):
        proc = self.face_proc
        if proc is None or proc.stderr is None:
            return
        for raw in iter(proc.stderr.readline, b""):
            self._face_stderr_tail.append(raw.decode("utf-8", "replace"))

    def _wait_screen_capturing(self, timeout):
        """Block until the screen ffmpeg has encoded its first frame (``_t0``
        is set) -- i.e. its avfoundation capture session is fully up -- or
        until ``timeout`` elapses, the process dies, or a stop is requested.

        This SERIALIZES avfoundation session startup. If the webcam capture's
        AVCaptureSession is configured at the same instant as the screen's,
        the screen input's configuration fails ("Configuration of video device
        failed" -> fallback to a camera the webcam already holds -> the screen
        ffmpeg dies with no raw.mov). Letting the screen session come up first
        avoids that race entirely. Returns True if the screen was confirmed
        capturing, False on timeout/death/stop.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._t0_lock:
                if self._t0 is not None:
                    return True
            if self._stop_requested.is_set():
                return False
            if self.proc is not None and self.proc.poll() is not None:
                return False   # screen already died; nothing to serialize
            time.sleep(0.02)
        return False

    def _start_watchdog(self, kb_listener):
        """Launch the process-death watchdog covering every child that
        would otherwise be silently orphaned if THIS process disappears
        without running _finalize() -- a crash (the exact SIGTRAP class
        _key_worker.py exists to contain, or any other), a SIGKILL, anything
        that skips normal cleanup. See _watchdog.py's docstring: this was
        found the hard way, as nine orphaned ffmpeg processes still running
        hours after a run of crashes, quietly recording and burning CPU and
        disk the whole time with nothing on screen to suggest it.

        Best-effort like everything else here -- if the watchdog itself
        can't launch, the recording proceeds exactly as it always has
        (i.e. as it did before this existed).
        """
        pids = [self.proc.pid]
        if kb_listener is not None and getattr(kb_listener, "proc", None):
            pids.append(kb_listener.proc.pid)
        if self.face_proc is not None:
            pids.append(self.face_proc.pid)
        self._watchdog_proc = spawn_watchdog(pids)

    def _stop_watchdog(self):
        """Tell the watchdog to stand down on a NORMAL stop -- its job is
        only for the abnormal case, and it must not linger as its own
        orphan after every ordinary, successful take."""
        proc = self._watchdog_proc
        self._watchdog_proc = None
        stop_watchdog(proc)

    def _start_face(self):
        """Launch the best-effort webcam capture and its reader threads.
        Returns (progress_thread, stderr_thread), or (None, None) if the
        process couldn't even be launched. MUST be called only after the
        screen session is confirmed running -- see _wait_screen_capturing for
        why concurrent avfoundation session setup kills the screen capture."""
        try:
            self.face_proc = subprocess.Popen(
                self._face_cmd(), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                start_new_session=True)
        except Exception:
            self.face_proc = None
            return None, None
        fp = threading.Thread(target=self._read_face_progress, daemon=True)
        fe = threading.Thread(target=self._read_face_stderr, daemon=True)
        fp.start()
        fe.start()
        return fp, fe

    def _start_shared_face(self):
        """Arm the face sink on the camera the preview already owns.

        No second avfoundation capture, so nothing to serialize against the
        screen session and nothing for the preview to give up. Best-effort:
        if the sink won't arm (camera closed, ffmpeg missing), the take just
        gets no face.mov, exactly as a failed _start_face would.
        """
        try:
            from . import camera_preview
            return camera_preview.start_recording(
                self.face_shared_ordinal, self.face_path, self.face_fps)
        except Exception:
            return False

    def _stop_face_capture(self):
        """Stop whichever facecam flavor is running (dedicated ffmpeg or the
        shared preview sink). Idempotent and best-effort -- used at the first
        PAUSE of a segmented take (the webcam must not record through the
        deleted gap) and safe to re-run from `_finalize`."""
        if self.face_proc is not None and self.face_proc.poll() is None:
            try:
                self.face_proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                self.face_proc.wait(timeout=6)
            except Exception:
                try:
                    self.face_proc.kill()
                except Exception:
                    pass
        self._stop_shared_face()

    def _stop_shared_face(self):
        """Close the face sink and adopt its t0 as the facecam anchor."""
        if self.face_shared_ordinal is None:
            return
        try:
            from . import camera_preview
            t0, frames = camera_preview.stop_recording(self.face_shared_ordinal)
        except Exception:
            return
        if t0 is not None and frames > 0:
            with self._face_t0_lock:
                self._face_t0 = t0

    # ---- capture window (full-display capture, render-time crop) ----------
    @staticmethod
    def _rect_points(entry):
        """`[x, y, w, h]` in POINTS, global top-left origin -- the meta.json
        `capture_window` rect contract. Points and not pixels deliberately:
        meta already carries logical_w/h and render.py derives the per-axis
        pixel scale from the REAL file (that's what keeps fractional Retina
        exact), so a stored pixel rect would be a second source of truth that
        can silently disagree with it."""
        return [float(entry["x"]), float(entry["y"]),
                float(entry["w"]), float(entry["h"])]

    def _snapshot_capture_window(self):
        """Fresh Quartz re-read of the chosen window's entry, or None.

        None means gone / minimized / moved offscreen / shrunk below the size
        gates / dragged to a secondary display -- every case where the caller
        should keep the rect it already had rather than trust a fresh one.
        Never raises: window geometry is best-effort exactly like the facecam
        and must never take down a recording. This is the seam tests
        monkeypatch (`record.dev.window_rect_points`).
        """
        return self._snapshot_window(self.capture_window)

    def _snapshot_window(self, entry):
        """`_snapshot_capture_window` for an arbitrary picked entry.

        Split out so the multi-window pick re-reads each of its windows
        through the exact same seam (and the exact same soft-failure
        posture) as the single one.
        """
        if not entry:
            return None
        try:
            wid = int(entry.get("id"))
        except (TypeError, ValueError):
            return None
        try:
            return dev.window_rect_points(
                wid, exclude_pids=(os.getpid(),),
                # Occlusion-free captures the window's whole surface, so its
                # rect must not be clipped to the display (`_window_native_space`).
                clip_to_display=not self._window_native_space())
        except Exception:
            return None

    # ---- window geometry track --------------------------------------------
    def _window_rect_changed(self, wid, rect):
        """Did this window move/resize beyond Quartz's 1pt jitter floor?"""
        prev = self._win_last_rect.get(wid)
        if prev is None:
            return True
        return any(abs(a - b) >= WINDOW_MOVE_EPS_PT for a, b in zip(rect, prev))

    def _write_window_event(self, wid, rect, z=None):
        """Append one geometry sample for one window.

        PRIVACY: geometry and an opaque window id ONLY -- never the app name
        or window title. Those would turn a rect log into a record of what
        you had open, which is a different thing from what this needs: render
        only has to know WHICH samples belong together over time, and the id
        does that without describing anything. meta.json's `capture_window`
        block still names the one window the user explicitly picked.
        `x`/`y` carry the window CENTRE purely as forward-compat padding, for
        the same reason the key tick pads them: an older checkout's
        load_events destructures x/y for every non-key line.

        `z` is the front-to-back rank (0 = frontmost) and is written ONLY
        when known, so a sample from before this existed -- and any caller
        that doesn't have a rank -- serializes byte-identically to before.
        It is an integer, so it stays on the right side of the privacy line:
        it says a window was covered, never by what.
        """
        now = time.monotonic()
        sample = {
            "t": now, "type": "window", "id": int(wid),
            "rect": [float(v) for v in rect],
            "x": float(rect[0] + rect[2] / 2.0),
            "y": float(rect[1] + rect[3] / 2.0),
        }
        if z is not None:
            sample["z"] = int(z)
        if not self._append(sample):
            # Dropped (paused gap, or the file is closed): the coalescing
            # state must NOT advance, or a window that appeared/moved during
            # the gap is marked already-written and its first post-resume
            # sample is suppressed until the next change or heartbeat --
            # exactly at the seam where a scene take's window set changes.
            # (docs/architecture.md, the geometry-coalescing seam fix.)
            return
        self._win_samples += 1
        self._win_last_rect[wid] = list(rect)
        self._win_last_write[wid] = now
        if z is not None:
            self._win_last_z[wid] = int(z)

    def _poll_window_geometry(self):
        """Sample EVERY on-screen window's rect until stop.

        Every window and not just the capture target, because the
        multi-window grid's cards are crops of arbitrary windows the user
        drew rects around -- they can only follow if their geometry was
        recorded too. Per-window coalescing keeps the cost of that
        proportional to how much actually moves.

        The display list is read once: it is ~6.9 ms of an ~8.6 ms
        enumeration, it cannot change mid-take in any configuration this
        tool supports (window capture is main-display-only), and this loop
        runs beside the encoder.

        Best-effort exactly like key capture and the facecam: Quartz refusing
        to answer must never take down a recording -- it just thins the
        track, and render falls back to a static crop for what it lacks.
        """
        try:
            displays = dev.displays_points()
        except Exception:
            displays = None
        own_pid = os.getpid()
        while not self._win_stop.is_set():
            try:
                entries = dev.list_windows(
                    exclude_pids=(own_pid,), displays=displays,
                    # MUST match the space `_snapshot_window` records into:
                    # render divides buffer pixels by these rects, and a track
                    # in a different space than the meta rect it is scaled
                    # against is wrong every frame instead of once.
                    clip_to_display=not self._window_native_space())
            except Exception:
                entries = []
            # list_windows is FRONT-TO-BACK, so the index IS the z rank --
            # free, and the only thing that can later tell "this window was
            # covered" apart from "this window was fine".
            for z, entry in enumerate(entries):
                try:
                    wid = int(entry["id"])
                    rect = self._rect_points(entry)
                except (KeyError, TypeError, ValueError):
                    continue
                stale = (time.monotonic() - self._win_last_write.get(wid, -1e18)
                         >= WINDOW_HEARTBEAT_SEC)
                raised = self._win_last_z.get(wid) != z
                if stale or raised or self._window_rect_changed(wid, rect):
                    self._write_window_event(wid, rect, z=z)
            try:
                self._raise_tick(entries)
            except Exception:
                pass                        # detection is best-effort, always
            try:
                self._shrink_tick(entries)
            except Exception:
                pass                        # detection is best-effort, always
            self._win_stop.wait(WINDOW_POLL_SEC)

    def _raise_tick(self, entries):
        """One pass of the "brought to the front" detector, riding the same
        20 Hz geometry poll (docs/architecture.md M2.4c).

        WHY this exists: the join-candidate feed used to be "windows whose id
        was not on screen when the take started" -- which reads as "a window I
        opened", but on macOS is not. Apps keep their windows around, so
        clicking Finder in the Dock or opening a document RAISES an existing
        window rather than creating one. A real take (recordings/20260831-092226)
        opened Finder and a Word doc and got no chip and no auto-add for
        either: both ids were already on screen at t0, so the baseline
        subtraction had ruled them out before the take began. The rank this
        reads is the same `z` the geometry track already logs, so a raise is
        observable for free.

        The rule is a TRANSITION to frontmost, held for
        WINDOW_RAISE_DWELL_SEC, and never the window that was already front
        on the first reading -- that one the user did not bring up, and on
        this machine it is typically the terminal the server was launched
        from. `entries` is front-to-back, so entry 0 IS the front.

        Best-effort like every other passenger on this loop: the caller
        swallows exceptions, and a missed raise costs a chip, never a take.
        """
        own = set(self.exclude_windows or ())
        front = None
        for entry in entries:
            try:
                wid = int(entry["id"])
            except (KeyError, TypeError, ValueError):
                break                       # unreadable front: treat as no reading
            # Our OWN chrome must never occupy the front slot: the pill and the
            # notes overlay are always-on-top, and one of them sitting at z=0
            # for the whole take would hide every real raise beneath it. They
            # are normally filtered out of `entries` already (they are not on
            # layer 0), so this is a cheap backstop, not the primary defense.
            if wid in own:
                continue
            front = wid
            break
        if front is None:
            # Nothing pickable on screen (or an unreadable entry): hold the
            # dwell where it is rather than crediting or resetting it.
            return
        if not self._win_front_seen:
            self._win_front_seen = True
            self._win_front0 = front
            return
        if front == self._win_front0 or front in self._win_raised:
            self._win_front_since = None
            return
        now = time.monotonic()
        if self._win_front_since is None or self._win_front_since[0] != front:
            self._win_front_since = (front, now)
        elif now - self._win_front_since[1] >= WINDOW_RAISE_DWELL_SEC:
            self._win_raised = self._win_raised | {front}
            self._win_front_since = None

    def _shrink_tick(self, entries):
        """One pass of the card-shrink detector (docs/architecture.md M3.2), run by
        the geometry poller it rides on -- the ONE owner of hide detection
        (parent clock, every take shape, ~20 Hz).

        Feeds `sck.plan_shrink_exits` (the pure decision) one reading per
        SHRINKABLE captured window: the mic anchor is exempt (decision 6:
        killing worker 0 kills the take's voiceover), the last live card
        never shrinks (a zero-card scene is inexpressible), an ever-paused
        take is disarmed (the S1 scene machine has no exit support), and the
        env kill-switch disarms everything. Absent-from-the-filtered-list is
        disambiguated with one direct per-id query -- present-but-filtered
        (other display, tiny, alpha shim) counts as SEEN, as does any Quartz
        error. Arrivals ride the worker's own STAT feed (appended - dup);
        `window_rebuild_gaveup` in a worker's tail promotes to a trigger
        (restored but the stream is dead forever -- decision 3).

        Fired exits land in the `_shrink_pending` mailbox for the run loop;
        this thread never touches workers.
        """
        if not self._shrink_enabled or self._ever_paused:
            return
        if self._paused.is_set():
            return
        workers = list(self._sck_workers)
        if not self._is_multi_window_native():
            return
        listed = set()
        for entry in entries:
            try:
                listed.add(int(entry["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        # The REJOIN watch runs regardless of live-card count -- a 2-card
        # take shrunk to 1 is exactly the flagship accidental-minimize case
        # that must come back. Exit DETECTION below needs >=2 live cards.
        self._watch_rejoins(workers, listed)
        if self._shrink_requested.is_set():
            return
        # Exit DETECTION needs >=2 live cards; the mic-anchor freeze NOTE
        # does not (a mic fleet shrunk to the anchor alone is exactly where
        # "Keep <App> visible" matters most -- the whole visible recording
        # is a frozen card while the mic keeps rolling). With `few`, only
        # the anchor is fed to the detector, whose exits are discarded into
        # the note logic below; nothing can reach the mailbox.
        few = len(workers) < 2
        onscreen, arrivals, gaveup = {}, {}, set()
        anchor_wid = None
        anchor_app = None
        for w in workers:
            try:
                wid = int((w.entry or w.window)["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if few and not (self.mic_idx is not None and w.index == 0):
                continue
            if self.mic_idx is not None and w.index == 0:
                # The mic anchor never shrinks (decision 6) -- but it still
                # COUNTS toward the global-vanish guard (decision 1 reads
                # "captured wids", not "shrinkable wids": anchor + one card
                # vanishing together is a Minimize-All / app event, not
                # per-window intent). Its own exits are discarded below.
                anchor_wid = wid
                anchor_app = str((w.entry or w.window).get("app")
                                 or "the recorded window")
            on = True if wid in listed else dev.window_onscreen(wid)
            onscreen[wid] = on
            st = w.stat or {}
            arrivals[wid] = (int(st.get("appended", 0))
                             - int(st.get("dup", 0)))
            if any("window_rebuild_gaveup" in line
                   for line in w.stderr_tail):
                gaveup.add(wid)
        self._shrink_tracker, exits = sck.plan_shrink_exits(
            self._shrink_tracker, onscreen, arrivals, time.monotonic(),
            SHRINK_HIDE_DEBOUNCE_SEC, gaveup=gaveup)
        if anchor_wid is not None:
            if onscreen.get(anchor_wid) is not False:
                if self._anchor_noted:
                    # The anchor is BACK: clear the server-side hint (a
                    # reloaded bar must not show a stale "keep it visible"
                    # for the rest of the take) and re-arm for the next
                    # episode. The bar resets its dedupe key on absence, so
                    # episode 2's identical app name still surfaces.
                    self._anchor_note_pending = ("seen", None)
                self._anchor_noted = False
            elif any(wid == anchor_wid for wid, _ in exits) \
                    and not self._anchor_noted:
                # The anchor met the full exit bar (hidden + delivery-dead
                # past the debounce) and was spared -- its card is frozen.
                # One hint per hide episode (decision 6); the mic itself
                # keeps recording (measured, M3.0d).
                self._anchor_noted = True
                self._anchor_note_pending = ("hidden", anchor_app)
        exits = [(wid, t) for wid, t in exits if wid != anchor_wid]
        if exits:
            pending = dict(self._shrink_pending or {})
            pending.update({int(wid): float(t) for wid, t in exits})
            self._shrink_pending = pending
            self._shrink_requested.set()

    def _watch_rejoins(self, workers, listed):
        """The poller-side half of auto-rejoin (M3.3): watch DEPARTED wids
        for a stable return to the pickable list and mail the ready ones to
        the run loop. Never touches workers or grow state -- eligibility is
        decided at drain time (`_try_rejoin`), where a pending manual pick
        always wins. One attempt per departure: `_rejoin_attempted` filters
        here; a re-departure clears it (`_try_shrink`)."""
        captured = set()
        for w in workers:
            try:
                captured.add(int((w.entry or w.window)["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        wids = []
        for d in self._departed:
            try:
                dwid = int((d.worker.entry or d.worker.window)["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if dwid in self._rejoin_attempted or dwid in captured:
                continue
            if len(self._sck_workers) >= _MAX_FLEET:
                continue                       # nothing to watch at the cap
            wids.append(dwid)
        if not wids:
            self._rejoin_watch = {}
            return
        self._rejoin_watch, ready = sck.plan_rejoin_ready(
            self._rejoin_watch, wids, listed, time.monotonic(),
            REJOIN_STABLE_SEC)
        if ready:
            pending = set(self._rejoin_pending or ())
            pending.update(int(w) for w in ready)
            self._rejoin_pending = pending
            self._rejoin_requested.set()

    def _start_window_track(self):
        """Launch the geometry poller.

        Runs for EVERY take, not just a window-targeted one: the multi-window
        grid is authored after the fact on an ordinary full-display
        recording, so the geometry has to already be on disk by then. A
        session where nothing is enumerable just gets no `window` lines and
        renders exactly as it would have before.
        """
        if self.capture_window:
            self._window_track = "failed"   # upgraded on the first real sample
        self._win_thread = threading.Thread(target=self._poll_window_geometry,
                                            daemon=True)
        self._win_thread.start()

    def _stop_window_track(self):
        self._win_stop.set()
        if self._win_thread is not None:
            self._win_thread.join(timeout=2.0)
            self._win_thread = None
        # "ok" reports the CAPTURE TARGET specifically -- samples of other
        # windows say nothing about whether the crop can be followed.
        if self._window_track is not None and self.capture_window:
            try:
                wid = int(self.capture_window.get("id"))
            except (TypeError, ValueError):
                return
            if wid in self._win_last_rect:
                self._window_track = "ok"

    def _resolve_capture_window(self):
        """Re-snapshot the window right before ffmpeg starts.

        Called AFTER the countdown on purpose: the countdown is exactly when
        the user brings their window forward, so a pick-time-only rect would
        make it useless. A failed re-read falls back to the pick-time rect and
        records `resnapshot: false` -- never abort a take that is already
        counting down.
        """
        if self.capture_windows:
            self._resolve_capture_windows()
            return
        if not self.capture_window:
            return
        fresh = self._snapshot_capture_window()
        self._cw_entry = fresh
        self._cw_resnapshot = fresh is not None
        if fresh is None:
            print("  window not found — using the rect from when you picked it.")

    def _resolve_capture_windows(self):
        """`_resolve_capture_window` for every window of a multi pick.

        A window that has gone away keeps its pick-time rect rather than
        dropping out of the composition: the layout was chosen with it in it,
        and a card frozen on stale geometry is a far smaller surprise
        mid-take than the grid silently reflowing to N-1 cards.
        """
        missing = 0
        for entry in self.capture_windows:
            fresh = self._snapshot_window(entry)
            self._cws_entries.append(fresh)
            self._cws_resnapshot.append(fresh is not None)
            if fresh is None:
                missing += 1
        if missing:
            print("  {} of {} windows not found — using the rects from when "
                  "you picked them.".format(missing, len(self.capture_windows)))

    def _window_native_space(self):
        """Does this take's window geometry live in UNCLIPPED window space?

        `devices` reports every window INTERSECTED with its display, which is
        right for a display-crop pick (an avfoundation crop cannot reach
        pixels outside the captured display) and wrong the moment SCK is
        capturing the window's own surface, because that surface includes
        whatever hangs off the edge of the screen. A window dragged half off
        the right edge then records a rect describing strictly LESS than the
        file contains, and everything downstream that divides one by the
        other is wrong by that ratio: the card comes out stretched, and
        `raw_w / logical_w` -- the event transform's scale -- is inflated on
        the clipped axis, so clicks land off-target inside that card.

        Deliberately LOOSER than `_is_window_native` / `_is_multi_window_native`:
        those answer "is a fleet running", which is false during the pick and
        the countdown, while this answers "which coordinate space do this
        take's rects belong to" -- and that is settled at construction, by
        the same condition __init__ uses to build the fleet at all. Reading a
        rect before the workers exist is exactly when it matters.

        See docs/architecture.md, "The display-CLIPPED rect".
        """
        return bool(self.window_native and self.backend == "sck")

    def _is_window_native(self):
        """Is this an occlusion-free, capture-the-window's-own-buffer take?

        The SINGLE guard every window-native consumer keys off, kept fail-safe:
        only true with a single capture window on the SCK backend, so an
        inconsistent caller can never half-enter the mode. When false, every
        path below runs exactly as it did before this feature.
        """
        return (self.window_native and self.backend == "sck"
                and self.capture_window is not None)

    def _is_multi_window_native(self):
        """Is this an occlusion-free native FLEET take (1-4 workers)?

        Constructor guarantees `_sck_workers` is empty unless every condition
        holds: window_native, SCK backend, and either 2-4 capture_windows OR
        a single mic-less/face-less capture_window (the scene-capable
        1-worker fleet -- docs/architecture.md). That makes THIS the single guard
        the fleet code path branches on; callers never re-derive the mode.
        N>=2 clears `capture_window` (manifest shapes never carry it); the
        1-worker fleet KEEPS it, deliberately -- `_is_window_native()` stays
        true, which is what lets the never-paused down-conversion
        (`_finalize_fleet_single`) write the pinned single-native meta
        through the unchanged `_capture_window_meta` / window-track
        machinery. No meta writer ever emits both keys.
        """
        return bool(self._sck_workers)

    def _capture_window_meta(self):
        """The optional meta.json `capture_window` block, or None when no
        window was requested (or its entry is unusable -- omitting the key
        fails safe to a normal full-frame render)."""
        src = self._cw_entry or self.capture_window
        if not src:
            return None
        try:
            origin = src.get("display_origin") or (0.0, 0.0)
            cw = {
                "id": int(src["id"]),
                "app": src.get("app"),
                "title": src.get("title"),
                "units": "points",
                "rect": self._rect_points(src),
                "display_origin": [float(origin[0]), float(origin[1])],
                "source": "quartz",
                "resnapshot": bool(self._cw_resnapshot),
                "end_rect": self._cw_end_rect,
                # "ok" -> events.jsonl carries a geometry track render can
                # follow; "failed" -> fall back to the single `rect` above,
                # bit-exact with pre-track renders.
                "track": self._window_track,
            }
            if self._is_window_native():
                # WINDOW-NATIVE marker: raw.mov IS the window, so render must
                # NOT crop. These keys appear ONLY here, so a display-crop
                # capture_window block stays byte-identical. `logical_w/h` are
                # the WINDOW's own point size (the scale denominator for
                # mapping global-point events into this source), distinct from
                # the top-level logical_w/h, which stay the display's.
                cw["mode"] = "window_native"
                cw["logical_w"] = float(src["w"])
                cw["logical_h"] = float(src["h"])
                if self._sck_size is not None:
                    # The encoded buffer size the worker actually pinned. Not
                    # derivable from the display here, unlike the crop path.
                    cw["buffer_w"] = int(self._sck_size.get("width", 0))
                    cw["buffer_h"] = int(self._sck_size.get("height", 0))
            return cw
        except (KeyError, IndexError, TypeError, ValueError):
            return None

    def _capture_windows_meta(self):
        """The optional meta.json `capture_windows` block (2-4 windows), or
        None when this wasn't a multi-window take.

        Deliberately a sibling of `capture_window` rather than a superset of
        it: the single-window block means "crop the frame to this", and this
        one means "composite these onto a background". Writing both would
        leave render guessing which one the session actually is.
        """
        if not self.capture_windows:
            return None
        origin = (0.0, 0.0)
        out = []
        for i, picked in enumerate(self.capture_windows):
            src = None
            if i < len(self._cws_entries):
                src = self._cws_entries[i]
            src = src or picked
            try:
                wid = int(src["id"])
                rect = self._rect_points(src)
            except (KeyError, IndexError, TypeError, ValueError):
                continue
            origin = src.get("display_origin") or origin
            end_rect = (self._cws_end_rects[i]
                        if i < len(self._cws_end_rects) else None)
            resnap = (bool(self._cws_resnapshot[i])
                      if i < len(self._cws_resnapshot) else False)
            out.append({
                "id": wid,
                "app": src.get("app"),
                "title": src.get("title"),
                "rect": rect,
                "end_rect": end_rect,
                "resnapshot": resnap,
                # Per window, same meaning as the single block's: "ok" means
                # this one's geometry is in events.jsonl and its card can
                # follow it.
                "track": "ok" if wid in self._win_last_rect else "failed",
            })
        if not out:
            return None
        try:
            display_origin = [float(origin[0]), float(origin[1])]
        except (IndexError, TypeError, ValueError):
            display_origin = [0.0, 0.0]
        return {
            "units": "points",
            "source": "quartz",
            "display_origin": display_origin,
            # "none"     -> nothing overlapped, nothing was touched
            # "moved"    -> windows were pulled apart before the take
            # "failed"   -> we tried and couldn't; expect occlusion
            # "declined" -> they overlapped and the user said don't
            "arrange": self._arrange_state,
            "windows": out,
        }

    # ---- lifecycle --------------------------------------------------------
    def start(self, countdown=3, duration=None, status_cb=None):
        """Record until Ctrl+C / duration. Returns the session dir on success,
        None if the user aborted during the countdown, and raises RecordError
        if ffmpeg failed to capture."""
        # Multi-window native (P3.1) is a strict fan-out of the single-window
        # SCK path: N independent workers, N raw_i.mov files, one manifest.
        # It runs on a parallel method so `start()` -- which handles avfoundation,
        # single SCK, facecam, watchdog, exclusion polling, etc. -- stays
        # exactly the code that shipped before this feature. `_is_multi_window_native`
        # is fail-safe (constructor guarantees the mode is fully consistent),
        # so the dispatch is a single line.
        if self._is_multi_window_native():
            return self._start_multi_native(countdown=countdown,
                                            duration=duration,
                                            status_cb=status_cb)

        from pynput import mouse

        def notify(state, **extra):
            if status_cb is None:
                return
            try:
                status_cb(state, extra)
            except Exception:
                pass

        self._stop_requested.clear()
        os.makedirs(self.session_dir, exist_ok=True)
        pts_w, pts_h, geom_src = dev.main_display_points()

        # Ctrl+C during the countdown aborts cleanly — nothing is open yet.
        try:
            for i in range(countdown, 0, -1):
                if self._stop_requested.is_set():
                    print("\nrecording stopped before start.")
                    return None
                notify("countdown", seconds=i)
                print("  recording in {}...".format(i), end="\r", flush=True)
                time.sleep(1)
            print(" " * 40, end="\r")
        except KeyboardInterrupt:
            print("\naborted before recording started.")
            return None

        sck_backend = self.backend == "sck"
        cmd = ([sys.executable, _SCK_WORKER_PATH] if sck_backend
               else self._screen_cmd())

        # Window rect is re-read HERE -- after the countdown, immediately before
        # ffmpeg starts -- because the countdown is exactly when the user brings
        # the target window forward. Best-effort; see _resolve_capture_window.
        self._resolve_capture_window()

        self._ev_file = open(self.events_path, "w")
        listener = None
        kb_listener = None
        prog = errt = None
        face_prog = face_errt = None
        start_wall = time.monotonic()
        self._seg_spawn_mono = start_wall
        died_early = False
        try:
            # Own session group so a terminal Ctrl+C doesn't kill ffmpeg
            # mid-write; we stop it explicitly (clean moov) in _finalize().
            # The SCK worker deliberately wears the same process shape --
            # SIGINT to stop, non-zero exit means failure, stderr tailed --
            # so _finalize, the watchdog and the death checks need no second
            # version. It differs in one way only: its config arrives on
            # stdin, so that pipe is open rather than DEVNULL.
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if sck_backend else subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, start_new_session=True)
            if sck_backend:
                self.proc.stdin.write(
                    (json.dumps(self._sck_config()) + "\n").encode("utf-8"))
                self.proc.stdin.flush()

            if sck_backend and self._exclude_provider is not None:
                threading.Thread(target=self._poll_exclusions,
                                 daemon=True).start()

            prog = threading.Thread(
                target=self._read_sck_stdout if sck_backend
                else self._read_progress, daemon=True)
            errt = threading.Thread(target=self._read_stderr, daemon=True)
            prog.start()
            errt.start()

            # Input listeners don't touch avfoundation -- start them now so no
            # early click/move/keypress is lost while we wait (below) for the
            # screen's capture session to come up before starting the webcam.
            listener = mouse.Listener(**self._mouse_listener_kwargs())
            listener.start()

            # Window geometry poller: same "start it before the avfoundation
            # wait" reasoning as the input listeners -- it touches only
            # Quartz, and an early drag must not be lost.
            self._start_window_track()

            # Keyboard capture is best-effort: a listener that can't start
            # (Input Monitoring not granted to this exact process tree) must
            # never abort an otherwise-good screen recording. The tri-state
            # key_capture flag in meta.json is what lets the app distinguish
            # "user never typed" from this silent failure -- see
            # _start_key_listener for why plain try/except can't detect it.
            if self.log_keys:
                kb_listener, self._key_capture = self._start_key_listener()
            else:
                self._key_capture = "disabled"

            # Facecam is best-effort AND must not race the screen's avfoundation
            # startup. Two AVCaptureSessions configuring at once makes the
            # SCREEN input's configuration fail ("Configuration of video device
            # failed" -> avfoundation falls back to a camera the webcam capture
            # already holds -> the screen ffmpeg dies with no raw.mov). So the
            # screen gets an uncontested head start before the webcam capture
            # touches avfoundation -- ending early the moment the screen
            # confirms a frame, so the bubble isn't held back once it's safe.
            # A launch/capture failure here still just leaves face_capture
            # "failed" and no face.mov.
            if self.face_idx is not None:
                self._wait_screen_capturing(timeout=FACE_START_DELAY_SEC)
                if self.proc is not None and self.proc.poll() is None:
                    if self.face_shared_ordinal is not None:
                        self._start_shared_face()
                    else:
                        face_prog, face_errt = self._start_face()

            self._start_watchdog(kb_listener)

            msg = "recording — press Ctrl+C to stop"
            if duration:
                msg += "  (auto-stop {:g}s)".format(duration)
            print(msg)
            notify("recording")

            # Run loop with pause/resume (segmented takes). A take that is
            # never paused never leaves the `recording` state and `_segments`
            # stays empty -- the single-file success path below then runs
            # byte-identically. pause()/resume() (from the HTTP/bar thread) set
            # request Events; the loop does the actual segment finalize/spawn so
            # those endpoints return immediately.
            state = "recording"
            while True:
                if self._stop_requested.is_set():
                    notify("stopping")
                    if state == "recording" and self._ever_paused:
                        if self._finalize_active_segment() is None:
                            died_early = True
                    break
                if state == "recording":
                    if duration and (time.monotonic() - start_wall
                                     - self._paused_accum) >= duration:
                        if self._ever_paused:
                            if self._finalize_active_segment() is None:
                                died_early = True
                        break
                    if self._pause_requested.is_set():
                        self._pause_requested.clear()
                        notify("pausing")
                        # Gate the event writers FIRST (the doc's pause step
                        # 1): the finalize below takes seconds (SIGINT + reap
                        # + moov write) during which the capture produces no
                        # frames -- a click or window drag logged in that
                        # window would land in the deleted gap and clamp onto
                        # the seam as a phantom cluster the auto-zoom would
                        # zoom on.
                        self._paused.set()
                        # Facecam is a Phase-1 cut on segmented takes -- and
                        # the cut must be enforced at the CAMERA, not just in
                        # meta: without this the webcam keeps recording
                        # through the privacy gap into an unreferenced
                        # face.mov. One-way stop at the first pause,
                        # best-effort like every facecam path.
                        self._stop_face_capture()
                        seg = self._finalize_active_segment()
                        prog.join(timeout=2.0)
                        errt.join(timeout=2.0)
                        if seg is None:
                            died_early = True
                            break
                        self._ever_paused = True
                        # Stale-replay fix, mirrored from the fleet loop: a
                        # duplicate resume during `resuming` re-set the
                        # consumed event, and the elif chain would fire it
                        # at the NEXT pause as a ghost self-resume.
                        self._resume_requested.clear()
                        self._scene_pending = None
                        self._pause_started = time.monotonic()
                        state = "paused"
                        notify("paused", segments=len(self._segments))
                    elif self.proc.poll() is not None:
                        died_early = True  # capture exited on its own -> failure
                        break
                elif self._resume_requested.is_set():
                    self._resume_requested.clear()
                    notify("resuming")
                    if self._pause_started is not None:
                        self._paused_accum += (time.monotonic()
                                               - self._pause_started)
                    prog, errt = self._spawn_screen_segment(len(self._segments))
                    # Re-arm the watchdog over the NEW segment's pid (the old
                    # dog covered the finalized segment's dead pid; it was kept
                    # through the gap so the key worker never went uncovered).
                    self._stop_watchdog()
                    self._start_watchdog(kb_listener)
                    if not self._wait_for_t0(SEGMENT_T0_TIMEOUT_SEC):
                        # A user Stop during the spin-up is NOT a failure:
                        # discard the segment that never delivered a frame
                        # and let the top-of-loop stop check finalize the
                        # take from the segments already on disk, exactly
                        # like stop-while-paused. Only a real spin-up
                        # failure (child death / T0 timeout) fails the take.
                        if self._stop_requested.is_set():
                            self._abort_spawned_segment(prog, errt)
                            continue
                        died_early = True
                        break
                    self._paused.clear()
                    state = "recording"
                    notify("recording")
                time.sleep(0.05)
        except KeyboardInterrupt:
            pass
        finally:
            if listener is not None:
                listener.stop()
            if kb_listener is not None:
                # Order matters: ask whether it survived BEFORE stopping it.
                self._settle_key_capture(kb_listener)
                try:
                    kb_listener.stop()
                except Exception:
                    pass
            self._finalize()
            if prog is not None:
                prog.join(timeout=2.0)
            if errt is not None:
                errt.join(timeout=2.0)
            if face_prog is not None:
                face_prog.join(timeout=2.0)
            if face_errt is not None:
                face_errt.join(timeout=2.0)

        # Segmented take (>=2 segments produced by pause/resume): a distinct
        # finalize -- moov-gate every segment, drop the top-level raw, write the
        # capture_segments manifest. A never-paused take, or a pause-then-stop
        # take with a single segment, has <2 segments and falls through to the
        # byte-identical single-file path below.
        if len(self._segments) >= 2:
            return self._finalize_segmented(pts_w, pts_h, geom_src, died_early)

        # Failure detection: ffmpeg dying on its own, or no usable file.
        rc = self.proc.returncode if self.proc is not None else None
        raw_ok = (os.path.exists(self.raw_path)
                  and os.path.getsize(self.raw_path) > 1024)
        if died_early or not raw_ok:
            message = self._failure_message(rc, raw_ok)
            self._write_error_log(message)
            raise RecordError(message)

        # SCK ONLY: frames present and size > 1024 is NOT proof of a usable
        # take. If the worker was stopped before it finished writing the movie
        # tail, raw.mov holds every captured frame but no `moov` index, and
        # opens to `moov atom not found` in every consumer -- the AVAssetWriter
        # tail-loss risk (docs/architecture.md) made concrete. Left
        # unchecked it wrote a meta.json and filed the corrupt take as a good
        # project, which then poisons the library. Fail it HONESTLY instead:
        # error.log, no meta.json, treated as a failed take like any other.
        # avfoundation is deliberately untouched -- ffmpeg writes its trailer
        # on SIGINT and has never been seen to lose it, and its success/meta
        # shape is pinned bit-exact.
        if self.backend == "sck" and not sck.has_moov(self.raw_path):
            message = self._sck_unfinalized_message(rc)
            self._write_error_log(message)
            raise RecordError(message)

        self._check_cfr()

        t0 = self._t0 if self._t0 is not None else start_wall

        # Facecam result: only "ok" if a usable face.mov was actually written.
        face_ok = (self.face_idx is not None
                   and os.path.exists(self.face_path)
                   and os.path.getsize(self.face_path) > 1024)
        if self.face_idx is not None:
            self._face_capture = "ok" if face_ok else "failed"

        meta = self._meta_dict(pts_w, pts_h, geom_src, t0, face_ok)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir

    def _meta_dict(self, pts_w, pts_h, geom_src, t0, face_ok):
        """The complete meta.json payload. Recorder is meta.json's SINGLE
        writer -- this is just the seam that lets tests inspect the exact key
        set without ffmpeg or permissions."""
        meta = {
            "fps": self.fps,
            "logical_w": pts_w, "logical_h": pts_h, "geom_source": geom_src,
            "t0_monotonic": t0,
            "video_index": self.video_idx, "mic_index": self.mic_idx,
            "raw": os.path.basename(self.raw_path),
            "events": os.path.basename(self.events_path),
            "cursor_mode": self.cursor_mode,
            "key_capture": self._key_capture,
            # Facecam sidecar (absent-capture -> face=None so render skips it).
            "face": os.path.basename(self.face_path) if face_ok else None,
            "face_index": self.face_idx,
            "face_fps": self.face_fps if face_ok else None,
            "face_t0_monotonic": self._face_t0 if face_ok else None,
            "face_capture": self._face_capture,
        }
        # OPTIONAL key, written only when a window was actually requested: a
        # session without it must stay byte-identical to a pre-feature one.
        cw = self._capture_window_meta()
        if cw is not None:
            meta["capture_window"] = cw
        cws = self._capture_windows_meta()
        if cws is not None:
            meta["capture_windows"] = cws
        # Same optional-key discipline for the capture backend: an
        # avfoundation session's key set must stay EXACTLY what it was, which
        # tests/test_record.py pins by dict equality. So these appear only on
        # the SCK path, where there is no pre-existing shape to preserve.
        if self.backend == "sck":
            meta["capture_backend"] = self.backend
            meta["cfr"] = self._cfr
            if self.exclude_windows:
                meta["excluded_windows"] = list(self.exclude_windows)
            if self._sck_filter is not None:
                # Records that our chrome really was kept out -- and, when
                # `missing` is non-zero, that some of it may not have been.
                meta["excluded_applied"] = int(self._sck_filter.get("applied", 0))
                meta["excluded_missing"] = int(self._sck_filter.get("missing", 0))
            if self._sck_stat is not None:
                meta["capture_stats"] = {
                    "appended": int(self._sck_stat.get("appended", 0)),
                    "duplicated": int(self._sck_stat.get("dup", 0)),
                    "dropped": int(self._sck_stat.get("dropped", 0)),
                    "notready": int(self._sck_stat.get("notready", 0)),
                    # Of the duplicates, how many the wall-clock tick wrote
                    # through spans where SCK delivered nothing at all. A
                    # still window is mostly this; without it those seconds
                    # were absent from the file entirely.
                    "idle_filled": int(self._sck_stat.get("idle", 0)),
                }
        return meta

    def stop(self):
        """Request a graceful stop for an in-flight recording."""
        self._stop_requested.set()

    # ---- segmented takes: pause / resume ---------------------------------

    def pause(self):
        """Request a pause (a true stop/start segment boundary). Returns fast;
        the run loop finalizes the active segment on its next tick. No-op if a
        pause/resume is already in flight or the take is already paused."""
        if not self._paused.is_set():
            self._pause_requested.set()

    def resume(self, scene=None):
        """Request a resume: spawn the next segment. Returns fast; the run loop
        spawns it and clears the paused state once the new segment's first
        frame lands. No-op unless currently paused.

        `scene` (scene takes, fleet loop only): a list of 1-4 validated window
        entries (devices.window_rect_points dicts) to RE-PICK the window set
        for the next scene; None resumes with the previous scene's windows.
        The whole-screen (P1) run loop ignores it -- callers gate on
        `pause_supported` / `_is_multi_window_native` before passing one.
        """
        if self._paused.is_set():
            self._scene_pending = scene
            self._resume_requested.set()

    def grow(self, entry):
        """Seamless window-JOIN: add ONE window to a LIVE fleet WITHOUT pausing
        (docs/architecture.md milestone 2). `entry` is a validated
        devices.window_rect_points dict. Returns fast; the fleet run loop
        spawns the worker, waits its first frame and appends it -- the existing
        windows never stop recording, so unlike resume there is no seam and no
        deleted span. The loop refuses a join it can't honor (ever paused,
        at the 4-card cap, or the window vanished) and simply keeps recording; a
        failed grow is never a failed take. Last-wins if two land before the
        loop drains (a second in-flight join is a rare follow-up)."""
        self._grow_pending = entry
        self._grow_requested.set()

    @property
    def departed_window_ids(self):
        """Window ids whose cards EXITED this take (M3.3/M3.4): the rejoin
        watch and the trigger surface read it; best-effort snapshot like
        `captured_window_ids`."""
        out = set()
        for d in list(self._departed):
            try:
                out.add(int((d.worker.entry or d.worker.window)["id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return out

    @property
    def raised_window_ids(self):
        """Window ids the user BROUGHT TO THE FRONT during this take (M2.4c):
        the trigger surface lets these ESCAPE the start-of-take baseline, so a
        window that was already open behind the cards can still be offered as
        a chip / auto-added. Populated by `_raise_tick` on the geometry poll;
        a frozenset that is rebound rather than mutated, so this needs no lock
        and can never tear."""
        return self._win_raised

    @property
    def grow_supported(self):
        """Can a window be JOINED to this live take right now? Occlusion-free
        fleets, below the 4-card cap, and never paused: finalize routes any
        2+-scene take through the SCENE manifest, which has no join support
        yet (`_join_marks` is ignored there) -- a joiner accepted after a
        pause would be silently dropped from the manifest, its file orphaned
        (docs/architecture.md M3.0c). Mic is fine -- channel 0's audio is one
        continuous file across the join."""
        if (not self._is_multi_window_native() or self._paused.is_set()
                or self._ever_paused):
            return False
        return len(self._sck_workers) < _MAX_FLEET

    @property
    def captured_window_ids(self):
        """The set of window ids the fleet is currently recording -- the
        trigger surface subtracts these from the live window list to offer
        only NEW windows for a join. Best-effort over a snapshot of the worker
        list (the fleet loop appends under grow); a torn read at worst
        mislists one candidate for one poll, and grow re-resolves at spawn."""
        ids = set()
        for w in tuple(self._sck_workers):
            try:
                ids.add(int((w.entry or w.window)["id"]))
            except (KeyError, TypeError, ValueError):
                pass
        return ids

    @property
    def pause_supported(self):
        """Can this take pause at a segment/scene boundary?

        True for a whole-screen take (P1 segmented) and for an occlusion-free
        fleet take (scene takes -- including the 1-worker fleet). False for
        every display-crop window capture: pausing one would run the P1
        machinery, whose `_segmented_meta_dict` carries no capture_window(s)
        block -- the take's load-bearing coordinate space would silently
        vanish and the export would come out full-screen.

        Also False while a MIC is recording an occlusion-free fleet: the scene
        (paused) render is video-only today, so a paused segment would silently
        drop the voiceover across the seam. Audio-across-a-scene-seam is the
        next milestone; until it lands, a mic take stays single-scene rather
        than muting half the narration.

        And False once the fleet has JOINED or SHRUNK (M3.2, decision 8):
        finalize routes any 2+-scene take through the SCENE manifest, which
        knows nothing of `_join_marks`/`_exit_marks` -- pausing after either
        would silently drop the joined/departed channel from the take. The
        shrink being AUTOMATIC makes "minimize, then pause" an ordinary
        gesture pair, so this gate is load-bearing, not defensive.
        """
        if self._is_multi_window_native():
            if self._join_marks or self._exit_marks or self._departed:
                return False
            return self.mic_idx is None
        return not self.capture_window and not self.capture_windows

    def _reset_active_segment_state(self):
        """Clear the instance fields the ACTIVE segment's reader writes, so the
        next segment starts from a clean slate (its own T0/stats). Only the
        active segment ever uses these; finalized segments keep their own
        snapshot in `_segments`."""
        with self._t0_lock:
            self._t0 = None
        self._sck_stat = None
        self._sck_filter = None
        self._sck_size = None
        self._sck_done = None
        self._sck_error = None

    def _spawn_screen_segment(self, index):
        """Spawn the whole-screen capture for segment `index` into its own
        file, and start its stdout/stderr readers. Returns (prog, errt).

        Segment 0 keeps `raw.mov` (so a never-paused take is byte-identical and
        a pause-then-stop take stays a plain single file); later segments write
        `seg_i.mov`. The exclusion poller (started once in `start()`) idles
        through the pause gap and re-reads `self.proc` each tick, so it follows
        the new segment with no restart."""
        self.raw_path = (os.path.join(self.session_dir, "raw.mov") if index == 0
                         else os.path.join(self.session_dir,
                                           "seg_{:d}.mov".format(index)))
        self._reset_active_segment_state()
        self._seg_spawn_mono = time.monotonic()
        sck_backend = self.backend == "sck"
        cmd = ([sys.executable, _SCK_WORKER_PATH] if sck_backend
               else self._screen_cmd())
        proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE if sck_backend else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True)
        if sck_backend:
            proc.stdin.write(
                (json.dumps(self._sck_config()) + "\n").encode("utf-8"))
            proc.stdin.flush()
        # Publish the child only AFTER its config line is written: the
        # exclusion poller stays alive through the pause and re-reads
        # `self.proc` each tick, and an EXCLUDE line landing as the worker's
        # FIRST stdin line would fail its config json.loads and kill the
        # take. (The start() spawn is safe by ordering alone -- the poller
        # thread starts after that config write.)
        self.proc = proc
        prog = threading.Thread(
            target=self._read_sck_stdout if sck_backend else self._read_progress,
            daemon=True)
        errt = threading.Thread(target=self._read_stderr, daemon=True)
        prog.start()
        errt.start()
        return prog, errt

    def _wait_for_t0(self, timeout):
        """Block until the active segment's reader has paired its first frame
        into `self._t0`, or `timeout` elapses. -> True if T0 landed. Keeping
        `_paused` set until this returns means no event is stamped into a
        segment whose clock origin is not yet known."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_requested.is_set():
                return False
            with self._t0_lock:
                if self._t0 is not None:
                    return True
            if self.proc is not None and self.proc.poll() is not None:
                return False        # the child died before delivering a frame
            time.sleep(0.02)
        with self._t0_lock:
            return self._t0 is not None

    def _abort_spawned_segment(self, prog=None, errt=None):
        """Stop arrived while a resumed segment was still spinning up: reap the
        just-spawned capture child, remove its frameless partial file, and
        restore the active-segment instance fields from the LAST finalized
        segment -- so the fallthrough finalize (segmented manifest for >=2
        segments, the plain single-file path for a pause-then-stop take with
        one) sees exactly the state it would after stop-while-paused."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
            except OSError:
                pass
            self._wait_clean()
        # Drain the dead worker's readers to EOF BEFORE restoring: their
        # STAT/FILTER/SIZE setters are unconditional, so a late line from the
        # discarded segment would overwrite the snapshot restored below.
        if prog is not None:
            prog.join(timeout=2.0)
        if errt is not None:
            errt.join(timeout=2.0)
        aborted = self.raw_path
        last = self._segments[-1] if self._segments else None
        if last is not None:
            self.raw_path = os.path.join(self.session_dir, last.file)
            with self._t0_lock:
                self._t0 = last.t0
            self._sck_stat = last.stat
            self._sck_filter = last.filter
            self._sck_size = last.size
        if aborted != self.raw_path:
            try:
                os.remove(aborted)
            except OSError:
                pass

    def _finalize_active_segment(self):
        """SIGINT + reap the active segment's capture child, verify its file has
        a `moov`, and snapshot its clock/stats into a `_Segment`. Returns the
        `_Segment`, or None on a finalize failure (which fails the whole take).
        Reuses the single-path `_wait_clean` grace exactly."""
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.send_signal(signal.SIGINT)
            except OSError:
                pass
            self._wait_clean()
        file_name = os.path.basename(self.raw_path)
        ok = (os.path.exists(self.raw_path)
              and os.path.getsize(self.raw_path) > 1024)
        if self.backend == "sck":
            ok = ok and sck.has_moov(self.raw_path)
        if not ok:
            return None
        with self._t0_lock:
            t0 = self._t0
        if t0 is None:
            # The T0 pairing rides the reader thread; a pause fired within the
            # segment's first ~second can beat it here even though frames are
            # on disk. Give the still-running reader a beat, then fall back to
            # the segment's spawn time -- the same approximation the
            # single-file path uses (start_wall) -- rather than writing a
            # null the SegmentClock would coerce to 0, which would clamp
            # every one of this segment's events onto the seam.
            deadline = time.monotonic() + 2.0
            while t0 is None and time.monotonic() < deadline:
                time.sleep(0.02)
                with self._t0_lock:
                    t0 = self._t0
            if t0 is None:
                t0 = self._seg_spawn_mono
        seg = _Segment(index=len(self._segments), file_name=file_name, t0=t0,
                       stat=self._sck_stat, filt=self._sck_filter,
                       size=self._sck_size)
        self._segments.append(seg)
        return seg

    def _segmented_meta_dict(self, pts_w, pts_h, geom_src):
        """meta.json for a segmented (pause/resume) take.

        Like `_multi_native_meta_dict`, a distinct shape on purpose: NO
        top-level `raw` (per-file lives in `capture_segments`, so every
        un-forked consumer fails loud rather than silently reading only segment
        0), and the session origin `t0_monotonic` is segment 0's. Mic rides
        every segment (Phase 1 keeps mic ON); facecam is a Phase-1 cut."""
        segs = []
        for s in self._segments:
            entry = {"file": s.file, "index": s.index,
                     "t0_monotonic": s.t0, "role": "screen"}
            if s.stat is not None:
                entry["capture_stats"] = {
                    "appended": int(s.stat.get("appended", 0)),
                    "duplicated": int(s.stat.get("dup", 0)),
                    "dropped": int(s.stat.get("dropped", 0)),
                    "notready": int(s.stat.get("notready", 0)),
                    "idle_filled": int(s.stat.get("idle", 0)),
                }
            segs.append(entry)
        return {
            "fps": self.fps,
            "logical_w": pts_w, "logical_h": pts_h, "geom_source": geom_src,
            "t0_monotonic": self._segments[0].t0,
            "video_index": self.video_idx, "mic_index": self.mic_idx,
            "events": os.path.basename(self.events_path),
            "cursor_mode": self.cursor_mode,
            "key_capture": self._key_capture,
            "face": None, "face_index": None, "face_fps": None,
            "face_t0_monotonic": None, "face_capture": None,
            "capture_backend": self.backend,
            "capture_segments": segs,
        }

    def _finalize_segmented(self, pts_w, pts_h, geom_src, died_early):
        """Finalize a segmented take: all-or-nothing moov gate over every
        segment, then drop the top-level `raw` (rename segment 0
        raw.mov -> seg_0.mov) and write the `capture_segments` manifest.
        Preserves the 'error.log + no meta.json = failed take' invariant."""
        failures = []
        if died_early:
            failures.append("a capture child exited during the take")
        for s in self._segments:
            path = os.path.join(self.session_dir, s.file)
            size_ok = os.path.exists(path) and os.path.getsize(path) > 1024
            moov_ok = size_ok and (self.backend != "sck" or sck.has_moov(path))
            if not moov_ok:
                failures.append("segment {} ({}) never finalized".format(
                    s.index, s.file))
        if len(self._segments) < 2:
            failures.append(
                "segmented finalize with {} segment(s)".format(
                    len(self._segments)))
        if failures:
            message = "segmented capture failed:\n  " + "\n  ".join(failures)
            self._write_error_log(message)
            raise RecordError(message)
        # Drop the top-level raw: rename segment 0's raw.mov -> seg_0.mov. The
        # writer is long reaped (finalized at the first pause), so no race.
        raw0 = os.path.join(self.session_dir, "raw.mov")
        seg0 = os.path.join(self.session_dir, "seg_0.mov")
        if os.path.exists(raw0):
            os.replace(raw0, seg0)
        self._segments[0].file = "seg_0.mov"
        meta = self._segmented_meta_dict(pts_w, pts_h, geom_src)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir

    # ---- multi-window native (P3.1) --------------------------------------

    def _multi_native_worker_cfg(self, worker):
        """The JSON config the worker will read on stdin, N-window variant.

        Same shape as `_sck_config` for a single-window native take, but
        rebuilt fresh here so this method has no dependency on `self.raw_path`
        or `self.capture_window` -- the multi path uses neither. Every worker
        differs in exactly two keys: `out` (its raw_i.mov) and
        `capture_window_id` (its target). Everything else stays identical
        across the fleet, which is what makes a rewrite that broke one worker
        catch fire in the pinned test.
        """
        cfg = {
            "out": worker.raw_path,
            "fps": int(self.fps),
            "show_cursor": self.cursor_mode != "synthetic",
            # The fleet passes no exclusions (a window-native buffer never
            # shows occluders, so the filter is inert) -- EXCEPT the 1-worker
            # fleet, which must stay behaviorally identical to the singleton
            # single-native path it replaces, and that path forwards the list
            # (equally inert, but recorded as intent in the worker config).
            "exclude": (list(self.exclude_windows) if self._fleet_single
                        else []),
            "capture_window_id": int(worker.window["id"]),
        }
        # Movie fragments: Phase C default is ON (worker's own module
        # constant). The env var overrides for A/B measurement; a value that
        # does not parse falls through to the worker default, same as
        # `_sck_config`.
        env = os.environ.get("AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC")
        if env is not None:
            try:
                cfg["movie_fragment_interval"] = float(env)
            except ValueError:
                pass
        # Audio: the mic rides ONE designated child -- channel 0, the session
        # anchor (docs/architecture.md "Audio ownership: mic on a
        # designated child (child 0)"). It MUST be exactly one worker: mic on
        # every worker would open N concurrent mic sessions, which macOS
        # refuses -- a silent capture failure for the whole fleet (pinned by
        # test_record.MultiNativeWorkerConfig). `_resolve_mic_uid` returns None
        # when no mic was requested, so a mic-less fleet is byte-identical to
        # before. Facecam is still OFF (a separate, non-orthogonal phase).
        if worker.index == 0:
            uid = self._resolve_mic_uid()
            if uid is not None:
                cfg["mic_unique_id"] = uid
        return cfg

    def _spawn_fleet(self, workers):
        """Spawn every worker of a fleet (procs + stdin config), THEN start
        their reader threads -- the same ordering the original inline block
        used, so an early startWriting failure is visible to startup logic.
        Returns (stdout_threads, stderr_threads)."""
        for worker in workers:
            worker.proc = subprocess.Popen(
                [sys.executable, _SCK_WORKER_PATH],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True)
            cfg = self._multi_native_worker_cfg(worker)
            worker.proc.stdin.write(
                (json.dumps(cfg) + "\n").encode("utf-8"))
            worker.proc.stdin.flush()
        stdout_threads, stderr_threads = [], []
        for worker in workers:
            t = threading.Thread(target=worker.read_stdout, daemon=True)
            t.start()
            stdout_threads.append(t)
            t = threading.Thread(target=worker.read_stderr, daemon=True)
            t.start()
            stderr_threads.append(t)
        return stdout_threads, stderr_threads

    def _harvest_fleet(self, workers):
        """Broadcast SIGINT to every child FIRST, then wait against ONE
        shared deadline (per the doc: not 40s x N). Idempotent on procs that
        are already reaped, so a stop-after-pause can call it again."""
        for worker in workers:
            if worker.proc is not None and worker.proc.poll() is None:
                try:
                    worker.proc.send_signal(signal.SIGINT)
                except OSError:
                    pass
        deadline = time.monotonic() + NATIVE_HARVEST_DEADLINE_SEC
        for worker in workers:
            if worker.proc is None:
                continue
            remaining = max(0.1, deadline - time.monotonic())
            try:
                worker.proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    worker.proc.kill()
                except OSError:
                    pass
                try:
                    worker.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass

    def _gate_fleet(self, workers):
        """The all-or-nothing verify over one fleet: every worker must have
        written a >1KB file with a `moov` and exited cleanly. Returns a list
        of (worker, why) failures -- empty means the fleet finalized clean."""
        failures = []
        for worker in workers:
            rc = worker.returncode()
            name = os.path.basename(worker.raw_path)
            size_ok = (os.path.exists(worker.raw_path)
                       and os.path.getsize(worker.raw_path) > 1024)
            has_moov = size_ok and worker.has_moov()
            if not size_ok:
                failures.append((worker, "no {} written (rc={})"
                                 .format(name, rc)))
            elif not has_moov:
                failures.append((worker, "{} never finalized (no moov, "
                                 "rc={})".format(name, rc)))
            elif rc not in (0, -signal.SIGINT):
                # Playable is enough. Phase C's movie fragments keep a moov'd
                # file decodable even when the worker had to be SIGKILLed
                # (losing <=~1s of tail), so an otherwise-complete take is worth
                # far more than discarding all N channels. Mirrors the shrink
                # path's moov salvage; warn so the abnormal exit is not silent.
                # The clean-stop path should never reach here now that each
                # worker unblocks its own stop signals (`_sck_worker.py`) -- a
                # blocked SIGINT was exactly what forced this SIGKILL before.
                print("note: window {} (worker {}) left with rc={} (err={!r}) "
                      "-- kept via moov salvage".format(
                          worker.window.get("id"), worker.index, rc,
                          worker.error), file=sys.stderr)
        return failures

    @staticmethod
    def _fleet_failure_message(failures):
        lines = ["multi-window native capture failed:"]
        for worker, why in failures:
            lines.append("  window {} ({}): {}".format(
                worker.window.get("id"),
                worker.window.get("app") or "?", why))
            if worker.stderr_tail:
                lines.append("    stderr: {}".format(
                    " | ".join(list(worker.stderr_tail))[:200]))
        return "\n".join(lines)

    def _wait_fleet_t0s(self, workers, timeout):
        """Block until EVERY worker of a freshly spawned scene has paired its
        first frame (t0), or timeout / child death / stop. -> True only when
        all N pairings landed. Keeping `_paused` set until this returns means
        no event is stamped into a scene whose clock origin is unknown."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._stop_requested.is_set():
                return False
            if all(w.t0 is not None for w in workers):
                return True
            if any(w.proc is not None and w.proc.poll() is not None
                   for w in workers):
                return False
            time.sleep(0.02)
        return all(w.t0 is not None for w in workers)

    def _build_scene_workers(self, scene_index, entries):
        """Fresh `_NativeWorker`s for scene `scene_index`, writing
        scene_{s}_raw_{c}.mov, each re-snapshot at spawn (live Quartz read;
        a window opened during the pause resolves here with no special
        casing -- window resolution was never start-time-only)."""
        workers = []
        for c, entry in enumerate(entries):
            worker = _NativeWorker(self.session_dir, c, entry)
            worker.raw_path = os.path.join(
                self.session_dir,
                "scene_{:d}_raw_{:d}.mov".format(scene_index, c))
            fresh = self._snapshot_window(entry)
            worker.entry = fresh
            worker.resnapshot = fresh is not None
            workers.append(worker)
        return workers

    def _finalize_active_scene(self, workers, threads):
        """Harvest + drain + gate the active fleet at a scene boundary.

        Every channel must land a moov'd file AND a first-frame t0 pairing --
        a null channel t0 fails the SCENE, never falls back to a spawn-time
        anchor: on a fleet, a spawn-time t0 shifts that channel's frame
        offset and silently misaligns every card (docs/architecture.md,
        "per-channel t0 is never null"). On success appends and returns a
        `_SceneRecord`; on failure returns None (caller fails the take).
        """
        # wall_end is stamped FIRST -- at the moment capture actually ends
        # (the SIGINT broadcast is the next thing that happens), NOT after
        # the multi-second harvest/join/gate below. The shortfall detector
        # compares `t0 + D_s` against this; stamping it late inflates every
        # measurement by the teardown latency and makes the design's #1
        # named hazard fire on every healthy pause (the review's top
        # finding -- the first gate take's warnings were exactly this bias).
        wall_end = time.monotonic()
        self._harvest_fleet(workers)
        for t in threads:
            t.join(timeout=2.0)
        for worker in workers:
            end = self._snapshot_window(worker.entry or worker.window)
            try:
                worker.end_rect = self._rect_points(end) if end else None
            except (KeyError, TypeError, ValueError):
                worker.end_rect = None
        failures = self._gate_fleet(workers)
        for worker in workers:
            if worker.t0 is None:
                failures.append((worker,
                                 "no first-frame clock pairing (t0)"))
        if failures:
            return None
        scene = _SceneRecord(index=len(self._scenes), workers=workers,
                             wall_end=wall_end)
        self._scenes.append(scene)
        return scene

    def _scene_fail_message(self, workers, scene_index):
        parts = ["scene take failed: scene {} did not finalize".format(
            scene_index)]
        failures = self._gate_fleet(workers)
        for worker in workers:
            if worker.t0 is None:
                failures.append((worker,
                                 "no first-frame clock pairing (t0)"))
        if failures:
            parts.append(self._fleet_failure_message(failures))
        return "\n".join(parts)

    def _abort_spawned_scene(self, workers, threads):
        """A resume's fleet never delivered (stop during spin-up, spawn
        failure, or T0 timeout): reap the children, drain their readers, and
        remove the frameless partial files. The take stays on its already-
        finalized scenes -- `self._sck_workers` was never swapped, so the
        fallthrough finalize sees exactly the stop-while-paused state."""
        self._harvest_fleet(workers)
        for t in threads:
            t.join(timeout=2.0)
        for worker in workers:
            try:
                os.remove(worker.raw_path)
            except OSError:
                pass

    def _scene_meta_dict(self, pts_w, pts_h, geom_src):
        """meta.json for a SCENE take (docs/architecture.md).

        A NEW top-level `capture_scenes` key -- not nested inside
        `capture_segments` (that would edit two pinned discriminators) and
        mutually exclusive with `raw` / `capture_channels` /
        `capture_segments` (the strict four-way partition). Each scene entry
        composes the two proven per-file dicts verbatim: scene-level
        index/t0/wall_end + a `channels` list of `_channel_meta` dicts. The
        scene t0 is channel 0's RAW pairing; the origin-adjusted composite
        t0 is derived at read time by `segments.scene_clock_entries` (no
        stored duration/count -- counts come from decode, the parent's
        one-source-of-truth rule).
        """
        scenes = []
        for scene in self._scenes:
            entry = {
                "index": scene.index,
                "t0_monotonic": scene.workers[0].t0,
                "channels": [self._channel_meta(w) for w in scene.workers],
            }
            if scene.wall_end is not None:
                entry["wall_end_monotonic"] = scene.wall_end
            scenes.append(entry)
        return {
            "fps": self.fps,
            "logical_w": pts_w, "logical_h": pts_h, "geom_source": geom_src,
            "t0_monotonic": scenes[0]["t0_monotonic"],
            "video_index": self.video_idx, "mic_index": None,
            "events": os.path.basename(self.events_path),
            "cursor_mode": self.cursor_mode,
            "key_capture": self._key_capture,
            "face": None, "face_index": None, "face_fps": None,
            "face_t0_monotonic": None, "face_capture": None,
            "capture_backend": self.backend,
            "capture_scenes": scenes,
        }

    def _finalize_scene_take(self, pts_w, pts_h, geom_src, died_early):
        """Finalize a >=2-scene take: verify every channel of every scene
        (moov-gated at each pause already; re-verified cheaply here), rename
        scene 0's raw_{c}.mov to scene_0_raw_{c}.mov (the N-file sibling of
        P1's raw.mov -> seg_0.mov -- writers long reaped, no race), and write
        the `capture_scenes` manifest. Preserves the 'error.log + no
        meta.json = failed take' invariant."""
        failures = []
        if died_early is not None:
            failures.append("worker {} exited during scene {} (rc={}, "
                            "err={!r})".format(died_early.index,
                                               len(self._scenes),
                                               died_early.returncode(),
                                               died_early.error))
        for scene in self._scenes:
            for worker in scene.workers:
                name = os.path.basename(worker.raw_path)
                size_ok = (os.path.exists(worker.raw_path)
                           and os.path.getsize(worker.raw_path) > 1024)
                moov_ok = size_ok and sck.has_moov(worker.raw_path)
                if not moov_ok:
                    failures.append("scene {} channel {} ({}) never "
                                    "finalized".format(scene.index,
                                                       worker.index, name))
                if worker.t0 is None:
                    failures.append("scene {} channel {} has no t0 pairing"
                                    .format(scene.index, worker.index))
        if failures:
            message = "scene take failed:\n  " + "\n  ".join(failures)
            self._write_error_log(message)
            raise RecordError(message)
        for worker in self._scenes[0].workers:
            new_path = os.path.join(
                self.session_dir,
                "scene_0_raw_{:d}.mov".format(worker.index))
            if worker.raw_path != new_path and os.path.exists(worker.raw_path):
                os.replace(worker.raw_path, new_path)
            worker.raw_path = new_path
        meta = self._scene_meta_dict(pts_w, pts_h, geom_src)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir

    def _worker_frame_count(self, worker):
        """Frames the worker wrote = its CFR slot count (incl. dup-fill),
        which equals the file's frame count and what render decodes. Prefer
        the exact count from the worker's DONE line -- `finish()` appends the
        idle tail AFTER the last <=1Hz STAT, so STAT can undercount the file
        by up to ~1s -- and fall back to STAT (DONE can be None when a slow
        pipe hasn't drained by finalize; its reader thread joins with a
        timeout). No re-probe at finalize either way."""
        if worker.done is not None:
            return int(worker.done.get("frames", 0))
        if worker.stat is not None:
            return int(worker.stat.get("appended", 0))
        return 0

    def _finalize_join_take(self, pts_w, pts_h, geom_src):
        """Finalize a seamless window-JOIN as a sub-range `capture_scenes`
        manifest (docs/architecture.md milestone 2). Unlike a pause/resume scene
        take, the survivor windows kept ONE continuous file each across every
        join, so NOTHING is renamed -- the scenes reference raw_i.mov with
        frame_start/frame_count (`_plan_join_scenes`). The caller already ran
        the all-or-nothing fleet gate (moov + t0 on every worker).

        Card-EXIT guard: a joiner that closed almost immediately after joining
        would leave its scene near-empty and, via n_out=min, could truncate its
        WHOLE scene (all the survivors). A joiner whose slice is shorter than a
        small floor is DEMOTED -- dropped from the manifest, its earlier scenes
        merged back -- rather than allowed to define a scene's length."""
        workers = self._sck_workers
        n_original = (self._n_original if self._n_original is not None
                      else len(workers) - len(self._join_marks))

        # Demote any joiner too short to carry a scene (closed right after it
        # joined). Keep survivors + the joiners that actually delivered.
        MIN_JOIN_FRAMES = max(2, int(self.fps // 4))     # ~0.25 s
        keep = list(workers[:n_original])
        for w in workers[n_original:]:
            if self._worker_frame_count(w) >= MIN_JOIN_FRAMES:
                keep.append(w)
            else:
                print("note: joined window {} delivered <{} frames -- dropped "
                      "from the take (it closed right after joining)."
                      .format(w.index, MIN_JOIN_FRAMES), file=sys.stderr)
        workers = keep

        if len(workers) <= n_original:
            # Every joiner was demoted -> no seam remains. Fall back to the
            # flat fleet manifest over the survivors (byte-identical to a take
            # that never grew).
            self._sck_workers = workers
            self._join_marks = []
            for w in workers:
                w.end_rect = self._rect_points(w.entry or w.window)
            t0 = workers[0].t0
            meta = self._multi_native_meta_dict(pts_w, pts_h, geom_src, t0)
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            return self.session_dir

        t0s = [w.t0 for w in workers]
        counts = [self._worker_frame_count(w) for w in workers]
        plan = _plan_join_scenes(t0s, counts, n_original, self.fps)

        for w in workers:
            w.end_rect = self._rect_points(w.entry or w.window)

        return self._write_fleet_scenes_meta(plan, workers, pts_w, pts_h,
                                             geom_src)

    def _write_fleet_scenes_meta(self, plan, workers, pts_w, pts_h, geom_src):
        """Write the sub-range `capture_scenes` manifest from a planner
        result (`_plan_join_scenes` / `_plan_fleet_scenes` -- the channel
        `index` in the plan is a POSITION into `workers`). Shared by the
        join and shrink finalizers so their manifests can never drift."""
        def display_key(cr):
            # Per-scene channel order IS the layout order (decision 11): a
            # rejoined card inherits its departed predecessor's rank so a
            # minimize/restore round-trip keeps its slot. Ranks default to
            # the channel index, so join-only takes sort exactly as before
            # (pinned byte-identical).
            w = workers[cr["index"]]
            return (self._display_rank.get(w.index, w.index), w.index)

        scenes = []
        for s_i, sc in enumerate(plan):
            chans = []
            for cr in sorted(sc["channels"], key=display_key):
                ch = self._channel_meta(workers[cr["index"]])
                ch["frame_start"] = int(cr["frame_start"])
                ch["frame_count"] = int(cr["frame_count"])
                # Re-anchor to the scene's seam instant so SegmentClock sees
                # ascending, aligned scenes (frame_start does the file seek).
                ch["t0_monotonic"] = float(cr["t0"])
                chans.append(ch)
            entry = {"index": s_i, "t0_monotonic": chans[0]["t0_monotonic"],
                     "channels": chans}
            if sc["end_t"] is not None:
                entry["wall_end_monotonic"] = float(sc["end_t"])
            scenes.append(entry)

        meta = {
            "fps": self.fps,
            "logical_w": pts_w, "logical_h": pts_h, "geom_source": geom_src,
            "t0_monotonic": scenes[0]["t0_monotonic"],
            # A join take CAN carry a mic (channel 0 is continuous across the
            # seam -- render muxes it). None keeps a mic-off join silent.
            "video_index": self.video_idx, "mic_index": self.mic_idx,
            "events": os.path.basename(self.events_path),
            "cursor_mode": self.cursor_mode,
            "key_capture": self._key_capture,
            "face": None, "face_index": None, "face_fps": None,
            "face_t0_monotonic": None, "face_capture": None,
            "capture_backend": self.backend,
            "capture_scenes": scenes,
        }
        if self.exclude_windows:
            meta["excluded_windows"] = list(self.exclude_windows)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir

    def _finalize_fleet_take(self, pts_w, pts_h, geom_src, start_wall):
        """Finalize a fleet take that had at least one card EXIT (M3.2) --
        and possibly joins too. The positional workers list is live
        survivors + gate-passed departed channels in channel-index order
        (originals first, then joiners in join order -- the planner's input
        contract, guaranteed by the monotone allocator); exits map each
        departed channel's POSITION to its back-dated seam.

        `_plan_fleet_scenes` owns every seam/demotion decision. A demoted
        channel (planner) or a gate-failed departure is excluded from the
        manifest -- file left on disk, stderr note. No scenes at all falls
        back to the FLAT manifest over the non-demoted survivors,
        byte-identical in shape to a take that never shrank. A single
        leftover channel down-converts through `_finalize_fleet_single`
        (render refuses a <2-channel `capture_channels` manifest -- pinned
        "need >=2"), seeding the singleton pick state from the survivor's
        own pick dict."""
        gated = [d for d in self._departed if d.gated]
        workers = sorted(self._sck_workers + [d.worker for d in gated],
                         key=lambda w: w.index)
        n_start = self._n_start if self._n_start is not None else 0
        n_original = sum(1 for w in workers if w.index < n_start)
        exits = {}
        for d in gated:
            exits[workers.index(d.worker)] = d.seam_t
        t0s = [w.t0 for w in workers]
        counts = [self._worker_frame_count(w) for w in workers]
        plan, demoted = _plan_fleet_scenes(t0s, counts, n_original, self.fps,
                                           exits=exits)
        for pos in demoted:
            print("note: window {} is not in the export (too little of it "
                  "was recorded); its file stays in the session folder."
                  .format(workers[pos].index), file=sys.stderr)
        keep = [w for pos, w in enumerate(workers) if pos not in set(demoted)]
        if not plan:
            # No seam survived: flat manifest over the non-demoted channels
            # (all near-equal t0s -- the planner demotes every joiner on
            # this path).
            self._sck_workers = keep
            for w in keep:
                if w.end_rect is None:
                    w.end_rect = self._rect_points(w.entry or w.window)
            if len(keep) == 1:
                # Render refuses a 1-channel capture_channels manifest, so
                # the lone survivor down-converts to the pinned SINGLE-native
                # shape, seeded with its own pick dict. The multi pick state
                # is cleared -- capture_window and capture_windows are
                # mutually exclusive composition contracts, and a stale
                # multi block would claim a display-crop composition this
                # file never had. The track verdict is derived the way
                # `_channel_meta` does it: `_stop_window_track` already ran
                # with no singleton pick, so without this the block says
                # track=None and render never follows window movement.
                self.capture_window = keep[0].window
                self.capture_windows = []
                if self._window_track is None:
                    try:
                        wid = int(keep[0].window["id"])
                    except (KeyError, TypeError, ValueError):
                        wid = None
                    self._window_track = ("ok" if wid in self._win_last_rect
                                          else "failed")
                return self._finalize_fleet_single(pts_w, pts_h, geom_src,
                                                   start_wall)
            t0 = keep[0].t0
            meta = self._multi_native_meta_dict(pts_w, pts_h, geom_src, t0)
            with open(self.meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            return self.session_dir
        for w in workers:
            if w.end_rect is None:
                w.end_rect = self._rect_points(w.entry or w.window)
        return self._write_fleet_scenes_meta(plan, workers, pts_w, pts_h,
                                             geom_src)

    def _alloc_channel_index(self):
        """The next channel index / raw_{i}.mov name, MONOTONE for the take.

        `len(self._sck_workers)` was the index source before cards could
        LEAVE (M3.2): after a shrink it re-issues a departed channel's index
        and a later join would collide with its file. Monotone-always is
        collision-proof; gaps (an aborted grow's removed file) are legal --
        manifest channels reference explicit file names.
        """
        if self._next_channel_idx is None:
            self._next_channel_idx = len(self._sck_workers)
        idx = self._next_channel_idx
        self._next_channel_idx += 1
        return idx

    def _respawn_watchdog(self, workers, kb_listener):
        """Re-arm the process watchdog over `workers` (+ the key child).
        Called when the fleet membership changes mid-take (a join)."""
        self._stop_watchdog()
        pids = [w.proc.pid for w in workers if w.proc is not None]
        if kb_listener is not None and getattr(kb_listener, "proc", None):
            pids.append(kb_listener.proc.pid)
        self._watchdog_proc = spawn_watchdog(pids)

    def _try_grow(self, entry, stdout_threads, stderr_threads, kb_listener,
                  notify):
        """Spawn ONE worker for a joining window and splice it into the LIVE
        fleet WITHOUT disturbing the running workers (no pause, no finalize --
        their continuous files just keep growing). Mutates `_sck_workers`, the
        thread lists and `_join_marks` on success (returns True). On any
        failure it aborts the spawned worker, restores the watchdog to the
        surviving fleet, and returns False -- a failed join is never a failed
        take; recording simply continues with the windows already in it."""
        try:
            wid = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            wid = None
        if wid is not None and (wid in self.captured_window_ids
                                or wid == self._growing_wid):
            # The recorder is AUTHORITATIVE against a DUPLICATE join. studio_app's
            # `captured_window_ids` guard is checked at request time, but a
            # joiner isn't in `_sck_workers` until the append far below (after
            # its ~2s T0 wait) -- so a manual chip RE-CLICK for the same window,
            # landing in that spawn-not-yet-appended gap with the last-wins
            # `_grow_pending` already drained to None, slips past both existing
            # guards and would spawn a SECOND worker on a window already in the
            # fleet (a cosmetic duplicate card; docs/architecture.md M3.2/M2.4 decision
            # 10, docs/architecture.md). `_growing_wid` closes the same gap
            # even if the append ever moves later than the spawn.
            #
            # Silent no-op, NOT a grow_error: the window IS recording, and
            # `on_record_state` keeps `joined`/`grow_error` mutually exclusive
            # -- an error here would MASK the first grow's standing `joined`.
            return False
        if self._ever_paused:
            # An ever-paused take finalizes through the scene manifest, which
            # ignores `_join_marks` -- accepting the worker would silently
            # drop it there (docs/architecture.md M3.0c).
            notify("recording", grow_error="can't add a window to a take "
                   "that was paused — still recording")
            return False
        if len(self._sck_workers) >= _MAX_FLEET:
            notify("recording", grow_error="already recording {} windows "
                   "(max {})".format(len(self._sck_workers), _MAX_FLEET))
            return False
        worker = _NativeWorker(self.session_dir, self._alloc_channel_index(),
                               entry)
        # Re-read through the ONE seam every other worker uses, so a joiner
        # lands in the same coordinate space as the windows already recording
        # (`_window_native_space`). studio_app resolved this entry moments ago
        # through the picker's CLIPPED read -- taking it verbatim was how a
        # partly-offscreen joiner got a rect describing less than its own
        # buffer. Falls back to the passed entry, same soft-failure posture as
        # `_resolve_capture_window`: a re-read that fails must not fail a join.
        fresh = self._snapshot_window(entry)
        worker.entry = fresh or entry
        worker.resnapshot = fresh is not None
        self._growing_wid = wid
        try:
            try:
                new_out, new_err = self._spawn_fleet([worker])
            except Exception:
                self._abort_spawned_scene([worker], [])
                notify("recording", grow_error="couldn't start the new window — "
                       "still recording")
                return False
            self._respawn_watchdog(self._sck_workers + [worker], kb_listener)
            print("[grow] joining window {} ({}); waiting up to {}s for its "
                  "first frame".format(wid, entry.get("app") or "?",
                                       GROW_T0_TIMEOUT_SEC), file=sys.stderr)
            if not self._wait_fleet_t0s([worker], GROW_T0_TIMEOUT_SEC):
                # The freshly picked window may have closed in the seconds since
                # it was validated -- OR it is on screen but visually STATIC, so
                # SCK has delivered nothing yet (docs/architecture.md). The
                # take is fine either way; the bar auto-add re-arms this wid and
                # retries while the window stays on screen (bar.js maybeRetryAutoAdd).
                print("[grow] window {} delivered no frame within {}s — join "
                      "abandoned, still recording".format(
                          wid, GROW_T0_TIMEOUT_SEC), file=sys.stderr)
                self._abort_spawned_scene([worker], new_out + new_err)
                self._respawn_watchdog(self._sck_workers, kb_listener)
                notify("recording", grow_error="the new window never delivered "
                       "a frame — still recording")
                return False
            if self._n_original is None:
                self._n_original = len(self._sck_workers)   # the survivor count
            self._sck_workers.append(worker)
            self._display_rank.setdefault(worker.index, worker.index)
            stdout_threads.extend(new_out)
            stderr_threads.extend(new_err)
            self._join_marks.append(worker.index)
            notify("recording", joined=worker.index,
                   windows=len(self._sck_workers))
            return True
        finally:
            # Cleared on every exit (append done, or abort) so the marker never
            # outlives the spawn and wedges a later legitimate re-add.
            self._growing_wid = None

    def _drain_shrink(self, kb_listener, notify):
        """Drain the shrink mailbox: clear the Event FIRST (a poller fire
        landing mid-drain re-sets it for the next tick, never lost), swap the
        pending dict out WHOLE, and retire each wid with ITS OWN back-dated
        seam. Extracted from the run loop so the drain itself is pinnable."""
        self._shrink_requested.clear()
        pending, self._shrink_pending = (self._shrink_pending or {}), None
        for wid in sorted(pending):
            self._try_shrink(wid, pending[wid], kb_listener, notify)

    def _drain_rejoin(self, stdout_threads, stderr_threads, kb_listener,
                      notify):
        """Drain the rejoin mailbox (the `_drain_shrink` discipline: clear
        the Event first, swap the pending set out whole). A wid deferred by
        a pending manual pick is simply DROPPED un-attempted: the watch
        re-arms it next tick and re-fires after a fresh stability window
        (~1s) -- deliberately NOT merged back into the mailbox, which would
        make the run loop a second `_rejoin_pending` writer racing the
        poller's own read-modify-write (adversarially caught: a poller mail
        landing mid-merge was overwritten)."""
        self._rejoin_requested.clear()
        pending, self._rejoin_pending = (self._rejoin_pending or set()), None
        for wid in sorted(pending):
            self._try_rejoin(wid, stdout_threads, stderr_threads,
                             kb_listener, notify)

    def _try_rejoin(self, wid, stdout_threads, stderr_threads, kb_listener,
                    notify, resolver=None):
        """One automatic re-add of a departed window (M3.3, decisions 10-11).

        Eligibility is decided HERE, on the run-loop thread: a pending
        MANUAL pick always wins (defer -- the grow branch drains first next
        tick, so this converges without a retry machine); everything else is
        the one attempt, marked up front so a failure falls back to the
        ordinary "+ App" chip (M3.4) instead of retrying. Rides `_try_grow`
        unchanged; on success the new worker INHERITS the departed card's
        display rank so the take's layout doesn't reshuffle.
        Returns "defer" | "joined" | "skipped" | "failed".
        """
        wid = int(wid)
        if self._grow_requested.is_set() or self._grow_pending is not None:
            return "defer"
        if len(self._sck_workers) >= _MAX_FLEET:
            # UN-burned: a cap skip neither resolved nor grew (decision 10's
            # attempt is "resolve -> grow"), and the watch's own cap filter
            # keeps it quiet until a later shrink frees a slot -- at which
            # point the window deserves its attempt (adversarially caught:
            # burning here left a still-on-screen window chip-only forever).
            return "skipped"
        self._rejoin_attempted.add(wid)
        if self._ever_paused:
            return "skipped"
        for w in self._sck_workers:
            try:
                if int((w.entry or w.window).get("id", -1)) == wid:
                    return "skipped"           # already back via a chip
            except (TypeError, ValueError):
                continue
        if resolver is None:
            resolver = (lambda w: dev.window_rect_points(
                w, exclude_pids=(os.getpid(),)))
        try:
            entry = resolver(wid)
        except Exception:
            entry = None
        if not entry:
            return "failed"                    # vanished again; chip later
        app = str(entry.get("app") or "the window")

        def auto_notify(state_name, **kw):
            # An AUTO attempt's failure must not read as a failed USER
            # action ("couldn't start the new window" is about a click the
            # user never made -- adversarially flagged). Success keeps the
            # plain `joined` -- "came back" beats "removed", deliberately.
            if kw.get("grow_error"):
                kw = dict(kw)
                kw["grow_error"] = ("couldn't automatically re-add {} -- "
                                    "use its chip to bring it back"
                                    .format(app))
            notify(state_name, **kw)

        rank = None
        for d in reversed(self._departed):
            try:
                if int((d.worker.entry or d.worker.window)["id"]) == wid:
                    rank = d.rank
                    break
            except (KeyError, TypeError, ValueError):
                continue
        if not self._try_grow(entry, stdout_threads, stderr_threads,
                              kb_listener, auto_notify):
            return "failed"
        if rank is not None:
            self._display_rank[self._sck_workers[-1].index] = rank
        return "joined"

    def _try_shrink(self, wid, seam_t, kb_listener, notify):
        """Retire ONE fleet worker whose window departed (docs/architecture.md
        M3.2): remove it from `_sck_workers` FIRST (same thread as the
        liveness poll, so once removed it can never be read as died_early),
        re-arm the watchdog over the survivors, SIGINT the child with its own
        bounded deadline, and departure-gate the file. The survivors never
        stop -- their continuous files simply span the seam, exactly like a
        join in reverse. A failed shrink is never a failed take.

        Gate: >1KB + moov + clean exit; on deadline SIGKILL then salvage on
        moov alone (movie fragments keep the file playable, losing <=~1s of
        tail the back-dated seam predates anyway). Even salvage failing just
        drops the channel from the manifest (file left on disk, stderr note)
        -- the survivors, whose files span the moment continuously, get no
        seam from it.
        """
        worker = None
        for w in self._sck_workers:
            try:
                if int((w.entry or w.window).get("id", -1)) == int(wid):
                    worker = w
                    break
            except (TypeError, ValueError):
                continue
        if worker is None or len(self._sck_workers) <= 1 or self._ever_paused:
            return False
        if self.mic_idx is not None and worker.index == 0:
            return False                       # the mic anchor never shrinks
        if dev.window_onscreen(wid) is True:
            # A restore raced the decision through the mailbox: the worker's
            # own rebuild machinery owns recovery; no seam.
            return False
        self._sck_workers.remove(worker)
        self._respawn_watchdog(self._sck_workers, kb_listener)
        # The departure record lands BEFORE the harvest, gated=False, and is
        # upgraded after the gate: an interrupt anywhere in the wait (Ctrl+C
        # lands on this thread) still leaves finalize a record to route on --
        # an unrecorded departure would make finalize treat the reduced fleet
        # as the whole take (adversarially measured: a 1-channel flat
        # manifest no dispatcher accepts, or a stale-prefix join fallback
        # that truncates every survivor to the join instant).
        dep = _DepartedChannel(
            worker, seam_t, gated=False,
            rank=self._display_rank.get(worker.index, worker.index))
        self._departed.append(dep)
        # A RE-departure (rejoined, then hidden again) earns a fresh
        # automatic attempt -- one attempt PER DEPARTURE, not per window.
        self._rejoin_attempted.discard(int(wid))
        if worker.proc is not None and worker.proc.poll() is None:
            try:
                worker.proc.send_signal(signal.SIGINT)
            except OSError:
                pass
            try:
                worker.proc.wait(timeout=SHRINK_HARVEST_DEADLINE_SEC)
            except subprocess.TimeoutExpired:
                try:
                    worker.proc.kill()
                except OSError:
                    pass
                try:
                    worker.proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        # Reader threads stay in the flat thread lists -- they idle once the
        # pipe closes and join at final finalize; the harvest wait above is
        # what lets the exact DONE count drain (M3.0b's preferred source).
        worker.end_rect = self._rect_points(worker.entry or worker.window)
        rc = worker.returncode()
        size_ok = (os.path.exists(worker.raw_path)
                   and os.path.getsize(worker.raw_path) > 1024)
        moov_ok = size_ok and worker.has_moov()
        gated = moov_ok and rc in (0, -signal.SIGINT)
        if not gated and moov_ok:
            gated = True                       # salvage: playable is enough
            print("note: window {} left with rc={} -- kept via moov salvage"
                  .format(worker.index, rc), file=sys.stderr)
        if gated:
            dep.gated = True
            self._exit_marks.append(worker.index)
            app = str((worker.entry or worker.window).get("app") or "window")
            # The one CLI line at the seam (the bar learns via the notify).
            print("note: stopped recording window {} ({}) -- it left the "
                  "screen; the other windows keep going."
                  .format(worker.index, app), file=sys.stderr)
            notify("recording", departed=worker.index, departed_app=app,
                   windows=len(self._sck_workers))
        else:
            print("note: window {} left the screen but its file never "
                  "finalized (rc={}) -- dropped from the take; the other "
                  "windows keep recording.".format(worker.index, rc),
                  file=sys.stderr)
            notify("recording", windows=len(self._sck_workers))
        return True

    def _finalize_fleet_single(self, pts_w, pts_h, geom_src, start_wall):
        """Down-convert a never-paused (or 1-scene-collapsed) 1-worker fleet
        to the pinned SINGLE-native shape: rename raw_0.mov -> raw.mov,
        restore the singleton view of the take (capture_window block, SCK
        size/stat/filter, window track verdict, t0), and write the ordinary
        `_meta_dict`. The fleet routing exists so the take could have paused;
        when it never did, the on-disk contract must be byte-compatible with
        the single-native path it replaced."""
        worker = self._sck_workers[0]
        raw = os.path.join(self.session_dir, "raw.mov")
        if worker.raw_path != raw and os.path.exists(worker.raw_path):
            os.replace(worker.raw_path, raw)
        self.raw_path = raw
        # capture_window stayed set for the 1-worker fleet (see __init__), so
        # `_capture_window_meta` and the `_stop_window_track` "ok"/"failed"
        # verdict ran unchanged; only the worker-held snapshots move over.
        self._cw_entry = worker.entry
        self._cw_resnapshot = worker.resnapshot
        # end_rect must be a LIVE snapshot, exactly like the singleton
        # `_finalize`'s -- the worker-held rect is the spawn-time one, and
        # copying it would make the "window moved during the take" warning
        # structurally impossible on every single occlusion-free take
        # (render compares end_rect against the start rect).
        end = self._snapshot_window(worker.entry or worker.window)
        try:
            self._cw_end_rect = self._rect_points(end) if end else None
        except (KeyError, TypeError, ValueError):
            self._cw_end_rect = None
        self._sck_size = worker.size
        self._sck_stat = worker.stat
        self._sck_filter = worker.filter
        with self._t0_lock:
            self._t0 = worker.t0
        self._check_cfr()
        t0 = worker.t0 if worker.t0 is not None else start_wall
        meta = self._meta_dict(pts_w, pts_h, geom_src, t0, face_ok=False)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir

    def _start_multi_native(self, countdown=3, duration=None, status_cb=None):
        """The N-window occlusion-free capture loop (P3.1).

        A strict fan-out of the single-window SCK path: N `_sck_worker.py`
        children, N raw_i.mov files, N clock pairings, one shared events file
        and one manifest. Written as a self-contained method (rather than an
        `if multi:` branch inside `start()`) so the single-window and
        avfoundation code paths above are provably unchanged -- no test on
        that code has to be re-run for confidence, only the ones this method
        owns.

        v1 scope, from docs/architecture.md's phasing:
        - static composite is P3.2, not here -- this method's job ends when
          N clean raw_i.mov files + manifest land on disk.
        - facecam and mic are OFF (doc: both are 'not orthogonal' and get
          their own phases).
        - all-or-nothing finalize: any child missing `moov` (via `sck.has_moov`
          -- Phase C guarantees a clean stop lands one) -> whole take fails
          with error.log and no meta.json, exactly the invariant single-window
          takes already carry.
        - teardown ordering per the doc: broadcast SIGINT to every child FIRST
          then harvest against ONE shared 40s deadline, not 40 * N.
        """
        from pynput import mouse

        def notify(state, **extra):
            if status_cb is None:
                return
            try:
                status_cb(state, extra)
            except Exception:
                pass

        # Facecam is still a doc-driven scope cut (a separate, non-orthogonal
        # phase), made loud so a caller who hoped for it does not silently get
        # a video-only take. Mic IS supported now -- it rides channel 0
        # (`_multi_native_worker_cfg`) and render muxes it from raw_0.mov.
        if self.face_idx is not None:
            print("note: facecam is not yet supported on multi-window native "
                  "takes; recording without face.mov.", file=sys.stderr)

        self._stop_requested.clear()
        os.makedirs(self.session_dir, exist_ok=True)
        pts_w, pts_h, geom_src = dev.main_display_points()

        try:
            for i in range(countdown, 0, -1):
                if self._stop_requested.is_set():
                    print("\nrecording stopped before start.")
                    return None
                notify("countdown", seconds=i)
                print("  recording in {}...".format(i), end="\r", flush=True)
                time.sleep(1)
            print(" " * 40, end="\r")
        except KeyboardInterrupt:
            print("\naborted before recording started.")
            return None

        # Per-worker rect re-snapshot: the countdown is when the user brings
        # each target forward, so a pick-time-only rect would be useless. A
        # failed re-read falls back to the pick-time entry -- same discipline
        # as `_resolve_capture_window` for the single-window path.
        for worker in self._sck_workers:
            fresh = self._snapshot_window(worker.window)
            worker.entry = fresh
            worker.resnapshot = fresh is not None
        missing = sum(1 for w in self._sck_workers if not w.resnapshot)
        if missing:
            print("  {} of {} windows not found — using the rects from when "
                  "you picked them.".format(missing, len(self._sck_workers)))

        self._ev_file = open(self.events_path, "w")
        listener = None
        kb_listener = None
        stdout_threads = []
        stderr_threads = []
        died_early = None
        scene_fail_msg = None
        start_wall = time.monotonic()
        try:
            # Spawn every worker BEFORE opening the readers -- an early
            # startWriting failure on one worker should be visible to the
            # startup logic, not laundered into an "everyone up, one weird
            # child" state. (_spawn_fleet preserves that ordering; it is the
            # same code, extracted so a scene resume can spawn a fresh fleet.)
            stdout_threads, stderr_threads = self._spawn_fleet(self._sck_workers)

            # Input listeners + geometry poller -- same as single-window, and
            # the poller ALREADY samples every on-screen window (see
            # `_poll_window_geometry`), so N-window native gets its geometry
            # track for free.
            listener = mouse.Listener(**self._mouse_listener_kwargs())
            listener.start()
            self._start_window_track()
            if self.log_keys:
                kb_listener, self._key_capture = self._start_key_listener()
            else:
                self._key_capture = "disabled"

            # Watchdog covers every SCK worker's pid PLUS the key worker --
            # the singleton path always covered it, and the N=1 occlusion-
            # free routing moved that config onto this loop, so dropping it
            # here would orphan the key worker on a parent SIGKILL. A parent
            # SIGKILL leaves exactly one deadlined child per raw_i.mov,
            # which C's fragments keep playable.
            pids = [w.proc.pid for w in self._sck_workers]
            if kb_listener is not None and getattr(kb_listener, "proc", None):
                pids.append(kb_listener.proc.pid)
            self._watchdog_proc = spawn_watchdog(pids)

            n = len(self._sck_workers)
            msg = ("recording (occlusion-free, {} window{}) — press Ctrl+C "
                   "to stop".format(n, "s" if n != 1 else ""))
            if duration:
                msg += "  (auto-stop {:g}s)".format(duration)
            print(msg)
            notify("recording")

            # Run loop with pause / re-pick / resume (scene takes). A take
            # that is never paused never leaves the `recording` state and
            # `_scenes` stays empty -- the fleet's single-manifest success
            # path below then runs byte-identically (the same off-switch
            # trick the P1 whole-screen loop uses). pause()/resume() set
            # request Events from the HTTP/bar thread; the loop does the
            # finalize/spawn so those endpoints return immediately.
            state = "recording"
            while True:
                if self._stop_requested.is_set():
                    notify("stopping")
                    if state == "recording" and self._ever_paused:
                        # The active fleet becomes the take's last scene --
                        # gate it NOW (fail-early), like the P1 stop branch.
                        scene = self._finalize_active_scene(
                            self._sck_workers,
                            stdout_threads + stderr_threads)
                        if scene is None:
                            scene_fail_msg = self._scene_fail_message(
                                self._sck_workers, len(self._scenes))
                    break
                if state == "recording":
                    if duration and (time.monotonic() - start_wall
                                     - self._paused_accum) >= duration:
                        if self._ever_paused:
                            scene = self._finalize_active_scene(
                                self._sck_workers,
                                stdout_threads + stderr_threads)
                            if scene is None:
                                scene_fail_msg = self._scene_fail_message(
                                    self._sck_workers, len(self._scenes))
                        break
                    if self._pause_requested.is_set():
                        self._pause_requested.clear()
                        notify("pausing")
                        # Gate the event writers FIRST (the P1 ordering,
                        # pinned): the finalize below takes seconds during
                        # which the fleet produces no frames -- input logged
                        # in that window would clamp onto the seam as a
                        # phantom cluster.
                        self._paused.set()
                        scene = self._finalize_active_scene(
                            self._sck_workers,
                            stdout_threads + stderr_threads)
                        if scene is None:
                            # A scene that cannot finalize is a failed take,
                            # caught AT THE PAUSE (fail-early, cheaper than
                            # discovering it at stop after more recording).
                            scene_fail_msg = self._scene_fail_message(
                                self._sck_workers, len(self._scenes))
                            break
                        self._ever_paused = True
                        # Stale-replay fix: a resume that raced the pause (or
                        # a duplicate click during `resuming`) must not fire a
                        # stale window set at the NEXT pause. Same for a grow
                        # that raced the pause -- replaying it after resume
                        # would join a window the scene finalize then drops.
                        self._resume_requested.clear()
                        self._scene_pending = None
                        self._grow_requested.clear()
                        self._grow_pending = None
                        # And the shrink + rejoin mailboxes (M3.2/M3.3): an
                        # exit or re-add armed just before the pause must not
                        # replay after resume -- the scene machine has no
                        # exit support (the rejoin replay would drain to a
                        # guarded no-op anyway; cleared for the same stale-
                        # replay hygiene as the others, not by necessity).
                        self._shrink_requested.clear()
                        self._shrink_pending = None
                        self._rejoin_requested.clear()
                        self._rejoin_pending = None
                        self._pause_started = time.monotonic()
                        state = "paused"
                        notify("paused", segments=len(self._scenes))
                        continue
                    if self._grow_requested.is_set():
                        # Seamless window-JOIN: splice in one new worker WITHOUT
                        # pausing -- the running workers never stop, so their
                        # files span the seam. A failed join keeps recording.
                        self._grow_requested.clear()
                        entry, self._grow_pending = self._grow_pending, None
                        if entry is not None:
                            self._try_grow(entry, stdout_threads,
                                           stderr_threads, kb_listener, notify)
                        continue
                    if self._shrink_requested.is_set():
                        # Card SHRINK (docs/architecture.md M3.2): a captured window
                        # was hidden AND delivery-dead past the debounce. Its
                        # worker dies at a back-dated seam; survivors never
                        # stop. Drained AFTER grow, BEFORE liveness (the tick
                        # ordering: stop > duration > pause > grow > shrink >
                        # liveness), one transition per tick.
                        self._drain_shrink(kb_listener, notify)
                        continue
                    if self._rejoin_requested.is_set():
                        # Auto-REJOIN (M3.3): a departed window is stably
                        # back on-screen -- re-add it as a NEW card riding
                        # _try_grow. After grow and shrink in the tick
                        # ordering; a failed rejoin is a failed grow, never
                        # a failed take.
                        self._drain_rejoin(stdout_threads, stderr_threads,
                                           kb_listener, notify)
                        continue
                    note = self._anchor_note_pending
                    if note is not None:
                        # The mic anchor's card froze / un-froze (decision
                        # 6) -- surface or clear the hint. Not a fleet
                        # transition, so no `continue`.
                        self._anchor_note_pending = None
                        kind, note_app = note
                        if kind == "hidden":
                            notify("recording", mic_anchor_hidden=note_app)
                        else:
                            notify("recording", mic_anchor_seen=True)
                    # Mid-take liveness detection (doc: "a dead child must be
                    # noticed *during* the take, not discovered at finalize
                    # after minutes of recording"). v1 fails the whole take on
                    # any child exit -- the surviving-windows-render policy
                    # improvement rides on a later phase. A SHRUNK worker was
                    # removed from `_sck_workers` before its SIGINT (same
                    # thread as this poll -- ordering, not a race), so a
                    # departed card can never be read as died_early.
                    for worker in self._sck_workers:
                        if worker.proc.poll() is not None:
                            died_early = worker
                            break
                    if died_early is not None:
                        break
                elif self._resume_requested.is_set():
                    self._resume_requested.clear()
                    pending, self._scene_pending = self._scene_pending, None
                    notify("resuming")
                    # Re-pick: the payload's window entries, else the previous
                    # scene's picks (re-snapshot fresh either way -- window
                    # resolution is live, so a window opened during the pause
                    # resolves with no special casing).
                    entries = pending or [w.window
                                          for w in self._scenes[-1].workers]
                    new_workers = self._build_scene_workers(
                        len(self._scenes), entries)
                    try:
                        new_threads = self._spawn_fleet(new_workers)
                    except Exception:
                        self._abort_spawned_scene(new_workers, [])
                        notify("paused", segments=len(self._scenes),
                               error="couldn't restart the capture — "
                                     "still paused")
                        continue
                    self._stop_watchdog()
                    new_pids = [w.proc.pid for w in new_workers]
                    if (kb_listener is not None
                            and getattr(kb_listener, "proc", None)):
                        new_pids.append(kb_listener.proc.pid)
                    self._watchdog_proc = spawn_watchdog(new_pids)
                    if not self._wait_fleet_t0s(new_workers,
                                                SEGMENT_T0_TIMEOUT_SEC):
                        # Reap and discard the frameless fleet either way.
                        self._abort_spawned_scene(
                            new_workers, new_threads[0] + new_threads[1])
                        if self._stop_requested.is_set():
                            # User stop during spin-up: the top-of-loop stop
                            # check finalizes the take from the scenes
                            # already on disk, exactly stop-while-paused.
                            continue
                        # A resume-spawn failure returns the take to
                        # `paused` -- NEVER take-death: the spawned thing is
                        # a freshly picked window that may have closed in
                        # the seconds since validation, and the finalized
                        # scenes on disk are worth more than the retry
                        # (docs/architecture.md, failure policy).
                        notify("paused", segments=len(self._scenes),
                               error="the new windows never delivered a "
                                     "frame — still paused")
                        continue
                    if self._pause_started is not None:
                        self._paused_accum += (time.monotonic()
                                               - self._pause_started)
                        self._pause_started = None
                    self._sck_workers = new_workers
                    stdout_threads, stderr_threads = new_threads
                    self._paused.clear()
                    state = "recording"
                    notify("recording")
                time.sleep(0.05)
        except KeyboardInterrupt:
            # A Ctrl+C on an ever-paused take must finalize the ACTIVE scene
            # like a stop would -- falling straight through would write a
            # manifest that silently omits everything since the last resume
            # (its files orphaned on disk, its events clamping to the final
            # seam). Unreachable via the HTTP surfaces (they stop()), so
            # this covers only a direct SIGINT to the recording process.
            if state == "recording" and self._ever_paused:
                scene = self._finalize_active_scene(
                    self._sck_workers, stdout_threads + stderr_threads)
                if scene is None:
                    scene_fail_msg = self._scene_fail_message(
                        self._sck_workers, len(self._scenes))
        finally:
            if listener is not None:
                listener.stop()
            if kb_listener is not None:
                self._settle_key_capture(kb_listener)
                try:
                    kb_listener.stop()
                except Exception:
                    pass
            self._stop_window_track()
            # Teardown: broadcast SIGINT to EVERY child first, then wait
            # against ONE shared deadline (per doc: not 40s * N). On a take
            # stopped while paused, the current fleet was already harvested at
            # the pause -- `_harvest_fleet` is idempotent on reaped procs.
            # Departed (shrunk) workers ride along for the same idempotent
            # reason: normally long reaped, but a Ctrl+C that interrupted a
            # shrink's own harvest wait must not orphan the departing child.
            self._harvest_fleet(self._sck_workers
                                + [d.worker for d in self._departed])
            for t in stdout_threads + stderr_threads:
                t.join(timeout=2.0)
            if self._ev_file is not None:
                try:
                    self._ev_file.close()
                except OSError:
                    pass
                self._ev_file = None
            self._stop_watchdog()

        # A scene that failed its finalize gate (at pause, or at stop on a
        # resumed take) fails the whole take -- error.log, no meta.json.
        if scene_fail_msg is not None:
            self._write_error_log(scene_fail_msg)
            raise RecordError(scene_fail_msg)

        # Scene take: >=2 finalized scenes write the `capture_scenes`
        # manifest. Exactly ONE scene (pause-then-stop-while-paused, or every
        # later resume aborted) collapses to the legacy shape below --
        # content-identical to a never-paused take, and the fully supported
        # read model is worth more than a degenerate 1-scene manifest.
        if len(self._scenes) >= 2:
            return self._finalize_scene_take(pts_w, pts_h, geom_src,
                                             died_early)
        if self._scenes:
            self._sck_workers = self._scenes[0].workers

        # Per-worker end-rect (drift note in the manifest, mirrors the
        # single-window `end_rect`).
        for worker in self._sck_workers:
            worker.end_rect = (self._rect_points(worker.entry
                                                 or worker.window))

        # All-or-nothing finalize gate. Every worker must have:
        # - written a raw_i.mov > 1 KB (frames actually flowed)
        # - the raw_i.mov must have a `moov` (Phase C guarantees clean stop)
        # A single failure fails the whole take, mirroring the invariant on
        # the single-window path (error.log + no meta.json).
        failures = self._gate_fleet(self._sck_workers)
        if died_early is not None:
            failures.insert(0, (died_early,
                                "worker {} exited during the take (rc={}, "
                                "err={!r})".format(died_early.index,
                                                   died_early.returncode(),
                                                   died_early.error)))
        if failures:
            message = self._fleet_failure_message(failures)
            self._write_error_log(message)
            raise RecordError(message)

        # Card SHRINK (M3.2): any departure -- gate-passed OR failed --
        # routes to the generalized fleet finalizer BEFORE the fleet-single
        # down-conversion below: a 1-window take that grew and then shrank
        # back to one card still has a departed channel whose footage and
        # seams only _finalize_fleet_take can express (adversarially
        # measured: the down-convert first silently discarded a fully-gated
        # departed original).
        if self._exit_marks or self._departed:
            return self._finalize_fleet_take(pts_w, pts_h, geom_src,
                                             start_wall)

        # A single occlusion-free pick that never became a scene take writes
        # the pinned SINGLE-native shape (raw.mov + capture_window block) --
        # the fleet-of-1 is an implementation detail the on-disk contract
        # must not leak.
        if self._fleet_single and len(self._sck_workers) == 1:
            return self._finalize_fleet_single(pts_w, pts_h, geom_src,
                                               start_wall)

        # Session timeline zero: v1 uses channel 0's t0 (per the doc, "pin
        # the output origin to the audio-bearing stream's t0" -- child 0 is
        # the designated audio owner when audio lands, and today's audio-
        # absent v1 keeps the same anchor so the future add is a no-op on
        # the shape). Fall back to start_wall if T0 never arrived (shouldn't
        # happen once has_moov passed above, but keep it defensive).
        t0 = self._sck_workers[0].t0
        if t0 is None:
            t0 = start_wall

        # Per-worker CFR check. Report-only (never fails a take, mirroring
        # the single-window path). Uses the same verify_cfr helper.
        for worker in self._sck_workers:
            try:
                sck.verify_cfr(worker.raw_path, self.fps)
            except Exception:
                pass

        # Seamless window-JOIN: the fleet grew mid-take, so the survivors'
        # continuous files are sub-ranged into a K+1-scene `capture_scenes`
        # manifest (`_finalize_join_take`). Every worker already passed the
        # all-or-nothing gate above. No joins -> the flat manifest below,
        # byte-identical.
        if self._join_marks:
            return self._finalize_join_take(pts_w, pts_h, geom_src)

        meta = self._multi_native_meta_dict(pts_w, pts_h, geom_src, t0)
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        return self.session_dir

    def _multi_native_meta_dict(self, pts_w, pts_h, geom_src, t0):
        """meta.json for a multi-window native take (P3.1).

        A DIFFERENT shape from the single-window meta on purpose: no `raw`
        key (per-file lives in the manifest), no `capture_window` /
        `capture_windows` (both imply a single-file-with-crop story that
        does not apply here), and the new `capture_channels` list is the
        source of truth. render (P3.2) branches on presence of
        `capture_channels`; every earlier consumer keys off keys the
        single-window path is still the sole writer of, so nothing prior
        to P3.2 changes behavior on a legacy take.
        """
        meta = {
            "fps": self.fps,
            "logical_w": pts_w, "logical_h": pts_h, "geom_source": geom_src,
            "t0_monotonic": t0,
            # `mic_index` records that the mic was requested; the audio track
            # itself lives inside channel 0's raw_0.mov (the designated owner),
            # which render probes directly. None (mic off) keeps the manifest
            # byte-identical to before.
            "video_index": self.video_idx, "mic_index": self.mic_idx,
            # Manifest v1 stores events + cursor mode at the session level;
            # per-channel `events` would only be needed if input mapping
            # differed per card, which it does not (one shared click set).
            "events": os.path.basename(self.events_path),
            "cursor_mode": self.cursor_mode,
            "key_capture": self._key_capture,
            "face": None, "face_index": None, "face_fps": None,
            "face_t0_monotonic": None, "face_capture": None,
            "capture_backend": self.backend,
            "capture_channels": [
                self._channel_meta(w) for w in self._sck_workers],
        }
        if self.exclude_windows:
            meta["excluded_windows"] = list(self.exclude_windows)
        return meta

    def _channel_meta(self, worker):
        """One entry of `capture_channels`. Mirrors the single-window
        `_capture_window_meta` shape for its `capture_window`-family keys, so
        a future render seam can consume either without a second parser."""
        src = worker.entry or worker.window
        origin = src.get("display_origin") or (0.0, 0.0)
        channel = {
            "role": "screen_window",
            "file": os.path.basename(worker.raw_path),
            "mode": "window_native",
            "id": int(src["id"]),
            "app": src.get("app"),
            "title": src.get("title"),
            "units": "points",
            "rect": self._rect_points(src),
            "display_origin": [float(origin[0]), float(origin[1])],
            "source": "quartz",
            "resnapshot": bool(worker.resnapshot),
            "end_rect": worker.end_rect,
            # Per-channel geometry track verdict, mirroring the single-window
            # `track` on capture_window: "ok" iff the shared geometry poller
            # actually logged a sample for this window's id (the poller runs
            # once and samples every on-screen window, so per-channel
            # verdicts fall out of one shared `_win_last_rect`). "failed"
            # keeps the same tri-state semantics render.py already knows.
            "track": ("ok" if int(src["id"]) in self._win_last_rect
                      else "failed"),
            # Per-window logical size = the WINDOW's own point dims, exactly
            # as the single-window native meta stores them.
            "logical_w": float(src["w"]),
            "logical_h": float(src["h"]),
            # Per-channel clock pairing, our own monotonic domain. Session-
            # level `t0_monotonic` above is anchored to channel 0.
            "t0_monotonic": worker.t0,
        }
        if worker.size is not None:
            channel["buffer_w"] = int(worker.size.get("width", 0))
            channel["buffer_h"] = int(worker.size.get("height", 0))
        if worker.stat is not None:
            channel["capture_stats"] = {
                "appended": int(worker.stat.get("appended", 0)),
                "duplicated": int(worker.stat.get("dup", 0)),
                "dropped": int(worker.stat.get("dropped", 0)),
                "notready": int(worker.stat.get("notready", 0)),
                # See the single-window note in `_capture_window_meta`.
                "idle_filled": int(worker.stat.get("idle", 0)),
            }
        return channel

    @staticmethod
    def _screen_denied_signature(tail):
        """True when ffmpeg's avfoundation couldn't configure the screen input
        and fell back to a camera -- the fingerprint of a denied/wedged Screen
        Recording grant. The screen device advertises camera-only pixel formats
        (uyvy422 etc.) only after that fallback, so it's an unambiguous tell."""
        low = (tail or "").lower()
        return ("configuration of video device failed" in low
                or ("falling back to default" in low and "uyvy422" in low))

    def _check_cfr(self):
        """Measure whether the file we just wrote is really CFR, and say so.

        Runs on EVERY backend, including today's ffmpeg one. That path forces
        CFR with `-r` and has never been observed to violate it — but nothing
        checked, and the assumption is load-bearing far downstream: `render`
        and `retime.TimeMap` turn frame index into time by dividing by fps,
        and they read that fps from cv2 rather than from `meta["fps"]`. A VFR
        file doesn't fail; it silently stretches time, so a zoom drifts off
        its click and the editor preview stops matching the export.

        Deliberately NEVER fails the take. The recording exists and is
        watchable; a check that has never run in the wild does not get to
        throw one away on its first day. It warns, and stores the verdict for
        `meta` on the backends that carry it.
        """
        try:
            status, detail = sck.verify_cfr(self.raw_path, self.fps)
        except Exception as exc:                # never let a probe kill a take
            status, detail = "unknown", "cfr check failed: {!r}".format(exc)
        self._cfr = status
        self._cfr_detail = detail
        if status == "suspect":
            print("warning: {} is not constant frame rate ({}). Zoom timing "
                  "may drift.".format(os.path.basename(self.raw_path), detail),
                  file=sys.stderr)
        return status

    def _sck_failure_message(self, rc, raw_ok):
        """Why an SCK take produced nothing usable.

        A separate function because the ffmpeg diagnosis CANNOT be reused:
        `_screen_denied_signature` works by string-matching ffmpeg's stderr
        for the camera-fallback pixel formats, and not one of those strings
        exists on this path. Reusing it would leave a denied SCK take with
        the generic message — which is the silent-failure shape this codebase
        already decided it does not accept (see the RecordError contract in
        the module docstring).

        The worker's own `ERR <domain> <detail>` line is the primary source:
        it knows whether it failed at content, filter, writer or start, which
        is more than an exit code can say.
        """
        parts = ["ScreenCaptureKit capture failed"]
        if rc is not None:
            parts[0] += " (exit code {})".format(rc)
        if self._sck_error:
            parts.append("worker reported: " + self._sck_error)
        tail = "".join(str(t) for t in self._stderr_tail).strip()
        if tail:
            parts.append("--- capture worker output (tail) ---\n" + tail)
        low = ((self._sck_error or "") + " " + tail).lower()
        if not raw_ok and ("tcc" in low or "declined" in low
                           or "not authorized" in low or "permission" in low
                           or "start " in low):
            parts.append(
                "This looks like the Screen Recording permission. Unlike the\n"
                "ffmpeg path, ScreenCaptureKit reports it as a typed error\n"
                "rather than by quietly recording a camera, so:\n"
                "  1. Grant Screen Recording to THIS app in System Settings >\n"
                "     Privacy & Security > Screen Recording.\n"
                "  2. FULLY QUIT and relaunch it -- the grant only takes\n"
                "     effect on relaunch.\n"
                "  3. If it is already granted and still fails, toggle it\n"
                "     off/on, quit, and reboot if needed.\n"
                "Or record with the other backend: --capture-backend "
                "avfoundation.")
        elif rc == -6 or rc == 134:
            # The hazard this worker exists to contain, reported plainly
            # rather than as a mystery exit code.
            parts.append(
                "The capture worker was aborted (SIGABRT). ScreenCaptureKit\n"
                "does this when an error surfaces inside its XPC reply, and\n"
                "it kills only this helper -- the app itself is unaffected.\n"
                "Please report the worker output above; --capture-backend\n"
                "avfoundation records in the meantime.")
        return "\n\n".join(parts)

    def _sck_unfinalized_message(self, rc):
        """Why an SCK take captured frames but produced an unplayable file.

        Distinct from `_sck_failure_message`: capture SUCCEEDED here -- the
        frames are in raw.mov -- but the `moov` index that `finishWriting`
        appends never landed, so every consumer reports `moov atom not found`.
        This is the `AVAssetWriter` tail-loss risk (docs/architecture.md,
        "we lose the tail"): stop the worker before
        `finishWritingWithCompletionHandler_` completes and the tail is gone
        even though the frames are not. We catch it rather than write a
        meta.json that files a corrupt take as a good project. A distinct
        message matters -- "capture failed" would send someone to check the
        Screen Recording grant that plainly worked.
        """
        parts = ["ScreenCaptureKit captured the recording but did not "
                 "finalize it"]
        if rc is not None:
            parts[0] += " (worker exit code {})".format(rc)
        appended = None
        if self._sck_stat is not None:
            try:
                appended = int(self._sck_stat.get("appended"))
            except (TypeError, ValueError):
                appended = None
        if appended:
            parts.append(
                "{} frames were captured, but the file's `moov` index -- "
                "written when the writer finalizes -- is missing, so it "
                "cannot be played or rendered.".format(appended))
        else:
            parts.append(
                "The file's `moov` index -- written when the writer finalizes "
                "-- is missing, so it cannot be played or rendered.")
        if self._sck_done is None:
            parts.append(
                "The capture worker never reported DONE: it was stopped before "
                "it finished writing the movie tail. That is usually the app "
                "being force-quit or crashing during the stop; a normal stop "
                "gives the worker the time it needs to finalize.")
        tail = "".join(str(t) for t in self._stderr_tail).strip()
        if tail:
            parts.append("--- capture worker output (tail) ---\n" + tail)
        return "\n\n".join(parts)

    def _sck_config(self):
        """The JSON config handed to `_sck_worker.py` on its stdin.

        stdin rather than argv because argv is visible in `ps` to every
        process on the machine, and this carries the output path and the list
        of windows the user is hiding.
        """
        cfg = {
            "out": self.raw_path,
            "fps": int(self.fps),
            "show_cursor": self.cursor_mode != "synthetic",
            "exclude": list(self.exclude_windows),
        }
        # Movie-fragment cadence, seconds. Present unless the env var is set
        # to a non-numeric string, in which case we omit the key so the worker
        # falls back to its module default (still on). Setting the env var
        # to 0 makes the worker skip `setMovieFragmentInterval_` entirely --
        # the byte-exact off switch that reproduces pre-C monolithic-file
        # layout. Resolved parent-side so its value is testable without
        # importing PyObjC or opening a real writer.
        env = os.environ.get("AUTOCINE_MOVIE_FRAGMENT_INTERVAL_SEC")
        if env is not None:
            try:
                cfg["movie_fragment_interval"] = float(env)
            except ValueError:
                pass    # ignore garbage; worker default fires
        if self._is_window_native():
            # The one field that flips the worker from display-capture to
            # capturing THIS window's own buffer. Absent on every other take,
            # so a non-native SCK config is byte-identical to before.
            cfg["capture_window_id"] = int(self.capture_window["id"])
        uid = self._resolve_mic_uid()
        if uid is not None:
            cfg["mic_unique_id"] = uid
        return cfg

    def _resolve_mic_uid(self):
        """The AVFoundation uniqueID string for `self.mic_idx`, or None.

        Resolved HERE, not in the worker: SCK wants an AVFoundation uniqueID
        string while everything else in this app speaks the avfoundation
        ordinal, and the ordinal must be translated while we can still
        cross-check it against the name ffmpeg reported. None means either no
        mic was requested OR we could not resolve it SAFELY -- both treated as
        no microphone, because recording the WRONG input is the failure that
        matters, and this app already warns that these indices shift whenever
        a device connects.

        Shared by the single-window `_sck_config` and the multi-native fleet's
        channel 0 (`_multi_native_worker_cfg`) so both resolve identically and
        the mic can only ever land on ONE input.
        """
        if self.mic_idx is None:
            return None
        names = dict(dev.list_avf_devices().get("audio", []))
        uid = dev.mic_unique_id(self.mic_idx, names.get(self.mic_idx))
        if uid is None:
            print("warning: audio index {} could not be matched to a "
                  "microphone; recording without audio"
                  .format(self.mic_idx), file=sys.stderr)
        return uid

    # How often the parent re-reads which windows must stay out of the take.
    # 2 Hz: fast enough that a bubble popped out mid-take is hidden within
    # half a second, slow enough that a CGWindowList sweep is free next to
    # encoding. Only CHANGES are sent, so a still desktop costs one list
    # walk and no IPC at all.
    EXCLUDE_POLL_SEC = 0.5

    def _poll_exclusions(self):
        """Keep the worker's content filter in step with our own chrome.

        A SAFETY NET, and worth being honest about how narrow it is. This was
        built for "the facecam bubble pops out mid-take, the picker opens",
        and that scenario does not exist:

          * the recording face of the bar is a stop button, a timer and a
            label — there is no control to open either one (`bar.html`);
          * `floatFacecam()` is reachable only from the idle face, so a
            floating bubble is already floating before the take starts;
          * and both windows are CREATED ONCE AND HIDDEN, not made on demand
            (observed: 'Facecam' and 'Pick Windows' sit in the window list
            with stable ids while off screen).

        That third bullet used to end "...so `capture_exclusions` reports them
        whether or not they are visible." It does not: nothing calls
        `capture_exclusions`, so hidden windows are reported by no one. What
        the provider actually returns is the on-screen pid sweep alone, which
        by construction cannot see an ordered-out window — so a hidden window
        is covered not by being pre-reported but by this poll noticing it
        within 500 ms of it appearing. Which, for a window that is hidden,
        is the same outcome: it is not in the frames either way.

        What is left is real but small: a window that neither was on screen
        when the take began nor is owned by a pid we know — a macOS tooltip
        over the stop button, a WebKit service window, or whatever a future
        UI adds. The on-screen pid sweep catches the ones we own on the next
        poll; the ones we do not own it never sees at all.

        Kept because it is cheap (one window-list walk per 500 ms, IPC only
        when the set changes) and because the failure it guards against is
        our own chrome burned into a user's recording, which is the whole
        reason this backend exists.

        Everything here fails toward "keep recording": a provider that raises,
        a closed pipe, a dead worker — all just end the poller. The take is
        worth more than the exclusion.
        """
        last = list(self.exclude_windows)
        while not self._stop_requested.is_set():
            if self._stop_exclude_poll.wait(self.EXCLUDE_POLL_SEC):
                return
            proc = self.proc
            if proc is None or proc.poll() is not None or proc.stdin is None:
                # Segmented pause: the child is legitimately dead for the
                # whole gap and a fresh one appears at resume -- idle through
                # it (the next tick re-reads self.proc) instead of exiting
                # for good, or every post-resume chrome change would be
                # burned into the footage. A child dying OUTSIDE a pause
                # still just ends the poller, as before.
                if self._paused.is_set() or self._pause_requested.is_set():
                    continue
                return
            try:
                ids = [int(w) for w in (self._exclude_provider() or [])]
            except Exception:
                continue
            if ids == last:
                continue
            try:
                proc.stdin.write(
                    ("EXCLUDE " + " ".join(str(i) for i in ids) + "\n")
                    .encode("utf-8"))
                proc.stdin.flush()
            except Exception:
                if self._paused.is_set() or self._pause_requested.is_set():
                    continue
                return
            last = ids
            self.exclude_windows = ids

    def _read_sck_stdout(self):
        """Consume the capture worker's control channel.

        This is the SCK backend's `_read_progress`. It has the same one job —
        set `self._t0` — but arrives at it differently: ffmpeg reports
        elapsed output time, while the worker reports frame 0's presentation
        time on the CoreMedia host clock, which we convert into our own
        `time.monotonic()` domain HERE, with two adjacent local reads. The
        child's own clock is never trusted; `time.monotonic()` is per-process
        on this stack, and trusting it across a process boundary is exactly
        what put every key timestamp a process-age in the past.
        """
        stream = self.proc.stdout
        if stream is None:
            return
        for raw in iter(stream.readline, b""):
            try:
                line = raw.decode("utf-8", "replace").strip()
            except Exception:
                continue
            kind, payload = sck.parse_worker_line(line)
            if kind == "t0":
                with self._t0_lock:
                    if self._t0 is None:
                        self._t0 = sck.host_pts_to_monotonic(payload["pts0"])
            elif kind == "size":
                # The encoded pixel dims. Stored for every SCK take; only the
                # window-native meta writes it out (see _capture_window_meta).
                self._sck_size = payload
            elif kind == "stat":
                self._sck_stat = payload
            elif kind == "done":
                # The worker finished writing the movie tail. Recorded so the
                # success check can tell a finalized take from one whose
                # `moov` never landed -- see _sck_unfinalized_message.
                self._sck_done = payload
            elif kind == "filter":
                self._sck_filter = payload
                if payload.get("missing"):
                    # Our own chrome silently left in the take is the exact
                    # bug this backend exists to prevent, so it is worth a
                    # line even though it does not fail the recording.
                    print("warning: {} window id(s) could not be excluded"
                          .format(payload["missing"]), file=sys.stderr)
            elif kind == "err":
                self._sck_error = "{} {}".format(payload.get("domain", ""),
                                                 payload.get("detail", "")).strip()
                self._stderr_tail.append(self._sck_error)
            elif kind == "warn":
                self._stderr_tail.append("warn: " + payload.get("detail", ""))

    def _screen_cmd(self):
        """The avfoundation screen-capture argv.

        Lifted out of `start()` with NOT ONE CHARACTER CHANGED, for two
        reasons. First, nothing pinned this command: `-thread_queue_size` is
        load-bearing (see the comment below) and silently does nothing if it
        ever drifts after `-i`, and a frozen literal in the tests is the only
        thing that would catch that. Second, a second capture backend needs a
        seam to sit beside rather than an `if` buried in the middle of a
        200-line `start()`.
        """
        # "synthetic" hides the OS cursor at capture time (capture_cursor 0) so
        # render.py can draw its own smoothed/enlarged cursor from the move
        # track without doubling up on the real one baked into the pixels.
        capture_cursor_flag = "0" if self.cursor_mode == "synthetic" else "1"
        cmd = ["ffmpeg", "-y", "-hide_banner",
               "-f", "avfoundation", "-capture_cursor", capture_cursor_flag,
               "-framerate", str(self.fps),
               # THE fix for staticky audio, and for video sliding behind it.
               # avfoundation feeds video and audio through one demuxer into
               # per-stream queues sized 8 packets by default. A Retina
               # display at 60fps keeps libx264 busy enough that the video
               # queue backs up, and when it blocks, AUDIO PACKETS ARE
               # DROPPED -- which is heard as static, and seen as the picture
               # falling behind the sound it was recorded with. Both symptoms,
               # one starved queue. This must come BEFORE -i: it is an input
               # option and ffmpeg silently ignores a trailing one.
               "-thread_queue_size", str(THREAD_QUEUE_SIZE),
               "-i", _fmt_input(self.video_idx, self.mic_idx),
               "-c:v", "libx264", "-preset", "ultrafast",
               "-tune", "zerolatency",  # don't buffer frames -> tight sync
               "-crf", str(self.crf), "-pix_fmt", "yuv420p",
               "-r", str(self.fps)]  # force CFR: frame index -> time is exact
        if self.mic_idx is not None:
            # Pin the rate rather than inheriting whatever the device reports:
            # a 44.1k mic resampled per-packet is its own source of grit, and
            # aac wants 48k anyway.
            cmd += ["-c:a", "aac", "-b:a", "160k", "-ar", "48000",
                    # Keep audio on the same clock as video for the whole
                    # take. Without this, a device whose clock drifts from the
                    # capture clock accumulates offset over a long recording
                    # -- silently, because nothing resyncs them.
                    "-af", "aresample=async=1:first_pts=0"]
        cmd += ["-progress", "pipe:1", "-nostats", self.raw_path]
        return cmd

    def _failure_message(self, rc, raw_ok):
        if self.backend == "sck":
            return self._sck_failure_message(rc, raw_ok)
        parts = ["ffmpeg screen capture failed"]
        if rc is not None:
            parts[0] += " (exit code {})".format(rc)
        tail = "".join(self._stderr_tail).strip()
        if tail:
            parts.append("--- ffmpeg output (tail) ---\n" + tail)
        if not raw_ok and self._screen_denied_signature(tail):
            # Specific, actionable diagnosis instead of the generic list.
            msg = (
                "The screen input could not be configured, so avfoundation fell\n"
                "back to a camera (note the camera-only pixel formats above).\n"
                "This is the Screen Recording permission signature -- almost\n"
                "always one of:\n"
                "  1. Screen Recording is not granted to THIS app in\n"
                "     System Settings > Privacy & Security > Screen Recording.\n"
                "  2. It IS granted but the grant is wedged (looks on, doesn't\n"
                "     take): toggle it off/on, FULLY QUIT the terminal (Cmd-Q),\n"
                "     and if it still fails, REBOOT -- macOS's screen-capture TCC\n"
                "     state often only clears on restart.\n"
                "  3. A macOS update broke this ffmpeg's avfoundation capture:\n"
                "     `brew upgrade ffmpeg` (or reinstall it).\n"
                "An active iPhone Continuity Camera can also trigger this -- turn\n"
                "Continuity Camera off if the device list shows one.")
            if self.face_idx is not None:
                msg += (
                    "\n  4. The --face webcam capture raced the screen capture's\n"
                    "     avfoundation session setup. This is normally serialized\n"
                    "     (screen first, then webcam); if you see this WITH --face,\n"
                    "     retry, or record without --face to confirm the screen\n"
                    "     capture works on its own.")
            parts.append(msg)
        elif not raw_ok:
            parts.append(
                "No usable recording was written. Common causes:\n"
                "  - Screen Recording permission not granted to this terminal\n"
                "    (System Settings > Privacy & Security > Screen Recording),\n"
                "  - wrong --display index (run `studio devices`),\n"
                "  - the requested --fps isn't supported by the display.")
        return "\n\n".join(parts)

    def _finalize(self):
        # Stop asking about exclusions before the pipe closes underneath the
        # poller -- it fails soft either way, but a shutdown that races is a
        # shutdown someone eventually has to debug.
        self._stop_exclude_poll.set()
        # Window geometry at stop, read BEFORE ffmpeg is asked to wind down so
        # it describes the window as of the last captured frames. None just
        # means "gone/minimized/moved away by then" -- which is exactly what
        # turns "my recording is ruined and I don't know why" into an explained
        # warning at render time. Best-effort like everything else here.
        if self.capture_window:
            end = self._snapshot_capture_window()
            try:
                self._cw_end_rect = self._rect_points(end) if end else None
            except (KeyError, TypeError, ValueError):
                self._cw_end_rect = None
        # Close the face sink BEFORE meta is written: its t0 and frame count
        # are what decide face_capture and face_t0_monotonic.
        self._stop_shared_face()
        for entry in self.capture_windows:
            end = self._snapshot_window(entry)
            try:
                self._cws_end_rects.append(
                    self._rect_points(end) if end else None)
            except (KeyError, TypeError, ValueError):
                self._cws_end_rects.append(None)
        # Stop the poller before the event file is closed below -- it writes
        # under _ev_lock and bails on a closed file, but joining here keeps
        # the sample count (and therefore meta's `track` state) settled
        # before _meta_dict reads it.
        self._stop_window_track()
        # Stop the facecam first (best-effort, clean trailer) so it isn't left
        # holding the webcam if the screen trailer write stalls.
        if self.face_proc is not None and self.face_proc.poll() is None:
            try:
                self.face_proc.send_signal(signal.SIGINT)
            except Exception:
                pass
            try:
                self.face_proc.wait(timeout=6)
            except Exception:
                try:
                    self.face_proc.kill()
                except Exception:
                    pass
        try:
            if self.proc is not None and self.proc.poll() is None:
                # SIGINT -> ffmpeg writes the trailer and exits cleanly.
                try:
                    self.proc.send_signal(signal.SIGINT)
                except Exception:
                    pass
                self._wait_clean()
        finally:
            with self._ev_lock:
                if self._ev_file is not None:
                    try:
                        self._ev_file.flush()
                        self._ev_file.close()
                    finally:
                        self._ev_file = None
            # Put the user's windows back LAST -- after the encoder has wound
            # down, or they'd snap back into the final frames of the take.
            self._restore_arrangement()
            # Tell the watchdog to stand down LAST of all -- it protects
            # the *entire* shutdown sequence above (a crash mid-finalize is
            # exactly the scenario it exists for), so it should only learn
            # its job is done once there's truly nothing left for it to do.
            self._stop_watchdog()

    def _restore_arrangement(self):
        """Undo the pre-take un-overlap, if one happened. Best-effort."""
        saved, self._arrange_restore = self._arrange_restore, None
        if not saved:
            return
        try:
            from . import arrange
            arrange.restore(saved)
        except Exception:
            pass

    def _wait_clean(self):
        """Wait for the capture process to finalize, tolerating an impatient
        2nd Ctrl+C.

        The SCK worker needs materially more grace than ffmpeg. Its finish()
        stops the stream (up to 5s) and then runs
        `finishWritingWithCompletionHandler_` (up to 30s) before it exits, and
        SIGKILLing it before that completes is exactly how the `moov` tail is
        lost -- the frames are written but the index never is (see
        docs/architecture.md). The worker always self-terminates within
        that budget, even on a hang, and `wait()` returns the instant it
        exits, so outwaiting it costs a normal take nothing (finalize is
        sub-second in practice) while never cutting off a slow-but-honest one.
        ffmpeg writes its trailer in well under a second, so it keeps the
        original tight budget.
        """
        grace = 40 if self.backend == "sck" else 8
        escalations = 0
        while True:
            try:
                self.proc.wait(timeout=grace)
                return
            except subprocess.TimeoutExpired:
                escalations += 1
                if escalations == 1:
                    self.proc.terminate()
                    grace = 8   # it ignored the trailer window; wind down fast
                else:
                    self.proc.kill()
                    return
            except KeyboardInterrupt:
                # Don't abort the trailer write on a repeated Ctrl+C.
                continue
