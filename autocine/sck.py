"""sck.py — parent-side helpers for the ScreenCaptureKit capture path.

**This module imports no ScreenCaptureKit and no PyObjC.** Everything that
touches SCK lives in `_sck_worker.py`, a disposable child process, for the
same reason `_key_worker.py` exists: a PyObjC call into SCK was measured
(2026-07-30) turning an ObjC exception into a `SIGABRT` of its host, arriving
through `com.apple.replayd`'s `NSXPCConnection` reply block, where no Python
`try/except` can reach it. Keeping the decision logic here means the parent
can reason about capture without ever loading the thing that can kill it —
and means these functions are unit-testable on any machine, with no
permissions and no display.

Today this holds the CFR gate, which is useful on its own: `raw.mov` being
constant frame rate is assumed all over the renderer — `render.py` and
`retime.TimeMap` map frame index to time by dividing by fps, and read that
fps from **cv2**, not from `meta["fps"]` — and until now nothing checked it.
A VFR file doesn't fail loudly; it silently stretches time, so zooms drift
off their clicks and the editor preview stops agreeing with the export.
"""

from fractions import Fraction
import json
import subprocess
import time

# How far the measured rate may sit from the requested one and still count as
# CFR. A whole frame per minute at 60fps is ~0.017 fps, so this is tight
# enough to catch a 59.94-vs-60 mismatch (0.06 fps) — which is the realistic
# failure, since a display that isn't a clean 60 Hz makes a nominal-60 grid
# accumulate about one frame every 16 seconds.
RATE_EPS = 0.01


# Refresh rates a Mac display plausibly runs at, ascending. Used to work out
# which grid the frames actually arrived on, which is NOT always the one we
# asked for: a nominal-60 grid on a 59.94 Hz panel drifts about one frame
# every 16 seconds, and it drifts silently.
GRID_CANDIDATES = (23.976, 24.0, 25.0, 29.97, 30.0, 47.952, 48.0, 50.0,
                   59.94, 60.0, 75.0, 90.0, 100.0, 119.88, 120.0, 144.0)

# How far a frame may sit from its slot before the grid is rejected, seconds.
# ~1ms is well under half a frame at 120 Hz (4.2ms) so a correct grid always
# passes, and far under the error a wrong grid accumulates.
GRID_TOL_SEC = 0.001

# Minimum span of samples needed to tell 59.94 from 60 apart. They differ by
# one part in a thousand, so a tenth of a second of frames cannot separate
# them however many frames it contains -- only elapsed time helps.
GRID_MIN_SPAN_SEC = 2.0


def infer_grid_hz(times, candidates=GRID_CANDIDATES, tol=GRID_TOL_SEC,
                  min_span=GRID_MIN_SPAN_SEC):
    """Which constant grid do these frame times actually lie on? -> hz or None.

    `times` are presentation times in seconds, relative to the first frame
    (so `times[0]` is 0.0). Returns the LOWEST candidate rate that every
    sample sits on to within `tol`, or None when nothing fits.

    Deliberately measured against CUMULATIVE time rather than inter-frame
    deltas. A single 60 Hz delta (16.667ms) is only 16 MICROseconds away from
    a 59.94 Hz one (16.683ms) — no per-delta test can separate them, and a
    test that claims to is lying. Over elapsed time the error accumulates:
    by 5 seconds the two grids are a third of a frame apart, which is
    unmissable. Hence `min_span` — with less than that, we return None
    ("don't know") rather than a coin flip.

    Lowest-fitting wins because the coarsest grid that explains the data is
    the honest one. 60 Hz samples also fit a 120 Hz grid — every frame lands
    on an even slot — so without the ordering we would claim a finer grid
    than we can support, and pad the file with duplicates nobody asked for.
    The ordering is safe in the other direction: genuine 120 Hz samples do
    NOT fit a 60 Hz grid, because the odd ones land halfway between slots.

    None is a REFUSAL, and the caller is expected to treat it as one: the
    whole point of running this before `startCapture` is to fall back to a
    backend we trust rather than record something that will drift.
    """
    ts = [float(t) for t in times if t is not None and float(t) >= 0.0]
    if len(ts) < 2:
        return None
    span = max(ts) - min(ts)
    if span < float(min_span):
        return None
    for hz in sorted(candidates):
        period = 1.0 / hz
        worst = 0.0
        for t in ts:
            slot = round(t / period)
            worst = max(worst, abs(t - slot * period))
            if worst > tol:
                break
        if worst <= tol:
            return hz
    return None


def plan_slots(pts, pts0, fps, last_slot, max_fill=None):
    """Which CFR slots does a frame arriving at `pts` occupy?

    -> (repeat_slots, slot, dropped)

      repeat_slots  slots to fill with the PREVIOUS frame, in order
      slot          slot for this frame, or None if it should be dropped
      dropped       True when the frame lands on a slot already written

    This is the whole CFR story in one pure function. ScreenCaptureKit does
    not deliver a frame when nothing on screen changed, so a stream of frames
    is not a stream of slots: sit still for two seconds at 60fps and SCK
    hands over a handful of frames spanning 120 slots. Writing them as they
    arrive produces a variable-rate file, which is the failure the whole
    renderer is built to not survive — `render` and `retime.TimeMap` turn
    frame index into time by dividing by fps. So the gaps are filled with the
    last frame, which is exactly what was on screen during them.

    Frames landing on or before `last_slot` are DROPPED rather than nudged
    forward. Nudging would push every subsequent frame off its true time to
    preserve a frame nobody can see (two frames in one 16ms slot means the
    display showed one of them), and that error never comes back.

    `max_fill` bounds the gap. A machine that slept, or a stream that stalled
    on a locked screen, otherwise asks for a fill of hundreds of thousands of
    frames — turning a hiccup into an out-of-disk. The caller reports the
    clamp; a clamped take is time-shifted from that point and must not be
    passed off as clean.
    """
    try:
        step = 1.0 / float(fps)
    except (TypeError, ValueError, ZeroDivisionError):
        return ([], None, True)
    if last_slot is None:
        # First frame defines the origin: it IS slot 0, whatever its pts.
        return ([], 0, False)
    slot = int(round((float(pts) - float(pts0)) / step))
    if slot <= int(last_slot):
        return ([], None, True)
    gap = slot - int(last_slot) - 1
    if max_fill is not None and gap > int(max_fill):
        gap = int(max_fill)
        slot = int(last_slot) + gap + 1
    return (list(range(int(last_slot) + 1, int(last_slot) + 1 + gap)),
            slot, False)


# How far BEHIND the wall clock the idle fill deliberately runs. `plan_slots`
# drops a frame that lands on an already-written slot, so filling right up to
# "now" would discard a frame still in flight from the capture queue (its pts
# is stamped when the frame was grabbed, not when it was handed over). Lagging
# is free: a real arrival still lands on its own true slot, and `finish` fills
# the remainder exactly. 0.5s is far beyond any measured delivery latency.
IDLE_FILL_LAG_SEC = 0.5


def now_host_pts():
    """"Now" on the SAME clock ScreenCaptureKit stamps frames with.

    `CMClockGetHostTimeClock()` and `CLOCK_UPTIME_RAW` are the same timebase
    (see `host_pts_to_monotonic`, verified 2026-07-30 to 14µs-270µs), so the
    worker can locate itself on the pts timeline without touching ObjC.
    """
    return time.clock_gettime(time.CLOCK_UPTIME_RAW)


def plan_idle_fill(now_pts, pts0, fps, last_slot,
                   lag_sec=IDLE_FILL_LAG_SEC, max_fill=None):
    """Slots to dup-fill so the timeline keeps up with the WALL CLOCK.

    -> list of slots to fill with the held frame (possibly empty)

    `plan_slots` fills the gap between two arrivals, which is only half the
    story: SCK delivers nothing at all while the screen is still, so nothing
    calls it. A window that sat motionless for nine of its ten seconds
    therefore produced a one-second movie -- every frame real, every gap
    between them filled, and nine seconds of the take simply absent, because
    the timeline only ever advanced when a frame arrived. Measured before this
    existed: an idle window recorded for 10.0s wrote 65 frames (1.08s), and a
    26.0s three-window take wrote 1.28s per channel, both reporting `dropped:
    0`. The tail is the worst case -- time after the LAST change has no
    following arrival to fill it, so it is lost outright rather than
    compressed.

    So the caller ticks this on a timer as well, and the two compose: this
    advances the timeline through the quiet, `plan_slots` still places every
    real frame on its own true slot.

    Returns [] until a first frame has established `pts0`/`last_slot` -- there
    is nothing to duplicate before then, and the first frame defines slot 0
    whenever it arrives.
    """
    if pts0 is None or last_slot is None:
        return []
    try:
        step = 1.0 / float(fps)
        target = int((float(now_pts) - float(lag_sec) - float(pts0)) / step)
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    last = int(last_slot)
    if target <= last:
        return []
    gap = target - last
    if max_fill is not None and gap > int(max_fill):
        gap = int(max_fill)
    return list(range(last + 1, last + 1 + gap))


def plan_restore_action(prev_onscreen, now_onscreen, fail_count, max_tries):
    """Window-native minimize/restore state machine -> one action string.

    -> "none" | "hidden" | "rebuild" | "giveup"

    A window captured via `initWithDesktopIndependentWindow_` loses its backing
    surface when minimized, and the stream does NOT re-attach on restore (nor
    does an in-place `updateContentFilter_` swap) -- only a fresh `SCStream`
    recovers it. Measured in `tools/sck_minimize_probe.py`. So the worker
    watches the target's on-screen state and rebuilds its stream on the
    minimized->restored edge. This pure function is that edge logic, split out
    so it is unit-tested without ScreenCaptureKit:

      prev_onscreen  last committed on-screen state (True at capture start)
      now_onscreen   this poll's reading; None on a transient read failure
      fail_count     consecutive failed rebuild attempts since going hidden
      max_tries      give up (stop hammering) after this many failures

    "hidden": the target just went off-screen (minimize/hide/other-Space);
    the caller records the frozen span and flips its state to off-screen.
    "rebuild": the target is back; attempt a fresh SCStream. On success the
    caller commits on-screen + resets fail_count; on failure it leaves the
    state off-screen so the NEXT poll re-attempts, until "giveup".
    "none": no edge (or an unreadable poll) -- change nothing.

    Occlusion never triggers this: a covered window stays on-screen, so
    `now_onscreen` never goes False for it.
    """
    if now_onscreen is None:
        return "none"
    now = bool(now_onscreen)
    prev = bool(prev_onscreen)
    if now == prev:
        return "none"
    if not now:
        return "hidden"                       # prev on-screen -> now hidden
    # now on-screen, prev hidden: the restore edge (or a retry of it).
    if int(fail_count) >= int(max_tries):
        return "giveup"
    return "rebuild"


# Two windows vanishing within this span are ONE gesture (a Space switch's
# staggered list-departures, Minimize-All), never per-window intent. A user
# deliberately minimizing two windows lands well outside it (two clicks +
# genies); measured Space stagger is ~0.05-0.5s.
GLOBAL_VANISH_WINDOW_SEC = 1.0


def plan_shrink_exits(tracker, onscreen, arrivals, now, debounce,
                      gaveup=()):
    """The fleet card-shrink detector's per-poll decision (M3.2). PURE.

    The parent's geometry poller calls this each pass with one reading per
    SHRINKABLE captured window (the caller already excludes the mic anchor
    and never calls with fewer than 2 live cards). The exit rule is
    docs/architecture.md M3 decisions 1-3, both signals REQUIRED:

      absent-from-screen  (onscreen False; True or None -- a read hiccup --
                           counts as SEEN and cancels)
      AND arrivals-dead   (the channel's delivered-frame count flat since the
                           hide -- a hidden-but-delivering window, e.g. on an
                           inactive Space, keeps recording; MEASURED)

    held continuously for `debounce` seconds. The seam is BACK-DATED to the
    hide start -- the debounce delays only the decision. If arrivals ADVANCE
    while hidden, the baseline re-arms (hidden_since = now): delivery proves
    the pixels are real, so the eventual seam sits where delivery actually
    stopped, not where the window left the list.

    Decision 1's global-vanish guard: >=2 windows transitioning to hidden in
    the SAME poll is a Space switch / fullscreen transition / multi-window
    Cmd-H -- "the user looked away", never per-window intent. None of them
    arm. Already-armed windows keep their clocks (their own solo hide
    predates the global event, and the arrivals gate still protects them).

    `gaveup` lists windows whose worker reported `window_rebuild_gaveup`:
    back on-screen but the stream is dead and stays dead (the one case the
    diagnostics channel earns a decision role -- decision 3). They are
    treated as hidden even while SEEN, with the seam at their LAST real hide
    start (`last_hide_started` survives cancels for exactly this).

    tracker: {wid: {"hidden_since": t|None, "arrivals_at_hide": int,
                    "last_hide_started": t|None}} -- caller-held state,
    replaced by the returned copy (windows absent from `onscreen` fall out).
    Returns (new_tracker, exits) with exits = [(wid, seam_t), ...].
    """
    new_tracker = {}
    exits = []
    gaveup = set(gaveup)
    newly_hidden = [
        wid for wid, on in onscreen.items()
        if on is False and wid not in gaveup
        and (tracker.get(wid) or {}).get("hidden_since") is None]
    # Global-vanish is a time WINDOW, not tick-coincidence: a Space switch
    # staggers the list-departures across ~50ms polls (measured on-device --
    # two cards left on adjacent ticks, read as two SOLO vanishes, and one
    # exited). Vanish history rides the tracker under a reserved key; any
    # two different wids vanishing within the window suppress each other,
    # RETRO-cancelling the earlier arm.
    recent = [v for v in (tracker.get("__vanishes__") or [])
              if now - v[1] <= GLOBAL_VANISH_WINDOW_SEC]
    global_vanish = len(newly_hidden) >= 2 or (
        len(newly_hidden) == 1
        and any(v[0] != newly_hidden[0] for v in recent))
    recent += [(wid, now) for wid in newly_hidden]
    suppressed = set(newly_hidden) if global_vanish else set()
    if global_vanish:
        for wid in list(tracker):
            if wid == "__vanishes__":
                continue
            prev = tracker[wid]
            if (prev.get("hidden_since") is not None
                    and now - prev["hidden_since"]
                    <= GLOBAL_VANISH_WINDOW_SEC):
                prev = dict(prev)
                prev["hidden_since"] = None    # part of the same gesture
                tracker = dict(tracker)
                tracker[wid] = prev
                suppressed.add(wid)            # and not re-armed this pass
    for wid, on in onscreen.items():
        prev = dict(tracker.get(wid) or {})
        arr = int(arrivals.get(wid, 0))
        seen = (on is not False) and wid not in gaveup
        if seen:
            prev["hidden_since"] = None          # cancel; keep last_hide_started
            new_tracker[wid] = prev
            continue
        if prev.get("hidden_since") is None:
            if wid in suppressed:
                new_tracker[wid] = prev          # suppressed: not per-window intent
                continue
            start = now
            if wid in gaveup and prev.get("last_hide_started") is not None:
                start = prev["last_hide_started"]    # the true freeze start
            prev["hidden_since"] = start
            prev["last_hide_started"] = start
            prev["arrivals_at_hide"] = arr
            new_tracker[wid] = prev
            continue
        if arr != int(prev.get("arrivals_at_hide", arr)):
            # Still hidden but DELIVERING (inactive Space): re-arm so the
            # seam tracks the moment delivery actually stops.
            prev["hidden_since"] = now
            prev["last_hide_started"] = now
            prev["arrivals_at_hide"] = arr
            new_tracker[wid] = prev
            continue
        if now - prev["hidden_since"] >= float(debounce):
            exits.append((wid, float(prev["hidden_since"])))
            continue                             # fired: drop from tracking
        new_tracker[wid] = prev
    if recent:
        new_tracker["__vanishes__"] = recent
    return new_tracker, exits


def plan_rejoin_ready(watch, wids, present, now, stable_sec):
    """The auto-REJOIN restore edge (M3.3). PURE.

    A departed fleet window seen back on-screen CONTINUOUSLY for
    `stable_sec` earns its one automatic re-add ("restoring means re-record
    it" -- docs/architecture.md M3 decision 10). The stability window absorbs the
    restore animation and SCShareableContent re-listing lag so the grow that
    follows doesn't burn its T0 wait on a window mid-genie.

      watch      {wid: seen_since | None} -- caller-held, replaced by the
                 returned copy (wids not passed this time fall out)
      wids       the watchable departed wids (caller already excluded
                 attempted and re-captured ones)
      present    set of wids on-screen this pass (the FILTERED pickable
                 list: a window restored somewhere unpickable shouldn't
                 auto-rejoin a take it can't be resolved for)
      now / stable_sec  parent clock

    Returns (new_watch, ready). A ready wid leaves the watch -- the caller
    marks it attempted (one attempt per departure) and the run loop decides
    eligibility at drain time.
    """
    new_watch, ready = {}, []
    for wid in wids:
        if wid not in present:
            new_watch[wid] = None            # absence resets the stability
            continue
        since = watch.get(wid)
        since = now if since is None else since
        if now - since >= float(stable_sec):
            ready.append(wid)
            continue
        new_watch[wid] = since
    return new_watch, ready


def has_moov(path, _open=open):
    """Does `path` carry a top-level QuickTime/MP4 `moov` box? -> bool.

    The `moov` box is the sample index — offsets, sizes, durations — and it is
    written LAST, only when the writer finalizes (`AVAssetWriter`'s
    `finishWriting`, ffmpeg's trailer). A file with frames in `mdat` but no
    `moov` is unplayable: every consumer reports `moov atom not found`. That is
    exactly the shape a take left behind when its writer is stopped mid-tail —
    the known `AVAssetWriter` tail-loss risk (see docs/architecture.md,
    "we lose the tail") — and the frames are all still there, so file size
    alone cannot tell a finalized take from a broken one.

    Scanning the top-level boxes answers the question for a handful of reads:
    no decode, no ffprobe, no ObjC, so it is safe to call from the recorder's
    success check and from the app's thumbnailer alike. A box with declared
    size 0 "extends to EOF" — it is the last box, so a `moov` has to appear
    at or before it or not at all.

    Best-effort by construction: ANY read or parse problem returns False
    ("not proven finalized"), never raises. The callers use it to decide
    whether a take is broken, and a checker that can itself throw would just
    be a new way to lose a take.
    """
    try:
        with _open(path, "rb") as f:
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    return False
                size = int.from_bytes(hdr[:4], "big")
                box = hdr[4:8]
                if box == b"moov":
                    return True
                if size == 0:
                    # Extends to EOF: nothing after it to find.
                    return False
                if size == 1:
                    ext = f.read(8)
                    if len(ext) < 8:
                        return False
                    size = int.from_bytes(ext, "big")
                    if size < 16:            # 8 hdr + 8 largesize, minimum
                        return False
                    f.seek(size - 16, 1)
                else:
                    if size < 8:             # a size that can't hold its header
                        return False
                    f.seek(size - 8, 1)
    except Exception:
        return False


BACKENDS = ("avfoundation", "sck")
DEFAULT_BACKEND = "avfoundation"


def resolve_backend(explicit=None, env=None, saved=None):
    """Which capture backend to use. -> one of BACKENDS.

    Precedence: an explicit argument (a CLI flag, a request field), then the
    environment, then the SAVED preference, then the default. Saved sits
    below the environment so a one-off `AUTOCINE_CAPTURE_BACKEND=...`
    still wins for that launch without quietly rewriting what the user chose.

    The default stays `avfoundation` and will keep staying it until the
    criterion in docs/architecture.md is met, because "every off switch must be
    bit-exact" cannot survive a backend change: no SCK file is byte-identical
    to an ffmpeg one, so the only honest off switch is the old path still
    being there and still being what runs by default. A user opting in on
    their own machine is a different thing from us changing what everyone
    gets.

    An unrecognized value falls back to the default rather than raising. A
    typo in an env var, or a hand-edited settings.json, must not be able to
    stop someone recording — and the CLI validates its own flag up front,
    where a typo can still be reported to the person who typed it.
    """
    for value in (explicit,
                  (env or {}).get("AUTOCINE_CAPTURE_BACKEND"),
                  saved):
        if value is None:
            continue
        value = str(value).strip().lower()
        if value in BACKENDS:
            return value
    return DEFAULT_BACKEND


def parse_worker_line(line):
    """One line of `_sck_worker.py` stdout -> (kind, payload dict).

    Returns ("", {}) for anything unrecognized, which the reader drops. That
    is deliberate: the worker's stdout is a control channel, and a line we do
    not understand must never be able to take down a recording in progress.

    Kinds mirror the protocol in `_sck_worker.py`'s docstring: ready, size,
    t0, filter, stat, warn, done, err.
    """
    if not line:
        return ("", {})
    parts = str(line).strip().split()
    if not parts:
        return ("", {})
    head, rest = parts[0], parts[1:]

    def nums(n):
        if len(rest) < n:
            return None
        try:
            return [float(x) for x in rest[:n]]
        except ValueError:
            return None

    if head == "READY":
        return ("ready", {})
    if head == "SIZE":
        got = nums(3)
        if got is None:
            return ("", {})
        return ("size", {"width": int(got[0]), "height": int(got[1]),
                         "fps": int(got[2])})
    if head == "T0":
        got = nums(1)
        return ("", {}) if got is None else ("t0", {"pts0": got[0]})
    if head == "FILTER":
        got = nums(2)
        if got is None:
            return ("", {})
        return ("filter", {"applied": int(got[0]), "missing": int(got[1])})
    if head == "STAT":
        got = nums(5)
        if got is None:
            return ("", {})
        out = {"slot": int(got[0]), "appended": int(got[1]),
               "dup": int(got[2]), "dropped": int(got[3]),
               "notready": int(got[4])}
        # `idle` (slots dup-filled by the wall-clock tick) is appended after
        # the original five, so a worker that predates it still parses.
        more = nums(6)
        if more is not None:
            out["idle"] = int(more[5])
        return ("stat", out)
    if head == "DONE":
        got = nums(2)
        if got is None:
            return ("", {})
        return ("done", {"frames": int(got[0]), "media": got[1]})
    if head == "WARN":
        return ("warn", {"detail": " ".join(rest)})
    if head == "ERR":
        return ("err", {"domain": rest[0] if rest else "",
                        "detail": " ".join(rest[1:])})
    return ("", {})


def host_pts_to_monotonic(pts0, mono=None, uptime=None):
    """A capture PTS on the CoreMedia host clock -> our own `time.monotonic()`.

    This is the SCK backend's replacement for ffmpeg's `out_time_us` anchor,
    and it is deliberately computed ENTIRELY IN THE PARENT from two adjacent
    local reads. Nothing about the child's clock is trusted or transmitted,
    because `time.monotonic()` is per-process on this stack — the exact trap
    that put every key timestamp a whole process-age in the past.

    It works because `CMClockGetHostTimeClock()` and `CLOCK_UPTIME_RAW` are
    the same timebase (verified 2026-07-30: agreement to 14µs–270µs, which is
    just the gap between the two reads themselves) and `CLOCK_UPTIME_RAW` is
    system-wide, so a PTS produced in another process is still meaningful
    here. Read the two clocks as close together as possible: the error in the
    result is exactly the time between them.
    """
    now_mono = time.monotonic() if mono is None else float(mono)
    now_up = (time.clock_gettime(time.CLOCK_UPTIME_RAW)
              if uptime is None else float(uptime))
    return now_mono - (now_up - float(pts0))


def _parse_rate(text):
    """ffprobe's "60/1" / "3000/1001" / "0/0" -> float fps, or None.

    "0/0" is ffprobe's way of saying it has no idea, and is NOT a rate of
    zero — treating it as a number is how a missing measurement turns into a
    confident wrong answer.
    """
    if not text:
        return None
    try:
        frac = Fraction(str(text).strip())
    except (ValueError, ZeroDivisionError):
        return None
    if frac <= 0:
        return None
    return float(frac)


def probe_rates(path, run=None):
    """(avg_frame_rate, r_frame_rate, nb_frames) for a file's video stream.

    Each may be None. `run` is injectable so tests never need ffprobe.

    Both rates are read because they disagree in exactly the interesting
    case: `r_frame_rate` is the base/ideal rate ffprobe infers, while
    `avg_frame_rate` is frames divided by duration. A true CFR file has them
    equal; a file whose frames are unevenly spaced does not.
    """
    runner = run or _run_ffprobe
    try:
        info = runner(path)
    except Exception:
        return (None, None, None)
    try:
        streams = [s for s in (info or {}).get("streams", [])
                   if s.get("codec_type") == "video"]
    except Exception:
        return (None, None, None)
    if not streams:
        return (None, None, None)
    s = streams[0]
    nb = s.get("nb_frames")
    try:
        nb = int(nb)
    except (TypeError, ValueError):
        nb = None
    return (_parse_rate(s.get("avg_frame_rate")),
            _parse_rate(s.get("r_frame_rate")), nb)


# Wall-clock ceiling on the CFR probe. Reading stream metadata off a local
# file is near-instant, so this is not a tuning knob -- it is the guard that
# keeps a wedged ffprobe from eating a good take. `_check_cfr` runs on EVERY
# backend, avfoundation included, and it runs BEFORE meta.json is written, so
# a probe that never returns would leave a complete raw.mov with no meta.json
# beside it -- which this project reads as a FAILED take. TimeoutExpired is an
# Exception, so `probe_rates` catches it and the verdict degrades to
# "unknown": "we didn't look", never "we looked and it's fine".
PROBE_TIMEOUT_SEC = 20.0


def _run_ffprobe(path):
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        timeout=PROBE_TIMEOUT_SEC)
    return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")


def verify_cfr(path, fps, run=None):
    """Is `path` constant frame rate at `fps`? -> (status, detail).

    status is one of:
      "ok"      — measured constant, and at the rate we asked for.
      "suspect" — the two rates disagree, or neither matches `fps`. NOT
                  "failed": the recording is still there and still watchable,
                  and refusing it would throw away a take over a check that
                  has never run in the wild. The caller warns; a human
                  decides.
      "unknown" — couldn't measure (no ffprobe, unreadable file, no video
                  stream). Deliberately distinct from "ok": "we didn't look"
                  and "we looked and it's fine" must never collapse into the
                  same value, which is the same tri-state lesson as
                  `meta["key_capture"]`.

    The detail string always names `avg_frame_rate`/`r_frame_rate` and the
    numbers, because the first question anyone asks about a drifting render
    is "was the file CFR?" and the answer should be in the log already.
    """
    avg, base, nb = probe_rates(path, run=run)
    if avg is None and base is None:
        return ("unknown", "could not read avg_frame_rate/r_frame_rate from "
                           "{}".format(path))
    try:
        want = float(fps)
    except (TypeError, ValueError):
        want = None
    shown = "avg_frame_rate={} r_frame_rate={}{}".format(
        "?" if avg is None else round(avg, 4),
        "?" if base is None else round(base, 4),
        "" if nb is None else " nb_frames={}".format(nb))
    if avg is None or base is None:
        return ("unknown", "only one rate available: " + shown)
    if abs(avg - base) > RATE_EPS:
        return ("suspect", "variable frame rate: " + shown)
    if want and abs(avg - want) > RATE_EPS:
        return ("suspect", "constant, but not at the requested {} fps: {}"
                           .format(want, shown))
    return ("ok", shown)
