"""Automatic speed-up of idle stretches ("Rush").

Detects spans of the recording where nothing is happening -- no clicks,
keys, scrolls, drags, or purposeful cursor motion -- optionally intersects
them with audio silence (chipmunk narration is the biggest failure mode of
naive speed-up), then warps them faster with a quintic ramp in and out.

The whole feature is behind one branch in `render()`: when the render's
`speedup` option is off (the default) `TimeMap([])` is a bit-exact identity
and nothing here is invoked. See the "Auto speed-up (Rush)" entry in
`docs/architecture.md`.

Design principles, inherited from the rest of the project:

  - Pure decision cores + never-raising I/O wrappers (the `vision.py`
    pattern). silencedetect failure -> `silence_spans` returns None and the
    planner treats it as "no silence known" (conservative -- degrades to
    nothing sped, so a broken probe never garbles a video).
  - Closed-form warp math (quintic smootherstep on slowness, C2-continuous
    in and out). Monotone by construction (g >= 1/r > 0), so `TimeMap.warp`
    is invertible and directly samplable at the source-frame grid.
    CAVEAT (cuts): when the map carries cut spans (ripple delete), warp is
    only WEAKLY monotone -- flat across each cut -- and therefore not
    invertible at seams. The seam instant belongs to the EARLIER kept
    segment (the segments.py tie-break), so any future output->source
    lookup must resolve a seam tau to the cut's start.
  - Camera runs in OUTPUT time (see render.py): every event timestamp goes
    through `TimeMap.warp` before entering `build_path`, so authored
    perceptual constants (zoom_out_dur, chain_gap, spring omegas, motion
    blur central differences) all still refer to what the viewer sees.
  - Off switch is the empty span list. TimeMap([]) -> `identity=True` and
    every hot path in render() short-circuits. Cuts (ripple delete) ride
    the same map as a third piece kind -- `identity` requires empty spans
    AND empty cuts, so a cut-free, speedup-free render is bit-exact.
"""

import re
import subprocess

import numpy as np

try:
    import cv2 as _cv2
except ImportError:   # pragma: no cover
    _cv2 = None


DEFAULTS = dict(
    rate=6.0,          # plateau speed factor; 10s idle -> ~2.1s output at r=6
    min_idle=3.0,      # s, smallest span worth compressing
    pad=0.8,           # s, activity padding on each side (protects ripples,
                       # reaction beats; matches camera hold_after)
    ramp=0.5,          # s, quintic speed-ramp duration each end
    move_speed=90.0,   # source px/s move-speed gate: below = drift/tremor
                       # (idle), above = deliberate travel (active)
    drag_cap=30.0,     # s, safety cap on a drag pair (== camera drag_max_hold)
    silence_gate=True, # intersect idle with audio silence when a mic track exists
    silence_db=-38.0,  # ffmpeg silencedetect noise threshold
    silence_min=0.6,   # s, silencedetect min silence duration (mid-sentence
                       # breaths stay "not silent")
    # Visual motion gate: an "idle" span the event stream sees is not
    # necessarily idle on SCREEN -- a playing video, scrolling build log,
    # progress bar, or download animation keeps changing pixels without
    # generating input events. Time-lapsing that turns it into a glitchy
    # blur. Gate splits/drops any span containing sustained on-screen
    # motion. Cost is proportional to IDLE time only (probes seek into
    # candidate spans, not the whole clip), and each probe uses cap.grab()
    # between sparse retrieves + a downscaled diff, so the extra decode
    # is a small fraction of the render's own cost.
    motion_gate=True,
    motion_probe_fps=4.0,   # sample cadence inside a candidate span
    motion_frac=0.02,       # fraction of the 256px-max frame that must
                            # change to count as "motion" (blinking caret
                            # is ~1e-4, video playback >= a few %)
    motion_menubar_frac=0.03,  # exclude the top strip (clock)
    motion_max_probe_dim=256,   # px; downscale to this max dim before diff
    motion_diff_thresh=10,      # per-pixel |delta| gate (uint8)
    max_spans=64,      # cap on emitted spans (filter graph size bound)
)


# ---- quintic smootherstep + its integral ---------------------------------


def _q(u):
    """C2-continuous quintic smootherstep 6u^5 - 15u^4 + 10u^3 on [0, 1]."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def _Q(u):
    """Antiderivative of _q from 0 to u: u^6 - 3u^5 + 2.5u^4. Q(1) = 0.5."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 0.5
    return u * u * u * u * (u * (u - 3.0) + 2.5)


def _q_vec(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


# ---- span-set algebra ---------------------------------------------------


def union_spans(A):
    """Merge/sort a possibly-overlapping list of (a, b) or dict spans.

    Accepts (start, end) tuples or dicts with 'start'/'end'. Zero-length
    spans dropped. Returns [(a, b)] sorted, non-overlapping.
    """
    pairs = []
    for s in A or []:
        if isinstance(s, dict):
            a, b = float(s.get("start", 0.0)), float(s.get("end", 0.0))
        else:
            a, b = float(s[0]), float(s[1])
        if b > a:
            pairs.append((a, b))
    if not pairs:
        return []
    pairs.sort()
    out = [list(pairs[0])]
    for a, b in pairs[1:]:
        if a <= out[-1][1]:
            if b > out[-1][1]:
                out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def intersect_spans(A, B):
    """Sweep-line intersection of two sorted non-overlapping span lists."""
    out = []
    i = j = 0
    while i < len(A) and j < len(B):
        a0, a1 = A[i]
        b0, b1 = B[j]
        s = max(a0, b0)
        e = min(a1, b1)
        if e > s:
            out.append((s, e))
        if a1 < b1:
            i += 1
        else:
            j += 1
    return out


def subtract_spans(A, B):
    """A minus B, both sorted non-overlapping."""
    out = []
    for a0, a1 in A:
        cur_s, cur_e = a0, a1
        for b0, b1 in B:
            if b1 <= cur_s:
                continue
            if b0 >= cur_e:
                break
            if b0 > cur_s:
                out.append((cur_s, b0))
            cur_s = max(cur_s, b1)
            if cur_s >= cur_e:
                break
        if cur_e > cur_s:
            out.append((cur_s, cur_e))
    return out


def quantize_cut_spans(cuts, fps, duration=None):
    """Union-merge cut ranges and snap them onto the frame grid.

    Accepts {start, end} dicts or (a, b) pairs. Each merged range becomes
    [floor(a*fps)/fps, ceil(b*fps)/fps) -- the same floor/ceil convention
    as render._trim_frame_bounds -- so the video emit mask and the audio
    atrim windows carve IDENTICAL rationals and "how much was removed" is
    exactly reportable (a 1e-9 epsilon keeps k/fps inputs on frame k).
    Ranges are clipped into [0, duration] when duration is given; empty
    and degenerate ranges drop. Returns [(a, b)] sorted, non-overlapping
    (snapping can make neighbours touch; they re-merge).

    Both the render and the MCP snapped-echo go through THIS function, so
    the number the user is told always matches what the encoder removes.
    """
    merged = union_spans(cuts)
    fps = float(fps)
    if fps <= 0:
        return merged
    out = []
    for a, b in merged:
        qa = np.floor(a * fps + 1e-9) / fps
        qb = np.ceil(b * fps - 1e-9) / fps
        qa = max(0.0, qa)
        if duration is not None:
            dur = max(0.0, float(duration))
            qa = min(qa, dur)
            qb = max(qa, min(qb, dur))
        if qb > qa + 1e-9:
            out.append((float(qa), float(qb)))
    return union_spans(out)


# ---- detection: activity, drags, idle candidates -------------------------


def activity_times(clicks_t, moves_t, moves_x, moves_y, ups_t, keys_t,
                   scrolls_t, move_speed=90.0):
    """Sorted np.ndarray of activity timestamps (media seconds).

    Clicks/ups/keys/scrolls count unconditionally. Move samples count only
    when instantaneous speed vs the previous sample exceeds `move_speed`
    (drift/tremor vs. deliberate travel; see DEFAULTS['move_speed'] for
    calibration). Empty/None inputs are safe.
    """
    parts = []
    for arr in (clicks_t, ups_t, keys_t, scrolls_t):
        a = np.asarray(arr if arr is not None else [], dtype=float).ravel()
        if a.size:
            a = a[np.isfinite(a)]
            if a.size:
                parts.append(a)
    mv_t = np.asarray(moves_t if moves_t is not None else [], dtype=float).ravel()
    if mv_t.size >= 2:
        mx = np.asarray(moves_x, dtype=float).ravel()
        my = np.asarray(moves_y, dtype=float).ravel()
        dt = np.diff(mv_t)
        d = np.hypot(np.diff(mx), np.diff(my))
        sp = np.where(dt > 0, d / np.maximum(dt, 1e-9), 0.0)
        fast = mv_t[1:][sp > float(move_speed)]
        if fast.size:
            parts.append(fast)
    if not parts:
        return np.array([], dtype=float)
    a = np.concatenate(parts)
    a.sort()
    return a


def drag_spans(clicks_t, ups_t, cap=30.0, min_drag=0.4):
    """Pair each click with the first up after it; return (down_t, up_t) spans.

    Mirrors the camera's `drag_hold` step-2b pairing: cap on stuck-button
    safety, and blips shorter than `min_drag` seconds are treated as plain
    clicks (not drags). None/empty inputs are safe.
    """
    ups = np.sort(np.asarray(ups_t if ups_t is not None else [], dtype=float).ravel())
    clicks = np.sort(np.asarray(clicks_t if clicks_t is not None else [], dtype=float).ravel())
    out = []
    for ct in clicks:
        i = int(np.searchsorted(ups, ct, side="right"))
        if i < ups.size:
            ut = float(ups[i])
            end = min(ut, float(ct) + float(cap))
            if end - float(ct) >= float(min_drag):
                out.append((float(ct), end))
    return out


def idle_candidates(activity_t, drags, duration, pad=0.8, min_idle=3.0):
    """Candidate idle spans (source seconds) in [0, duration].

    Point events are padded by `pad` on each side, drag spans are treated
    as active over their whole duration then padded. The complement, after
    the min-idle filter, is the candidate list. Empty activity + long
    enough duration -> the whole clip is one candidate.
    """
    duration = max(0.0, float(duration))
    if duration <= 0:
        return []
    a = np.asarray(activity_t, dtype=float).ravel()
    if a.size:
        a = a[np.isfinite(a)]
        a = a[(a >= 0.0) & (a <= duration)]
        a.sort()
    intervals = []
    for t in a:
        intervals.append((max(0.0, float(t) - pad),
                          min(duration, float(t) + pad)))
    for a0, b0 in (drags or []):
        intervals.append((max(0.0, float(a0) - pad),
                          min(duration, float(b0) + pad)))
    if not intervals:
        return [(0.0, duration)] if duration >= min_idle else []
    active = union_spans(intervals)
    idle = []
    prev = 0.0
    for s, e in active:
        if s - prev >= min_idle:
            idle.append((prev, s))
        prev = e
    if duration - prev >= min_idle:
        idle.append((prev, duration))
    return idle


# ---- audio silence probe ------------------------------------------------


_SILENCE_START = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?\d+(?:\.\d+)?)")


def parse_silencedetect(stderr_text, duration):
    """Pair silence_start/end lines from ffmpeg silencedetect stderr.

    A trailing unclosed silence_start (mic is quiet at EOF) closes at
    `duration`. Garbage lines skipped. Never raises; pure decision core.
    """
    if not stderr_text:
        return []
    starts = []
    ends = []
    for line in stderr_text.splitlines():
        m = _SILENCE_START.search(line)
        if m:
            starts.append(float(m.group(1)))
            continue
        m = _SILENCE_END.search(line)
        if m:
            ends.append(float(m.group(1)))
    out = []
    dur = float(duration)
    for i, s in enumerate(starts):
        e = ends[i] if i < len(ends) else dur
        s = max(0.0, min(dur, float(s)))
        e = max(0.0, min(dur, float(e)))
        if e > s:
            out.append((s, e))
    out.sort()
    return out


def silence_spans(raw_path, duration, db=-38.0, min_dur=0.6, timeout=60):
    """[(s, e)] silence in `raw_path` via one audio-only ffmpeg pass, or None.

    None means "probe failed" -- the planner treats that as "silence
    unknown" and (per plan_speed_spans's silence contract) conservatively
    speeds nothing. Sessions with no audio stream should not call this.
    """
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", str(raw_path),
             "-vn",
             "-af", "silencedetect=noise={:g}dB:d={:g}".format(float(db), float(min_dur)),
             "-f", "null", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except Exception:
        return None
    text = proc.stderr.decode("utf-8", "replace") if proc.stderr else ""
    return parse_silencedetect(text, duration)


# ---- visual motion gate ------------------------------------------------


def motion_mask(frames, thresh_frac=0.02, diff_thresh=10, menubar_frac=0.03):
    """Per-consecutive-pair booleans: True = static (no motion detected).

    Pure decision core: takes a list of >= 2 grayscale (H, W) arrays (any
    dtype convertible to int16), returns a bool array of length len(frames)-1
    where True means the pair is "static" (below the motion threshold).

    Uses the same |delta| gate as vision.py: per-pixel |a-b| > diff_thresh
    counts as changed, then fraction-of-frame changed is compared against
    `thresh_frac`. The top `menubar_frac` of the frame is excluded (the
    macOS menubar clock flickers every second and would trigger every gate).
    """
    if not frames or len(frames) < 2:
        return np.array([], dtype=bool)
    out = np.empty(len(frames) - 1, dtype=bool)
    prev = np.asarray(frames[0], dtype=np.int16)
    h = prev.shape[0]
    top = int(round(h * float(menubar_frac)))
    for i in range(1, len(frames)):
        cur = np.asarray(frames[i], dtype=np.int16)
        d = np.abs(cur - prev)
        if top > 0:
            d[:top, :] = 0
        changed = float((d > int(diff_thresh)).mean())
        out[i - 1] = (changed < float(thresh_frac))
        prev = cur
    return out


def static_sub_spans(times, static_mask, min_span, edge_pad=0.0):
    """Extract contiguous static sub-spans (in seconds) from a probe result.

    times: sorted times (s) at which frames were sampled inside a candidate
    span. Length T (>= 2).
    static_mask: bool[T-1] returned by `motion_mask` (True = pair static).
    min_span: minimum sub-span length to keep, in seconds.
    edge_pad: shrink each surviving sub-span by this on each side (s), to
    keep the ramps of the eventual TimeMap warp comfortably inside the
    verified-static region.

    Returns [(a, b), ...] sorted, non-overlapping, each >= min_span long
    AFTER edge_pad shrink. Any span whose entire probe is static returns
    itself (as one sub-span from times[0] to times[-1]).
    """
    times = np.asarray(times, dtype=float).ravel()
    if times.size < 2 or static_mask.size == 0:
        return []
    static = np.asarray(static_mask, dtype=bool).ravel()
    if static.size != times.size - 1:
        raise ValueError("static_mask must be one shorter than times")
    out = []
    i = 0
    T = static.size
    while i < T:
        if not static[i]:
            i += 1
            continue
        j = i
        while j < T and static[j]:
            j += 1
        a = float(times[i])
        b = float(times[j])  # end of last static pair == start of first
                             # non-static (or the last probe time if we hit
                             # the end without a break)
        aa = a + float(edge_pad)
        bb = b - float(edge_pad)
        if bb - aa >= float(min_span):
            out.append((aa, bb))
        i = j + 1
    return out


def _probe_frames_in_span(raw_path, a, b, probe_fps, max_dim):
    """Decode grayscale downscaled frames sparsely across [a, b].

    Returns (times, frames) — times are the actual media seconds sampled,
    frames are grayscale uint8 arrays downscaled so max(H, W) <= max_dim.
    Any decode failure returns ([], []). Uses one seek per span + grab()
    between sample retrieves so cost scales with idle time, not clip
    length.
    """
    if _cv2 is None or b <= a:
        return [], []
    try:
        cap = _cv2.VideoCapture(str(raw_path))
        if not cap.isOpened():
            return [], []
    except Exception:
        return [], []
    try:
        src_fps = cap.get(_cv2.CAP_PROP_FPS) or 60.0
        if src_fps <= 0:
            src_fps = 60.0
        n_frames = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT))
        i0 = int(round(a * src_fps))
        i1 = int(round(b * src_fps))
        if n_frames > 0:
            i1 = min(max(0, n_frames - 1), i1)
        if i1 <= i0:
            return [], []
        step = max(1, int(round(src_fps / max(1e-6, float(probe_fps)))))
        cap.set(_cv2.CAP_PROP_POS_FRAMES, i0)
        times = []
        frames = []
        idx = i0
        while idx <= i1:
            ok, fr = cap.read()
            if not ok:
                break
            times.append(idx / src_fps)
            gray = _cv2.cvtColor(fr, _cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            m = max(h, w)
            if m > int(max_dim):
                scale = float(max_dim) / m
                gray = _cv2.resize(gray, (int(round(w * scale)),
                                          int(round(h * scale))),
                                   interpolation=_cv2.INTER_AREA)
            frames.append(gray)
            idx += 1
            # sparse: skip step-1 frames without retrieve (grab is cheap
            # relative to BGR decode + resize)
            for _ in range(step - 1):
                if not cap.grab():
                    idx = i1 + 1
                    break
                idx += 1
        return times, frames
    finally:
        try:
            cap.release()
        except Exception:
            pass


def visually_static_spans(raw_path, spans, params=None):
    """Split each candidate span at moving stretches; keep the static
    sub-spans (each >= min_idle seconds after edge_pad shrink).

    Never raises: decode failure on a span drops that span (conservative
    -- worst case the output is merely longer than it could have been).
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    if _cv2 is None:
        return list(spans)   # no cv2: gate can't run -> pass through
    out = []
    min_idle = float(p["min_idle"])
    # Shrink surviving sub-spans by half the probe step so a ramp does
    # not overlap the first-moving frame after a genuinely static stretch.
    edge_pad = 0.5 / max(1e-6, float(p["motion_probe_fps"]))
    for span in spans:
        if isinstance(span, dict):
            a = float(span["start"])
            b = float(span["end"])
            extra = {k: v for k, v in span.items() if k not in ("start", "end")}
        else:
            a, b = float(span[0]), float(span[1])
            extra = {}
        if b - a < min_idle:
            continue
        try:
            times, frames = _probe_frames_in_span(
                raw_path, a, b,
                probe_fps=float(p["motion_probe_fps"]),
                max_dim=int(p["motion_max_probe_dim"]))
        except Exception:
            times, frames = [], []
        if not frames:
            # decode failure or empty probe -> drop this span (conservative)
            continue
        if len(frames) < 2:
            # only one sample fits (too short to gate meaningfully) -- keep
            continue
        mask = motion_mask(frames,
                           thresh_frac=float(p["motion_frac"]),
                           diff_thresh=int(p["motion_diff_thresh"]),
                           menubar_frac=float(p["motion_menubar_frac"]))
        subs = static_sub_spans(times, mask, min_idle, edge_pad=edge_pad)
        for (sa, sb) in subs:
            item = {"start": sa, "end": sb}
            item.update(extra)
            out.append(item)
    return out


# ---- plan assembly ------------------------------------------------------


def plan_speed_spans(activity_t, drags, duration, rate, params=None,
                     silence=None, overrides=None, motion_static=None):
    """Compose detection + gates + user overrides into the final span list.

    Returns [{start, end, rate}] sorted, non-overlapping. Rate applied to
    each auto span is `rate`; a `force` override may carry its own rate.

    silence: None means "unknown/no probe" (v1 = respect nothing sped,
    conservative). [] means "no silence found" (also nothing sped, since
    intersect with [] is []). A list of spans intersects with idle to keep
    only silent stretches -- narration guard.

    motion_static: optional callable(spans) -> spans that keeps only the
    visually-static sub-spans of each input (see visually_static_spans;
    the plan calls this AFTER silence gating so we only decode inside
    the already-narrowed candidates -- proportional to remaining idle,
    not total clip length). Skipped when None. Force overrides are
    inserted AFTER this gate so an author-forced range is not vetoed by
    on-screen motion the author explicitly wants time-lapsed (e.g. a
    long file download whose animation the viewer doesn't need at 1x).

    overrides: [{mode: "off"|"force", start, end, rate?}]. "off" ranges
    subtract from auto detection; "force" ranges are added unconditionally.
    """
    p = dict(DEFAULTS)
    p.update(params or {})
    rate = max(1.0, float(rate))
    idle = idle_candidates(activity_t, drags, duration,
                           pad=p["pad"], min_idle=p["min_idle"])
    if silence is not None:
        idle = intersect_spans(idle, silence)
        idle = [(a, b) for (a, b) in idle if b - a >= p["min_idle"]]
    spans = [{"start": a, "end": b, "rate": rate} for (a, b) in idle]
    if motion_static is not None and spans:
        gated = motion_static(spans)
        # motion_static returns dicts that may or may not carry rate; if
        # not, re-attach the auto rate.
        spans = []
        for s in gated:
            if isinstance(s, dict):
                a = float(s["start"]); b = float(s["end"])
                r = float(s.get("rate", rate))
                spans.append({"start": a, "end": b, "rate": r})
            else:
                a, b = float(s[0]), float(s[1])
                spans.append({"start": a, "end": b, "rate": rate})

    off = union_spans([(o.get("start"), o.get("end"))
                       for o in (overrides or [])
                       if o.get("mode") == "off"
                       and o.get("start") is not None
                       and o.get("end") is not None])
    if off:
        remain = subtract_spans(
            [(s["start"], s["end"]) for s in spans], off)
        spans = [{"start": a, "end": b, "rate": rate} for (a, b) in remain
                 if b - a >= p["min_idle"]]

    for o in (overrides or []):
        if o.get("mode") != "force":
            continue
        try:
            a = float(o["start"])
            b = float(o["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if b <= a:
            continue
        r = o.get("rate")
        r = max(1.0, float(r)) if r is not None else rate
        spans.append({"start": a, "end": b, "rate": r})

    spans.sort(key=lambda s: (s["start"], -s["rate"]))
    merged = []
    for s in spans:
        if merged and s["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], s["end"])
            merged[-1]["rate"] = max(merged[-1]["rate"], s["rate"])
        else:
            merged.append(dict(s))

    max_spans = int(p["max_spans"])
    if len(merged) > max_spans:
        merged = sorted(merged, key=lambda s: -(s["end"] - s["start"]))[:max_spans]
        merged.sort(key=lambda s: s["start"])
    return merged


# ---- TimeMap ------------------------------------------------------------


class TimeMap(object):
    """Piecewise time warp: identity outside sped spans, quintic-ramped
    inside them (entry ramp -> plateau at speed r -> exit ramp).

    `warp(t)` is C1-continuous, strictly monotonic (g = dtau/dt in
    [1/r, 1] > 0), and thus invertible. Empty spans -> the exact identity
    (see `identity`), so every downstream consumer stays bit-exact when
    the feature is off.

    `cuts` (ripple delete) are source spans whose output-time contribution
    is exactly ZERO: no frame inside one emits, the audio graph omits the
    range entirely, and warp is flat (weakly monotone, NOT invertible)
    across it -- a seam tau belongs to the cut's start (the earlier kept
    segment, matching the segments.py tie-break). A cut is an explicit
    piece kind, never rate=inf: inf would infinite-loop `atempo_chain`
    and trip `segments_for_audio`'s dtau==0 -> rate_eff=1.0 fallback into
    splicing the removed audio back in. Cuts WIN over speed spans: any
    overlap is subtracted from the span before pieces are built. Callers
    should pass cuts through `quantize_cut_spans` first so video and
    audio carve identical frame-grid boundaries.

    Duration semantics: when `duration` is given, spans are clipped into
    [0, duration] first, the map extends identity out to `duration`, and
    `output_duration` = warp(duration). When omitted, the map is defined
    on [0, max_span_end].
    """

    __slots__ = ("_spans", "_cuts", "_ramp", "_duration", "_pieces",
                 "_output_duration")

    def __init__(self, spans, ramp=0.5, duration=None, cuts=None):
        spans = spans or []
        cleaned = []
        for s in spans:
            try:
                a = float(s["start"])
                b = float(s["end"])
                r = float(s["rate"])
            except (KeyError, TypeError, ValueError):
                continue
            if r <= 1.0 + 1e-9 or b <= a:
                continue
            if duration is not None:
                dur = max(0.0, float(duration))
                a = max(0.0, min(a, dur))
                b = max(a, min(b, dur))
            if b > a:
                cleaned.append([a, b, r])
        # merge overlaps: sort by start, max-rate-wins
        cleaned.sort(key=lambda x: (x[0], -x[2]))
        merged = []
        for a, b, r in cleaned:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
                merged[-1][2] = max(merged[-1][2], r)
            else:
                merged.append([a, b, r])
        cut_list = []
        for c in (cuts or []):
            if isinstance(c, dict):
                try:
                    a, b = float(c["start"]), float(c["end"])
                except (KeyError, TypeError, ValueError):
                    continue
            else:
                try:
                    a, b = float(c[0]), float(c[1])
                except (TypeError, ValueError, IndexError):
                    continue
            a = max(0.0, a)
            if duration is not None:
                dur = max(0.0, float(duration))
                a = min(a, dur)
                b = min(b, dur)
            if b > a:
                cut_list.append((a, b))
        cut_list = union_spans(cut_list)
        if cut_list:
            # cut wins over speedup: remove cut ranges from the speed
            # spans (splitting where a cut lands mid-span) so no source
            # instant is claimed by both region kinds.
            survived = []
            for a, b, r in merged:
                for ka, kb in subtract_spans([(a, b)], cut_list):
                    if kb - ka > 1e-9:
                        survived.append([ka, kb, r])
            merged = survived
        self._cuts = cut_list
        # per-span d clipped to (b - a)/2 for short spans (spec: ramp
        # shrinks so entry and exit exactly meet)
        self._spans = [(a, b, r, min(float(ramp), (b - a) / 2.0))
                       for (a, b, r) in merged]
        self._ramp = float(ramp)
        self._duration = None if duration is None else max(0.0, float(duration))
        self._pieces = self._build_pieces()
        self._output_duration = (self._pieces[-1][1]
                                 + self._piece_len_tau(self._pieces[-1],
                                                       self._piece_end(-1)))

    # --- piece table --------------------------------------------------

    def _build_pieces(self):
        """Ordered list of (t_start, tau_start, kind, r, d, a_ref).

        kind: 'id' (identity), 'pl' (plateau at 1/r), 'en'/'ex' (ramps),
        'cut' (zero output time -- tau flat across the piece).
        a_ref = span-start `a` (needed by 'en'/'ex' to compute u).
        """
        pieces = []
        t = 0.0
        tau = 0.0
        regions = [(a, b, "speed", r, d) for (a, b, r, d) in self._spans]
        regions += [(a, b, "cut", 1.0, 0.0) for (a, b) in self._cuts]
        regions.sort(key=lambda x: x[0])
        for a, b, kind, r, d in regions:
            if a > t:
                pieces.append((t, tau, "id", 1.0, 0.0, 0.0))
                tau += (a - t)
                t = a
            if kind == "cut":
                # A splice: no ramp (a hard cut has no ease), no output
                # time. tau stays flat over [a, b); the seam instant
                # belongs to the EARLIER kept segment.
                pieces.append((a, tau, "cut", 1.0, 0.0, 0.0))
                t = b
                continue
            if d > 0.0:
                pieces.append((a, tau, "en", r, d, a))
                tau += d * (1.0 + 1.0 / r) / 2.0
                t = a + d
            pl_end = b - d
            if pl_end > t + 1e-12:
                pieces.append((t, tau, "pl", r, 0.0, 0.0))
                tau += (pl_end - t) / r
                t = pl_end
            if d > 0.0:
                pieces.append((b - d, tau, "ex", r, d, b - d))
                tau += d * (1.0 + 1.0 / r) / 2.0
                t = b
        if self._duration is not None and self._duration > t:
            pieces.append((t, tau, "id", 1.0, 0.0, 0.0))
            # tau tail added when computing output_duration below via _piece_end
        if not pieces:
            pieces.append((0.0, 0.0, "id", 1.0, 0.0, 0.0))
        return pieces

    def _piece_end(self, idx):
        """t at the end of piece[idx] (== start of the next, or duration
        for the tail)."""
        if idx == -1:
            idx = len(self._pieces) - 1
        if idx + 1 < len(self._pieces):
            return self._pieces[idx + 1][0]
        # last piece: extends to _duration (if given) or its own start
        if self._duration is not None:
            return max(self._duration, self._pieces[idx][0])
        # no duration and no span past this: identity piece with zero
        # length (only happens for an empty-spans + duration=None map)
        return self._pieces[idx][0]

    def _piece_len_tau(self, piece, t_end):
        t_start, tau_start, kind, r, d, ref = piece
        dt = max(0.0, t_end - t_start)
        if dt == 0.0:
            return 0.0
        if kind == "id":
            return dt
        if kind == "cut":
            return 0.0
        if kind == "pl":
            return dt / r
        if kind == "en":
            u = min(1.0, dt / d) if d > 0 else 1.0
            return d * (u - (1.0 - 1.0 / r) * _Q(u))
        if kind == "ex":
            u = min(1.0, dt / d) if d > 0 else 1.0
            return d * (u / r + (1.0 - 1.0 / r) * _Q(u))
        return dt

    # --- public API ---------------------------------------------------

    @property
    def identity(self):
        return not self._spans and not self._cuts

    @property
    def output_duration(self):
        return self._output_duration

    @property
    def spans(self):
        return [(a, b, r) for (a, b, r, _d) in self._spans]

    @property
    def cut_spans(self):
        """[(a, b)] union-merged cut ranges in SOURCE seconds."""
        return list(self._cuts)

    def keep_mask(self, times):
        """bool array over `times`: True = outside every cut.

        Half-open seam ownership: t == cut start is DROPPED, t == cut end
        is KEPT -- the seam belongs to the earlier kept segment, matching
        the segments.py tie-break, so any future output->source inverse
        agrees with this mask at seams.
        """
        times = np.asarray(times, dtype=float)
        keep = np.ones(times.shape, dtype=bool)
        for a, b in self._cuts:
            keep &= ~((times >= a) & (times < b))
        return keep

    def _warp_scalar(self, t):
        if t <= 0.0:
            return float(t)
        pieces = self._pieces
        # linear scan is O(S); binary search below is O(log S); either
        # is negligible next to the render loop.
        lo, hi = 0, len(pieces) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if pieces[mid][0] <= t:
                lo = mid
            else:
                hi = mid - 1
        p = pieces[lo]
        return p[1] + self._piece_len_tau(p, t)

    def warp(self, t):
        """Scalar or 1-D array warp t -> tau. Vectorized for arrays."""
        if np.isscalar(t):
            return self._warp_scalar(float(t))
        arr = np.asarray(t, dtype=float)
        if arr.size == 0:
            return np.array([], dtype=float)
        if self.identity:
            return arr.copy()
        flat = arr.ravel()
        out = np.empty_like(flat)
        for i in range(flat.size):
            out[i] = self._warp_scalar(float(flat[i]))
        return out.reshape(arr.shape)

    def warp_spans(self, spans):
        """Warp start/end of a list of {start, end, ...} dicts or tuples.

        Returns copies (dicts get a shallow copy with start/end replaced;
        tuples become (a', b')). Non-recognized items pass through.
        """
        out = []
        for s in spans or []:
            if isinstance(s, dict) and "start" in s and "end" in s:
                d = dict(s)
                d["start"] = float(self._warp_scalar(float(s["start"])))
                d["end"] = float(self._warp_scalar(float(s["end"])))
                out.append(d)
            elif isinstance(s, (tuple, list)) and len(s) >= 2:
                a = float(self._warp_scalar(float(s[0])))
                b = float(self._warp_scalar(float(s[1])))
                out.append((a, b) if isinstance(s, tuple) else [a, b])
            else:
                out.append(s)
        return out

    def slowness(self, times):
        """dtau/dt at each time (vectorized). Identity returns all-1.0."""
        times = np.asarray(times, dtype=float)
        g = np.ones_like(times)
        if self.identity or times.size == 0:
            return g
        for a, b, r, d in self._spans:
            if d > 0.0:
                mask = (times >= a) & (times < a + d)
                if mask.any():
                    u = (times[mask] - a) / d
                    g[mask] = 1.0 - (1.0 - 1.0 / r) * _q_vec(u)
                mask = (times >= b - d) & (times <= b)
                if mask.any():
                    u = (times[mask] - (b - d)) / d
                    g[mask] = 1.0 / r + (1.0 - 1.0 / r) * _q_vec(u)
            # plateau
            pl_lo = a + d
            pl_hi = b - d
            if pl_hi > pl_lo:
                mask = (times >= pl_lo) & (times < pl_hi)
                if mask.any():
                    g[mask] = 1.0 / r
            elif d == 0.0:
                mask = (times >= a) & (times <= b)
                if mask.any():
                    g[mask] = 1.0 / r
        for a, b in self._cuts:
            # dtau/dt is 0 inside a cut (no output time passes). Half-open
            # to match keep_mask: the seam instant reads as kept.
            mask = (times >= a) & (times < b)
            if mask.any():
                g[mask] = 0.0
        return g

    def emission(self, frame_times):
        """(emit: bool[T], out_ord: int[T], n_out_total: int) for a source-
        time frame grid.

        `emit[i]` = True iff source frame i produces an output frame.
        `out_ord[i]` = 0-based output ordinal AT this source frame (== the
        index that emit-frame i writes to; for dropped frames it is the
        most recently emitted ordinal).

        `n_out_total` = total output frames the audio graph will produce
        for the whole clip. Comes from the ANALYTIC `output_duration *
        fps`, not from the sum of discretely-sampled slowness -- the two
        can disagree by up to O(1/fps) due to rectangle-vs-analytic
        quadrature error over ramp regions, and that ~1-frame drift used
        to leak into the audio filter graph (audio built from the exact
        closed-form warp; video built from the discrete cumsum). Now the
        two agree by construction.

        Identity path: emit all-True, out_ord == arange, n_out_total == T
        -- bit-exact with the pre-feature render loop.
        """
        frame_times = np.asarray(frame_times, dtype=float)
        T = frame_times.size
        if T == 0:
            return (np.array([], dtype=bool),
                    np.array([], dtype=np.int64), 0)
        if self.identity:
            return (np.ones(T, dtype=bool),
                    np.arange(T, dtype=np.int64), T)
        # Sample the CLOSED-FORM warp on the source-frame grid; the fps
        # is recovered from the grid spacing so the map stays fps-agnostic.
        # Each source frame i "covers" [t_i, t_{i+1}] of source time; we
        # key emission on the END of that interval (tau_end) so the LAST
        # source frame reaches the true clip end -- if we used tau[i]
        # instead, the last output frame (ordinal n_out_total - 1)
        # emerges past the last sampled source time and would silently
        # drop, leaving audio one frame longer than video and A/V-desync
        # every render.
        tau = self.warp(frame_times)
        dt = float(frame_times[1] - frame_times[0]) if T > 1 else 1.0
        fps = 1.0 / dt if dt > 0 else 1.0
        tau_end = np.empty_like(tau)
        tau_end[:-1] = tau[1:]
        tau_end[-1] = self.output_duration
        # `completed[i]` = number of output frames whose START lies AT or
        # BEFORE the end of source frame i's coverage.
        completed = np.floor(fps * tau_end + 1e-9).astype(np.int64)
        prev = np.concatenate(([0], completed[:-1]))
        emit = completed > prev
        out_ord = np.maximum(0, completed - 1)
        n_out_total = int(completed[-1])
        if n_out_total < 1:
            n_out_total = 1
        return emit, out_ord, n_out_total

    def segments_for_audio(self, t0, t1):
        """[(loc_start, loc_end, rate_eff)] covering [0, t1-t0] EXCEPT cut
        ranges, which are OMITTED entirely (no tuple -- the concat butts
        their neighbours together). A cut-free map still covers the window
        gaplessly, exactly as before cuts existed.

        `t0`/`t1` are SOURCE-time bounds (typically the trim window and,
        by construction, the same window the encoder gives ffmpeg with
        -ss/-t). Coordinates in the returned tuples are TRIMMED-LOCAL
        seconds (= source-t minus t0). Each span's `rate_eff` matches the
        video warp at the span boundaries exactly:

            rate_eff = (b - a) / (tau(b) - tau(a))

        Identity segments carry rate_eff == 1.0. When the map is identity
        the whole window is one such segment (callers may still short-
        circuit and skip the retime filter entirely -- see render._encode_cmd,
        whose gate must treat a GAPPED all-1.0 list as needing the filter
        graph: the gaps are where the removed audio would otherwise stay).
        """
        t0 = float(t0)
        t1 = float(t1)
        if t1 <= t0:
            return []
        regions = [(a, b, r) for (a, b, r, _d) in self._spans]
        regions += [(a, b, None) for (a, b) in self._cuts]  # None = cut
        regions.sort(key=lambda x: x[0])
        out = []
        prev = t0
        for a, b, r in regions:
            aa = max(a, t0)
            bb = min(b, t1)
            if bb <= aa:
                continue
            if aa > prev + 1e-9:
                out.append((prev - t0, aa - t0, 1.0))
            if r is None:
                # cut: emit nothing, advance past the removed range
                prev = bb
                continue
            dtau = self._warp_scalar(bb) - self._warp_scalar(aa)
            rate_eff = ((bb - aa) / dtau) if dtau > 1e-9 else 1.0
            out.append((aa - t0, bb - t0, rate_eff))
            prev = bb
        if prev < t1 - 1e-9:
            out.append((prev - t0, t1 - t0, 1.0))
        # collapse near-zero segments (numeric hygiene on the trim edges)
        return [(a, b, r) for (a, b, r) in out if b - a > 1e-6]


# ---- audio atempo decomposition -----------------------------------------


def atempo_chain(rate):
    """Decompose an audio rate (>= 1) into a list of atempo factors, each
    in the safe (0.5, 2.0] range ffmpeg documents.

    Returns [] for rates within 1e-6 of 1.0 (no filter needed).
    """
    r = float(rate)
    if not np.isfinite(r):
        # Belt: a cut must never reach the audio chain as a rate (inf
        # would loop the 2.0-splitting below forever). Cuts are OMITTED
        # segments in TimeMap.segments_for_audio, so this only fires on
        # a caller bug -- fail loud, not hang.
        raise ValueError("atempo rate must be finite, got {!r}".format(rate))
    if r <= 1.0 + 1e-6:
        return []
    steps = []
    # split off 2.0 factors from the top while r stays > 2.0 by a margin
    while r > 2.0 + 1e-9:
        steps.append(2.0)
        r /= 2.0
    if abs(r - 1.0) > 1e-6:
        steps.append(r)
    return steps
