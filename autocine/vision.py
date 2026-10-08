"""Visual typing anchor: find WHERE typing happened by watching the video.

The last-click heuristic for anchoring typing zooms fails whenever the user
types into a field that was already focused (click the sidebar, then type
into the chat box) -- the camera zooms toward the stale click while the
text appears elsewhere. But the renderer HAS the video: during a typing
burst the caret/text region is almost the only thing changing on screen,
so accumulating frame differences across the burst window and locating the
DOMINANT changing component finds the real typing location -- no
Accessibility APIs, works retroactively on any recording with key events
(validated on a real session: found the chat input 1400px from the
misleading click).

Guards make it conservative (a wrong visual anchor outranks a correct
click anchor downstream, so every ambiguous case must return None):
- a global-motion window (scrolling, video playing) yields None;
- near-zero or too-diffuse change yields None;
- multiple comparably-strong changing regions (typing here, a blinking
  terminal cursor or spinner there) yield None unless one component
  clearly dominates -- the centroid of a UNION of disjoint blobs would
  land where nothing changed at all;
- decode failures never raise.

Costs are bounded: diffs are accumulated streamingly (previous frame +
accumulator only, never a frame list) and only the first _MAX_WINDOW_SEC
of a burst is analyzed -- the anchor is where typing STARTS, and an
unbounded window would decode minutes of video inside the editor's first
preview request.
"""

import threading
from collections import OrderedDict

import cv2
import numpy as np

_DIFF_THRESH = 12        # per-pixel |delta| at/below this is compression noise
_MENUBAR_FRAC = 0.03     # exclude the top strip (menu-bar clock flicker)
_MAX_CHANGED_FRAC = 0.08  # more of the frame changing -> scroll/video, not typing
_MIN_CHANGED_FRAC = 1e-5  # less -> nothing visibly happened; no anchor
_MAX_MASK_FRAC = 0.35    # half-peak mask spanning more of the frame -> ambiguous
_DOMINANCE = 2.0         # top component must carry 2x the energy of the next
_SAMPLE_STEP = 0.3       # s between sampled frames inside a burst window
_PRE_PAD = 0.2           # s sampled before the first key tick
_POST_PAD = 0.35         # s sampled after the last tick (trailing caret/text)
_MAX_WINDOW_SEC = 10.0   # analyze at most this much of a burst (from its start)
_BLUR_SIGMA = 25         # px; merges caret + letters into one blob
_CACHE_MAX = 8


class _DiffAccumulator(object):
    """Streaming |frame delta| accumulator: holds only the previous frame
    and the running accumulation, never a frame list (burst windows are
    unbounded in principle; memory must not scale with them)."""

    def __init__(self):
        self._prev = None
        self._acc = None
        self._n = 0

    def push(self, gray):
        cur = np.asarray(gray, dtype=np.int16)
        if self._prev is not None:
            d = np.abs(cur - self._prev).astype(np.float32)
            d[d <= _DIFF_THRESH] = 0.0
            self._acc = d if self._acc is None else self._acc + d
        self._prev = cur
        self._n += 1

    def anchor(self, menubar_frac=_MENUBAR_FRAC):
        """Dominant-component centroid (x, y), or None. Runs all gates."""
        if self._acc is None or self._n < 2:
            return None
        acc = self._acc
        h, w = acc.shape
        acc[:int(round(h * menubar_frac)), :] = 0.0

        changed = float((acc > 0).mean())
        if changed < _MIN_CHANGED_FRAC or changed > _MAX_CHANGED_FRAC:
            return None

        blur = cv2.GaussianBlur(acc, (0, 0), _BLUR_SIGMA)
        peak = float(blur.max())
        if peak <= 0.0:
            return None
        mask = (blur > peak * 0.5)
        if float(mask.mean()) > _MAX_MASK_FRAC:
            return None   # change is everywhere-ish; no single typing locus

        # Disjoint half-peak regions (typing here, a blinking terminal
        # cursor there) must NOT be centroid-averaged into empty space:
        # pick the component carrying the most accumulated energy, and only
        # trust it when it clearly dominates the runner-up.
        n_comp, labels = cv2.connectedComponents(mask.astype(np.uint8), 8)
        if n_comp <= 1:
            return None
        energies = np.zeros(n_comp, np.float64)
        for c in range(1, n_comp):
            energies[c] = float(blur[labels == c].sum())
        order = np.argsort(energies)[::-1]
        best = int(order[0])
        if len(order) > 1 and energies[int(order[1])] > 0:
            if energies[best] < _DOMINANCE * energies[int(order[1])]:
                return None   # two credible loci -> ambiguous, stay safe
        sel = labels == best
        weights = blur * sel
        total = float(weights.sum())
        if total <= 0.0:
            return None
        ys, xs = np.nonzero(sel)
        wsel = blur[ys, xs]
        return (float((xs * wsel).sum() / total),
                float((ys * wsel).sum() / total))


def anchor_from_frames(frames, menubar_frac=_MENUBAR_FRAC):
    """Anchor for an in-memory frame list (the pure, testable entry point).

    frames: >= 2 grayscale (H, W) arrays sampled across a typing burst, in
    time order.
    """
    if not frames or len(frames) < 2:
        return None
    acc = _DiffAccumulator()
    for f in frames:
        acc.push(f)
    return acc.anchor(menubar_frac=menubar_frac)


def _burst_anchor(cap, fps, n_frames, start, end):
    """Stream frames across the (capped) burst window into the accumulator.

    One seek, then sequential read/grab -- and the window is clamped to
    _MAX_WINDOW_SEC from the burst start: the anchor is where typing
    begins, and grab() still decodes every inter-coded frame it skips, so
    an uncapped window would decode arbitrary amounts of video.
    """
    t0 = max(0.0, float(start) - _PRE_PAD)
    t1 = min(float(end) + _POST_PAD, t0 + _MAX_WINDOW_SEC)
    i0 = int(t0 * fps)
    i1 = int(t1 * fps)
    if n_frames > 0:   # frame-countless containers: trust fps, EOF stops us
        i1 = min(max(0, n_frames - 1), i1)
    if i1 <= i0:
        return None
    step = max(1, int(round(_SAMPLE_STEP * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    acc = _DiffAccumulator()
    idx = i0
    while idx <= i1:
        ok, fr = cap.read()
        if not ok:
            break
        acc.push(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        idx += 1
        skip = min(step - 1, i1 - idx + 1)
        for _ in range(skip):
            if not cap.grab():
                idx = i1 + 1
                break
            idx += 1
    return acc.anchor()


def typing_visual_anchors(raw_path, bursts):
    """Per-burst visual anchors for `bursts` = [(start_s, end_s), ...] in
    media time. Returns a list aligned with bursts; entries are
    {"start", "end", "x", "y"} or None where no confident anchor exists.
    Any decode failure degrades to None (never raises) -- the click
    fallback in camera._plan_typing_intervals covers it.
    """
    if not bursts:
        return []
    try:
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            return [None] * len(bursts)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            out = []
            for (b0, b1) in bursts:
                try:
                    a = _burst_anchor(cap, fps, n, b0, b1)
                except Exception:
                    a = None
                out.append(None if a is None else
                           {"start": float(b0), "end": float(b1),
                            "x": a[0], "y": a[1]})
            return out
        finally:
            cap.release()
    except Exception:
        return [None] * len(bursts)


_cache = OrderedDict()
_cache_lock = threading.Lock()
_pending = {}


def cached_typing_anchors(raw_path, events_mtime, bursts):
    """typing_visual_anchors behind a thread-safe LRU with in-flight
    de-duplication.

    studio_app serves every request on its own thread (plus the render
    worker), and the editor fires /api/preview and /api/camera-path
    near-simultaneously on open -- without the lock the LRU races
    (get/move_to_end vs eviction), and without the pending-event dedup two
    concurrent misses would each pay the full decode. Never raises.
    """
    key = (str(raw_path), float(events_mtime or 0.0),
           tuple((round(float(a), 4), round(float(b), 4)) for a, b in bursts))
    waited = False
    while True:
        with _cache_lock:
            hit = _cache.get(key)
            if hit is not None:
                _cache.move_to_end(key)
                return hit
            ev = _pending.get(key)
            if ev is None or waited:
                # First arrival computes; a waiter whose wait timed out
                # takes over rather than looping forever on a computing
                # thread that may have died (double compute is harmless).
                if ev is None:
                    _pending[key] = threading.Event()
                break
        ev.wait(timeout=300.0)   # someone else is computing this key
        waited = True

    try:
        val = typing_visual_anchors(raw_path, bursts)
    except Exception:
        val = [None] * len(bursts)
    with _cache_lock:
        _cache[key] = val
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
        ev = _pending.pop(key, None)
    if ev is not None:
        ev.set()
    return val
