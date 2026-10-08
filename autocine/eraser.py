"""Erase the burned-in macOS cursor from a `cursor_mode: "system"` take.

A take recorded with the system cursor has the OS pointer COMPOSITED INTO THE
PIXELS -- there is no separate layer to switch off after the fact. Re-recording
a narration pass over that footage with a live pointer then puts two cursors on
screen. This module removes the recorded one.

**It is not object detection.** `events.jsonl` already logs where the pointer
was, and the logged position is accurate to a pixel or two against the frame at
that instant, so the pointer's location is KNOWN for every frame. The job is
"repaint a small box at a known place", not "find the arrow".

**Temporal recovery, not spatial inpainting.** A screen recording is mostly
static, so the pixels under the pointer were genuinely visible in other frames
-- before it arrived, and after it left. Those are the REAL pixels. Recovering
them beats hallucinating a plausible patch, and it is what makes the result
read as footage rather than as a smudge.

The structure follows from one observation: a pixel's coverage by the cursor
box is a set of disjoint RUNS. Inside one run `[a, b]` the pixel is hidden the
whole time, so there are exactly two clean samples that can repair any frame in
it -- the value just before the run (`pre`, observed at `a-1`) and the value
just after (`post`, observed at `b+1`). Both are constant across the run. So
the per-frame fill is a choice between two values, not a search.

## The three passes

1. **Schedule** (`boxes_for_track`, no decode). Turn the event track into a
   per-frame integer box. Pure rect arithmetic; a frame's onset/offset pixel
   sets are `box[i] \\ box[i-1]` and `box[i-1] \\ box[i]`, computed as up to
   four sub-rectangles (`_rect_minus`) rather than by masking the whole
   visited region -- that is what keeps this O(box) per frame instead of
   O(screen), which on a 2880x1800 take is a 5-megapixel scan 14,000 times.
2. **Prepass** (`plan`, ONE extra decode of the source). Collects the `post`
   samples, which is the half of the data a forward render cannot know, and
   seeds the `pre` plates up to the trim start.
3. **Apply** (`step`/`observe`/`erase`, riding the render's own decode). Keeps
   the `pre` plate current by sampling one frame ahead of each run start, and
   repaints each frame's cursor.

## The box is a SEARCH region, not the replacement region

Only the connected blob of changed pixels reaching the hotspot is repainted
(`_footprint`) -- the cursor is opaque, connected, and drawn at a known point,
so that IS the cursor. The rest of the box is untouched footage that merely
happens to sit near the pointer, and overwriting it turns every imperfection
in the recovered plate into a full-box artifact. Measured before this existed:
a sample from the wrong side of a change put a slab of a dialog's rounded
corner into the middle of an Excel toolbar.

## What it refuses to fake

If the pointer parks over content that CHANGES while it is parked -- a playing
video, a dialog animating in, a document being scrolled -- the pixels
underneath never existed uncovered and no amount of temporal search will find
them. Two independent detectors catch it, and both were put here by a measured
failure rather than by anticipation:

* the **bracket test**, per pixel: `pre` and `post` are the same pixel seen
  before and after the run, so `pre != post` IS the statement "this changed
  while it was hidden";
* the **verification ring** (`_verify_ring`), per frame: agreement is NOT
  proof on its own. A pointer that sat still for seventeen minutes bracketed a
  document that scrolled away and back, both samples agreed on a paragraph the
  frame did not show, and the eraser confidently painted the wrong sentence
  over the right one -- worse than leaving the cursor there. Keeping the
  plates a little wider than the box gives a band of pixels that can be
  compared at the sample instant and at now, which settles both which side of
  a change a sample is on and whether either describes this scene at all.

What survives neither goes to `cv2.inpaint`. `stats()`/`report()` say how many
frames needed it, so falling back is never a silent outcome -- on footage this
cannot support (that seventeen-minute take) the honest answer is "all of
them", and the report is how anyone finds that out.

## Cost

One extra decode of the source -- the reason the whole feature is opt-in --
plus ~40 bytes/px of plates over the region the pointer actually visited. The
prepass only pays the BGR conversion on frames that owe a sample (`grab()`
elsewhere), which on a still take is a handful of frames and on a busy one is
most of them.
"""

import numpy as np

try:
    import cv2
except ImportError:                                   # pragma: no cover
    cv2 = None


# -- The box ------------------------------------------------------------
#
# In POINTS, converted per-axis at call time by the same `scale_x`/`scale_y`
# the rest of the renderer derives from the real file -- so this is correct
# under fractional Retina scaling and on a non-Retina display, and hardcoding
# pixels here would be wrong on both.
#
# Measured on this machine's 2x footage (tools/measure: diff a parked-cursor
# frame against the same frame after the pointer left): the arrow occupies
# dx [+1, +30], dy [+1, +40] px from the hotspot, drop shadow included. The
# hotspot is the arrow's TIP, so the arrow is entirely down-and-right of it --
# but an I-beam's hotspot is its CENTRE and a resize cursor's is the middle of
# its span, so the box has to open in every direction or those shapes lose
# their top half. Hence a small "back" and a large "forward", both generous:
# a box a few px too big costs a few more px of recovery, a box one px too
# small leaves a shadow fringe, and a fringe is what gives an erase away.
CURSOR_BOX_BACK_PT = 16.0
CURSOR_BOX_FWD_PT = 24.0

# A pixel is "unchanged across the run" when pre and post agree to this, per
# channel. Not zero: even a static screen is re-encoded every frame, so the
# same pixel drifts by a few levels of codec noise between two distant frames.
CURSOR_AGREE_TOL = 14

# How far a sample is trusted when the bracket test has refuted it, or when
# there is only one sample and so no test at all. Agreement overrides this
# entirely -- an agreed pixel is exact at any distance.
#
# Its floor is `CURSOR_POST_MIN_SEC`: the pixels that genuinely depend on it
# are the ones in runs too short to have earned a `post` sample, and those
# are at most that stale. Tried at exactly that floor (0.12) and it was
# measurably worse -- a clean repair over a file listing picked up a smudge
# because pixels a fifth of a second from a clean sample were pushed to the
# inpaint, while the frames it was meant to rescue (a dialog animating under
# the pointer) were not rescued at all. The ring veto handles those instead,
# and this stays generous.
CURSOR_STALE_SEC = 0.25

# Runs shorter than this get `pre` only, no bracket test. Their staleness is
# bounded by this same number, which is below the threshold of noticing.
#
# The bound exists because the record count is the number of COVERAGE ONSETS.
# Measured on a 4-minute take with a busy pointer: bracketing EVERY run costs
# 13.0M records (195 MB), 0.1s costs 2.0M (31 MB) and 0.4s costs 0.77M. The
# knee is exactly where you would expect -- almost all runs are the one-frame
# slivers a moving pointer sheds off its trailing edge, and none of them can
# go stale enough to see. Every run long enough to matter is bracketed.
CURSOR_POST_MIN_SEC = 0.1

# Hard ceiling on stored `post` records (~15 bytes each, so ~90 MB). A take
# adversarial enough to blow through this degrades to `pre`-only for the
# overflow rather than to an unbounded allocation; `stats()` says so.
CURSOR_MAX_POST_RECORDS = 6_000_000

# The cursor's own footprint inside the search box: how far the frame has to
# differ from its recovered plate to count as "the cursor drew here", and how
# many px to grow the result by. Both are deliberately generous downward --
# the drop shadow fades to a couple of levels above the background, and a
# missed shadow pixel is a visible fringe, while an extra repainted pixel is
# repainted with its own recovered value and therefore costs nothing.
CURSOR_FOOTPRINT_TOL = 6
CURSOR_FOOTPRINT_GROW = 2
# How far the footprint may reach past the last pixel the plate vouches for.
# The cursor's rim and shadow tail do sit on unrecoverable pixels sometimes;
# this is what still lets those come off, while keeping the reach bounded.
CURSOR_FOOTPRINT_RIM = 3

# The verification ring: how far outside the search box the plates are also
# kept, in POINTS.
#
# This is what makes the bracket test honest. `pre == post` was originally
# read as proof that the covered pixels never changed -- and measured on a
# real take, that is FALSE over a long park: a pointer sat still for 17
# minutes, the document under it scrolled away and back, and both samples
# agreed on text that was NOT what the frame showed in between. The eraser
# confidently painted the wrong sentence over the right one, which is a worse
# outcome than leaving the cursor alone.
#
# A time limit would be arbitrary (a pointer resting on a static toolbar for
# three minutes IS exactly recoverable). Evidence is not: the pixels just
# OUTSIDE the box are visible the whole time, so keeping the plates a little
# wider than the box gives the same pixels at the sample instant and at now.
# If that ring still matches, the neighbourhood did not change and the
# recovery stands however old it is; if it does not, this frame's plate
# belongs to a different scene and nothing in it is trusted.
CURSOR_VERIFY_MARGIN_PT = 7.0

# A ring pixel counts as CHANGED when its max-channel mismatch against the
# plate exceeds this; the plate is declared to belong to a different scene
# when more than `CURSOR_RING_MAX_FRAC` of the ring has changed.
#
# Measured on six frames judged by eye first, then scored (`frac > 24`, the
# better of the two plates): the three clean repairs came in at 0.000, 0.000
# and 0.010, the three the recording could not support at 0.022, 0.085 and
# 0.089. 0.05 sits between them with roughly 5x of margin on each side. The
# 0.022 is the one the ring genuinely cannot see -- a dialog corner that
# appeared INSIDE the box while its surroundings held still -- and it is
# caught, imperfectly, by the per-pixel bracket test instead.
CURSOR_RING_TOL = 24
CURSOR_RING_MAX_FRAC = 0.05

# How close in time a sample must be to survive a ring veto. Two frames: if
# the neighbourhood this plate came from is gone, the only samples still
# describing the frame are the ones taken essentially at the same instant.
CURSOR_VETOED_STALE_FRAMES = 2

# What to do with a pixel the recording never showed uncovered.
CURSOR_FALLBACK = "inpaint"

# `cv2.inpaint` context margin and radius, in px. The margin matters: when a
# whole box is unrecoverable there is no repaired content inside it to
# inpaint from, so the call is handed a slightly larger window whose border
# is real footage.
CURSOR_INPAINT_MARGIN = 16
CURSOR_INPAINT_RADIUS = 4

_BIG = np.int64(1 << 30)


def _param(params, key, default):
    if not isinstance(params, dict):
        return default
    v = params.get(key)
    if v is None:
        return default
    return v


def cursor_track(frame_times, ev_t, ev_x, ev_y):
    """Per-frame `(x, y)` pointer position, as a STEP function.

    `move` lines are only written when the pointer actually moves, so the
    position between two of them is the earlier one held -- the pointer is
    still on screen while it is still. Before the first sample and after the
    last, the nearest known position is held, matching how `render.py`
    already interpolates the spotlight follower.

    Held rather than interpolated on purpose: a linear interpolation invents
    positions the pointer never occupied, and `boxes_for_track` widens each
    frame's box over the events that fall inside it anyway, which covers the
    real motion without guessing at its shape.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    ev_t = np.asarray(ev_t, dtype=float)
    if ev_t.size == 0:
        return None, None
    order = np.argsort(ev_t, kind="stable")
    ev_t = ev_t[order]
    ev_x = np.asarray(ev_x, dtype=float)[order]
    ev_y = np.asarray(ev_y, dtype=float)[order]
    # side="right" - 1 is "the last event at or before this frame's instant".
    idx = np.searchsorted(ev_t, frame_times, side="right") - 1
    idx = np.clip(idx, 0, ev_t.size - 1)
    return ev_x[idx], ev_y[idx]


def boxes_for_track(frame_times, ev_t, ev_x, ev_y, W, H,
                    back_x, back_y, fwd_x, fwd_y, hot_pad=3, ring=0.0):
    """`(boxes, hots, tracked)`, each `(N, 4)` int32 `[x0,y0,x1,y1]` in W x H.

    `boxes` is the SEARCH region -- everywhere the cursor might have drawn
    this frame. `hots` is the much smaller rect the HOTSPOT itself could have
    been in, which is what `erase` seeds its connected-component search from.
    `tracked` is `boxes` grown by `ring`: the plates are kept over THAT, so
    every frame has a margin of pixels it holds a sample for AND can see
    uncovered right now -- see `CURSOR_VERIFY_MARGIN_PT`.

    Each is the union over every pointer position the frame could have
    caught: the held position at the frame's instant, plus every event inside
    the frame's own interval. A 60fps frame is exposed over 16.7ms and the
    pointer is logged at up to ~50Hz, so a fast drag really does put the arrow
    somewhere between two logged points -- a box pinned to the held position
    alone would then miss it by the distance travelled and leave a smear of
    arrow behind. Widening is cheap and is self-correcting: a pixel covered by
    the widened box for one frame has clean samples one frame away on both
    sides.
    """
    frame_times = np.asarray(frame_times, dtype=float)
    n = frame_times.size
    empty = np.zeros((0, 4), dtype=np.int32)
    if n == 0:
        return empty, empty, empty
    ev_t = np.asarray(ev_t, dtype=float)
    if ev_t.size == 0:
        return empty, empty, empty
    order = np.argsort(ev_t, kind="stable")
    ev_t = ev_t[order]
    ev_x = np.asarray(ev_x, dtype=float)[order]
    ev_y = np.asarray(ev_y, dtype=float)[order]

    held_x, held_y = cursor_track(frame_times, ev_t, ev_x, ev_y)
    lo_x = np.array(held_x, dtype=float)
    hi_x = np.array(held_x, dtype=float)
    lo_y = np.array(held_y, dtype=float)
    hi_y = np.array(held_y, dtype=float)

    # Fold in the events landing inside each frame's interval. `np.minimum.at`
    # is the scatter-reduce that lets this stay one vectorized pass instead of
    # a Python loop over ~10^5 events.
    edges = frame_times
    slot = np.searchsorted(edges, ev_t, side="right") - 1
    inside = (slot >= 0) & (slot < n)
    if inside.any():
        s = slot[inside]
        np.minimum.at(lo_x, s, ev_x[inside])
        np.maximum.at(hi_x, s, ev_x[inside])
        np.minimum.at(lo_y, s, ev_y[inside])
        np.maximum.at(hi_y, s, ev_y[inside])

    W, H = int(W), int(H)

    def _rects(mx0, my0, mx1, my1):
        out = np.empty((n, 4), dtype=np.int32)
        out[:, 0] = np.clip(np.floor(lo_x - mx0), 0, W)
        out[:, 1] = np.clip(np.floor(lo_y - my0), 0, H)
        out[:, 2] = np.clip(np.ceil(hi_x + mx1) + 1.0, 0, W)
        out[:, 3] = np.clip(np.ceil(hi_y + my1) + 1.0, 0, H)
        return out

    boxes = _rects(back_x, back_y, fwd_x, fwd_y)
    hots = _rects(hot_pad, hot_pad, hot_pad, hot_pad)
    tracked = _rects(back_x + ring, back_y + ring, fwd_x + ring, fwd_y + ring)
    # A box clamped to nothing (pointer off the captured area, or off a
    # window crop) is an EMPTY box, which the rest of this module already
    # treats as "no coverage this frame" -- not a degenerate 1px one.
    dead = (boxes[:, 2] <= boxes[:, 0]) | (boxes[:, 3] <= boxes[:, 1])
    boxes[dead] = 0
    hots[dead] = 0
    tracked[dead] = 0
    return boxes, hots, tracked


def _rect_minus(c, p):
    """`c` minus `p`, as up to four disjoint rects.

    Both are `[x0, y0, x1, y1]`. This is the whole reason the per-frame cost
    is O(box): the alternative -- a boolean mask over everything the pointer
    ever visited -- is O(screen) per frame, which on a 2880x1800 take is a
    5-megapixel scan 14,000 times over.
    """
    cx0, cy0, cx1, cy1 = int(c[0]), int(c[1]), int(c[2]), int(c[3])
    if cx1 <= cx0 or cy1 <= cy0:
        return []
    px0, py0, px1, py1 = int(p[0]), int(p[1]), int(p[2]), int(p[3])
    ix0, iy0 = max(cx0, px0), max(cy0, py0)
    ix1, iy1 = min(cx1, px1), min(cy1, py1)
    if ix1 <= ix0 or iy1 <= iy0:
        return [(cx0, cy0, cx1, cy1)]
    out = []
    if iy0 > cy0:
        out.append((cx0, cy0, cx1, iy0))
    if iy1 < cy1:
        out.append((cx0, iy1, cx1, cy1))
    if ix0 > cx0:
        out.append((cx0, iy0, ix0, iy1))
    if ix1 < cx1:
        out.append((ix1, iy0, cx1, iy1))
    return out


def _flat_indices(rects, rx0, ry0, rw):
    """Flat offsets into a `region`-shaped plate for a list of rects."""
    parts = []
    for (x0, y0, x1, y1) in rects:
        if x1 <= x0 or y1 <= y0:
            continue
        rows = np.arange(y0 - ry0, y1 - ry0, dtype=np.int64)[:, None] * rw
        cols = np.arange(x0 - rx0, x1 - rx0, dtype=np.int64)[None, :]
        parts.append((rows + cols).ravel())
    if not parts:
        return np.zeros(0, dtype=np.int64)
    if len(parts) == 1:
        return parts[0]
    return np.concatenate(parts)


def region_of(boxes):
    """Bounding box of every non-empty frame box, or None when there is no
    coverage at all (an events file with no positions, or a pointer that never
    entered the captured area)."""
    if boxes is None or len(boxes) == 0:
        return None
    b = np.asarray(boxes)
    live = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    if not live.any():
        return None
    b = b[live]
    return (int(b[:, 0].min()), int(b[:, 1].min()),
            int(b[:, 2].max()), int(b[:, 3].max()))


class CursorEraser(object):
    """Per-render eraser state. Build with `plan()`, never directly.

    The three entry points must be driven IN FRAME ORDER over the render's
    own decode:

    * `step(i)`   -- bookkeeping only, no pixels. MUST be called for every
                     source frame in the render window, including ones the
                     speed-up plan drops, because it is what opens and closes
                     runs. Skipping it leaves a pixel's `post` pointing at a
                     previous run's sample.
    * `observe(fr, i)` -- keep the `pre` plate current. Reads only pixels
                     OUTSIDE this frame's box, so it composes with `erase`
                     in either order.
    * `erase(fr, i)`   -- repaint this frame's box, in place.
    """

    def __init__(self, boxes, hots, tracked, region, start_idx, end_idx,
                 pre_val, pre_t, post_a, post_p, post_val, post_t,
                 stale_frames, ring_tol, ring_max_frac, agree_tol,
                 footprint_tol, footprint_grow, footprint_rim, fallback,
                 inpaint_margin, inpaint_radius, post_dropped=0):
        self.boxes = boxes          # what may be REPLACED
        self.hots = hots
        self.tracked = tracked      # what the plates COVER (boxes + ring)
        self.region = region
        self.start_idx = int(start_idx)
        self.end_idx = int(end_idx)
        rx0, ry0, rx1, ry1 = region
        self._rx0, self._ry0 = rx0, ry0
        self._rw, self._rh = rx1 - rx0, ry1 - ry0
        self._pre_val = pre_val                     # (rh, rw, 3) uint8
        self._pre_t = pre_t                         # (rh, rw) int32
        self._post_val = np.zeros_like(pre_val)
        self._post_t = np.full((self._rh, self._rw), -1, dtype=np.int32)
        # `post` records grouped by the frame their run starts on, so the
        # scatter in `step` is a slice rather than a search.
        self._post_a = post_a
        self._post_p = post_p
        self._post_rec_val = post_val
        self._post_rec_t = post_t
        self._post_cursor = 0
        self.post_dropped = int(post_dropped)
        self.stale_frames = int(stale_frames)
        self.ring_tol = int(ring_tol)
        self.ring_max_frac = float(ring_max_frac)
        self.agree_tol = int(agree_tol)
        self.footprint_tol = int(footprint_tol)
        self.footprint_grow = int(footprint_grow)
        self.footprint_rim = int(footprint_rim)
        self.fallback = fallback
        self.inpaint_margin = int(inpaint_margin)
        self.inpaint_radius = int(inpaint_radius)
        self._stepped = start_idx - 1
        self.frames_repaired = 0
        self.frames_fallback = 0
        self.frames_declined = 0
        self.px_repaired = 0
        self.px_fallback = 0

    # -- bookkeeping ----------------------------------------------------

    def _box(self, i):
        if i < 0 or i >= len(self.boxes):
            return (0, 0, 0, 0)
        b = self.boxes[i]
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def _hot(self, i):
        if i < 0 or i >= len(self.hots):
            return (0, 0, 0, 0)
        b = self.hots[i]
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def _tbox(self, i):
        if i < 0 or i >= len(self.tracked):
            return (0, 0, 0, 0)
        b = self.tracked[i]
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    def step(self, i):
        """Open the runs that start at frame `i`.

        Newly covered pixels get their `post` INVALIDATED first and then set
        from the record table if this run earned one. The invalidation is not
        optional: the plate is reused across runs, and a pixel entering a new
        run while still holding the previous run's `post` would be repaired
        from a sample belonging to a different moment entirely.
        """
        i = int(i)
        self._stepped = i
        onset = _rect_minus(self._tbox(i), self._tbox(i - 1))
        if onset:
            idx = _flat_indices(onset, self._rx0, self._ry0, self._rw)
            if idx.size:
                self._post_t.reshape(-1)[idx] = -1
        # Records are sorted by run-start frame, so everything for frame `i`
        # is one contiguous block at the cursor.
        a = self._post_a
        if self._post_cursor < a.size:
            lo = self._post_cursor
            hi = lo + int(np.searchsorted(a[lo:], i, side="right"))
            if hi > lo:
                sel = slice(lo, hi)
                p = self._post_p[sel]
                self._post_val.reshape(-1, 3)[p] = self._post_rec_val[sel]
                self._post_t.reshape(-1)[p] = self._post_rec_t[sel]
                self._post_cursor = hi

    def observe(self, frame_bgr, i):
        """Record the clean value of every pixel a run is about to cover.

        Called at frame `i`, it samples the pixels that `box[i+1]` will cover
        and `box[i]` does not -- so the sample is, by construction, the pixel
        the instant BEFORE it was hidden. That is `pre`, and taking it here
        rather than from a maintained whole-region plate is what keeps the
        per-frame cost proportional to the box rather than the screen.
        """
        i = int(i)
        nxt = _rect_minus(self._tbox(i + 1), self._tbox(i))
        if not nxt:
            return
        idx = _flat_indices(nxt, self._rx0, self._ry0, self._rw)
        if not idx.size:
            return
        sub = frame_bgr[self._ry0:self._ry0 + self._rh,
                        self._rx0:self._rx0 + self._rw]
        self._pre_val.reshape(-1, 3)[idx] = sub.reshape(-1, 3)[idx]
        self._pre_t.reshape(-1)[idx] = i

    # -- the repair -----------------------------------------------------

    def erase(self, frame_bgr, i):
        """Repaint frame `i`'s cursor box, in place. Returns True when it
        touched anything."""
        i = int(i)
        bx0, by0, bx1, by1 = self._box(i)
        if bx1 <= bx0 or by1 <= by0:
            return False
        sy = slice(by0 - self._ry0, by1 - self._ry0)
        sx = slice(bx0 - self._rx0, bx1 - self._rx0)
        pv = self._pre_val[sy, sx]
        pt = self._pre_t[sy, sx].astype(np.int64)
        qv = self._post_val[sy, sx]
        qt = self._post_t[sy, sx].astype(np.int64)

        has_pre = pt >= 0
        has_post = qt >= 0
        d_pre = np.where(has_pre, i - pt, _BIG)
        d_post = np.where(has_post, qt - i, _BIG)
        # A sample from the "wrong" side of now (the plate holding a value
        # from a run that has not opened yet) would read as a negative
        # distance; clamp so it can never win the comparison.
        d_pre = np.where(d_pre < 0, _BIG, d_pre)
        d_post = np.where(d_post < 0, _BIG, d_post)

        # Causally nearest wins by default, so a change in the underlying
        # content is picked up as fast as the recording allows...
        use_post = d_post < d_pre
        have = (d_pre < _BIG) | (d_post < _BIG)
        both = (d_pre < _BIG) & (d_post < _BIG)
        # ...but WHICH SIDE of a change we are on beats how close it is, and
        # the ring answers that from the current frame -- same pixels, both
        # instants. It can also answer "neither", which is the veto.
        era, live = self._verify_ring(frame_bgr, i, pv, qv)
        if era is not None and both.any():
            use_post = np.where(both, era, use_post)
        fill = np.where(use_post[:, :, None], qv, pv)
        # The bracket test: the same pixel, seen before the run and after it.
        # Agreement means the content sat still the whole time it was hidden.
        # Necessary but NOT sufficient on its own -- content can change and
        # change back, which is why `live` gates it (see the ring constant).
        agree = both & (np.abs(pv.astype(np.int16)
                               - qv.astype(np.int16)).max(axis=2)
                        <= self.agree_tol)
        d_near = np.minimum(d_pre, d_post)
        # `~both` is "one sample, so no bracket to test": a run that opened
        # before the clip did or has not closed by the end of it. With the
        # ring vouching for the plate that sample is the best evidence there
        # is, and refusing it would blur a pointer that simply parked and
        # stayed -- the single most ordinary thing a pointer does.
        good = have & (agree | (d_near <= self.stale_frames) | ~both)
        if not live:
            # The neighbourhood this plate came from is gone, so the bracket
            # test proves nothing and only samples from essentially this
            # instant survive. Everything else goes to the fallback -- the
            # cursor still comes out, it just comes out inpainted.
            good = good & (d_near <= CURSOR_VETOED_STALE_FRAMES)

        out = frame_bgr[by0:by1, bx0:bx1]
        fill = fill.astype(np.uint8)
        foot = self._footprint(out, fill, (bx0, by0, bx1, by1), i, good)
        if foot is None:
            # Nothing the plate can vouch for reaches the pointer, so there is
            # no evidence of where the cursor is -- and no licence to repaint.
            self.frames_declined += 1
            return False
        write = foot & good
        bad = foot & ~good
        n_write = int(write.sum())
        n_bad = int(bad.sum())
        if not (n_write or n_bad):
            return False
        np.copyto(out, fill, where=write[:, :, None])
        self.frames_repaired += 1
        self.px_repaired += n_write
        if n_bad:
            self.frames_fallback += 1
            self.px_fallback += n_bad
            if self.fallback == "inpaint":
                self._inpaint(frame_bgr, (bx0, by0, bx1, by1), bad)
        return True

    def _footprint(self, cur, fill, box, i, good):
        """Which pixels in the search box the CURSOR actually drew on.

        The box is deliberately generous, so most of it is untouched footage
        that happens to sit near the pointer. Overwriting all of it makes
        every imperfection in the recovered plate a full-box artifact -- which
        is exactly how a repair goes from invisible to a slab of the wrong
        shade sitting in the middle of a toolbar.

        So the box is treated as a search region and only the cursor is
        replaced. The cursor is a CONNECTED, opaque blob drawn at a known
        point, so: threshold `|current - recovered|`, dilate to pick up the
        antialiased rim and the drop shadow's faint tail, take connected
        components, and keep the ones reaching the hotspot rect.

        **The component may only grow through pixels the plate can vouch for**
        (`good`), and that restriction is the whole difference between a
        repair and corruption. "The frame differs from the plate here" means
        two completely different things depending on whether the plate is
        trustworthy: where it is, the difference IS the cursor; where it is
        not, the difference is just the plate being wrong, and following it is
        how the blob walks off the pointer and into real content.

        Measured on a full 4-minute export before this restriction: the median
        footprint was a correct 1331 px, but 14.6% of frames came out at 3-6x
        a real cursor and one frame inpainted 4668 px (69% of its box) across
        a live spreadsheet, destroying three numbers -- on a frame where the
        ring veto had ALREADY concluded that nothing was recoverable
        (`good` was 0.0% of the box). It knew, and damaged the frame anyway.
        With the restriction that frame yields an empty footprint and is left
        untouched: the pointer stays visible, which is a visible and honest
        outcome, where deleted spreadsheet cells are silent corruption.

        Returns None when nothing survives -- a pointer that left the display,
        a frame already identical to its own recovery, or one whose plate
        cannot support the erase at all.
        """
        bx0, by0, bx1, by1 = box
        d = np.abs(cur.astype(np.int16) - fill.astype(np.int16)).max(axis=2)
        changed = d > self.footprint_tol
        mask = (changed & good).astype(np.uint8)
        if not mask.any():
            return None
        if cv2 is None:                                # pragma: no cover
            return mask.astype(bool)
        if self.footprint_grow > 0:
            k = 2 * int(self.footprint_grow) + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        n_lab, lab = cv2.connectedComponents(mask, connectivity=8)
        if n_lab <= 1:
            return None
        hx0, hy0, hx1, hy1 = self._hot(i)
        hx0, hy0 = max(hx0, bx0), max(hy0, by0)
        hx1, hy1 = min(hx1, bx1), min(hy1, by1)
        if hx1 <= hx0 or hy1 <= hy0:
            return None
        seeds = np.unique(lab[hy0 - by0:hy1 - by0, hx0 - bx0:hx1 - bx0])
        seeds = seeds[seeds > 0]
        if not seeds.size:
            return None
        foot = np.isin(lab, seeds)
        # The cursor's own antialiased rim and the tail of its drop shadow can
        # land on pixels the plate cannot vouch for. Those are cursor and have
        # to come off, so the component is allowed a bounded reach into
        # untrusted territory -- bounded being the point: a few px of rim is a
        # fringe, an unbounded walk is the corruption above.
        if self.footprint_rim > 0:
            k = 2 * int(self.footprint_rim) + 1
            grown = cv2.dilate(foot.astype(np.uint8), np.ones((k, k), np.uint8))
            foot = foot | (grown.astype(bool) & changed)
        return foot

    def _verify_ring(self, frame_bgr, i, pre, post):
        """`(prefer_post, live)` from the ring of pixels around the box.

        The plates are kept `CURSOR_VERIFY_MARGIN_PT` wider than the box, so
        for every frame there is a band of pixels this holds a sample of AND
        can see uncovered right now. Comparing the SAME pixels at the sample
        instant and at this frame answers the two questions the temporal
        evidence cannot:

        * which era -- the candidate whose ring matches the frame is the one
          taken from this side of whatever changed;
        * whether either is usable at all. If NEITHER ring matches, the plate
          describes a scene that is gone, and `pre == post` proves nothing:
          measured on a real take, a pointer parked for 17 minutes bracketed
          a document that scrolled away and back, so both samples agreed on
          text the frame did not show. Without this the eraser painted the
          wrong sentence over the right one -- confidently, and much worse
          than leaving the cursor there.

        `(None, True)` when there is no ring to read (the box is flush with
        the frame edge): no opinion, and no grounds to veto either.
        """
        bx0, by0, bx1, by1 = self._box(i)
        tx0, ty0, tx1, ty1 = self._tbox(i)
        ring = _rect_minus((tx0, ty0, tx1, ty1), (bx0, by0, bx1, by1))
        if not ring:
            return None, True
        idx = _flat_indices(ring, self._rx0, self._ry0, self._rw)
        if not idx.size:
            return None, True
        cur = frame_bgr[self._ry0:self._ry0 + self._rh,
                        self._rx0:self._rx0 + self._rw].reshape(-1, 3)
        ref = cur[idx].astype(np.int16)
        pre_t = self._pre_t.reshape(-1)[idx]
        post_t = self._post_t.reshape(-1)[idx]

        def _changed(plate, valid):
            """Fraction of the ring this plate gets WRONG.

            A fraction, not a median: the thing being detected is a LOCAL
            change -- a dialog sliding in, a paragraph rewritten -- and a
            median over a ring that is mostly untouched background never
            moves for those. It moved for nothing on the take that motivated
            this, which is how a stale plate got through. Counting how much
            of the neighbourhood disagrees is the question actually being
            asked.
            """
            if not valid.any():
                return None
            d = np.abs(plate.reshape(-1, 3)[idx][valid].astype(np.int16)
                       - ref[valid]).max(axis=1)
            return float((d > self.ring_tol).mean())

        fp = _changed(self._pre_val, pre_t >= 0)
        fq = _changed(self._post_val, post_t >= 0)
        if fp is None and fq is None:
            return None, True
        best = min(v for v in (fp, fq) if v is not None)
        live = best <= self.ring_max_frac
        if fp is None or fq is None:
            return None, live
        return (fq < fp), live

    def _inpaint(self, frame_bgr, box, bad):
        """Spatial fallback for pixels the recording never showed uncovered.

        Deliberately last and deliberately small: it is the only step here
        that INVENTS pixels, so it only ever runs on the subset the temporal
        pass declared unrecoverable, and it runs after that pass has filled
        its neighbours with real content for it to grow from.
        """
        if cv2 is None:
            return
        bx0, by0, bx1, by1 = box
        h, w = frame_bgr.shape[:2]
        m = self.inpaint_margin
        gx0, gy0 = max(0, bx0 - m), max(0, by0 - m)
        gx1, gy1 = min(w, bx1 + m), min(h, by1 + m)
        sub = np.ascontiguousarray(frame_bgr[gy0:gy1, gx0:gx1])
        mask = np.zeros(sub.shape[:2], dtype=np.uint8)
        mask[by0 - gy0:by1 - gy0, bx0 - gx0:bx1 - gx0] = bad.astype(np.uint8) * 255
        fixed = cv2.inpaint(sub, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        frame_bgr[by0:by1, bx0:bx1] = fixed[by0 - gy0:by1 - gy0,
                                            bx0 - gx0:bx1 - gx0]

    # -- reporting ------------------------------------------------------

    def stats(self):
        px = max(1, self.px_repaired + self.px_fallback)
        return {
            "frames": self.frames_repaired,
            "frames_fallback": self.frames_fallback,
            "frames_declined": self.frames_declined,
            "px_fallback_frac": self.px_fallback / float(px),
            "post_dropped": self.post_dropped,
            "region": self.region,
        }

    def report(self):
        s = self.stats()
        if not s["frames"]:
            return "  cursor erase: nothing to erase (pointer never in frame)"
        line = ("  cursor erase: {} frames repaired, {} needed the spatial "
                "fallback ({:.1%} of repaired pixels)".format(
                    s["frames"], s["frames_fallback"], s["px_fallback_frac"]))
        if s["frames_declined"]:
            line += ("\n  cursor erase: {} frames LEFT ALONE -- the recording "
                     "never showed those pixels uncovered, so the pointer is "
                     "still in them".format(s["frames_declined"]))
        if s["post_dropped"]:
            line += "\n  cursor erase: {} run records over the cap, those " \
                    "runs used the forward sample only".format(
                        s["post_dropped"])
        return line


def box_padding(scale_x, scale_y, params=None):
    """`(back_x, back_y, fwd_x, fwd_y, ring)` in SOURCE px, from the
    point-space constants.

    Per-axis, from the scales `render.py` derives from the real file -- the
    same treatment `capture_window`'s rect gets, and for the same reason:
    a constant in pixels would be wrong on any display scaling but this one.
    """
    back = float(_param(params, "box_back_pt", CURSOR_BOX_BACK_PT))
    fwd = float(_param(params, "box_fwd_pt", CURSOR_BOX_FWD_PT))
    ring = float(_param(params, "verify_margin_pt", CURSOR_VERIFY_MARGIN_PT))
    sx = abs(float(scale_x)) or 1.0
    sy = abs(float(scale_y)) or 1.0
    return back * sx, back * sy, fwd * sx, fwd * sy, ring * max(sx, sy)


def plan(capture, crop_fn, boxes, hots, tracked, src_fps, start_idx=0,
         end_idx=None, walk_start=0, walk_limit=None, params=None,
         progress=None):
    """Run the prepass and return a ready `CursorEraser`, or None.

    `capture` is anything with cv2's `grab()`/`read()`, already positioned at
    `walk_start`; `crop_fn(frame, i)` maps a decoded frame into the render's
    SOURCE space (the capture-window crop, its geometry track, then the editor
    crop) so every sample this takes lands in the same coordinates as `boxes`.
    Both are injected rather than opened here, which is what lets the tests
    drive this with a list of numpy arrays and no video file.

    `start_idx`/`end_idx` are the frames the caller will actually RENDER;
    `walk_start`/`walk_limit` bound how much of the source this is allowed to
    read looking for samples. An export walks the whole file
    (`walk_start=0`, no limit) because a clean sample is worth having however
    far away it is. The editor's single-frame preview cannot afford that, so
    it walks a window around the frame and accepts a worse result outside it
    -- see `render._preview_cursor_eraser`.

    None means "nothing to do": no coverage in the walked span, which is a
    session whose events never put the pointer inside the captured area.

    The single decode this costs is spent on the half of the problem a
    forward render cannot solve -- `post`, the clean sample from AFTER a run.
    `pre` needs no storage at all, because `observe` can sample it one frame
    ahead of each run start during the render's own pass; the only part the
    prepass owes it is the plate as of the trim start.
    """
    boxes = np.asarray(boxes, dtype=np.int32)
    hots = np.asarray(hots, dtype=np.int32)
    tracked = np.asarray(tracked, dtype=np.int32)
    count = len(boxes)
    start_idx = max(0, int(start_idx))
    end_idx = count if end_idx is None else min(count, int(end_idx))
    walk_start = max(0, min(int(walk_start), start_idx))
    walk_limit = count if walk_limit is None else min(count, int(walk_limit))
    if end_idx <= start_idx or walk_limit <= walk_start:
        return None
    # Sized to the span actually walked, not to the whole take: a preview
    # window a second wide must not allocate plates for every pixel the
    # pointer visited over ten minutes.
    region = region_of(tracked[walk_start:walk_limit])
    if region is None:
        return None

    fps = float(src_fps) or 60.0
    stale_frames = max(0, int(round(
        float(_param(params, "stale_sec", CURSOR_STALE_SEC)) * fps)))
    ring_tol = int(_param(params, "ring_tol", CURSOR_RING_TOL))
    ring_max_frac = float(_param(params, "ring_max_frac",
                                 CURSOR_RING_MAX_FRAC))
    post_min = max(1, int(round(
        float(_param(params, "post_min_sec", CURSOR_POST_MIN_SEC)) * fps)))
    max_records = int(_param(params, "max_post_records",
                             CURSOR_MAX_POST_RECORDS))
    agree_tol = int(_param(params, "agree_tol", CURSOR_AGREE_TOL))
    foot_tol = int(_param(params, "footprint_tol", CURSOR_FOOTPRINT_TOL))
    foot_grow = int(_param(params, "footprint_grow", CURSOR_FOOTPRINT_GROW))
    foot_rim = int(_param(params, "footprint_rim", CURSOR_FOOTPRINT_RIM))
    fallback = str(_param(params, "fallback", CURSOR_FALLBACK))
    if fallback not in ("inpaint", "keep"):
        fallback = CURSOR_FALLBACK
    margin = int(_param(params, "inpaint_margin", CURSOR_INPAINT_MARGIN))
    radius = int(_param(params, "inpaint_radius", CURSOR_INPAINT_RADIUS))

    rx0, ry0, rx1, ry1 = region
    rw, rh = rx1 - rx0, ry1 - ry0
    pre_val = np.zeros((rh, rw, 3), dtype=np.uint8)
    pre_t = np.full((rh, rw), -1, dtype=np.int32)
    # `a` of the run currently covering each pixel, -1 when uncovered. This
    # is the only whole-region array the prepass needs, and it is what turns
    # "a pixel came back into view" into "a run of known length just ended".
    run_start = np.full(rh * rw, -1, dtype=np.int32)

    rec_a, rec_p, rec_v, rec_t = [], [], [], []
    n_records = 0
    dropped = 0
    pending = 0          # runs opened inside the window and still open

    def _box(i):
        # The TRACKED box throughout the prepass: the plates have to cover
        # the verification ring too, so runs are tracked on the grown rect.
        if i < 0 or i >= count:
            return (0, 0, 0, 0)
        b = tracked[i]
        return (int(b[0]), int(b[1]), int(b[2]), int(b[3]))

    i = walk_start
    while i < walk_limit and (i < end_idx or pending > 0):
        # At the first walked frame there is no history, so everything the
        # box covers counts as a run STARTING here: no `pre` (correctly --
        # we never saw those pixels clean) but a `post` still gets collected
        # when the pointer moves off them. Reading `_box(i - 1)` instead
        # would leave those runs invisible to the offset test below, which
        # keys on `run_start >= 0`, and a bounded preview walk would then
        # silently collect nothing at all.
        cur = _box(i)
        prev = _box(i - 1) if i > walk_start else (0, 0, 0, 0)
        onset = _rect_minus(cur, prev)
        offset = _rect_minus(prev, cur)
        off_idx = _flat_indices(offset, rx0, ry0, rw) if offset else None

        # Which of the just-ended runs owe a `post` sample: long enough to
        # matter, and overlapping the frames this render will actually write.
        qual = None
        if off_idx is not None and off_idx.size:
            a_of = run_start[off_idx]
            qual = ((a_of >= 0) & (a_of < end_idx) & ((i - a_of) >= post_min)
                    & (i - 1 >= start_idx))
            if not qual.any():
                qual = None
        need_pre = i < start_idx and bool(_rect_minus(_box(i + 1), cur))
        want_frame = (qual is not None) or need_pre

        if want_frame:
            ok, raw = capture.read()
            if not ok:
                break
            fr = crop_fn(raw, i) if crop_fn is not None else raw
            sub = fr[ry0:ry0 + rh, rx0:rx0 + rw].reshape(-1, 3)
        else:
            if not capture.grab():
                break
            sub = None

        if qual is not None and n_records < max_records:
            sel = off_idx[qual]
            room = max_records - n_records
            if sel.size > room:
                dropped += int(sel.size - room)
                sel = sel[:room]
            rec_a.append(np.maximum(run_start[sel], start_idx).astype(np.int32))
            rec_p.append(sel.astype(np.int32))
            rec_v.append(sub[sel].copy())
            rec_t.append(np.full(sel.size, i, dtype=np.int32))
            n_records += int(sel.size)
        elif qual is not None:
            dropped += int(qual.sum())

        if off_idx is not None and off_idx.size:
            closed = run_start[off_idx]
            pending -= int(((closed >= 0) & (closed < end_idx)).sum())
            run_start[off_idx] = -1
        if onset:
            on_idx = _flat_indices(onset, rx0, ry0, rw)
            if on_idx.size:
                run_start[on_idx] = i
                if i < end_idx:
                    pending += int(on_idx.size)

        if need_pre and sub is not None:
            nxt = _flat_indices(_rect_minus(_box(i + 1), cur), rx0, ry0, rw)
            if nxt.size:
                pre_val.reshape(-1, 3)[nxt] = sub[nxt]
                pre_t.reshape(-1)[nxt] = i

        i += 1
        if progress is not None and i % 600 == 0:
            progress(i, walk_limit)

    if rec_a:
        post_a = np.concatenate(rec_a)
        post_p = np.concatenate(rec_p)
        post_v = np.concatenate(rec_v)
        post_t = np.concatenate(rec_t)
        # `step` walks these with a moving cursor, so they have to be in
        # run-start order. They are ALMOST sorted already (runs close in end
        # order, not start order), so this is a cheap stable fix-up rather
        # than a real sort cost.
        order = np.argsort(post_a, kind="stable")
        post_a, post_p = post_a[order], post_p[order]
        post_v, post_t = post_v[order], post_t[order]
    else:
        post_a = np.zeros(0, dtype=np.int32)
        post_p = np.zeros(0, dtype=np.int32)
        post_v = np.zeros((0, 3), dtype=np.uint8)
        post_t = np.zeros(0, dtype=np.int32)

    return CursorEraser(boxes, hots, tracked, region, start_idx, end_idx,
                        pre_val, pre_t, post_a, post_p, post_v, post_t,
                        stale_frames=stale_frames,
                        ring_tol=ring_tol, ring_max_frac=ring_max_frac,
                        agree_tol=agree_tol,
                        footprint_tol=foot_tol, footprint_grow=foot_grow,
                        footprint_rim=foot_rim, fallback=fallback,
                        inpaint_margin=margin, inpaint_radius=radius,
                        post_dropped=dropped)
