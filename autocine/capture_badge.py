"""Erase macOS's per-window CAPTURE INDICATOR from an occlusion-free take.

A `--occlusion-free` take asks ScreenCaptureKit for one window's own buffer,
and macOS composites a small rounded "someone is capturing this window" pill
into that buffer -- over the window's traffic lights, in the top-left corner.
It is in the PIXELS, there is no layer to switch off, and it is present in
every frame from the first, so it is nobody's idea of a demo video.

**It only exists on the window-native path.** A whole-screen take of the same
windows shows the real traffic lights; that capture's indicator lives in the
menu bar instead (and is genuinely purple, which is where the name comes
from). So everything here is gated on `mode: "window_native"` -- see
`applies()`.

## Why it has to be FOUND rather than computed

The pill's SIZE is a constant: 52 x 20 points, in every take in
`recordings/` -- 60 of 60 window-native channels, across Claude, Chrome,
Terminal, Finder, TextEdit, Calculator, Safari and Excel, light and dark.

Its POSITION is not. macOS puts it over the window's close/minimise/zoom
buttons, and where those sit is the app's business: measured offsets from the
window's own top-left run Claude/Terminal/TextEdit (8, 4) pt, Excel (13, 6),
Chrome (21, 10), Finder/Safari (20, 16). It can also MOVE mid-take
(`recordings/20260831-183141/raw_0.mov`: (8, 4) pt for 90 s, then (18, 15)).
So a fixed offset would be wrong per app and wrong within one take, and the
detector below is not a nicety -- it is the only thing that is correct.

## Lock, verify, re-acquire

Searching every frame would be waste: the pill is a static overlay, so once
it is found its pixels are IDENTICAL frame to frame. So:

* **lock** -- one contour search on the first frame (`find`);
* **verify** -- each later frame, the mean absolute difference between the
  locked box and the reference patch. A 104x40 patch compare is a rounding
  error next to the decode that produced the frame;
* **re-acquire** -- only when that check fails, and only every
  `RESEARCH_EVERY` frames after a miss, so a pointer parked on top of the
  pill cannot turn the render into a per-frame contour search.

When a re-acquire finds nothing the last box is HELD, not dropped. The pill
is drawn for the whole capture, so "I can't see it" means something is in
front of it (the pointer, a menu) -- not that it left. Dropping the box there
would flash the badge back for exactly the frames it is hardest to ignore.

## Two independent checks, because a size filter is not a detector

A 104x40 box in the corner of a window is not rare -- a toolbar button, a tab,
a sidebar row. Erasing one would repaint real content with flat grey and
nobody would ever see the frame it happened on. So a candidate must also
carry the GLYPH: `_signature` reduces the pill's interior to a 16x8 bitmap
(binarised against its own median, which is what makes it survive both a dark
glyph on a light pill and the reverse), and it is matched against `GLYPH`.

The measured margin is wide. Across the six apps above the signatures sit
within 6 bits of each other out of 128; `MAX_GLYPH_BITS` is 20.

The glyph check is also what lets the size filter stay loose enough to catch
a RESIZED window: SCK fits an enlarged window back into the fixed buffer, and
the pill shrinks with everything else, so candidates are accepted down to
half size on the nominal aspect and identified by their glyph.

## What the fill is

The pill sits on a title bar, which is flat, so the honest repair is the
local background colour: the median of a ring just outside the box, painted
over it. Not `cv2.inpaint` -- with the window's rounded corner and its
transparent surround a few pixels away, inpainting drags a smear of black
into the corner, and it looked worse than the badge on every take it was
tried on.

Nothing is invented in its place. The traffic lights the pill covers are NOT
redrawn: they cannot be recovered from the footage (they are hidden in every
frame), so drawing them would be fabricating window chrome, and a window
whose buttons are missing is a look, while a window with buttons it never had
is a lie. `report()` says how many frames were painted and how many held, so
a take where this went wrong is findable rather than silent.
"""

import cv2
import numpy as np


# The pill, in POINTS. Measured on 60/60 window-native channels; the only
# thing about this overlay that is constant across apps and appearances.
PILL_PT = (52.0, 20.0)
_ASPECT = PILL_PT[0] / PILL_PT[1]

# Where to look, in POINTS from the window's top-left. Generous next to the
# widest offset seen (Finder/Safari at 20,16) so an app that indents its
# buttons further still lands inside, and small enough that the search never
# wanders into content.
SEARCH_PT = (260.0, 160.0)

# 16x8 bitmap of the pill's interior (see `_signature`), as a 128-bit int.
# Taken from a Claude window; every other app measured lands within 6 bits.
GLYPH = 0xf00fe02fe06fe047e04fe0c7e083f087
_GLYPH_W, _GLYPH_H = 16, 8
MAX_GLYPH_BITS = 20

# Canny pairs: the pill's edge against a title bar can be a 6-level step
# (light mode) or a 40-level one (dark), and no single threshold finds both.
_CANNY = ((8, 24), (20, 60), (40, 120))
_DILATE = np.ones((2, 2), np.uint8)

# Accept a candidate this much smaller than nominal -- an SCK letterbox fit
# after a window resize shrinks the whole buffer, pill included.
MIN_SIZE_FRAC = 0.5
_ASPECT_TOL = 0.28

# Verification: mean abs difference (0..255) between the locked box and the
# reference patch, above which the lock is considered stale.
DRIFT_TOL = 5.0
# After a failed re-acquire, wait this many frames before searching again.
RESEARCH_EVERY = 15
# Give up acquiring after this many failed searches. `probe` has usually
# already sampled the file by then, and a channel with no badge at all is a
# real case -- a sheet or a dialog has no window buttons for macOS to cover,
# so it gets no indicator (measured: an Excel "Open" sheet, a Chrome
# "Close Tab?" dialog). Without a cap those takes would pay a contour search
# every RESEARCH_EVERY frames for the whole render, looking for nothing.
MAX_ACQUIRE_TRIES = 40

# Frames `probe` samples, in order. NOT frame 0: macOS draws the indicator a
# few frames INTO the capture (measured onset: frame 0-5 across the takes in
# recordings/), so a locate that only looked at the first frame would miss on
# most takes and leave the badge on screen until the loop happened to find
# it. Sampling ahead also means frame 0 is erased on the SAME rect as every
# other frame -- the alternative is ~80 ms of real traffic lights that then
# blink out, which reads as a glitch rather than as a clean corner.
PROBE_FRAMES = (8, 30, 90, 240)

# Fill geometry, in PIXELS of the decoded buffer.
_PAD = 3        # grown over the tight box: the pill is antialiased and casts
                # a faint shadow, and both must go with it.
_RING = 10      # width of the background-sample ring outside the pad.


def applies(spec):
    """Whether `spec` -- a `meta.capture_window` or one `capture_channels`
    entry -- is a window-native capture, and therefore carries the badge.

    Deliberately narrow: a whole-screen or display-crop take has no per-window
    indicator, and running the search on one would be looking for a thing that
    is not there in the corner of a desktop full of content.
    """
    return isinstance(spec, dict) and spec.get("mode") == "window_native"


def scale_for(spec, frame_w):
    """points -> buffer pixels for one window-native channel.

    From the manifest (`buffer_w` / `logical_w`) so it is the value the
    recorder measured, with the decoded width as the fallback for a session
    written before those keys, and 2.0 -- Retina -- as the last resort.
    """
    spec = spec if isinstance(spec, dict) else {}
    lw = spec.get("logical_w")
    try:
        lw = float(lw)
    except (TypeError, ValueError):
        lw = 0.0
    if lw <= 0:
        return 2.0
    bw = spec.get("buffer_w")
    try:
        bw = float(bw)
    except (TypeError, ValueError):
        bw = 0.0
    if bw <= 0:
        bw = float(frame_w or 0)
    if bw <= 0:
        return 2.0
    return bw / lw


def _signature(gray, rect):
    """The pill's interior as a 128-bit glyph bitmap.

    Inset past the rounded ends before sampling (the border is the one part of
    the pill that changes shape with the fit scale), reduce to 16x8, and
    threshold on the patch's OWN median -- which is what makes one reference
    match both a dark glyph on a light pill and a light one on a dark pill,
    without carrying two templates or guessing the appearance.
    """
    x, y, w, h = rect
    ix, iy = int(round(x + 0.22 * w)), int(round(y + 0.15 * h))
    iw, ih = int(round(0.56 * w)), int(round(0.70 * h))
    if iw < _GLYPH_W or ih < _GLYPH_H:
        return None
    patch = gray[iy:iy + ih, ix:ix + iw]
    if patch.shape[0] < _GLYPH_H or patch.shape[1] < _GLYPH_W:
        return None
    small = cv2.resize(patch, (_GLYPH_W, _GLYPH_H),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
    bits = (small > np.median(small)).astype(np.uint8).reshape(-1)
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def _hamming(a, b):
    return bin(a ^ b).count("1")


def glyph_distance(gray, rect):
    """Bits between this box's interior and `GLYPH`, polarity-agnostic.

    Both polarities are tried because the appearance the pill adopts is the
    window's, not ours, and a take that switches to dark mode mid-recording
    must not quietly stop being erased.
    """
    sig = _signature(gray, rect)
    if sig is None:
        return None
    d = _hamming(sig, GLYPH)
    n = _GLYPH_W * _GLYPH_H
    return min(d, n - d)


def _candidates(gray, tw):
    """Boxes in the search ROI whose size and aspect could be the pill."""
    lo_w, hi_w = tw * MIN_SIZE_FRAC, tw * 1.10
    out = []
    for lo, hi in _CANNY:
        edges = cv2.dilate(cv2.Canny(gray, lo, hi), _DILATE)
        found = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # OpenCV 3 returns (image, contours, hierarchy); 4 returns
        # (contours, hierarchy). Take the contours from either shape.
        contours = found[-2]
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if h <= 0 or not (lo_w - 1.5 <= w <= hi_w):
                continue
            if abs(w / float(h) - _ASPECT) > _ASPECT_TOL:
                continue
            out.append((int(x), int(y), int(w), int(h)))
    return out


def _refine(frame, rect):
    """Shrink a Canny bounding box onto the pill's own fill.

    The edge box overshoots by a pixel or two on each side. Recovering the
    tight box keeps the painted rectangle as small as the artefact -- and
    gives `BadgeEraser` a stable reference patch whose bounds do not shift
    with the Canny threshold that happened to win.
    """
    x, y, w, h = rect
    pad = 6
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(frame.shape[1], x + w + pad)
    y1 = min(frame.shape[0], y + h + pad)
    sub = frame[y0:y1, x0:x1].astype(np.int16)
    if sub.size == 0:
        return rect
    ring = _ring_samples(frame, (x, y, w, h), pad=pad, ring=_RING)
    if ring is None or ring.size == 0:
        return rect
    bg = np.median(ring, axis=0).astype(np.int16)
    mask = (np.abs(sub - bg).sum(axis=2) > 10).astype(np.uint8)
    n, _lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    best = None
    for i in range(1, n):
        bx, by, bw, bh, area = stats[i]
        if best is None or area > best[4]:
            best = (bx, by, bw, bh, area)
    if best is None:
        return rect
    bx, by, bw, bh, _a = best
    # Only accept the refinement if it stayed the same object.
    if bw < w * 0.7 or bh < h * 0.6 or bw > w * 1.2 or bh > h * 1.2:
        return rect
    return (x0 + int(bx), y0 + int(by), int(bw), int(bh))


def find(frame_bgr, scale=2.0, search_pt=SEARCH_PT):
    """Locate the capture pill in a decoded window-native frame.

    Returns a tight `(x, y, w, h)` in buffer pixels, or None. Two checks have
    to agree -- size/aspect from the contour pass, then the glyph -- and among
    survivors the best glyph match wins, so a lookalike sitting beside the
    real pill loses to it rather than racing it.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    scale = float(scale) if scale and scale > 0 else 2.0
    h_all, w_all = frame_bgr.shape[:2]
    sw = min(w_all, max(1, int(round(search_pt[0] * scale))))
    sh = min(h_all, max(1, int(round(search_pt[1] * scale))))
    roi = frame_bgr[:sh, :sw]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    tw = PILL_PT[0] * scale
    best = None
    seen = set()
    for rect in _candidates(gray, tw):
        if rect in seen:
            continue
        seen.add(rect)
        d = glyph_distance(gray, rect)
        if d is None or d > MAX_GLYPH_BITS:
            continue
        if best is None or d < best[0]:
            best = (d, rect)
    if best is None:
        return None
    return _refine(frame_bgr, best[1])


def probe(path, scale=2.0, at_frames=PROBE_FRAMES, search_pt=SEARCH_PT):
    """Locate the badge by sampling a few frames of `path` up front.

    Returns `(rect, frame)` -- the box and the frame it was found in, which
    seeds `BadgeEraser`'s reference patch -- or `(None, None)`.

    One extra open + a handful of seeks per channel, paid once. See
    PROBE_FRAMES for why it does not simply use the first frame.
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, None
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for k in at_frames:
            if total > 0 and k >= total:
                k = max(0, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
            ok, fr = cap.read()
            if not ok:
                continue
            rect = find(fr, scale, search_pt)
            if rect is not None:
                return rect, fr
    finally:
        cap.release()
    return None, None


def _ring_strips(frame, rect, pad=_PAD, ring=_RING):
    """The four background bands around the (padded) box, as VIEWS.

    Views, and four of them, rather than one masked gather: this runs on every
    frame of every channel, and a boolean mask over the neighbourhood costs a
    copy plus a fancy-index gather where four slices cost nothing at all.
    """
    x, y, w, h = rect
    px0, py0 = max(0, x - pad), max(0, y - pad)
    px1 = min(frame.shape[1], x + w + pad)
    py1 = min(frame.shape[0], y + h + pad)
    ox0, oy0 = max(0, px0 - ring), max(0, py0 - ring)
    ox1 = min(frame.shape[1], px1 + ring)
    oy1 = min(frame.shape[0], py1 + ring)
    out = []
    for band in (frame[oy0:py0, ox0:ox1], frame[py1:oy1, ox0:ox1],
                 frame[py0:py1, ox0:px0], frame[py0:py1, px1:ox1]):
        if band.size:
            out.append(band.reshape(-1, 3))
    return out


def _ring_samples(frame, rect, pad=_PAD, ring=_RING):
    """Background pixels in a band just outside the (padded) box."""
    strips = _ring_strips(frame, rect, pad=pad, ring=ring)
    if not strips:
        return None
    return np.concatenate(strips, axis=0)


def fill_color(frame, rect, pad=_PAD, ring=_RING):
    """The local title-bar colour: the MEDIAN of the ring around the box.

    Median, not mean, because the ring routinely catches things that are not
    background -- the window's rounded corner and the transparent black
    outside it on one side, a toolbar control on the other. A mean lets any
    of them tint the patch; the median ignores them until they are most of
    the ring, and `BadgeEraser` counts the frames where they are.
    """
    samples = _ring_samples(frame, rect, pad=pad, ring=ring)
    if samples is None or samples.size == 0:
        return None, 0.0
    med = np.median(samples, axis=0)
    spread = float(np.median(np.abs(samples.astype(np.float32) - med)))
    return med, spread


def erase(frame_bgr, rect, pad=_PAD, ring=_RING):
    """Paint the pill out, IN PLACE. Returns the spread of the ring samples
    (a flatness score for the background that was matched; large means the
    fill had no single colour to match and the patch may show)."""
    col, spread = fill_color(frame_bgr, rect, pad=pad, ring=ring)
    if col is None:
        return None
    x, y, w, h = rect
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(frame_bgr.shape[1], x + w + pad)
    y1 = min(frame_bgr.shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    frame_bgr[y0:y1, x0:x1] = col.astype(frame_bgr.dtype)
    return spread


def patch_box(rect, frame_shape, pad=_PAD):
    """The rectangle `erase` actually paints: the tight box plus the pad,
    clipped to the frame. Exposed so a second compositor (the editor's live
    canvas player) can fill the SAME pixels rather than a near-miss."""
    x, y, w, h = rect
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(frame_shape[1], x + w + pad)
    y1 = min(frame_shape[0], y + h + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))


def patch(frame_bgr, rect):
    """`{"rect": [x, y, w, h], "color": [b, g, r]}` for `rect` in this frame,
    WITHOUT painting it -- the description of the repair, for a caller that
    has to perform it somewhere else (the browser). None when it does not
    land inside the frame."""
    box = patch_box(rect, frame_bgr.shape)
    if box is None:
        return None
    col, _spread = fill_color(frame_bgr, rect)
    if col is None:
        return None
    return {"rect": list(box), "color": [int(round(v)) for v in col]}


class BadgeEraser(object):
    """One channel's badge, tracked across that channel's frames.

    Stateful on purpose: `apply` is called once per decoded frame from inside
    a render loop, and the state is what keeps the per-frame cost at one small
    array compare instead of one contour search. See the module docstring for
    the lock/verify/re-acquire rule and why a miss HOLDS the box.
    """

    # Ring spread above which the background is called non-flat. Titlebar
    # noise measures ~1-3; a box straddling real content runs far higher.
    NOISY_SPREAD = 12.0

    # Recompute the fill when the ring's mean moves by more than this. A title
    # bar holds still for thousands of frames, so the median -- a sort, and the
    # one part of this that is not free -- is paid a handful of times per take
    # instead of 60 times a second, while a real change (dark mode, a control
    # sliding past, a card of a different app scrolling under the corner)
    # still moves the fill within one frame.
    RING_EPS = 0.75

    def __init__(self, scale=2.0, search_pt=SEARCH_PT, rect=None,
                 seed_frame=None):
        self.scale = float(scale) if scale and scale > 0 else 2.0
        self.search_pt = search_pt
        self.rect = tuple(rect) if rect else None
        self._ref = None            # reference pixels of the locked box
        self._miss = 0              # frames since the last failed re-acquire
        self._tries = 0             # failed acquisition searches
        self._fill = None           # cached background colour + its ring mean
        self._ring_key = None
        self.frames = 0
        self.painted = 0
        self.held = 0
        self.relocks = 0
        self.noisy = 0
        if self.rect is not None and seed_frame is not None:
            self._snapshot(seed_frame)

    @classmethod
    def for_channel(cls, path, spec, frame_w=None, search_pt=SEARCH_PT):
        """Build one for a window-native channel, pre-located from its file.

        Returns None when the channel is not window-native -- the caller then
        holds None and never calls a method, which is what keeps an
        unaffected take running the pre-feature loop untouched.
        """
        if not applies(spec):
            return None
        scale = scale_for(spec, frame_w)
        rect, frame = probe(path, scale, search_pt=search_pt)
        return cls(scale, search_pt=search_pt, rect=rect, seed_frame=frame)

    def clone(self):
        """A second eraser with this one's lock, for a second pass over the
        same file (the cursor eraser's prepass). State is per-pass: two passes
        sharing one instance would each see the other's frame counter and
        drift reference."""
        twin = type(self)(self.scale, search_pt=self.search_pt, rect=self.rect)
        if self._ref is not None:
            twin._ref = self._ref.copy()
        return twin

    def _lock(self, frame):
        rect = find(frame, self.scale, self.search_pt)
        if rect is None:
            return False
        self.rect = rect
        self._snapshot(frame)
        return True

    def _snapshot(self, frame):
        x, y, w, h = self.rect
        self._ref = frame[y:y + h, x:x + w].copy()

    def _drifted(self, frame):
        x, y, w, h = self.rect
        cur = frame[y:y + h, x:x + w]
        if self._ref is None or cur.shape != self._ref.shape:
            return True
        d = np.abs(cur.astype(np.int16) - self._ref.astype(np.int16))
        return float(d.mean()) > DRIFT_TOL

    def apply(self, frame_bgr):
        """Erase this frame's badge in place; returns the frame."""
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr
        self.frames += 1
        if self.rect is None:
            # Not locked: `probe` missed, so this take may have no badge at
            # all. Keep looking, but slowly, and stop after MAX_ACQUIRE_TRIES.
            if self._tries >= MAX_ACQUIRE_TRIES:
                return frame_bgr
            if self._miss and self.frames % RESEARCH_EVERY:
                self._miss += 1
                return frame_bgr
            if not self._lock(frame_bgr):
                self._miss += 1
                self._tries += 1
                return frame_bgr
            self._miss = 0
        elif self._drifted(frame_bgr):
            # Something changed inside the box: the pill moved (a real case --
            # see the module docstring), or something is drawn over it.
            if not self._miss or self.frames % RESEARCH_EVERY == 0:
                rect = find(frame_bgr, self.scale, self.search_pt)
                if rect is None:
                    self._miss += 1
                    self.held += 1
                else:
                    if rect != self.rect:
                        self.relocks += 1
                    self.rect = rect
                    self._snapshot(frame_bgr)
                    self._miss = 0
            else:
                self._miss += 1
                self.held += 1
        if self._paint(frame_bgr):
            self.painted += 1
        return frame_bgr

    def _paint(self, frame):
        """Fill the locked box with the local background, cached."""
        strips = _ring_strips(frame, self.rect)
        if not strips:
            return False
        n = sum(s.shape[0] for s in strips)
        key = float(sum(float(s.sum()) for s in strips)) / max(1, n * 3)
        if (self._fill is None or self._ring_key is None
                or abs(key - self._ring_key) > self.RING_EPS):
            samples = np.concatenate(strips, axis=0)
            med = np.median(samples, axis=0)
            spread = float(np.median(np.abs(samples.astype(np.float32) - med)))
            self._fill = med.astype(frame.dtype)
            self._ring_key = key
            if spread > self.NOISY_SPREAD:
                self.noisy += 1
        x, y, w, h = self.rect
        x0, y0 = max(0, x - _PAD), max(0, y - _PAD)
        x1 = min(frame.shape[1], x + w + _PAD)
        y1 = min(frame.shape[0], y + h + _PAD)
        if x1 <= x0 or y1 <= y0:
            return False
        frame[y0:y1, x0:x1] = self._fill
        return True

    def stats(self):
        return {"frames": self.frames, "painted": self.painted,
                "held": self.held, "relocks": self.relocks,
                "noisy": self.noisy,
                "rect": list(self.rect) if self.rect else None}

    def report(self):
        """One line for the render log, or None when there is nothing to say.

        Silent on the ordinary outcome (found it, painted every frame on a
        flat title bar). It speaks up for the two results worth knowing --
        never found, and found but painted over something that was not flat --
        because "the badge is still in my export" and "there is a grey smear
        in the corner" are the only two ways this feature fails, and neither
        should have to be discovered by watching the video.
        """
        if self.rect is None:
            if not self.frames:
                return None
            return ("capture badge: not found in {} frames -- left as "
                    "recorded".format(self.frames))
        bits = []
        if self.held:
            bits.append("{} held".format(self.held))
        if self.relocks:
            bits.append("{} relocks".format(self.relocks))
        if self.noisy:
            bits.append("background was non-flat {}x".format(self.noisy))
        if not bits:
            return None
        return "capture badge: erased {} frames ({})".format(
            self.painted, ", ".join(bits))
