"""Optional 'framed' presentation: a background (gradient / solid / wallpaper
image), padding, rounded corners and a soft shadow around the recording — the
polished cinematic look. Default style is 'clean' (pure zoom, native res).

The background + shadow are static, so `FramePainter` bakes them once and each
frame only composites the (changing) recording on top.

Backgrounds (`--background`):
  - a preset name        e.g. `aurora`, `midnight`, `sunset`, `ocean`
  - a solid color        e.g. `#101018`  or  `graphite`
  - an image file path   used as a wallpaper (cover-cropped, softly blurred)
"""

import os
import re
from math import log, sqrt

import cv2
import numpy as np

# Gradient presets as (bottom_rgb, top_rgb); rendered top->bottom.
_PRESETS = {
    "aurora":   ((116, 66, 92),   (36, 30, 32)),
    "midnight": ((30, 40, 90),    (8, 10, 24)),
    "sunset":   ((250, 120, 90),  (70, 30, 70)),
    "ocean":    ((40, 150, 170),  (12, 40, 70)),
    "forest":   ((70, 150, 90),   (12, 34, 26)),
    "graphite": ((70, 72, 78),    (26, 27, 30)),
    "mist":     ((232, 234, 238), (188, 194, 205)),
    "grape":    ((150, 80, 200),  (40, 20, 70)),
}
# Solid-color aliases (rgb).
_SOLIDS = {
    "black": (12, 12, 14),
    "white": (244, 245, 248),
    "gray": (128, 130, 134),
    "grey": (128, 130, 134),
}
DEFAULT_BACKGROUND = "aurora"

# The margin every framed/multi-card composition leaves around its content, as
# a fraction of the canvas WIDTH -- spent on BOTH axes, so it reads as one
# uniform border rather than a proportional one. THE single source of truth on
# the Python side; `studio_web/editor.js` carries the same number twice (the
# framed viewport preview, and `fitPlan`'s "Fit to frame"), and
# `test_web_sources.py` pins the two languages against each other by reading
# the literal back out of the JS -- so this cannot be changed alone.
#
# Was 0.055, lowered 2026-08-31: on a 4-card occlusion-free take it spent
# 211px a side on a 3840x2160 canvas, and the cards read as small islands in a
# large empty background. Measured on recordings/20260831-103923 scene 3,
# canvas coverage: desktop 52.5% -> 64.8%, feature 62.0% -> 69.5%, grid
# 53.2% -> 66.2%. The remaining gap is `_scale_boxes_into` filling one axis and
# centring the other, which no margin can recover.
PAD_FRAC = 0.03


def _even(n):
    """libx264 requires even dimensions."""
    n = int(round(n))
    return n if n % 2 == 0 else n + 1


def resolve_aspect_canvas(spec, src_w, src_h):
    """Resolve an --aspect spec into an (out_w, out_h) render canvas.

    - None / "auto" (default): passes the source dimensions through
      unchanged -- today's behavior, and the identity case every other
      aspect-aware code path (camera window sizing, effect scaling) is built
      to reduce to exactly.
    - "W:H" (e.g. "9:16", "1:1", "4:5"): a target aspect ratio. The canvas
      keeps ~the source's total pixel budget while conforming to that ratio,
      so a vertical export isn't gratuitously up- or down-scaled.
    - "WxH" (e.g. "1080x1920"): an exact custom canvas size.
    - Anything unparsable falls back to "auto" rather than crashing (same
      never-crash philosophy as resolve_background).
    """
    src_w = max(2, int(src_w))
    src_h = max(2, int(src_h))
    if spec is None:
        return src_w, src_h
    text = str(spec).strip().lower()
    if text in ("", "auto", "source", "native"):
        return src_w, src_h

    m = re.match(r"^(\d+)\s*x\s*(\d+)$", text)
    if m:
        w, h = _even(int(m.group(1))), _even(int(m.group(2)))
        if w >= 2 and h >= 2:
            return w, h
        return src_w, src_h

    m = re.match(r"^(\d+)\s*:\s*(\d+)$", text)
    if m:
        rw, rh = int(m.group(1)), int(m.group(2))
        if rw > 0 and rh > 0:
            scale = sqrt((float(src_w) * float(src_h)) / (rw * rh))
            w = _even(max(2, rw * scale))
            h = _even(max(2, rh * scale))
            return w, h
    return src_w, src_h


def output_size(W, H, style, aspect=None):
    """The real output canvas for both 'clean' and 'framed' styles.

    `style` doesn't change the resolved size -- clean fills this canvas
    edge to edge, framed insets the recording inside it with padding/
    background -- but both need the same canvas math, so it stays a
    parameter for API clarity even though this function doesn't branch on it.
    """
    return resolve_aspect_canvas(aspect, W, H)


def fit_max_height(w, h, max_h):
    """Scale (w, h) DOWN so the height is at most `max_h`, keeping the aspect
    ratio and even (libx264) dimensions. The render-resolution control: it
    trades sharpness for a smaller canvas, which is what makes the composite
    fast (cost is per-pixel, per-frame).

    A no-op -- bit-exact -- when `max_h` is falsy or the canvas already fits,
    so `resolution: auto` reproduces today's output exactly. Never scales UP
    (that would only cost pixels without adding detail)."""
    try:
        max_h = int(max_h)
    except (TypeError, ValueError):
        return int(w), int(h)
    w, h = int(w), int(h)
    if max_h <= 0 or h <= max_h:
        return w, h
    scale = float(max_h) / float(h)
    return _even(max(2, w * scale)), _even(max(2, h * scale))


def _hex_to_rgb(s):
    s = s.lstrip("#")
    if len(s) != 6:
        return None
    try:
        return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    except ValueError:
        return None


def _gradient(w, h, bottom_rgb, top_rgb):
    top = np.array(top_rgb[::-1], np.float32)     # rgb -> bgr
    bot = np.array(bottom_rgb[::-1], np.float32)
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    col = top[None, :] * (1.0 - ramp) + bot[None, :] * ramp
    return np.repeat(col[:, None, :], w, axis=1).astype(np.float32)


def _solid(w, h, rgb):
    return np.full((h, w, 3), np.array(rgb[::-1], np.float32), np.float32)


def _wallpaper(w, h, img):
    """Cover-crop `img` to w x h, soften it, and darken slightly so the
    recording placed on top stays the focal point."""
    ih, iw = img.shape[:2]
    scale = max(w / float(iw), h / float(ih))
    rw, rh = int(np.ceil(iw * scale)), int(np.ceil(ih * scale))
    resized = cv2.resize(img, (rw, rh), interpolation=cv2.INTER_AREA)
    x = (rw - w) // 2
    y = (rh - h) // 2
    crop = resized[y:y + h, x:x + w].astype(np.float32)
    k = max(1, int(0.01 * w) | 1)          # odd kernel, ~1% of width
    crop = cv2.GaussianBlur(crop, (k, k), 0)
    return crop * 0.82                       # gentle darken


def resolve_background(value, W, H):
    """Return an (H, W, 3) float32 BGR base image for the given spec."""
    if value is None:
        value = DEFAULT_BACKGROUND
    if isinstance(value, str):
        key = value.strip()
        low = key.lower()
        if low in _PRESETS:
            return _gradient(W, H, *_PRESETS[low])
        if low in _SOLIDS:
            return _solid(W, H, _SOLIDS[low])
        rgb = _hex_to_rgb(key)
        if rgb is not None:
            return _solid(W, H, rgb)
        if os.path.isfile(key):
            img = cv2.imread(key, cv2.IMREAD_COLOR)
            if img is not None:
                return _wallpaper(W, H, img)
        # Unknown spec: fall back to the default preset rather than crash.
        return _gradient(W, H, *_PRESETS[DEFAULT_BACKGROUND])
    # already an array-like base
    return np.asarray(value, np.float32)


def _blur_downscale(sigma, area):
    """How coarse a grid a `sigma`-wide Gaussian over `area` px may use.

    Same trick, same reason as `_SHADOW_DOWNSCALE` on the moving-cells path
    (see the note by that constant): a blur this wide is heavily band-limited,
    so computing it at 1/d and scaling back up costs a count or two of the
    255 the mask is expressed in.

    Two gates, and the AREA one is the point. A downscale that is merely
    accurate is not good enough here -- the baked plate is an export pixel
    path, and drifting one LSB on every ordinary card render to save 0.2s is
    a bad trade. So this stays byte-identical for every canvas whose blur is
    already cheap (1920x1080, 2560x1440, 2880x1800 all return 1) and engages
    only for the oversized composites where the full blur genuinely hurts: a
    multi-window scene canvas sizes itself to render its cards ~1:1
    (`render._multi_native_canvas`), which on a Retina 4-window take is
    3636x2046 = 7.4MP and ~1.4s of GaussianBlur PER SCENE.
    """
    if area < _BLUR_MIN_AREA:
        return 1
    d = int(float(sigma) // _BLUR_MIN_SIGMA_PX)
    return max(1, min(_SHADOW_DOWNSCALE, d))


def _wide_blur(mask, sigma):
    """`cv2.GaussianBlur(mask, (0,0), sigma)`, computed on a 1/d grid."""
    h, w = mask.shape[:2]
    d = _blur_downscale(sigma, w * h)
    if d <= 1:
        return cv2.GaussianBlur(mask, (0, 0), sigma)
    small = cv2.resize(mask, (max(1, (w + d - 1) // d), max(1, (h + d - 1) // d)),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / float(d)))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    r = int(min(r, w // 2, h // 2))
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    return m


def draw_drop_shadow(out, cell, sigma, dy, alpha, d):
    """Darken `out` in place with one blurred, offset, rounded drop shadow.

    Module-level so both the multi-window compositor (`MultiFramePainter.
    _shadow_layer`, which delegates here) and render's whole-screen
    "grow the active window" pass can cast the same shadow the framed look
    uses. `cell` is `(x, y, w, h)` in `out` pixels; the shadow is the rounded
    `cell` mask, blurred at `sigma`, offset down by `dy`, multiplied in at
    `alpha`. Computed at 1/`d` resolution and scaled back up, and bounded to
    the cell's own 3-sigma neighbourhood -- both measured affordable at the
    cast layer's sigma (see `_shadow_layer`'s own note for the numbers).
    """
    H, W = out.shape[:2]
    ix, iy, fw, fh = (int(v) for v in cell)
    margin = int(3 * sigma) + d
    x0, y0 = max(0, ix - margin), max(0, iy + dy - margin)
    x1 = min(W, ix + fw + margin)
    y1 = min(H, iy + dy + fh + margin)
    rw, rh = x1 - x0, y1 - y0
    if rw <= 0 or rh <= 0:
        return
    sw, sh_ = max(1, rw // d), max(1, rh // d)
    small = np.zeros((sh_, sw), np.uint8)
    m = _rounded_mask(max(1, fw // d), max(1, fh // d),
                      max(1, int(0.02 * fw) // d))
    my = max(0, min(sh_ - m.shape[0], (iy + dy - y0) // d))
    mx = max(0, min(sw - m.shape[1], (ix - x0) // d))
    roi = small[my:my + m.shape[0], mx:mx + m.shape[1]]
    np.maximum(roi, m[:roi.shape[0], :roi.shape[1]], out=roi)
    small = cv2.GaussianBlur(small, (0, 0), max(0.5, sigma / d))
    shadow = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_LINEAR)
    # 255 - shadow*alpha, then a scaled multiply: the same darkening the
    # baked path does, without ever leaving uint8. Layers therefore stack as
    # a product, which is what two real shadows do.
    k = cv2.addWeighted(shadow, -float(alpha), shadow, 0.0, 255.0)
    region = out[y0:y1, x0:x1]
    region[:] = cv2.multiply(region, cv2.cvtColor(k, cv2.COLOR_GRAY2BGR),
                             scale=1.0 / 255.0, dtype=cv2.CV_8U)


class FramePainter:
    """Composites a zoomed recording onto a precomputed framed background."""

    def __init__(self, W, H, style="framed", background=None,
                 pad_frac=PAD_FRAC):
        self.W, self.H, self.style = W, H, style
        pad = int(pad_frac * W)
        inner_w = W - 2 * pad
        inner_h = int(round(inner_w * H / float(W)))
        if inner_h > H - 2 * pad:
            inner_h = H - 2 * pad
            inner_w = int(round(inner_h * W / float(H)))
        self.iw, self.ih = inner_w, inner_h
        self.x = (W - inner_w) // 2
        self.y = (H - inner_h) // 2

        radius = max(8, int(0.02 * inner_w))
        self._mask = _rounded_mask(inner_w, inner_h, radius)  # 0/255, ih x iw
        self._radius = radius

        base = resolve_background(background, W, H).copy()
        shadow = np.zeros((H, W), np.uint8)
        sy = min(H - inner_h, self.y + int(0.012 * H))
        shadow[sy:sy + inner_h, self.x:self.x + inner_w] = self._mask
        shadow = cv2.GaussianBlur(shadow, (0, 0), max(1.0, 0.02 * W))
        sh = (shadow.astype(np.float32) / 255.0)[:, :, None] * 0.45
        # Quantize the static backdrop ONCE. `paint` returns uint8, and every
        # pixel it doesn't overwrite is exactly this truncation of that float
        # composite -- so carrying the float canvas per frame (copy 62 MB,
        # then convert the whole thing back down) is ~50ms/frame of arithmetic
        # on pixels that never change. Bit-exact: same values, same cast.
        self._plate = (base * (1.0 - sh)).astype(np.uint8)
        self._corners = self._corner_repairs(
            int(min(radius, inner_w // 2, inner_h // 2)))

    def _corner_repairs(self, radius):
        """The only pixels of the inner rect that are NOT the recording: the
        rounded corners. `paint` resizes the frame straight over the whole
        rect and then puts these back, which is why it needs no per-pixel
        blend at all.

        Safe because `_rounded_mask`'s two rectangles already cover
        everything outside the four `radius`-sized corner squares, so the
        mask's zeros can't live anywhere else. `radius` is the CLAMPED one
        the mask was actually drawn with, not the requested one.
        """
        iw, ih = self.iw, self.ih
        plate = self._plate[self.y:self.y + ih, self.x:self.x + iw]
        out = []
        for ys, xs in ((slice(0, radius), slice(0, radius)),
                       (slice(0, radius), slice(iw - radius, iw)),
                       (slice(ih - radius, ih), slice(0, radius)),
                       (slice(ih - radius, ih), slice(iw - radius, iw))):
            outside = (self._mask[ys, xs] == 0)[:, :, None]
            if not outside.any():
                continue
            out.append((ys, xs, outside, plate[ys, xs].copy()))
        return out

    def paint(self, img_bgr):
        """Composite one recording frame onto the baked framed backdrop.

        Composited in uint8, with no per-pixel alpha math: `_rounded_mask`
        draws with aliased fills, so the mask is only ever 0 or 255 and the
        old `rec * m + bg * (1 - m)` was a *select* wearing a blend's
        clothes. Every value it could produce is reachable exactly here --
        inside the inner rect, `rec` round-trips 0..255 through float32
        unchanged; outside it, the plate is the same truncation of the same
        float. So this is byte-for-byte what the float version returned, far
        cheaper. A fresh array per call, since callers draw on top of it in
        place (the facecam bubble) and would otherwise smear across frames.
        """
        out = self._plate.copy()
        roi = out[self.y:self.y + self.ih, self.x:self.x + self.iw]
        # Resize STRAIGHT into the canvas -- no full-size temporary.
        fit = cv2.resize(img_bgr, (self.iw, self.ih), dst=roi,
                         interpolation=cv2.INTER_AREA)
        if fit is not roi:
            # OpenCV refused the in-place destination (it reallocates on any
            # dtype mismatch -- a non-uint8 frame) and handed back its own
            # buffer. Assigning casts exactly like the old .astype().
            roi[:] = fit
        for ys, xs, outside, plate in self._corners:
            np.copyto(roi[ys, xs], plate, where=outside)
        return out


def make_painter(W, H, style, background=None):
    """Return a painter for non-clean styles, else None."""
    if style == "clean":
        return None
    return FramePainter(W, H, style, background=background)


# ---- multi-window framing ---------------------------------------------------
_GRID_CANDIDATES = {
    1: [(1, 1)],
    2: [(1, 2), (2, 1)],
    3: [(1, 3), (3, 1)],
    4: [(1, 4), (4, 1), (2, 2)],
}


def _choose_grid(n, canvas_w, canvas_h):
    """Pick a (rows, cols) grid for `n` cells (1-4), zero wasted cells,
    choosing among the small candidate set for `n` the layout whose
    row/column ratio best matches the canvas orientation -- e.g. a 9:16
    canvas with 2 windows stacks them (2 rows x 1 col) rather than the
    side-by-side layout a 16:9 canvas gets. `n` is clamped to [1, 4].

    Explicit v1 call: N=3 only considers the 1x3/3x1 candidates, not a
    2x2 grid with one empty cell.
    """
    n = max(1, min(4, int(n)))
    candidates = _GRID_CANDIDATES[n]
    canvas_ratio = max(1e-6, float(canvas_w)) / max(1e-6, float(canvas_h))
    target = log(canvas_ratio)
    best = candidates[0]
    best_err = None
    for rows, cols in candidates:
        err = abs(log(float(cols) / float(rows)) - target)
        if best_err is None or err < best_err:
            best_err, best = err, (rows, cols)
    return best


def _fit_aspect(box_w, box_h, content_w, content_h):
    """Largest (content_w:content_h)-aspect box that fits inside
    (box_w, box_h). Same shape as render._contain_fit, duplicated locally
    since render.py already imports framing (importing back would be
    circular)."""
    box_w = max(1.0, float(box_w))
    box_h = max(1.0, float(box_h))
    content_w = max(1.0, float(content_w))
    content_h = max(1.0, float(content_h))
    scale = min(box_w / content_w, box_h / content_h)
    return content_w * scale, content_h * scale


def _rect_wh(rect):
    """Accepts a {"w","h",...} dict (an edits.json window entry) or a
    plain (w, h) pair."""
    if isinstance(rect, dict):
        return float(rect.get("w", 1.0)), float(rect.get("h", 1.0))
    w, h = rect
    return float(w), float(h)


def _rect_xywh(rect):
    """`(x, y, w, h)` of an edits.json window entry, or None.

    None for the plain `(w, h)` tuples `MultiFramePainter` also accepts:
    those carry no origin, so there is no desktop arrangement to preserve
    and the caller has to fall back to the grid.
    """
    if not isinstance(rect, dict):
        return None
    try:
        return [float(rect["x"]), float(rect["y"]),
                float(rect["w"]), float(rect["h"])]
    except (KeyError, TypeError, ValueError):
        return None


def _separate_boxes(boxes, max_passes=64):
    """Push overlapping boxes apart, along each pair's shallower axis.

    Shallower axis on purpose: two windows overlapping by 40px horizontally
    and 400px vertically are side-by-side windows with a small collision,
    and shoving them 400px apart vertically would destroy the arrangement
    the user is looking at. Each box takes half the push, so neither one is
    privileged, and ties (identical rects) break by index so the result is
    deterministic rather than dependent on dict ordering.

    Mutates and returns `boxes`.
    """
    n = len(boxes)
    for _ in range(max_passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = boxes[i], boxes[j]
                ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                if ox <= 0.0 or oy <= 0.0:
                    continue
                moved = True
                axis = 0 if ox <= oy else 1
                depth = (ox if axis == 0 else oy) / 2.0
                ac = a[axis] + a[axis + 2] / 2.0
                bc = b[axis] + b[axis + 2] / 2.0
                sign = 1.0 if ac <= bc else -1.0
                a[axis] -= sign * depth
                b[axis] += sign * depth
        if not moved:
            break
    return boxes


def _compact_axis(boxes, axis, gap):
    """Slide every box toward the origin on one axis, order preserved.

    A box only blocks another when they overlap on the OTHER axis -- that is
    what keeps a two-column arrangement two columns instead of collapsing
    every window into one stack. Boxes are processed in their current order
    along `axis`, so "left of" and "above" survive; only the dead space
    between them goes.

    Mutates and returns `boxes`.
    """
    other = 1 - axis
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][axis], i))
    origin = min(b[axis] for b in boxes) if boxes else 0.0
    placed = []
    for idx in order:
        box = boxes[idx]
        limit = origin
        for pidx in placed:
            prev = boxes[pidx]
            overlaps_other = (box[other] < prev[other] + prev[other + 2]
                              and prev[other] < box[other] + box[other + 2])
            if overlaps_other:
                limit = max(limit, prev[axis] + prev[axis + 2] + gap)
        box[axis] = limit
        placed.append(idx)
    return boxes


def _fit_scale(W, H, boxes, pad):
    """The uniform scale `_scale_boxes_into` would use for this arrangement.

    Split out because it is also the SCORE for choosing between arrangements:
    a bigger scale means the windows end up bigger on screen, which is the
    whole reason to compact at all.
    """
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    span_w = max(1e-6, max(b[0] + b[2] for b in boxes) - x0)
    span_h = max(1e-6, max(b[1] + b[3] for b in boxes) - y0)
    return min(max(1.0, float(W) - 2.0 * pad) / span_w,
               max(1.0, float(H) - 2.0 * pad) / span_h)


def _scale_boxes_into(W, H, boxes, pad):
    """Uniformly scale a box set to fit the canvas, then centre it.

    UNIFORM and not per-box: that is what preserves both each window's own
    aspect and the size difference between them, so a full-height editor
    stays visibly bigger than the terminal next to it.

    The price of that, paid by every caller: `min()` of the two axes fills
    ONE axis exactly and leaves the other centred with slack, in proportion
    to how far the arrangement's bounding-box aspect is from the canvas's.
    Nothing downstream can recover that slack without either cropping a card
    or breaking its aspect, so "fill the frame" is never what this returns --
    see the measured table under "named arrangements" below for what that
    costs each layout on four canvases.
    """
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    span_w = max(1e-6, max(b[0] + b[2] for b in boxes) - x0)
    span_h = max(1e-6, max(b[1] + b[3] for b in boxes) - y0)
    avail_w = max(1.0, float(W) - 2.0 * pad)
    avail_h = max(1.0, float(H) - 2.0 * pad)
    scale = min(avail_w / span_w, avail_h / span_h)
    off_x = (float(W) - span_w * scale) / 2.0
    off_y = (float(H) - span_h * scale) / 2.0
    out = []
    for x, y, w, h in boxes:
        fw = max(2, min(W, int(round(w * scale))))
        fh = max(2, min(H, int(round(h * scale))))
        ix = max(0, min(W - fw, int(round(off_x + (x - x0) * scale))))
        iy = max(0, min(H - fh, int(round(off_y + (y - y0) * scale))))
        out.append((ix, iy, fw, fh))
    return out


def _grid_placements(W, H, rects, pad, gap):
    """Canvas placement for the original uniform rows x cols grid."""
    rows, cols = _choose_grid(len(rects), W, H)
    usable_w = max(1, W - 2 * pad)
    usable_h = max(1, H - 2 * pad)
    cell_w = (usable_w - gap * (cols - 1)) / float(cols)
    cell_h = (usable_h - gap * (rows - 1)) / float(rows)
    out = []
    for i, rect in enumerate(rects):
        r, c = divmod(i, cols)
        cell_x = pad + c * (cell_w + gap)
        cell_y = pad + r * (cell_h + gap)
        rw, rh = _rect_wh(rect)
        fw_f, fh_f = _fit_aspect(cell_w, cell_h, rw, rh)
        fw = max(2, min(W, int(round(fw_f))))
        fh = max(2, min(H, int(round(fh_f))))
        ix = int(round(cell_x + (cell_w - fw) / 2.0))
        iy = int(round(cell_y + (cell_h - fh) / 2.0))
        ix = max(0, min(W - fw, ix))
        iy = max(0, min(H - fh, iy))
        out.append((ix, iy, fw, fh))
    return out


def _desktop_placements(W, H, rects, pad, gap):
    """Canvas placement that keeps the windows' desktop arrangement.

    Three passes over the source rects, all in source-pixel space before a
    single scale into the canvas:

      1. separate  -- pull any overlapping pair apart (`_separate_boxes`);
      2. compact   -- squeeze the dead space out of each axis, preserving
                      left-of / above-of order (`_compact_axis`);
      3. fit       -- one uniform scale + centre (`_scale_boxes_into`).

    Aspects and relative sizes survive all three, which is the whole point:
    the result reads as the user's own desktop, tidied, rather than as a
    grid that happens to contain their windows.

    Returns None when any rect has no origin (the `(w, h)` tuple form), so
    the caller can fall back to the grid rather than stacking everything at
    the top-left corner.
    """
    boxes = []
    for rect in rects:
        box = _rect_xywh(rect)
        if box is None:
            return None
        box[2] = max(1.0, box[2])
        box[3] = max(1.0, box[3])
        boxes.append(box)
    if not boxes:
        return []
    # Gap and pad are canvas-space, but separation and compaction run in
    # source space -- convert once, using the ratio the final fit will use.
    span_w = max(1e-6, max(b[0] + b[2] for b in boxes)
                 - min(b[0] for b in boxes))
    src_gap = gap * (span_w / max(1.0, float(W)))
    _separate_boxes(boxes)

    # Which axis to compact FIRST changes the answer, and neither order is
    # right in general. Compacting x first turns two windows sitting at
    # opposite corners into a narrow column; compacting y first leaves them
    # side by side. Compacting at all is wrong when the desktop was already
    # tight and squeezing only distorts it.
    #
    # So build all three candidates and keep the one whose windows end up
    # BIGGEST -- every candidate preserves the arrangement (that is what
    # `_compact_axis` guarantees), so the only thing left to choose on is how
    # well it uses the frame. Ties break by candidate order, which keeps this
    # deterministic.
    candidates = [
        [list(b) for b in boxes],                                   # as-is
        _compact_axis(_compact_axis([list(b) for b in boxes], 0, src_gap),
                      1, src_gap),                                  # x then y
        _compact_axis(_compact_axis([list(b) for b in boxes], 1, src_gap),
                      0, src_gap),                                  # y then x
    ]
    best = max(candidates, key=lambda c: _fit_scale(W, H, c, pad))
    return _scale_boxes_into(W, H, best, pad)


# ---- named arrangements -----------------------------------------------------
# The three presets below all build their boxes in an abstract UNIT space --
# one dimension pinned to 1.0, the other being the card's own aspect -- and
# then hand the whole set to `_scale_boxes_into`. Going through that helper is
# the point: it is the same uniform scale + centre the desktop layout uses, so
# they inherit its aspect and relative-size guarantees instead of re-deriving
# them, and there is no dead space left INSIDE the arrangement afterwards.
#
# WHAT THAT IS NOT: it is not "fills the frame". `_scale_boxes_into` applies
# one scale, `min(avail_w/span_w, avail_h/span_h)`, which fills exactly ONE
# axis and centres on the other -- letterboxing the second by however the
# arrangement's bounding-box aspect differs from the canvas's. That is
# unavoidable for aspect-locked rectangles, so the useful question is not
# "does this preset fill the frame" (none of them can, in general) but "which
# preset suits this canvas". Measured, as the percentage of the canvas the
# cards actually cover, on the THREE real recorded windows this feature came
# from -- 1418x1718 at (0,0), 1418x852 at (1462,0), 1418x854 at (1462,900);
# only `desktop` reads those origins, the rest see aspects only:
#
#   canvas            grid    desktop  feature   row     column
#   16:10 landscape   34.9%   80.7%    81.8%    33.0%   20.0%
#   16:9              38.7%   71.0%    71.6%    36.5%   17.5%
#   9:16 vertical     73.2%   29.1%    74.9%    11.8%   67.5%
#   1:1               21.8%   51.8%    56.6%    20.9%   35.4%
#
# (Re-measured 2026-08-31 when PAD_FRAC dropped 0.055 -> 0.03. Every figure
# rose; the RANKING did not move, and `feature` went from within 0.3 points
# of best to outright best on all four.)
#
# Reading it: `feature` is the only preset that RE-ORIENTS (see
# `_feature_placements`), which is why it lands within 0.3 points of the best
# layout on all four canvases and is the safe default. `grid` adapts too --
# `_choose_grid` picks rows x cols against the canvas orientation -- which is
# exactly why it stays competitive on the vertical canvas where the
# fixed-axis presets collapse, and why it narrowly wins there. `column` and
# `row` are pinned to an axis by name and swing hardest: column runs 16.5% ->
# 63.6% from the landscape canvas to the vertical one, row 29.5% -> 10.7% the
# other way. With only TWO cards the ranking moves again (feature and column
# tie at 80.9% on 9:16 against the grid's 58.4%), so none of this is a rule --
# it is the shape of the trade-off. `test_framing.InkCoverage` pins the table
# so this comment cannot quietly rot.
#
# Fewer than two cards goes to `_grid_placements` unchanged: a single card has
# no arrangement to speak of, and it keeps `_scale_boxes_into` (whose spans are
# `min()`/`max()` over the box list) from ever seeing an empty one.


def _unit_gap(H, pad, gap):
    """`gap`, a canvas-pixel distance, in unit space -- where 1.0 spans the
    padded canvas height."""
    return float(gap) / max(1.0, float(H) - 2.0 * pad)


def _rect_aspect(rect):
    """w/h of a rect, defended against a zero or negative dimension the same
    way `_fit_aspect` defends itself."""
    w, h = _rect_wh(rect)
    return max(1.0, w) / max(1.0, h)


def _row_placements(W, H, rects, pad, gap):
    """One row, in card order: every card full height, each as wide as its own
    aspect makes it.

    A row of N cards has an N-times-wider bounding box than one card, so it
    wants a very wide canvas and starves on anything else: measured on the
    three reference windows it covers 29.5% of a 16:10 canvas and 10.7% of a
    9:16 one. Rarely the best choice for three cards -- it exists because a
    user who asks for a row means a row, and unlike `feature` it never
    transposes to something else behind their back.
    """
    if len(rects) < 2:
        return _grid_placements(W, H, rects, pad, gap)
    gap_u = _unit_gap(H, pad, gap)
    boxes = []
    x = 0.0
    for rect in rects:
        aspect = _rect_aspect(rect)
        boxes.append([x, 0.0, aspect, 1.0])
        x += aspect + gap_u
    return _scale_boxes_into(W, H, boxes, pad)


def _column_placements(W, H, rects, pad, gap):
    """One column, in card order: every card full width, each as tall as its
    own aspect makes it.

    The exact mirror of `_row_placements`, and the exact mirror of its
    numbers: it is the pick for a VERTICAL export (63.6% of a 9:16 canvas on
    the three reference windows, 80.9% with two) and the worst pick for a
    landscape one (16.5% at 16:10). Like `row`, it never transposes.
    """
    if len(rects) < 2:
        return _grid_placements(W, H, rects, pad, gap)
    gap_u = _unit_gap(H, pad, gap)
    boxes = []
    y = 0.0
    for rect in rects:
        height = 1.0 / _rect_aspect(rect)
        boxes.append([0.0, y, 1.0, height])
        y += height + gap_u
    return _scale_boxes_into(W, H, boxes, pad)


def _feature_boxes(hero, rest, gap_u, transposed):
    """Unit-space boxes for ONE orientation of the feature arrangement.

    `transposed` False = hero on the LEFT with the rest in a column beside it;
    True = hero on TOP with the rest in a row beneath it.

    The secondary block is sized as ONE block -- its widths, its heights and
    the gaps between them decided together -- until it spans exactly the
    hero's facing edge. Sizing each card on its own instead would leave the
    block ragged; a shared width (or, transposed, a shared height) so it
    presents one straight edge to the hero is the look being asked for. Each
    card still keeps its own aspect, so the block is only regular on the one
    axis.

    The hero is `(hero, 1.0)` in BOTH orientations deliberately. That is what
    makes `_fit_scale` a fair score between them in `_feature_placements`:
    the same object at the same unit size, so a larger scale genuinely means
    a larger hero on screen rather than a difference in how the two
    candidates happened to be normalized.

    EVERY gap here is `gap_u`, the hero-to-block one included. The block's
    size is solved for *after* subtracting its internal gaps rather than the
    gaps being scaled along with the block -- one requested gap, one visual
    gap. It used to emit three different ones for a single request (gap_u
    between hero and stack, gap_u * scale inside it, then `_scale_boxes_into`
    over the top). Measured on a 2880x1800 canvas, whose requested gap is
    57px: a 3-card feature came out 58 / 46, a 4-card one 58 / 24 / 25. Both
    now measure 57 across.
    """
    boxes = [[0.0, 0.0, hero, 1.0]]
    gaps = (len(rest) - 1) * gap_u
    if not transposed:
        # A card `width` wide is width/aspect tall, so the column's height is
        # width * sum(1/aspect) + gaps; solve that for exactly the hero's 1.0.
        width = max(1e-6, (1.0 - gaps) / max(1e-6, sum(1.0 / a for a in rest)))
        x, y = hero + gap_u, 0.0
        for aspect in rest:
            height = width / aspect
            boxes.append([x, y, width, height])
            y += height + gap_u
        return boxes
    # Transposed: a card `height` tall is height*aspect wide, so solve the
    # row's width for exactly the hero's.
    height = max(1e-6, (hero - gaps) / max(1e-6, sum(rest)))
    x, y = 0.0, 1.0 + gap_u
    for aspect in rest:
        width = height * aspect
        boxes.append([x, y, width, height])
        x += width + gap_u
    return boxes


def _feature_placements(W, H, rects, pad, gap):
    """One card large, the rest in a block beside it sharing a straight edge.

    THE ORIENTATION ADAPTS TO THE CANVAS. "One big window plus the others
    beside it" is an intent, not an axis, and the same reasoning `_choose_grid`
    already applies to rows-vs-columns applies here: on a canvas taller than
    the left-hero arrangement wants, the hero goes on TOP with the rest in a
    row beneath it. Both candidates are built and scored with `_fit_scale`,
    exactly how `_desktop_placements` picks among its three compaction
    candidates, and the bigger scale wins; `max` keeps the first maximum, so
    a tie is broken by candidate order (left-hero) and stays deterministic.

    Measured on the three windows this preset was built for, transposing
    closes essentially the whole gap to the grid on a vertical export: 26.6%
    -> 67.1% of a 1080x1920 canvas covered, against the grid's 67.4%. A
    landscape canvas keeps picking the left-hero candidate (16:10 scores it
    1484 against the transpose's 1164), so the choice there is a no-op; its
    coverage moved 68.2% -> 67.7% only because the gap fix above made the
    stack's own gaps as wide as the hero-to-stack one. (Both figures are at
    the historical `pad_frac=0.055`; the point is the delta, not the level --
    the same layout measures 81.8% at today's PAD_FRAC.)

    'row' and 'column' deliberately do NOT do this: they are named after an
    axis, and a user who asks for a row and is handed a column has been lied
    to. 'feature' is named after a *role*, which is why it may move.
    """
    if len(rects) < 2:
        return _grid_placements(W, H, rects, pad, gap)
    gap_u = _unit_gap(H, pad, gap)
    hero = _rect_aspect(rects[0])
    rest = [_rect_aspect(r) for r in rects[1:]]
    candidates = [_feature_boxes(hero, rest, gap_u, False),
                  _feature_boxes(hero, rest, gap_u, True)]
    best = max(candidates, key=lambda c: _fit_scale(W, H, c, pad))
    return _scale_boxes_into(W, H, best, pad)


# Everything that is not the grid, keyed by its `window_layout` name. A miss
# lands on `None` and therefore on the grid, which is how an unknown layout
# string has always behaved; a member that returns None (the desktop layout,
# on rects with no origin) falls back the same way.
_ARRANGEMENTS = {
    "desktop": _desktop_placements,
    "feature": _feature_placements,
    "row": _row_placements,
    "column": _column_placements,
}


def fit_placements(W, H, placements, pad_frac=PAD_FRAC):
    """Take the slack out of an arbitrary arrangement, preserving every card's
    aspect and their relative sizes.

    This is `_scale_boxes_into` and nothing more, so it inherits that
    function's limit exactly: it grows the arrangement until it touches the
    padding on ONE axis and centres the other. It is not a different, better
    fit that the presets are missing.

    Which is why it does two very different things depending on what it is
    handed, and both are correct:

      - on a HAND-DRAGGED arrangement it is the whole point. A drop lands
        exactly where it was dropped, dead space and all, because re-flowing
        the other cards out from under the pointer would be fighting the
        user -- so the slack is real and only an explicit action can take it
        out. Measured on `test_framing.FitPlacements.DRAGGED` -- the
        reproducible stand-in for the real hand-dragged arrangement that
        prompted this, three cards adrift in the middle of a 2880x1800
        canvas: 8.2% of the canvas covered before, 76.9% after -- 9.4x the
        ink, which is one uniform 3.1x on each card's width and height.
        Quote whichever of the two you mean and SAY WHICH: ink scales as the
        square of edge length, so a reader who takes 9.4x for the card's
        width is off by a factor of three. (Re-measured at PAD_FRAC 0.03;
        the earlier text quoted 31.1% -> 67.0% for a differently-scaled
        starting arrangement, which nothing pinned and nobody could
        reproduce.)
      - on PRESET output there is no SCALE left to take: a preset already
        went through this same fit, so it comes back at scale 1.0, to within
        a pixel of re-centring an odd span.
      - EXCEPT `grid`, which is not a fixed point and is the one case a
        docstring here used to deny outright. `_grid_placements` aspect-fits
        each crop inside its OWN uniform cell, so the cards' bounding box can
        sit off-centre while every cell is exactly where it belongs; fitting
        re-centres that box without resizing anything. Measured with two
        windows, scale exactly 1.0 both times: 78px of re-centring on a
        1080x1920 canvas, 7-8px on 2880x1800; `test_framing.py`'s
        `FitPlacements` pins the 2880x1800 case. So a UI may offer this on
        grid output -- it just has to promise a re-centre and not a rescale.

    `pad` comes from the WIDTH and is then used on BOTH axes, matching
    `MultiFramePainter`, so a fitted arrangement is framed identically to one
    an auto-layout produced. Takes and returns `(x, y, w, h)` in CANVAS
    pixels; `.cells` dicts are accepted too. Empty in, empty out.

    NOT the only implementation: the editor's "Fit to frame" button does this
    same arithmetic in JS (`fitCardsToFrame` in `studio_web/editor.js`) so it
    can write the per-card `layout` fractions without a round trip, and it
    works in float `pad` where this rounds. The MCP tool calls this function.
    Two copies of one formula -- keep them in step by hand.
    """
    W, H = int(W), int(H)
    boxes = []
    for cell in placements:
        x, y, w, h = _cell_xywh(cell)
        boxes.append([float(x), float(y),
                      max(1.0, float(w)), max(1.0, float(h))])
    if not boxes:
        return []
    return _scale_boxes_into(W, H, boxes, int(pad_frac * W))


# Window focus, stage 1: how much bigger the subject gets, as a linear scale
# on its own cell. Deliberately moderate -- this is only the FIRST rung of
# the ladder, and stage 2 zooms the whole composition in on the grown card,
# so growing too far here would leave the second click nothing to do (at
# 1.8x on a 3-up grid the card already fills the canvas height and the screen
# zoom computes to ~1.0). Capped by the canvas, so a card that is already
# large just grows less.
_FOCUS_GROW = 1.4

# Window focus redraws the drop shadows every frame. At full canvas
# resolution that is a 400ms GaussianBlur; the blur is so wide (sigma ~ 58px
# on a 2880-wide canvas) that the result is heavily band-limited, so it is
# computed on a 1/8 grid and scaled back up. See `_draw_shadow`.
_SHADOW_DOWNSCALE = 8

# A card at rest sits on the BACKDROP; a card lifted by window focus sits on
# its NEIGHBOURS, and the shadow that separates it from the first does not
# separate it from the second. The cast layer is wide by design (sigma =
# 0.02*W, 58px on a 2880 canvas), so where it falls across another window's
# bright content it reads as a gradient rather than a boundary and the grown
# card looks pasted on flat. Darkening it alone does not fix that -- 58px of
# ramp is 58px of ramp. So a lifted card gets TWO layers: the cast one,
# deepened and thrown further (a card further off the plane casts a bigger,
# darker shadow), plus a tight one hugging its edge, which is what actually
# draws the boundary against a neighbour.
#
# All of it scales with `lift_fraction`, and at lift 0 the layer list is
# exactly the single pass that was here before -- so a demoted card, an
# un-emphasized one, and every frame of an un-focused render are unchanged.
_SHADOW_ALPHA = 0.45           # cast layer, resting on the backdrop
_SHADOW_LIFT_ALPHA = 0.66      # cast layer, fully grown
_SHADOW_LIFT_SIGMA = 1.7       # cast blur, x this at full lift
_SHADOW_LIFT_THROW = 2.2       # cast offset, x this at full lift
_CONTACT_SIGMA_FRAC = 0.008    # tight layer's blur, as a fraction of W
_CONTACT_ALPHA = 0.55          # tight layer at full lift (absent at rest)
# The tight layer cannot afford the cast layer's 1/8 grid: that grid places
# the mask by integer division, so it can sit up to 7px off, which is an
# eighth of the cast sigma but a THIRD of this one -- visible as a halo
# heavier on the top-left than the bottom-right. At 1/3 the same error is
# ~1px against a 23px sigma.
_CONTACT_DOWNSCALE = 3
# A layer this faint cannot darken a uint8 pixel by even one level
# (0.002 * 255 = 0.5), so a barely-lifted card skips it rather than paying a
# blur for a no-op.
_SHADOW_MIN_ALPHA = 0.002
_BLUR_MIN_SIGMA_PX = 18.0

# Canvas area below which the baked-plate blur is computed exactly. Sits above
# 2880x1800 (5.2MP), the largest canvas an ordinary framed/multi-window export
# reaches, and below the 7.4MP a Retina multi-window scene canvas reaches --
# so `_blur_downscale` is a no-op for every take that was never slow.
_BLUR_MIN_AREA = 6000000


def lift_fraction(base_w, target_w, w):
    """How far a card has grown from its base cell toward its FOCUSED one,
    0..1 -- the one number a drop shadow needs in order to know the card is
    lifted off the plane rather than lying on it.

    Measured against that card's OWN target rather than a fixed scale
    because `focus_placements` caps growth by the canvas: a card with room
    for only 1.05x is fully focused at 1.05x, and its shadow should say so.
    A card with no room at all (target == base) never lifts, which is the
    honest answer -- nothing about it moved. Demoted cards shrink, so they
    come back 0 and keep the resting shadow.
    """
    base_w, target_w = float(base_w), float(target_w)
    if target_w - base_w <= 1e-6:
        return 0.0
    return max(0.0, min(1.0, (float(w) - base_w) / (target_w - base_w)))


def focus_placements(W, H, rects, base, hero, pad, gap):
    """Placement with `hero` as the subject: it grows IN PLACE and overlaps
    its neighbours, which do not move at all.

    The neighbours holding still is the whole point -- what the eye is meant
    to track is one card getting bigger, not the composition rearranging
    itself. Two earlier cuts got this wrong in opposite directions: one
    pushed a camera into the composed grid (which magnified the grid and
    cropped the subject at the frame edge), the other shrank the neighbours
    into a side strip (which moved everything at once and read as a re-layout
    rather than a focus).

    Growth is about the card's own CENTRE, then clamped inside the padding,
    so a card at the edge of the canvas grows inward instead of off it. The
    scale is uniform, so the card's aspect is exact and nothing is cropped --
    the point of focusing a window is to see more of it.
    """
    out = [tuple(int(v) for v in b[:4]) for b in base]
    n = len(out)
    if not n or not (0 <= int(hero) < n):
        return out
    bx, by, bw, bh = out[int(hero)]
    usable_w = max(1, W - 2 * pad)
    usable_h = max(1, H - 2 * pad)
    scale = min(_FOCUS_GROW, usable_w / float(bw), usable_h / float(bh))
    if scale <= 1.0:
        return out                      # already as big as the canvas allows
    nw = max(2, int(round(bw * scale)))
    nh = max(2, int(round(bh * scale)))
    cx, cy = bx + bw / 2.0, by + bh / 2.0
    nx = max(pad, min(W - pad - nw, int(round(cx - nw / 2.0))))
    ny = max(pad, min(H - pad - nh, int(round(cy - nh / 2.0))))
    out[int(hero)] = (nx, ny, nw, nh)
    return out


def _cell_xywh(c):
    """`(x, y, w, h)` ints from either a placement tuple or a `.cells` dict."""
    if isinstance(c, dict):
        return (int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"]))
    return tuple(int(v) for v in c[:4])


# A card on its way OUT of the main area finishes shrinking well before it
# finishes travelling. Both ends of the blend are collision-free layouts, but
# a straight lerp of two boxes is not: measured on a 3-card grid, the two
# demoted cards overlapped by ~56x16px at the halfway point, and stage 1
# RESTS at halfway, so it is a steady-state artefact rather than a flicker.
# Shrinking ahead of the move ("get out of the way, then go") keeps them
# clear without changing either endpoint.
_FOCUS_SHRINK_LEAD = 1.8


def blend_placements(base, targets, weights):
    """Lerp `base` placements toward each `targets[i]` by `weights[i]`.

    One continuous blend rather than "switch layout at a threshold", so a
    card handing the subject over to another is a rearrangement rather than
    a cut -- the outgoing hero shrinks while the incoming one grows.

    Weights are normalized when they sum past 1 (they overlap while one card
    eases out and the next eases in), because past that the blend would
    extrapolate beyond BOTH layouts and throw cards off the canvas.

    Returns `(placements, order)` -- `order` is the draw order, least
    emphasized first, so the subject is painted OVER its neighbours, which
    is what makes the stage-1 overlap read as the focused card being in
    front rather than as a hole punched in it. Without it a demoted card
    mid-travel is painted UNDER the growing hero and simply vanishes
    (measured: at stage 1 of a 3-card take, one card was completely hidden).
    """
    n = len(base)
    total = sum(max(0.0, float(w)) for w in weights)
    order = sorted(range(n), key=lambda j: float(weights[j])
                   if j < len(weights) else 0.0)
    if total <= 1e-6:
        return list(base), order
    k = (1.0 / total) if total > 1.0 else 1.0
    out = []
    for j, b in enumerate(base):
        x, y, w, h = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        for i, wt in enumerate(weights):
            wt = max(0.0, float(wt)) * k
            if wt <= 1e-6:
                continue
            t = targets[i][j]
            # Shrinking cards lead with the size; growing ones (the hero)
            # stay linear, so it can never outrun its own box and clip the
            # canvas edge mid-move.
            st = min(1.0, wt * _FOCUS_SHRINK_LEAD) if t[2] < b[2] else wt
            x += wt * (t[0] - b[0])
            y += wt * (t[1] - b[1])
            w += st * (t[2] - b[2])
            h += st * (t[3] - b[3])
        out.append((int(round(x)), int(round(y)),
                    max(2, int(round(w))), max(2, int(round(h)))))
    return out, order


def _apply_card_overrides(W, H, placements, rects):
    """Replace any card's placement with its manual `layout` fraction.

    Applied on top of whichever auto-layout ran, so dragging a card works
    identically under every arrangement. Fractions are of the CANVAS, so a
    dragged card keeps its position when the export aspect changes.
    """
    out = list(placements)
    for i, rect in enumerate(rects):
        if not isinstance(rect, dict):
            continue
        lay = rect.get("layout")
        if not isinstance(lay, dict):
            continue
        try:
            lx, ly = float(lay["x"]), float(lay["y"])
            lw, lh = float(lay["w"]), float(lay["h"])
        except (KeyError, TypeError, ValueError):
            continue
        fw = max(2, min(W, int(round(lw * W))))
        fh = max(2, min(H, int(round(lh * H))))
        ix = max(0, min(W - fw, int(round(lx * W))))
        iy = max(0, min(H - fh, int(round(ly * H))))
        out[i] = (ix, iy, fw, fh)
    return out


class MultiFramePainter:
    """Composites N statically-cropped 'window' cards onto one shared
    framed background -- the padded/rounded/shadowed `FramePainter` look,
    generalized from one recording to a grid of 1-4.

    Unlike `FramePainter` (whose recording is always pre-fit to the canvas
    aspect upstream by the camera plan), each window's crop can be any
    aspect independent of its grid cell, so every cell gets its own
    aspect-fit + letterbox against the shared background rather than one
    canvas-wide inner box. A 1-window layout is therefore not an alias for
    today's `style="framed"` -- it's this same per-cell computation with
    n=1, and (unlike framed style) has no auto-zoom camera driving it.
    """

    def __init__(self, W, H, rects, background=None, pad_frac=PAD_FRAC,
                 gap_frac=0.02, layout="grid"):
        W, H = int(W), int(H)
        self.W, self.H = W, H
        pad = int(pad_frac * W)
        gap = max(0, int(gap_frac * W))

        placements = None
        arrange = _ARRANGEMENTS.get(layout)
        if arrange is not None:
            placements = arrange(W, H, rects, pad, gap)
        if placements is None:
            placements = _grid_placements(W, H, rects, pad, gap)
        placements = _apply_card_overrides(W, H, placements, rects)
        self.layout = layout
        # Kept for the animated (window-focus) path, which has to recompute
        # placements and rebuild the backdrop for cells that move each frame.
        self._rects = list(rects)
        self._pad, self._gap = pad, gap
        self._placements = [tuple(p) for p in placements]
        self._focus_widths = None   # lazily, for the lift-scaled shadow
        self._bg = resolve_background(background, W, H)

        shadow = np.zeros((H, W), np.uint8)
        self._cells = []  # (ix, iy, fw, fh, mask) -- mask is 0/255, HxW
        self._radii = []  # corner radius per cell, parallel to _cells
        clamped = []      # radius as _rounded_mask actually applied it
        for (ix, iy, fw, fh) in placements:
            radius = max(8, int(0.02 * fw))
            mask = _rounded_mask(fw, fh, radius)
            self._cells.append((ix, iy, fw, fh, mask))
            self._radii.append(radius)
            clamped.append(int(min(radius, fw // 2, fh // 2)))

            shadow_y = max(0, min(H - fh, iy + int(0.012 * H)))
            shadow[shadow_y:shadow_y + fh, ix:ix + fw] = np.maximum(
                shadow[shadow_y:shadow_y + fh, ix:ix + fw], mask)

        # One shared blur pass for all N cells' shadows, not N separate ones.
        shadow = _wide_blur(shadow, max(1.0, 0.02 * W))
        sh = (shadow.astype(np.float32) / 255.0)[:, :, None] * _SHADOW_ALPHA
        # `self._bg` is what `resolve_background` just returned, so building it
        # a second time here only re-ran the gradient (0.08s a scene at these
        # canvas sizes). Aliased rather than copied because nothing writes
        # through `base` -- the composite below ALLOCATES -- and `self._bg`'s
        # only other readers hand out `.astype(np.uint8)` copies of their own
        # (`background_plate`, `_backdrop`). Keep it read-only.
        base = self._bg
        # Quantize the static backdrop ONCE. `paint` returns uint8, and every
        # pixel it doesn't overwrite is exactly this truncation of that float
        # composite -- so carrying the float canvas per frame (copy it, then
        # convert the whole thing back down) is 50ms/frame of arithmetic on
        # pixels that never change. Bit-exact: same values, same cast.
        self._plate = (base * (1.0 - sh)).astype(np.uint8)
        self._corners = [
            self._corner_repairs(cell, r) for cell, r in zip(self._cells, clamped)
        ]

    def _corner_repairs(self, cell, radius):
        """The only pixels of a cell's rect that are NOT its recording: the
        rounded corners. `paint` resizes each crop straight over its whole
        rect and then puts these back, which is why it needs no per-pixel
        blend at all.

        Safe because `_rounded_mask`'s two rectangles already cover
        everything outside the four `radius`-sized corner squares, so the
        mask's zeros can't live anywhere else.
        """
        ix, iy, fw, fh, mask = cell
        plate = self._plate[iy:iy + fh, ix:ix + fw]
        out = []
        for ys, xs in ((slice(0, radius), slice(0, radius)),
                       (slice(0, radius), slice(fw - radius, fw)),
                       (slice(fh - radius, fh), slice(0, radius)),
                       (slice(fh - radius, fh), slice(fw - radius, fw))):
            outside = (mask[ys, xs] == 0)[:, :, None]
            if not outside.any():
                continue
            out.append((ys, xs, outside, plate[ys, xs].copy()))
        return out

    @property
    def cells(self):
        """Placement geometry, canvas px, one dict per window in build order.

        Exposed so the editor's live preview can redraw this exact grid in a
        browser canvas (a <video> element can only ever show one crop). The
        alternative -- reimplementing the grid in JS -- is a second source of
        truth that drifts the moment pad/gap/aspect-fit changes here.
        """
        return [{"x": int(ix), "y": int(iy), "w": int(fw), "h": int(fh),
                 "radius": int(r)}
                for (ix, iy, fw, fh, _mask), r in zip(self._cells, self._radii)]

    def background_plate(self):
        """The backdrop with NO shadows baked in.

        Window focus moves the cells, so the baked shadows belong to a
        layout that is no longer on screen. The editor draws its own over
        this instead; the export rebuilds them properly in `_plate_for`.
        """
        return self._bg.astype(np.uint8)

    def base_plate(self):
        """Background + drop shadows with no windows painted over it -- what
        `paint` starts from. The live preview draws this once and blits the
        video crops on top, so the backdrop needs no JS-side color math."""
        return self._plate.copy()

    def paint(self, crops):
        """`crops`: a list of raw (uncropped-to-cell-size) BGR arrays, one
        per window, in the same order the painter was built with.

        Composited in uint8, with no per-pixel alpha math: `_rounded_mask`
        draws with aliased fills, so a cell's mask is only ever 0 or 255 and
        the old `rec * m + bg * (1 - m)` was a *select* wearing a blend's
        clothes. Every value it could produce is reachable exactly here --
        inside the card, `rec` round-trips 0..255 through float32 unchanged;
        outside it, the plate is the same truncation of the same float. So
        this is byte-for-byte what the float version returned, ~8x cheaper.
        A fresh array per call, since callers draw on top of it in place
        (the facecam bubble) and would otherwise smear across frames.
        """
        out = self._plate.copy()
        for (ix, iy, fw, fh, _mask), corners, crop in zip(
                self._cells, self._corners, crops):
            roi = out[iy:iy + fh, ix:ix + fw]
            # Resize STRAIGHT into the canvas -- no full-cell temporary.
            fit = cv2.resize(crop, (fw, fh), dst=roi,
                             interpolation=cv2.INTER_AREA)
            if fit is not roi:
                # OpenCV refused the in-place destination (it reallocates on
                # any dtype mismatch -- a non-uint8 crop) and handed back its
                # own buffer. Assigning casts exactly like the old .astype().
                roi[:] = fit
            for ys, xs, outside, plate in corners:
                np.copyto(roi[ys, xs], plate, where=outside)
        return out


    # -- window focus: cells that move ------------------------------------
    #
    # Everything above is built for a layout that never changes: the cells,
    # their masks, the drop shadows and the backdrop are all baked once, and
    # `paint` is then a resize per card into a fixed box. Window focus moves
    # the cells, so the backdrop has to be rebuilt every frame -- which is
    # affordable only because of the two tricks below.

    @property
    def source_sizes(self):
        """`(w, h)` of each card's SOURCE rect, in recording px.

        What bounds how far window focus may magnify a card: the cell is
        already some fraction of this, and pushing the camera past it is
        inventing detail the capture never had.
        """
        return [_rect_wh(r) for r in self._rects]

    @property
    def base_placements(self):
        """The un-focused layout, `(x, y, w, h)` per card. What
        `focus_placements` blends away from."""
        return list(self._placements)

    def focus_targets(self, hero):
        """`focus_placements` for this painter's own canvas, rects and base
        layout."""
        return focus_placements(self.W, self.H, self._rects,
                                self._placements, hero, self._pad, self._gap)

    @property
    def focus_widths(self):
        """Each card's width in its OWN focused layout -- the denominator of
        `lift_fraction`. Computed once, on the first frame that needs it,
        because `paint()` never does."""
        if self._focus_widths is None:
            self._focus_widths = [self.focus_targets(i)[i][2]
                                  for i in range(len(self._placements))]
        return self._focus_widths

    def lift_of(self, index, w):
        """`lift_fraction` for card `index` currently drawn `w` px wide."""
        if not (0 <= int(index) < len(self._placements)):
            return 0.0
        return lift_fraction(self._placements[int(index)][2],
                             self.focus_widths[int(index)], w)

    def _shadow_layers(self, lift):
        """`(sigma, dy, alpha, downscale)` per shadow layer for a card at
        `lift` (0..1, from `lift_fraction`).

        At rest this is ONE layer: the same wide, gently-thrown cast shadow
        the baked backdrop carries, at the same alpha. A lifted card gets
        that layer deepened and thrown further, plus the tight contact layer
        that separates it from the neighbour it is now sitting on. See the
        constants above for why both are needed.
        """
        W, H = self.W, self.H
        sigma = max(1.0, 0.02 * W)
        dy = int(0.012 * H)
        lift = max(0.0, min(1.0, float(lift)))
        if lift <= 0.0:
            return [(sigma, dy, _SHADOW_ALPHA, _SHADOW_DOWNSCALE)]
        def _at(rest, full):
            return rest + lift * (full - rest)
        return [
            (sigma * _at(1.0, _SHADOW_LIFT_SIGMA),
             int(dy * _at(1.0, _SHADOW_LIFT_THROW)),
             _at(_SHADOW_ALPHA, _SHADOW_LIFT_ALPHA),
             _SHADOW_DOWNSCALE),
            (max(1.0, _CONTACT_SIGMA_FRAC * W), 0,
             _CONTACT_ALPHA * lift, _CONTACT_DOWNSCALE),
        ]

    def _draw_shadow(self, out, cell, lift=0.0):
        """Darken `out` in place with one card's drop shadow.

        Drawn immediately before its own card rather than baked into a
        single backdrop, because window focus makes the subject OVERLAP its
        neighbours -- a shadow that only ever falls on the background gives
        no depth cue at precisely the moment the composition depends on one,
        and the grown card reads as pasted flat on top.

        `lift` (0 for a card at its base size, 1 for a fully grown one)
        picks the layers: one at rest, two once the card is off the plane.
        """
        for sigma, dy, alpha, d in self._shadow_layers(lift):
            if alpha > _SHADOW_MIN_ALPHA:
                self._shadow_layer(out, cell, sigma, dy, alpha, d)

    def _shadow_layer(self, out, cell, sigma, dy, alpha, d):
        """One blurred, offset, multiplied layer of a card's shadow.

        Delegates to the module-level `draw_drop_shadow` (shared verbatim with
        render's whole-screen grow pass); `out` is the full canvas, so
        `out.shape[:2]` is exactly this painter's `(H, W)`. Kept as a method
        so the compositor call sites read the same and the existing per-card
        shadow tests stay a fair comparison against the baked plate.

        Two measurements keep it affordable (see `draw_drop_shadow`): computed
        at 1/d resolution and scaled back up, and bounded to the card's own
        3-sigma neighbourhood, composited in integer via cv2.
        """
        draw_drop_shadow(out, cell, sigma, dy, alpha, d)

    def paint_at(self, crops, cells, order=None):
        """Composite `crops` into per-frame `cells` (window focus).

        `order` is the draw order (least emphasized first). It matters
        because a demoted card travelling toward the strip passes BEHIND the
        growing hero, and painting it after would punch a hole in the
        subject; painting the hero last is also just what "this is the thing
        you are looking at" should mean.

        Falls through to the baked `paint()` whenever the cells are exactly
        the base layout, so the un-emphasized stretches of a focused render
        stay byte-identical to the un-focused one -- the animation only
        costs what it actually animates.
        """
        cells = [_cell_xywh(c) for c in cells]
        if cells == self._placements:
            return self.paint(crops)
        out = self._bg.astype(np.uint8)
        seq = order if order is not None else range(len(cells))
        for j in seq:
            if j >= len(cells) or j >= len(crops):
                continue
            (ix, iy, fw, fh), crop = cells[j], crops[j]
            # Shadow first, onto whatever is already there -- background for
            # a card standing alone, a neighbour for the grown subject. How
            # far this card has grown is also how far off the plane it reads
            # as being, so it picks the shadow (see `_shadow_layers`).
            self._draw_shadow(out, (ix, iy, fw, fh), self.lift_of(j, fw))
            fw = max(2, min(self.W, fw))
            fh = max(2, min(self.H, fh))
            ix = max(0, min(self.W - fw, ix))
            iy = max(0, min(self.H - fh, iy))
            radius = max(8, int(0.02 * fw))
            r = int(min(radius, fw // 2, fh // 2))
            roi = out[iy:iy + fh, ix:ix + fw]
            mask = _rounded_mask(fw, fh, radius)
            # The four corner squares are the only pixels of the cell that
            # are NOT the recording (same fact `_corner_repairs` rests on),
            # so stash just those before the resize scribbles over them.
            keep = [(ys, xs, (mask[ys, xs] == 0)[:, :, None],
                     roi[ys, xs].copy())
                    for ys, xs in ((slice(0, r), slice(0, r)),
                                   (slice(0, r), slice(fw - r, fw)),
                                   (slice(fh - r, fh), slice(0, r)),
                                   (slice(fh - r, fh), slice(fw - r, fw)))]
            fit = cv2.resize(crop, (fw, fh), dst=roi,
                             interpolation=cv2.INTER_AREA)
            if fit is not roi:
                roi[:] = fit
            for ys, xs, outside, plate in keep:
                np.copyto(roi[ys, xs], plate, where=outside)
        return out


def placements_for(W, H, rects, layout="grid", pad_frac=PAD_FRAC,
                   gap_frac=0.02):
    """The cell rects a `MultiFramePainter` WOULD lay out, without building one.

    Exactly the placement half of its constructor -- same arrangement, same
    truncated pad/gap -- minus the expensive half (background resolve plus a
    baked full-canvas shadow blur, measured at ~1.8s on a 2880x1800 canvas).

    Exists so a caller can SIZE the canvas from the layout before committing
    to it: you cannot know how far a window buffer will be downscaled until
    you know its cell, and building a painter per candidate size to find out
    would cost more than the render.
    """
    W, H = int(W), int(H)
    pad = int(pad_frac * W)
    gap = max(0, int(gap_frac * W))
    placements = None
    arrange = _ARRANGEMENTS.get(layout)
    if arrange is not None:
        placements = arrange(W, H, rects, pad, gap)
    if placements is None:
        placements = _grid_placements(W, H, rects, pad, gap)
    return [tuple(p) for p in _apply_card_overrides(W, H, placements, rects)]


def make_multi_painter(W, H, windows, background=None, layout="grid"):
    """Return a `MultiFramePainter` for 1-4 window specs, else None."""
    if not windows:
        return None
    return MultiFramePainter(W, H, windows, background=background,
                             layout=layout)
