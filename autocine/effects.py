"""Render-time overlays: click highlights and an optional cursor spotlight.

Both effects are drawn in *output* pixel space, on top of the camera-warped
frame (before any framing/background is composited). They are camera-aware:
each screen-space point (recorded pixels) is mapped through the same window the
renderer used for that frame — top-left (x0, y0) and zoom z — so a ripple stays
glued to the UI element that was clicked and scales naturally with the zoom.

    out_x = (screen_x - x0) * z
    out_y = (screen_y - y0) * z

Nothing here needs macOS permissions: it operates purely on the recorded
event stream + the per-frame camera transform, so it is fully unit-testable
with a synthetic session.
"""

import os
from math import exp, hypot

import cv2
import numpy as np


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


# ---- click ripple -----------------------------------------------------------
CLICK_DEFAULTS = dict(
    duration=0.62,        # s, ring lifetime
    r0_frac=0.006,        # start radius, fraction of recorded W (screen space)
    r_max_frac=0.055,     # end radius
    thickness_frac=0.0035,  # ring stroke, fraction of recorded W
    color=(244, 244, 244),  # BGR (soft white)
    peak_alpha=0.55,      # ring opacity at t=click
    pulse_frac=0.22,      # pulse lives in the first `pulse_frac` of duration
    pulse_alpha=0.34,     # filled-dot opacity at t=click
)

# named tints for --click-color (BGR)
CLICK_COLORS = {
    "white": (244, 244, 244),
    "black": (24, 24, 24),
    "yellow": (60, 220, 250),
    "blue": (240, 170, 70),
    "green": (120, 210, 90),
    "pink": (200, 120, 240),
    "red": (70, 70, 235),
}


def resolve_color(value, default=(244, 244, 244)):
    """Accept a named color, a '#rrggbb' hex string, or an (B,G,R) tuple."""
    if value is None:
        return default
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    s = str(value).strip().lower()
    if s in CLICK_COLORS:
        return CLICK_COLORS[s]
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6:
        try:
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
            return (b, g, r)  # BGR
        except ValueError:
            pass
    return default


def _ease_out(u):
    """Cubic ease-out: fast start, gentle finish (1 - (1-u)^3)."""
    v = 1.0 - u
    return 1.0 - v * v * v


def _alpha_circle(img, cx, cy, radius, color, alpha, thickness):
    """Alpha-composite a (possibly filled) circle onto img within a tight ROI.

    Draws the circle into a copy of the ROI then blends: untouched pixels are
    `alpha*roi + (1-alpha)*roi = roi` (identity), so only the drawn pixels
    change. thickness < 0 fills the disc.
    """
    if alpha <= 0.0 or radius <= 0:
        return
    h, w = img.shape[:2]
    pad = (abs(thickness) if thickness > 0 else 0) + 2
    x0 = int(np.floor(cx - radius - pad)); x1 = int(np.ceil(cx + radius + pad))
    y0 = int(np.floor(cy - radius - pad)); y1 = int(np.ceil(cy + radius + pad))
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(w, x1); y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return  # entirely off-frame
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    cv2.circle(layer, (int(round(cx)) - x0, int(round(cy)) - y0),
               int(round(radius)), color, thickness, cv2.LINE_AA)
    a = float(min(1.0, max(0.0, alpha)))
    cv2.addWeighted(layer, a, roi, 1.0 - a, 0.0, dst=roi)


class ClickFX:
    """Draws an expanding, fading ring (+ a quick pulse) at every click."""

    def __init__(self, W, H, clicks_t, clicks_x, clicks_y, params=None):
        self.W, self.H = W, H
        p = dict(CLICK_DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.color = resolve_color(p.get("color"), CLICK_DEFAULTS["color"])
        self.dur = float(p["duration"])
        self.r0 = float(p["r0_frac"]) * W
        self.rmax = float(p["r_max_frac"]) * W
        self.th = max(1, int(round(float(p["thickness_frac"]) * W)))
        # (t, x, y) sorted by time; times are media-relative seconds.
        clicks = sorted(zip([float(t) for t in clicks_t],
                            [float(x) for x in clicks_x],
                            [float(y) for y in clicks_y]))
        self.clicks = clicks
        self._lo = 0  # moving lower bound (draw() is called with rising t)

    def draw(self, img, t, x0, y0, z):
        dur, clicks = self.dur, self.clicks
        # advance past clicks whose ripple has fully faded
        while self._lo < len(clicks) and clicks[self._lo][0] < t - dur:
            self._lo += 1
        i = self._lo
        n = len(clicks)
        while i < n and clicks[i][0] <= t:
            ct, cx, cy = clicks[i]
            i += 1
            u = (t - ct) / dur
            if u < 0.0 or u > 1.0:
                continue
            ox = (cx - x0) * z
            oy = (cy - y0) * z
            # expanding ring
            r = (self.r0 + (self.rmax - self.r0) * _ease_out(u)) * z
            fade = (1.0 - u)
            a_ring = self.p["peak_alpha"] * fade * fade
            _alpha_circle(img, ox, oy, r, self.color, a_ring,
                          max(1, int(round(self.th * z))))
            # quick filled pulse at the very start ("tap" feedback)
            pf = self.p["pulse_frac"]
            if u < pf:
                pu = u / pf
                a_dot = self.p["pulse_alpha"] * (1.0 - pu)
                _alpha_circle(img, ox, oy, self.r0 * z, self.color, a_dot, -1)


# ---- cursor spotlight -------------------------------------------------------
SPOTLIGHT_DEFAULTS = dict(
    radius_frac=0.16,   # bright-core radius, fraction of output W
    feather_frac=0.10,  # soft falloff width, fraction of output W
    dim=0.45,           # how dark the surround gets (0=none, 1=black)
    zoom_only=True,     # only dim while zoomed in (z > 1.02)
)


class Spotlight:
    """Darkens the frame except a soft radial disc around the cursor.

    The radial falloff texture is precomputed once; each frame multiplies a
    dark copy of the frame back toward full brightness inside the disc ROI —
    O(disc area) per frame, not O(frame).
    """

    def __init__(self, W, H, params=None):
        p = dict(SPOTLIGHT_DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.dim = float(p["dim"])
        self.zoom_only = bool(p["zoom_only"])
        r = max(1.0, float(p["radius_frac"]) * W)
        f = max(1.0, float(p["feather_frac"]) * W)
        self.R = int(np.ceil(r + f))
        yy, xx = np.ogrid[-self.R:self.R + 1, -self.R:self.R + 1]
        dist = np.sqrt(xx * xx + yy * yy).astype(np.float32)
        # smooth 1 (inside core) -> 0 (past core+feather)
        m = np.clip((r + f - dist) / f, 0.0, 1.0)
        m = m * m * (3.0 - 2.0 * m)  # smoothstep
        # brightness factor tex in [1-dim, 1]; multiply frame by this in ROI
        self.tex = (1.0 - self.dim * (1.0 - m)).astype(np.float32)[:, :, None]

    def draw(self, img, cx, cy, z):
        if self.dim <= 0.0:
            return
        if self.zoom_only and z <= 1.02:
            return
        h, w = img.shape[:2]
        # 1) darken the whole frame toward the surround level
        cv2.convertScaleAbs(img, dst=img, alpha=1.0 - self.dim, beta=0.0)
        # 2) restore brightness inside the disc via the precomputed falloff
        R = self.R
        cxi, cyi = int(round(cx)), int(round(cy))
        x0 = cxi - R; y0 = cyi - R
        tx0 = max(0, -x0); ty0 = max(0, -y0)
        ix0 = max(0, x0); iy0 = max(0, y0)
        ix1 = min(w, cxi + R + 1); iy1 = min(h, cyi + R + 1)
        if ix1 <= ix0 or iy1 <= iy0:
            return
        tw = ix1 - ix0; th = iy1 - iy0
        tex = self.tex[ty0:ty0 + th, tx0:tx0 + tw]
        # img was multiplied by (1-dim); dividing the disc by (1-dim) and
        # multiplying by tex lands each pixel at original*tex.
        roi = img[iy0:iy1, ix0:ix1].astype(np.float32)
        roi *= tex / max(1e-3, (1.0 - self.dim))
        np.clip(roi, 0, 255, out=roi)
        img[iy0:iy1, ix0:ix1] = roi.astype(np.uint8)


# ---- synthetic cursor --------------------------------------------------------
CURSOR_DEFAULTS = dict(
    size_frac=0.028,          # base cursor height, fraction of output W
    scale=1.0,                # user-facing extra size multiplier (--cursor-size)
    smoothing_tau=0.05,       # s, EMA time constant for the drawn position
    idle_after=1.2,           # s of no significant movement before fading out
    fade_dur=0.35,            # s, idle fade in/out duration
    move_threshold_px=1.5,    # screen-space px to count as "moved" (anti-jitter)
    tilt_gain=0.05,           # deg of lean per (px/s) of horizontal velocity
    tilt_max_deg=14.0,        # cap on lean angle
    click_pulse=0.22,         # extra scale fraction right after a click
    click_pulse_dur=0.16,     # s, decay time of the click "tap" pulse
    color="white",            # fill color: name or #rrggbb
    outline=(30, 30, 30),     # BGR outline
    shadow_alpha=0.35,
    shadow_offset_frac=0.12,  # fraction of cursor size
)

# Normalized arrow-cursor silhouette (tip at the origin, unit height ~1.18).
_CURSOR_SHAPE = np.array([
    [0.00, 0.00],
    [0.00, 1.00],
    [0.25, 0.78],
    [0.42, 1.18],
    [0.58, 1.10],
    [0.42, 0.70],
    [0.72, 0.70],
], dtype=np.float32)


class CursorFX:
    """Draws a smoothed, size-scaled synthetic cursor from the move track.

    Wants a frame with no pointer already in it -- drawing this on top of a
    recording that still has the system cursor baked into the pixels would
    double up. Two ways to get one: record with `--cursor synthetic` (the real
    OS cursor hidden at capture time, see record.py), or turn the cursor
    eraser on, which lifts the recorded pointer out of an ordinary take
    upstream of the warp. `render._cursor_fx_draws` is the gate; it decides
    before this is ever constructed.

    Precomputes a per-frame smoothed position/tilt/visibility array over the
    full `frame_times` up front (the same pattern camera.build_path uses),
    rather than smoothing incrementally as `draw()` is called. That makes a
    single-frame preview correctly reflect the *cumulative* smoothing since
    the start of the clip instead of resetting at the preview instant --
    smoothing is purely causal (no settle-tail-style lookahead), so truncating
    `frame_times` to end at the preview point is always safe here.
    """

    def __init__(self, W, H, frame_times, moves_t, moves_x, moves_y,
                clicks_t=None, params=None):
        p = dict(CURSOR_DEFAULTS)
        if params:
            p.update(params)
        self.p = p
        self.W, self.H = W, H
        self.size = max(2.0, float(p["size_frac"]) * W * float(p["scale"]))
        self.color = resolve_color(p.get("color"), (255, 255, 255))
        self.outline = resolve_color(p.get("outline"), (30, 30, 30))

        T = len(frame_times)
        self.sx = np.zeros(T)
        self.sy = np.zeros(T)
        self.tilt = np.zeros(T)
        self.alpha = np.zeros(T)
        self.pulse = np.zeros(T)
        if T == 0:
            return

        moves_t = moves_t if moves_t is not None else np.array([])
        if moves_t.size:
            raw_x = np.interp(frame_times, moves_t, moves_x,
                              left=moves_x[0], right=moves_x[-1])
            raw_y = np.interp(frame_times, moves_t, moves_y,
                              left=moves_y[0], right=moves_y[-1])
        else:
            raw_x = np.full(T, W / 2.0)
            raw_y = np.full(T, H / 2.0)

        clicks = sorted(float(c) for c in (clicks_t if clicks_t is not None else []))
        ci = 0
        cx, cy = float(raw_x[0]), float(raw_y[0])
        last_move_t = float(frame_times[0])
        last_click_t = None
        prev_t = float(frame_times[0])
        tau = max(1e-4, float(p["smoothing_tau"]))
        idle_after = float(p["idle_after"])
        fade_dur = max(1e-4, float(p["fade_dur"]))
        move_thresh = float(p["move_threshold_px"])
        tilt_gain = float(p["tilt_gain"])
        tilt_max = float(p["tilt_max_deg"])
        pulse_dur = max(1e-4, float(p["click_pulse_dur"]))
        pulse_amt = float(p["click_pulse"])

        for i in range(T):
            t = float(frame_times[i])
            dt = (t - prev_t) if i > 0 else (1.0 / 60.0)
            dt = max(1e-4, dt)
            prev_t = t

            dx = float(raw_x[i]) - cx
            dy = float(raw_y[i]) - cy
            moved = hypot(dx, dy) > move_thresh
            a = 1.0 - exp(-dt / tau)
            nx, ny = cx + a * dx, cy + a * dy
            vx = (nx - cx) / dt
            cx, cy = nx, ny
            if moved:
                last_move_t = t

            while ci < len(clicks) and clicks[ci] <= t:
                last_click_t = clicks[ci]
                ci += 1

            idle_elapsed = t - last_move_t
            if idle_elapsed <= idle_after:
                alpha = 1.0
            else:
                alpha = 1.0 - (idle_elapsed - idle_after) / fade_dur
            self.alpha[i] = _clamp(alpha, 0.0, 1.0)
            self.tilt[i] = _clamp(vx * tilt_gain, -tilt_max, tilt_max)

            if last_click_t is not None:
                since_click = t - last_click_t
                if since_click < pulse_dur:
                    self.pulse[i] = pulse_amt * (1.0 - since_click / pulse_dur)

            self.sx[i] = cx
            self.sy[i] = cy

    def draw(self, img, idx, x0, y0, z_eff, z_eff_y=None, pos=None):
        """Paint the cursor onto `img`, mapping source px through
        `(p - x0) * z_eff`.

        `pos` overrides the smoothed position for this call, for the
        multi-window compositor: there the same instant is drawn into several
        cards, each of which wants the cursor expressed in ITS OWN space. The
        smoothing, fade and pulse still come from `idx` -- only the point
        moves -- so this stays one cursor seen through N transforms rather
        than N cursors.

        `z_eff_y` exists for the multi-window compositor, the one caller whose
        source->output scale can differ per axis: a card bound to a window
        that CHANGED ASPECT mid-take is cropped to the tracked rect and
        resized back to the drawn rect's size, so x and y are scaled by
        different factors. Left None (every other caller) it is z_eff and this
        is bit-exact with the single-scale form. The glyph itself is always
        sized by z_eff -- a stretched cursor would read as a rendering bug,
        not as the window having been resized.
        """
        if idx < 0 or idx >= len(self.sx):
            return
        alpha = float(self.alpha[idx])
        if alpha <= 0.0:
            return
        zy = z_eff if z_eff_y is None else z_eff_y
        px = float(self.sx[idx]) if pos is None else float(pos[0])
        py = float(self.sy[idx]) if pos is None else float(pos[1])
        ox = (px - x0) * z_eff
        oy = (py - y0) * zy
        size = self.size * z_eff * (1.0 + float(self.pulse[idx]))
        ang = np.radians(float(self.tilt[idx]))
        c, s = np.cos(ang), np.sin(ang)
        rot = np.array([[c, -s], [s, c]], dtype=np.float32)
        pts = (_CURSOR_SHAPE * size) @ rot.T
        pts[:, 0] += ox
        pts[:, 1] += oy
        self._paint(img, pts, alpha)

    def sample_track(self, stride=1):
        """Everything a NON-Python renderer needs to draw this same cursor.

        The editor's live multi-window composite is a browser canvas, so it
        cannot call `draw`. Handing over the precomputed per-frame state (plus
        the glyph and its size) keeps `_CURSOR_SHAPE`, the EMA smoothing, the
        idle fade and the click pulse defined exactly once, here -- the same
        reason `MultiFramePainter` exposes `.cells`/`.base_plate()` instead of
        letting JS recompute the grid.

        Positions are in SOURCE px and `size` is in source px too, so the
        consumer applies whatever card transform it is already using for the
        video crop. Thinned by `stride` frames; the consumer interpolates.
        """
        stride = max(1, int(stride))
        sel = slice(None, None, stride)
        return {
            "shape": [[float(a), float(b)] for a, b in _CURSOR_SHAPE],
            "size": float(self.size),
            "stride": stride,
            "x": [round(float(v), 2) for v in self.sx[sel]],
            "y": [round(float(v), 2) for v in self.sy[sel]],
            "a": [round(float(v), 3) for v in self.alpha[sel]],
            "tilt": [round(float(v), 2) for v in self.tilt[sel]],
            "pulse": [round(float(v), 3) for v in self.pulse[sel]],
            "shadow_alpha": float(self.p["shadow_alpha"]),
            "shadow_offset_frac": float(self.p["shadow_offset_frac"]),
        }

    def _paint(self, img, pts, alpha):
        h, w = img.shape[:2]
        pad = 6
        x0i = int(np.floor(pts[:, 0].min())) - pad
        x1i = int(np.ceil(pts[:, 0].max())) + pad
        y0i = int(np.floor(pts[:, 1].min())) - pad
        y1i = int(np.ceil(pts[:, 1].max())) + pad
        rx0, ry0 = max(0, x0i), max(0, y0i)
        rx1, ry1 = min(w, x1i), min(h, y1i)
        if rx1 <= rx0 or ry1 <= ry0:
            return  # entirely off-frame
        roi = img[ry0:ry1, rx0:rx1]
        local = pts.copy()
        local[:, 0] -= rx0
        local[:, 1] -= ry0
        poly = np.round(local).astype(np.int32).reshape(1, -1, 2)

        shadow_off = max(1, int(round(self.size * float(self.p["shadow_offset_frac"]))))
        shadow_poly = poly.copy()
        shadow_poly[..., 0] += shadow_off
        shadow_poly[..., 1] += shadow_off
        shadow_layer = roi.copy()
        cv2.fillPoly(shadow_layer, shadow_poly, (0, 0, 0), cv2.LINE_AA)
        a_shadow = float(self.p["shadow_alpha"]) * alpha
        cv2.addWeighted(shadow_layer, a_shadow, roi, 1.0 - a_shadow, 0.0, dst=roi)

        cursor_layer = roi.copy()
        outline_th = max(1, int(round(self.size * 0.05)))
        cv2.fillPoly(cursor_layer, poly, self.color, cv2.LINE_AA)
        cv2.polylines(cursor_layer, poly, True, self.outline, outline_th, cv2.LINE_AA)
        cv2.addWeighted(cursor_layer, alpha, roi, 1.0 - alpha, 0.0, dst=roi)


# ---- facecam bubble ---------------------------------------------------------
FACECAM_DEFAULTS = dict(
    position="bottom-left",   # corner: bottom-left|bottom-right|top-left|top-right
    size_frac=0.20,           # bubble diameter as a fraction of OUTPUT height
    margin_frac=0.035,        # gap from the edges, fraction of output height
    shape="circle",           # "circle" | "rounded"
    # Ring thickness as a fraction of the bubble diameter. OFF by default:
    # a soft-white ring reads as a sticker pasted onto the video, which is
    # exactly the "outer frame" complaint -- and the drop shadow below
    # already separates the bubble from whatever is behind it. Set
    # render.facecam_border to put it back.
    border_frac=0.0,
    border_color=(248, 244, 240),  # BGR (soft white)
    mirror=True,              # selfie-mirror, like every webcam self-view
    shadow=True,
    # Background blur strength in [0, 1]. 0 = OFF (bit-exact -- no vignette
    # precompute, the draw path is untouched). A real person matte needs an ML
    # segmenter (a heavy dep this project forbids), so instead this keeps the
    # centre of the bubble -- where a centre-cropped webcam puts the face --
    # sharp and eases into a blur toward the rim, softening the room behind you
    # without cutting you out. See `_vignette` below.
    blur=0.0,
)


def _bubble_alpha(D, shape, radius_frac=0.32):
    """Feathered [0,1] alpha mask of a DxD bubble (float32, shape (D,D,1))."""
    yy, xx = np.ogrid[:D, :D]
    c = (D - 1) / 2.0
    if shape == "rounded":
        r = radius_frac * D
        dx = np.abs(xx - c) - (D / 2.0 - r)
        dy = np.abs(yy - c) - (D / 2.0 - r)
        dx = np.maximum(dx, 0.0)
        dy = np.maximum(dy, 0.0)
        dist = np.sqrt(dx * dx + dy * dy) - r
        a = np.clip(-dist, 0.0, 1.0)
    else:  # circle
        dist = np.sqrt((xx - c) ** 2 + (yy - c) ** 2) - (D / 2.0 - 1.0)
        a = np.clip(-dist, 0.0, 1.0)
    return a.astype(np.float32)[:, :, None]


class FacecamOverlay:
    """Composites a webcam bubble (face.mov) onto the final output frame.

    Drawn AFTER framing, in output pixel space, at a chosen corner. The face
    track is aligned to the screen timeline via the two monotonic anchors:
    for a screen frame at media time ``t`` the face media time is
    ``t + (t0_screen - face_t0)``. Frames are decoded forward (cheap for the
    monotonic render walk) and only re-seek on the rare backwards step.

    Fully unit-testable: give it a synthetic face.mov and draw onto a blank
    frame -- no macOS permissions, no live camera.
    """

    def __init__(self, face_path, out_w, out_h, t0_screen, face_t0,
                 face_fps=30.0, params=None):
        p = dict(FACECAM_DEFAULTS)
        if params:
            p.update({k: v for k, v in params.items() if v is not None})
        self.p = p
        self.cap = cv2.VideoCapture(face_path)
        # Some codecs report 0/negative frame counts; treat that as "unknown
        # length" and let forward reads hit EOF instead of hard-clamping.
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.n = n if n > 0 else None
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.face_fps = float(fps) if fps and fps > 0 else float(face_fps or 30.0)
        self.offset = float(t0_screen) - float(face_t0)
        self._cur_idx = -1
        self._last = None

        D = max(16, int(round(float(p["size_frac"]) * out_h)))
        D = min(D, out_w, out_h)
        self.D = D
        margin = int(round(float(p["margin_frac"]) * out_h))
        pos = str(p["position"]).lower()
        px = margin if "left" in pos else max(0, out_w - D - margin)
        py = margin if "top" in pos else max(0, out_h - D - margin)
        self.px, self.py = int(px), int(py)
        self.alpha = _bubble_alpha(D, str(p["shape"]))
        self.border_th = max(0, int(round(float(p["border_frac"]) * D / 2.0)))
        self.mirror = bool(p["mirror"])
        self.shadow = bool(p["shadow"])
        # Background blur (a radial vignette, not a person matte -- see the
        # FACECAM_DEFAULTS note). Precomputed once because the bubble never
        # changes size: `_vignette` is a DxD weight in [0,1] that is 0 across a
        # sharp centre core and eases to 1 at the rim, and `_blur_sigma` sets
        # how hard the rim blurs. Left None when off, so `draw` skips it
        # entirely and the output stays byte-identical to the pre-feature path.
        self.blur = max(0.0, min(1.0, float(p.get("blur", 0.0) or 0.0)))
        self._vignette = None
        self._blur_sigma = 0.0
        if self.blur > 0.0:
            self._blur_sigma = max(0.6, self.blur * D * 0.06)
            yy, xx = np.ogrid[:D, :D]
            cc = (D - 1) / 2.0
            rr = np.sqrt((xx - cc) ** 2 + (yy - cc) ** 2) / (D / 2.0)
            core = 0.5                       # the inner half stays fully sharp
            tt = np.clip((rr - core) / (1.0 - core), 0.0, 1.0)
            w = tt * tt * (3.0 - 2.0 * tt)   # smoothstep: no hard blur seam
            self._vignette = w.astype(np.float32)[:, :, None]
        # Precomputed soft shadow: the bubble never moves or changes size, so
        # the blur is one cost at init rather than one per frame. It used to
        # ride on the border being present (a hard black disc peeking out from
        # under the ring); with the ring off by default that would have left a
        # dark crescent, so the shadow is now its own thing -- feathered, and
        # drawn whether or not there is a ring.
        self._shadow = None
        if self.shadow:
            pad = max(2, D // 8)
            self._shadow_pad = pad
            off = max(1, D // 40)
            box = D + 2 * pad
            m = np.zeros((box, box), np.float32)
            m[pad + off:pad + off + D, pad + off:pad + off + D] = \
                self.alpha[:, :, 0]
            sigma = max(1.0, D * 0.05)
            self._shadow = (cv2.GaussianBlur(m, (0, 0), sigma) * 0.38)[:, :, None]
        self._ok = self.cap.isOpened()

    def available(self):
        return self._ok

    def _frame_at(self, t):
        fm = t + self.offset
        idx = 0 if fm < 0 else int(round(fm * self.face_fps))
        idx = max(0, idx)
        if self.n is not None:
            idx = min(idx, self.n - 1)
        if idx == self._cur_idx and self._last is not None:
            return self._last
        if idx < self._cur_idx:                 # rare backward step (retime)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self._cur_idx = idx - 1
        while self._cur_idx < idx:
            ok, fr = self.cap.read()
            if not ok:
                break
            self._cur_idx += 1
            self._last = fr
        return self._last

    def draw(self, img, t):
        if not self._ok:
            return
        frame = self._frame_at(t)
        if frame is None:
            return
        D, px, py = self.D, self.px, self.py
        h, w = img.shape[:2]
        if px < 0 or py < 0 or px + D > w or py + D > h:
            return  # bubble wouldn't fit the frame
        # center-crop the face frame to a square, then fit the bubble
        fh, fw = frame.shape[:2]
        s = min(fh, fw)
        cy0 = (fh - s) // 2
        cx0 = (fw - s) // 2
        sq = frame[cy0:cy0 + s, cx0:cx0 + s]
        if self.mirror:
            sq = sq[:, ::-1]
        bub = cv2.resize(sq, (D, D), interpolation=cv2.INTER_AREA)

        # Background blur: blend a blurred copy in over the rim, leaving the
        # sharp centre untouched. `bub` becomes float32 here; the alpha
        # composite below already casts to float32, so nothing downstream cares.
        if self._vignette is not None:
            blurred = cv2.GaussianBlur(bub, (0, 0), self._blur_sigma)
            bub = bub * (1.0 - self._vignette) + blurred * self._vignette

        if self._shadow is not None:
            pad = self._shadow_pad
            sy0, sx0 = py - pad, px - pad
            sy1, sx1 = sy0 + self._shadow.shape[0], sx0 + self._shadow.shape[1]
            # Clip against the frame -- the bubble sits one margin in, but a
            # large size_frac can still push the feathered edge off it.
            cy0, cx0 = max(0, sy0), max(0, sx0)
            cy1, cx1 = min(h, sy1), min(w, sx1)
            if cy1 > cy0 and cx1 > cx0:
                sh = self._shadow[cy0 - sy0:cy1 - sy0, cx0 - sx0:cx1 - sx0]
                roi_s = img[cy0:cy1, cx0:cx1]
                roi_s[:] = (roi_s.astype(np.float32) * (1.0 - sh)).astype(np.uint8)

        roi = img[py:py + D, px:px + D]
        a = self.alpha
        roi[:] = (bub.astype(np.float32) * a
                  + roi.astype(np.float32) * (1.0 - a)).astype(np.uint8)

        if self.border_th > 0:
            cv2.circle(img, (px + D // 2, py + D // 2),
                       D // 2 - self.border_th // 2,
                       tuple(int(c) for c in self.p["border_color"]),
                       self.border_th, cv2.LINE_AA)

    def release(self):
        try:
            self.cap.release()
        except Exception:
            pass


# ---- shared helper ----------------------------------------------------------
def load_wallpaper(path):
    """Read an image file for use as a framed background, or None if unreadable."""
    if not path or not os.path.isfile(path):
        return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img  # None if cv2 couldn't decode it
