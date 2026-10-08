"""bar_native.py — Cocoa-side behaviors for the frameless recording pill.

Eight things the always-on-top pywebview window needs that plain HTML
can't do:

1. **Fit-to-content sizing.** The pill's width changes with its face (idle is
   ~702pt wide, "recording" only ~265pt). A fixed 880x136 window leaves up to
   84% of an always-on-top window as invisible, click-eating dead space.
   `fit_frame` recomputes the window frame from the size the page reports.
2. **Click-through for the transparent margin.** Even a snug window keeps a
   little slack so the pill's drop shadow isn't clipped. `ClickThrough` watches
   the mouse and flips `NSWindow.ignoresMouseEvents`, so only the pill itself
   eats clicks and every transparent pixel falls through to the app behind.
3. **Position persistence** across runs, for the pill and for the detached
   facecam bubble (`bar-pos.json`).
4. **An attempt at staying out of the recording.** The pill and the bubble
   float over the very display ffmpeg captures, so they end up burned into
   every take. `exclude_from_capture` sets `NSWindowSharingNone` to ask macOS
   to omit them. Untested against real pixels — see that function.
5. **Following the user onto every Space.** A full-screen app owns its own
   Space, and a plain always-on-top NSWindow is pinned to the Space it was
   born on — so the pill used to vanish over exactly the apps people most want
   to record. Takes BOTH halves: `use_panel_windows` (pywebview must build a
   non-activating `NSPanel`, which cannot be retrofitted) and
   `join_all_spaces` (the collection-behavior bits).
6. **Docking onto a picked window.** `union_rect`/`dock_anchor`/`dock_frame`
   park the pill bottom-centre on whatever the window picker just elected.
7. **Reaching a second display.** `origin_screen` pins pywebview's coordinate
   anchor to the primary display and `clamp_to_screens` keeps the pill on
   whichever display it was dragged onto, instead of on one hard-coded one.
8. **Real frosted glass.** CSS `backdrop-filter` only blurs PAGE content — a
   transparent native window has nothing behind it in the page, so the desktop
   can't be blurred from CSS at all. `Glass` puts an `NSVisualEffectView`
   (behind-window blending) under the WKWebView's content for each rect
   `content_layout` returns, masked to the CSS capsule.

Coordinate systems, because two of them meet here:
  * **layout coords** — origin top-left of the PRIMARY display (never
    `mainScreen()`, which follows keyboard focus — see
    `_origin_screen_frame`), y down. What CSS, `webview.create_window(x=, y=)`
    and `Window.move()` use. Persisted.
  * **Cocoa coords** — origin bottom-left, y up. What `NSWindow.frame()` and
    `NSEvent.mouseLocation()` use. Used for the live frame math so no screen
    height is needed and multi-monitor setups stay correct.

Everything Cocoa-specific is behind a lazy import and fails soft: without
pywebview/pyobjc every helper no-ops and the caller keeps the old behavior.
The geometry is pure and unit-tested (see tests/test_bar_native.py).
"""

import json
import os
import threading

# Transparent slack kept on every side of the pill so its drop shadow isn't
# clipped by the window edge. Cosmetic only: ClickThrough makes it
# click-through, and fit_frame is what keeps it from being 800px wide. Sized
# to just contain the pill's shadow (0 8px 20px -> 20 on three sides, 28
# below) so that even with click-through off the dead ring stays small.
# MUST match --bar-pad in bar.css.
SHADOW_PAD = 28

_POS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "bar-pos.json"))

# The notes overlay's content lives in its OWN file, not bar-pos.json: that
# file is small position state we rewrite on every drag, and folding a
# multi-KB script into it would mean re-serializing the whole blob on each
# nudge. Plain UTF-8, not JSON — it is one free-text field, and a human should
# be able to open it. Same soft-fail rules as the positions file (see below).
_NOTES_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "notepad.txt"))


# --------------------------------------------------------------------------
# position persistence
# --------------------------------------------------------------------------

def _read_positions(path=None):
    try:
        with open(path or _POS_PATH) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_positions(data, path=None):
    try:
        with open(path or _POS_PATH, "w") as f:
            json.dump(data, f)
        return True
    except Exception:
        return False


def _load_xy(kx, ky, path=None):
    data = _read_positions(path)
    x, y = data.get(kx), data.get(ky)
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return (int(x), int(y))
    return None


def _save_xy(kx, ky, x, y, path=None):
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        return False
    data = _read_positions(path)
    data[kx] = int(x)
    data[ky] = int(y)
    return _write_positions(data, path)


def load_bar_position(path=None):
    """Saved pill position as layout-coord (x, y), or None. Never raises."""
    return _load_xy("x", "y", path)


def save_bar_position(x, y, path=None):
    return _save_xy("x", "y", x, y, path)


def load_face_position(path=None):
    """Saved floating-facecam position as layout-coord (x, y), or None."""
    return _load_xy("face_x", "face_y", path)


def save_face_position(x, y, path=None):
    return _save_xy("face_x", "face_y", x, y, path)


def load_notepad_geometry(path=None):
    """Saved notes-overlay geometry as layout-coord (x, y, w, h), or None.

    Unlike the pill and the bubble, the notes window is user-resizable, so its
    SIZE is saved alongside its position — otherwise every reopen would forget
    a widened panel. All four values must be present and numeric or the whole
    thing reads as None (fall back to the default box).
    """
    data = _read_positions(path)
    vals = [data.get(k) for k in ("notepad_x", "notepad_y",
                                  "notepad_w", "notepad_h")]
    if all(isinstance(v, (int, float)) for v in vals):
        x, y, w, h = (int(v) for v in vals)
        if w > 0 and h > 0:
            return (x, y, w, h)
    return None


def save_notepad_geometry(x, y, w, h, path=None):
    """Persist the notes window's (x, y, w, h). Rejects non-numeric or
    non-positive sizes rather than writing a box that can't be reopened."""
    vals = (x, y, w, h)
    if not all(isinstance(v, (int, float)) for v in vals):
        return False
    if w <= 0 or h <= 0:
        return False
    data = _read_positions(path)
    data["notepad_x"], data["notepad_y"] = int(x), int(y)
    data["notepad_w"], data["notepad_h"] = int(w), int(h)
    return _write_positions(data, path)


def load_notepad_text(path=None):
    """The saved notes text, or "" when there is none. Never raises — a
    corrupt or unreadable file just reads as empty, like every other bit of
    runtime state the bar writes beside itself."""
    try:
        with open(path or _NOTES_PATH, encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


def save_notepad_text(text, path=None):
    """Persist the notes text. Coerces None/non-strings to "" so a stray
    bridge value can't raise; swallows write failures (a preferences write
    must never be able to stop someone recording)."""
    try:
        with open(path or _NOTES_PATH, "w", encoding="utf-8") as f:
            f.write("" if text is None else str(text))
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# pure geometry
# --------------------------------------------------------------------------

def fit_frame(frame, new_w, new_h):
    """Resize a window to (new_w, new_h) without the pill appearing to move.

    `frame` is Cocoa-coord (x, y, w, h) — y is the BOTTOM edge. The pill is
    centered horizontally in the window and sits a fixed SHADOW_PAD below its
    top edge, so pinning the window's center-x and top edge pins the pill.
    Returns a new Cocoa-coord (x, y, w, h) tuple.
    """
    x, y, w, h = frame
    new_w = max(1, int(round(new_w)))
    new_h = max(1, int(round(new_h)))
    return (
        int(round(x + (w - new_w) / 2.0)),   # keep center-x
        int(round(y + h - new_h)),           # keep the top edge (y + h)
        new_w,
        new_h,
    )


def clamp_frame(frame, screen, margin=0):
    """Nudge a Cocoa-coord frame back inside `screen` (also Cocoa (x,y,w,h)).

    A window larger than the screen on an axis is pinned to that axis' low
    edge rather than being pushed off the opposite one.
    """
    x, y, w, h = frame
    sx, sy, sw, sh = screen
    lo_x, hi_x = sx + margin, sx + sw - margin - w
    lo_y, hi_y = sy + margin, sy + sh - margin - h
    x = lo_x if hi_x < lo_x else min(max(x, lo_x), hi_x)
    y = lo_y if hi_y < lo_y else min(max(y, lo_y), hi_y)
    return (int(round(x)), int(round(y)), int(w), int(h))


def _overlap_area(frame, screen):
    """Area of the intersection of two Cocoa-coord (x, y, w, h) boxes."""
    fx, fy, fw, fh = frame
    sx, sy, sw, sh = screen
    ox = min(fx + fw, sx + sw) - max(fx, sx)
    oy = min(fy + fh, sy + sh) - max(fy, sy)
    return 0.0 if ox <= 0 or oy <= 0 else float(ox) * float(oy)


def _centre_dist2(frame, screen):
    """Squared distance between two boxes' centres. Tie-break only, so the
    units never matter -- no sqrt."""
    fx, fy, fw, fh = frame
    sx, sy, sw, sh = screen
    dx = (fx + fw / 2.0) - (sx + sw / 2.0)
    dy = (fy + fh / 2.0) - (sy + sh / 2.0)
    return dx * dx + dy * dy


def clamp_to_screens(frame, screens, margin=0):
    """Clamp a Cocoa-coord frame onto the display it actually sits on.

    `clamp_frame` against ONE screen is what used to pin the pill to a single
    display: a frame the user dragged onto a second monitor is outside that
    screen on both axes, so the very next fit yanked it straight back. Pick
    the screen the frame overlaps most -- nearest centre when it overlaps none
    (mid-drag over the dead gap in a staggered arrangement) -- and clamp
    against that one.

    An empty screen list returns the frame untouched: leaving a window where
    the user put it beats moving it somewhere we cannot justify.
    """
    usable = [s for s in (screens or ()) if s and s[2] > 0 and s[3] > 0]
    if not usable:
        return frame
    best, best_area = None, -1.0
    for s in usable:
        area = _overlap_area(frame, s)
        if area > best_area:
            best, best_area = s, area
    if best_area <= 0:
        best = min(usable, key=lambda s: _centre_dist2(frame, s))
    return clamp_frame(frame, best, margin=margin)


def hit_rects_to_screen(frame, rects, grow=0):
    """Layout-coord rects relative to the window's top-left -> Cocoa coords.

    `rects` is an iterable of (x, y, w, h) as measured by JS inside the page.
    Malformed entries are skipped rather than raising — this data crosses the
    JS bridge. `grow` inflates each rect on every side (see ClickThrough's
    HYSTERESIS_PT).
    """
    fx, fy, fw, fh = frame
    out = []
    for r in rects or ():
        try:
            rx, ry, rw, rh = (float(v) for v in r[:4])
        except Exception:
            continue
        if rw <= 0 or rh <= 0:
            continue
        out.append((fx + rx - grow, fy + fh - ry - rh - grow,
                    rw + 2 * grow, rh + 2 * grow))
    return out


def point_in_rects(px, py, rects):
    """True when (px, py) falls inside any (x, y, w, h) rect."""
    for rx, ry, rw, rh in rects or ():
        if rx <= px <= rx + rw and ry <= py <= ry + rh:
            return True
    return False


# Row gap between the pill and the hint strip below it — matches
# .bar-body { gap: 8px } in bar.css.
BODY_GAP = 8


def content_layout(pill_w, pill_h, hint_w=0, hint_h=0, pad=SHADOW_PAD,
                   gap=BODY_GAP):
    """Window box that fits the pill (+ hint strip below it) plus shadow slack.

    Returns `(w, h, rects)` — the window size and the click-catching regions
    inside it, in layout coords relative to the window's top-left. The page
    reports raw element sizes and this owns the arithmetic, so the window box
    and the click-through mask can never disagree.
    """
    pill_w, pill_h = max(0.0, float(pill_w)), max(0.0, float(pill_h))
    hint_w, hint_h = max(0.0, float(hint_w)), max(0.0, float(hint_h))
    w = max(pill_w, hint_w) + 2 * pad
    h = pill_h + 2 * pad + ((hint_h + gap) if hint_h > 0 else 0.0)
    rects = [((w - pill_w) / 2.0, float(pad), pill_w, pill_h)]
    if hint_h > 0:
        rects.append(((w - hint_w) / 2.0, pad + pill_h + gap, hint_w, hint_h))
    return (int(round(w)), int(round(h)), rects)


# How far the pill's VISIBLE bottom edge sits above the bottom of the window
# it is docked onto. Measured to the pill and not to its window box on
# purpose: `content_layout` always leaves `SHADOW_PAD` of transparent slack
# below the content, so placing the frame naively would read as ~60pt of gap.
# Keep it > SHADOW_PAD — that is what keeps the window BOX inside the target
# too, so docking onto a full-screen window doesn't immediately need
# `clamp_frame` to pull it back off the bottom edge.
DOCK_MARGIN = 32


def union_rect(rects):
    """Smallest (x, y, w, h) containing every rect, or None when there is
    nothing usable. Layout coords in, layout coords out.

    A multi-window pick docks to the union rather than to whichever window
    happened to be clicked first: with 2-4 cards on screen there is no single
    "the window", and the union is the region the take is about. For the
    ordinary one-window pick the union IS that window, so both the picker's
    modes go through one path.

    Malformed and degenerate entries are skipped rather than raising — this
    data comes back from a Quartz read that can return anything.
    """
    box = None
    for r in rects or ():
        try:
            x, y, w, h = (float(v) for v in r[:4])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        if box is None:
            box = [x, y, x + w, y + h]
        else:
            box = [min(box[0], x), min(box[1], y),
                   max(box[2], x + w), max(box[3], y + h)]
    if box is None:
        return None
    return (box[0], box[1], box[2] - box[0], box[3] - box[1])


def dock_anchor(target, margin=DOCK_MARGIN):
    """Layout-coord `(centre_x, bottom_y)` to park the pill's visible bottom
    on, given the window rect it should sit in.

    Bottom-centre, because the top of a window is its own chrome — the part a
    viewer actually reads — and because that is where every screen recorder
    puts its controls. Pure: no screen read, so it tests offline.
    """
    x, y, w, h = (float(v) for v in target[:4])
    return (x + w / 2.0, y + h - float(margin))


def dock_frame(anchor, size, pad=SHADOW_PAD, screen=None):
    """Cocoa-coord frame that lands a `size` window on a layout-coord `anchor`.

    The anchor names the pill's VISIBLE bottom edge, so `pad` — the transparent
    shadow slack `content_layout` leaves below the content — is added back
    before the flip into Cocoa coords. `size` is passed in rather than read off
    the window because the caller usually knows a size the window has not
    reached yet (`bar_fit` docks the box it is about to apply, not the one on
    screen). Returns None when the screen can't be read.
    """
    cx, by = (float(v) for v in anchor[:2])
    w, h = (float(v) for v in size[:2])
    return to_cocoa_frame(cx - w / 2.0, by - h + float(pad), w, h, screen=screen)


# The glass sits this far INSIDE each rect the page reports. The CSS pill
# draws a 1px hairline border over its own edge, and bar.js reports
# Math.ceil'd sizes while the HTML pill is centred on its true (fractional)
# width, so the real edge can sit up to 0.5pt inside the reported rect.
# Insetting by the border width lands every glass pixel UNDER that border
# instead of leaving a sliver of blur haloing outside it.
GLASS_INSET = 1

# NSAutoresizingMaskOptions bits, hardcoded like every other AppKit value in
# this module (they have to exist even when AppKit doesn't import).
NSVIEW_MIN_X_MARGIN = 1 << 0
NSVIEW_MAX_X_MARGIN = 1 << 2
NSVIEW_MIN_Y_MARGIN = 1 << 3
NSVIEW_MAX_Y_MARGIN = 1 << 5


def capsule_radius(w, h):
    """The corner radius CSS actually draws for `border-radius: 999px` on a
    w x h box. The spec scales overlapping radii down until they fit, which
    for one radius on every corner is half the SHORTER side. The pill and
    `.bar-hint` both use 999px, so both are capsules -- a hint that wraps to
    two lines is a taller capsule, not a rounded rectangle."""
    return max(0.0, min(float(w), float(h)) / 2.0)


def glass_frames(rects, host_h, flipped, inset=GLASS_INSET):
    """Page rects -> the frame and mask radius of the glass behind each one.

    `rects` are layout coords relative to the window's top-left -- exactly
    what `content_layout` returns, so the glass, the window box and the click
    mask all come out of one piece of arithmetic and cannot disagree.

    Returns `(x, y, w, h, radius)` per usable rect, in the HOST view's
    coordinates. A flipped host takes layout y as is; an unflipped one needs
    y flipped against its own height, which is the only time `host_h` is
    read. The host is the WKWebView, and that one IS flipped (measured:
    `isFlipped()` is True on pywebview's WebKitHost), but which way it goes
    is read off the live view rather than assumed.

    Malformed entries, and rects the inset swallows whole, are skipped rather
    than raising -- this data crosses the JS bridge.
    """
    out = []
    for r in rects or ():
        try:
            rx, ry, rw, rh = (float(v) for v in r[:4])
        except Exception:
            continue
        rx, ry = rx + inset, ry + inset
        rw, rh = rw - 2 * inset, rh - 2 * inset
        if rw <= 0 or rh <= 0:
            continue
        y = ry if flipped else float(host_h) - ry - rh
        out.append((rx, y, rw, rh, capsule_radius(rw, rh)))
    return out


def glass_autoresizing_mask(flipped):
    """Autoresizing that holds a glass view still on screen while the window
    resizes around it: both x margins flexible (the pill is centred, and two
    equal flexible margins split a width change evenly) and the BOTTOM margin
    flexible (the pill hangs a fixed SHADOW_PAD below the window's top).

    It is `fit_frame`'s anti-jump rule one level down, and it is what covers
    the gap between `set_frame` landing and the explicit re-frame queued
    behind it: `setFrame:display:YES` draws at once, so without it a height
    change (the hint strip coming or going) would paint a frame of glass
    displaced by the height delta. "Bottom" is min-y in an unflipped view and
    max-y in a flipped one.
    """
    bottom = NSVIEW_MAX_Y_MARGIN if flipped else NSVIEW_MIN_Y_MARGIN
    return NSVIEW_MIN_X_MARGIN | NSVIEW_MAX_X_MARGIN | bottom


# --------------------------------------------------------------------------
# Cocoa glue (all soft-failing)
# --------------------------------------------------------------------------

def _ns_window(window):
    """The NSWindow behind a pywebview window, or None."""
    # Importing pywebview's Cocoa backend can initialize native AppKit state.
    # A missing/stand-in window cannot resolve to an NSWindow, so reject it
    # before that import.  This keeps the advertised soft-failure path safe in
    # permission-free tests and during partial native-window startup.
    uid = getattr(window, "uid", None)
    if uid is None:
        return None
    try:
        from webview.platforms import cocoa
        inst = cocoa.BrowserView.instances.get(uid)
        return None if inst is None else inst.window
    except Exception:
        return None


def screen_frames():
    """Cocoa-coord (x, y, w, h) for every attached display, primary first.

    `[]` when AppKit isn't importable, which every caller reads as "don't
    move the window" rather than as an error.
    """
    try:
        import AppKit
        out = []
        for s in AppKit.NSScreen.screens():
            f = s.frame()
            out.append((f.origin.x, f.origin.y, f.size.width, f.size.height))
        return out
    except Exception:
        return []


def _origin_screen_frame():
    """Cocoa-coord (x, y, w, h) of the screen LAYOUT coords are measured from.

    That is `NSScreen.screens()[0]`, the PRIMARY display (the one holding the
    menu bar, and the one `Quartz.CGMainDisplayID` reports, so this agrees
    with `devices.displays_points`). Cocoa's global space puts its bottom-left
    at (0, 0), and JS `ev.screenX/screenY`, `webview.create_window(x=, y=)`
    and `devices.list_windows` all measure from its top-left.

    It is deliberately NOT `NSScreen.mainScreen()`. That is "the screen
    holding the window with keyboard focus", so on a two-display setup it
    follows whatever the user last clicked -- and this function feeds every
    layout<->Cocoa conversion here plus the on-screen clamp. Reading focus
    instead of the primary display silently re-based the pill's coordinate
    space each time focus moved between monitors, which is what made the pill
    impossible to drag off whichever display had focus.
    """
    frames = screen_frames()
    return frames[0] if frames else None


def origin_screen():
    """The pywebview `Screen` for `_origin_screen_frame`'s display, or None.

    Passed as `create_window(screen=...)` so the cocoa backend anchors its
    `move()` and `center()` math to the PRIMARY display. Left alone, that
    backend caches `NSScreen.mainScreen()` **at window-creation time** and
    then treats every later `move(x, y)` as an offset from that screen -- but
    the drag region reports `ev.screenX/screenY`, which are global. Open the
    pill while a second display holds focus and every drag is then off by
    that screen's origin, so the pill cannot be dragged onto another display
    at all. Pinning the anchor to the primary makes the two agree.

    None when pywebview can't be asked; the caller then keeps pywebview's own
    (focus-dependent) choice.
    """
    try:
        import webview
        # `webview.screens` is a `@module_property` (proxy_tools) — it is
        # ACCESSED, never called. `webview.screens()` raises "'list' object is
        # not callable", which this used to swallow to None, quietly reverting
        # to pywebview's mainScreen() anchor and un-fixing the drag.
        screens = list(webview.screens)
    except Exception:
        return None
    return screens[0] if screens else None


def frame_of(window):
    """Cocoa-coord (x, y, w, h) of a pywebview window, or None."""
    nswin = _ns_window(window)
    if nswin is None:
        return None
    try:
        f = nswin.frame()
        return (f.origin.x, f.origin.y, f.size.width, f.size.height)
    except Exception:
        return None


def window_number(window):
    """The CoreGraphics window id of a pywebview window, or None.

    This is the handle ScreenCaptureKit's `SCContentFilter` excludes by, and
    it is the RIGHT handle: `NSWindowSharingNone` (see `exclude_from_capture`)
    was measured 2026-07-30 to be ignored by ffmpeg's avfoundation screen
    input, while SCK exclusion by window id was measured to work — the window
    stays visible on the user's own display and is absent from the capture.

    Measured stable across `setFrame:display:`, so `fit_frame` resizing the
    pill mid-take does not invalidate an id already handed out. It is NOT
    stable across window destruction/recreation (the facecam bubble and the
    picker are created and destroyed on demand), which is why the ids are
    re-reported rather than captured once.
    """
    nswin = _ns_window(window)
    if nswin is None:
        return None
    try:
        return int(nswin.windowNumber())
    except Exception:
        return None


def main_screen_layout():
    """The whole main screen as LAYOUT coords `(x, y, w, h)`, or None.

    "Main" here means the PRIMARY display -- `_origin_screen_frame`, i.e. the
    one `devices.displays_points` marks `main` and the only one
    `devices.list_windows(main_only=True)` reports windows from. Sizing the
    overlay off `NSScreen.mainScreen()` instead was wrong twice over: the
    picker would take a focused SECOND display's dimensions while still being
    placed at layout (0, 0), so it covered the wrong box on the wrong screen.
    A genuinely multi-display picker is separate work (`devices` filters to
    the main display first).

    Layout coords are what `webview.create_window` takes, and their origin is
    the primary screen's top-left -- which is also the origin of the global
    top-left POINT space `devices.list_windows` reports window rects in. So a
    window placed at this frame can treat a window rect as a CSS position
    with no conversion at all, which is exactly what the picker overlay does.

    Includes the menu-bar strip (`frame`, not `visibleFrame`): the picker
    dims the whole display, and a lit strip across the top would read as a
    bug.
    """
    screen = _origin_screen_frame()
    if screen is None:
        return None
    _sx, _sy, sw, sh = screen
    return (0, 0, int(round(sw)), int(round(sh)))


def to_layout_xy(frame):
    """Cocoa-coord frame -> layout-coord (x, y) top-left, as pywebview wants."""
    screen = _origin_screen_frame()
    if screen is None or frame is None:
        return None
    sx, sy, _sw, sh = screen
    x, y, _w, h = frame
    return (int(round(x - sx)), int(round(sy + sh - (y + h))))


def to_cocoa_frame(x, y, w, h, screen=None):
    """Layout-coord top-left box -> Cocoa-coord frame. Inverse of
    `to_layout_xy`, and the only conversion the docking path needs: window
    rects arrive from `devices.list_windows` already in layout coords.

    `screen` is injectable so the geometry tests offline; None reads the main
    screen and returns None when there isn't one.
    """
    if screen is None:
        screen = _origin_screen_frame()
    if screen is None:
        return None
    sx, sy, _sw, sh = screen
    return (int(round(sx + x)), int(round(sy + sh - (y + h))),
            int(round(w)), int(round(h)))


def set_frame(window, frame):
    """Atomically move+resize (one setFrame: — no two-step jump). Soft-fails."""
    nswin = _ns_window(window)
    if nswin is None:
        return False
    try:
        import AppKit
        from PyObjCTools import AppHelper
        x, y, w, h = frame
        rect = AppKit.NSMakeRect(float(x), float(y), float(w), float(h))

        def _apply():
            try:
                nswin.setFrame_display_(rect, True)
            except Exception:
                pass

        AppHelper.callAfter(_apply)
        return True
    except Exception:
        return False


# NSWindowSharingNone — the AppKit knob for asking macOS to leave a window out
# of screen captures. Hardcoded rather than read off AppKit so the value is
# there even when the import isn't.
NSWINDOW_SHARING_NONE = 0


def _apply_sharing_none(nswin):
    """Set sharingType on a raw NSWindow. Split out so a test can drive it with
    a stand-in window and no Cocoa runloop.

    True only means the selector accepted the call — it says nothing about
    whether a recording actually drops the window.
    """
    try:
        nswin.setSharingType_(NSWINDOW_SHARING_NONE)
        return True
    except Exception:
        return False


def exclude_from_capture(window):
    """Ask macOS to keep a pywebview window out of screen recordings.

    Both of our always-on-top windows sit over the display ffmpeg records, so
    today they are part of every take. This is the documented way to ask for
    the opposite; whether the avfoundation screen input we capture through
    honors sharingType is UNVERIFIED — nobody has checked the resulting pixels
    — so treat it as an attempt, not a fix.

    Soft-fails like the rest of this section: no pywebview/pyobjc, or an
    NSWindow that won't take the call, and the caller keeps today's behavior.

    `AUTOCINE_NO_CAPTURE_EXCLUDE=1` skips it entirely. That is the ONLY way
    to get the pill, the bubble or the notes overlay into a screenshot --
    `screencapture` and every CGWindowList-based grabber honor sharingType,
    so without it a product screenshot (a website, a README, a bug report)
    literally cannot show the bar. Off by default because a take is worth
    more than a screenshot: with it set, the chrome lands in the recording.
    """
    if os.environ.get("AUTOCINE_NO_CAPTURE_EXCLUDE") == "1":
        return False
    nswin = _ns_window(window)
    if nswin is None:
        return False
    try:
        from PyObjCTools import AppHelper

        def _apply():
            _apply_sharing_none(nswin)

        AppHelper.callAfter(_apply)
        return True
    except Exception:
        return False


# NSWindowCollectionBehavior bits, hardcoded for the same reason as
# NSWINDOW_SHARING_NONE: the values have to exist even when AppKit doesn't.
NSWINDOW_BEHAVIOR_CAN_JOIN_ALL_SPACES = 1 << 0
NSWINDOW_BEHAVIOR_FULLSCREEN_AUXILIARY = 1 << 8
ALL_SPACES_BEHAVIOR = (NSWINDOW_BEHAVIOR_CAN_JOIN_ALL_SPACES
                       | NSWINDOW_BEHAVIOR_FULLSCREEN_AUXILIARY)


# NSWindowStyleMaskNonactivatingPanel. Only an NSPanel may carry it -- AppKit
# logs "NSWindow does not support nonactivating panel styleMask 0x80" and
# strips it, which is exactly how this was measured.
NSWINDOW_STYLE_MASK_NONACTIVATING_PANEL = 1 << 7

_panel_host = None


def _make_panel_host():
    """Define (once) an NSPanel subclass shaped like pywebview's WindowHost.

    Cached in a module global because a pyobjc class name may only be
    registered with the Objective-C runtime once; defining it per call raises
    on the second window.
    """
    global _panel_host
    if _panel_host is not None:
        return _panel_host
    import objc
    import AppKit

    class _AutocinePanelHost(AppKit.NSPanel):
        def initWithContentRect_styleMask_backing_defer_(self, rect, mask,
                                                         backing, defer):
            this = objc.super(
                _AutocinePanelHost, self
            ).initWithContentRect_styleMask_backing_defer_(
                rect, mask | NSWINDOW_STYLE_MASK_NONACTIVATING_PANEL,
                backing, defer)
            if this is not None:
                # NSPanel defaults hidesOnDeactivate to True -- for a floating
                # recorder pill that means vanishing the moment the user
                # clicks the app they are demoing, which is worse than the bug
                # this whole class swap exists to fix.
                this.setHidesOnDeactivate_(False)
            return this

        def canBecomeKeyWindow(self):
            # Mirrors pywebview's WindowHost. `focus` is assigned by pywebview
            # AFTER init, so this must survive being asked before then --
            # defaulting to True keeps the notes overlay typable.
            return bool(getattr(self, "focus", True))

    _panel_host = _AutocinePanelHost
    return _panel_host


def use_panel_windows():
    """Make pywebview build NSPanels instead of NSWindows, process-wide.

    **This is the only thing that lets our chrome follow the user onto a
    full-screen app's Space**, and it has to happen at CREATION: measured
    2026-08-13, a window must be BOTH an `NSPanel` AND carry
    `NSWindowStyleMaskNonactivatingPanel`, neither bit is sufficient alone,
    and AppKit flatly refuses the mask on a live NSWindow (`setStyleMask_`
    logs and strips it). See `join_all_spaces` for the full matrix.

    pywebview instantiates `BrowserView.WindowHost`, a plain NSWindow
    subclass, by class-attribute lookup at window-creation time -- so swapping
    that attribute is the whole patch, and it is a far smaller reach into
    pywebview than re-hosting its WKWebView in a window we own.

    Process-wide is the correct scope here and not laziness: every pywebview
    window this codebase creates is bar chrome (pill, facecam bubble, picker,
    notes overlay). The projects library and the editor are a Chrome app-mode
    window or a browser tab (`open_app_window`), so no document window can be
    caught by this.

    Must run BEFORE the first window is created -- for window one that means
    before `webview.start()`, since pywebview defers its NSWindow until the
    GUI loop is up. Returns True when the swap is in place. Soft-fails: no
    pywebview/pyobjc and the caller keeps plain NSWindows, i.e. today's
    behavior minus the Spaces fix.
    """
    if os.environ.get("AUTOCINE_NO_PANEL_WINDOWS") == "1":
        return False
    try:
        from webview.platforms import cocoa
        host = _make_panel_host()
        if host is None:
            return False
        cocoa.BrowserView.WindowHost = host
        return True
    except Exception:
        return False


def _apply_all_spaces(nswin):
    """Set collectionBehavior on a raw NSWindow. Split out for the same reason
    as `_apply_sharing_none`: a test can drive it with a stand-in window and no
    Cocoa runloop.

    REPLACES the behavior rather than OR-ing into whatever is there. Both bits
    are mutually exclusive with `NSWindowCollectionBehaviorManaged`, so reading
    the current value back to merge would only risk carrying that bit in.
    """
    try:
        nswin.setCollectionBehavior_(ALL_SPACES_BEHAVIOR)
        return True
    except Exception:
        return False


def join_all_spaces(window):
    """Let a pywebview window follow the user onto every Space, including the
    one a full-screen app owns.

    Without this the bar is unusable over exactly the apps people most want to
    record. A full-screen app gets its OWN Space, and a window created with the
    default collection behavior belongs to the Space it was born on: switch to
    the full-screen app and the pill, the facecam bubble and the notes overlay
    all disappear, leaving no way to stop a take but Ctrl+C. pywebview only
    ever calls `setCollectionBehavior_` inside `toggle_fullscreen`, which we
    never use, so the default is what we inherit and this is the one place it
    gets fixed. `on_top=True` already puts these windows at
    `NSStatusWindowLevel`, which is above a full-screen window once the
    behavior lets us onto its Space at all.

    `CanJoinAllSpaces` puts the window on every Space and `FullScreenAuxiliary`
    permits it *alongside* a full-screen window rather than deferring it until
    the app leaves full screen — but **these bits are necessary and not
    sufficient. `use_panel_windows` is the other half**, and neither works
    alone. Measured 2026-08-13, four windows walked across two Spaces at once:

        NSWindow + behavior 257, level 3   -> pinned
        NSWindow + behavior 257, level 25  -> pinned   (the pill before this)
        NSPanel  + behavior 257, level 3   -> FOLLOWS
        NSPanel  + behavior 257, level 25  -> FOLLOWS

    So the window LEVEL is irrelevant (we keep pywebview's 25), and the class
    is everything. A second pass split class from style mask: a bare `NSPanel`
    without `NSWindowStyleMaskNonactivatingPanel` is pinned too, and an
    `NSWindow` given that mask — at creation OR after — has it stripped, with
    AppKit logging "NSWindow does not support nonactivating panel styleMask
    0x80". Both, or nothing. Accessory/LSUIElement activation policy was also
    tested and is NOT required, so the bar keeps its Dock icon.

    Do NOT measure this with a screenshot. `exclude_from_capture` applies
    `NSWindowSharingNone` to the same windows, so `screencapture` cannot see
    them at all and every frame reads as "the pill isn't there" — a clean
    false negative that cost a debugging cycle here. Use
    `CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly)`, which lists
    only the ACTIVE Space, against the window id.

    **This makes our chrome visible to the capture where it previously
    wasn't.** On a full-screen Space the pill used to be absent because it was
    stuck on another Space, so an avfoundation take of a full-screen app came
    out clean by accident; it won't any more. Only the SCK backend actually
    excludes these windows (docs/architecture.md, "Capture exclusion, and what is
    actually wired"). `AUTOCINE_NO_ALL_SPACES=1` restores the old behavior for
    anyone recording a full-screen app on avfoundation.

    Soft-fails like the rest of this section.
    """
    if os.environ.get("AUTOCINE_NO_ALL_SPACES") == "1":
        return False
    nswin = _ns_window(window)
    if nswin is None:
        return False
    try:
        from PyObjCTools import AppHelper

        def _apply():
            _apply_all_spaces(nswin)

        AppHelper.callAfter(_apply)
        return True
    except Exception:
        return False


# NSVisualEffectView values and NSWindowBelow, hardcoded for the usual reason.
NSVISUAL_EFFECT_MATERIAL_HUD_WINDOW = 13
NSVISUAL_EFFECT_BLENDING_BEHIND_WINDOW = 0
NSVISUAL_EFFECT_STATE_ACTIVE = 1
NSWINDOW_BELOW = -1

# What the page is told if the glass is torn down after it went translucent:
# bar.css's `html.glass` rules are the only thing that makes the pill see-
# through, so dropping the class restores the opaque pill. The flag makes the
# loss final in the page too: a bar_fit reply computed before the teardown
# (it can land after this) must not put the class back (bar.js applyGlass).
GLASS_LOST_JS = ("window.autocineGlassLost=true;"
                 "document.documentElement.classList.remove('glass')")

# What the page is told once the glass is actually framed and visible. The
# ordinary path does not need it (bar_fit's reply already says "glass"), but
# a LAZY install -- `shown` never installed -- is only queued by the fit that
# triggers it, so that fit's reply goes out before the views exist and bar.js,
# having latched the size, would never ask again. Idempotent, and a no-op
# once the glass was lost.
GLASS_ON_JS = ("window.autocineGlassLost||"
               "document.documentElement.classList.add('glass')")


def _web_view(window):
    """The WKWebView behind a pywebview window, or None.

    Read off the BrowserView instance rather than `contentView()`: pywebview
    only makes the web view the content view in `webView:didFinishNavigation:`,
    which is after `shown` fires, while `inst.webview` exists from creation.
    """
    uid = getattr(window, "uid", None)
    if uid is None:
        return None
    try:
        from webview.platforms import cocoa
        inst = cocoa.BrowserView.instances.get(uid)
        return None if inst is None else inst.webview
    except Exception:
        return None


def _call_on_main(fn):
    """Queue `fn` on the Cocoa main thread (FIFO). False when it can't be."""
    try:
        from PyObjCTools import AppHelper
        AppHelper.callAfter(fn)
        return True
    except Exception:
        return False


def _make_effect_view(flipped):
    """One hidden NSVisualEffectView, configured as the pill's glass.

    * `behindWindow` blending: the whole point -- blur what is on the DESKTOP
      behind this window. (`withinWindow` would blur the page, which is all
      CSS backdrop-filter could ever do here.)
    * state `active`, not the default follows-window-active: the pill is a
      non-activating panel that is almost never key, and an inactive effect
      view renders flat grey with no blur at all.
    * `HUDWindow` material under a forced DarkAqua appearance: Apple's
      material for floating heads-up controls, dark in every system mode --
      the pill's chassis is near-black whether the user runs light or dark,
      and a light material would put the white UI text on a pale ground.
    """
    import AppKit
    view = AppKit.NSVisualEffectView.alloc().initWithFrame_(
        AppKit.NSMakeRect(0, 0, 1, 1))
    view.setMaterial_(NSVISUAL_EFFECT_MATERIAL_HUD_WINDOW)
    view.setBlendingMode_(NSVISUAL_EFFECT_BLENDING_BEHIND_WINDOW)
    view.setState_(NSVISUAL_EFFECT_STATE_ACTIVE)
    view.setWantsLayer_(True)
    dark = AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameDarkAqua)
    if dark is not None:
        view.setAppearance_(dark)
    view.setAutoresizingMask_(glass_autoresizing_mask(flipped))
    view.setHidden_(True)
    return view


def _capsule_mask(radius):
    """A resizable capsule mask: a (2r+1)pt square holding a radius-r rounded
    rect, cap insets of r on every side, stretch mode. AppKit then stretches
    only the 1pt centre, so one image masks the pill at ANY width -- the view
    re-applies the mask through its cap insets on every resize.

    A mask image and not `layer.cornerRadius`: behind-window blending is done
    by the window server, and a layer corner radius does not reliably clip
    it. `maskImage` is the documented shape control for NSVisualEffectView.
    Drawn by a handler so it re-renders at whatever backing scale the display
    has.
    """
    import AppKit
    r = float(radius)
    side = 2.0 * r + 1.0

    def _draw(rect):
        AppKit.NSColor.blackColor().set()
        AppKit.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            rect, r, r).fill()
        return True

    image = AppKit.NSImage.imageWithSize_flipped_drawingHandler_(
        AppKit.NSMakeSize(side, side), False, _draw)
    image.setCapInsets_(AppKit.NSEdgeInsets(r, r, r, r))
    image.setResizingMode_(AppKit.NSImageResizingModeStretch)
    return image


class Glass(object):
    """Native frosted glass under the pill and the hint strip.

    CSS `backdrop-filter` cannot do this: it blurs what is behind an element
    IN THE PAGE, and in a transparent native window the page has nothing
    behind the pill -- the desktop is on the far side of the window, where
    only the window server can reach it. So the glass is AppKit's:
    `NSVisualEffectView`s added to the WKWebView BELOW WebKit's own content
    view (the web content stays transparent and draws over them), one per
    rect `content_layout` returns, each masked to the CSS capsule.

    * Created once (`install`, off the pill's `shown` event) and re-framed on
      every `bar_fit` (`update`); a view with no rect -- the hint strip went
      away -- is hidden, never destroyed and rebuilt.
    * Every AppKit write goes through `callAfter`. The main thread's queue is
      FIFO, so an `update` queued after `set_frame` runs against the size the
      window was just given; `glass_autoresizing_mask` covers the one frame
      AppKit draws in between.
    * The views sit under WebKit's content view, which spans the whole web
      view, so hit-testing never reaches them: click-through, the drag region
      and every control behave exactly as without glass.
    * `installed` is True only once the views really exist in the host.
      `bar_fit` reports "glass" off it, and bar.js only turns the pill
      translucent when it does -- so any failure here leaves today's opaque
      pill, never a see-through one over nothing. The first update that
      actually shows a view also tells the page directly (`GLASS_ON_JS`),
      because on the lazy-install path the fit that installs has already
      replied "no glass" by the time the queued install runs.
    * Soft-fails like everything else in this module: no pyobjc, no web
      view, or any exception -> not installed, and a failed install is not
      retried. `AUTOCINE_NO_VIBRANCY=1` switches it off entirely.

    `host_reader`, `call_on_main`, `make_view` and `make_mask` are seams so
    the logic tests with stand-ins and no Cocoa.
    """

    MAX_VIEWS = 2          # the pill + the hint strip: all content_layout makes

    def __init__(self, window, host_reader=None, call_on_main=None,
                 make_view=None, make_mask=None):
        self.window = window
        self.installed = False     # written on the main thread only
        self._failed = False       # an install/update blew up: stay CSS-only
        self._announced = False    # GLASS_ON_JS sent (main thread only)
        self._host = None
        self._flipped = True
        self._views = []
        self._radii = []           # mask radius per view, so it's set once
        self._masks = {}           # radius -> image, shared + kept alive
        self._host_reader = host_reader or (lambda: _web_view(self.window))
        self._call_on_main = call_on_main or _call_on_main
        self._make_view = make_view or _make_effect_view
        self._make_mask = make_mask or _capsule_mask

    @staticmethod
    def enabled():
        return os.environ.get("AUTOCINE_NO_VIBRANCY") != "1"

    def install(self):
        """Queue the one-time creation of the glass views. True = queued."""
        if not self.enabled() or self._failed:
            return False
        return self._call_on_main(self._install_now)

    def update(self, rects):
        """Queue a re-frame to `rects` (layout coords, as `content_layout`
        returns them). Call it AFTER `set_frame`. Installs lazily if `install`
        never ran (a pywebview with no `shown` event). True = queued."""
        if not self.enabled() or self._failed:
            return False
        rects = list(rects or ())
        return self._call_on_main(lambda: self._update_now(rects))

    # --- main thread only ---------------------------------------------------
    def _install_now(self):
        if self.installed or self._failed:
            return self.installed
        host = self._host_reader()
        if host is None:
            return False           # not a failure: a later update retries
        views = []
        try:
            flipped = bool(host.isFlipped())
            for _ in range(self.MAX_VIEWS):
                view = self._make_view(flipped)
                # relativeTo None + below = the very bottom of the subview
                # list, under WebKit's content view.
                host.addSubview_positioned_relativeTo_(view, NSWINDOW_BELOW,
                                                       None)
                views.append(view)
        except Exception:
            for view in views:
                try:
                    view.removeFromSuperview()
                except Exception:
                    pass
            self._failed = True
            return False
        self._host, self._flipped = host, flipped
        self._views = views
        self._radii = [None] * len(views)
        self.installed = True
        return True

    def _update_now(self, rects):
        if not self._install_now():
            return False
        try:
            host_h = 0.0
            if not self._flipped:
                host_h = float(self._host.bounds().size.height)
            frames = glass_frames(rects, host_h, self._flipped)
            for i, view in enumerate(self._views):
                if i >= len(frames):
                    view.setHidden_(True)
                    continue
                x, y, w, h, radius = frames[i]
                if self._radii[i] != radius:
                    mask = self._masks.get(radius)
                    if mask is None:
                        mask = self._masks[radius] = self._make_mask(radius)
                    view.setMaskImage_(mask)
                    self._radii[i] = radius
                view.setFrame_(((x, y), (w, h)))
                view.setHidden_(False)
        except Exception:
            self._teardown()
            return False
        if frames and not self._announced:
            self._announced = True
            self._tell_page(GLASS_ON_JS)
        return True

    def _tell_page(self, script):
        """Run `script` in the pill's page, from a helper thread: this runs
        on the main thread, and `evaluate_js` blocks on a reply the main
        thread would have to deliver. Soft-fails (no window, no evaluate_js,
        any exception)."""
        evaluate = getattr(self.window, "evaluate_js", None)
        if evaluate is None:
            return

        def _run():
            try:
                evaluate(script)
            except Exception:
                pass
        threading.Thread(target=_run, daemon=True).start()

    def _teardown(self):
        """Hide every view and fall back to CSS for good.

        The page may already be wearing the translucent pill, and a bar_fit
        reply only ever ADDS the class (bar.js applyGlass), so this is the one
        path that takes it away: `GLASS_LOST_JS`, which also marks the loss so
        a late reply cannot undo it. Sent from a thread (`_tell_page`).
        """
        self.installed = False
        self._failed = True
        for view in self._views:
            try:
                view.setHidden_(True)
            except Exception:
                pass
        self._tell_page(GLASS_LOST_JS)


def _mouse_state():
    """(x, y, pressed_buttons) in Cocoa screen coords, or None without AppKit."""
    try:
        import AppKit
        loc = AppKit.NSEvent.mouseLocation()
        return (loc.x, loc.y, AppKit.NSEvent.pressedMouseButtons())
    except Exception:
        return None


class ClickThrough(object):
    """Flip `NSWindow.ignoresMouseEvents` by where the pointer is.

    The window is a big transparent rectangle; only the pill (and the hint
    strip) should catch clicks. Cocoa has no per-pixel hit testing for a
    WKWebView-backed window, so we watch the pointer instead: inside a live
    rect the window is interactive, everywhere else it's transparent to clicks
    and whatever is behind it gets them.

    **Everything here is built to fail toward "interactive."** An unclickable
    pill is a far worse bug than a bit of dead space, so:

    * Rects are held in LAYOUT coords and converted against the window's LIVE
      frame on every check. Freezing them in screen coords at report time was
      wrong: `set_frame` is async (`callAfter`), so a second `bar_fit` reads a
      stale frame, and the rects end up describing a position the window isn't
      at — the pointer is then never "inside" and the pill latches unclickable.
    * The pointer is POLLED rather than watched through NSEvent monitors.
      Monitors have to be right about local-vs-global delivery, install order,
      and threading, and any gap there means the flag never gets cleared.
      `NSEvent.mouseLocation()` is cheap and always readable.
    * `ignore=True` is only ever possible while the poller is confirmed
      running (`_armed`), and `stop()` restores interactivity.
    * Any failure — no pyobjc, no frame, no mouse read — means interactive.

    Set AUTOCINE_NO_CLICKTHROUGH=1 to switch it off entirely; the window
    is then simply always interactive, dead space and all.
    """

    POLL_SEC = 0.04
    # The rect is grown a touch when testing, so arriving at the pill's edge
    # arms it slightly early and a pixel of jitter can't flicker the state.
    HYSTERESIS_PT = 6

    def __init__(self, window, mouse_state=None, apply_ignore=None,
                 frame_reader=None):
        self.window = window
        self._rects = []           # LAYOUT coords, relative to window top-left
        self._ignoring = None      # None = unknown / not yet applied
        self._armed = False        # only the running poller may set True
        self._stop = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        # seams, so the decision logic is testable without a live NSWindow
        self._mouse_state = mouse_state or _mouse_state
        self._apply_ignore = apply_ignore or self._set_ignores_mouse_events
        self._frame_reader = frame_reader or (lambda: frame_of(self.window))

    def set_rects(self, rects):
        """Set the interactive regions, in layout coords (x, y, w, h) relative
        to the window's top-left — the same space `content_layout` returns."""
        with self._lock:
            self._rects = list(rects or ())
        self._update()

    def should_ignore(self, x, y):
        """True when the pointer is over transparent slack, so clicks at
        (x, y) belong to whatever is behind the window.

        No rects reported yet, or no readable window frame -> False. A page
        that never calls back must not render the pill permanently unclickable.
        """
        with self._lock:
            rects = list(self._rects)
        if not rects:
            return False
        frame = self._frame_reader()
        if frame is None:
            return False
        live = hit_rects_to_screen(frame, rects, grow=self.HYSTERESIS_PT)
        return bool(live) and not point_in_rects(x, y, live)

    def start(self):
        """Begin polling. False = click-through stays off (window fully
        interactive). Safe to call from any thread."""
        if os.environ.get("AUTOCINE_NO_CLICKTHROUGH") == "1":
            return False
        if self._mouse_state() is None:      # no pyobjc -> never arm
            return False
        self._armed = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self._armed = False
        if self._ignoring:                   # never leave it unclickable
            self._apply_ignore(False)
            self._ignoring = False

    def _loop(self):
        try:
            while not self._stop.wait(self.POLL_SEC):
                self._update()
        finally:
            # if the watcher ever dies, it must not leave the window ignoring
            # mouse events with nothing left to turn that back off
            self._armed = False
            if self._ignoring:
                self._apply_ignore(False)
                self._ignoring = False

    def _set_ignores_mouse_events(self, ignore):
        """Queue the flag onto the main thread — AppKit wants it there, and we
        reach this from the poller and the JS bridge thread."""
        nswin = _ns_window(self.window)
        if nswin is None:
            return False
        try:
            from PyObjCTools import AppHelper

            def _apply():
                try:
                    nswin.setIgnoresMouseEvents_(ignore)
                except Exception:
                    pass

            AppHelper.callAfter(_apply)
            return True
        except Exception:
            return False

    def _update(self):
        try:
            state = self._mouse_state()
            if state is None:
                return
            x, y, buttons = state
            # Never re-arm mid-gesture: flipping the flag while the user is
            # dragging the pill would drop the drag on the floor.
            if buttons:
                return
            ignore = self._armed and self.should_ignore(x, y)
            if ignore == self._ignoring:
                return
            if self._apply_ignore(ignore):
                self._ignoring = ignore
        except Exception:
            pass
