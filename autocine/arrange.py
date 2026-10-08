"""Pull overlapping windows apart before a multi-window take.

WHY THIS EXISTS. avfoundation captures a DISPLAY, and render crops each
window out of that capture -- so a window sitting on top of a picked window
is physically inside that window's card. Separating the cards in the
composition cannot undo it: the pixels underneath were never recorded. The
only fix that doesn't need ScreenCaptureKit is to make sure nothing IS on top
when the take starts, which means moving the user's real windows and putting
them back afterwards.

Split in two on purpose:

  * `plan_separation` is pure geometry -- no Quartz, no Accessibility, no
    permissions -- so the part with the interesting failure modes is the part
    that unit-tests offline;
  * `apply_arrangement` / `restore` touch the Accessibility API and soft-fail
    at every step, the same posture as the geometry poller and the facecam.
    A window that refuses to move is a worse recording, never a failed one.

Accessibility is ALREADY required to record at all (`permissions.build_report`
lists it in `required`, and `_start_record` refuses a take when it's missing),
so this adds no new grant -- it uses one the user has necessarily given.
"""

import os

from . import devices as dev

# Breathing room left between windows after separation, in points. Bigger
# than the compositor's gap on purpose: this one has to survive a shadow,
# and a 1pt sliver of the window behind is worse than an obvious gap.
DEFAULT_GAP_PT = 24.0
# Below this a window stops being worth recording, so shrink-to-fit gives up
# rather than producing four unreadable slivers.
MIN_W_PT = 240.0
MIN_H_PT = 180.0
# How closely an AX window's frame must match Quartz's before we believe
# they're the same window. Both report global top-left points, so this is
# rounding slack, not a search radius.
MATCH_TOL_PT = 6.0


# ---------------------------------------------------------------- pure ----

def _inflated_overlap(a, b, gap):
    """Overlap depths of two rects, each inflated by `gap`/2 per side."""
    h = gap / 2.0
    ax, ay, aw, ah = a[0] - h, a[1] - h, a[2] + gap, a[3] + gap
    bx, by, bw, bh = b[0] - h, b[1] - h, b[2] + gap, b[3] + gap
    return (min(ax + aw, bx + bw) - max(ax, bx),
            min(ay + ah, by + bh) - max(ay, by))


def any_overlap(rects, gap=0.0):
    """Does any pair of rects (inflated by `gap`) intersect?"""
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            ox, oy = _inflated_overlap(rects[i], rects[j], gap)
            if ox > 0.0 and oy > 0.0:
                return True
    return False


def _separate(boxes, gap, max_passes=64):
    """Push overlapping boxes apart along each pair's shallower axis.

    Same idea as `framing._separate_boxes`, but with a gap and in POINTS on
    the real desktop rather than in source pixels on a canvas -- kept
    separate because this one also has to survive the screen clamp below,
    and merging them would mean one function with two contradictory jobs.
    """
    n = len(boxes)
    for _ in range(max_passes):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = boxes[i], boxes[j]
                ox, oy = _inflated_overlap(a, b, gap)
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
            return
    return


def _clamp_into(boxes, screen):
    """Slide every box fully back inside the screen (never resize here)."""
    sx, sy, sw, sh = (float(v) for v in screen)
    for b in boxes:
        b[0] = max(sx, min(sx + sw - b[2], b[0]))
        b[1] = max(sy, min(sy + sh - b[3], b[1]))


def plan_separation(rects, screen, gap=DEFAULT_GAP_PT,
                    min_w=MIN_W_PT, min_h=MIN_H_PT):
    """Non-overlapping on-screen targets for `rects`, or None.

    None has two distinct meanings and both are correct outcomes, not
    errors: *nothing overlapped, so there is nothing to do*, and *these
    windows cannot be made to fit side by side on this display*. The caller
    treats both as "record as-is" -- which is why this returns None rather
    than a best-effort arrangement that still overlaps. Handing back a plan
    that doesn't actually separate them would move the user's windows for no
    benefit at all, which is the one outcome with no upside.

    Windows are shrunk only if they have to be, uniformly, and never below
    `min_w` x `min_h`.
    """
    boxes = []
    for r in rects:
        try:
            boxes.append([float(r[0]), float(r[1]),
                          max(1.0, float(r[2])), max(1.0, float(r[3]))])
        except (IndexError, TypeError, ValueError):
            return None
    if len(boxes) < 2 or not any_overlap(boxes, gap=0.0):
        return None
    try:
        sx, sy, sw, sh = (float(v) for v in screen)
    except (IndexError, TypeError, ValueError):
        return None
    if sw <= 0.0 or sh <= 0.0:
        return None

    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        cand = []
        for x, y, w, h in boxes:
            nw = max(min_w, w * scale) if scale < 1.0 else w
            nh = max(min_h, h * scale) if scale < 1.0 else h
            # Shrinking about the window's own centre keeps it near where the
            # user last saw it, which makes the move read as "tidied" rather
            # than "scattered".
            cand.append([x + (w - nw) / 2.0, y + (h - nh) / 2.0, nw, nh])
        if any(b[2] > sw or b[3] > sh for b in cand):
            continue
        # Separate and clamp alternately: clamping a box off the screen edge
        # can push it back into a neighbour, so neither pass alone converges.
        for _ in range(8):
            _separate(cand, gap)
            _clamp_into(cand, screen)
            if not any_overlap(cand, gap=0.0):
                return [[round(v, 1) for v in b] for b in cand]
    return None


# ------------------------------------------------------- Accessibility ----

def _ax():
    """The ApplicationServices symbols we need, or None. Never raises."""
    try:
        from ApplicationServices import (
            AXUIElementCreateApplication, AXUIElementCopyAttributeValue,
            AXUIElementSetAttributeValue, AXValueCreate, AXValueGetValue,
            kAXWindowsAttribute, kAXPositionAttribute, kAXSizeAttribute,
            kAXValueCGPointType, kAXValueCGSizeType)
        from Quartz import CGPoint, CGSize
    except Exception:
        return None
    return {
        "app": AXUIElementCreateApplication,
        "get": AXUIElementCopyAttributeValue,
        "set": AXUIElementSetAttributeValue,
        "mkval": AXValueCreate,
        "getval": AXValueGetValue,
        "windows": kAXWindowsAttribute,
        "pos": kAXPositionAttribute,
        "size": kAXSizeAttribute,
        "pt_t": kAXValueCGPointType,
        "sz_t": kAXValueCGSizeType,
        "Point": CGPoint,
        "Size": CGSize,
    }


def _ax_frame(ax, win):
    """`[x, y, w, h]` of one AX window in global top-left points, or None."""
    try:
        err_p, pos = ax["get"](win, ax["pos"], None)
        err_s, size = ax["get"](win, ax["size"], None)
        if err_p or err_s or pos is None or size is None:
            return None
        ok_p, pt = ax["getval"](pos, ax["pt_t"], None)
        ok_s, sz = ax["getval"](size, ax["sz_t"], None)
        if not ok_p or not ok_s:
            return None
        return [float(pt.x), float(pt.y), float(sz.width), float(sz.height)]
    except Exception:
        return None


def _ax_window_matching(ax, pid, rect, tol=MATCH_TOL_PT):
    """The app's AX window whose frame matches `rect`, or None.

    Matched on geometry through PUBLIC API rather than through the private
    `_AXUIElementGetWindow`, which is the usual way to turn a CGWindowID into
    an AX element. Both AX and CGWindowBounds report global top-left points,
    so an exact-ish frame match is unambiguous in practice -- and when it
    isn't (two identically placed windows in one app) the caller just doesn't
    move that window, which is the safe direction to be wrong in.
    """
    try:
        app = ax["app"](int(pid))
        err, wins = ax["get"](app, ax["windows"], None)
        if err or not wins:
            return None
    except Exception:
        return None
    for win in wins:
        frame = _ax_frame(ax, win)
        if frame is None:
            continue
        if all(abs(a - b) <= tol for a, b in zip(frame, rect)):
            return win
    return None


def _ax_set_frame(ax, win, rect):
    """Move+resize one AX window. True only if it actually landed there."""
    x, y, w, h = (float(v) for v in rect)
    try:
        # Position, then size, then position again: an app that clamps its
        # size against the screen can shove itself back after the first move,
        # and re-asserting the origin is cheaper than reasoning about which
        # apps do it.
        ax["set"](win, ax["pos"], ax["mkval"](ax["pt_t"], ax["Point"](x, y)))
        ax["set"](win, ax["size"], ax["mkval"](ax["sz_t"], ax["Size"](w, h)))
        ax["set"](win, ax["pos"], ax["mkval"](ax["pt_t"], ax["Point"](x, y)))
    except Exception:
        return False
    landed = _ax_frame(ax, win)
    if landed is None:
        return False
    # Generous tolerance: plenty of apps snap to a row/column grid (terminals)
    # or enforce a minimum size, and a window that ended up 12pt off is still
    # un-overlapped, which is the whole point.
    return all(abs(a - b) <= 24.0 for a, b in zip(landed, rect))


def apply_arrangement(entries, targets):
    """Move `entries` to `targets`. Returns `(moved_count, restore_token)`.

    `entries` are `devices.list_windows` Entries (they carry the window id and
    the CURRENT rect, which is what identifies the AX element). `targets` is
    what `plan_separation` returned, index-aligned.

    `restore_token` is opaque -- pass it to `restore`. It holds live AX
    element references, so it is only meaningful inside this process, which
    is exactly the lifetime it needs: the Recorder that starts the take is
    the one that ends it.
    """
    ax = _ax()
    if ax is None or not targets:
        return 0, None
    saved = []
    moved = 0
    for entry, target in zip(entries, targets):
        try:
            wid = int(entry["id"])
            current = [float(entry["x"]), float(entry["y"]),
                       float(entry["w"]), float(entry["h"])]
        except (KeyError, TypeError, ValueError):
            continue
        pid = dev.window_owner_pid(wid, exclude_pids=(os.getpid(),))
        if pid is None:
            continue
        win = _ax_window_matching(ax, pid, current)
        if win is None:
            continue
        if _ax_set_frame(ax, win, target):
            moved += 1
            saved.append((win, current))
        else:
            # It half-moved. Put it back immediately rather than leaving the
            # user's desktop in a state nobody asked for.
            _ax_set_frame(ax, win, current)
    return moved, (saved or None)


def restore(token):
    """Put windows back where `apply_arrangement` found them. Best-effort."""
    if not token:
        return
    ax = _ax()
    if ax is None:
        return
    for win, rect in token:
        try:
            _ax_set_frame(ax, win, rect)
        except Exception:
            continue
