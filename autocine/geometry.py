"""Load and normalize the recorded input-event stream."""

import json
from typing import Dict

import numpy as np


def frontmost_owner(state, x, y):
    """Which window a point landed in, plus whether the choice was ambiguous.

    `state` maps window_id -> (rect, z), rect = (x, y, w, h), z = front-to-back
    rank (0 = frontmost, -1 = unknown). Returns `(owner_id | None, ambiguous)`:

    - the FRONTMOST containing window when any candidate has a known z;
    - else the SMALLEST containing window -- the one a user would say they
      clicked, a dialog over its parent -- with `ambiguous=True` when that
      fallback had to pick among MORE THAN ONE containing window with no z to
      separate them (a genuinely undecidable overlap).

    A track with no z at all is not treated as "everything is frontmost". The
    owner is None (ambiguous False) when the point is inside no window.

    This is the single authority for point -> window: `beats._hit` delegates
    here, and render's whole-screen "grow the active window" resolver calls it
    with the same `state` shape, so the beat sheet and the render agree on who
    owns a click.
    """
    cands = []
    for wid, (r, z) in state.items():
        if r[0] <= x <= (r[0] + r[2]) and r[1] <= y <= (r[1] + r[3]):
            cands.append((int(wid), float(r[2]) * float(r[3]), int(z)))
    if not cands:
        return None, False
    known = [c for c in cands if c[2] >= 0]
    if known:
        return int(min(known, key=lambda c: (c[2], c[1]))[0]), False
    owner = int(min(cands, key=lambda c: c[1])[0])
    return owner, (len(cands) > 1)


def load_events(path: str) -> Dict[str, np.ndarray]:
    """Read events.jsonl into sorted numpy arrays (raw monotonic-second times).

    Returns clicks (mouse-down), mouse-up times, a movement track,
    key-activity tick times, and scroll ticks. Every click is also appended to the movement track so the camera
    has a position sample there. Key lines carry only a quantized timestamp
    (their x/y, when present, is just the last cursor position padded in for
    forward-compat -- not a real position sample, so it is NOT added to the
    move track). Sessions recorded before key capture simply yield an empty
    keys_t.

    Scroll lines carry a REAL cursor position (macOS delivers scroll events
    to the window under the cursor), but like ups they are deliberately NOT
    added to the move track: scroll influence on the camera is a toggleable
    feature (render.scroll_zoom), and keeping the events in their own
    scrolls_* arrays is what makes the off switch -- and every pre-scroll
    session -- bit-exact with the historical camera. Sessions recorded
    before scroll capture simply yield empty scrolls_*.

    Window lines (`type: "window"`) are record-time geometry samples of every
    on-screen window, in POINTS with a global top-left origin -- the track
    render follows so a window that is moved or resized mid-take stays framed,
    both for a `--capture-window` crop and for each card of the multi-window
    grid. They yield `windows_t`, `windows_rect` (an (N, 4) array of x/y/w/h)
    and `windows_id` (which window each sample belongs to; -1 for the
    capture-target-only tracks written before the id was recorded). Like keys
    they never join the move track: their x/y is padding, not a cursor sample.
    Sessions recorded before geometry tracking yield an empty track and fall
    back to a static crop.

    `windows_z` is the front-to-back rank at each sample (0 = frontmost), or
    -1 where the recorder didn't write one. It is what lets render tell that a
    window was COVERED during part of a take -- the one thing display-capture-
    plus-crop cannot see for itself -- without ever recording what covered it.
    An all -1 track means "no occlusion information", not "nothing was on top".
    """
    moves_t, moves_x, moves_y = [], [], []
    clicks_t, clicks_x, clicks_y = [], [], []
    scrolls_t, scrolls_x, scrolls_y = [], [], []
    ups_t = []
    keys_t = []
    wins_t, wins_rect, wins_id = [], [], []
    wins_z = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                e = json.loads(line)
                etype = e.get("type")
                if etype == "key":   # no positional payload -- check first
                    keys_t.append(e["t"])
                    continue
                if etype == "window":
                    # Window-geometry sample (POINTS, global top-left origin).
                    # Its x/y are forward-compat padding, not a cursor
                    # position, so like keys it never joins the move track.
                    r = e.get("rect")
                    if isinstance(r, (list, tuple)) and len(r) == 4:
                        try:
                            wins_rect.append([float(v) for v in r])
                        except (TypeError, ValueError):
                            continue
                        wins_t.append(e["t"])
                        # Front-to-back rank, 0 = frontmost. -1 = "unknown",
                        # which is every sample written before the poller
                        # recorded it; consumers must treat an all -1 track
                        # as "no occlusion information", not "all frontmost".
                        try:
                            zv = e.get("z")
                            wins_z.append(-1 if zv is None else int(zv))
                        except (TypeError, ValueError):
                            wins_z.append(-1)
                        # -1 = "unlabelled", which is what the first
                        # capture-window-only tracks wrote. Consumers that
                        # filter by id must treat an all -1 track as "these
                        # all belong to the capture target".
                        try:
                            wins_id.append(int(e.get("id", -1)))
                        except (TypeError, ValueError):
                            wins_id.append(-1)
                    continue
                t, x, y = e["t"], e["x"], e["y"]
                if etype == "move":
                    moves_t.append(t); moves_x.append(x); moves_y.append(y)
                elif etype == "down":
                    clicks_t.append(t); clicks_x.append(x); clicks_y.append(y)
                    moves_t.append(t); moves_x.append(x); moves_y.append(y)
                elif etype == "up":
                    # Recorded since day one; used for drag-aware zoom holds
                    # (down->up pairing). Deliberately NOT added to the move
                    # track, so sessions rendered with drag_hold off stay
                    # bit-exact with the pre-feature renderer.
                    ups_t.append(t)
                elif etype == "scroll":
                    scrolls_t.append(t); scrolls_x.append(x); scrolls_y.append(y)
    except FileNotFoundError:
        pass

    if moves_t:
        order = np.argsort(moves_t)
        mt = np.asarray(moves_t)[order]
        mx = np.asarray(moves_x)[order]
        my = np.asarray(moves_y)[order]
    else:
        mt = mx = my = np.array([])

    if scrolls_t:
        s_order = np.argsort(scrolls_t)
        st = np.asarray(scrolls_t)[s_order]
        sx = np.asarray(scrolls_x)[s_order]
        sy = np.asarray(scrolls_y)[s_order]
    else:
        st = sx = sy = np.array([])

    if wins_t:
        w_order = np.argsort(wins_t)
        wt = np.asarray(wins_t, dtype=float)[w_order]
        wr = np.asarray(wins_rect, dtype=float)[w_order]
        wi = np.asarray(wins_id, dtype=int)[w_order]
        wz = np.asarray(wins_z, dtype=int)[w_order]
    else:
        wt = np.array([])
        wr = np.zeros((0, 4))
        wi = np.array([], dtype=int)
        wz = np.array([], dtype=int)

    return {
        "windows_t": wt, "windows_rect": wr, "windows_id": wi,
        "windows_z": wz,
        "moves_t": mt, "moves_x": mx, "moves_y": my,
        "clicks_t": np.asarray(clicks_t),
        "clicks_x": np.asarray(clicks_x),
        "clicks_y": np.asarray(clicks_y),
        "ups_t": np.sort(np.asarray(ups_t)),
        "keys_t": np.sort(np.asarray(keys_t)),
        "scrolls_t": st, "scrolls_x": sx, "scrolls_y": sy,
    }
