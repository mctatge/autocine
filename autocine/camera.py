"""Auto-zoom camera model.

One concept: `zoomRanges`. Auto-cluster clicks into proposed ranges;
render each as a smooth spring-driven zoom that leads the first click
and holds through the last. No per-frame cursor following, no typing/
scroll/drag heuristics -- those layered auto-detectors are what made
the previous camera feel wrong. The legacy file survives at
`camera_legacy.py` for reference during the rewrite.

Signature (`build_path`) matches the legacy call for source-compat with
`render.py`; the extra kwargs (keys_t, ups_t, scrolls_*, typing_anchors)
are accepted and ignored.

Springs use published reference values (screen mass 2.25,
stiffness 200, damping 40 -> omega=9.43, zeta=0.94, near-critically
damped). Integration is the closed-form analytic step so it's
fps-independent.
"""

from math import cos, exp, sin, sqrt

import numpy as np


# Cinematic defaults. The pre-roll is DELIBERATELY long
# (2.5s) -- the reference project shows zooms starting ~2.8s before
# the first click of a cluster. Short pre-rolls feel reactive; long
# pre-rolls read as intentional cinematography.
DEFAULTS = dict(
    max_zoom=2.0,
    pre_roll=2.5,        # s, zoom-in leads first click of cluster
    tail=0.5,            # s, zoom-out follows last click of cluster
    chain_gap=4.0,       # s, clicks within this join one cluster
    min_room=0.5,        # s, drop a cluster whose last click is nearer than
                         # this to the clip end (matches the reference
                         # trailing-click behavior on the reference session
                         # whose last click landed *past* clip end).
    snap_ratio=0.25,     # magnetic pull to edges (frac of half-window)
    # Overview framing: a cluster whose clicks are spread WIDER than a
    # max_zoom window can't be held -- chasing each one whip-pans across
    # the screen (motion sickness). Instead, drop that range's zoom to
    # fit the activity bbox and LOCK the center on it: a still overview,
    # the way a cinematic camera frames scattered clicks. Bit-exact off with
    # overview=False.
    overview=True,
    overview_fill=0.8,   # activity bbox fills at most this frac of window
    overview_zoom_min=1.3,  # never zoom out past this for an overview
    # Screen spring (m=2.25, k=200, c=40)
    omega=9.43,          # sqrt(k/m)  rad/s
    zeta=0.94,           # c / (2*sqrt(k*m))
    # Zoom spring (same characteristic; can diverge later)
    z_omega=9.43,
    z_zeta=0.94,
    always_zoomed=False,
    # Window focus: the two-stage ladder (see `build_focus_emphasis`).
    # Stage 1 leans the layout partway toward the clicked card; stage 2 --
    # reached on that card's SECOND click -- goes all the way, which is
    # emphasis 1.0 by definition since it IS the focused layout.
    focus_lean=0.5,      # stage 1: how far toward the focused layout
    focus_full_min=1.0,  # s, stage 2 needs at least this long to be worth it
    focus_hold=1.8,      # s, a cluster that EARNS stage 2 holds this long
                         # after its second click. Without it `tail` (0.5s)
                         # is all the room a two-click cluster ever has, and
                         # since that is always below `focus_full_min` the
                         # escalation could never fire at all -- "click twice
                         # and move on" is the common case, not an edge one.
    # Legacy no-ops -- accepted so render.py's flag surface still works.
    # None of these affect the new plan; they're here to prevent
    # KeyErrors in params dicts that came from edits.json render options.
    typing_bucket=0.1,   # pinned by test_record.py; must equal record.KEY_BUCKET_SEC
)

# Legacy zoom-speed presets. In the new model there's no separate
# in/out duration to scale; a slower preset just widens the pre-roll
# (giving the zoom more time to breathe).
SPEED_FACTORS = {"slow": 1.5, "normal": 1.0, "fast": 0.7}


def build_params(max_zoom, params):
    P = dict(DEFAULTS)
    if params:
        for k, v in params.items():
            if v is None:
                continue
            P[k] = v
    P["max_zoom"] = float(max_zoom)
    # Apply zoom_speed preset: stretch pre_roll (holds untouched).
    speed = str(P.get("zoom_speed") or "normal").lower()
    factor = SPEED_FACTORS.get(speed, 1.0)
    P["pre_roll"] = float(P["pre_roll"]) * factor
    # Coerce numerics
    for k in ("max_zoom", "pre_roll", "tail", "chain_gap", "min_room",
             "snap_ratio", "omega", "zeta", "z_omega", "z_zeta",
             "overview_fill", "overview_zoom_min",
             "focus_lean", "focus_full_min", "focus_hold"):
        P[k] = float(P[k])
    P["always_zoomed"] = bool(P.get("always_zoomed"))
    P["overview"] = bool(P.get("overview"))

    class _P: pass
    p = _P()
    for k, v in P.items():
        setattr(p, k, v)
    return p


def _damped_step(x, v, target, omega, zeta, dt):
    """Closed-form analytic step for a linear damped harmonic oscillator.

    Exact for arbitrary dt (fps-independent). For zeta ~ 1 (our regime),
    behaves like a critically-damped follower with minimal overshoot.
    """
    if dt <= 0.0:
        return x, v
    if omega <= 0.0:
        return target, 0.0
    A = x - target
    if zeta < 1.0:
        wd = omega * sqrt(1.0 - zeta * zeta)
        B = (v + zeta * omega * A) / wd
        e = exp(-zeta * omega * dt)
        c = cos(wd * dt)
        s = sin(wd * dt)
        x1 = target + e * (A * c + B * s)
        v1 = (-zeta * omega * (x1 - target)
              + e * wd * (-A * s + B * c))
    elif zeta == 1.0:
        e = exp(-omega * dt)
        x1 = target + e * (A + (v + omega * A) * dt)
        v1 = e * ((v + omega * A) - omega * (A + (v + omega * A) * dt))
    else:
        r = omega * sqrt(zeta * zeta - 1.0)
        s1 = -zeta * omega + r
        s2 = -zeta * omega - r
        c1 = (v - s2 * A) / (s1 - s2)
        c2 = A - c1
        e1 = exp(s1 * dt)
        e2 = exp(s2 * dt)
        x1 = target + c1 * e1 + c2 * e2
        v1 = c1 * s1 * e1 + c2 * s2 * e2
    return x1, v1


# -- Planner (offline) ---------------------------------------------------

def _in_suppressed(t, suppressed):
    """Is `t` inside any suppressed span? (the password / sensitive-content
    escape hatch). Module level because `build_card_paths` filters per-card
    segments with the same rule `plan_zoom` applies to its click stream."""
    for sp in (suppressed or []):
        s = float(sp.get("start", 0.0))
        e = float(sp.get("end", 0.0))
        if e > s and s <= t <= e:
            return True
    return False


def cluster_clicks(clicks, chain_gap):
    """Chain sorted (t, x, y) clicks into clusters, breaking on `chain_gap`.

    Public with `cluster_to_range` and `build_params` because the planner is
    not the only thing that needs to know where the camera will move: the
    beat sheet (`beats.py`) reports each cluster's zoom span, and it has to
    be the span render actually plans, not a lookalike that drifts. There is
    one further hand-written copy of this contract -- `edits
    .auto_zoom_proposals`, kept hand-written so edits.py stays numpy-free --
    and `test_beats.py` pins the two against each other on the same input.
    """
    if not clicks:
        return []
    out = [[clicks[0]]]
    for c in clicks[1:]:
        if c[0] - out[-1][-1][0] <= chain_gap:
            out[-1].append(c)
        else:
            out.append([c])
    return out


def cluster_to_range(cluster, P, T_end):
    """Turn a click cluster into a zoom range.

    Returns None ONLY for a lone trailing click with no room for a
    zoom-out arc (a single click within `min_room` of T_end, or past
    it) -- matches the reference behavior on the reference session's
    ninth click, which lands past clip end.

    A *multi-click* cluster near the end is real activity the user cared
    about (they clicked right up until they hit stop). Dropping the whole
    span would leave seconds of clicking un-zoomed, so instead we zoom
    and HOLD the level through the clip end -- no zoom-out arc, because
    there's no room for one (`hold_out`)."""
    first = cluster[0]
    last = cluster[-1]
    start = max(0.0, first[0] - P.pre_roll)
    no_room = (T_end - last[0]) < P.min_room
    if no_room and len(cluster) <= 1:
        # Lone trailing click: no zoom (pinned by test).
        return None
    hold_out = no_room
    end = T_end if hold_out else min(last[0] + P.tail, T_end)
    return {
        "startTime": start,
        "endTime": end,
        "zoom": P.max_zoom,
        "type": "follow-click-groups",
        "clicks": [(float(c[0]), float(c[1]), float(c[2])) for c in cluster],
        "isSystem": True,
        # Auto-clustered ranges are eligible for the whole-screen "grow the
        # active window" reshape (`window_resolver`). Manual zooms are not,
        # unless materialized from an auto proposal (see `_normalize_manual`).
        "screenAuto": True,
        "holdOut": hold_out,
    }


def _normalize_manual(manual_zooms, P):
    """Convert edits.json `zooms` entries to internal range shape."""
    out = []
    for m in (manual_zooms or []):
        s = m.get("start")
        e = m.get("end")
        if s is None or e is None:
            continue
        s = float(s); e = float(e)
        if e <= s:
            continue
        level = m.get("level")
        z = float(level) if level is not None else P.max_zoom
        pin_x = m.get("x")
        pin_y = m.get("y")
        r = {
            "startTime": s,
            "endTime": e,
            "zoom": z,
            "type": "fixed" if pin_x is not None else "follow-click-groups",
            "clicks": [],
            "isSystem": False,
            # A materialized auto-zoom (edits.auto_zoom_proposals stamps
            # `auto: True`) stays eligible for the whole-screen window reshape
            # even though it rides the manual pool; a user-drawn zoom does not.
            "screenAuto": bool(m.get("auto")),
        }
        if pin_x is not None:
            r["x"] = float(pin_x)
            r["y"] = float(pin_y) if pin_y is not None else 0.0
        out.append(r)
    return out


def _range_overlaps_suppress(rng, suppressed):
    for sp in suppressed:
        s = float(sp.get("start", 0.0))
        e = float(sp.get("end", 0.0))
        if e <= s:
            continue
        if s <= rng["startTime"] and e >= rng["endTime"]:
            return True
    return False


def _range_is_auto(r):
    """Is this range eligible for the whole-screen window reshape?

    True for auto-clustered ranges and materialized auto-zooms, False for
    user-drawn manual zooms. `screenAuto` is stamped by every creator; the
    fallback keeps ranges built directly in tests (which set only `isSystem`)
    behaving sensibly.
    """
    if "screenAuto" in r:
        return bool(r["screenAuto"])
    return bool(r.get("isSystem", True))


def _merge_ranges(ranges):
    """Merge overlapping ranges. Manual overrides win at overlaps
    (zoom level, pin target). Auto ranges' click lists are pooled so
    the follower still tracks each click in the merged span."""
    if not ranges:
        return []
    ranges = sorted((dict(r) for r in ranges), key=lambda r: r["startTime"])
    merged = [dict(ranges[0])]
    merged[0]["screenAuto"] = _range_is_auto(ranges[0])
    for r in ranges[1:]:
        prev = merged[-1]
        if r["startTime"] <= prev["endTime"] + 1e-6:
            prev["endTime"] = max(prev["endTime"], r["endTime"])
            # Pool clicks from both
            prev_clicks = prev.get("clicks", []) or []
            r_clicks = r.get("clicks", []) or []
            all_c = sorted({(round(c[0], 6), c[1], c[2])
                            for c in (prev_clicks + r_clicks)})
            prev["clicks"] = [tuple(c) for c in all_c]
            # A merged span is auto-reshape-eligible only if EVERY contributor
            # was -- so a user-drawn zoom overlapping an auto one protects the
            # whole merged range from the window reshape.
            prev["screenAuto"] = bool(prev.get("screenAuto")) and _range_is_auto(r)
            # Manual wins for zoom/type/target
            if not r.get("isSystem", True):
                prev["zoom"] = r["zoom"]
                prev["type"] = r.get("type", prev.get("type"))
                if "x" in r:
                    prev["x"] = r["x"]
                    prev["y"] = r.get("y", 0.0)
                prev["isSystem"] = False
        else:
            nr = dict(r)
            nr["screenAuto"] = _range_is_auto(r)
            merged.append(nr)
    return merged


def plan_zoom(clicks, W, H, T_end, P, manual=None, suppressed=None,
              auto_cluster=True, win_w=None, win_h=None,
              window_resolver=None):
    """Return a list of zoom ranges (auto + manual, merged).

    auto_cluster=True (default): also cluster the click stream into
    proposed ranges. Callers who already materialize auto zooms into
    their edits doc (studio_app, MCP) should pass False so the same
    clicks don't produce doubled ranges. CLI-only renders that don't
    load edits.json rely on auto_cluster=True.

    win_w/win_h are the zoom-1.0 crop window (default W/H); they drive
    the overview-vs-follow framing decision so a non-source aspect export
    reasons about what's actually visible.

    `window_resolver` (whole-screen "grow the active window") is a
    render-side closure that MAY reshape an auto follow range whose clicks
    are owned by one on-screen window into a fixed window-anchored target.
    It runs after the click lists are populated and BEFORE `_apply_overview`
    (which then skips the now-`fixed` range via its existing type guard --
    the two are mutually exclusive per range). `None` (every caller that
    doesn't opt in) leaves the ranges byte-identical to before.
    """
    suppressed = list(suppressed or [])
    if suppressed:
        clicks = [c for c in clicks if not _in_suppressed(c[0], suppressed)]
    auto = []
    if auto_cluster:
        for cl in cluster_clicks(sorted(clicks), P.chain_gap):
            r = cluster_to_range(cl, P, T_end)
            if r is not None:
                auto.append(r)
    manual_ranges = _normalize_manual(manual, P)
    if suppressed:
        manual_ranges = [r for r in manual_ranges
                         if not _range_overlaps_suppress(r, suppressed)]
    merged = _merge_ranges(auto + manual_ranges)
    # For any follow-click-groups range that doesn't carry its own
    # click list (e.g. a materialized range loaded from edits.json),
    # populate one from the current click stream so the runtime
    # follower still tracks each click in the merged span.
    clicks_sorted = sorted(clicks)
    for r in merged:
        if r.get("type") == "follow-click-groups" and not r.get("clicks"):
            r["clicks"] = [(t, x, y) for (t, x, y) in clicks_sorted
                           if r["startTime"] <= t <= r["endTime"]]
    # Whole-screen window reshape, before overview: an eligible follow range
    # may become a `fixed` window-anchored target here, which `_apply_overview`
    # then leaves alone (it only touches follow ranges).
    if window_resolver is not None:
        window_resolver(merged)
    _apply_overview(merged, W, H,
                    win_w if win_w is not None else W,
                    win_h if win_h is not None else H, P)
    return merged


def _apply_overview(ranges, W, H, win_w, win_h, P):
    """Convert follow ranges whose clicks are too spread to hold at full
    zoom into a still, lower-zoom overview locked on the activity center.

    Runs after merge so it sees each range's FINAL pooled click list, and
    at render time -- so it reasons about the actual `win_w`/`win_h` and
    stays out of the stored edits (a vertical export re-decides for its
    own window). Applies to ANY follow-click-groups range, auto or
    manual: it only ever *reduces* zoom to fit and never exceeds the
    range's own level, so it's a pure anti-whip measure, not an override
    of the user's ceiling. Manual PINNED ranges (type=fixed) are an
    explicit target and are left untouched. `overview=False` is the
    bit-exact escape hatch back to always-full-zoom chasing.
    """
    if not P.overview:
        return
    fill = P.overview_fill
    for r in ranges:
        if r.get("type") != "follow-click-groups":
            continue
        clks = r.get("clicks") or []
        if len(clks) < 2:
            continue
        xs = [float(c[1]) for c in clks]
        ys = [float(c[2]) for c in clks]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        z = max(1.0, float(r["zoom"]))
        # Zoom at which the bbox (plus margin) exactly fills the window.
        z_fit = min(win_w * fill / bw if bw > 1.0 else 1e9,
                    win_h * fill / bh if bh > 1.0 else 1e9)
        if z_fit >= z:
            continue  # already fits at full zoom -- keep the tight follow
        # Scattered activity: still overview. Fit the bbox (floored so we
        # never zoom out to nothing) and lock dead-center on it.
        r["zoom"] = max(P.overview_zoom_min, min(z, z_fit))
        r["type"] = "fixed"
        r["x"] = (min(xs) + max(xs)) / 2.0
        r["y"] = (min(ys) + max(ys)) / 2.0
        r["overview"] = True


# -- Runtime (per-frame) -------------------------------------------------

def _target_for_range(rng, t, W, H, win_w, win_h, snap_ratio):
    kind = rng.get("type", "follow-click-groups")
    if kind == "fixed" and "x" in rng:
        tx, ty = float(rng["x"]), float(rng["y"])
    else:
        clicks = rng.get("clicks", []) or []
        if not clicks:
            tx, ty = W / 2.0, H / 2.0
        else:
            done = [c for c in clicks if c[0] <= t]
            src = done[-1] if done else clicks[0]
            tx, ty = float(src[1]), float(src[2])

    z = max(1.0, float(rng["zoom"]))
    half_w = (win_w / z) / 2.0
    half_h = (win_h / z) / 2.0
    lo_x, hi_x = half_w, W - half_w
    lo_y, hi_y = half_h, H - half_h

    if lo_x > hi_x:
        tx = W / 2.0
    else:
        tx = max(lo_x, min(hi_x, tx))
        if snap_ratio > 0.0:
            edge = snap_ratio * half_w
            if tx - lo_x < edge:
                tx = lo_x
            elif hi_x - tx < edge:
                tx = hi_x
    if lo_y > hi_y:
        ty = H / 2.0
    else:
        ty = max(lo_y, min(hi_y, ty))
        if snap_ratio > 0.0:
            edge = snap_ratio * half_h
            if ty - lo_y < edge:
                ty = lo_y
            elif hi_y - ty < edge:
                ty = hi_y
    return tx, ty


def _active_range(ranges, t):
    for r in ranges:
        if r["startTime"] <= t <= r["endTime"]:
            return r
        if r["startTime"] > t:
            break
    return None


def simulate_path(frame_times, ranges, W, H, win_w, win_h, P,
                  initial_center=None):
    T = len(frame_times)
    if T == 0:
        return np.zeros((0, 3))
    cx = float(initial_center[0]) if initial_center else W / 2.0
    cy = float(initial_center[1]) if initial_center else H / 2.0
    vx = vy = 0.0
    z = 1.0
    vz = 0.0
    out = np.zeros((T, 3))
    last_reached_z = 1.0
    prev_t = float(frame_times[0])
    for i in range(T):
        t = float(frame_times[i])
        dt = t - prev_t if i > 0 else 0.0
        rng = _active_range(ranges, t)
        if rng is not None:
            tx, ty = _target_for_range(rng, t, W, H, win_w, win_h, P.snap_ratio)
            z_target = float(rng["zoom"])
            if z_target > last_reached_z:
                last_reached_z = z_target
        else:
            if P.always_zoomed and last_reached_z > 1.05:
                # Hold last-reached zoom + freeze at current position.
                tx, ty = cx, cy
                z_target = last_reached_z
            else:
                tx, ty = W / 2.0, H / 2.0
                z_target = 1.0
        cx, vx = _damped_step(cx, vx, tx, P.omega, P.zeta, dt)
        cy, vy = _damped_step(cy, vy, ty, P.omega, P.zeta, dt)
        z, vz = _damped_step(z, vz, z_target, P.z_omega, P.z_zeta, dt)
        # Clamp the zoom to [1.0, max_zoom] with velocity anti-windup.
        # With zeta ~ 0.94 the spring is slightly under-damped and would
        # otherwise overshoot the ceiling by ~0.001, which trips
        # tests expecting a hard band.
        if z < 1.0:
            z = 1.0
            vz = max(0.0, vz)
        elif z > P.max_zoom:
            z = P.max_zoom
            vz = min(0.0, vz)
        out[i, 0] = cx
        out[i, 1] = cy
        out[i, 2] = z
        prev_t = t
    return out


# -- Public entry point (source-compat with the legacy signature) --------

def build_path(frame_times, W, H,
               clicks_t, clicks_x, clicks_y,
               moves_t, moves_x, moves_y,
               max_zoom=2.0, fps=60.0, params=None,
               manual_zooms=None, suppressed_ranges=None,
               plan_duration=None, win_w=None, win_h=None,
               window_resolver=None,
               # Legacy kwargs accepted for source-compat with render.py.
               # The new model does not consume them.
               keys_t=None, ups_t=None, typing_anchors=None,
               scrolls_t=None, scrolls_x=None, scrolls_y=None):
    # `manual_zooms=None` (the default; used by the CLI which doesn't
    # load edits.json) means "please auto-cluster clicks". An empty list
    # from an authoritative caller (studio_app, MCP) suppresses that so
    # already-materialized zooms don't get duplicated.
    auto_cluster = manual_zooms is None
    manual_zooms = list(manual_zooms or [])
    suppressed_ranges = list(suppressed_ranges or [])
    manual_levels = [float(m.get("level")) for m in manual_zooms
                     if m.get("level") is not None]
    eff_max_zoom = max([max_zoom] + manual_levels) if manual_levels else max_zoom
    P = build_params(eff_max_zoom, params)

    T = len(frame_times)
    if T == 0:
        return np.zeros((0, 3))

    win_w = float(win_w) if win_w is not None else float(W)
    win_h = float(win_h) if win_h is not None else float(H)
    T_end = float(plan_duration) if plan_duration is not None else float(frame_times[-1])

    clicks = sorted((float(clicks_t[i]),
                     float(clicks_x[i]),
                     float(clicks_y[i]))
                    for i in range(len(clicks_t)))
    ranges = plan_zoom(clicks, W, H, T_end, P,
                       manual=manual_zooms,
                       suppressed=suppressed_ranges,
                       auto_cluster=auto_cluster,
                       win_w=win_w, win_h=win_h,
                       window_resolver=window_resolver)

    # The camera starts every clip at the source center -- the cursor
    # is decorative, never a camera anchor by itself.
    return simulate_path(frame_times, ranges, W, H, win_w, win_h, P,
                         initial_center=(W / 2.0, H / 2.0))


def _cluster_clicks_by_owner(owned, chain_gap):
    """Cluster `(t, x, y, owner)` clicks, breaking on a CHANGE OF OWNER as
    well as on `chain_gap`. Returns `[(owner, [(t, x, y), ...]), ...]`.

    This is what makes multi-card zoom track attention instead of collapsing.
    Clustering each card's clicks in isolation is wrong the moment the user
    works in two windows at once: `chain_gap` is 4.0s, so alternating every
    1.5s chains EACH card's clicks into one cluster spanning the whole take,
    and the two ranges then cover each other end to end. Arbitration can only
    resolve that by silencing one card for the entire recording -- which is
    what it did, measured: 12 clicks in a card produced zero zoom, and a
    3-card round robin left exactly one surviving range out of 36.

    Breaking on the owner turns the same stream into the alternating short
    ranges the user actually performed, and arbitration is then only trimming
    `pre_roll` overlaps at the seams.
    """
    if not owned:
        return []
    out = [(owned[0][3], [owned[0][:3]])]
    for c in owned[1:]:
        owner, last = out[-1][0], out[-1][1][-1]
        if c[3] == owner and c[0] - last[0] <= chain_gap:
            out[-1][1].append(c[:3])
        else:
            out.append((c[3], [c[:3]]))
    return out


def _arbitrate_card_ranges(per_card):
    """Given [(card_index, ranges)], return the same lists with cross-card
    overlaps removed so AT MOST ONE card is zoomed at any instant.

    The rule is *the most recent activity wins*: ranges are swept in start
    order, and when a range belonging to a different card opens while another
    is still running, the running one is TRUNCATED to end there. It is not a
    hard cut -- `simulate_path` finds no active range for the loser from that
    moment and its spring eases back to 1.0/centre, which is the same
    zoom-out arc a range ending naturally produces.

    Preemption is deliberately by START, and a range starts `pre_roll` (2.5s)
    BEFORE its first click, so the outgoing card begins settling while the
    user is still travelling toward the next window rather than after they
    have arrived. That is the same lead the single-camera model uses to make
    a zoom read as intentional rather than reactive.

    Why arbitrate at all: N cards zooming at once is exactly the whip-pan
    motion sickness this module's whole two-pass structure exists to prevent
    (see `_apply_overview`). One subject at a time is the framing decision,
    made at plan time on click geometry -- not a per-frame heuristic.
    """
    ordered = sorted(
        ((float(r["startTime"]), ci, r)
         for ci, ranges in per_card for r in ranges),
        key=lambda item: item[0])
    kept = [[] for _ in range(len(per_card))]
    active = None          # (card_index, range) currently holding the screen
    for start, ci, rng in ordered:
        if active is not None:
            a_ci, a_rng = active
            if a_ci != ci and float(a_rng["endTime"]) > start:
                a_rng["endTime"] = start
                clicks = a_rng.get("clicks") or []
                first_click = float(clicks[0][0]) if clicks else None
                # Drop rather than truncate when the subject never arrived:
                # a range is `pre_roll` (2.5s) of lead before its first click,
                # so one preempted inside that lead is a card that started
                # easing toward something the user then didn't do. Truncating
                # it leaves the spring mid-rise with upward velocity, and you
                # see two cards briefly moving at once -- the exact thing this
                # arbitration exists to prevent. Measured: dropping these takes
                # "both cards climbing" from 23 frames to 0.
                dead = float(a_rng["endTime"]) <= float(a_rng["startTime"])
                stillborn = first_click is not None and first_click >= start
                if dead or stillborn:
                    kept[a_ci].remove(a_rng)
                else:
                    # It did zoom to a real click, so let it play its ease-out
                    # instead of holding to the end of the clip.
                    a_rng["holdOut"] = False
        kept[ci].append(rng)
        active = (ci, rng)
    return kept


def build_card_paths(frame_times, cards, max_zoom=2.0, params=None,
                     suppressed_ranges=None, plan_duration=None):
    """One camera path per multi-window card, arbitrated so only the card
    with the most recent activity is zoomed.

    Multi-window mode composites N static crops and therefore has no single
    camera window -- which is why the camera was skipped there entirely. But
    a card IS a source rect with its own output cell, i.e. exactly the
    (W, H) + (win_w, win_h) pair this module already plans for; record-time
    window capture proved that by rebinding those to a window and keeping
    auto-zoom alive. So each card gets a real plan/simulate pass in ITS OWN
    coordinate space, and the only new idea is `_arbitrate_card_ranges`.

    `cards` is a list of dicts with `w`/`h` (the card's source rect size),
    `win_w`/`win_h` (its zoom-1.0 window, i.e. the cell aspect fitted into
    that rect), `clicks` as (t, x, y) triples ALREADY translated into card
    space, and optional `manual` zoom ranges. Returns one (T, 3) array per
    card, in the same order.

    Cards with no ranges of their own come back as a flat zoom-1.0 path, so a
    caller can cheaply detect "this card never zooms" and keep compositing it
    the untouched way.
    """
    T = len(frame_times)
    if not cards:
        return []
    if T == 0:
        return [np.zeros((0, 3)) for _ in cards]
    suppressed_ranges = list(suppressed_ranges or [])
    T_end = (float(plan_duration) if plan_duration is not None
             else float(frame_times[-1]))

    # `always_zoomed` is a WHOLE-CAMERA idea ("hold the last zoom to the end")
    # and does not compose with N of them: honoured per card it makes every
    # card that ever zoomed hold forever, so every card ends up zoomed at once
    # and the arbitration below is defeated (measured: 404 of 600 frames with
    # both cards at 2.0x). Cards hold their framing by NOT zooming, which is
    # the same end state, so it is dropped here rather than fought.
    if params and params.get("always_zoomed"):
        params = dict(params)
        params.pop("always_zoomed", None)

    # One clustering across ALL cards, broken on the owner, so alternating
    # attention becomes alternating ranges instead of two take-length ones.
    owned = []
    for ci, card in enumerate(cards):
        for t, x, y in (card.get("clicks") or []):
            owned.append((float(t), float(x), float(y), ci))
    owned.sort(key=lambda c: c[0])
    chain_gap = build_params(max_zoom, params).chain_gap
    grouped = _cluster_clicks_by_owner(owned, chain_gap)
    per_owner = {}
    for owner, cl in grouped:
        per_owner.setdefault(owner, []).append(cl)

    # Plan, arbitrate, THEN merge -- the order is load-bearing. Merging first
    # (what `plan_zoom` does in one shot) unions each card's own segments back
    # into a single take-length range, because continuous work in one window
    # legitimately chains: 12 clicks 3s apart become ranges that abut, and
    # `_merge_ranges` is right to fuse them. But two cards each holding a
    # take-length range cover each other end to end, and arbitration can then
    # only silence one card for the whole recording -- measured before this
    # ordering was fixed. Arbitrating the per-segment ranges first cuts the
    # timeline into who-owns-when, and the merge afterwards can only fuse a
    # card's own adjacent spans, which is exactly what it should do.
    per_card, planned = [], []
    for ci, card in enumerate(cards):
        manual = list(card.get("manual") or [])
        levels = [float(m.get("level")) for m in manual
                  if m.get("level") is not None]
        eff_max_zoom = max([max_zoom] + levels) if levels else max_zoom
        P = build_params(eff_max_zoom, params)
        W = float(card["w"])
        H = float(card["h"])
        win_w = float(card.get("win_w") or W)
        win_h = float(card.get("win_h") or H)
        segs = []
        for cl in per_owner.get(ci, []):
            if suppressed_ranges:
                cl = [c for c in cl
                      if not _in_suppressed(c[0], suppressed_ranges)]
                if not cl:
                    continue
            r = cluster_to_range(cl, P, T_end)
            if r is not None:
                segs.append(r)
        planned.append((P, W, H, win_w, win_h, manual))
        per_card.append((ci, segs))

    kept = _arbitrate_card_ranges(per_card)

    out = []
    for ci, card in enumerate(cards):
        P, W, H, win_w, win_h, manual = planned[ci]
        clicks = sorted((float(t), float(x), float(y))
                        for t, x, y in (card.get("clicks") or []))
        # plan_zoom's tail, on the arbitrated segments: manual ranges join the
        # pool, overlapping spans of THIS card fuse, follow ranges get their
        # click list, and overview re-decides the framing per card.
        manual_ranges = _normalize_manual(manual, P)
        if suppressed_ranges:
            manual_ranges = [r for r in manual_ranges
                             if not _range_overlaps_suppress(r,
                                                             suppressed_ranges)]
        merged = _merge_ranges(kept[ci] + manual_ranges)
        for r in merged:
            if r.get("type") == "follow-click-groups" and not r.get("clicks"):
                r["clicks"] = [(t, x, y) for (t, x, y) in clicks
                               if r["startTime"] <= t <= r["endTime"]]
        _apply_overview(merged, W, H, win_w, win_h, P)
        out.append(simulate_path(frame_times, merged, W, H, win_w, win_h, P,
                                 initial_center=(W / 2.0, H / 2.0)))
    return out


def _focus_range(start, end, card, level, amount):
    """One span of the focus plan: which card is the subject, when, how far.

    `amount` is the 0..1 emphasis the layout blends by, NOT a zoom level --
    window focus re-weights the composition (the subject's card grows, the
    others shrink toward an edge) rather than pushing a camera into the
    grid. `level` is the symbolic name of that amount, carried so the same
    plan can be handed to the editor as objects rather than numbers.
    """
    return {
        "startTime": float(start),
        "endTime": float(end),
        "card": int(card),
        "level": level,
        "amount": float(amount),
    }


def _focus_amounts(P):
    """`(stage1, stage2)` emphasis. Stage 2 is 1.0 by definition -- it IS the
    focused layout -- so only the lean is a tunable."""
    return max(0.0, min(1.0, P.focus_lean)), 1.0


def _split_at_second_click(rng, click_times, ci, a1, a2, P):
    """The escalation: one span in, one or two out.

    Stage 1 holds from the range's `pre_roll` lead until the card's SECOND
    surviving click, and stage 2 owns the rest. Two abutting spans rather
    than a ramp inside one, because the emphasis spring already turns a
    changed target into a smooth chase -- so the escalation is a continuous
    re-weighting, and the split costs nothing but the right `endTime`.

    Two ways a span declines to escalate and stays at stage 1:

    - **Its second click never arrives** inside the surviving span (a lone
      click, or a cluster arbitration truncated before the escalation). The
      user only ever committed to that window once.
    - **Stage 2 wouldn't have room to play.** A cluster whose 2nd click is
      also its LAST leaves only `tail` (0.5s) before the ease-out, and the
      spring needs longer than that just to rise -- so the layout would
      lurch halfway over and immediately back. Declining is the same posture
      `cluster_to_range` takes on a lone trailing click and
      `_arbitrate_card_ranges` takes on a stillborn range: when there isn't
      room to do the move properly, don't start it.
    """
    start, end = float(rng["startTime"]), float(rng["endTime"])
    inside = [t for t in click_times if start <= t <= end]
    if (len(inside) >= 2 and start < inside[1] < end
            and (end - inside[1]) >= P.focus_full_min):
        t2 = inside[1]
        return [_focus_range(start, t2, ci, "focus", a1),
                _focus_range(t2, end, ci, "full", a2)]
    return [_focus_range(start, end, ci, "focus", a1)]


def _normalize_focus_manual(manual, n_cards, P):
    """Editor-authored focus spans (`edits.focus`) -> internal spans.

    `level` is `"focus"` / `"full"`, or a raw 0..1 number for a hand-set
    amount. Overlaps are resolved the same way `_arbitrate_card_ranges`
    resolves them -- the later span truncates the running one -- so the
    composition still only ever has one subject.
    """
    a1, a2 = _focus_amounts(P)
    out = []
    for m in (manual or []):
        s, e = m.get("start"), m.get("end")
        if s is None or e is None:
            continue
        s, e = float(s), float(e)
        if e <= s:
            continue
        try:
            ci = int(m.get("card"))
        except (TypeError, ValueError):
            continue
        if not (0 <= ci < n_cards):
            continue
        level = m.get("level")
        if level is None or level == "focus":
            amount = a1
        elif level == "full":
            amount = a2
        else:
            try:
                amount = max(0.0, min(1.0, float(level)))
            except (TypeError, ValueError):
                level, amount = "focus", a1
        out.append(_focus_range(s, e, ci, level, amount))
    out.sort(key=lambda r: r["startTime"])
    kept = []
    for r in out:
        while kept and r["startTime"] < kept[-1]["endTime"]:
            kept[-1]["endTime"] = r["startTime"]
            if kept[-1]["endTime"] > kept[-1]["startTime"]:
                break
            kept.pop()
        kept.append(r)
    return kept


def _plan_focus_auto(cards, P, T_end, suppressed_ranges):
    """Auto-planned focus spans from the cards' click times.

    Same three-step order `build_card_paths` documents: cluster across ALL
    cards breaking on the owner, arbitrate so one card owns the composition
    at a time, and only then merge each card's own spans. The arbitration is
    the whole reason this composes -- it already knows "most recent activity
    wins", with a `pre_roll` lead so the outgoing card starts giving up the
    frame while the user is still travelling toward the next window.
    """
    a1, a2 = _focus_amounts(P)
    owned = []
    for ci, card in enumerate(cards):
        for t in (card.get("click_times") or []):
            t = float(t)
            if suppressed_ranges and _in_suppressed(t, suppressed_ranges):
                continue
            # x/y are unused here -- the subject of this plan is the CARD,
            # so only the time and the owner matter.
            owned.append((t, 0.0, 0.0, ci))
    owned.sort(key=lambda c: c[0])
    if not owned:
        return []
    per_owner = {}
    for owner, cl in _cluster_clicks_by_owner(owned, P.chain_gap):
        per_owner.setdefault(owner, []).append(cl)
    per_card = []
    for ci in range(len(cards)):
        segs = []
        for cl in per_owner.get(ci, []):
            r = cluster_to_range(cl, P, T_end)
            if r is None:
                continue
            # A cluster with a second click is going to ask for stage 2, so
            # give it room to actually play. `tail` (0.5s) is tuned for a
            # zoom easing out, not for a whole second move starting. Applied
            # BEFORE arbitration on purpose: the extension is a request, and
            # a later card claiming the screen still truncates it, so this
            # can never hold a window past the point attention left it.
            if len(cl) >= 2:
                r["endTime"] = min(float(T_end),
                                   max(float(r["endTime"]),
                                       float(cl[1][0]) + P.focus_hold))
            segs.append(r)
        per_card.append((ci, segs))
    kept = _arbitrate_card_ranges(per_card)
    spans = []
    for ci in range(len(cards)):
        for r in _merge_ranges(kept[ci]):
            times = sorted(float(c[0]) for c in (r.get("clicks") or []))
            spans.extend(_split_at_second_click(r, times, ci, a1, a2, P))
    spans.sort(key=lambda r: r["startTime"])
    return spans


def plan_focus_ranges(cards, max_zoom=2.0, params=None,
                      suppressed_ranges=None, plan_duration=None):
    """The focus plan as EDITABLE objects: `[{start, end, card, level}, ...]`.

    What `edits.focus` is materialized from, so every move the composition
    would make is a real timeline object the user can retime, retarget to
    another card, re-level or delete -- rather than an opaque track they can
    only accept or switch off wholesale.

    `level` stays symbolic ("focus"/"full") rather than a number, because
    what those mean is decided by `framing.focus_placements` against the
    live canvas: the same span still means "make this the subject" after a
    card is added, a window is resized, or the export aspect changes.
    """
    if not cards:
        return []
    P = build_params(max_zoom, params)
    T_end = float(plan_duration) if plan_duration is not None else 0.0
    spans = _plan_focus_auto(cards, P, T_end, list(suppressed_ranges or []))
    return [{"start": r["startTime"], "end": r["endTime"],
             "card": r["card"], "level": r["level"]} for r in spans]


def build_focus_emphasis(frame_times, cards, max_zoom=2.0, params=None,
                         suppressed_ranges=None, plan_duration=None,
                         manual=None):
    """Per-card emphasis over time -- a `(T, N)` array in 0..1, or None.

    This is "focus the window I'm working in": click in a window and the
    composition re-weights toward that card, click again and it becomes the
    clear subject while the others shrink to an edge. Emphasis is fed to
    `framing.focus_placements` + `blend_placements`, so what actually moves
    is the LAYOUT.

    Deliberately not a camera over the composed grid. A camera magnifies
    whatever the grid put in the middle and crops the subject at the frame
    edge -- you end up seeing *less* of the window you just focused, which
    is backwards. Re-laying-out keeps every card aspect-fit and uncropped.

    `cards` is a list of `{"click_times": [t, ...]}`. No geometry: where the
    cards sit is `framing`'s business, and keeping it out of here is what
    lets one plan drive the export and the editor's compositor alike.

    `manual` follows `build_path`'s convention -- None means "auto-plan from
    the clicks", while a list (even an empty one) is authoritative and
    suppresses auto-planning, so a caller that has already materialized the
    plan into `edits.focus` doesn't get it twice.

    Returns None when nothing is ever emphasized -- the signal to composite
    exactly the way the un-focused path does, not merely an optimization.
    """
    T, N = len(frame_times), len(cards or [])
    if not N or T == 0:
        return None
    P = build_params(max_zoom, params)
    T_end = (float(plan_duration) if plan_duration is not None
             else float(frame_times[-1]))
    suppressed_ranges = list(suppressed_ranges or [])

    if manual is not None:
        spans = _normalize_focus_manual(manual, N, P)
        if suppressed_ranges:
            spans = [r for r in spans
                     if not _range_overlaps_suppress(r, suppressed_ranges)]
    else:
        spans = _plan_focus_auto(cards, P, T_end, suppressed_ranges)
    if not spans or max(r["amount"] for r in spans) <= 1e-6:
        return None

    # One spring per card, chasing its own target. Same closed-form step the
    # zoom camera uses, so emphasis accelerates and settles like every other
    # motion in the export instead of reading as a linear slide.
    out = np.zeros((T, N))
    e = [0.0] * N
    v = [0.0] * N
    prev_t = float(frame_times[0])
    for i in range(T):
        t = float(frame_times[i])
        dt = t - prev_t if i > 0 else 0.0
        rng = _active_range(spans, t)
        for ci in range(N):
            target = (rng["amount"]
                      if rng is not None and rng["card"] == ci else 0.0)
            e[ci], v[ci] = _damped_step(e[ci], v[ci], target,
                                        P.z_omega, P.z_zeta, dt)
            if e[ci] < 0.0:
                e[ci], v[ci] = 0.0, max(0.0, v[ci])
            elif e[ci] > 1.0:
                e[ci], v[ci] = 1.0, min(0.0, v[ci])
            out[i, ci] = e[ci]
        prev_t = t
    return out


def typing_bursts(keys_t, max_zoom=2.0, params=None, suppressed_ranges=None):
    """Legacy shim: the new model doesn't auto-zoom on typing bursts,
    so no bursts are ever produced. render.py's typing-anchor cache
    layer collapses to a no-op when this returns []."""
    return []
