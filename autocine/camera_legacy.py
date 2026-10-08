"""The auto-zoom virtual camera (two-pass model).

PASS 1 (offline, `plan_zoom`): chain clicks into focus segments and bake a
single quintic-smootherstep ZOOM envelope with lookahead — snappy zoom-in,
2x-gentler zoom-out, a small pre-roll so the zoom leads the click, and an
OVERVIEW fallback that fits scattered rapid clicks instead of chasing them.
Planning offline means we never zoom out then straight back in (the biggest
source of seasickness) — it's eliminated structurally.

PASS 2 (per-frame, `Camera.update`): the CENTER is a reactive follower. The
cursor is de-jittered (zoom-scaled dead-zone + EMA), held inside a comfort box
(no pan for micro/medium motion), given a small predictive lead, then eased by
an EXACT closed-form critically-damped spring (overshoot-free, frame-rate
independent). Target and result are clamped to the zoom's safe band with
velocity anti-windup so the camera never fights an edge.

`build_path` wraps both passes behind the renderer's stable interface: it
returns a per-frame (T, 3) array of (center_x, center_y, zoom) in pixels.

Design distilled by a multi-agent design panel (spring vs. keyframe vs.
comfort-hybrid) into this synthesis.
"""

from math import exp, hypot

import numpy as np

# Zoom transition speed presets: a single factor scaling the three
# transition timings (zoom_in_dur / zoom_out_dur / pre_roll) together, so
# the snappy-in/gentle-out ratio and the anticipation grammar are preserved.
# Holds (hold_after, min_hold, typing_hold_after) are dwell time, not
# transition time, and deliberately don't scale. Note the merge horizon
# (t_he + zoom_out_dur) scales along with it: slower zooms merge more
# aggressively, which is exactly what prevents them from bouncing.
SPEED_FACTORS = {"slow": 1.5, "normal": 1.0, "fast": 0.65}

DEFAULTS = dict(
    max_zoom=2.0,          # tunable 1.6-2.4
    zoom_in_dur=0.45,      # s, snappy quintic zoom-in
    zoom_out_dur=0.90,     # s, gentle quintic zoom-out (2x the in)
    pre_roll=0.12,         # s, start zoom-in before the click (anticipation)
    zoom_speed="normal",   # transition speed preset (see SPEED_FACTORS)
    hold_after=0.80,       # s, dwell after last activity before zoom-out
    min_hold=0.60,         # s, floor on hold so single clicks aren't twitchy
    chain_gap=1.20,        # s, clicks within this join one segment
    rapid_gap=0.40,        # s, mean inter-click gap to count as a flurry
    overview_min_clicks=4,
    overview_span_frac=0.50,   # click bbox must exceed this frac of W/H
    overview_zoom_min=1.25,
    overview_margin_frac=0.08,  # bbox padding (fraction of W/H)
    move_speed_on=220.0,   # px/s, fast-move keep-alive threshold
    zoom_on_fast_move=False,
    always_zoomed=False,   # hold the last-reached zoom instead of settling to 1.0
    drag_hold=True,        # a down->up drag holds the zoom for its whole span
    drag_max_hold=30.0,    # s, safety cap on one drag's keep-alive (stuck button)
    # Typing-triggered zoom (see _plan_typing_intervals; designed by a second
    # multi-agent panel). Keys arrive as bare 100ms-bucketed activity ticks.
    typing_gap=2.5,        # s, ticks within this chain into one burst
    typing_min_keys=4,     # min activity ticks for a burst (rejects chords)
    typing_min_span=0.8,   # s, min first->last tick span (rejects "ok<CR>")
    typing_hold_after=1.0, # s, dwell after last tick (the proofread beat)
    typing_zoom_frac=0.85, # level = 1 + frac*(max_zoom-1): slightly wide, anchor is inferred
    typing_anchor_max_age=8.0,  # s, newest click older than this can't anchor a burst
    typing_anchor_slack=0.5,    # s, a click landing just AFTER the burst start still anchors
    typing_bucket=0.1,     # s, recorder's coalescing bucket (== record.KEY_BUCKET_SEC)
    # Scroll-aware camera (designed by a third multi-agent panel). Scroll
    # ticks arrive rate-limited (<= ~10/s) at the cursor position -- and on
    # macOS scroll events go to the window UNDER the cursor, so that position
    # is where the scrolled content lives. Two effects, both gated so empty
    # scrolls reproduce the scroll-less plan bit-exactly: ticks extend click
    # clusters' keep-alive (reading the page you just clicked), and sustained
    # scroll-READING triggers its own gentle follow-mode interval. The
    # trigger gate is CLUMP structure, not tick count: a lone trackpad
    # flick's momentum coast is one continuous gapless event stream (pynput
    # exposes no momentum phase), so only >= scroll_min_clumps gesture
    # clumps separated by >= scroll_pause of quiet -- or one very long
    # deliberate run (scroll_long_span) -- reads as "scrolling through a
    # read". The zoom-in is anchored at the first SETTLE (first quiet gap),
    # never at peak flick velocity: zoom into settled text, not streaming
    # text. Hold is reading-scale (scroll_hold_after) so read-scroll-read
    # cadences merge into one calm hold instead of pumping.
    scroll_gap=3.5,        # s, ticks within this chain into one burst
    scroll_min_ticks=4,    # noise floor on a burst's tick count
    scroll_min_span=1.0,   # s, min first->last tick span
    scroll_pause=0.4,      # s, quiet gap splitting a burst into gesture clumps
    scroll_min_clumps=2,   # momentum gate: a lone flick coasts in ONE clump
    scroll_long_span=3.5,  # s, a single gapless run this long also qualifies
    scroll_hold_after=4.5, # s, reading dwell after the last tick
    scroll_zoom_frac=0.55, # level = 1 + frac*(max_zoom-1): reading wants context
    scroll_span_frac=0.35, # tick-bbox stability gate (frac of W/H), else skip
    omega_pan=6.5,         # rad/s, center spring (half-life ~0.258s)
    comfort_frac=0.20,     # comfort-box half-extent as frac of visible dim
    deadzone_px=3.0,       # px at zoom 1 (scaled by 1/z); kills jitter
    cursor_tau=0.06,       # s, cursor position EMA
    vel_tau=0.08,          # s, cursor velocity EMA
    lead_time=0.10,        # s, predictive lead = lead_time * cursor velocity
    max_lead_frac=0.06,    # cap lead at this frac of visible dim
    pan_cap_frac=0.60,     # max on-screen pan speed as frac of W/s
    # -- The "Focused" center controller -------------------------------------
    # Two screen-animation modes:
    #   "focused" (default): piecewise-CONSTANT anchor, event-driven glides.
    #     The camera is rock-still by construction between clicks/segment-
    #     boundaries; it only moves when authored evidence (a click) or a
    #     sustained out-of-window dwell fires a discrete retarget, and the
    #     existing critically-damped spring plays each jump as ONE decisive
    #     glide. Ordinary cursor motion inside the hold is IGNORED --
    #     killing the "just follows the cursor around" complaint at its
    #     root instead of tuning it. Drags stay coupled via the legacy
    #     branch (drag = the content).
    #   "smooth": the legacy comfort-box + predictive-lead follower, kept
    #     verbatim as the escape hatch (and for A/B). edits render.screen_anim
    #     switches at export time.
    screen_anim="focused",
    contain_frac=0.80,     # E3 recapture arms only when cursor is beyond
                           # this frac of the half-window from the anchor
                           # (i.e. in the outer 20% of the visible crop) --
                           # an order of magnitude wider than the old
                           # comfort box, so ordinary mousing never moves
                           # the camera.
    recapture_dwell=0.45,  # s, out-of-window dwell before an E3 glide fires
    dwell_speed=60.0,      # src px/s cursor-vel threshold: above == transit
                           # (idle waves / travel to a click) -> ignored;
                           # below == parked -> dwell timer accumulates.
    retarget_cooldown=1.2, # s, min spacing between non-click retargets so
                           # they never chain; clicks bypass this (authored).
    glide_max_dur=1.10,    # s, an in-flight glide's vmax is raised so it
                           # completes in <= this bound (distance-scaled cap
                           # raise) -- decisive volley re-aims, no crawl.
    arrive_eps=1.5,        # src px, arrival-snap radius: once the spring is
                           # within this AND essentially still, the center
                           # snaps to the anchor exactly (identical floats
                           # every subsequent frame -- the dead-still hold).
    recenter_frac=0.55,    # E1 skip-gate: a click whose target lies inside
                           # the central `recenter_frac` of the current
                           # window per axis is already comfortably visible
                           # -- don't retarget. Kills the form-field
                           # staircase (5 fields 150px apart at z=2 becomes
                           # ONE hold, not 5 glides).
    regrip_gap=3.5,        # s, max full-out gap between two intervals whose
                           # anchors coincide that the same-anchor bridge
                           # will span (see plan_zoom step 4 disjunct).
                           # Kills the measured "zoom out and straight back
                           # in at the same anchor" pump. 0 disables the
                           # bridge, restoring today's reducer bit-exactly.
    regrip_frac=0.05,      # anchor proximity for the bridge, frac of W.
)


class _P(object):
    """Lightweight attribute bag for parameters."""


def _params(max_zoom, overrides):
    p = _P()
    for k, v in DEFAULTS.items():
        setattr(p, k, v)
    p.max_zoom = max_zoom
    if overrides:
        for k, v in overrides.items():
            setattr(p, k, v)
    # Apply the speed preset AFTER overrides so an explicit zoom_in_dur/
    # zoom_out_dur override composes with it (scaled like everything else).
    # Unknown preset strings fall back to 1.0 (defensive, like edits.py).
    factor = SPEED_FACTORS.get(getattr(p, "zoom_speed", "normal"), 1.0)
    if factor != 1.0:
        p.zoom_in_dur *= factor
        p.zoom_out_dur *= factor
        p.pre_roll *= factor
    return p


def _clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


def quintic(u):
    """Smootherstep 6u^5 - 15u^4 + 10u^3 (C2-continuous)."""
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    return u * u * u * (u * (u * 6.0 - 15.0) + 10.0)


def cd_step(x, v, target, omega, dt):
    """Exact critically-damped spring step (no overshoot, fps-independent)."""
    y = x - target
    e = exp(-omega * dt)
    B = v + omega * y
    return target + (y + B * dt) * e, (v - omega * B * dt) * e


def _sample(track, t, idx):
    """Quintic-interpolate a (t, value) track at time t, advancing `idx`."""
    n = len(track)
    while idx + 1 < n and track[idx + 1][0] <= t:
        idx += 1
    while idx > 0 and track[idx][0] > t:
        idx -= 1
    if t <= track[0][0]:
        return track[0][1], 0
    if t >= track[-1][0]:
        return track[-1][1], n - 1
    ta, va = track[idx]
    tb, vb = track[idx + 1]
    if tb == ta:
        return vb, idx
    return va + (vb - va) * quintic((t - ta) / (tb - ta)), idx


def _range_start(r):
    return float(r.get("start", r.get("t_start", 0.0)))


def _range_end(r):
    return float(r.get("end", r.get("t_end", _range_start(r))))


def _in_any_range(t, ranges):
    for r in ranges:
        if _range_start(r) <= t < _range_end(r):
            return True
    return False


def _manual_interval(m, P):
    """Convert one edits.json manual-zoom-range dict into the same interval
    shape auto clusters produce: {t_in, t_he, level, ov, lock}.

    x/y both set -> hard-locked to that point (like the overview lock-to-point
    mode); otherwise the camera follows the cursor with the normal comfort-box
    logic for the duration of the range, same as an auto cluster would.
    """
    t_start = _range_start(m)
    t_end = _range_end(m)
    if t_end <= t_start:
        t_end = t_start + 1e-3
    level = m.get("level")
    level = P.max_zoom if level is None else _clamp(float(level), 1.0, P.max_zoom)
    x, y = m.get("x"), m.get("y")
    pinned = x is not None and y is not None
    return {
        "t_in": t_start - P.zoom_in_dur,
        "t_he": t_end,
        "level": level,
        "ov": pinned,
        "lock": (float(x), float(y)) if pinned else None,
    }


def _chain_bursts(keys, P, suppressed=None):
    """Chain key ticks into gated typing-burst spans [(first, last), ...].

    Maximal runs with inter-tick gap <= typing_gap that never chain ACROSS
    a suppressed range, then gated by typing_min_keys AND typing_min_span
    (rejects lone keys and chords). Shared by planning and by
    typing_bursts() so a renderer analyzing burst windows always sees the
    exact spans the plan will use.
    """
    suppressed = suppressed or []
    if not keys:
        return []
    runs = []
    cur = [keys[0]]
    for k in keys[1:]:
        bridges = any(_range_start(r) < k and _range_end(r) > cur[-1]
                      for r in suppressed)
        if (k - cur[-1]) <= P.typing_gap and not bridges:
            cur.append(k)
        else:
            runs.append(cur)
            cur = [k]
    runs.append(cur)
    return [(b[0], b[-1]) for b in runs
            if len(b) >= P.typing_min_keys
            and (b[-1] - b[0]) >= P.typing_min_span]


def typing_bursts(keys_t, max_zoom=2.0, params=None, suppressed_ranges=None):
    """Public: gated typing-burst spans in media time -- the windows a
    renderer should analyze for visual anchors (see autocine.vision). Applies
    the same suppression filter and gates as plan_zoom, so the spans align
    exactly with the intervals the plan will build."""
    P = _params(max_zoom, params)
    keys = sorted(float(k) for k in np.asarray(list(keys_t if keys_t is not None else [])).ravel())
    suppressed = list(suppressed_ranges or [])
    if suppressed:
        keys = [k for k in keys if not _in_any_range(k, suppressed)]
    return _chain_bursts(keys, P, suppressed)


def _match_visual_anchor(visual_anchors, first, last):
    """The visual anchor whose span matches this burst, or None."""
    if not visual_anchors:
        return None
    for va in visual_anchors:
        if not isinstance(va, dict):
            continue
        try:
            if (abs(float(va.get("start")) - first) < 0.05
                    and abs(float(va.get("end")) - last) < 0.05):
                return (float(va["x"]), float(va["y"]))
        except (TypeError, ValueError):
            continue
    return None


def _plan_typing_intervals(keys, clicks, P, suppressed=None,
                           visual_anchors=None):
    """Chain key-activity ticks into typing bursts, anchor each, and shape
    them as interval dicts for the same merge pool as click clusters (so
    all the no-bounce machinery applies for free).

    keys/clicks are POST-suppression. Two extra suppression guarantees on
    top of the key filter (the password escape hatch): a burst never chains
    ACROSS a suppressed range, and a burst's hold never extends INTO one.

    Anchoring, in priority order:
    1. VISUAL anchor (visual_anchors, from autocine.vision): where the screen
       actually changed during the burst -- direct evidence of the caret,
       immune to the type-into-an-already-focused-field failure where the
       last click points somewhere else entirely. Also rescues
       keyboard-only flows (Spotlight, Cmd+L) that have no click at all.
    2. The newest click within [first_key - typing_anchor_max_age,
       first_key + typing_anchor_slack].
    3. Neither -> the burst is skipped: wide-but-correct beats
       zoomed-but-wrong.
    """
    suppressed = suppressed or []
    out = []
    level = 1.0 + P.typing_zoom_frac * (P.max_zoom - 1.0)
    for first, last in _chain_bursts(keys, P, suppressed):
        anchor = _match_visual_anchor(visual_anchors, first, last)
        if anchor is None:
            for c in clicks:   # sorted by t; the newest qualifying click wins
                if (c[0] <= first + P.typing_anchor_slack
                        and (first - c[0]) <= P.typing_anchor_max_age):
                    anchor = (c[1], c[2])
        if anchor is None:
            continue
        t_in = first - P.pre_roll
        # The recorder floor-quantizes tick times; the true last keystroke
        # can trail its bucket by up to one bucket width.
        t_act = last + P.typing_bucket
        t_he = max(t_act + P.typing_hold_after,
                   t_in + P.zoom_in_dur + P.min_hold)
        for r in suppressed:
            rs = _range_start(r)
            if last <= rs < t_he:
                t_he = max(min(t_he, rs), t_in + P.zoom_in_dur)
        out.append({"t_in": t_in, "t_he": t_he, "level": level,
                    "ov": False, "lock": anchor,
                    "manual": False, "typing": True,
                    # Typing-mode protection starts HERE, not at whatever
                    # earlier t_in a merge produces: a drag-held cluster
                    # flowing into a typing burst must stay cursor-followed
                    # through the drag (review-confirmed defect) -- the
                    # "cursor lies" rationale only applies once typing has
                    # actually begun.
                    "typing_from": t_in,
                    # (from, activity-end) member record for the merged
                    # interval's typing-protection WINDOWS: a scroll burst
                    # beginning after t_act ends this burst's protection
                    # (see the meta emission in plan_zoom step 5).
                    "_tm": [(t_in, t_act)]})
    return out


def _chain_scroll_ticks(scrolls, P, suppressed=None):
    """Chain scroll ticks into bursts: maximal runs of (t, x, y) triples with
    inter-tick gap <= scroll_gap that never chain ACROSS a suppressed range
    (same rule as typing bursts -- the password/privacy escape hatch)."""
    suppressed = suppressed or []
    if not scrolls:
        return []
    runs = [[scrolls[0]]]
    for s in scrolls[1:]:
        bridges = any(_range_start(r) < s[0] and _range_end(r) > runs[-1][-1][0]
                      for r in suppressed)
        if (s[0] - runs[-1][-1][0]) <= P.scroll_gap and not bridges:
            runs[-1].append(s)
        else:
            runs.append([s])
    return runs


def _plan_scroll_intervals(scrolls, W, H, P, suppressed=None):
    """Chain scroll ticks into bursts, gate them down to sustained
    scroll-READS, and shape those as gentle follow-mode intervals for the
    same merge pool as click clusters.

    scrolls is POST-suppression [(t, x, y)] sorted by t (positions in source
    px -- the renderer scales points->px before calling, so the stability
    gate compares like with like against W/H).

    Follow mode (ov=False, lock=None) on purpose: macOS delivers scroll
    events to the window under the cursor, so the parked cursor IS the
    region being read, and the existing comfort-box follower centers there
    with click re-aim for free. (The centroid of the ticks is that same
    cursor position, so a lock would buy nothing -- and a cursor parked on
    a scrollbar mis-anchors either way; fixing THAT needs a visual anchor,
    a la vision.py. Deliberately v2.)

    Gates, in order (see DEFAULTS for the momentum rationale):
      - scroll_min_ticks AND scroll_min_span: noise floor.
      - >= scroll_min_clumps gesture clumps (quiet gaps >= scroll_pause),
        OR a single run spanning >= scroll_long_span: rejects the lone
        flick + momentum coast, which is always one gapless clump.
      - tick bbox within scroll_span_frac of W/H: scrolling several panes
        around the screen -> wide-but-correct beats zoomed-but-wrong.

    The zoom-in lands at the first SETTLE (the tick before the first quiet
    gap; for a single long run, the burst end): the scale ramp plays over
    settled text during the reading pause, never over streaming content.
    """
    suppressed = suppressed or []
    out = []
    level = 1.0 + P.scroll_zoom_frac * (P.max_zoom - 1.0)
    for run in _chain_scroll_ticks(scrolls, P, suppressed):
        first, last = run[0][0], run[-1][0]
        span = last - first
        if len(run) < P.scroll_min_ticks or span < P.scroll_min_span:
            continue
        settles = [run[i][0] for i in range(len(run) - 1)
                   if run[i + 1][0] - run[i][0] >= P.scroll_pause]
        if len(settles) + 1 < P.scroll_min_clumps and span < P.scroll_long_span:
            continue
        xs = [p[1] for p in run]
        ys = [p[2] for p in run]
        if (max(xs) - min(xs) > P.scroll_span_frac * W
                or max(ys) - min(ys) > P.scroll_span_frac * H):
            continue
        t_in = settles[0] if settles else last
        t_he = max(last + P.scroll_hold_after,
                   t_in + P.zoom_in_dur + P.min_hold)
        for r in suppressed:
            rs = _range_start(r)
            if last <= rs < t_he:
                t_he = max(min(t_he, rs), t_in + P.zoom_in_dur)
        out.append({"t_in": t_in, "t_he": t_he, "level": level,
                    "ov": False, "lock": None, "manual": False,
                    # The merge reducer must key on when scrolling BEGAN,
                    # not on the settle-anchored t_in: a burst whose ticks
                    # start inside the previous hold's zoom-out horizon has
                    # to fold into that hold even when its first settle
                    # lands past it -- otherwise the camera zooms fully out
                    # and straight back in WHILE the user is scrolling (the
                    # exact bounce the pool exists to prevent;
                    # review-confirmed defect). The settle-anchored t_in
                    # only shapes a FRESH zoom-in, where it belongs.
                    "merge_from": first,
                    # Burst START (not the settle): the moment the user's
                    # attention verifiably moved -- what ends a preceding
                    # typing burst's protection window on merge.
                    "_ss": [first]})
    return out


def _typing_window_active(seg, t):
    """True while typing-lock protection applies at time t within seg.

    Windows are (start, until) pairs, one per merged typing burst, with
    until=None meaning "through the hold" (the historical behavior, and
    exactly what every keys-without-scrolls plan produces). A window closed
    early (until set) hands the camera back to normal cursor-follow -- and
    a LATER window re-arms protection, so type / scroll-read / type keeps
    each burst protected without pinning the read in between.
    """
    tw = seg.get("typing_windows")
    if not tw:
        return t >= seg.get("typing_from", seg["t_in"])
    for f, u in tw:
        if f <= t and (u is None or t < u):
            return True
    return False


def _settle_tail(Z, T_end, zoom_out_dur):
    """Guarantee the zoom returns to 1.0 by T_end.

    A click near the end would otherwise leave the last frame mid-zoom; here we
    graft a gentle zoom-out ending exactly at T_end onto whatever level the
    plan is at when the settle window starts.
    """
    if _sample(Z, T_end, 0)[0] <= 1.0 + 1e-3:
        return Z
    ts = T_end - zoom_out_dur
    kept = [kf for kf in Z if kf[0] < ts]
    if not kept:
        kept = [(0.0, 1.0)]
    zs = _sample(Z, max(ts, kept[-1][0]), 0)[0]
    if ts <= kept[-1][0]:
        ts = kept[-1][0] + 1e-4
    kept.append((ts, zs))
    kept.append((T_end if T_end > ts else ts + 1e-4, 1.0))
    return kept


# --------------------------------------------------------------------------
# PASS 1 — plan the zoom envelope (offline)
# --------------------------------------------------------------------------
def plan_zoom(clicks, moves, W, H, T_end, P, manual=None, suppressed=None,
              keys=None, ups=None, typing_anchors=None, scrolls=None):
    """clicks/moves: lists of (t, x, y[, speed]) tuples, sorted by t.

    ups: optional sorted list of mouse-up times, for drag-aware holds (see
    step 2b). None/empty (or P.drag_hold False) reproduces the ups-less
    plan exactly.

    scrolls: optional sorted list of (t, x, y) scroll-tick triples (cursor
    position at scroll time, source px). They extend cluster keep-alives
    (step 2) and sustained scroll-reads become gentle follow intervals in
    the merge pool (see _plan_scroll_intervals). None/empty reproduces the
    scroll-less plan exactly.

    manual: optional list of edits.json-shaped zoom-range dicts
    ({start, end, x, y, level}) authored by hand in the editor. Each becomes
    an interval alongside the auto-detected click clusters (see step 4) so a
    manual range that overlaps an auto cluster merges into one continuous
    hold exactly like two close auto clusters do -- no special-casing needed.

    suppressed: optional list of {start, end} ranges. Any click whose time
    falls inside one is dropped before clustering, so it can never trigger an
    auto-zoom (e.g. to keep the camera pulled back while typing a password).

    keys: optional sorted list of key-activity tick times (bare floats, no
    key identity -- see record.KEY_BUCKET_SEC). Typing bursts become anchored
    `typing` intervals in the same merge pool (see _plan_typing_intervals);
    None/empty reproduces the keys-less plan exactly.

    Returns (Z, meta): Z is a list of (t, zoom) quintic keyframes; meta is a
    list of per-segment dicts {t_in, t_he, t_oe, ov, lock, typing}.
    """
    manual = manual or []
    suppressed = suppressed or []
    keys = list(keys or [])
    ups = list(ups or [])
    scrolls = list(scrolls or [])
    if suppressed:
        clicks = [c for c in clicks if not _in_any_range(c[0], suppressed)]
        keys = [k for k in keys if not _in_any_range(k, suppressed)]
        scrolls = [s for s in scrolls if not _in_any_range(s[0], suppressed)]

    # 1) Chain clicks into clusters by inter-click gap.
    clusters = []
    for c in clicks:
        ct, cx, cy = c[0], c[1], c[2]
        if clusters and ct - clusters[-1]["t_last"] <= P.chain_gap:
            k = clusters[-1]
            k["pts"].append((ct, cx, cy))
            k["t_last"] = ct
        else:
            clusters.append({"pts": [(ct, cx, cy)], "t0": ct, "t_last": ct})

    # 2) Fast cursor moves extend a cluster's keep-alive time (drags/reading);
    #    scroll ticks do too, at ANY speed -- scrolling is activity by
    #    definition (reading through the page you just clicked). One merged
    #    time-sorted stream, not two passes: an extension by either kind
    #    re-opens the chain_gap window for later events of the other kind
    #    (a scroll->move->scroll chain would drop its middle link under
    #    per-stream scans). No scrolls -> exactly the old moves-only scan
    #    (moves arrive sorted; the sort is stable).
    activity = [(m[0], (m[3] if len(m) > 3 else 0.0), False) for m in moves]
    activity += [(s[0], 0.0, True) for s in scrolls]
    activity.sort(key=lambda a: a[0])
    for k in clusters:
        tmax = k["t_last"]
        for at, sp, is_scroll in activity:
            if (at >= k["t0"] - P.pre_roll
                    and (is_scroll or sp > P.move_speed_on)
                    and 0.0 <= at - tmax <= P.chain_gap):
                tmax = at
        # 2b) Drag-aware hold: a down->up span keeps the zoom alive for its
        # WHOLE duration, however slowly the cursor moves (the fast-move
        # rule above misses slow deliberate drags: careful slider drags,
        # slow text selection, dragging a file). Each click pairs with the
        # first up after it; extensions are capped (stuck-button safety)
        # and clamped at the start of any suppressed range the drag enters.
        # The paired (down, up_clamped) spans are also emitted into the
        # cluster's `drags` list so Pass 2 can route them through the
        # legacy comfort-box controller (drag = content).
        k["drags"] = []
        if getattr(P, "drag_hold", True) and ups:
            for (ct, _cx, _cy) in k["pts"]:
                for ut in ups:
                    if ut > ct:
                        ext = min(ut, ct + P.drag_max_hold)
                        for r in suppressed:
                            rs = _range_start(r)
                            if ct < rs < ext:
                                ext = rs
                        if ext > tmax:
                            tmax = ext
                        # short blip is just a click, not a drag
                        if ext - ct >= 0.4:
                            k["drags"].append((ct, ext))
                        break
        k["t_act"] = tmax

    # 3) Classify each cluster and compute its desired in/hold-end times.
    for k in clusters:
        xs = [p[1] for p in k["pts"]]
        ys = [p[2] for p in k["pts"]]
        bw = max(xs) - min(xs)
        bh = max(ys) - min(ys)
        n = len(k["pts"])
        rapid = (n >= P.overview_min_clicks
                 and (k["t_last"] - k["t0"]) / max(1, n - 1) < P.rapid_gap)
        wide = bw > P.overview_span_frac * W or bh > P.overview_span_frac * H
        if rapid and wide:
            # max(bw, 1) guards against a degenerate (collinear) bbox.
            zx = W / (max(bw, 1.0) + 2 * P.overview_margin_frac * W)
            zy = H / (max(bh, 1.0) + 2 * P.overview_margin_frac * H)
            k["level"] = _clamp(min(zx, zy), P.overview_zoom_min, P.max_zoom)
            k["ov"] = True
            k["lock"] = ((max(xs) + min(xs)) / 2.0, (max(ys) + min(ys)) / 2.0)
        else:
            k["level"] = P.max_zoom
            k["ov"] = False
            k["lock"] = None
        k["t_in"] = k["t0"] - P.pre_roll
        k["t_he"] = max(k["t_act"] + P.hold_after,
                        k["t_in"] + P.zoom_in_dur + P.min_hold)

    # 4) Combine auto (cluster) intervals with manual zoom ranges, sort by
    #    start time, then merge any whose zoom-in would begin before the
    #    previous one finishes zooming out -> one continuous hold (the camera
    #    pans between them) instead of a zoom-out-then-straight-back-in
    #    bounce. This is the same reducer as before; manual ranges just enter
    #    the pool as additional candidate intervals, so a manual punch-in that
    #    overlaps an auto-detected cluster blends for free.
    raw_intervals = [
        {"t_in": k["t_in"], "t_he": k["t_he"], "level": k["level"],
         "ov": k["ov"], "lock": k["lock"], "manual": False,
         # entry_xy is the CLICK that caused this zoom -- the point the
         # camera anchors to during the hold (see the segment
         # E2 case in Camera.update). None means "fall back to filtered
         # cursor at entry" (scroll-only / manual-follow / typing lock).
         "entry_xy": (k["pts"][0][1], k["pts"][0][2]),
         "drags": list(k["drags"])}
        for k in clusters
    ]
    for spec in manual:
        iv = _manual_interval(spec, P)
        iv["manual"] = True
        raw_intervals.append(iv)
    raw_intervals.extend(_plan_typing_intervals(keys, clicks, P,
                                                suppressed=suppressed,
                                                visual_anchors=typing_anchors))
    raw_intervals.extend(_plan_scroll_intervals(scrolls, W, H, P,
                                                suppressed=suppressed))
    # Merge decisions (and pool order) key on when the interval's ACTIVITY
    # begins -- merge_from, which scroll intervals set to their first tick
    # because their t_in is settle-anchored (see _plan_scroll_intervals).
    # Every other interval kind's t_in already tracks activity onset.
    raw_intervals.sort(key=lambda iv: iv.get("merge_from", iv["t_in"]))

    # Defensive defaults for interval kinds that don't populate the enriched
    # keys themselves (typing, scroll, manual). The reducer below and the
    # meta emission at step 5 both read these unconditionally.
    for iv in raw_intervals:
        iv.setdefault("entry_xy", None)
        iv.setdefault("drags", [])

    def _regrip_anchor(iv):
        """The anchor used for the same-anchor bridge disjunct: the fixed
        lock if set, else the click that opened the interval, else None
        (typing/scroll intervals without anchors don't bridge -- they'll
        merge on the normal horizon or not at all)."""
        return iv.get("lock") or iv.get("entry_xy")

    def _regrip_bridges(cur, nxt):
        """Same-anchor peephole: two intervals whose anchors coincide and
        whose full-out gap fits within regrip_gap merge as if they were
        the normal reducer's "zoom-out overlaps next zoom-in" case, so the
        camera holds ONE zoom across the gap instead of pumping out-and-
        straight-back-in at the same spot (the measured defect on both
        real recordings). Gated tight -- ANY of: bridge disabled, missing
        anchors, distant anchors, gap over horizon, or a suppressed range
        living inside the gap (the airtight escape hatch -- password
        typing between two clicks on the same login field must NOT bridge
        across the suppressed range).
        """
        rgap = float(getattr(P, "regrip_gap", 0.0) or 0.0)
        if rgap <= 0.0:
            return False
        gap = nxt.get("merge_from", nxt["t_in"]) - cur["t_he"]
        if gap <= P.zoom_out_dur:
            return False    # normal reducer already merges this pair
        if gap > P.zoom_out_dur + rgap:
            return False
        a, b = _regrip_anchor(cur), _regrip_anchor(nxt)
        if a is None or b is None:
            return False
        r = P.regrip_frac * W
        if abs(a[0] - b[0]) > r or abs(a[1] - b[1]) > r:
            return False
        for sr in suppressed:
            rs, re = _range_start(sr), _range_end(sr)
            if rs < nxt.get("merge_from", nxt["t_in"]) and re > cur["t_he"]:
                return False
        return True

    intervals = []
    for iv in raw_intervals:
        if intervals and (
                iv.get("merge_from", iv["t_in"])
                    <= intervals[-1]["t_he"] + P.zoom_out_dur
                or _regrip_bridges(intervals[-1], iv)):
            cur = intervals[-1]
            # Attention bookkeeping for the typing-protection windows
            # emitted with the meta below: typing members carry
            # (from, activity-end) pairs, scroll members their burst-start
            # times. Accumulated on every merge; only consulted when the
            # merged interval still carries the typing flag.
            if iv.get("_tm"):
                cur["_tm"] = cur.get("_tm", []) + iv["_tm"]
            if iv.get("_ss"):
                cur["_ss"] = cur.get("_ss", []) + iv["_ss"]
            cur["t_he"] = max(cur["t_he"], iv["t_he"])
            cur["level"] = max(cur["level"], iv["level"])
            if iv["manual"] and iv["ov"]:
                # an explicit manual pin always wins the merged lock point,
                # even if it's folded into a preceding/following auto "follow"
                # cluster -- the user asked for this exact spot on purpose.
                cur["ov"] = True
                cur["lock"] = iv["lock"]
                cur["typing"] = False
            elif cur["manual"] and cur["ov"]:
                pass   # keep the already-merged-in manual pin
            elif ((cur.get("typing") or iv.get("typing"))
                    and not (cur["manual"] or iv["manual"])):
                # A typing anchor survives merging with auto clusters (the
                # dominant click-then-type flow): during typing the cursor is
                # exactly the signal that lies, so cursor-follow must not win
                # the merged hold. Manual ranges (pin OR follow) still
                # outrank typing -- the user said so on purpose -- via the
                # branches above/below. Later typing anchor wins if both.
                src = iv if iv.get("typing") else cur
                cur["ov"] = False
                cur["lock"] = src["lock"]
                # Compute BEFORE flagging cur as typing: a non-typing cur
                # (e.g. a drag-held click cluster) must not contribute its
                # own t_in here, or the protection window would swallow the
                # whole drag span.
                cur["typing_from"] = min(
                    x.get("typing_from", x["t_in"])
                    for x in (cur, iv) if x.get("typing"))
                cur["typing"] = True
            elif not cur["ov"] or not iv["ov"]:   # any auto follow -> tight
                cur["ov"] = False
                cur["lock"] = None
                cur["typing"] = False
            cur["manual"] = cur["manual"] or iv["manual"]
            # Anchor / drag bookkeeping for Pass 2:
            #   entry_xy = the anchor of the EARLIEST-BY-ACTIVITY member
            #   of the merged interval. cur already carries its own
            #   entry_xy (from insertion); we only borrow iv's when cur
            #   didn't have one AND iv started AT or BEFORE cur --
            #   which is impossible under the reducer's merge_from sort
            #   (cur was inserted first ⇒ cur.merge_from ≤ iv.merge_from).
            #   Practical consequence: a scroll-read interval that starts
            #   BEFORE a click cluster keeps entry_xy=None so E2 anchors
            #   at the filtered cursor (= the read region), NOT the
            #   future click that user hasn't performed yet -- which
            #   would kick off a huge pre-emptive glide to a point the
            #   viewer isn't looking at. The click will retarget via E1
            #   at its actual time. Verified against the session 155340
            #   scroll->click->click sequence.
            cur["drags"] = list(cur.get("drags") or []) + list(iv.get("drags") or [])
            continue
        intervals.append(dict(iv))

    # 5) Bake quintic zoom keyframes.
    Z = [(0.0, 1.0)]
    meta = []

    def push(t, v):
        if t <= Z[-1][0]:
            t = Z[-1][0] + 1e-4
        Z.append((t, v))

    always_zoomed = bool(getattr(P, "always_zoomed", False))
    n_intervals = len(intervals)
    for idx, iv in enumerate(intervals):
        lvl = iv["level"]
        push(iv["t_in"], 1.0)
        push(iv["t_in"] + P.zoom_in_dur, lvl)
        push(iv["t_he"], lvl)
        # "Always zoomed" holds the *last* interval's level for the rest of
        # the clip instead of zooming back out -- e.g. for a fixed-focus demo
        # where the camera should stay punched in once it gets there.
        if always_zoomed and idx == n_intervals - 1:
            t_oe = max(T_end, iv["t_he"])
        else:
            t_oe = iv["t_he"] + P.zoom_out_dur
            push(t_oe, 1.0)
        # Typing-protection WINDOWS: one (start, until) pair per merged
        # typing burst. `until` is None (protection through the hold, the
        # historical behavior) unless a scroll burst BEGAN after that
        # burst's last keystroke -- the user demonstrably moved on to
        # scroll-reading, so the cursor stops lying and follow mode must
        # take back over (type-then-scroll, panel-confirmed defect). A
        # scroll that starts DURING typing ends nothing.
        windows = None
        if iv.get("typing"):
            starts = sorted(iv.get("_ss") or [])
            members = iv.get("_tm") or [(iv.get("typing_from", iv["t_in"]),
                                         None)]
            windows = []
            for f, act in sorted(members, key=lambda m: m[0]):
                until = None
                if act is not None:
                    for s in starts:
                        if s > act:
                            until = s
                            break
                windows.append((f, until))
        meta.append({"t_in": iv["t_in"], "t_he": iv["t_he"], "t_oe": t_oe,
                     "ov": iv["ov"], "lock": iv["lock"],
                     "typing": bool(iv.get("typing", False)),
                     "typing_from": iv.get("typing_from", iv["t_in"]),
                     "typing_windows": windows,
                     # Pass-2 focused-mode inputs (harmless in smooth mode).
                     "entry_xy": iv.get("entry_xy"),
                     "drags": list(iv.get("drags") or [])})

    if always_zoomed and intervals:
        push(max(T_end, Z[-1][0]), intervals[-1]["level"])
    else:
        push(max(T_end, Z[-1][0]), 1.0)
        Z = _settle_tail(Z, T_end, P.zoom_out_dur)
    return Z, meta


# --------------------------------------------------------------------------
# PASS 2 — per-frame center follower + zoom sampling
# --------------------------------------------------------------------------
class Camera(object):
    def __init__(self, W, H, fps, Z, meta, P, win_w=None, win_h=None):
        self.W = W
        self.H = H
        # win_w/win_h: the zoom-1.0 crop window size in source pixels. When
        # the render canvas matches the source aspect (the default), this is
        # just W/H, so every formula below is unchanged from before. A
        # non-source-aspect export (see framing.resolve_aspect_canvas) passes
        # a narrower/shorter window here -- e.g. a vertical strip of a
        # landscape source -- while W/H stay the source's own clamp bounds.
        self.win_w = W if win_w is None else win_w
        self.win_h = H if win_h is None else win_h
        self.dt = 1.0 / float(fps)
        self.P = P
        self.Z = Z
        self.meta = meta
        self.zi = 0
        self.mi = 0
        self.cx = W / 2.0
        self.cy = H / 2.0
        self.vx = 0.0
        self.vy = 0.0
        self.ax = W / 2.0
        self.ay = H / 2.0
        self.cfx = W / 2.0
        self.cfy = H / 2.0
        self.pfx = W / 2.0
        self.pfy = H / 2.0
        self.vcx = 0.0
        self.vcy = 0.0
        # Most recent click position (source px) and its time; the `typing`
        # segment mode targets this so mid-burst clicks re-aim the camera
        # while cursor shoves/parking are ignored. None until the first
        # click commits.
        self.last_click_xy = None
        self.last_click_t = -1e18
        # Focused-mode state (the focused controller):
        #   dwell -- E3 hysteresis timer: accumulates dt each frame the
        #     cursor is (outside the containment box AND quasi-stationary);
        #     resets to 0 on re-entry or on fast motion. Fires one glide
        #     when it reaches recapture_dwell, subject to the cooldown.
        #   last_retarget -- t of the most recent E2/E3/E4/click retarget,
        #     for the cooldown gate that keeps non-click glides rare.
        #   glide_d -- distance of the current in-flight glide (0 while
        #     the camera is at rest). Raises vmax for far retargets so
        #     they complete in <= glide_max_dur instead of crawling at
        #     the base pan cap. Cleared by the arrival snap.
        #   prev_prot -- last frame's typing-protection state, for the
        #     falling-edge detector that fires E4 (protection handoff).
        #   seg_key -- segment identity, for the rising-edge detector
        #     that fires E2 (segment entry).
        self.dwell = 0.0
        self.last_retarget = -1e18
        self.glide_d = 0.0
        self.prev_prot = False
        self.seg_key = None

    def _retarget(self, tx, ty, t):
        """Discrete E1/E2/E3/E4 retarget: latch the anchor, capture the
        glide distance so the vmax cap raise decides its duration, and
        reset the dwell timer so E3 can't chain-fire immediately after."""
        self.glide_d = hypot(self.cx - float(tx), self.cy - float(ty))
        self.ax, self.ay = float(tx), float(ty)
        self.last_retarget = t
        self.dwell = 0.0

    def _drag_active(self, seg, t):
        """True while t is inside any (down, up_clamped) drag span emitted
        for this segment (plan_zoom step 2b). Empty list / no segment ->
        False; keys-absent / drag_hold-off sessions leave `drags` empty
        so the focused-mode latched hold applies uniformly."""
        if seg is None:
            return False
        for (d0, d1) in seg.get("drags") or []:
            if d0 <= t < d1:
                return True
        return False

    def _seg(self, t):
        m = self.meta
        if not m:
            return None
        while self.mi + 1 < len(m) and t >= m[self.mi]["t_oe"]:
            self.mi += 1
        while self.mi > 0 and t < m[self.mi]["t_in"]:
            self.mi -= 1
        s = m[self.mi]
        return s if (s["t_in"] <= t < s["t_oe"]) else None

    def update(self, t, cursor, click_pos):
        W, H, dt, P = self.W, self.H, self.dt, self.P
        win_w, win_h = self.win_w, self.win_h
        z, self.zi = _sample(self.Z, t, self.zi)
        z = _clamp(z, 1.0, P.max_zoom)
        seg = self._seg(t)
        zooming_out = seg is not None and t >= seg["t_he"]
        focused = getattr(P, "screen_anim", "focused") == "focused"

        # De-jitter cursor / hard-commit on click.
        if click_pos is not None:
            self.cfx, self.cfy = click_pos
            # E1 -- click retarget, gated by the recenter frac so a click
            # already comfortably visible in the current framing doesn't
            # trigger a fidget glide (kills the form-field staircase, the
            # worst-looking focused-mode residual). Legacy/smooth mode
            # snaps ax/ay to the click as before -- no gating there.
            recenter_frac = float(getattr(P, "recenter_frac", 0.0) or 0.0)
            if focused and recenter_frac > 0.0:
                hx_click = recenter_frac * (win_w / (2.0 * z))
                hy_click = recenter_frac * (win_h / (2.0 * z))
                if (abs(click_pos[0] - self.ax) <= hx_click
                        and abs(click_pos[1] - self.ay) <= hy_click):
                    pass    # click is centrally visible -- don't retarget
                else:
                    self._retarget(click_pos[0], click_pos[1], t)
            else:
                self.ax, self.ay = click_pos
            self.last_click_xy = click_pos
            self.last_click_t = t
        else:
            dx = cursor[0] - self.cfx
            dy = cursor[1] - self.cfy
            d = hypot(dx, dy)
            dz = P.deadzone_px / z
            if d > dz:
                a = 1.0 - exp(-dt / P.cursor_tau)
                k = (d - dz) / d
                self.cfx += a * dx * k
                self.cfy += a * dy * k
        b = 1.0 - exp(-dt / P.vel_tau)
        self.vcx += b * (((self.cfx - self.pfx) / dt) - self.vcx)
        self.vcy += b * (((self.cfy - self.pfy) / dt) - self.vcy)
        self.pfx, self.pfy = self.cfx, self.cfy

        # E2 -- segment entry (rising edge of segment identity): latch the
        # anchor at the click that CAUSED this zoom (seg["entry_xy"], set
        # by plan_zoom step 3), or at the pin/lock, or fall back to the
        # filtered cursor. This is the load-bearing "anchor on the click,
        # not the cursor" rule that makes the hold rock-still by default.
        # Skipped in smooth mode -- there the old comfort-box branch owns
        # the anchor continuously.
        if focused:
            key = (self.mi, seg["t_in"]) if seg is not None else None
            if key != self.seg_key:
                self.seg_key = key
                if seg is not None and not zooming_out and not seg["ov"]:
                    # A typing segment's lock only rules once typing has
                    # actually begun (typing_from). A merged scroll-then-
                    # type hold enters at the SCROLL's t_in -- gliding to
                    # the typing anchor there would pan the camera to a
                    # field the user hasn't touched yet, then E3 would
                    # drag it back to the read region, then the typing
                    # branch would pan it out again: three pans where the
                    # merge promises one hold (adversarial-review-confirmed
                    # defect; see test_scroll_then_type_does_not_preglide_
                    # to_typing_anchor). Before typing_from the entry
                    # evidence is entry_xy (the opening click) or the
                    # cursor (the read region); the typing branch itself
                    # retargets to the lock when protection begins.
                    lock = seg.get("lock")
                    if (lock is not None and seg.get("typing")
                            and t < seg.get("typing_from", seg["t_in"])):
                        lock = None
                    ex = lock or seg.get("entry_xy") or (self.cfx, self.cfy)
                    self._retarget(ex[0], ex[1], t)

            # E4 -- protection handoff: when a typing window closes mid-
            # segment (a scroll burst begins after the last keystroke,
            # per plan_zoom's typing_windows machinery), hand the camera
            # back to the current cursor so it pans to the region the
            # user is now reading. Falling-edge detected on prev_prot so
            # it fires exactly once per handoff.
            prot_now = (seg is not None and seg.get("typing")
                        and not zooming_out and _typing_window_active(seg, t))
            if self.prev_prot and not prot_now and seg is not None and not zooming_out:
                self._retarget(self.cfx, self.cfy, t)
            self.prev_prot = prot_now

        # Pick the raw center target by phase.
        if seg is not None and seg["ov"]:
            tx, ty = seg["lock"]
        elif (seg is not None and seg.get("typing") and not zooming_out
                and _typing_window_active(seg, t)):
            # Anchored typing hold: point at the most recent click WITHIN
            # this segment; only clicks move the target. Cursor motion
            # (shoves, parking, idle drift) is ignored for the duration --
            # during typing the cursor is precisely the positional signal
            # that lies. Until a click commits inside the segment, the
            # target is the PLAN-TIME anchor (seg["lock"]): a click from
            # long before the segment may be staler than the planner's
            # typing_anchor_max_age rule allows, and briefly zooming toward
            # it (then reversing at full zoom when the real anchoring click
            # lands -- the slack-anchored flow) is exactly the motion
            # artifact this mode exists to prevent.
            # Only clicks that land AFTER typing began re-aim the camera:
            # before that the plan-time anchor rules (it may be a VISUAL
            # anchor pointing at the real caret, which the pre-typing
            # anchoring click must not override -- the click-then-type
            # merged hold would otherwise snap back to the click).
            if (self.last_click_xy is not None
                    and self.last_click_t >= seg.get("typing_from",
                                                     seg["t_in"])):
                tx, ty = self.last_click_xy
            else:
                tx, ty = seg["lock"]
            # Keep the follower anchor in sync so the zoom-out phase (which
            # targets ax/ay) continues from the same point without a yank.
            # A target change here (mid-burst click re-aim to a different
            # field) routes through _retarget so glide_d bookkeeping is
            # correct and the arrival snap can freeze at the new anchor.
            if focused and (tx != self.ax or ty != self.ay):
                self._retarget(tx, ty, t)
            else:
                self.ax, self.ay = tx, ty
        elif seg is not None and not zooming_out:
            drag_now = self._drag_active(seg, t)
            if drag_now or not focused:
                # Legacy comfort-box + predictive-lead follower. Two uses:
                # (a) actual drags -- drag IS the content, so the camera
                #     should track the cursor pixel-for-pixel (careful
                #     slider drags, slow text selection). Focused-mode
                #     drops back into this branch for the span. On drag
                #     end, ax/ay stay wherever the box last dragged them
                #     (E5 latch is implicit -- nothing touches ax/ay in
                #     the focused-latched branch below).
                # (b) screen_anim="smooth" -- the escape-hatch mode that
                #     reproduces the pre-redesign look verbatim.
                cw = P.comfort_frac * (win_w / z)
                ch = P.comfort_frac * (win_h / z)
                if self.cfx > self.ax + cw:
                    self.ax = self.cfx - cw
                elif self.cfx < self.ax - cw:
                    self.ax = self.cfx + cw
                if self.cfy > self.ay + ch:
                    self.ay = self.cfy - ch
                elif self.cfy < self.ay - ch:
                    self.ay = self.cfy + ch
                mlx = P.max_lead_frac * (win_w / z)
                mly = P.max_lead_frac * (win_h / z)
                tx = self.ax + _clamp(P.lead_time * self.vcx, -mlx, mlx)
                ty = self.ay + _clamp(P.lead_time * self.vcy, -mly, mly)
                # Reset the dwell timer so E3 doesn't fire immediately
                # when the drag ends and control returns to the focused
                # branch (the cursor may well be outside the containment
                # box, but we just handed the camera to it).
                self.dwell = 0.0
            else:
                # FOCUSED latched hold + E3 dwell recapture. The anchor
                # is piecewise-CONSTANT -- ordinary cursor motion inside
                # the visible window never touches it. The camera moves
                # only when a click commits (E1 above), a segment enters
                # (E2 above), a typing window closes (E4 above), or the
                # cursor sits still beyond the outer containment margin
                # for recapture_dwell seconds without a click coming in.
                # Result: the hold is rock-still by construction.
                bx = float(getattr(P, "contain_frac", 0.8)) * (win_w / (2.0 * z))
                by = float(getattr(P, "contain_frac", 0.8)) * (win_h / (2.0 * z))
                outside = (abs(self.cfx - self.ax) > bx
                           or abs(self.cfy - self.ay) > by)
                slow = hypot(self.vcx, self.vcy) < float(getattr(P, "dwell_speed", 60.0))
                if outside and slow:
                    self.dwell += dt
                else:
                    self.dwell = 0.0
                if (self.dwell >= float(getattr(P, "recapture_dwell", 0.45))
                        and (t - self.last_retarget)
                            >= float(getattr(P, "retarget_cooldown", 1.2))):
                    self._retarget(self.cfx, self.cfy, t)
                tx, ty = self.ax, self.ay
        else:
            # Zooming-out phase (t >= t_he inside a segment) or between
            # segments: target is the latched anchor. The safe-band clamp
            # below handles the recentering as z ramps back to 1.0 --
            # today's mechanism, and correct under `always_zoomed` (the
            # graft the panel flagged) because ax/ay is left alone, not
            # projected to the z=1 band.
            tx, ty = self.ax, self.ay

        # Pre-clamp target to the current zoom's safe band.
        hx = win_w / (2.0 * z)
        hy = win_h / (2.0 * z)
        tx = _clamp(tx, hx, W - hx)
        ty = _clamp(ty, hy, H - hy)

        # Exact critically-damped center spring.
        px, py = self.cx, self.cy
        self.cx, self.vx = cd_step(self.cx, self.vx, tx, P.omega_pan, dt)
        self.cy, self.vy = cd_step(self.cy, self.vy, ty, P.omega_pan, dt)

        # Velocity clamp (bound on-screen pan speed). Base is the pan_cap
        # frac like before; on top, an in-flight glide (glide_d > 0) may
        # raise it so the pan completes within glide_max_dur regardless
        # of distance. Otherwise a 1500 px volley retarget at the base
        # cap crawls for 2+ seconds -- the "swimmy" tail the metrics
        # caught. Motion blur (already default-on) covers the higher peak.
        vmax = (P.pan_cap_frac * win_w) / z
        if focused and self.glide_d > 0.0:
            gmax = float(getattr(P, "glide_max_dur", 1.10))
            if gmax > 0.0:
                vmax = max(vmax, self.glide_d / gmax)
        sp = hypot(self.vx, self.vy)
        if sp > vmax:
            s = vmax / sp
            self.vx *= s
            self.vy *= s
            self.cx = px + self.vx * dt
            self.cy = py + self.vy * dt

        # Hard clamp + anti-windup.
        if self.cx < hx:
            self.cx = hx
            self.vx = max(self.vx, 0.0)
        if self.cx > W - hx:
            self.cx = W - hx
            self.vx = min(self.vx, 0.0)
        if self.cy < hy:
            self.cy = hy
            self.vy = max(self.vy, 0.0)
        if self.cy > H - hy:
            self.cy = H - hy
            self.vy = min(self.vy, 0.0)

        # Arrival snap: once the spring is within arrive_eps of the
        # anchor AND essentially still, freeze the center at the anchor
        # exactly. Every subsequent frame in the hold emits identical
        # floats, so np.ptp(hold) == 0 exactly -- the dead-still hold
        # the complaint asked for. Skipped in smooth mode (that path is
        # supposed to reproduce the pre-redesign look bit-for-bit).
        if focused and self.glide_d > 0.0:
            eps = float(getattr(P, "arrive_eps", 1.5))
            if (abs(self.cx - self.ax) < eps and abs(self.cy - self.ay) < eps
                    and hypot(self.vx, self.vy) < 40.0):
                self.cx, self.cy = self.ax, self.ay
                self.vx = self.vy = 0.0
                self.glide_d = 0.0

        return self.cx, self.cy, z


# --------------------------------------------------------------------------
# Renderer-facing wrapper (stable interface)
# --------------------------------------------------------------------------
def build_path(frame_times, W, H,
               clicks_t, clicks_x, clicks_y,
               moves_t, moves_x, moves_y,
               max_zoom=2.0, fps=60.0, params=None,
               manual_zooms=None, suppressed_ranges=None,
               plan_duration=None, win_w=None, win_h=None,
               keys_t=None, ups_t=None, typing_anchors=None,
               scrolls_t=None, scrolls_x=None, scrolls_y=None):
    """Return an (T, 3) array of (center_x, center_y, zoom) in pixel units.

    manual_zooms/suppressed_ranges are edits.json-shaped lists (see
    plan_zoom's docstring) threaded straight through from the renderer.

    keys_t: optional array/list of key-activity tick times (media-relative
    seconds, no key identity). None/empty reproduces the keys-less path
    bit-exactly; see plan_zoom/_plan_typing_intervals for the semantics.

    ups_t: optional array/list of mouse-up times, for drag-aware zoom
    holds (plan_zoom step 2b). None/empty reproduces the ups-less path
    bit-exactly.

    scrolls_t/scrolls_x/scrolls_y: optional scroll-tick times + cursor
    positions (source px) for the scroll-aware camera (cluster keep-alive
    extension + gentle scroll-read zooms; see _plan_scroll_intervals).
    None/empty reproduces the scroll-less path bit-exactly -- which is
    also every session recorded before scroll capture existed.

    typing_anchors: optional list of visual typing anchors (dicts with
    start/end/x/y, aligned to typing_bursts() spans; None entries allowed)
    -- see autocine.vision. They outrank the last-click anchor rule.

    plan_duration: the true end-of-clip time used for pass-1 planning (in
    particular `_settle_tail`'s "the zoom must be back to 1.0 by the end"
    guarantee). Defaults to frame_times[-1], which is correct whenever
    frame_times spans the whole clip (the normal `render()` case). Callers
    that only simulate a *prefix* of the clip -- e.g. a single-frame preview
    at time t, which only needs frame_times up to t for performance -- MUST
    pass the real clip duration here, or the tail-settle logic will mistake
    "the frame we stopped simulating at" for "the end of the clip" and force
    the zoom to prematurely flatten back to 1.0 by that point.

    win_w/win_h: the zoom-1.0 crop window size in source pixels, for a
    render canvas whose aspect doesn't match the source (see
    framing.resolve_aspect_canvas + render.py's `_contain_fit`). Defaults to
    W/H, reproducing today's behavior exactly.
    """
    manual_zooms = list(manual_zooms or [])
    suppressed_ranges = list(suppressed_ranges or [])
    # A manual range may ask for more zoom than the render's own --zoom/max
    # setting; raise the effective ceiling so it isn't silently clipped by
    # Camera.update's per-frame clamp to P.max_zoom.
    manual_levels = [float(m.get("level")) for m in manual_zooms
                     if m.get("level") is not None]
    eff_max_zoom = max([max_zoom] + manual_levels) if manual_levels else max_zoom
    P = _params(eff_max_zoom, params)
    T = len(frame_times)
    if T == 0:
        return np.zeros((0, 3))

    clicks = [(float(clicks_t[i]), float(clicks_x[i]), float(clicks_y[i]))
              for i in range(len(clicks_t))]
    if suppressed_ranges:
        # Keep click_frame (below) in sync with the plan: a suppressed click
        # must never snap the camera to it, even if the camera happens to be
        # zoomed in at that moment for an unrelated reason.
        clicks = [c for c in clicks if not _in_any_range(c[0], suppressed_ranges)]

    moves = []
    if moves_t.size:
        for i in range(len(moves_t)):
            if i > 0:
                dtm = moves_t[i] - moves_t[i - 1]
                sp = (hypot(moves_x[i] - moves_x[i - 1],
                            moves_y[i] - moves_y[i - 1]) / dtm) if dtm > 0 else 0.0
            else:
                sp = 0.0
            moves.append((float(moves_t[i]), float(moves_x[i]),
                          float(moves_y[i]), float(sp)))

    keys = ([] if keys_t is None
            else sorted(float(k) for k in np.asarray(keys_t).ravel()))
    ups = ([] if ups_t is None
           else sorted(float(u) for u in np.asarray(ups_t).ravel()))

    scrolls = []
    if scrolls_t is not None:
        s_t = np.asarray(scrolls_t, dtype=float).ravel()
        s_x = (np.zeros(s_t.size) if scrolls_x is None
               else np.asarray(scrolls_x, dtype=float).ravel())
        s_y = (np.zeros(s_t.size) if scrolls_y is None
               else np.asarray(scrolls_y, dtype=float).ravel())
        scrolls = sorted((float(s_t[i]), float(s_x[i]), float(s_y[i]))
                         for i in range(s_t.size))

    T_end = float(plan_duration) if plan_duration is not None else float(frame_times[-1])
    Z, meta = plan_zoom(clicks, moves, W, H, T_end, P,
                        manual=manual_zooms, suppressed=suppressed_ranges,
                        keys=keys, ups=ups, typing_anchors=typing_anchors,
                        scrolls=scrolls)
    cam = Camera(W, H, fps, Z, meta, P, win_w=win_w, win_h=win_h)

    if moves_t.size >= 1:
        cxp = np.interp(frame_times, moves_t, moves_x,
                        left=moves_x[0], right=moves_x[-1])
        cyp = np.interp(frame_times, moves_t, moves_y,
                        left=moves_y[0], right=moves_y[-1])
        cam.cfx = cam.pfx = float(cxp[0])
        cam.cfy = cam.pfy = float(cyp[0])
    elif scrolls:
        # No move/click samples at all (cursor parked since before the
        # recording started) but scroll ticks exist: their x/y IS the
        # parked cursor position, and it is the only positional signal in
        # the whole session -- without this a scroll-read zoom would aim
        # at the frame center instead of where the user is reading
        # (review-confirmed defect). Scroll-less sessions never reach
        # this branch, so the bit-exactness contract is untouched.
        s_t = np.array([s[0] for s in scrolls])
        s_x = np.array([s[1] for s in scrolls])
        s_y = np.array([s[2] for s in scrolls])
        cxp = np.interp(frame_times, s_t, s_x, left=s_x[0], right=s_x[-1])
        cyp = np.interp(frame_times, s_t, s_y, left=s_y[0], right=s_y[-1])
        cam.cfx = cam.pfx = float(cxp[0])
        cam.cfy = cam.pfy = float(cyp[0])
    else:
        cxp = np.full(T, W / 2.0)
        cyp = np.full(T, H / 2.0)

    click_frame = {}
    for c in clicks:
        fi = int(round(c[0] * fps))
        if 0 <= fi < T:
            click_frame[fi] = (c[1], c[2])

    out = np.zeros((T, 3))
    for i in range(T):
        cp = click_frame.get(i)
        out[i] = cam.update(float(frame_times[i]),
                            (float(cxp[i]), float(cyp[i])), cp)
    return out
