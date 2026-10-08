"""Event-derived beat sheet: what happened in a take, without decoding a frame.

`describe_session` hands an agent counts and flat float arrays -- 53 click
timestamps and 303 scroll timestamps say that something happened 356 times and
nothing at all about *what*. Everything needed to say what is already sitting
in events.jsonl: the geometry track records EVERY on-screen window's rect and
front-to-back rank at 20Hz, so a click can be attributed to the window it
landed in, a scroll to the window it scrolled, and a window coming to front is
a scene change that costs nothing to detect.

This module is the read model over that. Pure functions over the arrays
`geometry.load_events` returns plus a duration: no video decode, no ffmpeg, no
sidecar file, and therefore no cache to invalidate and no staleness bug. On a
4-minute session with 1679 geometry samples the derivation is 20-38ms (47-56ms
through `render.session_beats`, which also re-reads events.jsonl), which is
why it can be inlined into `describe_session` rather than built behind a tool
call. Cost scales with the GEOMETRY TRACK, not the video: a synthetic 2-hour
take with 40 windows (277k samples) is ~1.8s and one with 200 windows (1.4M
samples) ~5.4s. Those are far outside anything a real desktop produces, but
they are the honest shape of the curve -- attribution is O(queries x live
windows) -- and worth knowing before anyone puts this on a hot path.

Three rules it inherits from the code it sits next to:

- **`zoom` is what will RENDER; `zoom_proposal` is what auto-zoom would
  suggest.** They are different fields because they are different facts:
  every surface but a bare CLI render plans the camera only from the
  materialized `edits.zooms` (`camera.build_path` sets
  `auto_cluster = manual_zooms is None`, and the MCP/web resolvers always
  pass a list), so a session whose auto-zooms were never materialized -- or
  whose zooms an agent has since deleted -- renders no camera move at all.
  Calling the proposal "the plan" would state the opposite of the truth.
  The proposal itself still comes from the planner's own functions
  (`camera.cluster_clicks` / `camera.cluster_to_range`), so it cannot drift
  from the camera the first time anyone retunes `chain_gap`.
- **Identity stays opaque.** This reads the same events.jsonl that
  deliberately records no app name and no window title, and it does not undo
  that: a beat says `window_id 36554`, never what 36554 was.
- **Nothing is silently dropped.** The beat list is capped (an agent reading
  a 30-minute take should not be handed thousands of rows), and the overflow
  is reported in `truncated` rather than quietly trimmed.

Every time is MEDIA seconds -- the same clock as `describe_session`'s
`click_times`, `edits.trim` and every zoom range -- so a beat's `start` can be
handed straight to `set_trim` / `add_zoom` / `add_speedup` / `add_marker`.
Positions are SOURCE PIXELS, the space `describe_session` reports as
`width`/`height`, so a `bbox` feeds `add_zoom`'s x/y without a unit
conversion. Events themselves are recorded in points; the caller supplies the
per-axis scale exactly as render derives it.
"""

import numpy as np

from . import camera
from . import geometry
from . import retime

# Whether a window is on screen at time t is decided by its own first/last
# sample, NOT by a staleness constant -- and the grace on each end is derived
# from the track's measured heartbeat rather than assumed. The recorder polls
# at 20Hz but only heartbeats each window about once a second (measured on a
# real take: median inter-sample gap 1.043s, max 1.079s, dead flat across all
# ten windows), so any hardcoded cut lands either just under the cadence --
# which silently drops every window for a third of all query times -- or far
# enough over it to keep closed windows alive. Deriving it removes the choice.
_FALLBACK_PERIOD = 1.05  # s, only for a track too short to measure
_GRACE_PERIODS = 1.5     # a presence interval extends this many periods past
                         # its last sample before the window counts as gone
_GAP_PERIODS = 2.5       # a hole wider than this splits a window's track into
                         # separate presence intervals (minimised / hidden /
                         # moved to another Space, then brought back)
_EDGE_PERIODS = 2.0      # first/last sample within this many periods of the
                         # track's own edge = already open at record start,
                         # which is state, not an open/close BEAT
SCROLL_GAP = 1.0        # s, scroll ticks within this join one run
TYPING_GAP = 2.0        # s, key ticks within this join one burst
MIN_IDLE = 3.0          # s, dead air shorter than this is a pause, not a hole
MAX_BEATS = 300         # cap on the returned list; overflow is REPORTED

_KINDS_BY_PRIORITY = ("clicks", "scroll", "typing", "front", "open",
                      "close", "idle")


def _finite_media(arr, t0, duration):
    """(times, keep_mask) in media seconds, clamped to [0, duration].

    Returns the mask too so callers can keep x/y aligned with t.
    """
    a = np.asarray(arr if arr is not None else [], dtype=float).ravel()
    if a.size == 0:
        return a, np.zeros(0, dtype=bool)
    t = a - float(t0)
    keep = np.isfinite(t) & (t >= 0.0)
    if duration and duration > 0:
        keep &= t <= (float(duration) + 1e-6)
    return t[keep], keep


def _runs(times, gap):
    """Split ascending `times` into runs, breaking on a gap wider than `gap`.

    Returns a list of (start_index, end_index_exclusive).
    """
    if times.size == 0:
        return []
    out = []
    s = 0
    for i in range(1, int(times.size)):
        if times[i] - times[i - 1] > gap:
            out.append((s, i))
            s = i
    out.append((s, int(times.size)))
    return out


def _track(ev, t0, duration):
    """The geometry track as ascending media-time samples, or None.

    None means the session has no window track at all (recorded before
    geometry tracking, or a backend that wrote none) -- every window-aware
    beat is then simply absent, which is the honest answer.
    """
    t = ev.get("windows_t")
    rects = ev.get("windows_rect")
    ids = ev.get("windows_id")
    zs = ev.get("windows_z")
    if t is None or rects is None or ids is None or len(t) == 0:
        return None
    t = np.asarray(t, dtype=float).ravel()
    rects = np.asarray(rects, dtype=float)
    ids = np.asarray(ids, dtype=int).ravel()
    if zs is None or len(zs) != len(t):
        zs = np.full(t.shape, -1, dtype=int)
    else:
        zs = np.asarray(zs, dtype=int).ravel()
    if rects.ndim != 2 or rects.shape[0] != t.shape[0] or rects.shape[1] != 4:
        return None
    # Media clock, same conversion every other consumer uses. Samples outside
    # the clip are kept: a window's geometry at t=-0.5 is the state the take
    # opened in, and dropping it would invent an "open" beat at t=0.
    tm = t - float(t0)
    order = np.argsort(tm, kind="stable")
    tm, rects, ids, zs = tm[order], rects[order], ids[order], zs[order]

    # Heartbeat period, measured per window and pooled. Median over the whole
    # pooled gap distribution, so one window that closed early can't skew it.
    # Group samples by window in ONE pass. The obvious `tm[ids == wid]` per
    # window is O(windows x samples) and really bites: on a 2-hour take with
    # 200 windows (1.38M samples) it was 5.7s, inlined into describe_session.
    # Sorting by id instead is O(n log n) -- same result, 0.2s.
    gaps = []
    per_window = {}
    by_id = np.argsort(ids, kind="stable")
    ids_sorted = ids[by_id]
    bounds = np.flatnonzero(np.diff(ids_sorted)) + 1
    for grp in np.split(by_id, bounds):
        if grp.size == 0:
            continue
        wid = int(ids[grp[0]])
        if wid < 0:
            continue
        # argsort by id is stable and tm is already time-ordered, so grp is
        # ascending in time -- sorted() is a cheap guarantee, not a fix.
        order_w = np.argsort(tm[grp], kind="stable")
        tw = tm[grp][order_w]
        per_window[wid] = (tw, zs[grp][order_w])
        if tw.size >= 2:
            gaps.append(np.diff(tw))
    period = float(np.median(np.concatenate(gaps))) if gaps else _FALLBACK_PERIOD
    if not np.isfinite(period) or period <= 0:
        period = _FALLBACK_PERIOD

    # Presence INTERVALS, not one (first, last) span. A window that is
    # minimised, Cmd-H'd or on another Space stops appearing in the poll
    # entirely -- devices._filter_windows drops anything without
    # kCGWindowIsOnscreen -- and comes back under the SAME id. Collapsing
    # that to (first, last) leaves the window "present" across a hole it was
    # not on screen for, so its stale rect keeps winning clicks (and, being
    # the smaller/frontmost of the two, wins them from the window that was
    # actually visible). Splitting on a gap wider than the heartbeat makes
    # the disappearance both correct and VISIBLE, as close/open beats.
    hole = period * _GAP_PERIODS
    life = {}
    for wid, (tw, _zw) in per_window.items():
        spans = []
        s = float(tw[0])
        for i in range(1, int(tw.size)):
            if float(tw[i]) - float(tw[i - 1]) > hole:
                spans.append((s, float(tw[i - 1])))
                s = float(tw[i])
        spans.append((s, float(tw[-1])))
        life[wid] = spans
    return {
        "t": tm, "rect": rects, "id": ids, "z": zs,
        "has_z": bool(zs.size and int(np.max(zs)) >= 0),
        "duration": float(duration or 0.0),
        "period": period, "life": life, "groups": per_window,
    }


def _states_at(track, queries):
    """Window state at each of `queries` (ascending media seconds).

    One merged pass over samples and queries: {window_id: (rect, z)} holding
    each window's newest sample at or before the query time, minus any window
    that has since gone away (past the end of its own recorded lifetime).
    """
    out = []
    if track is None:
        return [{} for _ in queries]
    ts, rects, ids, zs = track["t"], track["rect"], track["id"], track["z"]
    life, grace = track["life"], track["period"] * _GRACE_PERIODS

    def _present(wid, q):
        for s, e in life.get(wid, ()):
            if (s - grace) <= q <= (e + grace):
                return True
        return False

    latest = {}
    i = 0
    n = int(ts.size)
    for q in queries:
        while i < n and ts[i] <= q:
            latest[int(ids[i])] = (rects[i], int(zs[i]))
            i += 1
        out.append(dict((wid, v) for wid, v in latest.items()
                        if wid >= 0 and _present(wid, q)))
    return out


def _hit(state, x, y):
    """Which window a point landed in: frontmost containing rect, or None.

    Thin wrapper over `geometry.frontmost_owner` (the shared authority): the
    beat sheet only needs the owner id, not the ambiguity flag render's
    grow-the-window resolver uses. Ties and unknown z fall back to the
    SMALLEST containing window (a dialog over its parent); a track with no z
    at all is not treated as "everything is frontmost".
    """
    return geometry.frontmost_owner(state, x, y)[0]


def _window_beats(track):
    """`front`, `open` and `close` beats from the geometry track alone."""
    out = []
    if track is None:
        return out
    ts, ids, zs = track["t"], track["id"], track["z"]
    duration = track["duration"]
    t_start = float(ts[0])
    t_end = float(ts[-1])
    edge = track["period"] * _EDGE_PERIODS
    if track["has_z"]:
        # A window is "brought to front" only on a real TRANSITION to z=0
        # from a known z above it; its first sample is initial state, not an
        # event. Vectorized per window -- the scalar loop over every sample
        # cost 0.9s on a 1.4M-sample track.
        for wid, (tw, zw) in track["groups"].items():
            if tw.size < 2:
                continue
            prev, cur = zw[:-1], zw[1:]
            at = tw[1:]
            mask = (prev > 0) & (cur == 0) & (at >= 0.0) & (at <= duration)
            for t in at[mask].tolist():
                out.append({"kind": "front", "t": float(t),
                            "window_id": int(wid)})
    # One open/close pair per PRESENCE INTERVAL, so a window that was
    # minimised and restored reads as close-then-open rather than silently
    # staying on screen. Only the edges of the track itself are exempt: a
    # window already open when recording began did not "open".
    for wid, spans in track["life"].items():
        for first_t, last_t in spans:
            if first_t > (t_start + edge) and 0.0 <= first_t <= duration:
                out.append({"kind": "open", "t": first_t, "window_id": wid})
            if last_t < (t_end - edge) and 0.0 <= last_t <= duration:
                out.append({"kind": "close", "t": last_t, "window_id": wid})
    return out


def _zoom_level(z):
    """A zoom range's level, defaulting the way `edits._normalize_zoom_range`
    does -- a hand-written edits.json may omit it."""
    try:
        return round(float(z.get("level", 2.0)), 3)
    except (TypeError, ValueError):
        return 2.0


def _overlapping_zooms(zooms, start, end):
    return [z for z in zooms
            if float(z.get("end", 0.0)) > start and float(z.get("start", 0.0)) < end]


def _click_beats(ev, t0, duration, track, sx, sy, ox, oy, max_zoom, params,
                 to_src=None, zooms=None):
    """One beat per click cluster.

    `zoom` is what will ACTUALLY render: the entries in `edits.zooms` that
    cover the cluster. It is not the planner's proposal -- that distinction is
    load-bearing, because every surface except a bare CLI render plans zoom
    ranges ONLY from the materialized edits doc (`camera.build_path` sets
    `auto_cluster = manual_zooms is None`, and the MCP/web resolvers always
    pass a list). A session whose auto-zooms were never materialized, or whose
    zooms an agent has since deleted, renders no camera move there at all, so
    reporting the proposal as the plan would state the opposite of the truth.
    What auto-zoom WOULD propose is still reported, under the honest name
    `zoom_proposal`.
    """
    ct, keep = _finite_media(ev.get("clicks_t"), t0, duration)
    if ct.size == 0:
        return []
    cx = np.asarray(ev.get("clicks_x"), dtype=float).ravel()[keep]
    cy = np.asarray(ev.get("clicks_y"), dtype=float).ravel()[keep]
    order = np.argsort(ct, kind="stable")
    ct, cx, cy = ct[order], cx[order], cy[order]
    owners = [_hit(st, x, y) for st, x, y
              in zip(_states_at(track, ct), cx, cy)]

    P = camera.build_params(max_zoom, params)
    clicks = [(float(t), float(x), float(y)) for t, x, y in zip(ct, cx, cy)]
    out = []
    i = 0
    for cluster in camera.cluster_clicks(clicks, P.chain_gap):
        n = len(cluster)
        seg = owners[i:i + n]
        i += n
        xs = [c[1] for c in cluster]
        ys = [c[2] for c in cluster]
        counts = {}
        for w in seg:
            if w is not None:
                counts[w] = counts.get(w, 0) + 1
        # Source pixels: the space describe_session reports, so this feeds
        # add_zoom's x/y with no conversion. `to_src` is render's OWN mapper
        # when the caller supplied it, which is the only thing that is right
        # on a --capture-window session whose window moved: the crop origin
        # is time-varying there, and a static one puts the bbox off by the
        # window's displacement (measured: 160 against a truth of 60, on a
        # 120px-wide frame).
        if to_src is not None:
            pts = [to_src(t, x, y) for t, x, y in cluster]
            px = [p[0] for p in pts]
            py = [p[1] for p in pts]
        else:
            px = [x * sx - ox for x in xs]
            py = [y * sy - oy for y in ys]
        beat = {
            "kind": "clicks",
            "start": float(cluster[0][0]),
            "end": float(cluster[-1][0]),
            "n": n,
            "bbox": [min(px), min(py), max(px) - min(px), max(py) - min(py)],
        }
        if counts:
            beat["window_id"] = max(sorted(counts), key=lambda w: counts[w])
            if len(counts) > 1:
                beat["windows"] = dict(
                    (str(w), counts[w]) for w in sorted(counts))
            # window_id is a plurality over the clicks that could be
            # attributed at all. Say so when that is not all of them --
            # otherwise a 2-of-20 plurality reads as authoritative.
            attributed = sum(counts.values())
            if attributed < n:
                beat["attributed"] = attributed
        hit = _overlapping_zooms(zooms or [], float(cluster[0][0]),
                                 float(cluster[-1][0]))
        if hit:
            beat["zoom"] = {
                "start": min(float(z["start"]) for z in hit),
                "end": max(float(z["end"]) for z in hit),
                # HOW FAR it pushes in, not just when. "that zoom was too
                # aggressive" is answered by changing this number
                # (`adjust_zoom`), so reading the beat sheet has to show it
                # -- and show the new one back on the next read.
                "level": max(_zoom_level(z) for z in hit),
                "ids": [z["id"] for z in hit if z.get("id")],
            }
        else:
            r = camera.cluster_to_range(cluster, P, float(duration))
            if r is not None:
                beat["zoom_proposal"] = {
                    "start": float(r["startTime"]),
                    "end": float(r["endTime"]),
                    "hold_out": bool(r.get("holdOut")),
                }
        out.append(beat)
    return out


def _scroll_beats(ev, t0, duration, track):
    st, keep = _finite_media(ev.get("scrolls_t"), t0, duration)
    if st.size == 0:
        return []
    sxs = np.asarray(ev.get("scrolls_x"), dtype=float).ravel()[keep]
    sys_ = np.asarray(ev.get("scrolls_y"), dtype=float).ravel()[keep]
    order = np.argsort(st, kind="stable")
    st, sxs, sys_ = st[order], sxs[order], sys_[order]
    owners = [_hit(s, x, y) for s, x, y in zip(_states_at(track, st), sxs, sys_)]
    out = []
    for a, b in _runs(st, SCROLL_GAP):
        span = float(st[b - 1] - st[a])
        counts = {}
        for w in owners[a:b]:
            if w is not None:
                counts[w] = counts.get(w, 0) + 1
        beat = {"kind": "scroll", "start": float(st[a]), "end": float(st[b - 1]),
                "n": int(b - a)}
        if span > 0 and (b - a) > 1:
            # (n-1) intervals across n ticks -- using n inflates a short run
            # by up to 2x (two ticks 0.1s apart are 10/s, not 20/s).
            beat["rate"] = (b - a - 1) / span
        if counts:
            beat["window_id"] = max(sorted(counts), key=lambda w: counts[w])
        out.append(beat)
    return out


def _typing_beats(ev, t0, duration):
    """Key-activity bursts.

    Deliberately carries no window_id: key lines record a quantized tick and
    nothing else -- their x/y is padded-in last-cursor position, not a real
    sample (see geometry.load_events) -- so attributing typing to a window by
    position would be a confident guess about the thing the log refuses to
    record.
    """
    kt, _ = _finite_media(ev.get("keys_t"), t0, duration)
    if kt.size == 0:
        return []
    kt = np.sort(kt)
    return [{"kind": "typing", "start": float(kt[a]), "end": float(kt[b - 1]),
             "n": int(b - a)}
            for a, b in _runs(kt, TYPING_GAP)]


def _idle_beats(ev, t0, duration):
    """Dead air, from the same detector `--speedup` uses to pick its spans."""
    if not duration or duration <= 0:
        return []
    clicks, _ = _finite_media(ev.get("clicks_t"), t0, duration)
    ups, _ = _finite_media(ev.get("ups_t"), t0, duration)
    keys, _ = _finite_media(ev.get("keys_t"), t0, duration)
    scrolls, _ = _finite_media(ev.get("scrolls_t"), t0, duration)
    moves, mkeep = _finite_media(ev.get("moves_t"), t0, duration)
    mx = np.asarray(ev.get("moves_x") if ev.get("moves_x") is not None else [],
                    dtype=float).ravel()
    my = np.asarray(ev.get("moves_y") if ev.get("moves_y") is not None else [],
                    dtype=float).ravel()
    if mx.size == mkeep.size and my.size == mkeep.size:
        mx, my = mx[mkeep], my[mkeep]
    else:
        # Positions out of step with times: drop the move track rather than
        # feed activity_times mismatched arrays. Clicks/keys/scrolls still
        # carry the idle detection.
        moves = mx = my = np.zeros(0, dtype=float)
    act = retime.activity_times(clicks, moves, mx, my, ups, keys, scrolls)
    drags = retime.drag_spans(clicks, ups)
    spans = retime.idle_candidates(act, drags, float(duration),
                                   min_idle=MIN_IDLE)
    return [{"kind": "idle", "start": float(s), "end": float(e)}
            for s, e in spans]


def _thin(beats, max_beats, duration):
    """Cap the list while keeping it representative of the WHOLE take.

    Ranking by span length (the obvious approach, and the first one here) is
    wrong in a way that is worse than the cap it implements: instants
    (`front`/`open`/`close`) and zero-length spans (lone clicks, single
    scroll ticks) all weigh 0, so they lose every tie to any span with
    duration, and among themselves the earlier index wins. The result on a
    long take is that window changes survive up to some point in the timeline
    and then stop dead -- measured on an 8x-tiled real session, the last kept
    instant was t=829s of a 1923s take. An agent reads that as "they stopped
    switching windows", which is a confident lie about the tail of the
    recording.

    So: stratify by TIME, keep the most informative beat in each stratum.
    Uniform coverage end to end, and within a stratum a click cluster beats a
    bare front change.
    """
    n_keep = int(max_beats)
    span = float(duration) if duration and duration > 0 else None
    if span is None:
        # No clock to stratify against: fall back to an even index sample,
        # which is still uniform over the take.
        step = len(beats) / float(n_keep)
        keep = sorted(set(int(i * step) for i in range(n_keep)))
        return [beats[i] for i in keep], len(beats) - len(keep)

    def _rank(b):
        # Lower is better. Kind first, then longer span.
        kind = _KINDS_BY_PRIORITY.index(b["kind"])
        length = float(b.get("end", b.get("t", 0.0))) - \
            float(b.get("start", b.get("t", 0.0)))
        return (kind, -length)

    buckets = {}
    for i, b in enumerate(beats):
        t = float(b.get("start", b.get("t", 0.0)))
        slot = min(n_keep - 1, int((t / span) * n_keep)) if span > 0 else 0
        cur = buckets.get(slot)
        if cur is None or _rank(b) < _rank(beats[cur]):
            buckets[slot] = i
    keep = sorted(buckets.values())
    # Empty strata leave room; spend it on the best of what's left, so a cap
    # of 300 really does return 300 beats rather than however many slots
    # happened to be occupied.
    if len(keep) < n_keep:
        chosen = set(keep)
        rest = sorted((i for i in range(len(beats)) if i not in chosen),
                      key=lambda i: _rank(beats[i]))
        keep = sorted(chosen | set(rest[:n_keep - len(keep)]))
    return [beats[i] for i in keep], len(beats) - len(keep)


def _round(v, nd=3):
    if isinstance(v, float):
        return round(v, nd)
    if isinstance(v, list):
        return [_round(x, nd) for x in v]
    if isinstance(v, dict):
        return dict((k, _round(x, nd)) for k, x in v.items())
    return v


def _remap_ev_times(ev, media_fn):
    """A shallow copy of `ev` whose event-TIME arrays are mapped by `media_fn`
    (the segmented-take clock: parent-monotonic -> concatenated output time).

    Only the `*_t` timestamp arrays are remapped; the paired x/y/rect/id/z
    arrays are shared unchanged. `media_fn` is shape-preserving (clamp-not-drop),
    so the pairing stays index-aligned. After this the beat families run
    exactly as on a single take, with `t0=0.0` -- because the times are ALREADY
    in output-media seconds. This is what keeps `beats` in step with `render`
    and `describe_session` on a segmented session (one shared mapper).
    """
    out = dict(ev)
    for k in ("clicks_t", "moves_t", "ups_t", "keys_t", "scrolls_t",
              "windows_t"):
        arr = ev.get(k)
        if arr is not None and getattr(arr, "size", len(arr) if arr is not None
                                       else 0):
            out[k] = media_fn(arr)
    return out


def beat_sheet(ev, duration, t0=0.0, scale_x=1.0, scale_y=1.0,
               origin_x=0.0, origin_y=0.0,
               max_zoom=2.0, params=None, max_beats=MAX_BEATS,
               to_src=None, zooms=None, media_fn=None):
    """A time-ordered account of a take, derived from events alone.

    `ev` is what `geometry.load_events` returns; `duration` is media seconds;
    `t0` is `meta["t0_monotonic"]`. `scale_x`/`scale_y` convert recorded
    points to recorded pixels (render derives them per-axis off the RAW file
    dims as `raw_width / logical_w`), and `origin_x`/`origin_y` are the
    record-time capture crop's origin -- together they are render's own
    `to_src`, so a `--capture-window` session reports window space, which is
    the space that session's `describe_session` reports too.
    `max_zoom`/`params` are the render options in effect, so the reported
    zoom spans match what this session would actually render -- pass the same
    values the renderer would get.

    Returns `{"beats": [...], "truncated": int, "notes": [...]}`. Beats are
    sorted by time; each carries `kind` plus either `start`/`end` (spans) or
    `t` (instants). When more than `max_beats` are found the LONGEST spans and
    every instant beat inside them are kept and `truncated` says how many were
    dropped -- a 30-minute take must not silently look like a quiet one.
    """
    duration = float(duration or 0.0)
    # Segmented take: pre-map every event time to the concatenated output
    # timeline (paused gaps deleted) via the ONE shared clock, then run the
    # families with t0=0 -- the times are already output-media seconds. When
    # `media_fn` is None (every non-segmented take) this is a no-op and the
    # path below is byte-identical to before.
    if media_fn is not None:
        ev = _remap_ev_times(ev, media_fn)
        t0 = 0.0
    track = _track(ev, t0, duration)
    notes = []
    if track is None:
        notes.append("no window geometry track: beats carry no window_id, "
                     "and window open/close/front changes are not detectable")
    elif not track["has_z"]:
        notes.append("geometry track has no front-to-back rank: no `front` "
                     "beats, and window attribution falls back to the "
                     "smallest containing window")

    if zooms is not None and not zooms and ev.get("clicks_t") is not None \
            and len(ev["clicks_t"]):
        notes.append("edits.zooms is empty: NOTHING will zoom in a render of "
                     "this session as it stands. Each clicks beat carries a "
                     "`zoom_proposal` -- what auto-zoom would propose there "
                     "-- which becomes real only via add_zoom.")

    beats = []
    beats += _click_beats(ev, t0, duration, track, float(scale_x),
                          float(scale_y), float(origin_x), float(origin_y),
                          max_zoom, params, to_src=to_src, zooms=zooms)
    beats += _scroll_beats(ev, t0, duration, track)
    beats += _typing_beats(ev, t0, duration)
    beats += _idle_beats(ev, t0, duration)
    beats += _window_beats(track)

    def _key(b):
        t = b.get("start", b.get("t", 0.0))
        return (float(t), _KINDS_BY_PRIORITY.index(b["kind"]))

    beats.sort(key=_key)

    truncated = 0
    if max_beats and len(beats) > max_beats:
        beats, truncated = _thin(beats, max_beats, duration)
        notes.append(
            "%d of %d beats omitted to stay under max_beats=%d. The kept "
            "beats are spread evenly across the whole take, so the shape is "
            "preserved but the detail is not; pass a larger max_beats for "
            "the full list." % (truncated, truncated + len(beats), max_beats))

    return {"beats": [_round(b) for b in beats],
            "truncated": int(truncated),
            "notes": notes}


def session_beats(ev, meta, info, crop=None, edits_render=None,
                  max_beats=MAX_BEATS, to_src=None, zooms=None, media_fn=None):
    """`beat_sheet` wired to `meta.json` + a `render.describe_session` payload.

    Keeps the unit derivation in ONE place and derives it the way render
    does: points -> pixels off the RAW file dims (fractional Retina scaling
    is real, hence per-axis), then minus the record-time crop origin. `crop`
    is `render._capture_crop_px`'s (x, y, w, h) or None.

    `to_src` overrides that with render's own mapper, which is the only thing
    correct on a tracked `--capture-window` session (moving window = moving
    crop origin). `zooms` is `edits.zooms` -- what will actually render.
    """
    duration = float(info.get("duration") or 0.0)
    rw = float(info.get("raw_width") or 0.0)
    rh = float(info.get("raw_height") or 0.0)
    try:
        lw = float(meta.get("logical_w") or 0.0)
        lh = float(meta.get("logical_h") or 0.0)
    except (TypeError, ValueError):
        lw = lh = 0.0
    sx = (rw / lw) if (rw > 0 and lw > 0) else 1.0
    sy = (rh / lh) if (rh > 0 and lh > 0) else 1.0
    ox, oy = (float(crop[0]), float(crop[1])) if crop is not None else (0.0, 0.0)
    render_opts = edits_render or {}
    try:
        max_zoom = float(render_opts.get("zoom") or 2.0)
    except (TypeError, ValueError):
        max_zoom = 2.0
    params = {"zoom_speed": render_opts.get("zoom_speed")}
    try:
        t0 = float(meta.get("t0_monotonic") or 0.0)
    except (TypeError, ValueError):
        t0 = 0.0
    return beat_sheet(ev, duration, t0=t0, scale_x=sx, scale_y=sy,
                      origin_x=ox, origin_y=oy, max_zoom=max_zoom,
                      params=params, max_beats=max_beats,
                      to_src=to_src, zooms=zooms, media_fn=media_fn)
