"""Session edit presets: trim + render options persisted to edits.json."""

import copy
import json
import os
import re
import uuid

EDIT_FILENAME = "edits.json"
_MIN_TRIM_SPAN = 1.0 / 120.0
_MIN_RANGE_SPAN = 0.15  # s, minimum span for a manual zoom / suppression range
_MAX_WINDOWS = 4
_MIN_WINDOW_DIM = 8.0  # px; guards a degenerate 0-size crop from reaching render.py
# Smallest editor crop, in source px. Matches `render._capture_crop_px`'s own
# `min_dim` so both crop stages reject a degenerate rect at the same size --
# they land on the same frame and a rect one stage keeps and the other drops
# would be a silent disagreement about what the source space even is.
_MIN_CROP_DIM = 16.0
_MIN_CARD_FRAC = 0.02  # smallest manual card placement, as a canvas fraction
_DEFAULT_ZOOM_LEVEL = 2.0
_MAX_ZOOM_LEVEL = 8.0
_MAX_MARKER_LABEL_LEN = 120

# Auto-zoom materialization parameters. Mirror camera.DEFAULTS so a session
# opened for the first time gets the same ranges the planner would have
# produced ephemerally in the previous model -- except now they land in
# edits.json as normal user-owned zoom entries the user can edit/delete.
_AUTO_ZOOM_PRE_ROLL = 2.5
_AUTO_ZOOM_TAIL = 0.5
_AUTO_ZOOM_CHAIN_GAP = 4.0
_AUTO_ZOOM_MIN_ROOM = 0.5
_DEFAULT_PRESET_ID = "preset-default"
_DEFAULT_PRESET_NAME = "Default"

_FACECAM_POSITIONS = ("bottom-left", "bottom-right", "top-left", "top-right")
_FACECAM_SHAPES = ("circle", "rounded")
# The canonical accepted set for render.window_layout, and the ONLY copy any
# other module should validate against -- window_layout has already shipped a
# bug where one surface accepted a value the next silently coerced back to
# "grid" (docs/architecture.md, "Every render option must reach ALL FOUR sinks").
# framing.py dispatches on these exact strings; anything else grids.
_WINDOW_LAYOUTS = ("grid", "desktop", "feature", "row", "column")

_DEFAULT_RENDER = {
    "zoom": 2.2,
    "zoom_speed": "normal",
    "screen_anim": "focused",
    "offset": 0.0,
    "style": "clean",
    "background": None,
    "click_fx": True,
    "click_color": None,
    "spotlight": False,
    "cursor_fx": False,
    "cursor_size": 1.0,
    # Erase the burned-in system cursor (autocine/eraser.py). OFF by default and
    # emphatically not a free win: it costs a second decode of the source, and
    # where the recording never showed the covered pixels uncovered it has to
    # fall back to an inpaint. It is also only meaningful on a `cursor_mode:
    # "system"` take -- a synthetic-cursor session has nothing burned in, and
    # render.py no-ops there rather than paying the pass.
    "cursor_erase": False,
    "aspect": "auto",
    # Export resolution: caps the output canvas HEIGHT, trading sharpness for
    # a smaller, faster composite (cost is per-pixel per-frame). "auto" keeps
    # the natural canvas -- the sharpness-preserving size a Retina multi-window
    # take drives to 4K -- and is bit-exact with pre-feature output. Otherwise
    # a pixel height ("2160" | "1440" | "1080" | "720"); the render scales the
    # whole canvas DOWN to fit, never up. framing.fit_max_height.
    "resolution": "auto",
    "always_zoomed": False,
    "motion_blur": True,
    # Overview framing: the anti-whip camera pass (camera._apply_overview) --
    # a click cluster spread wider than a max_zoom window is held as a still,
    # lower-zoom overview instead of chasing each click. ON by default, matching
    # camera.py's DEFAULTS. See docs/architecture.md.
    "overview": True,
    # typing_zoom/drag_hold/scroll_zoom/screen_anim are legacy camera no-ops in
    # the rewritten camera (see camera.py DEFAULTS). Kept in the render dict for
    # backward-compat with edits.json written by older builds; the editor no
    # longer surfaces them.
    "typing_zoom": True,
    "drag_hold": True,
    "scroll_zoom": True,
    # Multi-window grid only: bind each drawn rect to the window it overlaps
    # and follow that window when it moves or resizes. Off = the static crops
    # the grid shipped with, bit-exact.
    "window_follow": True,
    # Multi-window only: how the cards are arranged. "grid" is the original
    # uniform rows x cols, and it adapts -- `framing._choose_grid` picks
    # rows x cols against the canvas orientation. "desktop" keeps the
    # windows' relative on-screen arrangement -- left stays left, a big
    # window stays big -- pulls any overlaps apart, squeezes out the dead
    # space and scales the result to fit. "feature", "row" and "column"
    # ignore where the windows sat and build from the cards' aspects alone:
    # a hero card with the rest in a block beside it, one row, one column.
    #
    # Those three do NOT "fill the canvas by construction" -- that claim
    # shipped in this comment and was wrong. EVERY arrangement, "desktop"
    # included, ends in `framing._scale_boxes_into`: ONE uniform scale, so it
    # meets the padding on the axis that binds and leaves a margin on the
    # other. Aspect-locked rectangles can do nothing else, so the pick is
    # decided by the EXPORT ASPECT. The measured coverage table lives above
    # `framing._unit_gap` and is pinned by `test_framing.InkCoverage`; the
    # short version is that "feature" is the safe pick at any aspect (it is
    # the one arrangement that TRANSPOSES -- hero on top with the rest in a
    # row beneath on a tall canvas -- so it is best or within 0.3 points of
    # best on all four canvases measured), "grid" matches it on a vertical
    # export, "desktop" matches it on a wide one, and "row"/"column" are
    # named after an axis and therefore never transpose.
    #
    # Sessions seeded from a record-time multi-window pick default to
    # "desktop"; hand-drawn rects keep "grid". A per-card `layout` override
    # (see `_normalize_card_layout`) still applies on top of whichever ran.
    "window_layout": "grid",
    # Per-card auto-zoom in multi-window mode. OFF by default: the whole point
    # of the compositor is a steady side-by-side layout, and zooming inside a
    # card is a different look, not a strictly better one -- so it stays an
    # explicit choice and the static-crop output stays bit-exact without it.
    # Only ONE card is ever zoomed at a time (camera._arbitrate_card_ranges).
    "window_zoom": False,
    # Window focus: the COMPOSITION camera. Click in a window and the whole
    # framed scene leans toward that card; click again and it pushes in until
    # that card is the frame. Distinct from window_zoom, which zooms the
    # footage INSIDE a card while the cell stays put -- the two compose, and
    # both are off by default so the steady side-by-side layout stays the
    # thing you get unless you ask for movement.
    "window_focus": False,
    # Whole-screen "grow the active window": on a plain whole-screen take with
    # several windows on screen, clicking inside ONE window grows it a bit to
    # overlap its neighbours while the frame eases in ("meeting in the
    # middle") -- instead of an auto-zoom that crops into a neighbour. ON by
    # default: it targets exactly the multi-window recordings the plain camera
    # framed badly, and the product is a CLI-less web app where good behavior
    # must be automatic. Bit-exact off switch for the invariant + when <2
    # windows were recorded.
    "screen_focus": True,
    # macOS's per-window capture indicator, painted out of an occlusion-free
    # (window-native) take. ON by default and by the same reasoning as
    # `screen_focus`: it is an artefact of HOW we capture, not a look anyone
    # chose, and the product is a CLI-less web app where a demo has to come
    # out clean without anyone knowing the artefact has a name. No-op on
    # every other capture mode -- a whole-screen take has no per-window
    # badge -- and the off switch is bit-exact.
    "badge_erase": True,
    # Facecam bubble (only shown when the session actually has a face track;
    # meta.face). Defaults ON so a face-recorded session shows it without
    # extra clicks; position/size/shape are the editor + MCP knobs.
    "facecam": True,
    "facecam_position": "bottom-left",
    "facecam_size": 0.20,
    "facecam_shape": "circle",
    # Ring thickness as a fraction of the bubble diameter. 0 = no ring,
    # which is the default: the drop shadow already separates the bubble,
    # and a white ring reads as a sticker on top of the video.
    "facecam_border": 0.0,
    # Background blur strength for the bubble, 0..1. 0 = off. Keeps the face
    # (centre of the bubble) sharp and softens toward the rim -- a vignette,
    # not a person cutout. See effects.FACECAM_DEFAULTS["blur"].
    "facecam_blur": 0.0,
    # Auto speed-up (Rush): compress idle stretches with a quintic ramp.
    # OFF by default -- it changes output duration, and (per research)
    # the real product ships this as a suggestion. The editor makes it
    # one click; MCP callers can flip render.speedup + save_edits.
    "speedup": False,
    "speedup_rate": 6.0,
    "speedup_silence_gate": True,
    "speedup_motion_gate": True,
    "fade": 0.0,
    "music": None,
    # Event SFX. Tri-state, and NULL IS ON: null/"" means the built-in
    # synthesized sound, "off" silences that kind, anything else is a path
    # to a sound file. Sounds are on by default (the product ships polish
    # you turn down, not polish you have to discover), so "no opinion
    # recorded" has to resolve to the built-in -- which also means every
    # edits.json written before this feature picks the sounds up.
    "click_sound": None,
    "key_sound": None,
    "sfx_volume": 1.0,
    "gif": False,
    "gif_fps": 15,
    "gif_width": 1000,
}

# 0 silences the bed entirely; 2.0 is already loud against a voiceover, and
# anything past it only invites clipping in sfx.build_bed's final clamp.
_MAX_SFX_VOLUME = 2.0

_MIN_SPEEDUP_RATE = 1.5
_MAX_SPEEDUP_RATE = 12.0

_ASPECT_PATTERN = re.compile(r"^(\d+\s*:\s*\d+|\d+\s*x\s*\d+)$")

_DEFAULT_TRIM = {
    "start": 0.0,
    "end": None,  # None means "through the end of the clip".
}


def edits_path(session_dir):
    return os.path.join(session_dir, EDIT_FILENAME)


def has_edits(session_dir):
    return os.path.isfile(edits_path(session_dir))


def default_edits():
    base_render = copy.deepcopy(_DEFAULT_RENDER)
    return {
        "version": 1,
        "rev": 0,
        "trim": copy.deepcopy(_DEFAULT_TRIM),
        "render": base_render,
        "active_preset_id": _DEFAULT_PRESET_ID,
        "presets": [{
            "id": _DEFAULT_PRESET_ID,
            "name": _DEFAULT_PRESET_NAME,
            "render": copy.deepcopy(base_render),
        }],
        "zooms": [],
        "suppressed": [],
        "markers": [],
        "speedups": [],
        "cuts": [],
        "windows": [],
        "focus": [],
        # None = uncropped, and the render path is then byte-identical to the
        # one that shipped before this field existed.
        "crop": None,
        # Flipped to True after the first successful load that materializes
        # click-cluster zooms into the `zooms` array. Prevents re-materializing
        # (which would resurrect ranges the user deleted).
        "auto_zooms_initialized": False,
        "capture_windows_initialized": False,
        "focus_initialized": False,
        "focus_plan_version": _FOCUS_PLAN_VERSION,
    }


def _cluster_click_times(click_times, chain_gap):
    if not click_times:
        return []
    ts = sorted(float(t) for t in click_times)
    out = [[ts[0]]]
    for t in ts[1:]:
        if t - out[-1][-1] <= chain_gap:
            out[-1].append(t)
        else:
            out.append([t])
    return out


def auto_zoom_proposals(click_times, duration,
                        pre_roll=_AUTO_ZOOM_PRE_ROLL,
                        tail=_AUTO_ZOOM_TAIL,
                        chain_gap=_AUTO_ZOOM_CHAIN_GAP,
                        min_room=_AUTO_ZOOM_MIN_ROOM,
                        level=_DEFAULT_ZOOM_LEVEL):
    """Compute zoom-range proposals from a list of click times.

    Ranges are the same shape edits.json stores manual zooms in
    (`{start, end, x, y, level}`) with x=y=None so the runtime treats
    them as follow-click-groups. A trailing LONE click without room for
    a zoom-out arc is dropped (matches the reference behavior on the
    reference project's trailing lone click); a trailing multi-click
    cluster is real activity, so it zooms and HOLDS to the clip end
    instead of being discarded -- users click right up until they stop
    recording, so this is the common case. Mirrors
    `camera.cluster_to_range`.
    """
    if duration is None or duration <= 0 or not click_times:
        return []
    clusters = _cluster_click_times(click_times, chain_gap)
    out = []
    for cl in clusters:
        first, last = cl[0], cl[-1]
        no_room = (duration - last) < min_room
        if no_room and len(cl) <= 1:
            continue  # lone trailing click: no zoom
        start = max(0.0, first - pre_roll)
        end = float(duration) if no_room else min(float(duration), last + tail)
        out.append({"start": float(start), "end": float(end),
                    "x": None, "y": None, "level": float(level),
                    # Marks this as an auto-generated (not user-drawn) zoom, so
                    # render's whole-screen "grow the active window" reshape may
                    # snap it to the clicked window. Preserved by
                    # `_normalize_zoom_range`; a user-drawn zoom omits it.
                    "auto": True})
    return out


def initialize_auto_zooms(edits_doc, click_times, duration):
    """Materialize auto-zoom proposals into edits_doc['zooms'] once.

    Idempotent: if `auto_zooms_initialized` is already True, returns
    the doc unchanged. Otherwise appends any proposal that doesn't
    already overlap an existing zoom and flips the flag.

    Returns (doc, changed_bool). The caller writes back only if changed.
    """
    doc = copy.deepcopy(edits_doc) if edits_doc is not None else default_edits()
    if doc.get("auto_zooms_initialized"):
        return doc, False
    proposals = auto_zoom_proposals(click_times, duration)
    existing = list(doc.get("zooms") or [])

    def _overlaps(p, e):
        return not (p["end"] <= e["start"] or p["start"] >= e["end"])

    fresh = [p for p in proposals
             if not any(_overlaps(p, e) for e in existing)]
    if fresh:
        doc["zooms"] = existing + fresh
    doc["auto_zooms_initialized"] = True
    return doc, True


def _seed_capture_window_render(raw_render):
    """The render defaults a record-time multi-window pick opens with.

    Two decisions, both about a session whose cards ARE windows the user
    picked off their own screen:

    - `window_layout: "desktop"` -- it should open looking like the screen it
      came from, not like a flat contact sheet.
    - `window_focus: True` -- the auto-zoom on a multi-window take is the
      COMPOSITION camera: the card being worked in grows in place, and a
      second click in the same burst pushes the whole frame in on it. That is
      the same posture `screen_focus` already takes by default on a plain
      whole-screen take with several windows on it ("meet in the middle"
      rather than crop into a neighbour), so the two multi-window modes now
      agree instead of one of them defaulting to no camera at all.

    `window_zoom` is deliberately NOT flipped. It zooms the footage INSIDE a
    card while the cell stays put -- a different look, and the one the user
    has to ask for (MCP `zoom_style: "inside"` / `--window-zoom`). Both
    remain plain render options, so every off switch stays exactly where it
    was: this seeds the session's STARTING value, it does not change what
    either flag means or what `default_edits()` returns.
    """
    render = dict(raw_render or {})
    render["window_layout"] = "desktop"
    render["window_focus"] = True
    return render


def initialize_capture_windows(edits_doc, specs, multi_native=False):
    """Materialize a record-time multi-window pick into edits_doc['windows'].

    `specs` is what `render.capture_window_specs` returns: source-pixel rects
    each carrying the real macOS `window_id` they came from, in pick order.

    Idempotent by the same trick `initialize_auto_zooms` uses -- a flag on
    the doc -- but with a second guard: it never overwrites a non-empty
    `windows` array, so re-opening a session after you have rearranged or
    deleted cards leaves your work alone.

    Also flips `render.window_layout` to "desktop", because a session whose
    cards ARE the windows you picked off your screen should open looking like
    your screen -- and `render.window_focus` ON, so the auto-zoom on such a
    take is the composition camera (the worked-in card GROWS, then the whole
    frame pushes in) rather than nothing at all. `window_zoom` -- zooming the
    footage INSIDE a card while its cell stays put -- stays off; it is a
    different look, not the default one. See `_seed_capture_window_render`.
    A hand-drawn `--window` set never comes through here and so keeps the
    grid, and keeps both cameras off.

    `multi_native` marks an occlusion-free session (one `raw_i.mov` per window,
    a `capture_channels` manifest, and NO `windows` array -- the cards ARE the
    recorded channels). It has no specs to seed, but its cards are still the
    windows you picked off your screen, so it earns the same "desktop" default:
    otherwise it falls through to the flat "grid", which rows N landscape
    windows into thumbnails so small the text is unreadable. See
    docs/architecture.md ("Default layout").

    Returns (doc, changed_bool). The caller writes back only if changed.
    """
    doc = copy.deepcopy(edits_doc) if edits_doc is not None else default_edits()
    if doc.get("capture_windows_initialized"):
        return doc, False
    doc["capture_windows_initialized"] = True
    if not specs or list(doc.get("windows") or []):
        # Nothing to seed, or the user already has cards. Still flip the flag
        # so this can't re-run and clobber them later. A multi-native session
        # has no specs and no windows array, but its cards are the channels --
        # give it the desktop arrangement instead of the tiny-thumbnail grid.
        if multi_native and not list(doc.get("windows") or []):
            doc["render"] = _seed_capture_window_render(doc.get("render"))
        return doc, True
    seeded = []
    for spec in list(specs)[:_MAX_WINDOWS]:
        if not isinstance(spec, dict):
            continue
        seeded.append({"x": spec.get("x"), "y": spec.get("y"),
                       "w": spec.get("w"), "h": spec.get("h"),
                       "window_id": spec.get("window_id")})
    if not seeded:
        return doc, True
    doc["windows"] = seeded
    doc["render"] = _seed_capture_window_render(doc.get("render"))
    return doc, True


def focus_plan_is_stale(edits_doc):
    """Does this doc still need a focus plan (or a re-plan)?"""
    doc = edits_doc or {}
    return (not doc.get("focus_initialized")
            or _as_int_or(doc.get("focus_plan_version"), 0)
            < _FOCUS_PLAN_VERSION)


def initialize_focus_ranges(edits_doc, ranges):
    """Materialize the composition camera's plan into `edits_doc['focus']`.

    `ranges` is what `camera.plan_focus_ranges` (via
    `render.multi_window_focus_ranges`) returns. Same materialize-once
    contract as `initialize_auto_zooms`: the plan becomes real timeline
    objects the user owns, and the flag stops it re-running and resurrecting
    arcs they deleted. Like `initialize_capture_windows` it also refuses to
    clobber a non-empty array.

    Deliberately NOT gated on `render.window_focus`. The ranges are inert
    while the toggle is off, and materializing regardless means flipping the
    switch on shows the plan immediately rather than only after a reload --
    and means turning it off and on again can't quietly re-plan over edits
    the user already made.

    Returns (doc, changed_bool). The caller writes back only if changed.
    """
    doc = copy.deepcopy(edits_doc) if edits_doc is not None else default_edits()
    if not focus_plan_is_stale(doc):
        return doc, False
    # A version bump re-plans: the stored spans were produced by a planner
    # since found wrong, so honouring them would preserve the bug forever.
    replan = bool(doc.get("focus_initialized"))
    doc["focus_initialized"] = True
    doc["focus_plan_version"] = _FOCUS_PLAN_VERSION
    if not ranges or (list(doc.get("focus") or []) and not replan):
        return doc, True
    fresh = []
    for r in ranges:
        if not isinstance(r, dict):
            continue
        fresh.append({"start": r.get("start"), "end": r.get("end"),
                      "card": r.get("card"), "level": r.get("level")})
    if not fresh:
        return doc, True
    doc["focus"] = fresh
    return doc, True


def _as_float(value, default):
    if default is None and value is None:
        return None
    try:
        if value is None:
            if default is None:
                return None
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        if default is None:
            return None
        return float(default)


def _as_int_or(value, default):
    """Int coercion that never raises -- a hand-edited edits.json can hold
    anything, and a bad value must degrade, not crash the load."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _clean_opt_str(value):
    if value is None:
        return None
    txt = str(value).strip()
    return txt or None


def _as_opt_int(value):
    """An int, or None for anything that isn't cleanly one.

    Bools are rejected on purpose: `True` is an int in Python, and a window
    id of 1 that came from a JSON `true` is a bug wearing a valid value.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _make_unique_id(prefix, existing_ids):
    while True:
        cand = "{}-{}".format(prefix, uuid.uuid4().hex[:8])
        if cand not in existing_ids:
            return cand


def _make_unique_preset_id(existing_ids):
    return _make_unique_id("preset", existing_ids)


def _make_unique_name(base_name, existing_names):
    name = _clean_opt_str(base_name) or "Preset"
    if name not in existing_names:
        return name
    i = 2
    while True:
        cand = "{} {}".format(name, i)
        if cand not in existing_names:
            return cand
        i += 1


def _preset_idx(presets, preset_id):
    for i, p in enumerate(presets):
        if p.get("id") == preset_id:
            return i
    return -1


def _normalize_trim(raw_trim, duration=None):
    raw_trim = raw_trim if isinstance(raw_trim, dict) else {}
    start = max(0.0, _as_float(raw_trim.get("start"), 0.0))
    end_raw = raw_trim.get("end")
    end = None if end_raw in (None, "") else _as_float(end_raw, None)

    if duration is not None:
        duration = max(0.0, float(duration))
        start = min(start, duration)
        if end is None:
            end = duration
        else:
            end = min(max(0.0, end), duration)
    elif end is not None:
        end = max(0.0, end)

    if end is not None and end < start + _MIN_TRIM_SPAN:
        end = start + _MIN_TRIM_SPAN
        if duration is not None:
            end = min(end, duration)
            if end <= start:
                start = max(0.0, duration - _MIN_TRIM_SPAN)
                end = duration
    return {
        "start": float(start),
        "end": None if duration is None and raw_trim.get("end") in (None, "") else float(end),
    }


def _normalize_time_bounds(start_raw, end_raw, duration=None, min_span=_MIN_RANGE_SPAN):
    """Clamp a (start, end) pair to [0, duration] with a minimum span.

    Shared by manual zoom ranges and suppression ranges, which (unlike trim)
    always have a concrete end -- there is no "through the end of the clip"
    sentinel for them.
    """
    start = max(0.0, _as_float(start_raw, 0.0))
    end = _as_float(end_raw, start + min_span)
    if duration is not None:
        duration = max(0.0, float(duration))
        start = min(start, duration)
        end = min(max(0.0, end), duration)
    if end < start + min_span:
        end = start + min_span
        if duration is not None:
            end = min(end, duration)
            if end <= start:
                start = max(0.0, duration - min_span)
                end = duration
    return float(start), float(end)


def _normalize_zoom_range(raw, existing_ids, duration=None):
    raw = raw if isinstance(raw, dict) else {}
    start, end = _normalize_time_bounds(raw.get("start"), raw.get("end"), duration=duration)
    level = _as_float(raw.get("level"), _DEFAULT_ZOOM_LEVEL)
    level = max(1.0, min(_MAX_ZOOM_LEVEL, level))
    x = _as_float(raw.get("x"), None)
    y = _as_float(raw.get("y"), None)
    if x is None or y is None:
        x = y = None
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("zoom", existing_ids)
    out = {
        "id": pid,
        "start": start,
        "end": end,
        "x": x,
        "y": y,
        "level": float(level),
    }
    # Preserve the auto marker (from auto_zoom_proposals) so the whole-screen
    # window reshape can tell an auto-generated zoom from a user-drawn one.
    # Only kept when truthy, so a user-drawn zoom serializes exactly as before.
    if raw.get("auto"):
        out["auto"] = True
    return out


def _normalize_zoom_list(raw_list, duration=None):
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src:
        z = _normalize_zoom_range(item, ids, duration=duration)
        ids.add(z["id"])
        out.append(z)
    out.sort(key=lambda z: (z["start"], z["end"]))
    return out


def _normalize_speedup_range(raw, existing_ids, duration=None):
    """Manual speed-up override range: {id, start, end, mode, rate}.

    mode: "off" (keep this stretch at real time; subtracts from auto
    detection) or "force" (retime this stretch even if auto wouldn't).
    rate: optional; None means "use render.speedup_rate at render time".
    """
    raw = raw if isinstance(raw, dict) else {}
    start, end = _normalize_time_bounds(raw.get("start"), raw.get("end"),
                                        duration=duration)
    mode = str(raw.get("mode") or "off").strip().lower()
    if mode not in ("off", "force"):
        mode = "off"
    rate_raw = raw.get("rate")
    if rate_raw is None:
        rate = None
    else:
        rate = _as_float(rate_raw, _DEFAULT_RENDER["speedup_rate"])
        rate = max(_MIN_SPEEDUP_RATE, min(_MAX_SPEEDUP_RATE, rate))
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("speedup", existing_ids)
    return {"id": pid, "start": start, "end": end, "mode": mode, "rate": rate}


def _normalize_speedup_list(raw_list, duration=None):
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src:
        s = _normalize_speedup_range(item, ids, duration=duration)
        ids.add(s["id"])
        out.append(s)
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


def _normalize_cut_range(raw, existing_ids, duration=None):
    """Cut (ripple delete) range: {id, start, end} in SOURCE seconds.

    The removed span. Deliberately shaped like a suppressed range: no
    per-entry options, and NO overlap merging here -- no list in edits.py
    merges (merging would destroy ids under the editor's adopt, and stale
    an MCP caller's returned cut_id). Overlaps union downstream inside
    retime.TimeMap, where the speedup spans already merge.

    min_span=0: the 0.15s _MIN_RANGE_SPAN floor is a manual-zoom/
    suppression UX rule; for a cut, widening is CONTENT LOSS (removing
    more than the author asked) and would also desync the MCP's snapped
    echo from what actually persists. A sub-frame cut is legitimate here
    -- quantize_cut_spans snaps it outward to one frame -- and a
    degenerate end<=start collapses to zero length, which the render's
    union/quantize simply drops (removes nothing). Non-finite values
    fall back rather than leaking NaN into the doc (the MCP/CLI reject
    them at write time; this is the belt for a hand-edited file).
    """
    raw = raw if isinstance(raw, dict) else {}

    def _finite_or_none(v):
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if f == f and f not in (float("inf"), float("-inf")) else None

    s_raw = _finite_or_none(raw.get("start"))
    e_raw = _finite_or_none(raw.get("end"))
    if s_raw is None or e_raw is None:
        # A half-broken pair (NaN start with a real end, etc.) must not
        # fall back into a REAL removal from 0 -- collapse the whole entry
        # to zero length instead; the render's union drops it.
        s_raw = e_raw = None
    start, end = _normalize_time_bounds(s_raw, e_raw,
                                        duration=duration, min_span=0.0)
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("cut", existing_ids)
    return {"id": pid, "start": start, "end": end}


def _normalize_cut_list(raw_list, duration=None):
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src:
        c = _normalize_cut_range(item, ids, duration=duration)
        ids.add(c["id"])
        out.append(c)
    out.sort(key=lambda c: (c["start"], c["end"]))
    return out


# --- the cuts WRITE CONTRACT, shared by every authoring surface ----------
#
# Normalization above is defensive and never raises -- it is the read path,
# and a hand-edited file must not brick the editor. These guards are the
# opposite: they are the WRITE path, and they reject loudly.
#
# They lived in mcp_server until the web editor learned to author a cut,
# at which point "the MCP validates, the browser does not" became a way to
# save a document the renderer refuses. Four sinks now share them: the MCP
# tools, `save_session_edits`, `studio.py render --cut`, and (through the
# first two) the editor. The message strings are load-bearing -- they are
# what tests/test_mcp_server.py pins, so moving them here byte-identical is
# the proof the port was faithful.
#
# The range cap aligns with retime.DEFAULTS["max_spans"] -- documented there
# as a filter-graph size bound -- and is a loud reject, never a truncation
# (a truncated cut list silently ships un-cut content). CUT_MIN_KEPT_SEC
# rejects a cut set that would swallow (almost) the whole trim window at
# WRITE time, instead of 40 seconds into an export (render's zero-output
# RuntimeError stays as the belt).
CUT_MAX_RANGES = 64
CUT_MIN_KEPT_SEC = 0.25


class CutsError(ValueError):
    """A cut set that must not be written. Each surface re-raises this as
    its own kind of refusal (ToolError / a 400 / a CLI exit) so the caller
    sees one vocabulary, but the RULE lives in one place."""


def cut_summary(cuts_list, trim, duration, fps):
    """(merged, removed_sec, output_duration) for a prospective cuts list.

    Runs through the SAME quantizer render uses (retime.quantize_cut_spans),
    so what a surface echoes is exactly what the encoder will remove.
    `output_duration` is the trim window minus the removed overlap (None
    when the session duration is unknown).

    retime is imported lazily on purpose: it pulls numpy, and edits.py is
    deliberately import-light so `load_edits` stays cheap for the library
    listing and every settings-shaped caller.
    """
    from . import retime

    merged = retime.quantize_cut_spans(
        [(float(c["start"]), float(c["end"])) for c in cuts_list],
        fps, duration=duration)
    t0 = max(0.0, float((trim or {}).get("start") or 0.0))
    t1 = (trim or {}).get("end")
    if t1 is None:
        t1 = float(duration) if duration else None
    else:
        t1 = float(t1)
    if t1 is None:
        removed = sum(b - a for (a, b) in merged)
        out_dur = None
    else:
        removed = sum(min(b, t1) - max(a, t0)
                      for (a, b) in merged if min(b, t1) > max(a, t0))
        out_dur = max(0.0, (t1 - t0) - removed)
    return merged, removed, out_dur


def validate_cuts(merged, out_dur):
    """Raise CutsError if this merged cut set must not be written."""
    if len(merged) > CUT_MAX_RANGES:
        raise CutsError(
            "too many cut ranges: {} after merging (max {}). Merge "
            "neighbouring cuts into wider ranges.".format(
                len(merged), CUT_MAX_RANGES))
    if out_dur is not None and out_dur < CUT_MIN_KEPT_SEC:
        raise CutsError(
            "these cuts would remove (almost) the whole trim window -- "
            "less than {:.2f}s would remain. Shrink a cut or clear the "
            "trim first.".format(CUT_MIN_KEPT_SEC))


def plan_cuts(cuts_list, trim, duration, fps):
    """Summarize AND validate in one call. Returns cut_summary's triple.

    Keyed on what is actually REMOVED, not on the raw list and not on the
    merged spans. A document whose cuts all fall outside the trim window
    takes nothing out of the export, so it must not be rejected for a
    short trim window that the trim handles themselves are allowed to
    create (their own floor is smaller than CUT_MIN_KEPT_SEC) — `merged`
    alone would reject exactly that case, since a cut at 5-6s is still a
    merged span when the trim is 0-0.1s.
    """
    merged, removed, out_dur = cut_summary(cuts_list, trim, duration, fps)
    if removed > 0:
        validate_cuts(merged, out_dur)
    return merged, removed, out_dur


def _normalize_suppressed_range(raw, existing_ids, duration=None):
    raw = raw if isinstance(raw, dict) else {}
    start, end = _normalize_time_bounds(raw.get("start"), raw.get("end"), duration=duration)
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("suppress", existing_ids)
    return {"id": pid, "start": start, "end": end}


def _normalize_suppressed_list(raw_list, duration=None):
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src:
        s = _normalize_suppressed_range(item, ids, duration=duration)
        ids.add(s["id"])
        out.append(s)
    out.sort(key=lambda s: (s["start"], s["end"]))
    return out


_FOCUS_LEVELS = ("focus", "full")

# Bump when a planner change makes previously-materialized `edits.focus`
# spans WRONG rather than merely different -- `initialize_focus_ranges` then
# re-plans that session on next open instead of honouring a stale array.
#
# v2: the live-preview helpers tracked windows at the wrong scale on Retina
#     takes, so clicks were attributed to no card at all and the stored plan
#     was a near-empty stub (measured: 10 of 11 clicks unattributed).
#
# A bump REPLACES the stored spans. That is the point -- they are wrong --
# but it means hand edits to focus spans do not survive one. Acceptable
# while nothing but the planner writes that array; if an editing UI lands,
# give user-touched spans a marker and preserve those.
_FOCUS_PLAN_VERSION = 2


def _normalize_focus_range(raw, existing_ids, duration=None):
    """One `edits.focus` entry: which card owns the composition, when, and
    how hard.

    `level` stays SYMBOLIC ("focus"/"full") rather than a number wherever it
    can, because the two stages are derived from the card's own cell -- a
    range retargeted to a differently-sized card, or the same project exported
    at another aspect, then still means "fill the frame" instead of a zoom
    factor that was only ever right for one layout. A number is accepted for a
    hand-set level and clamped to [0, 1] -- the SAME range
    `camera._normalize_focus_manual` reads a numeric level as: a 0..1 emphasis
    AMOUNT the layout blends by (0 = no lean, 1 = the fully focused layout),
    NOT a zoom factor. (This used to floor at 1.0, which silently turned every
    numeric level into the strongest setting once the camera clamped it back to
    1.0 -- so `level: 0.25` meaning "a quieter grow" rendered as "full".)
    """
    raw = raw if isinstance(raw, dict) else {}
    start, end = _normalize_time_bounds(raw.get("start"), raw.get("end"),
                                        duration=duration)
    try:
        card = max(0, int(raw.get("card")))
    except (TypeError, ValueError):
        card = 0
    level = raw.get("level")
    if isinstance(level, str) and level in _FOCUS_LEVELS:
        pass
    elif level is None:
        level = "focus"
    else:
        try:
            level = max(0.0, min(1.0, float(level)))
        except (TypeError, ValueError):
            level = "focus"
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("focus", existing_ids)
    return {"id": pid, "start": start, "end": end, "card": card,
            "level": level}


def _normalize_focus_list(raw_list, duration=None):
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src:
        f = _normalize_focus_range(item, ids, duration=duration)
        ids.add(f["id"])
        out.append(f)
    out.sort(key=lambda f: (f["start"], f["end"]))
    return out


def _normalize_marker(raw, existing_ids, duration=None):
    raw = raw if isinstance(raw, dict) else {}
    t = max(0.0, _as_float(raw.get("time"), 0.0))
    if duration is not None:
        t = min(t, max(0.0, float(duration)))
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("marker", existing_ids)
    label = _clean_opt_str(raw.get("label"))
    if label is not None:
        label = label[:_MAX_MARKER_LABEL_LEN]
    return {"id": pid, "time": float(t), "label": label}


def _normalize_marker_list(raw_list, duration=None):
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src:
        m = _normalize_marker(item, ids, duration=duration)
        ids.add(m["id"])
        out.append(m)
    out.sort(key=lambda m: (m["time"], m["id"]))
    return out


def _normalize_card_layout(raw):
    """One manual card placement, as 0-1 CANVAS fractions, or None.

    Normalized rather than absolute pixels because this describes a position
    in the OUTPUT, whose size changes with `--aspect` and the render preset --
    a fraction survives all of that, where a pixel rect would silently mean
    something different at 1080p and in a 9:16 export.

    Returned only when every field parses, so a half-written override falls
    back to auto-layout rather than placing a card at (0, 0).
    """
    if not isinstance(raw, dict):
        return None
    vals = {}
    for key in ("x", "y", "w", "h"):
        v = _as_float(raw.get(key), None)
        if v is None:
            return None
        vals[key] = v
    if vals["w"] <= 0.0 or vals["h"] <= 0.0:
        return None
    return {"x": max(0.0, min(1.0, vals["x"])),
            "y": max(0.0, min(1.0, vals["y"])),
            "w": max(_MIN_CARD_FRAC, min(1.0, vals["w"])),
            "h": max(_MIN_CARD_FRAC, min(1.0, vals["h"]))}


def _normalize_window_rect(raw, existing_ids):
    """One cropped-window entry: {id, x, y, w, h} in source-video pixels.

    Deliberately no frame-size parameter (unlike the time-bounds
    normalizers, which clamp against `duration`) -- the existing zoom pin
    (`x`/`y` in `_normalize_zoom_range`) has zero spatial clamping here
    either; spatial safety against the real decoded frame is enforced
    downstream at render time (mirrors `_camera_window`'s clamp), the same
    split this schema follows.
    """
    raw = raw if isinstance(raw, dict) else {}
    x = max(0.0, _as_float(raw.get("x"), 0.0))
    y = max(0.0, _as_float(raw.get("y"), 0.0))
    w = max(_MIN_WINDOW_DIM, _as_float(raw.get("w"), _MIN_WINDOW_DIM))
    h = max(_MIN_WINDOW_DIM, _as_float(raw.get("h"), _MIN_WINDOW_DIM))
    pid = _clean_opt_str(raw.get("id"))
    if not pid or pid in existing_ids:
        pid = _make_unique_id("window", existing_ids)
    out = {"id": pid, "x": float(x), "y": float(y),
           "w": float(w), "h": float(h)}
    # `window_id` (the real macOS window this card is OF) and `layout` (a
    # manual placement) are both emitted ONLY when set. Adding them
    # unconditionally would rewrite the serialized shape of every window
    # entry ever saved; keeping them optional is what makes a hand-drawn
    # rect from before this feature normalize to the exact same five keys.
    wid = _as_opt_int(raw.get("window_id"))
    if wid is not None:
        out["window_id"] = wid
    layout = _normalize_card_layout(raw.get("layout"))
    if layout is not None:
        out["layout"] = layout
    return out


def _normalize_crop(raw):
    """The editor's crop rect -- `{x, y, w, h}` in source-video pixels -- or
    None, which is OFF and must stay the literal pre-feature render path.

    Source-video pixels means the space `describe_session` reports and
    `source_frame` returns, i.e. AFTER any record-time `capture_window` crop.
    That is the same space `edits.windows` rects and zoom pins already live
    in, so the editor needs no second coordinate mapping to place this one.

    Spatially unclamped here, exactly like `_normalize_window_rect`: this
    schema has no frame size to clamp against, and render.py clamps into the
    real decoded frame at the point it slices. What IS enforced is the shape
    -- a non-dict, a non-finite number or a sub-`_MIN_CROP_DIM` side all
    normalize to None (off) rather than to a rect nothing downstream can
    honor, because a crop that silently became 16x16 would be far worse than
    one that silently did nothing.
    """
    if not isinstance(raw, dict):
        return None
    x = _as_float(raw.get("x"), float("nan"))
    y = _as_float(raw.get("y"), float("nan"))
    w = _as_float(raw.get("w"), float("nan"))
    h = _as_float(raw.get("h"), float("nan"))
    vals = (x, y, w, h)
    if any(v != v or v in (float("inf"), float("-inf")) for v in vals):
        return None
    if w < _MIN_CROP_DIM or h < _MIN_CROP_DIM:
        return None
    return {"x": max(0.0, float(x)), "y": max(0.0, float(y)),
            "w": float(w), "h": float(h)}


def normalize_crop(raw):
    """`_normalize_crop` for callers outside this module.

    The web editor's live option dict is UNSAVED and has therefore never been
    through `normalize_edits`, so the render-kwargs resolver has to validate
    the crop itself rather than trust it -- the same reason `window_layout`
    is gated against `_WINDOW_LAYOUTS` there.
    """
    return _normalize_crop(raw)


def _normalize_window_list(raw_list):
    """Up to `_MAX_WINDOWS` window entries, order preserved.

    Order is grid position, not a timeline -- unlike every other list in
    this file, this one must NOT be sorted.
    """
    out = []
    ids = set()
    src = raw_list if isinstance(raw_list, list) else []
    for item in src[:_MAX_WINDOWS]:
        w = _normalize_window_rect(item, ids)
        ids.add(w["id"])
        out.append(w)
    return out


def _normalize_channel_layouts(raw):
    """Per-CHANNEL manual card placements for an occlusion-free multi-native
    take -- POSITIONAL: slot i is channel i (`meta.capture_channels[i]`), up to
    `_MAX_WINDOWS`. A slot is a `_normalize_card_layout` dict (0-1 CANVAS
    fractions) or `None` (that card keeps its preset placement).

    This does NOT reuse `_normalize_window_list`, which COMPACTS the list: a
    null slot must survive AT ITS INDEX, or every later override shifts onto
    the wrong channel. Malformed entries fall back to `None` (auto) for that
    slot, never a (0,0) placement (`_normalize_card_layout`'s all-or-nothing).

    Absent / not-a-list -> `[]` (every card auto, the None state). Trailing
    nulls are trimmed so an all-auto list canonicalizes to `[]` and the
    round-trip is idempotent; interior nulls are preserved.
    """
    if not isinstance(raw, list):
        return []
    out = [_normalize_card_layout(item) for item in raw[:_MAX_WINDOWS]]
    while out and out[-1] is None:
        out.pop()
    return out


def _normalize_scene_layouts(raw):
    """Per-SCENE manual card placements for a scene take (`capture_scenes`) --
    a dict keyed by scene-index STRING, each value a positional channel-indexed
    null-preserving list (exactly `_normalize_channel_layouts`). A missing scene
    key means that scene is fully auto; a null slot means that card keeps its
    preset. Empty / all-null scene lists are dropped so `{}` canonicalizes
    "everything auto" and the round-trip is idempotent.

    The scene index is stable (`capture_scenes` is frozen at record time), so
    the key already separates a window that recurs across scenes -- binding
    stays positional within a scene, no channel id needed.
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    for key, val in raw.items():
        try:
            s = int(key)
        except (TypeError, ValueError):
            continue
        if s < 0:
            continue
        lay = _normalize_channel_layouts(val)
        if lay:
            out[str(s)] = lay
    return out


def _normalize_hidden_channels(raw):
    """Captured cards the user REMOVED from the render (a reversible hide) --
    a list of channel FILE basenames (e.g. "raw_2.mov"). Keyed by file, not
    index, because a scene take's channel recurs across scenes under the same
    file and one hide should drop it from EVERY scene; it is also mode-agnostic
    (multi-native `capture_channels` and scene `capture_scenes` both name a
    `file` per channel). Order-preserving, deduped, capped at `_MAX_WINDOWS`.

    Absent / not-a-list -> `[]` (nothing hidden, the off state, byte-identical
    to today). The render applies its own guards -- it never hides the session
    anchor (scene 0's channel 0) and never empties a scene -- so a stale or
    over-eager entry here can only ever be a no-op, never a broken take.
    """
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, str) and item and item not in out:
            out.append(item)
        if len(out) >= _MAX_WINDOWS:
            break
    return out


def marker_chapters(edits_obj, duration=None):
    """Derive chapter ranges from markers within the active trim window.

    Markers are treated as chapter *starts*; each chapter runs until the next
    marker (strictly later in time) or trim end. The chapter id is the marker id
    so both manual UI and automation can address the same stable identifier.
    """
    normalized = normalize_edits(edits_obj or {}, duration=duration)
    trim = normalized.get("trim", {})
    start = max(0.0, _as_float(trim.get("start"), 0.0))
    end = trim.get("end")
    if end is None:
        if duration is not None:
            end = max(start, float(duration))
        else:
            marker_times = [float(m.get("time", start)) for m in normalized.get("markers", [])]
            end = max([start] + marker_times)
    end = max(start, float(end))
    if end <= start:
        return []

    in_trim = [
        m for m in normalized.get("markers", [])
        if start <= float(m.get("time", 0.0)) < end
    ]
    chapters = []
    last_start = None
    for idx, marker in enumerate(in_trim):
        marker_start = max(start, float(marker.get("time", start)))
        if last_start is not None and marker_start <= (last_start + 1e-6):
            continue
        marker_id = marker.get("id")
        next_marker_id = None
        marker_end = end
        for j in range(idx + 1, len(in_trim)):
            next_t = float(in_trim[j].get("time", marker_start))
            if next_t > marker_start + 1e-6:
                marker_end = min(end, next_t)
                next_marker_id = in_trim[j].get("id")
                break
        if marker_end <= marker_start:
            continue
        label = _clean_opt_str(marker.get("label")) or "Chapter {}".format(len(chapters) + 1)
        chapters.append({
            "id": marker_id,
            "marker_id": marker_id,
            "next_marker_id": next_marker_id,
            "label": label,
            "start": float(marker_start),
            "end": float(marker_end),
            "duration": float(marker_end - marker_start),
        })
        last_start = marker_start
    return chapters


def _normalize_aspect(value):
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("", "auto", "source", "native"):
        return "auto"
    if _ASPECT_PATTERN.match(text):
        return text
    return "auto"


# Export-resolution presets: the output canvas HEIGHT cap. "auto" = the
# natural (sharpness-preserving) canvas. Kept a small closed set so the UI
# and the render agree; an unknown value falls back to "auto" (never crash,
# never a surprise size) -- the same philosophy as _normalize_aspect.
_RESOLUTION_CHOICES = ("auto", "2160", "1440", "1080", "720")


def _normalize_resolution(value):
    if value is None:
        return "auto"
    text = str(value).strip().lower()
    if text in ("", "auto", "source", "native"):
        return "auto"
    # tolerate "1080p" / "1080P"
    if text.endswith("p"):
        text = text[:-1]
    return text if text in _RESOLUTION_CHOICES else "auto"


def resolution_max_height(value):
    """The pixel height a `resolution` value caps to, or None for 'auto'.
    The single place the string preset becomes the `max_height` the render
    understands, so the editor, CLI and MCP cannot disagree."""
    text = _normalize_resolution(value)
    if text == "auto":
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _normalize_render(raw_render):
    raw_render = raw_render if isinstance(raw_render, dict) else {}
    style = str(raw_render.get("style", _DEFAULT_RENDER["style"]))
    if style not in ("clean", "framed"):
        style = _DEFAULT_RENDER["style"]
    bg = _clean_opt_str(raw_render.get("background"))
    if bg and style == "clean":
        style = "framed"
    zoom = max(1.0, _as_float(raw_render.get("zoom"), _DEFAULT_RENDER["zoom"]))
    speed = str(raw_render.get("zoom_speed") or "normal").strip().lower()
    if speed not in ("slow", "normal", "fast"):
        speed = "normal"
    anim = str(raw_render.get("screen_anim") or "focused").strip().lower()
    if anim not in ("focused", "smooth"):
        anim = "focused"
    return {
        "zoom": float(zoom),
        "zoom_speed": speed,
        "screen_anim": anim,
        "offset": _as_float(raw_render.get("offset"), _DEFAULT_RENDER["offset"]),
        "style": style,
        "background": bg,
        "click_fx": _as_bool(raw_render.get("click_fx"), _DEFAULT_RENDER["click_fx"]),
        "click_color": _clean_opt_str(raw_render.get("click_color")),
        "spotlight": _as_bool(raw_render.get("spotlight"), _DEFAULT_RENDER["spotlight"]),
        "cursor_fx": _as_bool(raw_render.get("cursor_fx"), _DEFAULT_RENDER["cursor_fx"]),
        "cursor_size": max(0.2, min(5.0, _as_float(raw_render.get("cursor_size"),
                                                    _DEFAULT_RENDER["cursor_size"]))),
        "cursor_erase": _as_bool(raw_render.get("cursor_erase"),
                                 _DEFAULT_RENDER["cursor_erase"]),
        "aspect": _normalize_aspect(raw_render.get("aspect")),
        "resolution": _normalize_resolution(raw_render.get("resolution")),
        "always_zoomed": _as_bool(raw_render.get("always_zoomed"),
                                  _DEFAULT_RENDER["always_zoomed"]),
        "motion_blur": _as_bool(raw_render.get("motion_blur"),
                                _DEFAULT_RENDER["motion_blur"]),
        "overview": _as_bool(raw_render.get("overview"),
                             _DEFAULT_RENDER["overview"]),
        "typing_zoom": _as_bool(raw_render.get("typing_zoom"),
                                _DEFAULT_RENDER["typing_zoom"]),
        "drag_hold": _as_bool(raw_render.get("drag_hold"),
                              _DEFAULT_RENDER["drag_hold"]),
        "scroll_zoom": _as_bool(raw_render.get("scroll_zoom"),
                                _DEFAULT_RENDER["scroll_zoom"]),
        "window_follow": _as_bool(raw_render.get("window_follow"),
                                  _DEFAULT_RENDER["window_follow"]),
        "window_layout": (raw_render.get("window_layout")
                          if raw_render.get("window_layout") in _WINDOW_LAYOUTS
                          else _DEFAULT_RENDER["window_layout"]),
        "window_zoom": _as_bool(raw_render.get("window_zoom"),
                                _DEFAULT_RENDER["window_zoom"]),
        "window_focus": _as_bool(raw_render.get("window_focus"),
                                 _DEFAULT_RENDER["window_focus"]),
        "screen_focus": _as_bool(raw_render.get("screen_focus"),
                                 _DEFAULT_RENDER["screen_focus"]),
        "badge_erase": _as_bool(raw_render.get("badge_erase"),
                                _DEFAULT_RENDER["badge_erase"]),
        "facecam": _as_bool(raw_render.get("facecam"),
                            _DEFAULT_RENDER["facecam"]),
        "facecam_position": (raw_render.get("facecam_position")
                             if raw_render.get("facecam_position") in _FACECAM_POSITIONS
                             else _DEFAULT_RENDER["facecam_position"]),
        "facecam_size": max(0.08, min(0.5, _as_float(
            raw_render.get("facecam_size"), _DEFAULT_RENDER["facecam_size"]))),
        "facecam_border": max(0.0, min(0.2, _as_float(
            raw_render.get("facecam_border"),
            _DEFAULT_RENDER["facecam_border"]))),
        "facecam_blur": max(0.0, min(1.0, _as_float(
            raw_render.get("facecam_blur"),
            _DEFAULT_RENDER["facecam_blur"]))),
        "facecam_shape": (raw_render.get("facecam_shape")
                          if raw_render.get("facecam_shape") in _FACECAM_SHAPES
                          else _DEFAULT_RENDER["facecam_shape"]),
        "speedup": _as_bool(raw_render.get("speedup"),
                            _DEFAULT_RENDER["speedup"]),
        "speedup_rate": max(_MIN_SPEEDUP_RATE, min(_MAX_SPEEDUP_RATE,
            _as_float(raw_render.get("speedup_rate"),
                      _DEFAULT_RENDER["speedup_rate"]))),
        "speedup_silence_gate": _as_bool(raw_render.get("speedup_silence_gate"),
                                         _DEFAULT_RENDER["speedup_silence_gate"]),
        "speedup_motion_gate": _as_bool(raw_render.get("speedup_motion_gate"),
                                        _DEFAULT_RENDER["speedup_motion_gate"]),
        "fade": max(0.0, _as_float(raw_render.get("fade"), _DEFAULT_RENDER["fade"])),
        "music": _clean_opt_str(raw_render.get("music")),
        "click_sound": _clean_opt_str(raw_render.get("click_sound")),
        "key_sound": _clean_opt_str(raw_render.get("key_sound")),
        "sfx_volume": max(0.0, min(_MAX_SFX_VOLUME,
            _as_float(raw_render.get("sfx_volume"),
                      _DEFAULT_RENDER["sfx_volume"]))),
        "gif": _as_bool(raw_render.get("gif"), _DEFAULT_RENDER["gif"]),
        "gif_fps": int(max(1, min(50, _as_float(raw_render.get("gif_fps"),
                                                _DEFAULT_RENDER["gif_fps"])))),
        "gif_width": int(max(64, min(4000, _as_float(raw_render.get("gif_width"),
                                                      _DEFAULT_RENDER["gif_width"])))),
    }


def _normalize_presets(raw_presets, fallback_render):
    out = []
    ids = set()
    names = set()
    src = raw_presets if isinstance(raw_presets, list) else []
    for item in src:
        if not isinstance(item, dict):
            continue
        pid = _clean_opt_str(item.get("id"))
        if not pid or pid in ids:
            pid = _make_unique_preset_id(ids)
        name = _make_unique_name(item.get("name") or "Preset", names)
        out.append({
            "id": pid,
            "name": name,
            "render": _normalize_render(item.get("render")),
        })
        ids.add(pid)
        names.add(name)
    if out:
        return out
    return [{
        "id": _DEFAULT_PRESET_ID,
        "name": _DEFAULT_PRESET_NAME,
        "render": copy.deepcopy(fallback_render),
    }]


def normalize_edits(raw, duration=None):
    raw = raw if isinstance(raw, dict) else {}
    trim_n = _normalize_trim(raw.get("trim"), duration=duration)
    render_n = _normalize_render(raw.get("render"))
    presets_n = _normalize_presets(raw.get("presets"), render_n)
    zooms_n = _normalize_zoom_list(raw.get("zooms"), duration=duration)
    suppressed_n = _normalize_suppressed_list(raw.get("suppressed"), duration=duration)
    markers_n = _normalize_marker_list(raw.get("markers"), duration=duration)
    speedups_n = _normalize_speedup_list(raw.get("speedups"), duration=duration)
    cuts_n = _normalize_cut_list(raw.get("cuts"), duration=duration)
    windows_n = _normalize_window_list(raw.get("windows"))
    channel_layouts_n = _normalize_channel_layouts(raw.get("channel_layouts"))
    scene_layouts_n = _normalize_scene_layouts(raw.get("scene_layouts"))
    hidden_channels_n = _normalize_hidden_channels(raw.get("hidden_channels"))
    focus_n = _normalize_focus_list(raw.get("focus"), duration=duration)
    crop_n = _normalize_crop(raw.get("crop"))
    active_id = _clean_opt_str(raw.get("active_preset_id"))
    idx = _preset_idx(presets_n, active_id)
    if idx < 0:
        idx = 0
        active_id = presets_n[0]["id"]
    # Keep the currently active preset and top-level render in sync.
    if "render" in raw:
        presets_n[idx]["render"] = copy.deepcopy(render_n)
    else:
        render_n = copy.deepcopy(presets_n[idx]["render"])
    # rev: optimistic-concurrency counter, bumped on every save_edits().
    try:
        rev_n = max(0, int(raw.get("rev", 0)))
    except (TypeError, ValueError):
        rev_n = 0
    return {
        "version": 1,
        "rev": rev_n,
        "trim": trim_n,
        "render": render_n,
        "active_preset_id": active_id,
        "presets": presets_n,
        "zooms": zooms_n,
        "suppressed": suppressed_n,
        "markers": markers_n,
        "speedups": speedups_n,
        "cuts": cuts_n,
        "windows": windows_n,
        "channel_layouts": channel_layouts_n,
        "scene_layouts": scene_layouts_n,
        "hidden_channels": hidden_channels_n,
        "focus": focus_n,
        "crop": crop_n,
        "auto_zooms_initialized": bool(raw.get("auto_zooms_initialized")),
        "capture_windows_initialized": bool(
            raw.get("capture_windows_initialized")),
        "focus_initialized": bool(raw.get("focus_initialized")),
        "focus_plan_version": _as_int_or(raw.get("focus_plan_version"), 0),
    }


def merge_edits(base, patch, duration=None):
    base_n = normalize_edits(base or {}, duration=duration)
    patch = patch if isinstance(patch, dict) else {}
    merged = {
        "version": 1,
        "rev": base_n["rev"],   # rev is never patchable — it tracks saves
        "trim": dict(base_n["trim"]),
        "render": dict(base_n["render"]),
        "active_preset_id": base_n["active_preset_id"],
        "presets": copy.deepcopy(base_n["presets"]),
        "zooms": copy.deepcopy(base_n["zooms"]),
        "suppressed": copy.deepcopy(base_n["suppressed"]),
        "markers": copy.deepcopy(base_n["markers"]),
        "speedups": copy.deepcopy(base_n["speedups"]),
        "cuts": copy.deepcopy(base_n["cuts"]),
        "windows": copy.deepcopy(base_n["windows"]),
        "channel_layouts": copy.deepcopy(base_n["channel_layouts"]),
        "scene_layouts": copy.deepcopy(base_n["scene_layouts"]),
        "hidden_channels": copy.deepcopy(base_n["hidden_channels"]),
        "focus": copy.deepcopy(base_n["focus"]),
        "crop": copy.deepcopy(base_n["crop"]),
        "auto_zooms_initialized": bool(base_n.get("auto_zooms_initialized")),
        "capture_windows_initialized": bool(
            base_n.get("capture_windows_initialized")),
        "focus_initialized": bool(base_n.get("focus_initialized")),
        "focus_plan_version": _as_int_or(
            base_n.get("focus_plan_version"), 0),
    }
    trim_patch = patch.get("trim")
    if isinstance(trim_patch, dict):
        for k in ("start", "end"):
            if k in trim_patch:
                merged["trim"][k] = trim_patch[k]
    # zooms/suppressed/markers are timeline content (like trim), not per-preset
    # "look", so a patch simply replaces the whole array; normalize_edits()
    # below re-validates/clamps/re-ids every entry regardless.
    if "zooms" in patch:
        merged["zooms"] = patch.get("zooms")
    if "suppressed" in patch:
        merged["suppressed"] = patch.get("suppressed")
    if "markers" in patch:
        merged["markers"] = patch.get("markers")
    if "speedups" in patch:
        merged["speedups"] = patch.get("speedups")
    if "cuts" in patch:
        merged["cuts"] = patch.get("cuts")
    if "windows" in patch:
        merged["windows"] = patch.get("windows")
    if "channel_layouts" in patch:
        merged["channel_layouts"] = patch.get("channel_layouts")
    # Whole-key replacement (like windows/channel_layouts). Safe on the web
    # path, where the client always re-POSTs the complete scene_layouts dict; a
    # future MCP writer sending a PARTIAL dict would need a deep per-scene merge.
    if "scene_layouts" in patch:
        merged["scene_layouts"] = patch.get("scene_layouts")
    # Whole-key replacement: the web client re-POSTs the full hidden list.
    if "hidden_channels" in patch:
        merged["hidden_channels"] = patch.get("hidden_channels")
    if "focus" in patch:
        merged["focus"] = patch.get("focus")
    # Membership, not truthiness: null IS the payload when the user
    # resets the crop back to the full frame.
    if "crop" in patch:
        merged["crop"] = patch.get("crop")
    active_changed = False
    if "active_preset_id" in patch:
        merged["active_preset_id"] = patch.get("active_preset_id")
        active_changed = True
    render_patch = patch.get("render")
    render_changed = False
    if isinstance(render_patch, dict):
        for k in _DEFAULT_RENDER.keys():
            if k in render_patch:
                merged["render"][k] = render_patch[k]
                render_changed = True

    active_idx = _preset_idx(merged["presets"], merged["active_preset_id"])
    if active_idx < 0:
        active_idx = 0
        merged["active_preset_id"] = merged["presets"][0]["id"]
    if active_changed and not render_changed:
        merged["render"] = copy.deepcopy(merged["presets"][active_idx]["render"])
    if render_changed:
        merged["presets"][active_idx]["render"] = copy.deepcopy(merged["render"])
    return normalize_edits(merged, duration=duration)


def set_active_preset(base, preset_id, duration=None):
    base_n = normalize_edits(base or {}, duration=duration)
    idx = _preset_idx(base_n["presets"], preset_id)
    if idx < 0:
        raise ValueError("unknown preset id: {}".format(preset_id))
    out = {
        "version": 1,
        "rev": base_n["rev"],
        "trim": dict(base_n["trim"]),
        "render": copy.deepcopy(base_n["presets"][idx]["render"]),
        "active_preset_id": base_n["presets"][idx]["id"],
        "presets": copy.deepcopy(base_n["presets"]),
        "zooms": copy.deepcopy(base_n["zooms"]),
        "suppressed": copy.deepcopy(base_n["suppressed"]),
        "markers": copy.deepcopy(base_n["markers"]),
        "speedups": copy.deepcopy(base_n["speedups"]),
        "cuts": copy.deepcopy(base_n["cuts"]),
        "windows": copy.deepcopy(base_n["windows"]),
        "channel_layouts": copy.deepcopy(base_n["channel_layouts"]),
        "scene_layouts": copy.deepcopy(base_n["scene_layouts"]),
        "hidden_channels": copy.deepcopy(base_n["hidden_channels"]),
        "focus": copy.deepcopy(base_n["focus"]),
        "crop": copy.deepcopy(base_n["crop"]),
        "auto_zooms_initialized": bool(base_n.get("auto_zooms_initialized")),
        "capture_windows_initialized": bool(
            base_n.get("capture_windows_initialized")),
        "focus_initialized": bool(base_n.get("focus_initialized")),
        "focus_plan_version": _as_int_or(
            base_n.get("focus_plan_version"), 0),
    }
    return normalize_edits(out, duration=duration)


def duplicate_preset(base, source_preset_id=None, name=None,
                     source_render=None, select_new=True, duration=None):
    base_n = normalize_edits(base or {}, duration=duration)
    presets = copy.deepcopy(base_n["presets"])
    source_id = _clean_opt_str(source_preset_id) or base_n["active_preset_id"]
    idx = _preset_idx(presets, source_id)
    if idx < 0:
        idx = _preset_idx(presets, base_n["active_preset_id"])
    if idx < 0:
        idx = 0
    src = presets[idx]
    src_render = (_normalize_render(source_render)
                  if isinstance(source_render, dict)
                  else copy.deepcopy(src["render"]))
    ids = set(p["id"] for p in presets)
    names = set(p["name"] for p in presets)
    new_id = _make_unique_preset_id(ids)
    new_name = _make_unique_name(name or (src.get("name", "Preset") + " Copy"), names)
    presets.append({
        "id": new_id,
        "name": new_name,
        "render": src_render,
    })
    out = {
        "version": 1,
        "rev": base_n["rev"],
        "trim": dict(base_n["trim"]),
        "render": dict(base_n["render"]),
        "active_preset_id": base_n["active_preset_id"],
        "presets": presets,
        "zooms": copy.deepcopy(base_n["zooms"]),
        "suppressed": copy.deepcopy(base_n["suppressed"]),
        "markers": copy.deepcopy(base_n["markers"]),
        "speedups": copy.deepcopy(base_n["speedups"]),
        "cuts": copy.deepcopy(base_n["cuts"]),
        "windows": copy.deepcopy(base_n["windows"]),
        "channel_layouts": copy.deepcopy(base_n["channel_layouts"]),
        "scene_layouts": copy.deepcopy(base_n["scene_layouts"]),
        "hidden_channels": copy.deepcopy(base_n["hidden_channels"]),
        "focus": copy.deepcopy(base_n["focus"]),
        "crop": copy.deepcopy(base_n["crop"]),
        "auto_zooms_initialized": bool(base_n.get("auto_zooms_initialized")),
        "capture_windows_initialized": bool(
            base_n.get("capture_windows_initialized")),
        "focus_initialized": bool(base_n.get("focus_initialized")),
        "focus_plan_version": _as_int_or(
            base_n.get("focus_plan_version"), 0),
    }
    if select_new:
        out["active_preset_id"] = new_id
        out["render"] = copy.deepcopy(src_render)
    return normalize_edits(out, duration=duration)


def load_edits(session_dir, duration=None):
    path = edits_path(session_dir)
    if not os.path.isfile(path):
        return normalize_edits(default_edits(), duration=duration)
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:
        raw = {}
    return normalize_edits(raw, duration=duration)


def _write_edits_atomic(path, normalized):
    # tmp + os.replace so a concurrent reader can never see a torn file
    # (load_edits treats unparseable JSON as "no edits", which would silently
    # reset everything).
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(normalized, f, indent=2)
    os.replace(tmp, path)


def save_edits(session_dir, edits_obj, duration=None):
    normalized = normalize_edits(edits_obj, duration=duration)
    # Callers pass a doc derived from a load (rev == disk rev at load time);
    # every save bumps it so other clients can detect external changes.
    normalized["rev"] = int(normalized.get("rev", 0)) + 1
    _write_edits_atomic(edits_path(session_dir), normalized)
    return normalized


def reset_edits(session_dir, duration=None):
    current = load_edits(session_dir, duration=duration)
    normalized = normalize_edits(default_edits(), duration=duration)
    normalized["rev"] = int(current.get("rev", 0)) + 1
    _write_edits_atomic(edits_path(session_dir), normalized)
    return normalized
