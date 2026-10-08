"""Command-line interface: `studio devices | record | render | app | ...`."""

import argparse
import errno
import json
import math
import os
import sys
import webbrowser
from datetime import datetime
from urllib.parse import quote

from . import __version__
from . import devices as dev
from . import edits as _edits
from . import record as rec
from . import render as ren
from . import sck
from . import settings
from . import paths

REC_ROOT = paths.recordings_root()


def _window_line(w):
    """One `devices` listing row for a window entry (rect in points)."""
    return "  [{}] {}  ({:.0f},{:.0f} {:.0f}x{:.0f} pt)".format(
        w["id"], w["label"], w["x"], w["y"], w["w"], w["h"])


def _print_windows():
    """List pickable windows so `record --capture-window <id>` has ids to use.

    Without this the flag would need an id nobody can obtain. Degrades to a
    one-line note when Quartz is missing (dev.list_windows returns [] rather
    than raising) or when nothing on the main display passes the size gates.
    Returns True when at least one window was listed.
    """
    wins = dev.list_windows(exclude_pids=(os.getpid(),))
    print("Windows (front to back, main display only):")
    if not wins:
        if not dev.displays_points():
            print("  (window enumeration unavailable -- Quartz isn't "
                  "importable here)")
        else:
            print("  (no capturable windows right now)")
        return False
    for w in wins:
        print(_window_line(w))
    return True


def _cmd_devices(args):
    d = dev.list_avf_devices()
    print("Video devices:")
    for idx, name in d["video"]:
        tag = "   <- screen capture" if "capture screen" in name.lower() else ""
        print("  [{}] {}{}".format(idx, name, tag))
    print("Audio devices:")
    for idx, name in d["audio"]:
        print("  [{}] {}".format(idx, name))
    w, h, src = dev.main_display_points()
    print("Main display: {:.0f} x {:.0f} points (via {})".format(w, h, src))
    have_windows = _print_windows()
    scr = dev.find_screen_device(d)
    if scr is not None:
        print("\nReady to go:  python3 studio.py record --display {}".format(scr))
        if have_windows:
            print("Just one window:  python3 studio.py record --display {} "
                  "--capture-window <id>".format(scr))
    return 0


def _session_dir(args):
    if args.out:
        return args.out
    return os.path.join(REC_ROOT, datetime.now().strftime("%Y%m%d-%H%M%S"))


def _resolve_style(args, background):
    """A --background implies the framed look unless the user asked otherwise.

    `background` is the resolved value (flag or saved), not `args.background`:
    a background carried over from the session's edits.json still promotes the
    style, or a saved background would paint behind a 'clean' full-bleed frame
    and the same edits.json would look different depending on the surface --
    the ALL FOUR sinks rule in docs/architecture.md.
    """
    style = args.style
    if background and style == "clean":
        style = "framed"
    return style


def _parse_window_spec(spec):
    """Parse one `--window X,Y,W,H` occurrence (source-video pixels)."""
    parts = spec.split(",")
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--window expects X,Y,W,H (got: {!r})".format(spec))
    try:
        x, y, w, h = (float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--window values must be numbers (got: {!r})".format(spec))
    return {"x": x, "y": y, "w": w, "h": h}


def _parse_cut_spec(spec):
    """Parse one `--cut START-END` occurrence (media seconds)."""
    parts = spec.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            "--cut expects START-END seconds (got: {!r})".format(spec))
    try:
        a, b = float(parts[0]), float(parts[1])
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--cut values must be numbers (got: {!r})".format(spec))
    if not (math.isfinite(a) and math.isfinite(b)):
        # float("nan") parses, and NaN then no-ops the b <= a guard below.
        raise argparse.ArgumentTypeError(
            "--cut values must be finite (got: {!r})".format(spec))
    if b <= a:
        raise argparse.ArgumentTypeError(
            "--cut end must be after start (got: {!r})".format(spec))
    return {"start": a, "end": b}


# --facecam-* flag -> FacecamOverlay param name, and the edits.render key the
# flag falls back to when it wasn't passed.
_FACECAM_ARGS = (("facecam_position", "position"),
                 ("facecam_size", "size_frac"),
                 ("facecam_shape", "shape"),
                 ("facecam_blur", "blur"))


def _facecam_enabled(args, saved_render):
    """Whether to composite the bubble: --no-facecam wins, else the session's
    saved `render.facecam`, else on (the pre-edits default)."""
    val = getattr(args, "facecam", None)
    if val is None:
        val = saved_render.get("facecam")
    return True if val is None else bool(val)


def _facecam_params(args, saved_render):
    """FacecamOverlay params from the --facecam-* flags.

    The flags default to None so "the user passed it" is distinguishable from
    "it happens to equal the default"; anything not passed falls back to the
    session's saved render options, so a CLI render doesn't silently undo the
    editor's Facecam panel. Returns None when nothing is set (the overlay then
    uses its own defaults, exactly as before these flags could be omitted).
    """
    p = {}
    for arg_key, param_key in _FACECAM_ARGS:
        val = getattr(args, arg_key, None)
        if val is None:
            val = saved_render.get(arg_key)
        if val is not None:
            p[param_key] = val
    return p or None


def _cursor_params(args, saved_render):
    """CursorFX params from --cursor-size, else the session's saved size.

    None when neither is set, so `effects.CURSOR_DEFAULTS` applies untouched.
    """
    size = getattr(args, "cursor_size", None)
    if size is None:
        size = saved_render.get("cursor_size")
    return {"scale": float(size)} if size else None


def _render_kwargs(args, saved_render=None):
    # saved_render: the session's edits.json `render` block, when there is one
    # (`record --render` has no saved doc yet -- an empty dict is the same as
    # "no session preference", i.e. the plain flag defaults).
    #
    # EVERY option here follows the same precedence: an explicit flag wins,
    # else the saved value, else the pre-edits default. Anything read from
    # `args.<name>` directly (with no saved fallback) silently drops what the
    # web editor or MCP `set_render_options` wrote, so one edits.json makes
    # two different videos depending on which surface exported it -- the
    # ALL FOUR sinks rule in docs/architecture.md. `window_layout` shipped that
    # bug on the web side and this CLI side keeps drifting into the same shape
    # (the reviewed drops: background / aspect / click_color / music /
    # click_sound / speedup{,_rate,_silence_gate,_motion_gate} / always_zoomed
    # / zoom_speed / overview / zoom itself). Flags whose argparse default is
    # already `None` (cursor_fx, cursor_erase, window_zoom, window_focus,
    # screen_focus, facecam) can distinguish "not passed" from "passed the
    # default"; the others were converted to `default=None` sentinels so this
    # resolver can tell them apart too.
    saved_render = saved_render or {}

    # Camera DEFAULTS overrides (params dict). Each falls back to saved:
    # `always_zoomed` and `zoom_speed` are editor knobs (edits._DEFAULT_RENDER
    # carries them), and `overview` has no CLI flag at all -- it is
    # saved-only. drag_hold / screen_anim are legacy no-ops (edits.py:71-72),
    # left alone.
    cam_params = {}
    always_z = getattr(args, "always_zoomed", None)
    if always_z is None:
        always_z = saved_render.get("always_zoomed")
    if always_z:
        cam_params["always_zoomed"] = True
    if not getattr(args, "drag_hold", True):
        cam_params["drag_hold"] = False
    zoom_speed = getattr(args, "zoom_speed", None)
    if zoom_speed is None:
        zoom_speed = saved_render.get("zoom_speed")
    if zoom_speed and zoom_speed != "normal":
        cam_params["zoom_speed"] = zoom_speed
    if getattr(args, "screen_anim", "focused") != "focused":
        cam_params["screen_anim"] = args.screen_anim
    if not saved_render.get("overview", True):
        cam_params["overview"] = False

    # Options whose flag defaults are None (converted from valued/store_true
    # so "not passed" is distinguishable from "passed the default"). Value
    # source: flag if given, else saved, else the pre-edits default (matches
    # `edits._DEFAULT_RENDER`; `render()`'s own signature default is the
    # ultimate belt-and-braces).
    background = (args.background if args.background is not None
                  else saved_render.get("background"))
    click_color = (args.click_color if args.click_color is not None
                   else saved_render.get("click_color"))
    aspect = args.aspect if args.aspect is not None else saved_render.get("aspect")
    # Export resolution: flag if given, else saved, else "auto". Mapped to the
    # render's `max_height` through the ONE converter the editor/MCP also use,
    # so the three surfaces cannot disagree.
    resolution = (getattr(args, "resolution", None)
                  if getattr(args, "resolution", None) is not None
                  else saved_render.get("resolution"))
    max_height = _edits.resolution_max_height(resolution)
    music = args.music if args.music is not None else saved_render.get("music")
    click_sound = (args.click_sound if args.click_sound is not None
                   else saved_render.get("click_sound"))
    key_sound = (args.key_sound if args.key_sound is not None
                 else saved_render.get("key_sound"))
    if getattr(args, "no_sound_fx", False):
        click_sound = key_sound = "off"
    sfx_volume = (args.sfx_volume if args.sfx_volume is not None
                  else float(saved_render.get("sfx_volume")
                             if saved_render.get("sfx_volume") is not None
                             else _edits._DEFAULT_RENDER["sfx_volume"]))
    max_zoom = (args.zoom if args.zoom is not None
                else float(saved_render.get("zoom")
                           or _edits._DEFAULT_RENDER["zoom"]))
    speedup = (args.speedup if args.speedup is not None
               else bool(saved_render.get("speedup", False)))
    speedup_rate = (args.speedup_rate if args.speedup_rate is not None
                    else float(saved_render.get("speedup_rate")
                               or _edits._DEFAULT_RENDER["speedup_rate"]))
    silence_arg = getattr(args, "speedup_silence_gate", None)
    speedup_silence_gate = (silence_arg if silence_arg is not None
                            else bool(saved_render.get(
                                "speedup_silence_gate", True)))
    motion_arg = getattr(args, "speedup_motion_gate", None)
    speedup_motion_gate = (motion_arg if motion_arg is not None
                           else bool(saved_render.get(
                               "speedup_motion_gate", True)))

    kwargs = dict(
        max_zoom=max_zoom,
        # _resolve_style takes the RESOLVED background (flag or saved) so a
        # session-only background still promotes style clean -> framed.
        style=_resolve_style(args, background),
        make_gif=args.gif,
        background=background,
        click_fx=not args.no_clicks,
        click_params=({"color": click_color} if click_color else None),
        spotlight=args.spotlight,
        fade=args.fade,
        music=music,
        click_sound=click_sound,
        key_sound=key_sound,
        sfx_volume=sfx_volume,
        # Falls back to the SAVED options for the same reason cursor_erase
        # does below -- and now with more at stake: on a system-cursor take
        # the two are a PAIR (the eraser lifts the recorded pointer out, the
        # fx draws its replacement). Reading one from edits.json and not the
        # other would export footage with no cursor at all.
        cursor_fx=(args.cursor_fx
                   if getattr(args, "cursor_fx", None) is not None
                   else bool(saved_render.get("cursor_fx", False))),
        cursor_params=_cursor_params(args, saved_render),
        # Falls back to the SAVED option like window_zoom/window_focus do,
        # rather than to the flag's own default: an eraser turned on in the
        # editor has to survive `studio.py render`, or one edits.json makes
        # two different videos depending on which surface exported it.
        cursor_erase=(args.cursor_erase
                      if getattr(args, "cursor_erase", None) is not None
                      else bool(saved_render.get("cursor_erase", False))),
        aspect=aspect,
        max_height=max_height,
        motion_blur=args.motion_blur,
        typing_zoom=args.typing_zoom,
        scroll_zoom=args.scroll_zoom,
        window_follow=args.window_follow,
        window_layout=getattr(args, "window_layout", None)
        or saved_render.get("window_layout") or "grid",
        window_zoom=(getattr(args, "window_zoom", None)
                     if getattr(args, "window_zoom", None) is not None
                     else bool(saved_render.get("window_zoom", False))),
        window_focus=(getattr(args, "window_focus", None)
                      if getattr(args, "window_focus", None) is not None
                      else bool(saved_render.get("window_focus", False))),
        # Whole-screen grow-the-active-window. ON by default (the CLI-less web
        # product wants it automatic), so the saved fallback is True and the
        # only override is --no-screen-focus.
        screen_focus=(args.screen_focus
                      if getattr(args, "screen_focus", None) is not None
                      else bool(saved_render.get("screen_focus", True))),
        badge_erase=(args.badge_erase
                     if getattr(args, "badge_erase", None) is not None
                     else bool(saved_render.get("badge_erase", True))),
        facecam=_facecam_enabled(args, saved_render),
        facecam_params=_facecam_params(args, saved_render),
        speedup=speedup,
        speedup_rate=speedup_rate,
        speedup_silence_gate=speedup_silence_gate,
        speedup_motion_gate=speedup_motion_gate,
        params=(cam_params or None),
        gif_fps=args.gif_fps,
        gif_width=args.gif_width,
    )
    windows = getattr(args, "windows", None)
    if windows:
        # cap at 4, same as edits._MAX_WINDOWS -- render.py/framing.py also
        # tolerate more, but keep the CLI's contract explicit here.
        kwargs["windows"] = windows[:4]
    cuts = getattr(args, "cuts", None)
    if cuts:
        kwargs["cuts"] = cuts
    return kwargs


# Sentinel for "--capture-window was given but can't be honored" -- distinct
# from None, which means "no window requested" (a normal full-display take).
_CW_ERROR = object()

# Matches edits._MAX_WINDOWS -- the compositor caps cards at 4, so
# accepting a fifth pick would only silently drop it later.
_MAX_CAPTURE_WINDOWS = 4


def _resolve_capture_window(args, d, display):
    """Turn `--capture-window <id>` (repeatable) into devices entries.

    Returns a LIST, empty when the flag wasn't passed. `_CW_ERROR` (caller
    exits non-zero) when it was but can't be honored -- silently recording
    full-screen after the user asked for one window is the failure mode worth
    being loud about, since they'd only find out after a 20-minute take. One
    id is a window CROP (auto-zoom survives); two to four is a multi-window
    composition, which is a different feature reached through the same flag
    because from the user's side it is the same sentence with more windows.
    """
    raw = getattr(args, "capture_window", None)
    if raw is None:
        return []
    win_ids = list(raw) if isinstance(raw, (list, tuple)) else [raw]
    if not win_ids:
        return []
    if len(win_ids) > _MAX_CAPTURE_WINDOWS:
        print("at most {} windows can be recorded together (got {})."
              .format(_MAX_CAPTURE_WINDOWS, len(win_ids)), file=sys.stderr)
        return _CW_ERROR
    # Multi-display is out of scope for v1: meta.json's logical_w/h always
    # describe the MAIN display, so cropping a rect out of a secondary
    # display's capture would use the wrong points->pixels scale. Refuse
    # rather than produce a plausible-looking wrong crop.
    primary = dev.find_screen_device(d)
    if primary is not None and display != primary:
        print("--capture-window only works with the primary screen device "
              "(--display {}); window capture on a secondary display isn't "
              "supported yet.".format(primary), file=sys.stderr)
        return _CW_ERROR
    entries = []
    for win_id in win_ids:
        entry = dev.window_rect_points(win_id, exclude_pids=(os.getpid(),))
        if entry is None:
            print("window {} isn't capturable: it's gone, minimized, offscreen, "
                  "too small, or on a secondary display.\nRun `python3 studio.py "
                  "devices` for the current window ids.".format(win_id),
                  file=sys.stderr)
            return _CW_ERROR
        entries.append(entry)
    if len(entries) == 1:
        print("capture window: {} -- bring it forward during the countdown "
              "(the rect is re-read right before recording starts)"
              .format(entries[0]["label"]))
    else:
        print("capture windows ({}):".format(len(entries)))
        for i, entry in enumerate(entries, 1):
            print("  {}. {}".format(i, entry["label"]))
    return entries


def _cmd_record(args):
    d = dev.list_avf_devices()
    display = args.display
    if display is None:
        display = dev.find_screen_device(d)
        if display is None:
            print("Could not auto-detect the screen device; pass --display N "
                  "(run `studio devices`).", file=sys.stderr)
            return 2
    elif not dev.is_screen_device(d, display):
        # avfoundation indices renumber when video devices come and go, so a
        # --display copied from an older `devices` listing can now name a
        # webcam. Warn rather than override: unlike the bar's remembered
        # pick, this number was typed for this run, and silently recording a
        # different device than the one asked for is its own bug.
        print("warning: --display {} isn't a screen device in the current "
              "avfoundation list (indices shift when cameras connect or "
              "disconnect). Run `python3 studio.py devices` to re-check; "
              "the screen is index {}."
              .format(display, dev.find_screen_device(d)), file=sys.stderr)
    if args.mic is None or str(args.mic).lower() == "none":
        mic = None
    else:
        mic = int(args.mic)

    face_idx = None
    if getattr(args, "face", False):
        face_idx = (args.face_index if args.face_index is not None
                    else dev.find_camera_device(d))
        if face_idx is None:
            print("No webcam found; recording without facecam.", file=sys.stderr)

    picked = _resolve_capture_window(args, d, display)
    if picked is _CW_ERROR:
        return 2
    capture_window = picked[0] if len(picked) == 1 else None
    capture_windows = picked if len(picked) > 1 else None

    session = _session_dir(args)
    print("session: {}".format(session))
    # Resolved here rather than read back off the Recorder: the CLI already
    # has everything it needs, and reaching into the object it just built
    # couples this call site to an attribute for no reason.
    backend = sck.resolve_backend(getattr(args, "capture_backend", None),
                                  os.environ,
                                  settings.get("capture_backend"))
    # --occlusion-free: capture each target window's OWN buffer via SCK.
    # 1 window   -> single-file native (P1, `raw.mov` IS the window)
    # 2-4 windows -> N-file multi-native (P3.1, one `raw_i.mov` per window;
    #               manifest in meta.json). Both imply the SCK backend and
    #               refuse an explicit avfoundation flag rather than silently
    #               falling back to a display crop that would still show
    #               occluders. The (N=0 / N>4) validation lives upstream in
    #               `_resolve_capture_window` so this branch never fires on
    #               an out-of-range pick.
    window_native = bool(getattr(args, "occlusion_free", False))
    if window_native:
        if not (1 <= len(picked) <= 4):
            print("--occlusion-free records 1-4 windows; pass one or more "
                  "--capture-window <id> arguments.", file=sys.stderr)
            return 2
        if getattr(args, "capture_backend", None) == "avfoundation":
            print("--occlusion-free needs the sck backend; avfoundation cannot "
                  "capture a window's own buffer.", file=sys.stderr)
            return 2
        backend = "sck"
    if backend != sck.DEFAULT_BACKEND:
        print("capture backend: {}".format(backend))
    r = rec.Recorder(session, display, mic_idx=mic, fps=args.fps,
                     cursor_mode=args.cursor, log_keys=args.log_keys,
                     face_idx=face_idx, capture_window=capture_window,
                     capture_windows=capture_windows,
                     backend=backend, window_native=window_native)
    try:
        result = r.start(countdown=args.countdown, duration=args.duration)
    except rec.RecordError as e:
        print("\nrecording failed:\n{}".format(e), file=sys.stderr)
        return 1
    if result is None:
        return 0  # user aborted during countdown
    print("saved recording to {}".format(session))
    if window_native:
        if len(picked) == 1:
            print("occlusion-free: raw.mov is the window's own buffer — nothing "
                  "in front of it appears in the recording. render maps clicks "
                  "into window space, so auto-zoom lands correctly.")
        else:
            print("occlusion-free (multi-native, {}): each raw_i.mov is one "
                  "window's own buffer -- inter-window overlap disappears. "
                  "Manifest lives in meta.json (`capture_channels`); render "
                  "composites them onto one canvas."
                  .format(len(picked)))
    if args.render:
        ren.render(session, **_record_render_kwargs(
            session, args,
            multi_native=bool(window_native and len(picked) > 1)))
    else:
        print("render it:  python3 studio.py render '{}'".format(session))
    return 0


def _seed_capture_windows(session, multi_native=False):
    """One-shot seed of a record-time window pick; returns the seeded doc.

    Only sessions that actually HAVE a pick are touched -- a whole-screen take
    gets no edits.json out of this -- and the flag inside
    `edits.initialize_capture_windows` makes it idempotent. A failure here
    costs the caller nothing worse than the flag defaults, so it returns an
    empty doc rather than raising.
    """
    try:
        specs = ren.capture_window_specs(session)
        if not specs and not multi_native:
            return {}
        doc = _edits.load_edits(session)
        if not doc.get("capture_windows_initialized"):
            doc, changed = _edits.initialize_capture_windows(
                doc, specs, multi_native=multi_native)
            if changed:
                doc = _edits.save_edits(session, doc)
        return doc
    except Exception:
        return {}


def _record_render_kwargs(session, args, multi_native=False):
    """The kwargs `record --render` hands `render.render()`.

    Materializes the record-time window pick FIRST, so this render is the same
    video `studio.py render` (and the editor, and MCP) would produce a minute
    later -- they all seed on first touch, and this path used to render
    straight off the flags.

    Both halves of the seed have to be threaded, which is the part that is
    easy to get wrong: `render.render` gates the whole cards path on
    `bool(windows)` and reads `window_focus` only INSIDE it, so passing the
    seeded render block alone left `--capture-window a b --render` exporting
    the plain whole-screen video (not even a grid) with the seeded camera
    inert, while `studio.py render` on that very edits.json exported the
    desktop composite. An explicit `--window` set still wins -- `_render_kwargs`
    has already put it in `kwargs`, and `setdefault` leaves it alone.

    A multi-native pick seeds no `windows` array (its cards ARE the recorded
    channels), so it passes `[]` and `render.render` dispatches off the
    manifest exactly as before.
    """
    seeded = _seed_capture_windows(session, multi_native=multi_native)
    kwargs = _render_kwargs(args, seeded.get("render") or {})
    kwargs.setdefault("windows", list(seeded.get("windows") or []))
    return kwargs


def _cmd_render(args):
    # Load edits.json (materializing auto zooms on first read) so a CLI
    # render matches what the editor would produce -- and honors any
    # ranges the user has already deleted. Without this pass the camera
    # falls back to on-the-fly click clustering (fine for a brand-new
    # session but resurrects deleted ranges after the editor has touched
    # the doc).
    session = args.session
    # Multi-window native takes a SEPARATE prep path from the whole-screen one
    # below: there is no single `raw.mov`, and manual zooms / crops / windows
    # stay inert (no per-card manual-zoom surface yet). But it still loads the
    # session's edits so the render block (window_zoom / window_focus /
    # window_layout / ...) reaches `render.render()`, which dispatches to
    # `_render_multi_native` off the manifest. It just skips the arc-seeding
    # (auto zooms, materialized focus ranges) the whole-screen prep does.
    _meta_path = os.path.join(session, "meta.json")
    if os.path.isfile(_meta_path):
        try:
            with open(_meta_path) as _f:
                _meta = json.load(_f)
        except (OSError, ValueError):
            _meta = None
        if isinstance(_meta, dict) and (
                isinstance(_meta.get("capture_channels"), list)
                or isinstance(_meta.get("capture_scenes"), list)):
            # Scene takes ride the same short-circuit as multi-native: both
            # are manifest sessions with no raw.mov, so the whole-screen prep
            # below (describe + auto-zoom arc seeding into edits.json) would
            # materialize junk zooms before render()'s own dispatch rescued
            # the take (docs/architecture.md, the missed-dispatch-site finding).
            # Honour the session's saved edits the same way every other render
            # path does: window_zoom / window_focus / window_layout /
            # background etc. set in the editor have to survive a plain
            # `studio.py render`, or one edits.json makes two different videos
            # depending on which surface exported it. `_render_kwargs` already
            # falls back to the saved block per key; the short-circuit used to
            # starve it of one. Manual zooms / crops / windows stay inert here
            # (there is no per-card manual-zoom surface yet), so only the
            # render block is threaded through -- not the arcs the non-native
            # path also seeds below.
            m_info = ren.describe_session(session, include_click_times=True)
            m_doc = _edits.load_edits(session, duration=m_info["duration"])
            if not m_doc.get("capture_windows_initialized"):
                # First render of a take never opened in the editor: seed the
                # "desktop" default (cards ARE the screen windows) so the CLI
                # and the editor agree on the arrangement.
                m_doc, m_changed = _edits.initialize_capture_windows(
                    m_doc, ren.capture_window_specs(session),
                    multi_native=True)
                if m_changed:
                    m_doc = _edits.save_edits(session, m_doc,
                                              duration=m_info["duration"])
            kwargs = _render_kwargs(args, m_doc.get("render") or {})
            # Suppressed ranges feed the per-card cameras / emphasis in
            # `_render_multi_native`, so honor them here too.
            # Manual zooms / crops / windows stay inert here.
            kwargs.setdefault("suppressed_ranges",
                              list(m_doc.get("suppressed") or []))
            # Manual card placement (multi-native): a top-level spatial edit
            # like crop, so thread it here too -- otherwise a plain
            # `studio.py render` ignores the drag the editor saved (one
            # edits.json, two different videos). Inert on scene takes (P2).
            kwargs.setdefault("channel_layouts",
                              list(m_doc.get("channel_layouts") or []))
            # Per-scene manual placement (scene takes); inert on multi-native.
            kwargs.setdefault("scene_layouts",
                              dict(m_doc.get("scene_layouts") or {}))
            # Cards the user removed from the render (scene + multi-native).
            kwargs.setdefault("hidden_channels",
                              list(m_doc.get("hidden_channels") or []))
            # Focus is auto-planned for multi-native on first open (the editor
            # deliberately does not materialize it), so `None` is the normal
            # answer and the render plans from the clicks. But MCP `adjust_zoom`
            # DOES materialize the plan when someone softens one of its moves --
            # and a materialized plan the CLI ignored would be the
            # window_layout/speedups/cuts bug again: one edits.json, two
            # different videos depending on which surface exported it. Same
            # None-vs-[] contract every other surface uses.
            kwargs.setdefault(
                "focus_ranges",
                list(m_doc.get("focus") or [])
                if not _edits.focus_plan_is_stale(m_doc) else None)
            # Trim is a top-level edit like `crop` -- the editor and MCP each
            # thread it into ren.render explicitly (studio_app:2215,
            # mcp_server:1478); a plain `studio.py render` that skipped it
            # exported the FULL take from an edits.json whose trim the other
            # two surfaces honored. `_render_multi_native` drops trim via
            # **_ignored today, so this is a no-op on that path -- and staying
            # symmetric with the whole-screen branch keeps the invariant even
            # if that dispatcher grows trim later.
            m_trim = m_doc.get("trim") or {}
            m_trim_end = m_trim.get("end")
            kwargs.setdefault("trim_start", float(m_trim.get("start") or 0.0))
            kwargs.setdefault(
                "trim_end",
                None if m_trim_end is None else float(m_trim_end))
            try:
                ren.render(session, out_path=args.out, offset=args.offset,
                           **kwargs)
            except Exception as exc:
                print("render failed: {!r}".format(exc), file=sys.stderr)
                return 1
            return 0
    info = ren.describe_session(session, include_click_times=True)
    doc = _edits.load_edits(session, duration=info["duration"])
    if not doc.get("auto_zooms_initialized"):
        doc, changed = _edits.initialize_auto_zooms(
            doc, info.get("click_times") or [], info["duration"])
        if changed:
            doc = _edits.save_edits(session, doc, duration=info["duration"])
    # Same one-shot materialization for a record-time multi-window pick: the
    # windows the user chose become the session's cards, once.
    if not doc.get("capture_windows_initialized"):
        doc, changed = _edits.initialize_capture_windows(
            doc, ren.capture_window_specs(session))
        if changed:
            doc = _edits.save_edits(session, doc, duration=info["duration"])
    kwargs = _render_kwargs(args, doc.get("render") or {})
    kwargs.setdefault("manual_zooms", list(doc.get("zooms") or []))
    kwargs.setdefault("suppressed_ranges", list(doc.get("suppressed") or []))
    kwargs.setdefault("windows", list(doc.get("windows") or []))
    # Manual card placement (docs/architecture.md), the editor-only authoring
    # surface whose SAVED result every render sink has to read. The
    # multi-native / scene branch above already threads both; repeating them
    # here keeps this branch's kwarg key set equal to the web resolver's
    # (`test_cli.RenderKwargsSurfaceParity`). They are inert on a whole-screen
    # take -- render() forwards them only on its capture_channels /
    # capture_scenes dispatch -- exactly like `trim_start`/`trim_end` are inert
    # over there, and for the same reason: staying symmetric is what keeps the
    # invariant true if a dispatcher later grows a use for them.
    kwargs.setdefault("channel_layouts",
                      list(doc.get("channel_layouts") or []))
    kwargs.setdefault("scene_layouts", dict(doc.get("scene_layouts") or {}))
    kwargs.setdefault("hidden_channels",
                      list(doc.get("hidden_channels") or []))
    # MCP/editor-authored cuts must export from the CLI too (one edits.json,
    # one output -- the window_layout lesson); --cut overrides for this render.
    kwargs.setdefault("cuts", list(doc.get("cuts") or []))
    # Manual force/off speed-up spans (MCP add_speedup writes them; render()
    # honors "force" ones even with speedup off) -- a saved edit, so the CLI
    # has to pass them like the web/MCP resolvers do or one edits.json makes
    # two different videos depending on which surface exported it.
    kwargs.setdefault("speedups", list(doc.get("speedups") or []))
    # Focus ranges are authoritative only once the editor has materialized
    # them (`focus_initialized`) -- an empty array then genuinely means "the
    # user deleted every arc", and must not be re-planned from the clicks.
    # Before that, None asks render to auto-plan, the same None-vs-[] contract
    # `camera.build_path` uses for manual zooms. The CLI never materializes:
    # that needs the canvas layout, which is the editor's context, and a
    # `--window` override here would plan against cards the session doesn't
    # actually have. Always setdefault the key (None when stale) so the CLI's
    # kwarg key set matches the web sink's -- one edits.json, one video.
    kwargs.setdefault(
        "focus_ranges",
        list(doc.get("focus") or []) if not _edits.focus_plan_is_stale(doc)
        else None)
    # The editor's crop is a saved edit, so a CLI render has to honor it or
    # `studio.py render` and the in-app Export produce different framings from
    # one edits.json -- exactly the split `window_layout` shipped with.
    kwargs.setdefault("crop_rect", _edits.normalize_crop(doc.get("crop")))
    # The editor's saved trim has to reach render() the same way `crop` does,
    # or `studio.py render` and the in-app Export produce different lengths
    # from one edits.json -- the split cli.py:483's comment warns about,
    # extended to the sibling top-level edit. Mirrors studio_app:2215 and
    # mcp_server:1478.
    trim = doc.get("trim") or {}
    trim_end = trim.get("end")
    kwargs.setdefault("trim_start", float(trim.get("start") or 0.0))
    kwargs.setdefault(
        "trim_end", None if trim_end is None else float(trim_end))
    # The cuts write contract, same rule the MCP tools and the web editor
    # enforce. `--cut` is an authoring surface too, and a cut set that
    # swallows the trim window should say so HERE rather than 40 seconds
    # into the encode. Last, because it has to see the resolved trim.
    try:
        _edits.plan_cuts(kwargs.get("cuts") or [],
                         {"start": kwargs.get("trim_start"),
                          "end": kwargs.get("trim_end")},
                         info["duration"], info.get("fps") or 60.0)
    except _edits.CutsError as exc:
        raise SystemExit("error: {}".format(exc))
    ren.render(session, out_path=args.out, offset=args.offset, **kwargs)
    return 0


def _cmd_app(args):
    from . import studio_app
    try:
        studio_app._validate_bind_host(args.host)
    except ValueError as exc:
        raise SystemExit("error: {}".format(exc))
    if getattr(args, "reload", False):
        return _run_app_with_reload(args)
    return studio_app.run_server(
        host=args.host,
        port=args.port,
        open_browser=not args.no_open,
        recordings_root=args.recordings_root,
    )


def _run_app_with_reload(args):
    """Supervisor mode: parent opens ONE browser tab (at startup only), then
    respawns the server on file changes. Open tabs poll /api/rev and reload
    themselves when the boot id changes — no new-tab clutter across restarts.
    """
    import os as _os
    import time as _time
    import webbrowser
    from . import livereload

    project_root = _os.path.abspath(
        _os.path.join(_os.path.dirname(__file__), ".."))
    watch_dirs = [_os.path.join(project_root, "autocine"),
                  _os.path.join(project_root, "studio_web")]

    child_argv = [sys.executable,
                  _os.path.abspath(_os.path.join(project_root, "studio.py")),
                  "app",
                  "--host", args.host,
                  "--port", str(args.port),
                  "--no-open",             # parent controls browser opening
                  "--recordings-root", args.recordings_root]

    display_host = ("127.0.0.1" if args.host in ("0.0.0.0", "::", "")
                    else args.host)
    url = "http://{}:{}/".format(display_host, args.port)

    if not args.no_open:
        # Wait until the child server answers before opening the tab, so the
        # first request doesn't race the bind. Poll /api/health with a short
        # timeout; give up after ~5s.
        import threading

        def _open_when_ready():
            deadline = _time.monotonic() + 5.0
            while _time.monotonic() < deadline:
                if _is_studio_server(url.rstrip("/")):
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                    return
                _time.sleep(0.15)

        threading.Thread(target=_open_when_ready, daemon=True).start()

    return livereload.run(child_argv, watch_dirs=watch_dirs)


def _studio_server_or_url(args):
    """Bind the studio server, or detect one already on host:port.

    Returns (server, url). `server` is None when another *studio* server is
    already listening (EADDRINUSE + /api/health answers) -- callers then just
    open windows/tabs against the returned url instead of serving. If the port
    is held by something that is NOT a studio server (e.g. a Vite dev server
    on 5173), exit with an error instead of opening windows at it.
    """
    from . import studio_app
    try:
        return studio_app.build_server(args.host, args.port,
                                       recordings_root=args.recordings_root)
    except ValueError as exc:
        raise SystemExit("error: {}".format(exc))
    except OSError as exc:
        if getattr(exc, "errno", None) != errno.EADDRINUSE:
            raise
    display_host = ("127.0.0.1" if args.host in ("0.0.0.0", "::", "")
                    else args.host)
    url = "http://{}:{}".format(display_host, args.port)
    if not _is_studio_server(url):
        print("port {} is in use by another application (not a studio "
              "server); pass --port N to use a different port".format(args.port),
              file=sys.stderr)
        raise SystemExit(2)
    return None, url


def _hard_exit(code=0):
    """Exit even if pywebview/Cocoa left non-daemon threads running.

    Only used after the native bar window has been torn down and (when we own
    it) the HTTP server has been shut down cleanly — all state is on disk.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def _install_signal_quit(on_quit=None):
    """Make Ctrl+C actually kill the native bar.

    Python only runs signal handlers on the MAIN thread, between bytecodes --
    and once `webview.start()` hands the main thread to Cocoa's run loop, that
    never happens again. So a plain `signal.signal` handler is registered,
    never fires, and Ctrl+C looks ignored: exactly the reported symptom.

    The fix is to take signals off the main thread entirely. Block them
    process-wide, then have one daemon thread sit in `sigwait`, which is a
    blocking C call that does not care what the main thread is doing. That
    thread gets to run real Python and can tear things down.

    CAUTION -- this block is INHERITED by every child we spawn (a blocked
    signal mask survives fork+exec), and `signal.signal` in a child sets only
    the disposition, never the mask. So a capture/key/watchdog child would
    start with its stop signals permanently pending: the parent's stop-SIGINT
    is silently ignored until a SIGKILL, and (for the SCK fleet) the whole take
    is discarded. Each such child therefore unblocks its own stop signals at
    startup -- see `_sck_worker.py`, `_key_worker.py`, `_watchdog.py`, and
    `record.Recorder._gate_fleet`'s moov salvage. ffmpeg is immune (it resets
    its own mask), so the avfoundation path needs nothing.

    `on_quit` is a best-effort chance to stop an in-flight recording so its
    moov atom is written -- a take killed mid-write is an unplayable file, and
    that is a much worse outcome than a slow quit. Then `os._exit`, because
    pywebview leaves non-daemon threads that would otherwise hang the process
    (the same reason `_hard_exit` exists).

    Soft-fails on any platform without pthread_sigmask: the caller simply
    keeps whatever behavior it had.
    """
    import signal
    if not hasattr(signal, "pthread_sigmask") or not hasattr(signal, "sigwait"):
        return False
    sigs = {signal.SIGINT, signal.SIGTERM}
    try:
        signal.pthread_sigmask(signal.SIG_BLOCK, sigs)
    except Exception:
        return False

    def _waiter():
        try:
            signal.sigwait(sigs)
        except Exception:
            return
        print("\nquitting...", file=sys.stderr)
        if on_quit is not None:
            try:
                on_quit()
            except Exception:
                pass
        _hard_exit(0)

    import threading as _threading
    _threading.Thread(target=_waiter, daemon=True).start()
    return True


def _is_studio_server(url):
    import json as _json
    import urllib.request
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=1.5) as resp:
            return bool(_json.loads(resp.read().decode("utf-8")).get("ok"))
    except Exception:
        return False


def _cmd_bar(args):
    import threading
    import time
    from . import studio_app
    try:
        dw, dh, _src = dev.main_display_points()
        dw, dh = float(dw), float(dh)
    except Exception:
        dw, dh = 1440.0, 900.0
    if dw <= 0 or dh <= 0:
        dw, dh = 1440.0, 900.0
    width, height = 880, 136
    x = int((dw - width) // 2)
    y = int(dh - height - 64)
    server, url = _studio_server_or_url(args)
    if server is not None:
        print("studio app listening on {}/".format(url))
        print("recordings root: {}".format(server.state.recordings_root))

    if studio_app.native_bar_available():
        # native pill: frameless + always-on-top (pywebview / WKWebView).
        # Cocoa needs the main thread, so the server serves from a thread.
        _install_signal_quit(lambda: studio_app.stop_native_bar())
        serve_thread = None
        if server is not None:
            serve_thread = threading.Thread(target=studio_app.serve,
                                            args=(server,), daemon=True)
            serve_thread.start()
        ran = studio_app.run_native_bar(url, width, height, x=x, y=y)
        if ran:
            if server is None:
                _hard_exit(0)   # pywebview leaves non-daemon threads behind
            # keep the backend alive for any editor tabs the bar opened
            print("recording bar closed - studio server still running at "
                  "{}/ (Ctrl+C to quit)".format(url))
            try:
                while serve_thread.is_alive():
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print("\nshutting down studio app...")
                server.shutdown()
                serve_thread.join(timeout=5)
            _hard_exit(0)
        # pywebview failed at runtime -- fall through to the browser window;
        # the server (if we own it) is already serving on its thread.
        studio_app.open_app_window("{}/bar.html".format(url), width, height,
                                   x=x, y=y)
        if server is None:
            print("studio server already running; opened {}/bar.html".format(url))
            _hard_exit(0)
        try:
            while serve_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nshutting down studio app...")
            server.shutdown()
            serve_thread.join(timeout=5)
        _hard_exit(0)

    studio_app.open_app_window("{}/bar.html".format(url), width, height,
                               x=x, y=y)
    if server is None:
        print("studio server already running; opened {}/bar.html".format(url))
        return 0
    return studio_app.serve(server)


def _cmd_open(args):
    from . import studio_app
    root = os.path.abspath(args.recordings_root)
    raw_arg = str(args.session)
    if os.sep in raw_arg or (os.altsep and os.altsep in raw_arg):
        path = os.path.abspath(raw_arg)
        if path != root and not path.startswith(root + os.sep):
            print("session dir is not inside the recordings root ({}): {}"
                  .format(root, path), file=sys.stderr)
            return 2
        name = os.path.basename(path.rstrip(os.sep))
    else:
        name = raw_arg
        path = os.path.join(root, name)
    if not os.path.isfile(os.path.join(path, "meta.json")):
        print("not a recording session (missing meta.json): {}".format(path),
              file=sys.stderr)
        return 2
    server, url = _studio_server_or_url(args)
    page = "{}/editor.html?session={}".format(url, quote(name))
    try:
        webbrowser.open(page)
    except Exception:
        pass
    if server is None:
        print("studio server already running; opened {}".format(page))
        return 0
    print("studio app listening on {}/".format(url))
    print("recordings root: {}".format(server.state.recordings_root))
    return studio_app.serve(server)


def _cmd_mcp(args):
    if args.print_config:
        config = {
            "mcpServers": {
                "autocine": {
                    "command": os.path.abspath(sys.executable),
                    "args": [
                        paths.studio_entrypoint(),
                        "mcp",
                        "--recordings-root",
                        os.path.abspath(args.recordings_root),
                    ],
                },
            },
        }
        print(json.dumps(config, indent=2))
        return 0
    try:
        from . import mcp_server
    except ImportError as exc:
        print("MCP server unavailable ({}). autocine/mcp_server.py is not "
              "present yet or its dependencies are missing.".format(exc),
            file=sys.stderr)
        return 1
    if args.print_tools:
        print(json.dumps({"tools": mcp_server.TOOL_DEFS}, indent=2))
        return 0
    return mcp_server.run(recordings_root=args.recordings_root)


def _add_server_opts(sp):
    """Bind/root options shared by the server-flavored subcommands."""
    sp.add_argument("--host", default="127.0.0.1",
                    help="loopback bind host only (default: 127.0.0.1)")
    sp.add_argument("--port", type=int, default=5173,
                    help="bind port (default: 5173)")
    sp.add_argument("--recordings-root", default=REC_ROOT,
                    help="recordings directory (default: %(default)s; override "
                         "with AUTOCINE_RECORDINGS_ROOT)")


def _add_render_opts(sp):
    """Presentation options shared by `render` and `record --render`."""
    # default=None so the resolver can tell "unpassed" from "passed the
    # default"; unpassed falls back to `edits._DEFAULT_RENDER["zoom"]` (2.2),
    # which is what the web editor and MCP already use. Before this the flag
    # default was 2.0, so a never-edited session diverged from the web/MCP
    # output on `studio.py render` alone.
    sp.add_argument("--zoom", type=float, default=None,
                    help="max zoom factor (default: whatever the session "
                         "saved, or 2.2 for a never-edited take)")
    sp.add_argument("--zoom-speed", default=None,
                    choices=["slow", "normal", "fast"],
                    help="zoom transition speed: slow (cinematic), normal, "
                         "or fast (snappy). Defaults to whatever the session "
                         "saved")
    sp.add_argument("--screen-anim", default="focused",
                    choices=["focused", "smooth"],
                    help="camera style while zoomed: focused (rock-still "
                         "holds, event-driven pans -- the cinematic "
                         "look) or smooth (legacy continuous cursor "
                         "follower)")
    sp.add_argument("--style", default="clean", choices=["clean", "framed"],
                    help="clean (native) or framed (background + rounded corners)")
    sp.add_argument("--background", default=None,
                    help="framed background: preset (aurora, midnight, sunset, "
                         "ocean, forest, graphite, mist, grape), #rrggbb, or an "
                         "image path (implies --style framed)")
    sp.add_argument("--no-clicks", action="store_true",
                    help="disable the click-highlight ripples (on by default)")
    sp.add_argument("--click-color", default=None,
                    help="ripple color: name (white, yellow, blue, ...) or #rrggbb")
    sp.add_argument("--spotlight", action="store_true",
                    help="dim everything except a soft disc around the cursor")
    sp.add_argument("--cursor-fx", action="store_true", default=None,
                    help="draw a smoothed, enlarged synthetic cursor. Needs a "
                         "frame with no pointer already in it: either record "
                         "with --cursor synthetic, or add --cursor-erase to "
                         "lift the recorded one out of an ordinary take first")
    sp.add_argument("--cursor-size", type=float, default=None,
                    help="synthetic cursor size multiplier (default 1.0)")
    sp.add_argument("--cursor-erase", action="store_true", default=None,
                    help="remove the RECORDED system cursor from the footage "
                         "(the opposite of --cursor-fx: that one draws a "
                         "pointer, this one takes the burned-in one out -- "
                         "pair them to REPLACE the recorded cursor with the "
                         "synthetic one on a take that was not recorded for "
                         "it). "
                         "Repaints each frame's pointer box with the real "
                         "pixels from before it arrived or after it left; "
                         "costs one extra decode of the source, and falls "
                         "back to a local inpaint where the recording never "
                         "showed those pixels uncovered")
    sp.add_argument("--aspect", default=None,
                    help="output aspect: auto (default, matches the source), "
                         "a ratio like 9:16 / 1:1 / 4:5 / 4:3, or an exact "
                         "WxH like 1080x1920. The camera auto-reflows to fit")
    sp.add_argument("--resolution", default=None,
                    choices=["auto", "2160", "1440", "1080", "720"],
                    help="cap the output HEIGHT: auto (default, the natural "
                         "sharpness-preserving canvas -- 4K for a Retina "
                         "multi-window take) or 2160/1440/1080/720. Scales the "
                         "whole canvas down, which is what makes the render "
                         "fast (cost is per-pixel per-frame)")
    sp.add_argument("--fade", type=float, default=0.0,
                    help="fade in/out to black over N seconds (e.g. 0.4)")
    sp.add_argument("--music", default=None,
                    help="audio file mixed under the recording as background music")
    sp.add_argument("--click-sound", default=None,
                    help="click sound: 'auto' (default, a built-in click), "
                         "'off', or an audio file to play at each click")
    sp.add_argument("--key-sound", default=None,
                    help="keystroke sound: 'auto' (default, a built-in "
                         "keystroke), 'off', or an audio file to play at "
                         "each recorded key-activity tick")
    sp.add_argument("--no-sound-fx", action="store_true", default=False,
                    help="silence BOTH the click and keystroke sounds "
                         "(shorthand for --click-sound off --key-sound off)")
    sp.add_argument("--sfx-volume", type=float, default=None,
                    help="loudness of the click/keystroke bed, 0 to 2 "
                         "(default 1.0; 0 is silent)")
    # default=None so the resolver can tell unpassed from "passed False": a
    # store_true action's implicit default (False) would clobber a saved
    # `always_zoomed: true` on `studio.py render` alone.
    sp.add_argument("--always-zoomed", action="store_true", default=None,
                    help="hold the last-reached zoom for the rest of the clip "
                         "instead of settling back out to full frame "
                         "(defaults to whatever the session saved)")
    sp.add_argument("--no-motion-blur", dest="motion_blur", action="store_false",
                    help="disable camera motion blur on fast pans/zooms "
                         "(on by default)")
    sp.add_argument("--no-typing-zoom", dest="typing_zoom", action="store_false",
                    help="disable automatic zoom on typing bursts "
                         "(on by default; needs key activity in the recording)")
    sp.add_argument("--no-drag-hold", dest="drag_hold", action="store_false",
                    help="don't hold the zoom through slow mouse drags "
                         "(on by default)")
    sp.add_argument("--no-scroll-zoom", dest="scroll_zoom", action="store_false",
                    help="disable scroll-aware camera behavior: no gentle zoom "
                         "on sustained scroll-reading and no zoom hold while "
                         "scrolling (on by default; needs scroll activity in "
                         "the recording)")
    # choices from edits, not a literal: argparse rejecting a value the editor
    # and MCP both accept is the worst version of this drifting apart -- the
    # flag would look broken rather than quietly wrong.
    sp.add_argument("--window-layout", choices=_edits._WINDOW_LAYOUTS,
                    default=None,
                    # The percent signs below are doubled because argparse
                    # runs every help string through `% dict(...)`; a bare
                    # one raises at --help time, not at import.
                    help="multi-window only: how the cards are arranged. "
                         "Cards are aspect-locked and scaled by ONE uniform "
                         "factor, so no arrangement fills both axes -- it "
                         "meets the padding on one and leaves margin on the "
                         "other, and which one wastes least is decided by the "
                         "export aspect. 'feature' gives the first window a "
                         "hero card with the rest in a block beside it, and "
                         "it is the one arrangement that TRANSPOSES -- hero "
                         "on top with the rest in a row beneath, when the "
                         "canvas is taller than the side-by-side form wants. "
                         "'grid' is uniform rows x cols and adapts too, "
                         "choosing rows/cols against the canvas orientation. "
                         "'desktop' keeps the windows' relative on-screen "
                         "arrangement, pulls any overlaps apart and squeezes "
                         "out the dead space. 'row' lays them side by side "
                         "and 'column' stacks them, both in card order and "
                         "neither ever transposed -- reach for those when the "
                         "reading order matters more than the coverage. "
                         "Measured on the three real windows this was built "
                         "for (1418x1718, 1418x852, 1418x854), fraction of "
                         "the canvas covered: 16:10 export -- feature 81.8%%, "
                         "desktop 80.7%%, grid 34.9%%, row 33.0%%, column "
                         "20.0%%; 9:16 export -- feature 74.9%%, grid 73.2%%, "
                         "column 67.5%%, desktop 29.1%%, row 11.8%%. So: "
                         "'feature' unless you want something specific -- "
                         "transposing is what makes it the best of the five "
                         "on all four canvases measured. "
                         "'grid' matches it on a vertical export and "
                         "'desktop' on a wide one; 'row' is never the best of "
                         "the five for three windows. Two windows rank "
                         "differently ('feature' and 'column' tie at 80.9%% "
                         "on a 9:16 export, against the grid's 58.4%%), so "
                         "this is a shape, not a law. Defaults to whatever "
                         "the session saved (a record-time multi-window pick "
                         "saves 'desktop')")
    sp.add_argument("--window-zoom", dest="window_zoom", action="store_true",
                    default=None,
                    help="multi-window only: auto-zoom INSIDE the cards. Each "
                         "card gets its own camera planned from the clicks "
                         "that landed in it, and only the card with the most "
                         "recent activity is zoomed at a time -- the others "
                         "hold their full framing. Off by default (the static "
                         "side-by-side layout is the point of this mode); "
                         "defaults to whatever the session saved")
    sp.add_argument("--no-window-zoom", dest="window_zoom",
                    action="store_false", default=None,
                    help="multi-window only: force per-card auto-zoom OFF for "
                         "this render even if the session saved it on")
    sp.add_argument("--window-focus", dest="window_focus",
                    action="store_true", default=None,
                    help="multi-window only: focus the window you're working "
                         "in. Clicking in a window leans the whole "
                         "composition toward that card; clicking again pushes "
                         "in until it fills the frame, and it eases back out "
                         "when your attention moves. Only one card is ever "
                         "the subject. ON by default for a session recorded "
                         "with a multi-window pick (--capture-window a b, "
                         "--occlusion-free), off for hand-drawn --window "
                         "cards; with neither flag you get whatever the "
                         "session saved")
    sp.add_argument("--no-window-focus", dest="window_focus",
                    action="store_false", default=None,
                    help="multi-window only: force the composition camera OFF "
                         "for this render even if the session saved it on")
    sp.add_argument("--screen-focus", dest="screen_focus",
                    action="store_true", default=None,
                    help="whole-screen only: when you click inside ONE window "
                         "and others are on screen, grow that window a bit to "
                         "overlap its neighbours while the frame eases in -- "
                         "instead of cropping into a neighbour. ON by default; "
                         "defaults to whatever the session saved")
    sp.add_argument("--no-screen-focus", dest="screen_focus",
                    action="store_false", default=None,
                    help="whole-screen only: force the grow-the-active-window "
                         "emphasis OFF for this render even if the session "
                         "saved it on")
    sp.add_argument("--badge-erase", dest="badge_erase",
                    action="store_true", default=None,
                    help="occlusion-free only: paint out macOS's per-window "
                         "capture indicator -- the pill it draws over each "
                         "recorded window's traffic lights. ON by default; "
                         "defaults to whatever the session saved")
    sp.add_argument("--no-badge-erase", dest="badge_erase",
                    action="store_false", default=None,
                    help="occlusion-free only: leave macOS's capture "
                         "indicator in the footage")
    sp.add_argument("--no-window-follow", dest="window_follow",
                    action="store_false",
                    help="multi-window only: don't bind each --window "
                         "rect to the recorded window it overlaps, so a window "
                         "moved or resized mid-take drifts out of its card "
                         "(following is on by default; needs a recording with "
                         "a window geometry track)")
    # The facecam flags default to None (not to their values) so `render` can
    # tell "not passed" from "passed the default" and fall back to whatever the
    # editor/MCP saved for the session -- same deal as --window and the saved
    # window layout.
    sp.add_argument("--no-facecam", dest="facecam", action="store_false",
                    default=None,
                    help="don't composite the webcam bubble even if the "
                         "session has a face.mov track (on by default)")
    sp.add_argument("--facecam-position", default=None,
                    choices=["bottom-left", "bottom-right", "top-left", "top-right"],
                    help="corner for the facecam bubble (default bottom-left, "
                         "or the session's saved position)")
    sp.add_argument("--facecam-size", type=float, default=None,
                    help="facecam bubble diameter as a fraction of output "
                         "height (default 0.20, or the session's saved size)")
    sp.add_argument("--facecam-shape", default=None,
                    choices=["circle", "rounded"],
                    help="facecam bubble outline: circle (default) or rounded "
                         "square; omit to use the session's saved shape")
    sp.add_argument("--facecam-blur", type=float, default=None,
                    help="background blur strength for the facecam bubble, "
                         "0..1 (0 = off, the default): keeps the face sharp and "
                         "softens toward the rim")
    # default=None on the whole family so the resolver can tell unpassed from
    # "passed the default": before this, `--speedup`'s implicit False would
    # clobber a saved `speedup: true`, and `--speedup-rate`'s default 6.0
    # would clobber a saved rate (silently rendering an add_speedup force
    # span at 6.0 instead of the saved value -- render.py:139 threads
    # `speedup_rate` into force spans whose own rate is null).
    sp.add_argument("--speedup", action="store_true", default=None,
                    help="automatically speed up idle stretches (no clicks, "
                         "keys, scrolls, or purposeful motion) with a smooth "
                         "quintic ramp in and out. Off by default; changes "
                         "output duration. Defaults to whatever the session "
                         "saved")
    sp.add_argument("--speedup-rate", type=float, default=None,
                    help="peak speed factor inside auto-detected idle spans "
                         "(default 6.0; clamped to [1.5, 12.0]). Defaults "
                         "to whatever the session saved")
    sp.add_argument("--no-speedup-silence-gate", dest="speedup_silence_gate",
                    action="store_false", default=None,
                    help="disable the audio silence gate: sped spans no "
                         "longer required to also be quiet -- risks "
                         "chipmunk narration. Only meaningful on sessions "
                         "with a mic track; ignored otherwise")
    sp.add_argument("--no-speedup-motion-gate", dest="speedup_motion_gate",
                    action="store_false", default=None,
                    help="disable the visual motion gate: sped spans no "
                         "longer required to also be visually still -- "
                         "risks time-lapsing a playing video, scrolling "
                         "build log, or download animation into a glitchy "
                         "blur")
    sp.add_argument("--window", dest="windows", action="append",
                    type=_parse_window_spec, default=None, metavar="X,Y,W,H",
                    help="crop this source-pixel rect into a framed "
                         "'window' card; repeat up to 4 times to arrange "
                         "multiple windows on one background "
                         "(--window-layout picks the arrangement). The "
                         "whole-screen camera is off in this mode; "
                         "--window-zoom and --window-focus are the two "
                         "cameras it has. Overrides the session's saved "
                         "window layout for this render; omit to use "
                         "whatever the editor/MCP has saved, same as "
                         "--zoom's relationship to saved zoom ranges")
    sp.add_argument("--cut", dest="cuts", action="append",
                    type=_parse_cut_spec, default=None, metavar="START-END",
                    help="CUT this range of seconds out of the export "
                         "(ripple delete): the range vanishes, what follows "
                         "slides earlier, audio stays in sync. Repeatable; "
                         "overlaps merge; boundaries snap outward to the "
                         "frame grid. Overrides the session's saved cuts "
                         "for this render; omit to use whatever the "
                         "editor/MCP has saved (same relationship as "
                         "--window). Not yet applied on scene, "
                         "multi-window-native, or card-layout renders (a "
                         "note prints)")
    sp.add_argument("--gif", action="store_true", help="also export a GIF")
    sp.add_argument("--gif-fps", type=int, default=15,
                    help="GIF frame rate (default 15)")
    sp.add_argument("--gif-width", type=int, default=1000,
                    help="GIF width in pixels, height auto-scales (default 1000)")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="studio",
        description="AutoCine: record your screen + render with "
                    "automatic click-driven zoom.")
    p.add_argument("--version", action="version",
                   version="AutoCine {}".format(__version__))
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("devices", help="list capture devices and display info")
    sp.set_defaults(func=_cmd_devices)

    sp = sub.add_parser("record", help="record the screen (Ctrl+C to stop)")
    sp.add_argument("--display", type=int, default=None,
                    help="avfoundation video index (default: auto-detect)")
    sp.add_argument("--mic", default=None,
                    help="avfoundation audio index, or 'none' (default)")
    sp.add_argument("--fps", type=int, default=60)
    sp.add_argument("--duration", type=float, default=None,
                    help="auto-stop after N seconds")
    sp.add_argument("--countdown", type=int, default=3)
    sp.add_argument("--cursor", default="system", choices=["system", "synthetic"],
                    help="system (default): keep the real OS cursor baked into "
                         "the recording. synthetic: hide it at capture time so "
                         "render --cursor-fx can draw a smoothed, enlarged one")
    sp.add_argument("--capture-backend", default=None,
                    choices=["avfoundation", "sck"],
                    help="how the screen is captured. avfoundation (default) "
                         "is ffmpeg's legacy screen input. sck uses "
                         "ScreenCaptureKit, which can keep chosen windows OUT "
                         "of the recording (the recording bar) — avfoundation "
                         "cannot, at all. Still being proven; the default will "
                         "not change until it is.")
    sp.add_argument("--no-key-log", dest="log_keys", action="store_false",
                    help="don't log key-activity timestamps (typing-triggered "
                         "zoom won't be available for this recording; key "
                         "identity is NEVER logged either way)")
    sp.add_argument("--face", action="store_true",
                    help="also capture the webcam to face.mov (facecam bubble); "
                         "best-effort — never aborts the screen recording")
    sp.add_argument("--face-index", type=int, default=None,
                    help="avfoundation camera index for --face (default: "
                         "auto-detect the first non-screen video device)")
    sp.add_argument("--capture-window", type=int, default=None, metavar="ID",
                    action="append",
                    help="record just this window (id from `studio devices`). "
                         "avfoundation can't target a window, so the whole "
                         "display is still captured and the window's rect is "
                         "cropped out at render time -- auto-zoom, click FX "
                         "and every other effect keep working. The rect is "
                         "re-read right after the countdown, so bring the "
                         "window forward then. Repeat up to 4 times to record "
                         "several windows together: they're composited onto a "
                         "background keeping their on-screen arrangement "
                         "(static cards, no auto-zoom). Main display only")
    sp.add_argument("--occlusion-free", action="store_true",
                    help="record each --capture-window as its OWN buffer via "
                         "ScreenCaptureKit, so windows in front of a target "
                         "never appear in the recording (unlike plain "
                         "--capture-window, which records the display and "
                         "crops). Implies --capture-backend sck; main display "
                         "only. With 1 window: single raw.mov IS the window. "
                         "With 2-4 windows (P3.1): one raw_i.mov per window "
                         "plus a `capture_channels` manifest in meta.json; the "
                         "N-window composite render lands in P3.2")
    sp.add_argument("--out", default=None, help="session dir (default timestamped)")
    sp.add_argument("--render", action="store_true",
                    help="render immediately after recording")
    _add_render_opts(sp)
    sp.set_defaults(func=_cmd_record)

    sp = sub.add_parser("render", help="render a session with auto-zoom")
    sp.add_argument("session", help="session dir produced by `record`")
    sp.add_argument("--out", default=None, help="output .mp4 path")
    sp.add_argument("--offset", type=float, default=0.0,
                    help="sync nudge in seconds (+ = clicks later)")
    _add_render_opts(sp)
    sp.set_defaults(func=_cmd_render)

    sp = sub.add_parser("app", help="launch the local Studio web app")
    _add_server_opts(sp)
    sp.add_argument("--no-open", action="store_true",
                    help="do not auto-open the browser")
    sp.add_argument("--reload", action="store_true",
                    help="auto-restart the server + reload open tabs on "
                         "source changes (autocine/ + studio_web/)")
    sp.set_defaults(func=_cmd_app)

    sp = sub.add_parser("bar", help="open the compact recorder bar as a "
                                    "bottom-center app window")
    _add_server_opts(sp)
    sp.set_defaults(func=_cmd_bar)

    sp = sub.add_parser("open", help="open a session in the Studio editor "
                                     "(session dir path or bare session name)")
    sp.add_argument("session", help="session dir path or bare session name")
    _add_server_opts(sp)
    sp.set_defaults(func=_cmd_open)

    sp = sub.add_parser("mcp", help="run the editing MCP server (stdio)")
    _add_server_opts(sp)
    output = sp.add_mutually_exclusive_group()
    output.add_argument("--print-config", action="store_true",
                        help="print a portable MCP client configuration and exit")
    output.add_argument("--print-tools", action="store_true",
                        help="print the live MCP tool schemas as JSON and exit")
    sp.set_defaults(func=_cmd_mcp)

    args = p.parse_args(argv)
    if not getattr(args, "func", None):
        p.print_help()
        return 1
    return args.func(args) or 0
