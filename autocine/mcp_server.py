"""Local MCP (Model Context Protocol) server for editing recorded sessions.

Exposes the deterministic render pipeline (raw.mov + events.jsonl + edits.json
-> video) as MCP tools over stdio, so an LLM client can inspect sessions,
adjust the edit spec, SEE the result via preview_frame, and re-render.

Protocol: newline-delimited JSON-RPC 2.0 on stdin/stdout (one JSON object per
line, no Content-Length framing). stdout carries ONLY protocol JSON lines;
all logging goes to stderr, and every call into the autocine render pipeline is
wrapped in redirect_stdout(sys.stderr) because render/preview print progress.

Hand-rolled on stdlib only: the system Python is 3.9 and the official `mcp`
package requires 3.10+.
"""

import base64
import collections
import contextlib
import datetime
import json
import math
import os
import sys
import traceback
import uuid

from . import __version__
from . import beats as beats_mod
from . import edits as ed
from . import framing
from . import paths
from . import render as ren
from . import retime
from . import segments as seg
from . import transcribe as tx

# Cuts (ripple delete) authoring guards. These now live in edits.py so the
# web editor and the CLI enforce the SAME rule -- "the MCP validates, the
# browser does not" was a way to save a document the renderer refuses. The
# aliases stay because this module's error vocabulary (ToolError) is not
# edits.py's, and because the rationale for each number is written there.
_MAX_CUT_RANGES = ed.CUT_MAX_RANGES
_MIN_KEPT_SEC = ed.CUT_MIN_KEPT_SEC

PROTOCOL_VERSION = "2025-03-26"
SERVER_INFO = {"name": "autocine", "version": __version__}
DEFAULT_RECORDINGS_ROOT = paths.recordings_root()

_SESSION_PROP = {
    "type": "string",
    "description": "Session name (directory basename, e.g. '20260101-120000') from list_sessions.",
}

# The layouts this surface accepts. One tuple for the schema enum, the write
# validator AND the render resolver: a name added to only some of them is
# accepted here and silently rendered as "grid" over there, which is the same
# class of bug the render-kwarg parity tests exist to catch. Must stay in step
# with edits._WINDOW_LAYOUTS -- that normalizer is what actually decides what
# survives a save, so a name missing THERE never reaches edits.json at all.
_WINDOW_LAYOUTS = ("grid", "desktop", "feature", "row", "column")

# The two multi-window cameras are stored as two independent booleans
# (`render.window_focus` / `render.window_zoom`) because they compose. Asking
# a model to reason about the pair every time is how you get the wrong one
# switched on, so the tool surface also takes ONE word for the four
# combinations. This is pure shorthand: it writes the same two flags, adds no
# new persisted field, and `zoom_style` is never stored.
_ZOOM_STYLES = collections.OrderedDict((
    # The default on a record-time multi-window pick (edits._seed_capture_
    # window_render): the worked-in card GROWS in place, then the whole frame
    # pushes in on it.
    ("frame", {"window_focus": True, "window_zoom": False}),
    # Zoom the footage INSIDE a card while its cell stays put.
    ("inside", {"window_focus": False, "window_zoom": True}),
    ("both", {"window_focus": True, "window_zoom": True}),
    ("off", {"window_focus": False, "window_zoom": False}),
))


def _zoom_style_of(render):
    """The one-word name for a render dict's (window_focus, window_zoom)."""
    pair = (bool(render.get("window_focus")), bool(render.get("window_zoom")))
    for name, flags in _ZOOM_STYLES.items():
        if (flags["window_focus"], flags["window_zoom"]) == pair:
            return name
    return "off"


def _build_stamp():
    """Identity of the code THIS PROCESS loaded, captured at import time.

    A `studio.py mcp` process keeps serving the module it imported, so a
    long-running server silently lacks every tool and argument added since it
    started -- which reads to the caller as "that feature was never built"
    (we hit exactly that with four stale servers). Sampling the mtime once at
    import, not per call, is what makes staleness visible: the value stays
    pinned to the code in memory even after the file on disk is edited.
    """
    src = os.path.abspath(__file__)
    try:
        built = datetime.datetime.fromtimestamp(
            os.path.getmtime(src)).replace(microsecond=0).isoformat()
    except OSError:
        built = None
    return {"version": SERVER_INFO["version"], "source": src, "built": built,
            "pid": os.getpid()}


SERVER_BUILD = _build_stamp()


class ToolError(Exception):
    """Tool-level failure: reported as an isError tool result, not a JSON-RPC error."""


def _log(msg):
    try:
        sys.stderr.write(str(msg) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _dumps(payload):
    return json.dumps(payload, separators=(",", ":"))


def _text_content(payload):
    return [{"type": "text", "text": _dumps(payload)}]


def _result(msg_id, result):
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _error(msg_id, code, message):
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": int(code), "message": str(message)}}


def _new_id(prefix):
    return "{}-{}".format(prefix, uuid.uuid4().hex[:8])


# ---------------------------------------------------------------------------
# Argument extraction/validation (tool-level: raise ToolError, never JSON-RPC)
# ---------------------------------------------------------------------------

def _require_str(args, key):
    v = args.get(key)
    if not isinstance(v, str) or not v.strip():
        raise ToolError("missing required argument: {}".format(key))
    return v


def _opt_str(args, key):
    if key not in args or args[key] is None:
        return None
    v = args[key]
    if not isinstance(v, str):
        raise ToolError("argument '{}' must be a string".format(key))
    return v


def _session_output_path(session_dir, value):
    """Resolve an MCP render output without letting it leave the session.

    `realpath` is required in addition to `commonpath`: a lexical child can
    traverse through an existing symlink.  Resolving the requested target also
    catches an existing output-file symlink before ffmpeg's `-y` follows it.
    """
    if "\x00" in value:
        raise ToolError("render output must stay inside the session directory")
    root = os.path.abspath(session_dir)
    root_real = os.path.realpath(root)
    target_real = None
    try:
        expanded = os.path.expanduser(value)
        candidate = (expanded if os.path.isabs(expanded)
                     else os.path.join(root, expanded))
        target = os.path.abspath(candidate)
        target_real = os.path.realpath(target)
        contained = os.path.commonpath((root_real, target_real)) == root_real
    except (OSError, TypeError, ValueError):
        contained = False
        target = None
    if not contained or target == root or target_real == root_real:
        raise ToolError("render output must stay inside the session directory")
    return target


def _num(args, key, required=False):
    if key not in args or args[key] is None:
        if required:
            raise ToolError("missing required argument: {}".format(key))
        return None
    v = args[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ToolError("argument '{}' must be a number".format(key))
    # json.loads accepts bare NaN/Infinity tokens; NaN then no-ops every
    # comparison guard downstream (a NaN cut start sailed past end<=start
    # AND the coverage validation, then persisted as a cut from 0 -- the
    # reviewed failure). Finite-only, for every numeric tool argument.
    if not math.isfinite(float(v)):
        raise ToolError("argument '{}' must be a finite number".format(key))
    return float(v)


def _opt_int(args, key):
    n = _num(args, key)
    return None if n is None else int(round(n))


def _parse_clock(text):
    """'1:23' / '1:23.5' / '01:02:03' / '83' -> seconds, or None."""
    txt = str(text).strip()
    if not txt:
        return None
    parts = txt.split(":")
    if len(parts) > 3:
        return None
    total = 0.0
    for part in parts:
        part = part.strip()
        if not part:
            return None
        try:
            v = float(part)
        except ValueError:
            return None
        if not math.isfinite(v) or v < 0:
            return None
        total = total * 60.0 + v
    return total


def _time_arg(args, key, required=True):
    """A time in seconds from a number OR a 'm:ss' string.

    The one argument a user types verbatim ("that zoom at 1:23"), so it takes
    the shape they say it in as well as the shape every other tool uses.
    """
    v = args.get(key)
    if v is None:
        if required:
            raise ToolError("missing required argument: {}".format(key))
        return None
    if isinstance(v, str):
        secs = _parse_clock(v)
        if secs is None:
            raise ToolError("argument '{}' must be seconds or 'm:ss' (got "
                            "{!r})".format(key, v))
        return secs
    return _num(args, key, required=required)


def _fmt_clock(seconds):
    """Seconds as 'm:ss.s' -- how the user said it, for messages."""
    secs = max(0.0, float(seconds))
    return "{:d}:{:04.1f}".format(int(secs // 60), secs % 60)


def _camera_params(r):
    """Camera DEFAULTS overrides from a normalized render-options dict.
    None when nothing deviates (mirrors studio_app._camera_params_from_render_opts)."""
    cam = {}
    if r.get("always_zoomed"):
        cam["always_zoomed"] = True
    if not r.get("overview", True):
        cam["overview"] = False
    if not r.get("drag_hold", True):
        cam["drag_hold"] = False
    speed = r.get("zoom_speed")
    if speed and speed != "normal":
        cam["zoom_speed"] = speed
    anim = r.get("screen_anim")
    if anim and anim != "focused":
        cam["screen_anim"] = anim
    return cam or None


def _facecam_params(r):
    """FacecamOverlay overrides from a normalized render-options dict.
    None when nothing deviates (mirrors
    studio_app._facecam_params_from_render_opts)."""
    fc = {}
    pos = r.get("facecam_position")
    if pos and pos != "bottom-left":
        fc["position"] = pos
    size = float(r.get("facecam_size") or 0.0)
    if size and abs(size - 0.20) > 1e-6:
        fc["size_frac"] = size
    shape = r.get("facecam_shape")
    if shape and shape != "circle":
        fc["shape"] = shape
    border = float(r.get("facecam_border") or 0.0)
    if border > 0:
        fc["border_frac"] = border
    blur = float(r.get("facecam_blur") or 0.0)
    if blur > 0:
        fc["blur"] = blur
    return fc or None


def _opt_bool(args, key):
    if key not in args or args[key] is None:
        return None
    v = args[key]
    if not isinstance(v, bool):
        raise ToolError("argument '{}' must be a boolean".format(key))
    return v


# ---------------------------------------------------------------------------
# Tool definitions (model-facing docs)
# ---------------------------------------------------------------------------

TOOL_DEFS = [
    {
        "name": "list_sessions",
        "description": (
            "List all recorded sessions (newest first). Each entry has the session "
            "name (used by every other tool), duration in seconds, click_count, "
            "has_edits (an edits.json exists), and has_output_mp4 (already rendered). "
            "Sessions that cannot be read are listed with an 'error' field. "
            "Also returns 'server': the version, pid, and build time of the code "
            "THIS server process is running. If a tool or argument documented "
            "elsewhere seems to be missing, check 'built' — a server left running "
            "keeps serving the code it started with, and the fix is to restart it, "
            "not to reimplement the feature."
        ),
        "inputSchema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "describe_session",
        "description": (
            "Full metadata for one session: fps, width/height in source-video pixels "
            "(the coordinate space for zoom pins), frame_count, duration (seconds), "
            "has_audio, click_count/move_count, cursor_mode, "
            "has_face (a webcam track the facecam* render options apply to), "
            "plus 'edits' (the current persisted edit document), 'chapters' "
            "(ranges derived from markers), 'has_transcript' (whether "
            "get_transcript would answer instantly or has to run an ASR pass "
            "first), and 'server' (this server process's build stamp — see "
            "list_sessions). Call this before editing a session.\n"
            "READ 'beats' FIRST: a time-ordered account of what actually "
            "happened, derived from the recorded events (no frames decoded). "
            "Each beat is 'clicks' (a click cluster, with the zoom span the "
            "camera will plan for it, and a bbox in source pixels), 'scroll', "
            "'typing', 'idle' (dead air — the spans worth speeding up or "
            "trimming), or a window change: 'front' (a window came to the "
            "front — the closest thing this recording has to a scene cut), "
            "'open', 'close'. Beats carry window_id where the recording can "
            "say so; ids are opaque by design — the event log records no app "
            "name or window title, so match them against list_recorded_windows "
            "geometry, or look with preview_frame(source=true). 'notes' says "
            "when a session cannot support part of this (e.g. no geometry "
            "track), and 'truncated' counts beats omitted on a long take — "
            "neither is ever silent. This replaces reading the raw "
            "click_times/scroll_times arrays, which say only that something "
            "happened N times; pass detail=true if you genuinely need them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "detail": {
                    "type": "boolean",
                    "description": (
                        "Include the raw per-event arrays (click_times, "
                        "key_times, scroll_times) and the full presets list. "
                        "Default false: 'beats' summarizes the same events in "
                        "a form you can act on, at a fraction of the size."
                    ),
                },
                "max_beats": {
                    "type": "integer",
                    "description": (
                        "Cap on the number of beats returned (default 300). "
                        "When a take has more, the ones kept are spread "
                        "evenly across it and beats.truncated says how many "
                        "were left out; raise this to see them all."
                    ),
                },
            },
            "required": ["session"],
        },
    },
    {
        "name": "get_edits",
        "description": (
            "Return the session's normalized edit document from edits.json: trim, "
            "render options, manual zoom ranges ('zooms'), auto-zoom suppression "
            "ranges ('suppressed'), and chapter markers. This exact document drives "
            "render_video and preview_frame, so rendering is deterministic."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session": _SESSION_PROP},
            "required": ["session"],
        },
    },
    {
        "name": "set_render_options",
        "description": (
            "Update the session's render 'look' options; ONLY the arguments you pass "
            "change, everything else is preserved (persisted to edits.json). "
            "zoom = max auto-zoom level (default 2.2) -- to change ONE moment "
            "rather than the ceiling for the whole take, use adjust_zoom; "
            "zoom_style is the one-word multi-window camera switch "
            "('frame' / 'inside' / 'both' / 'off'); offset shifts click timestamps "
            "vs the video in seconds (sync nudge); style is 'clean' or 'framed'; "
            "background (gradient preset name like 'sunset', '#hex', or an image "
            "path) implies framed style; aspect is 'auto', a 'W:H' ratio like "
            "'9:16', or exact 'WxH' pixels; fade is intro/outro fade seconds; "
            "cursor_size scales the synthetic cursor; music/click_sound are audio "
            "file paths (pass null to clear); the facecam* options place the "
            "webcam bubble (only visible on sessions recorded with a face track "
            "-- see describe_session's has_face). Verify visually with preview_frame."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "zoom": {"type": "number", "description": "Max auto-zoom level, >= 1 (e.g. 2.2)."},
                "zoom_speed": {"type": "string", "enum": ["slow", "normal", "fast"], "description": "Zoom transition speed preset (default 'normal')."},
                "offset": {"type": "number", "description": "Click-vs-video sync offset in seconds."},
                "style": {"type": "string", "enum": ["clean", "framed"]},
                "background": {"type": ["string", "null"], "description": "Gradient preset, '#hex' color, or image path; implies framed style. Null clears."},
                "aspect": {"type": "string", "description": "'auto', 'W:H' ratio (e.g. '9:16'), or 'WxH' exact pixels."},
                "click_fx": {"type": "boolean", "description": "Draw click ripple rings (default true)."},
                "click_color": {"type": ["string", "null"], "description": "Ripple color name or '#hex'. Null clears."},
                "spotlight": {"type": "boolean", "description": "Dim around the cursor while zoomed."},
                "cursor_fx": {"type": "boolean", "description": "Draw the synthetic cursor. Needs a frame with no pointer already in it: either a recording made with cursor_mode 'synthetic', or cursor_erase true to lift the recorded one out first -- set both in one call to REPLACE the recorded cursor on an ordinary take. Draws nothing on a 'system' take without cursor_erase."},
                "cursor_size": {"type": "number", "description": "Synthetic cursor scale (1.0 = normal)."},
                "cursor_erase": {"type": "boolean", "description": "Erase the RECORDED system cursor from the footage (opposite of cursor_fx, which draws one). Repaints each frame's pointer box from the real pixels visible before it arrived or after it left; costs a second decode of the source and inpaints where the recording never showed those pixels uncovered. Pair it with cursor_fx to swap the recorded pointer for the synthetic one. No-op on a session recorded with cursor_mode 'synthetic'."},
                "fade": {"type": "number", "description": "Fade in/out duration in seconds."},
                "music": {"type": ["string", "null"], "description": "Background music file path. Null clears."},
                "click_sound": {"type": ["string", "null"], "description": "Mouse-click sound. Null/'auto' (the default) plays a built-in click at every recorded click; 'off' silences it; anything else is an audio file path to play instead."},
                "key_sound": {"type": ["string", "null"], "description": "Keystroke sound. Null/'auto' (the default) plays a built-in keystroke at every recorded key-activity tick (see describe_session's key_count/key_capture); 'off' silences it; anything else is an audio file path."},
                "sfx_volume": {"type": "number", "description": "Loudness of the click/keystroke bed, 0 to 2 (default 1.0). 0 silences both."},
                "always_zoomed": {"type": "boolean", "description": "Hold the final zoom level to the end instead of settling back to 1.0."},
                "motion_blur": {"type": "boolean", "description": "Blur fast camera pans/zooms across a shutter window (default true)."},
                "overview": {"type": "boolean", "description": "Overview framing (default true): when a click cluster is spread wider than a full-zoom window can hold, pull back to a still, lower-zoom overview locked on the activity center instead of whip-panning between clicks."},
                "typing_zoom": {"type": "boolean", "description": "Auto-zoom on typing bursts, anchored at the last click (default true; needs key activity in the recording — see describe_session's key_count)."},
                "drag_hold": {"type": "boolean", "description": "Hold the zoom through slow mouse drags, down to up (default true)."},
                "scroll_zoom": {"type": "boolean", "description": "Scroll-aware camera: gentle auto-zoom on sustained scroll-reading, plus holding an active zoom while scrolling (default true; needs scroll activity in the recording — see describe_session's scroll_count)."},
                "window_follow": {"type": "boolean", "description": "Multi-window only: bind each set_windows rect to the recorded window it overlaps and follow that window when it is moved or resized mid-take (default true; needs a recording with a window geometry track). Independent of window_layout -- it decides what each card SHOWS, not where the card sits. Off = the static crops windows mode shipped with."},
                "window_layout": {"type": "string", "enum": list(_WINDOW_LAYOUTS), "description": "Multi-window only: how the cards are arranged. Cards are aspect-locked rectangles scaled by ONE uniform factor, so NO arrangement fills both axes: it meets the padding on one and leaves margin on the other, and which arrangement wastes least is decided by the export aspect -- pick against the `aspect` you are exporting at. THE RULE: use 'feature' unless you want something specific; it is the only arrangement that TRANSPOSES (hero on the left with the rest in a column beside it, or hero on TOP with the rest in a row beneath when the canvas is taller than that wants), which is what makes it the best of the five on all four canvases measured. 'grid' runs it closest on a vertical export, 'desktop' on a wide one, and 'row' is never the best of the five for three windows. The arrangements: 'grid' (default) gives every card an equal cell and adapts by choosing rows x cols against the canvas orientation; 'desktop' keeps the windows' relative on-screen arrangement -- same left-to-right and top-to-bottom order, same relative sizes -- with overlaps pulled apart, dead space squeezed out and the result scaled to fit; 'feature' gives the first window a large panel with the remaining cards as one block beside it (the '1 big + 2 small' arrangement); 'row' lays every card side by side and 'column' stacks them top to bottom, both in card order and neither ever transposed -- reach for those when the reading order matters more than the coverage. Measured on the three real recorded windows this was built for (1418x1718, 1418x852, 1418x854), fraction of the canvas actually covered -- 16:10 export: feature 81.8%, desktop 80.7%, grid 34.9%, row 33.0%, column 20.0%; 9:16 export: feature 74.9%, grid 73.2%, column 67.5%, desktop 29.1%, row 11.8%; 1:1: feature 56.6%, desktop 51.8%, column 35.4%, grid 21.8%, row 20.9%. TWO windows rank differently from three -- on a 9:16 export 'feature' and 'column' tie at 90.2% against grid's 63.4% -- so treat this as the shape of the trade-off, not a law. Sessions recorded with a multi-window pick default to 'desktop'."},
                "zoom_style": {"type": "string", "enum": list(_ZOOM_STYLES), "description": "Multi-window only, and the ONE-WORD shorthand for the window_focus/window_zoom pair below -- prefer it over setting the two booleans by hand. 'frame': the card being worked in GROWS in place and then the whole composition pushes in on it (the outer-frame zoom). This is what a take recorded with a multi-window pick already defaults to, so you rarely have to set it -- reach for this to put it back. 'inside': no composition move at all; the footage zooms INSIDE each card while its cell stays put. 'both': the two compose (the frame leans toward a card AND that card's own camera pushes in). 'off': the layout holds steady and nothing zooms. It writes render.window_focus / render.window_zoom and nothing else, so an explicit boolean passed in the same call wins over it."},
                "window_zoom": {"type": "boolean", "description": "Multi-window only: auto-zoom INSIDE the cards (zoom_style 'inside'). Each card gets its own camera planned from the clicks that landed in it, and only the card with the most recent activity is zoomed at any moment -- the others hold their full framing, so the composition stays readable. OFF by default even on a take recorded with a multi-window pick: that take's default camera is the outer-frame one (window_focus), and zooming inside a card is a different look rather than a stronger version of it. The two compose -- zoom_style 'both'."},
                "window_focus": {"type": "boolean", "description": "Multi-window only: focus the window being worked in (zoom_style 'frame'). Clicking in a window grows that card in place and leans the whole composition toward it; clicking again in the same burst pushes the frame in until that card fills it, and it eases back out when attention moves to another window. Only one card is ever the subject. Distinct from window_zoom, which zooms the footage INSIDE a card while its cell stays put — the two compose. ON by default on any session recorded with a multi-window pick (that pick is what makes the cards, so the camera that moves the cards is the one that take opens with); off on hand-drawn set_windows cards. Soften or drop ONE of its moves with adjust_zoom rather than switching the whole thing off."},
                "badge_erase": {"type": "boolean", "description": "Occlusion-free (window-native) takes only: paint out macOS's per-window capture indicator -- the rounded pill the system draws over each recorded window's traffic lights, which is burned into the captured pixels. ON by default. No-op on every other capture mode (a whole-screen take has no per-window badge) and on a window macOS never marked (a sheet or dialog has no window buttons to cover). Set false to leave the footage exactly as recorded."},
                "screen_focus": {"type": "boolean", "description": "Whole-screen (non-multi-window) takes only: when an auto-zoom's clicks land inside ONE on-screen window and other windows are present, grow that window a bit in place to overlap its neighbours while the frame eases in slightly ('meeting in the middle'), instead of framing a crop that cuts into a neighbour. Uses the recorded per-window geometry to pick the owner. ON by default; a no-op when fewer than two windows were recorded, when the clicks span several windows, or when the window nearly fills the frame."},
                "facecam": {"type": "boolean", "description": "Composite the webcam bubble over the video (default true; a no-op unless the session was recorded with a face track — see describe_session's has_face)."},
                "facecam_position": {"type": "string", "enum": ["bottom-left", "bottom-right", "top-left", "top-right"], "description": "Corner the webcam bubble sits in (default 'bottom-left')."},
                "facecam_size": {"type": "number", "description": "Bubble diameter as a fraction of output height (default 0.20; clamped to [0.08, 0.5])."},
                "facecam_shape": {"type": "string", "enum": ["circle", "rounded"], "description": "Bubble outline: 'circle' (default) or 'rounded' square."},
                "facecam_blur": {"type": "number", "description": "Background blur strength for the bubble, 0..1 (default 0 = off). Keeps the face sharp and softens toward the rim -- a vignette, not a person cutout."},
                "speedup": {"type": "boolean", "description": "Automatically speed up idle stretches (no clicks/keys/scrolls/purposeful motion) with a smooth quintic ramp. OFF by default; when ON, output duration shrinks -- combine with add_speedup(mode='force') or add_speedup(mode='off') to override auto detection."},
                "speedup_rate": {"type": "number", "description": "Peak speed factor inside auto-detected idle spans (default 6.0; clamped to [1.5, 12.0])."},
                "speedup_silence_gate": {"type": "boolean", "description": "Only speed spans that are ALSO audio-silent (default true). Turning off risks chipmunk narration; ignored on sessions without a mic track."},
                "speedup_motion_gate": {"type": "boolean", "description": "Only speed spans that are ALSO visually still (default true). Turning off risks time-lapsing a playing video, scrolling build log, or download animation into a glitchy blur."},
                "gif": {"type": "boolean", "description": "Also produce an animated GIF when rendering."},
                "gif_fps": {"type": "integer", "description": "GIF frame rate (default 15)."},
                "gif_width": {"type": "integer", "description": "GIF width in pixels (default 1000)."},
            },
            "required": ["session"],
        },
    },
    {
        "name": "set_trim",
        "description": (
            "Set the export trim range in seconds from the start of the recording. "
            "Only provided keys change; end=null means 'through the end of the "
            "clip'. Values are clamped to the clip duration. Persists to edits.json."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "start": {"type": "number", "description": "Trim start in seconds."},
                "end": {"type": ["number", "null"], "description": "Trim end in seconds, or null for end of clip."},
            },
            "required": ["session"],
        },
    },
    {
        "name": "add_zoom",
        "description": (
            "Add a manual zoom-in over [start, end] seconds. level is the zoom "
            "factor (default 2.0, clamped to [1, 8]). Pass x AND y (source-video "
            "pixels — see describe_session width/height) to pin the zoom to a fixed "
            "point; omit both to have the camera follow the cursor. Manual ranges "
            "that overlap auto-detected click zooms merge into one smooth hold. "
            "Returns the saved (clamped/normalized) zooms array — note the entry's "
            "id for remove_zoom. Pick x/y off preview_frame with source=true (scaling "
            "by its source_scale); the ordinary preview is already zoomed, so a pin "
            "read off it lands somewhere else. Verify with preview_frame at a time "
            "inside the range."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "start": {"type": "number", "description": "Zoom-range start in seconds."},
                "end": {"type": "number", "description": "Zoom-range end in seconds."},
                "level": {"type": "number", "description": "Zoom factor (default 2.0, max 8)."},
                "x": {"type": "number", "description": "Pin X in source-video pixels (requires y)."},
                "y": {"type": "number", "description": "Pin Y in source-video pixels (requires x)."},
            },
            "required": ["session", "start", "end"],
        },
    },
    {
        "name": "adjust_zoom",
        "description": (
            "Change how hard the camera moves at ONE moment, addressed by TIME. "
            "This is the tool for \"that zoom at 1:23 was too aggressive\" "
            "(change='softer'), \"push in harder there\" ('stronger') and "
            "\"don't zoom there at all\" ('off') -- reach for it instead of "
            "removing and re-adding a range, and instead of set_render_options "
            "zoom, which is the ceiling for the WHOLE take. `at` takes seconds "
            "or 'm:ss'; it is source-media time, the same clock as "
            "describe_session's beats, click_times, trim and every zoom range "
            "(so on a take with a trim/cuts/speed-ups the time you read off the "
            "exported video is NOT this clock -- the response says so when they "
            "differ). It finds the camera move covering that moment, or the "
            "nearest one within 3s, and reports which it matched plus the "
            "before/after -- report those numbers rather than the ones you "
            "asked for. It handles both kinds of take so you don't have to know "
            "which this is: on a whole-screen take it retunes the zoom range in "
            "edits.zooms (softer divides the level by 1.3; below 1.2 the range "
            "is removed instead, and the response carries the start/end to "
            "add_zoom it back), materializing the auto-zoom proposals first if "
            "this session never had them so softening one arc doesn't silently "
            "delete the rest. On a MULTI-WINDOW take the move is the "
            "composition camera: softer demotes a whole-frame push-in back to "
            "the card merely growing in place, and a move that is already just "
            "a grow is left alone (change='off' removes it -- there is no tool "
            "to add one back). Verify with preview_frame inside the range, or "
            "by re-reading describe_session's beats. A SCENE take (the window "
            "set changes mid-recording) is the one shape with no per-move "
            "surface yet -- every scene is its own fleet, planned at render "
            "time -- and is refused with that reason plus the take-wide "
            "levers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "at": {"type": ["number", "string"], "description": "When the move you mean happens: seconds (83.0) or a clock string ('1:23'). Source-media time."},
                "change": {"type": "string", "enum": ["softer", "stronger", "off"], "description": "Which way to move it (default 'softer'). 'off' removes the move entirely."},
                "level": {"type": "number", "description": "Whole-screen takes only: set this exact zoom factor instead of stepping (>= 1, clamped to 8). Ignored by 'off'; rejected on a multi-window take, whose move has no zoom factor."},
            },
            "required": ["session", "at"],
        },
    },
    {
        "name": "remove_zoom",
        "description": "Remove a manual zoom range by its id (from add_zoom or get_edits). Returns the remaining zooms.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "zoom_id": {"type": "string", "description": "The zoom range id, e.g. 'zoom-1a2b3c4d'."},
            },
            "required": ["session", "zoom_id"],
        },
    },
    {
        "name": "add_speedup",
        "description": (
            "Add a manual auto-speed-up override for [start, end] seconds. "
            "mode='force' retimes this stretch even if auto detection would "
            "not (great for a long file-download wait with visible activity "
            "you still want compressed). mode='off' keeps this stretch at "
            "real time, subtracting from auto detection (protects a moment "
            "the viewer must see even if the event stream is quiet). rate "
            "is optional (defaults to render.speedup_rate; clamped to "
            "[1.5, 12.0]). Returns the saved speedups array; note the id "
            "for remove_speedup. Force ranges take effect even when "
            "render.speedup is false."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "start": {"type": "number", "description": "Range start in seconds."},
                "end": {"type": "number", "description": "Range end in seconds."},
                "mode": {"type": "string", "enum": ["force", "off"], "description": "'force' to speed this stretch, 'off' to keep it at real time. Default 'off'."},
                "rate": {"type": "number", "description": "Peak speed factor (default: render.speedup_rate). Clamped to [1.5, 12.0]."},
            },
            "required": ["session", "start", "end"],
        },
    },
    {
        "name": "remove_speedup",
        "description": "Remove a manual speed-up override by its id. Returns the remaining speedups.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "speedup_id": {"type": "string", "description": "The speedup range id, e.g. 'speedup-1a2b3c4d'."},
            },
            "required": ["session", "speedup_id"],
        },
    },
    {
        "name": "add_cut",
        "description": (
            "CUT [start, end] seconds OUT of the exported video (ripple "
            "delete): the removed range vanishes, what follows slides "
            "earlier, mic audio stays in sync, and no zoom or click sound "
            "fires for events inside it. Times are the same source-media "
            "clock every other tool uses (transcript word times, add_zoom, "
            "set_trim), so a find_in_transcript hit can be cut directly. "
            "Boundaries snap outward to the frame grid -- the response's "
            "`snapped` range is what the export will actually remove -- "
            "and the response reports the resulting output duration. "
            "Returns the saved cuts array; note the id for remove_cut. "
            "Overlapping cuts are fine (they merge at render time). The "
            "editor timeline shows cuts as dimmed ranges and skips them "
            "during playback, but SCRUBBING still shows removed frames -- "
            "only the export ripples; that is a known v1 scope limit, not "
            "a bug. Cuts do not yet apply to scene or multi-window takes "
            "(the render prints a note and exports un-cut)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "start": {"type": "number", "description": "Cut start in seconds (source-media clock)."},
                "end": {"type": "number", "description": "Cut end in seconds."},
            },
            "required": ["session", "start", "end"],
        },
    },
    {
        "name": "remove_cut",
        "description": "Remove one cut by its id (from add_cut/set_cuts/get_edits). Returns the remaining cuts.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "cut_id": {"type": "string", "description": "The cut id, e.g. 'cut-1a2b3c4d'."},
            },
            "required": ["session", "cut_id"],
        },
    },
    {
        "name": "set_cuts",
        "description": (
            "Replace the WHOLE cuts array in one call -- the way to apply "
            "several transcript-located cuts at once ('remove all three "
            "stutters'): pass every range you want cut; pass [] to clear "
            "them all. Same clock, snapping, and response fields as "
            "add_cut. Unlike add_cut this REPLACES existing cuts, so read "
            "get_edits first if you mean to keep any."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "cuts": {
                    "type": "array",
                    "description": "Every cut to keep, as {start, end} in seconds ({id} optional to preserve an existing entry's id).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "id": {"type": "string"},
                        },
                        "required": ["start", "end"],
                    },
                },
            },
            "required": ["session", "cuts"],
        },
    },
    {
        "name": "add_marker",
        "description": (
            "Add a chapter marker at time (seconds) with an optional label. Markers "
            "define chapters: each chapter runs from its marker to the next marker "
            "or the trim end. Returns the saved markers plus the derived chapters."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "time": {"type": "number", "description": "Marker time in seconds."},
                "label": {"type": "string", "description": "Optional chapter label."},
            },
            "required": ["session", "time"],
        },
    },
    {
        "name": "remove_marker",
        "description": "Remove a chapter marker by its id. Returns the remaining markers and recomputed chapters.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "marker_id": {"type": "string", "description": "The marker id, e.g. 'marker-1a2b3c4d'."},
            },
            "required": ["session", "marker_id"],
        },
    },
    {
        "name": "list_recorded_windows",
        "description": (
            "Every window whose geometry this recording captured, as "
            "{window_id, x, y, w, h, samples, occluded_sec} in source-video "
            "pixels. Use it to build a multi-window layout WITHOUT measuring "
            "rects off a preview frame: pass a window_id on each set_windows "
            "entry and its card is pinned to that exact window, so it follows "
            "the window instead of render inferring which one you meant by "
            "overlap. `occluded_sec` is how long something else sat on top of "
            "that window during the take -- capture is display-capture-plus-"
            "crop, so those seconds are recorded into that window's card and "
            "a non-zero value is a real quality warning, not a hint. The rect "
            "is the window's median position over the take. Geometry and "
            "opaque ids only -- app names and titles are deliberately never "
            "recorded. Empty for sessions recorded before geometry tracking."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session": _SESSION_PROP},
            "required": ["session"],
        },
    },
    {
        "name": "set_crop",
        "description": (
            "Crop the recording: one {x, y, w, h} rect in source-video pixels "
            "(the same space add_zoom's x/y and set_windows' rects use — see "
            "describe_session's width/height). Every frame is sliced to it "
            "before the camera runs, so the export comes out at the crop's "
            "size and the auto-zoom, click FX, cursor and framing all keep "
            "working inside it — unlike set_windows, which replaces the "
            "whole-screen camera. This is the spatial sibling of set_trim: "
            "use it to cut away a menu bar, a second monitor's worth of "
            "empty desk, or any dead margin. Sides are rounded to even "
            "numbers (h264 rejects odd ones) and clamped into the frame; a "
            "rect under 16px a side, or one covering the whole frame, is "
            "treated as no crop. Pass null to go back to the full frame. "
            "MEASURE the rect on preview_frame with source=true, scaling by "
            "the source_scale it reports — the ordinary preview is "
            "auto-zoomed, so a rect read off it is silently wrong."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "crop": {
                    "type": ["object", "null"],
                    "description": "Crop rect, e.g. {\"x\":0,\"y\":48,\"w\":2880,\"h\":1752}. null clears it.",
                    "properties": {
                        "x": {"type": "number"},
                        "y": {"type": "number"},
                        "w": {"type": "number"},
                        "h": {"type": "number"},
                    },
                    "required": ["x", "y", "w", "h"],
                },
            },
            "required": ["session", "crop"],
        },
    },
    {
        "name": "set_windows",
        "description": (
            "Replace the session's multi-window layout: an array of up to "
            "4 {x, y, w, h} crop rects, in source-video pixels (same "
            "coordinate space as add_zoom's x/y — see describe_session's "
            "width/height). When non-empty, rendering switches from the "
            "whole-screen auto-zoom camera to framed 'window' cards on a "
            "background (a polished window-capture look, generalized to "
            "1-4 sources cropped from this one recording). What that mode "
            "then offers: set_render_options `window_layout` picks one of "
            "five arrangements (grid/desktop/feature/row/column, chosen "
            "against your export aspect); `window_zoom` gives each card its "
            "own click-driven camera, with only the most recently active "
            "card ever zoomed; `window_focus` leans the composition toward "
            "the card you are working in and then pushes the whole frame in "
            "on it; cards can also be placed by hand (a 'layout' box on the "
            "entry, which is what dragging a card in the editor writes) and "
            "fit_windows grows a hand-placed arrangement until it meets the "
            "canvas padding. motion_blur/click_fx/spotlight need a single "
            "frame-filling camera window and are skipped here with a printed "
            "note; cursor_fx is NOT skipped -- the cursor is positional, so "
            "it is drawn through each card's own crop. Pass an empty "
            "array to turn "
            "windows mode back off; passing rects again also drops any "
            "hand placement, which is the way back to a pure arrangement. "
            "MEASURE the rects on preview_frame with source=true, scaling by "
            "the source_scale it reports — the ordinary preview is auto-zoomed, "
            "so rects read off it are silently wrong. Verify with a normal "
            "preview_frame afterwards."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "windows": {
                    "type": "array",
                    "description": "Up to 4 crop rects, e.g. [{\"x\":0,\"y\":0,\"w\":800,\"h\":600}].",
                    "items": {
                        "type": "object",
                        "properties": {
                            "x": {"type": "number"},
                            "y": {"type": "number"},
                            "w": {"type": "number"},
                            "h": {"type": "number"},
                            "window_id": {"type": "integer", "description": "Optional: pin this card to a real recorded window (ids from list_recorded_windows) so it follows that window exactly, instead of render inferring which window the rect is of by overlap."},
                        },
                        "required": ["x", "y", "w", "h"],
                    },
                },
            },
            "required": ["session", "windows"],
        },
    },
    {
        "name": "add_window",
        "description": (
            "Append one crop rect (source-video pixels) to the session's "
            "window layout, up to 4 total. Turns on multi-window mode if "
            "this is the first entry. Returns the saved windows array — "
            "note the entry's id for remove_window."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "x": {"type": "number", "description": "Crop rect X in source-video pixels."},
                "y": {"type": "number", "description": "Crop rect Y in source-video pixels."},
                "w": {"type": "number", "description": "Crop rect width in source-video pixels."},
                "h": {"type": "number", "description": "Crop rect height in source-video pixels."},
            },
            "required": ["session", "x", "y", "w", "h"],
        },
    },
    {
        "name": "fit_windows",
        "description": (
            "Rescale the session's CURRENT multi-window arrangement to take "
            "up the canvas. Takes the cards exactly as they sit now -- "
            "whichever window_layout is set, plus any placement overrides "
            "already on them -- and grows the whole composition about its own "
            "bounding box until it meets the canvas padding, so every card "
            "keeps its aspect, its relative position and its relative size, "
            "and only the empty margin goes away. WHERE IT PAYS: a "
            "HAND-PLACED arrangement -- cards dragged in the editor, or "
            "'layout' boxes written directly -- because nothing else ever "
            "rescales a hand placement. Measured on a real dragged layout, "
            "canvas coverage went 8% -> 77% -- 9.4x the ink, which is one "
            "uniform 3.1x on each card's width and height. On a "
            "freshly applied window_layout it is a NO-OP: every arrangement "
            "is already produced by this same uniform fit, so there is no "
            "slack left to take up and no card changes size ('grid' can "
            "still re-centre the composition, but nothing grows). That is "
            "NOT the same as an arrangement filling the frame -- one uniform "
            "scale can only meet the padding on ONE axis, and the margin on "
            "the other is answered by choosing a different window_layout or "
            "export aspect, never by fitting. "
            "The fit is written as a per-card 'layout' override (0-1 "
            "canvas fractions -- the same thing dragging a card in the editor "
            "writes), which means it OUTLIVES the arrangement it was computed "
            "from: change window_layout afterwards and the overrides still "
            "win. Call set_windows to drop them and go back to the pure "
            "arrangement. Returns the saved windows array plus the canvas the "
            "fractions are of. Verify with preview_frame."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session": _SESSION_PROP},
            "required": ["session"],
        },
    },
    {
        "name": "remove_window",
        "description": "Remove one window rect by its id (from add_window/set_windows/get_edits). Returns the remaining windows.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "window_id": {"type": "string", "description": "The window rect id, e.g. 'window-1a2b3c4d'."},
            },
            "required": ["session", "window_id"],
        },
    },
    {
        "name": "reset_edits",
        "description": (
            "Reset the session's edits.json to defaults: no trim, no manual zooms, "
            "no suppressions, no markers, default render options. Returns the "
            "defaults document. On a session recorded with a multi-window pick "
            "the defaults include that pick (its cards, the 'desktop' "
            "arrangement and the composition camera), because that is what the "
            "take opens as. This cannot be undone."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"session": _SESSION_PROP},
            "required": ["session"],
        },
    },
    {
        "name": "render_video",
        "description": (
            "Render the session to MP4 using the saved edits (trim, manual zooms, "
            "suppressions, markers-agnostic render options). Blocking — may take a "
            "while for long recordings. Rendering is deterministic: the same "
            "recording + edits.json always produces the same video, so it is always "
            "safe to re-render after changing edits. 'out' overrides the output "
            "path (default <session>/output.mp4); 'gif' also writes an animated "
            "GIF. Returns {\"out_path\": ...}."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "out": {"type": "string", "description": "Output file path inside the session directory; relative paths are resolved from that directory."},
                "gif": {"type": "boolean", "description": "Also write an animated GIF next to the MP4 (overrides the saved gif option)."},
            },
            "required": ["session"],
        },
    },
    {
        "name": "preview_frame",
        "description": (
            "Render ONE frame at time (seconds, clamped to the trim range) with all "
            "saved edits applied, returned as a JPEG image. THIS IS HOW YOU SEE "
            "YOUR EDITS: after add_zoom / set_trim / set_render_options, call this "
            "at a relevant time to visually verify framing, zoom level, background, "
            "and effects, and self-correct before the (slower) render_video. "
            "max_width (default 900) downscales the preview image. "
            "Pass source=true to get the RAW source frame instead (no camera, "
            "no effects, no background) -- do that whenever you need to READ "
            "COORDINATES off the image for set_windows or add_zoom, because "
            "the normal preview is auto-zoomed and rects measured on it are "
            "silently wrong."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "time": {"type": "number", "description": "Time in seconds to preview (clamped to the trim range)."},
                "max_width": {"type": "integer", "description": "Max preview width in pixels (default 900)."},
                "source": {
                    "type": "boolean",
                    "description": (
                        "Return the unedited SOURCE frame (the coordinate space "
                        "set_windows / add_zoom use) instead of the rendered "
                        "output. Use this to measure crop rects and zoom pins."
                    ),
                },
            },
            "required": ["session", "time"],
        },
    },
    {
        "name": "get_transcript",
        "description": (
            "Speech-to-text for the session's audio: 'words' ({t, dur, text, conf} "
            "in SECONDS on the same clock as click_times, trim and every zoom "
            "range), 'segments' (sentences), 'text' (the whole narration), and "
            "'silences' (stretches with no speech -- the dead air). USE THIS to "
            "locate a moment by what was SAID instead of guessing timestamps, then "
            "express the edit with the normal tools (add_cut for 'remove "
            "the part where I...', set_trim, add_zoom, add_speedup, "
            "add_marker). "
            "Pass start/end to fetch only part of a long take. "
            "The FIRST call on a session runs a local ASR pass over the whole "
            "recording and can take minutes; it is cached in transcript.json and "
            "instant afterwards. When status is not 'ok' there is no transcript "
            "and 'reason' says why (no ASR installed, no audio track, no speech) "
            "-- report that rather than inventing timings."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "start": {"type": "number", "description": "Only return words at/after this time (seconds)."},
                "end": {"type": "number", "description": "Only return words at/before this time (seconds)."},
                "include_words": {
                    "type": "boolean",
                    "description": (
                        "Include the word-level array (default true). Set false "
                        "for a compact sentences-only view of a long take."
                    ),
                },
            },
            "required": ["session"],
        },
    },
    {
        "name": "find_in_transcript",
        "description": (
            "Find where a phrase was SPOKEN. Returns every match as {t, dur, text, "
            "context} in seconds -- 'context' is the surrounding sentence, so you "
            "can tell which of three identical phrases you have. Matching ignores "
            "case, punctuation and word-splitting. "
            "This is the fast way to answer 'zoom in when I say X', 'remove the "
            "part where I stutter', 'mark each section': find the span, then hand "
            "its times to add_cut/set_cuts (removals), set_trim, add_zoom, or "
            "add_marker. Transcribes on first use, "
            "like get_transcript; an empty 'matches' means the phrase was not said "
            "(check 'status' first -- a session with no transcript also matches "
            "nothing)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "session": _SESSION_PROP,
                "query": {"type": "string", "description": "Words to find, e.g. 'the demo'."},
                "limit": {"type": "integer", "description": "Max matches to return (default 20)."},
            },
            "required": ["session", "query"],
        },
    },
]


class McpServer(object):
    """Newline-delimited JSON-RPC 2.0 MCP server over stdio.

    The protocol loop (serve/handle_line) is separable from the logic:
    handle_message(dict) -> dict-or-None can be driven directly in tests.
    """

    def __init__(self, recordings_root=None):
        self.recordings_root = os.path.abspath(
            recordings_root or DEFAULT_RECORDINGS_ROOT)
        self._handlers = {
            "list_sessions": self._tool_list_sessions,
            "describe_session": self._tool_describe_session,
            "get_edits": self._tool_get_edits,
            "set_render_options": self._tool_set_render_options,
            "set_trim": self._tool_set_trim,
            "add_zoom": self._tool_add_zoom,
            "adjust_zoom": self._tool_adjust_zoom,
            "remove_zoom": self._tool_remove_zoom,
            "add_speedup": self._tool_add_speedup,
            "remove_speedup": self._tool_remove_speedup,
            "add_cut": self._tool_add_cut,
            "remove_cut": self._tool_remove_cut,
            "set_cuts": self._tool_set_cuts,
            "add_marker": self._tool_add_marker,
            "remove_marker": self._tool_remove_marker,
            "list_recorded_windows": self._tool_list_recorded_windows,
            "set_crop": self._tool_set_crop,
            "set_windows": self._tool_set_windows,
            "add_window": self._tool_add_window,
            "fit_windows": self._tool_fit_windows,
            "remove_window": self._tool_remove_window,
            "reset_edits": self._tool_reset_edits,
            "render_video": self._tool_render_video,
            "preview_frame": self._tool_preview_frame,
            "get_transcript": self._tool_get_transcript,
            "find_in_transcript": self._tool_find_in_transcript,
        }

    # -- protocol loop ------------------------------------------------------

    def serve(self, stdin=None, stdout=None):
        stdin = stdin if stdin is not None else sys.stdin
        stdout = stdout if stdout is not None else sys.stdout
        _log("autocine MCP server on stdio (recordings root: {})".format(
            self.recordings_root))
        while True:
            try:
                line = stdin.readline()
            except KeyboardInterrupt:
                return 0
            if line == "":  # EOF -> clean exit
                return 0
            line = line.strip()
            if not line:
                continue
            resp = self.handle_line(line)
            if resp is None:
                continue
            try:
                stdout.write(_dumps(resp) + "\n")
                stdout.flush()
            except (BrokenPipeError, OSError):
                return 0

    def handle_line(self, line):
        try:
            msg = json.loads(line)
        except Exception:
            return _error(None, -32700, "Parse error")
        if not isinstance(msg, dict):
            return _error(None, -32600, "Invalid Request")
        return self.handle_message(msg)

    def handle_message(self, msg):
        """Handle one decoded JSON-RPC message. Returns a response dict for
        requests (messages carrying an "id"), None for notifications."""
        method = msg.get("method")
        has_id = "id" in msg
        msg_id = msg.get("id")
        params = msg.get("params")

        if not has_id:
            # Notification: never respond. Known ones (notifications/initialized,
            # notifications/cancelled) are no-ops; unknown ones are ignored.
            return None
        if not isinstance(method, str) or not method:
            return _error(msg_id, -32600, "Invalid Request")

        if method == "initialize":
            proto = None
            if isinstance(params, dict):
                proto = params.get("protocolVersion")
            if not (isinstance(proto, str) and proto):
                proto = PROTOCOL_VERSION
            return _result(msg_id, {
                "protocolVersion": proto,
                "capabilities": {"tools": {}},
                "serverInfo": dict(SERVER_INFO),
            })
        if method == "ping":
            return _result(msg_id, {})
        if method == "tools/list":
            return _result(msg_id, {"tools": [dict(t) for t in TOOL_DEFS]})
        if method == "tools/call":
            return self._handle_tools_call(msg_id, params)
        return _error(msg_id, -32601, "Method not found: {}".format(method))

    def _handle_tools_call(self, msg_id, params):
        if not isinstance(params, dict):
            return _error(msg_id, -32602, "Invalid params: expected an object")
        name = params.get("name")
        if not isinstance(name, str) or not name:
            return _error(msg_id, -32602, "Invalid params: missing tool name")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return _error(msg_id, -32602, "Invalid params: arguments must be an object")
        handler = self._handlers.get(name)
        if handler is None:
            return _error(msg_id, -32602, "Unknown tool: {}".format(name))
        try:
            # The render pipeline prints progress to stdout; stdout must carry
            # only protocol JSON, so divert everything to stderr for the call.
            with contextlib.redirect_stdout(sys.stderr):
                content = handler(arguments)
            return _result(msg_id, {"content": content, "isError": False})
        except ToolError as exc:
            return _result(msg_id, {
                "content": [{"type": "text", "text": "Error: {}".format(exc)}],
                "isError": True,
            })
        except Exception as exc:
            _log("tool '{}' failed: {}\n{}".format(name, exc, traceback.format_exc()))
            return _result(msg_id, {
                "content": [{"type": "text", "text": "Error: {}".format(exc)}],
                "isError": True,
            })

    # -- session helpers -----------------------------------------------------

    def _session_dir(self, name):
        """Resolve + validate a session name exactly like the web app does."""
        root = self.recordings_root
        ok = bool(name) and isinstance(name, str)
        if ok and os.path.basename(name) != name:
            ok = False
        path = os.path.abspath(os.path.join(root, name)) if ok else None
        if ok and path != root and not path.startswith(root + os.sep):
            ok = False
        if ok and not os.path.isdir(path):
            ok = False
        if ok and not os.path.isfile(os.path.join(path, "meta.json")):
            ok = False
        if not ok:
            raise ToolError("unknown session: {}".format(name))
        return path

    def _duration_or_none(self, session_dir):
        """Clip duration for edits clamping; None if the recording is unreadable
        (edits-only tools must still work on a session with a broken raw.mov)."""
        try:
            return float(ren.describe_session(session_dir)["duration"])
        except Exception:
            return None

    def _edit_session(self, session):
        session_dir = self._session_dir(session)
        duration = self._duration_or_none(session_dir)
        current = self._load_seeded_edits(session_dir, duration)
        return session_dir, duration, current

    def _load_seeded_edits(self, session_dir, duration):
        """Load edits, materializing a record-time multi-window pick once.

        `studio_app._load_edits_with_auto` and the CLI render path both do
        this on first touch, and the MCP being the one surface that didn't
        meant a take recorded with a window pick answered here with the
        HAND-DRAWN defaults -- flat grid, no camera -- while the same
        edits.json opened in the app showed the desktop arrangement with the
        composition camera on. One edits.json, one video: whichever surface
        touches the session first is the one that materializes the pick.

        Only sessions that actually HAVE a pick are seeded (a whole-screen
        take is left alone, disk included), and the flag makes it a one-shot.
        A failed write is not an error: the in-memory doc is still correct and
        the next call retries.
        """
        doc = ed.load_edits(session_dir, duration=duration)
        if doc.get("capture_windows_initialized"):
            return doc
        native = self._is_multi_native(session_dir)
        try:
            specs = ren.capture_window_specs(session_dir)
        except Exception:
            specs = []
        if not native and not specs:
            return doc
        seeded, changed = ed.initialize_capture_windows(
            doc, specs, multi_native=native)
        if not changed:
            return doc
        try:
            return ed.save_edits(session_dir, seeded, duration=duration)
        except Exception:
            return seeded

    def _is_multi_native(self, session_dir):
        """True for a session with N window buffers and NO single source
        coordinate space: a P3.1 multi-native take OR a scene take (whose
        scenes are each a fleet -- docs/architecture.md). Both refuse manual
        spatial edits for the same reason.

        Read straight off meta.json rather than describe_session so the guard
        is cheap and cannot itself fail on an unreadable channel file."""
        try:
            with open(os.path.join(session_dir, "meta.json")) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return False
        return ren._is_multi_native_meta(meta) or seg.is_scene_meta(meta)

    def _refuse_manual_spatial_edit(self, session_dir, what):
        """Raise if `session_dir` is multi-native and `what` is a spatial edit
        with no per-card addressing surface (v1 scope cut).

        Manual add_zoom (with coords) / set_crop / set_windows all assume a
        single source coordinate space; a multi-native take has N window
        buffers and no way yet to say WHICH card a coordinate belongs to.
        Rather than silently apply it to the wrong space (or nothing), refuse
        with a pointer to the automatic per-card camera, which is how zoom is
        expressed on this take. See docs/architecture.md."""
        if self._is_multi_native(session_dir):
            raise ToolError(
                "{} is not supported on a multi-window native take: it has "
                "N window buffers and no per-card coordinate surface yet. "
                "Per-card zoom is automatic -- set render.window_zoom / "
                "render.window_focus via set_render_options instead."
                .format(what))

    def _save_patch(self, session_dir, current, patch, duration):
        merged = ed.merge_edits(current, patch, duration=duration)
        return ed.save_edits(session_dir, merged, duration=duration)

    def _render_kwargs(self, edits_obj):
        """Map a normalized edits doc to render()/preview_frame() kwargs,
        exactly like studio_app.StudioState._render_kwargs_from_edits.

        KEEP THE KEY SET IDENTICAL to that method's (modulo this one's
        `params` being its `camera_params`) -- an option resolved there but
        not here renders differently depending on which surface you asked,
        silently, because the missing kwarg falls back to render()'s
        signature default instead of erroring. `speedup*` did exactly that:
        edits.speedups written by this server's own add_speedup were dropped
        on the floor by render_video. `tests/test_mcp_server.py`'s
        `RenderKwargsParityTests` pins the two together, key and value.
        """
        r = edits_obj.get("render", {})
        click_color = r.get("click_color")
        return {
            "max_zoom": float(r.get("zoom", 2.0)),
            "offset": float(r.get("offset", 0.0)),
            "style": str(r.get("style", "clean")),
            "background": r.get("background"),
            "click_fx": bool(r.get("click_fx", True)),
            "click_params": ({"color": click_color} if click_color else None),
            "spotlight": bool(r.get("spotlight", False)),
            "cursor_fx": bool(r.get("cursor_fx", False)),
            "cursor_params": {"scale": float(r.get("cursor_size", 1.0))},
            "cursor_erase": bool(r.get("cursor_erase", False)),
            "aspect": r.get("aspect") or "auto",
            # Export-resolution preset -> the pixel-height cap render()
            # understands. Same converter studio_app uses, so the two
            # surfaces cannot resolve it differently (RenderKwargsParityTests).
            "max_height": ed.resolution_max_height(r.get("resolution")),
            "motion_blur": bool(r.get("motion_blur", True)),
            "typing_zoom": bool(r.get("typing_zoom", True)),
            "scroll_zoom": bool(r.get("scroll_zoom", True)),
            "window_follow": bool(r.get("window_follow", True)),
            "window_layout": (r.get("window_layout")
                              if r.get("window_layout") in _WINDOW_LAYOUTS
                              else "grid"),
            "window_zoom": bool(r.get("window_zoom", False)),
            "window_focus": bool(r.get("window_focus", False)),
            "screen_focus": bool(r.get("screen_focus", True)),
            "badge_erase": bool(r.get("badge_erase", True)),
            "focus_ranges": (None
                             if ed.focus_plan_is_stale(edits_obj)
                             else list(edits_obj.get("focus") or [])),
            "params": _camera_params(r),
            "facecam": bool(r.get("facecam", True)),
            "facecam_params": _facecam_params(r),
            "fade": max(0.0, float(r.get("fade", 0.0) or 0.0)),
            "make_gif": bool(r.get("gif", False)),
            "music": r.get("music"),
            "click_sound": r.get("click_sound"),
            "key_sound": r.get("key_sound"),
            "sfx_volume": float(r.get("sfx_volume", 1.0)
                                if r.get("sfx_volume") is not None else 1.0),
            "gif_fps": int(r.get("gif_fps", 15) or 15),
            "gif_width": int(r.get("gif_width", 1000) or 1000),
            "manual_zooms": list(edits_obj.get("zooms") or []),
            "suppressed_ranges": list(edits_obj.get("suppressed") or []),
            "speedup": bool(r.get("speedup", False)),
            "speedup_rate": float(r.get("speedup_rate", 6.0) or 6.0),
            "speedup_silence_gate": bool(r.get("speedup_silence_gate", True)),
            "speedup_motion_gate": bool(r.get("speedup_motion_gate", True)),
            # Manual force/off spans -- this server's own add_speedup writes
            # them, and render() honors "force" ones even with speedup off.
            "speedups": list(edits_obj.get("speedups") or []),
            # Ripple-delete ranges (add_cut/set_cuts) -- top-level timeline
            # content like speedups; render() unions + frame-quantizes them.
            "cuts": list(edits_obj.get("cuts") or []),
            "windows": list(edits_obj.get("windows") or []),
            # Manual card placement (docs/architecture.md). AUTHORED only in
            # the web editor -- there is no MCP tool and no CLI flag that
            # writes one -- but CONSUMED by every render sink, so it has to
            # resolve here too: a multi-native/scene take whose cards were
            # dragged in the editor must export the SAME composite from
            # render_video as it does from Export. Web-only authoring is not
            # a licence to drop it on read; that is the "one edits.json, two
            # videos" split this resolver exists to prevent. Inert on a
            # whole-screen take (render() forwards them only on the
            # capture_channels / capture_scenes dispatch branches).
            # isinstance rather than `or []`: a hand-written doc reaching this
            # server has not been through normalize_edits, and `list("ab")` is
            # not what studio_app resolves for the same junk.
            "channel_layouts": (list(edits_obj.get("channel_layouts"))
                                if isinstance(edits_obj.get("channel_layouts"),
                                              list) else []),
            "scene_layouts": (dict(edits_obj.get("scene_layouts"))
                              if isinstance(edits_obj.get("scene_layouts"),
                                            dict) else {}),
            "hidden_channels": (list(edits_obj.get("hidden_channels"))
                                if isinstance(edits_obj.get("hidden_channels"),
                                              list) else []),
            # Top-level like `trim`, not a render option -- see
            # studio_app._render_kwargs_from_edits.
            "crop_rect": ed.normalize_crop(edits_obj.get("crop")),
        }

    # -- tools ----------------------------------------------------------------

    def _tool_list_sessions(self, args):
        out = []
        root = self.recordings_root
        if os.path.isdir(root):
            for name in sorted(os.listdir(root), reverse=True):
                d = os.path.join(root, name)
                if not os.path.isdir(d):
                    continue
                if not os.path.isfile(os.path.join(d, "meta.json")):
                    continue
                try:
                    info = ren.describe_session(d)
                    out.append({
                        "session": info["session"],
                        "duration": info["duration"],
                        "click_count": info["click_count"],
                        "has_edits": ed.has_edits(d),
                        "has_output_mp4": info["has_output_mp4"],
                    })
                except Exception as exc:
                    out.append({"session": name, "error": str(exc)})
        return _text_content({"sessions": out, "server": dict(SERVER_BUILD)})

    def _tool_describe_session(self, args):
        session_dir = self._session_dir(_require_str(args, "session"))
        detail = bool(args.get("detail"))
        info = ren.describe_session(session_dir, include_click_times=detail)
        duration = info["duration"]
        saved = self._load_seeded_edits(session_dir, duration)
        if not detail:
            # `presets` is a byte-for-byte duplicate of `render` plus the
            # saved presets; get_edits still returns it in full. Nothing an
            # agent orienting on a session does with it justifies ~1.4KB in
            # every describe call.
            saved = dict(saved)
            saved.pop("presets", None)
        info["edits"] = saved
        # The two multi-window camera flags, as the one word the tools take.
        # Only on a take that HAS cards: on a whole-screen take the pair is
        # inert, and reporting "frame" there would read as a promise.
        if self._is_multi_window_session(session_dir, saved):
            info["zoom_style"] = _zoom_style_of(saved.get("render") or {})
        info["chapters"] = ed.marker_chapters(saved, duration=duration)
        # What actually HAPPENED, derived from events alone -- no frames
        # decoded, no cache, ~50ms. This is the field to read before
        # deciding where to trim/zoom; the raw *_times arrays it summarizes
        # are behind `detail` precisely because they cannot be read.
        try:
            info["beats"] = ren.session_beats(
                session_dir, info=info, edits_render=saved.get("render"),
                zooms=list(saved.get("zooms") or []),
                max_beats=_opt_int(args, "max_beats") or beats_mod.MAX_BEATS)
        except Exception as exc:   # never fail a describe over a derived view
            info["beats"] = {"beats": [], "truncated": 0,
                             "notes": ["beat sheet unavailable: %s" % (exc,)]}
        # Cache read only -- describe_session must never block on an ASR pass.
        cached = tx.load_transcript(session_dir)
        info["has_transcript"] = bool(cached and cached.get("words"))
        info["server"] = dict(SERVER_BUILD)
        return _text_content(info)

    def _transcript(self, session):
        """(session_dir, doc, duration). Runs ASR on a cache miss.

        Slow ONCE per session and cheap forever after -- and it must stay on
        the tool's own thread rather than being pushed to a background job,
        because a model that gets "transcribing, try later" back has no way to
        wait and will simply guess timestamps instead.
        """
        session_dir = self._session_dir(session)
        duration = self._duration_or_none(session_dir)
        with contextlib.redirect_stdout(sys.stderr):
            doc = tx.transcribe_session(session_dir)
        return session_dir, doc, duration

    def _tool_get_transcript(self, args):
        session = _require_str(args, "session")
        start = _num(args, "start")
        end = _num(args, "end")
        include_words = args.get("include_words", True)
        _, doc, duration = self._transcript(session)
        payload = {
            "session": session,
            "status": doc.get("status"),
            "reason": doc.get("reason", ""),
            "language": doc.get("language", ""),
            "model": doc.get("model", ""),
            "duration": duration,
            "text": tx.transcript_text(doc),
            "segments": doc.get("segments") or [],
            "silences": tx.silence_spans(doc, duration=duration),
        }
        if include_words is not False:
            # Ends are DERIVED (tx.words_with_ends) and computed on the FULL
            # list before slicing -- the repair looks ahead to the next word,
            # so a sliced list would repair against the wrong neighbour.
            # `end` is the field to cut on; `t + dur` is not, and this is the
            # same value the editor uses.
            payload["words"] = tx.slice_words(
                {"words": tx.words_with_ends(doc.get("words"))}, start, end)
        return _text_content(payload)

    def _tool_find_in_transcript(self, args):
        session = _require_str(args, "session")
        query = _require_str(args, "query")
        limit = _opt_int(args, "limit") or 20
        _, doc, _ = self._transcript(session)
        return _text_content({
            "session": session,
            "status": doc.get("status"),
            "reason": doc.get("reason", ""),
            "query": query,
            "matches": tx.find_phrase(doc, query, limit=max(1, limit)),
        })

    def _tool_get_edits(self, args):
        _, _, current = self._edit_session(_require_str(args, "session"))
        return _text_content(current)

    def _tool_set_render_options(self, args):
        session = _require_str(args, "session")
        patch_render = {}
        for key in ("zoom", "offset", "cursor_size", "fade", "speedup_rate",
                    "facecam_size", "facecam_blur", "sfx_volume"):
            if key in args and args[key] is not None:
                patch_render[key] = _num(args, key)
        for key in ("gif_fps", "gif_width"):
            if key in args and args[key] is not None:
                patch_render[key] = _opt_int(args, key)
        for key in ("click_fx", "spotlight", "cursor_fx", "cursor_erase",
                    "always_zoomed",
                    "motion_blur", "overview",
                    "typing_zoom", "drag_hold", "scroll_zoom",
                    "window_follow", "window_zoom", "window_focus",
                    "screen_focus", "badge_erase",
                    "facecam", "speedup", "speedup_silence_gate",
                    "speedup_motion_gate", "gif"):
            if key in args and args[key] is not None:
                patch_render[key] = _opt_bool(args, key)
        for key in ("style", "background", "aspect", "click_color", "music",
                    "click_sound", "key_sound", "zoom_speed",
                    "facecam_position",
                    "facecam_shape", "window_layout"):
            if key in args:  # explicit null clears the option
                patch_render[key] = _opt_str(args, key)
        # zoom_style is SHORTHAND, never a stored field: it expands to the
        # window_focus/window_zoom pair. An explicit boolean in the same call
        # wins, so `zoom_style='both', window_zoom=false` means what it says
        # rather than silently disagreeing with itself.
        if args.get("zoom_style") is not None:
            style = _opt_str(args, "zoom_style")
            if style not in _ZOOM_STYLES:
                raise ToolError("zoom_style must be one of: {}".format(
                    ", ".join("'{}'".format(v) for v in _ZOOM_STYLES)))
            explicit = dict((k, patch_render[k])
                            for k in ("window_focus", "window_zoom")
                            if k in patch_render)
            patch_render.update(_ZOOM_STYLES[style])
            patch_render.update(explicit)
        if patch_render.get("style") is not None and \
                patch_render["style"] not in ("clean", "framed"):
            raise ToolError("style must be 'clean' or 'framed'")
        if patch_render.get("zoom_speed") is not None and \
                patch_render["zoom_speed"] not in ("slow", "normal", "fast"):
            raise ToolError("zoom_speed must be 'slow', 'normal', or 'fast'")
        if patch_render.get("facecam_position") is not None and \
                patch_render["facecam_position"] not in (
                    "bottom-left", "bottom-right", "top-left", "top-right"):
            raise ToolError("facecam_position must be 'bottom-left', "
                            "'bottom-right', 'top-left', or 'top-right'")
        if patch_render.get("facecam_shape") is not None and \
                patch_render["facecam_shape"] not in ("circle", "rounded"):
            raise ToolError("facecam_shape must be 'circle' or 'rounded'")
        if patch_render.get("window_layout") is not None and \
                patch_render["window_layout"] not in _WINDOW_LAYOUTS:
            raise ToolError("window_layout must be one of: {}".format(
                ", ".join("'{}'".format(v) for v in _WINDOW_LAYOUTS)))
        if not patch_render:
            raise ToolError("no render options provided")
        session_dir, duration, current = self._edit_session(session)
        saved = self._save_patch(session_dir, current, {"render": patch_render},
                                 duration)
        return _text_content({"render": saved["render"],
                              "zoom_style": _zoom_style_of(saved["render"])})

    def _tool_set_trim(self, args):
        session = _require_str(args, "session")
        patch_trim = {}
        if "start" in args and args["start"] is not None:
            patch_trim["start"] = _num(args, "start")
        if "end" in args:
            patch_trim["end"] = None if args["end"] is None else _num(args, "end")
        if not patch_trim:
            raise ToolError("provide start and/or end")
        session_dir, duration, current = self._edit_session(session)
        # Trim is the OTHER input to the cut budget: narrowing the window
        # shrinks what survives exactly as adding a cut does. add_cut and
        # set_cuts both validate; without this, `set_trim` is the way to
        # write the very document they refuse. Same delta rule the web sink
        # uses -- only block a trim that BREAKS a currently-valid document.
        prospective = dict(current.get("trim") or {})
        prospective.update(patch_trim)
        cuts_now = current.get("cuts") or []
        fps = self._session_fps(session_dir)
        try:
            ed.plan_cuts(cuts_now, prospective, duration, fps)
        except ed.CutsError as exc:
            try:
                ed.plan_cuts(cuts_now, current.get("trim"), duration, fps)
            except ed.CutsError:
                pass          # already invalid — this trim is not the cause
            else:
                raise ToolError(str(exc))
        saved = self._save_patch(session_dir, current, {"trim": patch_trim},
                                 duration)
        return _text_content({"trim": saved["trim"]})

    def _tool_add_zoom(self, args):
        session = _require_str(args, "session")
        start = _num(args, "start", required=True)
        end = _num(args, "end", required=True)
        level = _num(args, "level")
        x = _num(args, "x")
        y = _num(args, "y")
        if (x is None) != (y is None):
            raise ToolError("x and y must be provided together (or neither)")
        session_dir = self._session_dir(session)
        # A manual zoom TARGETING a coordinate has no meaning on a multi-native
        # take (which card's buffer is (x, y) in?). A whole-frame zoom with no
        # coords is equally moot -- there is no whole-screen camera in cards
        # mode. Refuse either way, pointing at the automatic per-card camera.
        self._refuse_manual_spatial_edit(session_dir, "add_zoom")
        entry = {"id": _new_id("zoom"), "start": start, "end": end}
        if level is not None:
            entry["level"] = level
        if x is not None:
            entry["x"] = x
            entry["y"] = y
        session_dir, duration, current = self._edit_session(session)
        zooms = list(current.get("zooms") or []) + [entry]
        saved = self._save_patch(session_dir, current, {"zooms": zooms}, duration)
        return _text_content({"zooms": saved["zooms"]})

    def _tool_remove_zoom(self, args):
        return self._remove_by_id(_require_str(args, "session"), "zooms",
                                  _require_str(args, "zoom_id"), "zoom")

    # -- adjust_zoom: retune ONE camera move, addressed by time -------------
    #
    # "that zoom at 1:23 was too aggressive" is the sentence this exists for.
    # Everything else about it follows from that: the time is the address (not
    # an id the user cannot see), one word says which way to move it, and the
    # tool -- not the caller -- works out which of the two cameras this take
    # actually has and what "softer" means for it. The alternative an agent
    # reached for before this was remove_zoom + add_zoom, which loses the
    # arc's pin and its auto marker, or set_render_options(zoom=...), which
    # quietly retunes the WHOLE take to fix one moment.

    _ADJUST_STEP = 1.3       # one rung of softer/stronger on a screen zoom
    _ADJUST_FLOOR = 1.2      # a screen zoom gentler than this is removed
    _ADJUST_NEAR = 3.0       # a move may sit this far from `at` and still match

    @staticmethod
    def _pick_move(ranges, at, near):
        """`(index, how)` of the range covering `at`, else the nearest one
        within `near` seconds, else `(None, None)`.

        Nearest-within-a-window rather than covering-only because the user is
        reading a time off a player, sometimes off an export whose clock the
        trim/cuts have shifted, and a zoom arc runs several seconds -- being
        strict here would answer "no zoom at 1:23" about the very move they
        just watched.
        """
        if not ranges:
            return None, None
        covering = [i for i, r in enumerate(ranges)
                    if float(r["start"]) <= at <= float(r["end"])]
        if covering:
            # Ranges can overlap only when hand-authored; take the one whose
            # centre is nearest so the answer is deterministic either way.
            i = min(covering,
                    key=lambda j: abs((float(ranges[j]["start"])
                                       + float(ranges[j]["end"])) / 2.0 - at))
            return i, "covering"
        best, best_d = None, None
        for i, r in enumerate(ranges):
            d = max(float(r["start"]) - at, at - float(r["end"]), 0.0)
            if best_d is None or d < best_d:
                best, best_d = i, d
        if best is not None and best_d <= near:
            return best, "nearest"
        return None, None

    @staticmethod
    def _range_digest(ranges, limit=6):
        """`['12.5-19.0', ...]` -- what to show when nothing matched."""
        return ["{:.1f}-{:.1f}".format(float(r["start"]), float(r["end"]))
                for r in ranges[:limit]]

    @staticmethod
    def _export_clock_note(doc):
        """A warning when the EXPORT's clock is not this tool's clock, or None.

        A trim, a cut or a speed-up all mean the time the user read off the
        finished video is not the time in `at`. Saying so is the difference
        between "I softened the wrong arc" being visible and being a mystery.
        """
        trim = doc.get("trim") or {}
        head = float(trim.get("start") or 0.0)
        cuts = len(doc.get("cuts") or [])
        sped = bool((doc.get("render") or {}).get("speedup"))
        why = []
        if head > 0.01:
            why.append("{:.2f}s is trimmed off the head".format(head))
        if cuts:
            why.append("{} cut range{} are rippled out".format(
                cuts, "" if cuts == 1 else "s"))
        if sped:
            why.append("idle stretches are sped up")
        if not why:
            return None
        note = ("`at` is SOURCE-media time and this export is retimed ({}), "
                "so a timestamp read off the exported video is not this clock"
                .format(", ".join(why)))
        if head > 0.01 and not cuts and not sped:
            note += " -- here source = exported + {:.2f}s".format(head)
        return note + "."

    def _is_scene_take(self, session_dir):
        """True for a SCENE take (`capture_scenes`: the window set changes
        mid-recording). Read straight off meta.json like `_is_multi_native`."""
        try:
            with open(os.path.join(session_dir, "meta.json")) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            return False
        return seg.is_scene_meta(meta)

    def _is_multi_window_session(self, session_dir, doc):
        """Does this take's camera live in CARDS rather than in one frame?"""
        if list(doc.get("windows") or []):
            return True
        return self._is_multi_native(session_dir)

    def _tool_adjust_zoom(self, args):
        session = _require_str(args, "session")
        at = _time_arg(args, "at")
        change = (_opt_str(args, "change") or "softer").strip().lower()
        if change not in ("softer", "stronger", "off"):
            raise ToolError("change must be 'softer', 'stronger' or 'off'")
        level = _num(args, "level")
        if level is not None and level < 1.0:
            raise ToolError("level must be >= 1")
        session_dir, duration, current = self._edit_session(session)
        if duration and at > float(duration) + self._ADJUST_NEAR:
            raise ToolError(
                "at={} is past the end of this recording ({:.1f}s)".format(
                    _fmt_clock(at), float(duration)))
        if self._is_multi_window_session(session_dir, current):
            if level is not None:
                raise ToolError(
                    "level is a whole-screen zoom factor, and this take's "
                    "camera is the multi-window composition one -- it has "
                    "rungs, not a factor. Use change='softer' / 'stronger' / "
                    "'off'.")
            return self._adjust_focus_move(session, session_dir, duration,
                                           current, at, change)
        return self._adjust_screen_zoom(session, session_dir, duration,
                                        current, at, change, level)

    def _adjust_screen_zoom(self, session, session_dir, duration, current,
                            at, change, level):
        """The whole-screen take: retune the `edits.zooms` range at `at`."""
        doc, notes, materialized = current, [], 0
        if not doc.get("auto_zooms_initialized"):
            # Materialize the auto proposals FIRST. Every surface but a bare
            # CLI render plans the camera from `edits.zooms` alone, so writing
            # one adjusted arc into an empty array would leave that arc as the
            # only zoom in the take -- silently deleting every other move the
            # user just watched. See beats.py's `zoom` vs `zoom_proposal`.
            try:
                info = ren.describe_session(session_dir,
                                            include_click_times=True)
                clicks, dur = info.get("click_times") or [], info["duration"]
            except Exception:
                clicks, dur = [], duration
            before = len(doc.get("zooms") or [])
            doc, _ = ed.initialize_auto_zooms(doc, clicks, dur)
            doc = ed.normalize_edits(doc, duration=duration)   # ids + order
            materialized = max(0, len(doc.get("zooms") or []) - before)
            if materialized:
                notes.append(
                    "this session's {} auto-zoom proposals were materialized "
                    "into edits.zooms first, so the rest of the take still "
                    "zooms".format(materialized))
        zooms = [dict(z) for z in (doc.get("zooms") or [])]
        if not zooms:
            raise ToolError(
                "nothing zooms in this take, so there is no move at {} to "
                "adjust -- add_zoom(start, end, level) creates one.".format(
                    _fmt_clock(at)))
        idx, how = self._pick_move(zooms, at, self._ADJUST_NEAR)
        if idx is None:
            raise ToolError(
                "no zoom covers {} (nor within {:.0f}s of it). The zoom "
                "ranges in this take are {} -- pass a time inside one, or "
                "add_zoom to create a move there.".format(
                    _fmt_clock(at), self._ADJUST_NEAR,
                    ", ".join(self._range_digest(zooms))))
        target = zooms[idx]
        before = {"id": target.get("id"), "start": float(target["start"]),
                  "end": float(target["end"]),
                  "level": round(float(target.get("level", 2.0)), 3)}
        if change == "off":
            zooms.pop(idx)
            action = "removed"
        elif level is not None:
            target["level"] = float(level)
            action = "set"
        elif change == "softer":
            softened = before["level"] / self._ADJUST_STEP
            if softened < self._ADJUST_FLOOR:
                # Below this the "zoom" is a 15% nudge nobody reads as a zoom;
                # holding still is the honest version of gentler. Recoverable:
                # the response carries the range to add_zoom back.
                zooms.pop(idx)
                action = "removed"
                notes.append(
                    "it was already at the gentlest useful level ({:.2f}), so "
                    "the camera now holds steady there -- add_zoom(start={:.2f}"
                    ", end={:.2f}, level=...) puts a move back".format(
                        before["level"], before["start"], before["end"]))
            else:
                target["level"] = round(softened, 3)
                action = "softened"
        else:
            target["level"] = round(before["level"] * self._ADJUST_STEP, 3)
            action = "strengthened"
        saved = self._save_patch(session_dir, doc, {"zooms": zooms}, duration)
        after = next((z for z in saved["zooms"]
                      if z.get("id") == before["id"]), None)
        clock = self._export_clock_note(saved)
        if clock:
            notes.append(clock)
        return _text_content({
            "session": session,
            "at": round(at, 3),
            "target": "zoom",
            "matched": how,
            "action": action,
            "before": before,
            "after": after,
            "zooms": saved["zooms"],
            "notes": notes,
        })

    # The composition camera's two rungs, gentlest first: the card grows in
    # place, then the whole frame pushes in on it (docs/architecture.md, "Window focus").
    _FOCUS_RUNGS = ("focus", "full")

    @staticmethod
    def _focus_rung(level):
        """0 = the card grows in place, 1 = the whole frame pushes in.

        `level` is normally the symbol "focus"/"full" (the only thing the
        planner emits). A numeric level is a 0..1 emphasis AMOUNT -- both
        `edits._normalize_focus_range` and `camera._normalize_focus_manual`
        clamp it to that range -- so only a full 1.0 amount is the
        frame-push-in rung; any smaller amount is a (possibly quieter) grow
        and classifies as the grow rung. (Before the [0, 1] clamp landed a
        number was floored at 1.0 and always read as "full"; classifying a
        0.25 grow as the top rung would resurrect that inversion.)
        """
        if level == "full":
            return 1
        try:
            return 1 if float(level) >= 1.0 else 0
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _focus_move(spans, idx, gap=0.05):
        """Indices of the one MOVE that `spans[idx]` belongs to.

        The planner splits one burst of attention into two abutting spans --
        the card growing, then the frame pushing in
        (`camera._split_at_second_click`). Someone pointing at a time inside
        the grow half and calling the move too aggressive means the push-in,
        so the unit this tool changes is the RUN of abutting same-card spans,
        not the one span the timestamp happened to land in.

        BOTH endpoints are card-checked, and that is load-bearing rather than
        belt-and-braces: spans belonging to DIFFERENT cards that touch exactly
        are the planner's ordinary output (a card hands the composition over
        at the instant the next one takes it), and `render._first_wins_spans`
        manufactures that shape by construction. Checking one endpoint let the
        forward walk swallow the neighbour's span, so softening card 0's move
        silently re-levelled -- or, under `change='off'`, DELETED -- a move
        belonging to card 1.
        """
        card = spans[idx].get("card")

        def abuts(a, b):
            return (spans[a].get("card") == card
                    and spans[b].get("card") == card
                    and abs(float(spans[a]["end"])
                            - float(spans[b]["start"])) <= gap)

        lo = idx
        while lo > 0 and abuts(lo - 1, lo):
            lo -= 1
        hi = idx
        while hi + 1 < len(spans) and abuts(hi, hi + 1):
            hi += 1
        return list(range(lo, hi + 1))

    def _focus_plan_for(self, session_dir, doc):
        """This take's window-focus plan as editable spans (or [])."""
        kw = self._render_kwargs(doc)
        try:
            if self._is_multi_native(session_dir):
                return ren.multi_native_focus_ranges(
                    session_dir, offset=kw["offset"], max_zoom=kw["max_zoom"],
                    params=kw["params"],
                    suppressed_ranges=kw["suppressed_ranges"])
            return ren.multi_window_focus_ranges(
                session_dir, list(doc.get("windows") or []),
                offset=kw["offset"], max_zoom=kw["max_zoom"],
                params=kw["params"],
                suppressed_ranges=kw["suppressed_ranges"],
                window_follow=kw["window_follow"],
                crop_rect=kw["crop_rect"])
        except Exception:
            return []

    def _adjust_focus_move(self, session, session_dir, duration, current, at,
                           change):
        """The multi-window take: retune the composition camera's move at `at`."""
        doc, notes, materialized = current, [], 0
        render = doc.get("render") or {}
        if not render.get("window_focus"):
            style = _zoom_style_of(render)
            if render.get("window_zoom"):
                raise ToolError(
                    "this take's camera is zoom_style '{}' -- the camera "
                    "INSIDE each card, which is planned from that card's own "
                    "clicks and has no per-move surface yet. Lower it for the "
                    "whole take with set_render_options(zoom=<level>), or "
                    "switch to the composition camera, whose moves this tool "
                    "can retune one at a time, with "
                    "set_render_options(zoom_style='frame').".format(style))
            raise ToolError(
                "nothing zooms in this take (zoom_style '{}'), so there is no "
                "move at {} to adjust. set_render_options(zoom_style='frame') "
                "turns the composition camera on -- the worked-in card grows, "
                "then the frame pushes in.".format(style, _fmt_clock(at)))
        if self._is_scene_take(session_dir):
            # A scene take's timeline is N scenes joined end to end, each its
            # own fleet with its own window set, and its composition camera is
            # planned per SCENE at render time. There is no single plan to
            # materialize (the fleet emitter reads a `capture_channels`
            # manifest a scene take by construction does not have), and
            # handing `render._scene_local_ranges` a materialized plan is not
            # a surface that exists yet. Say that, rather than reporting the
            # fleet emitter's empty result as if the take had no moves.
            raise ToolError(
                "this is a scene take (the window set changes mid-recording), "
                "and its composition camera is planned per scene at render "
                "time -- there is no per-move surface to retune at {} yet. "
                "What applies to the whole take: "
                "set_render_options(zoom_style='off') holds every scene's "
                "layout steady, and zoom_style='inside' swaps the "
                "outer-frame camera for the one inside each card."
                .format(_fmt_clock(at)))
        if ed.focus_plan_is_stale(doc):
            # Materialize this take's own plan so ONE of its moves can be
            # owned and changed. The spans are the same ones the render
            # auto-plans (`render.multi_native_focus_ranges` builds the export's
            # cards), so this fixes the plan in place rather than replacing it
            # with a different one -- and an empty plan is never materialized,
            # because an empty `edits.focus` reads as "the user deleted every
            # arc" and would switch the camera off.
            plan = self._focus_plan_for(session_dir, doc)
            if not plan:
                raise ToolError(
                    "this take's composition camera is auto-planned and its "
                    "moves could not be materialized into editable spans, so "
                    "there is nothing to retune at {}. The levers that do "
                    "apply to the whole take: set_render_options(zoom_style="
                    "'off') to hold the layout steady, or a different "
                    "window_layout to change what the move has room to "
                    "do.".format(_fmt_clock(at)))
            doc, _ = ed.initialize_focus_ranges(doc, plan)
            doc = ed.normalize_edits(doc, duration=duration)   # ids + order
            materialized = len(doc.get("focus") or [])
            if materialized:
                notes.append(
                    "the composition camera's {} moves were materialized into "
                    "edits.focus first (same plan, now editable)".format(
                        materialized))
        spans = [dict(f) for f in (doc.get("focus") or [])]
        if not spans:
            raise ToolError(
                "the composition camera makes no move in this take, so there "
                "is nothing to adjust at {}.".format(_fmt_clock(at)))
        idx, how = self._pick_move(spans, at, self._ADJUST_NEAR)
        if idx is None:
            raise ToolError(
                "no composition move covers {} (nor within {:.0f}s of it). "
                "The moves in this take are {}.".format(
                    _fmt_clock(at), self._ADJUST_NEAR,
                    ", ".join(self._range_digest(spans))))
        move = self._focus_move(spans, idx)
        rungs = [self._focus_rung(spans[i].get("level")) for i in move]
        top = max(rungs)
        before = {"card": spans[move[0]].get("card"),
                  "start": float(spans[move[0]]["start"]),
                  "end": float(spans[move[-1]]["end"]),
                  "level": self._FOCUS_RUNGS[top],
                  "ids": [spans[i].get("id") for i in move]}
        after = dict(before)
        if change == "off":
            for i in reversed(move):
                spans.pop(i)
            action, after = "removed", None
            notes.append(
                "the composition holds steady through that stretch now. There "
                "is no tool to add a focus move back -- reset_edits re-plans "
                "the whole take if you need it.")
        elif change == "softer":
            if top == 0:
                # A card growing in place IS the gentlest move the composition
                # camera has. Removing it is a one-way door (nothing adds one
                # back), so say what the next step is instead of taking it.
                action = "unchanged"
                notes.append(
                    "that move is already the gentlest one there is: the card "
                    "grows in place and the frame does not move. "
                    "change='off' removes it entirely.")
            else:
                # The WHOLE move drops a rung, not just the half the timestamp
                # landed in -- "that zoom was too aggressive" is about the move
                # the viewer saw, and a pair left at [grow, push-in] would
                # still push in a second later.
                for i in move:
                    spans[i]["level"] = self._FOCUS_RUNGS[0]
                action, after["level"] = "softened", self._FOCUS_RUNGS[0]
                notes.append(
                    "the frame no longer pushes in there; the card still "
                    "grows in place.")
        else:
            strongest = self._FOCUS_RUNGS[-1]
            if min(rungs) >= len(self._FOCUS_RUNGS) - 1:
                action = "unchanged"
                notes.append(
                    "that move is already the strongest one there is: the "
                    "whole frame pushes in until the card fills it.")
            else:
                # Promoting the whole run means the push-in starts with the
                # move instead of at its second click -- which is what asking
                # for more of it means when part of the run is already there.
                for i in move:
                    spans[i]["level"] = strongest
                action, after["level"] = "strengthened", strongest
        if action == "unchanged" and not materialized:
            saved = doc          # nothing to write
        else:
            saved = self._save_patch(session_dir, doc, {"focus": spans},
                                     duration)
        clock = self._export_clock_note(saved)
        if clock:
            notes.append(clock)
        return _text_content({
            "session": session,
            "at": round(at, 3),
            "target": "focus",
            "matched": how,
            "action": action,
            "before": before,
            "after": after,
            "focus": saved.get("focus") or [],
            "notes": notes,
        })

    def _tool_add_speedup(self, args):
        session = _require_str(args, "session")
        start = _num(args, "start", required=True)
        end = _num(args, "end", required=True)
        mode = _opt_str(args, "mode") or "off"
        rate = _num(args, "rate")
        entry = {"id": _new_id("speedup"), "start": start, "end": end,
                 "mode": mode}
        if rate is not None:
            entry["rate"] = rate
        session_dir, duration, current = self._edit_session(session)
        speedups = list(current.get("speedups") or []) + [entry]
        saved = self._save_patch(session_dir, current, {"speedups": speedups},
                                 duration)
        return _text_content({"speedups": saved["speedups"]})

    def _tool_remove_speedup(self, args):
        return self._remove_by_id(_require_str(args, "session"), "speedups",
                                  _require_str(args, "speedup_id"), "speedup")

    # -- cuts (ripple delete) ---------------------------------------------

    def _session_fps(self, session_dir):
        try:
            with open(os.path.join(session_dir, "meta.json")) as f:
                return float(json.load(f).get("fps") or 60.0)
        except (OSError, ValueError, TypeError):
            return 60.0

    def _cut_summary(self, cuts_list, trim, duration, fps):
        """(merged, removed_sec, output_duration) for a prospective cuts
        list -- see edits.cut_summary. Kept as a method so the tools read
        the same as they did before the rule moved."""
        return ed.cut_summary(cuts_list, trim, duration, fps)

    def _validate_cuts(self, merged, out_dur):
        """edits.validate_cuts, re-raised in this surface's vocabulary."""
        try:
            ed.validate_cuts(merged, out_dur)
        except ed.CutsError as exc:
            raise ToolError(str(exc))

    def _tool_add_cut(self, args):
        session = _require_str(args, "session")
        start = _num(args, "start", required=True)
        end = _num(args, "end", required=True)
        if end <= start:
            raise ToolError("cut end must be after start")
        entry = {"id": _new_id("cut"), "start": start, "end": end}
        session_dir, duration, current = self._edit_session(session)
        fps = self._session_fps(session_dir)
        cuts = list(current.get("cuts") or []) + [entry]
        # Validate and echo from the NORMALIZED list -- what will actually
        # persist and render -- never the raw request (the reviewed echo-vs-
        # saved drift). With cuts' min_span=0 normalize the two only differ
        # on clamps, but the contract is "the echo IS the encoder's input".
        prospective = ed._normalize_cut_list(cuts, duration=duration)
        merged, removed, out_dur = self._cut_summary(
            prospective, current.get("trim"), duration, fps)
        self._validate_cuts(merged, out_dur)
        saved = self._save_patch(session_dir, current, {"cuts": cuts},
                                 duration)
        snapped = retime.quantize_cut_spans(
            [(c["start"], c["end"]) for c in saved["cuts"]
             if c["id"] == entry["id"]], fps, duration=duration)
        payload = {"cuts": saved["cuts"], "removed_sec": removed}
        if snapped:
            payload["snapped"] = {"start": snapped[0][0],
                                  "end": snapped[-1][1]}
        if out_dur is not None:
            payload["output_duration"] = out_dur
        return _text_content(payload)

    def _tool_remove_cut(self, args):
        return self._remove_by_id(_require_str(args, "session"), "cuts",
                                  _require_str(args, "cut_id"), "cut")

    def _tool_set_cuts(self, args):
        session = _require_str(args, "session")
        raw = args.get("cuts")
        if not isinstance(raw, list):
            raise ToolError("cuts must be an array of {start, end} ranges")
        parsed = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ToolError("cuts[{}] must be an object".format(i))
            entry = {}
            for key in ("start", "end"):
                v = item.get(key)
                if (isinstance(v, bool) or not isinstance(v, (int, float))
                        or not math.isfinite(float(v))):
                    raise ToolError(
                        "cuts[{}].{} must be a finite number".format(i, key))
                entry[key] = float(v)
            if entry["end"] <= entry["start"]:
                raise ToolError(
                    "cuts[{}] end must be after start".format(i))
            cid = item.get("id")
            entry["id"] = (str(cid) if cid and not isinstance(cid, bool)
                           else _new_id("cut"))
            parsed.append(entry)
        session_dir, duration, current = self._edit_session(session)
        fps = self._session_fps(session_dir)
        # Same normalized-list discipline as add_cut: validate + echo what
        # will persist and render, not the raw request.
        prospective = ed._normalize_cut_list(parsed, duration=duration)
        merged, removed, out_dur = self._cut_summary(
            prospective, current.get("trim"), duration, fps)
        if parsed:
            self._validate_cuts(merged, out_dur)
        saved = self._save_patch(session_dir, current, {"cuts": parsed},
                                 duration)
        payload = {"cuts": saved["cuts"], "removed_sec": removed,
                   "snapped": [{"start": a, "end": b} for (a, b) in merged]}
        if out_dur is not None:
            payload["output_duration"] = out_dur
        return _text_content(payload)

    def _tool_add_marker(self, args):
        session = _require_str(args, "session")
        t = _num(args, "time", required=True)
        label = _opt_str(args, "label")
        entry = {"id": _new_id("marker"), "time": t, "label": label}
        session_dir, duration, current = self._edit_session(session)
        markers = list(current.get("markers") or []) + [entry]
        saved = self._save_patch(session_dir, current, {"markers": markers},
                                 duration)
        return _text_content({
            "markers": saved["markers"],
            "chapters": ed.marker_chapters(saved, duration=duration),
        })

    def _tool_remove_marker(self, args):
        return self._remove_by_id(_require_str(args, "session"), "markers",
                                  _require_str(args, "marker_id"), "marker")

    def _tool_list_recorded_windows(self, args):
        session = _require_str(args, "session")
        session_dir = self._session_dir(session)
        with contextlib.redirect_stdout(sys.stderr):
            found = ren.recorded_windows(session_dir)
        return _text_content({"windows": found})

    def _tool_set_crop(self, args):
        session = _require_str(args, "session")
        # A crop is a single-source-space rect; a multi-native take has N
        # window buffers and no shared frame to crop. Refuse a non-null crop
        # before parsing, but let a null-clear pass -- clearing a crop a
        # multi-native session never had is a harmless no-op, and blocking it
        # would make a blanket "clear all crops" script fail on these takes.
        if args.get("crop") is not None:
            self._refuse_manual_spatial_edit(self._session_dir(session),
                                             "set_crop")
        # Membership, not truthiness: null is a real argument here (it clears
        # the crop), so an absent key has to be told apart from an explicit
        # null rather than both falling into the same branch.
        if "crop" not in args:
            raise ToolError("crop is required (pass null to clear it)")
        raw = args.get("crop")
        if raw is None:
            parsed = None
        elif not isinstance(raw, dict):
            raise ToolError("crop must be an {x, y, w, h} object, or null")
        else:
            parsed = {}
            for key in ("x", "y", "w", "h"):
                if key not in raw or raw[key] is None:
                    raise ToolError("crop missing '{}'".format(key))
                v = raw[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ToolError("crop.{} must be a number".format(key))
                parsed[key] = float(v)
            # Reject here rather than let normalize_edits quietly return None:
            # a caller that asked for a 4px crop and got silence back has no
            # way to tell that from a crop that worked.
            if ed.normalize_crop(parsed) is None:
                raise ToolError(
                    "crop is degenerate (each side must be at least "
                    "{:.0f}px)".format(ed._MIN_CROP_DIM))
        session_dir, duration, current = self._edit_session(session)
        saved = self._save_patch(session_dir, current, {"crop": parsed},
                                 duration)
        return _text_content({"crop": saved["crop"]})

    def _tool_set_windows(self, args):
        session = _require_str(args, "session")
        # `set_windows` draws crop cards on the SINGLE source frame. A
        # multi-native take's cards ARE its channels -- bound to real windows
        # by the manifest, not arbitrary crop rects -- so arbitrary rects have
        # no meaning here. Refuse a non-empty set; an empty clear is allowed
        # for the same reason set_crop's null-clear is.
        raw = args.get("windows")
        if isinstance(raw, list) and raw:
            self._refuse_manual_spatial_edit(self._session_dir(session),
                                             "set_windows")
        if not isinstance(raw, list):
            raise ToolError("windows must be an array of {x, y, w, h} rects")
        parsed = []
        for i, item in enumerate(raw):
            if not isinstance(item, dict):
                raise ToolError("windows[{}] must be an object".format(i))
            entry = {}
            for key in ("x", "y", "w", "h"):
                if key not in item or item[key] is None:
                    raise ToolError("windows[{}] missing '{}'".format(i, key))
                v = item[key]
                if isinstance(v, bool) or not isinstance(v, (int, float)):
                    raise ToolError(
                        "windows[{}].{} must be a number".format(i, key))
                entry[key] = float(v)
            # Optional: pin this card to a real recorded window rather than
            # letting render guess by overlap. list_recorded_windows is where
            # the ids come from.
            wid = item.get("window_id")
            if wid is not None and not isinstance(wid, bool):
                try:
                    entry["window_id"] = int(wid)
                except (TypeError, ValueError):
                    raise ToolError(
                        "windows[{}].window_id must be an integer".format(i))
            parsed.append(entry)
        session_dir, duration, current = self._edit_session(session)
        saved = self._save_patch(session_dir, current, {"windows": parsed},
                                 duration)
        return _text_content({"windows": saved["windows"]})

    def _tool_add_window(self, args):
        session = _require_str(args, "session")
        x = _num(args, "x", required=True)
        y = _num(args, "y", required=True)
        w = _num(args, "w", required=True)
        h = _num(args, "h", required=True)
        entry = {"id": _new_id("window"), "x": x, "y": y, "w": w, "h": h}
        wid = args.get("window_id")
        if wid is not None and not isinstance(wid, bool):
            try:
                entry["window_id"] = int(wid)
            except (TypeError, ValueError):
                raise ToolError("window_id must be an integer")
        session_dir, duration, current = self._edit_session(session)
        windows = list(current.get("windows") or []) + [entry]
        saved = self._save_patch(session_dir, current, {"windows": windows},
                                 duration)
        return _text_content({"windows": saved["windows"]})

    def _tool_fit_windows(self, args):
        """Grow the composition to the canvas padding, as a placement override.

        Not "fill the canvas": `framing.fit_placements` applies ONE uniform
        scale, which meets the padding on the binding axis and leaves the
        other margined by however the arrangement's bbox aspect differs from
        the canvas. So this pays on a HAND-PLACED arrangement (measured on a
        real dragged layout: 31% -> 67% of the canvas) and finds no SLACK in
        a freshly applied `window_layout`, since those are already scaled by
        this same helper. `grid` is the one that still moves: it aspect-fits
        each crop inside its own cell, so the cards' bounding box can sit
        off-centre and this re-centres it at scale 1.0 (measured, two
        windows: 78px on 1080x1920, 7-8px on 2880x1800). Nothing grows.

        The arrangement it fits is whatever `multi_window_layout` reports --
        the SAME call the editor's camera-path endpoint makes, so the cards
        this fits are the cards the editor is showing, card overrides and all.
        Deliberately not a render option: fitting is an ACTION on the current
        arrangement, and re-deriving it every render would fight the next hand
        placement instead of preserving it.
        """
        session = _require_str(args, "session")
        session_dir, duration, current = self._edit_session(session)
        windows = list(current.get("windows") or [])
        if not windows:
            raise ToolError(
                "session has no windows to fit: fit_windows rescales an "
                "existing multi-window arrangement, so add cards with "
                "set_windows or add_window first")
        kw = self._render_kwargs(current)
        with contextlib.redirect_stdout(sys.stderr):
            layout = ren.multi_window_layout(
                session_dir, kw["windows"], background=kw["background"],
                style=kw["style"], aspect=kw["aspect"],
                window_layout=kw["window_layout"],
                crop_rect=kw["crop_rect"])
        canvas_w, canvas_h = layout["canvas"]
        fitted = framing.fit_placements(canvas_w, canvas_h, layout["cells"])
        # Fractions of the OUTPUT, not source pixels: the canvas resizes with
        # --aspect and the framed style, and a placement authored in px would
        # silently mean something else at the next export size.
        patched = []
        for win, box in zip(windows, fitted):
            entry = dict(win)
            entry["layout"] = {"x": box[0] / float(canvas_w),
                               "y": box[1] / float(canvas_h),
                               "w": box[2] / float(canvas_w),
                               "h": box[3] / float(canvas_h)}
            patched.append(entry)
        saved = self._save_patch(session_dir, current, {"windows": patched},
                                 duration)
        return _text_content({"windows": saved["windows"],
                              "canvas": [int(canvas_w), int(canvas_h)]})

    def _tool_remove_window(self, args):
        return self._remove_by_id(_require_str(args, "session"), "windows",
                                  _require_str(args, "window_id"), "window")

    def _remove_by_id(self, session, key, item_id, label):
        session_dir, duration, current = self._edit_session(session)
        items = current.get(key) or []
        kept = [it for it in items if it.get("id") != item_id]
        if len(kept) == len(items):
            raise ToolError("unknown {} id: {}".format(label, item_id))
        saved = self._save_patch(session_dir, current, {key: kept}, duration)
        payload = {key: saved[key]}
        if key == "markers":
            payload["chapters"] = ed.marker_chapters(saved, duration=duration)
        return _text_content(payload)

    def _tool_reset_edits(self, args):
        session_dir = self._session_dir(_require_str(args, "session"))
        duration = self._duration_or_none(session_dir)
        ed.reset_edits(session_dir, duration=duration)
        # Defaults for THIS session, which for a take recorded with a window
        # pick means the pick is back (cards, `desktop`, the composition
        # camera) -- the same thing the editor shows after a reset and a
        # reload. Re-seeding here rather than on the next call keeps the
        # returned document equal to what the next render will use.
        return _text_content(self._load_seeded_edits(session_dir, duration))

    def _tool_render_video(self, args):
        session = _require_str(args, "session")
        session_dir, duration, saved = self._edit_session(session)
        kw = self._render_kwargs(saved)
        trim = saved.get("trim", {})

        gif_arg = _opt_bool(args, "gif")
        make_gif = kw["make_gif"] if gif_arg is None else gif_arg

        out_path = _session_output_path(
            session_dir, _opt_str(args, "out") or "output.mp4")
        if make_gif:
            _session_output_path(
                session_dir, os.path.splitext(out_path)[0] + ".gif")

        trim_end = trim.get("end")
        final_out = ren.render(
            session_dir,
            out_path=out_path,
            max_zoom=kw["max_zoom"],
            offset=kw["offset"],
            style=kw["style"],
            make_gif=make_gif,
            params=kw["params"],
            background=kw["background"],
            click_fx=kw["click_fx"],
            click_params=kw["click_params"],
            spotlight=kw["spotlight"],
            fade=kw["fade"],
            music=kw["music"],
            click_sound=kw["click_sound"],
            key_sound=kw["key_sound"],
            sfx_volume=kw["sfx_volume"],
            trim_start=float(trim.get("start") or 0.0),
            trim_end=(None if trim_end is None else float(trim_end)),
            manual_zooms=kw["manual_zooms"],
            suppressed_ranges=kw["suppressed_ranges"],
            cursor_fx=kw["cursor_fx"],
            cursor_params=kw["cursor_params"],
            aspect=kw["aspect"],
            motion_blur=kw["motion_blur"],
            typing_zoom=kw["typing_zoom"],
            scroll_zoom=kw["scroll_zoom"],
            speedup=kw["speedup"],
            speedup_rate=kw["speedup_rate"],
            speedup_silence_gate=kw["speedup_silence_gate"],
            speedup_motion_gate=kw["speedup_motion_gate"],
            speedups=kw["speedups"],
            cuts=kw["cuts"],
            facecam=kw["facecam"],
            facecam_params=kw["facecam_params"],
            gif_fps=kw["gif_fps"],
            gif_width=kw["gif_width"],
            windows=kw["windows"],
            window_follow=kw["window_follow"],
            window_layout=kw["window_layout"],
            window_zoom=kw["window_zoom"],
            window_focus=kw["window_focus"],
            screen_focus=kw["screen_focus"],
            badge_erase=kw["badge_erase"],
            focus_ranges=kw["focus_ranges"],
            crop_rect=kw["crop_rect"],
            cursor_erase=kw["cursor_erase"],
            max_height=kw["max_height"],
            channel_layouts=kw["channel_layouts"],
            scene_layouts=kw["scene_layouts"],
            hidden_channels=kw["hidden_channels"],
        )
        return _text_content({"out_path": final_out})

    def _tool_preview_frame(self, args):
        import cv2  # deferred: render already depends on it

        session = _require_str(args, "session")
        t = _num(args, "time", required=True)
        max_width = _opt_int(args, "max_width")
        max_width = 900 if max_width is None else max(16, int(max_width))
        want_source = bool(args.get("source"))

        session_dir = self._session_dir(session)
        info = ren.describe_session(session_dir)
        duration = float(info["duration"])
        # Seeded like every other tool: a preview that showed the hand-drawn
        # defaults while `render_video` exported the pick would be the
        # "one edits.json, two videos" bug wearing a JPEG.
        saved = self._load_seeded_edits(session_dir, duration)
        kw = self._render_kwargs(saved)

        trim = saved.get("trim", {})
        trim_start = float(trim.get("start") or 0.0)
        trim_end = trim.get("end")
        trim_end = duration if trim_end is None else float(trim_end)
        t = max(trim_start, min(t, trim_end))

        if want_source:
            # Deliberately ignores every edit: the point is the coordinate
            # space rects are authored in, which the rendered output is not.
            frame = ren.source_frame(session_dir, t_sec=t, offset=kw["offset"])
            return self._encode_preview(frame, max_width, t, source=True)

        frame = ren.preview_frame(
            session_dir,
            t_sec=t,
            max_zoom=kw["max_zoom"],
            offset=kw["offset"],
            style=kw["style"],
            params=kw["params"],
            background=kw["background"],
            click_fx=kw["click_fx"],
            click_params=kw["click_params"],
            spotlight=kw["spotlight"],
            fade=kw["fade"],
            manual_zooms=kw["manual_zooms"],
            suppressed_ranges=kw["suppressed_ranges"],
            cursor_fx=kw["cursor_fx"],
            cursor_params=kw["cursor_params"],
            aspect=kw["aspect"],
            motion_blur=kw["motion_blur"],
            typing_zoom=kw["typing_zoom"],
            scroll_zoom=kw["scroll_zoom"],
            facecam=kw["facecam"],
            facecam_params=kw["facecam_params"],
            windows=kw["windows"],
            window_follow=kw["window_follow"],
            window_layout=kw["window_layout"],
            window_zoom=kw["window_zoom"],
            window_focus=kw["window_focus"],
            screen_focus=kw["screen_focus"],
            badge_erase=kw["badge_erase"],
            focus_ranges=kw["focus_ranges"],
            crop_rect=kw["crop_rect"],
            cursor_erase=kw["cursor_erase"],
            max_height=kw["max_height"],
        )
        return self._encode_preview(
            frame, max_width, t,
            source_dims=(info["width"], info["height"]))

    @staticmethod
    def _encode_preview(frame, max_width, t, source=False, source_dims=None):
        """Downscale + JPEG-encode a preview frame into an MCP content list.

        `source_scale` is what makes a downscaled SOURCE frame usable for
        measurement: multiply any coordinate read off the returned image by it
        to get source pixels. Without it a caller who passed max_width would
        author rects in the wrong units -- the same class of silent error the
        `source` flag exists to prevent.

        Which is exactly why only the SOURCE path may report it. On the
        rendered path these dimensions are the output canvas -- the camera has
        applied a time-varying zoom and pan, and aspect framing has re-boxed
        the result -- so no single scalar maps the image back to source
        coordinates, and a `source_scale` there is a wrong answer that looks
        like a right one. That path reports `frame_*` instead and says where
        to go for measurable pixels. (Measured before the split: a 9:16 framed
        preview of a 2880-wide source reported source_width 1708.)
        """
        import cv2  # deferred: render already depends on it

        full_h, full_w = frame.shape[:2]
        if full_w > max_width:
            scale = max_width / float(full_w)
            frame = cv2.resize(
                frame, (int(max_width), max(1, int(round(full_h * scale)))),
                interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        jpeg = ren.encode_preview_jpeg(frame)
        meta = {"time": float(t), "width": int(w), "height": int(h)}
        if source:
            meta["source_width"] = int(full_w)
            meta["source_height"] = int(full_h)
            meta["source_scale"] = (float(full_w) / float(w)) if w else 1.0
        else:
            meta["is_source"] = False
            meta["frame_width"] = int(full_w)
            meta["frame_height"] = int(full_h)
            meta["frame_scale"] = (float(full_w) / float(w)) if w else 1.0
            if source_dims:
                meta["source_width"] = int(source_dims[0])
                meta["source_height"] = int(source_dims[1])
            meta["note"] = (
                "Rendered frame: these are OUTPUT-CANVAS pixels after the "
                "camera's zoom/pan and aspect framing, so no scale converts "
                "them back to source coordinates. To measure a rect or a zoom "
                "pin, call preview_frame again with source=true."
            )
        return [
            {
                "type": "image",
                "data": base64.b64encode(jpeg).decode("ascii"),
                "mimeType": "image/jpeg",
            },
            {"type": "text", "text": _dumps(meta)},
        ]


def run(recordings_root=None):
    """Start the stdio MCP server loop. Returns a process exit code."""
    server = McpServer(recordings_root=recordings_root)
    return server.serve()


if __name__ == "__main__":
    sys.exit(run())
