"""Local web Studio app for recording, previewing, and rendering sessions."""

import base64
import hmac
import ipaddress
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

# Quiet libavformat under cv2 BEFORE the module loads its ffmpeg backend.
# cv2's VideoCapture prints decoder errors (`moov atom not found` and friends)
# straight to C-level stderr, which no Python try/except can catch -- and the
# library thumbnails every session on every refresh, so one broken or
# still-recording take floods the console indefinitely. -8 is AV_LOG_QUIET.
# setdefault so an operator debugging a decode can still override it.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np

from . import arrange
from . import bar_native
from . import camera_preview
from . import devices as dev
from . import edits
from . import framing
from . import paths
from . import permissions as perms
from . import record as rec
from . import sck
from . import settings
from . import transcribe
from . import render as ren

DEFAULT_REC_ROOT = paths.recordings_root()

# The Studio API can start a screen/mic/camera capture and read or delete local
# recordings, so treating "loopback" as authentication is not enough.  A web
# page can submit simple cross-origin requests to localhost, while DNS
# rebinding can turn a same-origin page into a readable localhost client.  A
# fresh unguessable token is therefore minted for each server process and is
# required on every non-bootstrap API request.  Fetch clients send the header;
# browser-managed media elements put it in their local resource URL because
# they cannot set headers.  Cookies are deliberately not authentication here:
# they are scoped to a hostname, not a port, so an unrelated localhost server
# could otherwise make authenticated cross-port image requests.
_TOKEN_HEADER = "X-AutoCine-Token"
_TOKEN_META = "autocine-token"
_TOKEN_QUERY = "autocine_token"
_TOKEN_EXEMPT_PATHS = frozenset(("/api/health", "/api/rev"))
_TOKEN_QUERY_PATHS = (
    "/api/camera/preview", "/api/media/", "/api/thumb/",
)
_MAX_JSON_BODY = 8 * 1024 * 1024

_ASSET_LINK_RE = re.compile(
    br'((?:href|src)=")(/?[\w./-]+\.(?:css|js))((?:\?[^"]*)?)(")')


def _version_asset_links(html, static_dir):
    """Stamp `?v=<mtime>` onto local .css/.js links in a served HTML page.

    `Cache-Control: no-store` stops a cache from being poisoned, but it can't
    clean one that already is: a browser holding a heuristically-cached
    stylesheet never asks the server again, so an edit on disk stays invisible
    however many times the app is relaunched. Changing the URL is the only
    thing that reliably forces a refetch. Absolute/remote URLs are left alone.
    """
    def stamp(m):
        pre, url, query, post = m.groups()
        try:
            rel = url.decode("utf-8").lstrip("/")
            mtime = int(os.path.getmtime(os.path.join(static_dir, rel)))
        except Exception:
            return m.group(0)          # unknown file: leave it exactly as-is
        sep = b"&" if query else b"?"
        return (pre + url + query + sep + b"v=" +
                str(mtime).encode("ascii") + post)

    try:
        return _ASSET_LINK_RE.sub(stamp, html)
    except Exception:
        return html                     # never break page serving over this


def _inject_security_token(html, token):
    """Put the per-launch API token in an app HTML shell.

    It lives in a meta element rather than a URL, so it cannot leak through
    browser history or Referer headers.  The page is same-origin and protected
    from framing; another site may request the shell but the browser's origin
    policy prevents that site from reading the token.
    """
    token_json = json.dumps(str(token)).encode("ascii")
    tag = (b'<meta name="' + _TOKEN_META.encode("ascii") +
           b'" content=' + token_json + b'>')
    match = re.search(br"<head(?:\s[^>]*)?>", html, flags=re.IGNORECASE)
    if match:
        return html[:match.end()] + b"\n  " + tag + html[match.end():]
    return tag + b"\n" + html


def shell_url(url_base, page, **params):
    """URL for one of the app's own pages, stamped with the page's mtime.

    The mtime query is what lets a *new* build escape an already-cached HTML
    entry: `Cache-Control: no-store` only governs responses the server
    actually gets asked for, and a browser holding a heuristically-cached
    page never asks. Once the fresh copy lands, no-store keeps it honest and
    `_version_asset_links` handles the css/js underneath.
    """
    static_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "studio_web"))
    try:
        params["v"] = int(os.path.getmtime(os.path.join(static_dir, page)))
    except Exception:
        pass
    query = "&".join("{}={}".format(k, params[k]) for k in sorted(params))
    return "{}/{}{}{}".format(url_base.rstrip("/"), page,
                              "?" if query else "", query)


PROJECT_FILENAME = "project.json"
_MAX_PROJECT_NAME_LEN = 120
_THUMB_FILENAME = "thumb.jpg"
_THUMB_WIDTH = 640
_THUMB_JPEG_QUALITY = 82
# Send-side bound on one MJPEG preview connection (see _stream_camera). This
# is the ONLY thing that can free the webcam from a client that holds its
# socket open but stops reading — the refcount that keeps the device alive is
# dropped nowhere else. Generous on purpose: a healthy client drains a ~30 KB
# JPEG over loopback in microseconds, so this can only fire on a wedged one.
CAMERA_STREAM_TIMEOUT_SEC = 15.0
_WAVEFORM_FILENAME = "waveform.json"
_WAVEFORM_BUCKETS = 600
_DESCRIBE_CACHE_MAX = 64


class StudioError(RuntimeError):
    """HTTP-friendly app error."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = int(status)


class EditsConflict(Exception):
    """Raised when a save's base_rev doesn't match the document on disk.

    Carries the current (winning) document so the handler can return it with
    the 409 and the client can adopt it instead of clobbering.
    """

    def __init__(self, current):
        super().__init__("edits changed outside this editor")
        self.current = current


class CutsRejected(Exception):
    """Raised when a save's `cuts` would produce an export the renderer
    refuses (see edits.plan_cuts).

    Carries the current (unchanged, still-winning) document for the SAME
    reason EditsConflict does: the editor autosaves a WHOLE document, so a
    refusal it cannot recover from would wedge every later edit behind a
    save that never succeeds. The client adopts this and says what happened,
    exactly as it does on a 409.
    """

    def __init__(self, message, current):
        super().__init__(message)
        self.current = current


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


def _as_int(value, default):
    try:
        if value is None:
            return int(default)
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
    text = str(value).strip()
    return text or None


def _camera_params_from_render_opts(render_opts):
    """Camera DEFAULTS overrides derived from render options. Returns None
    when nothing deviates from the camera's own defaults."""
    cam = {}
    if _as_bool(render_opts.get("always_zoomed"), False):
        cam["always_zoomed"] = True
    if not _as_bool(render_opts.get("overview"), True):
        cam["overview"] = False
    if not _as_bool(render_opts.get("drag_hold"), True):
        cam["drag_hold"] = False
    speed = _clean_opt_str(render_opts.get("zoom_speed"))
    if speed and speed != "normal":
        cam["zoom_speed"] = speed
    anim = _clean_opt_str(render_opts.get("screen_anim"))
    if anim and anim != "focused":
        cam["screen_anim"] = anim
    return cam or None


def _facecam_params_from_render_opts(render_opts):
    """FacecamOverlay params from render options (position / size / shape).
    None when everything matches the overlay's own defaults."""
    fc = {}
    pos = _clean_opt_str(render_opts.get("facecam_position"))
    if pos and pos != "bottom-left":
        fc["position"] = pos
    size = _as_float(render_opts.get("facecam_size"), 0.0)
    if size and abs(size - 0.20) > 1e-6:
        fc["size_frac"] = size
    shape = _clean_opt_str(render_opts.get("facecam_shape"))
    if shape and shape != "circle":
        fc["shape"] = shape
    border = _as_float(render_opts.get("facecam_border"), 0.0)
    if border and border > 0:
        fc["border_frac"] = border
    blur = _as_float(render_opts.get("facecam_blur"), 0.0)
    if blur and blur > 0:
        fc["blur"] = blur
    return fc or None


def _patch_from_options(options):
    options = options if isinstance(options, dict) else {}
    patch = {"trim": {}, "render": {}}
    if "trim_start" in options:
        patch["trim"]["start"] = options.get("trim_start")
    if "trim_end" in options:
        patch["trim"]["end"] = options.get("trim_end")
    for key in (
        "zoom", "zoom_speed", "screen_anim", "offset", "style", "background",
        "click_fx", "click_color", "spotlight", "cursor_fx", "cursor_size",
        "cursor_erase",
        "aspect", "resolution", "always_zoomed", "motion_blur", "overview",
        "typing_zoom", "drag_hold",
        "scroll_zoom", "window_follow", "window_layout", "window_zoom",
        "window_focus", "screen_focus", "badge_erase",
        "facecam_border",
        "facecam", "facecam_position", "facecam_size",
        "facecam_shape", "facecam_blur",
        "speedup", "speedup_rate", "speedup_silence_gate",
        "speedup_motion_gate",
        "fade", "music", "click_sound", "key_sound", "sfx_volume",
        "gif", "gif_fps", "gif_width",
    ):
        if key in options:
            patch["render"][key] = options.get(key)
    if not patch["trim"]:
        patch["trim"] = None
    if not patch["render"]:
        patch["render"] = None
    # zooms/suppressed/markers are timeline content, not render "look" -- they ride
    # alongside trim/render in the patch so live (unsaved) edits preview
    # correctly, same as the saved-edits flow in save_session_edits().
    if "zooms" in options:
        patch["zooms"] = options.get("zooms")
    if "suppressed" in options:
        patch["suppressed"] = options.get("suppressed")
    if "markers" in options:
        patch["markers"] = options.get("markers")
    if "speedups" in options:
        patch["speedups"] = options.get("speedups")
    if "cuts" in options:
        patch["cuts"] = options.get("cuts")
    if "windows" in options:
        patch["windows"] = options.get("windows")
    if "channel_layouts" in options:
        patch["channel_layouts"] = options.get("channel_layouts")
    if "scene_layouts" in options:
        patch["scene_layouts"] = options.get("scene_layouts")
    if "hidden_channels" in options:
        patch["hidden_channels"] = options.get("hidden_channels")
    return patch


def waveform_peaks(pcm_bytes, buckets=_WAVEFORM_BUCKETS):
    """Bucketed |peak| envelope (0..1) of mono s16le PCM bytes.

    Pure function (bytes in, list of floats out) so it's unit-testable
    without ffmpeg. Empty/undecodable input -> [].
    """
    buckets = max(1, int(buckets))
    if not pcm_bytes:
        return []
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return []
    samples = np.frombuffer(pcm_bytes[:usable], dtype="<i2")
    n = int(samples.size)
    if n == 0:
        return []
    mags = np.abs(samples.astype(np.float32)) / 32768.0
    buckets = min(buckets, n)
    edges = np.linspace(0, n, buckets + 1).astype(np.int64)
    peaks = []
    for i in range(buckets):
        lo = int(edges[i])
        hi = max(int(edges[i + 1]), lo + 1)
        peaks.append(round(float(mags[lo:hi].max()), 4))
    return peaks


def _rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(round(c)))) for c in rgb]
    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def background_presets():
    """framing's gradient presets as swatch-friendly hex color stops.

    framing stores each preset as (bottom_rgb, top_rgb) and renders the
    gradient top->bottom, so colors are ordered [top, bottom] -- the order
    the gradient actually goes.
    """
    presets = []
    for name, (bottom_rgb, top_rgb) in framing._PRESETS.items():
        presets.append({
            "id": name,
            "colors": [_rgb_to_hex(top_rgb), _rgb_to_hex(bottom_rgb)],
        })
    return presets


def _normalize_project_name(name):
    if name is None:
        return None
    text = str(name).strip()
    if not text:
        return None
    return text[:_MAX_PROJECT_NAME_LEN]


def load_project_name(session_dir):
    """Display name from the session's project.json sidecar, or None."""
    path = os.path.join(session_dir, PROJECT_FILENAME)
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return _normalize_project_name(data.get("name"))


def save_project_name(session_dir, name):
    """Persist (or clear) the session display name; returns the saved value.

    The name lives in a project.json sidecar (NOT edits.json/meta.json --
    edits normalization strips unknown keys). Clearing removes the file.
    """
    clean = _normalize_project_name(name)
    path = os.path.join(session_dir, PROJECT_FILENAME)
    if clean is None:
        try:
            os.remove(path)
        except OSError:
            pass
        return None
    with open(path, "w") as f:
        json.dump({"name": clean}, f, indent=2)
    return clean


def _thumbnail_jpeg(raw_path, width=_THUMB_WIDTH, quality=_THUMB_JPEG_QUALITY,
                    badge_spec=None):
    """Decode one representative frame of `raw_path` into JPEG bytes.

    `badge_spec` is the capture_window / channel block this file came from,
    when there is one: an occlusion-free take's frame carries macOS's capture
    indicator, and a library card is a picture of the take -- it should show
    what the export shows, not the artefact the export removes."""
    if not os.path.isfile(raw_path):
        raise StudioError("recording not found: {}".format(raw_path), status=404)
    # A take whose writer never finalized -- or one still recording -- has
    # frames but no `moov` index. Handing it to cv2 gets `moov atom not found`
    # on C-level stderr (uncatchable) for every library refresh. Detect it
    # first and fail cleanly, so a broken/in-progress take is skipped in
    # silence instead of flooding the console. (OPENCV_FFMPEG_LOGLEVEL quiets
    # what does reach the decoder elsewhere; this keeps it away entirely here.)
    if not sck.has_moov(raw_path):
        raise StudioError(
            "recording is not finalized (no moov atom): {}".format(raw_path),
            status=422)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise StudioError("cannot open recording: {}".format(raw_path), status=404)
    ok, frame = False, None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / float(fps) if frame_count > 0 and fps > 0 else 0.0
        t = min(max(0.5, 0.1 * duration), duration * 0.5) if duration > 0 else 0.0
        idx = int(t * fps) if fps > 0 else 0
        if frame_count > 0:
            idx = max(0, min(idx, frame_count - 1))
        if idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok and idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise StudioError("cannot decode a thumbnail frame from {}".format(raw_path),
                          status=404)
    if badge_spec is not None:
        # Before the downscale: the box is in buffer pixels.
        ren._badge_erase_once(frame, raw_path, badge_spec, frame.shape[1])
    h, w = frame.shape[:2]
    if w != width and w > 0:
        new_h = max(2, int(round(h * (width / float(w)))))
        frame = cv2.resize(frame, (int(width), new_h),
                           interpolation=cv2.INTER_AREA)
    ok, enc = cv2.imencode(".jpg", frame,
                           [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise StudioError("failed to encode thumbnail", status=500)
    return enc.tobytes()


def open_app_window(url, width, height, x=None, y=None):
    """Open `url` as a chromeless Chrome app window; fall back to a normal
    browser tab. Never raises."""
    try:
        if os.path.isdir("/Applications/Google Chrome.app"):
            args = ["open", "-na", "Google Chrome", "--args",
                    "--app={}".format(url),
                    "--window-size={},{}".format(int(width), int(height))]
            if x is not None and y is not None:
                args.append("--window-position={},{}".format(int(x), int(y)))
            proc = subprocess.run(args, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
            if proc.returncode == 0:
                return True
    except Exception:
        pass
    try:
        webbrowser.open(url)
    except Exception:
        pass
    return False


def native_bar_available():
    """True when pywebview (optional dependency) can drive a native window."""
    try:
        import webview  # noqa: F401
        return True
    except Exception:
        return False


def _destroy_async(win):
    """destroy() from inside a js_api callback can deadlock the Cocoa main
    loop (the bridge thread ends up waiting on itself) — hand it off."""
    if win is None:
        return

    def _destroy():
        try:
            win.destroy()
        except Exception:
            pass

    threading.Thread(target=_destroy, daemon=True).start()


def _native_window_setup(win):
    """The two Cocoa knobs every bar window wants, once its NSWindow exists.

    Kept together because they are two halves of one promise — visible to the
    USER wherever they are, invisible to the CAPTURE — and because getting one
    without the other is the bad state: all-Spaces without exclusion burns our
    own chrome into full-screen takes that used to come out clean by accident,
    which is a live concern now that the pill really does follow (see
    `bar_native.join_all_spaces`) and only SCK actually excludes it.

    The picker deliberately does NOT go through here: it is a full-screen
    overlay for pointing at windows on the CURRENT Space, so following the user
    to every other one would be meaningless at best.

    Both calls soft-fail on their own; this returns nothing and raises nothing.
    """
    bar_native.exclude_from_capture(win)
    bar_native.join_all_spaces(win)


class _NativeFaceApi:
    """JS bridge for the detached facecam bubble (facecam.html?native=1)."""

    def __init__(self, bar_api):
        self.window = None
        self._bar_api = bar_api

    def face_moved(self):
        """Persist the bubble's position after a drag."""
        xy = bar_native.to_layout_xy(bar_native.frame_of(self.window))
        if xy is not None:
            bar_native.save_face_position(xy[0], xy[1])
        return {"ok": True}

    def face_dock(self):
        """The bubble's own "dock" button — hand it back to the pill."""
        return self._bar_api.facecam_dock()


class _NativeNotepadApi:
    """JS bridge for the invisible notes overlay (notepad.html?native=1).

    The overlay is a plain script-holder the user reads from while demoing.
    Its whole trick is being a window the BAR PROCESS owns: the recorder's pid
    sweep enumerates it and ScreenCaptureKit's content filter drops it from the
    take, so it stays on the user's screen and out of the video — the same path
    the pill and the facecam bubble ride, and the same SCK-only caveat.

    Text and geometry persist through bar_native to files beside the app.
    """

    def __init__(self, bar_api):
        self.window = None
        self._bar_api = bar_api

    def notepad_load(self):
        """The saved notes text, so the page can seed its textarea on open."""
        return {"text": bar_native.load_notepad_text()}

    def notepad_save(self, text=""):
        """Persist the notes text, and snapshot the window geometry while we're
        at it — a native edge-resize fires no drag-end of our own, so folding a
        geometry save into every text save is what catches it."""
        bar_native.save_notepad_text(text)
        self._persist_geometry()
        return {"ok": True}

    def notepad_moved(self):
        """Persist position + size after a header drag or a grip resize."""
        self._persist_geometry()
        return {"ok": True}

    def notepad_resize(self, w, h):
        """Resize the frameless window from the grip, keeping the TOP-LEFT
        corner fixed so the panel grows down-and-right under the pointer.

        Frameless WKWebView windows don't reliably offer native edge-resize,
        so the page drives this: the pointer inside the window is the new
        bottom-right corner (frameless is 1:1 CSS px to points).
        """
        win = self.window
        frame = bar_native.frame_of(win)
        if frame is None:
            return {"ok": False}
        try:
            new_w, new_h = max(1, int(w)), max(1, int(h))
        except (TypeError, ValueError):
            return {"ok": False}
        x, y, _w, _h = frame
        # Cocoa y is the BOTTOM edge; keep the top edge (y + _h) and left (x).
        target = (x, y + _h - new_h, new_w, new_h)
        # Whichever display the overlay was dragged onto, not one fixed one --
        # see bar_native.clamp_to_screens.
        target = bar_native.clamp_to_screens(target, bar_native.screen_frames())
        bar_native.set_frame(win, target)
        return {"ok": True}

    def notepad_close(self):
        """The overlay's own close button — hand back to the bar to tear down."""
        return self._bar_api.notepad_close()

    def _persist_geometry(self):
        frame = bar_native.frame_of(self.window)
        if frame is None:
            return
        xy = bar_native.to_layout_xy(frame)
        if xy is None:
            return
        bar_native.save_notepad_geometry(xy[0], xy[1], int(frame[2]),
                                         int(frame[3]))


class _NativePickerApi:
    """JS bridge for the full-screen window picker (picker.html?native=1).

    Thin on purpose: the picker's whole job is to answer one question, so the
    surface is confirm, cancel, and a boot-time handshake that lets the page
    check it is really covering the display it thinks it is.
    """

    def __init__(self, bar_api):
        self.window = None
        self.display_w = 0.0
        self._bar = bar_api

    def on_shown(self):
        """Post-creation setup, once the NSWindow actually exists."""
        # Keep the picker out of any capture, same attempt as the pill. It
        # should be gone before recording starts, but "should" is doing a lot
        # of work there and the call is free.
        bar_native.exclude_from_capture(self.window)
        win = self.window
        if win is None:
            return
        try:
            win.evaluate_js(
                "window.__pickerInit && window.__pickerInit({})".format(
                    float(self.display_w)))
        except Exception:
            pass

    def picker_done(self, ids=None, arrange=False):
        return self._bar.picker_result(ids or [], arrange)

    def picker_cancel(self):
        return self._bar.picker_cancelled()


class _NativeBarApi:
    """JS bridge for the bar page (bar.html?native=1).

    Beyond close_window() this is what lets the page drive its own native
    window: reporting how big it actually needs to be (`bar_fit`), persisting
    where it was dragged to (`bar_moved`), and popping the facecam preview out
    into its own floating window (`facecam_float`/`facecam_dock`).
    """

    def __init__(self, url_base=""):
        self.window = None
        self.url_base = url_base
        self.click_through = None
        self.face_window = None
        self._face_api = None
        self._face_lock = threading.Lock()
        self.picker_window = None
        self._picker_api = None
        self._picker_lock = threading.Lock()
        self.notepad_window = None
        self._notepad_api = None
        self._notepad_lock = threading.Lock()
        # Pending "park the pill on the window that was just picked", as a
        # layout-coord (centre_x, bottom_y). Consumed by the next bar_fit --
        # see dock_to_windows for why it is a pending anchor and not just a
        # move. Reached from two bridge threads, hence the lock.
        self._dock_anchor = None
        self._dock_lock = threading.Lock()
        # bar_fit is serialized end to end: pywebview runs every bridge call
        # on its own thread, and two fits in one JS tick (showFace + setHint)
        # could otherwise queue their window resize and glass re-frame onto
        # the main thread interleaved -- the glass (and click-through rects)
        # then settled on the OTHER fit than the window did. See bar_fit.
        self._fit_lock = threading.Lock()
        self._fit_page = None      # bar.js page id + newest applied fit seq
        self._fit_seq = None

    # --- capture exclusion ----------------------------------------------
    def capture_exclusions(self):
        """CoreGraphics window ids for every window this bar puts on screen.

        These are MEANT to reach `SCContentFilter(display:excludingWindows:)`
        so our own chrome stays visible to the USER while being absent from
        the recording, and they have to come from this process: only it can
        read its own NSWindows.

        NOTHING CALLS THIS TODAY. No `studio_web/` code invokes it over the
        pywebview bridge, so `POST /api/bar/windows` is never sent and the
        server's `_bar_window_ids` stays empty. What actually excludes the
        bar in every working topology is the pid sweep in
        `StudioState._excluded_pids` — both recorded passes, the Stage-1
        probe and the 2026-07-30 end-to-end take, were produced with this
        method dead. Do not read a green `tests/test_exclusions.py` as
        evidence that the bar reports anything.

        The gap the push would close is a bar the recording server never
        spawned (`studio.py bar` attaching to an app server that already
        holds the port), where neither source knows the pill. Whether that
        case actually burns the pill is UNMEASURED — `exclude_from_capture`
        already applies `NSWindowSharingNone`, and the measurement that
        avfoundation ignores it does not transfer to SCK. See
        docs/architecture.md's "Capture exclusion, and what is actually wired"
        for the control run that settles it, and for the two traps in wiring
        it (never push a pid; never push once at start-up).

        All four windows, because any of them can be on screen during a take:
        the pill always, the facecam bubble when popped out, the picker if it
        is somehow still up, and the notes overlay whenever it's open (which,
        unlike the others, is often the WHOLE take — it's a teleprompter). Ids
        are re-read on every call rather than cached — these windows are
        created and destroyed on demand, so a cached id would name a dead
        window (or, worse, a recycled one).

        Soft-fails per window: a `None` is dropped rather than failing the
        whole report, since excluding three of four windows still beats
        excluding none.
        """
        ids = []
        for win in (self.window, self.face_window, self.picker_window,
                    self.notepad_window):
            if win is None:
                continue
            wid = bar_native.window_number(win)
            if wid is not None and wid not in ids:
                ids.append(wid)
        return {"ids": ids}

    # --- window fitting -------------------------------------------------
    def bar_fit(self, pill_w, pill_h, hint_w=0, hint_h=0, page=None, seq=None):
        """Size the native window to the page's real content.

        The page reports the measured pill (and hint strip) in CSS px — which
        equal points for this frameless window — and bar_native.content_layout
        turns that into the window box plus the click-catching regions inside
        it. Everything outside those regions becomes click-through, so the
        transparent slack around the pill stops eating clicks meant for the
        app behind it.

        Serialized, and newest-wins. pywebview runs each bridge call on its
        own thread, and bar.js can send two fits in one tick (a face swap and
        a hint change). Unserialized, their `set_frame`, click-through rects
        and glass re-frame reached the main queue interleaved, so the glass
        could settle on the other fit than the window (measured on the real
        pill: 12 of 60 paired fits) and bar.js, having latched its last size,
        never sent the fit that would repair it. Under `_fit_lock` each fit's
        main-thread work is queued as one unit. `page`/`seq` (bar.js: a
        per-page id and a counter) drop a fit whose thread lost the race to
        a NEWER one from the same page, so the unit that lands last is the
        one the page sent last; a new page id (a reload) starts afresh.
        Callers that pass neither keep the plain serialized behavior.
        """
        with self._fit_lock:
            if seq is not None:
                try:
                    seq = float(seq)
                except (TypeError, ValueError):
                    seq = None
            if seq is not None:
                if (page == self._fit_page and self._fit_seq is not None
                        and seq <= self._fit_seq):
                    return {"ok": False, "stale": True}
                self._fit_page, self._fit_seq = page, seq
            return self._bar_fit_locked(pill_w, pill_h, hint_w, hint_h)

    def _bar_fit_locked(self, pill_w, pill_h, hint_w, hint_h):
        win = self.window
        if win is None:
            return {"ok": False}
        frame = bar_native.frame_of(win)
        if frame is None:
            return {"ok": False}
        try:
            w, h, rects = bar_native.content_layout(
                pill_w, pill_h, hint_w, hint_h)
        except (TypeError, ValueError):
            return {"ok": False}
        # A pending dock wins over the anti-jump rule: the user just told us
        # where the pill belongs, and this fit is the one that knows the size
        # it will be when it gets there.
        fitted = (self._take_dock_frame((w, h))
                  or bar_native.fit_frame(frame, w, h))
        # Clamp onto the display the pill is ON, not onto one chosen display.
        # This used to be `clamp_frame` against `NSScreen.mainScreen()`, which
        # follows keyboard focus: with the browser focused on a second
        # monitor, the first fit after a drag hauled the pill back onto that
        # monitor, so it could never be moved to the other display.
        fitted = bar_native.clamp_to_screens(fitted, bar_native.screen_frames())
        bar_native.set_frame(win, fitted)
        if self.click_through is not None:
            # layout coords, NOT screen: set_frame is async, so anything
            # resolved against `fitted` here could describe a position the
            # window hasn't reached. ClickThrough converts against the live
            # frame instead.
            self.click_through.set_rects(rects)
        out = {"ok": True, "w": fitted[2], "h": fitted[3]}
        # Native frosted glass (bar_native.Glass), re-framed to the same rects.
        # Queued AFTER set_frame -- callAfter is FIFO -- so it lands against
        # the size the window was just given. "glass" is reported only once
        # the effect views really exist: bar.js keeps the opaque pill until
        # then, so a failed install can never leave a see-through one. On the
        # lazy path (`shown` never installed) this fit's reply can't say so
        # yet -- the install is only queued -- so Glass tells the page itself
        # once the views are framed (bar_native.GLASS_ON_JS).
        # getattr: `glass` is attached by run_native_bar, not __init__.
        glass = getattr(self, "glass", None)
        if glass is not None and glass.update(rects) and glass.installed:
            out["glass"] = True
        return out

    def bar_moved(self):
        """Persist the pill's position after a drag."""
        xy = bar_native.to_layout_xy(bar_native.frame_of(self.window))
        if xy is not None:
            bar_native.save_bar_position(xy[0], xy[1])
        return {"ok": True}

    # --- docking onto a picked window -----------------------------------
    def dock_to_windows(self, ids=None):
        """Park the pill bottom-centre on the window(s) just elected.

        Layout coords the whole way: `devices.window_rect_points` reports rects
        in the global top-left POINT space, which is the space `bar_native`'s
        dock helpers use, so nothing converts until the final flip into Cocoa.

        The placement is deliberately NOT persisted. `bar-pos.json` holds where
        the user DRAGGED the pill, and picking a window in the overlay must not
        silently overwrite that — dock for this session, and a drag afterwards
        saves exactly as it always did.

        The anchor is left pending as well as applied, because a `bar_fit`
        almost always follows within milliseconds (the picked window's name
        goes into the hint strip, which resizes the window) and `set_frame` is
        async: a fit that raced this call would read a pre-dock frame and drag
        the pill straight back. Whichever lands last, both agree on the anchor.

        Every failure is a quiet no-op — a stale id, no Quartz, an unreadable
        frame. The pill staying put costs nothing; an exception crossing the JS
        bridge mid-pick could cost a recording.
        """
        rects = []
        for wid in (ids or []):
            try:
                entry = dev.window_rect_points(int(wid),
                                               exclude_pids=(os.getpid(),))
            except (TypeError, ValueError):
                continue
            if entry:
                rects.append((entry["x"], entry["y"],
                              entry["w"], entry["h"]))
        target = bar_native.union_rect(rects)
        if target is None:
            return {"ok": False, "reason": "no window rect"}
        anchor = bar_native.dock_anchor(target)
        with self._dock_lock:
            self._dock_anchor = anchor
        frame = bar_native.frame_of(self.window)
        if frame is not None:
            self._apply_dock_frame(
                bar_native.dock_frame(anchor, (frame[2], frame[3])))
        return {"ok": True}

    def _take_dock_frame(self, size):
        """The Cocoa frame for a pending dock at `size`, clearing the anchor.

        None in the ordinary case (nothing pending), which is what keeps
        `bar_fit`'s anti-jump behavior byte-for-byte what it was before docking
        existed.
        """
        with self._dock_lock:
            anchor = self._dock_anchor
            self._dock_anchor = None
        if anchor is None:
            return None
        return bar_native.dock_frame(anchor, size)

    def _apply_dock_frame(self, frame):
        """Clamp a docked frame to its display and move the pill onto it."""
        if frame is None or self.window is None:
            return False
        frame = bar_native.clamp_to_screens(frame, bar_native.screen_frames())
        return bar_native.set_frame(self.window, frame)

    # --- detached facecam bubble ----------------------------------------
    def close_picker_if_open(self):
        """Tear the picker down on any bar exit path (quit, record, reload)."""
        return {"ok": True, "closed": self._close_picker()}

    def facecam_float(self, ordinal=0):
        """Open the webcam preview as its own floating always-on-top window.

        The caller (bar.js) has already released its preview — macOS hands the
        camera to one consumer at a time, and the floating bubble becomes that
        consumer until it docks again. `ordinal` is the OpenCV camera ordinal
        the bubble should stream (see camera_preview.list_cameras).
        """
        try:
            import webview
        except Exception:
            return {"ok": False, "reason": "pywebview unavailable"}
        with self._face_lock:
            if self.face_window is not None:
                return {"ok": True, "reason": "already open"}
            size = 168
            pos = bar_native.load_face_position()
            api = _NativeFaceApi(self)
            try:
                win = webview.create_window(
                    "AutoCine — Facecam",
                    url=shell_url(self.url_base, "facecam.html", native=1,
                                  ordinal=_as_int(ordinal, 0)),
                    js_api=api,
                    width=size,
                    height=size,
                    x=None if pos is None else pos[0],
                    y=None if pos is None else pos[1],
                    screen=bar_native.origin_screen(),   # see run_native_bar
                    frameless=True,
                    easy_drag=False,
                    on_top=True,
                    resizable=False,
                    transparent=True,
                    background_color="#0b0b0f",
                )
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}
            api.window = win
            # Same capture-exclusion attempt as the pill (see run_native_bar),
            # and the same all-Spaces behavior -- a bubble that stayed behind
            # on the desktop Space while the pill followed would be worse than
            # either alone. The bubble is created while the GUI loop is already
            # running, so its NSWindow may not exist yet -- hang the calls off
            # `shown` when pywebview exposes it, and fall back to trying
            # immediately.
            try:
                win.events.shown += lambda: _native_window_setup(win)
            except Exception:
                _native_window_setup(win)
            self._face_api = api
            self.face_window = win
            return {"ok": True}

    # --- window picker --------------------------------------------------
    def pick_windows(self):
        """Open the full-screen hover picker over the whole main display.

        Same second-window pattern as `facecam_float`, with three deliberate
        differences, each one a lesson from the countdown overlay on
        `bar-native-polish` that had to be ripped back out:

          * `background_color` is a SIX-digit hex. pywebview validates it
            against `^#(?:[0-9a-fA-F]{3}){1,2}$` and RAISES on anything else
            -- an 8-digit "#00000000" is exactly what killed that attempt.
            Transparency comes from `transparent=True` plus CSS, never from
            an alpha channel here.
          * never `fullscreen=True`: that is native macOS fullscreen, which
            would open a whole new Space and hide the very windows the user
            is trying to point at. We size ourselves to NSScreen.frame().
          * every failure returns `{"ok": False}` instead of raising, and the
            bar falls straight back to its <select>. A picker that won't open
            must never be able to cost someone a recording.
        """
        try:
            import webview
        except Exception:
            return {"ok": False, "reason": "pywebview unavailable"}
        if os.environ.get("AUTOCINE_NO_PICKER") == "1":
            return {"ok": False, "reason": "picker disabled"}
        with self._picker_lock:
            if self.picker_window is not None:
                return {"ok": True, "reason": "already open"}
            screen = bar_native.main_screen_layout()
            if screen is None:
                return {"ok": False, "reason": "no screen"}
            sx, sy, sw, sh = screen
            api = _NativePickerApi(self)
            try:
                win = webview.create_window(
                    "AutoCine — Pick Windows",
                    url=shell_url(self.url_base, "picker.html", native=1),
                    js_api=api,
                    width=int(sw), height=int(sh),
                    x=int(sx), y=int(sy),
                    screen=bar_native.origin_screen(),   # see run_native_bar
                    frameless=True,
                    easy_drag=False,
                    on_top=True,
                    resizable=False,
                    shadow=False,
                    transparent=True,
                    background_color="#0b0b0f",
                )
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}
            api.window = win
            api.display_w = float(sw)
            try:
                win.events.shown += lambda: api.on_shown()
            except Exception:
                api.on_shown()
            self._picker_api = api
            self.picker_window = win
            return {"ok": True}

    def _close_picker(self):
        with self._picker_lock:
            win = self.picker_window
            self.picker_window = None
            self._picker_api = None
        _destroy_async(win)
        return win is not None

    def picker_result(self, ids, arrange):
        """The picker confirmed. Hand the ids to the bar, dock onto them, and
        close.

        Docking happens AFTER the eval, not before: `__barWindowsPicked` is
        what triggers the resize this dock wants to be sized against, and the
        pending anchor is picked up by that fit either way.
        """
        self._close_picker()
        ids = [int(i) for i in (ids or [])]
        payload = json.dumps({"ids": ids, "arrange": bool(arrange)})
        self._eval_bar("window.__barWindowsPicked && "
                       "window.__barWindowsPicked({})".format(payload))
        self.dock_to_windows(ids)
        return {"ok": True}

    def picker_cancelled(self):
        """The picker was dismissed. Leave the bar's selection alone."""
        self._close_picker()
        self._eval_bar("window.__barPickerClosed && window.__barPickerClosed()")
        return {"ok": True}

    def facecam_dock(self):
        """Close the floating bubble; the pill takes the camera back."""
        with self._face_lock:
            win = self.face_window
            self.face_window = None
            self._face_api = None
        _destroy_async(win)
        # tell the pill to re-acquire and show its inline preview
        self._eval_bar("window.__barFacecamDocked && window.__barFacecamDocked()")
        return {"ok": True, "closed": win is not None}

    def facecam_release(self):
        """Drop the bubble's camera grip so the capture ffmpeg can open it,
        and take the bubble off screen for the duration of the take.

        The bubble is an always-on-top window floating over the display being
        captured, so whatever it shows once the camera is gone — a placeholder,
        or just the empty disc — gets burned into raw.mov, and from there into
        the cached thumbnail, forever. Hidden rather than destroyed so the
        position the user parked it at survives the recording.
        """
        self._eval_face("window.__faceRelease && window.__faceRelease()")
        self._face_visible(False)
        return {"ok": True}

    def facecam_resume(self):
        """Re-acquire the camera after a recording ends."""
        # back on screen first: a hidden web view is a throttled one, and the
        # MJPEG stream __faceResume starts should not be racing that
        self._face_visible(True)
        self._eval_face("window.__faceResume && window.__faceResume()")
        return {"ok": True}

    def _face_visible(self, visible):
        """Show/hide the floating bubble's native window. Soft-fails like
        everything else Cocoa-side: no pywebview (or a window that never
        finished showing) and the bubble simply stays where it is."""
        win = self.face_window
        if win is None:
            return False
        try:
            if visible:
                win.show()
            else:
                win.hide()
        except Exception:
            return False
        return True

    def _eval_face(self, script):
        win = self.face_window
        if win is None:
            return
        try:
            win.evaluate_js(script)
        except Exception:
            pass

    def _eval_bar(self, script):
        win = self.window
        if win is None:
            return
        try:
            win.evaluate_js(script)
        except Exception:
            pass

    # --- notes overlay --------------------------------------------------
    def notepad_open(self):
        """Open the invisible notes overlay as its own floating window.

        Same second-window pattern as `facecam_float`, with two deliberate
        differences: it is RESIZABLE (a script wants room, and the user drags
        the grip in notepad.js) and it is NOT click-through — the whole point
        is to type into it, so every pixel is interactive and there is no
        transparent slack to fall through.

        Capture exclusion is inherited, not re-plumbed: this window belongs to
        the bar process, so the recorder's pid sweep (`capture_exclude_ids`)
        already enumerates it and SCK drops it from the take. `NSWindowSharing
        None` is applied too, as on every other bar window. Both only bite on
        the SCK backend; bar.js warns when Notes is opened on avfoundation.
        """
        try:
            import webview
        except Exception:
            return {"ok": False, "reason": "pywebview unavailable"}
        with self._notepad_lock:
            if self.notepad_window is not None:
                return {"ok": True, "reason": "already open"}
            geom = bar_native.load_notepad_geometry()
            if geom is not None:
                x, y, w, h = geom
            else:
                x, y, w, h = None, None, 360, 420
            api = _NativeNotepadApi(self)
            try:
                win = webview.create_window(
                    "AutoCine — Notes",
                    url=shell_url(self.url_base, "notepad.html", native=1),
                    js_api=api,
                    width=int(w),
                    height=int(h),
                    x=None if x is None else int(x),
                    y=None if y is None else int(y),
                    screen=bar_native.origin_screen(),   # see run_native_bar
                    min_size=(220, 150),
                    frameless=True,
                    easy_drag=False,
                    on_top=True,
                    resizable=True,
                    transparent=True,
                    background_color="#0b0b0f",
                )
            except Exception as exc:
                return {"ok": False, "reason": str(exc)}
            api.window = win
            # Same capture-exclusion attempt as the pill, and the same
            # all-Spaces behavior -- this one is the teleprompter, so a Space
            # switch hiding it would blank the script mid-take. Hung off
            # `shown` since the NSWindow may not exist the instant create
            # returns.
            try:
                win.events.shown += lambda: _native_window_setup(win)
            except Exception:
                _native_window_setup(win)
            self._notepad_api = api
            self.notepad_window = win
            return {"ok": True}

    def notepad_close(self):
        """Close the notes overlay, saving its final geometry first, and tell
        the pill so it can un-press its Notes toggle."""
        with self._notepad_lock:
            api = self._notepad_api
            win = self.notepad_window
            self.notepad_window = None
            self._notepad_api = None
        if api is not None:
            try:
                api._persist_geometry()
            except Exception:
                pass
        _destroy_async(win)
        self._eval_bar("window.__barNotesClosed && window.__barNotesClosed()")
        return {"ok": True, "closed": win is not None}

    # --- teardown -------------------------------------------------------
    def close_window(self):
        with self._face_lock:
            face = self.face_window
            self.face_window = None
            self._face_api = None
        _destroy_async(face)
        with self._notepad_lock:
            note = self.notepad_window
            self.notepad_window = None
            self._notepad_api = None
        _destroy_async(note)
        # The picker is always-on-top and full-screen; leaving it behind
        # after the bar quits would be an unclosable dimmed desktop.
        self._close_picker()
        _destroy_async(self.window)


_ACTIVE_STATE = None

# The bar API of the process that owns the native window, so a signal handler
# on another thread can reach it. There is exactly one native bar per process.
_ACTIVE_BAR_API = None


def stop_native_bar():
    """Best-effort teardown from OUTSIDE the Cocoa main thread (Ctrl+C).

    Stopping an in-flight recording first is the whole point: ffmpeg needs its
    SIGINT to write the moov atom, and a take killed without it is an
    unplayable file. Everything here is wrapped -- a failed tidy-up must not
    stop the process from exiting, which is what the user actually asked for.
    """
    api = _ACTIVE_BAR_API
    state = _ACTIVE_STATE
    if state is not None:
        try:
            if state.snapshot().get("record", {}).get("status") in (
                    "countdown", "recording", "pausing", "paused",
                    "resuming"):
                state.stop_record()
                # Give _finalize its window to write the trailer.
                deadline = time.time() + 8.0
                while time.time() < deadline:
                    if state.snapshot().get("record", {}).get("status") not in (
                            "recording", "stopping", "pausing", "paused",
                            "resuming"):
                        break
                    time.sleep(0.15)
        except Exception:
            pass
    if api is not None:
        try:
            api.close_picker_if_open()
        except Exception:
            pass


def run_native_bar(url_base, width, height, x=None, y=None):
    """Open the recording bar as a frameless, always-on-top native window
    (pywebview / WKWebView). Blocks until the window is closed.

    `width`/`height` are only the *initial* size — once the page loads it
    reports its real content box through `api.bar_fit` and the window shrinks
    to the pill (see bar_native). A previously dragged position wins over the
    caller's (x, y).

    Returns True when the window ran (and has now closed), False when
    pywebview is unavailable or window creation failed -- callers fall back
    to open_app_window(). Must be called from the main thread (Cocoa).
    """
    try:
        import webview
    except Exception:
        return False
    api = _NativeBarApi(url_base=url_base)
    global _ACTIVE_BAR_API
    _ACTIVE_BAR_API = api
    # BEFORE any window exists, including this process's later bubble/picker/
    # notes windows: pywebview's plain NSWindow can never join another app's
    # full-screen Space, and the class cannot be changed once the window is
    # built. See bar_native.use_panel_windows.
    bar_native.use_panel_windows()
    # Ask for camera access here, on the main thread, before the web view
    # exists. WKWebView has no mediaDevices to prompt on our behalf and
    # OpenCV can't prompt from its capture worker, so without this the
    # facecam preview fails with no prompt ever shown. No-op once answered.
    camera_preview.request_access()
    saved = bar_native.load_bar_position()
    if saved is not None:
        x, y = saved
    try:
        window = webview.create_window(
            "AutoCine — New Recording",
            url=shell_url(url_base, "bar.html", native=1),
            js_api=api,
            width=int(width),
            height=int(height),
            x=None if x is None else int(x),
            y=None if y is None else int(y),
            # Anchor every coordinate pywebview computes for this window to
            # the PRIMARY display. Unset, the cocoa backend caches
            # `NSScreen.mainScreen()` at creation and then reads the drag
            # region's GLOBAL `ev.screenX/screenY` as offsets from it -- so a
            # pill opened while a second monitor had focus could not be
            # dragged onto the other display. See bar_native.origin_screen.
            screen=bar_native.origin_screen(),
            frameless=True,
            easy_drag=False,   # default True swallows clicks (whole window
                               # becomes a drag surface); the pill marks its
                               # own .pywebview-drag-region instead
            on_top=True,
            resizable=False,
            transparent=True,
            background_color="#0b0b0f",
        )
        api.window = window
        # MEASURED 2026-08-13: `webview.start(func)` runs `func` on a thread
        # BEFORE `guilib.create_window(windows[0])`, so inside `_on_start`
        # there is no NSWindow yet and every bar_native call silently no-ops
        # (`_ns_window` returns None). The pill's `exclude_from_capture` lived
        # there and had therefore never actually applied `NSWindowSharingNone`
        # -- unlike the bubble and the notes overlay, which are created while
        # the loop is up and already hang their setup off `shown`. Hang the
        # pill's off `shown` too; that is the first point the window exists.
        # `tools/spaces_pill_probe.py` pins both halves.
        try:
            window.events.shown += lambda: _native_window_setup(window)
        except Exception:
            pass                       # no `shown` -> the _on_start fallback
        # Only the pill's own rect should eat clicks; the transparent slack
        # kept for its drop shadow falls through to the app behind. If this
        # can't arm, the window just stays fully interactive — dead slack is
        # a much smaller problem than an unclickable pill
        # (AUTOCINE_NO_CLICKTHROUGH=1 forces that fallback).
        click_through = bar_native.ClickThrough(window)
        api.click_through = click_through
        # Real frosted glass under the pill: NSVisualEffectViews below the
        # web content, framed by every bar_fit. Created off `shown` for the
        # same reason as _native_window_setup (no NSWindow before it); the
        # first bar_fit installs lazily if that never ran. The pill only --
        # the bubble/notes/picker windows keep their own look.
        # AUTOCINE_NO_VIBRANCY=1 keeps the opaque CSS pill.
        glass = bar_native.Glass(window)
        api.glass = glass
        try:
            window.events.shown += lambda: glass.install()
        except Exception:
            pass

        def _on_start():
            click_through.start()
            # Fallback only -- the `shown` handler above is what lands. Kept
            # for a pywebview that exposes no `shown` event, where this is
            # still strictly better than nothing even though it usually races
            # the window into existence and loses.
            _native_window_setup(window)

        webview.start(_on_start)
        click_through.stop()
        return True
    except Exception as exc:
        print("native bar window failed ({}); falling back to a browser "
              "window".format(exc), file=sys.stderr)
        return False


class StudioState:
    def __init__(self, recordings_root):
        # Reachable from a signal handler on another thread: Ctrl+C has to be
        # able to stop an in-flight take so ffmpeg writes its trailer.
        global _ACTIVE_STATE
        _ACTIVE_STATE = self
        self.recordings_root = os.path.abspath(recordings_root)
        os.makedirs(self.recordings_root, exist_ok=True)
        self._lock = threading.Lock()
        self._record_thread = None
        self._active_recorder = None
        # Window ids on screen when a fleet take STARTED -- the seamless-join
        # trigger offers only windows that appeared SINCE (new since start).
        # None except during a fleet take.
        self._grow_baseline_ids = None
        self._record_status = {"status": "idle", "message": "", "session": None}
        self._render_thread = None
        self._transcribe_jobs = {}
        self._render_status = {"status": "idle", "message": "", "session": None}
        self._describe_cache = OrderedDict()
        self._describe_lock = threading.Lock()
        # serializes load->merge->save cycles within this process so two
        # concurrent HTTP saves can't interleave (cross-process safety comes
        # from the rev compare-and-swap in save_session_edits)
        self._edits_lock = threading.Lock()
        self._bar_lock = threading.Lock()
        self._bar_process = None
        # Window ids the bar has reported for its own chrome, for capture
        # exclusion. Guarded by _bar_lock along with _bar_process.
        self._bar_window_ids = []
        self._bar_watchdog = None
        # boot_id changes on every server process start; the frontend polls
        # /api/rev and reloads open tabs when it sees a different value.
        # Live-reload mode is signaled by AUTOCINE_LIVE=1 (set by the
        # supervisor in livereload.run); default is off.
        self.boot_id = "{:d}".format(time.time_ns())
        self.live_reload = os.environ.get("AUTOCINE_LIVE") == "1"

    def _excluded_pids(self):
        """PIDs whose windows must never show up in the window picker.

        os.getpid() alone is not enough: the native pill is usually a
        DETACHED CHILD (launch_native_bar spawns `studio.py bar --port N` in
        its own session), so its always-on-top window belongs to a different
        process. devices also filters our chrome by title as a backstop.
        """
        pids = {os.getpid()}
        with self._bar_lock:
            proc = self._bar_process
        if proc is not None and proc.poll() is None:
            try:
                pids.add(int(proc.pid))
            except (TypeError, ValueError):
                pass
        return pids

    def settings_payload(self):
        """Current preferences, plus the backend that a take would ACTUALLY
        use right now.

        `capture_backend` is what is saved; `capture_backend_effective` is
        what would run, which differs whenever the environment overrides it.
        The UI shows the effective one — the whole point of this is that
        nobody should have to guess which engine a recording used.
        """
        saved = settings.get("capture_backend")
        # Tri-state, deliberately: None means the user has never expressed a
        # preference, and the bar turns auto-add ON in that case. Collapsing
        # it to a bool here would make "never chosen" indistinguishable from
        # "explicitly turned off".
        auto_add = settings.get("auto_add_windows")
        return {
            "capture_backend": saved,
            "capture_backend_effective": sck.resolve_backend(
                None, os.environ, saved),
            "capture_backend_locked": bool(
                os.environ.get("AUTOCINE_CAPTURE_BACKEND")),
            "backends": list(sck.BACKENDS),
            "auto_add_windows": (None if auto_add is None else bool(auto_add)),
        }

    def save_settings(self, values):
        """Persist preferences. Unknown keys and bad values are dropped."""
        clean = {}
        backend = (values or {}).get("capture_backend")
        if backend is not None:
            # Store only what resolve_backend would honor, so a typo lands
            # as a visible no-op now rather than a silent fallback later.
            value = str(backend).strip().lower()
            if value not in sck.BACKENDS:
                raise StudioError("unknown capture backend: {}".format(backend),
                                  status=400)
            clean["capture_backend"] = value
        auto_add = (values or {}).get("auto_add_windows")
        if auto_add is not None:
            # Only ever written from an explicit toggle, so the stored value
            # always means "the user chose this" -- never a default echoed
            # back by a client that merely rendered one.
            clean["auto_add_windows"] = bool(auto_add)
        settings.save(clean)
        return self.settings_payload()

    def set_bar_window_ids(self, ids):
        """Record the window ids the bar reports for itself.

        The bar is the only process that can read its own NSWindows, so the
        design is that it pushes and we never pull — stored rather than
        fetched on demand because the recorder needs them at ffmpeg-start
        time, when a round trip to a possibly-wedged GUI process is exactly
        what we don't want.

        NO PRODUCTION CALLER pushes today (see `_NativeBarApi
        .capture_exclusions`), so in practice this stays `[]` and
        `capture_exclude_ids` returns the pid sweep alone. Reached only by
        tests and by hand.

        REPLACES rather than accumulates, and that is load-bearing if this is
        ever wired: the ids have no expiry — not even `shutdown()` clears
        them — and macOS recycles CGWindowIDs, so a stale id can come to name
        a stranger's window and silently cut it from a later recording.
        Replacement is the only thing that self-heals, which is why a wiring
        must push per take rather than once at start-up.
        """
        clean = []
        for wid in ids or []:
            try:
                wid = int(wid)
            except (TypeError, ValueError):
                continue
            if wid > 0 and wid not in clean:
                clean.append(wid)
        with self._bar_lock:
            self._bar_window_ids = clean
        return clean

    def capture_exclude_ids(self):
        """Every window id that must be kept out of a recording.

        Union of two independent sources, because each one alone has a hole:

          * what the bar REPORTED — authoritative for its own windows, but
            only as fresh as the last push, and blind to windows it doesn't
            own (WebKit spawns service windows in the same process);
          * a live pid sweep — catches everything on screen right now, but
            only for pids we know about, and only while they're on screen.

        Union, not either/or: a window missed by both is a window burned into
        the user's recording, and that is the failure this whole path exists
        to prevent.
        """
        with self._bar_lock:
            reported = list(self._bar_window_ids)
        out = list(reported)
        for wid in dev.window_ids_for_pids(self._excluded_pids()):
            if wid not in out:
                out.append(wid)
        return out

    def _describe_cache_key(self, session_dir):
        """(dir, raw mtime, events mtime, meta mtime): changes whenever the
        inputs do.

        meta.json is in the key because the describe payload now derives
        width/height from its `capture_window` block — i.e. meta.json defines
        the editor's source coordinate space. Nothing in the app rewrites
        meta.json today (the Recorder is its only writer, at stop), so this
        costs one stat and never actually fires; it is here so that a future
        meta rewrite can't silently hand the editor a stale coordinate space.
        """
        raw_name, events_name = "raw.mov", "events.jsonl"
        try:
            with open(os.path.join(session_dir, "meta.json")) as f:
                meta = json.load(f)
            raw_name = meta.get("raw", raw_name)
            events_name = meta.get("events", events_name)
        except Exception:
            pass

        def _mtime(p):
            try:
                return os.path.getmtime(p)
            except OSError:
                return -1.0

        return (session_dir,
                _mtime(os.path.join(session_dir, raw_name)),
                _mtime(os.path.join(session_dir, events_name)),
                _mtime(os.path.join(session_dir, "meta.json")))

    def _describe(self, session_dir, include_click_times=False):
        """Cached ren.describe_session (keyed on raw/events mtimes).

        The full include_click_times=True result is cached once; the lighter
        variant is derived from it. Output-file flags are re-checked on every
        call (they change without the key changing, e.g. after a render).
        """
        key = self._describe_cache_key(session_dir)
        with self._describe_lock:
            cached = self._describe_cache.get(key)
            if cached is not None:
                self._describe_cache.move_to_end(key)
        if cached is None:
            cached = ren.describe_session(session_dir, include_click_times=True)
            with self._describe_lock:
                self._describe_cache[key] = cached
                self._describe_cache.move_to_end(key)
                while len(self._describe_cache) > _DESCRIBE_CACHE_MAX:
                    self._describe_cache.popitem(last=False)
        out = dict(cached)
        out["has_output_mp4"] = os.path.isfile(
            os.path.join(session_dir, "output.mp4"))
        out["has_output_gif"] = os.path.isfile(
            os.path.join(session_dir, "output.gif"))
        if include_click_times:
            out["click_times"] = list(cached.get("click_times") or [])
            out["key_times"] = list(cached.get("key_times") or [])
            out["scroll_times"] = list(cached.get("scroll_times") or [])
        else:
            out.pop("click_times", None)
            out.pop("key_times", None)
            out.pop("scroll_times", None)
        return out

    def _drop_describe_cache(self, session_dir):
        with self._describe_lock:
            stale = [k for k in self._describe_cache if k[0] == session_dir]
            for k in stale:
                self._describe_cache.pop(k, None)

    def _session_dir(self, session_name):
        if not session_name:
            raise StudioError("missing session name", status=400)
        if os.path.basename(session_name) != session_name:
            raise StudioError("invalid session name", status=400)
        path = os.path.abspath(os.path.join(self.recordings_root, session_name))
        root = self.recordings_root
        if path != root and not path.startswith(root + os.sep):
            raise StudioError("invalid session path", status=400)
        if not os.path.isdir(path):
            raise StudioError("session not found: {}".format(session_name), status=404)
        return path

    def list_sessions(self):
        out = []
        for name in sorted(os.listdir(self.recordings_root), reverse=True):
            d = os.path.join(self.recordings_root, name)
            if not os.path.isdir(d):
                continue
            meta_path = os.path.join(d, "meta.json")
            if not os.path.isfile(meta_path):
                continue
            try:
                info = self._describe(d)
                saved = edits.load_edits(d, duration=info["duration"])
                chapters = edits.marker_chapters(saved, duration=info["duration"])
                trim_start = saved["trim"]["start"]
                trim_end = saved["trim"]["end"]
                trimmed = max(0.0, float(trim_end) - float(trim_start))
                out.append({
                    "session": info["session"],
                    "name": load_project_name(d),
                    "duration": info["duration"],
                    "trimmed_duration": trimmed,
                    "fps": info["fps"],
                    "size": [info["width"], info["height"]],
                    "click_count": info["click_count"],
                    "trim": saved["trim"],
                    "chapter_count": len(chapters),
                    "has_edits": edits.has_edits(d),
                    "has_output_mp4": info["has_output_mp4"],
                    "has_output_gif": info["has_output_gif"],
                })
            except Exception as exc:
                out.append({
                    "session": name,
                    "name": load_project_name(d),
                    "error": str(exc),
                    "has_edits": edits.has_edits(d),
                    "has_output_mp4": os.path.isfile(os.path.join(d, "output.mp4")),
                    "has_output_gif": os.path.isfile(os.path.join(d, "output.gif")),
                })
        return out

    def _focus_plan(self, session_dir, doc):
        """The window-focus plan for `doc`'s cards, or [].

        Purely temporal -- which card is the subject and when -- so it needs
        no canvas layout at all; what "focused" looks like is decided at
        render time against the live canvas. Like every other one-shot here,
        a failure costs the caller nothing worse than an un-materialized
        plan, because the render path still auto-plans from the clicks.
        """
        # A multi-native take never reaches here: `_load_edits_with_auto`
        # skips focus materialization for it (its focus is AUTO-planned at
        # render time and not editable per card in v1, because every card
        # shares the same click TIMES -- clamp-not-drop -- so a materialized
        # per-card span plan would arbitrate differently than the auto-plan.
        # See the guard in `_load_edits_with_auto` and
        # docs/architecture.md, "Window-focus stays auto").
        windows = list(doc.get("windows") or [])
        if not windows:
            return []
        kw = self._render_kwargs_from_edits(doc)
        try:
            return ren.multi_window_focus_ranges(
                session_dir, windows,
                offset=kw["offset"], max_zoom=kw["max_zoom"],
                params=kw["camera_params"],
                suppressed_ranges=kw["suppressed_ranges"],
                window_follow=kw["window_follow"],
                crop_rect=kw["crop_rect"])
        except Exception:
            return []

    def _load_edits_with_auto(self, session_dir, info):
        """Load edits + one-shot materialize the auto zoom proposals.

        Idempotent: after the first call the doc carries
        `auto_zooms_initialized: True` and later calls are pure reads.
        Acquires the edits lock only when a write is needed."""
        # Multi-native focus is AUTO-planned, never materialized into editable
        # spans: every card shares the same click TIMES (clamp-not-drop), so
        # `camera.plan_focus_ranges` would arbitrate the shared cluster into
        # per-card spans that don't reproduce the auto-plan emphasis -- unlike
        # the display-crop path, whose clicks are owned by position. Leaving it
        # stale keeps the render, the export, and the live preview all on the
        # same auto-plan (`_render_kwargs_from_edits` passes focus_ranges=None
        # while stale). See docs/architecture.md, "Window-focus
        # stays auto".
        # Scene takes share the multi-native posture end to end: focus stays
        # auto-planned (per-scene, at render time) and the capture-window
        # seeding flips the layout default to `desktop`.
        native = bool(info.get("multi_native") or info.get("scene_take"))
        loaded = edits.load_edits(session_dir, duration=info["duration"])
        if (loaded.get("auto_zooms_initialized")
                and loaded.get("capture_windows_initialized")
                and (native or not edits.focus_plan_is_stale(loaded))):
            return loaded
        click_times = info.get("click_times")
        if click_times is None:
            full = self._describe(session_dir, include_click_times=True)
            click_times = full.get("click_times") or []
        with self._edits_lock:
            current = edits.load_edits(session_dir, duration=info["duration"])
            changed_any = False
            if not current.get("auto_zooms_initialized"):
                current, changed = edits.initialize_auto_zooms(
                    current, click_times, info["duration"])
                changed_any = changed_any or changed
            # A record-time multi-window pick becomes the session's cards
            # exactly once, on first open -- same posture as the auto zooms,
            # and it never touches a `windows` array the user already has.
            if not current.get("capture_windows_initialized"):
                current, changed = edits.initialize_capture_windows(
                    current, ren.capture_window_specs(session_dir),
                    multi_native=native)
                changed_any = changed_any or changed
            # The composition camera's plan becomes editable arcs, once. AFTER
            # the capture-window seeding above, because it plans against the
            # cards that seeding may have just created. Skipped for multi-
            # native (focus is auto-planned there -- see the note above).
            if not native and edits.focus_plan_is_stale(current):
                current, changed = edits.initialize_focus_ranges(
                    current, self._focus_plan(session_dir, current))
                changed_any = changed_any or changed
            if not changed_any:
                return current
            return edits.save_edits(session_dir, current,
                                    duration=info["duration"])

    def get_session(self, session_name):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir, include_click_times=True)
        info["name"] = load_project_name(session_dir)
        info["edits"] = self._load_edits_with_auto(session_dir, info)
        info["chapters"] = edits.marker_chapters(info["edits"], duration=info["duration"])
        info["has_edits"] = edits.has_edits(session_dir)
        return info

    def get_edits(self, session_name):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        return self._load_edits_with_auto(session_dir, info)

    @staticmethod
    def _cuts_ok(doc, info):
        try:
            edits.plan_cuts(doc.get("cuts") or [], doc.get("trim"),
                            info["duration"], info.get("fps") or 60.0)
        except edits.CutsError as exc:
            return str(exc)
        return None

    def _guard_cuts(self, merged, current, info):
        """Refuse a save that BREAKS the cut contract; allow one that merely
        inherits an already-broken document. See save_session_edits."""
        problem = self._cuts_ok(merged, info)
        if problem is None:
            return
        if self._cuts_ok(current, info) is not None:
            return          # already invalid on disk — this save is not the cause
        raise CutsRejected(problem, current)

    def save_session_edits(self, session_name, edits_obj, base_rev=None):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        with self._edits_lock:
            current = edits.load_edits(session_dir, duration=info["duration"])
            if base_rev is not None:
                try:
                    base_rev = int(base_rev)
                except (TypeError, ValueError):
                    base_rev = None
            if base_rev is not None and base_rev != current.get("rev", 0):
                # another client (second tab, MCP server) saved since this
                # client last loaded — refuse instead of clobbering their work
                raise EditsConflict(current)
            merged = edits.merge_edits(current, edits_obj,
                                       duration=info["duration"])
            # The cuts write contract, same rule the MCP tools enforce.
            # Checked on the MERGED doc (a patch carrying only `cuts` still
            # has to answer for the trim already on disk) and AFTER the CAS,
            # so a stale-rev save reports the conflict rather than a budget
            # it was never really writing against.
            #
            # It is a DELTA check, not a state check. The editor autosaves
            # the whole document, so a state check would mean one bad
            # edits.json — hand-written, or left by an older build — locks
            # the user out of every unrelated edit forever, with an error
            # about cuts they may not have made. A save that leaves an
            # already-invalid document no worse is allowed through; only a
            # save that BREAKS a valid one is refused.
            self._guard_cuts(merged, current, info)
            return edits.save_edits(session_dir, merged,
                                    duration=info["duration"])

    def reset_session_edits(self, session_name):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        return edits.reset_edits(session_dir, duration=info["duration"])

    def duplicate_preset(self, session_name, name=None, source_preset_id=None,
                         source_render=None):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        current = edits.load_edits(session_dir, duration=info["duration"])
        updated = edits.duplicate_preset(
            current,
            source_preset_id=source_preset_id,
            name=name,
            source_render=source_render,
            select_new=True,
            duration=info["duration"],
        )
        return edits.save_edits(session_dir, updated, duration=info["duration"])

    def activate_preset(self, session_name, preset_id):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        current = edits.load_edits(session_dir, duration=info["duration"])
        updated = edits.set_active_preset(
            current,
            preset_id=preset_id,
            duration=info["duration"],
        )
        return edits.save_edits(session_dir, updated, duration=info["duration"])

    def devices(self):
        d = dev.list_avf_devices()
        return {
            "video": [{"index": idx, "name": name} for idx, name in d.get("video", [])],
            "audio": [{"index": idx, "name": name} for idx, name in d.get("audio", [])],
            "auto_screen": dev.find_screen_device(d),
            # cameras only (screens stripped), each with the OpenCV ordinal the
            # preview stream needs. The bar uses this instead of
            # enumerateDevices, which does not exist in WKWebView.
            "cameras": camera_preview.list_cameras(d),
        }

    def _new_windows(self, excluded, captured, baseline, raised=()):
        """Candidate windows to JOIN into a live fleet: on screen NOW, minus
        the ones already being recorded, minus the ones present at take start
        THAT THE USER NEVER BROUGHT UP. Same source + exclusions as the picker
        (`windows()`), so the app's own chrome (bar / picker / facecam, via pid
        + the title backstop) never appears. Capped to the 2 front-most
        (list_windows is front-to-back z-order, so ~most-recently-focused) to
        bound the pill width and the flood after a Space switch. App-name-only
        -- no title, no rect -- to keep continuous exposure on the
        high-frequency /api/state poll to the minimum a one-click chip needs.
        Runs OUTSIDE the lock (a Quartz sweep).

        `raised` (record.Recorder.raised_window_ids) is what makes the baseline
        a suppression of CLUTTER rather than a hard wall: the baseline exists
        so every window that merely happened to be open doesn't become a chip,
        but a window the user deliberately raised mid-take is exactly the one
        they want recorded -- and on macOS "opening Finder" or "opening a
        document" usually raises an existing window rather than creating one,
        so without this the flagship case was silently unreachable
        (docs/architecture.md M2.4c). Defaults to empty: a recorder that predates the
        property (or any caller that omits it) gets the old id-only rule,
        byte-identical.
        """
        out = []
        for w in dev.list_windows(exclude_pids=excluded):
            try:
                wid = int(w["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if wid in captured:
                continue
            if wid in baseline and wid not in raised:
                continue
            out.append({"id": w["id"], "app": w.get("app") or "Window"})
            if len(out) >= 2:
                break
        return out

    def windows(self):
        """Pickable on-screen windows for the record-time window picker.

        NOT to be confused with edits.json's `windows` (the render-time
        multi-window crop rects, in source-video pixels). These are live
        macOS windows in POINTS, and picking one only snapshots a rect that
        record.py stores in meta.json -- avfoundation can't target a window,
        so render.py crops the full-display capture to it later.

        Deliberately its own endpoint rather than a key in devices(): that
        payload is cached and polled by the bar, and a window list goes stale
        within seconds (any move, close, minimize or Space switch).

        `available` is False only when Quartz itself isn't there — an empty
        list with `available: true` genuinely means "nothing pickable right
        now". The UI should HIDE the picker in the former case instead of
        offering a dropdown that can never fill.
        """
        return {
            "windows": dev.list_windows(exclude_pids=self._excluded_pids()),
            "available": bool(dev.displays_points()),
        }

    def permissions(self):
        return perms.check_permissions()

    def media_path(self, session_name, kind):
        session_dir = self._session_dir(session_name)
        with open(os.path.join(session_dir, "meta.json")) as f:
            meta = json.load(f)
        if kind == "raw":
            path = os.path.join(session_dir, meta.get("raw", "raw.mov"))
        elif kind == "output":
            path = os.path.join(session_dir, "output.mp4")
        elif kind == "face":
            # The webcam track, so the editor's live composite can draw the
            # bubble instead of dropping it the moment you press play.
            path = os.path.join(session_dir, str(meta.get("face") or "face.mov"))
        elif kind.startswith("channel"):
            # One window's own buffer from a multi-native take. These sessions
            # have NO raw.mov at all, so this is the only way the editor can
            # play them -- it loads one <video> per channel and composites.
            try:
                idx = int(kind[len("channel"):])
            except ValueError:
                raise StudioError("bad channel: {}".format(kind), status=400)
            chans = meta.get("capture_channels") or []
            if not 0 <= idx < len(chans):
                raise StudioError("no channel {} in session".format(idx),
                                  status=404)
            path = os.path.join(session_dir, str(chans[idx].get("file")))
        elif kind.startswith("scene"):
            # scene{s}channel{c}: one channel of one scene of a scene take.
            # Bounds-checked against the manifest; no collision with the flat
            # legacy `channelN` (which multi-native owns).
            m = re.match(r"^scene(\d+)channel(\d+)$", kind)
            if not m:
                raise StudioError("bad scene media kind: {}".format(kind),
                                  status=400)
            s_i, c_i = int(m.group(1)), int(m.group(2))
            scenes = meta.get("capture_scenes") or []
            if not 0 <= s_i < len(scenes):
                raise StudioError("no scene {} in session".format(s_i),
                                  status=404)
            chans = scenes[s_i].get("channels") or []
            if not 0 <= c_i < len(chans):
                raise StudioError(
                    "no channel {} in scene {}".format(c_i, s_i), status=404)
            path = os.path.join(session_dir, str(chans[c_i].get("file")))
        else:
            raise StudioError("unknown media kind: {}".format(kind), status=400)
        if not os.path.isfile(path):
            raise StudioError("media file not found: {}".format(path), status=404)
        return path

    def reveal_session(self, session_name):
        """Open the session folder in the OS file browser (macOS `open`)."""
        session_dir = self._session_dir(session_name)
        if sys.platform != "darwin":
            raise StudioError(
                "revealing in a file browser is only supported on macOS",
                status=400)
        try:
            subprocess.Popen(["open", session_dir])
        except Exception as exc:
            raise StudioError("failed to open Finder: {}".format(exc), status=500)
        return {"opened": session_dir}

    def get_project_name(self, session_name):
        session_dir = self._session_dir(session_name)
        return load_project_name(session_dir)

    def set_project_name(self, session_name, name):
        session_dir = self._session_dir(session_name)
        return save_project_name(session_dir, name)

    def thumbnail(self, session_name):
        """JPEG thumbnail bytes, cached at <session>/thumb.jpg keyed on the
        recording's mtime."""
        session_dir = self._session_dir(session_name)
        try:
            with open(os.path.join(session_dir, "meta.json")) as f:
                meta = json.load(f)
        except Exception:
            meta = {}
        raw_path = os.path.join(session_dir, meta.get("raw", "raw.mov"))
        # The manifest block the thumbnail frame comes FROM, so an
        # occlusion-free card shows the same erased corner the export does.
        badge_spec = meta.get("capture_window")
        if not os.path.isfile(raw_path):
            # Manifest sessions have no raw.mov: fall back to the first
            # channel file (multi-native) or the first scene's first channel
            # (scene takes) so the library card gets a real thumbnail.
            fallback = None
            chans = meta.get("capture_channels") or []
            if chans:
                fallback, badge_spec = chans[0].get("file"), chans[0]
            else:
                scenes = meta.get("capture_scenes") or []
                if scenes and scenes[0].get("channels"):
                    ch0 = scenes[0]["channels"][0]
                    fallback, badge_spec = ch0.get("file"), ch0
            if fallback:
                raw_path = os.path.join(session_dir, str(fallback))
        if not os.path.isfile(raw_path):
            raise StudioError("recording not found for session: {}".format(
                session_name), status=404)
        cache_path = os.path.join(session_dir, _THUMB_FILENAME)
        try:
            raw_mtime = os.path.getmtime(raw_path)
        except OSError:
            raw_mtime = None
        if raw_mtime is not None and os.path.isfile(cache_path):
            try:
                if os.path.getmtime(cache_path) >= raw_mtime:
                    with open(cache_path, "rb") as f:
                        data = f.read()
                    if data:
                        return data
            except OSError:
                pass
        data = _thumbnail_jpeg(raw_path, badge_spec=badge_spec)
        try:
            with open(cache_path, "wb") as f:
                f.write(data)
        except OSError:
            pass
        return data

    def waveform(self, session_name):
        """{"peaks": [0..1 floats], "duration": s}; cached per raw mtime.

        Sessions without audio (or any ffmpeg failure) yield empty peaks
        rather than an error, so the timeline UI can always render.
        """
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        duration = float(info.get("duration", 0.0))
        if not info.get("has_audio"):
            return {"peaks": [], "duration": duration}
        raw_path = info.get("raw_path")
        if not raw_path:
            # A segmented take has no single raw (the manifest is the truth),
            # and joining the segments just to draw peaks is render-grade
            # work -- Phase 1 degrades to the no-audio shape so the timeline
            # still renders (docs/architecture.md: read-model breadth, pending).
            return {"peaks": [], "duration": duration}
        try:
            raw_mtime = os.path.getmtime(raw_path)
        except OSError:
            raw_mtime = None
        cache_path = os.path.join(session_dir, _WAVEFORM_FILENAME)
        if raw_mtime is not None:
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                if (isinstance(cached, dict)
                        and cached.get("mtime") == raw_mtime
                        and isinstance(cached.get("peaks"), list)):
                    return {
                        "peaks": cached["peaks"],
                        "duration": float(cached.get("duration", duration)),
                    }
            except Exception:
                pass
        try:
            proc = subprocess.run(
                ["ffmpeg", "-v", "error", "-i", raw_path, "-map", "a:0?",
                 "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            pcm = proc.stdout if proc.returncode == 0 else b""
        except Exception:
            pcm = b""
        peaks = waveform_peaks(pcm, buckets=_WAVEFORM_BUCKETS)
        if raw_mtime is not None:
            try:
                with open(cache_path, "w") as f:
                    json.dump({"mtime": raw_mtime, "peaks": peaks,
                               "duration": duration}, f)
            except OSError:
                pass
        return {"peaks": peaks, "duration": duration}

    def transcript(self, session_name):
        """The cached transcript + this session's transcribe-job state.

        NEVER runs ASR: the editor paints this on every session open, and a
        multi-minute decode on the paint path would hang the panel. A session
        with no cache comes back `status: "none"`, which is the UI's cue to
        offer the button rather than an error.
        """
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        doc = transcribe.load_transcript(session_dir)
        with self._lock:
            job = dict(self._transcribe_jobs.get(session_name) or
                       {"status": "idle", "message": ""})
        if doc is None:
            doc = {"status": "none", "reason": "", "words": [], "segments": [],
                   "language": "", "model": ""}
        # Each word gets a DERIVED `end` -- the removal-repaired end time
        # (transcribe.word_end_times). Derived on read and never written to
        # transcript.json: the file stays the ASR's own output, and the rule
        # can be corrected without invalidating anybody's cache.
        #
        # This exists so the editor's select-to-cut can turn a word range
        # into a time range as `words[first].t -> words[last].end`, carrying
        # no copy of the repair rule in JavaScript. A second implementation
        # in the browser is exactly the drift the cluster-contract invariant
        # warns about, and there is no JS harness here to pin it against.
        words = transcribe.words_with_ends(doc.get("words"))
        return {
            "session": session_name,
            "status": doc.get("status", "none"),
            "reason": doc.get("reason", ""),
            "language": doc.get("language", ""),
            "model": doc.get("model", ""),
            "duration": float(info.get("duration", 0.0)),
            "has_audio": bool(info.get("has_audio")),
            "words": words,
            "segments": doc.get("segments") or [],
            "cut_limits": {"max_ranges": edits.CUT_MAX_RANGES,
                           "min_kept_sec": edits.CUT_MIN_KEPT_SEC},
            "job": job,
            "asr": transcribe.availability(),
        }

    def start_transcribe(self, session_name, force=False):
        """Kick off (or re-report) the ASR pass for one session.

        Same shape as the render job: a daemon thread plus a status the client
        polls. One job per session, and asking twice while it runs is a
        no-op rather than a second decoder over the same audio.
        """
        session_dir = self._session_dir(session_name)
        with self._lock:
            job = self._transcribe_jobs.get(session_name)
            if job and job.get("status") == "running":
                return dict(job)
            self._transcribe_jobs[session_name] = {
                "status": "running", "message": "transcribing"}

        def worker():
            try:
                doc = transcribe.transcribe_session(session_dir, force=force)
                status = "done" if doc.get("status") == "ok" else "empty"
                message = doc.get("reason", "") or "transcript ready"
            except Exception as exc:          # pragma: no cover - defensive
                status, message = "error", str(exc)
            with self._lock:
                self._transcribe_jobs[session_name] = {
                    "status": status, "message": message}

        threading.Thread(target=worker, daemon=True).start()
        with self._lock:
            return dict(self._transcribe_jobs[session_name])

    def camera_path(self, session_name, options):
        """Sampled camera path (ren.camera_path) + the resolved trim window.

        Short-circuits to a windows_mode payload when the resolved edits have
        `windows` set -- there is no per-frame camera path to sample in that
        mode (static crops, see render.py's `use_multi`), and computing one
        anyway would be wasted work for a payload nothing downstream would use
        meaningfully. Instead that payload carries the composite's LAYOUT, so
        the editor can redraw the grid live during playback rather than
        falling back to the bare recording.
        """
        options = options or {}
        session_dir, info, resolved = self._resolved_edits(session_name, options)
        trim = resolved.get("trim", {})
        trim_start = _as_float(trim.get("start"), 0.0)
        trim_end = trim.get("end")
        if trim_end is None:
            trim_end = float(info.get("duration", 0.0))
        else:
            trim_end = _as_float(trim_end, info.get("duration", 0.0))
        kwargs = self._render_kwargs_from_edits(resolved)
        # Scene take: N per-scene fleets, no raw.mov, and a DIFFERENT window
        # set per scene, so the editor plays each scene's channels directly and
        # swaps the fleet at each seam (the S3 live player, docs/architecture.md).
        # Hand over the per-scene composite plan; canvas/cells live UNDER
        # scenes[] so the single-scene `multiReady()` gate is never armed, and
        # a preview nicety must never 500 -- on any failure we fall back to the
        # still-only stub the editor already degrades to (scene_preview_frame
        # stays the correctness floor).
        if info.get("scene_take"):
            payload = {"scene_take": True, "windows_mode": True, "path": [],
                       "trim": {"start": trim_start, "end": trim_end}}
            stride = max(1, min(10, _as_int(options.get("stride"), 2)))
            try:
                data = ren.scene_camera_path(
                    session_dir, background=kwargs["background"],
                    style=kwargs["style"], aspect=kwargs["aspect"],
                    window_layout=kwargs.get("window_layout", "grid"),
                    window_zoom=kwargs["window_zoom"],
                    window_focus=kwargs["window_focus"],
                    max_zoom=kwargs["max_zoom"], params=kwargs["camera_params"],
                    suppressed_ranges=kwargs["suppressed_ranges"],
                    focus_ranges=kwargs["focus_ranges"],
                    max_height=kwargs["max_height"], stride=stride,
                    scene_layouts=kwargs["scene_layouts"],
                    hidden_channels=kwargs["hidden_channels"],
                    badge_erase=kwargs["badge_erase"])
            except Exception:
                return payload
            payload.update(data)
            return payload
        # Multi-native: N per-window buffers and NO raw.mov, so `ren.camera_path`
        # has nothing to open. Hand over the composite layout instead and let the
        # editor play the channels directly -- without this the player asks for a
        # raw.mov that cannot exist and shows a blank canvas.
        if info.get("multi_native"):
            payload = {"multi_native": True, "windows_mode": True,
                       "trim": {"start": trim_start, "end": trim_end}}
            try:
                layout = ren.multi_native_layout(
                    session_dir, background=kwargs["background"],
                    style=kwargs["style"], aspect=kwargs["aspect"],
                    window_layout=kwargs.get("window_layout", "grid"),
                    max_height=kwargs["max_height"],
                    channel_layouts=kwargs["channel_layouts"],
                    badge_erase=kwargs["badge_erase"])
            except Exception:
                # A preview nicety must never take down the request; the editor
                # falls back to its "cannot play this take" state.
                return payload
            payload["canvas"] = list(layout["canvas"])
            payload["cells"] = layout["cells"]
            payload["channels"] = layout["channels"]
            payload["fps"] = layout["fps"]
            payload["duration"] = layout["duration"]
            payload["plate_jpeg_base64"] = base64.b64encode(
                ren.encode_preview_jpeg(layout["plate"], quality=90)
            ).decode("ascii")
            # Per-card zoom + window-focus in the LIVE preview, mirroring the
            # display-crop branch below. Without these the editor showed the
            # static arrangement while the export animated, the same paused-vs-
            # playing divergence the display-crop path was fixed for. The
            # emitters call the SAME planners the export does, so the browser
            # (which re-derives the crop/cell from these numbers) frames
            # identically. Each is a preview nicety -- never fail the request.
            stride = max(1, min(10, _as_int(options.get("stride"), 2)))
            try:
                payload["card_paths"] = ren.multi_native_card_paths(
                    session_dir, offset=kwargs["offset"],
                    window_zoom=kwargs["window_zoom"],
                    max_zoom=kwargs["max_zoom"], params=kwargs["camera_params"],
                    suppressed_ranges=kwargs["suppressed_ranges"],
                    window_follow=kwargs["window_follow"], stride=stride)
            except Exception:
                payload["card_paths"] = None
            try:
                payload["focus_cells"] = ren.multi_native_focus_cells(
                    session_dir, layout["painter"], offset=kwargs["offset"],
                    window_focus=kwargs["window_focus"],
                    max_zoom=kwargs["max_zoom"], params=kwargs["camera_params"],
                    suppressed_ranges=kwargs["suppressed_ranges"],
                    window_follow=kwargs["window_follow"],
                    focus_ranges=kwargs["focus_ranges"], stride=stride)
            except Exception:
                payload["focus_cells"] = None
            if payload.get("focus_cells"):
                # Moving cells make the baked plate shadows wrong, so the
                # browser needs the bare backdrop to draw its own over.
                payload["bg_jpeg_base64"] = base64.b64encode(
                    ren.encode_preview_jpeg(layout["background"], quality=90)
                ).decode("ascii")
            return payload
        if resolved.get("windows"):
            payload = {"windows_mode": True,
                       "trim": {"start": trim_start, "end": trim_end}}
            try:
                layout = ren.multi_window_layout(
                    session_dir, kwargs["windows"],
                    background=kwargs["background"], style=kwargs["style"],
                    aspect=kwargs["aspect"],
                    window_layout=kwargs.get("window_layout", "grid"),
                    crop_rect=kwargs["crop_rect"],
                    max_height=kwargs["max_height"])
            except Exception:
                # Layout is a preview nicety: without it the editor falls back
                # to the plain recording during playback, exactly as before.
                # Never let it take down the whole camera-path request.
                return payload
            payload["canvas"] = list(layout["canvas"])
            payload["cells"] = layout["cells"]
            payload["plate_jpeg_base64"] = base64.b64encode(
                ren.encode_preview_jpeg(layout["plate"], quality=90)
            ).decode("ascii")
            stride = max(1, min(10, _as_int(options.get("stride"), 2)))
            # Same posture as the layout above: a preview nicety must never
            # cost the caller the whole camera-path response.
            try:
                payload["cursor"] = ren.multi_window_cursor_track(
                    session_dir, offset=kwargs["offset"],
                    cursor_fx=kwargs["cursor_fx"],
                    cursor_params=kwargs["cursor_params"],
                    stride=stride)
            except Exception:
                payload["cursor"] = None
            try:
                payload["card_paths"] = ren.multi_window_card_paths(
                    session_dir, kwargs["windows"], layout["cells"],
                    offset=kwargs["offset"], window_zoom=kwargs["window_zoom"],
                    max_zoom=kwargs["max_zoom"], params=kwargs["camera_params"],
                    suppressed_ranges=kwargs["suppressed_ranges"],
                    window_follow=kwargs["window_follow"], stride=stride,
                    crop_rect=kwargs["crop_rect"])
            except Exception:
                payload["card_paths"] = None
            try:
                payload["focus_cells"] = ren.multi_window_focus_cells(
                    session_dir, kwargs["windows"], layout["painter"],
                    offset=kwargs["offset"],
                    window_focus=kwargs["window_focus"],
                    max_zoom=kwargs["max_zoom"],
                    params=kwargs["camera_params"],
                    suppressed_ranges=kwargs["suppressed_ranges"],
                    window_follow=kwargs["window_follow"],
                    focus_ranges=kwargs["focus_ranges"], stride=stride,
                    crop_rect=kwargs["crop_rect"])
            except Exception:
                payload["focus_cells"] = None
            if payload.get("focus_cells"):
                # Moving cells make the baked shadows wrong, so the browser
                # needs the bare backdrop to draw its own over.
                payload["bg_jpeg_base64"] = base64.b64encode(
                    ren.encode_preview_jpeg(layout["background"], quality=90)
                ).decode("ascii")
            return payload
        stride = max(1, min(10, _as_int(options.get("stride"), 2)))
        data = ren.camera_path(
            session_dir,
            t_start=_as_float(options.get("t_start"), None),
            t_end=_as_float(options.get("t_end"), None),
            stride=stride,
            max_zoom=kwargs["max_zoom"],
            offset=kwargs["offset"],
            style=kwargs["style"],
            params=kwargs["camera_params"],
            background=kwargs["background"],
            manual_zooms=kwargs["manual_zooms"],
            suppressed_ranges=kwargs["suppressed_ranges"],
            aspect=kwargs["aspect"],
            typing_zoom=kwargs["typing_zoom"],
            scroll_zoom=kwargs["scroll_zoom"],
            crop_rect=kwargs["crop_rect"],
            screen_focus=kwargs["screen_focus"],
        )
        data["trim"] = {"start": trim_start, "end": trim_end}
        return data

    def delete_session(self, session_name):
        """Move a session directory to the Trash (macOS Finder)."""
        session_dir = self._session_dir(session_name)
        if sys.platform != "darwin":
            raise StudioError(
                "deleting to Trash is only supported on macOS", status=400)
        with self._lock:
            rec_st = self._record_status
            if (rec_st.get("session") == session_name
                    and rec_st.get("status") in ("countdown", "recording",
                                                 "stopping", "pausing",
                                                 "paused", "resuming")):
                raise StudioError(
                    "a recording is running for this session", status=409)
            ren_st = self._render_status
            if (ren_st.get("session") == session_name
                    and ren_st.get("status") == "running"):
                raise StudioError(
                    "a render is running for this session", status=409)
        escaped = session_dir.replace("\\", "\\\\").replace('"', '\\"')
        script = 'tell application "Finder" to delete POSIX file "{}"'.format(escaped)
        try:
            proc = subprocess.run(["osascript", "-e", script],
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
        except Exception as exc:
            raise StudioError(
                "failed to move session to Trash: {}".format(exc), status=500)
        if proc.returncode != 0:
            err = (proc.stderr or b"").decode("utf-8", "replace").strip()
            raise StudioError(
                "failed to move session to Trash: {}".format(
                    err or "osascript exited {}".format(proc.returncode)),
                status=500)
        self._drop_describe_cache(session_dir)
        return {"deleted": session_name}

    def _resolved_edits(self, session_name, options):
        session_dir = self._session_dir(session_name)
        info = self._describe(session_dir)
        saved = edits.load_edits(session_dir, duration=info["duration"])
        patch = _patch_from_options(options)
        merged = edits.merge_edits(saved, patch, duration=info["duration"])
        return session_dir, info, merged

    def _render_kwargs_from_edits(self, edits_obj):
        render_opts = edits_obj.get("render", {})
        click_color = _clean_opt_str(render_opts.get("click_color"))
        zooms = edits_obj.get("zooms")
        suppressed = edits_obj.get("suppressed")
        focus = edits_obj.get("focus")
        speedups = edits_obj.get("speedups")
        cuts = edits_obj.get("cuts")
        windows = edits_obj.get("windows")
        channel_layouts = edits_obj.get("channel_layouts")
        scene_layouts = edits_obj.get("scene_layouts")
        hidden_channels = edits_obj.get("hidden_channels")
        # `crop` is top-level, not a render option: it is a spatial trim, the
        # sibling of `trim`, and lives in source px like `windows` and the
        # zoom pins. It still has to reach all four sinks below.
        crop_rect = edits.normalize_crop(edits_obj.get("crop"))
        return dict(
            max_zoom=_as_float(render_opts.get("zoom"), 2.0),
            offset=_as_float(render_opts.get("offset"), 0.0),
            style=str(render_opts.get("style", "clean")),
            make_gif=_as_bool(render_opts.get("gif"), False),
            background=_clean_opt_str(render_opts.get("background")),
            click_fx=_as_bool(render_opts.get("click_fx"), True),
            click_params=({"color": click_color} if click_color else None),
            spotlight=_as_bool(render_opts.get("spotlight"), False),
            cursor_fx=_as_bool(render_opts.get("cursor_fx"), False),
            cursor_params={"scale": _as_float(render_opts.get("cursor_size"), 1.0)},
            cursor_erase=_as_bool(render_opts.get("cursor_erase"), False),
            aspect=_clean_opt_str(render_opts.get("aspect")) or "auto",
            # Export-resolution preset -> the pixel-height cap the render
            # understands. None ("auto") is bit-exact with the natural canvas.
            max_height=edits.resolution_max_height(render_opts.get("resolution")),
            motion_blur=_as_bool(render_opts.get("motion_blur"), True),
            typing_zoom=_as_bool(render_opts.get("typing_zoom"), True),
            scroll_zoom=_as_bool(render_opts.get("scroll_zoom"), True),
            window_follow=_as_bool(render_opts.get("window_follow"), True),
            # Against edits._WINDOW_LAYOUTS rather than a second literal set:
            # the options dict from the editor is live (unsaved) and has NOT
            # been through normalize_edits, so this is a real gate, and a
            # hand-copied tuple here is how an arrangement the editor offers
            # ends up rendering as the grid.
            window_layout=(render_opts.get("window_layout")
                           if render_opts.get("window_layout") in edits._WINDOW_LAYOUTS
                           else "grid"),
            window_zoom=_as_bool(render_opts.get("window_zoom"), False),
            window_focus=_as_bool(render_opts.get("window_focus"), False),
            # Whole-screen grow-the-active-window; ON by default (see
            # edits._DEFAULT_RENDER["screen_focus"]).
            screen_focus=_as_bool(render_opts.get("screen_focus"), True),
            # macOS's per-window capture indicator, painted out of an
            # occlusion-free take; ON by default (edits._DEFAULT_RENDER).
            badge_erase=_as_bool(render_opts.get("badge_erase"), True),
            # None until the editor has materialized the plan: [] would
            # mean "the user deleted every arc" and suppress auto-planning.
            focus_ranges=(list(focus) if isinstance(focus, list)
                          and not edits.focus_plan_is_stale(edits_obj)
                          else None),
            facecam=_as_bool(render_opts.get("facecam"), True),
            facecam_params=_facecam_params_from_render_opts(render_opts),
            camera_params=_camera_params_from_render_opts(render_opts),
            fade=max(0.0, _as_float(render_opts.get("fade"), 0.0)),
            music=_clean_opt_str(render_opts.get("music")),
            click_sound=_clean_opt_str(render_opts.get("click_sound")),
            key_sound=_clean_opt_str(render_opts.get("key_sound")),
            sfx_volume=_as_float(render_opts.get("sfx_volume"), 1.0),
            gif_fps=_as_int(render_opts.get("gif_fps"), 15),
            gif_width=_as_int(render_opts.get("gif_width"), 1000),
            manual_zooms=list(zooms) if isinstance(zooms, list) else [],
            suppressed_ranges=list(suppressed) if isinstance(suppressed, list) else [],
            speedup=_as_bool(render_opts.get("speedup"), False),
            speedup_rate=_as_float(render_opts.get("speedup_rate"), 6.0),
            speedup_silence_gate=_as_bool(
                render_opts.get("speedup_silence_gate"), True),
            speedup_motion_gate=_as_bool(
                render_opts.get("speedup_motion_gate"), True),
            speedups=list(speedups) if isinstance(speedups, list) else [],
            cuts=list(cuts) if isinstance(cuts, list) else [],
            windows=list(windows) if isinstance(windows, list) else [],
            channel_layouts=(list(channel_layouts)
                             if isinstance(channel_layouts, list) else []),
            scene_layouts=(dict(scene_layouts)
                           if isinstance(scene_layouts, dict) else {}),
            hidden_channels=(list(hidden_channels)
                             if isinstance(hidden_channels, list) else []),
            crop_rect=crop_rect,
        )

    def start_render(self, session_name, options):
        options = options or {}
        session_dir, _, resolved = self._resolved_edits(session_name, options)
        # The browser never chooses an output sink.  Passing `options.out`
        # through to ffmpeg accepted both arbitrary writable paths and URL
        # outputs, which turned a compromised/rebound local page into a way to
        # stream a private take elsewhere.  MCP/CLI have their own explicit
        # output contracts; the web Export action always writes beside the
        # project under this fixed name.
        out_path = os.path.join(session_dir, "output.mp4")
        kwargs = self._render_kwargs_from_edits(resolved)
        trim = resolved.get("trim", {})
        trim_start = _as_float(trim.get("start"), 0.0)
        trim_end = trim.get("end")
        with self._lock:
            if self._render_status.get("status") == "running":
                raise StudioError("a render is already running", status=409)
            self._render_status = {
                "status": "running",
                "message": "rendering...",
                "session": session_name,
                "out_path": out_path,
            }

        def worker():
            try:
                final_out = ren.render(
                    session_dir,
                    out_path=out_path,
                    max_zoom=kwargs["max_zoom"],
                    offset=kwargs["offset"],
                    style=kwargs["style"],
                    make_gif=kwargs["make_gif"],
                    background=kwargs["background"],
                    click_fx=kwargs["click_fx"],
                    click_params=kwargs["click_params"],
                    spotlight=kwargs["spotlight"],
                    cursor_fx=kwargs["cursor_fx"],
                    cursor_params=kwargs["cursor_params"],
                    aspect=kwargs["aspect"],
                    motion_blur=kwargs["motion_blur"],
                    typing_zoom=kwargs["typing_zoom"],
                    scroll_zoom=kwargs["scroll_zoom"],
                    facecam=kwargs["facecam"],
                    facecam_params=kwargs["facecam_params"],
                    params=kwargs["camera_params"],
                    fade=kwargs["fade"],
                    music=kwargs["music"],
                    click_sound=kwargs["click_sound"],
                    key_sound=kwargs["key_sound"],
                    sfx_volume=kwargs["sfx_volume"],
                    gif_fps=kwargs["gif_fps"],
                    gif_width=kwargs["gif_width"],
                    trim_start=trim_start,
                    trim_end=(None if trim_end is None else float(trim_end)),
                    manual_zooms=kwargs["manual_zooms"],
                    suppressed_ranges=kwargs["suppressed_ranges"],
                    speedup=kwargs["speedup"],
                    speedup_rate=kwargs["speedup_rate"],
                    speedup_silence_gate=kwargs["speedup_silence_gate"],
                    speedup_motion_gate=kwargs["speedup_motion_gate"],
                    speedups=kwargs["speedups"],
                    cuts=kwargs["cuts"],
                    windows=kwargs["windows"],
                    window_follow=kwargs["window_follow"],
                    window_layout=kwargs["window_layout"],
                    window_zoom=kwargs["window_zoom"],
                    window_focus=kwargs["window_focus"],
                    focus_ranges=kwargs["focus_ranges"],
                    crop_rect=kwargs["crop_rect"],
                    cursor_erase=kwargs["cursor_erase"],
                    badge_erase=kwargs["badge_erase"],
                    max_height=kwargs["max_height"],
                    channel_layouts=kwargs["channel_layouts"],
                    scene_layouts=kwargs["scene_layouts"],
                    hidden_channels=kwargs["hidden_channels"],
                )
                with self._lock:
                    self._render_status = {
                        "status": "done",
                        "message": "render complete",
                        "session": session_name,
                        "out_path": final_out,
                    }
            except Exception as exc:
                with self._lock:
                    self._render_status = {
                        "status": "error",
                        "message": str(exc),
                        "session": session_name,
                        "out_path": out_path,
                    }

        th = threading.Thread(target=worker, daemon=True)
        with self._lock:
            self._render_thread = th
        th.start()
        return dict(self._render_status)

    def preview(self, session_name, t_sec, options):
        options = options or {}
        session_dir, info, resolved = self._resolved_edits(session_name, options)
        kwargs = self._render_kwargs_from_edits(resolved)
        if info.get("scene_take"):
            # Scene take (occlusion-free, window set changes mid-recording):
            # composite one frame off the SAME per-scene planner Export runs,
            # so the scrub JPEG is bit-identical to the exported frame. Trim
            # is ignored on this export path (v1), so the preview clamps to
            # the full joined timeline rather than a trim that never lands.
            duration = float(info.get("duration", 0.0))
            t = max(0.0, min(_as_float(t_sec, 0.0), duration))
            frame = ren.scene_preview_frame(
                session_dir, t_sec=t, max_zoom=kwargs["max_zoom"],
                style=kwargs["style"], background=kwargs["background"],
                aspect=kwargs["aspect"], window_layout=kwargs["window_layout"],
                window_zoom=kwargs["window_zoom"],
                window_focus=kwargs["window_focus"],
                cursor_fx=kwargs["cursor_fx"],
                cursor_params=kwargs["cursor_params"],
                params=kwargs["camera_params"],
                suppressed_ranges=kwargs["suppressed_ranges"],
                focus_ranges=kwargs["focus_ranges"],
                max_height=kwargs["max_height"],
                scene_layouts=kwargs["scene_layouts"],
                hidden_channels=kwargs["hidden_channels"],
                badge_erase=kwargs["badge_erase"])
            img = ren.encode_preview_jpeg(frame, quality=88)
            h, w = frame.shape[:2]
            return {
                "width": int(w),
                "height": int(h),
                "time": float(t),
                "trim": {"start": 0.0, "end": duration},
                "image_jpeg_base64": base64.b64encode(img).decode("ascii"),
            }
        trim = resolved.get("trim", {})
        trim_start = _as_float(trim.get("start"), 0.0)
        trim_end = trim.get("end")
        if trim_end is None:
            trim_end = float(info.get("duration", 0.0))
        else:
            trim_end = _as_float(trim_end, info.get("duration", 0.0))
        t = _as_float(t_sec, trim_start)
        t = max(trim_start, min(t, trim_end))
        frame = ren.preview_frame(
            session_dir,
            t_sec=t,
            max_zoom=kwargs["max_zoom"],
            offset=kwargs["offset"],
            style=kwargs["style"],
            background=kwargs["background"],
            click_fx=kwargs["click_fx"],
            click_params=kwargs["click_params"],
            spotlight=kwargs["spotlight"],
            cursor_fx=kwargs["cursor_fx"],
            cursor_params=kwargs["cursor_params"],
            aspect=kwargs["aspect"],
            motion_blur=kwargs["motion_blur"],
            typing_zoom=kwargs["typing_zoom"],
            scroll_zoom=kwargs["scroll_zoom"],
            facecam=kwargs["facecam"],
            facecam_params=kwargs["facecam_params"],
            params=kwargs["camera_params"],
            fade=kwargs["fade"],
            manual_zooms=kwargs["manual_zooms"],
            suppressed_ranges=kwargs["suppressed_ranges"],
            windows=kwargs["windows"],
            window_follow=kwargs["window_follow"],
            window_layout=kwargs["window_layout"],
            window_zoom=kwargs["window_zoom"],
            window_focus=kwargs["window_focus"],
            focus_ranges=kwargs["focus_ranges"],
            crop_rect=kwargs["crop_rect"],
            cursor_erase=kwargs["cursor_erase"],
            screen_focus=kwargs["screen_focus"],
            badge_erase=kwargs["badge_erase"],
            max_height=kwargs["max_height"],
        )
        h, w = frame.shape[:2]
        img = ren.encode_preview_jpeg(frame, quality=88)
        return {
            "width": int(w),
            "height": int(h),
            "time": float(t),
            "trim": {"start": trim_start, "end": trim_end},
            "image_jpeg_base64": base64.b64encode(img).decode("ascii"),
        }

    def _resolve_capture_window(self, window_id):
        """Fresh rect for the picked window, or None for a full-screen take.

        Raises StudioError(400) when a window WAS picked but no longer
        resolves — gone, minimized, moved offscreen, shrunk below the size
        gates, or dragged to a secondary display (multi-display window
        capture is out of scope for v1 and must fail safe rather than crop
        the wrong pixels). Never falls back to full screen.
        """
        if window_id is None or str(window_id).strip() == "":
            return None
        entry = dev.window_rect_points(_as_int(window_id, -1),
                                       exclude_pids=self._excluded_pids())
        if entry is None:
            raise StudioError(
                "that window is no longer on screen — pick it again, or "
                "choose Full screen", status=400)
        return entry

    def _resolve_capture_windows(self, window_ids):
        """Fresh rects for a multi-window pick: `(entries, dropped_labels)`.

        Deliberately NOT the single-window contract. There, a stale id is a
        400 because the whole take was about that one window. Here, closing
        one of three picked windows between picking and hitting record should
        not throw the other two away -- so a window that no longer resolves is
        DROPPED and named in the response, and only an empty result is an
        error. Falling to one survivor hands back a single entry, which the
        Recorder then treats as an ordinary window crop.
        """
        if not isinstance(window_ids, (list, tuple)) or not window_ids:
            return [], []
        seen = set()
        entries, dropped = [], []
        excluded = self._excluded_pids()
        for raw in list(window_ids)[:edits._MAX_WINDOWS]:
            wid = _as_int(raw, -1)
            if wid < 0 or wid in seen:
                continue
            seen.add(wid)
            entry = dev.window_rect_points(wid, exclude_pids=excluded)
            if entry is None:
                dropped.append(str(wid))
                continue
            entries.append(entry)
        if not entries:
            raise StudioError(
                "none of those windows are on screen any more — pick them "
                "again, or choose Full screen", status=400)
        return entries, dropped

    def _arrange_windows(self, entries, options):
        """Un-overlap the picked windows before the take: `(state, token)`.

        Opt-in (`options.arrange`), because moving someone's windows is a
        visible thing to do to their machine and must be their call. It is
        also the only way a multi-window composition can be CORRECT when the
        picks overlap — see autocine/arrange.py — so the picker asks for it
        rather than the server assuming either way.

        Every failure lands on "record anyway": the take is worth more than
        the tidy-up, and `occluded_sec` in describe_session will say what it
        cost.
        """
        if len(entries or []) < 2:
            return "none", None
        if not _as_bool(options.get("arrange"), False):
            rects = [[e["x"], e["y"], e["w"], e["h"]] for e in entries]
            return ("declined" if arrange.any_overlap(rects) else "none"), None
        try:
            dw, dh, _src = dev.main_display_points()
            rects = [[e["x"], e["y"], e["w"], e["h"]] for e in entries]
            targets = arrange.plan_separation(rects, (0.0, 0.0, dw, dh))
            if targets is None:
                # Either nothing overlapped or nothing fits — tell those apart
                # so the editor doesn't warn about a take that was always fine.
                return ("none" if not arrange.any_overlap(rects)
                        else "failed"), None
            moved, token = arrange.apply_arrangement(entries, targets)
            if not moved:
                return "failed", None
            # Re-read: the windows are somewhere new, and the rects that go
            # into meta.json have to describe where they ACTUALLY are, not
            # where we asked them to be.
            excluded = self._excluded_pids()
            for i, entry in enumerate(entries):
                fresh = dev.window_rect_points(entry["id"],
                                               exclude_pids=excluded)
                if fresh is not None:
                    entries[i] = fresh
            return ("moved" if moved == len(entries) else "failed"), token
        except Exception:
            return "failed", None

    def start_record(self, options):
        # The camera is NOT suspended any more when the facecam can ride the
        # preview capture: one process reads the device and tees frames into
        # face.mov, so the bubble keeps updating through the take. Suspending
        # only still happens on the fallback path, where the recorder opens
        # its own avfoundation capture and macOS would refuse the second
        # opener. _start_record decides which, since it is what knows whether
        # a camera was picked at all.
        try:
            return self._start_record(options or {})
        except Exception:
            camera_preview.resume()
            raise

    def _start_record(self, options):
        options = options or {}
        report = self.permissions()
        missing = report.get("missing_required", [])
        if missing:
            labels = [report["checks"][k]["label"] for k in missing]
            raise StudioError(
                "recording blocked: missing permissions ({})".format(
                    ", ".join(labels)),
                status=400,
            )
        d = dev.list_avf_devices()
        display = options.get("display")
        if display is None or str(display).strip() == "":
            display = dev.find_screen_device(d)
            if display is None:
                raise StudioError("could not auto-detect screen device", status=400)
        else:
            display = _as_int(display, 0)
            # An explicitly-picked index is a number the CLIENT chose, and the
            # bar builds its picker once in loadDevices() and then holds those
            # raw avfoundation indices for as long as its window lives. Those
            # indices shift underneath it whenever a video device comes or goes
            # (Continuity Camera, a virtual cam), so a long-lived bar can hand
            # back an index that now names the WEBCAM -- and we would hand it
            # straight to ffmpeg and record the user's face. Re-check it
            # against the listing we just read; a stale pick falls back to
            # auto-detect, because recording the wrong SCREEN is a recoverable
            # annoyance and recording a camera is not.
            if not dev.is_screen_device(d, display):
                display = dev.find_screen_device(d)
                if display is None:
                    raise StudioError("could not auto-detect screen device",
                                      status=400)
        mic = options.get("mic")
        if mic is None or str(mic).strip().lower() in ("", "none"):
            mic = None
        else:
            mic = _as_int(mic, 0)
        fps = max(1, _as_int(options.get("fps"), 60))
        countdown = max(0, _as_int(options.get("countdown"), 3))
        duration = options.get("duration")
        duration = None if duration in (None, "", 0, "0") else max(0.1, _as_float(duration, 20.0))
        cursor_mode = options.get("cursor")
        cursor_mode = cursor_mode if cursor_mode in rec.CURSOR_MODES else "system"
        log_keys = _as_bool(options.get("log_keys"), True)
        # Facecam: the bar sends face=true when a webcam is picked (its own
        # getUserMedia preview uses a browser deviceId that doesn't map to an
        # avfoundation index, so capture just uses the default camera).
        face_idx = None
        face_shared = None
        if _as_bool(options.get("face"), False):
            face_idx = dev.find_camera_device(d)
            # The OpenCV ordinal the bar is previewing. When we have it, the
            # facecam rides that capture instead of opening its own -- which
            # is the whole reason the preview no longer goes dark mid-take.
            face_shared = _as_int(options.get("face_ordinal"), -1)
            if face_shared < 0:
                face_shared = None
            if face_shared is None:
                # No preview to ride: fall back to the second avfoundation
                # capture, and that one DOES need the device to itself.
                camera_preview.suspend()
        # Record-time window capture. avfoundation can't target a window, so
        # we snapshot the chosen window's rect (points, global top-left) here
        # and render.py crops the full-display capture to it later. Resolved
        # BEFORE the session exists so a doomed take never starts, and a
        # window that can't be resolved is a hard error: the user asked for a
        # window, and silently handing them a full-screen take they didn't
        # want is a wasted recording.
        capture_window = self._resolve_capture_window(options.get("window_id"))
        capture_windows, dropped = self._resolve_capture_windows(
            options.get("window_ids"))
        if capture_windows:
            # A multi pick supersedes any stale single `window_id` the client
            # left in the payload -- one of the two has to win, and the newer
            # control is the one the user just used.
            capture_window = None
        # Occlusion-free: capture each picked window's OWN buffer via SCK, so
        # nothing in front of a target appears in that channel -- as opposed
        # to the display-crop above. 1 window is single-file native; 2-4 is
        # P3.1 multi-native (one raw_i.mov per window + a `capture_channels`
        # manifest). Validate loudly rather than silently downgrade to a
        # crop that would still show occluders. The bar mirrors the same
        # 1-4 range, but the server is the authority.
        window_native = _as_bool(options.get("occlusion_free"), False)
        if window_native:
            if not capture_window and not capture_windows:
                raise StudioError(
                    "occlusion-free needs at least one window selected",
                    status=400)
        # Arrange physically moves the user's real windows. That's meaningless
        # for a native take (the composite doesn't care about the on-screen
        # arrangement -- each raw_i.mov IS a window's own buffer), and MOVING
        # THE USER'S WINDOWS FOR ZERO BENEFIT is exactly what the doc says
        # never to do (docs/architecture.md, "Arrange must be
        # disabled in native mode"). Skip the helper entirely -- it never runs,
        # never even measures overlap ("declined"), and the state is an honest
        # "none" (not applicable to this mode).
        if window_native:
            arranged = ("none", None)
        else:
            arranged = self._arrange_windows(capture_windows, options)

        session_name = datetime.now().strftime("%Y%m%d-%H%M%S")
        session_dir = os.path.join(self.recordings_root, session_name)
        # Keeping our own chrome out of the take. Resolved HERE, at start,
        # rather than inside the Recorder: only this process knows which
        # windows belong to the app (the bar reports its own, and we sweep
        # our pids), and the ids must be read while those windows are on
        # screen. Ignored by the avfoundation backend, which has no window
        # channel at all -- that is the whole reason the SCK path exists.
        backend = sck.resolve_backend(options.get("capture_backend"),
                                      os.environ,
                                      settings.get("capture_backend"))
        if window_native:
            # Occlusion-free is SCK-only; avfoundation cannot capture a window's
            # own buffer. This wins over the picker/env for THIS take.
            backend = "sck"
        exclude = self.capture_exclude_ids() if backend == "sck" else []
        recorder = rec.Recorder(session_dir, display, mic_idx=mic, fps=fps,
                                cursor_mode=cursor_mode, log_keys=log_keys,
                                face_idx=face_idx,
                                capture_window=capture_window,
                                capture_windows=capture_windows or None,
                                backend=backend, exclude_windows=exclude,
                                window_native=window_native,
                                # Re-read during the take as a safety net for
                                # windows that appear later and that nobody
                                # reported -- a tooltip, a WebKit service
                                # window. NOT for the bubble or picker: both
                                # exist with stable ids before the take and
                                # cannot be opened from the recording face.
                                # See Recorder._poll_exclusions.
                                exclude_provider=(self.capture_exclude_ids
                                                  if backend == "sck" else None))
        recorder.face_shared_ordinal = face_shared
        recorder._arrange_state = arranged[0]
        recorder._arrange_restore = arranged[1]

        with self._lock:
            if self._record_status.get("status") in (
                    "countdown", "recording", "stopping",
                    "pausing", "paused", "resuming"):
                raise StudioError("a recording is already running", status=409)
            self._active_recorder = recorder
            self._record_status = {
                "status": "countdown" if countdown > 0 else "recording",
                "message": "recording is starting...",
                "session": session_name,
                "display": display,
                "mic": mic,
                "fps": fps,
                # Server truth for the bar's pause button. `pause_supported`
                # is the authority (whole-screen + occlusion-free fleet
                # takes pause; display-crop captures don't); a bar that did
                # NOT start this take (reloaded, or a second client) has no
                # local picker state to gate on. `window_capture` is kept
                # for older clients and for the paused-face re-pick
                # affordance (it says "this take has windows to re-pick").
                "window_capture": bool(recorder.capture_window
                                       or recorder.capture_windows),
                "pause_supported": bool(recorder.pause_supported),
            }
            # Baseline for the seamless-join candidate list: the windows
            # already on screen at start. Only a fleet take can grow, so only
            # it pays the one extra list_windows sweep; every other take
            # leaves the baseline None and the /api/state payload unchanged.
            if getattr(recorder, "_is_multi_window_native", lambda: False)():
                try:
                    self._grow_baseline_ids = {
                        int(w["id"]) for w in dev.list_windows(
                            exclude_pids=self._excluded_pids())}
                except Exception:
                    self._grow_baseline_ids = set()
            else:
                self._grow_baseline_ids = None

        def on_record_state(state_name, payload):
            payload = payload or {}
            with self._lock:
                if self._record_status.get("session") != session_name:
                    return
                if state_name == "countdown":
                    secs = payload.get("seconds")
                    self._record_status["status"] = "countdown"
                    if secs is None:
                        self._record_status["message"] = "recording countdown..."
                    else:
                        self._record_status["message"] = "recording in {}...".format(secs)
                elif state_name == "recording":
                    self._record_status["status"] = "recording"
                    self._record_status["message"] = "recording..."
                    # wall-clock start so any client (even one opened later)
                    # can show the true elapsed time of a running take
                    self._record_status.setdefault("started_wall", time.time())
                    # Back from a pause: slide started_wall forward by the
                    # paused span, so `now - started_wall` stays CONTENT time
                    # (the paused gap is deleted from the output; the timer
                    # must not count it either).
                    paused_wall = self._record_status.pop("paused_wall", None)
                    if paused_wall is not None:
                        self._record_status["started_wall"] += max(
                            0.0, time.time() - paused_wall)
                    # Fleet membership outcomes (docs/architecture.md milestones 2
                    # + 3): `_try_grow` emits EXACTLY ONE of {grow_error} or
                    # {joined, windows} per join; `_try_shrink` emits
                    # {departed, departed_app, windows} per card exit; a
                    # plain start/resume `recording` notify carries none.
                    # The keys stay MUTUALLY EXCLUSIVE so a stale `joined`
                    # can never mask a later `grow_error` -- and a fast
                    # restore's `joined` SUPERSEDING an unread `departed` is
                    # deliberate ("came back" beats "removed"). A plain
                    # notify clears them all. `mic_anchor_hidden` (decision
                    # 6's freeze hint) rides independently: set when
                    # carried, cleared by any plain notify.
                    ge = payload.get("grow_error")
                    j = payload.get("joined")
                    d = payload.get("departed")
                    mic_note = payload.get("mic_anchor_hidden")
                    if mic_note:
                        self._record_status["mic_anchor_hidden"] = mic_note
                    if payload.get("mic_anchor_seen"):
                        # The anchor is back on screen: retire the hint (a
                        # reloaded bar must not keep "keep it visible" for
                        # the rest of the take; its absence also resets the
                        # bar's dedupe key so the NEXT episode surfaces).
                        self._record_status.pop("mic_anchor_hidden", None)
                    if ge:
                        self._record_status["grow_error"] = ge
                        self._record_status.pop("joined", None)
                        self._record_status.pop("windows", None)
                        self._record_status.pop("departed", None)
                        self._record_status.pop("departed_app", None)
                    elif j is not None:
                        self._record_status["joined"] = j
                        w = payload.get("windows")
                        if w is not None:
                            self._record_status["windows"] = w
                        self._record_status.pop("grow_error", None)
                        self._record_status.pop("departed", None)
                        self._record_status.pop("departed_app", None)
                    elif d is not None:
                        self._record_status["departed"] = d
                        da = payload.get("departed_app")
                        if da:
                            self._record_status["departed_app"] = da
                        w = payload.get("windows")
                        if w is not None:
                            self._record_status["windows"] = w
                        self._record_status.pop("grow_error", None)
                        self._record_status.pop("joined", None)
                        # A from-start window that departed must be able to
                        # come back as an ordinary "+ App" chip once its
                        # auto-rejoin gives up -- without this, the baseline
                        # subtraction hides it from `new_windows` forever.
                        rec_obj = self._active_recorder
                        if self._grow_baseline_ids and rec_obj is not None:
                            try:
                                self._grow_baseline_ids -= set(
                                    rec_obj.departed_window_ids)
                            except Exception:
                                pass
                    elif payload.get("windows") is not None:
                        # A windows-carrying notify with no outcome key (a
                        # gate-failed shrink): store the fresh count, touch
                        # nothing else. Reading it as "plain" wiped a
                        # standing mic hint / unread departed mid-take
                        # (adversarially caught) -- and the recorder never
                        # re-notes the same hide episode, so the wipe was
                        # permanent.
                        self._record_status["windows"] = payload["windows"]
                    elif not mic_note and not payload.get("mic_anchor_seen"):
                        # A truly PLAIN recording notify (start / resume):
                        # clear everything.
                        self._record_status.pop("grow_error", None)
                        self._record_status.pop("joined", None)
                        self._record_status.pop("windows", None)
                        self._record_status.pop("departed", None)
                        self._record_status.pop("departed_app", None)
                        self._record_status.pop("mic_anchor_hidden", None)
                elif state_name == "stopping":
                    self._record_status["status"] = "stopping"
                    self._record_status["message"] = "stopping recording..."
                # Segmented-take pause/resume (see docs/architecture.md).
                # `paused` stamps paused_wall; the resume-side `recording`
                # branch above then shifts started_wall forward by the gap
                # (see below), so every client's `now - started_wall` elapsed
                # math keeps reporting CONTENT time, not wall time, with no
                # client-side change.
                elif state_name == "pausing":
                    self._record_status["status"] = "pausing"
                    self._record_status["message"] = "pausing recording..."
                elif state_name == "paused":
                    self._record_status["status"] = "paused"
                    self._record_status["segments"] = payload.get("segments")
                    # A FAILED resume re-enters `paused` with an `error`
                    # payload (scene takes: the picked window closed during
                    # spin-up). Surface it -- dropping it left the bar
                    # flipping "Resuming…" -> "Paused" with no explanation.
                    err = payload.get("error")
                    if err:
                        self._record_status["message"] = err
                        self._record_status["resume_error"] = err
                    else:
                        self._record_status["message"] = "recording paused"
                        self._record_status.pop("resume_error", None)
                    # setdefault, NOT overwrite: a failed resume re-notifies
                    # `paused` while the original stamp is still standing
                    # (the `recording` branch pops it only on a real
                    # resume), and re-stamping would drop the failed
                    # attempt's span from the started_wall slide -- the
                    # bar's content timer would run ahead of the recorder's
                    # own paused-time accounting for the rest of the take.
                    self._record_status.setdefault("paused_wall", time.time())
                elif state_name == "resuming":
                    self._record_status["status"] = "resuming"
                    self._record_status["message"] = "resuming recording..."

        def worker():
            try:
                session = recorder.start(
                    countdown=countdown,
                    duration=duration,
                    status_cb=on_record_state,
                )
                if session is None:
                    status = {
                        "status": "idle",
                        "message": "recording cancelled",
                        "session": session_name,
                    }
                else:
                    status = {
                        "status": "done",
                        "message": "recording complete",
                        "session": session_name,
                        "session_dir": session,
                    }
            except Exception as exc:
                status = {
                    "status": "error",
                    "message": str(exc),
                    "session": session_name,
                }
                # Say so ON THE CONSOLE too, not just in the HTTP status.
                # `Recorder.start()` has already printed "recording — press
                # Ctrl+C to stop" by this point, so without this the terminal's
                # last word on a FAILED take is that it is recording -- which
                # is exactly how a dead take gets mistaken for a live one
                # (reported verbatim: "not sure if it's still supposed to be
                # recording"). The full diagnosis goes out because it is the
                # actionable part (the avfoundation permission fingerprint,
                # etc), same as the `studio record` CLI path prints.
                print("\nrecording failed:\n{}".format(exc), file=sys.stderr)
            finally:
                # the recorder is done with the webcam — let previews reopen it
                camera_preview.resume()
                with self._lock:
                    self._active_recorder = None
                    self._grow_baseline_ids = None
                    self._record_status = status

        th = threading.Thread(target=worker, daemon=True)
        with self._lock:
            self._record_thread = th
        th.start()
        return dict(self._record_status)

    def stop_record(self):
        with self._lock:
            recorder = self._active_recorder
            st = self._record_status.get("status")
        # A PAUSED take must still be stoppable -- the run loop handles
        # stop-while-paused (the segments already on disk finalize as the take).
        if recorder is None or st not in ("countdown", "recording", "stopping",
                                          "pausing", "paused", "resuming"):
            raise StudioError("no active recording to stop", status=409)
        recorder.stop()
        with self._lock:
            self._record_status["status"] = "stopping"
            self._record_status["message"] = "stopping recording..."
            return dict(self._record_status)

    def pause_record(self):
        """Pause the running take at a segment boundary (segmented takes).
        Valid only from `recording`; the recorder's run loop finalizes the
        active segment and flips the status to `paused` via its status_cb."""
        with self._lock:
            recorder = self._active_recorder
            st = self._record_status.get("status")
        if recorder is None or st != "recording":
            raise StudioError("no active recording to pause", status=409)
        # Pause is supported on whole-screen takes (P1 segmented) and on
        # occlusion-free fleet takes (scene takes). A DISPLAY-CROP window
        # capture is refused OUTRIGHT: its pause would run the P1 machinery,
        # whose manifest carries no capture_window(s) block -- the take's
        # coordinate space would silently vanish and the export would come
        # out full-screen. The recorder is the authority
        # (`Recorder.pause_supported`); the attribute fallback keeps older
        # callers/stubs on the previous whole-screen-only contract.
        supported = getattr(recorder, "pause_supported", None)
        if supported is None:
            supported = not (recorder.capture_window
                             or recorder.capture_windows)
        if not supported:
            raise StudioError(
                "pause is not supported on this take (display-crop window "
                "captures cannot pause; occlusion-free takes can)",
                status=400)
        recorder.pause()
        with self._lock:
            self._record_status["status"] = "pausing"
            self._record_status["message"] = "pausing recording..."
            return dict(self._record_status)

    def _resolve_scene_windows(self, window_ids):
        """Fresh rects for a mid-take RE-PICK: ALL-OR-400, never
        drop-and-degrade. `_resolve_capture_windows`' lenient contract is
        right at record START (losing one of three picked windows shouldn't
        throw the others away) and wrong at a resume: a silently shrunk
        window set betrays the intent the user just expressed on the paused
        face, and the 1-survivor crop fallback isn't even a fleet scene.
        A refused pick leaves the take `paused` -- it is never a failure."""
        if not isinstance(window_ids, (list, tuple)) or not window_ids:
            raise StudioError("re-pick needs 1-4 window ids", status=400)
        ids, seen, bad = [], set(), []
        for raw in window_ids:
            wid = _as_int(raw, -1)
            if wid < 0:
                # Strict means STRICT: an unparseable or negative id 400s
                # instead of silently shrinking the set (the exact
                # drop-and-degrade this resolver exists to forbid).
                # Deduping valid repeats below is fine.
                bad.append(repr(raw))
                continue
            if wid in seen:
                continue
            seen.add(wid)
            ids.append(wid)
        if bad:
            raise StudioError(
                "re-pick has invalid window id(s): {} — the take is still "
                "paused; pick again".format(", ".join(bad)), status=400)
        if not 1 <= len(ids) <= edits._MAX_WINDOWS:
            raise StudioError("re-pick needs 1-4 windows", status=400)
        excluded = self._excluded_pids()
        entries, missing = [], []
        for wid in ids:
            entry = dev.window_rect_points(wid, exclude_pids=excluded)
            if entry is None:
                missing.append(str(wid))
            else:
                entries.append(entry)
        if missing:
            raise StudioError(
                "window(s) {} are no longer on screen — the take is still "
                "paused; pick again".format(", ".join(missing)), status=400)
        return entries

    def resume_record(self, options=None):
        """Resume a paused take: the recorder spawns the next segment/scene.
        Valid only from `paused` (a `pausing` take hasn't finalized its
        segment yet -- retry once it lands).

        Optional body `{"window_ids": [...]}` RE-PICKS the window set for
        the next scene (scene takes; occlusion-free fleet only), validated
        by the strict resolver above. No body = same-config resume."""
        options = options or {}
        with self._lock:
            recorder = self._active_recorder
            st = self._record_status.get("status")
        if recorder is None or st != "paused":
            raise StudioError("no paused recording to resume", status=409)
        window_ids = options.get("window_ids")
        if window_ids:
            is_fleet = getattr(recorder, "_is_multi_window_native",
                               lambda: False)
            if not is_fleet():
                raise StudioError(
                    "re-picking windows is only supported on occlusion-free "
                    "takes", status=400)
            scene = self._resolve_scene_windows(window_ids)
            recorder.resume(scene)
        else:
            recorder.resume()
        with self._lock:
            self._record_status["status"] = "resuming"
            self._record_status["message"] = "resuming recording..."
            return dict(self._record_status)

    def _resolve_grow_window(self, window_id):
        """Fresh rect for a mid-take JOIN of ONE window -- the single-id,
        strict analog of `_resolve_scene_windows`. All-or-400: a missing /
        non-int / negative id or a window no longer on screen raises, and the
        caller returns BEFORE touching the recorder, so a bad pick never
        perturbs the live take (it keeps recording). No 1-4 range check -- the
        cap is enforced by `grow_supported`."""
        wid = _as_int(window_id, -1)
        if not isinstance(window_id, (int, float, str)) or wid < 0:
            raise StudioError("a valid window id is required to add a window",
                              status=400)
        entry = dev.window_rect_points(
            wid, exclude_pids=self._excluded_pids())
        if entry is None:
            raise StudioError(
                "that window ({}) is no longer on screen — the take is still "
                "recording".format(wid), status=400)
        return entry

    def grow_record(self, body=None):
        """Seamless window-JOIN: add ONE window to a LIVE occlusion-free take
        WITHOUT pausing (docs/architecture.md milestone 2). Body `{"window_id": N}`.

        Mirrors `resume_record` but for the RECORDING state, and every guard
        rejects BEFORE `recorder.grow` so a refused join leaves the take
        recording untouched. Returns the status dict UNCHANGED -- grow is
        seamless, the take stays `recording`; the bar learns the outcome from
        the `joined` / `grow_error` fields the next poll carries."""
        body = body or {}
        with self._lock:
            recorder = self._active_recorder
            st = self._record_status.get("status")
        if recorder is None or st != "recording":
            raise StudioError("no active recording to add a window to",
                              status=409)
        is_fleet = getattr(recorder, "_is_multi_window_native",
                           lambda: False)
        if not is_fleet():
            raise StudioError(
                "adding a window is only supported on occlusion-free takes",
                status=400)
        if not getattr(recorder, "grow_supported", False):
            raise StudioError(
                "can't add a window to this take (at the {}-window "
                "maximum, or the take was paused)".format(
                    rec._MAX_FLEET), status=400)
        wid = body.get("window_id")
        try:
            captured = recorder.captured_window_ids or set()
        except Exception:
            captured = set()
        try:
            if wid is not None and int(wid) in captured:
                # Duplicate-wid guard (M3.3, decision 10): after an
                # auto-rejoin, a user's in-flight chip click for the same
                # window must not spawn a SECOND worker on it.
                raise StudioError("that window is already being recorded",
                                  status=400)
        except (TypeError, ValueError):
            pass
        entry = self._resolve_grow_window(wid)
        recorder.grow(entry)
        with self._lock:
            return dict(self._record_status)

    def snapshot(self):
        with self._lock:
            rec = dict(self._record_status)
            render = dict(self._render_status)
            recorder = self._active_recorder
            st = rec.get("status")
            captured = baseline = None
            raised = None
            grow_ok = False
            if recorder is not None and st == "recording":
                try:
                    captured = recorder.captured_window_ids
                    grow_ok = bool(recorder.grow_supported)
                    baseline = self._grow_baseline_ids or set()
                    # Windows the user raised mid-take escape the baseline
                    # (M2.4c). getattr, not attribute access: a stub / older
                    # recorder without it falls back to the id-only rule.
                    raised = getattr(recorder, "raised_window_ids", None)
                except Exception:
                    captured = None
        # Seamless-join candidates: only for a LIVE fleet take, and the Quartz
        # sweep runs OUTSIDE the lock (it contends with on_record_state on every
        # record transition, and /api/state is the hottest endpoint). Wrapped
        # so this most-polled endpoint can never gain a new 500; any failure
        # falls back to the un-augmented record block, byte-identical to before.
        if captured is not None and st == "recording":
            try:
                native = getattr(recorder, "_is_multi_window_native",
                                 lambda: False)()
                if native:
                    rec["grow_supported"] = grow_ok
                    rec["new_windows"] = (
                        self._new_windows(self._excluded_pids(), captured,
                                          baseline or set(), raised or set())
                        if grow_ok else [])
            except Exception:
                rec.pop("grow_supported", None)
                rec.pop("new_windows", None)
        return {"record": rec, "render": render}

    def launch_native_bar(self, port):
        """Spawn `python3 studio.py bar --port <port>` as a detached child so
        the native pywebview pill runs on ITS own main thread; the child
        health-probes this server (see cli._studio_server_or_url) and reuses
        it. Deduped: a second call while the child is alive is a no-op."""
        if not native_bar_available():
            return {"native": False, "launched": False,
                    "reason": "pywebview not installed"}
        studio_py = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "studio.py"))
        if not os.path.isfile(studio_py):
            return {"native": False, "launched": False,
                    "reason": "studio.py not found"}
        with self._bar_lock:
            existing = self._bar_process
            if existing is not None and existing.poll() is None:
                return {"native": True, "launched": False,
                        "reason": "already running", "pid": existing.pid}
            argv = [sys.executable, studio_py, "bar", "--port", str(int(port))]
            try:
                proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )
            except Exception as exc:
                return {"native": False, "launched": False,
                        "reason": "spawn failed: {}".format(exc)}
            self._bar_process = proc
            # The bar is spawned into its OWN session (above) so it gets its
            # own Cocoa main thread. That detachment has a cost the hard way:
            # it puts the child outside this process's group, so a Ctrl+C to
            # the server never reaches it, and if the server dies the bar
            # keeps running with a backend that no longer exists -- a GUI
            # window that can't record, can't be reached by the API it polls,
            # and that the user has no obvious way to account for (reported:
            # "ctrl c doesn't kill python app/pill ... it kept the bar
            # alive"). `shutdown()` handles the normal Ctrl+C case; this
            # watchdog handles the server dying abnormally, where nothing
            # runs to do it politely.
            self._bar_watchdog = rec.spawn_watchdog([proc.pid])
            return {"native": True, "launched": True, "pid": proc.pid}

    def shutdown(self):
        """Tear down what this server owns, on the way out.

        Called from `serve()`'s `finally`, so it runs on Ctrl+C and on any
        other exit from the serve loop. Three things outlive this process
        otherwise, and all three were observed doing exactly that:
          * the detached native bar (see launch_native_bar) -- a window whose
            backend is about to vanish;
          * an in-flight recording's ffmpeg -- which is in its own session
            too, and which the Recorder only stops from `_finalize()`;
          * the WEBCAM. This process is where the device actually lives (a
            GET /api/camera/preview opens it), and until this call site
            existed `camera_preview.shutdown()` had NONE anywhere in the
            codebase -- the only release in the whole system was a preview
            client's socket dying. So the camera's release was delegated
            entirely to a window in a different process, and a server told to
            quit would hand the device back to macOS only as a side effect of
            its own exit.
        Everything here is best-effort: shutdown must not raise on the way
        out and leave the socket open.
        """
        try:
            camera_preview.shutdown()
        except Exception:
            pass
        try:
            if self._record_status.get("status") in ("countdown", "recording",
                                                     "pausing", "paused",
                                                     "resuming"):
                print("stopping the recording in flight...")
                self.stop_record()
                deadline = time.time() + 10.0
                while time.time() < deadline:
                    if self._record_status.get("status") not in (
                            "recording", "stopping", "countdown",
                            "pausing", "paused", "resuming"):
                        break
                    time.sleep(0.15)
        except Exception:
            pass
        with self._bar_lock:
            bar, self._bar_process = self._bar_process, None
            dog, self._bar_watchdog = self._bar_watchdog, None
        # Stand the watchdog down FIRST: we are about to close the bar
        # ourselves, and a watchdog that noticed us exiting mid-teardown
        # would race us to SIGKILL a window that is already going away
        # politely.
        rec.stop_watchdog(dog)
        if bar is not None and bar.poll() is None:
            print("closing the recording bar...")
            try:
                bar.terminate()
            except Exception:
                pass
            try:
                bar.wait(timeout=5)
            except Exception:
                try:
                    bar.kill()
                except Exception:
                    pass


# Media (raw_i.mov / face.mov / scene channels) must NEVER be heuristically
# cached. The responses carry no validator (no ETag/Last-Modified) and the
# server speaks HTTP/1.0, so with no directive at all Chrome caches them by
# heuristic -- and an aborted partial load (every editor reload tears its
# <video> fleet down mid-fetch) can leave a poisoned entry the browser then
# serves forever: the element sits in networkState=LOADING, readyState=0, no
# error, nothing buffered, and live playback hangs. MEASURED 2026-08-29: the
# identical URL with a cache-busting query loaded instantly (readyState 4)
# while the plain one stalled indefinitely. `no-store` is also correct on the
# merits -- a re-record reuses these filenames, so a stale hit is wrong data.
_MEDIA_CACHE_CONTROL = "no-store, no-cache, must-revalidate"


class StudioHttpServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, state, static_dir, public_host=None,
                 security_token=None):
        _validate_bind_host(public_host if public_host is not None
                            else server_address[0])
        super().__init__(server_address, StudioRequestHandler)
        self.state = state
        self.static_dir = static_dir
        self.security_token = str(security_token or secrets.token_urlsafe(32))
        # DNS rebinding is stopped by comparing the HTTP Host to this exact
        # set.  The validated loopback bind is an explicit additional
        # authority, not a suffix/pattern.
        self.allowed_hosts = {"127.0.0.1", "localhost"}
        for candidate in (public_host, server_address[0], self.server_address[0]):
            value = str(candidate or "").strip().lower().rstrip(".")
            if value:
                self.allowed_hosts.add(value)

    def host_allowed(self, host):
        return host in self.allowed_hosts

    def handle_error(self, request, client_address):
        # clients hanging up mid-response (video scrubbing, closed pill
        # windows) is normal — don't spray tracebacks into the terminal
        exc = sys.exc_info()[1]
        if isinstance(exc, (BrokenPipeError, ConnectionResetError)):
            return
        super().handle_error(request, client_address)


class StudioRequestHandler(BaseHTTPRequestHandler):
    server_version = "autocine-studio/0.1"
    # The MJPEG preview (_stream_camera) writes several small chunks per
    # frame on an unbuffered socket; without this, Nagle's algorithm can
    # hold a small chunk back waiting for the peer's ACK before sending the
    # next one, adding tens of ms of extra latency to every frame.
    disable_nagle_algorithm = True

    def log_message(self, fmt, *args):
        # Keep the terminal cleaner; app state/errors are returned via JSON.
        return

    def end_headers(self):
        # The app never needs to be framed, sniffed as another type, or send
        # local URLs as referrers.  `frame-ancestors` also prevents a hostile
        # page from clickjacking the real local UI after loading it in an
        # iframe (the token itself remains same-origin-only).
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Content-Security-Policy",
                         "frame-ancestors 'none'; base-uri 'none'")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    @staticmethod
    def _authority(value):
        """Return a normalized (hostname, port), or None for bad authority."""
        raw = str(value or "")
        if not raw or raw != raw.strip() or any(c.isspace() for c in raw):
            return None
        try:
            parsed = urlparse("//" + raw)
            if (parsed.username is not None or parsed.password is not None
                    or parsed.path not in ("", "/") or parsed.params
                    or parsed.query or parsed.fragment):
                return None
            host = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port
        except (TypeError, ValueError):
            return None
        if not host:
            return None
        return host, port

    def _host_allowed(self):
        values = self.headers.get_all("Host") or []
        if len(values) != 1:
            return False
        authority = self._authority(values[0])
        if authority is None:
            return False
        host, port = authority
        if not self.server.host_allowed(host):
            return False
        actual_port = int(self.server.server_address[1])
        # A Host without a port denotes the HTTP default (80), not whichever
        # ephemeral port this process happens to own.  Accepting it on 5173
        # would blur the exact authority boundary the DNS-rebinding gate is
        # meant to enforce.
        return (80 if port is None else port) == actual_port

    def _origin_allowed(self):
        values = self.headers.get_all("Origin") or []
        if not values:
            return True
        if len(values) != 1:
            return False
        raw = str(values[0] or "")
        if raw == "null":
            return False
        try:
            parsed = urlparse(raw)
            if (parsed.scheme != "http" or parsed.username is not None
                    or parsed.password is not None
                    or parsed.path not in ("", "/") or parsed.params
                    or parsed.query or parsed.fragment):
                return False
            host = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port if parsed.port is not None else 80
        except (TypeError, ValueError):
            return False
        return (self.server.host_allowed(host)
                and port == int(self.server.server_address[1]))

    def _presented_token(self, path):
        header = self.headers.get(_TOKEN_HEADER)
        if header:
            return str(header)
        if any(path == prefix or path.startswith(prefix)
               for prefix in _TOKEN_QUERY_PATHS):
            values = parse_qs(urlparse(self.path).query).get(_TOKEN_QUERY) or []
            if len(values) == 1:
                return str(values[0])
        return ""

    def _token_allowed(self, path):
        presented = self._presented_token(path)
        expected = self.server.security_token
        if not presented or len(presented) != len(expected):
            return False
        return hmac.compare_digest(presented, expected)

    def _authorize(self, path, method):
        # Gate BEFORE reading a request body or opening a camera/media file.
        if not self._host_allowed() or not self._origin_allowed():
            raise StudioError("forbidden", status=403)
        if path.startswith("/api/") and path not in _TOKEN_EXEMPT_PATHS:
            if not self._token_allowed(path):
                raise StudioError("forbidden", status=403)
        if method == "POST":
            content_type = (self.headers.get("Content-Type") or "").split(
                ";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise StudioError("Content-Type must be application/json",
                                  status=415)

    def _send_json(self, payload, status=200):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_error_json(self, message, status=400):
        self._send_json({"error": str(message)}, status=status)

    def _send_bytes(self, data, content_type, cache_control=None):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        if cache_control:
            self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _own_base_url(self):
        """This server's browsable base URL (host shown as 127.0.0.1 when
        bound to a wildcard address)."""
        host, port = self.server.server_address[0], self.server.server_address[1]
        display_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
        return "http://{}:{}".format(display_host, port)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except (TypeError, ValueError):
            raise StudioError("invalid Content-Length", status=400)
        if length < 0:
            raise StudioError("invalid Content-Length", status=400)
        if length > _MAX_JSON_BODY:
            raise StudioError("JSON body is too large", status=413)
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            raise StudioError("invalid JSON body", status=400)

    def _serve_static(self, req_path):
        if req_path in ("", "/"):
            rel = "index.html"
        else:
            rel = req_path.lstrip("/")
        # Extensionless aliases for app pages (only when the page exists;
        # otherwise fall through to the normal 404).
        alias = {"bar": "bar.html", "editor": "editor.html"}.get(rel)
        if alias and os.path.isfile(os.path.join(self.server.static_dir, alias)):
            rel = alias
        rel = os.path.normpath(rel)
        if rel.startswith(".."):
            raise StudioError("not found", status=404)
        path = os.path.join(self.server.static_dir, rel)
        if not os.path.isfile(path):
            raise StudioError("not found", status=404)
        with open(path, "rb") as f:
            data = f.read()
        ctype = "application/octet-stream"
        if path.endswith(".html"):
            ctype = "text/html; charset=utf-8"
            data = _version_asset_links(data, self.server.static_dir)
            data = _inject_security_token(data, self.server.security_token)
        elif path.endswith(".css"):
            ctype = "text/css; charset=utf-8"
        elif path.endswith(".js"):
            ctype = "application/javascript; charset=utf-8"
        elif path.endswith(".png"):
            ctype = "image/png"
        elif path.endswith(".jpg") or path.endswith(".jpeg"):
            ctype = "image/jpeg"
        elif path.endswith(".svg"):
            ctype = "image/svg+xml"
        elif path.endswith(".ico"):
            ctype = "image/x-icon"
        elif path.endswith(".json"):
            ctype = "application/json"
        elif path.endswith(".woff2"):
            ctype = "font/woff2"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        # The app shell must never be served from a stale cache. With no
        # cache headers at all, WKWebView applies heuristic caching and can
        # reuse bar.css/bar.js indefinitely without revalidating — its disk
        # cache outlives the process, so an edit on disk silently never
        # reaches the window even after a relaunch. This cost a whole
        # debugging round trip once; don't remove it.
        if path.endswith((".html", ".css", ".js")):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream_camera(self, ordinal):
        """MJPEG webcam preview: multipart/x-mixed-replace into an <img>.

        Runs until the client goes away (which shows up as a broken pipe) —
        ThreadingHTTPServer gives each request its own thread, so a long-lived
        stream doesn't block anything else.
        """
        boundary = "ssframe"
        try:
            stream = camera_preview.frames(ordinal)
            first = next(stream)
        except StopIteration:
            raise StudioError("camera produced no frames", status=503)
        except RuntimeError as exc:
            raise StudioError(str(exc), status=503)
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "multipart/x-mixed-replace; boundary={}".format(boundary))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # The camera's lifetime is this socket's lifetime: the refcount that
        # keeps the device open is only dropped by the `finally` below, which
        # is only reached when this loop ends. A peer that closes gets caught
        # (the next write raises EPIPE), but a peer that stays ESTABLISHED and
        # simply stops draining does not — without a timeout this thread parks
        # in sendall forever and the webcam stays lit with nothing able to
        # release it short of killing the process. The bound is deliberately
        # enormous compared to a healthy client (a ~30 KB JPEG over loopback,
        # 30 times a second) so it can only ever fire on a genuinely wedged
        # one.
        try:
            self.connection.settimeout(CAMERA_STREAM_TIMEOUT_SEC)
        except Exception:
            pass
        try:
            frame = first
            while True:
                # one write per frame, not three — each extra write is a
                # chance for Nagle/delayed-ACK to stall the next frame
                # behind a small trailing segment (see disable_nagle_algorithm)
                header = ("--{}\r\nContent-Type: image/jpeg\r\n"
                          "Content-Length: {}\r\n\r\n").format(
                              boundary, len(frame)).encode("ascii")
                self.wfile.write(header + frame + b"\r\n")
                frame = next(stream)
        except (StopIteration, BrokenPipeError, ConnectionResetError,
                socket.timeout):
            pass                      # client closed the <img>, or wedged
        except Exception:
            pass
        finally:
            try:
                stream.close()        # drops this client's refcount
            except Exception:
                pass

    def _serve_file(self, path):
        size = os.path.getsize(path)
        ext = os.path.splitext(path)[1].lower()
        ctype = "application/octet-stream"
        if ext == ".mp4":
            ctype = "video/mp4"
        elif ext == ".mov":
            ctype = "video/quicktime"

        rng = self.headers.get("Range")
        if rng and rng.startswith("bytes="):
            spec = rng[len("bytes="):].strip()
            start_s, end_s = spec.split("-", 1) if "-" in spec else (spec, "")
            if start_s:
                start = int(start_s)
                end = int(end_s) if end_s else (size - 1)
            else:
                # Suffix range: bytes=-N
                n = int(end_s or "0")
                start = max(0, size - n)
                end = size - 1
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", _MEDIA_CACHE_CONTROL)
            self.send_header("Content-Range",
                             "bytes {}-{}/{}".format(start, end, size))
            self.send_header("Content-Length", str(length))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(64 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            return

        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", _MEDIA_CACHE_CONTROL)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_GET(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            self._authorize(path, "GET")
            if path == "/api/sessions":
                self._send_json({"sessions": self.server.state.list_sessions()})
                return
            if path.startswith("/api/sessions/"):
                name = unquote(path[len("/api/sessions/"):])
                self._send_json({"session": self.server.state.get_session(name)})
                return
            if path.startswith("/api/edits/"):
                name = unquote(path[len("/api/edits/"):]).strip("/")
                self._send_json({"edits": self.server.state.get_edits(name)})
                return
            if path == "/api/devices":
                self._send_json(self.server.state.devices())
                return
            if path == "/api/windows":
                # separate from /api/devices on purpose -- that response is
                # cached/polled, and a window list is stale within seconds
                self._send_json(self.server.state.windows())
                return
            if path == "/api/camera/preview":
                qs = parse_qs(urlparse(self.path).query)
                self._stream_camera(_as_int((qs.get("ordinal") or ["0"])[0], 0))
                return
            if path == "/api/state":
                self._send_json(self.server.state.snapshot())
                return
            if path == "/api/permissions":
                self._send_json(self.server.state.permissions())
                return
            if path == "/api/settings":
                self._send_json(self.server.state.settings_payload())
                return
            if path == "/api/bar/windows":
                # What the recorder will ask to keep out of the capture.
                # Readable so the Stage-1 probe (tools/sck_pill_probe.py) can
                # aim at the real pill without reaching into the GUI process.
                self._send_json({"ids": self.server.state.capture_exclude_ids()})
                return
            if path.startswith("/api/media/"):
                parts = path[len("/api/media/"):].split("/")
                if len(parts) != 2:
                    raise StudioError("invalid media path", status=404)
                session_name = unquote(parts[0])
                kind = unquote(parts[1]).strip().lower()
                media_path = self.server.state.media_path(session_name, kind)
                self._serve_file(media_path)
                return
            if path.startswith("/api/thumb/"):
                name = unquote(path[len("/api/thumb/"):]).strip("/")
                data = self.server.state.thumbnail(name)
                self._send_bytes(data, "image/jpeg", cache_control="max-age=30")
                return
            if path.startswith("/api/waveform/"):
                name = unquote(path[len("/api/waveform/"):]).strip("/")
                self._send_json(self.server.state.waveform(name))
                return
            if path.startswith("/api/transcript/"):
                name = unquote(path[len("/api/transcript/"):]).strip("/")
                self._send_json(self.server.state.transcript(name))
                return
            if path.startswith("/api/project/"):
                name = unquote(path[len("/api/project/"):]).strip("/")
                self._send_json({"name": self.server.state.get_project_name(name)})
                return
            if path == "/api/backgrounds":
                self._send_json({"presets": background_presets()})
                return
            if path == "/api/health":
                self._send_json({"ok": True})
                return
            if path == "/api/rev":
                # Cheap poll target for live-reload.js: a boot id that bumps
                # every time the server process starts, plus a flag telling
                # the client whether polling is meaningful. When live_reload
                # is False the client stops polling on the first response.
                self._send_json({"rev": self.server.state.boot_id,
                                 "reload": self.server.state.live_reload})
                return
            self._serve_static(path)
        except StudioError as exc:
            self._send_error_json(str(exc), status=exc.status)
        except Exception as exc:
            self._send_error_json(str(exc), status=500)

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            path = parsed.path
            self._authorize(path, "POST")
            body = self._read_json()
            # Exact-suffix carve-out BEFORE any prefix handling: nothing else
            # may treat "/api/sessions/<name>/delete" as a session name.
            if path.startswith("/api/sessions/") and path.endswith("/delete"):
                name = unquote(
                    path[len("/api/sessions/"):-len("/delete")].strip("/"))
                self._send_json(self.server.state.delete_session(name))
                return
            if path.startswith("/api/transcript/"):
                name = unquote(path[len("/api/transcript/"):]).strip("/")
                self._send_json(self.server.state.start_transcribe(
                    name, force=bool(body.get("force"))))
                return
            if path == "/api/camera-path":
                session = body.get("session")
                options = body.get("options") or {}
                self._send_json(self.server.state.camera_path(session, options))
                return
            if path.startswith("/api/project/"):
                name = unquote(path[len("/api/project/"):]).strip("/")
                saved = self.server.state.set_project_name(name, body.get("name"))
                self._send_json({"name": saved})
                return
            if path == "/api/camera/release":
                # The ONLY release that actually works. A preview client
                # cannot end its own stream -- WKWebView keeps the multipart
                # load (and the socket, and the camera) alive after the <img>
                # drops its src -- so the UI asks US to let go instead. See
                # camera_preview._Manager.release; a camera feeding a take's
                # face.mov is skipped rather than truncated.
                self._send_json({"released": camera_preview.release()})
                return
            if path == "/api/open":
                session = body.get("session")
                base = self._own_base_url()
                if session:
                    self.server.state._session_dir(session)  # validate
                    url = "{}/editor.html?session={}".format(
                        base, quote(str(session)))
                else:
                    url = base + "/"
                try:
                    webbrowser.open(url)
                except Exception:
                    pass
                self._send_json({"opened": url})
                return
            if path == "/api/preview":
                session = body.get("session")
                t_sec = body.get("time", 0.0)
                options = body.get("options") or {}
                data = self.server.state.preview(session, t_sec, options)
                self._send_json(data)
                return
            if path == "/api/render":
                session = body.get("session")
                options = body.get("options") or {}
                status = self.server.state.start_render(session, options)
                self._send_json({"render": status})
                return
            if path.startswith("/api/edits/"):
                suffix = path[len("/api/edits/"):].strip("/")
                if suffix.endswith("/reset"):
                    name = unquote(suffix[:-len("/reset")].strip("/"))
                    saved = self.server.state.reset_session_edits(name)
                else:
                    name = unquote(suffix)
                    payload = body.get("edits", body)
                    try:
                        saved = self.server.state.save_session_edits(
                            name, payload, base_rev=body.get("base_rev"))
                    except EditsConflict as conflict:
                        self._send_json({
                            "error": "edits changed outside this editor",
                            "conflict": True,
                            "edits": conflict.current,
                        }, status=409)
                        return
                    except CutsRejected as rejected:
                        # 400 with the winning document, shaped like the 409
                        # above so the client has ONE recovery path. `code`
                        # is what lets it tell this from a generic 400 (a bad
                        # session name, a torn body) and avoid unwinding an
                        # edit the user never connected to the failure.
                        self._send_json({
                            "error": str(rejected),
                            "code": "cuts",
                            "edits": rejected.current,
                        }, status=400)
                        return
                self._send_json({"edits": saved})
                return
            if path.startswith("/api/presets/"):
                suffix = path[len("/api/presets/"):].strip("/")
                if suffix.endswith("/duplicate"):
                    name = unquote(suffix[:-len("/duplicate")].strip("/"))
                    saved = self.server.state.duplicate_preset(
                        name,
                        name=body.get("name"),
                        source_preset_id=body.get("source_preset_id"),
                        source_render=body.get("source_render"),
                    )
                    self._send_json({"edits": saved})
                    return
                if suffix.endswith("/activate"):
                    name = unquote(suffix[:-len("/activate")].strip("/"))
                    preset_id = body.get("preset_id")
                    if not preset_id:
                        raise StudioError("missing preset_id", status=400)
                    saved = self.server.state.activate_preset(name, preset_id)
                    self._send_json({"edits": saved})
                    return
                raise StudioError("not found", status=404)
            if path == "/api/bar":
                port = self.server.server_address[1]
                self._send_json(self.server.state.launch_native_bar(int(port)))
                return
            if path == "/api/settings":
                self._send_json(self.server.state.save_settings(body))
                return
            if path == "/api/bar/windows":
                # The bar pushing its own window ids. Only that process can
                # read its NSWindows, so a push is the only way they could
                # reach the recorder -- but NOTHING SENDS THIS. No caller
                # exists in studio_web/; the endpoint is reached by tests and
                # by hand only. The `studio.py app` topology named here in an
                # earlier version of this comment is in fact the one case the
                # pid sweep already covers, since the server spawns that bar
                # and knows its pid. See _NativeBarApi.capture_exclusions.
                ids = self.server.state.set_bar_window_ids(body.get("ids"))
                self._send_json({"ids": ids})
                return
            if path == "/api/record/start":
                options = body.get("options") or {}
                status = self.server.state.start_record(options)
                self._send_json({"record": status})
                return
            if path == "/api/record/stop":
                status = self.server.state.stop_record()
                self._send_json({"record": status})
                return
            if path == "/api/record/pause":
                status = self.server.state.pause_record()
                self._send_json({"record": status})
                return
            if path == "/api/record/resume":
                status = self.server.state.resume_record(body)
                self._send_json({"record": status})
                return
            if path == "/api/record/grow":
                # Seamless mid-take window-join. An async ACK -- the join
                # outcome arrives on the next /api/state poll (joined /
                # grow_error). A StudioError (bad/vanished pick, at cap) is
                # sent as a JSON error by the handler below and the take keeps
                # recording untouched.
                status = self.server.state.grow_record(body)
                self._send_json({"record": status})
                return
            if path.startswith("/api/reveal/"):
                name = unquote(path[len("/api/reveal/"):]).strip("/")
                data = self.server.state.reveal_session(name)
                self._send_json(data)
                return
            raise StudioError("not found", status=404)
        except StudioError as exc:
            self._send_error_json(str(exc), status=exc.status)
        except Exception as exc:
            self._send_error_json(str(exc), status=500)

    def do_OPTIONS(self):
        # Same-origin requests never need a CORS preflight.  Refusing every
        # OPTIONS request (and emitting no ACAO headers) makes a foreign page's
        # custom-header attempt fail before it can reach an API handler.
        self._send_error_json("forbidden", status=403)


def _validate_bind_host(host):
    """Return *host* when it is an explicit loopback-only bind.

    The per-launch browser token is a CSRF/DNS-rebinding boundary, not network
    authentication: every legitimate app shell has to receive it.  Binding to
    a wildcard or LAN address would therefore hand that credential to any
    client that can fetch the shell.  Keep the source-development server local
    instead of presenting the bootstrap token as a LAN access control.
    """
    raw = str(host or "")
    if not raw or raw != raw.strip():
        raise ValueError("Studio --host must be an explicit loopback address")
    normalized = raw.lower().rstrip(".")
    if normalized == "localhost":
        return host
    try:
        address = ipaddress.ip_address(normalized)
        # ThreadingHTTPServer is AF_INET here. Reject IPv6 rather than claiming
        # support and returning an invalid unbracketed http://::1:PORT URL.
        loopback = address.version == 4 and address.is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        raise ValueError(
            "Studio --host must be loopback-only (localhost or 127.0.0.0/8)")
    return host


def build_server(host="127.0.0.1", port=5173, recordings_root=DEFAULT_REC_ROOT):
    """Bind the studio HTTP server; returns (httpd, url).

    `url` has no trailing slash so callers can compose page paths
    (e.g. url + "/bar.html"). Raises ValueError for a non-loopback bind and
    OSError (EADDRINUSE) if the port is already taken -- callers like `studio
    bar` use the latter to detect an already-running studio server.
    """
    _validate_bind_host(host)
    static_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "studio_web"))
    if not os.path.isdir(static_dir):
        raise RuntimeError("studio web assets not found: {}".format(static_dir))

    state = StudioState(recordings_root=recordings_root)
    server = StudioHttpServer(
        (host, int(port)), state=state, static_dir=static_dir,
        public_host=host)
    actual_host, actual_port = server.server_address[0], server.server_address[1]
    browser_host = "127.0.0.1" if actual_host in ("0.0.0.0", "::") else actual_host
    url = "http://{}:{}".format(browser_host, actual_port)
    return server, url


def serve(httpd):
    """Serve until interrupted; always closes the socket."""
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down studio app...")
    finally:
        # Before the socket goes: stop anything this server spawned that
        # would otherwise outlive it (the detached native bar, an in-flight
        # recording's ffmpeg). Without this, Ctrl+C left both running --
        # the bar as a window with no backend, ffmpeg as a silent recorder.
        try:
            httpd.state.shutdown()
        except Exception:
            pass
        httpd.server_close()
    return 0


def run_server(host="127.0.0.1", port=5173, open_browser=True,
               recordings_root=DEFAULT_REC_ROOT, open_page=None):
    server, url = build_server(host, port, recordings_root=recordings_root)
    print("studio app listening on {}/".format(url))
    print("recordings root: {}".format(server.state.recordings_root))
    if open_browser:
        if open_page:
            target = url + "/" + str(open_page).lstrip("/")
        else:
            target = url + "/"
        try:
            webbrowser.open(target)
        except Exception:
            pass
    return serve(server)
