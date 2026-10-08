"""Capture-device discovery and display/window geometry for macOS."""

import math
import re
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def list_avf_devices() -> Dict[str, List[Tuple[int, str]]]:
    """Parse `ffmpeg -f avfoundation -list_devices` into video/audio lists."""
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "avfoundation",
         "-list_devices", "true", "-i", ""],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    text = proc.stderr
    video: List[Tuple[int, str]] = []
    audio: List[Tuple[int, str]] = []
    section: Optional[str] = None
    for line in text.splitlines():
        if "AVFoundation video devices" in line:
            section = "video"
            continue
        if "AVFoundation audio devices" in line:
            section = "audio"
            continue
        m = re.search(r"\[(\d+)\]\s+(.*)$", line)
        if m and section:
            idx, name = int(m.group(1)), m.group(2).strip()
            (video if section == "video" else audio).append((idx, name))
    return {"video": video, "audio": audio}


def find_screen_device(devices: Dict[str, List[Tuple[int, str]]]) -> Optional[int]:
    """Return the avfoundation index of the first 'Capture screen' device."""
    for idx, name in devices.get("video", []):
        if "capture screen" in name.lower():
            return idx
    return None


def is_screen_device(devices: Dict[str, List[Tuple[int, str]]],
                     idx: Optional[int]) -> bool:
    """True when `idx` names a screen-capture device in THIS device list.

    avfoundation indices are POSITIONAL, not stable identifiers: the video
    list is webcams and screens interleaved in one numbering, so plugging in
    a Continuity Camera or starting a virtual cam renumbers everything after
    it. An index captured from an earlier listing can therefore come to name
    a *webcam* -- and recording then silently produces a video of the user's
    face instead of their screen, which is the "screen fell back to camera"
    failure seen from the other end.

    So any index that didn't come from a listing we just read has to be
    re-checked against one before it reaches ffmpeg. Uses the same
    "capture screen" name test as `find_screen_device`, so auto-detection and
    validation can never disagree about what counts as a screen. An unknown
    index is False: absent from the list means it names nothing we can vouch
    for, which is not a screen.
    """
    if idx is None:
        return False
    for i, name in devices.get("video", []):
        if i == idx:
            return "capture screen" in name.lower()
    return False


def find_camera_device(devices: Dict[str, List[Tuple[int, str]]]) -> Optional[int]:
    """Return the avfoundation index of the first real camera.

    A camera is any video device that isn't a screen-capture device
    (on a Mac that's the FaceTime/webcam). Returns None when the machine
    exposes no webcam, so facecam capture degrades to 'no face track'.
    """
    for idx, name in devices.get("video", []):
        if "capture screen" in name.lower():
            continue
        return idx
    return None


def main_display_points() -> Tuple[float, float, str]:
    """Logical size of the main display, in points. Returns (w, h, source).

    avfoundation records at the physical (Retina) backing resolution, but
    pynput reports the cursor in points. We store the point size here so the
    renderer can compute the exact points->pixels scale from the recorded file
    (this correctly handles fractional Retina scaling, e.g. 1.7x).
    """
    # 1) Quartz — accurate; installed alongside pynput's dependencies.
    try:
        from Quartz import CGMainDisplayID, CGDisplayBounds
        b = CGDisplayBounds(CGMainDisplayID())
        return float(b.size.width), float(b.size.height), "quartz"
    except Exception:
        pass
    # 2) AppleScript fallback.
    try:
        out = subprocess.check_output(
            ["osascript", "-e",
             'tell application "Finder" to get bounds of window of desktop'],
            text=True).strip()
        parts = [float(p) for p in out.replace(" ", "").split(",")]
        if len(parts) == 4:
            return parts[2] - parts[0], parts[3] - parts[1], "osascript"
    except Exception:
        pass
    # 3) Last resort.
    return 1440.0, 900.0, "fallback"


# --- window enumeration (for record-time "capture this window") -------------
#
# avfoundation cannot target a window, so the picker only snapshots a RECT (in
# points, global top-left origin) that render.py later crops out of the
# full-display capture. Everything here soft-fails exactly like
# main_display_points() above: no Quartz -> empty list, never an exception.

# Size gates, in points. Both are independently necessary: the ~24pt-tall
# menubar strips only fail on height, a 35pt-wide Grammarly side rail only
# fails on width.
WINDOW_MIN_W_PT = 160.0
WINDOW_MIN_H_PT = 120.0

# Windows whose alpha is at or below this are invisible shims, not UI.
WINDOW_MIN_ALPHA = 0.05

# Longest title we put in a picker label before eliding.
WINDOW_TITLE_MAX = 60

# Deliberately MINIMAL. The real work is done by the layer==0 and size gates —
# a big owner blacklist is brittle (names change between macOS releases) and
# locale-dependent (they're localized), and it would happily hide a real app
# that happens to share a name with a system agent.
WINDOW_OWNER_BLACKLIST = ("Window Server", "Dock", "Wallpaper")

# Our own always-on-top chrome. PID is the primary test; these titles are the
# fallback for the hand-launched-bar case (studio_app spawns `studio.py bar` as
# a detached child, so os.getpid() alone doesn't cover it).
WINDOW_OWN_TITLES = (
    "AutoCine — New Recording",
    "AutoCine — Facecam",
    "AutoCine — Pick Windows",
)


def displays_points() -> List[Dict[str, Any]]:
    """Active displays as {"id","x","y","w","h","main"} dicts, in points.

    Global top-left origin, i.e. the same space kCGWindowBounds uses (the main
    display sits at (0,0) and displays above/left of it have negative origins).
    Returns [] when Quartz is unavailable — callers treat that as "no window
    capture", never as an error.
    """
    try:
        from Quartz import (CGDisplayBounds, CGGetActiveDisplayList,
                            CGMainDisplayID)
        err, ids, count = CGGetActiveDisplayList(16, None, None)
        if err:
            return []
        main_id = int(CGMainDisplayID())
        out: List[Dict[str, Any]] = []
        for did in list(ids)[:int(count)]:
            b = CGDisplayBounds(did)
            out.append({
                "id": int(did),
                "x": float(b.origin.x), "y": float(b.origin.y),
                "w": float(b.size.width), "h": float(b.size.height),
                "main": int(did) == main_id,
            })
        return out
    except Exception:
        return []


def _finite(v: Any) -> Optional[float]:
    """float(v) if it's a real finite number, else None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _intersect(rect: Tuple[float, float, float, float],
               disp: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    """Overlap of a window rect with a display's bounds, or None if disjoint."""
    x0 = max(rect[0], float(disp["x"]))
    y0 = max(rect[1], float(disp["y"]))
    x1 = min(rect[0] + rect[2], float(disp["x"]) + float(disp["w"]))
    y1 = min(rect[1] + rect[3], float(disp["y"]) + float(disp["h"]))
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1 - x0, y1 - y0)


def _window_label(app: str, title: str) -> str:
    """Ready-made picker option text: "App — Title" (or just "App")."""
    t = title.strip()
    if not t or t == app:
        return app
    if len(t) > WINDOW_TITLE_MAX:
        t = t[:WINDOW_TITLE_MAX - 1].rstrip() + "…"
    return "%s — %s" % (app, t)


def _filter_windows(infos: Iterable[Any],
                    displays: Sequence[Dict[str, Any]],
                    exclude_pids: Iterable[int] = (),
                    main_only: bool = True,
                    min_w: float = WINDOW_MIN_W_PT,
                    min_h: float = WINDOW_MIN_H_PT,
                    clip_to_display: bool = True) -> List[Dict[str, Any]]:
    """Turn raw CGWindowListCopyWindowInfo dicts into picker entries.

    Pure: no Quartz, no I/O — all the junk-filtering rules live here so they're
    unit-testable offline. Input order is PRESERVED (the window list comes back
    front-to-back in z-order, which is exactly what a picker wants).

    Dropped: anything not on layer 0 (menu-bar extras are 25, Notification
    Centre 23, Dock icons 20, side rails 5, the cursor is a huge negative),
    near-transparent shims, offscreen/minimized windows, windows that don't
    overlap any display, windows whose visible part is smaller than the size
    gates, non-main-display windows (when `main_only`), our own recording
    chrome, and a minimal owner blacklist.

    A window that hangs off the edge of its display is KEPT but reported
    INTERSECTED with that display — an avfoundation crop can only ever cover
    pixels the capture actually contains, so the visible part is the honest
    answer for the display-crop window pick, which is what this was written
    for.

    `clip_to_display=False` reports the window's TRUE bounds instead. That is
    what OCCLUSION-FREE (window-native SCK) capture needs, because there the
    capture IS the window's own surface and includes whatever hangs off the
    edge of the screen — so the clipped rect describes strictly less than the
    recording contains. Getting this wrong is silent and shows up as a
    stretched card at render time; see docs/architecture.md, "The
    display-CLIPPED rect". Every GATE still runs on the visible part
    (display ownership by biggest overlap, the size floors, `main_only`) —
    only what is REPORTED changes, so a window that is 95% off screen is
    still refused rather than offered at its true size.
    """
    pids = set()
    for p in exclude_pids:
        try:
            pids.add(int(p))
        except (TypeError, ValueError):
            continue
    out: List[Dict[str, Any]] = []
    for raw in infos:
        try:
            info = dict(raw)
        except Exception:
            continue

        # (2) real windows live on layer 0.
        layer = _finite(info.get("kCGWindowLayer", 0))
        if layer is None or int(layer) != 0:
            continue

        # (3) invisible shims.
        alpha = _finite(info.get("kCGWindowAlpha", 1.0))
        if alpha is None or alpha <= WINDOW_MIN_ALPHA:
            continue

        # (4) minimized / hidden windows omit this key entirely.
        if not info.get("kCGWindowIsOnscreen"):
            continue

        # (5) bounds must be finite and non-degenerate.
        try:
            b = dict(info.get("kCGWindowBounds") or {})
        except Exception:
            continue
        x = _finite(b.get("X"))
        y = _finite(b.get("Y"))
        w = _finite(b.get("Width"))
        h = _finite(b.get("Height"))
        if None in (x, y, w, h) or w <= 0 or h <= 0:
            continue

        # (6) owning display = biggest overlap; gate on the VISIBLE part.
        best = None
        best_area = 0.0
        for disp in displays:
            hit = _intersect((x, y, w, h), disp)
            if hit is None:
                continue
            area = hit[2] * hit[3]
            if area > best_area:
                best, best_area = (hit, disp), area
        if best is None:
            continue
        rect, disp = best
        if rect[2] < min_w or rect[3] < min_h:
            continue
        if main_only and not disp.get("main"):
            continue
        # Gates above ran on the VISIBLE part, always. Only the reported
        # geometry follows `clip_to_display`.
        if not clip_to_display:
            rect = (x, y, w, h)

        app = str(info.get("kCGWindowOwnerName") or "").strip()
        title = str(info.get("kCGWindowName") or "").strip()

        # (7) never offer our own recording chrome.
        pid = _finite(info.get("kCGWindowOwnerPID"))
        if pid is not None and int(pid) in pids:
            continue
        if title in WINDOW_OWN_TITLES:
            continue

        # (8) minimal owner blacklist.
        if app in WINDOW_OWNER_BLACKLIST:
            continue

        wid = _finite(info.get("kCGWindowNumber"))
        if wid is None:
            continue

        # kCGWindowName is '' (or absent) without Screen Recording permission.
        if not app:
            app = title or "Window"
        out.append({
            "id": int(wid),
            "app": app,
            "title": title or app,
            "label": _window_label(app, title),
            "x": rect[0], "y": rect[1], "w": rect[2], "h": rect[3],
            "display_id": int(disp["id"]),
            "display_origin": [float(disp["x"]), float(disp["y"])],
            "main_display": bool(disp.get("main")),
        })
    return out


def _copy_window_info(options: int, window_id: int) -> List[Any]:
    """CGWindowListCopyWindowInfo as a plain list. Raises without Quartz —
    both callers below wrap it in the usual soft-failing try/except."""
    from Quartz import CGWindowListCopyWindowInfo
    return list(CGWindowListCopyWindowInfo(options, window_id) or [])


def window_onscreen(window_id: int) -> Optional[bool]:
    """Is this ONE window on-screen right now? -> True / False / None.

    The parent-side sibling of `_sck_worker._target_onscreen`, for the fleet
    card-shrink detector (docs/architecture.md M3): a direct per-id query that
    BYPASSES `list_windows`' display/size/alpha filters, so a window that is
    merely filtered (other display, shrunk tiny, alpha shim) still reads
    on-screen -- only genuinely not-onscreen (minimized, closed) is False.
    Measured caveats the caller must respect: a Cmd-H-hidden window stays
    ONSCREEN here, and a window on an inactive Space reads False while SCK
    keeps delivering -- so this is never sufficient for an exit by itself
    (decision 2's arrivals-dead requirement). None on any read hiccup: a
    transient Quartz failure must count as SEEN, never toward a card exit.
    """
    try:
        from Quartz import kCGWindowListOptionIncludingWindow
        wid = int(window_id)
        for info in _copy_window_info(kCGWindowListOptionIncludingWindow, wid):
            if int(info.get("kCGWindowNumber", -1)) == wid:
                return bool(info.get("kCGWindowIsOnscreen", False))
        return False                      # absent -> minimized / closed
    except Exception:
        return None


def list_windows(exclude_pids: Iterable[int] = (),
                 main_only: bool = True,
                 displays: Optional[List[Dict[str, Any]]] = None,
                 clip_to_display: bool = True) -> List[Dict[str, Any]]:
    """Pickable on-screen windows, front-to-back. [] without Quartz.

    `displays` lets a caller reuse an already-read display list. That is not
    micro-optimization: `displays_points()` is ~6.9 ms of this call's ~8.6 ms,
    and the record-time geometry poller runs it many times a second while
    ffmpeg is encoding. Displays cannot change mid-recording in any way this
    tool supports (window capture is main-display-only), so the poller reads
    them once. Omit it and behavior is exactly as before.
    """
    try:
        from Quartz import (kCGNullWindowID, kCGWindowListExcludeDesktopElements,
                            kCGWindowListOptionOnScreenOnly)
        infos = _copy_window_info(
            kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements,
            kCGNullWindowID)
        return _filter_windows(infos,
                               displays_points() if displays is None else displays,
                               exclude_pids=exclude_pids, main_only=main_only,
                               clip_to_display=clip_to_display)
    except Exception:
        return []


def window_rect_points(window_id: int,
                       exclude_pids: Iterable[int] = (),
                       main_only: bool = True,
                       clip_to_display: bool = True
                       ) -> Optional[Dict[str, Any]]:
    """Re-read ONE window's entry, or None if it's no longer capturable.

    None means gone, minimized, moved offscreen, shrunk below the size gates,
    or (with `main_only`) dragged to a secondary display — every case where the
    caller should keep the rect it already had rather than trust a fresh one.
    """
    try:
        from Quartz import kCGWindowListOptionIncludingWindow
        infos = _copy_window_info(kCGWindowListOptionIncludingWindow,
                                  int(window_id))
        found = _filter_windows(infos, displays_points(),
                                exclude_pids=exclude_pids, main_only=main_only,
                                clip_to_display=clip_to_display)
        for entry in found:
            if entry["id"] == int(window_id):
                return entry
        return None
    except Exception:
        return None


def audio_unique_ids() -> List[Tuple[str, str]]:
    """[(uniqueID, localizedName)] for capture mics, in ffmpeg's own order.

    ScreenCaptureKit identifies a microphone by an AVFoundation uniqueID
    string (e.g. "BuiltInMicrophoneDevice"), while this app's bar, CLI and
    `meta["mic_index"]` all speak an avfoundation ORDINAL. This is the bridge.

    The order is not a guess: ffmpeg 8.1's libavdevice enumerates audio
    devices with `discoverySessionWithDeviceTypes:mediaType:position:` over
    `AVCaptureDeviceTypeMicrophone`/`AVMediaTypeAudio` (confirmed by reading
    the selector out of libavdevice), so making the identical call returns
    the same devices in the same order — ordinal N is index N here. The
    caller cross-checks the name anyway, because an ordinal that has silently
    shifted must never be allowed to select a different microphone.

    [] without the ObjC runtime. No new dependency: importing
    ScreenCaptureKit or Quartz already loads AVFoundation into the runtime.
    """
    try:
        import objc
        # Quartz is imported for its SIDE EFFECT, not its API: the class
        # below is registered by AVFoundation, and `objc.lookUpClass` finds
        # nothing until some framework has pulled it into the ObjC runtime.
        # Without this the whole function silently returned [] -- which read
        # exactly like "this machine has no microphones". Quartz rather than
        # AVFoundation because pynput already makes it a hard dependency, so
        # this path stays available on a checkout with no SCK wheels.
        import Quartz  # noqa: F401
        session = objc.lookUpClass("AVCaptureDeviceDiscoverySession")
        got = session.discoverySessionWithDeviceTypes_mediaType_position_(
            # "AVCaptureDeviceTypeMicrophone" is the macOS 14+ spelling and
            # covers built-in AND external inputs. The older
            # ...TypeBuiltInMicrophone returns ZERO devices here, so it is
            # not a safe fallback -- it looks like "no mic" rather than an
            # error. "soun" is AVMediaTypeAudio's real value.
            ["AVCaptureDeviceTypeMicrophone"], "soun", 0)
        return [(str(dev.uniqueID()), str(dev.localizedName()))
                for dev in got.devices()]
    except Exception:
        return []


def mic_unique_id(ordinal: Optional[int],
                  expect_name: Optional[str] = None) -> Optional[str]:
    """avfoundation audio ordinal -> AVFoundation uniqueID, or None.

    None means "could not resolve safely", and the caller must treat that as
    no microphone rather than as a default one. Recording the wrong input is
    the failure that matters here: this app already carries a warning that
    avfoundation indices are not stable (anything connecting or
    disconnecting renumbers everything after it), and the screen-device
    equivalent of this mistake records the user's face.

    `expect_name` is the name ffmpeg reported for that ordinal. When it is
    given and does not match what the discovery session returns at that
    index, we refuse rather than pick — a mismatch means the numbering moved
    under us, which is exactly when guessing is most harmful.
    """
    if ordinal is None:
        return None
    try:
        idx = int(ordinal)
    except (TypeError, ValueError):
        return None
    devices = audio_unique_ids()
    if idx < 0 or idx >= len(devices):
        return None
    uid, name = devices[idx]
    if expect_name:
        want = str(expect_name).strip().lower()
        if want and want not in name.strip().lower():
            return None
    return uid


def _ids_for_pids(infos: Iterable[Any], pids: Iterable[int]) -> List[int]:
    """Pure: every `kCGWindowNumber` owned by one of `pids`, in z-order.

    Deliberately does NOT reuse `_filter_windows`, which exists to answer a
    different question ("what may the user pick to record?") and would drop
    every window we need here:

      * it keeps only layer 0, and our chrome is `on_top=True` — the pill
        sits at layer 25, so `list_windows` structurally cannot see it;
      * it drops near-transparent shims, and the pill's window is mostly
        transparent slack kept for its drop shadow;
      * it applies size gates, and the pill is small by design.

    A window we merely *fail to enumerate* is a window that lands in the
    user's recording, so the rule here is the opposite of the picker's: keep
    everything the pid owns and let the caller decide.

    PRIVACY: reads `kCGWindowNumber` and `kCGWindowOwnerPID` only. Window
    titles are never touched, matching `_window_label`'s posture — an id is
    an opaque handle, a title is the user's content.
    """
    want = set()
    for p in pids:
        try:
            want.add(int(p))
        except (TypeError, ValueError):
            continue
    out: List[int] = []
    seen = set()
    for raw in infos:
        try:
            info = dict(raw)
        except Exception:
            continue
        pid = _finite(info.get("kCGWindowOwnerPID"))
        if pid is None or int(pid) not in want:
            continue
        num = _finite(info.get("kCGWindowNumber"))
        if num is None:
            continue
        wid = int(num)
        if wid in seen:
            continue
        seen.add(wid)
        out.append(wid)
    return out


def window_ids_for_pids(pids: Iterable[int]) -> List[int]:
    """On-screen window ids owned by `pids`. [] without Quartz.

    Designed as the backstop half of capture exclusion — the bar reports the
    ids it knows about, and this catches anything it doesn't (a WebKit
    service window, a window created after the last report). In practice it
    is the ONLY half: nothing pushes, so this is what excludes the bar in
    every take today. Its limits are therefore the system's limits — pid-
    keyed, so a bar the server never spawned is invisible to it, and
    on-screen only, so a window is covered from the poll after it appears
    rather than in advance.

    Soft-fails to [] like every other Quartz call here — an empty list means
    "exclude nothing", which is today's behavior, not a crash.
    """
    try:
        from Quartz import (kCGNullWindowID, kCGWindowListOptionOnScreenOnly)
        infos = _copy_window_info(kCGWindowListOptionOnScreenOnly,
                                  kCGNullWindowID)
        return _ids_for_pids(infos, pids)
    except Exception:
        return []


def window_owner_pid(window_id: int,
                     exclude_pids: Iterable[int] = ()) -> Optional[int]:
    """The pid that owns one window, or None. Deliberately NOT on `Entry`.

    The Accessibility API is per-process (`AXUIElementCreateApplication`
    takes a pid), so anything that wants to MOVE a window needs this. It is a
    separate lookup rather than a twelfth `Entry` key because the Entry shape
    is a pinned contract that a lot of code reads, and because a pid is a
    handle onto the user's running processes — it belongs on the call that
    actually needs one, not in every window list the app passes around.

    Goes through the same `_filter_windows` gates as `window_rect_points`, so
    a window we would refuse to capture is also one we refuse to move.
    """
    try:
        from Quartz import kCGWindowListOptionIncludingWindow
        infos = _copy_window_info(kCGWindowListOptionIncludingWindow,
                                  int(window_id))
        keep = _filter_windows(infos, displays_points(),
                               exclude_pids=exclude_pids, main_only=True)
        if not any(e["id"] == int(window_id) for e in keep):
            return None
        for info in infos:
            if not isinstance(info, dict) and not hasattr(info, "get"):
                continue
            num = _finite(info.get("kCGWindowNumber"))
            if num is None or int(num) != int(window_id):
                continue
            pid = _finite(info.get("kCGWindowOwnerPID"))
            return int(pid) if pid is not None else None
        return None
    except Exception:
        return None
