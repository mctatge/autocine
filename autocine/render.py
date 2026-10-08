"""Render a recorded session into an auto-zoom MP4 (and optional GIF).

On top of the two-pass auto-zoom camera (see camera.py) the renderer can layer
presentation polish, all driven purely by the recorded event stream:

  - click highlights   expanding ring + pulse at every click (effects.ClickFX)
  - cursor spotlight    optional radial dim around the cursor (effects.Spotlight)
  - framed background   gradient / solid / wallpaper (framing.FramePainter)
  - intro/outro fade    fade from / to black over `fade` seconds
  - background music     mixed under the recording's own audio via ffmpeg
  - click / key sounds   a pre-mixed SFX bed at every recorded click and
                         keystroke tick (sfx.py; on by default)
"""

import base64
import json
import os
import re
import subprocess
import tempfile
from math import hypot

import cv2
import numpy as np

from . import (beats, camera, capture_badge, effects, eraser, framing,
               geometry, retime, segments, sfx, vision)

# Runaway guard on the SFX bed. The old value was 1200, sized for the
# per-click ffmpeg `adelay` taps the bed replaced (see sfx.py's docstring for
# why those went). Keystroke ticks are bucketed at 0.1s, so a talkative take
# carries far more of them than clicks, and the bed costs O(events) rather
# than O(events) filter nodes -- there is no reason to be stingy any more.
_MAX_CLICK_SFX_EVENTS = sfx.MAX_SFX_EVENTS

# How far either side of the frame the editor's single-frame preview is
# allowed to search for clean pixels. The export walks the whole file, which
# is right for a background job and wrong for a still the editor repaints on
# every edit. See `_preview_cursor_eraser`.
_ERASE_PREVIEW_REACH_SEC = 0.5


def _session_paths(session_dir):
    meta_path = os.path.join(session_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    raw_path = os.path.join(session_dir, meta.get("raw", "raw.mov"))
    events_path = os.path.join(session_dir, meta.get("events", "events.jsonl"))
    return meta, raw_path, events_path


# -- macOS capture-indicator erase (`badge_erase`) -----------------------
#
# An occlusion-free (window-native) capture comes back with a system-drawn
# "this window is being captured" pill burned into the top-left corner of
# every frame. `autocine/capture_badge.py` finds it and paints it out; this
# section is only the wiring -- where each pixel path picks up an eraser, and
# the memo that keeps the locate off the scrub path.
#
# EVERY path here is None when `badge_erase` is off or the take is not
# window-native, and a None eraser is never called: an unaffected take runs
# the pre-feature loop unchanged, which is the bit-exact off switch the
# project's invariants require.

# Locating the badge costs an open plus a few seeks. The editor's scrub calls
# `preview_frame` once per drag frame on a file that cannot change after the
# recording stopped, so the result is memoised per (path, size, mtime) for
# the life of the process -- the web app is long-running, and re-probing per
# scrub frame was the difference between a responsive scrub and a stuttering
# one.
_BADGE_RECTS = {}
_BADGE_RECTS_MAX = 64


def _badge_key(path):
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (os.path.abspath(path), st.st_size, st.st_mtime)


def _badge_probe(path, spec, frame_w=None):
    """Memoised `(rect, patch)` for one window-native file -- the box, and the
    repair described for a caller that paints elsewhere. `(None, None)` when
    there is nothing to find (not window-native, or a sheet/dialog macOS
    never marked -- neither has window buttons for it to cover)."""
    if not capture_badge.applies(spec):
        return None, None
    key = _badge_key(path)
    if key is not None and key in _BADGE_RECTS:
        return _BADGE_RECTS[key]
    rect, frame = capture_badge.probe(
        path, capture_badge.scale_for(spec, frame_w))
    found = (rect, capture_badge.patch(frame, rect)
             if rect is not None and frame is not None else None)
    if key is not None:
        if len(_BADGE_RECTS) >= _BADGE_RECTS_MAX:
            _BADGE_RECTS.clear()
        _BADGE_RECTS[key] = found
    return found


def _badge_rect(path, spec, frame_w=None):
    return _badge_probe(path, spec, frame_w)[0]


def _badge_patch(path, spec, frame_w=None, enabled=True):
    """The badge repair as JSON for the editor's live canvas player: the same
    box and the same flat colour the render paints, so a scrub and a play of
    the same take show the same corner. None when there is nothing to erase.
    """
    if not enabled:
        return None
    found = _badge_probe(path, spec, frame_w)[1]
    if not found:
        return None
    b, g, r = found["color"]
    # CSS, not the BGR triple OpenCV works in: the consumer is a canvas
    # fillStyle, and a channel order swapped on the way out is exactly the
    # kind of bug that shows up as a blue corner nobody can explain.
    return {"rect": list(found["rect"]),
            "color": "#{:02x}{:02x}{:02x}".format(r, g, b)}


def _badge_eraser(path, spec, frame_w=None, enabled=True):
    """A `BadgeEraser` for one window-native channel, or None."""
    if not enabled or not capture_badge.applies(spec):
        return None
    scale = capture_badge.scale_for(spec, frame_w)
    rect = _badge_rect(path, spec, frame_w)
    return capture_badge.BadgeEraser(scale, rect=rect)


def _badge_erasers(session_dir, channels, dims=None, enabled=True):
    """One eraser per manifest channel (None where the channel has no badge)."""
    out = []
    for i, ch in enumerate(channels):
        w = dims[i][0] if dims and i < len(dims) else None
        out.append(_badge_eraser(
            os.path.join(session_dir, ch.get("file", "")), ch, w,
            enabled=enabled))
    return out if any(e is not None for e in out) else None


def _badge_report(erasers, label=None):
    """Print whatever the erasers have to say. Silent on the ordinary
    outcome -- see `BadgeEraser.report`."""
    for i, er in enumerate(erasers or []):
        if er is None:
            continue
        line = er.report()
        if line:
            print("  {}{}".format(
                "" if label is None else "{} {}: ".format(label, i), line))


def _badge_erase_once(frame, path, spec, frame_w=None, enabled=True):
    """Single-frame erase for the preview paths, which decode out of order
    and so cannot carry the loop's lock/verify state."""
    if not enabled:
        return frame
    rect = _badge_rect(path, spec, frame_w)
    if rect is not None:
        capture_badge.erase(frame, rect)
    return frame


def _facecam_overlay(session_dir, meta, out_w, out_h, enabled=True, params=None):
    """Build a facecam bubble overlay from the session's face track, or None
    when there's no face.mov / the facecam is disabled. Alignment uses the two
    monotonic anchors stored at record time (t0_monotonic, face_t0_monotonic)."""
    if not enabled:
        return None
    face_name = meta.get("face")
    if not face_name:
        return None
    face_path = os.path.join(session_dir, face_name)
    if not os.path.isfile(face_path):
        return None
    t0 = float(meta.get("t0_monotonic", 0.0) or 0.0)
    face_t0 = meta.get("face_t0_monotonic")
    face_t0 = float(face_t0) if face_t0 is not None else t0
    ov = effects.FacecamOverlay(face_path, out_w, out_h, t0, face_t0,
                                face_fps=meta.get("face_fps") or 30.0,
                                params=params)
    return ov if ov.available() else None


def _probe_has_audio(path):
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            text=True).strip()
        return bool(out)
    except Exception:
        return False


# Layout names we will paste into a filter graph. ffprobe is trusted here,
# but a graph string is a command surface, so the value is whitelisted
# rather than interpolated blind.
_SAFE_CHANNEL_LAYOUTS = ("mono", "stereo", "2.1", "3.0", "4.0", "quad",
                         "5.0", "5.1", "6.1", "7.1")


def _probe_audio_layout(path):
    """The channel layout of `path`'s first audio stream, or None.

    Exists because the SFX bed is MONO and the recording may not be. Feeding
    a mono bed into `amix` beside a stereo mic makes ffmpeg resolve the
    mismatch by downmixing the RECORDING -- measured: a stereo mic comes out
    mono and 3 dB hot, and a mic with signal on one channel only is folded
    into the other. That is silent damage to the user's voice caused purely
    by an effect being on, so the bed is pinned to the recording's layout
    instead (`aformat` in `_encode_cmd` / `_multi_native_enc_cmd`).

    Returns None when unknown, which leaves the graph exactly as it was --
    correct for the mono case, where there is no mismatch to resolve.
    """
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a:0",
             "-show_entries", "stream=channel_layout,channels",
             "-of", "csv=p=0", path], text=True).strip()
    except Exception:
        return None
    if not out:
        return None
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    for value in parts:
        if value in _SAFE_CHANNEL_LAYOUTS:
            return value
    # Older ffprobe builds can report an empty/"unknown" layout; fall back to
    # the channel COUNT for the only two shapes a capture realistically has.
    for value in parts:
        if value == "1":
            return "mono"
        if value == "2":
            return "stereo"
    return None


def _multi_native_has_audio(session_dir, meta):
    """Does an occlusion-free fleet take carry a mic track? True iff channel 0
    (the designated audio owner) has an audio stream. Cheap probe of one file;
    a mic-off take has no audio track there, so this is False, byte-for-byte
    the previous behavior."""
    channels = meta.get("capture_channels") or []
    if not channels:
        return False
    ch0 = channels[0].get("file")
    if not ch0:
        return False
    return _probe_has_audio(os.path.join(session_dir, ch0))


def _scene_has_continuous_audio(session_dir, meta):
    """Does a scene take carry a continuous channel-0 mic track? True only for
    a gapfree JOIN: channel 0 is the SAME file in every scene (one continuous
    recording across the seam) and it has an audio stream. A pause/resume
    scene take (per-scene channel-0 files) returns False -- its audio would
    need per-scene placement, which milestone 1's guard keeps off the table."""
    scenes = meta.get("capture_scenes") or []
    if not scenes:
        return False
    ch0_files = [((sc.get("channels") or [{}])[0].get("file")) for sc in scenes]
    if not ch0_files[0] or any(f != ch0_files[0] for f in ch0_files):
        return False
    return _probe_has_audio(os.path.join(session_dir, ch0_files[0]))


def _probe_duration(path):
    """Container duration in seconds, or None."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], text=True).strip()
        return float(out)
    except Exception:
        return None


def _typing_anchor_hints(raw_path, events_path, keys_t, max_zoom, params,
                         suppressed_ranges):
    """Visual typing anchors for this session's bursts, or None.

    Cheap when there is nothing to do (no keys / no gated bursts) and
    cached across calls (the editor previews per scrub). Any failure
    degrades to None -- planning falls back to the click-anchor rule.

    Sync note (review-settled): burst spans are in NUDGED media time, and
    that IS the video timeline by the --offset contract -- the user tunes
    offset until events align with the on-screen action -- so the video is
    sampled at the spans directly. Subtracting offset here would undo the
    calibration and miss the typing by exactly the nudge.
    """
    keys_t = np.asarray(keys_t if keys_t is not None else [])
    if keys_t.size == 0:
        return None
    bursts = camera.typing_bursts(keys_t, max_zoom=max_zoom, params=params,
                                  suppressed_ranges=suppressed_ranges)
    if not bursts:
        return None
    try:
        mtime = os.path.getmtime(events_path)
    except OSError:
        mtime = 0.0
    return vision.cached_typing_anchors(raw_path, mtime, bursts)


def _plan_timemap(speedup, rate, silence_gate, speedups, params,
                  clicks_t, moves_t, moves_x, moves_y, ups_t, keys_t,
                  scrolls_t, raw_path, duration, has_audio,
                  motion_gate=True, verbose=False, cuts=None):
    """Build the TimeMap for a render.

    Returns TimeMap([]) (the exact identity) whenever the feature is off,
    the plan finds nothing to speed, or `speedups` contains only
    "off"/keep-1x overrides. The renderer's every non-identity code path
    is guarded by `tm.identity`, so this returning identity keeps the
    off-switch bit-exact.

    `cuts` (already union-merged + frame-quantized -- see
    retime.quantize_cut_spans) is threaded into EVERY TimeMap constructed
    here, identity exits included: a cuts-only render must produce a
    non-identity map even when speed-up is off or its gates found nothing.
    """
    cuts = cuts or []
    # `speedups` is manual per-span overrides ({start, end, mode, rate?})
    # -- kept even when `speedup=False` so an author can add "force" spans
    # to test the pipeline on a session that would otherwise generate no
    # auto spans. Only force ranges matter in that case (auto detection
    # is off).
    overrides = list(speedups or [])
    if not speedup:
        force = [o for o in overrides if o.get("mode") == "force"]
        if not force:
            return retime.TimeMap([], duration=duration, cuts=cuts)
        spans = [{"start": float(o["start"]),
                  "end": float(o["end"]),
                  "rate": max(1.0, float(o.get("rate") or rate))}
                 for o in force
                 if o.get("start") is not None and o.get("end") is not None]
        p = dict(retime.DEFAULTS)
        p.update(params or {})
        return retime.TimeMap(spans, ramp=float(p["ramp"]),
                              duration=duration, cuts=cuts)

    activity = retime.activity_times(
        clicks_t, moves_t, moves_x, moves_y, ups_t, keys_t, scrolls_t,
        move_speed=float((params or {}).get(
            "move_speed", retime.DEFAULTS["move_speed"])))
    drags = retime.drag_spans(clicks_t, ups_t,
                              cap=float((params or {}).get(
                                  "drag_cap", retime.DEFAULTS["drag_cap"])))
    silence = None
    if has_audio and silence_gate:
        p = dict(retime.DEFAULTS)
        p.update(params or {})
        silence = retime.silence_spans(raw_path, duration,
                                       db=float(p["silence_db"]),
                                       min_dur=float(p["silence_min"]))
        if silence is None:
            # Fail-closed: the probe failed (ffmpeg missing, timeout,
            # unreadable audio codec, ...). The user asked for the
            # narration guard AND has audio; we don't know whether it's
            # silence or narration, so we must NOT chipmunk-speed it.
            # Auto detection short-circuits to identity; explicit
            # 'force' overrides still fire (author knows what they're
            # doing). Disable --speedup-silence-gate to override.
            if verbose:
                print("  warning: silence detection failed; auto speed-up "
                      "disabled (disable --speedup-silence-gate to force)")
            force = [o for o in overrides if o.get("mode") == "force"]
            if not force:
                return retime.TimeMap([], duration=duration, cuts=cuts)
            spans = [{"start": float(o["start"]),
                      "end": float(o["end"]),
                      "rate": max(1.0, float(o.get("rate") or rate))}
                     for o in force
                     if o.get("start") is not None and o.get("end") is not None]
            p2 = dict(retime.DEFAULTS)
            p2.update(params or {})
            return retime.TimeMap(spans, ramp=float(p2["ramp"]),
                                  duration=duration, cuts=cuts)
    motion_fn = None
    if motion_gate:
        p2 = dict(retime.DEFAULTS)
        p2.update(params or {})
        motion_fn = (lambda s: retime.visually_static_spans(raw_path, s,
                                                            params=p2))
    spans = retime.plan_speed_spans(activity, drags, duration, rate,
                                    params=params, silence=silence,
                                    overrides=overrides,
                                    motion_static=motion_fn)
    if not spans:
        return retime.TimeMap([], duration=duration, cuts=cuts)
    p = dict(retime.DEFAULTS)
    p.update(params or {})
    return retime.TimeMap(spans, ramp=float(p["ramp"]),
                              duration=duration, cuts=cuts)


def _plan_moves_mask(moves_t, tm):
    """Boolean mask over `moves_t`: True = keep in the CAMERA plan.

    Move samples whose timestamp lies strictly inside a sped span (past
    the entry ramp and before the exit ramp) are dropped -- warping would
    turn drift into a fake fast-move that triggers the cluster keep-alive
    and holds a zoom through a time-lapse. Samples in identity/ramp
    regions are kept so intra-burst gaps remain identity-exact.
    """
    if moves_t is None or moves_t.size == 0 or tm.identity:
        return np.ones_like(np.asarray(moves_t, dtype=bool), dtype=bool)
    mask = np.ones(moves_t.size, dtype=bool)
    for (a, b, r) in tm.spans:
        d = min(getattr(tm, "_ramp", 0.5), (b - a) / 2.0)
        lo = a + d
        hi = b - d
        if hi > lo:
            mask &= ~((moves_t > lo) & (moves_t < hi))
    return mask


def _drop_degenerate_spans(spans):
    """Remove {start, end} dicts / (a, b) pairs a cut collapsed to (near-)
    zero length under `TimeMap.warp_spans`. Items without a recognizable
    span shape pass through untouched (mirroring warp_spans itself)."""
    if not spans:
        return spans
    out = []
    for s in spans:
        try:
            if isinstance(s, dict) and "start" in s and "end" in s:
                if float(s["end"]) - float(s["start"]) <= 1e-6:
                    continue
            elif isinstance(s, (tuple, list)) and len(s) >= 2:
                if float(s[1]) - float(s[0]) <= 1e-6:
                    continue
        except (TypeError, ValueError):
            pass
        out.append(s)
    return out


def _trim_frame_bounds(total_count, fps, trim_start=0.0, trim_end=None):
    """Frame bounds [start, end) for a trim interval in seconds."""
    total = int(total_count)
    if total <= 0:
        return 0, 0
    rate = max(1e-6, float(fps))
    start_s = max(0.0, float(trim_start or 0.0))
    start_idx = int(np.floor(start_s * rate))
    start_idx = max(0, min(start_idx, total - 1))
    if trim_end is None:
        end_idx = total
    else:
        end_s = max(start_s, float(trim_end))
        end_idx = int(np.ceil(end_s * rate))
        end_idx = min(total, max(start_idx + 1, end_idx))
    return start_idx, end_idx


def _click_times_in_trim(clicks_t, clip_start, clip_duration,
                         max_events=_MAX_CLICK_SFX_EVENTS):
    """Click times mapped into a local clip timeline.

    Returns (times_sec, dropped_count), where times are relative to clip start
    and clipped to [0, clip_duration].
    """
    arr = np.asarray(clicks_t if clicks_t is not None else [], dtype=np.float64)
    if arr.size == 0:
        return [], 0
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return [], 0
    clip_start = float(clip_start or 0.0)
    clip_duration = max(0.0, float(clip_duration or 0.0))
    if clip_duration <= 0.0:
        return [], 0
    clip_end = clip_start + clip_duration + 1e-6
    arr = arr[(arr >= clip_start) & (arr <= clip_end)]
    if arr.size == 0:
        return [], 0
    local = arr - clip_start
    dropped = 0
    max_events = int(max(0, int(max_events)))
    if max_events and local.size > max_events:
        dropped = int(local.size - max_events)
        local = local[:max_events]
    return [float(x) for x in local.tolist()], dropped


def _sfx_layers(click_choice, key_choice, click_times, release_times,
                key_times):
    """The `sfx.Layer` list for one render, or [] when everything is off.

    All three time arrays are already on the OUTPUT timeline (trimmed, cut
    events dropped, speed-up warped) -- the same arrays the encoder sees --
    so the bed cannot drift from the picture.

    A user-supplied file replaces ONLY the press sound and plays at the old
    `volume=0.7`, so an existing `--click-sound` take keeps the character
    (and roughly the level) it always had. The mouse-up release sound is a
    built-in-only flourish: pairing an arbitrary file with itself 6 ms later
    would read as a double-click, not as a button coming back up.
    """
    layers = []
    if click_choice != sfx.OFF and len(click_times):
        if click_choice == sfx.AUTO:
            layers.append(sfx.Layer(click_times, sfx.builtin_bank("click"),
                                    jitter=0.10))
            if len(release_times):
                layers.append(sfx.Layer(release_times,
                                        sfx.builtin_bank("release")))
        else:
            samples = sfx.decode_file(click_choice)
            if samples is None:
                print("  warning: click sound file not found or not "
                      "decodable, skipping: {}".format(click_choice))
            else:
                layers.append(sfx.Layer(click_times, [samples], gain=0.7))
    if key_choice != sfx.OFF and len(key_times):
        if key_choice == sfx.AUTO:
            layers.append(sfx.Layer(key_times, sfx.builtin_bank("key"),
                                    jitter=0.18))
        else:
            samples = sfx.decode_file(key_choice)
            if samples is None:
                print("  warning: key sound file not found or not "
                      "decodable, skipping: {}".format(key_choice))
            else:
                layers.append(sfx.Layer(key_times, [samples], gain=0.7))
    return layers


def _write_sfx_bed(duration_s, click_sound, key_sound, sfx_volume,
                   click_times, release_times, key_times):
    """Pre-mix the event bed and write it to a temp wav. None when there is
    nothing to play.

    Shared by all three render paths (single-file, multi-window native
    fleet, scene take) so there is exactly ONE sound engine -- the measured
    reason for that is in sfx.py: the per-event ffmpeg `adelay` taps this
    replaced ramped 25x in loudness across a take.

    Returning None rather than a silent wav IS the off switch: it is what
    keeps `_encode_cmd` / `_multi_native_enc_cmd` byte-identical to their
    pre-bed argv. A silent wav would still add an input, a map and
    `-c:a aac -shortest`, turning a mic-less take from "no audio stream"
    into "a silent AAC track".

    `duration_s` and every time array must already be on the CALLER's output
    clock. Nothing downstream shifts them: `-ss` binds only to the mic input
    it precedes (measured -- see `_multi_native_enc_cmd`). `duration_s` must
    also come from the caller's planned FRAME COUNT, never from the event
    times: `build_bed` sizes the buffer from it, and a bed shorter than the
    video makes `-shortest` truncate the VIDEO (measured: a 0.8s bed against
    2.0s of video exported 24 frames instead of 60).
    """
    bed = sfx.build_bed(
        duration_s,
        _sfx_layers(sfx.resolve(click_sound), sfx.resolve(key_sound),
                    click_times, release_times, key_times),
        volume=sfx_volume)
    if bed is None:
        return None
    fd, path = tempfile.mkstemp(prefix="autocine-sfx-", suffix=".wav")
    os.close(fd)
    try:
        sfx.write_wav(path, bed)
    except Exception:
        # The path never escapes on this branch, so no caller's `finally`
        # could ever clean it up -- this is the only place that can.
        _remove_sfx_bed(path)
        raise
    return path


def _remove_sfx_bed(path):
    """Delete a temp bed. Call only AFTER the encoder is done reading it."""
    if not path:
        return
    try:
        os.remove(path)
    except OSError:
        pass


def describe_session(session_dir, include_click_times=False):
    """Return recording/session metadata used by the Studio app."""
    meta, raw_path, events_path = _session_paths(session_dir)
    # Multi-window native (P3.4): there is no single raw.mov -- the session is
    # N raw_i.mov channels + a manifest. Return a channel-aware description so
    # the editor and MCP don't try to open a top-level raw that isn't there.
    if _is_multi_native_meta(meta):
        return _describe_multi_native_session(
            session_dir, meta, events_path,
            include_click_times=include_click_times)
    # Scene take: K sequential fleet scenes, no single raw. Checked before
    # the segmented branch for symmetry with render()'s dispatch order.
    if segments.is_scene_meta(meta):
        return _describe_scene_session(
            session_dir, meta, events_path,
            include_click_times=include_click_times)
    # Segmented (pause/resume): no single raw -- N sequential seg_i.mov whose
    # paused gaps are deleted. Report the joined (gaps-removed) timeline.
    if segments.is_segmented_meta(meta):
        return _describe_segmented_session(
            session_dir, meta, events_path,
            include_click_times=include_click_times)
    if not os.path.isfile(raw_path):
        raise RuntimeError("cannot open recording: " + raw_path)
    if not os.path.isfile(events_path):
        raise RuntimeError("cannot open events: " + events_path)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open recording: " + raw_path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps", 60))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if frame_count > 0 and fps > 0:
        duration = frame_count / float(fps)
    else:
        duration = _probe_duration(raw_path) or 0.0
    ev = geometry.load_events(events_path)
    t0 = float(meta.get("t0_monotonic", 0.0) or 0.0)
    # Record-time window capture: `width`/`height` are THE source coordinate
    # space for every consumer (the editor maps viewport px -> source px with
    # them, and clamps zoom pins / window rects against them), so they must
    # report the CROPPED frame. The file's own dimensions stay available as
    # raw_width/raw_height.
    crop = _capture_crop_px(meta, width, height)
    info = {
        "session": os.path.basename(os.path.abspath(session_dir)),
        "session_dir": os.path.abspath(session_dir),
        "raw_path": raw_path,
        "events_path": events_path,
        "fps": float(fps),
        "width": int(crop[2]) if crop is not None else int(width),
        "height": int(crop[3]) if crop is not None else int(height),
        "raw_width": int(width),
        "raw_height": int(height),
        "capture_window": _describe_capture_window(meta, crop),
        "capture_windows": _describe_capture_windows(meta, ev, t0),
        "frame_count": int(frame_count),
        "duration": float(duration),
        "has_audio": bool(_probe_has_audio(raw_path)),
        "click_count": int(ev["clicks_t"].size),
        "move_count": int(ev["moves_t"].size),
        "key_count": int(ev["keys_t"].size),
        "scroll_count": int(ev["scrolls_t"].size),
        "key_capture": meta.get("key_capture"),
        "cursor_mode": str(meta.get("cursor_mode") or "system"),
        "has_face": bool(meta.get("face")) and os.path.isfile(
            os.path.join(session_dir, str(meta.get("face") or ""))),
        "face_capture": meta.get("face_capture"),
        # Screen media time -> face media time, the one number the editor's
        # live composite needs to seek face.mov alongside raw.mov. Derived
        # here from the two record-time monotonic anchors so the browser
        # never has to know they exist -- same value FacecamOverlay uses.
        "face_offset": (float(t0) - float(meta.get("face_t0_monotonic", t0)
                                          if meta.get("face_t0_monotonic")
                                          is not None else t0)),
        "has_output_mp4": os.path.isfile(os.path.join(session_dir, "output.mp4")),
        "has_output_gif": os.path.isfile(os.path.join(session_dir, "output.gif")),
    }
    if include_click_times:
        def _media_times(arr):
            times = (arr - t0) if arr.size else arr
            if not times.size:
                return []
            times = times[np.isfinite(times)]
            times = times[times >= 0.0]
            if duration > 0:
                times = times[times <= (duration + 1e-6)]
            return [float(x) for x in times.tolist()]
        info["click_times"] = _media_times(ev["clicks_t"])
        info["key_times"] = _media_times(ev["keys_t"])
        info["scroll_times"] = _media_times(ev["scrolls_t"])
    return info


def _multi_native_composite_t0(session_t0, origin, src_fps):
    """Parent-monotonic instant a FLAT fleet composite's OUTPUT FRAME 0 shows.

    Every channel is `grab()`-skipped to the shared origin, so output frame k
    is SESSION frame `origin + k`: channel i reads its file frame
    `(origin - offsets[i]) + k`, whose monotonic time is
    `session_t0 + offsets[i]/fps + (origin - offsets[i])/fps` -- i.e.
    `session_t0 + origin/fps` for EVERY i, which is what makes it the
    composite's t0 rather than any one channel's.

    So the composite's own clock is `t - (session_t0 + origin/fps)`, NOT
    `t - session_t0`: an event mapped the latter way lands `origin/fps` late
    against the picture (measured: exactly `origin` frames), while the mic
    is not -- `audio_skip_s` seeks channel 0's audio by that same `origin`
    skip, so the sound was already right and only the picture drifted.
    This is the flat-manifest twin of `segments.scene_clock_entries`'s
    `t0_s = t0_ch0 + origin/fps`, where the same reasoning is already
    load-bearing for scene takes -- the flat path was the one fleet timeline
    that never got it.

    `origin` is small in practice (0-8 frames measured across the local fleet
    takes, 0-133ms at 60fps) but unbounded in principle: it is exactly how far
    the SLOWEST channel's SCK worker lagged session t0.
    """
    src_fps = float(src_fps or 0.0)
    if src_fps <= 0.0:
        return float(session_t0)
    return float(session_t0) + (int(origin) / src_fps)


def _describe_multi_native_session(session_dir, meta, events_path,
                                    include_click_times=False):
    """`describe_session` for a P3.1 multi-window native take.

    No single `raw_path` -- the source is N `raw_i.mov` channels. Reports the
    OUTPUT canvas dims (the multi-native composite default, honoring nothing
    per-channel here since the render canvas is one shared frame) as
    `width`/`height`, plus a `capture_channels` array carrying each channel's
    file, window id, point rect, buffer dims and per-channel t0. Duration is
    the shortest channel's decodable span (the composite clips to the common
    overlap, exactly what `_render_multi_native` does).
    """
    channels = meta.get("capture_channels") or []
    fps = float(meta.get("fps") or 60)
    session_t0 = float(meta.get("t0_monotonic", 0.0) or 0.0)
    # load_events wraps its body in try/except and returns empty arrays for a
    # missing/unreadable file, so it is safe to call unconditionally.
    ev = geometry.load_events(events_path)

    # Per-channel span in output frames -> shortest common duration.
    chans_out = []
    min_span = None
    for ch in channels:
        path = os.path.join(session_dir, ch.get("file") or "")
        n_frames = 0
        if os.path.isfile(path):
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        ch_t0 = float(ch.get("t0_monotonic") or session_t0)
        offset = int(round((ch_t0 - session_t0) * fps))
        # Frames available from the shared origin (max offset) onward.
        chans_out.append({
            "file": ch.get("file"),
            "id": ch.get("id"),
            "app": ch.get("app"),
            "title": ch.get("title"),
            "rect": ch.get("rect"),
            "logical_w": ch.get("logical_w"),
            "logical_h": ch.get("logical_h"),
            "buffer_w": ch.get("buffer_w"),
            "buffer_h": ch.get("buffer_h"),
            "t0_monotonic": ch.get("t0_monotonic"),
            "track": ch.get("track"),
            "frame_count": n_frames,
            "frame_offset": offset,
        })
    offsets = [c["frame_offset"] for c in chans_out] or [0]
    origin = max(0, max(offsets))
    for c in chans_out:
        span = c["frame_count"] - max(0, origin - c["frame_offset"])
        if span > 0:
            min_span = span if min_span is None else min(min_span, span)
    n_out = min_span or 0
    duration = (n_out / fps) if fps > 0 else 0.0

    out_w, out_h = framing.output_size(
        _MULTI_NATIVE_DEFAULT_W, _MULTI_NATIVE_DEFAULT_H, "clean", aspect=None)

    info = {
        "session": os.path.basename(os.path.abspath(session_dir)),
        "session_dir": os.path.abspath(session_dir),
        "raw_path": None,           # no single raw -- the manifest is the truth
        "events_path": events_path,
        "fps": float(fps),
        "width": int(out_w),
        "height": int(out_h),
        "raw_width": int(out_w),
        "raw_height": int(out_h),
        "capture_window": None,
        "capture_windows": None,
        "capture_channels": chans_out,
        "multi_native": True,
        "frame_count": int(n_out),
        "duration": float(duration),
        # Mic (when present) rides channel 0; probe its file so describe/editor
        # report audio truthfully. Mic-off channel 0 has no audio track -> False,
        # exactly as before.
        "has_audio": bool(_multi_native_has_audio(session_dir, meta)),
        "click_count": int(ev["clicks_t"].size),
        "move_count": int(ev["moves_t"].size),
        "key_count": int(ev["keys_t"].size),
        "scroll_count": int(ev["scrolls_t"].size),
        "key_capture": meta.get("key_capture"),
        "cursor_mode": str(meta.get("cursor_mode") or "system"),
        "has_face": False,
        "face_capture": None,
        "face_offset": 0.0,
        "has_output_mp4": os.path.isfile(
            os.path.join(session_dir, "output.mp4")),
        "has_output_gif": os.path.isfile(
            os.path.join(session_dir, "output.gif")),
    }
    if include_click_times:
        # Same clock the composite is on: output time 0 is session frame
        # `origin`, so events rebase through the composite's t0, not session
        # t0 (`_multi_native_composite_t0`). Anchoring on session t0 reported
        # every time `origin/fps` late against a `duration` that IS the
        # composite's -- so the [0, duration] window below both admitted
        # pre-roll clicks the composite never shows and dropped real ones off
        # its tail. This is the surface `beats` and the MCP locate moments on.
        composite_t0 = _multi_native_composite_t0(session_t0, origin, fps)

        def _media_times(arr):
            times = (arr - composite_t0) if arr.size else arr
            if not times.size:
                return []
            times = times[np.isfinite(times)]
            times = times[times >= 0.0]
            if duration > 0:
                times = times[times <= (duration + 1e-6)]
            return [float(x) for x in times.tolist()]
        info["click_times"] = _media_times(ev["clicks_t"])
        info["key_times"] = _media_times(ev["keys_t"])
        info["scroll_times"] = _media_times(ev["scrolls_t"])
    return info


# ---- segmented takes (pause/resume) -----------------------------------------
# A segmented take is one recording paused/resumed into K sequential seg_i.mov
# files (`capture_segments` manifest, no top-level `raw`), whose paused
# wall-clock gaps are DELETED at render. The event->output-time map is the
# shared `segments.SegmentClock` (see docs/architecture.md). These
# helpers join the files for the render body and describe the joined timeline.


def _segment_paths(session_dir, meta):
    """Absolute paths of the segment files, in manifest order."""
    segs = meta.get("capture_segments") or []
    return [os.path.join(session_dir, s.get("file") or "") for s in segs]


def _segment_frame_counts(paths):
    """Decoded frame count of each segment file (0 for a missing/unreadable
    one). This is the ONE source of truth the SegmentClock and the joined
    frame grid both derive from."""
    counts = []
    for p in paths:
        n = 0
        if os.path.isfile(p):
            cap = cv2.VideoCapture(p)
            if cap.isOpened():
                n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        counts.append(n)
    return counts


def _concat_segments(session_dir, meta, fps):
    """Join the segment files into one continuous working file.

    Returns (joined_path, seg_frame_counts, total_count, tmp_paths). VIDEO is
    stream-copied by the concat demuxer -- frame-exact for the identical
    per-segment config (same codec/pixfmt/fps/resolution in Phase 1), so the
    joined frame count is exactly the sum of the per-segment counts.

    AUDIO (mic on) is NOT copied through the demuxer -- that concatenates the
    audio streams independently and lets each segment's ~0.26 s SCK spin-up
    lead-in stack into a growing A/V skew. Instead each segment's audio is
    rebuilt at its RECORDED placement: SCK audio starts ~0.26 s after video
    frame 0 and that start_time is real content placement, not a spurious
    delay (the worker rebases mic pts onto the video timeline --
    `_sck_worker.on_audio`; the single-file path preserves it via the
    container's start_time). `aresample=async=1:first_pts=0` reproduces that
    by padding the [0, start_time) window with silence -- never by sliding the
    audio early -- then each segment is padded/capped to EXACTLY
    D_i = count_i/fps and `concat`-butted: total audio == sum(D_i) == the
    joined video length, no per-seam drift, no per-segment shift. A segment
    with NO audio stream (the SCK worker is video-first and drops a failed mic
    rather than the take, so a resume can lose the mic mid-take) contributes
    D_i of synthesized silence instead of crashing the join. Everything is
    normalized to one sample format so concat accepts heterogeneous segments.
    Mic-off takes (no segment has audio) take the `-an` copy path. On any
    failure the partial joined file and the list file are removed before the
    error propagates -- a failed export must not strand a multi-GB orphan in
    the session dir.
    """
    paths = _segment_paths(session_dir, meta)
    for p in paths:
        if not os.path.isfile(p):
            raise RuntimeError("segment file missing: " + p)
    seg_counts = _segment_frame_counts(paths)
    total = sum(seg_counts)
    joined = os.path.join(session_dir, "._segjoin.{}.mov".format(os.getpid()))
    listfile = os.path.join(session_dir, "._segjoin.{}.txt".format(os.getpid()))
    try:
        with open(listfile, "w") as f:
            for p in paths:
                # concat-demuxer list syntax: single-quote the path, escaping
                # any embedded quote as '\'' per ffmpeg's rules.
                f.write("file '{}'\n".format(
                    os.path.abspath(p).replace("'", "'\\''")))
        seg_has_audio = [_probe_has_audio(p) for p in paths]
        if not any(seg_has_audio):
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                 "-i", listfile, "-c", "copy", "-an", joined], check=True)
            return joined, seg_counts, total, [joined, listfile]
        # Mic-on: copy video frame-exact (input 0 = the concat demuxer),
        # rebuild audio drift-free from the audio-carrying segment files as
        # separate inputs 1..K.
        inputs = ["-f", "concat", "-safe", "0", "-i", listfile]
        filt = []
        labels = []
        norm = ("aformat=sample_fmts=fltp:sample_rates=48000:"
                "channel_layouts=stereo")
        aud_idx = 0
        for i, p in enumerate(paths):
            d_i = (seg_counts[i] / float(fps)) if fps else 0.0
            if seg_has_audio[i]:
                aud_idx += 1
                inputs += ["-i", p]
                filt.append(
                    "[{idx}:a]aresample=async=1:first_pts=0,{norm},"
                    "atrim=0:{d:.6f},apad,atrim=0:{d:.6f}[a{i}]".format(
                        idx=aud_idx, norm=norm, d=d_i, i=i))
            else:
                filt.append(
                    "anullsrc=r=48000:cl=stereo,{norm},atrim=0:{d:.6f}[a{i}]"
                    .format(norm=norm, d=d_i, i=i))
            labels.append("[a{}]".format(i))
        filt.append("{}concat=n={}:v=0:a=1[a]".format(
            "".join(labels), len(paths)))
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y"] + inputs +
            ["-filter_complex", ";".join(filt),
             "-map", "0:v", "-map", "[a]", "-c:v", "copy", "-c:a", "aac",
             joined],
            check=True)
        return joined, seg_counts, total, [joined, listfile]
    except Exception:
        for tmp in (joined, listfile):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise


def _render_segmented(session_dir, out_path=None, **kwargs):
    """Render a segmented take: join the segment files, then run the ordinary
    single-file body on the joined file with the shared SegmentClock as
    `to_media` and `screen_focus` forced off (it resamples the geometry track,
    which can jump across a deleted pause gap -- a stated Phase-1 cut)."""
    with open(os.path.join(session_dir, "meta.json")) as f:
        meta = json.load(f)
    fps = float(meta.get("fps") or 60)
    if out_path is None:
        out_path = os.path.join(session_dir, "output.mp4")
    joined, seg_counts, total, tmp_paths = _concat_segments(
        session_dir, meta, fps)
    try:
        clock = segments.SegmentClock.from_meta(meta, seg_counts, fps)
        return render(
            session_dir, out_path=out_path,
            _src_override=joined, _to_media_override=clock.media,
            _bed_clock=clock,
            _count_override=total, screen_focus=False, **kwargs)
    finally:
        for tmp in tmp_paths:
            try:
                os.remove(tmp)
            except OSError:
                pass


def _describe_segmented_session(session_dir, meta, events_path,
                                include_click_times=False):
    """`describe_session` for a segmented take.

    No single `raw` -- the source is K sequential seg_i.mov whose paused gaps
    are deleted. Reports the JOINED (gaps-removed) timeline: `duration` is the
    sum of the per-segment content durations, `width`/`height` come from
    segment 0 (all share the config), and click/key/scroll times are mapped
    through the SAME `SegmentClock` the render and beats surfaces use -- so the
    editor, the MCP beat sheet, and the export cannot disagree across a seam.
    """
    seg_paths = _segment_paths(session_dir, meta)
    fps = float(meta.get("fps") or 60)
    seg_counts = _segment_frame_counts(seg_paths)
    total = sum(seg_counts)
    clock = segments.SegmentClock.from_meta(meta, seg_counts, fps)
    duration = clock.duration()
    width = height = 0
    if seg_paths and os.path.isfile(seg_paths[0]):
        cap = cv2.VideoCapture(seg_paths[0])
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    ev = geometry.load_events(events_path)
    segs_meta = meta.get("capture_segments") or []
    segs_out = [{
        "file": s.get("file"), "index": i,
        "t0_monotonic": s.get("t0_monotonic"),
        "frame_count": seg_counts[i] if i < len(seg_counts) else 0,
    } for i, s in enumerate(segs_meta)]
    info = {
        "session": os.path.basename(os.path.abspath(session_dir)),
        "session_dir": os.path.abspath(session_dir),
        "raw_path": None,           # no single raw -- the manifest is the truth
        "events_path": events_path,
        "fps": float(fps),
        "width": int(width), "height": int(height),
        "raw_width": int(width), "raw_height": int(height),
        "capture_segments": segs_out,
        "segmented": True,
        "frame_count": int(total),
        "duration": float(duration),
        "has_audio": bool(seg_paths and _probe_has_audio(seg_paths[0])),
        "click_count": int(ev["clicks_t"].size),
        "move_count": int(ev["moves_t"].size),
        "key_count": int(ev["keys_t"].size),
        "scroll_count": int(ev["scrolls_t"].size),
        "key_capture": meta.get("key_capture"),
        "cursor_mode": str(meta.get("cursor_mode") or "system"),
        "has_face": False,          # facecam is a Phase-1 cut on segmented takes
        "face_capture": meta.get("face_capture"),
        "face_offset": 0.0,
        "has_output_mp4": os.path.isfile(
            os.path.join(session_dir, "output.mp4")),
        "has_output_gif": os.path.isfile(
            os.path.join(session_dir, "output.gif")),
    }
    if include_click_times:
        def _mt(arr):
            if not getattr(arr, "size", 0):
                return []
            t = clock.media(arr)
            t = t[np.isfinite(t)]
            return [float(x) for x in t.tolist()]
        info["click_times"] = _mt(ev["clicks_t"])
        info["key_times"] = _mt(ev["keys_t"])
        info["scroll_times"] = _mt(ev["scrolls_t"])
    return info


def _hidden_file_set(hidden_channels, channel_lists):
    """The channel FILES to actually hide, `hidden_channels` (edits.py) minus
    two guards that keep a hide from ever breaking a take:

      * the session ANCHOR (the FIRST channel of the FIRST list) is never
        hidden -- it carries the clock origin (and, on a mic take, the audio);
      * a hide that would EMPTY any channel list is dropped for that list's
        files -- a zero-card scene/composite is inexpressible.

    `channel_lists` is the per-scene channel lists (scene take) or a single
    `[channels]` (multi-native). Returns a set of basenames; empty when nothing
    hides, so every caller's no-hide path is byte-identical to today."""
    if not hidden_channels:
        return set()
    hide = set(hidden_channels)
    lists = [lst for lst in channel_lists if lst]
    if lists and lists[0]:
        hide.discard((lists[0][0] or {}).get("file"))   # anchor is never hidden
    for lst in lists:
        files = [(c or {}).get("file") for c in lst]
        if files and not [f for f in files if f not in hide]:
            for f in files:                              # would empty this list
                hide.discard(f)
    return hide


def _visible_channels(channels, hide):
    """`channels` with hidden files dropped, plus the KEPT indices (so a
    positional layout list can be filtered in lockstep)."""
    keep = [i for i, c in enumerate(channels or [])
            if (c or {}).get("file") not in hide]
    return [channels[i] for i in keep], keep


def _filter_positional(lst, keep):
    """A positional list (channel_layouts / a scene's slice) re-indexed to the
    kept channels, trailing-None trimmed (the `_normalize_channel_layouts`
    canonical form)."""
    if not lst:
        return lst
    out = [lst[i] if i < len(lst) else None for i in keep]
    while out and out[-1] is None:
        out.pop()
    return out


def _apply_hidden_scenes(scenes, hidden_channels, scene_layouts):
    """Filtered `(scenes, scene_layouts)` for a scene take: every scene's
    channel list has the hidden files removed and its manual-layout slice is
    re-indexed in lockstep. No-hide -> the inputs unchanged (identity)."""
    hide = _hidden_file_set(hidden_channels,
                            [sc.get("channels") or [] for sc in scenes])
    if not hide:
        return scenes, scene_layouts
    out_scenes, out_layouts = [], {}
    for s_i, sc in enumerate(scenes):
        vis, keep = _visible_channels(sc.get("channels") or [], hide)
        new_sc = dict(sc)
        new_sc["channels"] = vis
        out_scenes.append(new_sc)
        if scene_layouts:
            lay = _filter_positional(scene_layouts.get(str(s_i)), keep)
            if lay:
                out_layouts[str(s_i)] = lay
    return out_scenes, (out_layouts if scene_layouts else scene_layouts)


def _apply_hidden_channels(channels, hidden_channels, channel_layouts):
    """Filtered `(channels, channel_layouts)` for a multi-native take."""
    hide = _hidden_file_set(hidden_channels, [channels or []])
    if not hide:
        return channels, channel_layouts
    vis, keep = _visible_channels(channels, hide)
    return vis, _filter_positional(channel_layouts, keep)


def _describe_scene_session(session_dir, meta, events_path,
                            include_click_times=False):
    """`describe_session` for a SCENE take (docs/architecture.md).

    No single `raw_path` -- the source is K scenes of 1-4 native channels.
    `duration` is the gaps-deleted sum of per-scene content durations from
    the SAME `segments.SegmentClock.from_scene_meta` builder render uses;
    each scene info entry carries the ORIGIN-ADJUSTED `t0_monotonic` and its
    aligned `frame_count` (the P1 key-name convention -- `session_beats`
    rebuilds the identical clock from exactly those two, which is the
    three-surface agreement the parents' review demanded). `width`/`height`
    report the default composite canvas, mirroring the multi-native
    describe. Clock warnings (shortfall/overlap) surface as
    `scene_warnings` rather than stdout -- the MCP owns stdout.
    """
    scenes = meta.get("capture_scenes") or []
    fps = float(meta.get("fps") or 60)
    ev = geometry.load_events(events_path)

    per_scene_counts = []
    scenes_out = []
    for s_i, scene in enumerate(scenes):
        counts = []
        chans_out = []
        for ch in scene.get("channels") or []:
            path = os.path.join(session_dir, ch.get("file") or "")
            n_frames = 0
            if os.path.isfile(path):
                cap = cv2.VideoCapture(path)
                if cap.isOpened():
                    # A gapfree JOIN keeps survivors on ONE continuous file
                    # whose scenes reference SUB-RANGES -- the clock must see
                    # the slice length, not the whole file, or every healthy
                    # join take trips the per-seam overlap guard and reports
                    # full-file per-channel counts (same helpers render uses).
                    n_frames = _scene_channel_count(cap, ch)
                cap.release()
            counts.append(n_frames)
            chans_out.append({
                "file": ch.get("file"),
                "id": ch.get("id"),
                "app": ch.get("app"),
                "title": ch.get("title"),
                "rect": ch.get("rect"),
                "logical_w": ch.get("logical_w"),
                "logical_h": ch.get("logical_h"),
                "buffer_w": ch.get("buffer_w"),
                "buffer_h": ch.get("buffer_h"),
                "t0_monotonic": ch.get("t0_monotonic"),
                "track": ch.get("track"),
                "frame_start": _scene_channel_start(ch),
                "frame_count": n_frames,
            })
        per_scene_counts.append(counts)
        scenes_out.append({"index": s_i, "channels": chans_out})
    clock, aligns, warnings = segments.SegmentClock.from_scene_meta(
        meta, per_scene_counts, fps)
    for s_i, entry in enumerate(scenes_out):
        origin, n_out, offsets = aligns[s_i]
        entry["t0_monotonic"] = clock.t0s[s_i]
        entry["frame_count"] = int(n_out)
        entry["duration"] = float(clock.durs[s_i])
        for c_i, ch in enumerate(entry["channels"]):
            ch["frame_offset"] = offsets[c_i] if c_i < len(offsets) else 0
    duration = clock.duration()

    out_w, out_h = framing.output_size(
        _MULTI_NATIVE_DEFAULT_W, _MULTI_NATIVE_DEFAULT_H, "clean", aspect=None)

    info = {
        "session": os.path.basename(os.path.abspath(session_dir)),
        "session_dir": os.path.abspath(session_dir),
        "raw_path": None,           # no single raw -- the manifest is the truth
        "events_path": events_path,
        "fps": float(fps),
        "width": int(out_w),
        "height": int(out_h),
        "raw_width": int(out_w),
        "raw_height": int(out_h),
        "capture_window": None,
        "capture_windows": None,
        "capture_scenes": scenes_out,
        "scene_take": True,
        "scene_warnings": warnings,
        "frame_count": int(sum(a[1] for a in aligns)),
        "duration": float(duration),
        # A mic scene take is a gapfree join (pause+mic is refused), whose
        # channel 0 is one continuous file across the seam -- probe it. A
        # mic-off scene take (pause/resume) has no audio -> False, as before.
        "has_audio": bool(meta.get("mic_index") is not None
                          and _scene_has_continuous_audio(session_dir, meta)),
        "click_count": int(ev["clicks_t"].size),
        "move_count": int(ev["moves_t"].size),
        "key_count": int(ev["keys_t"].size),
        "scroll_count": int(ev["scrolls_t"].size),
        "key_capture": meta.get("key_capture"),
        "cursor_mode": str(meta.get("cursor_mode") or "system"),
        "has_face": False,
        "face_capture": None,
        "face_offset": 0.0,
        "has_output_mp4": os.path.isfile(
            os.path.join(session_dir, "output.mp4")),
        "has_output_gif": os.path.isfile(
            os.path.join(session_dir, "output.gif")),
    }
    if include_click_times:
        def _mt(arr):
            if not getattr(arr, "size", 0):
                return []
            t = clock.media(arr)
            t = t[np.isfinite(t)]
            return [float(x) for x in t.tolist()]
        info["click_times"] = _mt(ev["clicks_t"])
        info["key_times"] = _mt(ev["keys_t"])
        info["scroll_times"] = _mt(ev["scrolls_t"])
    return info


class _SkipTrack(Exception):
    """Internal: no capture crop, so no window track is needed."""


def session_beats(session_dir, info=None, edits_render=None, zooms=None,
                  max_beats=beats.MAX_BEATS, ev=None):
    """`beats.beat_sheet` for a session dir -- what happened, no frames decoded.

    Lives here rather than in beats.py so the points -> source-pixel mapping
    stays in ONE place: `beats` gets the SAME `to_src` that `preview_frame`
    and `render` use, built from the same window track. That matters beyond
    tidiness -- on a `--capture-window` session whose window moved, the crop
    origin is time-varying, and a static one puts every bbox off by the
    window's displacement (measured on a synthetic tracked session: bbox x
    160 against a truth of 60, on a 120px-wide frame). beats.py itself stays
    a pure function over arrays and takes the mapper as an argument.

    `zooms` is `edits.zooms` -- the ONLY thing every non-CLI surface plans
    the camera from -- so that a beat reports the zoom that will really
    render rather than the proposal that would have been made.

    Cheap enough to inline into a describe call: measured 47-56ms end to end
    on a 4-minute take with 1679 geometry samples, a good part of which is
    re-reading events.jsonl (pass `ev` to skip that). No cache, so nothing
    to invalidate. Cost scales with the geometry track -- see beats.py's
    module docstring for the curve.
    """
    meta, raw_path, events_path = _session_paths(session_dir)
    if info is None:
        info = describe_session(session_dir)
    if ev is None:
        ev = geometry.load_events(events_path)
    # Segmented take: hand beats the SAME clock render and describe use, so all
    # three cluster on the identical (gaps-deleted) output times -- the
    # trailing-cluster contract. `media_fn=None` (every other take) leaves
    # beats byte-identical.
    media_fn = None
    if segments.is_segmented_meta(meta):
        seg_counts = [int(s.get("frame_count") or 0)
                      for s in (info.get("capture_segments") or [])]
        fps = float(info.get("fps") or meta.get("fps") or 0.0)
        if seg_counts and fps > 0:
            media_fn = segments.SegmentClock.from_meta(
                meta, seg_counts, fps).media
    elif segments.is_scene_meta(meta):
        # Scene take: rebuild the clock from the DERIVED numbers describe
        # emitted -- the origin-ADJUSTED per-scene t0 plus the aligned
        # frame_count -- never from the manifest's raw channel t0s, which
        # would map every event `origin/fps` late per scene relative to the
        # render (the export-right/editor-wrong class the P1 review taught).
        scene_infos = info.get("capture_scenes") or []
        fps = float(info.get("fps") or meta.get("fps") or 0.0)
        if scene_infos and fps > 0:
            t0s = [float(s.get("t0_monotonic") or 0.0) for s in scene_infos]
            durs = [int(s.get("frame_count") or 0) / fps
                    for s in scene_infos]
            media_fn = segments.SegmentClock(t0s, durs).media
    elif _is_multi_native_meta(meta):
        # Flat fleet take: its composite starts at session frame `origin`, so
        # beats must cluster on the SAME clock `_render_multi_native` and
        # `_describe_multi_native_session` use -- the trailing-cluster contract
        # the three consumers share. Rebuilt from the DERIVED per-channel
        # `frame_offset` describe emitted, for the same reason the scene branch
        # above rebuilds from the derived per-scene t0. `origin` is 0 on most
        # takes, and this is then the identity map.
        chans = info.get("capture_channels") or []
        fps = float(info.get("fps") or meta.get("fps") or 0.0)
        offsets = [int(c.get("frame_offset") or 0) for c in chans]
        if offsets and fps > 0:
            comp_t0 = _multi_native_composite_t0(
                float(meta.get("t0_monotonic") or 0.0),
                max(0, max(offsets)), fps)

            def media_fn(arr, _t0=comp_t0):
                return arr - _t0
    raw_W = int(info.get("raw_width") or 0)
    raw_H = int(info.get("raw_height") or 0)
    crop = _capture_crop_px(meta, raw_W, raw_H)
    to_src = None
    # Only a record-time window crop makes the origin time-varying. Without
    # one, render's own fallback IS the static mapping (`ax * scale_x`), so
    # building a window track here would cost ~40ms to compute the same
    # numbers the cheap path already gives.
    try:
        if crop is None and not _is_window_native_meta(meta):
            raise _SkipTrack
        scale_x = raw_W / float(meta.get("logical_w", raw_W) or raw_W)
        scale_y = raw_H / float(meta.get("logical_h", raw_H) or raw_H)
        fps = float(info.get("fps") or 0.0)
        n_frames = int(info.get("frame_count") or 0)
        if fps > 0 and n_frames > 0:
            t0 = float(meta.get("t0_monotonic", 0.0) or 0.0)

            def _to_media(arr):
                return (arr - t0) if getattr(arr, "size", 0) else arr

            track = _build_window_track(
                ev, crop, raw_W, raw_H, scale_x, scale_y,
                np.arange(n_frames) / float(fps), _to_media, fps,
                window_id=_capture_window_id(meta), meta=meta)
            if track is not None:
                to_src = track.to_src
    except _SkipTrack:
        to_src = None   # no crop: the static origin below IS render's mapping
    except (TypeError, ValueError, ZeroDivisionError, KeyError):
        to_src = None   # static origin below is the correct fallback
    return beats.session_beats(ev, meta, info, crop=crop,
                               edits_render=edits_render,
                               max_beats=max_beats, to_src=to_src,
                               zooms=zooms, media_fn=media_fn)


def preview_frame(session_dir, t_sec, max_zoom=2.0, offset=0.0,
                  style="clean", params=None,
                  background=None, click_fx=True, click_params=None,
                  spotlight=False, spotlight_params=None, fade=0.0,
                  manual_zooms=None, suppressed_ranges=None,
                  cursor_fx=False, cursor_params=None, aspect=None,
                  motion_blur=True, typing_zoom=True, scroll_zoom=True,
                  facecam=True, facecam_params=None, windows=None,
                  window_follow=True, window_layout="grid",
                  window_zoom=False, window_focus=False, focus_ranges=None,
                  crop_rect=None, cursor_erase=False,
                  cursor_erase_params=None, screen_focus=True,
                  max_height=None, badge_erase=True):
    """Render a single preview frame at `t_sec` (seconds)."""
    meta, raw_path, events_path = _session_paths(session_dir)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open recording: " + raw_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps", 60))
    if src_fps <= 0:
        src_fps = float(meta.get("fps", 60) or 60.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        dur = _probe_duration(raw_path)
        if dur is None:
            cap.release()
            raise RuntimeError("cannot determine recording duration")
        n_frames = max(1, int(dur * src_fps) + 1)

    idx = int(round(max(0.0, float(t_sec)) * src_fps))
    idx = max(0, min(idx, n_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot decode frame {} from {}".format(idx, raw_path))

    # points -> recorded pixels, per-axis (exact from the real file).
    scale_x = W / float(meta.get("logical_w", W) or W)
    scale_y = H / float(meta.get("logical_h", H) or H)
    # Record-time window capture: crop the decoded frame and rebind (W, H) to
    # the window's size -- everything below then runs in WINDOW space.
    raw_W, raw_H = W, H
    crop = _capture_crop_px(meta, W, H)
    if crop is not None:
        crop_x, crop_y, W, H = crop

    t0 = meta.get("t0_monotonic", 0.0)

    ev = geometry.load_events(events_path)

    def to_media(arr):
        return (arr - t0) + offset if arr.size else arr

    track = _build_window_track(ev, crop, raw_W, raw_H, scale_x, scale_y,
                                np.arange(n_frames) / float(src_fps),
                                to_media, src_fps,
                                window_id=_capture_window_id(meta), meta=meta)
    if track is not None:
        fr = track.apply(fr, idx)
        to_src = track.to_src
    else:
        if crop is not None:
            fr = _apply_capture_crop(fr, crop)

            def to_src(_t, ax, ay):
                return ax * scale_x - crop_x, ay * scale_y - crop_y
        else:
            def to_src(_t, ax, ay):
                return ax * scale_x, ay * scale_y

    # preview == export. One-shot: a scrub seeks, so there is no forward walk
    # for the loop's verify step to ride -- the box comes from the memo
    # (`_badge_rect`), which is why dragging the playhead does not re-probe.
    _badge_erase_once(fr, raw_path, meta.get("capture_window"), raw_W,
                      enabled=badge_erase)

    # Editor crop, stage two -- see `_user_crop_px`.
    user_crop = _user_crop_px(crop_rect, W, H)
    if user_crop is not None:
        W, H = user_crop[2], user_crop[3]
        fr = _apply_user_crop(fr, user_crop)
        to_src = _user_crop_to_src(to_src, user_crop)

    clicks_t = to_media(ev["clicks_t"])
    clicks_x, clicks_y = to_src(clicks_t, ev["clicks_x"], ev["clicks_y"])
    moves_t = to_media(ev["moves_t"])
    moves_x, moves_y = to_src(moves_t, ev["moves_x"], ev["moves_y"])
    keys_t = to_media(ev["keys_t"]) if typing_zoom else np.array([])
    ups_t = to_media(ev["ups_t"])
    if scroll_zoom:
        scrolls_t = to_media(ev["scrolls_t"])
        scrolls_x, scrolls_y = to_src(scrolls_t, ev["scrolls_x"],
                                      ev["scrolls_y"])
    else:
        scrolls_t = scrolls_x = scrolls_y = np.array([])
    use_multi = bool(windows)
    typing_anchors = (None if use_multi else
                      _typing_anchor_hints(raw_path, events_path, keys_t,
                                          max_zoom, params, suppressed_ranges))
    typing_anchors = _offset_typing_anchors(typing_anchors, crop, track,
                                            user=user_crop)

    # Cursor eraser, on the source frame, before anything reads it. Bounded
    # in time (see `_preview_cursor_eraser`) -- the still is allowed to be
    # WORSE than the export here, never better.
    if cursor_erase and _erase_applies(meta):
        esx, esy = _erase_scale(track, scale_x, scale_y)
        e_boxes, e_hots, e_track = _erase_boxes(
            ev, to_media, to_src, np.arange(n_frames) / float(src_fps),
            W, H, esx, esy, cursor_erase_params)
        if e_boxes is not None:
            e = _preview_cursor_eraser(
                raw_path, e_boxes, e_hots, e_track, idx, src_fps,
                _erase_crop_fn(crop, track, user_crop),
                params=cursor_erase_params)
            if e is not None:
                # `fr` may still be a VIEW of the decoded frame here; the
                # eraser writes in place, and everything below reads `fr`
                # after this point, so the order is the whole contract.
                e.step(idx)
                e.erase(fr, idx)

    out_w, out_h = framing.output_size(W, H, style, aspect=aspect)
    out_w, out_h = framing.fit_max_height(out_w, out_h, max_height)
    t = idx / float(src_fps)

    if use_multi:
        multi_painter = framing.make_multi_painter(
            out_w, out_h, windows, background=background,
            layout=window_layout)
        grid_tracks = _build_grid_tracks(
            ev, windows, crop, track, W, H, scale_x, scale_y,
            np.arange(n_frames) / float(src_fps), to_media, src_fps,
            enabled=window_follow)
        multi_cells = multi_painter.cells
        # Like the card cameras below, planned over the FULL frame grid: a
        # trailing hold otherwise mistakes the preview instant for the end.
        focus_emphasis = _build_focus_emphasis(
            windows, grid_tracks, clicks_t, clicks_x, clicks_y,
            np.arange(n_frames) / float(src_fps), W, H, max_zoom, params,
            suppressed_ranges, n_frames / float(src_fps), src_fps,
            manual=focus_ranges, enabled=window_focus)
        focus_layout = (None if focus_emphasis is None
                        else _FocusLayout(multi_painter, focus_emphasis,
                                          lean=camera.build_params(max_zoom,
                                                              params).focus_lean))
        multi_cells, focus_order = _focus_cells_at(
            focus_layout, idx, multi_cells)
        # The FULL frame grid, not a prefix: a card's plan must reason about
        # the real end of clip or a trailing hold mistakes the preview instant
        # for the end (see build_path's plan_duration).
        card_paths = _build_card_cameras(
            windows, multi_cells, grid_tracks, clicks_t, clicks_x, clicks_y,
            np.arange(n_frames) / float(src_fps), W, H, max_zoom, params,
            suppressed_ranges, n_frames / float(src_fps), src_fps,
            enabled=window_zoom)
        crops = _multi_crops(fr, windows, grid_tracks, idx, W, H)
        crops = _apply_card_cameras(crops, card_paths, windows, multi_cells,
                                    idx, W, H)
        out = _paint_multi(multi_painter, crops, multi_cells,
                           focus_order, focus_layout)
        if _cursor_fx_draws(cursor_fx, meta, cursor_erase):
            # Causal smoothing (see CursorFX's docstring), so simulating only
            # the prefix up to this frame gives the same cursor a full render
            # would put here.
            cursorfx = effects.CursorFX(
                W, H, np.arange(idx + 1) / float(src_fps),
                moves_t, moves_x, moves_y, clicks_t=clicks_t,
                params=cursor_params)
            _draw_multi_cursor(out, cursorfx, multi_cells, windows,
                               grid_tracks, idx, W, H, card_paths=card_paths)
        if focus_layout is not None:
            out = _apply_focus_camera(out, focus_layout.camera_at(idx),
                                      out_w, out_h)
        _warn_ignored_multi_window_options(
            motion_blur=motion_blur, click_fx=click_fx, spotlight=spotlight)
    else:
        win_w, win_h = _contain_fit(out_w, out_h, W, H)

        # Simulate one frame past the preview index (when the clip has one) so
        # the motion-blur central difference has a forward neighbor -- the still
        # then blurs exactly like the same frame would in a full render().
        sim_count = min(idx + 2, n_frames)
        frame_times = np.arange(sim_count) / float(src_fps)
        # Whole-screen "grow the active window": same reshape the export runs,
        # so the rendered still matches. The ctx must span the FULL clip (like
        # the export's), NOT the simulated prefix -- the resolver reads owner
        # geometry over the whole reshaped range, which extends past the
        # previewed frame; a prefix ctx would clamp those queries and anchor a
        # moved window differently than the export. Only the SPRING is a prefix.
        screen_ctx = None
        screen_resolver = None
        screen_spans = None
        if screen_focus and crop is None and track is None:
            full_ft = np.arange(n_frames) / float(src_fps)
            screen_ctx = _build_screen_focus_ctx(ev, to_media, to_src,
                                                 full_ft, W, H, src_fps)
            if screen_ctx is not None:
                _sp = camera.build_params(max_zoom, params)
                screen_resolver, screen_spans = _make_screen_focus_resolver(
                    screen_ctx, win_w, win_h, _sp.overview_fill, src_fps, out_w)
        path = camera.build_path(frame_times, W, H,
                                 clicks_t, clicks_x, clicks_y,
                                 moves_t, moves_x, moves_y,
                                 max_zoom=max_zoom, fps=src_fps,
                                 params=params or {},
                                 manual_zooms=manual_zooms,
                                 suppressed_ranges=suppressed_ranges,
                                 keys_t=keys_t, ups_t=ups_t,
                                 scrolls_t=scrolls_t, scrolls_x=scrolls_x,
                                 scrolls_y=scrolls_y,
                                 typing_anchors=typing_anchors,
                                 win_w=win_w, win_h=win_h,
                                 # We only simulate a prefix (up to the preview
                                 # frame) for speed, but the zoom plan itself must
                                 # still reason about the *real* clip length --
                                 # otherwise _settle_tail mistakes "now" for "the
                                 # end" and prematurely flattens an in-progress
                                 # zoom back to 1.0. See build_path's docstring.
                                 plan_duration=n_frames / float(src_fps),
                                 window_resolver=screen_resolver)
        screen_track = None
        if screen_ctx is not None and screen_spans:
            screen_track = _build_screen_focus_track(
                screen_spans, screen_ctx, path, src_fps,
                camera.build_params(max_zoom, params).always_zoomed)

        cx, cy, z = path[idx]
        x0, y0 = _camera_window(cx, cy, z, win_w, win_h, W, H)
        if motion_blur:
            cams = _motion_blur_cams(path, idx, win_w, win_h, out_w, out_h)
            out = _warp_blend(fr, cams, win_w, win_h, out_w, out_h, W, H)
        else:
            out = _warp(fr, x0, y0, z, win_w, win_h, out_w, out_h)
        z_eff = _zoom_to_output_scale(z, win_w, out_w)

        clickfx = (effects.ClickFX(W, H, clicks_t, clicks_x, clicks_y, click_params)
                   if click_fx and clicks_t.size else None)
        if clickfx is not None:
            clickfx.draw(out, t, x0, y0, z_eff)
        if spotlight:
            if moves_t.size >= 1:
                cur_x = float(np.interp(t, moves_t, moves_x,
                                        left=moves_x[0], right=moves_x[-1]))
                cur_y = float(np.interp(t, moves_t, moves_y,
                                        left=moves_y[0], right=moves_y[-1]))
            else:
                cur_x, cur_y = W / 2.0, H / 2.0
            effects.Spotlight(W, H, spotlight_params).draw(
                out, (cur_x - x0) * z_eff, (cur_y - y0) * z_eff, z_eff)

        if _cursor_fx_draws(cursor_fx, meta, cursor_erase):
            cursorfx = effects.CursorFX(W, H, frame_times, moves_t, moves_x, moves_y,
                                        clicks_t=clicks_t, params=cursor_params)
            cursorfx.draw(out, idx, x0, y0, z_eff)

        if screen_track is not None:
            out = _apply_screen_grow(out, screen_track, idx, x0, y0, z_eff)

        painter = framing.make_painter(out_w, out_h, style, background=background)
        if painter is not None:
            out = painter.paint(out)

    facecam_ov = _facecam_overlay(session_dir, meta, out_w, out_h,
                                  enabled=facecam, params=facecam_params)
    if facecam_ov is not None:
        facecam_ov.draw(out, t)
        facecam_ov.release()
    if fade > 0.0:
        total_dur = n_frames / float(src_fps)
        out = _apply_fade(out, t, total_dur, fade)
    return out


def source_frame(session_dir, t_sec, offset=0.0, crop_rect=None):
    """The decoded frame at `t_sec` in SOURCE coordinate space: no camera, no
    effects, no framing.

    This is the space `edits.windows` rects and zoom pins are expressed in,
    and the space `describe_session` reports as `width`/`height`.
    `preview_frame` CANNOT serve that purpose -- it returns the auto-zoomed,
    framed OUTPUT, so coordinates read off it are wrong by whatever the camera
    happened to be doing at that instant. Both images look equally plausible,
    which is exactly what makes the mistake silent, so placing a rect needs
    this and not that.

    A record-time window capture IS applied (following the geometry track when
    there is one), because that crop is what *defines* this session's source
    space -- `describe_session` already reports the cropped dims.
    """
    meta, raw_path, events_path = _session_paths(session_dir)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open recording: " + raw_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps", 60))
    if src_fps <= 0:
        src_fps = float(meta.get("fps", 60) or 60.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames <= 0:
        dur = _probe_duration(raw_path)
        n_frames = max(1, int(dur * src_fps) + 1) if dur else 1
    idx = int(round(max(0.0, float(t_sec)) * src_fps))
    idx = max(0, min(idx, n_frames - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("cannot decode frame {} from {}".format(idx, raw_path))

    crop = _capture_crop_px(meta, W, H)
    if crop is None:
        # No record-time capture crop: the editor's crop is the only stage,
        # and stays identity when it is unset.
        return _apply_user_crop(fr, _user_crop_px(crop_rect, W, H))
    raw_W, raw_H = W, H
    scale_x = W / float(meta.get("logical_w", W) or W)
    scale_y = H / float(meta.get("logical_h", H) or H)
    t0 = meta.get("t0_monotonic", 0.0)
    ev = geometry.load_events(events_path)

    def to_media(arr):
        return (arr - t0) + offset if arr.size else arr

    track = _build_window_track(ev, crop, raw_W, raw_H, scale_x, scale_y,
                                np.arange(n_frames) / float(src_fps),
                                to_media, src_fps,
                                window_id=_capture_window_id(meta), meta=meta)
    if track is not None:
        fr = track.apply(fr, idx)
    else:
        fr = _apply_capture_crop(fr, crop)
    return _apply_user_crop(fr, _user_crop_px(crop_rect, fr.shape[1],
                                              fr.shape[0]))


def multi_window_layout(session_dir, windows, background=None, style="clean",
                        aspect=None, window_layout="grid", crop_rect=None,
                        max_height=None):
    """Grid geometry + background plate for the multi-window composite.

    `render()`/`preview_frame()` composite this layout with NumPy; the editor's
    live playback has to redraw it in the browser instead (a <video> element
    can only ever show ONE crop), so it needs the same geometry plus something
    to draw it on. Both come straight off `framing.MultiFramePainter`, which
    keeps that class the only place the grid is ever computed.

    Returns `{"canvas": (w, h), "cells": [...], "plate": BGR ndarray}`. Each
    cell carries its destination rect + corner radius in canvas px, and a
    `src` rect in RAW FILE pixels -- raw, not window space, because that is
    the coordinate system of the <video> element the browser samples from.
    """
    if not windows:
        raise ValueError("multi_window_layout requires at least one window")
    meta, raw_path, _events_path = _session_paths(session_dir)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open recording: " + raw_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Same rebind into window space that preview_frame does, so a
    # window-captured session gets the grid it will actually render with.
    crop = _capture_crop_px(meta, W, H)
    if crop is not None:
        crop_x, crop_y, W, H = crop
    else:
        crop_x = crop_y = 0

    # Editor crop, stage two. `cell["src"]` stays in RAW FILE px (the browser
    # samples the <video> element in that space), so the crop's origin adds
    # into the same offset the capture crop already contributes.
    user_crop = _user_crop_px(crop_rect, W, H)
    if user_crop is not None:
        crop_x += user_crop[0]
        crop_y += user_crop[1]
        W, H = user_crop[2], user_crop[3]

    out_w, out_h = framing.output_size(W, H, style, aspect=aspect)
    out_w, out_h = framing.fit_max_height(out_w, out_h, max_height)
    painter = framing.make_multi_painter(out_w, out_h, windows,
                                         background=background,
                                         layout=window_layout)
    cells = painter.cells
    for cell, win in zip(cells, windows):
        x, y, w, h = _clamp_window_rect(win, W, H)
        cell["src"] = [x + crop_x, y + crop_y, w, h]
    return {"canvas": (int(out_w), int(out_h)), "cells": cells,
            "plate": painter.base_plate(),
            # Shadow-free backdrop + the painter itself, for window focus:
            # moving cells mean the baked shadows are wrong, so the browser
            # draws its own over this, and `multi_window_focus_cells` needs
            # the painter to compute the same placements the export will.
            "background": painter.background_plate(),
            "painter": painter}


# A point rect and its capture buffer describe the SAME window, so their
# aspects must agree. Beyond this much relative disagreement we treat the rect
# as clipped -- see `_channel_rect`.
_RECT_ASPECT_TOL = 0.01


def _unclip_point_size(lw, lh, bw, bh):
    """`(lw, lh)` corrected against the capture buffer when the recorded rect
    was display-CLIPPED, and returned UNCHANGED when it was not.

    The shared half of `_channel_rect` (which also has to place an origin) and
    of the two window-native track builders (which only need the size, as the
    points->pixels denominator). Clipping can only ever shrink a rect, so the
    clipped axis reports an inflated `buffer/points` ratio and the SMALLER
    ratio is the true backing scale.

    Recording writes unclipped rects as of 2026-08-31
    (`record.Recorder._window_native_space`); this is what lets takes made
    BEFORE that still render and map clicks correctly.
    """
    try:
        lw, lh, bw, bh = float(lw), float(lh), float(bw), float(bh)
    except (TypeError, ValueError):
        return lw, lh
    if not (lw > 0.0 and lh > 0.0 and bw > 1.0 and bh > 1.0):
        return lw, lh
    rx, ry = bw / lw, bh / lh
    lo = min(rx, ry)
    if lo <= 0.0 or not (np.isfinite(rx) and np.isfinite(ry)):
        return lw, lh
    if abs(rx - ry) <= _RECT_ASPECT_TOL * lo:
        return lw, lh                   # agree -> untouched, the common path
    return bw / lo, bh / lo


def _channel_rect(ch, dim):
    """One occlusion-free channel's card rect, in POINTS, reconciled against
    what the capture file actually contains.

    WHY this is not just `ch["rect"]`: the recorded rect comes from
    `devices._filter_windows`, which INTERSECTS every window with its display
    -- correct for the display-crop window pick (an avfoundation crop can only
    ever cover pixels the capture contains), and wrong here, because SCK
    captures the window's whole frame including the part hanging off the edge
    of the screen. When they disagree the card is drawn stretched: the cell is
    built from the clipped aspect and the full buffer is resized into it.

    Measured on recordings/20260831-103923: a Word window at
    `[723, 57, 717, 835]` -- note 723 + 717 = 1440, exactly the display width
    -- against a 2764x1670 buffer. Aspect 0.859 vs 1.655; the card came out
    squeezed to about half its true width, and every other channel in the same
    take agreed to 4 decimal places.

    Recovery is exact rather than heuristic, because clipping can only ever
    make a rect SMALLER: a clipped axis therefore reports an INFLATED
    buffer/rect ratio, so the SMALLER of the two ratios is the true backing
    scale and `buffer / scale` is the true size in points. (Both axes clipped
    -- a window larger than the display in both directions -- is unrecoverable;
    min() is still the least-wrong answer available.)

    The ORIGIN moves only when the clip is provably on the left/top edge, i.e.
    the rect starts flush against the display. A right/bottom clip leaves the
    origin alone, which is the common case (a window dragged off the right
    edge). `desktop` is the only layout that reads origins at all.

    Returns the rect UNTOUCHED whenever the aspects agree, which is every
    well-behaved channel -- so this is bit-exact for every take that does not
    have the bug.
    """
    rect = ch.get("rect") or [0.0, 0.0, ch.get("logical_w") or 1.0,
                              ch.get("logical_h") or 1.0]
    out = {"x": float(rect[0]), "y": float(rect[1]),
           "w": float(rect[2]), "h": float(rect[3])}
    try:
        bw, bh = float(dim[0]), float(dim[1])
    except (TypeError, ValueError, IndexError):
        return out
    if not (bw > 1.0 and bh > 1.0 and out["w"] > 0.0 and out["h"] > 0.0):
        return out
    true_w, true_h = _unclip_point_size(out["w"], out["h"], bw, bh)
    if true_w == out["w"] and true_h == out["h"]:
        return out                      # agree -> the untouched common path
    ox = float((ch.get("display_origin") or (0.0, 0.0))[0])
    oy = float((ch.get("display_origin") or (0.0, 0.0))[1])
    # Flush against the display's left/top edge means the clip took the part
    # BEFORE the origin, so the true window starts further back.
    if true_w > out["w"] and abs(out["x"] - ox) <= 1.0:
        out["x"] = out["x"] - (true_w - out["w"])
    if true_h > out["h"] and abs(out["y"] - oy) <= 1.0:
        out["y"] = out["y"] - (true_h - out["h"])
    out["w"], out["h"] = true_w, true_h
    return out


def _multi_native_canvas(rects, dims, style, aspect, window_layout,
                         max_height=None):
    """Canvas big enough to show the window buffers at ~1:1.

    The default was a hardcoded 1920x1080, and on a Retina Mac that quietly
    threw away two thirds of every window: a 1200px-wide buffer landed in a
    664px cell -- 0.55x, ~31% of the captured pixels -- and the text came out
    mushy for reasons nothing in the output hinted at. The whole-screen path
    never had this problem because `framing.output_size` passes the REAL
    source dimensions through, so "auto" renders at native resolution.

    So size from the sources here too: lay the cards out once at the reference
    canvas (cheaply, via `framing.placements_for`), find the card that ends up
    downscaled the most, and scale the canvas by exactly that factor. Cells
    scale linearly with the canvas, so one measurement is enough.

    Capped, because N Retina windows can ask for more than 4K and the encode
    cost is real; a capped take is merely as soft as it used to be, never
    softer. An explicit `--aspect WxH` is an exact request and wins outright.
    """
    base_w, base_h = framing.output_size(
        _MULTI_NATIVE_DEFAULT_W, _MULTI_NATIVE_DEFAULT_H, style, aspect=aspect)
    if not dims:
        return framing.fit_max_height(base_w, base_h, max_height)
    # An exact "WxH" aspect means the caller has named the canvas; honour it
    # (the resolution clamp still applies -- it only ever scales DOWN, and a
    # user asking for both an exact size and a smaller cap gets the cap).
    if isinstance(aspect, str) and re.match(r"^\s*\d+\s*x\s*\d+\s*$", aspect):
        return framing.fit_max_height(base_w, base_h, max_height)
    try:
        cells = framing.placements_for(base_w, base_h, rects,
                                       layout=window_layout)
    except Exception:
        return base_w, base_h
    need = 1.0
    for (_ix, _iy, fw, fh), (bw, bh) in zip(cells, dims):
        if fw > 0:
            need = max(need, float(bw) / float(fw))
        if fh > 0:
            need = max(need, float(bh) / float(fh))
    if need <= 1.0 + 1e-6:
        return framing.fit_max_height(base_w, base_h, max_height)
    need = min(need,
               _MULTI_NATIVE_MAX_DIM / float(max(base_w, base_h)),
               (_MULTI_NATIVE_MAX_PIXELS / float(base_w * base_h)) ** 0.5)
    if need <= 1.0 + 1e-6:
        return framing.fit_max_height(base_w, base_h, max_height)
    # The resolution clamp lands LAST -- it caps the sharpness-driven
    # scale-up, which is exactly the 4K-forcing path a user wants to escape.
    return framing.fit_max_height(
        framing._even(base_w * need), framing._even(base_h * need),
        max_height)


def _stamp_channel_layouts(rects, channel_layouts):
    """Stamp manual per-card placement (0-1 CANVAS fractions) onto the painter
    rects for a multi-native composite. POSITIONAL: slot i -> card i.

    Must run AFTER the canvas is sized (from the UN-stamped rects) and BEFORE
    `make_multi_painter`: `framing._apply_card_overrides` (framing.py:1061)
    then honors `rect['layout']` on top of whatever preset placed the base
    cell, while the canvas resolution stays independent of the override -- so a
    manual resize never changes the output size. A no-op when `channel_layouts`
    is empty/None (the off-switch: byte-identical to the pre-feature render).
    """
    if not channel_layouts:
        return
    for i, lay in enumerate(channel_layouts):
        if i < len(rects) and isinstance(lay, dict):
            rects[i]["layout"] = lay


def multi_native_layout(session_dir, background=None, style="clean",
                        aspect=None, window_layout="grid", max_height=None,
                        channel_layouts=None, badge_erase=True):
    """Composite geometry + plate for a MULTI-NATIVE session's live preview.

    The N-source analog of `multi_window_layout`, and the reason the editor
    could not play these takes at all: a multi-native session has no single
    `raw.mov` -- each window is its own `raw_i.mov` -- so one <video> element
    has nothing to point at. The browser needs N of them, the cell each is
    drawn into, and the per-channel time offset that puts them on one clock.

    Geometry comes off the same `framing.MultiFramePainter` the export builds,
    from the same POINT rects, so preview and export frame identically -- the
    single-source-of-truth rule `multi_window_layout` already follows.

    Each cell's `src` is the WHOLE channel buffer. That is the real difference
    from the display-crop path, where a cell samples a sub-rect of one shared
    recording: here the recording IS the window.

    `start` is per channel, in seconds, and mirrors `_render_multi_native`'s
    alignment exactly -- channels start at slightly different wall-clock
    instants, so each is skipped forward to the shared origin. `duration` is
    the shared overlap, clipped at whichever channel runs out first.
    """
    meta_path = os.path.join(session_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    channels = meta.get("capture_channels") or []
    if len(channels) < 2:
        raise ValueError("not a multi-window native session")

    src_fps = float(meta.get("fps") or 60)
    session_t0 = float(meta.get("t0_monotonic") or 0.0)
    rects, offsets, counts, dims = [], [], [], []
    for ch in channels:
        path = os.path.join(session_dir, ch["file"])
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError("cannot open channel file: " + path)
        try:
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            dims.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        finally:
            cap.release()
        ch_t0 = float(ch.get("t0_monotonic") or session_t0)
        offsets.append(int(round((ch_t0 - session_t0) * src_fps)))
        # POINTS, exactly as the export passes them -- `desktop` keys off the
        # origins to reproduce the on-screen arrangement. Reconciled against
        # the decoded buffer so a display-clipped rect can't stretch the card
        # (`_channel_rect`).
        rects.append(_channel_rect(ch, dims[-1]))

    origin = max(0, max(offsets)) if offsets else 0
    remaining = [counts[i] - max(0, origin - offsets[i])
                 for i in range(len(counts))]
    n_out = min(remaining) if remaining else 0

    out_w, out_h = _multi_native_canvas(rects, dims, style, aspect,
                                        window_layout, max_height=max_height)
    # Manual card placement, stamped AFTER the canvas is sized so a resize
    # never changes the output resolution (framing._apply_card_overrides).
    _stamp_channel_layouts(rects, channel_layouts)
    painter = framing.make_multi_painter(out_w, out_h, rects,
                                         background=background,
                                         layout=window_layout)
    cells = painter.cells
    out_channels = []
    for i, ch in enumerate(channels):
        bw, bh = dims[i]
        if i < len(cells):
            cells[i]["src"] = [0, 0, bw, bh]
        out_channels.append({
            "index": i,
            "media": "channel{}".format(i),
            "buffer_w": bw,
            "buffer_h": bh,
            # Seconds to skip so this channel sits at the shared origin.
            "start": round(max(0, origin - offsets[i]) / src_fps, 4),
            "app": ch.get("app"),
            # The capture-indicator repair, in this channel's BUFFER pixels.
            # The live player composites in the browser, so it has to paint
            # the corner itself or play would show the badge the paused
            # scrub and the export both hide.
            "badge": _badge_patch(
                os.path.join(session_dir, ch.get("file", "")), ch, bw,
                enabled=badge_erase),
        })
    return {"canvas": (int(out_w), int(out_h)), "cells": cells,
            "channels": out_channels,
            "fps": float(src_fps),
            "duration": float(n_out / src_fps) if src_fps else 0.0,
            "plate": painter.base_plate(),
            "background": painter.background_plate(),
            "painter": painter}


class _SourceSpace(object):
    """The decoded-file facts + event tracks that `preview_frame`/`render`
    derive before doing any work, for the LIVE-PREVIEW helpers below.

    Those helpers hand the editor things the browser cannot compute (the
    cursor track, the per-card cameras) and each needs the same preamble:
    dimensions, fps, the capture crop, and events mapped into source space.
    Deriving it once here keeps a third and fourth hand-copy of that preamble
    from drifting away from the two real render paths.
    """

    def __init__(self, session_dir, offset=0.0, crop_rect=None):
        self.ok = False
        meta, raw_path, events_path = _session_paths(session_dir)
        self.meta = meta
        cap = cv2.VideoCapture(raw_path)
        if not cap.isOpened():
            return
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps", 60))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if src_fps <= 0:
            src_fps = float(meta.get("fps", 60) or 60.0)
        if n_frames <= 0:
            dur = _probe_duration(raw_path)
            if dur is None:
                return
            n_frames = max(1, int(dur * src_fps) + 1)

        scale_x = W / float(meta.get("logical_w", W) or W)
        scale_y = H / float(meta.get("logical_h", H) or H)
        raw_W, raw_H = W, H
        crop = _capture_crop_px(meta, W, H)
        crop_x = crop_y = 0
        if crop is not None:
            crop_x, crop_y, W, H = crop
        t0 = meta.get("t0_monotonic", 0.0)
        ev = geometry.load_events(events_path)

        def to_media(arr):
            return (arr - t0) + offset if arr.size else arr

        frame_times = np.arange(n_frames) / float(src_fps)
        track = _build_window_track(ev, crop, raw_W, raw_H, scale_x, scale_y,
                                    frame_times, to_media, src_fps,
                                    window_id=_capture_window_id(meta), meta=meta)
        if track is not None:
            to_src = track.to_src
        elif crop is not None:
            def to_src(_t, ax, ay):
                return ax * scale_x - crop_x, ay * scale_y - crop_y
        else:
            def to_src(_t, ax, ay):
                return ax * scale_x, ay * scale_y

        # Editor crop, stage two -- see `_user_crop_px`. Its origin folds into
        # crop_x/crop_y because those exist to map back into RAW file px, and
        # the two crops stack in that direction. (With a geometry track that
        # mapping was already a snapshot-origin approximation; this does not
        # change that.)
        user_crop = _user_crop_px(crop_rect, W, H)
        self.user_crop = user_crop
        # The space as the CAPTURE stage left it. `_build_grid_tracks` callers
        # re-derive the capture crop against these, so they keep passing the
        # exact dims they passed before this field existed.
        self.capture_W, self.capture_H = W, H
        if user_crop is not None:
            crop_x += user_crop[0]
            crop_y += user_crop[1]
            W, H = user_crop[2], user_crop[3]
            to_src = _user_crop_to_src(to_src, user_crop)

        self.W, self.H = W, H
        self.src_fps = float(src_fps)
        self.n_frames = int(n_frames)
        # Points -> recorded pixels. Load-bearing for every helper that
        # tracks windows: the geometry samples in events.jsonl are in POINTS,
        # so on a Retina take this is 2.0 and hardcoding 1.0 puts every
        # tracked window at half its true position -- cards then follow their
        # windows into the top-left quadrant and the clicks (correctly
        # scaled) land in none of them. See `_build_grid_tracks`.
        self.scale_x, self.scale_y = scale_x, scale_y
        self.crop_x, self.crop_y = crop_x, crop_y
        self.ev = ev
        self.frame_times = frame_times
        self.to_media = to_media
        self.to_src = to_src
        self.ok = True

    def clicks(self):
        t = self.to_media(self.ev["clicks_t"])
        x, y = self.to_src(t, self.ev["clicks_x"], self.ev["clicks_y"])
        return t, x, y

    def moves(self):
        t = self.to_media(self.ev["moves_t"])
        x, y = self.to_src(t, self.ev["moves_x"], self.ev["moves_y"])
        return t, x, y


def multi_window_card_paths(session_dir, windows, cells, offset=0.0,
                            window_zoom=False, max_zoom=2.0, params=None,
                            suppressed_ranges=None, window_follow=True,
                            stride=2, crop_rect=None):
    """Sampled per-card camera paths for the editor's live composite, or None.

    `window_zoom` makes each card zoom on its own clicks, which the browser
    composites by varying the SOURCE rect it samples per cell -- so this hands
    over `(cx, cy, z)` per card and the JS turns it into that rect with the
    same `_camera_window` arithmetic. Without it the live composite would keep
    showing static crops while the export zoomed, which is the paused-vs-
    playing divergence this whole area just got fixed for.

    Entries are None for cards that never zoom, so a card the camera never
    picked keeps the plain static crop in the browser too.
    """
    if not window_zoom or not windows:
        return None
    space = _SourceSpace(session_dir, offset=offset,
                         crop_rect=crop_rect)
    if not space.ok:
        return None
    clicks_t, clicks_x, clicks_y = space.clicks()
    if not clicks_t.size:
        return None
    grid_tracks = _build_grid_tracks(
        space.ev, windows, _capture_crop_px(space.meta, space.capture_W,
                                            space.capture_H),
        None, space.W, space.H, space.scale_x, space.scale_y,
        space.frame_times, space.to_media, space.src_fps,
        enabled=window_follow)
    paths = _build_card_cameras(
        windows, cells, grid_tracks, clicks_t, clicks_x, clicks_y,
        space.frame_times, space.W, space.H, max_zoom, params,
        suppressed_ranges, space.n_frames / space.src_fps, space.src_fps,
        enabled=True)
    stride = max(1, int(stride))
    sel = slice(None, None, stride)
    out = []
    for path in paths:
        if path is None:
            out.append(None)
            continue
        out.append({
            "cx": [round(float(v), 2) for v in path[sel, 0]],
            "cy": [round(float(v), 2) for v in path[sel, 1]],
            "z": [round(float(v), 4) for v in path[sel, 2]],
        })
    if not any(p is not None for p in out):
        return None
    return {"stride": stride, "fps": space.src_fps, "cards": out}


def multi_window_focus_ranges(session_dir, windows, offset=0.0,
                              max_zoom=2.0, params=None,
                              suppressed_ranges=None, window_follow=True,
                              crop_rect=None):
    """The window-focus plan as editable spans, for materializing into
    `edits.focus` (`edits.initialize_focus_ranges`).

    Not gated on `window_focus`: the plan is what the editor shows the user
    so they can decide, and inert until the toggle is on.
    """
    if not windows:
        return []
    space = _SourceSpace(session_dir, offset=offset,
                         crop_rect=crop_rect)
    if not space.ok:
        return []
    clicks_t, clicks_x, clicks_y = space.clicks()
    if not clicks_t.size:
        return []
    grid_tracks = _build_grid_tracks(
        space.ev, windows, _capture_crop_px(space.meta, space.capture_W,
                                            space.capture_H),
        None, space.W, space.H, space.scale_x, space.scale_y,
        space.frame_times, space.to_media, space.src_fps,
        enabled=window_follow)
    times = _card_click_times(windows, grid_tracks, clicks_t, clicks_x,
                              clicks_y, space.W, space.H, space.src_fps)
    cards = [{"click_times": ts} for ts in times]
    return camera.plan_focus_ranges(
        cards, max_zoom=max_zoom, params=params,
        suppressed_ranges=suppressed_ranges,
        plan_duration=space.n_frames / space.src_fps)


def multi_window_focus_cells(session_dir, windows, painter, offset=0.0,
                             window_focus=False, max_zoom=2.0, params=None,
                             suppressed_ranges=None, window_follow=True,
                             focus_ranges=None, stride=2, crop_rect=None):
    """Sampled per-frame CELL geometry for the editor's live composite, or None.

    The browser gets finished placements rather than the emphasis track,
    deliberately: `framing.focus_placements` + `blend_placements` then stay
    the single place a layout is ever decided, and the JS side only has to
    lerp between the boxes it is handed -- which it already knows how to
    draw. Re-deriving the layout in JS is exactly how the paused server
    render and the playing canvas would drift apart.

    Shape mirrors `multi_window_layout`'s `cells`: `frames[i][card]` is
    `[x, y, w, h, radius]` in canvas px, plus the draw order and -- on the
    frames where anything has actually grown -- each card's `lift_fraction`,
    which is what its drop shadow is scaled by. The lift is computed here
    rather than in JS for the same reason the placements are: one place
    decides, so the paused server render and the playing canvas cannot
    disagree about how far off the plane a card is.
    """
    if not window_focus or not windows:
        return None
    space = _SourceSpace(session_dir, offset=offset,
                         crop_rect=crop_rect)
    if not space.ok:
        return None
    clicks_t, clicks_x, clicks_y = space.clicks()
    if not clicks_t.size and focus_ranges is None:
        return None
    grid_tracks = _build_grid_tracks(
        space.ev, windows, _capture_crop_px(space.meta, space.capture_W,
                                            space.capture_H),
        None, space.W, space.H, space.scale_x, space.scale_y,
        space.frame_times, space.to_media, space.src_fps,
        enabled=window_follow)
    emphasis = _build_focus_emphasis(
        windows, grid_tracks, clicks_t, clicks_x, clicks_y,
        space.frame_times, space.W, space.H, max_zoom, params,
        suppressed_ranges, space.n_frames / space.src_fps, space.src_fps,
        manual=focus_ranges, enabled=True)
    if emphasis is None:
        return None
    layout = _FocusLayout(painter, emphasis)
    stride = max(1, int(stride))
    frames = []
    for i in range(0, len(emphasis), stride):
        cells, order = layout.cells_at(i)
        cam = layout.camera_at(i)
        lifts = [round(painter.lift_of(j, c["w"]), 3)
                 for j, c in enumerate(cells)]
        frame = {
            "c": [[c["x"], c["y"], c["w"], c["h"], c["radius"]]
                  for c in cells],
            "o": list(order),
            "z": ([round(float(cam[0]), 1), round(float(cam[1]), 1),
                   round(float(cam[2]), 4)] if cam else None),
        }
        # Omitted on the (many) frames where nothing has grown -- the browser
        # reads a missing `l` as all-zero, which is the resting shadow.
        if any(v > 0.0 for v in lifts):
            frame["l"] = lifts
        frames.append(frame)
    return {"stride": stride, "fps": space.src_fps, "frames": frames}


def _native_camera_inputs(session_dir, offset=0.0):
    """The multi-native preamble both live-preview emitters below share.

    `_render_multi_native` derives the exact same values (per-channel tracks,
    the shared click set on the media clock, the aligned output-frame grid)
    before it composites; the display-crop path factors its equivalent into
    `_SourceSpace` for precisely this reason -- a third and fourth hand-copy
    of the setup would drift away from the render it must match numerically
    (`test_live_preview_track_matches_the_render`). So compute it once here.

    Returns a dict, or None when the session isn't multi-native or has no
    overlapping frames. Mirrors `_render_multi_native`'s alignment exactly:
    per-channel `t0_monotonic - session_t0` quantized to a frame offset, the
    shared origin at `max(offsets)`, and `n_out` the shortest channel's
    remaining frames -- so `frame_times` (and every path indexed by it) lines
    up with the export frame-for-frame.
    """
    meta_path = os.path.join(session_dir, "meta.json")
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    channels = meta.get("capture_channels") or []
    if len(channels) < 2:
        return None
    src_fps = float(meta.get("fps") or 60)
    session_t0 = float(meta.get("t0_monotonic") or 0.0)

    counts, frame_offsets = [], []
    for ch in channels:
        path = os.path.join(session_dir, ch.get("file", ""))
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None
        try:
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        finally:
            cap.release()
        ch_t0 = float(ch.get("t0_monotonic") or session_t0)
        frame_offsets.append(int(round((ch_t0 - session_t0) * src_fps)))
    origin = max(0, max(frame_offsets)) if frame_offsets else 0
    remaining = [counts[i] - max(0, origin - frame_offsets[i])
                 for i in range(len(counts))]
    n_out = min(remaining) if remaining else 0
    if n_out <= 0:
        return None

    frame_times = np.arange(n_out, dtype=float) / float(src_fps)
    events_path = os.path.join(session_dir, meta.get("events", "events.jsonl"))
    ev = geometry.load_events(events_path)

    composite_t0 = _multi_native_composite_t0(session_t0, origin, src_fps)

    def to_media(arr):
        arr = np.asarray(arr, dtype=float)
        # Match `_render_multi_native`: the composite's clock is
        # `t - (session_t0 + origin/fps)`, not `t - session_t0`
        # (`_multi_native_composite_t0`). The two preambles are hand-copies,
        # so the anchor has to be changed in both or the browser animates a
        # card path the export doesn't composite.
        #
        # They are NOT identical in general, and this is where they part:
        # `offset` is folded in here the way the display-crop preview does it,
        # but `render()` never passes it to `_render_multi_native` (its
        # signature absorbs it), so a non-zero offset moves the preview and
        # not the export. It is 0 in the editor flow, which is the only flow
        # that reaches this -- so the two coincide in practice, not by
        # construction.
        return (arr - composite_t0) + offset if arr.size else arr

    def _arr(name):
        v = ev.get(name)
        return np.asarray(v, dtype=float) if v is not None else np.array([])

    native_tracks = [
        _native_track_for_channel(ch, ev, frame_times, to_media, src_fps)
        for ch in channels]
    return {
        "channels": channels,
        "src_fps": src_fps,
        "n_out": n_out,
        "frame_times": frame_times,
        "native_tracks": native_tracks,
        "clicks_t": to_media(_arr("clicks_t")),
        "clicks_x": _arr("clicks_x"),
        "clicks_y": _arr("clicks_y"),
    }


def multi_native_card_paths(session_dir, offset=0.0, window_zoom=False,
                            max_zoom=2.0, params=None, suppressed_ranges=None,
                            window_follow=True, stride=2):
    """Sampled per-card camera paths for a MULTI-NATIVE live composite, or None.

    The occlusion-free analog of `multi_window_card_paths`: same wire shape
    (`{stride, fps, cards:[{cx,cy,z}|None]}`, same rounding, same per-card None
    convention) so the browser's `cardSrcRect` animates it with no change.
    The difference is entirely upstream -- each card is a WHOLE channel buffer
    (not a sub-rect of one raw.mov), so `(cx, cy)` are in that channel's own
    buffer pixels and `cell.src` is `[0, 0, buffer_w, buffer_h]`.

    `window_follow` is accepted for signature parity but inert: a native card
    always follows its window's geometry track (there is no static-crop mode
    here), matching `_render_multi_native`, which takes no such flag.
    """
    if not window_zoom:
        return None
    space = _native_camera_inputs(session_dir, offset=offset)
    if space is None:
        return None
    paths = _build_native_card_cameras(
        space["channels"], space["native_tracks"],
        space["clicks_t"], space["clicks_x"], space["clicks_y"],
        space["frame_times"], max_zoom=max_zoom, params=params,
        suppressed_ranges=suppressed_ranges,
        plan_duration=float(space["n_out"]) / float(space["src_fps"]),
        src_fps=space["src_fps"], enabled=True)
    stride = max(1, int(stride))
    sel = slice(None, None, stride)
    out = []
    for path in paths:
        if path is None:
            out.append(None)
            continue
        out.append({
            "cx": [round(float(v), 2) for v in path[sel, 0]],
            "cy": [round(float(v), 2) for v in path[sel, 1]],
            "z": [round(float(v), 4) for v in path[sel, 2]],
        })
    if not any(p is not None for p in out):
        return None
    return {"stride": stride, "fps": space["src_fps"], "cards": out}


def _first_wins_spans(spans):
    """Rewrite a focus plan so materializing it renders EXACTLY as auto does.

    The two ways a plan reaches the emphasis spring resolve overlaps
    differently: an auto plan is read straight by `camera._active_range`,
    which returns the FIRST span in start order that covers `t`, while a
    materialized one goes through `camera._normalize_focus_manual`, which
    truncates the running span when a later one opens. Identical inputs,
    different subject.

    That only bites on a multi-native take, and it is the clamp-not-drop
    click set that causes it: every card owns every click there, so a card's
    post-arbitration merge (`camera._plan_focus_auto` merges AFTER
    arbitrating, deliberately) can re-extend past a neighbour's start and
    leave two spans genuinely overlapping. MEASURED on a 2-card shared-click
    plan: 98 of 360 frames emphasized a different card, up to 0.5 apart --
    i.e. materializing the plan verbatim would silently re-cut the video.
    Display-crop cards own their clicks by position and never produce this
    (measured: bit-identical on three shapes), which is why only the native
    emitter calls this.

    So: give each span only the times where it would have won under
    first-wins -- clipping it against every EARLIER span's full extent, which
    is precisely `_active_range`'s rule -- and the two readers agree. A span
    can come back in more than one piece; each piece is a span like any other.
    """
    out, claimed = [], []
    for sp in spans:
        pieces = [(float(sp["start"]), float(sp["end"]))]
        for a, b in claimed:
            nxt = []
            for s0, e0 in pieces:
                if b <= s0 or a >= e0:
                    nxt.append((s0, e0))
                    continue
                if s0 < a:
                    nxt.append((s0, a))
                if b < e0:
                    nxt.append((b, e0))
            pieces = nxt
        for s0, e0 in pieces:
            if e0 - s0 > 1e-6:
                piece = dict(sp)
                piece["start"], piece["end"] = s0, e0
                out.append(piece)
        claimed.append((float(sp["start"]), float(sp["end"])))
    out.sort(key=lambda r: (r["start"], r["end"]))
    return out


def multi_native_focus_ranges(session_dir, offset=0.0, max_zoom=2.0,
                              params=None, suppressed_ranges=None):
    """The window-focus plan of a MULTI-NATIVE take as editable spans.

    The occlusion-free analog of `multi_window_focus_ranges`, and the reason
    it exists separately: that one builds its cards from `edits.windows` via
    the display grid tracks, and a multi-native session HAS no `windows`
    array -- calling it there returns `[]`, which materialized into
    `edits.focus` would read as "the user deleted every arc" and silently
    switch the composition camera off. This one builds the same cards the
    export builds (`_native_card_click_times` over the per-channel tracks),
    so the spans it returns reproduce the auto-plan exactly rather than
    replacing it with a different one.

    Nothing calls this on session load: a multi-native take is deliberately
    left auto-planned (studio_app._load_edits_with_auto). It is for the
    surface that has to make ONE arc editable on demand -- MCP `adjust_zoom`
    -- where materializing the identical plan is what turns "soften this
    move" into an edit the export and the editor both honour.
    """
    space = _native_camera_inputs(session_dir, offset=offset)
    if space is None:
        return []
    per_card_times = _native_card_click_times(
        space["native_tracks"], space["clicks_t"], space["clicks_x"],
        space["clicks_y"])
    cards = [{"click_times": list(ts)} for ts in per_card_times]
    if not cards:
        return []
    return _first_wins_spans(camera.plan_focus_ranges(
        cards, max_zoom=max_zoom, params=params or {},
        suppressed_ranges=suppressed_ranges,
        plan_duration=float(space["n_out"]) / float(space["src_fps"])))


def multi_native_focus_cells(session_dir, painter, offset=0.0,
                             window_focus=False, max_zoom=2.0, params=None,
                             suppressed_ranges=None, window_follow=True,
                             focus_ranges=None, stride=2):
    """Sampled per-frame CELL geometry for a MULTI-NATIVE live composite, or None.

    The occlusion-free analog of `multi_window_focus_cells`, emitting the
    identical wire shape (`{stride, fps, frames:[{c, o, z, l?}]}` in canvas
    px) so the browser's `focusFrameAt` animates it unchanged. `painter` is
    the SAME `MultiFramePainter` `multi_native_layout` built for this payload,
    so the focus placements are relative to the cells already on screen.

    Everything after the emphasis is byte-identical to the display-crop
    emitter -- the only native-specific part is the emphasis itself, built
    from per-channel `_NativeWindowTrack` click ownership rather than a single
    source's grid tracks. `_FocusLayout` gets the SAME `lean` the native
    export passes, so preview and export never disagree about the two rungs.
    """
    if not window_focus or painter is None:
        return None
    space = _native_camera_inputs(session_dir, offset=offset)
    if space is None:
        return None
    native_tracks = space["native_tracks"]
    per_card_times = _native_card_click_times(
        native_tracks, space["clicks_t"], space["clicks_x"], space["clicks_y"])
    windows_for_focus = [
        {"w": (tr.raw_w if tr is not None else 1),
         "h": (tr.raw_h if tr is not None else 1)}
        for tr in native_tracks]
    emphasis = _build_native_focus_emphasis(
        windows_for_focus, per_card_times, space["frame_times"],
        max_zoom, params or {}, suppressed_ranges,
        float(space["n_out"]) / float(space["src_fps"]), space["src_fps"],
        manual=focus_ranges)
    if emphasis is None:
        return None
    layout = _FocusLayout(
        painter, emphasis,
        lean=camera.build_params(max_zoom, params or {}).focus_lean)
    stride = max(1, int(stride))
    frames = []
    for i in range(0, len(emphasis), stride):
        cells, order = layout.cells_at(i)
        cam = layout.camera_at(i)
        lifts = [round(painter.lift_of(j, c["w"]), 3)
                 for j, c in enumerate(cells)]
        frame = {
            "c": [[c["x"], c["y"], c["w"], c["h"], c["radius"]]
                  for c in cells],
            "o": list(order),
            "z": ([round(float(cam[0]), 1), round(float(cam[1]), 1),
                   round(float(cam[2]), 4)] if cam else None),
        }
        if any(v > 0.0 for v in lifts):
            frame["l"] = lifts
        frames.append(frame)
    return {"stride": stride, "fps": space["src_fps"], "frames": frames}


def multi_window_cursor_track(session_dir, offset=0.0, cursor_fx=False,
                              cursor_params=None, stride=2, crop_rect=None):
    """Sampled synthetic-cursor state for the editor's live composite, or None.

    Windows mode now draws the cursor (`_draw_multi_cursor`), so the paused
    still has one -- and a live composite without it would put the cursor back
    in the blink-on-pause state the facecam was just rescued from. The browser
    cannot run `CursorFX.draw`, so it gets the precomputed track instead and
    replays it through the SAME per-cell transform it already uses for the
    video crop. Coordinates are RAW FILE px (like `multi_window_layout`'s
    `cell["src"]`), because that is the space the <video> element samples in.

    None whenever nothing would be drawn -- the effect is off, the session
    kept its real cursor, or there is no move track to smooth.

    That middle case stays strict on purpose. The export and the paused still
    let `cursor_erase` unlock the synthetic cursor on a system take
    (`_cursor_fx_draws`), because both of them actually erase. Live playback
    does not: it is the raw <video> under a browser compositor, so the
    recorded pointer is still in those pixels while the clip is playing, and
    a second one drawn over it is exactly the doubling the gate exists to
    prevent. Same split the eraser already has between the still and playback.
    """
    if not cursor_fx:
        return None
    space = _SourceSpace(session_dir, offset=offset,
                         crop_rect=crop_rect)
    if not space.ok or space.meta.get("cursor_mode") != "synthetic":
        return None
    W, H = space.W, space.H
    src_fps, n_frames = space.src_fps, space.n_frames
    crop_x, crop_y = space.crop_x, space.crop_y
    frame_times = space.frame_times
    moves_t, moves_x, moves_y = space.moves()
    if not moves_t.size:
        return None
    clicks_t = space.to_media(space.ev["clicks_t"])
    cfx = effects.CursorFX(W, H, frame_times, moves_t, moves_x, moves_y,
                           clicks_t=clicks_t, params=cursor_params)
    data = cfx.sample_track(stride)
    data["fps"] = float(src_fps)
    # Back into RAW file px, the space `cell["src"]` is expressed in.
    if crop_x or crop_y:
        data["x"] = [round(v + crop_x, 2) for v in data["x"]]
        data["y"] = [round(v + crop_y, 2) for v in data["y"]]
    return data


_CAMERA_PATH_IGNORED = frozenset((
    # render-only options whose value has no bearing on the camera path
    # -- callers may thread the full render option dict through and we
    # silently drop these. Anything OUTSIDE this set is a bug (typo /
    # missing plumbing) and we want to hear about it, not swallow it.
    "click_fx", "click_params", "spotlight", "spotlight_params",
    # multi-window compositing only; the camera path is not used in that mode
    "window_follow",
    "window_layout",
    "cursor_fx", "cursor_params", "motion_blur",
    "fade", "music", "click_sound", "make_gif", "gif_fps", "gif_width",
    "trim_start", "trim_end", "out_path",
    # SPEED-UP: preview + camera_path DO NOT reflect the retiming in v1
    # (editor previews at source-time; export retimes). Accepted and
    # dropped so studio_app can pass the full kwargs dict without a
    # KeyError, but this is a KNOWN scope limitation -- see the "Auto
    # speed-up (Rush)" entry in docs/architecture.md for the follow-up
    # editor commit that will honor these.
    "speedup", "speedup_rate", "speedup_silence_gate", "speedups",
    "speedup_params",
    # CUTS (ripple delete): same scope limitation as speed-up -- the
    # editor previews the SOURCE timeline (cut ranges drawn as removed
    # overlays, playback skips them client-side); only the export
    # re-times. See the "Cuts / ripple delete" entry in docs/architecture.md.
    "cuts",
))


def camera_path(session_dir, t_start=None, t_end=None, stride=2,
                max_zoom=2.0, offset=0.0, style="clean", params=None,
                background=None, manual_zooms=None, suppressed_ranges=None,
                aspect=None, typing_zoom=True, scroll_zoom=True,
                crop_rect=None, screen_focus=True, **_ignored):
    """Sampled camera path for the whole clip, for timeline UIs / automation.

    Simulates the FULL clip (same planning + spring as render(): full
    `frame_times`, `plan_duration` = the true clip length), then subsamples
    every `stride` frames for the payload. `t_start`/`t_end` restrict only
    the *returned* (subsampled) range -- never the simulation, so the values
    at any time are identical no matter what window is requested.

    Only camera-affecting options matter here (style/background/aspect drive
    the canvas/window; click_fx etc. do not) -- extra render-option kwargs
    listed in `_CAMERA_PATH_IGNORED` are accepted and dropped so callers
    can pass the full option dict. Anything else raises TypeError.

    Returns a JSON-friendly dict:
      {"times", "cx", "cy", "z", "fps", "source", "win", "canvas", "duration"}
    with cx/cy rounded to 3 decimals and z to 4 to keep payloads small.
    """
    unknown = set(_ignored) - _CAMERA_PATH_IGNORED
    if unknown:
        raise TypeError("camera_path got unexpected keyword argument(s): "
                        + ", ".join(sorted(unknown)))
    meta, raw_path, events_path = _session_paths(session_dir)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open recording: " + raw_path)
    try:
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps", 60))
        if src_fps <= 0:
            src_fps = float(meta.get("fps", 60) or 60.0)
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if n_frames <= 0:
        dur = _probe_duration(raw_path)
        if dur is None:
            raise RuntimeError("cannot determine recording duration")
        n_frames = max(1, int(dur * src_fps) + 1)

    # points -> recorded pixels, per-axis (exact from the real file).
    scale_x = W / float(meta.get("logical_w", W) or W)
    scale_y = H / float(meta.get("logical_h", H) or H)
    # Record-time window capture: rebind (W, H) to the window's size -- the
    # returned cx/cy live in WINDOW space, which is what "source" reports.
    raw_W, raw_H = W, H
    crop = _capture_crop_px(meta, W, H)
    if crop is not None:
        crop_x, crop_y, W, H = crop

    t0 = meta.get("t0_monotonic", 0.0)

    ev = geometry.load_events(events_path)

    def to_media(arr):
        return (arr - t0) + offset if arr.size else arr

    track = _build_window_track(ev, crop, raw_W, raw_H, scale_x, scale_y,
                                np.arange(n_frames) / float(src_fps),
                                to_media, src_fps,
                                window_id=_capture_window_id(meta), meta=meta)
    if track is not None:
        to_src = track.to_src
    elif crop is not None:
        def to_src(_t, ax, ay):
            return ax * scale_x - crop_x, ay * scale_y - crop_y
    else:
        def to_src(_t, ax, ay):
            return ax * scale_x, ay * scale_y

    # Editor crop, stage two -- see `_user_crop_px`. The camera plan has to
    # run in the SAME space the export does, or the editor's timeline shows a
    # camera line the exported file never followed.
    user_crop = _user_crop_px(crop_rect, W, H)
    if user_crop is not None:
        W, H = user_crop[2], user_crop[3]
        to_src = _user_crop_to_src(to_src, user_crop)
    # The rect the EDITOR must clip its <video> to during live playback: the
    # two crop stages composed back into one raw-file rect. Deliberately not
    # folded into `crop` below -- the editor reads that one to decide whether
    # the session was window-TARGETED (`captureCropApplied`, which drives a
    # banner), and an editor crop must not answer yes to that question.
    if user_crop is None:
        stage_crop = crop
    else:
        _bx, _by = (crop[0], crop[1]) if crop is not None else (0, 0)
        stage_crop = (_bx + user_crop[0], _by + user_crop[1],
                      user_crop[2], user_crop[3])

    clicks_t = to_media(ev["clicks_t"])
    clicks_x, clicks_y = to_src(clicks_t, ev["clicks_x"], ev["clicks_y"])
    moves_t = to_media(ev["moves_t"])
    moves_x, moves_y = to_src(moves_t, ev["moves_x"], ev["moves_y"])
    keys_t = to_media(ev["keys_t"]) if typing_zoom else np.array([])
    ups_t = to_media(ev["ups_t"])
    if scroll_zoom:
        scrolls_t = to_media(ev["scrolls_t"])
        scrolls_x, scrolls_y = to_src(scrolls_t, ev["scrolls_x"],
                                      ev["scrolls_y"])
    else:
        scrolls_t = scrolls_x = scrolls_y = np.array([])
    typing_anchors = _typing_anchor_hints(raw_path, events_path, keys_t,
                                          max_zoom, params, suppressed_ranges)
    typing_anchors = _offset_typing_anchors(typing_anchors, crop, track,
                                            user=user_crop)

    out_w, out_h = framing.output_size(W, H, style, aspect=aspect)
    win_w, win_h = _contain_fit(out_w, out_h, W, H)

    duration = n_frames / float(src_fps)
    frame_times = np.arange(n_frames) / float(src_fps)
    # Whole-screen "grow the active window": same planner reshape the export
    # runs, so the editor's camera line shows the SAME push (zero JS change);
    # the grow overlay rides the returned `screen_focus` payload. Plain take
    # only (no capture crop / cards).
    screen_ctx = None
    screen_resolver = None
    screen_spans = None
    if screen_focus and crop is None and track is None:
        screen_ctx = _build_screen_focus_ctx(ev, to_media, to_src, frame_times,
                                             W, H, src_fps)
        if screen_ctx is not None:
            _sp = camera.build_params(max_zoom, params)
            screen_resolver, screen_spans = _make_screen_focus_resolver(
                screen_ctx, win_w, win_h, _sp.overview_fill, src_fps, out_w)
    path = camera.build_path(frame_times, W, H,
                             clicks_t, clicks_x, clicks_y,
                             moves_t, moves_x, moves_y,
                             max_zoom=max_zoom, fps=src_fps,
                             params=params or {},
                             manual_zooms=manual_zooms,
                             suppressed_ranges=suppressed_ranges,
                             keys_t=keys_t, ups_t=ups_t,
                             scrolls_t=scrolls_t, scrolls_x=scrolls_x,
                             scrolls_y=scrolls_y,
                             typing_anchors=typing_anchors,
                             win_w=win_w, win_h=win_h,
                             plan_duration=duration,
                             window_resolver=screen_resolver)

    step = max(1, int(stride or 1))
    idx = np.arange(0, n_frames, step)
    if t_start is not None:
        idx = idx[frame_times[idx] >= float(t_start) - 1e-9]
    if t_end is not None:
        idx = idx[frame_times[idx] <= float(t_end) + 1e-9]
    times = frame_times[idx]
    sub = path[idx] if idx.size else np.zeros((0, 3))
    # Grow overlay samples (OUTPUT px on the pushed canvas), parallel to
    # times/cx/cy/z. None when the feature didn't fire -- byte-stable for
    # every take that doesn't use it.
    screen_grow = None
    if screen_ctx is not None and screen_spans and idx.size:
        screen_grow = _screen_focus_payload(
            screen_spans, screen_ctx, path, src_fps, out_w, out_h, win_w,
            step, idx, camera.build_params(max_zoom, params).always_zoomed)
    return {
        "times": [round(float(t), 4) for t in times.tolist()],
        "cx": [round(float(v), 3) for v in sub[:, 0].tolist()],
        "cy": [round(float(v), 3) for v in sub[:, 1].tolist()],
        "z": [round(float(v), 4) for v in sub[:, 2].tolist()],
        "screen_focus": screen_grow,
        "fps": float(src_fps),
        # "source" is the space cx/cy live in -- the CROPPED frame when this
        # session was recorded window-targeted. "raw_source"/"crop" expose the
        # file's own dimensions and the crop that got us here.
        "source": [int(W), int(H)],
        "raw_source": [int(raw_W), int(raw_H)],
        "crop": ([int(v) for v in crop] if crop is not None else None),
        "stage_crop": ([int(v) for v in stage_crop]
                       if stage_crop is not None else None),
        "win": [round(float(win_w), 3), round(float(win_h), 3)],
        "canvas": [int(out_w), int(out_h)],
        "duration": float(duration),
    }


def encode_preview_jpeg(frame_bgr, quality=88):
    """JPEG-encode a BGR frame for browser previews."""
    q = int(max(1, min(100, int(quality))))
    ok, enc = cv2.imencode(".jpg", frame_bgr,
                           [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        raise RuntimeError("failed to encode preview frame")
    return enc.tobytes()


def render(session_dir, out_path=None, max_zoom=2.0, offset=0.0,
           style="clean", make_gif=False, params=None,
           background=None, click_fx=True, click_params=None,
           spotlight=False, spotlight_params=None, fade=0.0, music=None,
           click_sound=None, key_sound=None, sfx_volume=1.0,
           trim_start=0.0, trim_end=None,
           manual_zooms=None, suppressed_ranges=None,
           cursor_fx=False, cursor_params=None, aspect=None,
           motion_blur=True, typing_zoom=True, scroll_zoom=True,
           speedup=False, speedup_rate=6.0,
           speedup_silence_gate=True, speedup_motion_gate=True,
           speedups=None, speedup_params=None,
           facecam=True, facecam_params=None,
           gif_fps=15, gif_width=1000, windows=None, window_follow=True,
           window_layout="grid", window_zoom=False, window_focus=False,
           focus_ranges=None, crop_rect=None, cursor_erase=False,
           cursor_erase_params=None, screen_focus=True, cuts=None,
           max_height=None, channel_layouts=None, scene_layouts=None,
           hidden_channels=None, join_entrance=True, badge_erase=True,
           _src_override=None, _to_media_override=None, _count_override=None,
           _bed_clock=None):
    with open(os.path.join(session_dir, "meta.json")) as f:
        meta = json.load(f)
    # Multi-window native (P3.2): meta carries a `capture_channels` manifest,
    # not a single `raw`. Delegated at the top so the single-file render
    # below runs exactly the code that shipped before this feature -- no
    # test on that path has to be re-run for confidence, only the ones the
    # multi-native path owns. `_is_multi_native_meta` is fail-safe (needs
    # >=2 channels, all `mode: window_native`).
    # `_src_override` is set only by `_render_segmented`, which has already
    # joined the segments and points us at the joined file; skip the manifest
    # dispatch so we don't recurse, and run the ordinary single-file body.
    # Scene take (docs/architecture.md): a `capture_scenes` manifest --
    # K sequential fleet scenes, one continuous output, layout change at the
    # seam. Checked FIRST (the discriminator is fail-safe and rejects every
    # sibling manifest, so the order is safety-redundant but explicit).
    if _src_override is None and segments.is_scene_meta(meta):
        if cuts:
            # v1 refusal, loud: the scene path has no event-override seam
            # (docs/architecture.md lists `_ev_override` as unbuilt S2 work), so
            # cuts cannot re-time it yet. Same posture as the flag list
            # _render_scenes itself prints.
            print("  note: cuts are not supported on scene takes yet -- "
                  "rendering without them.")
        return _render_scenes(
            session_dir, out_path=out_path, style=style, background=background,
            aspect=aspect, window_layout=window_layout,
            window_zoom=window_zoom, window_focus=window_focus,
            cursor_fx=cursor_fx, cursor_params=cursor_params,
            max_zoom=max_zoom, params=params,
            suppressed_ranges=suppressed_ranges, focus_ranges=focus_ranges,
            motion_blur=motion_blur, click_fx=click_fx, spotlight=spotlight,
            cursor_erase=cursor_erase, facecam=facecam, speedup=speedup,
            make_gif=make_gif, trim_start=trim_start, trim_end=trim_end,
            fade=fade, music=music, click_sound=click_sound,
            key_sound=key_sound, sfx_volume=sfx_volume,
            max_height=max_height, scene_layouts=scene_layouts,
            hidden_channels=hidden_channels, join_entrance=join_entrance,
            badge_erase=badge_erase)
    if _src_override is None and _is_multi_native_meta(meta):
        if cuts:
            print("  note: cuts are not supported on multi-window native "
                  "takes yet -- rendering without them.")
        return _render_multi_native(
            session_dir, out_path=out_path, style=style, background=background,
            aspect=aspect, window_layout=window_layout,
            motion_blur=motion_blur, click_fx=click_fx, spotlight=spotlight,
            cursor_fx=cursor_fx, cursor_params=cursor_params,
            cursor_erase=cursor_erase, facecam=facecam,
            speedup=speedup, make_gif=make_gif, gif_fps=gif_fps,
            gif_width=gif_width,
            window_zoom=window_zoom, window_focus=window_focus,
            max_zoom=max_zoom, params=params,
            suppressed_ranges=suppressed_ranges, focus_ranges=focus_ranges,
            max_height=max_height, channel_layouts=channel_layouts,
            hidden_channels=hidden_channels,
            click_sound=click_sound, key_sound=key_sound,
            sfx_volume=sfx_volume, badge_erase=badge_erase)
    # Segmented take (pause/resume): meta carries a `capture_segments` manifest
    # and no single `raw`. Delegate to `_render_segmented`, which joins the
    # segment files (paused gaps deleted) and calls back into this body with
    # `_src_override`/`_to_media_override`/`_count_override` set.
    if _src_override is None and segments.is_segmented_meta(meta):
        return _render_segmented(
            session_dir, out_path=out_path, max_zoom=max_zoom, offset=offset,
            style=style, make_gif=make_gif, params=params, background=background,
            click_fx=click_fx, click_params=click_params, spotlight=spotlight,
            spotlight_params=spotlight_params, fade=fade, music=music,
            click_sound=click_sound, key_sound=key_sound,
            sfx_volume=sfx_volume, trim_start=trim_start, trim_end=trim_end,
            manual_zooms=manual_zooms, suppressed_ranges=suppressed_ranges,
            cursor_fx=cursor_fx, cursor_params=cursor_params, aspect=aspect,
            motion_blur=motion_blur, typing_zoom=typing_zoom,
            scroll_zoom=scroll_zoom, speedup=speedup, speedup_rate=speedup_rate,
            speedup_silence_gate=speedup_silence_gate,
            speedup_motion_gate=speedup_motion_gate, speedups=speedups,
            speedup_params=speedup_params, facecam=facecam,
            facecam_params=facecam_params, gif_fps=gif_fps, gif_width=gif_width,
            cursor_erase=cursor_erase, cursor_erase_params=cursor_erase_params,
            windows=windows, window_follow=window_follow,
            window_layout=window_layout, window_zoom=window_zoom,
            window_focus=window_focus, focus_ranges=focus_ranges,
            crop_rect=crop_rect, cuts=cuts, max_height=max_height)
    raw_path = _src_override or os.path.join(
        session_dir, meta.get("raw", "raw.mov"))
    events_path = os.path.join(session_dir, meta.get("events", "events.jsonl"))
    if out_path is None:
        out_path = os.path.join(session_dir, "output.mp4")

    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        raise RuntimeError("cannot open recording: " + raw_path)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or float(meta.get("fps", 60))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # points -> recorded pixels, per-axis (exact, from the real file: handles
    # Retina backing store even under fractional display scaling).
    scale_x = W / float(meta.get("logical_w", W) or W)
    scale_y = H / float(meta.get("logical_h", H) or H)
    # Frame count first: the geometry track below is resampled onto this
    # frame grid, so it has to be known before any coordinate is mapped.
    if _count_override is not None:
        # Segmented: the joined frame count is the sum of the per-segment
        # decoded counts (the SAME numbers the SegmentClock was built from --
        # one source of truth). Assert the joined file agrees, so a concat that
        # silently dropped/added a frame per seam fails loud instead of
        # misplacing every post-seam event.
        count = int(_count_override)
        if n_frames > 0 and n_frames != count:
            raise RuntimeError(
                "segmented join frame count {} != sum of segment counts {} "
                "-- concat is not frame-exact".format(n_frames, count))
    elif n_frames > 0:
        count = n_frames
    else:
        # Some containers don't report a frame count; size the plan from the
        # real duration (falling back to a modest cap) rather than 3600s.
        dur = _probe_duration(raw_path)
        count = int(dur * src_fps) + 2 if dur else int(src_fps * 60)

    # Record-time window capture: crop every decoded frame (below, in the
    # loop) and rebind (W, H) to the window's size -- everything from here on
    # runs in WINDOW space, auto-zoom and effects included.
    raw_W, raw_H = W, H
    crop = _capture_crop_px(meta, W, H)
    if crop is not None:
        crop_x, crop_y, W, H = crop

    t0 = meta.get("t0_monotonic", 0.0)

    ev = geometry.load_events(events_path)

    if _to_media_override is not None:
        # Segmented: the piecewise clock maps a parent-monotonic event time to
        # its position in the concatenated (gaps-deleted) output timeline. It
        # is shape-preserving (clamp-not-drop), so it drops in as `to_media`
        # with no downstream change -- clicks/moves/keys/geometry all route
        # through it, exactly as they route through the affine map below.
        # A saved sync `offset` still applies, as the same post-map shift the
        # affine map gives it.
        if offset:
            def to_media(arr, _clock=_to_media_override):
                return _clock(arr) + offset
        else:
            to_media = _to_media_override
    else:
        def to_media(arr):
            return (arr - t0) + offset if arr.size else arr

    # Geometry track: follows a window that was moved/resized mid-take. None
    # (and therefore the single-snapshot crop, bit-exact) for every session
    # without one.
    track = _build_window_track(ev, crop, raw_W, raw_H, scale_x, scale_y,
                                np.arange(count) / float(src_fps),
                                to_media, src_fps,
                                window_id=_capture_window_id(meta), meta=meta)
    if track is not None:
        to_src = track.to_src
    elif crop is not None:
        def to_src(_t, ax, ay):
            return ax * scale_x - crop_x, ay * scale_y - crop_y
    else:
        def to_src(_t, ax, ay):
            return ax * scale_x, ay * scale_y

    # Editor crop, stage two. Resolved AFTER the track is built, never folded
    # into the rect handed to it -- see `_user_crop_px`.
    user_crop = _user_crop_px(crop_rect, W, H)
    if user_crop is not None:
        W, H = user_crop[2], user_crop[3]
        to_src = _user_crop_to_src(to_src, user_crop)

    clicks_t = to_media(ev["clicks_t"])
    clicks_x, clicks_y = to_src(clicks_t, ev["clicks_x"], ev["clicks_y"])
    moves_t = to_media(ev["moves_t"])
    moves_x, moves_y = to_src(moves_t, ev["moves_x"], ev["moves_y"])
    def _bed_src(raw):
        """A raw event array on the media clock, ready for the sound bed.

        On a SEGMENTED take `to_media` IS a `SegmentClock`, which is
        clamp-not-drop (see `_clock_contained`) -- so the events are masked
        to the segment that contains them BEFORE mapping, or every click
        made during a deleted pause would stack into one loud click at the
        seam. `_bed_clock` is None on every other path, where this is a
        plain `to_media` and costs nothing.
        """
        raw = np.asarray(raw if raw is not None else [], dtype=float)
        if _bed_clock is not None and raw.size:
            raw = raw[_clock_contained(_bed_clock, raw)]
        return to_media(raw)

    # Separate CAMERA and SOUND views of the same tracks. The camera keeps
    # the clamp-not-drop arrays (it needs x/y to stay positionally zipped);
    # the bed gets the masked ones. `keys_t` is additionally gated by
    # `typing_zoom` for the camera, and the bed's is deliberately NOT --
    # "don't zoom on typing" must not mean "go silent while I type".
    keys_t = to_media(ev["keys_t"]) if typing_zoom else np.array([])
    ups_t = to_media(ev["ups_t"])
    sfx_clicks_t = _bed_src(ev["clicks_t"])
    sfx_ups_t = _bed_src(ev["ups_t"])
    sfx_keys_t = _bed_src(ev["keys_t"])
    if scroll_zoom:
        scrolls_t = to_media(ev["scrolls_t"])
        scrolls_x, scrolls_y = to_src(scrolls_t, ev["scrolls_x"],
                                      ev["scrolls_y"])
    else:
        scrolls_t = scrolls_x = scrolls_y = np.array([])
    use_multi = bool(windows)
    typing_anchors = (None if use_multi else
                      _typing_anchor_hints(raw_path, events_path, keys_t,
                                          max_zoom, params, suppressed_ranges))
    typing_anchors = _offset_typing_anchors(typing_anchors, crop, track)

    start_idx, end_idx = _trim_frame_bounds(count, src_fps, trim_start, trim_end)
    if end_idx <= start_idx:
        raise RuntimeError("invalid trim range: start={} end={}".format(
            trim_start, trim_end))
    out_w, out_h = framing.output_size(W, H, style, aspect=aspect)
    out_w, out_h = framing.fit_max_height(out_w, out_h, max_height)
    win_w, win_h = _contain_fit(out_w, out_h, W, H)
    frame_times = np.arange(count) / float(src_fps)
    has_audio = _probe_has_audio(raw_path)

    # -- Auto speed-up (Rush) planning ---------------------------------
    # Every non-identity branch below the `if not tm.identity` gate is
    # guarded by that gate, so `speedup=False` (or "gates found nothing to
    # speed") compiles to today's path verbatim.
    total_dur_src = count / float(src_fps)
    # -- Cuts (ripple delete) -----------------------------------------
    # Union-merge + frame-quantize ONCE (retime.quantize_cut_spans) so the
    # video emit mask and the audio atrim windows carve identical
    # boundaries; the same helper feeds the MCP's snapped echo, so the
    # user is told exactly what the encoder removes. Empty `cuts` leaves
    # `cut_spans` empty and every branch below on the pre-feature path.
    cut_spans = []
    if cuts:
        if use_multi:
            # v1 refusal, loud: the card/focus/grid planners
            # (_build_grid_tracks / _build_focus_emphasis /
            # _build_card_cameras) are source-clock and read UNMASKED
            # clicks -- a card would grow for a click the cut deleted.
            # Masking those planners is the follow-up; until then, refuse.
            print("  note: cuts are ignored in multi-window card layouts "
                  "(v1) -- render without --window cards to apply them.")
        else:
            cut_spans = retime.quantize_cut_spans(cuts, src_fps,
                                                  duration=total_dur_src)
    tm = _plan_timemap(speedup, speedup_rate, speedup_silence_gate,
                       speedups, speedup_params,
                       clicks_t, moves_t, moves_x, moves_y,
                       ups_t, keys_t, scrolls_t,
                       raw_path=raw_path, duration=total_dur_src,
                       has_audio=has_audio,
                       motion_gate=speedup_motion_gate, verbose=True,
                       cuts=cut_spans)

    if tm.identity:
        emit = None
        out_ord = None
        n_out_total = count
        plan_clicks_t = clicks_t
        plan_clicks_x = clicks_x
        plan_clicks_y = clicks_y
        plan_moves_t = moves_t
        plan_moves_x = moves_x
        plan_moves_y = moves_y
        plan_keys_t = keys_t
        plan_ups_t = ups_t
        plan_scrolls_t = scrolls_t
        plan_scrolls_x = scrolls_x
        plan_scrolls_y = scrolls_y
        plan_manual = manual_zooms
        plan_suppress = suppressed_ranges
        plan_typing_anchors = typing_anchors
        plan_win_w = win_w
        plan_win_h = win_h
        plan_frame_times = frame_times
        plan_duration = None
    else:
        emit, out_ord, n_out_total = tm.emission(frame_times)
        # CUTS: events inside a removed range lose their visual context --
        # drop them from the camera plan (and, downstream, the click SFX)
        # BEFORE warping. Warping alone would collapse every one of them
        # onto the seam instant: a phantom click cluster for the camera
        # and a burst of stacked adelay taps in the audio graph.
        # Positionally-zipped families (t, x, y) are masked TOGETHER --
        # dropping t without x/y pairs every later click with a
        # neighbour's pixel (the segments.py clamp-not-drop trap, inverted).
        # With no cuts these are aliases: the speedup-only path computes
        # exactly what it always did.
        if tm.cut_spans:
            _ck = tm.keep_mask(clicks_t)
            cut_clicks_t = clicks_t[_ck]
            plan_clicks_x = clicks_x[_ck]
            plan_clicks_y = clicks_y[_ck]
            _mk = tm.keep_mask(moves_t)
            cut_moves_t = moves_t[_mk]
            cut_moves_x = moves_x[_mk]
            cut_moves_y = moves_y[_mk]
            cut_keys_t = (keys_t[tm.keep_mask(keys_t)]
                          if keys_t is not None and keys_t.size else keys_t)
            cut_ups_t = (ups_t[tm.keep_mask(ups_t)]
                         if ups_t is not None and ups_t.size else ups_t)
            if scrolls_t.size:
                _sk = tm.keep_mask(scrolls_t)
                cut_scrolls_t = scrolls_t[_sk]
                cut_scrolls_x = scrolls_x[_sk]
                cut_scrolls_y = scrolls_y[_sk]
            else:
                cut_scrolls_t = scrolls_t
                cut_scrolls_x = scrolls_x
                cut_scrolls_y = scrolls_y
        else:
            cut_clicks_t = clicks_t
            plan_clicks_x = clicks_x
            plan_clicks_y = clicks_y
            cut_moves_t, cut_moves_x, cut_moves_y = moves_t, moves_x, moves_y
            cut_keys_t = keys_t
            cut_ups_t = ups_t
            cut_scrolls_t = scrolls_t
            cut_scrolls_x = scrolls_x
            cut_scrolls_y = scrolls_y
        # Drop drift-speed cursor motion inside sped spans: after warping,
        # 80 px/s drift becomes 80*r px/s -- a fake "fast move" that would
        # trigger the camera's cluster keep-alive and hold the zoom through
        # a time-lapse. Move samples that BOOK-END or LIE ENTIRELY IN a
        # span's interior are dropped from the plan; the render loop's
        # spotlight interp still uses the full unwarped arrays.
        plan_moves_mask = _plan_moves_mask(cut_moves_t, tm)
        plan_moves_t = tm.warp(cut_moves_t[plan_moves_mask])
        plan_moves_x = cut_moves_x[plan_moves_mask]
        plan_moves_y = cut_moves_y[plan_moves_mask]
        plan_clicks_t = tm.warp(cut_clicks_t)
        plan_keys_t = tm.warp(cut_keys_t) if cut_keys_t is not None and cut_keys_t.size else cut_keys_t
        plan_ups_t = tm.warp(cut_ups_t) if cut_ups_t is not None and cut_ups_t.size else cut_ups_t
        if cut_scrolls_t.size:
            plan_scrolls_t = tm.warp(cut_scrolls_t)
            plan_scrolls_x = cut_scrolls_x
            plan_scrolls_y = cut_scrolls_y
        else:
            plan_scrolls_t = plan_scrolls_x = plan_scrolls_y = cut_scrolls_t
        plan_manual = tm.warp_spans(manual_zooms) if manual_zooms else manual_zooms
        plan_suppress = tm.warp_spans(suppressed_ranges) if suppressed_ranges else suppressed_ranges
        plan_typing_anchors = (tm.warp_spans(typing_anchors)
                               if typing_anchors else typing_anchors)
        if tm.cut_spans:
            # A range wholly inside a cut collapses to zero length at the
            # seam -- drop it rather than hand the camera a degenerate span.
            plan_manual = _drop_degenerate_spans(plan_manual)
            plan_suppress = _drop_degenerate_spans(plan_suppress)
            plan_typing_anchors = _drop_degenerate_spans(plan_typing_anchors)
        plan_win_w = win_w
        plan_win_h = win_h
        plan_frame_times = np.arange(n_out_total) / float(src_fps)
        plan_duration = tm.output_duration

    # Whole-screen "grow the active window": plain take only (no capture crop,
    # no cards) and not sped up (window geometry is source-clock; retime would
    # desync the grow -- gated OFF and re-anchored as later work). Builds the
    # per-window ctx + the planner resolver; both stay None otherwise, so the
    # plan and the loop below are byte-identical to today.
    screen_ctx = None
    screen_resolver = None
    screen_spans = None
    # `track is None and crop is None` == a PLAIN whole-screen take: a
    # capture-window crop has crop!=None and a single-window-native take has a
    # non-None `track` with crop None -- neither is "several windows on one
    # screen", so both are excluded.
    if (screen_focus and not use_multi and crop is None and track is None
            and tm.cut_spans and not tm.spans):
        # The speedup case stays silent (its shipped stdout is expected by
        # callers); cuts are new, and silently losing window emphasis on a
        # take that never used speed-up would read as a regression.
        print("  note: screen focus is off on this render -- cuts re-time "
              "the take and the window-grow track is source-clock "
              "(re-anchoring it is follow-up work).")
    if (screen_focus and not use_multi and crop is None and track is None
            and tm.identity):
        screen_ctx = _build_screen_focus_ctx(ev, to_media, to_src, frame_times,
                                             W, H, src_fps)
        if screen_ctx is not None:
            _sp = camera.build_params(max_zoom, params)
            screen_resolver, screen_spans = _make_screen_focus_resolver(
                screen_ctx, plan_win_w, plan_win_h, _sp.overview_fill, src_fps,
                out_w)

    if use_multi:
        # No per-frame camera in windows mode (static crops) -- skip the
        # planning pass entirely rather than computing and discarding it.
        path = None
    else:
        path = camera.build_path(plan_frame_times, W, H,
                                 plan_clicks_t, plan_clicks_x, plan_clicks_y,
                                 plan_moves_t, plan_moves_x, plan_moves_y,
                                 max_zoom=max_zoom, fps=src_fps,
                                 params=params or {},
                                 manual_zooms=plan_manual,
                                 suppressed_ranges=plan_suppress,
                                 keys_t=plan_keys_t, ups_t=plan_ups_t,
                                 scrolls_t=plan_scrolls_t,
                                 scrolls_x=plan_scrolls_x,
                                 scrolls_y=plan_scrolls_y,
                                 typing_anchors=plan_typing_anchors,
                                 win_w=plan_win_w, win_h=plan_win_h,
                                 plan_duration=plan_duration,
                                 window_resolver=screen_resolver)

    # Per-output-frame grow track (owner rect + weight off the same spring),
    # or None when nothing was claimed.
    screen_track = None
    if screen_ctx is not None and screen_spans:
        screen_track = _build_screen_focus_track(
            screen_spans, screen_ctx, path, src_fps,
            camera.build_params(max_zoom, params).always_zoomed)

    # Per-frame cursor position (recorded px) for the spotlight follower.
    # Spotlight is drawn at SOURCE-time src_idx (below), so this stays
    # indexed by source frames -- the full (unfiltered) move track.
    if moves_t.size >= 1:
        cur_x = np.interp(frame_times, moves_t, moves_x,
                          left=moves_x[0], right=moves_x[-1])
        cur_y = np.interp(frame_times, moves_t, moves_y,
                          left=moves_y[0], right=moves_y[-1])
    else:
        cur_x = np.full(count, W / 2.0)
        cur_y = np.full(count, H / 2.0)

    # A skipped synthetic cursor used to be silent -- and silence is the
    # wrong answer now that there IS something to do about it.
    if cursor_fx and not _cursor_fx_draws(cursor_fx, meta, cursor_erase):
        print("  note: synthetic cursor skipped -- this take has the real "
              "cursor burned into its pixels. Add --cursor-erase to lift it "
              "out first (costs one extra decode of the source).")

    # -- Capture-indicator erase ---------------------------------------
    # Built BEFORE the cursor eraser so its prepass can run the same stage
    # (`_erase_crop_fn`): the prepass samples clean plates from a second
    # decode, and a plate taken from an un-erased frame would disagree with
    # the loop's erased one wherever the pointer visited the badge -- the
    # eraser's own bracket test would then read that disagreement as "the
    # screen changed here" and fall back to inpainting a corner that was
    # never in doubt.
    badge = _badge_eraser(raw_path, meta.get("capture_window"), raw_W,
                          enabled=badge_erase)

    # -- Cursor eraser -------------------------------------------------
    # Planned here, applied in the loop below on the SOURCE frame -- before
    # the camera warp, the compositor and every effect, so it is the one
    # stage that composes with all of them for free (they each see a frame
    # that simply never had a pointer in it). Costs one extra decode of the
    # source, which is why it is opt-in.
    erase = None
    if cursor_erase and _erase_applies(meta):
        esx, esy = _erase_scale(track, scale_x, scale_y)
        e_boxes, e_hots, e_track = _erase_boxes(
            ev, to_media, to_src, frame_times, W, H, esx, esy,
            cursor_erase_params)
        if e_boxes is not None:
            print("  cursor erase: scanning the source for clean pixels...")
            e_cap = cv2.VideoCapture(raw_path)
            try:
                erase = eraser.plan(
                    e_cap, _erase_crop_fn(crop, track, user_crop,
                                          badge and badge.clone()),
                    e_boxes, e_hots, e_track, src_fps,
                    start_idx=start_idx, end_idx=end_idx,
                    params=cursor_erase_params,
                    progress=lambda i, n: print(
                        "  cursor erase: scanned {}/{} frames...".format(i, n),
                        end="\r", flush=True))
            finally:
                e_cap.release()
            print()

    facecam_ov = _facecam_overlay(session_dir, meta, out_w, out_h,
                                  enabled=facecam, params=facecam_params)
    if use_multi:
        painter = None
        multi_painter = framing.make_multi_painter(
            out_w, out_h, windows, background=background,
            layout=window_layout)
        grid_tracks = _build_grid_tracks(
            ev, windows, crop, track, W, H, scale_x, scale_y, frame_times,
            to_media, src_fps, enabled=window_follow)
        # `.cells` rebuilds its list on every access; the layout is static, so
        # hoist it out of the per-frame loop.
        multi_cells = multi_painter.cells
        focus_emphasis = _build_focus_emphasis(
            windows, grid_tracks, clicks_t, clicks_x, clicks_y,
            frame_times, W, H, max_zoom, params, suppressed_ranges,
            total_dur_src, src_fps, manual=focus_ranges,
            enabled=window_focus)
        focus_layout = (None if focus_emphasis is None
                        else _FocusLayout(multi_painter, focus_emphasis,
                                          lean=camera.build_params(max_zoom,
                                                              params).focus_lean))
        card_paths = _build_card_cameras(
            windows, multi_cells, grid_tracks, clicks_t, clicks_x, clicks_y,
            frame_times, W, H, max_zoom, params, suppressed_ranges,
            total_dur_src, src_fps, enabled=window_zoom)
        clickfx = spot = None
        cursorfx = None
        if _cursor_fx_draws(cursor_fx, meta, cursor_erase):
            cursorfx = effects.CursorFX(W, H, frame_times, moves_t, moves_x,
                                        moves_y, clicks_t=clicks_t,
                                        params=cursor_params)
        _warn_ignored_multi_window_options(
            motion_blur=motion_blur, click_fx=click_fx, spotlight=spotlight)
    else:
        painter = framing.make_painter(out_w, out_h, style, background=background)
        multi_painter = None
        multi_cells = None
        grid_tracks = None
        card_paths = None
        focus_layout = None
        clickfx = (effects.ClickFX(W, H, clicks_t, clicks_x, clicks_y, click_params)
                   if click_fx and clicks_t.size else None)
        spot = effects.Spotlight(W, H, spotlight_params) if spotlight else None
        cursorfx = None
        if _cursor_fx_draws(cursor_fx, meta, cursor_erase):
            cursorfx = effects.CursorFX(W, H, frame_times, moves_t, moves_x, moves_y,
                                        clicks_t=clicks_t, params=cursor_params)

    # Output-window frame counts + total output duration for the fade / audio.
    if tm.identity:
        out_count = end_idx - start_idx
    else:
        out_count = int(emit[start_idx:end_idx].sum())
        if out_count <= 0:
            raise RuntimeError(
                "speed-up/cuts left zero output frames in the trim window "
                "(rate={}, spans={}, cuts={})".format(
                    speedup_rate, tm.spans, tm.cut_spans))
    total_dur = out_count / float(src_fps)

    # Warp click times to output for both the click-SFX overlay AND the
    # audio filter graph.
    if tm.identity:
        audio_start = start_idx / float(src_fps)
        retime_segments = None
    else:
        audio_start = float(tm.warp(np.array([start_idx / float(src_fps)]))[0])
        # audio_end in the SOURCE domain (for atrim windows) is the source
        # time whose warp equals audio_start + total_dur. We already know
        # both bounds in source time (start_idx and end_idx) -- feed the
        # source window straight in and let TimeMap compute the atempo
        # rate at boundaries exactly.
        src_end = min(end_idx / float(src_fps), total_dur_src)
        retime_segments = tm.segments_for_audio(start_idx / float(src_fps),
                                                src_end)

    def _sfx_out(arr):
        """Source-time events -> the output timeline, cut events removed.

        Exactly the treatment `clicks_out` gets just above, factored out so
        the key and mouse-up tracks cannot drift from the click track: mask
        FIRST (an event inside a removed range must make no sound), warp
        second -- masking after warping would pile every cut event onto the
        seam instant as one burst.
        """
        arr = np.asarray(arr, dtype=np.float64)
        if arr.size == 0 or tm.identity:
            return arr
        if tm.cut_spans:
            arr = arr[tm.keep_mask(arr)]
        return tm.warp(arr) if arr.size else arr

    click_sound_times, dropped_sfx = _click_times_in_trim(
        _sfx_out(sfx_clicks_t),
        clip_start=audio_start,
        clip_duration=total_dur,
        max_events=_MAX_CLICK_SFX_EVENTS,
    )
    release_times, _ = _click_times_in_trim(
        _sfx_out(sfx_ups_t), clip_start=audio_start, clip_duration=total_dur,
        max_events=_MAX_CLICK_SFX_EVENTS)
    key_sound_times, dropped_keys = _click_times_in_trim(
        _sfx_out(sfx_keys_t), clip_start=audio_start,
        clip_duration=total_dur, max_events=_MAX_CLICK_SFX_EVENTS)

    if music and not os.path.isfile(music):
        print("  warning: --music file not found, skipping: {}".format(music))
    if dropped_sfx > 0:
        print("  warning: click sounds capped at {} events ({} skipped)".format(
            len(click_sound_times), dropped_sfx))
    if dropped_keys > 0:
        print("  warning: key sounds capped at {} events ({} skipped)".format(
            len(key_sound_times), dropped_keys))

    # Pre-mix every event sound into ONE wav and hand the encoder a single
    # input (sfx.py's docstring explains why this is not N ffmpeg taps).
    # None when both kinds are off or nothing was recorded -- which is what
    # keeps the encoder argv byte-identical to the pre-feature one.
    if not tm.identity:
        saved = (end_idx - start_idx - out_count) / float(src_fps)
        if tm.cut_spans:
            # Report cuts and speed-up separately: `saved` is TOTAL dropped
            # time; the cut share is the cut spans' overlap with the trim
            # window (exact -- the spans are frame-quantized).
            w0 = start_idx / float(src_fps)
            w1 = end_idx / float(src_fps)
            cut_removed = sum(min(b, w1) - max(a, w0)
                              for (a, b) in tm.cut_spans
                              if min(b, w1) > max(a, w0))
            print("  cuts: {} ranges, ~{:.1f}s removed ({} frames)".format(
                len(tm.cut_spans), cut_removed,
                int(round(cut_removed * src_fps))))
            saved -= cut_removed
        if tm.spans:
            print("  speed-up: {} spans, ~{:.1f}s saved ({} -> {} output frames)".format(
                len(tm.spans), saved, end_idx - start_idx, out_count))
    # The bed is created here, one statement before the try/except that
    # owns its cleanup -- anything that raises between writing it and the
    # big encode `finally` would otherwise orphan the file.
    sfx_path = _write_sfx_bed(total_dur, click_sound, key_sound, sfx_volume,
                              click_sound_times, release_times,
                              key_sound_times)
    try:
        enc = _encode_cmd(
            out_w, out_h, src_fps, raw_path, has_audio, music, out_path,
            audio_start=(start_idx / float(src_fps)),
            audio_duration=(end_idx - start_idx) / float(src_fps),
            sfx_path=sfx_path,
            # The bed is mono; pin it to whatever forms the base of the mix
            # so `amix` cannot resolve the mismatch by downmixing the USER's
            # audio.
            audio_layout=(_probe_audio_layout(raw_path) if has_audio
                          else (_probe_audio_layout(music)
                                if music and os.path.isfile(music) else None)),
            retime_segments=retime_segments,
        )
        proc = subprocess.Popen(enc, stdin=subprocess.PIPE)
    except BaseException:
        _remove_sfx_bed(sfx_path)
        raise
    i = 0
    src_idx = start_idx
    last = None if use_multi else len(path) - 1
    broken = False
    try:
        if start_idx > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        while True:
            if src_idx >= end_idx:
                break
            if emit is not None and not emit[src_idx]:
                # Dropped by the speed-up plan: grab (no BGR retrieve/copy)
                # and move on. This is the source of the perf win at high
                # rates; only the emitted 1-in-r frames pay the retrieve
                # + effects + pipe-write cost.
                #
                # ...unless the eraser is on, which needs the PIXELS of every
                # frame: its clean-plate sample for a run starting at `a` is
                # taken at `a-1`, and a dropped `a-1` would leave that run
                # with nothing to repair from. So with it on, a dropped frame
                # is decoded and observed, then discarded. `cursor_erase` off
                # keeps the literal `cap.grab()` branch.
                if erase is not None:
                    ok, fr = cap.read()
                    if not ok:
                        break
                    if track is not None:
                        fr = track.apply(fr, src_idx)
                    else:
                        fr = _apply_capture_crop(fr, crop)
                    if badge is not None:
                        badge.apply(fr)
                    fr = _apply_user_crop(fr, user_crop)
                    erase.step(src_idx)
                    erase.observe(fr, src_idx)
                elif not cap.grab():
                    break
                src_idx += 1
                continue
            ok, fr = cap.read()
            if not ok:
                break
            # Window capture crops FIRST (a view, no copy), so `edits.windows`
            # rects below are interpreted inside the cropped frame. No-op
            # (identity) when this session wasn't window-targeted. With a
            # geometry track the rect is this frame's own, resized back to the
            # snapshot size so (W, H) stays fixed for everything downstream.
            if track is not None:
                fr = track.apply(fr, src_idx)
            else:
                fr = _apply_capture_crop(fr, crop)
            # ...then macOS's capture indicator, painted out in the window
            # buffer's own coordinates -- before any crop, so a session
            # cropped past the corner is unaffected either way. Identity
            # (None) on every take that is not window-native.
            if badge is not None:
                badge.apply(fr)
            # ...then the editor's crop, also a view. Identity when unset, so
            # an uncropped session reaches the compositor with the same object
            # it always did.
            fr = _apply_user_crop(fr, user_crop)
            # ...then the burned-in pointer comes out, on the SOURCE frame,
            # so everything downstream composites a frame that never had one.
            if erase is not None:
                erase.step(src_idx)
                erase.erase(fr, src_idx)
                erase.observe(fr, src_idx)
            # Effects are keyed to SOURCE-time src_idx (their content is
            # source-frame content); animations play faster inside a sped
            # span, which is the correct time-lapse semantic.
            t = src_idx / float(src_fps)
            if use_multi:
                # Window focus moves the CELLS, so everything keyed to a cell
                # -- the per-card cameras, the composite, the cursor -- takes
                # this frame's layout rather than the baked one.
                cells, order = _focus_cells_at(focus_layout, src_idx,
                                               multi_cells)
                crops = _multi_crops(fr, windows, grid_tracks, src_idx, W, H)
                crops = _apply_card_cameras(crops, card_paths, windows,
                                            cells, src_idx, W, H)
                out = _paint_multi(multi_painter, crops, cells, order,
                                   focus_layout)
                _draw_multi_cursor(out, cursorfx, cells, windows,
                                   grid_tracks, src_idx, W, H,
                                   card_paths=card_paths)
                if focus_layout is not None:
                    out = _apply_focus_camera(
                        out, focus_layout.camera_at(src_idx), out_w, out_h)
            else:
                if out_ord is None:
                    pi = src_idx if src_idx <= last else last
                else:
                    pi = int(out_ord[src_idx])
                    if pi > last:
                        pi = last
                cx, cy, z = path[pi]
                x0, y0 = _camera_window(cx, cy, z, win_w, win_h, W, H)
                if motion_blur:
                    cams = _motion_blur_cams(path, pi, win_w, win_h, out_w, out_h)
                    out = _warp_blend(fr, cams, win_w, win_h, out_w, out_h, W, H)
                else:
                    out = _warp(fr, x0, y0, z, win_w, win_h, out_w, out_h)
                z_eff = _zoom_to_output_scale(z, win_w, out_w)
                if clickfx is not None:
                    clickfx.draw(out, t, x0, y0, z_eff)
                if spot is not None:
                    j = src_idx if src_idx < count else count - 1
                    spot.draw(out, (cur_x[j] - x0) * z_eff, (cur_y[j] - y0) * z_eff, z_eff)
                if cursorfx is not None:
                    cursorfx.draw(out, src_idx, x0, y0, z_eff)
                # Grow the active window in place, AFTER the effects so overlays
                # inside it ride the magnification, BEFORE the framed backdrop.
                if screen_track is not None:
                    out = _apply_screen_grow(out, screen_track, pi, x0, y0, z_eff)
                if painter is not None:
                    out = painter.paint(out)
            if facecam_ov is not None:
                # On top of everything (incl. the framed background), aligned
                # to SOURCE time t so it plays through retime like the screen.
                facecam_ov.draw(out, t)
            if fade > 0.0:
                # Fade is an OUTPUT-time effect (the viewer sees N/fps of
                # video, not the source's stretched idea of it).
                t_local = i / float(src_fps)
                out = _apply_fade(out, t_local, total_dur, fade)
            try:
                proc.stdin.write(out.tobytes())
            except (BrokenPipeError, OSError):
                broken = True  # ffmpeg died; break and report its exit code
                break
            i += 1
            src_idx += 1
            if i % 60 == 0:
                print("  rendered {} frames...".format(i), end="\r", flush=True)
    finally:
        cap.release()
        if facecam_ov is not None:
            facecam_ov.release()
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        rc = proc.wait()
        # After wait(): ffmpeg reads the bed from disk for the whole encode,
        # so removing it any earlier would pull the file out from under a
        # still-running process.
        _remove_sfx_bed(sfx_path)

    if rc != 0 or broken:
        raise RuntimeError(
            "ffmpeg encode failed (exit {}) after {} frames".format(rc, i))
    if i == 0:
        raise RuntimeError("render wrote no frames (trim outside recording?)")
    if not tm.identity and tm.cut_spans and i != out_count:
        # Belt for cut renders (the segmented-join counting discipline): the
        # emit plan promised out_count frames. A shortfall usually means the
        # container over-reported its frame count and the decode ended early
        # -- the output is shorter than planned and audio was carved for the
        # full plan (-shortest reconciles the tail). Loud, not fatal:
        # container frame-count lies are common enough that failing every
        # such take would be worse than the warning.
        print("  warning: cuts expected {} output frames but {} were "
              "written -- the source decode ended early; the exported "
              "tail may be shorter than planned".format(out_count, i))
    if erase is not None:
        # Falling back is a real outcome, not an internal detail: it means
        # the recording never showed those pixels uncovered, and the viewer
        # is looking at an inpaint rather than at recovered footage.
        print(erase.report())
    _badge_report([badge])
    print("\nwrote {} ({} frames)".format(out_path, i))

    if make_gif:
        gif_path = os.path.splitext(out_path)[0] + ".gif"
        _make_gif(out_path, gif_path, fps=max(1, int(gif_fps or 15)),
                  width=max(64, int(gif_width or 1000)))
        print("wrote {}".format(gif_path))
    return out_path


def _covers_window(segments, window_dur):
    """True iff `segments` tile [0, window_dur] gaplessly (any rates).

    The audio no-op collapse in `_encode_cmd` is only sound when the
    segment list covers the whole trim window with no holes: a GAPPED
    all-rate-1.0 list (cuts) still needs the filter graph -- the gaps are
    exactly where the removed audio would otherwise be left in, silently
    desyncing A/V by the total cut length. Tolerance 1e-5: safely above
    segments_for_audio's own 1e-6/1e-9 epsilons (its gapless speedup
    lists must always classify as covering) and far below one frame.
    """
    if not segments:
        return False
    if abs(float(segments[0][0])) > 1e-5:
        return False
    for i in range(len(segments) - 1):
        if abs(float(segments[i][1]) - float(segments[i + 1][0])) > 1e-5:
            return False
    if window_dur is not None:
        if abs(float(segments[-1][1]) - float(window_dur)) > 1e-5:
            return False
    return True


def _encode_cmd(out_w, out_h, src_fps, raw_path, has_audio, music, out_path,
                audio_start=0.0, audio_duration=None,
                sfx_path=None, audio_layout=None,
                retime_segments=None):
    """Build ffmpeg cmd: raw BGR frames + optional audio/music/event SFX.

    `sfx_path`: a pre-mixed click/keystroke bed (see sfx.py) added as ONE
    input. It is mixed with `normalize=0` so the recording keeps exactly its
    own level and the bed adds on top -- the default `amix` normalization
    would halve a voiceover the moment the first click landed.

    `audio_layout`: the recording's channel layout, from
    `_probe_audio_layout`. The bed is mono; without this, `amix` resolves
    the mismatch by downmixing the RECORDING, so turning on click sounds
    silently collapsed a stereo mic to mono (and shifted it 3 dB). Pinning
    the bed to the recording's layout preserves it exactly -- measured
    across mono, stereo, and stereo-with-signal-on-one-channel. None leaves
    the graph untouched, which is right for a mono recording.

    `retime_segments`: if not None and it either contains a segment with
    rate != 1.0 (speedup) or does NOT tile the trim window gaplessly
    (cuts -- see `_covers_window`), the recording audio is rebuilt by an
    atrim+atempo+concat chain per KEPT segment (rate_eff pinned to the
    video warp at segment boundaries; see retime.TimeMap.segments_for_audio;
    cut ranges are simply absent from the list, so the concat butts their
    neighbours). None -- and the gapless "all rate == 1.0" case -- both
    produce the pre-feature graph verbatim, so the speedup off-switch is
    byte-identical.
    """
    enc = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pixel_format", "bgr24",
           "-video_size", "{}x{}".format(out_w, out_h),
           "-framerate", "{:.6f}".format(src_fps), "-i", "-"]
    raw_idx = music_idx = click_idx = None
    idx = 1
    # When retiming, we CANNOT let ffmpeg's -ss / -t clip the audio input
    # -- the atrim windows below use absolute source seconds, and moving
    # the window with -ss would double-clip. Feed the whole audio track
    # and let the filter graph carve it up.
    has_retime = bool(retime_segments) and (
        any(abs(float(r) - 1.0) > 1e-6 for (_a, _b, r) in retime_segments)
        or not _covers_window(retime_segments, audio_duration))
    if has_audio:
        if not has_retime:
            ss = max(0.0, float(audio_start or 0.0))
            if ss > 0.0:
                enc += ["-ss", "{:.6f}".format(ss)]
            if audio_duration is not None:
                d = max(0.0, float(audio_duration))
                if d > 0.0:
                    enc += ["-t", "{:.6f}".format(d)]
        enc += ["-i", raw_path]; raw_idx = idx; idx += 1
    has_music = bool(music) and os.path.isfile(music)
    if has_music:
        enc += ["-i", music]; music_idx = idx; idx += 1
    has_sfx = bool(sfx_path) and os.path.isfile(sfx_path)
    if has_sfx:
        enc += ["-i", sfx_path]; click_idx = idx; idx += 1

    enc += ["-map", "0:v"]
    any_audio = has_audio or has_music or has_sfx
    filter_parts = []
    base_mix = None
    # Retimed recording audio: atrim per segment (absolute source seconds
    # relative to the input, since -ss is disabled when has_retime),
    # atempo chain per non-identity segment, concat into one stream.
    if has_audio and has_retime:
        offset = max(0.0, float(audio_start or 0.0))
        labels = []
        for k, (a_loc, b_loc, rate_eff) in enumerate(retime_segments):
            a_abs = offset + float(a_loc)
            b_abs = offset + float(b_loc)
            if b_abs <= a_abs:
                continue
            lbl = "rt{}".format(k)
            chain = "asetpts=PTS-STARTPTS"
            for step in retime.atempo_chain(rate_eff):
                chain += ",atempo={:.6f}".format(step)
            filter_parts.append(
                "[{}:a]atrim={:.6f}:{:.6f},{}[{}]".format(
                    raw_idx, a_abs, b_abs, chain, lbl))
            labels.append(lbl)
        if labels:
            if len(labels) == 1:
                # Single retimed segment: still relabel to [a_rt] so the
                # downstream music/click/mix graph is agnostic.
                filter_parts.append(
                    "[{}]anull[a_rt]".format(labels[0]))
            else:
                filter_parts.append(
                    "{}concat=n={}:v=0:a=1[a_rt]".format(
                        "".join("[{}]".format(lbl) for lbl in labels),
                        len(labels)))
            rec_label = "[a_rt]"
        else:
            rec_label = "[{}:a]".format(raw_idx)
    else:
        rec_label = "[{}:a]".format(raw_idx) if has_audio else None

    if has_audio and has_music:
        # Keep existing behavior: attenuate the recording track when mixed.
        filter_parts.append("{}volume=0.55[a_rec]".format(rec_label))
        # duration=longest, not `shortest`: with `shortest` a music file
        # SHORTER than the take cut the mic off at the music's end. That
        # used to be loud (the whole export was truncated with it, via
        # `-shortest`), but the SFX bed keeps the audio stream alive to the
        # video's length, so it would now read as narration mysteriously
        # dropping out mid-video. `-shortest` on the output still clips
        # music that runs LONGER than the picture.
        filter_parts.append("[{}:a][a_rec]amix=inputs=2:duration=longest:"
                            "dropout_transition=0[a_base]".format(music_idx))
        base_mix = "[a_base]"
    elif has_audio:
        base_mix = rec_label
    elif has_music:
        base_mix = "[{}:a]".format(music_idx)

    audio_map = None
    if has_sfx:
        sfx_in = "[{}:a]".format(click_idx)
        if base_mix is not None and audio_layout in _SAFE_CHANNEL_LAYOUTS:
            filter_parts.append("{}aformat=channel_layouts={}[a_sfx]".format(
                sfx_in, audio_layout))
            sfx_in = "[a_sfx]"
        if base_mix is not None:
            # normalize=0: the bed ADDS to the recording instead of halving
            # it. Both sides are level-controlled upstream (sfx.py keeps its
            # peaks low, and it clips its own sum), so the sum stays sane.
            # duration=longest (not `first`): a mic track that stopped early
            # must not drag the mix -- and through -shortest, the VIDEO --
            # down to its own length.
            filter_parts.append(
                "{}{}amix=inputs=2:duration=longest:dropout_transition=0:"
                "normalize=0[a_mix]".format(base_mix, sfx_in))
            audio_map = "[a_mix]"
        else:
            # Unbracketed: with no base there is no filter graph at all, so
            # this has to be a stream specifier, not a filter linklabel.
            audio_map = "{}:a".format(click_idx)
    elif has_audio and has_music:
        audio_map = "[a_base]"
    elif has_audio and has_retime:
        # Retimed graph but no music/clicks: [a_rt] is the finished stream.
        audio_map = "[a_rt]" if rec_label == "[a_rt]" else "{}:a".format(raw_idx)
    elif has_audio:
        audio_map = "{}:a".format(raw_idx)
    elif has_music:
        audio_map = "{}:a".format(music_idx)

    if filter_parts:
        enc += ["-filter_complex", ";".join(filter_parts)]
    if audio_map:
        enc += ["-map", audio_map]

    enc += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if any_audio:
        enc += ["-c:a", "aac", "-b:a", "160k", "-shortest"]
    enc += [out_path]
    return enc


def _contain_fit(out_w, out_h, W, H):
    """Largest (out_w, out_h)-aspect window that fits inside (W, H).

    This *is* the zoom-1.0 crop window in source pixels for a render canvas
    whose aspect doesn't match the source: exporting a 1920x1080 recording at
    9:16 yields a tall, narrow ~608x1080 window (the biggest portrait
    rectangle that fits inside the landscape source), which the camera then
    pans/zooms exactly like it would a full-frame window. When out_w:out_h
    already matches W:H (the default "auto" aspect), this reduces to
    (W, H) exactly.
    """
    out_w = max(1.0, float(out_w))
    out_h = max(1.0, float(out_h))
    W = max(1.0, float(W))
    H = max(1.0, float(H))
    scale = min(W / out_w, H / out_h)
    return out_w * scale, out_h * scale


def _camera_window(cx, cy, z, win_w, win_h, W, H):
    """Top-left of the (win_w/z, win_h/z) window centered at (cx, cy),
    clamped to stay inside the source (W, H) bounds."""
    z = max(1.0, float(z))
    cw = win_w / z
    ch = win_h / z
    x0 = min(max(cx - cw / 2.0, 0.0), W - cw)
    y0 = min(max(cy - ch / 2.0, 0.0), H - ch)
    return x0, y0


def _clamp_window_rect(rect, W, H, min_dim=8):
    """Defensive clamp of a multi-window crop rect into the real decoded
    frame bounds, mirroring `_camera_window`'s clamp style.

    `edits._normalize_window_rect` deliberately does NOT clamp spatially
    (see its docstring) -- the zoom pin's x/y are unclamped there too, and
    spatial safety is enforced downstream against the real frame instead.
    This is that downstream enforcement for windows.
    """
    W, H = int(W), int(H)
    min_dim = max(1, int(min_dim))

    def _f(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    x = max(0.0, min(_f(rect.get("x"), 0.0), W - min_dim))
    y = max(0.0, min(_f(rect.get("y"), 0.0), H - min_dim))
    w = max(min_dim, min(_f(rect.get("w"), min_dim), W - x))
    h = max(min_dim, min(_f(rect.get("h"), min_dim), H - y))
    return int(round(x)), int(round(y)), int(round(w)), int(round(h))


def _crop_window_rect(frame_bgr, rect, W, H):
    x, y, w, h = _clamp_window_rect(rect, W, H)
    return frame_bgr[y:y + h, x:x + w]


def _multi_crops(frame_bgr, windows, grid_tracks, frame_index, W, H):
    """One crop per grid card: the card's tracked rect where it bound to a
    real window, else the static rect it was drawn as.

    A bound card's crop is resized back to the drawn rect's size by
    `_WindowTrack.apply`, which is what keeps every cell's aspect -- and so
    the whole grid layout -- fixed while its window moves or resizes.
    """
    out = []
    for i, spec in enumerate(windows):
        tr = grid_tracks[i] if grid_tracks and i < len(grid_tracks) else None
        if tr is None:
            out.append(_crop_window_rect(frame_bgr, spec, W, H))
        else:
            out.append(tr.apply(frame_bgr, frame_index))
    return out


_CARD_ZOOM_EPS = 1e-3


def _multi_card_rect(spec, track, frame_index, W, H):
    """The source-px rect a card is SHOWING at `frame_index` -- its tracked
    rect where it bound to a real window, else the static rect it was drawn
    as. The exact rect `_multi_crops` slices, expressed as numbers."""
    if track is not None:
        return track.rect_at(frame_index)
    return _clamp_window_rect(spec, W, H)


def _source_to_card(spec, track, frame_index, W, H, px, py):
    """Map a SOURCE point into a card's own stabilized space, or None if the
    card isn't showing that point at this frame.

    Two rects, and the difference is the whole subtlety here. The card is
    SHOWING `_multi_card_rect` (which moves when it follows a window), but the
    pixels handed to the compositor are always in the DRAWN rect's space,
    because `_WindowTrack.apply` resizes every tracked crop back to that size
    to keep the grid geometry fixed. So the map is "into the shown rect, then
    scaled by drawn/shown" -- the same composition `_WindowTrack.to_src_px`
    does for a single-window capture.
    """
    sx, sy, sw, sh = _multi_card_rect(spec, track, frame_index, W, H)
    if sw <= 0 or sh <= 0:
        return None
    if not (sx <= px < sx + sw and sy <= py < sy + sh):
        return None
    _dx, _dy, dw, dh = _clamp_window_rect(spec, W, H)
    return (px - sx) * dw / float(sw), (py - sy) * dh / float(sh)


def _card_to_cell(spec, cell, path, frame_index, W, H):
    """`(x0, y0, zx, zy)` mapping a card's own space onto its output cell.

    Without a camera this is the plain fit `paint()` performs. With one it is
    the camera window, exactly as the single-source path composes
    `_camera_window` with `_warp` -- which is why the cursor needs no special
    case for zooming cards: it goes through this same transform either way.
    """
    _dx, _dy, dw, dh = _clamp_window_rect(spec, W, H)
    if path is None or not len(path):
        return 0.0, 0.0, cell["w"] / float(dw), cell["h"] / float(dh)
    idx = int(min(max(int(frame_index), 0), len(path) - 1))
    cx, cy, z = path[idx]
    x0, y0 = _camera_window(cx, cy, z, dw, dh, dw, dh)
    return x0, y0, cell["w"] / (dw / z), cell["h"] / (dh / z)


def _build_card_cameras(windows, cells, grid_tracks, clicks_t, clicks_x,
                        clicks_y, frame_times, W, H, max_zoom, params,
                        suppressed_ranges, plan_duration, src_fps,
                        enabled=False):
    """One camera path per card, in that card's own space, or None.

    Returns a list parallel to `windows`: a (T, 3) `(cx, cy, z)` path for a
    card that actually zooms, None for one that never does. None is not just
    an optimization -- it keeps a still card going through the untouched
    `paint()` resize rather than a warp, so switching this on does not
    resample every card in the grid.

    Clicks are assigned to a card by containment at the moment they happened
    (so a followed window keeps its clicks while it moves) and translated into
    card space. A click inside two overlapping cards counts for both, matching
    how the composite duplicates those pixels -- the same rule
    `_draw_multi_cursor` uses.
    """
    n = len(windows or [])
    if not enabled or not n:
        return [None] * n
    cards = []
    for i, spec in enumerate(windows):
        tr = grid_tracks[i] if grid_tracks and i < len(grid_tracks) else None
        _dx, _dy, dw, dh = _clamp_window_rect(spec, W, H)
        picked = []
        for k in range(len(clicks_t)):
            t = float(clicks_t[k])
            pt = _source_to_card(spec, tr, int(round(t * src_fps)), W, H,
                                 float(clicks_x[k]), float(clicks_y[k]))
            if pt is not None:
                picked.append((t, pt[0], pt[1]))
        cards.append({"w": dw, "h": dh, "clicks": picked})
    paths = camera.build_card_paths(
        frame_times, cards, max_zoom=max_zoom, params=params,
        suppressed_ranges=suppressed_ranges, plan_duration=plan_duration)
    out = []
    for path in paths:
        zoomed = path.size and float(np.max(path[:, 2])) > 1.0 + _CARD_ZOOM_EPS
        out.append(path if zoomed else None)
    return out


def _card_click_times(windows, grid_tracks, clicks_t, clicks_x, clicks_y,
                      W, H, src_fps):
    """Per-card lists of click TIMES, one list per window, by containment at
    the moment each click happened -- the same ownership rule
    `_build_card_cameras` and `_draw_multi_cursor` use, so a followed window
    keeps its clicks while it moves and a click inside two overlapping cards
    counts for both.

    Only the times: the composition camera's subject is the CARD, so where in
    the window the click landed is not its business.
    """
    out = []
    for i, spec in enumerate(windows or []):
        tr = grid_tracks[i] if grid_tracks and i < len(grid_tracks) else None
        picked = []
        for k in range(len(clicks_t)):
            t = float(clicks_t[k])
            if _source_to_card(spec, tr, int(round(t * src_fps)), W, H,
                               float(clicks_x[k]),
                               float(clicks_y[k])) is not None:
                picked.append(t)
        out.append(picked)
    return out


def _build_focus_emphasis(windows, grid_tracks, clicks_t, clicks_x, clicks_y,
                          frame_times, W, H, max_zoom, params,
                          suppressed_ranges, plan_duration, src_fps,
                          manual=None, enabled=False):
    """Per-card emphasis over time for window focus, or None.

    None means "nothing is ever the subject", which callers take as the
    signal to composite through the untouched baked layout -- so switching
    this on costs nothing on a take where it has nothing to do.
    """
    if not enabled or not windows:
        return None
    times = _card_click_times(windows, grid_tracks, clicks_t, clicks_x,
                              clicks_y, W, H, src_fps)
    cards = [{"click_times": ts} for ts in times]
    return camera.build_focus_emphasis(
        frame_times, cards, max_zoom=max_zoom, params=params,
        suppressed_ranges=suppressed_ranges, plan_duration=plan_duration,
        manual=manual)


# Window focus is a TWO-RUNG ladder and the rungs are different mechanisms:
#
#   stage 1 (first click)  -- the card GROWS in place, its neighbours hold
#   stage 2 (second click) -- the whole composition ZOOMS IN on that card
#
# One emphasis track drives both. `focus_lean` (0.5) is the stage-1 plateau,
# so growth completes there and the screen zoom owns the run from there to
# 1.0. Splitting it that way is what makes the second click feel like a
# separate, bigger move rather than more of the first.
#
# Doing the growth FIRST also keeps the sampling honest: the card is already
# composited into a larger cell before the canvas is warped, so the export
# stays near 1:1 with the source instead of upscaling a small cell.
_FOCUS_ZOOM_MARGIN = 0.05     # stage 2 leaves this much backdrop around it
_FOCUS_MAX_UPSCALE = 1.3      # never magnify a window this far past capture


def _focus_weights(emphasis_row, lean):
    """Split one frame's emphasis into `(grow, zoom)` weights per card."""
    lean = max(1e-6, min(0.999, float(lean)))
    grow = np.minimum(1.0, emphasis_row / lean)
    zoom = np.maximum(0.0, (emphasis_row - lean) / (1.0 - lean))
    return grow, zoom


class _FocusLayout:
    """Per-frame card placement AND composition camera for window focus.

    Holds the base layout and one grown layout per card, and blends them by
    that frame's growth weight. Both endpoints are computed ONCE -- only the
    blend is per frame -- so the cost of the animation is a handful of lerps
    plus the backdrop rebuild in `paint_at`, not a re-layout.

    `cells_at` returns the same dict shape `MultiFramePainter.cells` does,
    because every downstream consumer (the per-card cameras, the synthetic
    cursor, the editor payload) already speaks it and none of them care
    whether the layout is moving.
    """

    def __init__(self, painter, emphasis, lean=0.5):
        self.base = painter.base_placements
        self.targets = [painter.focus_targets(i)
                        for i in range(len(self.base))]
        self.emphasis = emphasis
        self.lean = lean
        self.W, self.H = painter.W, painter.H
        self.source_sizes = painter.source_sizes
        # Stage 2's destination, per card: how far the canvas must zoom for
        # that card's GROWN cell to fill the frame, and where to aim. Both
        # constant, because growth has finished by the time the zoom starts
        # -- so the two animations never feed back into each other.
        self.cam = []
        for i, t in enumerate(self.targets):
            x, y, w, h = t[i]
            z = min(self.W / float(max(1, w)), self.H / float(max(1, h)))
            z = max(1.0, z * (1.0 - _FOCUS_ZOOM_MARGIN))
            # Don't blow a window up far past the pixels the recording
            # actually has. The grown cell is already a real magnification of
            # the source crop; this bounds the TOTAL, which is what stops a
            # short wide card (whose cell is a big downscale) from being
            # pushed to 2.4x of soft.
            src_w = float(max(1.0, self.source_sizes[i][0]))
            cap = _FOCUS_MAX_UPSCALE * src_w / float(max(1, w))
            self.cam.append((x + w / 2.0, y + h / 2.0, max(1.0, min(z, cap))))

    def _weights(self, frame_index):
        i = int(min(max(int(frame_index), 0), len(self.emphasis) - 1))
        return _focus_weights(self.emphasis[i], self.lean)

    def _cells(self, frame_index):
        grow, _zoom = self._weights(frame_index)
        if float(np.max(grow)) <= 1e-4:
            return list(self.base), list(range(len(self.base)))
        return framing.blend_placements(self.base, self.targets, grow)

    def cells_at(self, frame_index):
        """`(cells, draw_order)` for one frame."""
        placements, order = self._cells(frame_index)
        return ([{"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                  "radius": max(8, int(0.02 * w))}
                 for (x, y, w, h) in placements], order)

    def camera_at(self, frame_index):
        """`(cx, cy, z)` in canvas px for stage 2, or None below it.

        Only ever one card is zooming (the emphasis arbitration guarantees
        it), so the weights are summed rather than argmaxed -- during the
        instant a hand-over overlaps, the two contributions pull the level
        down toward 1.0, i.e. the composition eases back out before pushing
        into the next subject. That is the right move, not a compromise
        framing between two windows.
        """
        _grow, zoom = self._weights(frame_index)
        total = float(np.sum(zoom))
        if total <= 1e-4:
            return None
        k = (1.0 / total) if total > 1.0 else 1.0
        cx = cy = 0.0
        z = 1.0
        for i, wt in enumerate(zoom):
            wt = float(wt) * k
            if wt <= 1e-6:
                continue
            tcx, tcy, tz = self.cam[i]
            cx += wt * tcx
            cy += wt * tcy
            z += wt * (tz - 1.0)
        if z <= 1.0 + 1e-6:
            return None
        # Cards with no weight contributed no centre, so re-anchor what is
        # left of the blend on the canvas middle.
        rest = max(0.0, 1.0 - min(1.0, total))
        cx += rest * self.W / 2.0
        cy += rest * self.H / 2.0
        return cx, cy, z


def _apply_focus_camera(canvas, cam, out_w, out_h):
    """Stage 2: warp the composed canvas so the subject fills the frame.

    Runs AFTER the cards, their own cameras and the synthetic cursor are all
    on the canvas -- by then the scene is one image, so the push-in moves the
    grown card, its neighbours and the backdrop together, which is what makes
    it read as the whole screen zooming rather than a second layout change.
    The facecam bubble is drawn after this, deliberately: an overlay is not
    part of the scene.
    """
    if cam is None:
        return canvas
    cx, cy, z = cam
    ch, cw = canvas.shape[:2]
    kx, ky = cw / float(out_w), ch / float(out_h)
    x0, y0 = _camera_window(cx * kx, cy * ky, z, cw, ch, cw, ch)
    return _warp(canvas, x0, y0, z, cw, ch, out_w, out_h)


def _focus_cells_at(focus_layout, frame_index, base_cells):
    """`(cells, order)` for this frame: the animated layout when focus is
    running, else the painter's own baked cells and natural order (which
    keeps the composite bit-exact)."""
    if focus_layout is None:
        return base_cells, None
    return focus_layout.cells_at(frame_index)


def _paint_multi(painter, crops, cells, order, focus_layout):
    """`paint()` when the layout is static, `paint_at()` when it moves."""
    if focus_layout is None:
        return painter.paint(crops)
    return painter.paint_at(crops, cells, order=order)


def _apply_card_cameras(crops, card_paths, windows, cells, frame_index, W, H):
    """Warp each zooming card's crop to its cell size through its own camera.

    `MultiFramePainter.paint` resizes every crop to the cell anyway and a
    same-size resize is an identity copy, so handing it an already-cell-sized
    warp slots in with no compositor change at all. Cards with no camera are
    left exactly as `_multi_crops` produced them.
    """
    if not card_paths:
        return crops
    out = list(crops)
    for i, path in enumerate(card_paths):
        if path is None or i >= len(out):
            continue
        cell = cells[i]
        _dx, _dy, dw, dh = _clamp_window_rect(windows[i], W, H)
        idx = int(min(max(int(frame_index), 0), len(path) - 1))
        cx, cy, z = path[idx]
        x0, y0 = _camera_window(cx, cy, z, dw, dh, dw, dh)
        # (dw/z, dh/z) -> the cell. At z == 1 that is the whole card stretched
        # to the cell, i.e. exactly the framing paint()'s resize gives -- which
        # is why the zoom-1.0 window must be the CARD and not a cell-aspect
        # sub-box: _contain_fit would centre-crop, silently throwing content
        # away whenever a cell's aspect differs from its card's (clamped rects,
        # hand-dragged cards, a changed --aspect).
        out[i] = _warp(out[i], x0, y0, z, dw, dh,
                       int(cell["w"]), int(cell["h"]))
    return out


def _draw_multi_cursor(out, cursorfx, cells, windows, grid_tracks,
                       frame_index, W, H, card_paths=None):
    """Draw the synthetic cursor into every card that is showing it.

    Multi-window mode has no camera path, which is why the camera-driven
    effects are off here -- but the cursor never needed one. Each card is an
    exact, known crop-and-scale of the source, so the cursor's source
    position maps into a card by that card's own transform: the same
    `(p - origin) * scale` shape `CursorFX.draw` already takes, with the
    card's rect as the origin and its cell-to-rect ratio as the scale. The
    camera's `(x0, y0, z_eff)` was only ever one instance of that.

    Two deliberate calls:
    - A card is drawn into only when the cursor is inside the rect that card
      is SHOWING, so a cursor over the terminal doesn't ghost onto the browser
      card next to it.
    - Overlapping cards each get it. Their source rects overlap, so the same
      pixels genuinely appear twice in the composite; drawing the cursor once
      would put it on a copy of the screen that no longer agrees with the
      other copy. (The picker already flags overlapping picks -- see
      docs/architecture.md -- because that duplication is a capture problem, not
      something the compositor can undo.)

    Clipping is free: each card's destination box is passed as a NumPy view,
    so `_paint`'s own bounds clamp cuts anything that would spill onto the
    background. Only the rounded corners are approximated -- a cursor in the
    outer ~2% corner arc can touch the background.
    """
    if cursorfx is None:
        return
    idx = int(min(max(int(frame_index), 0), len(cursorfx.sx) - 1))
    if float(cursorfx.alpha[idx]) <= 0.0:
        return
    cx, cy = float(cursorfx.sx[idx]), float(cursorfx.sy[idx])
    for i, spec in enumerate(windows):
        tr = grid_tracks[i] if grid_tracks and i < len(grid_tracks) else None
        pt = _source_to_card(spec, tr, frame_index, W, H, cx, cy)
        if pt is None:
            continue
        cell = cells[i]
        ix, iy, fw, fh = cell["x"], cell["y"], cell["w"], cell["h"]
        if fw <= 0 or fh <= 0:
            continue
        view = out[iy:iy + fh, ix:ix + fw]
        if view.size == 0:
            continue
        path = card_paths[i] if card_paths and i < len(card_paths) else None
        x0, y0, zx, zy = _card_to_cell(spec, cell, path, frame_index, W, H)
        # `draw` maps (p - x0) * z, so hand it the cursor's CARD-space
        # position and the card's own transform -- with a camera that is the
        # zoom window, without one it's the plain fit. x and y scales differ
        # only when a tracked window changed aspect (its crop is resized back
        # to the drawn rect's size); see CursorFX.draw's z_eff_y.
        cursorfx.draw(view, idx, x0, y0, zx, z_eff_y=zy, pos=pt)


# -- Record-time window capture ---------------------------------------------
# avfoundation cannot target a window, so `record --window` captures the whole
# display and snapshots the window's rect (Quartz, POINTS, global top-left
# origin) into meta["capture_window"]. The crop happens HERE, at render time,
# right after the points->pixels scale is derived from the real file -- so the
# window rect goes through the exact same, per-axis, fractional-Retina-exact
# scale as every recorded event coordinate. One source of truth; storing
# pixels in meta.json too would be a second one that can silently disagree.
#
# Applying the crop rebinds (W, H) to the WINDOW's size for everything
# downstream: camera.build_path, framing.output_size/_contain_fit,
# _camera_window, _zoom_to_output_scale and the effects classes all take the
# source bounds as parameters, so they move into window space for free -- and
# the headline auto-zoom keeps working, unlike the multi-window compositor
# path (which force-disables click FX / spotlight / cursor FX / motion blur).

_CAPTURE_MOVE_TOL_PT = 4.0    # per-edge slop before we call the window "moved"


def _capture_crop_px(meta, W, H, min_dim=16):
    """Source-pixel crop rect `(x, y, w, h)` for a window-targeted recording,
    or None when this session isn't one (or the snapshot can't be trusted).

    `W`/`H` are the RAW decoded dimensions of `raw.mov`; the scale comes from
    meta's `logical_w`/`logical_h`, per axis.

    Integerization is OUTWARD (floor the origin, ceil the far edge) so a
    fractional rect never shaves a fringe off the window, then the size is
    even-ized INWARD. **The even dimensions are load-bearing, not cosmetic**:
    `_encode_cmd` feeds `-video_size {out_w}x{out_h}` into
    `libx264 -pix_fmt yuv420p`, which rejects odd dimensions -- an odd crop
    would break every export. (`framing._even` rounds UP, which here would
    overrun the frame, hence the local inward even-ize.)

    Fails safe to None -- i.e. today's full-frame render, bit-exact -- on
    anything unexpected: key absent or not a dict, units other than points,
    a malformed/non-finite rect, a sub-`min_dim` window, or a rect with no
    overlap at all with the recorded frame (which is what a secondary-display
    capture looks like; multi-display is out of scope for v1).
    """
    if not isinstance(meta, dict):
        return None
    cw = meta.get("capture_window")
    if not isinstance(cw, dict):
        return None
    if cw.get("mode") == "window_native":
        # raw.mov IS the window's own buffer -- there is nothing to crop, and
        # cropping the display rect out of a bare-window frame would be a
        # double-crop. render branches on `mode` HERE, before the crop math;
        # the window-native event mapper lives in _build_window_track instead.
        return None
    if cw.get("units") != "points":
        return None
    rect = cw.get("rect")
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    origin = cw.get("display_origin")
    if origin is None:
        origin = (0.0, 0.0)
    if not isinstance(origin, (list, tuple)) or len(origin) != 2:
        return None
    try:
        rx, ry, rw, rh = (float(v) for v in rect)
        ox, oy = (float(v) for v in origin)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (rx, ry, rw, rh, ox, oy)):
        return None

    W, H = int(W), int(H)
    if W < 2 or H < 2:
        return None
    try:
        scale_x = W / float(meta.get("logical_w", W) or W)
        scale_y = H / float(meta.get("logical_h", H) or H)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not (np.isfinite(scale_x) and np.isfinite(scale_y)):
        return None
    if scale_x <= 0.0 or scale_y <= 0.0:
        return None

    x_px = (rx - ox) * scale_x
    y_px = (ry - oy) * scale_y
    w_px = rw * scale_x
    h_px = rh * scale_y
    min_dim = max(2, int(min_dim))
    if w_px < min_dim or h_px < min_dim:
        return None
    if x_px + w_px <= 0.0 or y_px + h_px <= 0.0 or x_px >= W or y_px >= H:
        return None    # wrong display / entirely off-frame -> full frame

    x0 = max(0, min(int(np.floor(x_px)), W - 2))
    y0 = max(0, min(int(np.floor(y_px)), H - 2))
    x1 = max(x0 + 2, min(int(np.ceil(x_px + w_px)), W))
    y1 = max(y0 + 2, min(int(np.ceil(y_px + h_px)), H))
    w = x1 - x0
    h = y1 - y0
    if w % 2:
        w -= 1
    if h % 2:
        h -= 1
    return int(x0), int(y0), int(w), int(h)


def _smooth1d(a, sigma_frames):
    """Zero-phase Gaussian smooth along a uniform grid, edges held.

    Zero-phase (centred, not causal) on purpose: this runs offline over the
    whole track, so there is no reason to accept the lag a causal filter
    would add -- a smoothed drag must not trail the real window.
    """
    a = np.asarray(a, dtype=float)
    if a.size < 3 or sigma_frames <= 0:
        return a
    radius = int(max(1, round(3.0 * sigma_frames)))
    if radius * 2 + 1 > a.size:
        radius = max(1, (a.size - 1) // 2)
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / float(sigma_frames)) ** 2)
    k /= k.sum()
    return np.convolve(np.pad(a, radius, mode="edge"), k, mode="valid")


# Smoothing width for the window geometry track, in seconds. Quartz reports
# integral point rects sampled at record.WINDOW_POLL_SEC, so a raw drag is a
# staircase; this is wide enough to read as continuous motion and narrow
# enough that the crop doesn't visibly lag the window.
_WINDOW_TRACK_SMOOTH_SEC = 0.12

_WINDOW_TRACK_MIN_PX = 16


class _WindowTrack:
    """Per-frame crop for a window capture whose geometry was logged over time.

    Without a track, `--capture-window` crops every frame to ONE snapshot: move
    the window and its content slides out of frame (revealing whatever is
    behind), resize it and the crop reframes nothing. With a track, each frame
    is cropped to where the window actually was, then **resized back to the
    snapshot's size**.

    That resize is what keeps the change contained. `(W, H)` stays fixed, so
    the output size, the camera plan, every effect's coordinate space and the
    editor's viewport->source mapping are all untouched, and a resized window
    scales smoothly into a stable canvas -- instead of the video changing
    dimensions mid-stream (impossible) or the framing jumping.
    """

    def __init__(self, rects_px, base, raw_w, raw_h, frame_times,
                 scale_x, scale_y):
        self.base_x, self.base_y, self.base_w, self.base_h = base
        self.times = np.asarray(frame_times, dtype=float)
        self.scale_x, self.scale_y = float(scale_x), float(scale_y)
        r = np.asarray(rects_px, dtype=float).copy()
        # Clamp into the real decoded frame, mirroring _clamp_window_rect's
        # style: a window dragged half off-screen must still slice.
        r[:, 2] = np.clip(r[:, 2], _WINDOW_TRACK_MIN_PX, raw_w)
        r[:, 3] = np.clip(r[:, 3], _WINDOW_TRACK_MIN_PX, raw_h)
        r[:, 0] = np.clip(r[:, 0], 0.0, raw_w - r[:, 2])
        r[:, 1] = np.clip(r[:, 1], 0.0, raw_h - r[:, 3])
        self._f = r
        self._i = np.rint(r).astype(int)

    def _idx(self, i):
        return int(min(max(int(i), 0), len(self._i) - 1))

    def rect_at(self, frame_index):
        x, y, w, h = self._i[self._idx(frame_index)]
        return int(x), int(y), int(w), int(h)

    def apply(self, frame_bgr, frame_index):
        """Crop this frame to the tracked rect and scale it back to the
        snapshot size, so downstream sees a constant (W, H)."""
        x, y, w, h = self.rect_at(frame_index)
        view = frame_bgr[y:y + h, x:x + w]
        if w == self.base_w and h == self.base_h:
            return view
        # INTER_AREA only genuinely helps when shrinking; a window that grew
        # past its snapshot size is being upscaled, where it just softens.
        interp = cv2.INTER_AREA if w > self.base_w else cv2.INTER_LINEAR
        return cv2.resize(view, (self.base_w, self.base_h), interpolation=interp)

    def to_src_px(self, t, x_px, y_px):
        """Map RAW-pixel coordinates at media time `t` into tracked window
        space -- the same space `apply` produces, so a click stays pinned to
        the UI it hit even while the window moves or resizes."""
        x_px = np.asarray(x_px, dtype=float)
        y_px = np.asarray(y_px, dtype=float)
        if x_px.size == 0:
            return x_px, y_px
        t = np.asarray(t, dtype=float)
        rx = np.interp(t, self.times, self._f[:, 0])
        ry = np.interp(t, self.times, self._f[:, 1])
        rw = np.maximum(np.interp(t, self.times, self._f[:, 2]), 1.0)
        rh = np.maximum(np.interp(t, self.times, self._f[:, 3]), 1.0)
        return ((x_px - rx) * (self.base_w / rw),
                (y_px - ry) * (self.base_h / rh))

    def to_src(self, t, ax, ay):
        """`to_src_px` for recorded POINT coordinates (the event streams)."""
        return self.to_src_px(t,
                              np.asarray(ax, dtype=float) * self.scale_x,
                              np.asarray(ay, dtype=float) * self.scale_y)


class _NativeWindowTrack:
    """Event mapping for a window-native (occlusion-free) take.

    raw.mov IS the window's own buffer, so `apply` is IDENTITY -- there is no
    display to crop. `to_src` maps a global-POINT event into that buffer:
    subtract the window's live top-left (points, from the geometry track),
    scale points->pixels by the display's backing scale, then by the letterbox
    FIT factor SCK applies when a resized window is fitted into the fixed
    buffer. That fit is aspect-preserved and TOP-LEFT anchored (measured
    2026-08-23), so there is no centering offset -- just one uniform scale that
    is exactly 1 at the capture-start size and shrinks only after a resize.

    Coordinates are CLAMPED into the buffer, never dropped: a click that
    strayed off the window keeps its TIME in the shared trailing-cluster set
    (camera.cluster_to_range / edits.auto_zoom_proposals / beats) -- the hard
    invariant -- while its position stays in bounds so it can't drag the camera
    off the window. See docs/architecture.md.
    """

    def __init__(self, rects_pt, raw_w, raw_h, frame_times, scale):
        # rects_pt: (N,4) window rect [x,y,w,h] in POINTS (global top-left) on
        # the frame grid. scale: points->pixels (display backing scale).
        self._r = np.asarray(rects_pt, dtype=float)
        self.raw_w, self.raw_h = int(raw_w), int(raw_h)
        self.times = np.asarray(frame_times, dtype=float)
        self.scale = float(scale)

    def _idx(self, i):
        return int(min(max(int(i), 0), len(self._r) - 1))

    def rect_at(self, frame_index):
        x, y, w, h = self._r[self._idx(frame_index)]
        return float(x), float(y), float(w), float(h)

    def apply(self, frame_bgr, frame_index):
        return frame_bgr            # the frame already IS the window

    def to_src(self, t, ax, ay):
        """Global-POINT event coords at media time `t` -> window-buffer pixels."""
        ax = np.asarray(ax, dtype=float)
        ay = np.asarray(ay, dtype=float)
        if ax.size == 0:
            return ax, ay
        t = np.asarray(t, dtype=float)
        ox = np.interp(t, self.times, self._r[:, 0])
        oy = np.interp(t, self.times, self._r[:, 1])
        ww = np.maximum(np.interp(t, self.times, self._r[:, 2]) * self.scale, 1.0)
        wh = np.maximum(np.interp(t, self.times, self._r[:, 3]) * self.scale, 1.0)
        # SCK fits the window's native-pixel size into the fixed buffer,
        # aspect-preserved; =1 at the start size, <1 after growing the window.
        fit = np.minimum(self.raw_w / ww, self.raw_h / wh) * self.scale
        return (np.clip((ax - ox) * fit, 0.0, self.raw_w),
                np.clip((ay - oy) * fit, 0.0, self.raw_h))

    def to_src_px(self, t, x_px, y_px):
        """Raw-decoded-pixel coords -> source. For a native take the raw buffer
        IS the source space, so this is identity plus a bounds clamp (used by
        the eraser's anchor mapping and raw-rect tracks)."""
        return (np.clip(np.asarray(x_px, dtype=float), 0.0, self.raw_w),
                np.clip(np.asarray(y_px, dtype=float), 0.0, self.raw_h))


def _is_window_native_meta(meta):
    """True for an occlusion-free, capture-the-window's-own-buffer take."""
    cw = meta.get("capture_window") if isinstance(meta, dict) else None
    return isinstance(cw, dict) and cw.get("mode") == "window_native"


def _is_multi_native_meta(meta):
    """True for a P3.1 multi-window native take: manifest lists N channels,
    each `mode: window_native`, and there is no top-level `raw` (per-file
    lives in the manifest, mutually exclusive with the single-file paths).
    """
    if not isinstance(meta, dict):
        return False
    channels = meta.get("capture_channels")
    if not isinstance(channels, list) or len(channels) < 2:
        return False
    return all(isinstance(ch, dict) and ch.get("mode") == "window_native"
               for ch in channels)


def _multi_native_enc_cmd(out_w, out_h, src_fps, out_path,
                          audio_path=None, audio_skip_s=0.0,
                          sfx_path=None, audio_layout=None):
    """ffmpeg argv for the multi-native composite: raw BGR frames on stdin,
    plus channel 0's audio when it carries a mic track, plus the event-sound
    bed when there is one.

    Off-switch: `audio_path=None, sfx_path=None` (a mic-less take with
    sounds off) returns the exact video-only command the fleet render always
    used -- pinned byte-for-byte -- and `audio_path` alone returns the exact
    mic command it used before the bed existed. Both are pinned.

    When audio is present, channel 0 is an input seeked by `audio_skip_s`
    (its own skip to the shared composite origin, so the voiceover stays in
    sync) and `-shortest` caps the muxed audio at the composite's video
    length.

    `audio_layout` pins the mono bed to the mic's channel layout before the
    mix; without it `amix` resolves the mismatch by downmixing the MIC (a
    stereo voiceover comes out mono and 3 dB hot). See `_probe_audio_layout`.

    The bed is NEVER seeked. `-ss` binds to the input it precedes, so it
    moves the mic only -- measured, not assumed. That is exactly why the
    bed's event times must already be on the composite OUTPUT clock rather
    than in any channel's own frame of reference: nothing downstream will
    shift them into place.
    """
    enc = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "rawvideo", "-pixel_format", "bgr24",
           "-video_size", "{}x{}".format(out_w, out_h),
           "-framerate", "{:.6f}".format(src_fps), "-i", "-"]
    idx = 1
    audio_idx = sfx_idx = None
    if audio_path is not None:
        if audio_skip_s and audio_skip_s > 1e-6:
            enc += ["-ss", "{:.6f}".format(audio_skip_s)]
        enc += ["-i", audio_path]
        audio_idx = idx
        idx += 1
    if sfx_path is not None and os.path.isfile(sfx_path):
        enc += ["-i", sfx_path]
        sfx_idx = idx
        idx += 1
    enc += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if audio_idx is not None and sfx_idx is not None:
        # normalize=0 so the bed ADDS to the mic instead of halving it;
        # duration=longest so a short mic cannot drag the mix -- and
        # through -shortest, the video -- down to its own length. Same
        # tokens, same reasons, as the single-file `_encode_cmd`.
        graph = ""
        sfx_in = "[{}:a]".format(sfx_idx)
        if audio_layout in _SAFE_CHANNEL_LAYOUTS:
            graph += "{}aformat=channel_layouts={}[a_sfx];".format(
                sfx_in, audio_layout)
            sfx_in = "[a_sfx]"
        graph += ("[{}:a]{}amix=inputs=2:duration=longest:"
                  "dropout_transition=0:normalize=0[a_mix]".format(
                      audio_idx, sfx_in))
        enc += ["-filter_complex", graph,
                "-map", "0:v:0", "-map", "[a_mix]",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    elif audio_idx is not None or sfx_idx is not None:
        # Exactly one audio input, whichever it is: map it straight through.
        # With only a bed this is input 1, so the argv is byte-identical in
        # shape to the mic-only one it replaces.
        only = audio_idx if audio_idx is not None else sfx_idx
        enc += ["-map", "0:v:0", "-map", "{}:a:0".format(only),
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    enc += [out_path]
    return enc


# Default output canvas for a multi-native composite when the caller does
# not pass --aspect. 1920x1080 matches the aspect users expect from
# "render a screen recording", and keeps MultiFramePainter's per-frame
# GaussianBlur affordable (a 2880x1800 default would triple the cost with
# no visible win at typical playback resolutions). Overridden by
# framing.output_size(default_w, default_h, style, aspect=aspect) when the
# user passes --aspect, so a 9:16 or a WxH custom lands exactly.
_MULTI_NATIVE_DEFAULT_W = 1920
_MULTI_NATIVE_DEFAULT_H = 1080
# Ceiling for the source-sized canvas (`_multi_native_canvas`). Four Retina
# windows can ask for well past 4K, and every extra pixel is paid on every
# frame of the encode. Expressed as a longest EDGE plus a pixel budget rather
# than a width/height pair, so a vertical export is allowed to grow tall --
# capping its height at 1080p's would leave 9:16 permanently soft.
_MULTI_NATIVE_MAX_DIM = 3840
_MULTI_NATIVE_MAX_PIXELS = 3840 * 2160

# Aligned lockstep decode across N raw_i.mov files, in output frames.
# Per-channel `t0_monotonic` in the manifest can differ from the session
# origin by SCK spin-up jitter; measured on the P3.1 demo it was ~10ms
# (0-1 frames at 60fps). A round-off larger than this bound warns the
# operator that per-card motion may drift by that many frames -- it does
# not silently degrade the composite.
_MULTI_NATIVE_DRIFT_WARN_FRAMES = 2


def _render_multi_native(session_dir, out_path=None, style="clean",
                          background=None, aspect=None, window_layout="grid",
                          motion_blur=False, click_fx=False,
                          spotlight=False,
                          cursor_fx=False, cursor_params=None,
                          cursor_erase=False,
                          facecam=False, speedup=False, make_gif=False,
                          gif_fps=15, gif_width=1000,
                          window_zoom=False, window_focus=False,
                          max_zoom=2.0, params=None,
                          suppressed_ranges=None, focus_ranges=None,
                          max_height=None, channel_layouts=None,
                          hidden_channels=None,
                          click_sound=None, key_sound=None, sfx_volume=1.0,
                          badge_erase=True,
                          **_ignored):
    """Composite render for a P3.1 multi-window native session
    (P3.2 static composite + P3.3 per-card camera & cursor).

    Reads meta's `capture_channels` manifest, opens N `cv2.VideoCapture`
    instances, and composites their frames onto ONE canvas via
    `framing.MultiFramePainter` -- the same painter the display-crop
    multi-window path uses, source-agnostic above the decode seam so it
    needs no math change here.

    Per-card camera (P3.3, on when `window_zoom=True`): each channel gets a
    `_NativeWindowTrack` built from the shared events.jsonl (via
    `_native_track_for_channel`), the shared click set projects into every
    card via `to_src` (clamped-not-dropped -- the times stay in the shared
    trailing-cluster set the three consumers pin), and per-card paths from
    `camera.build_card_paths` warp each crop before compositing. Window-
    focus (`window_focus=True`) rides the same click ownership through
    `_build_focus_emphasis` + `_FocusLayout`, exactly like the display-crop
    path. Per-card cursor-fx (`cursor_fx=True`) draws one `effects.CursorFX`
    per card in that card's buffer space.

    v1 scope cuts (stated loudly, per docs/architecture.md):
    - **click_fx / spotlight / motion_blur**: OFF -- per-card versions
      would need their own new work. A note prints for anything a caller
      passed truthy.
    - **cursor_erase**: OFF v1 -- the eraser plans off a single raw file;
      N-file plans are the next follow-up.
    - **--speedup**: not applied. With N streams the "which timeline owns
      output" question is deferred.
    - **facecam**: not composited (session-level t0 anchor is channel 0's,
      so a future add is a shape no-op).
    - **manual add_zoom / set_crop**: inert -- no per-card addressing yet.
    - **make_gif**: GIF post-pass keys off a finalized mp4; wire it after
      the composite is stable.

    Frame alignment: per-channel `t0_monotonic - session t0` is quantized to
    a frame index; a max drift over ~2 frames prints a warning. Lockstep
    reads keep the loop simple -- for the ~10ms deltas measured on the P3.1
    demo (0-1 frames at 60fps), the composite looks fine even without a
    per-channel warmup skip.
    """
    ignored = [n for n, on in (("motion_blur", motion_blur),
                                ("click_fx", click_fx),
                                ("spotlight", spotlight),
                                ("cursor_erase", cursor_erase),
                                ("facecam", facecam),
                                ("speedup", speedup),
                                ("make_gif", make_gif)) if on]
    if ignored:
        print("  note: {} ignored on multi-window native render (P3.3 covers "
              "per-card zoom/focus/cursor-fx; the rest land later)."
              .format(", ".join(ignored)))

    meta_path = os.path.join(session_dir, "meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    channels = meta.get("capture_channels") or []
    if len(channels) < 2:
        raise RuntimeError(
            "not a multi-window native session (need >=2 capture_channels)")
    channels, channel_layouts = _apply_hidden_channels(
        channels, hidden_channels, channel_layouts)
    if out_path is None:
        out_path = os.path.join(session_dir, "output.mp4")

    src_fps = float(meta.get("fps") or 60)
    session_t0 = float(meta.get("t0_monotonic") or 0.0)

    caps = []
    counts = []
    frame_offsets = []
    rects = []
    bed_path = None
    try:
        for ch in channels:
            path = os.path.join(session_dir, ch["file"])
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("cannot open channel file: " + path)
            caps.append(cap)
            counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            ch_t0 = float(ch.get("t0_monotonic") or session_t0)
            offset = int(round((ch_t0 - session_t0) * src_fps))
            frame_offsets.append(offset)
            # MultiFramePainter's `desktop` layout keys off (x, y) origins to
            # reproduce the on-screen arrangement; `grid`/`feature`/`row`/
            # `column` use aspects only. Passing rects in POINTS (from
            # meta.rect) preserves both without any conversion -- the painter
            # takes ratios and origins in a unit-consistent space.
            # Reconciled against the decoded buffer -- see `_channel_rect`.
            rects.append(_channel_rect(
                ch, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))))
        # Alignment window: each channel spans [frame_offsets[i], frame_offsets[i]+counts[i])
        # in session output frames. Common overlap = [max(offsets), min(offsets+counts)).
        # Skip leading frames per channel so we sit at the shared origin; the
        # loop then reads in lockstep for the shared duration.
        origin = max(0, max(frame_offsets)) if frame_offsets else 0
        max_drift = max(frame_offsets) - min(frame_offsets)
        if max_drift > _MULTI_NATIVE_DRIFT_WARN_FRAMES:
            print("  note: channel t0_monotonic drift is {} frames "
                  "(>{}). Cards may be offset by that many frames of motion; "
                  "the composite geometry is unaffected."
                  .format(max_drift, _MULTI_NATIVE_DRIFT_WARN_FRAMES))
        # Skip each channel to its shared-origin frame. cv2.CAP_PROP_POS_FRAMES
        # is a keyframe-approximate seek on H.264, but for tiny (0-2 frame)
        # skips it lands exactly in practice; grab() rather than read() so we
        # don't decode-and-throw-away.
        for i, cap in enumerate(caps):
            skip = origin - frame_offsets[i]
            for _ in range(max(0, skip)):
                if not cap.grab():
                    break
        # Common output duration -- clip at whichever channel runs out first.
        remaining = [counts[i] - max(0, origin - frame_offsets[i])
                     for i in range(len(caps))]
        n_out = min(remaining) if remaining else 0
        if n_out <= 0:
            raise RuntimeError("no overlapping frames across the {} channels "
                               "-- one of them may be empty".format(len(caps)))

        # Size the canvas from the real buffers, not a hardcoded 1080p -- see
        # `_multi_native_canvas`. The preview calls the same helper, so the
        # editor and the export cannot land on different resolutions.
        dims = [(int(c.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))) for c in caps]
        # One capture-indicator eraser per channel -- every card carries its
        # own badge, at its own app-specific offset. None (and never called)
        # when `badge_erase` is off.
        badges = _badge_erasers(session_dir, channels, dims,
                                enabled=badge_erase)
        out_w, out_h = _multi_native_canvas(rects, dims, style, aspect,
                                            window_layout,
                                            max_height=max_height)
        # Manual card placement, stamped AFTER the canvas is sized so a resize
        # never changes the output resolution -- the preview stamps at the same
        # point (multi_native_layout), so editor and export stay identical.
        _stamp_channel_layouts(rects, channel_layouts)
        painter = framing.make_multi_painter(
            out_w, out_h, rects, background=background,
            layout=window_layout)
        multi_cells = painter.cells

        # -- P3.3 per-card camera / focus / cursor-fx --------------------
        # OUTPUT-relative frame grid for camera + cursor: output frame k is
        # `k/src_fps` of the composite, which is why every edits-authored span
        # (suppressed/focus ranges, `plan_duration`) drops straight in -- the
        # editor authors them against this same 0-based timeline.
        frame_times = np.arange(n_out, dtype=float) / float(src_fps)
        events_path = os.path.join(
            session_dir, meta.get("events", "events.jsonl"))
        ev = geometry.load_events(events_path)

        # Events are parent-monotonic; the composite starts at session frame
        # `origin`, so they rebase through the composite's t0 and NOT through
        # session t0 -- see `_multi_native_composite_t0`. Anchoring on
        # session t0 fired every per-card zoom, focus move and synthetic
        # cursor `origin` frames after the picture (and after the mic, which
        # `audio_skip_s` already seeks to the origin).
        composite_t0 = _multi_native_composite_t0(session_t0, origin, src_fps)

        def to_media(arr):
            arr = np.asarray(arr, dtype=float)
            return arr - composite_t0 if arr.size else arr

        # geometry.load_events returns numpy arrays; `or []` would
        # raise "truth value of an array is ambiguous", so guard on None
        # explicitly. Empty arrays are the natural pass-through here --
        # every downstream path handles zero-length input.
        def _arr(name):
            v = ev.get(name)
            return np.asarray(v, dtype=float) if v is not None else np.array([])
        clicks_t = to_media(_arr("clicks_t"))
        moves_t = to_media(_arr("moves_t"))
        # Sound-bed-only tracks: the mouse-up release and the key ticks.
        # No `typing_zoom` gate exists on this path; if one ever lands, take
        # the UNGATED key array here for the same reason the single-file
        # body does -- "don't zoom on typing" must not mean "go silent while
        # I type".
        ups_t = to_media(_arr("ups_t"))
        sfx_keys_t = to_media(_arr("keys_t"))
        moves_x = _arr("moves_x")
        moves_y = _arr("moves_y")
        clicks_x = _arr("clicks_x")
        clicks_y = _arr("clicks_y")

        # Per-channel geometry tracks. Same math as the single-window native
        # `_native_track`, one instance per manifest channel. clamp-not-drop
        # in `_NativeWindowTrack.to_src` preserves the shared click set's
        # time membership across every card.
        native_tracks = [
            _native_track_for_channel(ch, ev, frame_times, to_media, src_fps)
            for ch in channels]

        # Per-card auto-zoom cameras. None per card means "this card never
        # zooms" and the still-card `paint()` resize runs untouched -- the
        # same optimization the display-crop multi path relies on.
        cam_params = params or {}
        card_paths = _build_native_card_cameras(
            channels, native_tracks, clicks_t, clicks_x, clicks_y,
            frame_times, max_zoom=max_zoom, params=cam_params,
            suppressed_ranges=suppressed_ranges,
            plan_duration=float(n_out) / float(src_fps),
            src_fps=src_fps, enabled=window_zoom)

        # Window-focus: which card is the emphasized subject, moment by moment.
        # The emphasis planner runs on PER-CARD click times, computed by
        # projecting the shared click set through each channel's native
        # track (clamped-not-dropped, so time membership stays shared).
        focus_layout = None
        if window_focus and native_tracks:
            per_card_times = _native_card_click_times(
                native_tracks, clicks_t, clicks_x, clicks_y)
            windows_for_focus = [
                {"w": (tr.raw_w if tr is not None else 1),
                 "h": (tr.raw_h if tr is not None else 1)}
                for tr in native_tracks]
            emphasis = _build_native_focus_emphasis(
                windows_for_focus, per_card_times, frame_times,
                max_zoom, cam_params, suppressed_ranges,
                float(n_out) / float(src_fps), src_fps,
                manual=focus_ranges)
            if emphasis is not None:
                focus_layout = _FocusLayout(
                    painter, emphasis,
                    lean=camera.build_params(max_zoom, cam_params).focus_lean)

        # Per-card cursor-fx. One `CursorFX` per channel, seeded with the
        # cursor's coordinates PROJECTED into that channel's buffer via
        # `to_src`. A card that never sees the cursor still gets a CursorFX,
        # but its `alpha` track collapses to zero (cursor always out of view
        # -> nothing drawn) -- cheaper than a per-frame containment check.
        card_cursors = None
        card_cursor_pos = None
        if cursor_fx and native_tracks and _cursor_fx_draws(
                cursor_fx, meta, cursor_erase=False):
            card_cursors, card_cursor_pos = _build_native_card_cursors(
                native_tracks, channels, frame_times,
                moves_t, moves_x, moves_y, clicks_t, cursor_params)

        # -- encoder pipe --
        # Audio: the mic rides channel 0 (record.py `_multi_native_worker_cfg`),
        # rebased by the worker onto channel 0's OWN video timeline. The
        # composite starts `origin - frame_offsets[0]` frames into channel 0
        # (its skip to the shared origin), so the audio is seeked by the same
        # amount to stay in sync. A mic-less take (no audio track in raw_0.mov)
        # produces the byte-identical video-only command.
        ch0_path = os.path.join(session_dir, channels[0]["file"])
        audio_skip_s = max(0, origin - frame_offsets[0]) / src_fps

        # Event sounds, on the COMPOSITE clock -- which is what `to_media`
        # already returns, because `_multi_native_composite_t0` folds the
        # `origin` skip into the anchor itself. So clip_start is 0.0, exactly
        # as on the scene path below ("`media` already returns take-wide
        # OUTPUT seconds"), and for the same reason.
        #
        # It carried an `origin / src_fps` term while `to_media` was still
        # session-anchored, which was right then and wrong the moment the
        # anchor landed: the skip came off twice and every sound fired
        # `origin/fps` EARLY (measured on the fixture below, origin = 3
        # frames: a click the picture shows at 0.400s sounded at 0.301s).
        # `audio_skip_s` above is NOT the same quantity -- it is a seek inside
        # channel 0's own file, hence its `frame_offsets[0]` term -- so it
        # stays.
        #
        # Duration comes from `n_out`, the planned frame count, NOT from the
        # events: a bed shorter than the video makes `-shortest` truncate the
        # video. `-ss` never touches the bed, so this is its final placement.
        bed_span = dict(clip_start=0.0,
                        clip_duration=n_out / src_fps)
        bed_clicks, dropped_c = _click_times_in_trim(clicks_t, **bed_span)
        bed_ups, _ = _click_times_in_trim(ups_t, **bed_span)
        bed_keys, dropped_k = _click_times_in_trim(sfx_keys_t, **bed_span)
        if dropped_c or dropped_k:
            print("  warning: event sounds capped ({} clicks, {} keys "
                  "skipped)".format(dropped_c, dropped_k))
        bed_path = _write_sfx_bed(n_out / src_fps, click_sound, key_sound,
                                  sfx_volume, bed_clicks, bed_ups, bed_keys)

        ch0_has_audio = _probe_has_audio(ch0_path)
        enc = _multi_native_enc_cmd(
            out_w, out_h, src_fps, out_path,
            audio_path=ch0_path if ch0_has_audio else None,
            audio_skip_s=audio_skip_s, sfx_path=bed_path,
            audio_layout=(_probe_audio_layout(ch0_path)
                          if ch0_has_audio else None))
        proc = subprocess.Popen(enc, stdin=subprocess.PIPE)

        last_frames = [None] * len(caps)
        wrote = 0
        try:
            for k in range(n_out):
                crops = []
                for i, cap in enumerate(caps):
                    ok, fr = cap.read()
                    if not ok:
                        # Channel ran out early despite the pre-flight math --
                        # hold on the previous frame rather than tear the
                        # composite. Happens only on a corrupted / truncated
                        # channel (Phase C's fragments keep tail-loss to <=1
                        # fragment, so 60 frames' worth at most).
                        if last_frames[i] is None:
                            raise RuntimeError(
                                "channel {} ({}): read failed at output "
                                "frame {} with no prior frame to hold"
                                .format(i, channels[i].get("file"), k))
                        fr = last_frames[i]
                    else:
                        # Erase BEFORE the frame becomes `last_frames[i]`: a
                        # held frame has already been through this, and a
                        # second pass would see its own flat fill where the
                        # badge used to be and go looking for a badge that is
                        # gone.
                        if badges and badges[i] is not None:
                            badges[i].apply(fr)
                        last_frames[i] = fr
                    crops.append(fr)

                # P3.3: focus moves cells around; the per-card camera has to
                # warp INTO the current cell so the composite adds up.
                cells, order = _focus_cells_at(focus_layout, k, multi_cells)
                if card_paths and any(p is not None for p in card_paths):
                    crops = _apply_native_card_cameras(
                        crops, card_paths, native_tracks, cells, k)
                out = _paint_multi(painter, crops, cells, order, focus_layout)
                if card_cursors:
                    _draw_native_multi_cursor(
                        out, card_cursors, card_cursor_pos, cells, k)
                if focus_layout is not None:
                    out = _apply_focus_camera(
                        out, focus_layout.camera_at(k), out_w, out_h)

                proc.stdin.write(out.tobytes())
                wrote += 1
                if wrote % 60 == 0:
                    print("  rendered {} frames...".format(wrote),
                          end="\r", flush=True)
            print("  rendered {} frames.  ".format(wrote))
            _badge_report(badges, label="channel")
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                # Adding the SFX bed made a SILENT failure reachable here:
                # ffmpeg can consume every frame and still fail at the mux
                # (a bad filter graph, an unreadable input), after which this
                # function used to print "wrote ..." and hand back a path to
                # a file that is broken or absent. The single-file body has
                # always raised on a non-zero exit; these two composites had
                # simply never been given a way to fail late.
                raise RuntimeError(
                    "ffmpeg encode failed (exit {}) after {} frames".format(
                        rc, wrote))
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
            raise
    finally:
        for cap in caps:
            cap.release()
        # After the encode. On the success path ffmpeg has been wait()ed on;
        # on the failure path it was killed. Either way it is done reading.
        _remove_sfx_bed(bed_path)
    print("wrote {} ({} frames from {} channels)".format(
        out_path, wrote, len(channels)))
    return out_path


def _clock_contained(clock, arr):
    """Boolean mask: True where a `segments.SegmentClock` segment/scene
    actually CONTAINS the timestamp.

    Both clock-driven paths need this and for the same reason. `clock.media`
    is clamp-not-drop -- it must return one value per input because it is
    positionally zipped with x/y arrays elsewhere, so an event recorded in a
    deleted pause gap (or before the first segment, or past the last) is
    parked on the nearest seam instead of disappearing. For the PICTURE that
    is the right call. For SOUND it is not: every click made during one
    pause would fire as a single N-times-louder click at the seam, on a take
    whose whole point is that the pause was deleted.

    Containment is decided by `clock.owner`, which runs the identical
    assignment pass as `media()`, so the mask and the mapping can never
    disagree about a seam.
    """
    a = np.asarray(arr, dtype=float).ravel()
    if a.size == 0:
        return np.zeros(0, dtype=bool)
    own = clock.owner(a)
    lo = np.asarray(clock.t0s, dtype=float)[own]
    hi = lo + np.asarray(clock.durs, dtype=float)[own]
    return np.isfinite(a) & (a >= lo) & (a < hi)


def _scene_bed_times(clock, arr, max_events=_MAX_CLICK_SFX_EVENTS):
    """Event times -> TAKE-WIDE output seconds for a scene take's sound bed.

    Not just `clock.media(arr)`. That mapper is CLAMP-NOT-DROP by design --
    it is positionally zipped with x/y arrays elsewhere, so it must return
    one value per input and parks a gap or tail event on the nearest seam
    (segments.SegmentClock.media). A sound bed has no such constraint, and
    the clamp would be audible: every click made during a PAUSE -- exactly
    when the user is fiddling with windows, so there can be many -- would
    fire as one burst at the seam, on a take whose whole point is that the
    paused wall-clock was deleted.

    So: keep only events a scene actually CONTAINS, then map the survivors
    through the same mapper the picture uses. Containment is decided by
    `clock.owner`, which runs the identical assignment pass as `media()`, so
    the mask and the mapping can never disagree about a seam.
    """
    a = np.asarray(arr if arr is not None else [], dtype=float).ravel()
    if a.size == 0:
        return [], 0
    a = a[_clock_contained(clock, a)]
    if a.size == 0:
        return [], 0
    # clip_start 0.0: `media` already returns take-wide OUTPUT seconds.
    return _click_times_in_trim(clock.media(a), clip_start=0.0,
                                clip_duration=clock.total,
                                max_events=max_events)


def _scene_local_ranges(ranges, start, dur):
    """Rebase output-media [a, b] spans into ONE scene's local time,
    dropping what falls outside. Used for suppressed/focus ranges, whose
    edits are stored on the take's gaps-deleted timeline."""
    out = []
    for pair in ranges or []:
        try:
            a, b = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, IndexError):
            continue
        lo = max(0.0, a - start)
        hi = min(dur, b - start)
        if hi > lo:
            out.append((lo, hi))
    return out


class _ScenePlan:
    """Everything needed to composite ONE scene's frames, built once per
    scene and reused for every frame in it. Shared by `_render_scenes` (the
    export loop) and `scene_preview_frame` (the editor's paused scrub) so the
    two CANNOT drift -- a preview frame is bit-identical to the exported one
    at the same output index. Caps are deliberately NOT held here: the plan is
    source-independent, and each caller seeks its own way (export grabs
    sequentially, preview seeks by frame index)."""

    __slots__ = ("painter", "multi_cells", "native_tracks", "card_paths",
                 "focus_layout", "card_cursors", "card_cursor_pos",
                 "origin", "n_out", "offsets", "channels")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _scene_channel_count(cap, ch):
    """Frame count for one scene channel.

    A gapfree window-JOIN scene (docs/architecture.md) keeps each survivor window on
    ONE continuous file across the seam, so its scene entry references a
    SUB-RANGE via an explicit `frame_count` (the slice length) -- without it
    the earlier scene would read the whole file and overrun into the next.
    Absent (pause/resume scenes, whose files ARE per-scene, and every
    non-gapfree take) -> the full file length, byte-identical to before.
    """
    return int(ch.get("frame_count") or cap.get(cv2.CAP_PROP_FRAME_COUNT))


def _scene_channel_start(ch):
    """Leading frames to skip so a continuous-file channel sits at its
    sub-range start (a gapfree join's later scene). 0 for a per-scene file."""
    return max(0, int(ch.get("frame_start") or 0))


# Seamless-join card ENTRANCE: how long the new card grows in and the survivors
# reflow at a join seam (docs/architecture.md milestone 2). Long enough to READ as a
# deliberate addition -- at 0.45s the grow-in was so brief it read as a hard cut,
# especially in the `desktop` layout where the survivors barely reflow (a real
# take surfaced this). Still a transition, not a set-piece: every frame is paid
# on the heavy scene compositor (roadmap #1), so it must not linger. The whole
# animation is live footage at animated card GEOMETRY -- no card is ever frozen
# or cut. The JS live player + paused scrub follow automatically (they read the
# frame count off `_scene_entrance`'s `n` in the server payload), so this is the
# single source of truth for every surface.
_JOIN_ENTRANCE_SEC = 0.8


def _smoothstep(t):
    """Classic 3t^2 - 2t^3 ease, clamped to [0, 1]. Zero velocity at both
    ends, so the card entrance starts and settles without a visible kick."""
    t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else float(t))
    return t * t * (3.0 - 2.0 * t)


def _is_join_seam(prev_channels, channels):
    """True when `channels` (scene s) is `prev_channels` (scene s-1) plus
    EXACTLY ONE new card -- the gapfree window-JOIN signature: every survivor
    keeps its SAME continuous file, in prev's relative order, and one new
    file is added AT ANY position. M2's grow appends the joiner; M3.3's
    auto-REJOIN inserts the returnee at its departed card's SLOT (decision
    11), so this is an ordered-subsequence check, not a prefix check -- the
    prefix version hard-cut every rejoin entrance (caught on the first real
    take). A pause/resume re-pick (different per-scene files, arbitrary
    add/remove) is still NOT a join seam, so it keeps its hard cut,
    byte-identical."""
    prev = [c.get("file") for c in prev_channels]
    cur = [c.get("file") for c in channels]
    if len(cur) != len(prev) + 1:
        return False
    new = [f for f in cur if f not in prev]
    if len(new) != 1:
        return False
    return [f for f in cur if f != new[0]] == prev


def _join_entrance_from_cells(prev_rects, to_cells, out_w, out_h, layout,
                              prev_files=None, cur_files=None):
    """The cells each scene-s card morphs FROM at a join seam: every survivor
    from the PREVIOUS scene's (fewer-card, so larger) layout on the shared
    canvas, and the one new card from a ~point at the centre of its final cell
    (so it grows in). `to_cells` are scene s's final cells (dicts or tuples).

    Survivors are matched by FILE when the file lists are given -- a rejoined
    card sits MID-list (slot inheritance), so position i of scene s is not
    position i of scene s-1. Without file lists the mapping stays positional
    (the appended-joiner shape, byte-identical to before)."""
    to = [framing._cell_xywh(c) for c in to_cells]
    prev = framing.placements_for(out_w, out_h, prev_rects, layout=layout)
    out = []
    for i, t in enumerate(to):
        j = None
        if prev_files is not None and cur_files is not None:
            if i < len(cur_files) and cur_files[i] in prev_files:
                j = prev_files.index(cur_files[i])
        elif i < len(prev):
            j = i
        if j is not None and j < len(prev):
            out.append(tuple(int(v) for v in prev[j][:4]))
        else:
            cx = int(t[0] + t[2] / 2.0)
            cy = int(t[1] + t[3] / 2.0)
            out.append((cx, cy, 2, 2))       # grows out from its centre
    return out, to


def _build_scene_plan(scene, s_i, aligns_entry, out_w, out_h, scene_rects_s,
                      clock, ev, src_fps, *, background, window_layout,
                      max_zoom, cam_params, window_zoom, window_focus,
                      suppressed_ranges, focus_ranges, cursor_fx,
                      cursor_params, meta, card_layouts=None):
    """Build the per-scene compositor state (painter + per-card cameras,
    focus emphasis, cursors) for scene `s_i`. Extracted verbatim from the
    body of `_render_scenes`' per-scene loop so the export path and the
    preview path plan a scene identically.

    `card_layouts` is this scene's manual per-card placement (scene_layouts[
    str(s_i)]); stamped onto the rects here -- the ONE site export, paused-still
    and live-preview all funnel through, so all three agree. `out_w/out_h` are
    already fixed (frozen from the un-overridden rects), so a resize never
    changes the shared canvas."""
    channels = scene.get("channels") or []
    origin, n_out, offsets = aligns_entry
    _stamp_channel_layouts(scene_rects_s, card_layouts)
    painter = framing.make_multi_painter(
        out_w, out_h, scene_rects_s, background=background,
        layout=window_layout)
    multi_cells = painter.cells
    frame_times = np.arange(n_out, dtype=float) / src_fps
    plan_dur = float(n_out) / src_fps

    # The scene-scoped events view + scene-local mapping. Ownership was
    # decided by the clock (seam-clamped events belong to the EARLIER scene);
    # the affine below never clamps because the view only contains this
    # scene's events -- except `scene_local`, which clips a tail event that
    # out-ran the decoded timeline to the scene's last frame.
    scene_ev = segments.scene_events_view(clock, s_i, ev)
    t0_s = clock.t0s[s_i]

    def to_media(arr, _t0=t0_s):
        arr = np.asarray(arr, dtype=float)
        return arr - _t0 if arr.size else arr

    def _arr(name):
        v = scene_ev.get(name)
        return (np.asarray(v, dtype=float) if v is not None
                else np.array([]))
    clicks_t = clock.scene_local(s_i, _arr("clicks_t"))
    clicks_x, clicks_y = _arr("clicks_x"), _arr("clicks_y")
    moves_t = clock.scene_local(s_i, _arr("moves_t"))
    moves_x, moves_y = _arr("moves_x"), _arr("moves_y")
    sup_s = _scene_local_ranges(
        suppressed_ranges, clock.starts[s_i], clock.durs[s_i])
    # None-vs-[] is load-bearing for the emphasis planner: a LIST -- even an
    # empty one -- is an authoritative manual plan, None asks for the
    # auto-plan. A scene whose slice of a manual take-wide plan is empty must
    # stay empty, not silently re-auto-plan.
    focus_s = (_scene_local_ranges(
        focus_ranges, clock.starts[s_i], clock.durs[s_i])
        if focus_ranges is not None else None)

    native_tracks = [
        _native_track_for_channel(ch, scene_ev, frame_times,
                                  to_media, src_fps)
        for ch in channels]
    card_paths = _build_native_card_cameras(
        channels, native_tracks, clicks_t, clicks_x, clicks_y,
        frame_times, max_zoom=max_zoom, params=cam_params,
        suppressed_ranges=sup_s, plan_duration=plan_dur,
        src_fps=src_fps, enabled=window_zoom)
    focus_layout = None
    if window_focus and native_tracks:
        per_card_times = _native_card_click_times(
            native_tracks, clicks_t, clicks_x, clicks_y)
        windows_for_focus = [
            {"w": (tr.raw_w if tr is not None else 1),
             "h": (tr.raw_h if tr is not None else 1)}
            for tr in native_tracks]
        emphasis = _build_native_focus_emphasis(
            windows_for_focus, per_card_times, frame_times,
            max_zoom, cam_params, sup_s, plan_dur, src_fps,
            manual=focus_s)
        if emphasis is not None:
            focus_layout = _FocusLayout(
                painter, emphasis,
                lean=camera.build_params(
                    max_zoom, cam_params).focus_lean)
    card_cursors = card_cursor_pos = None
    if cursor_fx and native_tracks and _cursor_fx_draws(
            cursor_fx, meta, cursor_erase=False):
        card_cursors, card_cursor_pos = _build_native_card_cursors(
            native_tracks, channels, frame_times,
            moves_t, moves_x, moves_y, clicks_t, cursor_params)

    return _ScenePlan(
        painter=painter, multi_cells=multi_cells, native_tracks=native_tracks,
        card_paths=card_paths, focus_layout=focus_layout,
        card_cursors=card_cursors, card_cursor_pos=card_cursor_pos,
        origin=origin, n_out=n_out, offsets=offsets, channels=channels)


def _scene_entrance(scenes, s_i, scene_rects, plan, src_fps, out_w, out_h,
                    window_layout, join_entrance):
    """The seamless-join card ENTRANCE state for scene `s_i`, or None.

    None on the first scene, a non-join seam (pause/resume), or when
    `join_entrance` is off -> a hard cut. Otherwise `(n, from_cells, to_cells)`:
    the new card grows in and the survivors reflow over the first `n` frames.
    Shared by the export loop and the editor's paused-scrub so the transition
    that Export produces is exactly what the editor previews."""
    if not (join_entrance and s_i > 0
            and _is_join_seam(scenes[s_i - 1].get("channels") or [],
                              scenes[s_i].get("channels") or [])):
        return None
    n = min(plan.n_out, max(2, int(round(_JOIN_ENTRANCE_SEC * src_fps))))
    from_cells, to_cells = _join_entrance_from_cells(
        scene_rects[s_i - 1], plan.multi_cells, out_w, out_h, window_layout,
        prev_files=[c.get("file")
                    for c in scenes[s_i - 1].get("channels") or []],
        cur_files=[c.get("file")
                   for c in scenes[s_i].get("channels") or []])
    return (n, from_cells, to_cells)


def _composite_scene_frame(plan, crops, k, out_w, out_h, entrance=None):
    """Paint ONE output frame (in-scene index `k`) from this scene's decoded
    channel `crops`, applying per-card cameras, the multi-cell paint, cursor
    overlays and the focus camera -- exactly the tail of `_render_scenes`'
    inner loop, factored so preview and export share it.

    `entrance` (from `_scene_entrance`) morphs the layout for the first frames
    of a join scene: the new card grows in and the survivors reflow, reusing
    the window-focus moving-cells compositor (`paint_at`). Per-card cameras /
    cursor / focus are suppressed during the ~0.8s morph (v1)."""
    if entrance is not None:
        n, from_cells, to_cells = entrance
        if k < n:
            w = _smoothstep(k / float(n - 1) if n > 1 else 1.0)
            cells_k, _order = framing.blend_placements(
                from_cells, [to_cells], [w])
            return plan.painter.paint_at(crops, cells_k)
    cells, order = _focus_cells_at(plan.focus_layout, k, plan.multi_cells)
    if plan.card_paths and any(p is not None for p in plan.card_paths):
        crops = _apply_native_card_cameras(
            crops, plan.card_paths, plan.native_tracks, cells, k)
    out = _paint_multi(plan.painter, crops, cells, order, plan.focus_layout)
    if plan.card_cursors:
        _draw_native_multi_cursor(
            out, plan.card_cursors, plan.card_cursor_pos, cells, k)
    if plan.focus_layout is not None:
        out = _apply_focus_camera(
            out, plan.focus_layout.camera_at(k), out_w, out_h)
    return out


def scene_preview_frame(session_dir, t_sec, max_zoom=2.0, style="clean",
                        background=None, aspect=None, window_layout="grid",
                        window_zoom=False, window_focus=False, cursor_fx=False,
                        cursor_params=None, params=None,
                        suppressed_ranges=None, focus_ranges=None,
                        max_height=None, scene_layouts=None,
                        hidden_channels=None, join_entrance=True,
                        badge_erase=True,
                        **_ignored):
    """Render ONE composited preview frame of a SCENE take at output time
    `t_sec` (seconds on the joined, gaps-deleted timeline the editor scrubs).

    The editor's paused-scrub preview: reuses the exact per-scene planner and
    single-frame compositor the export loop runs, so the JPEG under the
    playhead is what Export produces at that frame. Effects the scene-take
    export ignores (motion-blur, click-fx, trim, ...) are swallowed by
    `**_ignored` -- the caller passes its whole kwargs dict; only the ones
    that survive on this path have any effect. Returns a BGR ndarray."""
    meta, _raw, events_path = _session_paths(session_dir)
    scenes = meta.get("capture_scenes") or []
    if len(scenes) < 2:
        raise RuntimeError("not a scene take (need >=2 capture_scenes)")
    scenes, scene_layouts = _apply_hidden_scenes(
        scenes, hidden_channels, scene_layouts)
    src_fps = float(meta.get("fps") or 60)
    cam_params = params or {}

    scene_caps, scene_counts, scene_dims, scene_rects = [], [], [], []
    try:
        for scene in scenes:
            caps, counts, dims, rects = [], [], [], []
            for ch in scene.get("channels") or []:
                path = os.path.join(session_dir, ch["file"])
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("cannot open channel file: " + path)
                caps.append(cap)
                counts.append(_scene_channel_count(cap, ch))
                dims.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                # Reconciled against the decoded buffer -- `_channel_rect`.
                rects.append(_channel_rect(ch, dims[-1]))
            scene_caps.append(caps)
            scene_counts.append(counts)
            scene_dims.append(dims)
            scene_rects.append(rects)

        clock, aligns, _warn = segments.SegmentClock.from_scene_meta(
            meta, scene_counts, src_fps)

        # The same shared-canvas rule as export: the largest scene's
        # source-native candidate wins, and every scene paints at that size.
        out_w, out_h = 0, 0
        for s_i in range(len(scenes)):
            w_s, h_s = _multi_native_canvas(
                scene_rects[s_i], scene_dims[s_i], style, aspect,
                window_layout, max_height=max_height)
            if w_s * h_s > out_w * out_h:
                out_w, out_h = w_s, h_s

        # Output time -> (scene, in-scene frame). Scenes concatenate on the
        # frame grid (n_out first, dur = n_out/fps), so this is exact, not a
        # float-drifting sum-of-durations.
        total = sum(a[1] for a in aligns)
        kg = int(round(max(0.0, float(t_sec)) * src_fps))
        kg = max(0, min(kg, total - 1))
        cum = 0
        s_i = 0
        for i, (_o, n_out_i, _off) in enumerate(aligns):
            if kg < cum + n_out_i:
                s_i, kg = i, kg - cum
                break
            cum += n_out_i
        else:
            s_i = len(aligns) - 1
            kg = aligns[s_i][1] - 1

        ev = geometry.load_events(events_path)
        plan = _build_scene_plan(
            scenes[s_i], s_i, aligns[s_i], out_w, out_h, scene_rects[s_i],
            clock, ev, src_fps, background=background,
            window_layout=window_layout, max_zoom=max_zoom,
            cam_params=cam_params, window_zoom=window_zoom,
            window_focus=window_focus, suppressed_ranges=suppressed_ranges,
            focus_ranges=focus_ranges, cursor_fx=cursor_fx,
            cursor_params=cursor_params, meta=meta,
            card_layouts=(scene_layouts or {}).get(str(s_i)))

        origin, _n_out, offsets = aligns[s_i]
        caps = scene_caps[s_i]
        scene_channels = scenes[s_i].get("channels") or []
        crops = []
        for i, cap in enumerate(caps):
            # A gapfree window-join's later scene is a SUB-RANGE of a continuous
            # file: seek past frame_start so the scrub shows THIS scene's slice,
            # not the head of the file. 0 for per-scene files -> unchanged.
            fs = (_scene_channel_start(scene_channels[i])
                  if i < len(scene_channels) else 0)
            abs_idx = fs + origin - offsets[i] + kg
            abs_idx = max(fs, min(abs_idx, fs + scene_counts[s_i][i] - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_idx)
            ok, fr = cap.read()
            if not ok:
                # Seek-decode can miss on a non-keyframe; fall back to frame 0
                # so the preview shows something rather than 500ing mid-scrub.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = cap.read()
                if not ok:
                    raise RuntimeError(
                        "scene {} channel {} ({}): cannot decode frame {}"
                        .format(s_i, i, plan.channels[i].get("file"), abs_idx))
            # preview == export: the scrub shows the same erased corner the
            # export writes. One-shot (the memoised box, no lock state) --
            # this path seeks, so there is no forward walk to verify along.
            ch_i = scene_channels[i] if i < len(scene_channels) else None
            _badge_erase_once(fr, os.path.join(session_dir,
                                               (ch_i or {}).get("file", "")),
                              ch_i, fr.shape[1], enabled=badge_erase)
            crops.append(fr)
        # Same seamless-join entrance the export produces, so scrubbing the
        # first ~0.8s of a join scene shows the card growing in, not a hard
        # cut (preview == export). The entrance blends FROM the previous
        # scene's placements, and by the time the export loop reaches this
        # seam it has already stamped that scene's manual layouts onto its
        # rects; only the scrubbed scene was planned (and stamped) above, so
        # stamp the neighbour here too -- without it a hand-placed card in
        # the previous scene would morph from its preset cell.
        if s_i > 0:
            _stamp_channel_layouts(scene_rects[s_i - 1],
                                   (scene_layouts or {}).get(str(s_i - 1)))
        entrance = _scene_entrance(
            scenes, s_i, scene_rects, plan, src_fps, out_w, out_h,
            window_layout, join_entrance)
        return _composite_scene_frame(plan, crops, kg, out_w, out_h,
                                      entrance=entrance)
    finally:
        for caps in scene_caps:
            for cap in caps:
                cap.release()


def _sample_scene_card_paths(paths, src_fps, stride):
    """Sample a scene plan's per-card camera arrays into the `card_paths` wire
    shape -- byte-identical to `multi_native_card_paths` (same keys, stride,
    rounding, per-card None, and the "None if every card is None" collapse) so
    the browser's `cardSrcRect` animates a scene card with no client change."""
    stride = max(1, int(stride))
    sel = slice(None, None, stride)
    out = []
    for path in paths or []:
        if path is None:
            out.append(None)
            continue
        out.append({
            "cx": [round(float(v), 2) for v in path[sel, 0]],
            "cy": [round(float(v), 2) for v in path[sel, 1]],
            "z": [round(float(v), 4) for v in path[sel, 2]],
        })
    if not any(p is not None for p in out):
        return None
    return {"stride": stride, "fps": float(src_fps), "cards": out}


def _sample_scene_focus_cells(layout, painter, n_out, src_fps, stride):
    """Sample a scene plan's `_FocusLayout` into the `focus_cells` wire shape
    -- byte-identical to `multi_native_focus_cells` (`{stride, fps, frames:[{c,
    o, z, l?}]}`, canvas px, same rounding) so `focusFrameAt` animates it
    unchanged."""
    stride = max(1, int(stride))
    frames = []
    for i in range(0, int(n_out), stride):
        cells, order = layout.cells_at(i)
        cam = layout.camera_at(i)
        lifts = [round(painter.lift_of(j, c["w"]), 3)
                 for j, c in enumerate(cells)]
        frame = {
            "c": [[c["x"], c["y"], c["w"], c["h"], c["radius"]] for c in cells],
            "o": list(order),
            "z": ([round(float(cam[0]), 1), round(float(cam[1]), 1),
                   round(float(cam[2]), 4)] if cam else None),
        }
        if any(v > 0.0 for v in lifts):
            frame["l"] = lifts
        frames.append(frame)
    return {"stride": stride, "fps": float(src_fps), "frames": frames}


def scene_camera_path(session_dir, style="clean", background=None, aspect=None,
                      window_layout="grid", window_zoom=False,
                      window_focus=False, max_zoom=2.0, params=None,
                      suppressed_ranges=None, focus_ranges=None,
                      max_height=None, stride=2, scene_layouts=None,
                      hidden_channels=None, badge_erase=True,
                      **_ignored):
    """The per-scene live-composite plan for a SCENE take's editor player.

    A scene take has NO single raw.mov and a DIFFERENT window set per scene, so
    the browser needs, per scene: the shared-canvas geometry, each channel's
    `<video>` source + cell + start offset, and (when enabled) the sampled
    per-card zoom / window-focus tracks. This is the scene-aware analog of the
    three `multi_native_*` live-preview emitters, but emitted for EVERY scene
    in one payload so the client can build its fleet-ring player.

    The single-source-of-truth rule the other emitters follow, held here too:
    each scene is planned by the SAME `_build_scene_plan` the export loop
    (`_render_scenes`) runs, on the SAME shared max-need canvas and the SAME
    `SegmentClock.from_scene_meta` clock -- so a live composite frame matches
    the exported frame (and the server still, `scene_preview_frame`, which is
    also `_build_scene_plan`-based) up to the Canvas2D-vs-cv2 compositor
    residue that multi-native already accepts.

    Returns `{fps, duration, total_frames, seams, scenes:[...]}` where
    `seams[s]` is the OUTPUT frame index at which scene s begins (prefix sums
    of per-scene `frame_count`, matching describe), and each scene entry is
    `{index, frame_start, frame_count, canvas:[W,H], cells, channels:[{index,
    media:"scene{s}channel{c}", buffer_w, buffer_h, start, app}],
    plate_jpeg_base64, bg_jpeg_base64|None, card_paths|None, focus_cells|None}`.
    Canvas/cells live UNDER each scene entry, never at top level, so the
    editor's `multiReady()` (single-scene) gate is never accidentally armed.
    """
    meta, _raw, events_path = _session_paths(session_dir)
    scenes = meta.get("capture_scenes") or []
    if len(scenes) < 2:
        raise RuntimeError("not a scene take (need >=2 capture_scenes)")
    scenes, scene_layouts = _apply_hidden_scenes(
        scenes, hidden_channels, scene_layouts)
    src_fps = float(meta.get("fps") or 60)
    cam_params = params or {}

    # Decode facts (counts + buffer dims + point rects) -- caps released
    # immediately, exactly like `multi_native_layout`; the plan builder is
    # source-independent and needs none of the caps held.
    scene_counts, scene_dims, scene_rects = [], [], []
    for scene in scenes:
        counts, dims, rects = [], [], []
        for ch in scene.get("channels") or []:
            path = os.path.join(session_dir, ch["file"])
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError("cannot open channel file: " + path)
            try:
                counts.append(_scene_channel_count(cap, ch))
                dims.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            finally:
                cap.release()
            # Reconciled against the decoded buffer -- `_channel_rect`.
            rects.append(_channel_rect(ch, dims[-1]))
        scene_counts.append(counts)
        scene_dims.append(dims)
        scene_rects.append(rects)

    clock, aligns, _warn = segments.SegmentClock.from_scene_meta(
        meta, scene_counts, src_fps)

    # ONE shared canvas for every scene -- the export rule verbatim.
    out_w, out_h = 0, 0
    for s_i in range(len(scenes)):
        w_s, h_s = _multi_native_canvas(
            scene_rects[s_i], scene_dims[s_i], style, aspect, window_layout,
            max_height=max_height)
        if w_s * h_s > out_w * out_h:
            out_w, out_h = w_s, h_s

    ev = geometry.load_events(events_path)
    stride = max(1, int(stride))
    out_scenes, seams = [], []
    frame_start = 0
    for s_i, scene in enumerate(scenes):
        plan = _build_scene_plan(
            scene, s_i, aligns[s_i], out_w, out_h, scene_rects[s_i], clock, ev,
            src_fps, background=background, window_layout=window_layout,
            max_zoom=max_zoom, cam_params=cam_params, window_zoom=window_zoom,
            window_focus=window_focus, suppressed_ranges=suppressed_ranges,
            focus_ranges=focus_ranges, cursor_fx=False, cursor_params=None,
            meta=meta, card_layouts=(scene_layouts or {}).get(str(s_i)))
        origin, n_out, offsets = aligns[s_i]
        cells = plan.multi_cells
        channels = scene.get("channels") or []
        out_channels = []
        for i, ch in enumerate(channels):
            bw, bh = scene_dims[s_i][i]
            if i < len(cells):
                cells[i]["src"] = [0, 0, bw, bh]
            out_channels.append({
                "index": i,
                "media": "scene{}channel{}".format(s_i, i),
                "buffer_w": bw,
                "buffer_h": bh,
                # Where in the CHANNEL'S OWN FILE this scene begins, in seconds:
                # the origin-skip PLUS `frame_start` for a gapfree-join survivor
                # (whose file is one continuous recording -- scene 1 lives at
                # frame_start, not 0). The browser fleet ring seeks the <video>
                # to `start + tLocal`, so without frame_start scene 1 would play
                # the head of the file (or stall at the seam). 0 for per-scene
                # files -> unchanged.
                "start": round(
                    (_scene_channel_start(ch) + max(0, origin - offsets[i]))
                    / src_fps, 4),
                "app": ch.get("app"),
                # The channel's source file -- the stable identity the editor's
                # "remove this window" writes into edits.hidden_channels (a
                # window recurs across scenes under the same file).
                "file": ch.get("file"),
                # Capture-indicator repair in BUFFER pixels -- see the same
                # field in `multi_native_layout`. Memoised per file, so a
                # survivor appearing in several scenes is probed once.
                "badge": _badge_patch(
                    os.path.join(session_dir, ch.get("file", "")), ch, bw,
                    enabled=badge_erase),
            })
        entry = {
            "index": s_i,
            "frame_start": frame_start,
            "frame_count": int(n_out),
            "canvas": [int(out_w), int(out_h)],
            "cells": cells,
            "channels": out_channels,
            "plate_jpeg_base64": base64.b64encode(
                encode_preview_jpeg(plan.painter.base_plate(), quality=90)
            ).decode("ascii"),
            "bg_jpeg_base64": None,
            "card_paths": _sample_scene_card_paths(
                plan.card_paths, src_fps, stride) if window_zoom else None,
            "focus_cells": None,
        }
        if window_focus and plan.focus_layout is not None:
            entry["focus_cells"] = _sample_scene_focus_cells(
                plan.focus_layout, plan.painter, n_out, src_fps, stride)
            # Moving cells make the baked plate shadows wrong; the browser
            # needs the bare backdrop to draw its own over (native branch rule).
            entry["bg_jpeg_base64"] = base64.b64encode(
                encode_preview_jpeg(plan.painter.background_plate(), quality=90)
            ).decode("ascii")
        # Seamless-join ENTRANCE for the live editor: the browser's drawMulti
        # morphs `from_cells` -> this scene's `cells` over the first `frames`
        # (the same _scene_entrance the export runs), so the paused scrub /
        # playback shows the card growing in, not a hard cut. The morph moves
        # the cells, so the browser also needs the bare backdrop to draw fresh
        # per-card shadows over (same reason as focus).
        ent = _scene_entrance(scenes, s_i, scene_rects, plan, src_fps,
                              out_w, out_h, window_layout, True)
        if ent is not None:
            n_ent, from_cells, _to = ent
            entry["entrance"] = {
                "frames": int(n_ent),
                "from_cells": [{"x": int(c[0]), "y": int(c[1]),
                                "w": int(c[2]), "h": int(c[3])}
                               for c in from_cells],
            }
            if entry.get("bg_jpeg_base64") is None:
                entry["bg_jpeg_base64"] = base64.b64encode(
                    encode_preview_jpeg(plan.painter.background_plate(),
                                        quality=90)).decode("ascii")
        out_scenes.append(entry)
        seams.append(frame_start)
        frame_start += int(n_out)

    return {
        "fps": float(src_fps),
        "duration": float(frame_start / src_fps) if src_fps else 0.0,
        "total_frames": int(frame_start),
        "seams": seams,
        "scenes": out_scenes,
    }


def _render_scenes(session_dir, out_path=None, style="clean",
                   background=None, aspect=None, window_layout="grid",
                   window_zoom=False, window_focus=False,
                   cursor_fx=False, cursor_params=None,
                   max_zoom=2.0, params=None,
                   suppressed_ranges=None, focus_ranges=None,
                   motion_blur=False, click_fx=False, spotlight=False,
                   cursor_erase=False, facecam=False, speedup=False,
                   make_gif=False, trim_start=0.0, trim_end=None,
                   fade=0.0, music=None, click_sound=None, key_sound=None,
                   sfx_volume=1.0, max_height=None,
                   scene_layouts=None, hidden_channels=None,
                   join_entrance=True, badge_erase=True,
                   **_ignored):
    """Render a SCENE take (docs/architecture.md): K sequential fleet
    scenes composited scene-by-scene into ONE shared encoder -- no
    intermediates, no concat (the review's single-encoder correction: with
    the canvas and fps fixed take-wide, scene boundaries are invisible to
    the encoder, so the uniform-SPS question, the concat argv, the per-seam
    count belt and the second full-size disk copy all vanish).

    Per scene, this is `_render_multi_native`'s composite: same painter,
    same per-card camera / focus / cursor-fx machinery, fed through the
    SCENE-SCOPED events view (`segments.scene_events_view`) so no other
    scene's events can pile onto this scene's boundaries as phantom
    clusters, and a scene-local affine `to_media` (the view guarantees
    in-scene membership, so the affine never needs to clamp; geometry keeps
    its nearest-prior anchor for the resampler).

    v1 scope cuts (stated loudly, a note prints for anything passed truthy):
    trim / fade / music / click-sound plus the inherited multi-native list
    (motion-blur, click-fx, spotlight, cursor-erase, facecam, speedup, GIF).
    """
    ignored = [n for n, on in (("motion_blur", motion_blur),
                               ("click_fx", click_fx),
                               ("spotlight", spotlight),
                               ("cursor_erase", cursor_erase),
                               ("facecam", facecam),
                               ("speedup", speedup),
                               ("make_gif", make_gif),
                               ("trim", bool(trim_start or trim_end)),
                               ("fade", bool(fade)),
                               ("music", bool(music))) if on]
    if ignored:
        print("  note: {} ignored on a scene-take render (v1 scope; see "
              "docs/architecture.md).".format(", ".join(ignored)))

    with open(os.path.join(session_dir, "meta.json")) as f:
        meta = json.load(f)
    scenes = meta.get("capture_scenes") or []
    if len(scenes) < 2:
        raise RuntimeError("not a scene take (need >=2 capture_scenes)")
    scenes, scene_layouts = _apply_hidden_scenes(
        scenes, hidden_channels, scene_layouts)
    if out_path is None:
        out_path = os.path.join(session_dir, "output.mp4")
    src_fps = float(meta.get("fps") or 60)
    events_path = os.path.join(
        session_dir, meta.get("events", "events.jsonl"))

    scene_caps, scene_counts, scene_dims, scene_rects = [], [], [], []
    proc = None
    bed_path = None
    try:
        for scene in scenes:
            caps, counts, dims, rects = [], [], [], []
            for ch in scene.get("channels") or []:
                path = os.path.join(session_dir, ch["file"])
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    raise RuntimeError("cannot open channel file: " + path)
                caps.append(cap)
                counts.append(_scene_channel_count(cap, ch))
                dims.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                # Reconciled against the decoded buffer -- `_channel_rect`.
                rects.append(_channel_rect(ch, dims[-1]))
            scene_caps.append(caps)
            scene_counts.append(counts)
            scene_dims.append(dims)
            scene_rects.append(rects)

        # The ONE clock builder every surface calls -- describe and beats use
        # the same function, so the three surfaces cannot disagree.
        clock, aligns, clock_warnings = segments.SegmentClock.from_scene_meta(
            meta, scene_counts, src_fps)
        for w in clock_warnings:
            print("  note: " + w)

        # ONE output canvas for the whole take: each scene's source-native
        # candidate shares the same base aspect (scaled by a scalar need), so
        # the largest one wins and every painter builds at that shared size.
        # A single-scene take would land exactly on today's multi-native
        # canvas -- pinned.
        out_w, out_h = 0, 0
        for s_i in range(len(scenes)):
            w_s, h_s = _multi_native_canvas(
                scene_rects[s_i], scene_dims[s_i], style, aspect,
                window_layout, max_height=max_height)
            if w_s * h_s > out_w * out_h:
                out_w, out_h = w_s, h_s

        ev = geometry.load_events(events_path)
        cam_params = params or {}

        # Audio across the seam. A mic scene take can ONLY be a gapfree
        # window-JOIN (pause+mic is refused at record time -- record.py
        # `pause_supported`), where channel 0 is ONE continuous file spanning
        # every scene, so its mic track is continuous across the seam with no
        # placement math: mux it once, seeked by scene 0's channel-0 lead
        # (frame_start + origin skip) to the shared origin. The guards keep
        # this narrow -- mic_index set, and channel 0 really is the same file
        # in every scene -- so a pause/resume scene take (mic-off, per-scene
        # files) stays video-only, byte-identical.
        audio_path, audio_skip_s = None, 0.0
        if meta.get("mic_index") is not None:
            ch0_files = [((sc.get("channels") or [{}])[0].get("file"))
                         for sc in scenes]
            if ch0_files[0] and all(f == ch0_files[0] for f in ch0_files):
                ch0_path = os.path.join(session_dir, ch0_files[0])
                if _probe_has_audio(ch0_path):
                    origin0, _n0, offs0 = aligns[0]
                    lead0 = (_scene_channel_start(scenes[0]["channels"][0])
                             + max(0, origin0 - offs0[0]))
                    audio_path, audio_skip_s = ch0_path, lead0 / src_fps

        # Frame budget, hoisted: it is both the belt below and the bed's
        # length. `aligns` (post-overlap-clip) is the authority -- sizing the
        # bed off anything shorter would make `-shortest` truncate the video.
        total_expected = sum(a[1] for a in aligns)

        # ONE bed on the TAKE-WIDE output clock. The scenes share a single
        # encoder, so a per-scene bed would stack every scene's sounds at
        # t=0; `_scene_bed_times` maps through the same clock the picture
        # uses, with the deleted pause wall-clock removed.
        bed_clicks, dropped_c = _scene_bed_times(clock, ev.get("clicks_t"))
        bed_ups, _ = _scene_bed_times(clock, ev.get("ups_t"))
        bed_keys, dropped_k = _scene_bed_times(clock, ev.get("keys_t"))
        if dropped_c or dropped_k:
            print("  warning: event sounds capped ({} clicks, {} keys "
                  "skipped)".format(dropped_c, dropped_k))
        bed_path = _write_sfx_bed(total_expected / src_fps, click_sound,
                                  key_sound, sfx_volume,
                                  bed_clicks, bed_ups, bed_keys)

        enc = _multi_native_enc_cmd(
            out_w, out_h, src_fps, out_path,
            audio_path=audio_path, audio_skip_s=audio_skip_s,
            sfx_path=bed_path,
            audio_layout=(_probe_audio_layout(audio_path)
                          if audio_path else None))
        proc = subprocess.Popen(enc, stdin=subprocess.PIPE)

        wrote = 0
        for s_i, scene in enumerate(scenes):
            caps = scene_caps[s_i]
            scene_channels = scene.get("channels") or []
            origin, n_out, offsets = aligns[s_i]
            if n_out <= 0:
                raise RuntimeError(
                    "scene {}: no overlapping frames across its {} "
                    "channels".format(s_i, len(caps)))
            for i, cap in enumerate(caps):
                # A gapfree window-join's later scene reuses the survivor's
                # CONTINUOUS file, so seek past the frames the earlier scene
                # already consumed (`frame_start`) BEFORE the per-scene origin
                # alignment. 0 for per-scene files -> byte-identical grab count.
                # (Naive grab-skip; a large join-lateness re-decodes [0,T) here
                # -- the shared-cap read-through optimization is a follow-up
                # once record.grow lands, docs/architecture.md.)
                lead = _scene_channel_start(scene_channels[i]) + max(
                    0, origin - offsets[i])
                for _ in range(lead):
                    if not cap.grab():
                        break
            plan = _build_scene_plan(
                scene, s_i, aligns[s_i], out_w, out_h, scene_rects[s_i],
                clock, ev, src_fps, background=background,
                window_layout=window_layout, max_zoom=max_zoom,
                cam_params=cam_params, window_zoom=window_zoom,
                window_focus=window_focus, suppressed_ranges=suppressed_ranges,
                focus_ranges=focus_ranges, cursor_fx=cursor_fx,
                cursor_params=cursor_params, meta=meta,
                card_layouts=(scene_layouts or {}).get(str(s_i)))
            channels = plan.channels
            # One capture-indicator eraser per channel of THIS scene. Fresh
            # per scene even where a gapfree join reuses the survivor's
            # continuous file: each is a forward walk from wherever its
            # scene starts, which is what the lock/verify state describes.
            badges = _badge_erasers(session_dir, channels,
                                    enabled=badge_erase)
            # Seamless-join card ENTRANCE (shared with the editor's paused
            # scrub via `_composite_scene_frame`, so preview == export): the
            # new card grows in + survivors reflow over the first frames of a
            # join scene; None on a hard-cut seam.
            entrance = _scene_entrance(
                scenes, s_i, scene_rects, plan, src_fps, out_w, out_h,
                window_layout, join_entrance)

            last_frames = [None] * len(caps)
            for k in range(n_out):
                crops = []
                for i, cap in enumerate(caps):
                    ok, fr = cap.read()
                    if not ok:
                        if last_frames[i] is None:
                            raise RuntimeError(
                                "scene {} channel {} ({}): read failed at "
                                "frame {} with no prior frame to hold"
                                .format(s_i, i, channels[i].get("file"), k))
                        fr = last_frames[i]
                    else:
                        # Before it becomes `last_frames[i]` -- see the same
                        # ordering in `_render_multi_native`.
                        if badges and badges[i] is not None:
                            badges[i].apply(fr)
                        last_frames[i] = fr
                    crops.append(fr)
                out = _composite_scene_frame(plan, crops, k, out_w, out_h,
                                             entrance=entrance)
                proc.stdin.write(out.tobytes())
                wrote += 1
                if wrote % 60 == 0:
                    print("  rendered {} frames...".format(wrote),
                          end="\r", flush=True)
            _badge_report(badges, label="scene {} channel".format(s_i))
            for cap in caps:
                cap.release()
            scene_caps[s_i] = []
        if wrote != total_expected:
            raise RuntimeError(
                "scene render wrote {} frames, expected {} -- a scene's "
                "decode ran short".format(wrote, total_expected))
        print("  rendered {} frames.  ".format(wrote))
        proc.stdin.close()
        rc = proc.wait()
        proc = None
        if rc != 0:
            # Same reason as the fleet path above: a late mux failure must
            # not be reported as a successful render.
            raise RuntimeError(
                "ffmpeg encode failed (exit {}) after {} frames".format(
                    rc, wrote))
    except Exception:
        if proc is not None:
            try:
                proc.kill()
            except OSError:
                pass
        raise
    finally:
        for caps in scene_caps:
            for cap in caps:
                cap.release()
        # After the encode. On the success path ffmpeg has been wait()ed on;
        # on the failure path it was killed. Either way it is done reading.
        _remove_sfx_bed(bed_path)
    print("wrote {} ({} frames from {} scenes)".format(
        out_path, wrote, len(scenes)))
    return out_path


def _build_native_focus_emphasis(windows, per_card_times, frame_times,
                                   max_zoom, params, suppressed_ranges,
                                   plan_duration, src_fps, manual=None):
    """`_build_focus_emphasis` reshaped for multi-native (P3.3).

    The display-crop version projects clicks via `grid_tracks`; here that's
    already done -- `per_card_times[i]` is the shared click set filtered by
    each channel's clamp semantics (all times survive; the position doesn't
    matter for the emphasis planner). The camera helper is the same one
    the display-crop path calls, so the two share the same on-screen story.
    """
    if not windows:
        return None
    cards = [{"click_times": list(ts)} for ts in per_card_times]
    return camera.build_focus_emphasis(
        frame_times, cards, max_zoom=max_zoom, params=params or {},
        suppressed_ranges=suppressed_ranges, plan_duration=plan_duration,
        manual=manual)


def _build_native_card_cursors(native_tracks, channels, frame_times,
                                moves_t, moves_x, moves_y, clicks_t,
                                cursor_params):
    """One `effects.CursorFX` per card, in that card's buffer space (P3.3).

    Projects the shared cursor track into each channel via `to_src` (the
    same clamp semantics the clicks travel through), so a cursor over
    card A also has a clamped position in card B -- but B's `alpha` will
    read zero at that moment because containment is what drives visibility
    (a cursor at a card's edge because it was clamped isn't shown). Per
    card also carries the raw cursor-in-window predicate for the draw seam,
    computed once here.
    """
    cursors = []
    positions = []
    for i, tr in enumerate(native_tracks):
        if tr is None:
            cursors.append(None)
            positions.append(None)
            continue
        # Project moves + clicks through this card's track. The projection
        # clamps, so the raw predicate (is the cursor inside the CARD's
        # rect right now?) has to be computed from the un-projected coords.
        if moves_t is not None and moves_t.size:
            mx, my = tr.to_src(moves_t, moves_x, moves_y)
        else:
            mx, my = np.array([]), np.array([])
        cx, _cy = None, None
        # Per-card cursor lives in the CARD's buffer space, so pass its raw
        # buffer dims as (W, H). `CursorFX` interpolates the projected move
        # track onto `frame_times` and computes alpha per frame; that alpha
        # would be nonzero even for a cursor clamped to an edge, so we mask
        # with a per-frame "cursor really inside the card's rect" predicate
        # further down.
        cursorfx = effects.CursorFX(
            tr.raw_w, tr.raw_h, frame_times, moves_t, mx, my,
            clicks_t=clicks_t, params=cursor_params)
        cursors.append(cursorfx)
        # In-rect predicate on the frame grid: interpolate the raw cursor
        # onto `frame_times`, compare against each frame's live rect.
        if moves_t is not None and moves_t.size:
            rx = np.interp(frame_times, moves_t, moves_x,
                           left=moves_x[0], right=moves_x[-1])
            ry = np.interp(frame_times, moves_t, moves_y,
                           left=moves_y[0], right=moves_y[-1])
        else:
            rx = np.full(len(frame_times), np.nan)
            ry = np.full(len(frame_times), np.nan)
        rects = tr._r          # (T,4) live rect on the frame grid
        # `rects` length may not match `frame_times` if the track was
        # built for a shorter grid -- clip to len(frame_times).
        m = min(len(frame_times), len(rects))
        inside = np.zeros(len(frame_times), dtype=bool)
        if m:
            ox = rects[:m, 0]; oy = rects[:m, 1]
            ww = rects[:m, 2]; wh = rects[:m, 3]
            inside[:m] = ((rx[:m] >= ox) & (rx[:m] < ox + ww)
                          & (ry[:m] >= oy) & (ry[:m] < oy + wh))
        positions.append(inside)
    return cursors, positions


def _draw_native_multi_cursor(out, card_cursors, card_inside, cells,
                               frame_index):
    """Draw the synthetic cursor onto each card that ACTUALLY contains the
    cursor at this frame (P3.3).

    Parallel to `_draw_multi_cursor` for the display-crop path -- the
    difference is the containment predicate: instead of asking each grid
    track `is this display-space cursor inside my card's rect?`, we
    precomputed a boolean per frame per card in `_build_native_card_cursors`
    (from the raw un-clamped cursor track). Cursor drawn only where in-rect.
    """
    for i, cfx in enumerate(card_cursors):
        if cfx is None:
            continue
        if not bool(card_inside[i][frame_index]):
            continue
        idx = int(min(max(int(frame_index), 0), len(cfx.sx) - 1))
        if float(cfx.alpha[idx]) <= 0.0:
            continue
        cell = cells[i]
        ix, iy, fw, fh = cell["x"], cell["y"], cell["w"], cell["h"]
        if fw <= 0 or fh <= 0:
            continue
        view = out[iy:iy + fh, ix:ix + fw]
        if view.size == 0:
            continue
        # Card-buffer to cell-pixel: full-card fit at zoom 1 is straight
        # cell dims / buffer dims. Under per-card window_zoom the warp
        # picked a sub-window; the fit is still cell/buffer for the fresh
        # cursor draw because we composite onto the (already-warped) card,
        # not onto the raw buffer.
        zx = float(fw) / float(cfx.W)
        zy = float(fh) / float(cfx.H)
        cfx.draw(view, idx, 0.0, 0.0, zx, z_eff_y=zy)


def _erase_scale(track, scale_x, scale_y):
    """Points->pixels scale for the cursor eraser's BOX SIZES.

    A window-native take must use the WINDOW's backing scale (`track.scale`,
    ~2.0 on Retina), NOT the top-level display-derived `scale_x/scale_y` --
    which for a native take is raw_buffer/display_points (~1.0), about half the
    truth, and would size the search boxes at half the pixels the baked-in
    cursor occupies, clipping the erase. Box CENTERS already come from the
    native `to_src`; only the sizing scale is corrected here.
    """
    if isinstance(track, _NativeWindowTrack):
        return track.scale, track.scale
    return scale_x, scale_y


def _capture_window_id(meta):
    """The picked window's id, or None. The poller records EVERY window, so
    the capture crop has to say which of them is its own."""
    cw = meta.get("capture_window") if isinstance(meta, dict) else None
    if not isinstance(cw, dict):
        return None
    try:
        return int(cw["id"])
    except (KeyError, TypeError, ValueError):
        return None


def _window_samples(ev, window_id=None):
    """(times, rects) for one window id, or None when there's nothing usable.

    `window_id=None` takes every sample. A track written before ids were
    recorded is all -1, and such a session only ever logged its capture
    target -- so an id filter that matches nothing there falls back to the
    whole track rather than silently losing it.
    """
    wt = ev.get("windows_t")
    wr = ev.get("windows_rect")
    if wt is None or wr is None or len(wt) == 0 or len(wr) != len(wt):
        return None
    t = np.asarray(wt, dtype=float)
    r = np.asarray(wr, dtype=float)
    if window_id is not None:
        wi = ev.get("windows_id")
        if wi is not None and len(wi) == len(t):
            wi = np.asarray(wi, dtype=int)
            sel = wi == int(window_id)
            if sel.any():
                t, r = t[sel], r[sel]
            elif not (wi == -1).all():
                return None      # ids ARE recorded; this window isn't in them
    return (t, r) if t.size >= 2 else None


def _resample_track(sample_t, rects_px, times, src_fps):
    """Put irregular samples on the frame grid, then smooth.

    That order matters: samples are coalesced (a still window logs at ~1 Hz),
    so smoothing over sample INDEX would weight a long still stretch the same
    as a fast drag.
    """
    sigma = _WINDOW_TRACK_SMOOTH_SEC * float(src_fps or 60.0)
    return np.column_stack([
        _smooth1d(np.interp(times, sample_t, rects_px[:, c]), sigma)
        for c in range(4)
    ])


# -- Whole-screen "grow the active window" emphasis (`screen_focus`) ------
#
# On a PLAIN whole-screen take (no capture-window crop, no multi-window cards)
# with several windows on screen, an auto zoom whose clicks are owned by ONE
# window is reshaped -- in the planner, via `_make_screen_focus_resolver` -- from
# a click-bbox follow into a fixed target on that window, pushed in only
# slightly. Render then GROWS that window in place over the already-pushed frame
# (`_apply_screen_grow`), so the two meet in the middle: the window overlaps its
# neighbours while the whole frame eases in. Everything below is inert unless
# `screen_focus` is on AND >=2 windows were recorded; off is byte-identical.

_SCREEN_GROW_MAX = 1.30        # window grows up to this in place (the dominant move)
_SCREEN_PUSH_CAP = 1.22        # frame pushes in at most this (slight)
_SCREEN_PUSH_FRAC = 0.5        # ...and only this fraction of the window's own fit
_SCREEN_MAX_UPSCALE = 1.30     # cap total on-canvas magnification vs the source
_SCREEN_MIN_WINDOWS = 2        # "multiple windows on the display"
_SCREEN_OWNER_FRAC = 0.8       # this share of a cluster's clicks must share an owner
_SCREEN_MIN_VISIBLE = 0.5      # owner must be at least half inside the frame
_SCREEN_EPS = 0.01
_SCREEN_PRESENCE_HOLE_SEC = 2.5   # a window is "live" within this of a sample
                                  # (record.WINDOW_HEARTBEAT_SEC is 1.0s)
_SCREEN_CORNER_FRAC = 0.02        # rounded-corner radius as a fraction of grown size
_SCREEN_SHADOW_SIGMA_FRAC = 0.02
_SCREEN_SHADOW_DY_FRAC = 0.012
_SCREEN_SHADOW_ALPHA = 0.5
_SCREEN_SHADOW_DOWNSCALE = 8


def _nearest_sample_gap(sample_t, times):
    """|t - nearest sample| for each t in `times` (both in seconds)."""
    s = np.sort(np.asarray(sample_t, dtype=float))
    t = np.asarray(times, dtype=float)
    if s.size == 0:
        return np.full(t.shape, np.inf)
    idx = np.clip(np.searchsorted(s, t), 0, s.size - 1)
    left = np.clip(idx - 1, 0, s.size - 1)
    return np.minimum(np.abs(s[idx] - t), np.abs(t - s[left]))


def _window_z_on_grid(sample_t, sample_z, times):
    """Hold-last-known z rank on the frame grid (z is discrete -- NEVER
    interpolated). Before the first sample, holds the first known rank."""
    s = np.asarray(sample_t, dtype=float)
    z = np.asarray(sample_z, dtype=int)
    t = np.asarray(times, dtype=float)
    if s.size == 0:
        return np.full(t.shape, -1, dtype=int)
    order = np.argsort(s, kind="stable")
    s, z = s[order], z[order]
    idx = np.clip(np.searchsorted(s, t, side="right") - 1, 0, z.size - 1)
    return z[idx].astype(int)


class _ScreenFocusCtx(object):
    """Per-window source-pixel rect grids + z + presence for a plain
    whole-screen take, all on the render frame grid. Answers the two questions
    the resolver and the grow track ask: who owns a click, and where the owner
    sits each frame."""

    def __init__(self, per, W, H, src_fps):
        # per: {wid: (rect_px (T,4), z_grid (T,), live (T,) bool)}
        self.per = per
        self.W, self.H = float(W), float(H)
        self.src_fps = float(src_fps)
        self.T = int(next(iter(per.values()))[0].shape[0])

    def fi_of(self, t):
        return max(0, min(self.T - 1, int(round(float(t) * self.src_fps))))

    def _clampi(self, i):
        return max(0, min(self.T - 1, int(i)))

    def state_at(self, fi):
        fi = self._clampi(fi)
        st = {}
        for wid, (rect, z, live) in self.per.items():
            if live[fi]:
                st[wid] = (rect[fi], int(z[fi]))
        return st

    def n_live_between(self, t0, t1):
        i0, i1 = self.fi_of(t0), self.fi_of(t1)
        if i1 < i0:
            i0, i1 = i1, i0
        return sum(1 for (_r, _z, live) in self.per.values()
                   if bool(np.any(live[i0:i1 + 1])))

    def owner_rect_median(self, wid, t0, t1):
        i0, i1 = self.fi_of(t0), self.fi_of(t1)
        if i1 < i0:
            i0, i1 = i1, i0
        return np.median(self.per[wid][0][i0:i1 + 1], axis=0)

    def rect_at(self, wid, fi):
        fi = self._clampi(fi)
        rect, _z, live = self.per[wid]
        return rect[fi] if live[fi] else None


def _build_screen_focus_ctx(ev, to_media, to_src, frame_times, W, H, src_fps,
                            min_windows=_SCREEN_MIN_WINDOWS):
    """Build a `_ScreenFocusCtx`, or None when the take can't support the
    feature: no geometry, fewer than `min_windows` identified windows, or an
    empty grid. Window rects map POINTS -> SOURCE PIXELS through the SAME
    `to_src` clicks use (an affine on the plain path), so windows and clicks
    share one space and one Retina scale -- the documented half-position bug is
    avoided by reusing that mapper rather than a fresh scale."""
    wt = ev.get("windows_t")
    wr = ev.get("windows_rect")
    wi = ev.get("windows_id")
    wz = ev.get("windows_z")
    if wt is None or wr is None or wi is None or len(wt) == 0:
        return None
    times = np.asarray(frame_times, dtype=float)
    if times.size == 0:
        return None
    wt = np.asarray(wt, dtype=float)
    wr = np.asarray(wr, dtype=float)
    wi = np.asarray(wi, dtype=int)
    wz = (np.asarray(wz, dtype=int) if wz is not None and len(wz) == len(wt)
          else np.full(wt.shape, -1, dtype=int))
    ids = sorted(set(int(v) for v in wi if v >= 0))
    if len(ids) < int(min_windows):
        return None
    per = {}
    for wid in ids:
        sel = wi == wid
        if not np.any(sel):
            continue
        sample_t = to_media(wt[sel])
        if sample_t.size < 1:
            continue
        on_grid_pt = _resample_track(sample_t, wr[sel], times, src_fps)
        tlx, tly = to_src(times, on_grid_pt[:, 0], on_grid_pt[:, 1])
        brx, bry = to_src(times, on_grid_pt[:, 0] + on_grid_pt[:, 2],
                          on_grid_pt[:, 1] + on_grid_pt[:, 3])
        rect_px = np.column_stack([tlx, tly, brx - tlx, bry - tly])
        z_grid = _window_z_on_grid(sample_t, wz[sel], times)
        live = _nearest_sample_gap(sample_t, times) <= _SCREEN_PRESENCE_HOLE_SEC
        per[wid] = (rect_px, z_grid, live)
    if len(per) < int(min_windows):
        return None
    return _ScreenFocusCtx(per, W, H, src_fps)


def _rect_visible_frac(x, y, w, h, W, H):
    if w <= 0.0 or h <= 0.0:
        return 0.0
    ix = max(0.0, min(x + w, W) - max(x, 0.0))
    iy = max(0.0, min(y + h, H) - max(y, 0.0))
    return (ix * iy) / (w * h)


def _make_screen_focus_resolver(ctx, win_w, win_h, fill, src_fps, out_w):
    """`(resolver, spans)` -- the closure `plan_zoom` calls to reshape eligible
    ranges, plus the claimed spans it records for the grow track.

    A range is claimed iff it is an auto follow range whose clicks are >=
    `_SCREEN_OWNER_FRAC`-owned by ONE unambiguous window, with >= 2 windows
    live during the range, the owner at least half in frame, a non-trivial
    push, AND the output actually has headroom to draw the grow. On claim it
    becomes a fixed target on the owner's centre at a slight, fill-adaptive
    push level (a small window earns a bit more push; a near-fullscreen one is
    left on today's follow).

    `out_w` gates the grow-headroom check: on an aspect-changed export
    (`--aspect` 9:16 / 1:1 / 4:5 ...) the base upscale `out_w/win_w` already
    consumes the `_SCREEN_MAX_UPSCALE` budget, so `_apply_screen_grow` would
    draw a grow of ~1.0 -- and reshaping the follow zoom into a gentler push
    would then emphasize the window LESS than the feature off. So we decline
    the claim there and keep today's follow zoom (the better emphasis when the
    frame is already cropped tight to a vertical/square canvas)."""
    base_upscale = float(out_w) / float(win_w) if win_w else 1.0
    spans = []

    def resolver(ranges):
        for r in ranges:
            if r.get("type") != "follow-click-groups" or "x" in r:
                continue
            if not r.get("screenAuto"):
                continue
            clks = r.get("clicks") or []
            if not clks:
                continue
            votes = {}
            ambiguous = False
            for c in clks:
                owner, amb = geometry.frontmost_owner(
                    ctx.state_at(ctx.fi_of(c[0])), float(c[1]), float(c[2]))
                if owner is None:
                    continue
                ambiguous = ambiguous or amb
                votes[owner] = votes.get(owner, 0) + 1
            if ambiguous or not votes:
                continue
            owner, cnt = max(votes.items(), key=lambda kv: kv[1])
            if cnt < _SCREEN_OWNER_FRAC * len(clks):
                continue
            t0, t1 = float(r["startTime"]), float(r["endTime"])
            if ctx.n_live_between(t0, t1) < _SCREEN_MIN_WINDOWS:
                continue
            wx, wy, ww, wh = (float(v)
                              for v in ctx.owner_rect_median(owner, t0, t1))
            if ww < 1.0 or wh < 1.0:
                continue
            if _rect_visible_frac(wx, wy, ww, wh, ctx.W, ctx.H) < _SCREEN_MIN_VISIBLE:
                continue
            z_fit = min(win_w * fill / ww if ww > 1.0 else 1e9,
                        win_h * fill / wh if wh > 1.0 else 1e9)
            push = 1.0 + min(_SCREEN_PUSH_CAP - 1.0,
                             max(0.0, (z_fit - 1.0) * _SCREEN_PUSH_FRAC))
            if push - 1.0 < _SCREEN_EPS:
                continue  # near-fullscreen owner: no room; keep today's follow
            # No on-canvas headroom for the grow (aspect-changed export): the
            # gentle push alone would emphasize the window LESS than the follow
            # zoom it replaces, so decline and keep the follow. Mirrors
            # `_apply_screen_grow`'s grow cap so "claimed" implies "grow > 1".
            if _SCREEN_MAX_UPSCALE / (push * base_upscale) <= 1.0 + _SCREEN_EPS:
                continue
            r["type"] = "fixed"
            r["x"] = wx + ww / 2.0
            r["y"] = wy + wh / 2.0
            r["zoom"] = push
            r["screenFocus"] = True
            spans.append({"start": t0, "end": t1, "owner": int(owner),
                          "push": float(push)})
    return resolver, spans


class _ScreenFocusTrack(object):
    def __init__(self, rects, wgrow):
        self._rects = rects
        self._wgrow = wgrow

    def at(self, pi):
        i = int(pi)
        if i < 0 or i >= len(self._rects):
            return None, 0.0
        return self._rects[i], float(self._wgrow[i])


def _build_screen_focus_track(spans, ctx, path, src_fps, always_zoomed):
    """Per-output-frame `(owner_rect_px | None, w_grow in 0..1)`.

    `w_grow` is read off the SAME simulated spring as the frame push
    (`path[pi].z` normalized by the span's push target), so grow and push rise
    and settle in lockstep -- genuinely simultaneous, no two-spring drift.
    `always_zoomed` holds the last span's grow to the clip end, matching the
    camera's held push."""
    if not spans or ctx is None or path is None or not len(path):
        return None
    T = len(path)
    rects = [None] * T
    wgrow = np.zeros(T, dtype=float)

    def _fill(i0, i1, owner, push, hold_rect=None):
        denom = max(_SCREEN_EPS, push - 1.0)
        for pi in range(max(0, i0), min(T, i1 + 1)):
            r = hold_rect if hold_rect is not None else ctx.rect_at(owner, pi)
            if r is None:
                continue
            w = (float(path[pi][2]) - 1.0) / denom
            rects[pi] = r
            wgrow[pi] = 0.0 if w < 0.0 else (1.0 if w > 1.0 else w)

    ordered = sorted(spans, key=lambda s: s["start"])
    for sp in ordered:
        _fill(int(round(sp["start"] * src_fps)),
              int(round(sp["end"] * src_fps)),
              int(sp["owner"]), float(sp["push"]))
    if always_zoomed:
        last = max(ordered, key=lambda s: s["end"])
        i1 = int(round(last["end"] * src_fps))
        hold = ctx.rect_at(int(last["owner"]), i1)
        if hold is not None:
            _fill(i1 + 1, T - 1, int(last["owner"]), float(last["push"]),
                  hold_rect=hold)
    if all(r is None for r in rects):
        return None
    return _ScreenFocusTrack(rects, wgrow)


def _apply_screen_grow(out, track, pi, x0, y0, z_eff):
    """Grow the active window in place over the already-pushed frame `out`.

    Grow-from-`out` (post-effects): whatever was drawn inside the window --
    ripples, spotlight, the synthetic cursor -- rides the magnification for
    free. The window's box on `out` is warped up around its own centre by
    `grow_eff`; only that rounded box is pasted back, so neighbours stay put
    and the grown window overlaps them, over a soft drop shadow. No-op when the
    track has nothing for this frame (so idle stretches stay byte-identical)."""
    if track is None:
        return out
    rect, w = track.at(pi)
    if rect is None or w < _SCREEN_EPS:
        return out
    sx, sy, sw, sh = (float(v) for v in rect)
    if sw < 1.0 or sh < 1.0:
        return out
    ox, oy = (sx - x0) * z_eff, (sy - y0) * z_eff
    ow, oh = sw * z_eff, sh * z_eff
    if ow < 2.0 or oh < 2.0:
        return out
    grow = 1.0 + w * (_SCREEN_GROW_MAX - 1.0)
    grow_eff = min(grow, max(1.0, _SCREEN_MAX_UPSCALE / max(1.0, z_eff)))
    if grow_eff <= 1.0 + _SCREEN_EPS:
        return out
    Ho, Wo = out.shape[:2]
    cx, cy = ox + ow / 2.0, oy + oh / 2.0
    gw, gh = ow * grow_eff, oh * grow_eff
    gx0, gy0 = int(round(cx - gw / 2.0)), int(round(cy - gh / 2.0))
    gw_i, gh_i = int(round(gw)), int(round(gh))
    if gw_i < 2 or gh_i < 2:
        return out
    # off-canvas guard: enough of the grown window must actually land on frame.
    vx0, vy0 = max(0, gx0), max(0, gy0)
    vx1, vy1 = min(Wo, gx0 + gw_i), min(Ho, gy0 + gh_i)
    if (vx1 - vx0) < 2 or (vy1 - vy0) < 2:
        return out
    if (vx1 - vx0) * (vy1 - vy0) < _SCREEN_MIN_VISIBLE * gw_i * gh_i:
        return out
    # Magnify the whole frame about the window centre, then paste back only the
    # window's own grown box -- neighbours are untouched, so the grown window
    # overlaps the still screen around it. BORDER_REPLICATE handles a window
    # box that runs slightly off frame.
    g = grow_eff
    M = np.float32([[g, 0.0, cx * (1.0 - g)], [0.0, g, cy * (1.0 - g)]])
    grown = cv2.warpAffine(out, M, (int(Wo), int(Ho)), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    framing.draw_drop_shadow(
        out, (gx0, gy0, gw_i, gh_i),
        max(1.0, _SCREEN_SHADOW_SIGMA_FRAC * Wo),
        int(_SCREEN_SHADOW_DY_FRAC * Ho),
        _SCREEN_SHADOW_ALPHA * w, _SCREEN_SHADOW_DOWNSCALE)
    radius = max(6, int(_SCREEN_CORNER_FRAC * min(gw_i, gh_i)))
    mask = framing._rounded_mask(gw_i, gh_i, radius)
    mx0, my0 = vx0 - gx0, vy0 - gy0
    sub = mask[my0:my0 + (vy1 - vy0), mx0:mx0 + (vx1 - vx0)]
    roi_out = out[vy0:vy1, vx0:vx1]
    roi_grown = grown[vy0:vy1, vx0:vx1]
    np.copyto(roi_out, roi_grown, where=(sub[:, :, None] == 255))
    return out


def _screen_focus_payload(spans, ctx, path, src_fps, out_w, out_h, win_w,
                          stride, idx, always_zoomed):
    """The editor's grow overlay data: per-sampled-frame owner rect (in OUTPUT
    px, on the pushed canvas) + grow weight, or None when nothing is claimed.

    Mirrors `_apply_screen_grow`'s geometry so the browser can draw the same
    magnified window box. `None` when the feature didn't fire, which keeps the
    served payload byte-stable for every take that doesn't use it."""
    track = _build_screen_focus_track(spans, ctx, path, src_fps, always_zoomed)
    if track is None:
        return None
    samples = []
    for pi in idx.tolist():
        rect, w = track.at(int(pi))
        if rect is None or w < _SCREEN_EPS:
            samples.append(None)
            continue
        cx, cy, z = path[int(pi)]
        z_eff = _zoom_to_output_scale(z, win_w, out_w)
        x0, y0 = _camera_window(cx, cy, z, win_w,
                                win_w * out_h / float(out_w), ctx.W, ctx.H)
        sx, sy, sw, sh = (float(v) for v in rect)
        grow = 1.0 + w * (_SCREEN_GROW_MAX - 1.0)
        grow_eff = min(grow, max(1.0, _SCREEN_MAX_UPSCALE / max(1.0, z_eff)))
        samples.append({
            "x": round((sx - x0) * z_eff, 2),
            "y": round((sy - y0) * z_eff, 2),
            "w": round(sw * z_eff, 2),
            "h": round(sh * z_eff, 2),
            "grow": round(float(grow_eff), 4),
        })
    if all(s is None for s in samples):
        return None
    return samples


def _native_track(meta, ev, raw_w, raw_h, times, to_media, src_fps, window_id):
    """`_NativeWindowTrack` for a window-native take, or None when unusable.

    Uses the window's OWN point size (`capture_window.logical_w/h`) as the
    points->pixels denominator -- NOT the top-level logical_w/h, which stay the
    display's. The window's live top-left comes from the geometry track (same
    `window` samples the crop path reads), in POINTS; with no track it falls
    back to the single meta rect (correct only if the window never moved).
    """
    cw = meta.get("capture_window") if isinstance(meta, dict) else None
    if not isinstance(cw, dict):
        return None
    raw_w, raw_h = int(raw_w), int(raw_h)
    if raw_w < 2 or raw_h < 2:
        return None
    # The window's own point size is the points->pixels denominator. Prefer the
    # explicit logical_w/h; fall back to the rect's size when they are absent or
    # malformed (older/partial meta), which is the same number at capture start
    # -- so a native take stays mapped in WINDOW space rather than escaping to
    # the display fallback at the call site.
    rect = cw.get("rect")
    try:
        r = [float(v) for v in rect] if isinstance(rect, (list, tuple)) else None
    except (TypeError, ValueError):
        r = None
    if r is not None and (len(r) != 4 or not all(np.isfinite(v) for v in r)):
        r = None
    try:
        lw, lh = float(cw["logical_w"]), float(cw["logical_h"])
        if not (np.isfinite(lw) and np.isfinite(lh)) or lw <= 0.0 or lh <= 0.0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        if r is None:
            return None
        lw, lh = r[2], r[3]
    if lw <= 0.0 or lh <= 0.0:
        return None
    # A take recorded before 2026-08-31 stored the DISPLAY-CLIPPED rect for a
    # window hanging off the edge of the screen, which makes this denominator
    # too small and the scale inflated on the clipped axis -- clicks then land
    # off-target inside the card. Identity when the rect and buffer agree.
    lw, lh = _unclip_point_size(lw, lh, raw_w, raw_h)
    scale = raw_w / lw          # points->pixels (uniform Retina); ~ raw_h/lh
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    samples = _window_samples(ev, window_id)
    if samples is not None:
        sample_t = to_media(samples[0])
        rects_pt = np.asarray(samples[1], dtype=float)      # POINTS
        on_grid = _resample_track(sample_t, rects_pt, times, src_fps)
    else:
        # No geometry track: a single static rect (correct only if the window
        # never moved -- documented). `r` was validated above.
        if r is None:
            return None
        on_grid = np.tile(np.asarray(r, dtype=float), (int(times.size), 1))
    return _NativeWindowTrack(on_grid, raw_w, raw_h, times, scale)


def _native_track_for_channel(channel, ev, times, to_media, src_fps):
    """`_NativeWindowTrack` for one channel of the multi-native manifest
    (P3.3). Same math as `_native_track` -- the only difference is the
    source of `logical_w/h`, `rect`, and `id`: the manifest channel dict
    rather than the top-level `capture_window` block. Returns None when
    the channel's meta can't produce a usable track.

    Buffer dims come from the manifest (`buffer_w/h`) so this doesn't
    have to open the raw_i.mov just to read them; the encoder already
    wrote those on the SIZE line captured into the channel entry, and
    they cannot change mid-take (`plan_slots` and AVAssetWriterInput both
    pin them).
    """
    if not isinstance(channel, dict):
        return None
    raw_w = int(channel.get("buffer_w") or 0)
    raw_h = int(channel.get("buffer_h") or 0)
    if raw_w < 2 or raw_h < 2:
        return None
    rect = channel.get("rect")
    try:
        r = [float(v) for v in rect] if isinstance(rect, (list, tuple)) else None
    except (TypeError, ValueError):
        r = None
    if r is not None and (len(r) != 4 or not all(np.isfinite(v) for v in r)):
        r = None
    try:
        lw = float(channel["logical_w"])
        lh = float(channel["logical_h"])
        if not (np.isfinite(lw) and np.isfinite(lh)) or lw <= 0.0 or lh <= 0.0:
            raise ValueError
    except (KeyError, TypeError, ValueError):
        if r is None:
            return None
        lw, lh = r[2], r[3]
    if lw <= 0.0 or lh <= 0.0:
        return None
    # Same display-clipped-rect correction as `_native_track` above.
    lw, lh = _unclip_point_size(lw, lh, raw_w, raw_h)
    scale = raw_w / lw
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    wid = channel.get("id")
    samples = _window_samples(ev, wid) if wid is not None else None
    if samples is not None:
        sample_t = to_media(samples[0])
        rects_pt = np.asarray(samples[1], dtype=float)
        on_grid = _resample_track(sample_t, rects_pt, times, src_fps)
    else:
        # No geometry track for this window id -- fall back to the manifest's
        # single rect, correct only while the window doesn't move. This is the
        # same tri-state single-window native carries; the per-channel `track`
        # field (from _channel_meta) already says "failed" in this case so a
        # consumer that surfaces it can warn.
        if r is None:
            return None
        on_grid = np.tile(np.asarray(r, dtype=float), (int(times.size), 1))
    return _NativeWindowTrack(on_grid, raw_w, raw_h, times, scale)


def _build_native_card_cameras(channels, native_tracks, clicks_t, clicks_x,
                                clicks_y, frame_times, max_zoom, params,
                                suppressed_ranges, plan_duration, src_fps,
                                enabled=False):
    """Per-card auto-zoom paths for a multi-native take (P3.3).

    Same shape as `_build_card_cameras` for the display-crop path -- the
    difference is CLICK OWNERSHIP: instead of `_source_to_card` (which
    picks clicks by containment inside a display crop's rect), each
    channel's `_NativeWindowTrack.to_src` maps EVERY shared click into
    that card's buffer, clamped-not-dropped -- so a click that landed on
    card A also appears (clamped to an edge) in card B's zoom planner.
    That's the doc's clamp-not-drop invariant preserved across the shared
    click set: dropping the click would remove its TIME from the trailing
    cluster set the three consumers (`camera.cluster_to_range`,
    `edits.auto_zoom_proposals`, `beats`) share.

    A "picked" click near a card's edge with no local activity on that
    card produces a mild pull -- accepted trade for the shared time set.
    None when a card never zooms, same optimization as the display-crop
    version: a still card stays on the untouched `paint()` resize.
    """
    n = len(channels or [])
    if not enabled or not n:
        return [None] * n
    cards = []
    for i, ch in enumerate(channels):
        tr = native_tracks[i] if i < len(native_tracks) else None
        if tr is None:
            cards.append({"w": 1, "h": 1, "clicks": []})
            continue
        picked = []
        if clicks_t is not None and len(clicks_t):
            # Native track does the whole array in one call. Clamping
            # preserves the click's TIME (and therefore its membership in
            # the shared trailing-cluster set), which is the load-bearing
            # invariant `test_beats.ClusterContractTests` pins.
            xs, ys = tr.to_src(clicks_t, clicks_x, clicks_y)
            for k in range(len(clicks_t)):
                picked.append((float(clicks_t[k]),
                               float(xs[k]), float(ys[k])))
        cards.append({"w": tr.raw_w, "h": tr.raw_h, "clicks": picked})
    paths = camera.build_card_paths(
        frame_times, cards, max_zoom=max_zoom, params=params,
        suppressed_ranges=suppressed_ranges, plan_duration=plan_duration)
    out = []
    for path in paths:
        zoomed = path.size and float(np.max(path[:, 2])) > 1.0 + _CARD_ZOOM_EPS
        out.append(path if zoomed else None)
    return out


def _apply_native_card_cameras(crops, card_paths, native_tracks, cells,
                                 frame_index):
    """Warp each zooming native card's crop to cell size through its own
    camera. Parallel to `_apply_card_cameras` for the display-crop path;
    differs only in where the card dims come from (per-channel
    `_NativeWindowTrack.raw_w/h`, not `_clamp_window_rect`), and the crops
    are already whole frames (each raw_i.mov IS its window).
    """
    if not card_paths:
        return crops
    out = list(crops)
    for i, path in enumerate(card_paths):
        if path is None or i >= len(out):
            continue
        tr = native_tracks[i] if i < len(native_tracks) else None
        if tr is None:
            continue
        cell = cells[i]
        dw, dh = tr.raw_w, tr.raw_h
        idx = int(min(max(int(frame_index), 0), len(path) - 1))
        cx, cy, z = path[idx]
        x0, y0 = _camera_window(cx, cy, z, dw, dh, dw, dh)
        out[i] = _warp(out[i], x0, y0, z, dw, dh,
                       int(cell["w"]), int(cell["h"]))
    return out


def _native_card_click_times(native_tracks, clicks_t, clicks_x, clicks_y):
    """Per-card click TIMES, one list per channel (P3.3).

    Ownership rule for multi-native: a click's TIME goes into every card's
    list -- the clamped position stays inside each card's buffer, so the
    single-shared-clicks-into-many-cards mapping preserves the trailing
    cluster set for every consumer. `_build_focus_emphasis` and per-card
    beats read this directly.
    """
    out = []
    if clicks_t is None or not len(clicks_t):
        return [[] for _ in native_tracks]
    for tr in native_tracks:
        if tr is None:
            out.append([])
            continue
        # Clamp lands the click inside the buffer, times unchanged. For the
        # focus emphasis the position isn't used -- only the times matter --
        # so this is essentially a no-op that returns the shared time set.
        out.append([float(t) for t in clicks_t])
    return out


def _build_window_track(ev, crop, raw_w, raw_h, scale_x, scale_y,
                        frame_times, to_media, src_fps, window_id=None,
                        meta=None):
    """`_WindowTrack` for the capture crop, or None to keep the snapshot rect.

    None -- and therefore today's exact single-snapshot behavior -- whenever
    the session isn't window-targeted, its capture_window was rejected, or it
    predates geometry tracking (no `window` lines). Fewer than two samples is
    also None: one sample is just a second, competing snapshot.

    A window-NATIVE take is the exception: raw.mov IS the window, so the mapper
    is built regardless of `crop` (which is None for native) and BEFORE the
    crop-follow logic below.
    """
    times = np.asarray(frame_times, dtype=float)
    if _is_window_native_meta(meta) and times.size:
        native = _native_track(meta, ev, raw_w, raw_h, times, to_media,
                               src_fps, window_id)
        if native is not None:
            return native
    if crop is None:
        return None
    samples = _window_samples(ev, window_id)
    if samples is None:
        return None
    times = np.asarray(frame_times, dtype=float)
    if times.size == 0:
        return None
    sample_t, rects = to_media(samples[0]), samples[1]
    # points -> recorded pixels, per axis, exactly as every other recorded
    # coordinate is converted.
    px = np.column_stack([
        rects[:, 0] * scale_x, rects[:, 1] * scale_y,
        rects[:, 2] * scale_x, rects[:, 3] * scale_y,
    ])
    on_grid = _resample_track(sample_t, px, times, src_fps)
    return _WindowTrack(on_grid, crop, raw_w, raw_h, times, scale_x, scale_y)


# Minimum overlap before a drawn grid rect is taken to BE a real window. Set
# well above chance: at 0.6 a rect drawn around half a window (~0.5) stays
# static, so "follow" only happens when the user clearly framed the window.
_GRID_BIND_MIN_IOU = 0.6


def _rect_iou_over_time(rects, target):
    """IoU of a fixed `target` rect against an (N, 4) track, per row."""
    tx, ty, tw, th = [float(v) for v in target]
    x0 = np.maximum(rects[:, 0], tx)
    y0 = np.maximum(rects[:, 1], ty)
    x1 = np.minimum(rects[:, 0] + rects[:, 2], tx + tw)
    y1 = np.minimum(rects[:, 1] + rects[:, 3], ty + th)
    inter = np.maximum(0.0, x1 - x0) * np.maximum(0.0, y1 - y0)
    union = rects[:, 2] * rects[:, 3] + tw * th - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _build_grid_tracks(ev, windows, crop, capture_track, W, H,
                       scale_x, scale_y, frame_times, to_media, src_fps,
                       enabled=True):
    """One `_WindowTrack` per multi-window card, None where it stays static.

    The grid's rects are drawn by hand on a frame, with no window identity
    attached -- so each is BOUND to whichever recorded window it best
    overlaps, and then follows it. Binding by overlap (rather than asking the
    user to name a window) is what lets this work on rects that already exist,
    including ones drawn before geometry was ever recorded.

    The match is the best IoU that rect achieves against a window at ANY point
    in the take, not at t=0: the rect was drawn at whatever playhead the user
    happened to be at, and a window that moved would otherwise fail to match
    itself.

    Everything is computed in SOURCE space -- the space `edits.windows` rects
    live in -- which for a window-captured session means inside the capture
    crop, tracked if that crop is itself following.
    """
    if not enabled or not windows:
        return [None] * len(windows or [])
    blank = [None] * len(windows)
    times = np.asarray(frame_times, dtype=float)
    if times.size == 0:
        return blank
    wi = ev.get("windows_id")
    if wi is None or len(wi) == 0:
        return blank
    ids = sorted(set(int(v) for v in np.asarray(wi, dtype=int) if int(v) >= 0))
    if not ids:
        return blank

    # Every candidate window, resampled once, expressed in source space.
    candidates = {}
    for wid in ids:
        samples = _window_samples(ev, wid)
        if samples is None:
            continue
        sample_t = to_media(samples[0])
        r = samples[1]
        px = np.column_stack([r[:, 0] * scale_x, r[:, 1] * scale_y,
                              r[:, 2] * scale_x, r[:, 3] * scale_y])
        grid = _resample_track(sample_t, px, times, src_fps)
        candidates[wid] = _raw_track_to_source(grid, crop, capture_track, times)

    out = []
    for spec in windows:
        base = _clamp_window_rect(spec, W, H)
        # A card seeded from a record-time pick already KNOWS which window it
        # is -- the picker had the real id in hand. Use it: the IoU search
        # below is a reconstruction for rects that were drawn freehand, and
        # it can genuinely pick the wrong window when one is dragged over
        # where another used to be. Absent `window_id` (every hand-drawn
        # rect, and every session authored before this) takes the unchanged
        # path.
        pinned = _spec_window_id(spec)
        if pinned is not None and pinned in candidates:
            out.append(_WindowTrack(candidates[pinned], base, W, H, times,
                                    scale_x, scale_y))
            continue
        best_id, best_iou = None, 0.0
        for wid, track in candidates.items():
            iou = float(_rect_iou_over_time(track, base).max())
            if iou > best_iou:
                best_id, best_iou = wid, iou
        if best_id is None or best_iou < _GRID_BIND_MIN_IOU:
            out.append(None)
            continue
        out.append(_WindowTrack(candidates[best_id], base, W, H, times,
                                scale_x, scale_y))
    return out


def _describe_capture_windows(meta, ev, t0):
    """The `capture_windows` block plus what render derived, or None.

    Derived per window: `tracked` (its geometry is on disk, so its card can
    follow it) and `occluded_sec` (how long something sat on top of it, from
    the z track). Both are STATED FACTS for the editor to render -- the same
    posture as `capture_window.applied`, which exists precisely so consumers
    stop inferring things the server already knows.
    """
    block = meta.get("capture_windows")
    if not isinstance(block, dict):
        return None
    entries = block.get("windows")
    if not isinstance(entries, list) or not entries:
        return None
    ids = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            ids.append(int(entry["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    covered = occlusion_report(ev, ids)
    out = dict(block)
    described = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        item = dict(entry)
        try:
            wid = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            described.append(item)
            continue
        item["tracked"] = entry.get("track") == "ok"
        item["occluded_sec"] = float(covered.get(wid, 0.0))
        described.append(item)
    out["windows"] = described
    out["any_occluded"] = any(w.get("occluded_sec", 0.0) > 0.25
                              for w in described)
    return out


def occlusion_report(ev, window_ids):
    """How long each of `window_ids` spent underneath another window.

    Display-capture-plus-crop physically cannot see through an occluding
    window -- whatever was on top is in the crop. This does not fix that; it
    DETECTS it, which is the difference between a card that is quietly wrong
    and one the editor can flag. A window counts as covered at a sample when
    some other window is in front of it (lower z) and their rects intersect.

    Everything is in raw sample time, so the caller converts to media time
    the same way every other consumer does. `{}` for a session with no z
    track, which is every session recorded before this existed.
    """
    t = ev.get("windows_t")
    rects = ev.get("windows_rect")
    ids = ev.get("windows_id")
    zs = ev.get("windows_z")
    if t is None or rects is None or ids is None or zs is None:
        return {}
    if len(t) == 0 or len(zs) != len(t) or int(np.max(zs)) < 0:
        return {}

    wanted = set(int(w) for w in (window_ids or []))
    if not wanted:
        return {}

    # Walk the samples in time order, keeping the newest rect+z per window,
    # and integrate covered time over the gaps between samples.
    covered = dict((w, 0.0) for w in wanted)
    latest = {}
    order = np.argsort(t, kind="stable")
    prev_t = None
    for k in order:
        now = float(t[k])
        if prev_t is not None and now > prev_t:
            dt = now - prev_t
            for w in wanted:
                me = latest.get(w)
                if me is None:
                    continue
                for other_id, other in latest.items():
                    if other_id == w or other[1] < 0 or me[1] < 0:
                        continue
                    if other[1] >= me[1]:
                        continue  # behind us, or same rank
                    a, b = me[0], other[0]
                    ox = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
                    oy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
                    if ox > 0 and oy > 0:
                        covered[w] += dt
                        break
        prev_t = now
        latest[int(ids[k])] = (rects[k], int(zs[k]))
    return dict((w, round(v, 3)) for w, v in covered.items() if v > 0.0)


def recorded_windows(session_dir):
    """Every window whose geometry the take recorded: id + a stable rect.

    Exists so a card can be pinned to a real window BY ID instead of by
    measuring a rect off a preview frame and hoping render's overlap match
    agrees with what you meant. The rect returned is the window's MEDIAN
    position over the take, not its first sample -- a window that was dragged
    once shouldn't report the corner it started in.

    Geometry and opaque ids only, never a name or title: this reads the same
    events.jsonl that deliberately doesn't record what you had open, and
    surfacing identity here would quietly undo that.
    """
    try:
        meta, raw_path, events_path = _session_paths(session_dir)
    except Exception:
        return []
    ev = geometry.load_events(events_path)
    ids = ev.get("windows_id")
    rects = ev.get("windows_rect")
    if ids is None or rects is None or len(ids) == 0:
        return []
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        return []
    try:
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if raw_w < 2 or raw_h < 2:
        return []
    try:
        scale_x = raw_w / float(meta.get("logical_w", raw_w) or raw_w)
        scale_y = raw_h / float(meta.get("logical_h", raw_h) or raw_h)
    except (TypeError, ValueError, ZeroDivisionError):
        return []

    covered = occlusion_report(ev, sorted(set(int(v) for v in ids if v >= 0)))
    out = []
    for wid in sorted(set(int(v) for v in ids if int(v) >= 0)):
        rows = rects[np.asarray(ids, dtype=int) == wid]
        if rows.shape[0] == 0:
            continue
        med = np.median(rows, axis=0)
        out.append({
            "window_id": int(wid),
            "x": float(med[0] * scale_x), "y": float(med[1] * scale_y),
            "w": float(med[2] * scale_x), "h": float(med[3] * scale_y),
            "samples": int(rows.shape[0]),
            "occluded_sec": float(covered.get(int(wid), 0.0)),
        })
    return out


def capture_window_specs(session_dir):
    """`meta["capture_windows"]` as source-pixel card rects, or [].

    The bridge between the record-time pick (POINTS, global top-left, because
    that is the only unit Quartz can give without knowing the backing-store
    scale) and `edits.windows` (source pixels, because that is the space the
    editor drags rects in). The per-axis scale comes off the REAL recorded
    file for exactly the reason `_capture_crop_px` does it that way -- it is
    what makes fractional Retina scaling come out exact.

    A multi-window take has no capture crop, so source space IS raw space
    here. Each spec carries the real macOS `window_id`, which is what lets
    the card bind to its window by identity instead of by IoU guess.
    """
    try:
        meta, raw_path, _events = _session_paths(session_dir)
    except Exception:
        return []
    block = meta.get("capture_windows")
    if not isinstance(block, dict) or block.get("units") != "points":
        return []
    entries = block.get("windows")
    if not isinstance(entries, list) or not entries:
        return []
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        return []
    try:
        raw_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if raw_w < 2 or raw_h < 2:
        return []
    try:
        scale_x = raw_w / float(meta.get("logical_w", raw_w) or raw_w)
        scale_y = raw_h / float(meta.get("logical_h", raw_h) or raw_h)
    except (TypeError, ValueError, ZeroDivisionError):
        return []
    if not (np.isfinite(scale_x) and np.isfinite(scale_y)) \
            or scale_x <= 0 or scale_y <= 0:
        return []
    origin = block.get("display_origin") or (0.0, 0.0)
    try:
        ox, oy = float(origin[0]), float(origin[1])
    except (IndexError, TypeError, ValueError):
        ox = oy = 0.0

    out = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        rect = entry.get("rect")
        if not isinstance(rect, (list, tuple)) or len(rect) != 4:
            continue
        try:
            rx, ry, rw, rh = (float(v) for v in rect)
            wid = int(entry["id"])
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite(v) for v in (rx, ry, rw, rh)):
            continue
        x = max(0.0, min(float(raw_w - 2), (rx - ox) * scale_x))
        y = max(0.0, min(float(raw_h - 2), (ry - oy) * scale_y))
        w = max(2.0, min(float(raw_w) - x, rw * scale_x))
        h = max(2.0, min(float(raw_h) - y, rh * scale_y))
        out.append({"x": x, "y": y, "w": w, "h": h, "window_id": wid})
    return out


def _spec_window_id(spec):
    """The real macOS window id a card is pinned to, or None."""
    if not isinstance(spec, dict):
        return None
    raw = spec.get("window_id")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _raw_track_to_source(rects_px, crop, capture_track, times):
    """Move a raw-pixel rect track into the session's SOURCE space.

    Identity for an ordinary full-display recording (source IS raw). For a
    window-captured one, source is inside the capture crop -- and when that
    crop is itself tracked, the mapping is time-varying, so the corners go
    through the same `to_src_px` every event coordinate does.
    """
    if crop is None:
        return rects_px
    if capture_track is None:
        out = rects_px.copy()
        out[:, 0] -= crop[0]
        out[:, 1] -= crop[1]
        return out
    x0, y0 = capture_track.to_src_px(times, rects_px[:, 0], rects_px[:, 1])
    x1, y1 = capture_track.to_src_px(times,
                                     rects_px[:, 0] + rects_px[:, 2],
                                     rects_px[:, 1] + rects_px[:, 3])
    return np.column_stack([x0, y0, np.maximum(x1 - x0, 1.0),
                            np.maximum(y1 - y0, 1.0)])


def _apply_capture_crop(frame_bgr, crop):
    """Crop a decoded frame to the recorded window rect.

    Returns a numpy VIEW (no copy) when cropping, and the frame object
    ITSELF -- identity, not a copy -- when `crop is None`, so a session
    with no `capture_window` runs the literal pre-feature code path.
    """
    if crop is None:
        return frame_bgr
    x, y, w, h = crop
    return frame_bgr[y:y + h, x:x + w]


# -- The editor's crop (`edits.crop`) ------------------------------------
#
# A SECOND, static crop stage that runs after the record-time `capture_window`
# stage above. Two separate stages rather than one composed rect, because the
# first one is not necessarily static: `_WindowTrack.apply` re-crops every
# frame to where the window actually was. What makes stacking them safe is
# that the track then *resizes back to the snapshot size*, so whatever the
# first stage does, the space it hands on is a constant (W, H) -- which is
# exactly the space `edits.crop` is expressed in, and the space
# `describe_session` reports and `source_frame` returns.
#
# The rect must NEVER be composed into the `crop` that reaches
# `_build_window_track`. That function reads `window_id=None` as "track every
# window" (see `_window_samples`), so handing it a user rect on a session with
# a geometry track would build a follow-path out of ten unrelated windows'
# rects averaged together and the crop would wander across the take.


def _user_crop_px(rect, W, H, min_dim=16):
    """`(x, y, w, h)` ints for the editor's crop, clamped into the `W` x `H`
    source space, or None when there is nothing usable to crop to.

    `W`/`H` are the dims AFTER the capture stage, not the raw file's.

    Fails safe to None -- today's uncropped path, byte-identical -- on a
    missing/malformed rect, a non-finite number, or a rect that ends up
    smaller than `min_dim` on either side once clamped. Sizes are even-ized
    INWARD for the same load-bearing reason `_capture_crop_px` does it:
    `_encode_cmd` feeds the dims to `libx264 -pix_fmt yuv420p`, which rejects
    odd ones, so an odd crop would break every export.
    """
    if not isinstance(rect, dict):
        return None
    try:
        rx = float(rect.get("x", 0.0))
        ry = float(rect.get("y", 0.0))
        rw = float(rect["w"])
        rh = float(rect["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (rx, ry, rw, rh)):
        return None
    W, H = int(W), int(H)
    if W < 2 or H < 2:
        return None
    x0 = int(np.floor(max(0.0, rx)))
    y0 = int(np.floor(max(0.0, ry)))
    x1 = int(np.ceil(min(float(W), rx + rw)))
    y1 = int(np.ceil(min(float(H), ry + rh)))
    w = (x1 - x0) & ~1
    h = (y1 - y0) & ~1
    min_dim = max(2, int(min_dim))
    if w < min_dim or h < min_dim:
        return None
    if x0 == 0 and y0 == 0 and w == W and h == H:
        return None          # a full-frame crop is not a crop
    return (x0, y0, int(w), int(h))


def _apply_user_crop(frame_bgr, user):
    """Slice a frame to the editor's crop.

    Returns a numpy VIEW when cropping and the frame object ITSELF -- identity,
    not a copy -- when `user is None`, so an uncropped session runs the literal
    pre-feature code path. Mirrors `_apply_capture_crop`.
    """
    if user is None:
        return frame_bgr
    x, y, w, h = user
    return frame_bgr[y:y + h, x:x + w]


def _user_crop_to_src(to_src, user):
    """Wrap a source-space mapper so event coordinates land in crop space.

    Returns `to_src` UNCHANGED when there is no crop, so the uncropped path
    keeps the exact function object it had before.
    """
    if user is None:
        return to_src
    ux, uy = float(user[0]), float(user[1])

    def _cropped(t, ax, ay):
        sx, sy = to_src(t, ax, ay)
        return sx - ux, sy - uy

    return _cropped


# -- The cursor eraser (`render.cursor_erase`) ---------------------------
#
# Everything here is a no-op when the option is off: `render()` holds `None`
# and never calls a method, so an un-erased export runs the literal
# pre-feature loop. See autocine/eraser.py for the algorithm.


def _erase_applies(meta):
    """Whether this session even has a burned-in cursor to erase.

    A `--cursor synthetic` take switched ffmpeg's `-capture_cursor` off, so
    there is nothing in the pixels; its pointer is drawn by `effects.CursorFX`
    at render time, AFTER the warp, where this stage cannot reach it anyway.
    An absent key means an older session, and `record.py`'s default has always
    been "system" -- the same reading `preview_frame` uses for `cursor_fx`.
    """
    return (meta or {}).get("cursor_mode", "system") == "system"


def _cursor_fx_draws(cursor_fx, meta, cursor_erase):
    """Whether `effects.CursorFX` may paint a pointer over this take.

    The rule has always been one thing -- never two cursors on screen -- and
    there are now two ways to satisfy it. A `--cursor synthetic` capture never
    burned one in. A system take did, but `cursor_erase` takes it back out
    upstream of the warp (see `_erase_applies` above and the loop below), so
    the drawn pointer REPLACES the recorded one instead of doubling it; that
    is what makes the synthetic cursor available retroactively, on footage
    that was not recorded for it.

    Without the eraser a system take keeps the original no-op, byte for byte
    (`test_render_windows.py::SyntheticCursorInWindowsMode::
    test_system_cursor_mode_still_suppresses_it`) -- an enabled `cursor_fx`
    alone must never start drawing over a pointer that is still there.

    The live-playback compositor deliberately does NOT follow this
    relaxation; `multi_window_cursor_track` says why.
    """
    if not cursor_fx:
        return False
    if (meta or {}).get("cursor_mode", "system") == "synthetic":
        return True
    return bool(cursor_erase)


def _erase_crop_fn(crop, track, user_crop, badge=None):
    """A `frame, index -> source-space frame` mapper for the eraser's prepass.

    The prepass decodes the file a second time and MUST land in exactly the
    space the render loop works in, or every sample it takes is offset by the
    capture crop. Built from the same stages, in the same order -- including
    the capture-indicator erase, which changes pixels the prepass may sample.
    `badge` is a CLONE of the loop's eraser, never the instance itself: both
    walk the file from the start, and one lock/verify state cannot serve two
    interleaved passes.
    """
    def _fn(fr, i):
        if track is not None:
            fr = track.apply(fr, i)
        else:
            fr = _apply_capture_crop(fr, crop)
        if badge is not None:
            badge.apply(fr)
        return _apply_user_crop(fr, user_crop)

    return _fn


def _erase_boxes(ev, to_media, to_src, frame_times, W, H, scale_x, scale_y,
                 params=None):
    """Per-frame cursor search boxes + hotspot rects, in source space.

    Reads the position streams DIRECTLY off `ev` rather than reusing the
    camera's arrays: those are gated by `typing_zoom`/`scroll_zoom`, which are
    zoom-behaviour switches. Where the pointer physically was is not a
    question about zooming, and a take rendered with `scroll_zoom: false` must
    not suddenly stop erasing the cursor during a scroll.
    """
    parts_t, parts_x, parts_y = [], [], []
    for key in ("moves", "clicks", "scrolls"):
        t = to_media(ev[key + "_t"])
        if not t.size:
            continue
        x, y = to_src(t, ev[key + "_x"], ev[key + "_y"])
        parts_t.append(np.asarray(t, dtype=float))
        parts_x.append(np.asarray(x, dtype=float))
        parts_y.append(np.asarray(y, dtype=float))
    if not parts_t:
        return None, None, None
    bx, by, fx, fy, ring = eraser.box_padding(scale_x, scale_y, params)
    return eraser.boxes_for_track(
        frame_times, np.concatenate(parts_t), np.concatenate(parts_x),
        np.concatenate(parts_y), W, H, bx, by, fx, fy, ring=ring)


def _preview_cursor_eraser(raw_path, boxes, hots, tracked, idx, src_fps,
                           crop_fn, params=None):
    """An eraser for ONE frame, from a bounded window around it.

    The export's prepass reads the whole file, which is the right trade for a
    background job and the wrong one for a still the editor repaints on every
    edit. So the preview walks `_ERASE_PREVIEW_REACH_SEC` either side and
    accepts a worse answer than the export where the pointer has been parked
    longer than that: those pixels have no clean sample inside the window and
    take the spatial fallback instead. The still is therefore a FLOOR on the
    export's quality, never a flattering one -- which is the direction a
    preview should be wrong in.
    """
    fps = float(src_fps) or 60.0
    reach = max(1, int(round(_ERASE_PREVIEW_REACH_SEC * fps)))
    lo = max(0, idx - reach)
    hi = min(len(boxes), idx + reach + 1)
    cap = cv2.VideoCapture(raw_path)
    if not cap.isOpened():
        return None
    try:
        if lo > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        return eraser.plan(cap, crop_fn, boxes, hots, tracked, fps,
                           start_idx=idx, end_idx=idx + 1,
                           walk_start=lo, walk_limit=hi, params=params)
    finally:
        cap.release()


def _offset_typing_anchors(anchors, crop, track=None, user=None):
    """Translate vision typing anchors from RAW source px into window space.

    `vision.cached_typing_anchors` watches the *file*, so its anchors know
    nothing about the crop. `camera.build_path` ignores typing_anchors
    today, so this is behaviorally inert -- but they're the one coordinate
    stream that doesn't ride the event scale path, and leaving them in raw
    space would be a silent landmine the day they're consumed.

    A geometry `track` SUPERSEDES `crop`: window space is then time-varying,
    so each anchor is mapped through the rect at its own burst start rather
    than through one fixed origin.

    Returns NEW dicts: the list comes straight out of vision's LRU and must
    never be mutated in place.
    """
    if not anchors or (crop is None and track is None and user is None):
        return anchors
    ux, uy = (float(user[0]), float(user[1])) if user is not None else (0.0, 0.0)
    out = []
    for a in anchors:
        if not isinstance(a, dict):
            out.append(a)
            continue
        b = dict(a)
        ax = float(a.get("x", 0.0))
        ay = float(a.get("y", 0.0))
        if track is not None:
            t = np.asarray([float(a.get("start", 0.0))])
            mx, my = track.to_src_px(t, np.asarray([ax]), np.asarray([ay]))
            b["x"], b["y"] = float(mx[0]), float(my[0])
        elif crop is not None:
            b["x"] = ax - crop[0]
            b["y"] = ay - crop[1]
        else:
            b["x"], b["y"] = ax, ay
        # The editor's crop is the second stage, so its origin comes off
        # whatever the capture stage produced -- including a tracked rect.
        b["x"] -= ux
        b["y"] -= uy
        out.append(b)
    return out


def _capture_window_moved(cw):
    """Did the recorded window move / resize / vanish during the take?

    `end_rect` is snapshotted at stop; null means the window was gone by
    then. Either way the fixed crop stops describing the whole window part
    way through, which is a warning for the UI to surface -- it never
    changes what the renderer does.
    """
    end = cw.get("end_rect")
    if end is None:
        return True
    rect = cw.get("rect")
    if not isinstance(rect, (list, tuple)) or not isinstance(end, (list, tuple)):
        return True
    if len(rect) != 4 or len(end) != 4:
        return True
    try:
        a = [float(v) for v in rect]
        b = [float(v) for v in end]
    except (TypeError, ValueError):
        return True
    if not all(np.isfinite(v) for v in a + b):
        return True
    return any(abs(p - q) > _CAPTURE_MOVE_TOL_PT for p, q in zip(a, b))


def _describe_capture_window(meta, crop=None):
    """meta's `capture_window` block for describe_session, plus two derived
    flags -- or None for an ordinary full-screen session.

    `moved`   -- the window drifted/vanished during the take.
    `tracked` -- geometry was logged over time AND render is following it, so
      `moved` is handled rather than a defect. The UI needs both: "it moved"
      and "we followed it" are a reassurance; "it moved" alone is a warning.
    `applied` -- the crop actually RESOLVED. Non-null `capture_window` does
      NOT imply a cropped render: every fail-safe path in `_capture_crop_px`
      (non-point units, a rect on a secondary display, a sub-min_dim window)
      keeps the block but renders full-frame. Without this, a consumer can
      only *infer* the answer by comparing raw vs cropped dims -- which is
      wrong for a window that exactly covers the display. Pass the `crop`
      that `_capture_crop_px` returned for the same meta.
    """
    cw = meta.get("capture_window") if isinstance(meta, dict) else None
    if not isinstance(cw, dict):
        return None
    out = dict(cw)
    out["moved"] = bool(_capture_window_moved(cw))
    # A window-native take has no crop (raw.mov IS the window), yet the render
    # honors it and follows the window's geometry -- so `applied`/`tracked`
    # can't key off `crop` here. `mode` is already carried through `dict(cw)`.
    native = cw.get("mode") == "window_native"
    out["applied"] = (crop is not None) or native
    # Following only actually happens when the crop resolved (display-crop) or
    # the take is native -- a rejected capture_window renders full-frame.
    out["tracked"] = bool(cw.get("track") == "ok") and (crop is not None or native)
    return out


def _warn_ignored_multi_window_options(motion_blur, click_fx, spotlight):
    """Windows mode is static crops with no per-frame camera path, so the
    camera-driven effects have nothing to key off of. Print an explained
    no-op (same spirit as the existing '--music file not found' warning)
    rather than silently dropping options a caller explicitly asked for.

    `cursor_fx` is deliberately NOT in this list any more: the cursor is
    positional, not camera-driven, so `_draw_multi_cursor` maps it through
    each card's own crop-and-scale instead. Everything still here genuinely
    needs a camera window -- motion blur needs its velocity, the spotlight
    needs a single frame-filling image to darken, and click ripples still
    need the per-card work the cursor just got.
    """
    ignored = [name for name, on in (
        ("motion_blur", motion_blur), ("click_fx", click_fx),
        ("spotlight", spotlight)) if on]
    if ignored:
        print("  note: {} ignored in multi-window mode (static crops, "
             "no per-frame camera path to key off)".format(", ".join(ignored)))


def _warp(frame_bgr, x0, y0, z, win_w, win_h, out_w, out_h,
          interp=cv2.INTER_CUBIC):
    """Warp the (win_w/z, win_h/z) source window at (x0, y0) up to (out_w, out_h).

    Sub-pixel affine warp (not integer slicing) so slow pans/zooms don't
    stair-step or shimmer. sx/sy come out equal in practice -- (win_w, win_h)
    always matches the (out_w, out_h) aspect by construction (see
    framing.resolve_aspect_canvas / _contain_fit) -- but are computed
    independently for clarity/robustness.
    """
    z = max(1.0, float(z))
    sx = out_w / (win_w / z)
    sy = out_h / (win_h / z)
    M = np.float32([[sx, 0.0, -x0 * sx], [0.0, sy, -y0 * sy]])
    return cv2.warpAffine(frame_bgr, M, (int(round(out_w)), int(round(out_h))),
                          flags=interp,
                          borderMode=cv2.BORDER_REPLICATE)


# Camera motion blur: when the virtual camera pans/zooms fast, one hard warp
# per frame strobes. Instead we average several warps of the SAME source frame
# at camera positions sub-sampled across a shutter window centered on the
# frame instant -- true camera-motion blur with zero ghosting of screen
# content (content motion between source frames is never blended).
_MB_SHUTTER = 0.5          # fraction of the frame interval the shutter is open (180deg)
_MB_PX_PER_SAMPLE = 2.0    # target on-screen px of motion between sub-samples
_MB_MAX_SAMPLES = 12       # hard cap on warps per frame (perf bound)
_MB_MIN_PX = 1.5           # below this shutter-window motion, skip blur entirely


def _motion_blur_cams(path, i, win_w, win_h, out_w, out_h,
                      shutter=_MB_SHUTTER, px_per_sample=_MB_PX_PER_SAMPLE,
                      max_samples=_MB_MAX_SAMPLES, min_px=_MB_MIN_PX):
    """Camera sub-samples (cx, cy, z) for frame i's shutter window.

    Uses the central-difference camera velocity so the sample cloud is
    centered on path[i] (no perceived lag), and sizes the sample count from
    the motion magnitude in OUTPUT pixels: pan displacement scaled by the
    true output scale, plus the zoom term's edge displacement. Returns a
    single-element list (== no blur) for a slow/static camera, so the caller
    can keep the crisp single-warp path.
    """
    n_path = len(path)
    i = max(0, min(int(i), n_path - 1))
    i0 = max(i - 1, 0)
    i1 = min(i + 1, n_path - 1)
    cx, cy, z = float(path[i][0]), float(path[i][1]), float(path[i][2])
    if i1 == i0:
        return [(cx, cy, z)]
    # Per-frame camera delta (central difference where both neighbors exist).
    dcx = (float(path[i1][0]) - float(path[i0][0])) / (i1 - i0)
    dcy = (float(path[i1][1]) - float(path[i0][1])) / (i1 - i0)
    dz = (float(path[i1][2]) - float(path[i0][2])) / (i1 - i0)

    z_safe = max(1.0, z)
    kx = out_w / float(win_w)          # output px per source px at zoom 1
    ky = out_h / float(win_h)
    pan_px = hypot(dcx * z_safe * kx, dcy * z_safe * ky)
    # A zoom change moves edge content by ~(dz/z) * (half output diagonal).
    zoom_px = abs(dz) / z_safe * 0.5 * hypot(float(out_w), float(out_h))
    motion_px = (pan_px + zoom_px) * shutter
    if motion_px < min_px:
        return [(cx, cy, z)]
    n = int(np.ceil(motion_px / px_per_sample))
    n = max(2, min(int(max_samples), n))
    cams = []
    for k in range(n):
        s = ((k + 0.5) / n - 0.5) * shutter   # frame-interval units, symmetric
        cams.append((cx + dcx * s, cy + dcy * s, max(1.0, z + dz * s)))
    return cams


def _warp_blend(frame_bgr, cams, win_w, win_h, out_w, out_h, W, H):
    """Warp once per camera sub-sample and box-average the results.

    One sample -> the ordinary crisp cubic warp (identical to no-blur).
    Several -> linear-interp warps (the averaging hides cubic's edge) summed
    in float32 to avoid quantization drift.
    """
    if len(cams) == 1:
        cx, cy, z = cams[0]
        x0, y0 = _camera_window(cx, cy, z, win_w, win_h, W, H)
        return _warp(frame_bgr, x0, y0, z, win_w, win_h, out_w, out_h)
    acc = None
    for cx, cy, z in cams:
        x0, y0 = _camera_window(cx, cy, z, win_w, win_h, W, H)
        w = _warp(frame_bgr, x0, y0, z, win_w, win_h, out_w, out_h,
                  interp=cv2.INTER_LINEAR)
        acc = w.astype(np.float32) if acc is None else acc + w
    return (acc / float(len(cams)) + 0.5).astype(np.uint8)


def _zoom_to_output_scale(z, win_w, out_w):
    """The true output-pixels-per-source-pixel scale for a given camera zoom.

    Effects (click ripples, spotlight, cursor) need *this*, not the raw
    camera `z`, to size/position themselves correctly once the render canvas
    can differ from the source: z_eff = out_w / (win_w / z). Reduces to
    exactly `z` when win_w == out_w (the default "auto" aspect).
    """
    return max(1.0, float(z)) * (float(out_w) / float(win_w))


def _apply_fade(out_bgr, t, total_dur, fade):
    """Fade to black over the first/last `fade` seconds of the clip."""
    f = 1.0
    if t < fade:
        f = t / fade
    tail = total_dur - t
    if tail < fade:
        f = min(f, max(0.0, tail) / fade)
    if f >= 1.0:
        return out_bgr
    f = max(0.0, min(1.0, f))
    return (out_bgr.astype(np.float32) * f).astype(np.uint8)


def _make_gif(mp4_path, gif_path, fps=15, width=1000):
    palette = gif_path + ".palette.png"
    vf = "fps={},scale={}:-1:flags=lanczos".format(fps, width)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", mp4_path, "-frames:v", "1",
                    "-vf", vf + ",palettegen", palette], check=True)
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", mp4_path, "-i", palette,
                    "-lavfi", vf + "[x];[x][1:v]paletteuse", gif_path], check=True)
    try:
        os.remove(palette)
    except OSError:
        pass
