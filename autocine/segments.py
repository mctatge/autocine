"""Pure math for SEGMENTED takes.

A segmented take is one recording PAUSED and RESUMED into K sequential files
(`capture_segments` in meta), whose paused wall-clock gaps are DELETED at
render. This module holds the two pure pieces every layer shares:

- `is_segmented_meta(meta)` -- the fail-safe discriminator (also used to keep
  the off switch bit-exact: a non-segmented take never enters a new branch).
- `SegmentClock` -- the event-time -> concatenated-output-time map. It lives
  HERE, in one place, because `render()`, `beats.session_beats`, and
  `describe_session` must agree EXACTLY on it: those three share the
  trailing-cluster contract (docs/architecture.md hard invariant; `ClusterContractTests`).
  A second, drifting copy of this map inside the editor/beats surface is the
  headline bug the design's adversarial review caught -- the export would be
  right while the editor silently showed only segment 0.

Design + provenance: docs/architecture.md.

Segments are the SEQUENTIAL sibling of multi-native's PARALLEL
`capture_channels`; the two are mutually exclusive (a meta carrying
`capture_channels` is never segmented).

SCENE TAKES (docs/architecture.md) compose the two axes: a
`capture_scenes` manifest is K sequential SCENES, each of which is itself a
fleet of 1-4 parallel native channels (or, later, one whole-screen file).
The scene-side pure math lives here for the same reason `SegmentClock` does:
`render()`, `describe_session`, and `beats` must agree EXACTLY on the
event-time -> output-time map, the per-scene channel alignment, and which
scene OWNS each event. `is_scene_meta` / `scene_channel_alignment` /
`scene_clock_entries` / `SegmentClock.owner` / `scene_events_view` are that
single copy.
"""

import numpy as np

# A fleet scene's content duration is min-across-channels of decoded frames --
# which the P3.0 re-measure showed can run SHORT of wall time (a 15s take gave
# 13.2-14.3s at N=3-4). Events in that shortfall clamp onto the seam; warn
# when the manifest's recorded wall end says the gap is real and material.
SCENE_SHORTFALL_WARN_SEC = 0.35


def is_segmented_meta(meta):
    """True for a segmented take -- a `capture_segments` list of >=2 entries,
    each carrying a `file`, and NO multi-native `capture_channels`.

    Fail-safe: anything malformed (not a dict, <2 entries, a missing `file`,
    or the presence of `capture_channels`) returns False, so the caller falls
    back to today's single-file path byte-for-byte. Mirrors the defensive
    posture of `render._is_multi_native_meta`.
    """
    if not isinstance(meta, dict):
        return False
    if meta.get("capture_channels"):          # multi-native owns that shape
        return False
    segs = meta.get("capture_segments")
    if not isinstance(segs, list) or len(segs) < 2:
        return False
    return all(isinstance(s, dict) and s.get("file") for s in segs)


def is_scene_meta(meta):
    """True for a SCENE take -- a `capture_scenes` list of >=2 entries, each
    a fleet of 1-4 native channels (`channels`, every one carrying a `file`)
    XOR one whole-screen file (`file`), with NO sibling manifest.

    Fail-safe like its siblings: anything malformed returns False and the
    caller falls through to the existing dispatch order, where an unknown
    shape still fails LOUD ("cannot open recording: raw.mov") rather than
    silently rendering the wrong thing. >=2 mirrors `is_segmented_meta`: a
    one-scene take collapses to its legacy shape at finalize, so a 1-entry
    manifest on disk is malformed by construction.
    """
    if not isinstance(meta, dict):
        return False
    if meta.get("capture_channels") or meta.get("capture_segments"):
        return False                       # strict four-way partition
    scenes = meta.get("capture_scenes")
    if not isinstance(scenes, list) or len(scenes) < 2:
        return False
    for s in scenes:
        if not isinstance(s, dict):
            return False
        chans = s.get("channels")
        has_file = bool(s.get("file"))
        if bool(chans) == has_file:        # exactly one of the two
            return False
        if chans is not None:
            if not isinstance(chans, list) or not (1 <= len(chans) <= 4):
                return False
            if not all(isinstance(c, dict) and c.get("file") for c in chans):
                return False
    return True


def scene_channel_alignment(channel_t0s, channel_counts, fps):
    """Lockstep alignment for ONE fleet scene: `(origin, n_out, offsets)`.

    The multi-native aligner's math (`render._render_multi_native`) as a pure
    function, so render, describe and the clock builder share one copy:
    per-channel frame offset relative to channel 0, shared origin at the
    latest starter, and the common overlap `n_out` clipped at whichever
    channel runs out first. A channel with a null t0 raises -- inheriting the
    reader-side `or session_t0` fallback here would anchor that channel to
    the TAKE's origin and silently misalign the whole scene (docs/architecture.md,
    "per-channel t0 is never null").
    """
    if not channel_t0s:
        return 0, 0, []
    if any(t is None for t in channel_t0s):
        raise ValueError("fleet scene has a channel with no t0_monotonic")
    fps = float(fps)
    if fps <= 0:
        raise ValueError("scene alignment needs fps > 0")
    base = float(channel_t0s[0])
    offsets = [int(round((float(t) - base) * fps)) for t in channel_t0s]
    origin = max(0, max(offsets))
    n_out = min(int(c) - max(0, origin - o)
                for c, o in zip(channel_counts, offsets))
    return origin, max(0, n_out), offsets


def scene_clock_entries(scenes, per_scene_counts, fps):
    """Per-scene clock inputs for a `capture_scenes` manifest.

    `per_scene_counts[s]` is the list of decoded frame counts for scene s's
    channels (a one-element list for a whole-screen scene). Returns
    `(t0s, durs, aligns, warnings)` where `t0s[s]` is the ORIGIN-ADJUSTED
    parent-monotonic time of the scene's composite frame 0

        t0_s = t0_ch0 + origin/fps,   D_s = n_out/fps,

    `aligns[s]` is `(origin, n_out, offsets)`, and `warnings` is a list of
    human-readable strings (the caller decides where they may be printed --
    this module never writes to stdout, the MCP owns it).

    Two guards the adversarial review forced:
    - **overlap**: `D_s` is clipped so scene s can never own scene s+1's
      opening events (an above-grid channel rate would otherwise let
      `t0_s + D_s` cross the next scene's t0).
    - **shortfall**: when the manifest recorded `wall_end_monotonic` and the
      decoded timeline undershoots it by more than SCENE_SHORTFALL_WARN_SEC,
      the gap is named -- events in it clamp onto the seam, and silence here
      is how a phantom-cluster generator ships.
    """
    if len(per_scene_counts) != len(scenes):
        raise ValueError("per_scene_counts ({}) != capture_scenes ({})".format(
            len(per_scene_counts), len(scenes)))
    fps = float(fps)
    t0s, durs, aligns, warnings = [], [], [], []
    for s, entry in enumerate(scenes):
        counts = per_scene_counts[s]
        chans = entry.get("channels")
        if chans:
            ch_t0s = [c.get("t0_monotonic") for c in chans]
            origin, n_out, offsets = scene_channel_alignment(
                ch_t0s, counts, fps)
            t0 = float(ch_t0s[0]) + origin / fps
        else:
            # A whole-screen (`file`) scene is a valid MANIFEST shape
            # (is_scene_meta accepts it -- the S2 forward contract) but no
            # S1 reader produces a count for one; a bare IndexError here
            # would replace the loud failure the discriminator promises.
            if not counts:
                raise RuntimeError(
                    "scene {} is a whole-screen (file) scene -- those are "
                    "an S2 follow-up; this build reads fleet scenes only "
                    "(docs/architecture.md)".format(s))
            t0 = float(entry.get("t0_monotonic") or 0.0)
            origin, n_out, offsets = 0, int(counts[0]), [0]
        dur = n_out / fps if fps > 0 else 0.0
        wall_end = entry.get("wall_end_monotonic")
        if wall_end is not None:
            short = float(wall_end) - (t0 + dur)
            if short > SCENE_SHORTFALL_WARN_SEC:
                warnings.append(
                    "scene {}: decoded timeline ends {:.2f}s before the "
                    "recorded pause -- events in that span clamp to the seam "
                    "(fleet channels can under-run wall time; prefer <=3 "
                    "windows)".format(s, short))
        t0s.append(t0)
        durs.append(dur)
        aligns.append((origin, n_out, offsets))
    # Overlap guard: a scene may never reach into its successor's timeline.
    # The clip lands ON THE FRAME GRID (n_out first, then dur = n_out/fps):
    # setting dur to the raw span would put this clock sub-frame off the
    # frame stream the render actually emits AND off the beats rebuild
    # (which derives durs from frame_count) -- the three-surface agreement
    # failing exactly where the warning fires. The epsilon keeps float
    # noise in the t0 deltas from triggering a spurious one-frame clip.
    for s in range(len(t0s) - 1):
        span = t0s[s + 1] - t0s[s]
        if durs[s] > span + 1e-6:
            n_out = max(0, int(span * fps + 1e-6))
            warnings.append(
                "scene {}: content duration {:.3f}s overlaps the next "
                "scene's start (span {:.3f}s) -- clipped; the channel may "
                "not be constant frame rate".format(s, durs[s], span))
            durs[s] = n_out / fps
            aligns[s] = (aligns[s][0], n_out, aligns[s][2])
    return t0s, durs, aligns, warnings


def scene_events_view(clock, s, ev):
    """A SCENE-SCOPED copy of `geometry.load_events`' dict for scene `s`.

    The load-bearing rule (docs/architecture.md, "A materialized scene-scoped
    EVENTS VIEW"): per-scene consumers must never map the FULL event set
    through a clamping scene-local mapper -- every other scene's events would
    pile onto this scene's boundaries as phantom clusters (the pause-gate bug
    class, a third time). So the subsetting happens HERE, once:

    - **point events** (clicks/moves/scrolls/keys/ups) are subset by
      `clock.owner(t) == s`, slicing the t/x/y arrays TOGETHER so the
      positional zip alignment every consumer relies on is preserved. A
      seam-clamped event belongs to the EARLIER scene (owner()'s contract).
    - **geometry samples** are subset by parent-time containment
      `[t0_s, t0_s + D_s)` PLUS, per window id, the nearest PRIOR sample as
      a left anchor for the resampler ((t, rect) pairs are self-aligned, so
      this asymmetry is legal where slicing clicks by value would not be).

    Times stay in PARENT clock; the caller applies its scene-local affine
    (`arr - t0_s`) knowing everything in the view is in-scene by
    construction, so the affine never needs to clamp.
    """
    t0 = clock.t0s[s]
    dur = clock.durs[s]
    out = {}
    families = (("clicks_t", ("clicks_x", "clicks_y")),
                ("moves_t", ("moves_x", "moves_y")),
                ("scrolls_t", ("scrolls_x", "scrolls_y")),
                ("ups_t", ()),
                ("keys_t", ()))
    for tkey, companions in families:
        t = ev.get(tkey)
        n = int(getattr(t, "size", len(t) if t is not None else 0) or 0)
        if n == 0:
            out[tkey] = np.array([], dtype=float)
            for c in companions:
                out[c] = np.array([], dtype=float)
            continue
        t = np.asarray(t, dtype=float)
        sel = clock.owner(t) == s
        out[tkey] = t[sel]
        for c in companions:
            v = ev.get(c)
            if v is not None and len(v) == n:
                out[c] = np.asarray(v, dtype=float)[sel]
            else:
                out[c] = np.array([], dtype=float)
    wt, wr = ev.get("windows_t"), ev.get("windows_rect")
    n = int(getattr(wt, "size", 0) or 0)
    if n and wr is not None and len(wr) == n:
        t = np.asarray(wt, dtype=float)
        wi = ev.get("windows_id")
        ids = (np.asarray(wi, dtype=int) if wi is not None and len(wi) == n
               else np.full(t.shape, -1, dtype=int))
        keep = (t >= t0) & (t < t0 + dur)
        for wid in np.unique(ids):
            prior = np.nonzero((ids == wid) & (t < t0))[0]
            if prior.size:
                keep[prior[-1]] = True
        wz = ev.get("windows_z")
        out["windows_t"] = t[keep]
        out["windows_rect"] = np.asarray(wr, dtype=float)[keep]
        out["windows_id"] = ids[keep]
        out["windows_z"] = (np.asarray(wz, dtype=int)[keep]
                            if wz is not None and len(wz) == n
                            else np.full(int(keep.sum()), -1, dtype=int))
    else:
        out["windows_t"] = np.array([], dtype=float)
        out["windows_rect"] = np.zeros((0, 4), dtype=float)
        out["windows_id"] = np.array([], dtype=int)
        out["windows_z"] = np.array([], dtype=int)
    return out


class SegmentClock:
    """Maps a parent-monotonic event timestamp to its position in the
    CONCATENATED output timeline, with the paused wall-clock gaps removed.

    Built from per-segment ``(t0_monotonic, content_duration)`` pairs. The
    content duration ``D_i = frame_count_i / fps`` is measured from the DECODED
    files at render time -- the caller passes the SAME per-segment frame counts
    the renderer uses to build the frame grid, so the mapper and the grid share
    one source of truth (an independent per-file count can diverge from the
    joined file's real placement, misplacing every post-seam event).

    The map, for the segment ``i`` that owns ``t`` (``t0_i <= t < t0_i + D_i``):

        media(t) = S_i + (t - t0_i),   where  S_i = sum_{j<i} D_j.

    Paused gaps vanish because the sum runs over content durations only; no
    inter-segment gap term appears. An event before segment 0, inside a deleted
    gap, or a sub-frame past a segment's content end is **clamped** into
    ``[0, total]`` -- NEVER dropped. Dropping would shorten `t` relative to the
    parallel x/y arrays it is positionally zipped against
    (``clicks_x, clicks_y = to_src(clicks_t, ...)``), pairing every later click
    with a neighbour's pixel. Clamp-not-drop keeps `media` shape-preserving,
    exactly like the window-native click transform.
    """

    def __init__(self, t0s, durs):
        t0s = [float(x) for x in t0s]
        durs = [float(x) for x in durs]
        if len(t0s) != len(durs) or len(t0s) < 1:
            raise ValueError("SegmentClock needs >=1 matching (t0, dur) pairs")
        self.t0s = t0s
        self.durs = durs
        # Output start offset S_i = sum of prior content durations.
        starts = []
        acc = 0.0
        for d in durs:
            starts.append(acc)
            acc += d
        self.starts = starts
        self.total = acc

    @classmethod
    def from_meta(cls, meta, seg_frame_counts, fps):
        """Build from a `capture_segments` manifest + per-segment decoded frame
        counts + fps. `seg_frame_counts[i]` is `CAP_PROP_FRAME_COUNT` of the
        i-th segment file (or the cumulative-count derivation from the joined
        file); it and the manifest must be the same length and order.
        """
        segs = meta.get("capture_segments") or []
        if len(seg_frame_counts) != len(segs):
            raise ValueError(
                "seg_frame_counts ({}) != capture_segments ({})".format(
                    len(seg_frame_counts), len(segs)))
        fps = float(fps) or 0.0
        t0s = [float(s.get("t0_monotonic") or 0.0) for s in segs]
        durs = [(float(c) / fps) if fps > 0 else 0.0 for c in seg_frame_counts]
        return cls(t0s, durs)

    def media(self, arr):
        """Map an array of parent-monotonic timestamps to output-media seconds.

        Vectorized, shape-preserving (same length in, same length out), and
        clamp-not-drop. Accepts scalars, lists, or numpy arrays; returns a
        numpy array of the same shape as the raveled input.
        """
        a = np.asarray(arr, dtype=float).ravel()
        if a.size == 0:
            return a
        out = np.full(a.shape, np.nan, dtype=float)
        remaining = np.ones(a.shape, dtype=bool)
        # Assign every event that lands inside a segment's content window.
        for i in range(len(self.t0s)):
            lo = self.t0s[i]
            hi = lo + self.durs[i]
            in_seg = remaining & (a >= lo) & (a < hi)
            if in_seg.any():
                out[in_seg] = self.starts[i] + (a[in_seg] - lo)
                remaining &= ~in_seg
        if not remaining.any():
            return out
        # Everything left is before segment 0, in a deleted gap, or past the
        # last segment's content -> clamp.
        before0 = remaining & (a < self.t0s[0])
        out[before0] = 0.0
        remaining &= ~before0
        if remaining.any():
            # For a gap/tail event, clamp to the seam at or before it: the
            # content end of the last segment whose t0 <= t. searchsorted on
            # the ascending t0s gives that index; its content end == the next
            # seam (== S_{i+1}), and for the final segment == total.
            seams = np.array([self.starts[j] + self.durs[j]
                              for j in range(len(self.durs))], dtype=float)
            idx = np.searchsorted(np.asarray(self.t0s), a[remaining],
                                  side="right") - 1
            idx = np.clip(idx, 0, len(self.durs) - 1)
            out[remaining] = np.minimum(seams[idx], self.total)
        return out

    def owner(self, arr):
        """Which segment/scene OWNS each timestamp -- an int array, same
        length as the raveled input, computed by the SAME assignment pass
        `media()` runs so the two can never disagree.

        The boundary rule the scene review forced: an event `media()` clamps
        onto a seam belongs to the EARLIER scene (the one it was clamped
        INTO), never to the next scene's frame 0 -- value-containment on the
        mapped time would hand every seam-clamped event to the wrong scene's
        coordinate space (`[S_s, S_s+D_s)` is half-open exactly at the clamp
        value). Before-segment-0 events belong to segment 0; a final-scene
        tail event belongs to the final scene.
        """
        a = np.asarray(arr, dtype=float).ravel()
        out = np.zeros(a.shape, dtype=int)
        if a.size == 0:
            return out
        remaining = np.ones(a.shape, dtype=bool)
        for i in range(len(self.t0s)):
            lo = self.t0s[i]
            in_seg = remaining & (a >= lo) & (a < lo + self.durs[i])
            out[in_seg] = i
            remaining &= ~in_seg
        if remaining.any():
            before0 = remaining & (a < self.t0s[0])
            out[before0] = 0
            remaining &= ~before0
        if remaining.any():
            idx = np.searchsorted(np.asarray(self.t0s), a[remaining],
                                  side="right") - 1
            out[remaining] = np.clip(idx, 0, len(self.durs) - 1)
        return out

    def scene_local(self, s, arr):
        """Scene-LOCAL media time for events already known to be owned by
        scene `s` (i.e. the output of `scene_events_view`): a plain affine
        `t - t0_s`, clipped into `[0, D_s)` so a tail event that out-ran the
        scene's decoded timeline clamps to its last frame instead of
        escaping the plan. Shape-preserving like `media()`."""
        a = np.asarray(arr, dtype=float).ravel()
        if a.size == 0:
            return a
        hi = max(0.0, self.durs[s] - 1e-6)
        return np.clip(a - self.t0s[s], 0.0, hi)

    @classmethod
    def from_scene_meta(cls, meta, per_scene_counts, fps):
        """Build the take's clock from a `capture_scenes` manifest + decoded
        per-channel counts. Returns `(clock, aligns, warnings)` -- see
        `scene_clock_entries`. The ONE builder every surface calls."""
        scenes = meta.get("capture_scenes") or []
        t0s, durs, aligns, warnings = scene_clock_entries(
            scenes, per_scene_counts, fps)
        return cls(t0s, durs), aligns, warnings

    def duration(self):
        """Total output duration = sum of content durations (gaps removed)."""
        return self.total
