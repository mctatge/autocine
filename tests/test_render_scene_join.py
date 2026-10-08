"""Render foundation for the SEAMLESS window-join (milestone 2).

A gapfree join keeps each surviving window on ONE continuous file across the
seam; the `capture_scenes` manifest expresses each scene as a SUB-RANGE of
that file via `frame_start` / `frame_count`. This pins the two render edits
that make that work:
  * `_scene_channel_count` -- a scene reads its slice, not the whole file, so
    the earlier scene does not overrun into the later one;
  * `_scene_channel_start` -- a later scene seeks past the frames the earlier
    scene consumed, so it shows the RIGHT frames (not a re-read of the head).

The off-switch (a pause/resume scene take, whose channels are per-scene files
with neither key) is covered byte-for-byte by test_render_scenes.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import framing, render


FPS = 30


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class JoinSeamHelpers(unittest.TestCase):
    """The pure gates that decide when the card ENTRANCE animates."""

    def _ch(self, *files):
        return [{"file": f} for f in files]

    def test_join_seam_is_prev_plus_one_same_files(self):
        # scene s = scene s-1's SAME continuous files, in order, + one new file
        self.assertTrue(render._is_join_seam(
            self._ch("raw_0.mov", "raw_1.mov"),
            self._ch("raw_0.mov", "raw_1.mov", "raw_2.mov")))

    def test_pause_resume_reseam_is_not_a_join(self):
        # per-scene files (different names) -> NOT a join, keeps its hard cut
        self.assertFalse(render._is_join_seam(
            self._ch("scene_0_raw_0.mov"),
            self._ch("scene_1_raw_0.mov", "scene_1_raw_1.mov")))

    def test_removal_or_reorder_is_not_a_join(self):
        self.assertFalse(render._is_join_seam(
            self._ch("raw_0.mov", "raw_1.mov"), self._ch("raw_0.mov")))
        self.assertFalse(render._is_join_seam(
            self._ch("raw_0.mov", "raw_1.mov"),
            self._ch("raw_1.mov", "raw_0.mov", "raw_2.mov")))

    def test_mid_list_insertion_is_a_join_seam(self):
        # The M3.3 auto-REJOIN shape: the returnee sits at its departed
        # card's SLOT, not appended (decision 11). The prefix version of
        # this gate hard-cut every rejoin entrance -- caught on the first
        # real end-to-end take.
        self.assertTrue(render._is_join_seam(
            self._ch("raw_0.mov", "raw_2.mov"),
            self._ch("raw_0.mov", "raw_3.mov", "raw_2.mov")))
        # Insertion at the FRONT (a rank-0 departure on a micless take).
        self.assertTrue(render._is_join_seam(
            self._ch("raw_1.mov", "raw_2.mov"),
            self._ch("raw_3.mov", "raw_1.mov", "raw_2.mov")))
        # Survivor order must still be PRESERVED around the insertion.
        self.assertFalse(render._is_join_seam(
            self._ch("raw_0.mov", "raw_2.mov"),
            self._ch("raw_2.mov", "raw_3.mov", "raw_0.mov")))

    def test_entrance_cells_map_survivors_by_file_on_mid_insertion(self):
        # Survivor at scene-s position 2 must morph FROM its own previous
        # cell (prev position 1), and the mid-list returnee grows from a
        # point at its final cell's centre -- positional mapping would hand
        # the returnee a survivor's cell and grow the survivor from a point.
        rects = [(400.0, 300.0), (400.0, 300.0)]
        to_cells = [(10, 10, 300, 200), (350, 10, 300, 200),
                    (690, 10, 300, 200)]
        prev_files = ["raw_0.mov", "raw_2.mov"]
        cur_files = ["raw_0.mov", "raw_3.mov", "raw_2.mov"]
        frm, to = render._join_entrance_from_cells(
            rects, to_cells, 1920, 1080, "feature",
            prev_files=prev_files, cur_files=cur_files)
        prev_place = framing.placements_for(1920, 1080, rects,
                                            layout="feature")
        self.assertEqual(frm[0], tuple(int(v) for v in prev_place[0][:4]))
        self.assertEqual(frm[2], tuple(int(v) for v in prev_place[1][:4]))
        cx, cy = 350 + 150, 10 + 100
        self.assertEqual(frm[1], (cx, cy, 2, 2))     # the returnee grows in
        # Without file lists the mapping stays positional (the appended
        # joiner, byte-identical to before).
        frm2, _ = render._join_entrance_from_cells(
            rects, to_cells, 1920, 1080, "feature")
        self.assertEqual(frm2[0], tuple(int(v) for v in prev_place[0][:4]))
        self.assertEqual(frm2[1], tuple(int(v) for v in prev_place[1][:4]))
        self.assertEqual(frm2[2][2:], (2, 2))

    def test_smoothstep_rests_at_both_ends(self):
        self.assertEqual(render._smoothstep(0.0), 0.0)
        self.assertEqual(render._smoothstep(1.0), 1.0)
        self.assertEqual(render._smoothstep(-5), 0.0)   # clamped
        self.assertEqual(render._smoothstep(9), 1.0)
        self.assertAlmostEqual(render._smoothstep(0.5), 0.5)


class HiddenChannelsFilter(unittest.TestCase):
    """`hidden_channels` — the reversible "remove this card" render filter.
    Pure, no ffmpeg. The editor writes FILE names; render drops those channels
    (and their positional layout slots) from every scene, under two guards."""

    def _scenes(self):
        return [
            {"channels": [{"file": "raw_0.mov", "app": "Chrome"},
                          {"file": "raw_1.mov", "app": "Claude"}]},
            {"channels": [{"file": "raw_0.mov", "app": "Chrome"},
                          {"file": "raw_1.mov", "app": "Claude"},
                          {"file": "raw_2.mov", "app": "Finder"}]},
        ]

    def test_off_switch_is_identity(self):
        sc = self._scenes()
        out, lay = render._apply_hidden_scenes(sc, [], {"1": [None, None]})
        self.assertIs(out, sc)                       # same object, untouched
        self.assertEqual(lay, {"1": [None, None]})
        self.assertEqual(render._apply_hidden_channels(
            [{"file": "a"}], None, [{"x": 0}]), ([{"file": "a"}], [{"x": 0}]))

    def test_hides_a_file_from_every_scene(self):
        out, _ = render._apply_hidden_scenes(self._scenes(), ["raw_2.mov"], {})
        self.assertEqual([[c["file"] for c in s["channels"]] for s in out],
                         [["raw_0.mov", "raw_1.mov"],
                          ["raw_0.mov", "raw_1.mov"]])   # scene 1 lost Finder

    def test_anchor_is_never_hidden(self):
        # scene 0 / channel 0 carries the clock: a request to hide it is dropped.
        out, _ = render._apply_hidden_scenes(self._scenes(), ["raw_0.mov"], {})
        self.assertEqual(len(out[1]["channels"]), 3)    # unchanged

    def test_never_empties_a_scene(self):
        # A one-card scene whose only file is asked hidden keeps that card.
        sc = [{"channels": [{"file": "raw_0.mov"}, {"file": "raw_1.mov"}]},
              {"channels": [{"file": "raw_1.mov"}]}]
        out, _ = render._apply_hidden_scenes(sc, ["raw_1.mov"], {})
        self.assertEqual([c["file"] for c in out[1]["channels"]], ["raw_1.mov"])

    def test_layout_slice_is_filtered_in_lockstep(self):
        # scene 1 places card index 2 (Finder); hiding Finder drops that slot.
        lay = {"1": [None, None, {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}]}
        _, out_lay = render._apply_hidden_scenes(
            self._scenes(), ["raw_2.mov"], lay)
        self.assertNotIn("1", out_lay)   # only slot was the hidden card -> gone

    def test_multi_native_filter_and_layout(self):
        ch = [{"file": "raw_0.mov"}, {"file": "raw_1.mov"},
              {"file": "raw_2.mov"}]
        out, lay = render._apply_hidden_channels(
            ch, ["raw_1.mov"], [None, {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}])
        self.assertEqual([c["file"] for c in out],
                         ["raw_0.mov", "raw_2.mov"])
        self.assertEqual(lay, [])   # the placed slot (idx 1) was hidden


class SceneChannelHelpers(unittest.TestCase):
    class _Cap:
        def __init__(self, n):
            self._n = n

        def get(self, _prop):
            return float(self._n)

    def test_count_prefers_frame_count_slice(self):
        cap = self._Cap(60)
        self.assertEqual(render._scene_channel_count(cap, {"frame_count": 30}), 30)

    def test_count_falls_back_to_full_file(self):
        cap = self._Cap(60)
        self.assertEqual(render._scene_channel_count(cap, {}), 60)
        # 0 / None frame_count are treated as absent, not an empty slice.
        self.assertEqual(render._scene_channel_count(cap, {"frame_count": 0}), 60)

    def test_start_defaults_zero(self):
        self.assertEqual(render._scene_channel_start({}), 0)
        self.assertEqual(render._scene_channel_start({"frame_start": 30}), 30)
        self.assertEqual(render._scene_channel_start({"frame_start": -5}), 0)


def _mk_clip(path, w, h, nframes, audio=False):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i",
           "testsrc2=size={}x{}:rate={}:duration={}".format(
               w, h, FPS, nframes / FPS)]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=440:duration={}".format(nframes / FPS)]
    cmd += ["-frames:v", str(nframes), "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac"]
    cmd += [path]
    subprocess.check_call(cmd)


def _chan(file, wid, rect, t0, w, h, frame_start=None, frame_count=None):
    d = {"role": "screen_window", "file": file, "mode": "window_native",
         "id": wid, "app": "A", "title": "t", "units": "points",
         "rect": [float(v) for v in rect], "display_origin": [0.0, 0.0],
         "logical_w": float(rect[2]), "logical_h": float(rect[3]),
         "buffer_w": int(w), "buffer_h": int(h), "t0_monotonic": float(t0)}
    if frame_start is not None:
        d["frame_start"] = int(frame_start)
    if frame_count is not None:
        d["frame_count"] = int(frame_count)
    return d


def _mk_join_session(root, survivor_frame_start=30, ch0_audio=False,
                     events=()):
    """A 2-scene gapfree join: two survivor windows on continuous 60-frame
    files, a joiner that appears at the seam (T = frame 30) on its own
    30-frame file. Scene 0 = [0:30) of the survivors (2 cards); scene 1 =
    [30:60) of the survivors + [0:30) of the joiner (3 cards). One continuous
    2 s / 60-frame export. `ch0_audio` puts a mic track on the continuous
    channel-0 file (raw_0.mov) and marks the manifest mic_index.
    """
    os.makedirs(root, exist_ok=True)
    _mk_clip(os.path.join(root, "raw_0.mov"), 320, 180, 60,   # survivor A
             audio=ch0_audio)
    _mk_clip(os.path.join(root, "raw_1.mov"), 320, 180, 60)   # survivor B
    _mk_clip(os.path.join(root, "raw_2.mov"), 320, 180, 30)   # joiner
    T = 100.0 + 30 / FPS                                       # join instant
    meta = {
        "fps": FPS, "logical_w": 1440.0, "logical_h": 900.0,
        "geom_source": "quartz", "t0_monotonic": 100.0,
        "video_index": 1, "mic_index": 0 if ch0_audio else None,
        "events": "events.jsonl",
        "cursor_mode": "system", "key_capture": "activity",
        "face": None, "capture_backend": "sck",
        "capture_scenes": [
            {"index": 0, "t0_monotonic": 100.0, "wall_end_monotonic": T,
             "channels": [
                 _chan("raw_0.mov", 500, [0, 0, 320, 180], 100.0, 320, 180,
                       frame_start=0, frame_count=30),
                 _chan("raw_1.mov", 501, [400, 0, 320, 180], 100.0, 320, 180,
                       frame_start=0, frame_count=30)]},
            {"index": 1, "t0_monotonic": T,
             "channels": [
                 # survivors: SAME continuous files, second slice
                 _chan("raw_0.mov", 500, [0, 0, 320, 180], T, 320, 180,
                       frame_start=survivor_frame_start, frame_count=30),
                 _chan("raw_1.mov", 501, [400, 0, 320, 180], T, 320, 180,
                       frame_start=survivor_frame_start, frame_count=30),
                 # joiner: its OWN file from its start
                 _chan("raw_2.mov", 502, [0, 200, 320, 180], T, 320, 180,
                       frame_start=0, frame_count=30)]},
        ],
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return meta


def _probe_frames(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height",
        "-of", "default=noprint_wrappers=1", path]).decode()
    kv = dict(line.split("=", 1) for line in out.strip().splitlines())
    return int(kv["nb_read_frames"]), int(kv["width"]), int(kv["height"])


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class SceneJoinRender(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="scene_join_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_join_renders_one_continuous_video_of_both_slices(self):
        _mk_join_session(self.td)
        out = render.render(self.td, aspect="320x180")
        self.assertTrue(os.path.isfile(out))
        n, w, h = _probe_frames(out)
        # 30 (scene 0) + 30 (scene 1) = 60; the survivors' second slice does
        # NOT overrun, and the joiner's 30 frames line up.
        self.assertEqual(n, 60)

    def test_frame_start_actually_seeks_the_second_slice(self):
        # With the correct frame_start=30, scene 1 shows [30:60) of the
        # survivors. A broken frame_start=0 would re-show [0:30) -- different
        # pixels. If the seek were ignored the two renders would be identical.
        good_dir = os.path.join(self.td, "good")
        bad_dir = os.path.join(self.td, "bad")
        _mk_join_session(good_dir, survivor_frame_start=30)
        _mk_join_session(bad_dir, survivor_frame_start=0)
        good = render.render(good_dir, aspect="320x180")
        bad = render.render(bad_dir, aspect="320x180")
        with open(good, "rb") as f:
            a = f.read()
        with open(bad, "rb") as f:
            b = f.read()
        self.assertNotEqual(a, b, "frame_start was ignored -- the second "
                                  "slice re-read the head of the file")

    def _out_has_audio(self, path):
        out = subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            text=True).strip()
        return bool(out)

    def test_continuous_channel0_audio_is_muxed_across_the_seam(self):
        # The join's voiceover rides channel 0's continuous file -> one audio
        # track spanning both scenes, muxed into the export.
        _mk_join_session(self.td, ch0_audio=True)
        out = render.render(self.td, aspect="320x180")
        self.assertEqual(_probe_frames(out)[0], 60)
        self.assertTrue(self._out_has_audio(out),
                        "channel 0 carried a mic track across the seam; the "
                        "export must have audio")
        self.assertTrue(render.describe_session(self.td)["has_audio"])

    def test_micless_scene_take_stays_silent(self):
        # Off-switch: no mic -> the scene export is byte-for-byte the
        # video-only path (no audio stream).
        _mk_join_session(self.td, ch0_audio=False)
        out = render.render(self.td, aspect="320x180")
        self.assertFalse(self._out_has_audio(out))
        self.assertFalse(render.describe_session(self.td)["has_audio"])


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class JoinEntranceRender(unittest.TestCase):
    """The seamless-join card ENTRANCE (milestone 2): the new card grows in and
    the survivors reflow at the seam instead of a hard cut."""

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="join_entrance_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_entrance_morphs_the_seam_and_off_is_a_hard_cut(self):
        a_dir = os.path.join(self.td, "on")
        b_dir = os.path.join(self.td, "off")
        _mk_join_session(a_dir)
        _mk_join_session(b_dir)
        on = render.render(a_dir, aspect="320x180", join_entrance=True)
        off = render.render(b_dir, aspect="320x180", join_entrance=False)
        # Same length either way -- the entrance re-pixels the first frames of
        # scene 1, it never adds or drops a frame.
        self.assertEqual(_probe_frames(on)[0], _probe_frames(off)[0])
        # ...but the pixels of the transition differ: the morph happened.
        with open(on, "rb") as f:
            a = f.read()
        with open(off, "rb") as f:
            b = f.read()
        self.assertNotEqual(a, b, "join_entrance=True must animate the seam; "
                                  "False must reproduce the hard cut")


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class JoinEntrancePreview(unittest.TestCase):
    """The S3 preview contract on a JOIN take: the editor's paused scrub is
    the exported frame, INCLUDING the first ~0.8s card-entrance morph of a
    joined scene (docs/architecture.md). The entrance blends from the PREVIOUS
    scene's placements, so `scene_preview_frame` must see enough of the
    neighbouring scene -- its rects AND its manual layouts -- not just the
    scrubbed one."""

    # The join session is 60 frames @30fps; scene 1 starts at t=1.0 and its
    # entrance runs round(0.8 * 30) = 24 frames, i.e. t in [1.0, 1.8).
    T_INSIDE = 1.2     # scene-1 frame 6: mid-morph
    T_OUTSIDE = 1.9    # scene-1 frame 27: clear of the 24-frame morph

    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="join_entrance_pv_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _assert_preview_matches_export(self, t, **kw):
        out = render.render(self.td, out_path=os.path.join(self.td, "e.mp4"),
                            aspect="320x180", **kw)
        pv = render.scene_preview_frame(self.td, t_sec=t, aspect="320x180",
                                        **kw)
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", out,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        ex = np.frombuffer(raw, dtype=np.uint8)[:180 * 320 * 3].reshape(
            180, 320, 3)
        # H.264 (crf20) vs the raw composite: structurally identical, so the
        # mean abs diff is compression noise. A hard cut where the export
        # morphs (or a morph from the wrong cells) is a re-layout -- far
        # above this.
        diff = np.abs(pv.astype(np.int16) - ex.astype(np.int16)).mean()
        self.assertLess(diff, 8.0)

    def test_preview_matches_export_inside_the_entrance_window(self):
        _mk_join_session(self.td)
        self._assert_preview_matches_export(self.T_INSIDE)
        # ...and the frame really is a morph frame, not a hard cut that
        # happened to match: the entrance off-switch changes the preview.
        on = render.scene_preview_frame(self.td, t_sec=self.T_INSIDE,
                                        aspect="320x180", join_entrance=True)
        off = render.scene_preview_frame(self.td, t_sec=self.T_INSIDE,
                                         aspect="320x180", join_entrance=False)
        self.assertGreater(
            float(np.abs(on.astype(np.int16) - off.astype(np.int16)).mean()),
            1.0, "t=%.2f must be inside the entrance window" % self.T_INSIDE)

    def test_preview_matches_export_outside_the_entrance_window(self):
        _mk_join_session(self.td)
        self._assert_preview_matches_export(self.T_OUTSIDE)

    def test_entrance_blends_from_the_previous_scenes_manual_layout(self):
        # A hand-placed card in scene 0 moves the placements the entrance
        # morphs FROM. Export stamps scene 0's layouts before it reaches the
        # seam; the preview plans only the scrubbed scene, so without the
        # neighbour stamp it would morph from the preset cell instead.
        _mk_join_session(self.td)
        layouts = {"0": [{"x": 0.05, "y": 0.5, "w": 0.4, "h": 0.42}, None]}
        self._assert_preview_matches_export(self.T_INSIDE,
                                            scene_layouts=layouts)
