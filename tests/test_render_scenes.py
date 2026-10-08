"""End-to-end (permissions-free) tests for the SCENE-take render + read
model (docs/architecture.md). Synthesizes a 2-scene session -- scene 0
one channel, scene 1 two channels with a real inter-channel offset -- via
testsrc2, then exercises `render()` dispatch, the single-encoder composite,
`describe_session`, and the three-surface clock agreement.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import render
from autocine import segments

FPS = 30


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _mk_channel_file(path, width, height, duration):
    subprocess.check_call([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "testsrc2=size={}x{}:rate={}:duration={}".format(
            width, height, FPS, duration),
        "-pix_fmt", "yuv420p", path])


def _chan_meta(file, wid, rect, t0, width, height,
               frame_start=None, frame_count=None):
    d = {
        "role": "screen_window", "file": file, "mode": "window_native",
        "id": wid, "app": "App", "title": "t", "units": "points",
        "rect": [float(v) for v in rect],
        "display_origin": [0.0, 0.0], "source": "quartz",
        "resnapshot": True,
        "end_rect": [float(v) for v in rect], "track": "ok",
        "logical_w": float(rect[2]), "logical_h": float(rect[3]),
        "buffer_w": int(width), "buffer_h": int(height),
        "t0_monotonic": float(t0),
    }
    if frame_start is not None:
        d["frame_start"] = int(frame_start)
    if frame_count is not None:
        d["frame_count"] = int(frame_count)
    return d


def _mk_scene_session(root, events=()):
    """2-scene session: scene 0 = one 320x180 channel at t0=100; scene 1 =
    two channels (320x180 at 110.0, 240x180 at 110.1 -- a real 3-frame
    offset, so the origin adjustment is exercised). 1s of video each."""
    os.makedirs(root, exist_ok=True)
    _mk_channel_file(os.path.join(root, "scene_0_raw_0.mov"), 320, 180, 1)
    _mk_channel_file(os.path.join(root, "scene_1_raw_0.mov"), 320, 180, 1)
    _mk_channel_file(os.path.join(root, "scene_1_raw_1.mov"), 240, 180, 1)
    meta = {
        "fps": FPS,
        "logical_w": 1440.0, "logical_h": 900.0, "geom_source": "quartz",
        "t0_monotonic": 100.0,
        "video_index": 1, "mic_index": None,
        "events": "events.jsonl", "cursor_mode": "system",
        "key_capture": "activity",
        "face": None, "face_index": None, "face_fps": None,
        "face_t0_monotonic": None, "face_capture": None,
        "capture_backend": "sck",
        "capture_scenes": [
            {"index": 0, "t0_monotonic": 100.0,
             "wall_end_monotonic": 101.05,
             "channels": [_chan_meta("scene_0_raw_0.mov", 500,
                                     [0, 0, 320, 180], 100.0, 320, 180)]},
            {"index": 1, "t0_monotonic": 110.0,
             "channels": [
                 _chan_meta("scene_1_raw_0.mov", 500,
                            [0, 0, 320, 180], 110.0, 320, 180),
                 _chan_meta("scene_1_raw_1.mov", 501,
                            [400, 0, 240, 180], 110.1, 240, 180)]},
        ],
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for line in events:
            f.write(json.dumps(line) + "\n")
    return meta


def _mk_join_scene_session(root, events=()):
    """2-scene gapfree JOIN session: the survivor stays on ONE continuous
    60-frame file whose scenes reference sub-ranges via frame_start /
    frame_count; a 30-frame joiner appears at the seam (T = 101.0). The
    healthy-take shape the describe read model must not warn about."""
    os.makedirs(root, exist_ok=True)
    _mk_channel_file(os.path.join(root, "raw_0.mov"), 320, 180, 2)
    _mk_channel_file(os.path.join(root, "raw_1.mov"), 240, 180, 1)
    T = 100.0 + 30.0 / FPS
    meta = {
        "fps": FPS,
        "logical_w": 1440.0, "logical_h": 900.0, "geom_source": "quartz",
        "t0_monotonic": 100.0,
        "video_index": 1, "mic_index": None,
        "events": "events.jsonl", "cursor_mode": "system",
        "key_capture": "activity",
        "face": None, "face_index": None, "face_fps": None,
        "face_t0_monotonic": None, "face_capture": None,
        "capture_backend": "sck",
        "capture_scenes": [
            {"index": 0, "t0_monotonic": 100.0, "wall_end_monotonic": T,
             "channels": [_chan_meta("raw_0.mov", 500,
                                     [0, 0, 320, 180], 100.0, 320, 180,
                                     frame_start=0, frame_count=30)]},
            {"index": 1, "t0_monotonic": T,
             "channels": [
                 _chan_meta("raw_0.mov", 500,
                            [0, 0, 320, 180], T, 320, 180,
                            frame_start=30, frame_count=30),
                 _chan_meta("raw_1.mov", 501,
                            [400, 0, 240, 180], T, 240, 180,
                            frame_start=0, frame_count=30)]},
        ],
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for line in events:
            f.write(json.dumps(line) + "\n")
    return meta


def _click(t, x, y):
    return {"t": t, "type": "down", "x": x, "y": y, "button": "Button.left"}


def _probe_frames(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,width,height",
        "-of", "default=noprint_wrappers=1", path]).decode()
    kv = dict(line.split("=", 1) for line in out.strip().splitlines())
    return (int(kv["nb_read_frames"]), int(kv["width"]), int(kv["height"]))


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class SceneRenderEndToEnd(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.meta = _mk_scene_session(
            self.td,
            events=[_click(100.5, 100.0, 100.0),
                    _click(110.7, 500.0, 100.0)])

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_render_dispatches_and_writes_one_continuous_video(self):
        out = render.render(self.td, aspect="320x180")
        self.assertTrue(os.path.isfile(out))
        w, width, height = _probe_frames(out)
        # Scene 0 contributes 30 frames; scene 1's channel 1 starts 3 frames
        # late (0.1s at 30fps), so its overlap is 27 -- one video of 57.
        self.assertEqual(w, 57)
        self.assertEqual((width, height), (320, 180))

    def test_per_scene_zoom_and_focus_compose(self):
        base = render.render(self.td, out_path=os.path.join(self.td, "a.mp4"),
                             aspect="320x180")
        zoomed = render.render(self.td,
                               out_path=os.path.join(self.td, "b.mp4"),
                               aspect="320x180",
                               window_zoom=True, window_focus=True)
        with open(base, "rb") as f:
            a = f.read()
        with open(zoomed, "rb") as f:
            b = f.read()
        # The clicks drive per-scene cameras -- the outputs must differ; the
        # off state is the byte-stable baseline.
        self.assertNotEqual(a, b)
        self.assertEqual(_probe_frames(zoomed)[0], 57)

    def test_v1_cut_flags_do_not_crash_and_print_a_note(self):
        out = render.render(self.td, aspect="320x180", speedup=True,
                            fade=0.5, make_gif=True)
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(os.path.isfile(
            os.path.join(self.td, "output.gif")))


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class ScenePreviewFrame(unittest.TestCase):
    """`scene_preview_frame` is the editor's paused scrub. It must (a) return a
    composited frame at the composite canvas size instead of raising on the
    missing raw.mov, (b) select the right scene either side of the seam, and
    (c) match the exported frame at the same output time -- proof it reuses the
    export compositor and maps time->(scene, frame) the same way."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _mk_scene_session(
            self.td,
            events=[_click(100.5, 100.0, 100.0),
                    _click(110.7, 500.0, 100.0)])

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_returns_a_composited_frame_not_a_raise(self):
        # scene 0 spans 0-1s, scene 1 spans 1-1.9s (57 frames @30 -> 1.9s).
        fr = render.scene_preview_frame(self.td, t_sec=0.3, aspect="320x180")
        self.assertEqual(fr.shape, (180, 320, 3))
        # not an all-black frame (testsrc2 is vivid)
        self.assertGreater(float((fr.reshape(-1, 3).max(axis=1) > 16).mean()),
                           0.5)

    def test_scene_selection_across_the_seam(self):
        # Scene 0 (one card) and scene 1 (two cards, different layout) composite
        # to visibly different frames. A time->scene mapping that picked the
        # wrong scene would make a cross-seam pair look like a within-scene one,
        # so we require the cross-seam change to dwarf the within-scene one.
        def frm(t):
            return render.scene_preview_frame(
                self.td, t_sec=t, aspect="320x180").astype(np.int16)
        def diff(a, b):
            return float(np.abs(a - b).mean())
        within_scene1 = diff(frm(1.3), frm(1.6))   # both scene 1
        across_seam = diff(frm(0.6), frm(1.3))     # scene 0 vs scene 1
        self.assertGreater(across_seam, within_scene1 * 1.5)

    def test_matches_the_exported_frame(self):
        out = render.render(self.td, out_path=os.path.join(self.td, "e.mp4"),
                            aspect="320x180", window_zoom=True)
        t = 1.4                            # mid scene 1
        pv = render.scene_preview_frame(self.td, t_sec=t, aspect="320x180",
                                        window_zoom=True)
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", out,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        ex = np.frombuffer(raw, dtype=np.uint8)[:180 * 320 * 3].reshape(
            180, 320, 3)
        # H.264 (crf20) vs the raw composite: structurally identical, so the
        # mean abs diff is compression noise, not a mis-mapped frame.
        diff = np.abs(pv.astype(np.int16) - ex.astype(np.int16)).mean()
        self.assertLess(diff, 8.0)


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class SceneResolutionCap(unittest.TestCase):
    """The export-resolution control (max_height) on the scene path. Off must
    be bit-exact; on must shrink the canvas, aspect-locked."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _mk_scene_session(self.td, events=[_click(100.5, 100.0, 100.0)])

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_off_is_bit_exact(self):
        a = render.render(self.td, out_path=os.path.join(self.td, "a.mp4"))
        b = render.render(self.td, out_path=os.path.join(self.td, "b.mp4"),
                          max_height=None)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            self.assertEqual(fa.read(), fb.read())

    def test_cap_shrinks_the_canvas_aspect_locked(self):
        natural = _probe_frames(render.render(
            self.td, out_path=os.path.join(self.td, "n.mp4")))
        capped = _probe_frames(render.render(
            self.td, out_path=os.path.join(self.td, "c.mp4"), max_height=120))
        self.assertEqual(capped[2], 120)                 # height capped
        self.assertLess(capped[1], natural[1])           # width shrank too
        self.assertAlmostEqual(capped[1] / float(capped[2]),
                               natural[1] / float(natural[2]), delta=0.02)
        self.assertEqual(capped[0], natural[0])          # same frame count

    def test_preview_matches_capped_export(self):
        out = render.render(self.td, out_path=os.path.join(self.td, "e.mp4"),
                            max_height=120)
        pv = render.scene_preview_frame(self.td, t_sec=0.3, max_height=120)
        self.assertEqual(pv.shape[0], 120)
        self.assertEqual((pv.shape[1], pv.shape[0]),
                         (_probe_frames(out)[1], _probe_frames(out)[2]))


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class SceneDescribeAndClockAgreement(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()
        _mk_scene_session(
            self.td,
            events=[_click(100.5, 100.0, 100.0),
                    _click(110.7, 500.0, 100.0)])

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_describe_reports_the_scene_shape(self):
        info = render.describe_session(self.td)
        self.assertTrue(info["scene_take"])
        self.assertIsNone(info["raw_path"])
        self.assertEqual(len(info["capture_scenes"]), 2)
        s0, s1 = info["capture_scenes"]
        self.assertEqual(s0["frame_count"], 30)
        self.assertEqual(s1["frame_count"], 27)
        # Scene 1's t0 is ORIGIN-ADJUSTED: channel 1 starts 3 frames after
        # channel 0, so composite frame 0 sits at 110.0 + 3/30.
        self.assertAlmostEqual(s1["t0_monotonic"], 110.0 + 3.0 / FPS)
        self.assertAlmostEqual(info["duration"], (30 + 27) / float(FPS))
        self.assertEqual(info["frame_count"], 57)
        self.assertFalse(info["has_audio"])

    def test_click_times_map_through_the_scene_clock(self):
        info = render.describe_session(self.td, include_click_times=True)
        times = info["click_times"]
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[0], 0.5, places=3)
        # Scene 1 starts at output 1.0; the click at parent 110.7 is 0.6s
        # after the adjusted scene start (110.1).
        self.assertAlmostEqual(times[1], 1.0 + 0.6, places=3)

    def test_three_surface_clock_agreement(self):
        # The beats-side rebuild (info t0/frame_count) and the render-side
        # builder (from_scene_meta over decoded counts) must produce the
        # SAME map -- the trailing-cluster contract across surfaces.
        with open(os.path.join(self.td, "meta.json")) as f:
            meta = json.load(f)
        info = render.describe_session(self.td)
        t0s = [s["t0_monotonic"] for s in info["capture_scenes"]]
        durs = [s["frame_count"] / float(FPS)
                for s in info["capture_scenes"]]
        beats_clock = segments.SegmentClock(t0s, durs)
        render_clock, _aligns, _w = segments.SegmentClock.from_scene_meta(
            meta, [[30], [30, 30]], FPS)
        probe = np.array([99.0, 100.5, 105.0, 110.7, 111.05, 130.0])
        np.testing.assert_allclose(beats_clock.media(probe),
                                   render_clock.media(probe))
        np.testing.assert_array_equal(beats_clock.owner(probe),
                                      render_clock.owner(probe))

    def test_session_beats_runs_with_the_scene_clock(self):
        info = render.describe_session(self.td)
        sheet = render.session_beats(self.td, info=info)
        clicks = [b for b in sheet["beats"] if b.get("kind") == "clicks"]
        self.assertTrue(clicks)
        starts = sorted(b["start"] for b in clicks)
        self.assertAlmostEqual(starts[0], 0.5, places=2)

    def test_join_take_describe_uses_sub_ranges_and_stays_warning_free(self):
        # A healthy gapfree JOIN keeps a survivor on ONE continuous file
        # whose scenes reference sub-ranges (frame_start/frame_count).
        # Describe must feed the SLICE counts into the clock -- full-file
        # counts made scene 0 look 2s long, firing a spurious per-seam
        # overlap warning and reporting frame_count 60 on every healthy
        # join take (docs/architecture.md M3.0a).
        td = tempfile.mkdtemp()
        try:
            _mk_join_scene_session(td)
            info = render.describe_session(td)
            self.assertEqual(info["scene_warnings"], [])
            s0, s1 = info["capture_scenes"]
            self.assertEqual(s0["frame_count"], 30)
            self.assertEqual(s1["frame_count"], 30)
            # Per-channel counts are the slice, not the 60-frame file; the
            # sub-range start is surfaced so read-model consumers can place
            # the slice in the file.
            self.assertEqual(s0["channels"][0]["frame_count"], 30)
            self.assertEqual(s0["channels"][0]["frame_start"], 0)
            self.assertEqual(s1["channels"][0]["frame_count"], 30)
            self.assertEqual(s1["channels"][0]["frame_start"], 30)
            self.assertEqual(s1["channels"][1]["frame_count"], 30)
            self.assertEqual(s1["channels"][1]["frame_start"], 0)
            self.assertAlmostEqual(info["duration"], 60 / float(FPS))
            self.assertEqual(info["frame_count"], 60)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_join_take_clock_agrees_with_the_render_builder(self):
        # Same three-surface agreement as the per-scene fixture, on the
        # join shape: the beats rebuild (describe t0/frame_count) and the
        # render builder (from_scene_meta over SLICE counts) produce the
        # same map, and neither emits a warning.
        td = tempfile.mkdtemp()
        try:
            meta = _mk_join_scene_session(td)
            info = render.describe_session(td)
            t0s = [s["t0_monotonic"] for s in info["capture_scenes"]]
            durs = [s["frame_count"] / float(FPS)
                    for s in info["capture_scenes"]]
            beats_clock = segments.SegmentClock(t0s, durs)
            render_clock, _aligns, warn = segments.SegmentClock.from_scene_meta(
                meta, [[30], [30, 30]], FPS)
            self.assertEqual(warn, [])
            probe = np.array([99.0, 100.5, 100.99, 101.0, 101.5, 130.0])
            np.testing.assert_allclose(beats_clock.media(probe),
                                       render_clock.media(probe))
            np.testing.assert_array_equal(beats_clock.owner(probe),
                                          render_clock.owner(probe))
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_scene_shortfall_warning_surfaces_in_describe(self):
        # Scene 0's manifest records wall_end 101.05 but only 1.0s decoded --
        # right at the threshold boundary; widen it to make the warning fire.
        with open(os.path.join(self.td, "meta.json")) as f:
            meta = json.load(f)
        meta["capture_scenes"][0]["wall_end_monotonic"] = 102.0
        with open(os.path.join(self.td, "meta.json"), "w") as f:
            json.dump(meta, f)
        info = render.describe_session(self.td)
        self.assertTrue(any("clamp to the seam" in w
                            for w in info["scene_warnings"]))


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class SceneCanvasRule(unittest.TestCase):
    def test_shared_canvas_is_the_max_need_scene(self):
        td = tempfile.mkdtemp()
        try:
            _mk_scene_session(td)
            with open(os.path.join(td, "meta.json")) as f:
                meta = json.load(f)
            per_scene = []
            for s in meta["capture_scenes"]:
                rects = [{"x": c["rect"][0], "y": c["rect"][1],
                          "w": c["rect"][2], "h": c["rect"][3]}
                         for c in s["channels"]]
                dims = [(c["buffer_w"], c["buffer_h"])
                        for c in s["channels"]]
                per_scene.append(render._multi_native_canvas(
                    rects, dims, "clean", None, "grid"))
            expected = max(per_scene, key=lambda wh: wh[0] * wh[1])
            out = render.render(td)
            _n, width, height = _probe_frames(out)
            self.assertEqual((width, height), expected)
        finally:
            shutil.rmtree(td, ignore_errors=True)


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class SceneCameraPath(unittest.TestCase):
    """`render.scene_camera_path` is the editor live-player's per-scene plan
    (the scene-aware analog of the three `multi_native_*` emitters). It must
    describe every scene's canvas/cells/channels + the sampled per-card zoom /
    focus tracks, on the SAME shared canvas + clock the export uses, so the
    live composite matches export and the server still."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _mk_scene_session(
            self.td,
            events=[_click(100.5, 100.0, 100.0),
                    _click(110.7, 500.0, 100.0)])

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_shape_seams_channels_and_json_safe(self):
        p = render.scene_camera_path(self.td, aspect="320x180")
        # Two scenes; scene 0 = 30 frames, scene 1's late channel clips it to
        # 27, matching the 57-frame export the end-to-end test pins.
        self.assertEqual(p["fps"], 30.0)
        self.assertEqual(p["total_frames"], 57)
        self.assertEqual(p["seams"], [0, 30])
        self.assertAlmostEqual(p["duration"], 57 / 30.0, places=6)
        self.assertEqual([s["index"] for s in p["scenes"]], [0, 1])
        self.assertEqual([(s["frame_start"], s["frame_count"])
                          for s in p["scenes"]], [(0, 30), (30, 27)])
        # Top-level canvas/cells stay ABSENT (nested under scenes[]) so the
        # editor's single-scene multiReady() gate is never armed by mistake.
        self.assertNotIn("cells", p)
        self.assertNotIn("canvas", p)
        s0, s1 = p["scenes"]
        # Shared max-need canvas -- both scenes paint at the same size.
        self.assertEqual(s0["canvas"], s1["canvas"])
        self.assertEqual(s0["canvas"], [320, 180])
        # Per-scene channel wiring: media kind, buffer dims, and the alignment
        # `start` (scene 1's ch1 is the 3-frame-late origin, so ch0 skips 0.1s).
        self.assertEqual([c["media"] for c in s0["channels"]],
                         ["scene0channel0"])
        self.assertEqual([c["media"] for c in s1["channels"]],
                         ["scene1channel0", "scene1channel1"])
        self.assertEqual([(c["buffer_w"], c["buffer_h"])
                          for c in s1["channels"]], [(320, 180), (240, 180)])
        self.assertAlmostEqual(s1["channels"][0]["start"], 0.1, places=4)
        self.assertAlmostEqual(s1["channels"][1]["start"], 0.0, places=4)
        self.assertAlmostEqual(s0["channels"][0]["start"], 0.0, places=4)
        # Every cell samples its WHOLE channel buffer.
        for s in p["scenes"]:
            for c, ch in zip(s["cells"], s["channels"]):
                self.assertEqual(c["src"], [0, 0, ch["buffer_w"],
                                            ch["buffer_h"]])
        # The whole payload is JSON-serializable (no numpy leaking through).
        json.dumps(p)

    def test_card_paths_and_focus_gate_on_the_toggles(self):
        off = render.scene_camera_path(self.td, aspect="320x180")
        for s in off["scenes"]:
            self.assertIsNone(s["card_paths"])
            self.assertIsNone(s["focus_cells"])
            self.assertIsNone(s["bg_jpeg_base64"])

        zoom = render.scene_camera_path(self.td, aspect="320x180",
                                        window_zoom=True)
        # Each scene has a click, so per-card zoom fires; the wire shape mirrors
        # multi_native_card_paths exactly.
        self.assertTrue(any(s["card_paths"] for s in zoom["scenes"]))
        cp = next(s["card_paths"] for s in zoom["scenes"] if s["card_paths"])
        self.assertEqual(set(cp), {"stride", "fps", "cards"})

        foc = render.scene_camera_path(self.td, aspect="320x180",
                                       window_focus=True)
        self.assertTrue(any(s["focus_cells"] for s in foc["scenes"]))
        fs = next(s for s in foc["scenes"] if s["focus_cells"])
        self.assertEqual(set(fs["focus_cells"]), {"stride", "fps", "frames"})
        # Moving focus cells => the bare backdrop is shipped for that scene.
        self.assertIsNotNone(fs["bg_jpeg_base64"])

    def test_seams_match_the_describe_frame_counts(self):
        # Three-surface agreement: the player's seams are the prefix sums of
        # the SAME per-scene frame_count describe emits (one clock builder).
        p = render.scene_camera_path(self.td, aspect="320x180")
        info = render.describe_session(self.td)
        counts = [s["frame_count"] for s in info["capture_scenes"]]
        prefix, acc = [], 0
        for c in counts:
            prefix.append(acc)
            acc += c
        self.assertEqual(p["seams"], prefix)
        self.assertEqual(p["total_frames"], acc)


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg")
class SceneManualCardLayout(unittest.TestCase):
    """`scene_layouts` — per-scene hand-placed cards (docs/architecture.md P2).
    Off-switch bit-exact; a per-scene override moves that scene's card without
    changing resolution or touching sibling scenes; export == paused-still."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        _mk_scene_session(
            self.td, events=[_click(100.5, 100.0, 100.0),
                             _click(110.7, 500.0, 100.0)])
        self.lay = {"1": [None, {"x": 0.5, "y": 0.5, "w": 0.3, "h": 0.3}]}

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _render(self, name, **kw):
        return render.render(self.td, out_path=os.path.join(self.td, name),
                             aspect="320x180", **kw)

    def test_empty_is_byte_identical_and_override_moves_the_card(self):
        base = self._render("base.mp4")
        off = self._render("off.mp4", scene_layouts={})
        ov = self._render("ov.mp4", scene_layouts=self.lay)
        with open(base, "rb") as f:
            b = f.read()
        with open(off, "rb") as f:
            o = f.read()
        with open(ov, "rb") as f:
            v = f.read()
        self.assertEqual(b, o)                       # off-switch
        self.assertNotEqual(b, v)                     # scene 1 card moved
        self.assertEqual(_probe_frames(ov), _probe_frames(base))   # frozen res

    def test_scene0_frame_is_untouched_by_a_scene1_override(self):
        # A scene-1-only override must not reflow scene 0 (per-scene isolation
        # + the shared-canvas freeze).
        a = render.scene_preview_frame(self.td, t_sec=0.3, aspect="320x180")
        b = render.scene_preview_frame(self.td, t_sec=0.3, aspect="320x180",
                                       scene_layouts=self.lay)
        self.assertTrue(np.array_equal(a, b))
        # ...but scene 1 DOES change
        c = render.scene_preview_frame(self.td, t_sec=1.4, aspect="320x180")
        e = render.scene_preview_frame(self.td, t_sec=1.4, aspect="320x180",
                                       scene_layouts=self.lay)
        self.assertGreater(float(np.abs(c.astype(np.int16)
                                        - e.astype(np.int16)).mean()), 1.0)

    def test_preview_override_matches_the_exported_frame(self):
        # The S3 single-source guarantee holds WITH an override: the paused
        # still equals the export at the same output frame.
        out = self._render("e.mp4", scene_layouts=self.lay)
        t = 1.4
        pv = render.scene_preview_frame(self.td, t_sec=t, aspect="320x180",
                                        scene_layouts=self.lay)
        raw = subprocess.check_output([
            "ffmpeg", "-v", "error", "-ss", str(t), "-i", out,
            "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        ex = np.frombuffer(raw, dtype=np.uint8)[:180 * 320 * 3].reshape(
            180, 320, 3)
        diff = np.abs(pv.astype(np.int16) - ex.astype(np.int16)).mean()
        self.assertLess(diff, 8.0)


if __name__ == "__main__":
    unittest.main()
