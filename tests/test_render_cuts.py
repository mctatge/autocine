"""Multi-range cuts ("ripple delete") -- end-to-end render pins.

The unit halves live next to the code they pin: TimeMap cut math in
test_retime.CutMapTests / QuantizeCutSpansTests, the hole-aware audio gate
in test_render_audio.CutsAudioGraph. This suite pins the RENDER:

  * the off switch (cuts absent / None / [] render byte-identical),
  * exact frame counts (quantized boundaries, non-aligned inputs),
  * A/V sync across a seam (PCM-RMS: a tone recorded at source [2,3]
    must land at output [1,2] after cutting [1,2]),
  * click SFX suppression for clicks inside a cut,
  * the camera-model consequences (cluster fusion across a seam, the
    trailing holdOut flip against the new clip end) -- intended behavior,
    pinned so a change is a decision rather than an accident,
  * dispatch breadth: segmented takes compose; scene / multi-native /
    multi-window-card takes refuse loudly and render un-cut.
"""

import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import camera, render, retime

from tests.test_capture_window import _mk_session


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _events(n=6):
    return [{"t": 0.4 + 0.3 * i, "type": "down", "x": 100 + 10 * i,
             "y": 80 + 5 * i} for i in range(n)]


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _frames(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", path],
        text=True).strip()
    return int(out)


def _duration(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", path], text=True).strip()
    return float(out)


def _mk_session_with_tone(root, events, duration=4, fps=30,
                          width=320, height=180):
    """Synthetic session whose mic track is silence except a 440-ish tone
    over source [2, 3] -- the marker the A/V-sync test tracks across a cut."""
    os.makedirs(root, exist_ok=True)
    raw = os.path.join(root, "raw.mov")
    subprocess.check_call([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "testsrc2=size={}x{}:rate={}:duration={}".format(
            width, height, fps, duration),
        "-f", "lavfi", "-i",
        "aevalsrc=if(between(t\\,2\\,3)\\,0.5*sin(880*PI*t)\\,0):d={}".format(
            duration),
        "-map", "0:v", "-map", "1:a",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", raw])
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    meta = {"raw": "raw.mov", "events": "events.jsonl", "fps": fps,
            "logical_w": width, "logical_h": height, "t0_monotonic": 0.0,
            "cursor_mode": "system", "duration": duration}
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class CutsOffSwitch(unittest.TestCase):
    """THE invariant: cuts sit on the path every export runs, so the off
    state is pinned on the bytes (same-process triple, the test_crop
    pattern -- never a stored golden, which drifts across ffmpeg builds)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sd = os.path.join(cls.tmp, "sess")
        _mk_session(cls.sd, _events(), duration=2, fps=30,
                    width=320, height=180)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, name, **kw):
        out = os.path.join(self.tmp, name)
        render.render(self.sd, out_path=out, motion_blur=False, facecam=False,
                      **kw)
        return out

    def test_off_is_byte_identical(self):
        a = self._render("off_a.mp4")
        b = self._render("off_b.mp4", cuts=None)
        c = self._render("off_c.mp4", cuts=[])
        self.assertEqual(_md5(a), _md5(b))
        self.assertEqual(_md5(a), _md5(c))


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class CutsEndToEnd(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_cut_removes_exactly_the_covered_frames(self):
        sd = os.path.join(self.td, "s")
        _mk_session(sd, _events(), duration=4, fps=30, width=320, height=180)
        out = os.path.join(self.td, "out.mp4")
        render.render(sd, out_path=out, motion_blur=False, facecam=False,
                      cuts=[{"start": 1.0, "end": 2.0}])
        self.assertEqual(_frames(out), 90)
        self.assertAlmostEqual(_duration(out), 3.0, delta=0.1)

    def test_non_frame_aligned_cut_snaps_to_the_frame_grid(self):
        sd = os.path.join(self.td, "s")
        _mk_session(sd, _events(), duration=4, fps=30, width=320, height=180)
        out = os.path.join(self.td, "out.mp4")
        # [0.51, 1.49] quantizes to [floor(15.3)/30, ceil(44.7)/30] =
        # [0.5, 1.5): exactly 30 frames removed.
        render.render(sd, out_path=out, motion_blur=False, facecam=False,
                      cuts=[{"start": 0.51, "end": 1.49}])
        self.assertEqual(_frames(out), 90)

    def test_multiple_and_overlapping_cuts_union(self):
        sd = os.path.join(self.td, "s")
        _mk_session(sd, _events(), duration=4, fps=30, width=320, height=180)
        out = os.path.join(self.td, "out.mp4")
        # [0.5,1.0] + [0.8,1.5] union to [0.5,1.5]; plus [3.5,4.0] tail cut
        render.render(sd, out_path=out, motion_blur=False, facecam=False,
                      cuts=[{"start": 0.5, "end": 1.0},
                            {"start": 0.8, "end": 1.5},
                            {"start": 3.5, "end": 4.0}])
        self.assertEqual(_frames(out), 75)

    def test_cuts_covering_everything_raise_a_cuts_naming_error(self):
        sd = os.path.join(self.td, "s")
        _mk_session(sd, _events(), duration=2, fps=30, width=320, height=180)
        out = os.path.join(self.td, "out.mp4")
        with self.assertRaises(RuntimeError) as ctx:
            render.render(sd, out_path=out, motion_blur=False, facecam=False,
                          cuts=[{"start": 0.0, "end": 2.0}])
        self.assertIn("cuts", str(ctx.exception))

    def test_audio_follows_the_video_across_the_seam(self):
        """PCM-RMS (the test_render_segmented technique): the tone recorded
        at source [2, 3] must land at output [1, 2] after cutting [1, 2],
        and the removed second must not leave a hole or an echo."""
        sd = os.path.join(self.td, "s")
        _mk_session_with_tone(sd, _events(), duration=4, fps=30)
        out = os.path.join(self.td, "out.mp4")
        # Event sounds off: they are ON by default and would put real
        # energy in the windows this test needs silent. `click_fx` is the
        # visual ripple; these are its audio siblings.
        render.render(sd, out_path=out, motion_blur=False, facecam=False,
                      click_fx=False, click_sound="off", key_sound="off",
                      cuts=[{"start": 1.0, "end": 2.0}])
        self.assertAlmostEqual(_duration(out), 3.0, delta=0.1)
        pcm = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", out, "-map", "a:0",
             "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
            stdout=subprocess.PIPE, check=True).stdout
        arr = np.frombuffer(pcm, np.int16).astype(np.float64)
        sr = 8000.0

        def rms(t_a, t_b):
            chunk = arr[int(t_a * sr):int(t_b * sr)]
            return float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0

        tone = rms(1.1, 1.9)
        self.assertGreater(tone, 500.0)
        self.assertLess(rms(0.1, 0.9), tone * 0.05)
        self.assertLess(rms(2.1, 2.9), tone * 0.05)

    def test_click_inside_a_cut_produces_no_click_sound(self):
        sd = os.path.join(self.td, "s")
        # one click kept (0.5), one inside the cut (1.5). Asserted on the
        # times that reach the SFX bed rather than on the decoded audio:
        # the contract is that a cut event never becomes a sound at all,
        # not merely that it is quiet near the seam.
        _mk_session_with_tone(
            sd, [{"t": 0.5, "type": "down", "x": 100, "y": 80},
                 {"t": 1.5, "type": "down", "x": 200, "y": 120}],
            duration=3, fps=30)
        click_wav = os.path.join(self.td, "click.wav")
        subprocess.check_call(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "sine=frequency=1000:duration=0.05",
             click_wav])
        out = os.path.join(self.td, "out.mp4")
        captured = {}
        orig = render._sfx_layers

        def spy(click_choice, key_choice, clicks, releases, keys):
            captured["clicks"] = list(clicks)
            return orig(click_choice, key_choice, clicks, releases, keys)

        render._sfx_layers = spy
        try:
            render.render(sd, out_path=out, motion_blur=False, facecam=False,
                          click_sound=click_wav,
                          cuts=[{"start": 1.0, "end": 2.0}])
        finally:
            render._sfx_layers = orig
        times = captured.get("clicks")
        self.assertEqual(len(times), 1)
        self.assertAlmostEqual(times[0], 0.5, delta=0.05)


class CutsCameraConsequences(unittest.TestCase):
    """Two INTENDED camera-model consequences of planning in output time,
    pinned so a future change is a decision, not an accident."""

    def test_cluster_fusion_across_a_removed_gap(self):
        """Cutting the 30s of dead air between two click groups can fuse
        them into ONE cluster (their output-time gap drops under
        chain_gap): the user cut dead air and the camera now treats both
        groups as one beat. Correct per the model -- and pinned."""
        P = camera.build_params(2.0, None)
        src = [(1.0, 100, 100), (1.5, 110, 100),
               (12.0, 500, 300), (12.5, 510, 300)]
        self.assertEqual(len(camera.cluster_clicks(src, P.chain_gap)), 2)
        tm = retime.TimeMap([], duration=20.0, cuts=[(3.0, 11.0)])
        warped = [(float(tm.warp(t)), x, y) for (t, x, y) in src]
        self.assertEqual(len(camera.cluster_clicks(warped, P.chain_gap)), 1)

    def test_trailing_cluster_flips_to_hold_out_against_new_end(self):
        """A tail cut can move a mid-clip cluster inside min_room of the
        NEW clip end, flipping its ease-out into holdOut -- the rendered
        camera legitimately differs from the source-time proposal."""
        P = camera.build_params(2.0, None)
        cluster = [(4.8, 100, 100), (5.2, 120, 100)]
        r_uncut = camera.cluster_to_range(cluster, P, 20.0)
        self.assertFalse(r_uncut["holdOut"])
        tm = retime.TimeMap([], duration=20.0, cuts=[(5.5, 20.0)])
        warped = [(float(tm.warp(t)), x, y) for (t, x, y) in cluster]
        r_cut = camera.cluster_to_range(warped, P, tm.output_duration)
        self.assertIsNotNone(r_cut)
        self.assertTrue(r_cut["holdOut"])


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class CutsDispatchBreadth(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_segmented_take_composes_with_cuts(self):
        """Cut coordinates are on the JOINED (gaps-deleted) timeline -- the
        one describe/editor show -- and act after the SegmentClock."""
        from tests.test_render_segmented import (_events as _seg_events,
                                                 _mk_segmented_session)
        _mk_segmented_session(self.td, _seg_events())
        out = os.path.join(self.td, "out.mp4")
        render.render(self.td, out_path=out, motion_blur=False,
                      click_fx=False, cuts=[{"start": 0.5, "end": 1.0}])
        # joined take is 2.0s @ 30fps = 60 frames; the cut removes 15
        self.assertEqual(_frames(out), 45)

    def test_multi_window_cards_refuse_cuts_loudly(self):
        sd = os.path.join(self.td, "s")
        _mk_session(sd, _events(), duration=2, fps=30, width=320, height=180)
        out = os.path.join(self.td, "out.mp4")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            render.render(sd, out_path=out, motion_blur=False, facecam=False,
                          windows=[{"x": 0, "y": 0, "w": 300, "h": 180}],
                          cuts=[{"start": 0.5, "end": 1.0}])
        self.assertIn("cuts are ignored in multi-window", buf.getvalue())
        # rendered UN-cut: the refusal never half-applies
        self.assertEqual(_frames(out), 60)

    def test_scene_take_refuses_cuts_loudly(self):
        from tests.test_render_scenes import _click, _mk_scene_session
        _mk_scene_session(self.td,
                          events=[_click(100.5, 100.0, 100.0),
                                  _click(110.7, 500.0, 100.0)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            out = render.render(self.td, cuts=[{"start": 0.2, "end": 0.6}])
        self.assertIn("cuts are not supported on scene takes", buf.getvalue())
        self.assertTrue(os.path.isfile(out))
        # 57 frames = the scene suite's own uncut pin for this session
        self.assertEqual(_frames(out), 57)


if __name__ == "__main__":
    unittest.main()
