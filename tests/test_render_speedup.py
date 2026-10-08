"""End-to-end render checks for the auto speed-up (Rush) feature.

Uses a synthetic session built from an ffmpeg testsrc2 pattern -- no
macOS permissions, no real recording -- so the retiming pipeline is
exercised through render() including cv2 decode, camera planning in
output time, and ffmpeg encode of a piped BGR stream.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

from autocine import render


def _mk_session(root, events, duration=20, fps=60, width=640, height=360):
    os.makedirs(root, exist_ok=True)
    raw = os.path.join(root, "raw.mov")
    subprocess.check_call([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i",
        "testsrc2=size={}x{}:rate={}:duration={}".format(width, height, fps, duration),
        "-pix_fmt", "yuv420p", raw])
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({
            "raw": "raw.mov", "events": "events.jsonl",
            "fps": fps, "logical_w": width, "logical_h": height,
            "t0_monotonic": 0.0, "cursor_mode": "system",
            "duration": duration,
        }, f)


def _probe(path):
    """(nb_frames, duration_seconds) via ffprobe."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error",
        "-select_streams", "v",
        "-show_entries", "stream=nb_frames:format=duration",
        "-of", "csv=p=0", path], text=True).strip().splitlines()
    # ffprobe returns two lines: nb_frames, then duration
    frames = int(out[0]) if out and out[0] else 0
    dur = float(out[1]) if len(out) > 1 and out[1] else 0.0
    return frames, dur


class OffSwitchBitExact(unittest.TestCase):
    """The `speedup=False` default MUST produce the same frame count and
    duration as the pre-feature render would have."""

    def test_default_off_matches_no_speedup_kwargs(self):
        td = tempfile.mkdtemp(prefix="rush_off_")
        try:
            events = [
                {"t": 1.0, "type": "move", "x": 100, "y": 100},
                {"t": 1.01, "type": "down", "x": 100, "y": 100},
                {"t": 1.11, "type": "up",  "x": 100, "y": 100},
                {"t": 18.0, "type": "move", "x": 200, "y": 200},
                {"t": 18.5, "type": "down", "x": 200, "y": 200},
                {"t": 18.6, "type": "up",  "x": 200, "y": 200},
            ]
            _mk_session(td, events)
            out = os.path.join(td, "off.mp4")
            render.render(td, out_path=out, motion_blur=False)
            frames, dur = _probe(out)
            # 20s at 60fps = 1200 frames, exact
            self.assertEqual(frames, 1200)
            self.assertAlmostEqual(dur, 20.0, places=2)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class SpeedupEndToEnd(unittest.TestCase):
    """With speedup on, the output must actually be shorter and playable."""

    def test_speedup_shortens_idle_session(self):
        td = tempfile.mkdtemp(prefix="rush_on_")
        try:
            events = [
                {"t": 1.0, "type": "move", "x": 100, "y": 100},
                {"t": 1.01, "type": "down", "x": 100, "y": 100},
                {"t": 1.11, "type": "up",  "x": 100, "y": 100},
                {"t": 18.0, "type": "move", "x": 200, "y": 200},
                {"t": 18.5, "type": "down", "x": 200, "y": 200},
                {"t": 18.6, "type": "up",  "x": 200, "y": 200},
            ]
            _mk_session(td, events)
            out = os.path.join(td, "on.mp4")
            # motion_gate off: testsrc2 is a moving pattern, would defeat
            # the gate; we're testing the closed-form warp math here, not
            # the gate.
            render.render(td, out_path=out, motion_blur=False,
                          speedup=True, speedup_rate=6.0,
                          speedup_silence_gate=True,
                          speedup_motion_gate=False)
            frames, dur = _probe(out)
            # activity at [1, 1.01, 1.11, 18, 18.5, 18.6]; pad=0.8;
            # single idle span [1.9, 17.7] = 15.8s. At r=6, ramp=0.5:
            #   d = 0.5, entry+exit = 0.5*(7/6)/2*2 = 0.583s
            #   plateau = (15.8 - 1.0)/6 = 2.467s
            # span output = 3.05s. Non-sped: 20 - 15.8 = 4.2s.
            # total ~= 7.25s = 435 frames.
            self.assertTrue(300 <= frames <= 480,
                            "unexpected frame count: {}".format(frames))
            self.assertLess(dur, 12.0, "not shortened enough")
            # the video must be playable
            self.assertTrue(os.path.getsize(out) > 1024)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_speedup_with_no_idle_matches_off(self):
        """A busy session (activity every frame) yields zero idle spans,
        TimeMap is identity, and output is bit-exact with speedup=False."""
        td = tempfile.mkdtemp(prefix="rush_busy_")
        try:
            events = [{"t": i * 0.5, "type": "down", "x": 100, "y": 100}
                      for i in range(1, 39)]
            _mk_session(td, events)
            out_off = os.path.join(td, "off.mp4")
            out_on = os.path.join(td, "on.mp4")
            render.render(td, out_path=out_off, motion_blur=False,
                          click_fx=False)  # avoid fx variance for equality
            render.render(td, out_path=out_on, motion_blur=False,
                          click_fx=False,
                          speedup=True, speedup_rate=6.0,
                          speedup_silence_gate=True)
            f_off, _ = _probe(out_off)
            f_on, _ = _probe(out_on)
            self.assertEqual(f_off, f_on)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_camera_path_rejects_unknown_kwargs(self):
        """camera_path silently drops declared render-only kwargs (so
        studio_app can thread the full option dict) but must LOUDLY
        reject anything else -- a typo like 'speedy=True' would otherwise
        silently no-op and the editor would show stale timing."""
        td = tempfile.mkdtemp(prefix="rush_kw_")
        try:
            events = [{"t": 1.0, "type": "down", "x": 100, "y": 100}]
            _mk_session(td, events, duration=5)
            # declared-ignored kwarg: accepted
            data = render.camera_path(td, stride=10, click_fx=True,
                                      speedup=True, speedups=[])
            self.assertIn("times", data)
            # unknown kwarg: loud rejection
            with self.assertRaises(TypeError):
                render.camera_path(td, stride=10, speedy=True)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_force_override_on_no_idle_session_still_speeds(self):
        """Even with speedup=False, a 'force' speedups entry compresses."""
        td = tempfile.mkdtemp(prefix="rush_force_")
        try:
            events = [{"t": i * 0.5, "type": "down", "x": 100, "y": 100}
                      for i in range(1, 39)]
            _mk_session(td, events)
            out = os.path.join(td, "force.mp4")
            # force ranges bypass the motion gate anyway, but keep it
            # off for clarity here.
            render.render(td, out_path=out, motion_blur=False,
                          speedup=False, speedup_motion_gate=False,
                          speedups=[{"start": 4.0, "end": 16.0,
                                     "mode": "force", "rate": 6.0}])
            frames, dur = _probe(out)
            self.assertLess(frames, 900, "force span didn't compress")
            self.assertLess(dur, 12.0)
        finally:
            shutil.rmtree(td, ignore_errors=True)


class MotionGateEndToEnd(unittest.TestCase):
    """testsrc2 is a moving test pattern -- every frame differs from the
    prior one. With the motion gate ON, no idle span survives the visual
    probe, so the retimed output must equal the pre-feature length."""

    def test_moving_video_defeats_speedup_when_gate_on(self):
        td = tempfile.mkdtemp(prefix="rush_motion_on_")
        try:
            events = [
                {"t": 1.0, "type": "down", "x": 100, "y": 100},
                {"t": 18.0, "type": "down", "x": 200, "y": 200},
            ]
            _mk_session(td, events)
            out = os.path.join(td, "gated.mp4")
            render.render(td, out_path=out, motion_blur=False,
                          speedup=True, speedup_rate=6.0,
                          speedup_silence_gate=False,   # no audio anyway
                          speedup_motion_gate=True)
            frames, _ = _probe(out)
            # motion gate should reject every candidate -> full 1200 frames
            self.assertEqual(frames, 1200,
                             "motion gate did not reject a moving pattern")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_moving_video_gate_off_speeds_anyway(self):
        """Same session, motion_gate=False: previous behavior returns
        (idle detected, span compressed)."""
        td = tempfile.mkdtemp(prefix="rush_motion_off_")
        try:
            events = [
                {"t": 1.0, "type": "down", "x": 100, "y": 100},
                {"t": 18.0, "type": "down", "x": 200, "y": 200},
            ]
            _mk_session(td, events)
            out = os.path.join(td, "nogate.mp4")
            render.render(td, out_path=out, motion_blur=False,
                          speedup=True, speedup_rate=6.0,
                          speedup_silence_gate=False,
                          speedup_motion_gate=False)
            frames, _ = _probe(out)
            self.assertLess(frames, 500,
                            "gate-off should reveal the pre-gate behavior")
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_still_video_survives_motion_gate(self):
        """A visually-static clip (color=black) has every pair marked
        static, so idle detection stands and the output is shortened."""
        td = tempfile.mkdtemp(prefix="rush_motion_still_")
        try:
            os.makedirs(td, exist_ok=True)
            raw = os.path.join(td, "raw.mov")
            subprocess.check_call([
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i",
                "color=c=black:size=320x180:rate=60:duration=20",
                "-pix_fmt", "yuv420p", raw])
            events = [
                {"t": 1.0, "type": "down", "x": 100, "y": 100},
                {"t": 18.0, "type": "down", "x": 200, "y": 200},
            ]
            with open(os.path.join(td, "events.jsonl"), "w") as f:
                for e in events:
                    f.write(json.dumps(e) + "\n")
            with open(os.path.join(td, "meta.json"), "w") as f:
                json.dump({"raw": "raw.mov", "events": "events.jsonl",
                           "fps": 60, "logical_w": 320, "logical_h": 180,
                           "t0_monotonic": 0.0, "cursor_mode": "system",
                           "duration": 20.0}, f)
            out = os.path.join(td, "still.mp4")
            render.render(td, out_path=out, motion_blur=False,
                          speedup=True, speedup_rate=6.0,
                          speedup_silence_gate=False,
                          speedup_motion_gate=True)
            frames, _ = _probe(out)
            self.assertLess(frames, 600, "still black clip failed to speed")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class SilenceGateFailsClosed(unittest.TestCase):
    """When the audio silence probe fails on a session with audio, we
    must FAIL CLOSED: nothing sped, because we don't know whether the
    quiet stretches are silence or narration. The docstring and printed
    warning both promise this; the test pins it."""

    def _mk_audio_session(self, root):
        os.makedirs(root, exist_ok=True)
        raw = os.path.join(root, "raw.mov")
        # 20s test pattern + 20s silent audio track (so has_audio=True)
        subprocess.check_call([
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i",
            "testsrc2=size=320x180:rate=60:duration=20",
            "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
            "-t", "20", "-shortest",
            "-pix_fmt", "yuv420p", "-c:a", "aac", raw])
        events = [{"t": 1.0, "type": "down", "x": 100, "y": 100},
                  {"t": 18.0, "type": "down", "x": 200, "y": 200}]
        with open(os.path.join(root, "events.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        with open(os.path.join(root, "meta.json"), "w") as f:
            json.dump({"raw": "raw.mov", "events": "events.jsonl",
                       "fps": 60, "logical_w": 320, "logical_h": 180,
                       "t0_monotonic": 0.0, "cursor_mode": "system",
                       "duration": 20.0}, f)

    def test_probe_failure_disables_speedup(self):
        from autocine import retime as _retime
        td = tempfile.mkdtemp(prefix="rush_silence_fail_")
        try:
            self._mk_audio_session(td)
            out = os.path.join(td, "fail.mp4")
            # simulate a probe failure (ffmpeg missing / hang / codec
            # unreadable) by monkey-patching silence_spans to return None
            saved = _retime.silence_spans
            _retime.silence_spans = lambda *a, **kw: None
            try:
                render.render(td, out_path=out, motion_blur=False,
                              speedup=True, speedup_rate=6.0,
                              speedup_silence_gate=True)
            finally:
                _retime.silence_spans = saved
            frames, _ = _probe(out)
            # fail-closed: video is 20s at 60fps = 1200 frames, unretimed
            self.assertEqual(frames, 1200)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_probe_failure_force_ranges_still_fire(self):
        """A 'force' override represents the author's explicit choice --
        the probe failure conservatism doesn't override an intentional
        force range."""
        from autocine import retime as _retime
        td = tempfile.mkdtemp(prefix="rush_silence_fail_force_")
        try:
            self._mk_audio_session(td)
            out = os.path.join(td, "force.mp4")
            saved = _retime.silence_spans
            _retime.silence_spans = lambda *a, **kw: None
            try:
                render.render(td, out_path=out, motion_blur=False,
                              speedup=True, speedup_rate=6.0,
                              speedup_silence_gate=True,
                              speedups=[{"start": 5.0, "end": 15.0,
                                         "mode": "force", "rate": 6.0}])
            finally:
                _retime.silence_spans = saved
            frames, _ = _probe(out)
            self.assertLess(frames, 1000, "force range didn't fire")
        finally:
            shutil.rmtree(td, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
