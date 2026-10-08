"""End-to-end tests for SEGMENTED (pause/resume) takes.

Synthesizes a 2-segment session (testsrc2 seg_i.mov + a `capture_segments`
manifest + events.jsonl) -- permissions-free, ffmpeg only, the same harness the
capture-window / multi-native render tests use.

The load-bearing checks:
  - the paused wall-clock gap is DELETED (joined duration = sum of content),
  - a click in segment 1 lands at its gaps-removed output time,
  - describe / beats / the SegmentClock AGREE (the trailing-cluster contract),
  - the off switch: a plain take is untouched.

Design: docs/architecture.md.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import render, segments
from autocine import beats as beats_mod


FPS = 30
D = 1.0                       # each segment is 1.0s of content
NF = int(FPS * D)            # 30 frames per segment
GAP = 5.0                    # paused wall-clock, must vanish from the output
T0_0 = 100.0
T0_1 = T0_0 + D + GAP        # segment 1's fresh per-process t0 (== 106.0)


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _mk_segment(path, width=320, height=180, with_audio=False,
                audio_delay=0.0):
    src = "testsrc2=size={}x{}:rate={}:duration={}".format(width, height, FPS, D)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i", src]
    if with_audio:
        if audio_delay:
            # A real audio start_time, like the SCK mic's ~0.26s spin-up:
            # the audio genuinely BEGINS at `audio_delay` on the segment's
            # video timeline, carried as container placement.
            cmd += ["-itsoffset", "{:.3f}".format(audio_delay)]
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=440:duration={}".format(D)]
    cmd += ["-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac"]
        if not audio_delay:
            cmd += ["-shortest"]
    cmd += [path]
    subprocess.check_call(cmd)


def _mk_segmented_session(root, events, with_audio=False,
                          width=320, height=180, audio_delay=0.0):
    os.makedirs(root, exist_ok=True)
    wa = (with_audio if isinstance(with_audio, tuple)
          else (with_audio, with_audio))
    _mk_segment(os.path.join(root, "seg_0.mov"), width, height, wa[0],
                audio_delay=audio_delay)
    _mk_segment(os.path.join(root, "seg_1.mov"), width, height, wa[1],
                audio_delay=audio_delay)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    meta = {
        "fps": FPS, "logical_w": width, "logical_h": height,
        "t0_monotonic": T0_0, "cursor_mode": "system",
        "events": "events.jsonl",
        "capture_segments": [
            {"file": "seg_0.mov", "index": 0, "t0_monotonic": T0_0,
             "role": "screen"},
            {"file": "seg_1.mov", "index": 1, "t0_monotonic": T0_1,
             "role": "screen"},
        ],
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    return meta


def _events():
    # A click 0.5s into segment 0, one 0.5s into segment 1, and one INSIDE the
    # deleted gap (must clamp to the seam, never vanish).
    return [
        {"t": T0_0 + 0.5, "type": "down", "x": 100, "y": 90},
        {"t": T0_0 + D + 2.0, "type": "down", "x": 110, "y": 95},   # in the gap
        {"t": T0_1 + 0.5, "type": "down", "x": 120, "y": 100},
    ]


def _probe_duration(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1", path], text=True).strip()
    return float(out)


def _probe_has_audio(path):
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "a",
        "-show_entries", "stream=index", "-of", "csv=p=0", path],
        text=True).strip()
    return bool(out)


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class SegmentedDescribe(unittest.TestCase):
    def test_duration_deletes_the_paused_gap(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events())
            info = render.describe_session(td, include_click_times=True)
            self.assertTrue(info["segmented"])
            self.assertIsNone(info["raw_path"])
            # 2 x 1.0s content, the 5.0s gap gone.
            self.assertAlmostEqual(info["duration"], 2.0, places=1)
            self.assertEqual(info["frame_count"], 2 * NF)
            self.assertEqual(info["width"], 320)

    def test_click_times_are_gaps_removed_output_time(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events())
            info = render.describe_session(td, include_click_times=True)
            ct = info["click_times"]
            self.assertEqual(len(ct), 3)          # clamp-not-drop: none lost
            # seg0 click -> 0.5 ; gap click -> seam 1.0 ; seg1 click -> 1.5.
            self.assertAlmostEqual(ct[0], 0.5, places=1)
            self.assertAlmostEqual(ct[1], 1.0, places=1)
            self.assertAlmostEqual(ct[2], 1.5, places=1)


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class SegmentedRenderEndToEnd(unittest.TestCase):
    def test_video_only_join_renders_continuous_output(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events(), with_audio=False)
            out = os.path.join(td, "out.mp4")
            render.render(td, out_path=out, motion_blur=False, click_fx=False)
            self.assertTrue(os.path.isfile(out))
            self.assertAlmostEqual(_probe_duration(out), 2.0, delta=0.2)
            # The temp joined file is cleaned up.
            self.assertFalse(any(n.startswith("._segjoin")
                                 for n in os.listdir(td)))

    def test_mic_on_join_carries_audio(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events(), with_audio=True)
            out = os.path.join(td, "out.mp4")
            render.render(td, out_path=out, motion_blur=False, click_fx=False)
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(_probe_has_audio(out))
            # Audio was rebuilt drift-free to exactly the joined video length.
            self.assertAlmostEqual(_probe_duration(out), 2.0, delta=0.2)

    def test_mixed_audio_segments_survive_a_mic_dropout(self):
        # The SCK worker is video-first: a resume can lose the mic and produce
        # a video-only segment. The join must synthesize that segment's D_i of
        # silence, not crash on a missing [i:a] stream.
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events(), with_audio=(True, False))
            out = os.path.join(td, "out.mp4")
            render.render(td, out_path=out, motion_blur=False, click_fx=False)
            self.assertTrue(os.path.isfile(out))
            self.assertTrue(_probe_has_audio(out))
            self.assertAlmostEqual(_probe_duration(out), 2.0, delta=0.2)

    def test_audio_keeps_its_recorded_placement(self):
        # SCK audio starts ~0.26s after video frame 0 and that start_time is
        # REAL placement (the worker rebases mic pts onto the video timeline).
        # The join must reproduce it as leading silence -- never slide the
        # audio early, which would put every segment's sound ~0.26s ahead of
        # its picture.
        with tempfile.TemporaryDirectory() as td:
            meta = _mk_segmented_session(td, _events(), with_audio=True,
                                         audio_delay=0.3)
            joined, seg_counts, total, tmps = render._concat_segments(
                td, meta, FPS)
            try:
                pcm = subprocess.run(
                    ["ffmpeg", "-v", "error", "-i", joined, "-map", "a:0",
                     "-ac", "1", "-ar", "8000", "-f", "s16le", "-"],
                    stdout=subprocess.PIPE, check=True).stdout
                arr = np.frombuffer(pcm, np.int16).astype(np.float64)
                sr = 8000.0

                def rms(t_a, t_b):
                    chunk = arr[int(t_a * sr):int(t_b * sr)]
                    return float(np.sqrt(np.mean(chunk ** 2))) if chunk.size \
                        else 0.0

                tone = rms(0.45, 0.9)          # the sine, where it was recorded
                self.assertGreater(tone, 500.0)
                # The [0, 0.3) lead-in stays (near-)silent in BOTH segments'
                # windows -- segment 1 starts at output 1.0.
                self.assertLess(rms(0.0, 0.25), tone * 0.05)
                self.assertLess(rms(1.0, 1.25), tone * 0.05)
                self.assertGreater(rms(1.45, 1.9), 500.0)
            finally:
                for t in tmps:
                    try:
                        os.remove(t)
                    except OSError:
                        pass

    def test_crop_rect_is_forwarded_to_the_joined_render(self):
        # The editor's saved crop must survive the segmented dispatch -- a
        # cropped segmented take exporting full-frame would silently disagree
        # with the editor.
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events())
            out = os.path.join(td, "out.mp4")
            render.render(td, out_path=out, motion_blur=False, click_fx=False,
                          crop_rect={"x": 0, "y": 0, "w": 160, "h": 90})
            probe = subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0",
                 out], text=True).strip()
            self.assertEqual(probe, "160,90")


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class SharedMapperAgreement(unittest.TestCase):
    """render, describe, and beats must all cluster on the SAME output times.
    The invariant the whole read-model fork exists to protect."""

    def test_describe_and_beats_agree_with_the_clock(self):
        with tempfile.TemporaryDirectory() as td:
            _mk_segmented_session(td, _events())
            info = render.describe_session(td, include_click_times=True)
            # The clock the describe payload implies.
            seg_counts = [s["frame_count"] for s in info["capture_segments"]]
            with open(os.path.join(td, "meta.json")) as mf:
                meta = json.load(mf)
            clock = segments.SegmentClock.from_meta(meta, seg_counts, FPS)
            expected = clock.media(np.array([T0_0 + 0.5,
                                             T0_0 + D + 2.0,
                                             T0_1 + 0.5]))
            for got, exp in zip(info["click_times"], expected.tolist()):
                self.assertAlmostEqual(got, exp, places=4)
            # beats' click beat must sit at the same gaps-removed time as the
            # describe click_times (not the raw-minus-t0 time that would still
            # contain the 5s gap).
            sheet = render.session_beats(td)
            click_beats = [b for b in sheet["beats"] if b["kind"] == "clicks"]
            self.assertTrue(click_beats)
            starts = [b["start"] for b in click_beats]
            # The seg-1 click at output 1.5 must be represented; nothing near
            # the un-deleted raw time (~6.5) may appear.
            self.assertTrue(min(starts) < 2.1)
            self.assertTrue(max(starts) < 2.1)


class OffSwitch(unittest.TestCase):
    def test_plain_meta_is_not_segmented(self):
        self.assertFalse(segments.is_segmented_meta(
            {"raw": "raw.mov", "events": "events.jsonl"}))

    def test_beats_media_fn_none_is_identity(self):
        # A 1-entry clock reduces to arr - t0; and beat_sheet with media_fn=None
        # is the pre-feature path. Guard the bit-exact off switch.
        ev = {"clicks_t": np.array([2.0, 3.0]),
              "clicks_x": np.array([10.0, 20.0]),
              "clicks_y": np.array([10.0, 20.0]),
              "moves_t": np.array([]), "moves_x": np.array([]),
              "moves_y": np.array([]), "ups_t": np.array([]),
              "keys_t": np.array([]), "scrolls_t": np.array([]),
              "scrolls_x": np.array([]), "scrolls_y": np.array([]),
              "windows_t": np.array([]),
              "windows_rect": np.zeros((0, 4)), "windows_id": np.array([]),
              "windows_z": np.array([])}
        a = beats_mod.beat_sheet(dict(ev), duration=5.0, t0=1.0)
        b = beats_mod.beat_sheet(dict(ev), duration=5.0, t0=1.0, media_fn=None)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
