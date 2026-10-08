"""Audio wiring for the occlusion-free multi-window (fleet) render.

Milestone 1 of the studio audio path: the mic rides channel 0
(record.py `_multi_native_worker_cfg`), and the multi-native export muxes
channel 0's audio track into the composite, seeked to the shared origin so it
stays in sync. A mic-less take must render byte-for-byte the video-only way.

Split in two, matching the house pattern (`test_render_audio` tests argv
construction; the end-to-end is ffmpeg-gated):
  * command-construction tests -- fast, device-free, pin the off-switch;
  * one end-to-end render with a real audio track -- proves the mux works.
"""
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from autocine import render


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class MultiNativeEncCmd(unittest.TestCase):
    """`_multi_native_enc_cmd` argv -- the off-switch and the audio path."""

    VIDEO_ONLY = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "1920x1080", "-framerate", "60.000000", "-i", "-",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "/tmp/out.mp4",
    ]

    def test_no_audio_is_byte_identical_to_the_legacy_video_only_command(self):
        # Off-switch: a mic-less fleet take must reproduce the exact command
        # the multi-native render always used -- no audio inputs, maps, codecs.
        cmd = render._multi_native_enc_cmd(1920, 1080, 60.0, "/tmp/out.mp4")
        self.assertEqual(cmd, self.VIDEO_ONLY)

    def test_audio_adds_a_second_input_and_maps_it(self):
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4",
            audio_path="/s/raw_0.mov", audio_skip_s=0.0)
        # video from stdin, audio from the channel-0 file
        self.assertEqual(cmd.count("-i"), 2)
        self.assertIn("/s/raw_0.mov", cmd)
        self.assertIn("-map", cmd)
        self.assertIn("0:v:0", cmd)
        self.assertIn("1:a:0", cmd)
        self.assertIn("-c:a", cmd)
        self.assertIn("-shortest", cmd)
        # No seek when channel 0 IS the origin (skip 0).
        self.assertNotIn("-ss", cmd)

    def test_audio_skip_seeks_the_audio_input_before_it(self):
        # When channel 0 started before the shared origin, its audio is seeked
        # by the same amount its video is skipped -- and the -ss must precede
        # the audio input, or it would seek the (piped) video instead.
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4",
            audio_path="/s/raw_0.mov", audio_skip_s=0.1)
        self.assertIn("-ss", cmd)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "0.100000")
        self.assertLess(cmd.index("-ss"), cmd.index("/s/raw_0.mov"))
        # The stdin video input ('-i', '-') must NOT be preceded by the -ss.
        self.assertGreater(cmd.index("-ss"), cmd.index("-"))


class MultiNativeEncCmdWithSfxBed(unittest.TestCase):
    """The event-sound bed as an extra input on the composite encoder.

    Two off-switches have to survive intact here, not one: a mic-less take
    with sounds off is still the video-only argv, and a mic take with sounds
    off is still the exact pre-bed mic argv.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.bed = os.path.join(self.td, "sfx.wav")
        with open(self.bed, "wb"):
            pass

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    # The mic-only argv, spelled out. Comparing the function against itself
    # with and without `sfx_path=None` proves nothing -- `sfx_path` already
    # DEFAULTS to None, so both calls are identical and any mutation moves
    # both sides together. Only a literal pins byte-identity.
    MIC_ONLY = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pixel_format", "bgr24",
        "-video_size", "1920x1080", "-framerate", "60.000000", "-i", "-",
        "-ss", "0.100000", "-i", "/s/raw_0.mov",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:a", "aac", "-b:a", "192k", "-shortest",
        "/tmp/out.mp4",
    ]

    def test_a_mic_take_with_no_bed_is_the_exact_pre_bed_command(self):
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4",
            audio_path="/s/raw_0.mov", audio_skip_s=0.1)
        self.assertEqual(cmd, self.MIC_ONLY)

    def test_passing_sfx_path_none_changes_nothing(self):
        for kw in ({}, {"audio_path": "/s/raw_0.mov", "audio_skip_s": 0.1}):
            before = render._multi_native_enc_cmd(
                1920, 1080, 60.0, "/tmp/out.mp4", **kw)
            after = render._multi_native_enc_cmd(
                1920, 1080, 60.0, "/tmp/out.mp4", sfx_path=None, **kw)
            self.assertEqual(before, after, kw)

    def test_a_missing_bed_file_is_ignored(self):
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4",
            sfx_path=os.path.join(self.td, "nope.wav"))
        self.assertEqual(cmd.count("-i"), 1)
        self.assertNotIn("-c:a", cmd)

    def test_bed_without_a_mic_becomes_input_one_and_is_mapped(self):
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4", sfx_path=self.bed)
        self.assertEqual(cmd.count("-i"), 2)
        self.assertIn(self.bed, cmd)
        self.assertIn("0:v:0", cmd)
        self.assertIn("1:a:0", cmd)
        # No mic to mix against, so no graph at all.
        self.assertNotIn("-filter_complex", cmd)
        self.assertIn("-shortest", cmd)

    def test_bed_with_a_mic_mixes_without_ducking_the_mic(self):
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4",
            audio_path="/s/raw_0.mov", audio_skip_s=0.1, sfx_path=self.bed)
        joined = " ".join(cmd)
        self.assertEqual(cmd.count("-i"), 3)
        self.assertIn("[1:a][2:a]amix=inputs=2:duration=longest:"
                      "dropout_transition=0:normalize=0[a_mix]", joined)
        self.assertIn("[a_mix]", cmd)
        self.assertIn("0:v:0", cmd)

    def test_the_bed_is_never_seeked_by_the_mic_skip(self):
        """`-ss` binds to the input it precedes. If the bed ever landed
        AFTER the -ss and before its own -i, every event sound would slide
        by the channel-0 skip -- silently, and only on takes whose channel 0
        started early."""
        cmd = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4",
            audio_path="/s/raw_0.mov", audio_skip_s=0.25, sfx_path=self.bed)
        # exactly one -ss, and the mic input sits between it and the bed
        self.assertEqual(cmd.count("-ss"), 1)
        self.assertLess(cmd.index("-ss"), cmd.index("/s/raw_0.mov"))
        self.assertLess(cmd.index("/s/raw_0.mov"), cmd.index(self.bed))

    def test_mic_stream_index_does_not_move_when_a_bed_is_added(self):
        """Index arithmetic: the mic must stay input 1 whether or not a bed
        follows it. Swapping the two would map the bed as the 'mic' and
        silently drop the voiceover."""
        no_bed = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4", audio_path="/s/raw_0.mov")
        with_bed = render._multi_native_enc_cmd(
            1920, 1080, 60.0, "/tmp/out.mp4", audio_path="/s/raw_0.mov",
            sfx_path=self.bed)
        self.assertEqual(no_bed.index("/s/raw_0.mov"),
                         with_bed.index("/s/raw_0.mov"))
        self.assertGreater(with_bed.index(self.bed),
                           with_bed.index("/s/raw_0.mov"))


def _mk_channel(path, w, h, dur, fps, audio=False):
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i",
           "testsrc2=size={}x{}:rate={}:duration={}".format(w, h, fps, dur)]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=440:duration={}".format(dur)]
    cmd += ["-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [path]
    subprocess.check_call(cmd)


def _probe_has_audio(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", path],
        text=True).strip()
    return bool(out)


def _mk_fleet_session(root, ch0_audio, events=(), t0_deltas=(0.0, 0.01),
                      duration=1.0):
    """A 2-channel fleet session.

    `t0_deltas` are per-channel `t0_monotonic` offsets from the session t0,
    in seconds. The default (0, 0.01) rounds to 0 frames at 30fps, i.e. the
    composite origin is 0 -- which is why a non-zero delta has to be
    passed explicitly to exercise the origin correction at all.
    """
    os.makedirs(root, exist_ok=True)
    fps = 30
    _mk_channel(os.path.join(root, "raw_0.mov"), 240, 160, duration, fps,
                audio=ch0_audio)
    _mk_channel(os.path.join(root, "raw_1.mov"), 240, 160, duration, fps,
                audio=False)
    channels = []
    for i in range(2):
        channels.append({
            "role": "screen_window", "file": "raw_{}.mov".format(i),
            "mode": "window_native", "id": 1000 + i, "app": "A", "title": "t",
            "rect": [float(i * 140), 0.0, 120.0, 80.0],
            "logical_w": 120.0, "logical_h": 80.0,
            "buffer_w": 240, "buffer_h": 160,
            "t0_monotonic": 100.0 + float(t0_deltas[i]),
        })
    meta = {
        "fps": fps, "logical_w": 1440.0, "logical_h": 900.0,
        "geom_source": "quartz", "t0_monotonic": 100.0,
        "video_index": 1, "mic_index": 0 if ch0_audio else None,
        "events": "events.jsonl", "cursor_mode": "system",
        "key_capture": "activity", "capture_backend": "sck",
        "capture_channels": channels,
    }
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return meta


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class MultiNativeAudioEndToEnd(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp(prefix="mn_audio_")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_channel0_audio_is_muxed_into_the_export(self):
        _mk_fleet_session(self.td, ch0_audio=True)
        out = render.render(self.td, aspect="240x160")
        self.assertTrue(os.path.isfile(out))
        self.assertTrue(_probe_has_audio(out),
                        "channel 0 had a mic track; the export must carry it")

    def test_micless_fleet_export_has_no_audio(self):
        # Off-switch end-to-end: no mic anywhere -> a silent export, exactly
        # the pre-audio behavior.
        _mk_fleet_session(self.td, ch0_audio=False)
        out = render.render(self.td, aspect="240x160")
        self.assertTrue(os.path.isfile(out))
        self.assertFalse(_probe_has_audio(out))

    def test_describe_reports_audio_truthfully(self):
        _mk_fleet_session(self.td, ch0_audio=True)
        info = render.describe_session(self.td)
        self.assertTrue(info["has_audio"])
        _mk_fleet_session(self.td, ch0_audio=False)
        info2 = render.describe_session(self.td)
        self.assertFalse(info2["has_audio"])


if __name__ == "__main__":
    unittest.main()
