"""End-to-end: recorded clicks and keystrokes become audible sound.

test_sfx.py pins the bed in isolation and test_render_audio.py pins the
encoder argv. These decode the EXPORTED file, which is the only place the
two meet -- a bed built perfectly and then mixed at the wrong level, mapped
to no stream, or thrown away by `-shortest` would pass both of those and
still ship a silent video.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from autocine import render


def _have_ffmpeg():
    return shutil.which("ffmpeg") is not None


def _mk_session(root, events, duration=3, fps=30, width=320, height=180,
                with_audio=False, mic_layout="mono", mic_hz=440):
    """A synthetic session: `testsrc2` video, optional mic track.

    The mic is a real TONE, not `anullsrc`. Digital silence makes any "the
    mic survives" assertion unfalsifiable -- the export measures the same
    whether the recording was kept or dropped.

    `mic_layout` matters just as much: the SCK recorder always writes TWO
    channels (`_sck_worker` pins `AVNumberOfChannelsKey: 2`), and a mono-only
    fixture cannot see a downmix of the user's voice.
    """
    os.makedirs(root, exist_ok=True)
    raw = os.path.join(root, "raw.mov")
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
           "-f", "lavfi", "-i",
           "testsrc2=size={}x{}:rate={}:duration={}".format(
               width, height, fps, duration)]
    if with_audio:
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency={}:duration={}:sample_rate=48000".format(
                    mic_hz, duration)]
        if mic_layout == "left_only":
            # Signal on ONE channel of a stereo pair -- the shape a downmix
            # damages worst, and the one a mono fixture cannot represent.
            cmd += ["-f", "lavfi", "-i",
                    "anullsrc=r=48000:cl=mono:d={}".format(duration),
                    "-filter_complex",
                    "[1:a][2:a]join=inputs=2:channel_layout=stereo[a]",
                    "-map", "0:v", "-map", "[a]"]
        elif mic_layout == "stereo":
            cmd += ["-filter_complex",
                    "[1:a]aformat=channel_layouts=stereo[a]",
                    "-map", "0:v", "-map", "[a]"]
        else:
            cmd += ["-map", "0:v", "-map", "1:a"]
        cmd += ["-c:a", "aac"]
    cmd += ["-pix_fmt", "yuv420p", "-shortest", raw]
    subprocess.check_call(cmd)
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump({"raw": "raw.mov", "events": "events.jsonl", "fps": fps,
                   "logical_w": width, "logical_h": height,
                   "t0_monotonic": 0.0, "cursor_mode": "system",
                   "key_capture": "activity", "duration": duration}, f)


def _pcm(path, rate=8000):
    """Exported audio as a mono float array, or None when there is no track."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "a:0", "-ac", "1",
         "-ar", str(rate), "-f", "s16le", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, np.int16).astype(np.float64) / 32768.0


def _rms(arr, t_a, t_b, rate=8000):
    chunk = arr[int(t_a * rate):int(t_b * rate)]
    return float(np.sqrt(np.mean(chunk ** 2))) if chunk.size else 0.0


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class EventSoundsAreAudible(unittest.TestCase):
    """Sounds are ON by default -- a take with no options set has them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sd = os.path.join(cls.tmp, "sess")
        # A click at 0.5s and a burst of key ticks around 2.0s, with real
        # silence either side so each can be measured on its own.
        events = [{"t": 0.5, "type": "down", "x": 100, "y": 80},
                  {"t": 0.56, "type": "up", "x": 100, "y": 80}]
        events += [{"t": 2.0 + 0.1 * i, "type": "key", "x": 100, "y": 80}
                   for i in range(6)]
        _mk_session(cls.sd, events, duration=3, fps=30)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _render(self, name, **kwargs):
        out = os.path.join(self.tmp, name + ".mp4")
        render.render(self.sd, out_path=out, motion_blur=False,
                      facecam=False, click_fx=False, **kwargs)
        return out

    def test_default_render_has_a_click_and_a_typing_burst(self):
        arr = _pcm(self._render("on"))
        self.assertIsNotNone(arr, "the default render produced no audio track")
        quiet = _rms(arr, 1.2, 1.8)
        self.assertGreater(_rms(arr, 0.45, 0.75), max(quiet, 1e-4) * 5,
                           "no click sound near the recorded click")
        self.assertGreater(_rms(arr, 1.95, 2.65), max(quiet, 1e-4) * 5,
                           "no keystroke sounds during the typing burst")

    def test_click_and_key_sounds_are_independently_silenceable(self):
        no_keys = _pcm(self._render("nokeys", key_sound="off"))
        self.assertGreater(_rms(no_keys, 0.45, 0.75), 1e-3)
        self.assertLess(_rms(no_keys, 1.95, 2.65), 1e-3)

        no_clicks = _pcm(self._render("noclicks", click_sound="off"))
        self.assertLess(_rms(no_clicks, 0.45, 0.75), 1e-3)
        self.assertGreater(_rms(no_clicks, 1.95, 2.65), 1e-3)

    def test_both_off_leaves_a_mic_less_take_with_no_audio_track_at_all(self):
        """The off switch's real shape: no bed means no encoder input, which
        on a take with no mic means the pre-feature video-only output."""
        out = self._render("off", click_sound="off", key_sound="off")
        self.assertIsNone(_pcm(out))

    def test_zero_volume_is_the_same_off_switch(self):
        out = self._render("mute", sfx_volume=0.0)
        self.assertIsNone(_pcm(out))

    def test_volume_scales_what_lands_in_the_export(self):
        loud = _pcm(self._render("loud", sfx_volume=1.0))
        quiet = _pcm(self._render("quiet", sfx_volume=0.25))
        self.assertGreater(_rms(loud, 0.45, 0.75),
                           _rms(quiet, 0.45, 0.75) * 2.0)

    def test_keyboard_sounds_survive_typing_zoom_being_off(self):
        """`typing_zoom` is a CAMERA switch (a legacy no-op at that). It
        gates the key array the planner sees, and gating the sound bed with
        it would silence the keyboard for anyone who turned it off."""
        arr = _pcm(self._render("notypezoom", typing_zoom=False))
        self.assertGreater(_rms(arr, 1.95, 2.65), 1e-3)

    def test_trim_shifts_the_sounds_with_the_picture(self):
        """Trimming 1.5s off the head moves the typing burst to ~0.5s."""
        arr = _pcm(self._render("trim", trim_start=1.5, click_sound="off"))
        self.assertGreater(_rms(arr, 0.45, 1.15), 1e-3)
        self.assertLess(_rms(arr, 0.0, 0.4), 1e-3)

    def test_every_click_in_a_long_burst_is_equally_loud(self):
        """The regression the bed exists to prevent: under the old
        per-click `adelay` graph, amix normalized by the number of taps
        still alive, so early clicks were mixed near-silent and the last
        played at full level (measured 25x across 50 clicks)."""
        sd = os.path.join(self.tmp, "burst")
        _mk_session(sd, [{"t": 0.3 + 0.2 * i, "type": "down",
                          "x": 100, "y": 80} for i in range(40)],
                    duration=10, fps=30)
        out = os.path.join(self.tmp, "burst.mp4")
        render.render(sd, out_path=out, motion_blur=False, facecam=False,
                      click_fx=False, key_sound="off")
        arr = _pcm(out)
        peaks = []
        for i in range(40):
            t = 0.3 + 0.2 * i
            chunk = arr[int(t * 8000):int((t + 0.1) * 8000)]
            if chunk.size:
                peaks.append(float(np.abs(chunk).max()))
        self.assertEqual(len(peaks), 40)
        self.assertGreater(min(peaks), 0.0)
        self.assertLess(max(peaks) / min(peaks), 2.0,
                        "click loudness depends on position in the take")


def _channels(path):
    """(channel count, layout name) of the export's first audio stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels,channel_layout",
         "-of", "csv=p=0", path],
        stdout=subprocess.PIPE, check=True).stdout.decode().strip()
    parts = out.split(",")
    return int(parts[0]), (parts[1] if len(parts) > 1 else "")


def _per_channel_rms(path, t_a, t_b):
    """RMS of each channel over [t_a, t_b), at the file's own rate."""
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-map", "a:0", "-f", "f32le",
         "-ar", "48000", "-"],
        stdout=subprocess.PIPE, check=True).stdout
    n, _ = _channels(path)
    a = np.frombuffer(raw, "<f4").reshape(-1, n)
    w = a[int(t_a * 48000):int(t_b * 48000)]
    return [float(np.sqrt((w[:, i] ** 2).mean())) for i in range(n)]


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class EventSoundsWithAMicTrack(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.sd = os.path.join(cls.tmp, "sess")
        _mk_session(cls.sd, [{"t": 0.5, "type": "down", "x": 100, "y": 80}],
                    duration=2, fps=30, with_audio=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_the_mic_track_is_kept_and_the_bed_added_over_it(self):
        out = os.path.join(self.tmp, "mix.mp4")
        render.render(self.sd, out_path=out, motion_blur=False,
                      facecam=False, click_fx=False)
        arr = _pcm(out)
        self.assertIsNotNone(arr)
        # The bed, at the click.
        self.assertGreater(_rms(arr, 0.45, 0.75), 1e-3)
        # ...and the MIC, in a window with no click in it. The fixture used
        # to be `anullsrc`, so this half of the test measured nothing.
        self.assertGreater(_rms(arr, 0.05, 0.35), 1e-3,
                           "the mic track was dropped from the mix")

    def test_the_bed_does_not_shorten_the_export(self):
        """`-shortest` plus a mis-sized bed is the quiet way to truncate a
        video, so the duration is pinned against the sounds-off render."""
        on = os.path.join(self.tmp, "on.mp4")
        off = os.path.join(self.tmp, "off.mp4")
        render.render(self.sd, out_path=on, motion_blur=False, facecam=False,
                      click_fx=False)
        render.render(self.sd, out_path=off, motion_blur=False, facecam=False,
                      click_fx=False, click_sound="off", key_sound="off")

        def frames(path):
            return int(subprocess.run(
                ["ffprobe", "-v", "error", "-count_frames", "-select_streams",
                 "v:0", "-show_entries", "stream=nb_read_frames",
                 "-of", "default=nk=1:nw=1", path],
                stdout=subprocess.PIPE, check=True).stdout.strip())

        self.assertEqual(frames(on), frames(off))


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class TheBedDoesNotRemixTheMic(unittest.TestCase):
    """A mono bed beside a non-mono mic used to make ffmpeg resolve the
    mismatch by downmixing the RECORDING: turning on click sounds collapsed
    a stereo voiceover to mono and shifted it 3 dB. The bed is now pinned to
    the mic's layout (`render._probe_audio_layout` -> `aformat`).

    The SCK recorder always writes two channels, so this is the DEFAULT
    shape of a real narrated take -- it was invisible only because every
    mic fixture in the suite was mono.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _pair(self, layout):
        """(sounds-off, sounds-on) exports of one session."""
        sd = os.path.join(self.tmp, layout)
        _mk_session(sd, [{"t": 0.5, "type": "down", "x": 100, "y": 80}],
                    duration=2, fps=30, with_audio=True, mic_layout=layout)
        paths = []
        for name, kw in (("off", {"click_sound": "off", "key_sound": "off"}),
                         ("on", {})):
            out = os.path.join(self.tmp, layout + "_" + name + ".mp4")
            render.render(sd, out_path=out, motion_blur=False, facecam=False,
                          click_fx=False, **kw)
            paths.append(out)
        return paths

    def _assert_mic_untouched(self, layout):
        off, on = self._pair(layout)
        self.assertEqual(_channels(off), _channels(on),
                         "{}: enabling event sounds changed the export's "
                         "channel layout".format(layout))
        # Measured before the click, so only the mic contributes.
        a = _per_channel_rms(off, 0.05, 0.35)
        b = _per_channel_rms(on, 0.05, 0.35)
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            self.assertAlmostEqual(
                x, y, delta=0.002,
                msg="{}: channel {} moved {:.5f} -> {:.5f} when event "
                    "sounds were enabled".format(layout, i, x, y))

    def test_a_stereo_mic_survives_the_bed_unchanged(self):
        self._assert_mic_untouched("stereo")

    def test_a_mono_mic_survives_the_bed_unchanged(self):
        """The other half: pinning must not push a mono take to stereo,
        which would fold it 3 dB the other way."""
        self._assert_mic_untouched("mono")

    def test_a_one_sided_stereo_mic_is_not_folded_together(self):
        """Signal on one channel only -- an interface with a single input,
        or a system-audio loopback. A downmix silently mixes it into both."""
        self._assert_mic_untouched("left_only")
        off, on = self._pair("left_only")
        left, right = _per_channel_rms(on, 0.05, 0.35)
        self.assertGreater(left, 1e-3)
        self.assertLess(right, 1e-4, "the silent channel picked up signal")


@unittest.skipUnless(_have_ffmpeg(), "ffmpeg required")
class TempBedIsCleanedUp(unittest.TestCase):
    def test_no_bed_file_is_left_behind(self):
        tmp = tempfile.mkdtemp()
        try:
            sd = os.path.join(tmp, "sess")
            _mk_session(sd, [{"t": 0.5, "type": "down", "x": 10, "y": 10}],
                        duration=2, fps=30)
            before = set(f for f in os.listdir(tempfile.gettempdir())
                         if f.startswith("autocine-sfx-"))
            render.render(sd, out_path=os.path.join(tmp, "o.mp4"),
                          motion_blur=False, facecam=False, click_fx=False)
            after = set(f for f in os.listdir(tempfile.gettempdir())
                        if f.startswith("autocine-sfx-"))
            self.assertEqual(after - before, set())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
