"""Click / keystroke sounds on the two COMPOSITE render paths.

`test_render_sfx.py` covers the single-file path. These cover the two that
share `_multi_native_enc_cmd`:

  * the multi-window-native fleet (`_render_multi_native`)
  * the scene take            (`_render_scenes`)

Both re-time events differently from the single-file body and from each
other, and both get their sounds from ONE pre-mixed bed. What is worth
testing is therefore almost entirely the CLOCK: a bed that plays is easy,
a bed that plays in the right place is the whole feature.

The argv layer is already pinned by
`test_render_multi_native_audio.MultiNativeEncCmdWithSfxBed`; nothing here
re-pins it.
"""

import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np

from autocine import render, segments

# House style (test_render_cuts.py does the same): reuse the synthetic
# session builders and PCM helpers rather than growing a fourth copy.
from tests.test_render_sfx import _pcm, _rms
from tests.test_render_multi_native_audio import (_mk_fleet_session,
                                                  _probe_has_audio)
from tests.test_render_scenes import _mk_scene_session
from tests.test_render_scene_join import _mk_join_session
from tests.test_render_segmented import (_mk_segmented_session, D as SEG_D,
                                         T0_0 as SEG_T0, T0_1 as SEG_T1)

FPS = 30


def _have_ffmpeg():
    return (shutil.which("ffmpeg") is not None
            and shutil.which("ffprobe") is not None)


def _click(t):
    return {"t": t, "type": "down", "x": 60.0, "y": 40.0,
            "button": "Button.left"}


def _key(t):
    return {"t": t, "type": "key", "x": 60.0, "y": 40.0}


def _frames(path):
    return int(subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of",
         "default=nk=1:nw=1", path],
        stdout=subprocess.PIPE, check=True).stdout.strip())


def _loudest(arr, t_a, t_b, rate=8000):
    """(time, amplitude) of the largest sample inside [t_a, t_b).

    Position, not window energy. An event sound is ~14 ms long, so an RMS
    window wide enough to be robust is also wide enough to straddle a wrong
    answer 100 ms away -- which is exactly how the first version of the
    origin test below passed against a deliberately broken renderer.
    """
    lo, hi = int(t_a * rate), int(t_b * rate)
    chunk = np.abs(arr[lo:hi])
    if chunk.size == 0:
        return None, 0.0
    i = int(np.argmax(chunk))
    return (lo + i) / float(rate), float(chunk[i])


class PrivateTempdir(object):
    """Point `tempfile` at a private directory for the duration of a test.

    The bed-cleanup checks list the tempdir for `autocine-sfx-*`. Against
    the SHARED machine-wide one that is a race: any other autocine render --
    another test process, the studio app, a parallel agent -- lands a file
    between the before and after snapshots and fails a correct
    implementation. Observed happening during review.
    """

    def __enter__(self):
        self._prev = tempfile.tempdir
        self.path = tempfile.mkdtemp(dir=self._prev)
        tempfile.tempdir = self.path
        return self

    def __exit__(self, *exc):
        tempfile.tempdir = self._prev
        shutil.rmtree(self.path, ignore_errors=True)
        return False

    def beds(self):
        return set(f for f in os.listdir(self.path)
                   if f.startswith("autocine-sfx-"))


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class FleetBed(unittest.TestCase):
    """`_render_multi_native`: the bed rides the COMPOSITE clock."""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _render(self, name, events=(), t0_deltas=(0.0, 0.01),
                ch0_audio=False, **kw):
        sd = os.path.join(self.td, name)
        _mk_fleet_session(sd, ch0_audio, events=events, t0_deltas=t0_deltas)
        out = os.path.join(self.td, name + ".mp4")
        render.render(sd, out_path=out, aspect="320x180", **kw)
        return out

    def test_a_recorded_click_becomes_an_audible_sound(self):
        out = self._render("basic", events=[_click(100.5)])
        arr = _pcm(out)
        self.assertIsNotNone(arr, "fleet render produced no audio track")
        self.assertGreater(_rms(arr, 0.45, 0.75),
                           max(_rms(arr, 0.05, 0.35), 1e-4) * 5)

    def test_recorded_typing_becomes_audible(self):
        keys = [_key(100.4 + 0.1 * i) for i in range(5)]
        out = self._render("keys", events=keys, click_sound="off")
        arr = _pcm(out)
        self.assertIsNotNone(arr)
        self.assertGreater(_rms(arr, 0.38, 0.95), 1e-3)

    def test_the_bed_sits_on_the_composite_origin_not_session_t0(self):
        """THE crux of this path.

        Every channel is grab()-skipped to the shared origin, so composite
        output frame 0 is SESSION frame `origin` -- not session frame 0. A
        bed built on the SESSION clock sits `origin/fps` LATE, and out of
        sync with the mic, which IS corrected by the same term.

        This is a TWO-SIDED pin, and the other side has already fired once:
        `to_media` now folds the origin skip into its anchor
        (`render._multi_native_composite_t0`), so a bed that ALSO subtracts
        `origin/fps` -- as this one did while `to_media` was session-anchored
        -- takes it off twice and lands `origin/fps` EARLY. That is a real
        regression this caught on the anchor merge, not a hypothetical.

        Here channel 1 starts 0.1 s late at 30 fps, so origin = 3 frames =
        0.1 s, and a click recorded at session +0.5 s must sound at 0.4 s.
        Note the default fixture deltas (0, 0.01) round to origin == 0, so
        no other test on this path can tell the two clocks apart.
        """
        out = self._render("origin", events=[_click(100.5)],
                           t0_deltas=(0.0, 0.1))
        arr = _pcm(out)
        self.assertIsNotNone(arr)
        when, amp = _loudest(arr, 0.0, 0.9)
        self.assertGreater(amp, 0.05, "no click sound in the export at all")
        # Measured: 0.4005 s correct; 0.5005 s on the raw session clock;
        # 0.3010 s when the origin skip comes off twice.
        self.assertAlmostEqual(
            when, 0.40, delta=0.02,
            msg="click sounded at {:.4f}s; 0.50 means the bed is on the "
                "SESSION clock instead of the composite's, and 0.30 means "
                "the origin skip was subtracted twice (`to_media` already "
                "folds it in -- clip_start must be 0.0)".format(when))

    def test_sounds_off_leaves_a_micless_fleet_take_with_no_audio(self):
        """The off switch AT THE CALL SITE. The argv tests can only see a
        None `sfx_path` arriving; this catches a silent wav being built and
        handed over anyway, which would turn a video-only export into one
        with a silent AAC track."""
        out = self._render("off", events=[_click(100.5), _key(100.7)],
                           click_sound="off", key_sound="off")
        self.assertFalse(_probe_has_audio(out))

    def test_zero_volume_is_the_same_off_switch(self):
        """Also pins that `sfx_volume` actually REACHES this path: at the
        default 1.0 every other test passes whether or not it is forwarded
        through the dispatch."""
        out = self._render("mute", events=[_click(100.5)], sfx_volume=0.0)
        self.assertFalse(_probe_has_audio(out))

    def test_volume_scales_the_bed(self):
        loud = _pcm(self._render("loud", events=[_click(100.5)],
                                 sfx_volume=1.0))
        quiet = _pcm(self._render("quiet", events=[_click(100.5)],
                                  sfx_volume=0.25))
        self.assertGreater(_rms(loud, 0.45, 0.75),
                           _rms(quiet, 0.45, 0.75) * 2.0)

    def test_the_bed_mixes_over_a_mic_track_without_shortening_the_video(self):
        """`-shortest` plus a mis-sized bed silently truncates the VIDEO
        (measured: a 0.8 s bed against 2.0 s of video exported 24 frames of
        60), so the frame count is pinned against the sounds-off render."""
        on = self._render("mix_on", events=[_click(100.5)], ch0_audio=True)
        off = self._render("mix_off", events=[_click(100.5)], ch0_audio=True,
                           click_sound="off", key_sound="off")
        self.assertTrue(_probe_has_audio(on))
        self.assertEqual(_frames(on), _frames(off))

        a_on, a_off = _pcm(on), _pcm(off)
        # The BED reached the mix. `_probe_has_audio` alone is satisfied by
        # the mic, so without this the whole test passes with the bed
        # dropped -- it is the mic-take half of the feature.
        # PEAK, not RMS: a 14 ms click barely moves the RMS of a window
        # that also holds a continuous mic tone (measured 0.089 vs 0.088),
        # so an RMS comparison here cannot see the bed at all.
        self.assertGreater(_loudest(a_on, 0.45, 0.75)[1],
                           _loudest(a_off, 0.45, 0.75)[1] * 1.8,
                           "the bed did not reach the mix")
        # ...and the mic is UN-DUCKED: measured against the sounds-off
        # render in a window with no click in it, not against a threshold
        # loose enough that amix's default halving would still pass.
        self.assertAlmostEqual(_rms(a_on, 0.05, 0.35),
                               _rms(a_off, 0.05, 0.35), delta=0.002,
                               msg="the mic level moved when sounds were on")

    def test_the_note_no_longer_claims_sounds_are_ignored(self):
        """A stale entry in the ignored list tells the user the exact
        opposite of what the render just did."""
        sd = os.path.join(self.td, "note")
        _mk_fleet_session(sd, False, events=[_click(100.5)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            render.render(sd, out_path=os.path.join(self.td, "note.mp4"),
                          aspect="320x180")
        text = buf.getvalue()
        self.assertNotIn("click_sound", text)
        self.assertNotIn("key_sound", text)

    def test_no_bed_file_is_left_behind_on_success(self):
        with PrivateTempdir() as tmp:
            self._render("clean", events=[_click(100.5)])
            self.assertEqual(tmp.beds(), set())

    def test_no_bed_file_is_left_behind_when_the_encode_fails(self):
        """Exercises the `finally`, which needs a failure AFTER the bed
        exists. Corrupting a channel file does NOT do that -- the capture
        fails to open at the top of the try, long before `_write_sfx_bed`
        (verified: the spy below never fires on that path). Failing the
        `Popen` is the smallest thing that lands in the right window."""
        sd = os.path.join(self.td, "boom")
        _mk_fleet_session(sd, False, events=[_click(100.5)])
        wrote = []
        real_bed = render._write_sfx_bed
        real_popen = render.subprocess.Popen

        def bed_spy(*a, **k):
            path = real_bed(*a, **k)
            wrote.append(path)
            return path

        def popen_boom(*a, **k):
            raise OSError("ffmpeg could not be launched")

        with PrivateTempdir() as tmp:
            render._write_sfx_bed = bed_spy
            render.subprocess.Popen = popen_boom
            try:
                with self.assertRaises(OSError):
                    render.render(sd,
                                  out_path=os.path.join(self.td, "boom.mp4"),
                                  aspect="320x180")
            finally:
                render._write_sfx_bed = real_bed
                render.subprocess.Popen = real_popen
            self.assertTrue(wrote and wrote[0],
                            "the bed was never written -- this test is not "
                            "exercising the cleanup it claims to")
            self.assertEqual(tmp.beds(), set())


class SceneBedTimes(unittest.TestCase):
    """`_scene_bed_times` on its own -- no encode, no ffmpeg.

    The fixture clock is the one `_mk_scene_session` produces: scene 0 owns
    [100.0, 101.0) -> output [0.0, 1.0), scene 1 owns [110.1, 111.0) ->
    output [1.0, 1.9).
    """

    def setUp(self):
        self.clock = segments.SegmentClock([100.0, 110.1], [1.0, 0.9])

    def test_events_map_through_the_take_wide_clock(self):
        times, _ = render._scene_bed_times(self.clock, [100.5, 110.7])
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[0], 0.5, places=6)
        # 1.0 (scene 0's whole duration) + (110.7 - 110.1)
        self.assertAlmostEqual(times[1], 1.6, places=6)

    def test_an_event_in_a_deleted_pause_gap_is_dropped_not_clamped(self):
        """`clock.media` alone maps t=105.0 to 1.0 -- the seam -- because it
        is clamp-not-drop for the x/y-zipped callers. Clicking during a
        pause is normal (that is when windows get rearranged), so clamping
        would fire a burst at every seam."""
        self.assertAlmostEqual(float(self.clock.media([105.0])[0]), 1.0)
        times, _ = render._scene_bed_times(self.clock, [105.0])
        self.assertEqual(times, [])

    def test_a_tail_event_past_the_last_scene_is_dropped(self):
        self.assertAlmostEqual(float(self.clock.media([120.0])[0]), 1.9)
        self.assertEqual(render._scene_bed_times(self.clock, [120.0])[0], [])

    def test_an_event_before_the_first_scene_is_dropped(self):
        self.assertAlmostEqual(float(self.clock.media([99.0])[0]), 0.0)
        self.assertEqual(render._scene_bed_times(self.clock, [99.0])[0], [])

    def test_seam_ownership_is_half_open(self):
        """t == a scene's content end belongs to NEITHER (it is the deleted
        gap's first instant); t == the next scene's t0 belongs to it."""
        self.assertEqual(render._scene_bed_times(self.clock, [101.0])[0], [])
        times, _ = render._scene_bed_times(self.clock, [110.1])
        self.assertEqual(len(times), 1)
        self.assertAlmostEqual(times[0], 1.0, places=6)

    def test_empty_and_non_finite_inputs_are_safe(self):
        for value in (None, [], [float("nan")], [float("inf")]):
            self.assertEqual(render._scene_bed_times(self.clock, value)[0],
                             [])


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class SceneBed(unittest.TestCase):
    """`_render_scenes`: ONE bed on the take-wide clock, across the seam."""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _render(self, name, events=(), **kw):
        sd = os.path.join(self.td, name)
        _mk_scene_session(sd, events=events)
        out = os.path.join(self.td, name + ".mp4")
        render.render(sd, out_path=out, aspect="320x180", **kw)
        return out

    def test_sounds_from_both_scenes_land_on_the_take_wide_clock(self):
        """The deleted pause wall-clock (9 s here) must come out of the bed
        too. A per-scene bed would stack scene 1's sounds at t=0; a raw
        `t - t0` would put the second click at 10.7 s, past the export."""
        out = self._render("both", events=[_click(100.5), _click(110.7)])
        arr = _pcm(out)
        self.assertIsNotNone(arr, "scene render produced no audio track")
        # Derived from the fixture's clock, which is t0s [100.0, 110.1] /
        # durs [1.0, 0.9]:  100.5 -> 0.5, and 110.7 -> 1.0 + 0.6 = 1.6.
        first, amp_a = _loudest(arr, 0.0, 1.0)
        second, amp_b = _loudest(arr, 1.0, 1.9)
        self.assertGreater(amp_a, 0.05, "scene 0's click is missing")
        self.assertGreater(amp_b, 0.05, "scene 1's click is missing")
        self.assertAlmostEqual(first, 0.50, delta=0.02)
        self.assertAlmostEqual(
            second, 1.60, delta=0.02,
            msg="scene 1's click sounded at {:.4f}s; 0.6 would mean a "
                "per-scene bed, 1.7 an un-clipped scene origin".format(second))
        # ...and the deleted 9 s pause really is gone: nothing in between.
        self.assertLess(_rms(arr, 0.75, 1.35), 1e-3)

    def test_a_click_during_the_deleted_pause_makes_no_sound_at_the_seam(self):
        """End-to-end counterpart of the unit test above: without the
        containment mask this click would fire exactly on the 1.0 s seam.

        A real click at 110.7 rides along deliberately -- with only the gap
        click, `build_bed` would return None and the export would have no
        audio track at all, so the seam assertion would pass without ever
        being exercised.
        """
        out = self._render("gap", events=[_click(105.0), _click(110.7)])
        arr = _pcm(out)
        self.assertIsNotNone(arr, "the companion click should have built a bed")
        self.assertGreater(_rms(arr, 1.55, 1.80), 1e-3,
                           "companion click missing -- test is not exercising "
                           "the seam")
        self.assertLess(_rms(arr, 0.90, 1.20), 1e-3,
                        "the gap click was clamped onto the seam")

    def test_sounds_off_leaves_a_scene_take_with_no_audio(self):
        out = self._render("off", events=[_click(100.5)],
                           click_sound="off", key_sound="off")
        self.assertFalse(_probe_has_audio(out))

    def test_zero_volume_is_the_same_off_switch(self):
        out = self._render("mute", events=[_click(100.5)], sfx_volume=0.0)
        self.assertFalse(_probe_has_audio(out))

    def test_the_bed_does_not_change_the_frame_count(self):
        on = self._render("len_on", events=[_click(100.5)])
        off = self._render("len_off", events=[_click(100.5)],
                           click_sound="off", key_sound="off")
        self.assertEqual(_frames(on), _frames(off))
        self.assertEqual(_frames(on), 57)   # the fixture's pinned length

    def test_the_note_no_longer_claims_sounds_are_ignored(self):
        sd = os.path.join(self.td, "note")
        _mk_scene_session(sd, events=[_click(100.5)])
        buf = io.StringIO()
        with redirect_stdout(buf):
            render.render(sd, out_path=os.path.join(self.td, "note.mp4"),
                          aspect="320x180")
        text = buf.getvalue()
        self.assertNotIn("click_sound", text)
        self.assertNotIn("key_sound", text)
        # ...but the things that ARE still dropped must keep saying so.
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            render.render(sd, out_path=os.path.join(self.td, "note2.mp4"),
                          aspect="320x180", music="/tmp/nope.mp3")
        self.assertIn("music", buf2.getvalue())

    def test_no_bed_file_is_left_behind(self):
        with PrivateTempdir() as tmp:
            self._render("clean", events=[_click(100.5)])
            self.assertEqual(tmp.beds(), set())


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class SegmentedTakeBed(unittest.TestCase):
    """A pause/resume SEGMENTED take -- the third clock-driven path.

    It reaches `render()` with `_to_media_override=clock.media`, which is
    clamp-not-drop: without a containment mask every click made during the
    deleted pause maps onto the seam and the bed SUMS them there, so three
    clicks during one pause fire as a single 3x-loud click. Same defect
    `_scene_bed_times` prevents on the scene path; the fix is shared
    (`render._clock_contained`, reached here via `render`'s `_bed_clock`).

    The fixture's two segments are 1.0 s of content each with a deleted gap
    between them, so the seam sits at output t = 1.0 s.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _render(self, name, events, **kw):
        sd = os.path.join(self.td, name)
        _mk_segmented_session(sd, events)
        out = os.path.join(self.td, name + ".mp4")
        render.render(sd, out_path=out, motion_blur=False, facecam=False,
                      click_fx=False, **kw)
        return out

    def test_clicks_inside_the_deleted_pause_make_no_sound_at_the_seam(self):
        """Three clicks during the pause plus one real click in segment 1.
        The real one must sound; the seam must stay at the noise floor."""
        events = [{"t": SEG_T0 + SEG_D + 0.5 + 0.3 * i, "type": "down",
                   "x": 100, "y": 90} for i in range(3)]
        events.append({"t": SEG_T1 + 0.5, "type": "down", "x": 120,
                       "y": 100})
        arr = _pcm(self._render("gap", events))
        self.assertIsNotNone(arr,
                             "the segment-1 click should have built a bed")
        when, amp = _loudest(arr, 1.35, 1.75)
        self.assertGreater(amp, 0.05, "the segment-1 click is missing")
        self.assertAlmostEqual(when, 1.50, delta=0.03)
        self.assertLess(_loudest(arr, 0.90, 1.20)[1], 0.02,
                        "pause clicks were clamped onto the seam")

    def test_a_real_click_in_each_segment_still_sounds(self):
        """The other half: the mask must not eat legitimate events."""
        events = [{"t": SEG_T0 + 0.5, "type": "down", "x": 100, "y": 90},
                  {"t": SEG_T1 + 0.5, "type": "down", "x": 120, "y": 100}]
        arr = _pcm(self._render("both", events))
        self.assertIsNotNone(arr)
        first, amp_a = _loudest(arr, 0.30, 0.80)
        second, amp_b = _loudest(arr, 1.30, 1.80)
        self.assertGreater(amp_a, 0.05)
        self.assertGreater(amp_b, 0.05)
        self.assertAlmostEqual(first, 0.50, delta=0.03)
        self.assertAlmostEqual(second, 1.50, delta=0.03)


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class JoinTakeBed(unittest.TestCase):
    """A gapfree window-join scene take -- the only scene shape that
    carries a continuous channel-0 mic, so the only one that exercises the
    scene call site's `amix` branch."""

    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _render(self, name, events=(), ch0_audio=True, **kw):
        sd = os.path.join(self.td, name)
        _mk_join_session(sd, ch0_audio=ch0_audio, events=events)
        out = os.path.join(self.td, name + ".mp4")
        render.render(sd, out_path=out, aspect="320x180", **kw)
        return out

    def test_the_bed_mixes_over_the_voiceover_without_truncating_it(self):
        events = [_click(100.0 + 0.5), _click(100.0 + 1.5)]
        on = self._render("join_on", events=events)
        off = self._render("join_off", events=events,
                           click_sound="off", key_sound="off")
        self.assertTrue(_probe_has_audio(on))
        self.assertTrue(_probe_has_audio(off))
        self.assertEqual(_frames(on), _frames(off))
        self.assertEqual(_frames(on), 60)

        a_on, a_off = _pcm(on), _pcm(off)
        # Both `_probe_has_audio` calls above are satisfied by the mic
        # alone, so on their own they cannot see whether the bed reached
        # the mix -- and this is the ONLY scene shape with a mic, i.e. the
        # only exercise of the scene call site's amix branch.
        for label, t in (("first", 0.5), ("second", 1.5)):
            # Peak, for the same reason as the fleet test above.
            self.assertGreater(_loudest(a_on, t - 0.05, t + 0.25)[1],
                               _loudest(a_off, t - 0.05, t + 0.25)[1] * 1.8,
                               "the {} click did not reach the mix".format(
                                   label))
        # The voiceover is not ducked to make room for it.
        self.assertAlmostEqual(_rms(a_on, 0.05, 0.35),
                               _rms(a_off, 0.05, 0.35), delta=0.002)

    def test_a_micless_join_take_still_gets_its_sounds(self):
        out = self._render("join_nomic", events=[_click(100.5)],
                           ch0_audio=False)
        arr = _pcm(out)
        self.assertIsNotNone(arr)
        self.assertGreater(_rms(arr, 0.45, 0.75),
                           max(_rms(arr, 0.05, 0.35), 1e-4) * 5)


if __name__ == "__main__":
    unittest.main()
