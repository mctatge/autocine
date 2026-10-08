"""Unit tests for the click / keystroke sound bed (autocine/sfx.py).

The bed replaced an in-ffmpeg `asplit` + N x `adelay` + `amix` graph whose
per-event level depended on how many sibling taps were still alive. These
pin the properties that made the replacement worth doing -- constant level,
per-event variation, deterministic output -- plus the tri-state option
semantics the CLI, the editor and the MCP all resolve against.
"""

import os
import tempfile
import unittest
import wave

import numpy as np

from autocine import sfx


class ResolveOption(unittest.TestCase):
    """`click_sound` / `key_sound` are tri-state, and NULL IS ON."""

    def test_absent_means_the_builtin_sound(self):
        for value in (None, "", "   ", "auto", "AUTO"):
            self.assertEqual(sfx.resolve(value), sfx.AUTO, repr(value))

    def test_off_words_silence_the_kind(self):
        for value in ("off", "OFF", " none ", "no", "false", "0"):
            self.assertEqual(sfx.resolve(value), sfx.OFF, repr(value))

    def test_anything_else_is_a_path(self):
        self.assertEqual(sfx.resolve(" /tmp/click.wav "), "/tmp/click.wav")

    def test_editor_mirror_agrees_on_every_case(self):
        """studio_web/editor.js reimplements this in JS to drive the two
        switches. A switch that disagrees with the renderer is worse than
        no switch, so the off-word list has to stay identical."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(here, "studio_web", "editor.js")) as f:
            js = f.read()
        for word in sfx._OFF_WORDS:
            self.assertIn('"{}"'.format(word), js.split("SFX_OFF_WORDS")[1][:200])


class BuiltinSounds(unittest.TestCase):
    def test_banks_are_non_empty_and_bounded(self):
        for kind in ("click", "release", "key"):
            bank = sfx.builtin_bank(kind)
            self.assertTrue(bank, kind)
            for sample in bank:
                self.assertGreater(sample.size, 0, kind)
                self.assertLessEqual(float(np.abs(sample).max()), 1.0, kind)
                self.assertTrue(np.all(np.isfinite(sample)), kind)

    def test_release_is_quieter_than_the_press(self):
        """The mouse-up flourish must sit UNDER the press, or a single
        click reads as a double-click."""
        press = max(float(np.abs(s).max()) for s in sfx.builtin_bank("click"))
        release = float(np.abs(sfx.builtin_bank("release")[0]).max())
        self.assertLess(release, press * 0.6)

    def test_keystroke_is_quieter_than_a_click(self):
        """A take holds hundreds of keystrokes and a handful of clicks; at
        equal level the typing buries everything else."""
        click = max(float(np.abs(s).max()) for s in sfx.builtin_bank("click"))
        key = max(float(np.abs(s).max()) for s in sfx.builtin_bank("key"))
        self.assertLess(key, click)

    def test_variants_differ(self):
        for kind, count in (("click", sfx._CLICK_VARIANTS),
                            ("key", sfx._KEY_VARIANTS)):
            bank = sfx.builtin_bank(kind)
            self.assertEqual(len(bank), count)
            for i in range(len(bank)):
                for j in range(i + 1, len(bank)):
                    self.assertFalse(np.array_equal(bank[i], bank[j]),
                                     "{} variants {}/{} identical".format(
                                         kind, i, j))

    def test_synthesis_is_deterministic(self):
        """Re-rendering a take twice must produce identical audio -- the
        noise is fixed-seed for exactly this reason."""
        first = [s.copy() for s in sfx.builtin_bank("key")]
        sfx._CACHE.clear()
        second = sfx.builtin_bank("key")
        for a, b in zip(first, second):
            self.assertTrue(np.array_equal(a, b))

    def test_sounds_start_from_silence(self):
        """Every component starts at full amplitude on sample 0; without
        the attack ramp that step is an audible extra tick."""
        for kind in ("click", "release", "key"):
            for sample in sfx.builtin_bank(kind):
                self.assertLess(abs(float(sample[0])), 0.02, kind)


class EventVariation(unittest.TestCase):
    def test_variant_picks_are_spread_not_cyclic(self):
        """`index % count` would put a fixed PERIOD into a typing burst,
        which is audible as a repeating figure. Coverage and uniformity are
        necessary but nowhere near sufficient -- `i % 6` is perfectly
        uniform and covers every variant, and an earlier version of this
        test passed against it. What separates them is periodicity, so
        that is what is asserted."""
        picks = [sfx._pick(i, 6) for i in range(600)]
        counts = np.bincount(picks, minlength=6)
        self.assertEqual(len(np.nonzero(counts)[0]), 6)
        self.assertLess(counts.max() - counts.min(), 40)

        def periodic(seq, period):
            """True if seq repeats exactly every `period` events."""
            return all(seq[i] == seq[i + period]
                       for i in range(len(seq) - period))

        for period in range(1, 13):
            self.assertFalse(periodic(picks, period),
                             "variant picks repeat every {} events".format(
                                 period))
        # And the sanity check that the guard itself has teeth.
        self.assertTrue(periodic([i % 6 for i in range(600)], 6))

    def test_gain_jitter_stays_inside_the_requested_band(self):
        values = [sfx._gain_jitter(i, 0.2) for i in range(500)]
        self.assertGreaterEqual(min(values), 0.8 - 1e-9)
        self.assertLessEqual(max(values), 1.2 + 1e-9)

    def test_gain_jitter_actually_varies(self):
        """The bounds above are satisfied by a constant 1.0 -- i.e. by the
        jitter being dead, which is exactly the failure that would make a
        typing burst sound mechanical again."""
        values = [sfx._gain_jitter(i, 0.2) for i in range(500)]
        self.assertGreater(len(set(values)), 50)
        self.assertLess(min(values), 0.95)
        self.assertGreater(max(values), 1.05)

    def test_zero_jitter_is_exactly_unity(self):
        self.assertEqual(sfx._gain_jitter(7, 0.0), 1.0)


class BuildBed(unittest.TestCase):
    def _layer(self, times, **kw):
        return sfx.Layer(times, sfx.builtin_bank("click"), **kw)

    def test_returns_none_when_there_is_nothing_to_play(self):
        """None is the off switch's mechanism: render() only adds an
        encoder input when a bed came back."""
        self.assertIsNone(sfx.build_bed(5.0, []))
        self.assertIsNone(sfx.build_bed(5.0, [self._layer([])]))
        self.assertIsNone(sfx.build_bed(0.0, [self._layer([1.0])]))
        self.assertIsNone(sfx.build_bed(5.0, [self._layer([1.0])], volume=0.0))
        self.assertIsNone(sfx.build_bed(5.0, [sfx.Layer([1.0], [])]))

    def test_sound_lands_at_the_requested_time(self):
        bed = sfx.build_bed(3.0, [self._layer([1.0])])
        loudest = int(np.argmax(np.abs(bed)))
        self.assertAlmostEqual(loudest / float(sfx.SAMPLE_RATE), 1.0,
                               delta=0.01)

    def test_every_event_is_placed_at_the_same_level(self):
        """The whole point of the rewrite. Under the old `adelay` graph the
        first of N events was mixed at ~1/N and the last at ~1/1 -- measured
        as a 25x ramp across 50 clicks."""
        times = [0.2 + 0.1 * i for i in range(40)]
        bed = sfx.build_bed(6.0, [self._layer(times)])
        peaks = []
        for t in times:
            start = int(t * sfx.SAMPLE_RATE)
            peaks.append(float(np.abs(
                bed[start:start + int(0.05 * sfx.SAMPLE_RATE)]).max()))
        self.assertGreater(min(peaks), 0.0)
        # Only per-event jitter separates them, never their position.
        self.assertLess(max(peaks) / min(peaks), 1.6)

    def test_events_outside_the_clip_are_dropped_not_clamped(self):
        """A negative or past-the-end time must vanish, not pile onto the
        first/last frame as a phantom burst."""
        bed = sfx.build_bed(1.0, [self._layer([-1.0, 50.0])])
        self.assertIsNone(bed)

    def test_a_tail_is_kept_past_the_clip_end(self):
        """A click on the final frame keeps its DECAY instead of being
        chopped mid-transient; -shortest trims the surplus.

        Asserted on the audio, not on the buffer length: `build_bed` sizes
        the buffer as `duration*sr + tail + 1`, so a length-only assertion
        is satisfied by the `+ 1` alone and says nothing about the tail.
        """
        click = sfx.builtin_bank("click")[0]
        bed = sfx.build_bed(1.0, [self._layer([0.99])])
        self.assertGreaterEqual(bed.size,
                                int(1.0 * sfx.SAMPLE_RATE) + click.size)
        # The sound is not truncated: its energy past the nominal end is
        # what a too-short buffer would have thrown away.
        past_end = bed[int(1.0 * sfx.SAMPLE_RATE):]
        self.assertGreater(float(np.abs(past_end).max()), 0.01)

    def test_volume_scales_the_whole_bed(self):
        loud = sfx.build_bed(2.0, [self._layer([0.5])], volume=1.0)
        quiet = sfx.build_bed(2.0, [self._layer([0.5])], volume=0.25)
        self.assertAlmostEqual(float(np.abs(quiet).max()),
                               float(np.abs(loud).max()) * 0.25, delta=1e-3)

    def test_layers_sum_together(self):
        clicks = sfx.build_bed(2.0, [self._layer([0.5])])
        both = sfx.build_bed(2.0, [
            self._layer([0.5]),
            sfx.Layer([0.5], sfx.builtin_bank("key"))])
        self.assertGreater(float(np.abs(both).max()),
                           float(np.abs(clicks).max()))

    def test_output_never_clips(self):
        """A user-supplied sound file can be anything; the final clamp is
        the safety net that keeps the mix inside full scale."""
        hot = np.ones(2000, dtype=np.float32)
        bed = sfx.build_bed(
            2.0, [sfx.Layer([0.1, 0.1, 0.1, 0.1], [hot], gain=4.0)])
        self.assertLessEqual(float(np.abs(bed).max()), 1.0)

    def test_non_finite_times_are_skipped(self):
        bed = sfx.build_bed(2.0, [self._layer([float("nan"), 0.5, None])])
        self.assertIsNotNone(bed)
        self.assertAlmostEqual(int(np.argmax(np.abs(bed)))
                               / float(sfx.SAMPLE_RATE), 0.5, delta=0.01)

    def test_bed_is_reproducible(self):
        times = [0.1 * i for i in range(1, 30)]
        a = sfx.build_bed(4.0, [self._layer(times, jitter=0.2)])
        b = sfx.build_bed(4.0, [self._layer(times, jitter=0.2)])
        self.assertTrue(np.array_equal(a, b))


class SfxLayerAssembly(unittest.TestCase):
    """`render._sfx_layers` -- which layers actually get built.

    Levels and timings are pinned elsewhere; this pins that each layer
    EXISTS, which nothing else did. The mouse-up release in particular is
    computed and threaded through all three render paths and could have
    vanished silently.
    """

    def _layers(self, click="auto", key="auto",
                clicks=(0.1,), ups=(0.2,), keys=(0.3,)):
        from autocine import render
        return render._sfx_layers(sfx.resolve(click), sfx.resolve(key),
                                  list(clicks), list(ups), list(keys))

    def test_auto_builds_click_release_and_key_layers(self):
        layers = self._layers()
        self.assertEqual(len(layers), 3)
        self.assertEqual([list(ly.times) for ly in layers],
                         [[0.1], [0.2], [0.3]])

    def test_the_release_layer_is_dropped_when_clicks_are_off(self):
        """The release rides the CLICK switch -- silencing clicks must not
        leave half a mouse press behind."""
        layers = self._layers(click="off")
        self.assertEqual(len(layers), 1)
        self.assertEqual(list(layers[0].times), [0.3])

    def test_no_release_layer_without_mouse_up_events(self):
        self.assertEqual(len(self._layers(ups=())), 2)

    def test_a_custom_click_file_replaces_only_the_press(self):
        """A user file plays on the press alone: pairing an arbitrary sound
        with itself milliseconds later would read as a double-click."""
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "c.wav")
            sfx.write_wav(path, np.ones(480, dtype=np.float32) * 0.3)
            layers = self._layers(click=path)
        self.assertEqual(len(layers), 2)
        self.assertEqual([list(ly.times) for ly in layers], [[0.1], [0.3]])

    def test_an_unusable_sound_file_drops_only_its_own_layer(self):
        layers = self._layers(click="/tmp/not-a-real-sound-file.wav")
        self.assertEqual(len(layers), 1)
        self.assertEqual(list(layers[0].times), [0.3])


class WriteWav(unittest.TestCase):
    def test_round_trips_as_16_bit_mono_pcm(self):
        bed = sfx.build_bed(1.0, [
            sfx.Layer([0.25, 0.6], sfx.builtin_bank("click"))])
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "bed.wav")
            self.assertEqual(sfx.write_wav(path, bed), path)
            handle = wave.open(path, "rb")
            try:
                self.assertEqual(handle.getnchannels(), 1)
                self.assertEqual(handle.getsampwidth(), 2)
                self.assertEqual(handle.getframerate(), sfx.SAMPLE_RATE)
                self.assertEqual(handle.getnframes(), bed.size)
                back = np.frombuffer(
                    handle.readframes(handle.getnframes()),
                    dtype="<i2").astype(np.float32) / 32767.0
            finally:
                handle.close()
        self.assertLess(float(np.abs(back - bed).max()), 1e-3)


class DecodeFile(unittest.TestCase):
    def test_missing_file_returns_none(self):
        self.assertIsNone(sfx.decode_file("/tmp/not-a-real-sound-file.wav"))
        self.assertIsNone(sfx.decode_file(None))
        self.assertIsNone(sfx.decode_file(""))

    def test_undecodable_file_returns_none(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "junk.wav")
            with open(path, "wb") as f:
                f.write(b"not audio at all")
            self.assertIsNone(sfx.decode_file(path))

    def test_wav_round_trips_through_ffmpeg(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "tone.wav")
            sfx.write_wav(path, np.sin(
                np.linspace(0, 200.0, sfx.SAMPLE_RATE // 10)
            ).astype(np.float32) * 0.5)
            got = sfx.decode_file(path)
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got.size, sfx.SAMPLE_RATE // 10, delta=64)
        self.assertLess(float(np.abs(got).max()), 1.01)


if __name__ == "__main__":
    unittest.main()
