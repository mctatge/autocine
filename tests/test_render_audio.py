"""Unit tests for render audio mixing command construction."""

import os
import tempfile
import unittest

from autocine import render


class ClickSoundTimes(unittest.TestCase):
    def test_click_times_are_localized_to_trim_window(self):
        times, dropped = render._click_times_in_trim(
            [0.2, 1.1, 2.9, 3.2],
            clip_start=1.0,
            clip_duration=2.0,
            max_events=10,
        )
        self.assertEqual(dropped, 0)
        self.assertEqual(len(times), 2)
        self.assertAlmostEqual(times[0], 0.1)
        self.assertAlmostEqual(times[1], 1.9)

    def test_click_times_cap_reports_dropped_count(self):
        times, dropped = render._click_times_in_trim(
            [0.0, 0.1, 0.2],
            clip_start=0.0,
            clip_duration=1.0,
            max_events=2,
        )
        self.assertEqual(len(times), 2)
        self.assertEqual(dropped, 1)


class EncodeCommand(unittest.TestCase):
    """The SFX bed reaches the encoder as ONE extra input.

    It replaced an `asplit` + N x `adelay` + `amix` graph, whose per-tap
    level depended on how many sibling taps were still alive -- see sfx.py.
    """

    def test_sfx_bed_alone_is_mapped_without_a_filter_graph(self):
        with tempfile.TemporaryDirectory() as td:
            bed = os.path.join(td, "sfx.wav")
            with open(bed, "wb"):
                pass
            cmd = render._encode_cmd(
                640, 360, 30.0,
                raw_path="/tmp/raw.mov",
                has_audio=False,
                music=None,
                out_path="/tmp/out.mp4",
                sfx_path=bed,
            )
            self.assertIn(bed, cmd)
            # With nothing to mix against there is no graph at all, so the
            # map has to be a STREAM specifier -- a bracketed "[1:a]" would
            # be read as a filter linklabel and fail to resolve.
            self.assertNotIn("-filter_complex", cmd)
            self.assertIn("1:a", cmd)
            self.assertNotIn("[1:a]", cmd)
            self.assertIn("-c:a", cmd)

    def test_sfx_bed_mixes_over_recording_audio_without_ducking_it(self):
        with tempfile.TemporaryDirectory() as td:
            bed = os.path.join(td, "sfx.wav")
            with open(bed, "wb"):
                pass
            cmd = render._encode_cmd(
                640, 360, 60.0,
                raw_path="/tmp/raw.mov",
                has_audio=True,
                music=None,
                out_path="/tmp/out.mp4",
                audio_start=0.25,
                audio_duration=5.0,
                sfx_path=bed,
            )
            joined = " ".join(cmd)
            self.assertIn("-filter_complex", cmd)
            # normalize=0 is the load-bearing token: amix's default would
            # halve the mic the moment a click landed. duration=longest so a
            # short mic track cannot drag the mix (and -shortest with it,
            # the video) down to its own length.
            self.assertIn("[1:a][2:a]amix=inputs=2:duration=longest:"
                          "dropout_transition=0:normalize=0[a_mix]", joined)
            self.assertIn("[a_mix]", cmd)

    def test_no_sfx_bed_is_bit_exact_with_the_pre_feature_command(self):
        """The off switch: every event sound silenced (so render() builds no
        bed) must produce the exact argv the renderer used before the
        feature -- no stray input, no filter graph."""
        cmd_off = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.25, audio_duration=5.0, sfx_path=None)
        cmd_pre = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.25, audio_duration=5.0)
        self.assertEqual(cmd_off, cmd_pre)
        self.assertNotIn("-filter_complex", cmd_off)

    def test_missing_bed_file_is_ignored(self):
        cmd = render._encode_cmd(
            640, 360, 60.0,
            raw_path="/tmp/raw.mov", has_audio=True, music=None,
            out_path="/tmp/out.mp4",
            sfx_path="/tmp/definitely-not-a-real-bed.wav")
        self.assertNotIn("-filter_complex", cmd)
        self.assertIn("1:a", cmd)


class RetimeAudioGraph(unittest.TestCase):
    """The atrim+atempo+concat retimed graph (speedup on)."""

    def test_no_retime_segments_kwarg_absent_is_bit_exact(self):
        """Bit-exact off-switch: absent retime_segments must produce the
        pre-feature argv EXACTLY. This is the audio-side of the off-switch
        contract, mirrored across every downstream consumer."""
        cmd_pre = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.25, audio_duration=5.0)
        cmd_none = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.25, audio_duration=5.0,
            retime_segments=None)
        self.assertEqual(cmd_pre, cmd_none)

    def test_all_rate_one_segments_are_bit_exact_no_op(self):
        """A retime_segments list whose every rate is 1.0 must also
        collapse to the pre-feature argv -- callers may pass the identity
        segment list and expect zero cost."""
        cmd_pre = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.25, audio_duration=5.0)
        cmd_id = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.25, audio_duration=5.0,
            retime_segments=[(0.0, 5.0, 1.0)])
        self.assertEqual(cmd_pre, cmd_id)

    def test_retime_graph_disables_input_ss_and_t(self):
        """When retiming, -ss / -t on the audio input MUST be omitted or
        the atrim windows below (absolute source seconds) get double-clipped."""
        cmd = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=2.0, audio_duration=10.0,
            retime_segments=[(0.0, 5.0, 1.0),
                             (5.0, 10.0, 4.8)])
        # -ss appears only on the raw video input? No: raw video is stdin
        # (-i -). audio input is the second one. There must be no -ss
        # immediately preceding "/tmp/raw.mov".
        raw_i = cmd.index("/tmp/raw.mov")
        self.assertNotIn("-ss", cmd[max(0, raw_i - 4):raw_i])
        self.assertNotIn("-t", cmd[max(0, raw_i - 4):raw_i])

    def test_retime_graph_builds_atrim_atempo_concat(self):
        cmd = render._encode_cmd(
            640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=60.0,
            retime_segments=[(0.0, 10.0, 1.0),
                             (10.0, 20.0, 4.8),
                             (20.0, 60.0, 1.0)])
        fc_i = cmd.index("-filter_complex")
        graph = cmd[fc_i + 1]
        # atrim windows are absolute source seconds
        self.assertIn("atrim=0.000000:10.000000", graph)
        self.assertIn("atrim=10.000000:20.000000", graph)
        self.assertIn("atrim=20.000000:60.000000", graph)
        # rate 4.8 decomposes to 2.0 * 2.0 * 1.2 -- all three atempo factors
        self.assertIn("atempo=2.000000,atempo=2.000000,atempo=1.200000", graph)
        # concat produces the final [a_rt] label
        self.assertIn("concat=n=3:v=0:a=1[a_rt]", graph)
        self.assertIn("[a_rt]", cmd)

    def test_retime_with_sfx_bed_mixes_a_rt_with_the_bed(self):
        with tempfile.TemporaryDirectory() as td:
            bed = os.path.join(td, "sfx.wav")
            with open(bed, "wb"):
                pass
            cmd = render._encode_cmd(
                640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
                music=None, out_path="/tmp/out.mp4",
                audio_start=0.0, audio_duration=60.0,
                sfx_path=bed,
                retime_segments=[(0.0, 10.0, 1.0),
                                 (10.0, 20.0, 4.8),
                                 (20.0, 60.0, 1.0)])
            graph = " ".join(cmd)
            # The bed is built on the OUTPUT timeline, so it is mixed with
            # the already-retimed [a_rt] -- never atempo'd itself.
            self.assertIn("[a_rt][2:a]amix=inputs=2:duration=longest:"
                          "dropout_transition=0:normalize=0[a_mix]", graph)

    def test_retime_with_music_ducks_a_rt(self):
        with tempfile.TemporaryDirectory() as td:
            music_path = os.path.join(td, "bg.mp3")
            with open(music_path, "wb"):
                pass
            cmd = render._encode_cmd(
                640, 360, 60.0, raw_path="/tmp/raw.mov", has_audio=True,
                music=music_path, out_path="/tmp/out.mp4",
                audio_start=0.0, audio_duration=60.0,
                retime_segments=[(0.0, 10.0, 1.0),
                                 (10.0, 20.0, 4.8),
                                 (20.0, 60.0, 1.0)])
            graph = " ".join(cmd)
            # retimed recording feeds the volume-duck node
            self.assertIn("[a_rt]volume=0.55[a_rec]", graph)


class CutsAudioGraph(unittest.TestCase):
    """Holed (cuts) segment lists: all-rate-1.0 but NOT tiling the trim
    window. These MUST take the filter-graph path -- the gaps are the
    removed audio -- while the gapless all-1.0 no-op above stays pinned."""

    def test_holed_all_rate_one_forces_filter_graph(self):
        """The design trap this feature was planned around: before the
        hole-aware gate, this argv came out identical to the pre-feature
        one and shipped the cut audio back in."""
        cmd_pre = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=4.0)
        cmd_cut = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=4.0,
            retime_segments=[(0.0, 1.0, 1.0), (2.0, 4.0, 1.0)])
        self.assertNotEqual(cmd_pre, cmd_cut)
        self.assertIn("-filter_complex", cmd_cut)

    def test_holed_graph_atrim_windows_skip_the_cut(self):
        """Byte-literal filter strings: the atrim windows are exactly the
        kept ranges (no atempo -- every kept rate is 1.0), concat butts
        them into [a_rt]."""
        cmd = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=4.0,
            retime_segments=[(0.0, 1.0, 1.0), (2.0, 4.0, 1.0)])
        fc_i = cmd.index("-filter_complex")
        graph = cmd[fc_i + 1]
        self.assertEqual(
            graph,
            "[1:a]atrim=0.000000:1.000000,asetpts=PTS-STARTPTS[rt0];"
            "[1:a]atrim=2.000000:4.000000,asetpts=PTS-STARTPTS[rt1];"
            "[rt0][rt1]concat=n=2:v=0:a=1[a_rt]")
        self.assertNotIn("atempo", graph)

    def test_holed_graph_maps_a_rt(self):
        """The [a_rt] selection branch (the easy line for a gate refactor
        to miss): a cuts-only, no-music render must -map [a_rt], not the
        raw audio stream -- '-map 1:a' here means uncut audio shipped."""
        cmd = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=4.0,
            retime_segments=[(0.0, 1.0, 1.0), (2.0, 4.0, 1.0)])
        map_i = len(cmd) - 1 - cmd[::-1].index("-map")
        self.assertEqual(cmd[map_i + 1], "[a_rt]")
        self.assertNotIn("1:a", cmd)

    def test_holed_graph_disables_input_ss_and_t(self):
        cmd = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=1.0, audio_duration=3.0,
            retime_segments=[(0.0, 0.5, 1.0), (1.5, 3.0, 1.0)])
        raw_i = cmd.index("/tmp/raw.mov")
        self.assertNotIn("-ss", cmd[max(0, raw_i - 4):raw_i])
        self.assertNotIn("-t", cmd[max(0, raw_i - 4):raw_i])

    def test_hole_shorter_than_window_end_forces_graph(self):
        """A single all-1.0 segment that stops short of the window end is
        a tail cut -- also a hole."""
        cmd = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=4.0,
            retime_segments=[(0.0, 3.0, 1.0)])
        self.assertIn("-filter_complex", cmd)

    def test_cuts_compose_with_speedup_in_one_graph(self):
        cmd = render._encode_cmd(
            640, 360, 30.0, raw_path="/tmp/raw.mov", has_audio=True,
            music=None, out_path="/tmp/out.mp4",
            audio_start=0.0, audio_duration=10.0,
            retime_segments=[(0.0, 2.0, 1.0),      # kept
                             (3.0, 5.0, 4.8),      # speedup survives the cut
                             (5.0, 10.0, 1.0)])    # [2,3] was cut
        fc_i = cmd.index("-filter_complex")
        graph = cmd[fc_i + 1]
        self.assertIn("atrim=0.000000:2.000000", graph)
        self.assertIn("atrim=3.000000:5.000000", graph)
        self.assertIn("atempo=2.000000,atempo=2.000000,atempo=1.200000", graph)
        self.assertIn("concat=n=3:v=0:a=1[a_rt]", graph)


class CoversWindowHelper(unittest.TestCase):
    def test_gapless_lists_classify_as_covering(self):
        # the exact shapes segments_for_audio builds for speedups
        self.assertTrue(render._covers_window([(0.0, 5.0, 1.0)], 5.0))
        self.assertTrue(render._covers_window(
            [(0.0, 10.0, 1.0), (10.0, 20.0, 4.8), (20.0, 60.0, 1.0)], 60.0))
        # sub-epsilon float noise stays "covering"
        self.assertTrue(render._covers_window(
            [(0.0, 9.9999999, 1.0), (10.0, 60.0, 1.0)], 60.0000001))

    def test_holes_classify_as_not_covering(self):
        self.assertFalse(render._covers_window(
            [(0.0, 1.0, 1.0), (2.0, 4.0, 1.0)], 4.0))       # interior hole
        self.assertFalse(render._covers_window(
            [(1.0, 4.0, 1.0)], 4.0))                        # head cut
        self.assertFalse(render._covers_window(
            [(0.0, 3.0, 1.0)], 4.0))                        # tail cut
        self.assertFalse(render._covers_window([], 4.0))


if __name__ == "__main__":
    unittest.main()
