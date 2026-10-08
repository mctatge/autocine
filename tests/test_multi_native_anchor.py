"""The FLAT fleet composite's time anchor.

`_render_multi_native` skips every channel forward to the shared origin, so
the composite's output frame 0 is SESSION frame `origin` -- output time is
`(session frame - origin)/fps`, not `session frame/fps`. Three consumers
already knew that (the mic's `audio_skip_s` and each channel's `start` in
`multi_native_layout`); the
picture-side effects grid did not, so every per-card zoom, focus move and
synthetic cursor fired exactly `origin` frames AFTER the picture showed the
click that caused it -- and `describe`/`beats` reported those clicks on a
timeline the composite does not have.

`origin` measured 0-8 frames across the local fleet takes (0-133ms at 60fps),
but it is exactly how far the SLOWEST channel's SCK worker lagged session t0
and is unbounded in principle, so these pin it with a deliberately large one.

The scene path solved this first -- `segments.scene_clock_entries` anchors on
`t0_ch0 + origin/fps` -- and these tests hold the flat path to the same rule.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from autocine import render as ren   # noqa: E402


def _have_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        try:
            subprocess.check_output([tool, "-version"],
                                    stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError):
            return False
    return True


FPS = 30
BW, BH = 240, 160
LW, LH = BW // 2, BH // 2          # points; scale 2 -> buffer
SESSION_T0 = 100.0
CLICKS_MEDIA = (3.9, 4.0, 4.1, 4.2)
FLASH_MEDIA = 4.0
DUR = 8.0


def _make_fleet(root, lag_s, flash=False, cursor=False):
    """A 2-channel fleet take whose channel 1 starts `lag_s` AFTER session t0,
    so `origin == round(lag_s * FPS)`. With `flash`, channel 0 turns white at
    media t=FLASH_MEDIA -- a burn-in that pins the PICTURE anchor, since a
    channel's own file time is `media - its t0`, so both channels must be
    generated at their own local instant for one shared media instant.

    With `cursor`, the take is a `--cursor synthetic` one carrying a move
    track that parks the pointer far outside every window until FLASH_MEDIA
    and jumps it into window 1 exactly then -- an EXPORT-side effect firing at
    the same media instant the burn-in marks. Clicks are omitted in that mode:
    `geometry.load_events` folds every click into the move track, so they
    would pull the pointer on-screen early and blunt the step.
    """
    os.makedirs(root, exist_ok=True)
    t0s = [SESSION_T0, SESSION_T0 + lag_s]
    for i, t0 in enumerate(t0s):
        vf = "null"
        if flash and i == 0:
            local = FLASH_MEDIA - (t0 - SESSION_T0)
            vf = ("drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:"
                  "enable='gte(t,{})'".format(local))
        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "testsrc2=size={}x{}:rate={}:duration={}".format(
                 BW, BH, FPS, DUR),
             "-vf", vf, "-pix_fmt", "yuv420p",
             os.path.join(root, "raw_{}.mov".format(i))])
    channels = [{
        "role": "screen_window", "file": "raw_{}.mov".format(i),
        "mode": "window_native", "id": 1000 + i, "app": "A",
        "title": "w{}".format(i),
        "rect": [float(i * (LW + 20)), 0.0, float(LW), float(LH)],
        "logical_w": float(LW), "logical_h": float(LH),
        "buffer_w": BW, "buffer_h": BH, "t0_monotonic": t0s[i],
    } for i in range(2)]
    with open(os.path.join(root, "events.jsonl"), "w") as f:
        if cursor:
            # Far outside both windows, then inside window 1 (rect
            # [140, 0, 120, 80] -> centre 200, 40). The last off-screen sample
            # sits ONE frame before the jump so the resampled track steps
            # rather than sweeps.
            t = 0.0
            while t < FLASH_MEDIA - 1.0 / FPS:
                f.write(json.dumps({"t": SESSION_T0 + t, "type": "move",
                                    "x": 9000, "y": 9000}) + "\n")
                t += 0.1
            f.write(json.dumps({"t": SESSION_T0 + FLASH_MEDIA - 1.0 / FPS,
                                "type": "move", "x": 9000, "y": 9000}) + "\n")
            t = FLASH_MEDIA
            while t <= DUR:
                f.write(json.dumps({"t": SESSION_T0 + t, "type": "move",
                                    "x": 200, "y": 40}) + "\n")
                t += 0.1
        else:
            for t in CLICKS_MEDIA:
                f.write(json.dumps({"t": SESSION_T0 + t, "type": "down",
                                    "x": 30, "y": 40}) + "\n")
    meta = {"fps": FPS, "t0_monotonic": SESSION_T0, "events": "events.jsonl",
            "width": BW, "height": BH, "logical_w": 1440.0,
            "logical_h": 900.0, "capture_backend": "sck",
            "cursor_mode": "synthetic" if cursor else "system",
            "key_capture": "activity",
            "capture_channels": channels}
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f)
    return root


class CompositeT0(unittest.TestCase):
    """The helper itself: the composite's t0 is `session_t0 + origin/fps`,
    which is the SAME instant whichever channel you derive it through."""

    def test_matches_every_channels_own_skip(self):
        fps, session_t0 = 60.0, 100.0
        for offsets in ([0, 7], [0, -18, 3], [0, 0], [0, -3]):
            origin = max(0, max(offsets))
            want = ren._multi_native_composite_t0(session_t0, origin, fps)
            for off in offsets:
                # Channel i's file frame (origin - off) is output frame 0; its
                # monotonic time is `session_t0 + off/fps + (origin-off)/fps`.
                self.assertAlmostEqual(
                    want, session_t0 + off / fps + (origin - off) / fps,
                    places=9, msg=str(offsets))

    def test_zero_origin_is_the_identity(self):
        # The off switch: a fleet take whose slowest channel did not lag keeps
        # session t0 exactly, so every take with origin 0 is untouched.
        self.assertEqual(
            ren._multi_native_composite_t0(100.0, 0, 60.0), 100.0)

    def test_degenerate_fps_falls_back_to_session_t0(self):
        for fps in (0, 0.0, None):
            self.assertEqual(
                ren._multi_native_composite_t0(100.0, 30, fps), 100.0)


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class FlatFleetAnchor(unittest.TestCase):

    LAG = 1.0                      # 30 frames at FPS -- deliberately large

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(prefix="mn_anchor_")
        cls.lagged = _make_fleet(os.path.join(cls.root, "lag"), cls.LAG)
        cls.aligned = _make_fleet(os.path.join(cls.root, "flat"), 0.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    @property
    def origin(self):
        return int(round(self.LAG * FPS))

    def test_effects_grid_is_on_the_composite_clock(self):
        # The live-preview emitter is the one place the render's own event
        # mapping is observable without decoding; the export shares it.
        sp = ren._native_camera_inputs(self.lagged)
        self.assertIsNotNone(sp)
        want = [t - self.origin / float(FPS) for t in CLICKS_MEDIA]
        self.assertEqual([round(float(x), 6) for x in sp["clicks_t"]],
                         [round(t, 6) for t in want])

    def test_describe_reports_click_times_on_the_composite_clock(self):
        info = ren.describe_session(self.lagged, include_click_times=True)
        want = [t - self.origin / float(FPS) for t in CLICKS_MEDIA]
        self.assertEqual([round(t, 6) for t in info["click_times"]],
                         [round(t, 6) for t in want])
        # And that clock is the one `duration` is on -- the composite clips to
        # the common overlap, so it is SHORTER than the session by `origin`.
        self.assertAlmostEqual(info["duration"],
                               DUR - self.origin / float(FPS), places=6)

    def test_beats_cluster_on_the_same_clock_as_the_render(self):
        # The trailing-cluster contract: render, describe and beats must agree
        # on when a click happened, or a `describe_session().beats` timestamp
        # points somewhere the zoom is not.
        sheet = ren.session_beats(self.lagged)
        clicks = [b for b in sheet["beats"] if b.get("kind") == "clicks"]
        self.assertTrue(clicks, "fixture produced no clicks beat")
        lo = min(float(b["start"]) for b in clicks)
        self.assertAlmostEqual(lo, CLICKS_MEDIA[0] - self.origin / float(FPS),
                               places=2)

    def test_an_unlagged_take_is_untouched(self):
        # Off switch: origin 0 -> every surface reports raw media times, so
        # the overwhelmingly common take is bit-identical to before.
        sp = ren._native_camera_inputs(self.aligned)
        self.assertEqual([round(float(x), 6) for x in sp["clicks_t"]],
                         [round(t, 6) for t in CLICKS_MEDIA])
        info = ren.describe_session(self.aligned, include_click_times=True)
        self.assertEqual([round(t, 6) for t in info["click_times"]],
                         [round(t, 6) for t in CLICKS_MEDIA])


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class PictureAndEffectsAgree(unittest.TestCase):
    """The end-to-end claim, measured off real pixels: the frame that SHOWS a
    given media instant is the frame an effect fires on. Without the anchor
    these differ by exactly `origin`, which is the whole bug.

    Both halves are read out of RENDERED OUTPUT, so this binds
    `_render_multi_native`'s OWN copy of the anchor. An earlier version took
    the effect time from `_native_camera_inputs` -- the live-preview preamble,
    which keeps a separate hand-copy of the same setup -- and so left the
    export half of the fix unpinned: reverting only the export's
    `composite_t0` kept the entire suite green while demonstrably putting the
    synthetic cursor `origin` frames behind the picture again.
    """

    LAG = 1.0                      # 30 frames at FPS -- deliberately large

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mn_pix_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    @property
    def origin(self):
        return int(round(self.LAG * FPS))

    def _render(self, d, name, **kw):
        return ren.render(d, out_path=os.path.join(self.root, name),
                          window_zoom=False, window_focus=False,
                          max_height=400, **kw)

    def _flash_frame(self, path):
        """First output frame whose card-0 area is white."""
        import cv2
        cap = cv2.VideoCapture(path)
        k, first = 0, None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            h, w = fr.shape[:2]
            if first is None and float(fr[h // 2, w // 4].mean()) > 200:
                first = k
            k += 1
        cap.release()
        return first

    def test_the_burn_in_lands_on_the_frame_the_planner_plans_for(self):
        d = _make_fleet(os.path.join(self.root, "s"), self.LAG, flash=True)
        first_white = self._flash_frame(self._render(d, "o.mp4"))
        self.assertIsNotNone(first_white, "flash never appeared in the export")
        # PICTURE: media FLASH_MEDIA is shown `origin` frames earlier than a
        # session-t0 reading of the output timeline would put it.
        self.assertEqual(first_white,
                         int(round(FLASH_MEDIA * FPS)) - self.origin)
        # EFFECTS (preview preamble): the planner's grid agrees. Read its OWN
        # mapping rather than deriving the expected time here, which would
        # assert the arithmetic instead of the code.
        sp = ren._native_camera_inputs(d)
        i = CLICKS_MEDIA.index(FLASH_MEDIA)
        planned = int(np.argmin(np.abs(
            sp["frame_times"] - float(sp["clicks_t"][i]))))
        self.assertEqual(planned, first_white)

    def test_the_exports_own_effects_fire_on_the_picture_frame(self):
        # The export half, with nothing but decoded frames on both sides.
        # `cursor_fx` is the cheapest export-side effect to observe: the
        # pointer is out of every window until FLASH_MEDIA, so a cursor-on and
        # a cursor-off render are identical until it arrives, and the first
        # differing frame IS the instant the export drew it.
        import cv2
        d = _make_fleet(os.path.join(self.root, "c"), self.LAG,
                        flash=True, cursor=True)
        off = self._render(d, "off.mp4", cursor_fx=False)
        on = self._render(d, "on.mp4", cursor_fx=True)

        a, b = cv2.VideoCapture(off), cv2.VideoCapture(on)
        k, cursor_frame, flash_frame = 0, None, None
        while True:
            oka, fa = a.read()
            okb, fb = b.read()
            if not (oka and okb):
                break
            h, w = fa.shape[:2]
            if flash_frame is None and float(fa[h // 2, w // 4].mean()) > 200:
                flash_frame = k
            if cursor_frame is None and int(np.abs(
                    fa.astype(int) - fb.astype(int)).max()) > 8:
                cursor_frame = k
            k += 1
        a.release()
        b.release()

        self.assertIsNotNone(flash_frame, "flash never appeared")
        self.assertIsNotNone(
            cursor_frame,
            "cursor_fx changed no pixel -- the probe stopped probing")
        # The picture instant, from pixels.
        self.assertEqual(flash_frame,
                         int(round(FLASH_MEDIA * FPS)) - self.origin)
        # The export's own effect, from pixels, on that same frame. Anchored
        # on session t0 this is `origin` frames later.
        self.assertEqual(cursor_frame, flash_frame)


@unittest.skipUnless(_have_ffmpeg(), "needs ffmpeg/ffprobe")
class MicRidesTheCompositeOrigin(unittest.TestCase):
    """`audio_skip_s` is the OTHER origin term on this path, and nothing
    observed it: setting it to 0.0 kept the whole suite green while sliding
    the voiceover `origin/fps` late against the picture.

    It is easy to mistake for the bed's `clip_start`, which had to become 0.0
    when the anchor moved -- and it sits three lines away. But it is a
    different quantity: a SEEK inside channel 0's own file (hence its
    `frame_offsets[0]` term), not a rebase of event times, so the anchor never
    overlapped it. This pins that distinction from decoded PCM rather than
    from argv, because the argv tests pass the value in as a literal.
    """

    LAG = 1.0

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mn_mic_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_the_mic_lands_on_the_frame_the_picture_shows(self):
        d = os.path.join(self.root, "s")
        os.makedirs(d, exist_ok=True)
        origin = int(round(self.LAG * FPS))
        t0s = [SESSION_T0, SESSION_T0 + self.LAG]
        # Channel 0 carries BOTH marks at one media instant: a white burn-in
        # and a 1 kHz burst, each at its own local time. Channel 1 is plain.
        local = FLASH_MEDIA - (t0s[0] - SESSION_T0)
        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "error",
             "-f", "lavfi", "-i", "testsrc2=size={}x{}:rate={}:duration={}"
             .format(BW, BH, FPS, DUR),
             "-f", "lavfi", "-i",
             "sine=frequency=1000:duration={}".format(DUR),
             "-vf", ("drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:"
                     "enable='gte(t,{})'".format(local)),
             # Per-frame gate: 1 inside the burst window, 0 everywhere else.
             # Two chained `volume=enable=` filters do NOT do this -- the one
             # that is disabled passes the signal through untouched.
             "-af", "volume='between(t,{},{})':eval=frame"
                    .format(local, local + 0.08),
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
             os.path.join(d, "raw_0.mov")])
        subprocess.check_call(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i",
             "testsrc2=size={}x{}:rate={}:duration={}".format(BW, BH, FPS, DUR),
             "-pix_fmt", "yuv420p", os.path.join(d, "raw_1.mov")])
        channels = [{
            "role": "screen_window", "file": "raw_{}.mov".format(i),
            "mode": "window_native", "id": 1000 + i, "app": "A",
            "title": "w{}".format(i),
            "rect": [float(i * (LW + 20)), 0.0, float(LW), float(LH)],
            "logical_w": float(LW), "logical_h": float(LH),
            "buffer_w": BW, "buffer_h": BH, "t0_monotonic": t0s[i],
        } for i in range(2)]
        open(os.path.join(d, "events.jsonl"), "w").close()
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"fps": FPS, "t0_monotonic": SESSION_T0,
                       "events": "events.jsonl", "width": BW, "height": BH,
                       "logical_w": 1440.0, "logical_h": 900.0,
                       "capture_backend": "sck", "cursor_mode": "system",
                       "mic_index": 0, "key_capture": "activity",
                       "capture_channels": channels}, f)

        out = ren.render(d, out_path=os.path.join(self.root, "o.mp4"),
                         window_zoom=False, window_focus=False, max_height=400)
        # Decode the export's audio and find the burst.
        raw = subprocess.check_output(
            ["ffmpeg", "-v", "error", "-i", out, "-map", "a:0",
             "-f", "s16le", "-ac", "1", "-ar", "8000", "-"])
        pcm = np.frombuffer(raw, dtype="<i2").astype(float) / 32768.0
        self.assertGreater(pcm.size, 0, "export carried no audio track")
        win = 80                                     # 10 ms at 8 kHz
        env = np.abs(pcm[:(pcm.size // win) * win].reshape(-1, win)).max(axis=1)
        # Self-scaling threshold: AAC round-tripping a gated sine leaves the
        # burst well under full scale (measured 0.127), so a fixed level finds
        # nothing while the burst is plainly there against the silence.
        self.assertGreater(float(env.max()), 0.01,
                           "exported audio is silent end to end")
        loud = np.nonzero(env > 0.25 * env.max())[0]
        self.assertTrue(loud.size, "no burst found in the exported audio")
        onset = float(loud[0]) * win / 8000.0
        # The picture shows FLASH_MEDIA at (FLASH_MEDIA*fps - origin)/fps, and
        # the mic must be there too. Anchored at 0 it is origin/fps late.
        want = (int(round(FLASH_MEDIA * FPS)) - origin) / float(FPS)
        self.assertAlmostEqual(
            onset, want, delta=0.05,
            msg="mic burst at {:.4f}s, picture shows it at {:.4f}s; "
                "{:.4f}s means audio_skip_s stopped seeking channel 0 to the "
                "composite origin".format(onset, want, want + origin / FPS))


if __name__ == "__main__":
    unittest.main()
