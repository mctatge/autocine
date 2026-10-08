"""CLI plumbing: `record --capture-window`, the `devices` window list, and
`render --window-layout`.

No ffmpeg, no permissions, no recording: `rec.Recorder`, `ren.render` and
every `devices` lookup are faked, so these pin the argument wiring and the
refusal paths.
"""

import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

from autocine import cli
from autocine import edits as _edits

# "Capture screen 0" is index 1 here on purpose -- the screen device is NOT
# guaranteed to be index 0, and this machine really does list two of them.
DEVS = {"video": [(0, "FaceTime HD Camera"),
                  (1, "Capture screen 0"),
                  (2, "Capture screen 1")],
        "audio": [(0, "MacBook Air Microphone")]}

ENTRY = {"id": 42, "app": "Google Chrome", "title": "Docs",
         "label": "Google Chrome — Docs",
         "x": 10.0, "y": 20.0, "w": 800.0, "h": 600.0,
         "display_id": 1, "display_origin": [0.0, 0.0], "main_display": True}


def _fake_recorder(calls):
    """Recorder stand-in that records its construction kwargs."""

    class _R(object):
        def __init__(self, *args, **kwargs):
            calls.append(kwargs)

        def start(self, countdown=3, duration=None):
            return "/tmp/fake-session"

    return _R


class RecordCaptureWindow(unittest.TestCase):
    """`record --capture-window <id>` -> rec.Recorder(capture_window=entry)."""

    def _run(self, argv, entry=ENTRY):
        """Run `main(argv)` with the device layer faked. Returns
        (exit_code, recorder_kwargs_list, window_rect_points_mock, stderr)."""
        calls = []
        rect = mock.Mock(return_value=entry)
        err = io.StringIO()
        with mock.patch.object(cli.dev, "list_avf_devices",
                               return_value=DEVS), \
                mock.patch.object(cli.dev, "window_rect_points", rect), \
                mock.patch.object(cli.rec, "Recorder", _fake_recorder(calls)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, calls, rect, err.getvalue()

    # --- the off switch: absent flag must reproduce today's behavior --------
    def test_absent_flag_passes_none_and_never_looks_a_window_up(self):
        code, calls, rect, _ = self._run(["record", "--duration", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertIsNone(calls[0]["capture_window"])
        rect.assert_not_called()

    def test_resolved_window_is_handed_to_the_recorder(self):
        code, calls, rect, _ = self._run(
            ["record", "--capture-window", "42", "--duration", "1"])
        self.assertEqual(code, 0)
        self.assertIs(calls[0]["capture_window"], ENTRY)
        self.assertEqual(rect.call_args[0][0], 42)

    def test_lookup_excludes_our_own_pid(self):
        # Otherwise the recording bar itself is offerable/resolvable.
        import os
        _, _, rect, _ = self._run(
            ["record", "--capture-window", "42", "--duration", "1"])
        self.assertIn(os.getpid(), tuple(rect.call_args[1]["exclude_pids"]))

    # --- refusals: never silently record full-screen instead ---------------
    def test_unresolvable_id_exits_nonzero_without_recording(self):
        code, calls, _, err = self._run(
            ["record", "--capture-window", "999", "--duration", "1"],
            entry=None)
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        self.assertIn("999", err)

    def test_secondary_display_is_refused_before_any_lookup(self):
        # Multi-display is out of scope for v1: meta's logical_w/h always
        # describe the MAIN display, so this crop would use the wrong scale.
        code, calls, rect, err = self._run(
            ["record", "--display", "2", "--capture-window", "42",
             "--duration", "1"])
        self.assertEqual(code, 2)
        self.assertEqual(calls, [])
        rect.assert_not_called()
        self.assertIn("--display 1", err)

    def test_explicit_primary_display_is_allowed(self):
        code, calls, _, _ = self._run(
            ["record", "--display", "1", "--capture-window", "42",
             "--duration", "1"])
        self.assertEqual(code, 0)
        self.assertIs(calls[0]["capture_window"], ENTRY)

    def test_no_screen_device_detected_skips_the_display_guard(self):
        # ffmpeg listed no "Capture screen" device but the user knows the
        # index -- don't refuse on a comparison we can't actually make.
        devs = {"video": [(0, "FaceTime HD Camera")], "audio": []}
        calls = []
        rect = mock.Mock(return_value=ENTRY)
        with mock.patch.object(cli.dev, "list_avf_devices",
                               return_value=devs), \
                mock.patch.object(cli.dev, "window_rect_points", rect), \
                mock.patch.object(cli.rec, "Recorder", _fake_recorder(calls)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["record", "--display", "7", "--capture-window",
                             "42", "--duration", "1"])
        self.assertEqual(code, 0)
        self.assertIs(calls[0]["capture_window"], ENTRY)


class DevicesWindowList(unittest.TestCase):
    """`studio devices` has to be where --capture-window ids come from."""

    def _run(self, windows, displays=({"id": 1, "main": True},)):
        out = io.StringIO()
        with mock.patch.object(cli.dev, "list_avf_devices",
                               return_value=DEVS), \
                mock.patch.object(cli.dev, "main_display_points",
                                  return_value=(1440.0, 900.0, "quartz")), \
                mock.patch.object(cli.dev, "list_windows",
                                  return_value=list(windows)), \
                mock.patch.object(cli.dev, "displays_points",
                                  return_value=list(displays)), \
                contextlib.redirect_stdout(out):
            code = cli.main(["devices"])
        self.assertEqual(code, 0)
        return out.getvalue()

    def test_lists_id_label_and_point_rect(self):
        text = self._run([ENTRY])
        self.assertIn("[42]", text)
        self.assertIn("Google Chrome — Docs", text)
        self.assertIn("(10,20 800x600 pt)", text)
        self.assertIn("--capture-window", text)

    def test_front_to_back_order_is_preserved(self):
        second = dict(ENTRY, id=7, label="Terminal")
        text = self._run([ENTRY, second])
        self.assertLess(text.index("[42]"), text.index("[7]"))

    def test_empty_list_degrades_to_a_note_not_a_bare_header(self):
        text = self._run([])
        self.assertIn("no capturable windows", text)
        # Don't advertise a flag that has no usable id.
        self.assertNotIn("--capture-window", text)

    def test_without_quartz_says_so(self):
        text = self._run([], displays=())
        self.assertIn("unavailable", text)
        self.assertNotIn("--capture-window", text)


class RecordDisplayValidation(unittest.TestCase):
    """`--display N` warns when N no longer names a screen.

    avfoundation indices renumber whenever a video device connects or
    disconnects, so a number copied from an older `devices` listing can now
    be a webcam -- and recording it produces video of the user's face. The
    CLI WARNS rather than overriding: unlike the bar's remembered pick, this
    number was typed for this run, and silently recording a different device
    than the one asked for would be its own bug.
    """

    def _run(self, argv):
        calls = []
        err = io.StringIO()
        with mock.patch.object(cli.dev, "list_avf_devices",
                               return_value=DEVS), \
                mock.patch.object(cli.rec, "Recorder",
                                  _fake_recorder(calls)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, err.getvalue()

    def test_a_camera_index_warns_and_names_the_real_screen(self):
        code, err = self._run(["record", "--display", "0", "--duration", "1"])
        self.assertEqual(code, 0)
        self.assertIn("warning", err)
        self.assertIn("isn't a screen device", err)
        self.assertIn("the screen is index 1", err)

    def test_an_unknown_index_warns_too(self):
        _, err = self._run(["record", "--display", "9", "--duration", "1"])
        self.assertIn("isn't a screen device", err)

    def test_a_screen_index_is_silent(self):
        for idx in ("1", "2"):
            _, err = self._run(
                ["record", "--display", idx, "--duration", "1"])
            self.assertNotIn("isn't a screen device", err)

    def test_auto_detect_is_silent(self):
        _, err = self._run(["record", "--duration", "1"])
        self.assertNotIn("isn't a screen device", err)

    def test_the_warning_never_blocks_the_recording(self):
        # It's advice, not a refusal -- the user may know something we don't
        # (a device the listing parser missed, say).
        calls = []
        err = io.StringIO()
        with mock.patch.object(cli.dev, "list_avf_devices",
                               return_value=DEVS), \
                mock.patch.object(cli.rec, "Recorder",
                                  _fake_recorder(calls)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            code = cli.main(["record", "--display", "0", "--duration", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)   # it really did start


class RenderWindowLayoutFlag(unittest.TestCase):
    """`render --window-layout <name>` -> `ren.render(window_layout=...)`.

    The flag is the far end of a chain (argparse choices -> args ->
    `cli._render_kwargs` -> `ren.render` -> framing's dispatch table) whose
    every hop fails silently: a name that doesn't survive one of them doesn't
    raise, it lands on the grid. argparse takes its `choices` from
    `edits._WINDOW_LAYOUTS` precisely so a name the editor and MCP accept
    can't be hard-rejected here, which is what these pin from the CLI side.
    """

    # `render` reads the real session dir for edits.json, so the describe
    # call is the only thing that needs the video -- fake it and the rest of
    # _cmd_render runs for real against a temp dir.
    INFO = {"duration": 10.0, "click_times": [1.0, 2.0]}

    def _run(self, argv, saved=None):
        """Run `main(["render", <temp session>] + argv)` with the render
        pipeline faked. Returns (exit code, render kwargs, stderr)."""
        session = tempfile.mkdtemp(prefix="cli-render-")
        self.addCleanup(shutil.rmtree, session, True)
        if saved is not None:
            _edits.save_edits(session, {"render": saved},
                              duration=self.INFO["duration"])
        calls = []
        err = io.StringIO()

        def fake_render(_session, **kwargs):
            calls.append(kwargs)

        with mock.patch.object(cli.ren, "describe_session",
                               return_value=dict(self.INFO)), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render", fake_render), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(err):
            code = cli.main(["render", session] + argv)
        return code, calls, err.getvalue()

    def test_every_arrangement_reaches_the_renderer(self):
        # Looping the canonical tuple keeps this from drifting, so state the
        # premise too -- a tuple that lost the presets would make the loop
        # pass by having nothing left to check.
        self.assertTrue({"feature", "row", "column"}
                        .issubset(_edits._WINDOW_LAYOUTS))
        for name in _edits._WINDOW_LAYOUTS:
            code, calls, _ = self._run(["--window-layout", name])
            self.assertEqual(code, 0, name)
            self.assertEqual(calls[0]["window_layout"], name)

    def test_absent_flag_with_nothing_saved_is_the_grid(self):
        # The off switch: a render that never mentions the arrangement must
        # reach ren.render() exactly as it did before the presets existed.
        _, calls, _ = self._run([])
        self.assertEqual(calls[0]["window_layout"], "grid")

    def test_absent_flag_uses_the_arrangement_the_session_saved(self):
        # Same relationship --zoom has to saved zoom ranges: the editor/MCP
        # own the value, the flag is a per-render override.
        _, calls, _ = self._run([], saved={"window_layout": "column"})
        self.assertEqual(calls[0]["window_layout"], "column")

    def test_the_flag_overrides_the_saved_arrangement(self):
        _, calls, _ = self._run(["--window-layout", "feature"],
                                saved={"window_layout": "column"})
        self.assertEqual(calls[0]["window_layout"], "feature")

    def test_badge_erase_defaults_on_and_both_switches_reach_render(self):
        # Same three-state chain as --screen-focus: absent means "whatever the
        # session saved", which itself defaults ON. The capture indicator is
        # an artefact of how occlusion-free capture works, not a look anyone
        # picked, so a render that never mentions it still erases it.
        _, calls, _ = self._run([])
        self.assertTrue(calls[0]["badge_erase"])
        _, calls, _ = self._run(["--no-badge-erase"])
        self.assertFalse(calls[0]["badge_erase"])
        _, calls, _ = self._run([], saved={"badge_erase": False})
        self.assertFalse(calls[0]["badge_erase"])
        _, calls, _ = self._run(["--badge-erase"],
                                saved={"badge_erase": False})
        self.assertTrue(calls[0]["badge_erase"])

    def test_render_help_survives_percent_expansion(self):
        # `--help` is the only place argparse EXPANDS a help string
        # (`help % dict(...)`), so a literal percent sign in one raises
        # ValueError there and NOWHERE else: import is clean, every render
        # still works, and the only broken thing is the documentation of the
        # flag you went looking for. This help quotes measured coverage
        # percentages (escaped as %%), so the trap is live rather than
        # hypothetical -- and no wording is pinned, only that it formats.
        out = io.StringIO()
        with contextlib.redirect_stdout(out), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["render", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertIn("--window-layout", out.getvalue())

    def test_an_unknown_arrangement_is_refused_by_argparse(self):
        # argparse HARD-rejects rather than coercing, unlike every other hop.
        # That's deliberate: a typo at the terminal should say so, not render
        # a grid the user didn't ask for.
        err = io.StringIO()
        with contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                cli.main(["render", "/nonexistent", "--window-layout", "cascade"])
        self.assertEqual(ctx.exception.code, 2)
        self.assertIn("invalid choice", err.getvalue())
        # ...and the message names the new arrangements, so the fix is visible
        # without opening the docs.
        for name in ("feature", "row", "column"):
            self.assertIn(name, err.getvalue())


class RenderSavedSpeedupsReachRender(unittest.TestCase):
    """Saved `edits.speedups` (MCP add_speedup's force/off spans) reach
    `ren.render` from `studio.py render`, like they do from the web app and
    MCP render_video. The CLI used to setdefault every other saved array
    (zooms, suppressed, windows, crop) but not this one, so a force span
    changed the export from two surfaces and silently not the third -- the
    same one-edits-json-two-outputs split `window_layout` shipped with.
    """

    INFO = {"duration": 10.0, "click_times": [1.0, 2.0]}

    def _run(self, doc=None):
        session = tempfile.mkdtemp(prefix="cli-render-")
        self.addCleanup(shutil.rmtree, session, True)
        if doc is not None:
            _edits.save_edits(session, doc, duration=self.INFO["duration"])
        calls = []

        def fake_render(_session, **kwargs):
            calls.append(kwargs)

        with mock.patch.object(cli.ren, "describe_session",
                               return_value=dict(self.INFO)), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render", fake_render), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["render", session])
        return code, calls

    def test_a_saved_force_span_reaches_the_renderer(self):
        code, calls = self._run(doc={"speedups": [
            {"start": 1.5, "end": 3.0, "mode": "force", "rate": 8.0}]})
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        spans = calls[0]["speedups"]
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0]["mode"], "force")
        self.assertEqual(spans[0]["start"], 1.5)
        self.assertEqual(spans[0]["end"], 3.0)
        self.assertEqual(spans[0]["rate"], 8.0)

    def test_nothing_saved_passes_no_overrides(self):
        # The off switch: render() treats [] like its None default (identity
        # timemap), so a session without spans behaves exactly as before.
        _, calls = self._run()
        self.assertEqual(calls[0]["speedups"], [])


class RecordRenderSeedsTheCards(unittest.TestCase):
    """`record --render` has to hand `render()` what `studio.py render` would
    hand it a minute later, on the very edits.json the record just seeded.

    Threading only the seeded RENDER BLOCK is not enough and fails silently:
    `render.render` gates the whole cards path on `bool(windows)` and reads
    `window_focus` only inside it, so a `--capture-window a b --render` came
    out as the plain whole-screen video with the seeded camera inert, while
    `studio.py render <dir>` on the same session exported the desktop
    composite. Pin both halves.
    """

    SPECS = [{"x": 0, "y": 0, "w": 100, "h": 80, "window_id": 7},
             {"x": 200, "y": 0, "w": 100, "h": 80, "window_id": 8}]

    def _session(self):
        d = tempfile.mkdtemp(prefix="cli-rec-seed-")
        self.addCleanup(shutil.rmtree, d, True)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"fps": 30, "raw": "raw.mov"}, f)
        return d

    def _args(self, argv):
        """A real parsed `record` namespace (defaults included)."""
        captured = {}

        def grab(a):
            captured["args"] = a
            return 0

        with mock.patch.object(cli, "_cmd_record", side_effect=grab), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            cli.main(["record"] + argv)
        return captured["args"]

    def _kwargs(self, session, argv=None, specs=None, multi_native=False):
        args = self._args(argv or [])
        with mock.patch.object(cli.ren, "capture_window_specs",
                               return_value=(self.SPECS if specs is None
                                             else specs)):
            return cli._record_render_kwargs(session, args,
                                             multi_native=multi_native)

    def test_the_seeded_cards_reach_render(self):
        d = self._session()
        kw = self._kwargs(d)
        self.assertIn("windows", kw, "the cards never reached render(), so "
                                     "the take exported as a whole screen")
        self.assertEqual([w["window_id"] for w in kw["windows"]], [7, 8])
        self.assertTrue(kw["window_focus"])
        self.assertFalse(kw["window_zoom"])
        self.assertEqual(kw["window_layout"], "desktop")

    def test_it_matches_what_studio_render_would_use(self):
        """The actual invariant: the record path and the render path resolve
        the same session to the same cards + camera."""
        d = self._session()
        rec_kw = self._kwargs(d)
        doc = _edits.load_edits(d)          # what the record path just seeded
        render_kw = cli._render_kwargs(self._args([]), doc.get("render") or {})
        render_kw.setdefault("windows", list(doc.get("windows") or []))
        for key in ("windows", "window_focus", "window_zoom", "window_layout"):
            self.assertEqual(rec_kw[key], render_kw[key], key)

    def test_an_explicit_window_flag_still_wins(self):
        d = self._session()
        kw = self._kwargs(d, ["--window", "0,0,10,10"])
        self.assertEqual(len(kw["windows"]), 1, "the seeded cards overrode "
                                                "the rect that was typed")
        self.assertEqual(
            [kw["windows"][0][k] for k in ("x", "y", "w", "h")],
            [0.0, 0.0, 10.0, 10.0])

    def test_a_whole_screen_take_is_left_alone(self):
        # No pick: nothing seeded, no edits.json written, no cards.
        d = self._session()
        kw = self._kwargs(d, specs=[])
        self.assertEqual(kw["windows"], [])
        self.assertFalse(kw["window_focus"])
        self.assertFalse(_edits.has_edits(d))

    def test_a_multi_native_pick_seeds_the_camera_with_no_cards(self):
        # Its cards ARE the recorded channels, so `windows` stays empty and
        # render dispatches off the manifest -- but the camera still seeds.
        d = self._session()
        kw = self._kwargs(d, specs=[], multi_native=True)
        self.assertEqual(kw["windows"], [])
        self.assertTrue(kw["window_focus"])
        self.assertEqual(kw["window_layout"], "desktop")


class MultiNativeRenderHonoursSavedEdits(unittest.TestCase):
    """`render <multi-native-dir>` reads the session's edits.json like every
    other render path. The short-circuit for a `capture_channels` manifest
    used to pass args-only, so window_zoom / window_focus / window_layout
    turned on in the editor silently vanished on a plain `studio.py render` --
    one edits.json making two different videos. These pin the fallback (and
    that an explicit flag still wins), plus the desktop seed for a take never
    opened in the editor.
    """

    def _session(self, saved_render, initialized=True):
        d = tempfile.mkdtemp(prefix="cli-mn-")
        self.addCleanup(shutil.rmtree, d, True)
        with open(os.path.join(d, "meta.json"), "w") as f:
            # A manifest is all the short-circuit needs; the files never open
            # because describe_session is mocked.
            json.dump({"capture_channels": [
                {"role": "screen_window", "file": "raw_0.mov"},
                {"role": "screen_window", "file": "raw_1.mov"}]}, f)
        doc = _edits.default_edits()
        doc["capture_windows_initialized"] = initialized
        doc["render"].update(saved_render)
        _edits.save_edits(d, doc, duration=12.0)
        return d

    def _run(self, session, argv):
        calls = []
        with mock.patch.object(cli.ren, "describe_session",
                               return_value={"duration": 12.0}), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render",
                                  side_effect=lambda _s, **kw: calls.append(kw)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["render", session] + argv)
        return code, calls

    def test_saved_zoom_focus_and_layout_reach_render(self):
        d = self._session({"window_zoom": True, "window_focus": True,
                           "window_layout": "desktop"})
        code, calls = self._run(d, [])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["window_zoom"])
        self.assertTrue(calls[0]["window_focus"])
        self.assertEqual(calls[0]["window_layout"], "desktop")

    def test_an_explicit_flag_still_overrides_the_saved_block(self):
        d = self._session({"window_zoom": True, "window_layout": "desktop"})
        code, calls = self._run(d, ["--window-layout", "grid"])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0]["window_layout"], "grid")   # flag wins
        self.assertTrue(calls[0]["window_zoom"])              # saved survives

    def test_a_take_never_opened_seeds_the_desktop_default(self):
        # No capture_windows_initialized and no `windows`: the render seeds
        # the desktop arrangement (cards ARE the screen windows) so the CLI
        # and the editor agree, instead of falling to the tiny-thumbnail grid.
        d = self._session({}, initialized=False)
        code, calls = self._run(d, [])
        self.assertEqual(code, 0)
        self.assertEqual(calls[0]["window_layout"], "desktop")


class RenderSavedTrimReachesRender(unittest.TestCase):
    """A saved trim on edits.json must reach `ren.render` from the CLI too.

    The editor (studio_app.start_render) and MCP (render_video) each read
    edits.json's `trim` and pass trim_start/trim_end through explicitly; the
    CLI used to call render() with no trim kwargs, so `render.render`'s
    signature defaults (0.0 / None) exported the FULL take -- one edits.json
    producing two different lengths depending on whether Export or
    `studio.py render` shipped it. Exactly the split cli.py's crop comment
    describes ("has to survive `studio.py render`"), applied to trim.
    """

    INFO = {"duration": 30.0, "click_times": [1.0]}

    def _run(self, trim=None, meta=None):
        session = tempfile.mkdtemp(prefix="cli-trim-")
        self.addCleanup(shutil.rmtree, session, True)
        if meta is not None:
            with open(os.path.join(session, "meta.json"), "w") as f:
                json.dump(meta, f)
        doc = _edits.default_edits()
        if trim is not None:
            doc["trim"] = dict(trim)
        _edits.save_edits(session, doc, duration=self.INFO["duration"])
        calls = []
        with mock.patch.object(cli.ren, "describe_session",
                               return_value=dict(self.INFO)), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render",
                                  side_effect=lambda _s, **kw: calls.append(kw)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["render", session])
        return code, calls

    # --- whole-screen path -------------------------------------------------
    def test_saved_trim_reaches_render(self):
        code, calls = self._run(trim={"start": 2.5, "end": 18.0})
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["trim_start"], 2.5)
        self.assertEqual(calls[0]["trim_end"], 18.0)

    def test_default_trim_matches_the_other_surfaces(self):
        # The off switch. A session with no trim edit MUST pass the same
        # normalized (start=0.0, end=duration) that studio_app.start_render
        # and mcp_server._tool_render_video pass from an unedited doc, so
        # all three sinks produce the same length -- the "one edits.json,
        # two outputs" bug this fix closes.
        _, calls = self._run(trim=None)
        self.assertEqual(calls[0]["trim_start"], 0.0)
        # _normalize_trim substitutes None with the session duration on
        # load, and studio_app/mcp pass that through; render() then clamps
        # to media length, so this is bit-equivalent to "through the end".
        self.assertEqual(calls[0]["trim_end"], self.INFO["duration"])

    # --- multi-native path (manifest short-circuit) ------------------------
    # The other short-circuit that skipped saved edits: same fallback shape,
    # so pin it here too. _render_multi_native drops trim via **_ignored
    # today, so this only proves the plumbing; adding trim to the dispatcher
    # later needs no CLI change.
    MULTI_META = {"capture_channels": [
        {"role": "screen_window", "file": "raw_0.mov"},
        {"role": "screen_window", "file": "raw_1.mov"}]}

    def test_saved_trim_reaches_render_on_multi_native_path(self):
        code, calls = self._run(trim={"start": 3.0, "end": 9.5},
                                meta=self.MULTI_META)
        self.assertEqual(code, 0)
        self.assertEqual(calls[0]["trim_start"], 3.0)
        self.assertEqual(calls[0]["trim_end"], 9.5)


class RenderSavedRenderOptionsReachRender(unittest.TestCase):
    """`render` with saved render options must reach `ren.render` -- the
    one-edits-json-two-outputs bug class (docs/architecture.md, "ALL FOUR sinks").
    `_render_kwargs` used to read `args.<name>` directly for several options,
    silently dropping what the web editor and MCP `set_render_options` had
    written. Same shape as `RenderWindowLayoutFlag` -- the flag is a per-render
    override, the saved value is the session's preference, and the pre-edits
    default is the fallback.
    """
    INFO = {"duration": 10.0, "click_times": [1.0, 2.0]}

    def _run(self, argv, saved=None):
        session = tempfile.mkdtemp(prefix="cli-render-")
        self.addCleanup(shutil.rmtree, session, True)
        if saved is not None:
            _edits.save_edits(session, {"render": saved},
                              duration=self.INFO["duration"])
        calls = []
        with mock.patch.object(cli.ren, "describe_session",
                               return_value=dict(self.INFO)), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render",
                                  side_effect=lambda _s, **kw: calls.append(kw)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["render", session] + argv)
        return code, calls

    def test_saved_background_reaches_render_and_promotes_style(self):
        # A background implies the framed look; the coupling must survive
        # the saved fallback too -- otherwise a saved background paints
        # behind a 'clean' full-bleed frame (_resolve_style).
        _, calls = self._run([], saved={"background": "midnight"})
        self.assertEqual(calls[0]["background"], "midnight")
        self.assertEqual(calls[0]["style"], "framed")

    def test_saved_aspect_reaches_render(self):
        # `9:16` is exactly the divergence case: a non-auto aspect saved
        # from the editor's vertical export panel would render 16:10 on
        # `studio.py render`.
        _, calls = self._run([], saved={"aspect": "9:16"})
        self.assertEqual(calls[0]["aspect"], "9:16")

    def test_saved_music_and_click_sound_reach_render(self):
        _, calls = self._run([], saved={"music": "/tmp/m.mp3",
                                        "click_sound": "/tmp/c.wav"})
        self.assertEqual(calls[0]["music"], "/tmp/m.mp3")
        self.assertEqual(calls[0]["click_sound"], "/tmp/c.wav")

    def test_saved_click_color_reaches_render(self):
        _, calls = self._run([], saved={"click_color": "#34d399"})
        self.assertEqual(calls[0]["click_params"], {"color": "#34d399"})

    def test_saved_speedup_and_gates_reach_render(self):
        # The whole speedup family: `speedup:true` gates auto detection on,
        # and both gates being False changes the plan. Before this fix, the
        # flag defaults (store_true False, store_false True) clobbered the
        # saved values on `studio.py render`.
        _, calls = self._run([], saved={"speedup": True,
                                        "speedup_silence_gate": False,
                                        "speedup_motion_gate": False})
        self.assertTrue(calls[0]["speedup"])
        self.assertFalse(calls[0]["speedup_silence_gate"])
        self.assertFalse(calls[0]["speedup_motion_gate"])

    def test_saved_speedup_rate_reaches_render_not_flag_default(self):
        # The concrete failure the fix targets: an add_speedup force span
        # with rate:null runs at `render.speedup_rate` (render.py:139
        # `max(1.0, float(o.get("rate") or rate))`), so the flag default
        # 6.0 silently overrode a saved 3.0 on `studio.py render`.
        _, calls = self._run([], saved={"speedup_rate": 3.0})
        self.assertEqual(calls[0]["speedup_rate"], 3.0)

    def test_saved_speedups_array_reaches_render(self):
        # Manual force/off spans (MCP add_speedup) are the TOP-LEVEL
        # `speedups` array -- render.py honors "force" ones even when
        # `render.speedup=False`. Passed at the _cmd_render setdefault
        # level, not through _render_kwargs. Missing setdefault silently
        # dropped MCP-authored force spans on `studio.py render`.
        session = tempfile.mkdtemp(prefix="cli-render-")
        self.addCleanup(shutil.rmtree, session, True)
        doc = _edits.default_edits()
        doc["speedups"] = [{"start": 1.0, "end": 2.0,
                            "mode": "force", "rate": 4.0}]
        _edits.save_edits(session, doc, duration=self.INFO["duration"])
        calls = []
        with mock.patch.object(cli.ren, "describe_session",
                               return_value=dict(self.INFO)), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render",
                                  side_effect=lambda _s, **kw: calls.append(kw)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["render", session])
        self.assertEqual(code, 0)
        self.assertEqual(len(calls[0]["speedups"]), 1)
        self.assertEqual(calls[0]["speedups"][0]["mode"], "force")
        self.assertEqual(calls[0]["speedups"][0]["rate"], 4.0)

    def test_saved_camera_params_reach_render(self):
        # `always_zoomed` and `zoom_speed` arrive via the camera `params`
        # dict (render()'s own kwarg for camera-DEFAULTS overrides), and
        # `overview` has NO CLI flag at all -- it is saved-only. All three
        # need saved fallbacks or the editor's Camera panel silently no-ops
        # on `studio.py render`.
        _, calls = self._run([], saved={"always_zoomed": True,
                                        "zoom_speed": "fast",
                                        "overview": False})
        p = calls[0].get("params") or {}
        self.assertIs(p.get("always_zoomed"), True)
        self.assertEqual(p.get("zoom_speed"), "fast")
        self.assertIs(p.get("overview"), False)

    def test_zoom_default_matches_the_saved_default(self):
        # --zoom used to default to 2.0 in argparse while
        # `_DEFAULT_RENDER["zoom"] = 2.2`, so even a never-edited session
        # diverged from what the web/MCP produced. The alignment is the
        # saved fallback -- an unpassed --zoom on a default doc uses 2.2.
        _, calls = self._run([])
        self.assertEqual(calls[0]["max_zoom"],
                         _edits._DEFAULT_RENDER["zoom"])

    def test_explicit_flags_still_override_saved(self):
        _, calls = self._run(
            ["--zoom", "3.5", "--speedup", "--speedup-rate", "8.0",
             "--music", "/flag.mp3", "--click-color", "yellow",
             "--aspect", "1:1", "--background", "sunset",
             "--always-zoomed", "--zoom-speed", "slow"],
            saved={"zoom": 2.2, "speedup": False, "speedup_rate": 3.0,
                   "music": "/saved.mp3", "click_color": "#000000",
                   "aspect": "9:16", "background": "midnight",
                   "always_zoomed": False, "zoom_speed": "fast"})
        self.assertEqual(calls[0]["max_zoom"], 3.5)
        self.assertTrue(calls[0]["speedup"])
        self.assertEqual(calls[0]["speedup_rate"], 8.0)
        self.assertEqual(calls[0]["music"], "/flag.mp3")
        self.assertEqual(calls[0]["click_params"], {"color": "yellow"})
        self.assertEqual(calls[0]["aspect"], "1:1")
        self.assertEqual(calls[0]["background"], "sunset")
        p = calls[0].get("params") or {}
        self.assertIs(p.get("always_zoomed"), True)
        self.assertEqual(p.get("zoom_speed"), "slow")


class RenderKwargsSurfaceParity(unittest.TestCase):
    """The CLI's `_render_kwargs` + `_cmd_render` setdefaults reach render()
    with the same key set the web/MCP resolvers do. An option resolved by
    one sink and not the other doesn't error -- it falls back to render()'s
    signature default, and one edits.json makes two different videos
    depending on whether you exported from the web app or ran
    `studio.py render` (the `window_layout` and `speedups` classes of bug).
    `tests/test_mcp_server.py`'s `RenderKwargsParityTests` pins MCP<->web;
    this pins CLI<->web.
    """
    INFO = {"duration": 10.0, "click_times": [1.0]}
    # mcp/web resolvers include `offset` and trim; the CLI passes those to
    # render() as its own separate arguments (offset from `--offset`, trim
    # from edits.json trim, wired up at the render() call site), NOT through
    # the resolver's kwarg dict.
    RESOLVER_ONLY = frozenset({"offset", "trim_start", "trim_end"})
    # The web sink names the camera-overrides dict `camera_params`; the CLI
    # (and mcp) use `params`, which is render()'s own kwarg name. Same value,
    # same destination.
    ALIASES = {"camera_params": "params"}

    def test_cli_reaches_render_with_the_web_kwarg_key_set(self):
        from autocine import studio_app
        # A default (never-edited) edits doc is enough: both sinks are pure
        # functions of the doc + args, and this test is about which KEYS
        # they produce, not the values.
        session = tempfile.mkdtemp(prefix="cli-parity-")
        self.addCleanup(shutil.rmtree, session, True)
        root = tempfile.mkdtemp(prefix="cli-parity-root-")
        self.addCleanup(shutil.rmtree, root, True)
        state = studio_app.StudioState(root)
        web_kw = state._render_kwargs_from_edits(
            _edits.normalize_edits({}, duration=self.INFO["duration"]))
        web_keys = {self.ALIASES.get(k, k) for k in web_kw} - self.RESOLVER_ONLY

        calls = []
        with mock.patch.object(cli.ren, "describe_session",
                               return_value=dict(self.INFO)), \
                mock.patch.object(cli.ren, "capture_window_specs",
                                  return_value=[]), \
                mock.patch.object(cli.ren, "render",
                                  side_effect=lambda _s, **kw: calls.append(kw)), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = cli.main(["render", session])
        self.assertEqual(code, 0)
        cli_keys = set(calls[0])

        missing = web_keys - cli_keys
        self.assertEqual(missing, set(),
                         "these render options resolve in the web sink but "
                         "NOT the CLI: {} -- an option added to one resolver "
                         "and not the other silently falls back to render()'s "
                         "signature default (docs/architecture.md ALL FOUR sinks)"
                         .format(sorted(missing)))

if __name__ == "__main__":
    unittest.main()
