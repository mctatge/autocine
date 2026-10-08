"""Unit tests for the stdio MCP server (autocine/mcp_server.py).

Drives McpServer.handle_message directly -- no pipes or subprocess for the
protocol. Session tools run against a synthetic session (testsrc2 raw.mov +
hand-written events.jsonl + meta.json) built in a temp recordings root, so no
macOS screen/input permissions are needed. Skips the session tests if ffmpeg
is unavailable.
"""

import contextlib
import datetime
import io
import inspect
import json
import os
import shutil
import subprocess
import tempfile
import unittest

from autocine import edits as ed
from autocine import mcp_server
from autocine import studio_app
from autocine import transcribe as tx
from autocine.mcp_server import McpServer, PROTOCOL_VERSION
from tests import arrangement_claims as claims

EXPECTED_TOOLS = {
    "list_sessions", "describe_session", "get_edits", "set_render_options",
    "set_trim", "set_crop", "add_zoom", "adjust_zoom", "remove_zoom",
    "add_speedup", "remove_speedup",
    "add_cut", "remove_cut", "set_cuts",
    "add_marker", "remove_marker",
    "list_recorded_windows", "set_windows", "add_window", "remove_window",
    "fit_windows",
    "reset_edits",
    "render_video", "preview_frame",
    "get_transcript", "find_in_transcript",
}


def _req(id_, method, params=None):
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def _call(server, name, arguments=None, id_=1):
    return server.handle_message(_req(id_, "tools/call", {
        "name": name,
        "arguments": arguments if arguments is not None else {},
    }))


def _tool_json(resp):
    """Parse the first text content block of a tool result as JSON."""
    for block in resp["result"]["content"]:
        if block.get("type") == "text":
            return json.loads(block["text"])
    raise AssertionError("no text content in tool result: {}".format(resp))


def _tool_text(resp):
    for block in resp["result"]["content"]:
        if block.get("type") == "text":
            return block["text"]
    raise AssertionError("no text content in tool result: {}".format(resp))


def _bbox(cells):
    """(x0, y0, x1, y1) around a list of (x, y, w, h) card rects."""
    return (min(c[0] for c in cells), min(c[1] for c in cells),
            max(c[0] + c[2] for c in cells), max(c[1] + c[3] for c in cells))


def _probe_frames(path):
    """Video frame count of a rendered file, via ffprobe."""
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v",
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-of", "csv=p=0", path], text=True).strip()
    return int(out)


class ProtocolTests(unittest.TestCase):
    """Protocol-shape tests against an empty recordings root."""

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mcp-proto-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.server = McpServer(recordings_root=self.root)

    def test_initialize_echoes_client_protocol_version(self):
        resp = self.server.handle_message(_req(1, "initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "0"},
        }))
        self.assertEqual(resp["jsonrpc"], "2.0")
        self.assertEqual(resp["id"], 1)
        result = resp["result"]
        self.assertEqual(result["protocolVersion"], "2025-06-18")
        self.assertEqual(result["capabilities"], {"tools": {}})
        self.assertEqual(result["serverInfo"]["name"], "autocine")
        self.assertEqual(result["serverInfo"]["version"], "0.1.0")

    def test_initialize_defaults_protocol_version(self):
        for params in (None, {}, {"protocolVersion": ""}, {"protocolVersion": 42}):
            resp = self.server.handle_message(_req(1, "initialize", params))
            self.assertEqual(resp["result"]["protocolVersion"], PROTOCOL_VERSION)

    def test_ping_returns_empty_object(self):
        resp = self.server.handle_message(_req(7, "ping"))
        self.assertEqual(resp["result"], {})
        self.assertEqual(resp["id"], 7)

    def test_tools_list_shapes(self):
        resp = self.server.handle_message(_req(2, "tools/list"))
        tools = resp["result"]["tools"]
        self.assertEqual({t["name"] for t in tools}, EXPECTED_TOOLS)
        # Guards a duplicate name silently collapsing in the set compare above.
        self.assertEqual(len(tools), len(EXPECTED_TOOLS))
        for tool in tools:
            self.assertTrue(tool["description"].strip())
            schema = tool["inputSchema"]
            self.assertEqual(schema["type"], "object")
            self.assertIsInstance(schema["properties"], dict)
            self.assertIsInstance(schema["required"], list)
            for key in schema["required"]:
                self.assertIn(key, schema["properties"])

    def test_set_render_options_schema_covers_facecam(self):
        # The facecam options were shipped in the editor/CLI but never reached
        # this schema, so the model could not use them; pin schema+enums to the
        # validators edits.py actually applies.
        resp = self.server.handle_message(_req(2, "tools/list"))
        tools = {t["name"]: t for t in resp["result"]["tools"]}
        props = tools["set_render_options"]["inputSchema"]["properties"]
        for key in ("facecam", "facecam_position", "facecam_size",
                    "facecam_shape"):
            self.assertIn(key, props)
        self.assertEqual(set(props["facecam_position"]["enum"]),
                         set(ed._FACECAM_POSITIONS))
        self.assertEqual(set(props["facecam_shape"]["enum"]),
                         set(ed._FACECAM_SHAPES))

    def test_set_render_options_schema_offers_every_window_layout(self):
        # Same failure the facecam pin above guards, one field over: this
        # module keeps its OWN copy of the accepted names (schema enum, write
        # validator, render resolver all read `mcp_server._WINDOW_LAYOUTS`),
        # and a copy that falls behind edits.py rejects an arrangement the
        # editor and CLI both take -- so the model is told a preset doesn't
        # exist while the user is looking at it.
        resp = self.server.handle_message(_req(2, "tools/list"))
        tools = {t["name"]: t for t in resp["result"]["tools"]}
        props = tools["set_render_options"]["inputSchema"]["properties"]
        self.assertEqual(set(props["window_layout"]["enum"]),
                         set(ed._WINDOW_LAYOUTS))
        self.assertEqual(set(mcp_server._WINDOW_LAYOUTS),
                         set(ed._WINDOW_LAYOUTS))
        # An enum entry the description never mentions is an arrangement the
        # model can name but has no reason to pick, which is how 'feature',
        # 'row' and 'column' first shipped -- offered, and described only as
        # a group. Names, not phrasing: this breaks when an arrangement is
        # added, which is exactly when the description has to be rewritten,
        # and survives any rewording of it.
        described = props["window_layout"]["description"]
        for name in ed._WINDOW_LAYOUTS:
            self.assertIn("'{}'".format(name), described)
        # ...and it must send the model to the option the choice actually
        # turns on. The overclaim this replaced ranked the arrangements
        # absolutely -- "feature/row/column fill the frame by construction" --
        # when cards are aspect-locked and get ONE uniform scale, so an
        # arrangement meets the padding on one axis and letterboxes the other
        # by however its bbox aspect differs from the canvas. The ranking
        # inverts with the export aspect, so any absolute one is wrong for
        # half of all exports. `aspect` is an identifier here, not a word
        # choice; what the description says ABOUT each arrangement is checked
        # against a fresh measurement in the two tests below.
        self.assertIn("aspect", described)

    def _window_layout_description(self):
        resp = self.server.handle_message(_req(2, "tools/list"))
        tools = {t["name"]: t for t in resp["result"]["tools"]}
        props = tools["set_render_options"]["inputSchema"]["properties"]
        return props["window_layout"]["description"]

    def test_the_window_layout_description_quotes_only_measured_coverage(self):
        """The description sells the arrangements with numbers, and this is
        the only thing that reads them.

        It shipped with the pre-transpose table -- "desktop 70%" on a 16:10
        canvas, "desktop 27%" on a 9:16 one -- long after `feature` learned to
        transpose and the whole ranking moved. Neither figure is a coverage of
        anything the code produces on any canvas, with two windows or three,
        and everything downstream of it was green: the enum matched, the
        resolvers agreed, the arrangement rendered. Only the sentence the
        model reads was wrong, and its whole job is deciding which
        arrangement to pick.

        So: measure the layouts here (`arrangement_claims`, off
        `autocine.framing`, at test time) and check every percentage the
        description asserts against that. A rewording passes; a figure the
        code no longer produces does not. Attribution inside a sentence is
        deliberately not attempted -- see `percent_claims`.
        """
        described = self._window_layout_description()
        found = claims.percent_claims(described)
        # A description that dropped the numbers entirely would make the loop
        # below vacuous. It is allowed to -- an unquoted figure cannot be
        # stale -- but not silently: this is the one recommendation surface
        # for a decision that inverts with the export aspect, and the numbers
        # are how a model is meant to make it.
        self.assertTrue(found, "the window_layout description no longer "
                               "quotes a single coverage figure")
        for names, pct in found:
            measured = [v for name in names
                        for v in claims.measured_values(name)]
            self.assertTrue(
                any(abs(pct - v) <= claims.PCT_TOL for v in measured),
                "the description claims {:.1f}% for {}; measured today: "
                "{}".format(pct, "/".join(names),
                            ", ".join("%.1f" % v for v in sorted(measured))))

    def test_the_window_layout_description_writes_off_no_layout_that_wins(self):
        """The other half: it must not steer the model AWAY from an
        arrangement the measurements put at the top.

        `row` is never the best of the five for three windows and the
        description is free to say so. `feature` is best on 16:10, 16:9 and
        1:1 and a quarter-point off best on 9:16, so there is no export shape
        it can be written off for -- and the editor's hint was doing exactly
        that ("a waste of a tall" export) from the same stale table.

        Bounded, and the bound is on `arrangement_claims.dismissals`: this
        catches a write-off in so many words, not steering by omission.
        """
        for name, shape, sentence in claims.dismissals(
                self._window_layout_description()):
            gap = claims.gap_from_best(name, shape)
            self.assertGreater(
                gap, claims.DISMISSAL_TOL,
                "the description writes '{}' off for a {} export, but it "
                "measures within {:.1f} points of the best layout there -- "
                "{!r}".format(name, shape, gap, sentence))

    def test_set_windows_describes_the_mode_that_exists(self):
        # STALE COPY, caught late: this description told the model windows
        # mode was "a static grid" with "no per-window zoom" long after
        # window_zoom and window_focus shipped -- and set_windows is the tool
        # an agent uses to CREATE the arrangement, so that was the one place
        # the whole feature was invisible. Pin the option NAMES it has to hand
        # the model onward to (identifiers, not prose); adding another option
        # that shapes this mode should fail here.
        resp = self.server.handle_message(_req(2, "tools/list"))
        tools = {t["name"]: t for t in resp["result"]["tools"]}
        described = tools["set_windows"]["description"]
        for option in ("window_layout", "window_zoom", "window_focus",
                       "fit_windows"):
            self.assertIn(option, described)

    def test_unknown_method_is_method_not_found(self):
        resp = self.server.handle_message(_req(3, "bogus/method"))
        self.assertEqual(resp["error"]["code"], -32601)
        self.assertEqual(resp["id"], 3)

    def test_notifications_return_none(self):
        for method in ("notifications/initialized", "notifications/cancelled",
                       "notifications/whatever"):
            self.assertIsNone(self.server.handle_message(
                {"jsonrpc": "2.0", "method": method}))

    def test_malformed_json_line_is_parse_error(self):
        resp = self.server.handle_line("{this is not json")
        self.assertEqual(resp["error"]["code"], -32700)
        self.assertIsNone(resp["id"])

    def test_non_object_json_line_is_invalid_request(self):
        resp = self.server.handle_line("[1,2,3]")
        self.assertEqual(resp["error"]["code"], -32600)

    def test_tools_call_bad_params_shape_is_invalid_params(self):
        # params not an object
        resp = self.server.handle_message(_req(4, "tools/call", None))
        self.assertEqual(resp["error"]["code"], -32602)
        # missing tool name
        resp = self.server.handle_message(_req(5, "tools/call", {"arguments": {}}))
        self.assertEqual(resp["error"]["code"], -32602)
        # arguments not an object
        resp = self.server.handle_message(_req(6, "tools/call", {
            "name": "list_sessions", "arguments": [1]}))
        self.assertEqual(resp["error"]["code"], -32602)

    def test_unknown_tool_is_invalid_params(self):
        resp = _call(self.server, "no_such_tool")
        self.assertEqual(resp["error"]["code"], -32602)

    def test_missing_required_argument_is_tool_error(self):
        resp = _call(self.server, "describe_session", {})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("session", _tool_text(resp))

    def test_wrong_argument_type_is_tool_error(self):
        resp = _call(self.server, "add_zoom", {
            "session": "x", "start": "zero", "end": 1.0})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("Error:", _tool_text(resp))

    def test_unknown_session_is_tool_error(self):
        resp = _call(self.server, "describe_session", {"session": "nope"})
        self.assertTrue(resp["result"]["isError"])
        self.assertIn("unknown session: nope", _tool_text(resp))

    def test_path_escaping_session_is_tool_error(self):
        for bad in ("../evil", "a/b", "..", "."):
            resp = _call(self.server, "get_edits", {"session": bad})
            self.assertTrue(resp["result"]["isError"], bad)
            self.assertIn("unknown session", _tool_text(resp))

    def test_list_sessions_empty_root(self):
        resp = _call(self.server, "list_sessions")
        self.assertFalse(resp["result"]["isError"])
        self.assertEqual(_tool_json(resp)["sessions"], [])

    def test_list_sessions_reports_the_running_build(self):
        # A stale `studio.py mcp` process serves the code it imported, so the
        # stamp must describe THIS process, not the file on disk right now.
        stamp = _tool_json(_call(self.server, "list_sessions"))["server"]
        self.assertEqual(stamp, mcp_server.SERVER_BUILD)
        self.assertEqual(stamp["version"], mcp_server.SERVER_INFO["version"])
        self.assertEqual(stamp["pid"], os.getpid())
        self.assertEqual(stamp["source"], os.path.abspath(mcp_server.__file__))
        self.assertEqual(
            stamp["built"],
            datetime.datetime.fromtimestamp(
                os.path.getmtime(mcp_server.__file__)
            ).replace(microsecond=0).isoformat())


# Every render option pushed OFF its default, so a resolver that quietly
# returns a constant (or drops the key) can't pass by accident.
_DEVIATING_EDITS = {
    "trim": {"start": 0.5, "end": 9.0},
    "zooms": [{"start": 1.0, "end": 2.0, "level": 3.0}],
    "suppressed": [{"start": 3.0, "end": 4.0}],
    "speedups": [{"start": 5.0, "end": 7.0, "mode": "force", "rate": 4.0}],
    "cuts": [{"start": 8.0, "end": 8.5}],
    "windows": [{"x": 0, "y": 0, "w": 150, "h": 100}],
    "crop": {"x": 10, "y": 20, "w": 200, "h": 120},
    # `focus_initialized` is load-bearing, not decoration: both resolvers hand
    # `focus_ranges` to render() only once the plan is materialized, and
    # without the flag both would resolve None and the parity check would pass
    # on two resolvers that agree only by both declining.
    "focus": [{"start": 1.0, "end": 3.0, "card": 0, "level": "full"}],
    # Both the flag AND a current plan version, or `focus_plan_is_stale`
    # reports the plan needs re-planning and both resolvers hand render()
    # None -- the parity check would then pass on two sides that agree only
    # by both declining.
    "focus_initialized": True,
    "focus_plan_version": ed._FOCUS_PLAN_VERSION,
    # Manual card placement (docs/architecture.md). Authored web-only, but
    # both resolvers have to READ it or a dragged card exports one way from
    # Export and another from render_video. Defaults are [] / {}, so a
    # populated list and a populated per-scene dict are the deviating values.
    "channel_layouts": [{"x": 0.02, "y": 0.55, "w": 0.30, "h": 0.30}],
    "scene_layouts": {"1": [None, {"x": 0.02, "y": 0.55,
                                   "w": 0.30, "h": 0.35}]},
    # A removed (reversibly hidden) card -- default [], so a populated list is
    # the deviating value both resolvers must READ (else Export drops a card the
    # MCP render keeps).
    "hidden_channels": ["raw_1.mov"],
    "render": {
        "zoom": 3.5, "zoom_speed": "fast", "screen_anim": "smooth",
        "offset": 0.25, "style": "framed", "background": "sunset",
        "click_fx": False, "click_color": "#34d399", "spotlight": True,
        "cursor_fx": True, "cursor_size": 1.4, "cursor_erase": True,
        "aspect": "9:16",
        # resolution defaults "auto" (max_height None), so a real height cap
        # is its deviating value.
        "resolution": "1080",
        "always_zoomed": True, "motion_blur": False, "overview": False,
        "typing_zoom": False, "drag_hold": False, "scroll_zoom": False,
        "window_follow": False, "window_layout": "desktop",
        "window_zoom": True, "window_focus": True,
        # screen_focus defaults ON, so False is its deviating value.
        "screen_focus": False,
        # badge_erase likewise defaults ON (macOS's capture indicator is
        # painted out of an occlusion-free take unless you ask for it).
        "badge_erase": False,
        "facecam": False, "facecam_position": "bottom-right",
        "facecam_size": 0.3, "facecam_shape": "rounded",
        "facecam_border": 0.05,
        "speedup": True, "speedup_rate": 3.0,
        "speedup_silence_gate": False, "speedup_motion_gate": False,
        "fade": 0.4, "music": "/tmp/track.mp3",
        # Both sound fields default to null (= the built-in sound), so a
        # file path is their deviating value; sfx_volume defaults to 1.0.
        "click_sound": "/tmp/click.wav",
        "key_sound": "/tmp/key.wav",
        "sfx_volume": 0.4,
        "gif": True, "gif_fps": 24, "gif_width": 800,
    },
}


class RenderKwargsParityTests(unittest.TestCase):
    """`mcp_server.McpServer._render_kwargs` is a hand-written second copy of
    `studio_app.StudioState._render_kwargs_from_edits`, and each render/preview
    call site spells its kwargs out by hand. An option resolved on one side and
    not the other does NOT error -- it falls back to render()'s signature
    default, so the same edits.json renders a different video depending on
    whether you exported from the web app or asked the MCP server. That is how
    `speedup`/`speedups` (dropped by MCP entirely, including spans written by
    its own add_speedup tool) and `facecam_border` (ring shown in the web
    preview, missing from MCP's) both shipped. Pin the two resolvers together.

    Needs no session on disk: both resolvers are pure functions of an edits doc.
    """

    # mcp_server names the camera-overrides dict `params` (render()'s own kwarg
    # name); studio_app names it `camera_params`. Same value, same destination.
    ALIASES = {"camera_params": "params"}

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mcp-parity-")
        self.addCleanup(shutil.rmtree, self.root, True)
        self.server = McpServer(recordings_root=self.root)
        self.state = studio_app.StudioState(self.root)
        self.doc = ed.normalize_edits(_DEVIATING_EDITS, duration=10.0)

    def _both(self, doc):
        return (self.server._render_kwargs(doc),
                self.state._render_kwargs_from_edits(doc))

    def test_key_sets_match(self):
        mcp_kw, web_kw = self._both(self.doc)
        self.assertEqual(
            set(mcp_kw), {self.ALIASES.get(k, k) for k in web_kw},
            "mcp_server._render_kwargs and studio_app's resolver disagree on "
            "WHICH options exist -- whichever side is missing a key renders "
            "with render()'s default instead")

    def test_every_option_resolves_to_the_same_value(self):
        mcp_kw, web_kw = self._both(self.doc)
        for key, value in web_kw.items():
            self.assertEqual(mcp_kw[self.ALIASES.get(key, key)], value,
                             "render option '{}' resolves differently in "
                             "mcp_server".format(key))

    def test_every_window_layout_resolves_identically_on_both_surfaces(self):
        """The fixture pins one arrangement ("desktop"); the accepted set is
        five, and each surface validates against its own copy of it. A name
        one resolver knows and the other doesn't isn't an error on either
        side -- both fall back to "grid" -- so the MCP tool reports the
        preset saved and Export quietly renders the old arrangement."""
        self.assertTrue({"feature", "row", "column"}
                        .issubset(ed._WINDOW_LAYOUTS))
        for name in ed._WINDOW_LAYOUTS:
            doc = ed.normalize_edits(
                dict(_DEVIATING_EDITS,
                     render=dict(_DEVIATING_EDITS["render"],
                                 window_layout=name)),
                duration=10.0)
            self.assertEqual(doc["render"]["window_layout"], name)
            mcp_kw, web_kw = self._both(doc)
            self.assertEqual(mcp_kw["window_layout"], name)
            self.assertEqual(web_kw["window_layout"], name)

    def test_an_unknown_arrangement_grids_on_both_surfaces(self):
        # The fallback has to agree too: one side gridding while the other
        # honours a junk name is the same divergence, arrived at backwards.
        # (`normalize_edits` already coerces, so this exercises the resolvers'
        # own membership checks against an UNNORMALIZED doc -- which is what
        # the editor's live, unsaved options dict is.)
        doc = ed.normalize_edits(_DEVIATING_EDITS, duration=10.0)
        doc["render"]["window_layout"] = "cascade"
        mcp_kw, web_kw = self._both(doc)
        self.assertEqual(mcp_kw["window_layout"], "grid")
        self.assertEqual(web_kw["window_layout"], "grid")

    def test_the_fixture_deviates_from_every_default(self):
        """Guard on the guard: if an option in `_DEVIATING_EDITS` happened to
        match its default, the two tests above would pass on two resolvers that
        both just returned the default."""
        _, web_kw = self._both(self.doc)
        _, default_kw = self._both(ed.normalize_edits({}, duration=10.0))
        same = sorted(k for k, v in web_kw.items() if default_kw[k] == v)
        self.assertEqual(same, [], "these options are not actually exercised "
                                   "by the fixture: {}".format(same))


class SessionToolTests(unittest.TestCase):
    """End-to-end tool tests against a synthetic 320x200, 1-second session."""

    SESSION = "20260101-000000"

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp(prefix="mcp-sessions-")
        sdir = os.path.join(cls.root, cls.SESSION)
        os.makedirs(sdir)
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump({"fps": 30, "raw": "raw.mov", "events": "events.jsonl",
                       "cursor_mode": "system"}, f)
        events = [
            {"t": 0.10, "type": "move", "x": 40, "y": 40},
            {"t": 0.30, "type": "move", "x": 120, "y": 80},
            {"t": 0.50, "type": "down", "x": 160, "y": 100},
            {"t": 0.70, "type": "move", "x": 200, "y": 120},
            {"t": 0.90, "type": "move", "x": 240, "y": 150},
        ]
        with open(os.path.join(sdir, "events.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=320x200:rate=30",
             "-t", "1.0", "-pix_fmt", "yuv420p",
             os.path.join(sdir, "raw.mov")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            shutil.rmtree(cls.root, ignore_errors=True)
            raise unittest.SkipTest("ffmpeg cannot build synthetic raw.mov: " +
                                    proc.stderr.decode("utf-8", "replace")[-300:])
        cls.session_dir = sdir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.server = McpServer(recordings_root=self.root)
        resp = _call(self.server, "reset_edits", {"session": self.SESSION})
        self.assertFalse(resp["result"]["isError"])

    # -- helpers ------------------------------------------------------------

    def _ok(self, name, arguments=None):
        resp = _call(self.server, name, arguments)
        self.assertNotIn("error", resp, "JSON-RPC error from {}".format(name))
        self.assertFalse(resp["result"]["isError"],
                         "{} failed: {}".format(name, _tool_text(resp)))
        return resp

    def _err(self, name, arguments=None):
        resp = _call(self.server, name, arguments)
        self.assertTrue(resp["result"]["isError"],
                        "{} unexpectedly succeeded".format(name))
        return _tool_text(resp)

    def _edits(self):
        return _tool_json(self._ok("get_edits", {"session": self.SESSION}))

    # -- describe / list ----------------------------------------------------

    # -- transcript ---------------------------------------------------------

    def _seed_transcript(self):
        """A cached transcript.json, so the tools never shell out to ASR."""
        words = [
            {"t": 0.10, "dur": 0.20, "text": "Here", "conf": 0.9},
            {"t": 0.30, "dur": 0.15, "text": "is", "conf": 0.9},
            {"t": 0.45, "dur": 0.30, "text": "the", "conf": 0.9},
            {"t": 0.75, "dur": 0.20, "text": "demo.", "conf": 0.8},
        ]
        doc = {
            "status": "ok", "reason": "",
            "mtime": os.path.getmtime(os.path.join(self.session_dir, "raw.mov")),
            "engine": tx.engine_id(model=tx.find_model() or "ggml-x.bin"),
            "model": "ggml-x.bin", "language": "en", "words": words,
            "segments": [{"t": 0.1, "dur": 0.85, "text": "Here is the demo."}],
        }
        path = os.path.join(self.session_dir, tx.TRANSCRIPT_FILENAME)
        with open(path, "w") as f:
            json.dump(doc, f)
        self.addCleanup(lambda: os.path.exists(path) and os.remove(path))

    def test_get_transcript_returns_words_text_and_silences(self):
        self._seed_transcript()
        payload = _tool_json(self._ok("get_transcript", {"session": self.SESSION}))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["text"], "Here is the demo.")
        self.assertEqual([w["text"] for w in payload["words"]],
                         ["Here", "is", "the", "demo."])
        # word times are session seconds -- the clock set_trim/add_zoom take
        self.assertLess(payload["words"][-1]["t"], payload["duration"])
        self.assertEqual(payload["silences"], [])   # 1s take, all speech

    def test_get_transcript_window_and_words_off(self):
        self._seed_transcript()
        payload = _tool_json(self._ok("get_transcript",
                                      {"session": self.SESSION, "start": 0.46}))
        self.assertEqual([w["text"] for w in payload["words"]], ["the", "demo."])
        compact = _tool_json(self._ok("get_transcript",
                                      {"session": self.SESSION,
                                       "include_words": False}))
        self.assertNotIn("words", compact)
        self.assertTrue(compact["segments"])

    def test_find_in_transcript_gives_spans_the_edit_tools_take(self):
        self._seed_transcript()
        payload = _tool_json(self._ok("find_in_transcript",
                                      {"session": self.SESSION,
                                       "query": "the DEMO"}))
        self.assertEqual(len(payload["matches"]), 1)
        hit = payload["matches"][0]
        self.assertAlmostEqual(hit["t"], 0.45)
        self.assertAlmostEqual(hit["dur"], 0.5)
        self.assertEqual(hit["context"], "Here is the demo.")
        # ...and the span really is usable as an edit range
        self._ok("add_zoom", {"session": self.SESSION, "start": hit["t"],
                              "end": hit["t"] + hit["dur"]})
        self.assertEqual(payload["matches"],
                         _tool_json(self._ok("find_in_transcript",
                                             {"session": self.SESSION,
                                              "query": "the demo"}))["matches"])

    def test_find_in_transcript_miss_is_empty_not_an_error(self):
        self._seed_transcript()
        payload = _tool_json(self._ok("find_in_transcript",
                                      {"session": self.SESSION,
                                       "query": "kubernetes"}))
        self.assertEqual(payload["matches"], [])

    def test_no_transcript_reports_why_instead_of_failing(self):
        # this synthetic session has no audio track (and the machine may have
        # no ASR at all) -- either way the tool answers, with a reason.
        payload = _tool_json(self._ok("get_transcript", {"session": self.SESSION}))
        self.addCleanup(
            lambda: os.path.exists(
                os.path.join(self.session_dir, tx.TRANSCRIPT_FILENAME))
            and os.remove(os.path.join(self.session_dir, tx.TRANSCRIPT_FILENAME)))
        self.assertIn(payload["status"], ("no_audio", "unavailable"))
        self.assertTrue(payload["reason"])
        self.assertEqual(payload["words"], [])

    def test_describe_session_reports_transcript_availability(self):
        self.assertFalse(_tool_json(self._ok(
            "describe_session", {"session": self.SESSION}))["has_transcript"])
        self._seed_transcript()
        self.assertTrue(_tool_json(self._ok(
            "describe_session", {"session": self.SESSION}))["has_transcript"])

    def test_describe_session_full(self):
        payload = _tool_json(self._ok("describe_session",
                                      {"session": self.SESSION}))
        self.assertEqual(payload["session"], self.SESSION)
        self.assertEqual(payload["width"], 320)
        self.assertEqual(payload["height"], 200)
        self.assertAlmostEqual(payload["duration"], 1.0, delta=0.2)
        self.assertEqual(payload["click_count"], 1)
        self.assertEqual(payload["move_count"], 5)  # 4 moves + click appended
        self.assertFalse(payload["has_face"])  # no webcam track in this session
        self.assertIn("edits", payload)
        self.assertIn("chapters", payload)
        self.assertEqual(payload["chapters"], [])
        self.assertEqual(payload["server"], mcp_server.SERVER_BUILD)

    def test_describe_session_carries_a_beat_sheet(self):
        payload = _tool_json(self._ok("describe_session",
                                      {"session": self.SESSION}))
        sheet = payload["beats"]
        self.assertIn("beats", sheet)
        self.assertEqual(sheet["truncated"], 0)
        clicks = [b for b in sheet["beats"] if b["kind"] == "clicks"]
        self.assertEqual(len(clicks), 1)
        self.assertAlmostEqual(clicks[0]["start"], 0.5, delta=0.05)

    def test_describe_session_hides_raw_event_arrays_by_default(self):
        """The beat sheet replaces them; a flat list of floats is telemetry an
        agent cannot act on, and it dominated the payload."""
        lean = _tool_json(self._ok("describe_session",
                                   {"session": self.SESSION}))
        for k in ("click_times", "key_times", "scroll_times"):
            self.assertNotIn(k, lean)
        self.assertNotIn("presets", lean["edits"])

    def test_describe_session_detail_restores_them(self):
        full = _tool_json(self._ok("describe_session",
                                   {"session": self.SESSION, "detail": True}))
        self.assertEqual(len(full["click_times"]), 1)
        self.assertAlmostEqual(full["click_times"][0], 0.5, delta=0.05)
        self.assertIn("key_times", full)
        self.assertIn("scroll_times", full)
        self.assertIn("presets", full["edits"])
        self.assertIn("beats", full)   # detail ADDS, it does not swap

    def test_describe_session_detail_does_not_mutate_saved_edits(self):
        """The lean path pops `presets` off the loaded doc -- if that were
        the live object, describing a session would delete the user's
        presets on the next save."""
        self._ok("describe_session", {"session": self.SESSION})
        after = _tool_json(self._ok("get_edits", {"session": self.SESSION}))
        self.assertIn("presets", after)

    def test_list_sessions_includes_session(self):
        payload = _tool_json(self._ok("list_sessions"))
        sessions = payload["sessions"]
        self.assertEqual(len(sessions), 1)
        entry = sessions[0]
        self.assertEqual(entry["session"], self.SESSION)
        self.assertAlmostEqual(entry["duration"], 1.0, delta=0.2)
        self.assertEqual(entry["click_count"], 1)
        self.assertTrue(entry["has_edits"])  # setUp reset wrote edits.json
        self.assertIn("has_output_mp4", entry)

    # -- zooms ----------------------------------------------------------------

    def test_add_zoom_roundtrip(self):
        payload = _tool_json(self._ok("add_zoom", {
            "session": self.SESSION, "start": 0.2, "end": 0.8, "level": 3.0}))
        zooms = payload["zooms"]
        self.assertEqual(len(zooms), 1)
        self.assertTrue(zooms[0]["id"].startswith("zoom-"))
        self.assertAlmostEqual(zooms[0]["start"], 0.2, places=5)
        self.assertAlmostEqual(zooms[0]["end"], 0.8, places=5)
        self.assertAlmostEqual(zooms[0]["level"], 3.0, places=5)
        self.assertIsNone(zooms[0]["x"])
        self.assertIsNone(zooms[0]["y"])
        # round-trips through edits.json
        self.assertEqual(self._edits()["zooms"], zooms)

    def test_add_zoom_pinned_and_clamped(self):
        payload = _tool_json(self._ok("add_zoom", {
            "session": self.SESSION, "start": -5.0, "end": 99.0,
            "x": 160, "y": 100}))
        z = payload["zooms"][0]
        self.assertEqual(z["start"], 0.0)
        self.assertLessEqual(z["end"], 1.1)  # clamped to clip duration
        self.assertAlmostEqual(z["x"], 160.0, places=5)
        self.assertAlmostEqual(z["y"], 100.0, places=5)
        self.assertAlmostEqual(z["level"], 2.0, places=5)  # default level

    def test_add_zoom_x_without_y_is_error(self):
        text = self._err("add_zoom", {
            "session": self.SESSION, "start": 0.1, "end": 0.6, "x": 100})
        self.assertIn("x and y", text)

    def test_remove_zoom_and_unknown_id(self):
        added = _tool_json(self._ok("add_zoom", {
            "session": self.SESSION, "start": 0.2, "end": 0.6}))
        zoom_id = added["zooms"][0]["id"]
        remaining = _tool_json(self._ok("remove_zoom", {
            "session": self.SESSION, "zoom_id": zoom_id}))
        self.assertEqual(remaining["zooms"], [])
        text = self._err("remove_zoom", {
            "session": self.SESSION, "zoom_id": zoom_id})
        self.assertIn("unknown zoom id", text)

    # -- windows -----------------------------------------------------------

    def test_add_window_roundtrip(self):
        payload = _tool_json(self._ok("add_window", {
            "session": self.SESSION, "x": 10, "y": 20, "w": 100, "h": 80}))
        windows = payload["windows"]
        self.assertEqual(len(windows), 1)
        self.assertTrue(windows[0]["id"].startswith("window-"))
        self.assertAlmostEqual(windows[0]["x"], 10.0)
        self.assertAlmostEqual(windows[0]["y"], 20.0)
        self.assertAlmostEqual(windows[0]["w"], 100.0)
        self.assertAlmostEqual(windows[0]["h"], 80.0)
        # round-trips through edits.json
        self.assertEqual(self._edits()["windows"], windows)

    def test_add_window_requires_all_four_fields(self):
        text = self._err("add_window", {
            "session": self.SESSION, "x": 10, "y": 20, "w": 100})
        self.assertIn("missing required argument: h", text)

    def test_remove_window_and_unknown_id(self):
        added = _tool_json(self._ok("add_window", {
            "session": self.SESSION, "x": 0, "y": 0, "w": 50, "h": 50}))
        window_id = added["windows"][0]["id"]
        remaining = _tool_json(self._ok("remove_window", {
            "session": self.SESSION, "window_id": window_id}))
        self.assertEqual(remaining["windows"], [])
        text = self._err("remove_window", {
            "session": self.SESSION, "window_id": window_id})
        self.assertIn("unknown window id", text)

    def test_set_windows_replaces_whole_array(self):
        self._ok("add_window", {
            "session": self.SESSION, "x": 0, "y": 0, "w": 50, "h": 50})
        payload = _tool_json(self._ok("set_windows", {
            "session": self.SESSION,
            "windows": [
                {"x": 0, "y": 0, "w": 120, "h": 90},
                {"x": 120, "y": 0, "w": 120, "h": 90},
            ]}))
        self.assertEqual(len(payload["windows"]), 2)
        self.assertAlmostEqual(payload["windows"][0]["w"], 120.0)

    def test_set_windows_rejects_non_array(self):
        text = self._err("set_windows", {
            "session": self.SESSION, "windows": "nope"})
        self.assertIn("windows must be an array", text)

    def test_set_windows_rejects_incomplete_rect(self):
        text = self._err("set_windows", {
            "session": self.SESSION,
            "windows": [{"x": 0, "y": 0, "w": 10}]})
        self.assertIn("missing", text)

    def test_set_windows_empty_array_turns_windows_mode_off(self):
        self._ok("add_window", {
            "session": self.SESSION, "x": 0, "y": 0, "w": 50, "h": 50})
        payload = _tool_json(self._ok("set_windows", {
            "session": self.SESSION, "windows": []}))
        self.assertEqual(payload["windows"], [])

    # -- fit_windows --------------------------------------------------------

    # Two side-by-side crops of the 320x200 source. Small enough that the
    # auto-layout has room to grow them, which is what fit_windows does.
    TWO_CARDS = [{"x": 0, "y": 0, "w": 150, "h": 100},
                 {"x": 150, "y": 50, "w": 150, "h": 100}]

    def _composite_cells(self):
        """The card rects the compositor would draw right now, as
        (canvas, [(x, y, w, h), ...]).

        Same `multi_window_layout` call the editor's camera-path endpoint and
        fit_windows itself make, so this is what the export would look like --
        not a re-derivation of the arithmetic under test.
        """
        kw = self.server._render_kwargs(self._edits())
        with contextlib.redirect_stdout(io.StringIO()):
            layout = mcp_server.ren.multi_window_layout(
                self.session_dir, kw["windows"], background=kw["background"],
                style=kw["style"], aspect=kw["aspect"],
                window_layout=kw["window_layout"])
        return layout["canvas"], [(c["x"], c["y"], c["w"], c["h"])
                                  for c in layout["cells"]]

    def _hand_place_cards_in_a_corner(self):
        """Drop both cards into the top-left quarter the way dragging them in
        the editor does, through the editor's own save path. Returns
        (StudioState, saved doc) -- the doc's `rev` is what a client holding
        this arrangement would send back."""
        state = studio_app.StudioState(self.root)
        doc = state.get_edits(self.SESSION)
        boxes = ({"x": 0.05, "y": 0.05, "w": 0.25, "h": 0.25},
                 {"x": 0.34, "y": 0.05, "w": 0.25, "h": 0.25})
        placed = []
        for win, box in zip(doc["windows"], boxes):
            entry = dict(win)
            entry["layout"] = dict(box)
            placed.append(entry)
        return state, state.save_session_edits(
            self.SESSION, {"windows": placed}, base_rev=doc["rev"])

    def test_fit_windows_writes_a_layout_override_for_every_card(self):
        self._ok("set_windows", {"session": self.SESSION,
                                 "windows": list(self.TWO_CARDS)})
        before_ids = [w["id"] for w in self._edits()["windows"]]
        payload = _tool_json(self._ok("fit_windows", {"session": self.SESSION}))
        canvas, _ = self._composite_cells()
        self.assertEqual(payload["canvas"], list(canvas))
        self.assertEqual(len(payload["windows"]), 2)
        for win in payload["windows"]:
            box = win["layout"]
            self.assertEqual(sorted(box), ["h", "w", "x", "y"])
            for key in ("x", "y", "w", "h"):
                self.assertGreaterEqual(box[key], 0.0, key)
                self.assertLessEqual(box[key], 1.0, key)
            self.assertGreater(box["w"], 0.0)
            self.assertGreater(box["h"], 0.0)
        # Ids survive: unlike set_windows (which rebuilds entries and reissues
        # them), fitting is a placement edit -- a caller holding an id from
        # add_window can still remove that card afterwards.
        self.assertEqual([w["id"] for w in payload["windows"]], before_ids)
        # ...and it round-trips through edits.json, not just the response.
        self.assertEqual(self._edits()["windows"], payload["windows"])

    def test_fit_windows_removes_the_dead_space_a_hand_drag_leaves(self):
        # The gap this tool exists to close: nothing rescales a dragged
        # arrangement, so it sits in the middle of a mostly-empty canvas.
        self._ok("set_windows", {"session": self.SESSION,
                                 "windows": list(self.TWO_CARDS)})
        self._hand_place_cards_in_a_corner()
        canvas, before = self._composite_cells()
        self._ok("fit_windows", {"session": self.SESSION})
        _, after = self._composite_cells()

        bx0, by0, bx1, by1 = _bbox(before)
        ax0, ay0, ax1, ay1 = _bbox(after)
        self.assertGreater(ax1 - ax0, bx1 - bx0, "composition did not widen")
        self.assertGreater(ay1 - ay0, by1 - by0, "composition did not grow")
        # The bbox growing is not the claim -- the CARDS getting bigger is
        # (measured on the real dragged layout this came from: 31% -> 67% of
        # the canvas, every card 2.15x). A fit that only spread the cards
        # apart would pass the two assertions above and be worthless.
        for got, was in zip(after, before):
            self.assertGreater(got[2] * got[3], was[2] * was[3],
                               "card did not get bigger")
        # It grows until it MEETS the canvas padding on the binding axis and
        # stops -- `framing.fit_placements` spends `framing.PAD_FRAC` off
        # the width on both axes. Uniform scale, so only one axis can touch.
        from autocine import framing
        pad = int(framing.PAD_FRAC * canvas[0])
        touches = (abs(ax0 - pad) <= 1 and abs(ax1 - (canvas[0] - pad)) <= 1) or \
                  (abs(ay0 - pad) <= 1 and abs(ay1 - (canvas[1] - pad)) <= 1)
        self.assertTrue(touches, "fitted bbox {} does not reach the {}px pad "
                                 "on either axis of {}".format(
                                     (ax0, ay0, ax1, ay1), pad, canvas))
        # ...without reflowing anything: same count, same left-to-right order.
        self.assertEqual(len(after), len(before))
        self.assertLess(after[0][0], after[1][0])

    def test_fit_windows_finds_no_slack_in_an_arrangement_the_layout_made(self):
        # This used to assert `after == before` for feature/row/column, on the
        # claim that those three "fill the frame by construction" so fitting
        # one is pixel-identical. Both halves were wrong. They do NOT fill the
        # frame -- every arrangement is scaled by ONE uniform factor, which can
        # only meet the padding on one axis -- and the equality only held for
        # the three-card set it happened to use: with two cards the fit
        # re-centres by a rounding pixel and it fails.
        #
        # The property that IS true, and the one the tool description now
        # promises, is about SLACK rather than pixels: every arrangement is
        # already produced by this same uniform fit, so a fit over one finds
        # nothing to take up and no card changes size. 'grid' is in the loop
        # deliberately -- it letterboxes cards inside fixed cells, so its
        # bounding box can sit off-centre and a fit shifts the composition,
        # still without growing anything. Nothing here reads a coordinate as
        # such, so an arrangement that re-flows itself for a portrait canvas
        # is judged the same way as one that doesn't.
        third = {"x": 40, "y": 40, "w": 100, "h": 140}
        # Both card counts (the two-card case is the one the old assertion
        # got away with not testing) against one landscape canvas and one
        # portrait. "auto" is the 320x200 source, i.e. landscape; a second
        # landscape ratio would add a `multi_window_layout` probe per
        # arrangement -- this test is already among the slowest in the file --
        # for a distinction the arrangements do not draw. Orientation is what
        # they branch on.
        for cards in (list(self.TWO_CARDS), list(self.TWO_CARDS) + [third]):
            for aspect in ("auto", "9:16"):
                for name in ed._WINDOW_LAYOUTS:
                    why = "{} {} cards, aspect {}".format(
                        name, len(cards), aspect)
                    self._ok("set_windows", {"session": self.SESSION,
                                             "windows": list(cards)})
                    self._ok("set_render_options", {
                        "session": self.SESSION, "window_layout": name,
                        "aspect": aspect})
                    _, before = self._composite_cells()
                    self._ok("fit_windows", {"session": self.SESSION})
                    _, after = self._composite_cells()

                    self.assertEqual(len(after), len(before), why)
                    for got, want in zip(after, before):
                        self.assertEqual(got[2:], want[2:],
                                         "card resized by a fit: " + why)
                    if name != "grid":
                        # The uniform-fit arrangements are already centred
                        # too, so all a fit can do is round differently.
                        for got, want in zip(after, before):
                            self.assertLessEqual(abs(got[0] - want[0]), 1, why)
                            self.assertLessEqual(abs(got[1] - want[1]), 1, why)

    def test_fit_windows_without_windows_is_a_clean_error(self):
        text = self._err("fit_windows", {"session": self.SESSION})
        self.assertIn("no windows to fit", text)
        # A refusal, not a half-write: nothing was saved.
        self.assertEqual(self._edits()["windows"], [])

    def test_fit_windows_unknown_session_is_a_tool_error(self):
        self.assertIn("unknown session: nope",
                      self._err("fit_windows", {"session": "nope"}))

    def test_fit_windows_bumps_rev_and_a_stale_base_rev_is_refused(self):
        # Same compare-and-swap every other edit obeys: the editor is very
        # likely to be OPEN on this session (it's where the drag came from),
        # and fitting behind its back must make its next save conflict rather
        # than silently undo the fit.
        self._ok("set_windows", {"session": self.SESSION,
                                 "windows": list(self.TWO_CARDS)})
        state, dragged = self._hand_place_cards_in_a_corner()
        self._ok("fit_windows", {"session": self.SESSION})
        after = self._edits()
        self.assertGreater(after["rev"], dragged["rev"])

        with self.assertRaises(studio_app.EditsConflict) as ctx:
            state.save_session_edits(self.SESSION,
                                     {"windows": dragged["windows"]},
                                     base_rev=dragged["rev"])
        self.assertEqual(ctx.exception.current["rev"], after["rev"])
        # the fit survived the refused save
        self.assertEqual(self._edits()["windows"], after["windows"])

    def test_set_windows_is_the_route_back_to_the_pure_arrangement(self):
        # The only MCP way to clear the overrides -- fit_windows deliberately
        # writes something that outlives the arrangement it came from, so
        # "reset placement" has to be an explicit rebuild.
        self._ok("set_windows", {"session": self.SESSION,
                                 "windows": list(self.TWO_CARDS)})
        self._ok("fit_windows", {"session": self.SESSION})
        self.assertTrue(all("layout" in w for w in self._edits()["windows"]))
        payload = _tool_json(self._ok("set_windows", {
            "session": self.SESSION, "windows": list(self.TWO_CARDS)}))
        self.assertFalse(any("layout" in w for w in payload["windows"]))
        self.assertFalse(any("layout" in w for w in self._edits()["windows"]))

    # -- trim ------------------------------------------------------------------

    def test_set_trim_and_null_end(self):
        payload = _tool_json(self._ok("set_trim", {
            "session": self.SESSION, "start": 0.1, "end": 0.9}))
        self.assertAlmostEqual(payload["trim"]["start"], 0.1, places=5)
        self.assertAlmostEqual(payload["trim"]["end"], 0.9, places=5)
        # end=null -> through end of clip (normalized to the duration)
        payload = _tool_json(self._ok("set_trim", {
            "session": self.SESSION, "end": None}))
        self.assertAlmostEqual(payload["trim"]["start"], 0.1, places=5)
        self.assertAlmostEqual(payload["trim"]["end"], 1.0, delta=0.1)

    def test_set_trim_requires_a_key(self):
        text = self._err("set_trim", {"session": self.SESSION})
        self.assertIn("start", text)

    # -- speed-up ranges --------------------------------------------------------

    def test_add_speedup_default_mode_off_and_roundtrip(self):
        payload = _tool_json(self._ok("add_speedup", {
            "session": self.SESSION, "start": 0.1, "end": 0.7}))
        self.assertEqual(len(payload["speedups"]), 1)
        s = payload["speedups"][0]
        self.assertTrue(s["id"].startswith("speedup-"))
        self.assertEqual(s["mode"], "off")
        self.assertIsNone(s["rate"])
        self.assertEqual(self._edits()["speedups"], payload["speedups"])

    def test_add_speedup_force_with_rate(self):
        payload = _tool_json(self._ok("add_speedup", {
            "session": self.SESSION, "start": 0.1, "end": 0.7,
            "mode": "force", "rate": 8.0}))
        s = payload["speedups"][0]
        self.assertEqual(s["mode"], "force")
        self.assertEqual(s["rate"], 8.0)

    def test_remove_speedup_and_unknown_id(self):
        added = _tool_json(self._ok("add_speedup", {
            "session": self.SESSION, "start": 0.1, "end": 0.6}))
        sid = added["speedups"][0]["id"]
        remaining = _tool_json(self._ok("remove_speedup", {
            "session": self.SESSION, "speedup_id": sid}))
        self.assertEqual(remaining["speedups"], [])
        text = self._err("remove_speedup", {
            "session": self.SESSION, "speedup_id": "speedup-deadbeef"})
        self.assertIn("unknown speedup id", text)

    # -- cuts (ripple delete) -----------------------------------------------

    def test_add_cut_roundtrip_with_snapped_echo(self):
        payload = _tool_json(self._ok("add_cut", {
            "session": self.SESSION, "start": 0.21, "end": 0.38}))
        self.assertEqual(len(payload["cuts"]), 1)
        c = payload["cuts"][0]
        self.assertTrue(c["id"].startswith("cut-"))
        self.assertEqual(self._edits()["cuts"], payload["cuts"])
        # snapped outward to the 30fps frame grid: floor(0.21*30)/30 = 0.2,
        # ceil(0.38*30)/30 = 0.4 -- what the export will actually remove
        self.assertAlmostEqual(payload["snapped"]["start"], 0.2)
        self.assertAlmostEqual(payload["snapped"]["end"], 0.4)
        self.assertAlmostEqual(payload["removed_sec"], 0.2)
        self.assertAlmostEqual(payload["output_duration"], 0.8)

    def test_add_cut_rejects_swallowing_the_whole_take(self):
        text = self._err("add_cut", {
            "session": self.SESSION, "start": 0.0, "end": 1.0})
        self.assertIn("whole trim window", text)
        # nothing was saved
        self.assertEqual(self._edits()["cuts"], [])

    def test_set_trim_cannot_write_past_the_cut_budget(self):
        # Trim is the OTHER input to the budget: narrowing the window
        # shrinks what survives exactly as adding a cut does. Without a
        # guard here, set_trim is the way to write the very document
        # add_cut and set_cuts refuse.
        self._ok("add_cut", {"session": self.SESSION,
                             "start": 0.2, "end": 0.4})
        text = self._err("set_trim", {"session": self.SESSION,
                                      "start": 0.2, "end": 0.6})
        self.assertIn("whole trim window", text)
        # the trim on disk is untouched
        self.assertAlmostEqual(self._edits()["trim"]["start"], 0.0)

    def test_set_trim_still_works_when_the_cuts_leave_room(self):
        self._ok("add_cut", {"session": self.SESSION,
                             "start": 0.2, "end": 0.3})
        self._ok("set_trim", {"session": self.SESSION,
                              "start": 0.0, "end": 0.9})
        self.assertAlmostEqual(self._edits()["trim"]["end"], 0.9)

    def test_get_transcript_words_carry_the_repaired_end(self):
        # `end`, not `t + dur`, is the field a cut is built from -- the
        # editor and the agent must not disagree about where a word stops.
        payload = self._ok("get_transcript", {"session": self.SESSION})
        for w in payload.get("words") or []:
            self.assertIn("end", w)

    def test_add_cut_rejects_inverted_range(self):
        text = self._err("add_cut", {
            "session": self.SESSION, "start": 0.5, "end": 0.2})
        self.assertIn("end must be after start", text)

    def test_remove_cut_and_unknown_id(self):
        added = _tool_json(self._ok("add_cut", {
            "session": self.SESSION, "start": 0.2, "end": 0.4}))
        cid = added["cuts"][0]["id"]
        remaining = _tool_json(self._ok("remove_cut", {
            "session": self.SESSION, "cut_id": cid}))
        self.assertEqual(remaining["cuts"], [])
        text = self._err("remove_cut", {
            "session": self.SESSION, "cut_id": "cut-deadbeef"})
        self.assertIn("unknown cut id", text)

    def test_set_cuts_replaces_whole_array_and_clears(self):
        self._ok("add_cut", {"session": self.SESSION,
                             "start": 0.1, "end": 0.2})
        payload = _tool_json(self._ok("set_cuts", {
            "session": self.SESSION,
            "cuts": [{"start": 0.3, "end": 0.4},
                     {"start": 0.6, "end": 0.7}]}))
        self.assertEqual(len(payload["cuts"]), 2)
        self.assertAlmostEqual(payload["removed_sec"], 0.2)
        self.assertEqual(len(payload["snapped"]), 2)
        cleared = _tool_json(self._ok("set_cuts", {
            "session": self.SESSION, "cuts": []}))
        self.assertEqual(cleared["cuts"], [])
        self.assertEqual(self._edits()["cuts"], [])

    def test_set_cuts_validates_shapes(self):
        text = self._err("set_cuts", {
            "session": self.SESSION, "cuts": [{"start": 0.1}]})
        self.assertIn("must be a finite number", text)
        text = self._err("set_cuts", {"session": self.SESSION,
                                      "cuts": "nope"})
        self.assertIn("must be an array", text)

    def test_cut_tools_reject_non_finite_numbers(self):
        """REGRESSION (review finding): json.loads accepts bare NaN/Infinity;
        NaN then no-ops the end<=start guard AND the coverage validation,
        while normalize persisted a real cut from 0 -- a take-swallowing
        write the _MIN_KEPT_SEC guard existed to prevent."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            text = self._err("add_cut", {
                "session": self.SESSION, "start": bad, "end": 0.9})
            self.assertIn("finite", text)
            text = self._err("set_cuts", {
                "session": self.SESSION,
                "cuts": [{"start": 0.1, "end": bad}]})
            self.assertIn("finite", text)
        self.assertEqual(self._edits()["cuts"], [])

    def test_sub_150ms_cut_roundtrips_and_echo_matches_saved(self):
        """REGRESSION (review finding): the echo/validation used to run on
        the RAW request while a 0.15s-widened list persisted -- the same
        response contradicted itself. Now saved == authored and the snapped
        echo is the quantization of what persists."""
        payload = _tool_json(self._ok("add_cut", {
            "session": self.SESSION, "start": 0.3, "end": 0.4}))
        c = payload["cuts"][0]
        self.assertEqual(c["start"], 0.3)
        self.assertEqual(c["end"], 0.4)
        self.assertEqual(self._edits()["cuts"], payload["cuts"])
        self.assertAlmostEqual(payload["snapped"]["start"], 0.3)
        self.assertAlmostEqual(payload["snapped"]["end"], 0.4)
        self.assertAlmostEqual(payload["removed_sec"], 0.1)
        self.assertAlmostEqual(payload["output_duration"], 0.9)

    def test_set_render_options_speedup_round_trip(self):
        payload = _tool_json(self._ok("set_render_options", {
            "session": self.SESSION,
            "speedup": True, "speedup_rate": 8.0,
            "speedup_silence_gate": False}))
        r = payload["render"]
        self.assertTrue(r["speedup"])
        self.assertEqual(r["speedup_rate"], 8.0)
        self.assertFalse(r["speedup_silence_gate"])

    # -- markers / chapters -------------------------------------------------------

    def test_markers_and_chapters(self):
        self._ok("add_marker", {
            "session": self.SESSION, "time": 0.0, "label": "Intro"})
        payload = _tool_json(self._ok("add_marker", {
            "session": self.SESSION, "time": 0.5, "label": "Demo"}))
        self.assertEqual(len(payload["markers"]), 2)
        chapters = payload["chapters"]
        self.assertEqual(len(chapters), 2)
        self.assertEqual(chapters[0]["label"], "Intro")
        self.assertAlmostEqual(chapters[0]["start"], 0.0, places=5)
        self.assertAlmostEqual(chapters[0]["end"], 0.5, places=5)
        self.assertEqual(chapters[1]["label"], "Demo")
        self.assertAlmostEqual(chapters[1]["start"], 0.5, places=5)

        first_id = payload["markers"][0]["id"]
        removed = _tool_json(self._ok("remove_marker", {
            "session": self.SESSION, "marker_id": first_id}))
        self.assertEqual(len(removed["markers"]), 1)
        self.assertEqual(len(removed["chapters"]), 1)
        self.assertEqual(removed["chapters"][0]["label"], "Demo")

        text = self._err("remove_marker", {
            "session": self.SESSION, "marker_id": first_id})
        self.assertIn("unknown marker id", text)

    # -- render options / reset ---------------------------------------------------

    def test_set_render_options_partial_patch_preserves_other_keys(self):
        self._ok("set_render_options", {"session": self.SESSION, "zoom": 3.0})
        self._ok("set_render_options", {
            "session": self.SESSION, "spotlight": True, "background": "sunset"})
        render = self._edits()["render"]
        self.assertAlmostEqual(render["zoom"], 3.0, places=5)
        self.assertTrue(render["spotlight"])
        self.assertEqual(render["background"], "sunset")
        self.assertEqual(render["style"], "framed")  # background implies framed
        self.assertTrue(render["click_fx"])  # untouched default preserved

    def test_set_render_options_typing_zoom_round_trip(self):
        self._ok("set_render_options", {"session": self.SESSION,
                                        "typing_zoom": False})
        self.assertFalse(self._edits()["render"]["typing_zoom"])
        self._ok("set_render_options", {"session": self.SESSION,
                                        "typing_zoom": True})
        self.assertTrue(self._edits()["render"]["typing_zoom"])

    def test_set_render_options_zoom_speed_round_trip_and_validation(self):
        self._ok("set_render_options", {"session": self.SESSION,
                                        "zoom_speed": "fast"})
        render_opts = self._edits()["render"]
        self.assertEqual(render_opts["zoom_speed"], "fast")
        self.assertEqual(mcp_server._camera_params(render_opts),
                         {"zoom_speed": "fast"})
        text = self._err("set_render_options", {"session": self.SESSION,
                                                "zoom_speed": "warp"})
        self.assertIn("zoom_speed", text)
        self._ok("set_render_options", {"session": self.SESSION,
                                        "zoom_speed": "normal"})
        self.assertIsNone(mcp_server._camera_params(self._edits()["render"]))

    def test_set_render_options_drag_hold_round_trip_and_params(self):
        self._ok("set_render_options", {"session": self.SESSION,
                                        "drag_hold": False})
        render_opts = self._edits()["render"]
        self.assertFalse(render_opts["drag_hold"])
        self.assertEqual(mcp_server._camera_params(render_opts),
                         {"drag_hold": False})
        self._ok("set_render_options", {"session": self.SESSION,
                                        "drag_hold": True})
        self.assertIsNone(mcp_server._camera_params(self._edits()["render"]))

    def test_set_render_options_overview_round_trip_and_params(self):
        self._ok("set_render_options", {"session": self.SESSION,
                                        "overview": False})
        render_opts = self._edits()["render"]
        self.assertFalse(render_opts["overview"])
        self.assertEqual(mcp_server._camera_params(render_opts),
                         {"overview": False})
        self._ok("set_render_options", {"session": self.SESSION,
                                        "overview": True})
        self.assertIsNone(mcp_server._camera_params(self._edits()["render"]))

    def test_set_render_options_scroll_zoom_round_trip(self):
        self._ok("set_render_options", {"session": self.SESSION,
                                        "scroll_zoom": False})
        edited = self._edits()
        self.assertFalse(edited["render"]["scroll_zoom"])
        # ...and the render-kwargs builder must actually carry it through
        # to render()/preview_frame() (mutation-proven gap: a hardcoded
        # True survived the old suite)
        self.assertFalse(self.server._render_kwargs(edited)["scroll_zoom"])
        self._ok("set_render_options", {"session": self.SESSION,
                                        "scroll_zoom": True})
        edited = self._edits()
        self.assertTrue(edited["render"]["scroll_zoom"])
        self.assertTrue(self.server._render_kwargs(edited)["scroll_zoom"])

    def test_set_render_options_facecam_round_trip_and_params(self):
        payload = _tool_json(self._ok("set_render_options", {
            "session": self.SESSION, "facecam_position": "bottom-right",
            "facecam_size": 0.3, "facecam_shape": "rounded"}))
        r = payload["render"]
        self.assertEqual(r["facecam_position"], "bottom-right")
        self.assertAlmostEqual(r["facecam_size"], 0.3, places=5)
        self.assertEqual(r["facecam_shape"], "rounded")
        self.assertTrue(r["facecam"])  # untouched default preserved
        # ...and the render-kwargs builder must actually carry them through to
        # render()/preview_frame(), same guard as scroll_zoom/windows.
        kw = self.server._render_kwargs(self._edits())
        self.assertTrue(kw["facecam"])
        self.assertEqual(kw["facecam_params"],
                         {"position": "bottom-right", "size_frac": 0.3,
                          "shape": "rounded"})

    def test_set_render_options_facecam_defaults_send_no_overrides(self):
        # Off switch stays bit-exact: an untouched facecam reaches render()
        # as facecam=True with no params, exactly as before these options
        # were exposed here.
        kw = self.server._render_kwargs(self._edits())
        self.assertTrue(kw["facecam"])
        self.assertIsNone(kw["facecam_params"])
        self._ok("set_render_options", {"session": self.SESSION,
                                        "facecam": False})
        edited = self._edits()
        self.assertFalse(edited["render"]["facecam"])
        self.assertFalse(self.server._render_kwargs(edited)["facecam"])

    def test_set_render_options_facecam_size_is_clamped(self):
        payload = _tool_json(self._ok("set_render_options", {
            "session": self.SESSION, "facecam_size": 9.0}))
        self.assertAlmostEqual(payload["render"]["facecam_size"], 0.5, places=5)

    def test_set_render_options_rejects_bad_facecam_enums(self):
        text = self._err("set_render_options", {
            "session": self.SESSION, "facecam_position": "middle"})
        self.assertIn("facecam_position", text)
        text = self._err("set_render_options", {
            "session": self.SESSION, "facecam_shape": "hexagon"})
        self.assertIn("facecam_shape", text)
        # a rejected patch must not have been persisted
        render_opts = self._edits()["render"]
        self.assertEqual(render_opts["facecam_position"], "bottom-left")
        self.assertEqual(render_opts["facecam_shape"], "circle")

    def test_set_render_options_requires_an_option(self):
        text = self._err("set_render_options", {"session": self.SESSION})
        self.assertIn("no render options", text)

    def test_set_render_options_rejects_bad_style(self):
        text = self._err("set_render_options", {
            "session": self.SESSION, "style": "fancy"})
        self.assertIn("style", text)

    def test_reset_edits_restores_defaults(self):
        self._ok("add_zoom", {"session": self.SESSION, "start": 0.2, "end": 0.6})
        self._ok("set_render_options", {"session": self.SESSION, "zoom": 4.0})
        payload = _tool_json(self._ok("reset_edits", {"session": self.SESSION}))
        self.assertEqual(payload["zooms"], [])
        self.assertAlmostEqual(payload["render"]["zoom"], 2.2, places=5)

    # -- preview / render -----------------------------------------------------------

    def test_preview_frame_returns_jpeg_image_content(self):
        self._ok("add_zoom", {
            "session": self.SESSION, "start": 0.2, "end": 0.8,
            "x": 160, "y": 100, "level": 2.5})
        resp = self._ok("preview_frame", {
            "session": self.SESSION, "time": 0.5, "max_width": 200})
        content = resp["result"]["content"]
        self.assertEqual(content[0]["type"], "image")
        self.assertEqual(content[0]["mimeType"], "image/jpeg")
        self.assertTrue(content[0]["data"].startswith("/9j/"),
                        "expected base64 JPEG magic, got: " +
                        content[0]["data"][:12])
        meta = json.loads(content[1]["text"])
        self.assertLessEqual(meta["width"], 200)
        self.assertGreater(meta["height"], 0)
        self.assertAlmostEqual(meta["time"], 0.5, places=5)

    def test_rendered_preview_reports_no_source_scale(self):
        """The rendered frame is output-canvas pixels after the camera's
        zoom/pan and aspect framing, so NO scalar maps it back to source
        coordinates. Reporting `source_scale` there was a wrong answer that
        looked like a right one -- measured, a 9:16 framed preview of a
        2880-wide source reported source_width 1708."""
        self._ok("set_render_options", {"session": self.SESSION,
                                        "aspect": "9:16"})
        resp = self._ok("preview_frame", {"session": self.SESSION,
                                          "time": 0.5, "max_width": 120})
        meta = json.loads(resp["result"]["content"][1]["text"])
        self.assertNotIn("source_scale", meta)
        self.assertFalse(meta["is_source"])
        self.assertGreater(meta["frame_scale"], 0)
        self.assertIn("source=true", meta["note"])
        # The true source dims are still reported, and are NOT the canvas.
        info = _tool_json(self._ok("describe_session",
                                   {"session": self.SESSION}))
        self.assertEqual(meta["source_width"], info["width"])
        self.assertEqual(meta["source_height"], info["height"])
        self.assertNotEqual(meta["frame_width"], meta["source_width"])

    def test_source_preview_still_reports_source_scale(self):
        """The measurement path is unchanged -- that contract is the reason
        the rendered path had to stop claiming it."""
        resp = self._ok("preview_frame", {"session": self.SESSION,
                                          "time": 0.5, "max_width": 120,
                                          "source": True})
        meta = json.loads(resp["result"]["content"][1]["text"])
        info = _tool_json(self._ok("describe_session",
                                   {"session": self.SESSION}))
        self.assertEqual(meta["source_width"], info["width"])
        self.assertAlmostEqual(meta["source_scale"],
                               meta["source_width"] / float(meta["width"]),
                               places=6)
        self.assertNotIn("is_source", meta)
        self.assertNotIn("frame_scale", meta)

    def test_preview_frame_time_clamped_to_trim(self):
        self._ok("set_trim", {"session": self.SESSION, "start": 0.2, "end": 0.8})
        resp = self._ok("preview_frame", {"session": self.SESSION, "time": 5.0})
        meta = json.loads(resp["result"]["content"][1]["text"])
        self.assertAlmostEqual(meta["time"], 0.8, places=5)

    def test_preview_frame_honors_windows_mode(self):
        # Regression guard for mcp_server.py having its own _render_kwargs,
        # separate from studio_app's -- set_windows writing edits.json must
        # actually be reflected here, not silently ignored.
        self._ok("set_windows", {
            "session": self.SESSION,
            "windows": [{"x": 0, "y": 0, "w": 150, "h": 100},
                       {"x": 150, "y": 50, "w": 150, "h": 100}]})
        resp = self._ok("preview_frame", {
            "session": self.SESSION, "time": 0.3, "max_width": 200})
        content = resp["result"]["content"]
        self.assertEqual(content[0]["type"], "image")
        meta = json.loads(content[1]["text"])
        self.assertGreater(meta["width"], 0)
        self.assertGreater(meta["height"], 0)

    def test_preview_frame_source_ignores_edits_and_reports_scale(self):
        """`source=true` is how a caller MEASURES rects for set_windows /
        add_zoom. It must show the unedited source, and it must say what a
        downscaled image's pixels are worth in source units -- authoring rects
        in the wrong space is the exact silent failure this flag prevents."""
        self._ok("set_windows", {
            "session": self.SESSION,
            "windows": [{"x": 0, "y": 0, "w": 150, "h": 100}]})
        self._ok("set_render_options", {"session": self.SESSION,
                                        "background": "sunset"})
        info = _tool_json(self._ok("describe_session", {"session": self.SESSION}))

        resp = self._ok("preview_frame", {
            "session": self.SESSION, "time": 0.3, "source": True,
            "max_width": 200})
        meta = json.loads(resp["result"]["content"][1]["text"])
        # The source frame is the space describe_session reports, regardless of
        # the windows/background edits above.
        self.assertEqual(meta["source_width"], info["width"])
        self.assertEqual(meta["source_height"], info["height"])
        self.assertLessEqual(meta["width"], 200)
        self.assertAlmostEqual(
            meta["source_scale"], meta["source_width"] / float(meta["width"]),
            places=5)

    def test_preview_frame_source_differs_from_the_rendered_preview(self):
        """The whole point: with windows mode on, the two must NOT agree."""
        self._ok("set_windows", {
            "session": self.SESSION,
            "windows": [{"x": 0, "y": 0, "w": 150, "h": 100},
                        {"x": 150, "y": 50, "w": 150, "h": 100}]})
        args = {"session": self.SESSION, "time": 0.3, "max_width": 160}
        rendered = self._ok("preview_frame", args)
        src = self._ok("preview_frame", dict(args, source=True))
        self.assertNotEqual(rendered["result"]["content"][0]["data"],
                            src["result"]["content"][0]["data"])

    def test_preview_frame_source_scale_is_one_when_not_downscaled(self):
        resp = self._ok("preview_frame", {
            "session": self.SESSION, "time": 0.3, "source": True,
            "max_width": 100000})
        meta = json.loads(resp["result"]["content"][1]["text"])
        self.assertAlmostEqual(meta["source_scale"], 1.0, places=6)
        self.assertEqual(meta["width"], meta["source_width"])

    def test_render_video_writes_output_and_keeps_stdout_clean(self):
        # The handler must internally redirect the render pipeline's progress
        # prints away from stdout (protocol channel): capture sys.stdout around
        # the call and require it to stay empty.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            resp = _call(self.server, "render_video", {
                "session": self.SESSION, "out": "mcp-test-output.mp4"})
        self.assertEqual(buf.getvalue(), "",
                         "render_video leaked output onto stdout")
        self.assertFalse(resp["result"]["isError"],
                         "render failed: " + _tool_text(resp))
        payload = _tool_json(resp)
        out_path = payload["out_path"]
        self.assertEqual(os.path.dirname(out_path), self.session_dir)
        self.assertTrue(os.path.isfile(out_path))
        self.assertGreater(os.path.getsize(out_path), 0)

    def test_render_video_rejects_outputs_outside_the_session(self):
        outside = os.path.join(self.root, "outside.mp4")
        for bad in ("../outside.mp4", outside, ".", "\x00.mp4"):
            with self.subTest(out=bad):
                text = self._err("render_video", {
                    "session": self.SESSION, "out": bad})
                self.assertIn("inside the session directory", text)
        self.assertFalse(os.path.exists(outside))

    def test_render_video_rejects_an_output_symlink_that_escapes(self):
        outside = os.path.join(self.root, "outside.mp4")
        link = os.path.join(self.session_dir, "output.mp4")
        with open(outside, "wb") as f:
            f.write(b"keep")
        os.symlink(outside, link)
        try:
            text = self._err("render_video", {"session": self.SESSION})
            self.assertIn("inside the session directory", text)
            with open(outside, "rb") as f:
                self.assertEqual(f.read(), b"keep")
        finally:
            if os.path.lexists(link):
                os.remove(link)
            if os.path.exists(outside):
                os.remove(outside)

    def test_render_video_rejects_a_derived_gif_symlink_that_escapes(self):
        outside = os.path.join(self.root, "outside.gif")
        link = os.path.join(self.session_dir, "safe.gif")
        with open(outside, "wb") as f:
            f.write(b"keep")
        os.symlink(outside, link)
        try:
            text = self._err("render_video", {
                "session": self.SESSION, "out": "safe.mp4", "gif": True})
            self.assertIn("inside the session directory", text)
            with open(outside, "rb") as f:
                self.assertEqual(f.read(), b"keep")
        finally:
            if os.path.lexists(link):
                os.remove(link)
            if os.path.exists(outside):
                os.remove(outside)

    def test_set_render_options_speedup_reaches_render_kwargs(self):
        self._ok("set_render_options", {
            "session": self.SESSION, "speedup": True, "speedup_rate": 3.0,
            "speedup_silence_gate": False, "speedup_motion_gate": False})
        self._ok("add_speedup", {"session": self.SESSION, "start": 0.1,
                                 "end": 0.9, "mode": "force", "rate": 5.0})
        kw = self.server._render_kwargs(self._edits())
        self.assertTrue(kw["speedup"])
        self.assertAlmostEqual(kw["speedup_rate"], 3.0, places=5)
        self.assertFalse(kw["speedup_silence_gate"])
        self.assertFalse(kw["speedup_motion_gate"])
        self.assertEqual(len(kw["speedups"]), 1)
        self.assertEqual(kw["speedups"][0]["mode"], "force")
        # ...and the off switch stays bit-exact: an untouched session reaches
        # render() exactly as it did before speed-up existed.
        self._ok("reset_edits", {"session": self.SESSION})
        kw = self.server._render_kwargs(self._edits())
        self.assertFalse(kw["speedup"])
        self.assertEqual(kw["speedups"], [])

    def test_set_render_options_window_layout_round_trip(self):
        # The tool schema advertises window_layout; the handler used to drop
        # it on the floor, so "no render options provided" was the only sign
        # anything was wrong -- and only when it was the sole argument.
        payload = _tool_json(self._ok("set_render_options", {
            "session": self.SESSION, "window_layout": "desktop"}))
        self.assertEqual(payload["render"]["window_layout"], "desktop")
        self.assertEqual(
            self.server._render_kwargs(self._edits())["window_layout"],
            "desktop")
        text = self._err("set_render_options", {"session": self.SESSION,
                                                "window_layout": "cascade"})
        self.assertIn("window_layout", text)
        self.assertEqual(self._edits()["render"]["window_layout"], "desktop")

    def test_set_render_options_accepts_every_arrangement(self):
        # Premise, so a shrunken tuple can't make the loop pass by emptiness.
        self.assertTrue({"feature", "row", "column"}
                        .issubset(ed._WINDOW_LAYOUTS))
        for name in ed._WINDOW_LAYOUTS:
            payload = _tool_json(self._ok("set_render_options", {
                "session": self.SESSION, "window_layout": name}))
            self.assertEqual(payload["render"]["window_layout"], name)
            self.assertEqual(self._edits()["render"]["window_layout"], name)
            # ...and reaches render()/preview_frame(). Resolving it here is a
            # separate hop from accepting it: the validator and the resolver
            # keep their own membership checks, and a resolver that hasn't
            # learned the new names saves "feature" and renders the grid.
            self.assertEqual(
                self.server._render_kwargs(self._edits())["window_layout"],
                name)

    def test_set_render_options_rejects_a_near_miss_arrangement(self):
        # The accepted set is exact -- edits.py does no case/whitespace
        # repair, so accepting "Feature" here would save a value that
        # normalization then silently turns back into the grid.
        self._ok("set_render_options", {"session": self.SESSION,
                                        "window_layout": "feature"})
        for near in ("Feature", "ROW", " column"):
            text = self._err("set_render_options", {
                "session": self.SESSION, "window_layout": near})
            self.assertIn("window_layout", text)
            self.assertIn("feature", text)   # the message lists the real names
            self.assertEqual(self._edits()["render"]["window_layout"],
                             "feature", near)

    def _capture_render_kwargs(self, attr, tool, args):
        """Call `tool` with `mcp_server.ren.<attr>` stubbed out, and return
        (kwargs it was called with, kwargs the resolver produced)."""
        seen = {}

        def fake(session_dir, **kwargs):
            seen.update(kwargs)
            # Only the kwargs are wanted; ToolError is the one exception the
            # server turns into an isError result without logging a traceback.
            raise mcp_server.ToolError("stub")

        orig = getattr(mcp_server.ren, attr)
        setattr(mcp_server.ren, attr, fake)
        self.addCleanup(setattr, mcp_server.ren, attr, orig)
        self._err(tool, args)  # the stub's RuntimeError -> isError result
        return seen, self.server._render_kwargs(self._edits())

    def _assert_call_site_passes_everything(self, attr, tool, args):
        """Every resolved option that `ren.<attr>` can accept must actually be
        handed to it. This is the OTHER half of the parity bug: resolving an
        option and then forgetting it in the call site's hand-written kwarg
        list fails exactly as silently."""
        import inspect

        # Read the signature BEFORE the stub replaces it, or `accepted` is
        # the stub's (**kwargs) and this asserts nothing at all.
        accepted = set(inspect.signature(getattr(mcp_server.ren, attr))
                       .parameters)
        seen, kw = self._capture_render_kwargs(attr, tool, args)
        expected = sorted(set(kw) & accepted)
        self.assertGreater(len(expected), 20,
                           "signature introspection went wrong: only {} "
                           "options to check".format(len(expected)))
        for key in expected:
            self.assertIn(key, seen,
                          "{} resolves '{}' but never passes it to ren.{}() "
                          "-- it silently renders with that function's own "
                          "default".format(tool, key, attr))
            self.assertEqual(seen[key], kw[key], key)

    def test_render_video_passes_every_resolved_option(self):
        self._ok("set_render_options", {
            "session": self.SESSION, "speedup": True, "speedup_rate": 3.0,
            "window_layout": "desktop", "background": "sunset",
            "gif_fps": 24, "music": "/tmp/track.mp3"})
        self._ok("add_speedup", {"session": self.SESSION, "start": 0.1,
                                 "end": 0.9, "mode": "force"})
        self._assert_call_site_passes_everything(
            "render", "render_video", {"session": self.SESSION})

    def test_preview_frame_passes_every_resolved_option(self):
        # preview_frame's signature has no gif/music/speedup kwargs, so the
        # intersection with the resolver is what it must carry -- notably
        # window_layout, which decides how the multi-window cards are laid out.
        self._ok("set_render_options", {
            "session": self.SESSION, "window_layout": "desktop",
            "cursor_fx": True, "facecam_position": "top-right"})
        self._assert_call_site_passes_everything(
            "preview_frame", "preview_frame",
            {"session": self.SESSION, "time": 0.3})

    @unittest.skipUnless(shutil.which("ffprobe"), "needs ffprobe")
    def test_render_video_honors_forced_speedup(self):
        """The whole point of the speed-up kwargs: MCP's render must be the
        SAME video the web app's Export produces from the same edits.json.
        A forced span must actually shorten the output -- before this was
        wired, edits.speedups written by this server's own add_speedup tool
        never reached ren.render() and the take rendered at full length."""
        resp = _call(self.server, "render_video", {
            "session": self.SESSION, "out": "mcp-speedup-baseline.mp4"})
        self.assertFalse(resp["result"]["isError"],
                         "baseline render failed: " + _tool_text(resp))
        baseline = _probe_frames(_tool_json(resp)["out_path"])

        # mode="force" so this holds without depending on idle/silence/motion
        # detection: it must fire even with render.speedup left off.
        self._ok("add_speedup", {"session": self.SESSION, "start": 0.1,
                                 "end": 0.9, "mode": "force", "rate": 6.0})
        resp = _call(self.server, "render_video", {
            "session": self.SESSION, "out": "mcp-speedup-forced.mp4"})
        self.assertFalse(resp["result"]["isError"],
                         "speedup render failed: " + _tool_text(resp))
        sped = _probe_frames(_tool_json(resp)["out_path"])
        self.assertLess(sped, baseline * 0.8,
                        "forced speed-up did not shorten the render: "
                        "{} frames vs {} baseline".format(sped, baseline))

    def test_render_video_honors_windows_mode(self):
        # Same regression guard as preview_frame's: render_video must
        # actually thread `windows` through mcp_server's own _render_kwargs
        # and its ren.render call, not just accept set_windows silently.
        self._ok("set_windows", {
            "session": self.SESSION,
            "windows": [{"x": 0, "y": 0, "w": 150, "h": 100},
                       {"x": 150, "y": 50, "w": 150, "h": 100}]})
        resp = _call(self.server, "render_video", {
            "session": self.SESSION, "out": "mcp-windows-output.mp4"})
        self.assertFalse(resp["result"]["isError"],
                         "windows-mode render failed: " + _tool_text(resp))
        payload = _tool_json(resp)
        self.assertTrue(os.path.isfile(payload["out_path"]))
        self.assertGreater(os.path.getsize(payload["out_path"]), 0)


class AdjustZoomTests(unittest.TestCase):
    """`adjust_zoom` -- \"that zoom at 1:23 was too aggressive\" as ONE call.

    A 12-second session with two click bursts 7s apart, so the auto planner
    produces two separate arcs and \"change this one\" is a question with a
    wrong answer available.
    """

    SESSION = "20260301-000000"

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp(prefix="mcp-adjust-")
        sdir = os.path.join(cls.root, cls.SESSION)
        os.makedirs(sdir)
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump({"fps": 30, "raw": "raw.mov", "events": "events.jsonl",
                       "cursor_mode": "system",
                       "logical_w": 320.0, "logical_h": 200.0}, f)
        events = [{"t": 1.0, "type": "down", "x": 60, "y": 60},
                  {"t": 1.8, "type": "down", "x": 70, "y": 70},
                  {"t": 8.0, "type": "down", "x": 240, "y": 140},
                  {"t": 8.9, "type": "down", "x": 250, "y": 150}]
        with open(os.path.join(sdir, "events.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=320x200:rate=30",
             "-t", "12.0", "-pix_fmt", "yuv420p",
             os.path.join(sdir, "raw.mov")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            shutil.rmtree(cls.root, ignore_errors=True)
            raise unittest.SkipTest("ffmpeg cannot build synthetic raw.mov")
        cls.session_dir = sdir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.server = McpServer(recordings_root=self.root)
        self._ok("reset_edits")

    def _ok(self, name, **arguments):
        resp = _call(self.server, name, dict(session=self.SESSION,
                                             **arguments))
        self.assertFalse(resp["result"]["isError"],
                         "{} failed: {}".format(name, _tool_text(resp)))
        return _tool_json(resp)

    def _err(self, name, **arguments):
        resp = _call(self.server, name, dict(session=self.SESSION,
                                             **arguments))
        self.assertTrue(resp["result"]["isError"],
                        "{} unexpectedly succeeded".format(name))
        return _tool_text(resp)

    # -- the whole-screen take ---------------------------------------------

    def test_softening_one_arc_keeps_every_other_arc(self):
        """The trap this tool exists to close: every surface but a bare CLI
        render plans the camera from `edits.zooms` ALONE, so writing one
        adjusted arc into a never-materialized session would leave it as the
        only zoom in the take."""
        got = self._ok("adjust_zoom", at=1.0)
        self.assertEqual(got["target"], "zoom")
        self.assertEqual(got["matched"], "covering")
        self.assertEqual(got["action"], "softened")
        self.assertEqual(got["before"]["level"], 2.0)
        self.assertLess(got["after"]["level"], 2.0)
        levels = [z["level"] for z in got["zooms"]]
        self.assertEqual(len(levels), 2, "the other arc was dropped")
        self.assertIn(2.0, levels, "the arc nobody complained about moved")
        self.assertTrue(any("materialized" in n for n in got["notes"]))

    def test_a_clock_string_addresses_the_same_move_as_seconds(self):
        by_secs = self._ok("adjust_zoom", at=8.2)
        self._ok("reset_edits")
        by_clock = self._ok("adjust_zoom", at="0:08.2")
        self.assertEqual(by_clock["at"], by_secs["at"])
        self.assertEqual(by_clock["before"]["start"],
                         by_secs["before"]["start"])
        self.assertEqual(by_clock["after"]["level"],
                         by_secs["after"]["level"])

    def test_stronger_pushes_further_in(self):
        got = self._ok("adjust_zoom", at=1.0, change="stronger")
        self.assertEqual(got["action"], "strengthened")
        self.assertGreater(got["after"]["level"], got["before"]["level"])

    def test_an_exact_level_can_be_set(self):
        got = self._ok("adjust_zoom", at=1.0, level=1.4)
        self.assertEqual(got["action"], "set")
        self.assertEqual(got["after"]["level"], 1.4)

    def test_off_removes_the_move_and_says_how_to_put_it_back(self):
        got = self._ok("adjust_zoom", at=1.0, change="off")
        self.assertEqual(got["action"], "removed")
        self.assertIsNone(got["after"])
        self.assertEqual(len(got["zooms"]), 1)

    def test_softening_past_the_floor_removes_it_rather_than_faking_a_zoom(self):
        # 2.0 -> 1.538 -> below 1.2, which is not a zoom anyone reads as one.
        self._ok("adjust_zoom", at=1.0)
        got = self._ok("adjust_zoom", at=1.0)
        self.assertEqual(got["action"], "removed")
        self.assertTrue(any("add_zoom(start=" in n for n in got["notes"]),
                        "removal must carry the range to add back")

    def test_a_time_beside_a_move_matches_it_and_says_so(self):
        got = self._ok("adjust_zoom", at=3.0)     # the arc ends at 2.3
        self.assertEqual(got["matched"], "nearest")
        self.assertLess(got["before"]["end"], 3.0)

    def test_a_time_far_from_every_move_is_refused_with_the_ranges(self):
        # Past the last arc's tail (9.4) by more than the match window, but
        # still inside the "did you mean the end of the take" tolerance.
        # Materialize first so the arcs are known, then aim between them.
        self._ok("adjust_zoom", at=1.0, level=2.0)
        arcs = [(z["start"], z["end"])
                for z in self._ok("get_edits")["zooms"]]
        self.assertEqual(arcs, [(0.0, 2.3), (5.5, 9.4)])
        msg = self._err("adjust_zoom", at=12.5)
        self.assertIn("no zoom covers", msg)
        # The refusal has to carry the ranges themselves -- a bare "no zoom
        # there" leaves the caller guessing a second time.
        self.assertIn("0.0-2.3", msg)
        self.assertIn("5.5-9.4", msg)

    def test_a_time_past_the_recording_is_refused(self):
        self.assertIn("past the end", self._err("adjust_zoom", at=60))

    def test_a_bad_clock_string_is_refused(self):
        self.assertIn("seconds or", self._err("adjust_zoom", at="soon"))

    def test_an_unknown_change_is_refused(self):
        self.assertIn("softer", self._err("adjust_zoom", at=1.0,
                                          change="gentler"))

    def test_a_retimed_export_warns_that_the_clocks_differ(self):
        """The one way to silently adjust the wrong arc: read 1:23 off an
        exported video whose head was trimmed."""
        self._ok("set_trim", start=2.0)
        got = self._ok("adjust_zoom", at=8.2)
        self.assertTrue(any("SOURCE-media" in n for n in got["notes"]))
        self.assertTrue(any("2.00s" in n for n in got["notes"]))

    def test_the_clock_warning_names_cuts_and_speed_ups_too(self):
        """The trim case is the easy one. A cut or a speed-up shifts the
        export clock NON-linearly, so the note must name them and must NOT
        offer the source = exported + Xs arithmetic that only holds for a
        bare trim."""
        self._ok("set_trim", start=2.0)
        self._ok("add_cut", start=3.0, end=4.0)
        self._ok("set_render_options", speedup=True)
        notes = " ".join(self._ok("adjust_zoom", at=8.2)["notes"])
        self.assertIn("SOURCE-media", notes)
        self.assertIn("cut range", notes)
        self.assertIn("sped up", notes)
        self.assertNotIn("source = exported +", notes)

    def test_the_beat_sheet_reads_the_new_level_back(self):
        self._ok("adjust_zoom", at=1.0, level=1.4)
        info = self._ok("describe_session")
        clicks = [b for b in info["beats"]["beats"] if b["kind"] == "clicks"]
        first = min(clicks, key=lambda b: b["start"])
        self.assertEqual(first["zoom"]["level"], 1.4)

    # -- the multi-window take ---------------------------------------------

    def _cards(self):
        self._ok("set_windows", windows=[
            {"x": 0, "y": 0, "w": 160, "h": 200},
            {"x": 160, "y": 0, "w": 160, "h": 200}])

    def test_a_multi_window_take_retunes_the_composition_camera(self):
        self._cards()
        self._ok("set_render_options", zoom_style="frame")
        got = self._ok("adjust_zoom", at=1.0)
        self.assertEqual(got["target"], "focus")
        self.assertEqual(got["action"], "softened")
        self.assertEqual(got["before"]["level"], "full")
        self.assertEqual(got["after"]["level"], "focus")
        # The whole move drops a rung -- a pair left at [grow, push-in] would
        # still push in a second later.
        moved = [f for f in got["focus"]
                 if f["id"] in got["before"]["ids"]]
        self.assertEqual([f["level"] for f in moved], ["focus", "focus"])

    # An exactly-touching cross-card pair: card 0 hands the composition to
    # card 1 at t=1.5, and card 1's own two spans are the grow/push-in split.
    # This is the planner's ordinary output shape (and what
    # `render._first_wins_spans` manufactures by construction), authored here
    # directly so the test pins the RULE rather than one fixture's timings.
    HANDOVER = [{"start": 0.0, "end": 1.5, "card": 0, "level": "focus"},
                {"start": 1.5, "end": 4.6, "card": 1, "level": "focus"},
                {"start": 4.6, "end": 6.4, "card": 1, "level": "full"}]

    def _author_focus(self, spans):
        """Put `spans` in edits.focus as a materialized (authoritative) plan."""
        doc = self._ok("get_edits")
        doc["focus"] = [dict(f) for f in spans]
        doc["focus_initialized"] = True
        doc["focus_plan_version"] = ed._FOCUS_PLAN_VERSION
        return ed.save_edits(self.session_dir, doc, duration=12.0)["focus"]

    def test_a_move_never_swallows_the_neighbour_cards_move(self):
        """The run this tool edits has to stop at the card boundary. It did
        not: the forward walk card-checked only ONE endpoint, so a move that
        touched the next card's move reached across it -- `off` at card 0's
        move DELETED card 1's grow span, and `softer`/`stronger` re-levelled
        it. Spans of different cards touching exactly is the handover the
        planner emits, not a hand-authored oddity."""
        self._cards()
        self._ok("set_render_options", zoom_style="frame")
        authored = self._author_focus(self.HANDOVER)
        card1 = [f for f in authored if f["card"] == 1]
        card0 = [f for f in authored if f["card"] == 0]
        got = self._ok("adjust_zoom", at=1.0, change="off")
        self.assertEqual(got["before"]["card"], 0)
        self.assertEqual(got["before"]["end"], 1.5,
                         "the move ran past its own card's handover")
        self.assertEqual(set(got["before"]["ids"]),
                         set(f["id"] for f in card0))
        self.assertEqual([f for f in got["focus"] if f["card"] == 1], card1,
                         "editing card 0's move changed card 1's")

    def test_softening_a_move_leaves_the_neighbour_cards_rung_alone(self):
        """The same reach, in the direction that is silent: a re-levelled
        neighbour is invisible in the response, which reports only the move
        you asked about."""
        self._cards()
        self._ok("set_render_options", zoom_style="frame")
        authored = self._author_focus(self.HANDOVER)
        card1 = [f for f in authored if f["card"] == 1]
        got = self._ok("adjust_zoom", at=5.0, change="softer")
        self.assertEqual(got["before"]["card"], 1)
        self.assertEqual(got["before"]["start"], 1.5)
        # Card 1's own two spans ARE one move and both drop a rung...
        self.assertEqual([f["level"] for f in got["focus"] if f["card"] == 1],
                         ["focus", "focus"])
        # ...and card 0's is untouched.
        self.assertEqual([f for f in got["focus"] if f["card"] == 0],
                         [f for f in authored if f["card"] == 0])
        self.assertEqual(len(card1), 2)

    def test_the_gentlest_move_is_left_alone_rather_than_deleted(self):
        """Removing a focus move is a one-way door (nothing adds one back),
        so 'softer' stops at the bottom rung and says what 'off' would do."""
        self._cards()
        self._ok("set_render_options", zoom_style="frame")
        self._ok("adjust_zoom", at=1.0)
        got = self._ok("adjust_zoom", at=1.0)
        self.assertEqual(got["action"], "unchanged")
        self.assertTrue(any("change='off'" in n for n in got["notes"]))

    def test_off_removes_the_whole_move(self):
        self._cards()
        self._ok("set_render_options", zoom_style="frame")
        before = self._ok("adjust_zoom", at=1.0)
        got = self._ok("adjust_zoom", at=1.0, change="off")
        self.assertEqual(got["action"], "removed")
        left = {f["id"] for f in got["focus"]}
        self.assertFalse(left & set(before["before"]["ids"]))

    def test_the_inside_the_card_camera_is_refused_with_the_lever_that_works(self):
        self._cards()
        self._ok("set_render_options", zoom_style="inside")
        msg = self._err("adjust_zoom", at=1.0)
        self.assertIn("set_render_options(zoom=", msg)
        self.assertIn("zoom_style='frame'", msg)

    def test_no_camera_at_all_points_at_the_switch(self):
        self._cards()
        msg = self._err("adjust_zoom", at=1.0)
        self.assertIn("zoom_style='frame'", msg)

    def test_a_zoom_factor_is_refused_on_a_multi_window_take(self):
        self._cards()
        self._ok("set_render_options", zoom_style="frame")
        msg = self._err("adjust_zoom", at=1.0, level=1.5)
        self.assertIn("rungs, not a factor", msg)

    # -- zoom_style ---------------------------------------------------------

    def test_zoom_style_is_one_word_for_the_pair(self):
        for style, (focus, inside) in (("frame", (True, False)),
                                       ("inside", (False, True)),
                                       ("both", (True, True)),
                                       ("off", (False, False))):
            got = self._ok("set_render_options", zoom_style=style)
            self.assertEqual(got["zoom_style"], style)
            self.assertEqual(got["render"]["window_focus"], focus)
            self.assertEqual(got["render"]["window_zoom"], inside)

    def test_an_explicit_flag_wins_over_the_shorthand(self):
        got = self._ok("set_render_options", zoom_style="both",
                       window_zoom=False)
        self.assertTrue(got["render"]["window_focus"])
        self.assertFalse(got["render"]["window_zoom"])
        self.assertEqual(got["zoom_style"], "frame")

    def test_an_unknown_zoom_style_is_refused(self):
        self.assertIn("zoom_style must be",
                      self._err("set_render_options", zoom_style="cinematic"))

    def test_zoom_style_is_derived_never_stored(self):
        self._ok("set_render_options", zoom_style="inside")
        self.assertNotIn("zoom_style", self._ok("get_edits")["render"])


class EverySurfaceSeedsThePick(unittest.TestCase):
    """A session recorded with a multi-window pick is materialized by
    whichever surface touches it first. Inside this server that means EVERY
    tool that reads edits goes through `_load_seeded_edits` -- a tool that
    called `ed.load_edits` directly would answer with the hand-drawn defaults
    (whole screen, grid, no composition camera) while `render_video` on the
    same session exported the pick.

    `preview_frame` shipped exactly that way: it is the surface whose whole
    job is showing what the export will look like, and it was the one left
    reading unseeded edits. Pin the rule at the source level so the next tool
    added inherits it instead of re-learning it."""

    SESSION = "20260501-000000"

    def test_only_the_seeding_helper_loads_edits_directly(self):
        direct = [ln.strip() for ln in
                  inspect.getsource(mcp_server).splitlines()
                  if "ed.load_edits(" in ln]
        self.assertEqual(
            len(direct), 1,
            "every edits read must go through McpServer._load_seeded_edits; "
            "these bypass it: {}".format(direct))
        self.assertIn("ed.load_edits(",
                      inspect.getsource(McpServer._load_seeded_edits))

    def test_a_first_touch_preview_materializes_the_pick(self):
        if shutil.which("ffmpeg") is None:
            self.skipTest("ffmpeg not available")
        root = tempfile.mkdtemp(prefix="mcp-pick-")
        self.addCleanup(shutil.rmtree, root, True)
        sdir = os.path.join(root, self.SESSION)
        os.makedirs(sdir)
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump({"fps": 30, "raw": "raw.mov", "events": "events.jsonl",
                       "cursor_mode": "system",
                       "logical_w": 320.0, "logical_h": 200.0,
                       "capture_windows": {
                           "units": "points",
                           "display_origin": [0.0, 0.0],
                           "windows": [
                               {"id": 7, "rect": [0, 0, 150, 200]},
                               {"id": 8, "rect": [160, 0, 150, 200]}]}}, f)
        with open(os.path.join(sdir, "events.jsonl"), "w") as f:
            f.write(json.dumps({"t": 0.5, "type": "down", "x": 60, "y": 60})
                    + "\n")
        proc = subprocess.run(
            ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
             "-f", "lavfi", "-i", "testsrc2=size=320x200:rate=30",
             "-t", "1.0", "-pix_fmt", "yuv420p",
             os.path.join(sdir, "raw.mov")],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            self.skipTest("ffmpeg cannot build synthetic raw.mov")
        server = McpServer(recordings_root=root)
        self.assertFalse(ed.has_edits(sdir), "fixture starts unseeded")
        # preview_frame FIRST -- nothing else has touched this session.
        resp = _call(server, "preview_frame",
                     {"session": self.SESSION, "time": 0.5})
        self.assertFalse(resp["result"]["isError"], _tool_text(resp))
        doc = ed.load_edits(sdir)
        self.assertEqual([w["window_id"] for w in doc["windows"]], [7, 8])
        self.assertTrue(doc["render"]["window_focus"])
        self.assertEqual(doc["render"]["window_layout"], "desktop")


class FocusMoveBoundary(unittest.TestCase):
    """`McpServer._focus_move` -- which spans count as ONE move.

    The planner splits a burst of attention into two abutting spans (the card
    grows, then the frame pushes in), so the unit `adjust_zoom` edits is the
    run, not the span a timestamp happened to land in. The run must stop at a
    CARD boundary, and cross-card spans that touch exactly are the planner's
    ordinary handover -- checking only one endpoint's card let the walk reach
    into the neighbour's move.
    """

    HANDOVER = [{"start": 0.0, "end": 1.5, "card": 0, "level": "focus"},
                {"start": 1.5, "end": 4.6, "card": 1, "level": "focus"},
                {"start": 4.6, "end": 6.4, "card": 1, "level": "full"}]

    def test_the_run_stops_at_a_card_handover(self):
        self.assertEqual(McpServer._focus_move(self.HANDOVER, 0), [0])

    def test_the_run_covers_one_cards_abutting_pair(self):
        self.assertEqual(McpServer._focus_move(self.HANDOVER, 1), [1, 2])
        self.assertEqual(McpServer._focus_move(self.HANDOVER, 2), [1, 2])

    def test_a_gap_breaks_the_run_even_on_the_same_card(self):
        spans = [{"start": 0.0, "end": 1.5, "card": 0, "level": "focus"},
                 {"start": 4.0, "end": 5.0, "card": 0, "level": "focus"}]
        self.assertEqual(McpServer._focus_move(spans, 0), [0])
        self.assertEqual(McpServer._focus_move(spans, 1), [1])

    def test_a_lone_span_is_its_own_move(self):
        spans = [{"start": 0.0, "end": 1.5, "card": 0, "level": "full"}]
        self.assertEqual(McpServer._focus_move(spans, 0), [0])


class SceneTakeMcpTests(unittest.TestCase):
    """A SCENE take (the window set changes mid-recording) has no single
    composition plan to retune: every scene is its own fleet, planned at
    render time. `adjust_zoom` must SAY that rather than reporting the fleet
    emitter's empty result as "this take has no moves" -- and it must never
    materialize a plan for it, because an empty `edits.focus` reads as "the
    user deleted every arc" and would switch the camera off."""

    SESSION = "20260401-000000"

    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="mcp-scene-")
        self.addCleanup(shutil.rmtree, self.root, True)
        sdir = os.path.join(self.root, self.SESSION)
        os.makedirs(sdir)
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump({"fps": 30, "t0_monotonic": 0.0,
                       "events": "events.jsonl",
                       "capture_scenes": [
                           {"index": 0, "channels": [{"file": "s0_c0.mov"},
                                                     {"file": "s0_c1.mov"}]},
                           {"index": 1, "channels": [{"file": "s1_c0.mov"},
                                                     {"file": "s1_c1.mov"},
                                                     {"file": "s1_c2.mov"}]},
                       ]}, f)
        with open(os.path.join(sdir, "events.jsonl"), "w") as f:
            f.write(json.dumps({"t": 1.0, "type": "down", "x": 5, "y": 5})
                    + "\n")
        self.session_dir = sdir
        self.server = McpServer(recordings_root=self.root)

    def test_adjust_zoom_names_the_scene_take_and_the_levers(self):
        resp = _call(self.server, "adjust_zoom",
                     {"session": self.SESSION, "at": 1.0})
        self.assertTrue(resp["result"]["isError"])
        msg = _tool_text(resp)
        self.assertIn("scene take", msg)
        self.assertIn("zoom_style", msg)

    def test_it_never_materializes_an_empty_plan(self):
        """The failure mode this guards: materializing [] would be read back
        as "every arc deleted" and silently kill the composition camera."""
        _call(self.server, "adjust_zoom", {"session": self.SESSION, "at": 1.0})
        doc = ed.load_edits(self.session_dir)
        self.assertFalse(doc.get("focus_initialized"))
        self.assertEqual(doc.get("focus") or [], [])
        self.assertTrue(doc["render"]["window_focus"],
                        "the seeded camera must survive the refusal")


class MultiNativeMcpTests(unittest.TestCase):
    """P3.4: the MCP tool surface on a multi-window native session.

    describe_session must return a channel-aware payload (no raw.mov to open),
    and the manual spatial-edit tools (add_zoom with coords, set_crop,
    set_windows) must refuse with a pointer to the automatic per-card camera
    rather than silently applying to a coordinate space that doesn't exist."""

    SESSION = "20260201-000000"

    @classmethod
    def setUpClass(cls):
        if shutil.which("ffmpeg") is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp(prefix="mcp-multi-native-")
        sdir = os.path.join(cls.root, cls.SESSION)
        os.makedirs(sdir)
        # Two synthetic channel files + a manifest, no top-level raw.mov.
        for i, (w, h) in enumerate([(320, 240), (400, 300)]):
            proc = subprocess.run(
                ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                 "-f", "lavfi", "-i",
                 "testsrc2=size={}x{}:rate=30".format(w, h),
                 "-t", "1.0", "-pix_fmt", "yuv420p",
                 os.path.join(sdir, "raw_{}.mov".format(i))],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                shutil.rmtree(cls.root, ignore_errors=True)
                raise unittest.SkipTest("ffmpeg cannot build channel file")
        meta = {
            "fps": 30, "t0_monotonic": 0.0,
            "logical_w": 1440.0, "logical_h": 900.0,
            "events": "events.jsonl", "cursor_mode": "system",
            "key_capture": "activity", "capture_backend": "sck",
            "capture_channels": [
                {"role": "screen_window", "file": "raw_0.mov",
                 "mode": "window_native", "id": 100, "app": "A", "title": "0",
                 "units": "points", "rect": [0, 0, 160, 120],
                 "logical_w": 160.0, "logical_h": 120.0,
                 "buffer_w": 320, "buffer_h": 240, "t0_monotonic": 0.0,
                 "track": "ok"},
                {"role": "screen_window", "file": "raw_1.mov",
                 "mode": "window_native", "id": 200, "app": "B", "title": "1",
                 "units": "points", "rect": [200, 0, 200, 150],
                 "logical_w": 200.0, "logical_h": 150.0,
                 "buffer_w": 400, "buffer_h": 300, "t0_monotonic": 0.02,
                 "track": "ok"},
            ],
        }
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump(meta, f)
        with open(os.path.join(sdir, "events.jsonl"), "w") as f:
            # Two clicks in the first window, early enough that the planner
            # keeps the cluster (a lone click near the end has no room for a
            # move and is dropped) -- so this take HAS a composition move for
            # adjust_zoom to address.
            for t in (0.2, 0.35, 0.5):
                f.write(json.dumps({"t": t, "type": "down",
                                    "x": 80, "y": 60}) + "\n")
        cls.session_dir = sdir

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        self.server = McpServer(recordings_root=self.root)
        # A PRISTINE session per test. These share one class fixture on disk,
        # and seeding is a one-shot that writes edits.json -- so without this
        # the seeding test was pinning a file some alphabetically-earlier test
        # had already created, and would have passed against a server that
        # never seeded at all.
        stale = os.path.join(self.session_dir, "edits.json")
        if os.path.exists(stale):
            os.remove(stale)

    def _err(self, name, arguments):
        resp = _call(self.server, name, arguments)
        self.assertTrue(resp["result"]["isError"],
                        "{} unexpectedly succeeded".format(name))
        return _tool_text(resp)

    def test_describe_session_returns_channels_without_a_raw(self):
        resp = _call(self.server, "describe_session", {"session": self.SESSION})
        self.assertFalse(resp["result"]["isError"], _tool_text(resp))
        info = _tool_json(resp)
        self.assertTrue(info.get("multi_native"))
        self.assertEqual(len(info["capture_channels"]), 2)
        self.assertIsNone(info["raw_path"])
        # Composite canvas dims, not a channel's.
        self.assertEqual((info["width"], info["height"]), (1920, 1080))

    def test_add_zoom_is_refused_with_a_pointer_to_window_zoom(self):
        msg = self._err("add_zoom", {"session": self.SESSION,
                                     "start": 0.1, "end": 0.5,
                                     "x": 80, "y": 60})
        self.assertIn("multi-window native", msg)
        self.assertIn("window_zoom", msg)

    def test_add_zoom_without_coords_is_also_refused(self):
        # A whole-frame zoom is equally moot -- no whole-screen camera in
        # cards mode.
        msg = self._err("add_zoom", {"session": self.SESSION,
                                     "start": 0.1, "end": 0.5})
        self.assertIn("multi-window native", msg)

    def test_set_crop_non_null_is_refused(self):
        msg = self._err("set_crop", {"session": self.SESSION,
                                     "crop": {"x": 0, "y": 0,
                                              "w": 100, "h": 100}})
        self.assertIn("multi-window native", msg)

    def test_set_crop_null_clear_is_allowed(self):
        # A null-clear is a harmless no-op even on multi-native -- a blanket
        # "clear all crops" script must not fail on these sessions.
        resp = _call(self.server, "set_crop",
                     {"session": self.SESSION, "crop": None})
        self.assertFalse(resp["result"]["isError"], _tool_text(resp))

    def test_set_windows_non_empty_is_refused(self):
        msg = self._err("set_windows", {"session": self.SESSION,
                                        "windows": [{"x": 0, "y": 0,
                                                     "w": 100, "h": 100}]})
        self.assertIn("multi-window native", msg)

    def test_it_opens_with_the_outer_frame_camera(self):
        """A fleet take's cards ARE windows picked off the screen, so the
        camera that moves the cards is the one it opens with -- and the MCP
        has to materialize that pick itself, or this surface answers with the
        hand-drawn defaults while the app shows something else."""
        info = _tool_json(_call(self.server, "describe_session",
                                {"session": self.SESSION}))
        self.assertEqual(info["zoom_style"], "frame")
        render_opts = info["edits"]["render"]
        self.assertTrue(render_opts["window_focus"])
        self.assertFalse(render_opts["window_zoom"])
        self.assertEqual(render_opts["window_layout"], "desktop")

    def test_adjust_zoom_materializes_the_plan_and_drops_one_move(self):
        """The per-move edit that `add_zoom` cannot do here: the fleet take's
        composition camera is auto-planned, so this is the surface that turns
        one of its moves into something the user owns."""
        resp = _call(self.server, "adjust_zoom",
                     {"session": self.SESSION, "at": 0.3, "change": "off"})
        self.assertFalse(resp["result"]["isError"], _tool_text(resp))
        got = _tool_json(resp)
        self.assertEqual(got["target"], "focus")
        self.assertEqual(got["action"], "removed")
        self.assertTrue(any("materialized" in n for n in got["notes"]))
        left = {f["id"] for f in got["focus"]}
        self.assertFalse(left & set(got["before"]["ids"]))

    def test_reset_hands_back_the_takes_own_defaults(self):
        """"Defaults" for a session recorded with a pick INCLUDE the pick --
        otherwise the reset response says the camera is off while the very
        next render seeds it back on."""
        got = _tool_json(_call(self.server, "reset_edits",
                               {"session": self.SESSION}))
        self.assertTrue(got["render"]["window_focus"])
        self.assertEqual(got["render"]["window_layout"], "desktop")

    def test_set_render_options_still_works(self):
        # The RIGHT way to express per-card zoom on a multi-native take:
        # render.window_zoom / window_focus via set_render_options must NOT
        # be refused -- that is what the add_zoom refusal points people to.
        resp = _call(self.server, "set_render_options",
                     {"session": self.SESSION,
                      "window_zoom": True, "window_focus": True})
        self.assertFalse(resp["result"]["isError"], _tool_text(resp))


if __name__ == "__main__":
    unittest.main()
