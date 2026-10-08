"""Unit tests for the studio_app backend additions: waveform peaks, project
names, background presets, camera-path payloads, thumbnails, and the
describe-session cache. No screen/input permissions needed -- sessions are
synthesized with ffmpeg's lavfi test source.
"""

import base64
import contextlib
import inspect
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

import cv2
import numpy as np

from autocine import camera_preview, edits, framing, render, studio_app, transcribe


FFMPEG = shutil.which("ffmpeg")


def _make_session(root, name="20240101-000000", w=320, h=200, dur=2.0, fps=30):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    raw = os.path.join(d, "raw.mov")
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=size={}x{}:rate={}:duration={}".format(w, h, fps, dur),
         "-pix_fmt", "yuv420p", raw],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    events = [
        {"t": 0.2, "type": "move", "x": 50, "y": 50},
        {"t": 0.5, "type": "down", "x": 100, "y": 80},
        {"t": 1.0, "type": "move", "x": 200, "y": 120},
        {"t": 1.2, "type": "down", "x": 220, "y": 140},
    ]
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    meta = {"fps": fps, "logical_w": w, "logical_h": h, "t0_monotonic": 0.0,
            "raw": "raw.mov", "events": "events.jsonl",
            "cursor_mode": "system"}
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d


class WaveformPeaks(unittest.TestCase):
    def test_sine_buffer_length_and_range(self):
        n = 8000
        t = np.arange(n, dtype=np.float64) / 8000.0
        samples = (0.5 * 32767 * np.sin(2 * np.pi * 440.0 * t)).astype("<i2")
        peaks = studio_app.waveform_peaks(samples.tobytes(), buckets=100)
        self.assertEqual(len(peaks), 100)
        for p in peaks:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
        # each bucket spans several 440 Hz periods, so every bucket peak
        # should sit near the 0.5 amplitude
        self.assertGreater(min(peaks), 0.45)
        self.assertLess(max(peaks), 0.51)

    def test_empty_buffer(self):
        self.assertEqual(studio_app.waveform_peaks(b"", buckets=600), [])

    def test_odd_byte_buffer_does_not_crash(self):
        self.assertEqual(studio_app.waveform_peaks(b"\x01", buckets=600), [])

    def test_bucket_count_capped_by_sample_count(self):
        samples = np.arange(10, dtype="<i2")
        peaks = studio_app.waveform_peaks(samples.tobytes(), buckets=600)
        self.assertEqual(len(peaks), 10)

    def test_full_scale_hits_close_to_one(self):
        samples = np.full(1000, -32768, dtype="<i2")
        peaks = studio_app.waveform_peaks(samples.tobytes(), buckets=4)
        self.assertEqual(len(peaks), 4)
        for p in peaks:
            self.assertAlmostEqual(p, 1.0, places=3)


class ProjectName(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def _path(self):
        return os.path.join(self.td, studio_app.PROJECT_FILENAME)

    def test_save_strips_and_round_trips(self):
        saved = studio_app.save_project_name(self.td, "  My Demo  ")
        self.assertEqual(saved, "My Demo")
        self.assertTrue(os.path.isfile(self._path()))
        self.assertEqual(studio_app.load_project_name(self.td), "My Demo")

    def test_clearing_removes_the_sidecar(self):
        studio_app.save_project_name(self.td, "Something")
        saved = studio_app.save_project_name(self.td, None)
        self.assertIsNone(saved)
        self.assertFalse(os.path.isfile(self._path()))
        self.assertIsNone(studio_app.load_project_name(self.td))

    def test_empty_string_normalizes_to_none(self):
        self.assertIsNone(studio_app.save_project_name(self.td, "   "))
        self.assertFalse(os.path.isfile(self._path()))

    def test_long_names_truncate_to_120(self):
        saved = studio_app.save_project_name(self.td, "x" * 300)
        self.assertEqual(len(saved), 120)
        self.assertEqual(studio_app.load_project_name(self.td), "x" * 120)

    def test_corrupt_sidecar_loads_as_none(self):
        with open(self._path(), "w") as f:
            f.write("{not json")
        self.assertIsNone(studio_app.load_project_name(self.td))

    def test_missing_sidecar_loads_as_none(self):
        self.assertIsNone(studio_app.load_project_name(self.td))


class LiveReloadRev(unittest.TestCase):
    """/api/rev exposes the server's boot id (bumps on every start) + a
    `reload` flag driven by AUTOCINE_LIVE. The frontend's live-reload
    poller uses these to detect a restart without cluttering Chrome with
    new tabs."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.env_prev = os.environ.get("AUTOCINE_LIVE")
        os.environ.pop("AUTOCINE_LIVE", None)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)
        if self.env_prev is None:
            os.environ.pop("AUTOCINE_LIVE", None)
        else:
            os.environ["AUTOCINE_LIVE"] = self.env_prev

    def test_default_mode_reports_reload_false(self):
        state = studio_app.StudioState(recordings_root=self.td)
        self.assertFalse(state.live_reload)
        self.assertTrue(state.boot_id)

    def test_env_var_flips_live_reload_on(self):
        os.environ["AUTOCINE_LIVE"] = "1"
        state = studio_app.StudioState(recordings_root=self.td)
        self.assertTrue(state.live_reload)

    def test_boot_id_differs_across_states(self):
        # Each StudioState instance stamps its own boot id — same shape the
        # supervisor's respawn will get across process starts.
        a = studio_app.StudioState(recordings_root=self.td)
        # small sleep guarantees time.time_ns() advances
        time.sleep(0.001)
        b = studio_app.StudioState(recordings_root=self.td)
        self.assertNotEqual(a.boot_id, b.boot_id)


class StudioServerSecurity(unittest.TestCase):
    """The browser-facing server is not an unauthenticated localhost API."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.server, self.base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        shutil.rmtree(self.td, ignore_errors=True)

    def _request(self, path, token=True, headers=None, data=None,
                 method=None):
        request_headers = dict(headers or {})
        if token:
            request_headers[studio_app._TOKEN_HEADER] = (
                self.server.security_token)
        req = urllib.request.Request(
            self.base + path, data=data, headers=request_headers,
            method=method)
        return urllib.request.urlopen(req)

    def _assert_http_error(self, status, *args, **kwargs):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self._request(*args, **kwargs)
        self.assertEqual(caught.exception.code, status)

    def test_html_bootstraps_token_and_anti_framing_headers(self):
        response = self._request("/bar.html", token=False)
        body = response.read().decode("utf-8")
        meta = re.search(
            r'<meta name="autocine-token" content="([^"]+)">', body)
        self.assertIsNotNone(meta)
        self.assertEqual(meta.group(1), self.server.security_token)
        self.assertIsNone(response.headers.get("Set-Cookie"))
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")
        self.assertIn("frame-ancestors 'none'",
                      response.headers.get("Content-Security-Policy") or "")

    def test_health_and_revision_are_the_only_token_free_api_bootstrap(self):
        health = json.loads(
            self._request("/api/health", token=False).read().decode())
        self.assertEqual(health, {"ok": True})
        rev = json.loads(
            self._request("/api/rev", token=False).read().decode())
        self.assertIn("rev", rev)
        self._assert_http_error(403, "/api/sessions", token=False)

    def test_token_header_allows_a_normal_api_request(self):
        payload = json.loads(self._request("/api/sessions").read().decode())
        self.assertEqual(payload, {"sessions": []})

    def test_query_token_is_limited_to_passive_resource_routes(self):
        query = "?autocine_token={}".format(self.server.security_token)
        self._assert_http_error(
            403, "/api/sessions" + query, token=False)
        frames_iter = iter([b"jpeg"])
        with mock.patch.object(studio_app.camera_preview, "frames",
                               return_value=frames_iter) as frames:
            response = self._request(
                "/api/camera/preview?ordinal=0&autocine_token={}".format(
                    self.server.security_token), token=False)
            response.read()
        frames.assert_called_once_with(0)

    def test_foreign_host_is_rejected_even_with_the_token(self):
        port = self.server.server_address[1]
        self._assert_http_error(
            403, "/api/sessions",
            headers={"Host": "evil.127.0.0.1.example:{}".format(port)})

    def test_host_without_the_nondefault_port_is_rejected(self):
        self._assert_http_error(
            403, "/api/sessions", headers={"Host": "127.0.0.1"})

    def test_non_loopback_and_wildcard_binds_are_rejected(self):
        # The shell gives its API token to its client, so that token cannot
        # authenticate arbitrary LAN clients. The socket itself stays local.
        for host in ("", "0.0.0.0", "::", "::1", "192.0.2.8",
                     "studio.example"):
            with self.subTest(host=host):
                with self.assertRaises(ValueError):
                    studio_app.build_server(
                        host, 0, recordings_root=self.td)
        # Keep the lower-level constructor safe too; build_server is not the
        # only callable symbol in this module.
        with self.assertRaises(ValueError):
            studio_app.StudioHttpServer(
                ("0.0.0.0", 0), state=self.server.state,
                static_dir=self.server.static_dir, public_host="0.0.0.0")

    def test_foreign_and_null_origins_are_rejected_with_the_token(self):
        self._assert_http_error(
            403, "/api/sessions",
            headers={"Origin": "https://attacker.example"})
        self._assert_http_error(
            403, "/api/sessions", headers={"Origin": "null"})

    def test_same_origin_is_accepted(self):
        payload = json.loads(self._request(
            "/api/sessions", headers={"Origin": self.base}).read().decode())
        self.assertEqual(payload, {"sessions": []})

    def test_unauthenticated_camera_get_cannot_open_the_device(self):
        with mock.patch.object(studio_app.camera_preview, "frames") as frames:
            self._assert_http_error(
                403, "/api/camera/preview?ordinal=0", token=False)
        frames.assert_not_called()

    def test_cross_port_cookie_cannot_authenticate_camera_get(self):
        # Cookies belong to a hostname, not a port.  A hostile app on another
        # localhost port can therefore send this cookie in an <img> request,
        # which has no Origin header.  The server must ignore it.
        with mock.patch.object(studio_app.camera_preview, "frames") as frames:
            self._assert_http_error(
                403, "/api/camera/preview?ordinal=0", token=False,
                headers={
                    "Cookie": "autocine_token={}".format(
                        self.server.security_token),
                    "Referer": "http://127.0.0.1:9999/hostile.html",
                })
        frames.assert_not_called()

    def test_post_requires_json_before_dispatch(self):
        with mock.patch.object(self.server.state, "stop_record") as stop:
            self._assert_http_error(
                415, "/api/record/stop",
                headers={"Content-Type": "text/plain"}, data=b"{}",
                method="POST")
        stop.assert_not_called()

    def test_post_requires_the_token_before_dispatch(self):
        with mock.patch.object(self.server.state, "stop_record") as stop:
            self._assert_http_error(
                403, "/api/record/stop", token=False,
                headers={"Content-Type": "application/json"}, data=b"{}",
                method="POST")
        stop.assert_not_called()

    def test_negative_content_length_is_rejected(self):
        self._assert_http_error(
            400, "/api/record/stop",
            headers={"Content-Type": "application/json",
                     "Content-Length": "-1"},
            method="POST")

    def test_cors_preflight_is_never_allowed(self):
        self._assert_http_error(
            403, "/api/record/start", token=False,
            headers={"Origin": "https://attacker.example",
                     "Access-Control-Request-Method": "POST",
                     "Access-Control-Request-Headers":
                         studio_app._TOKEN_HEADER},
            method="OPTIONS")


class WebRenderOutputContainment(unittest.TestCase):
    def test_client_out_is_ignored_and_export_stays_in_the_session(self):
        td = tempfile.mkdtemp()
        state = studio_app.StudioState(recordings_root=td)
        session_dir = os.path.join(td, "take")
        os.makedirs(session_dir)
        resolved = edits.default_edits()
        called = {}

        def fake_render(path, out_path=None, **kwargs):
            called["session_dir"] = path
            called["out_path"] = out_path
            return out_path

        try:
            with mock.patch.object(
                    state, "_resolved_edits",
                    return_value=(session_dir, {"duration": 1.0}, resolved)):
                with mock.patch.object(studio_app.ren, "render",
                                       side_effect=fake_render):
                    started = state.start_render(
                        "take", {"out": "http://attacker.example/upload"})
                    self.assertIn(started["status"], ("running", "done"))
                    state._render_thread.join(timeout=2)
            expected = os.path.join(session_dir, "output.mp4")
            self.assertEqual(called["session_dir"], session_dir)
            self.assertEqual(called["out_path"], expected)
            self.assertEqual(state.snapshot()["render"]["out_path"], expected)
            self.assertEqual(state.snapshot()["render"]["status"], "done")
        finally:
            shutil.rmtree(td, ignore_errors=True)


class WebSecuritySourcePins(unittest.TestCase):
    """The browser clients must present the token the server now requires."""

    @classmethod
    def setUpClass(cls):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.sources = {}
        for name in ("shared.js", "live-reload.js", "picker.js", "bar.js",
                     "facecam.js", "library.js", "editor.js"):
            with open(os.path.join(root, "studio_web", name)) as source:
                cls.sources[name] = source.read()

    def test_shared_api_reads_the_bootstrap_meta_and_sets_the_header(self):
        shared = self.sources["shared.js"]
        self.assertIn('meta[name="autocine-token"]', shared)
        self.assertIn('headers["X-AutoCine-Token"] = token', shared)
        api_body = shared.split("async function api(", 1)[1].split(
            "\n}", 1)[0]
        self.assertIn("autocineHeaders", api_body)

    def test_direct_fetch_clients_also_attach_the_token(self):
        self.assertIn("headers: autocineHeaders()",
                      self.sources["live-reload.js"])
        picker = self.sources["picker.js"]
        self.assertIn('headers["X-AutoCine-Token"] = token', picker)
        load_body = picker.split("function load()", 1)[1].split(
            "\n  }", 1)[0]
        self.assertIn("requestHeaders", load_body)

    def test_the_only_direct_post_writes_json_through_api(self):
        shared = self.sources["shared.js"]
        recorder = shared.split("function openRecorderBar()", 1)[1].split(
            "\n}", 1)[0]
        self.assertIn('api("/api/bar", { body: {} })', recorder)
        self.assertNotIn('fetch("/api/bar"', recorder)

    def test_passive_camera_media_and_thumbnails_use_tokenized_urls(self):
        shared = self.sources["shared.js"]
        self.assertIn('"autocine_token=" + encodeURIComponent(token)', shared)
        for name in ("bar.js", "facecam.js", "library.js", "editor.js"):
            source = self.sources[name]
            self.assertIn("autocineUrl(", source, name)
        self.assertNotRegex(
            self.sources["bar.js"] + self.sources["facecam.js"],
            r'\.src\s*=\s*"/api/camera/preview')
        self.assertNotRegex(
            self.sources["library.js"], r'\.src\s*=\s*"/api/thumb/')
        self.assertNotRegex(
            self.sources["editor.js"], r'\.src\s*=\s*"/api/media/')

    def test_pagehide_flush_uses_the_authenticated_keepalive_path(self):
        shared = self.sources["shared.js"]
        editor = self.sources["editor.js"]
        self.assertIn("opts.keepalive = true", shared)
        self.assertIn(
            "const keepalive = Boolean(requestOptions && requestOptions.keepalive)",
            editor)
        keepalive_error = editor.split("} catch (e) {", 1)[1]
        self.assertLess(keepalive_error.index("if (keepalive)"),
                        keepalive_error.index("e.status === 409"))
        pagehide = editor.split(
            'window.addEventListener("pagehide"', 1)[1].split(
                "\n  });", 1)[0]
        self.assertIn("scheduleSave.cancel()", pagehide)
        self.assertIn("saveNow({ keepalive: true })", pagehide)
        self.assertNotIn("navigator.sendBeacon", editor)
        self.assertNotRegex(pagehide, r"savedMutCount\s*=")


class StaticAssetFreshness(unittest.TestCase):
    """The app shell must never render from a stale cache.

    Served with no cache headers at all, WKWebView heuristically caches
    bar.css/bar.js and its disk cache outlives the process — so an edit on
    disk can stay invisible in the native bar through any number of
    relaunches. That burned a real debugging cycle: a CSS fix was on disk,
    verified, and still "didn't work" in the pill. Two defences, both pinned
    here: no-store going forward, and an mtime query that changes the URL so
    an already-poisoned cache can't answer from memory.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.static = os.path.join(self.td, "static")
        os.makedirs(self.static)
        with open(os.path.join(self.static, "bar.css"), "w") as f:
            f.write(".x{}")
        with open(os.path.join(self.static, "bar.js"), "w") as f:
            f.write("//")

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_local_css_and_js_links_get_an_mtime_query(self):
        html = b'<link rel="stylesheet" href="/bar.css"><script src="/bar.js">'
        out = studio_app._version_asset_links(html, self.static).decode()
        css_v = int(os.path.getmtime(os.path.join(self.static, "bar.css")))
        js_v = int(os.path.getmtime(os.path.join(self.static, "bar.js")))
        self.assertIn("/bar.css?v={}".format(css_v), out)
        self.assertIn("/bar.js?v={}".format(js_v), out)

    def test_the_query_changes_when_the_file_does(self):
        html = b'<link href="/bar.css">'
        before = studio_app._version_asset_links(html, self.static)
        os.utime(os.path.join(self.static, "bar.css"), (1_700_000_000, 1_700_000_000))
        after = studio_app._version_asset_links(html, self.static)
        self.assertNotEqual(before, after)
        self.assertIn(b"?v=1700000000", after)

    def test_unknown_and_remote_urls_are_left_alone(self):
        html = (b'<script src="https://cdn.example.com/x.js">'
                b'<link href="/does-not-exist.css">')
        self.assertEqual(studio_app._version_asset_links(html, self.static), html)

    def test_an_existing_query_is_preserved(self):
        html = b'<script src="/bar.js?mod=1">'
        out = studio_app._version_asset_links(html, self.static).decode()
        self.assertIn("/bar.js?mod=1&v=", out)

    def test_serving_never_breaks_on_a_bad_static_dir(self):
        html = b'<link href="/bar.css">'
        self.assertEqual(
            studio_app._version_asset_links(html, "/nope/not/here"), html)

    def test_shell_files_are_served_no_store(self):
        server, base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            for path, want in (("/bar.html", True), ("/bar.css", True),
                               ("/bar.js", True), ("/favicon.ico", False)):
                try:
                    r = urllib.request.urlopen(base + path)
                except urllib.error.HTTPError:
                    continue          # asset not present in this checkout
                cc = r.headers.get("Cache-Control") or ""
                self.assertEqual("no-store" in cc, want,
                                 "%s -> %r" % (path, cc))
        finally:
            server.shutdown()
            server.server_close()

    def test_media_is_served_no_store(self):
        """MEDIA must never be heuristically cacheable.

        Regression pin for a real freeze: media carried NO Cache-Control (and
        no validator, over HTTP/1.0), so Chrome cached it by heuristic. Every
        editor reload tears its <video> fleet down mid-fetch, and a poisoned
        partial entry was then served forever -- the element sat in
        networkState=LOADING / readyState=0 with no error and live playback
        hung at a scene seam indefinitely. Measured 2026-08-29: the same URL
        with a cache-buster loaded instantly while the plain one never did.
        """
        sess = os.path.join(self.td, "20200101-000000")
        os.makedirs(sess, exist_ok=True)
        with open(os.path.join(sess, "meta.json"), "w") as f:
            json.dump({"fps": 60, "raw": "raw.mov"}, f)
        with open(os.path.join(sess, "raw.mov"), "wb") as f:
            f.write(b"\x00" * 4096)
        server, base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            r = urllib.request.urlopen(
                base + "/api/media/20200101-000000/raw?autocine_token="
                + server.security_token)
            cc = r.headers.get("Cache-Control") or ""
            self.assertIn("no-store", cc)
        finally:
            server.shutdown()
            server.server_close()

    def test_the_real_bar_page_ships_versioned_assets(self):
        """End-to-end: the page the native pill actually loads."""
        server, base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            body = urllib.request.urlopen(base + "/bar.html").read().decode()
        finally:
            server.shutdown()
            server.server_close()
        links = re.findall(r'(?:href|src)="(/[^"]+\.(?:css|js))(\?[^"]*)?"', body)
        self.assertTrue(links, "bar.html should reference local css/js")
        for url, query in links:
            self.assertTrue(query and "v=" in query,
                            "unversioned asset in bar.html: {}".format(url))


class LaunchNativeBar(unittest.TestCase):
    """POST /api/bar spawns `python3 studio.py bar --port <same>` as a
    detached child; the child health-probes back and reuses this server, then
    opens the pywebview pill on its own main thread. Dedupes while alive."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    def test_pywebview_unavailable_returns_native_false(self):
        with mock.patch.object(studio_app, "native_bar_available",
                               return_value=False):
            with mock.patch.object(studio_app.subprocess, "Popen") as popen:
                r = self.state.launch_native_bar(5273)
        self.assertEqual(r["native"], False)
        self.assertEqual(r["launched"], False)
        popen.assert_not_called()

    @staticmethod
    def _bar_calls(popen):
        """Only the spawns that launched the BAR.

        `launch_native_bar` also spawns a watchdog over the bar (so an
        abnormal server death still reaps that detached window), and these
        tests patch the shared `subprocess` module, so both show up here."""
        out = []
        for call in popen.call_args_list:
            argv = call[0][0]
            if len(argv) > 2 and argv[2] == "bar":
                out.append(argv)
        return out

    @staticmethod
    def _watchdog_calls(popen):
        return [c[0][0] for c in popen.call_args_list
                if len(c[0][0]) > 1 and str(c[0][0][1]).endswith("_watchdog.py")]

    def test_launches_child_with_matching_port(self):
        fake = mock.Mock()
        fake.pid = 4242
        fake.poll.return_value = None
        with mock.patch.object(studio_app, "native_bar_available",
                               return_value=True):
            with mock.patch.object(studio_app.subprocess, "Popen",
                                   return_value=fake) as popen:
                r = self.state.launch_native_bar(5273)
        self.assertTrue(r["native"])
        self.assertTrue(r["launched"])
        self.assertEqual(r["pid"], 4242)
        argv = self._bar_calls(popen)[0]
        self.assertEqual(argv[0], sys.executable)
        self.assertTrue(argv[1].endswith("studio.py"))
        self.assertEqual(argv[2], "bar")
        self.assertEqual(argv[3], "--port")
        self.assertEqual(argv[4], "5273")
        kwargs = popen.call_args_list[0][1]
        # detached: no stdio inheritance, own session. That detachment is
        # why the bar needs reaping at all -- see the watchdog assertion
        # below and StudioState.shutdown().
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stdout"], subprocess.DEVNULL)
        self.assertEqual(kwargs["stderr"], subprocess.DEVNULL)
        self.assertTrue(kwargs.get("start_new_session"))
        # a watchdog is armed over the bar's pid, so a server that dies
        # abnormally still takes the detached window down with it
        dogs = self._watchdog_calls(popen)
        self.assertEqual(len(dogs), 1)
        self.assertIn("4242", dogs[0])

    def test_second_call_while_alive_does_not_respawn(self):
        alive = mock.Mock()
        alive.pid = 1111
        alive.poll.return_value = None  # still running
        with mock.patch.object(studio_app, "native_bar_available",
                               return_value=True):
            with mock.patch.object(studio_app.subprocess, "Popen",
                                   return_value=alive) as popen:
                self.state.launch_native_bar(5273)
                r = self.state.launch_native_bar(5273)
        self.assertEqual(len(self._bar_calls(popen)), 1)
        self.assertTrue(r["native"])
        self.assertFalse(r["launched"])
        self.assertEqual(r["pid"], 1111)

    def test_respawns_after_previous_child_exited(self):
        dead = mock.Mock()
        dead.pid = 1111
        dead.poll.return_value = 0    # exited
        fresh = mock.Mock()
        fresh.pid = 2222
        fresh.poll.return_value = None
        dog1, dog2 = mock.Mock(), mock.Mock()
        dog1.poll.return_value = None
        dog2.poll.return_value = None
        with mock.patch.object(studio_app, "native_bar_available",
                               return_value=True):
            # each launch spawns the bar AND a watchdog over it
            with mock.patch.object(studio_app.subprocess, "Popen",
                                   side_effect=[dead, dog1, fresh, dog2]) as popen:
                self.state.launch_native_bar(5273)
                r = self.state.launch_native_bar(5273)
        self.assertEqual(len(self._bar_calls(popen)), 2)
        self.assertEqual(r["pid"], 2222)

    def test_spawn_failure_returns_native_false(self):
        with mock.patch.object(studio_app, "native_bar_available",
                               return_value=True):
            with mock.patch.object(studio_app.subprocess, "Popen",
                                   side_effect=OSError("boom")):
                r = self.state.launch_native_bar(5273)
        self.assertFalse(r["native"])
        self.assertFalse(r["launched"])
        self.assertIn("boom", r["reason"])


def _window_entry(wid=9856, app="Google Chrome", title="Docs", **over):
    """A devices.list_windows entry (the exact 11-key contract shape)."""
    entry = {
        "id": wid, "app": app, "title": title,
        "label": "{} — {}".format(app, title),
        "x": 120.0, "y": 64.0, "w": 900.0, "h": 620.0,
        "display_id": 1, "display_origin": [0.0, 0.0], "main_display": True,
    }
    entry.update(over)
    return entry


class WindowPicker(unittest.TestCase):
    """GET /api/windows — the record-time "capture this window" picker.

    Its own endpoint on purpose: /api/devices is cached and polled by the
    bar, and a window list is stale seconds after it's built.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    @contextlib.contextmanager
    def _quartz(self, windows, displays):
        with mock.patch.object(studio_app.dev, "list_windows",
                               return_value=windows) as lw:
            with mock.patch.object(studio_app.dev, "displays_points",
                                   return_value=displays):
                yield lw

    def test_payload_shape(self):
        entries = [_window_entry(), _window_entry(wid=42, app="Terminal",
                                                  title="zsh")]
        with self._quartz(entries, [{"id": 1, "main": True}]):
            payload = self.state.windows()
        self.assertEqual(sorted(payload.keys()), ["available", "windows"])
        self.assertTrue(payload["available"])
        self.assertEqual(payload["windows"], entries)
        self.assertEqual([w["id"] for w in payload["windows"]], [9856, 42])

    def test_no_quartz_degrades_to_available_false(self):
        # devices returns [] for both when pyobjc/Quartz is missing; the UI
        # must be able to HIDE the picker rather than show an empty dropdown.
        with self._quartz([], []):
            payload = self.state.windows()
        self.assertFalse(payload["available"])
        self.assertEqual(payload["windows"], [])

    def test_no_pickable_windows_is_still_available(self):
        # An empty list with a working Quartz means "nothing pickable right
        # now" -- a different state from "window capture is impossible here".
        with self._quartz([], [{"id": 1, "main": True}]):
            payload = self.state.windows()
        self.assertTrue(payload["available"])
        self.assertEqual(payload["windows"], [])

    def test_excludes_this_process_and_the_detached_bar_child(self):
        child = mock.Mock()
        child.pid = 4242
        child.poll.return_value = None          # still alive
        self.state._bar_process = child
        with self._quartz([], [{"id": 1, "main": True}]) as lw:
            self.state.windows()
        pids = set(lw.call_args[1]["exclude_pids"])
        self.assertIn(os.getpid(), pids)
        # the pill runs in its own detached process, so os.getpid() alone
        # would happily offer the recording pill as a capture target
        self.assertIn(4242, pids)

    def test_dead_bar_child_pid_is_not_excluded(self):
        dead = mock.Mock()
        dead.pid = 4242
        dead.poll.return_value = 0              # exited
        self.state._bar_process = dead
        with self._quartz([], [{"id": 1, "main": True}]) as lw:
            self.state.windows()
        pids = set(lw.call_args[1]["exclude_pids"])
        self.assertEqual(pids, {os.getpid()})

    def test_http_route_serves_the_payload(self):
        entries = [_window_entry()]
        server, base = studio_app.build_server(
            "127.0.0.1", 0, recordings_root=self.td)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        try:
            with self._quartz(entries, [{"id": 1, "main": True}]):
                req = urllib.request.Request(
                    base + "/api/windows",
                    headers={studio_app._TOKEN_HEADER: server.security_token})
                body = json.loads(
                    urllib.request.urlopen(req).read().decode())
        finally:
            server.shutdown()
            server.server_close()
        self.assertTrue(body["available"])
        self.assertEqual(body["windows"], entries)

    def test_windows_are_not_folded_into_devices(self):
        # /api/devices is polled and cached by the bar; a window list there
        # would be served stale.
        with mock.patch.object(studio_app.dev, "list_avf_devices",
                               return_value={"video": [], "audio": []}):
            with mock.patch.object(studio_app.dev, "find_screen_device",
                                   return_value=None):
                with mock.patch.object(studio_app.camera_preview,
                                       "list_cameras", return_value=[]):
                    payload = self.state.devices()
        self.assertNotIn("windows", payload)


class StartRecordWindowCapture(unittest.TestCase):
    """_start_record resolves the picked window BEFORE the session exists.

    A window that no longer resolves is a hard 400 -- never a silent
    full-screen take, which would be a wasted recording.
    """

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    @contextlib.contextmanager
    def _ready(self, window_entry):
        """Everything _start_record touches before the window check, stubbed."""
        with contextlib.ExitStack() as stack:
            p = stack.enter_context
            p(mock.patch.object(self.state, "permissions",
                                return_value={"missing_required": [],
                                              "checks": {}}))
            p(mock.patch.object(studio_app.dev, "list_avf_devices",
                                return_value={"video": [], "audio": []}))
            p(mock.patch.object(studio_app.dev, "find_screen_device",
                                return_value=2))
            rect = p(mock.patch.object(studio_app.dev, "window_rect_points",
                                       return_value=window_entry))
            recorder = p(mock.patch.object(studio_app.rec, "Recorder"))
            # keep the worker thread from ever running the mocked recorder
            p(mock.patch.object(studio_app.threading, "Thread"))
            yield rect, recorder

    def test_stale_window_id_is_a_400_and_starts_nothing(self):
        with self._ready(None) as (_, recorder):
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state._start_record({"window_id": 9856})
        self.assertEqual(ctx.exception.status, 400)
        self.assertIn("no longer on screen", str(ctx.exception))
        recorder.assert_not_called()
        # nothing was created on disk and no take is in flight
        self.assertEqual(os.listdir(self.td), [])
        self.assertEqual(self.state.snapshot()["record"]["status"], "idle")

    def test_unparseable_window_id_is_also_a_400(self):
        with self._ready(None) as (rect, recorder):
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state._start_record({"window_id": "not-a-number"})
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(rect.call_args[0][0], -1)
        recorder.assert_not_called()

    def test_resolved_window_is_forwarded_to_the_recorder(self):
        entry = _window_entry()
        with self._ready(entry) as (rect, recorder):
            self.state._start_record({"window_id": 9856})
        self.assertEqual(rect.call_args[0][0], 9856)
        self.assertIn(os.getpid(), set(rect.call_args[1]["exclude_pids"]))
        self.assertEqual(recorder.call_args[1]["capture_window"], entry)

    def test_full_screen_take_passes_no_capture_window(self):
        # The off switch: no window picked -> capture_window is None, the
        # recorder writes no meta key, and render.py takes the literal
        # pre-feature path.
        with self._ready(None) as (rect, recorder):
            self.state._start_record({})
        rect.assert_not_called()
        self.assertIsNone(recorder.call_args[1]["capture_window"])

    def test_recorder_actually_accepts_the_kwarg(self):
        # The mocked Recorder above would happily swallow a kwarg record.py
        # doesn't have; pin the real signature so the seam can't rot.
        import inspect
        sig = inspect.signature(studio_app.rec.Recorder.__init__)
        self.assertIn("capture_window", sig.parameters)
        self.assertIsNone(sig.parameters["capture_window"].default)

    def test_empty_window_id_means_full_screen(self):
        for value in (None, "", "   "):
            # each _start_record leaves a take "in flight" (the worker thread
            # is stubbed out, so nothing ever clears it) -- reset between runs
            self.state._record_status = {"status": "idle", "message": "",
                                         "session": None}
            with self._ready(None) as (rect, recorder):
                self.state._start_record({"window_id": value})
            rect.assert_not_called()
            self.assertIsNone(recorder.call_args[1]["capture_window"],
                              "window_id={!r}".format(value))


class StartRecordMultiNative(unittest.TestCase):
    """P3.4: `occlusion_free` + `window_ids` of 2-4 goes through as a
    multi-window native take -- SCK forced, arrange disabled, capture_windows
    + window_native forwarded to the Recorder."""

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    @contextlib.contextmanager
    def _ready(self, entries):
        """Stub everything _start_record touches; `entries` is what
        window_rect_points returns per id (a dict, or None to drop)."""
        with contextlib.ExitStack() as stack:
            p = stack.enter_context
            p(mock.patch.object(self.state, "permissions",
                                return_value={"missing_required": [],
                                              "checks": {}}))
            p(mock.patch.object(studio_app.dev, "list_avf_devices",
                                return_value={"video": [], "audio": []}))
            p(mock.patch.object(studio_app.dev, "find_screen_device",
                                return_value=2))

            def _rect(wid, exclude_pids=None):
                return entries.get(int(wid))
            p(mock.patch.object(studio_app.dev, "window_rect_points",
                                side_effect=_rect))
            # arrange must never run in native mode -- fail loudly if it does.
            arr = p(mock.patch.object(studio_app.arrange, "plan_separation",
                                      side_effect=AssertionError(
                                          "arrange ran in native mode")))
            recorder = p(mock.patch.object(studio_app.rec, "Recorder"))
            p(mock.patch.object(studio_app.threading, "Thread"))
            yield recorder, arr

    def test_multi_native_forces_sck_and_forwards_windows(self):
        e0 = _window_entry(wid=100, app="A", title="0")
        e1 = _window_entry(wid=200, app="B", title="1")
        with self._ready({100: e0, 200: e1}) as (recorder, _arr):
            self.state._start_record({
                "window_ids": [100, 200], "occlusion_free": True,
                "capture_backend": "avfoundation",   # must be overridden
                "arrange": True,                      # must be forced off
            })
        kw = recorder.call_args[1]
        self.assertEqual(kw["backend"], "sck")
        self.assertTrue(kw["window_native"])
        self.assertEqual(len(kw["capture_windows"]), 2)
        self.assertEqual([w["id"] for w in kw["capture_windows"]], [100, 200])

    def test_arrange_is_disabled_for_native(self):
        # The arrange planner is patched to raise if reached; reaching the
        # Recorder without it raising proves _start_record never ran arrange
        # in native mode -- even with arrange:true in the payload. The state
        # stamped on the recorder is an honest "none" (not applicable), not
        # the "declined" a display-crop overlapping pick would get.
        e0 = _window_entry(wid=100)
        e1 = _window_entry(wid=200)
        with self._ready({100: e0, 200: e1}) as (recorder, _arr):
            self.state._start_record({
                "window_ids": [100, 200], "occlusion_free": True,
                "arrange": True})
        self.assertTrue(recorder.call_args[1]["window_native"])
        # _arrange_state is set on the recorder instance after construction.
        self.assertEqual(self.state._active_recorder._arrange_state, "none")

    def test_occlusion_free_with_no_windows_is_a_400(self):
        with self._ready({}) as (recorder, _arr):
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state._start_record({"occlusion_free": True})
        self.assertEqual(ctx.exception.status, 400)
        recorder.assert_not_called()

    def test_single_window_occlusion_free_still_works(self):
        # N=1 occlusion-free is the existing single-file native path -- it must
        # keep going through as capture_window (not capture_windows).
        e0 = _window_entry(wid=100)
        with self._ready({100: e0}) as (recorder, _arr):
            self.state._start_record({
                "window_id": 100, "occlusion_free": True})
        kw = recorder.call_args[1]
        self.assertEqual(kw["backend"], "sck")
        self.assertTrue(kw["window_native"])
        self.assertEqual(kw["capture_window"]["id"], 100)


class CaptureWindowReachesTheApi(unittest.TestCase):
    """render.describe_session's window-capture fields must survive the
    describe cache and land in the session payload the editor reads."""

    @classmethod
    def setUpClass(cls):
        if FFMPEG is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp()
        cls.plain = "20240101-000010"
        cls.cropped = "20240101-000011"
        _make_session(cls.root, name=cls.plain)
        d = _make_session(cls.root, name=cls.cropped)
        meta_path = os.path.join(d, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        # logical size == file size in a synthetic session, so scale is 1:1
        # and the crop rect is also the expected pixel rect.
        meta["capture_window"] = {
            "id": 9856, "app": "Google Chrome", "title": "Docs",
            "units": "points",
            "rect": [40.0, 20.0, 160.0, 100.0],
            "display_origin": [0.0, 0.0],
            "source": "quartz",
            "resnapshot": True,
            "end_rect": [90.0, 20.0, 160.0, 100.0],   # moved 50pt right
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _state(self):
        return studio_app.StudioState(recordings_root=self.root)

    def test_session_payload_reports_cropped_and_raw_dims(self):
        info = self._state().get_session(self.cropped)
        # width/height are THE source coordinate space for editor.js
        self.assertEqual([info["width"], info["height"]], [160, 100])
        self.assertEqual([info["raw_width"], info["raw_height"]], [320, 200])
        cw = info["capture_window"]
        self.assertEqual(cw["id"], 9856)
        self.assertEqual(cw["app"], "Google Chrome")
        self.assertTrue(cw["moved"])          # end_rect drifted 50pt

    def test_fields_survive_the_describe_cache(self):
        state = self._state()
        state._describe(os.path.join(self.root, self.cropped))   # warm
        info = state.get_session(self.cropped)                   # cached path
        self.assertEqual([info["width"], info["height"]], [160, 100])
        self.assertIsNotNone(info["capture_window"])
        self.assertEqual(info["raw_width"], 320)

    def test_listing_reports_the_cropped_size(self):
        listed = {s["session"]: s for s in self._state().list_sessions()}
        self.assertEqual(listed[self.cropped]["size"], [160, 100])
        self.assertEqual(listed[self.plain]["size"], [320, 200])

    def test_a_session_without_the_key_is_unchanged(self):
        info = self._state().get_session(self.plain)
        self.assertIsNone(info["capture_window"])
        self.assertEqual([info["width"], info["height"]], [320, 200])
        self.assertEqual([info["raw_width"], info["raw_height"]], [320, 200])


class BackgroundPresets(unittest.TestCase):
    def test_ids_match_framing_presets_in_order(self):
        presets = studio_app.background_presets()
        self.assertEqual([p["id"] for p in presets],
                         list(framing._PRESETS.keys()))

    def test_colors_are_hex_top_then_bottom(self):
        presets = {p["id"]: p["colors"] for p in studio_app.background_presets()}
        for name, (bottom_rgb, top_rgb) in framing._PRESETS.items():
            colors = presets[name]
            self.assertEqual(len(colors), 2)
            for c in colors:
                self.assertRegex(c, r"^#[0-9a-f]{6}$")
            self.assertEqual(
                colors[0],
                "#{:02x}{:02x}{:02x}".format(*[int(v) for v in top_rgb]))
            self.assertEqual(
                colors[1],
                "#{:02x}{:02x}{:02x}".format(*[int(v) for v in bottom_rgb]))

    def test_aurora_specifically(self):
        presets = {p["id"]: p["colors"] for p in studio_app.background_presets()}
        # aurora: top (36, 30, 32), bottom (116, 66, 92)
        self.assertEqual(presets["aurora"], ["#241e20", "#74425c"])


class SyntheticSession(unittest.TestCase):
    """Tests that need a real (tiny, synthetic) recording on disk."""

    @classmethod
    def setUpClass(cls):
        if FFMPEG is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp()
        cls.name = "20240101-000000"
        cls.session_dir = _make_session(cls.root, name=cls.name)
        cls.raw_path = os.path.join(cls.session_dir, "raw.mov")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        # Per-test cleanup of cache/sidecar files the tests may create.
        for fn in (studio_app.PROJECT_FILENAME, studio_app._THUMB_FILENAME,
                   studio_app._WAVEFORM_FILENAME, "edits.json", "output.mp4",
                   "output.gif"):
            try:
                os.remove(os.path.join(self.session_dir, fn))
            except OSError:
                pass

    def _state(self):
        return studio_app.StudioState(recordings_root=self.root)

    # -- render.camera_path ------------------------------------------------

    def test_camera_path_shape_and_metadata(self):
        data = render.camera_path(self.session_dir, stride=2, max_zoom=2.0)
        for key in ("times", "cx", "cy", "z", "fps", "source", "win",
                    "canvas", "duration"):
            self.assertIn(key, data)
        n = len(data["times"])
        self.assertEqual(len(data["cx"]), n)
        self.assertEqual(len(data["cy"]), n)
        self.assertEqual(len(data["z"]), n)
        self.assertGreater(n, 0)
        self.assertEqual(data["source"], [320, 200])
        self.assertEqual(data["canvas"], [320, 200])
        self.assertEqual(data["win"], [320.0, 200.0])
        self.assertAlmostEqual(data["fps"], 30.0, places=3)
        self.assertAlmostEqual(data["duration"], 2.0, places=2)
        n_frames = int(round(data["duration"] * data["fps"]))
        self.assertEqual(n, int(math.ceil(n_frames / 2.0)))
        self.assertAlmostEqual(data["times"][0], 0.0)

    def test_camera_path_zoom_stays_in_band_and_activates(self):
        data = render.camera_path(self.session_dir, stride=1, max_zoom=2.0)
        z = data["z"]
        self.assertGreaterEqual(min(z), 1.0)
        self.assertLessEqual(max(z), 2.0 + 1e-6)
        # there are clicks, so the auto-zoom must actually do something
        self.assertGreater(max(z), 1.0)

    def test_camera_path_window_restricts_payload_not_simulation(self):
        full = render.camera_path(self.session_dir, stride=1, max_zoom=2.0)
        part = render.camera_path(self.session_dir, stride=1, max_zoom=2.0,
                                  t_start=0.5, t_end=1.5)
        self.assertLess(len(part["times"]), len(full["times"]))
        for t in part["times"]:
            self.assertGreaterEqual(t, 0.5 - 1e-6)
            self.assertLessEqual(t, 1.5 + 1e-6)
        # windowed values must be identical to the full simulation's samples
        lookup = dict(zip(full["times"], zip(full["cx"], full["cy"], full["z"])))
        for i, t in enumerate(part["times"]):
            self.assertIn(t, lookup)
            cx, cy, z = lookup[t]
            self.assertEqual(part["cx"][i], cx)
            self.assertEqual(part["cy"][i], cy)
            self.assertEqual(part["z"][i], z)

    def test_camera_path_ignores_declared_render_only_options(self):
        # Declared-render-only kwargs (see _CAMERA_PATH_IGNORED) are
        # dropped so studio_app can thread the full option dict through
        # without having to filter it per-endpoint.
        data = render.camera_path(self.session_dir, stride=4,
                                  click_fx=True, spotlight=False,
                                  fade=0.4)
        self.assertGreater(len(data["times"]), 0)

    def test_camera_path_rejects_unknown_kwargs(self):
        # But anything NOT in the ignore-list must be loud, not silent,
        # or a typo silently corrupts editor previews.
        with self.assertRaises(TypeError):
            render.camera_path(self.session_dir, stride=4,
                               some_future_option=123)

    def test_camera_path_ignores_cuts_without_changing_the_path(self):
        # Cuts are in _CAMERA_PATH_IGNORED (editor previews SOURCE time,
        # only the export ripples -- the speedup precedent): threading the
        # full kwargs dict with a cuts array must neither TypeError nor
        # alter the simulated path.
        base = render.camera_path(self.session_dir, stride=4)
        with_cuts = render.camera_path(
            self.session_dir, stride=4,
            cuts=[{"start": 0.2, "end": 0.6}])
        self.assertEqual(with_cuts["times"], base["times"])
        self.assertEqual(with_cuts["cx"], base["cx"])
        self.assertEqual(with_cuts["cy"], base["cy"])
        self.assertEqual(with_cuts["z"], base["z"])

    def test_patch_from_options_carries_cuts(self):
        # Live (unsaved) editor options must reach the resolved-edits
        # patch, or an unsaved cuts change would preview differently from
        # a saved one -- the hand-written-list gap `focus` fell into.
        patch = studio_app._patch_from_options(
            {"cuts": [{"start": 1.0, "end": 2.0}], "zoom": 2.5})
        self.assertEqual(patch["cuts"], [{"start": 1.0, "end": 2.0}])

    def test_state_camera_path_includes_trim(self):
        state = self._state()
        data = state.camera_path(self.name, {"stride": 3})
        self.assertIn("trim", data)
        self.assertAlmostEqual(data["trim"]["start"], 0.0)
        self.assertAlmostEqual(data["trim"]["end"], data["duration"], places=2)
        # an ephemeral trim patch resolves into the reported window
        data2 = state.camera_path(self.name, {"stride": 3, "trim_end": 1.0})
        self.assertAlmostEqual(data2["trim"]["end"], 1.0)
        # trim never restricts the simulated/returned path itself
        self.assertEqual(len(data2["times"]), len(data["times"]))

    def test_state_camera_path_windows_mode_carries_the_layout(self):
        """Windows mode has no per-frame path, so the editor redraws the
        composite itself -- the payload has to hand it the grid and the
        backdrop, or live playback falls back to the bare recording."""
        state = self._state()
        wins = [{"x": 0, "y": 0, "w": 100, "h": 80},
                {"x": 100, "y": 40, "w": 120, "h": 90}]
        data = state.camera_path(self.name, {"stride": 3, "windows": wins})
        self.assertTrue(data["windows_mode"])
        self.assertNotIn("times", data)          # still no camera path
        self.assertIn("trim", data)
        self.assertEqual(len(data["canvas"]), 2)
        self.assertEqual(len(data["cells"]), len(wins))
        for cell in data["cells"]:
            for key in ("x", "y", "w", "h", "radius", "src"):
                self.assertIn(key, cell)
            self.assertEqual(len(cell["src"]), 4)
        self.assertTrue(data["plate_jpeg_base64"])
        base64.b64decode(data["plate_jpeg_base64"])    # decodes cleanly

    def test_state_camera_path_without_windows_has_no_layout(self):
        """The layout keys are windows-mode-only -- an ordinary session must
        not start paying for a plate render it will never draw."""
        data = self._state().camera_path(self.name, {"stride": 3})
        self.assertNotIn("windows_mode", data)
        self.assertNotIn("cells", data)
        self.assertNotIn("plate_jpeg_base64", data)

    def test_state_camera_path_clamps_stride(self):
        state = self._state()
        a = state.camera_path(self.name, {"stride": 0})     # -> 1
        b = state.camera_path(self.name, {"stride": 999})   # -> 10
        self.assertGreater(len(a["times"]), len(b["times"]))
        n_frames = int(round(a["duration"] * a["fps"]))
        self.assertEqual(len(a["times"]), n_frames)
        self.assertEqual(len(b["times"]), int(math.ceil(n_frames / 10.0)))

    # -- thumbnails ----------------------------------------------------------

    def test_thumbnail_jpeg_bytes_and_width(self):
        data = studio_app._thumbnail_jpeg(self.raw_path)
        self.assertTrue(data.startswith(b"\xff\xd8"))
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img)
        self.assertEqual(img.shape[1], 640)

    def test_state_thumbnail_writes_and_reuses_cache(self):
        state = self._state()
        cache = os.path.join(self.session_dir, studio_app._THUMB_FILENAME)
        data = state.thumbnail(self.name)
        self.assertTrue(os.path.isfile(cache))
        self.assertTrue(data.startswith(b"\xff\xd8"))
        # a fresh cache file is served as-is (prove it by planting bytes)
        planted = b"\xff\xd8planted-cache-bytes"
        with open(cache, "wb") as f:
            f.write(planted)
        raw_mtime = os.path.getmtime(self.raw_path)
        os.utime(cache, (raw_mtime + 10, raw_mtime + 10))
        self.assertEqual(state.thumbnail(self.name), planted)
        # touching the recording invalidates the cache -> regenerated
        os.utime(self.raw_path, (raw_mtime + 20, raw_mtime + 20))
        regenerated = state.thumbnail(self.name)
        self.assertNotEqual(regenerated, planted)
        self.assertTrue(regenerated.startswith(b"\xff\xd8"))

    # -- waveform --------------------------------------------------------

    def test_waveform_without_audio_is_empty(self):
        state = self._state()
        data = state.waveform(self.name)
        self.assertEqual(data["peaks"], [])
        self.assertAlmostEqual(data["duration"], 2.0, places=2)

    # -- describe cache -----------------------------------------------------

    def test_describe_cache_hits_and_mtime_invalidation(self):
        state = self._state()
        real = render.describe_session
        calls = []

        def counting(session_dir, include_click_times=False):
            calls.append(session_dir)
            return real(session_dir,
                        include_click_times=include_click_times)

        with mock.patch.object(studio_app.ren, "describe_session", counting):
            a = state._describe(self.session_dir)
            b = state._describe(self.session_dir, include_click_times=True)
            self.assertEqual(len(calls), 1)
            self.assertNotIn("click_times", a)
            self.assertIn("click_times", b)
            self.assertEqual(len(b["click_times"]), 2)
            # invalidate by touching the recording
            mtime = os.path.getmtime(self.raw_path)
            os.utime(self.raw_path, (mtime + 5, mtime + 5))
            state._describe(self.session_dir)
            self.assertEqual(len(calls), 2)

    def test_describe_cache_refreshes_output_flags(self):
        state = self._state()
        first = state._describe(self.session_dir)
        self.assertFalse(first["has_output_mp4"])
        with open(os.path.join(self.session_dir, "output.mp4"), "wb") as f:
            f.write(b"\x00")
        real = render.describe_session
        calls = []

        def counting(session_dir, include_click_times=False):
            calls.append(session_dir)
            return real(session_dir,
                        include_click_times=include_click_times)

        with mock.patch.object(studio_app.ren, "describe_session", counting):
            second = state._describe(self.session_dir)
        self.assertEqual(calls, [])          # cache hit...
        self.assertTrue(second["has_output_mp4"])  # ...but flags are fresh

    # -- project names in session payloads --------------------------------

    def test_sessions_and_detail_include_project_name(self):
        state = self._state()
        self.assertIsNone(state.get_session(self.name)["name"])
        state.set_project_name(self.name, "  Launch Video ")
        listed = [s for s in state.list_sessions()
                  if s["session"] == self.name]
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["name"], "Launch Video")
        self.assertEqual(state.get_session(self.name)["name"], "Launch Video")
        self.assertEqual(state.get_project_name(self.name), "Launch Video")
        self.assertIsNone(state.set_project_name(self.name, None))
        self.assertIsNone(state.get_project_name(self.name))

    # -- delete guards ------------------------------------------------------

    def test_delete_refuses_while_render_running(self):
        state = self._state()
        state._render_status = {"status": "running", "session": self.name}
        with self.assertRaises(studio_app.StudioError) as ctx:
            state.delete_session(self.name)
        self.assertEqual(ctx.exception.status, 409)
        self.assertTrue(os.path.isdir(self.session_dir))

    def test_delete_refuses_while_recording_running(self):
        state = self._state()
        state._record_status = {"status": "recording", "session": self.name}
        with self.assertRaises(studio_app.StudioError) as ctx:
            state.delete_session(self.name)
        self.assertEqual(ctx.exception.status, 409)
        self.assertTrue(os.path.isdir(self.session_dir))


class WebCutAuthoring(unittest.TestCase):
    """The editor can author a cut, and cannot author one the renderer
    refuses. Before this, `cuts` was absent from the editor's save payload
    (so nothing it wrote reached disk) and the write guards lived only in
    mcp_server (so the web path enforced nothing)."""

    @classmethod
    def setUpClass(cls):
        if FFMPEG is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp()
        cls.name = "20240101-000000"
        cls.session_dir = _make_session(cls.root, name=cls.name, dur=12.0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def setUp(self):
        try:
            os.remove(os.path.join(self.session_dir, "edits.json"))
        except OSError:
            pass
        self.state = studio_app.StudioState(recordings_root=self.root)

    def test_a_saved_cut_round_trips(self):
        saved = self.state.save_session_edits(
            self.name, {"cuts": [{"id": "c1", "start": 1.0, "end": 2.0}]})
        self.assertEqual(len(saved["cuts"]), 1)
        self.assertAlmostEqual(saved["cuts"][0]["start"], 1.0)
        self.assertAlmostEqual(saved["cuts"][0]["end"], 2.0)
        self.assertEqual(saved["cuts"][0]["id"], "c1")
        # and it is on disk, not just in the response
        again = self.state.get_edits(self.name)
        self.assertEqual(len(again["cuts"]), 1)

    def test_clearing_cuts_round_trips(self):
        self.state.save_session_edits(
            self.name, {"cuts": [{"id": "c1", "start": 1.0, "end": 2.0}]})
        saved = self.state.save_session_edits(self.name, {"cuts": []})
        self.assertEqual(saved["cuts"], [])

    def test_a_cut_that_swallows_the_take_is_rejected(self):
        with self.assertRaises(studio_app.CutsRejected) as ctx:
            self.state.save_session_edits(
                self.name, {"cuts": [{"id": "c1", "start": 0.0, "end": 12.0}]})
        self.assertIn("whole trim window", str(ctx.exception))
        # nothing was written
        self.assertEqual(self.state.get_edits(self.name)["cuts"], [])

    def test_the_rejection_carries_the_winning_document(self):
        # The editor autosaves a WHOLE document, so a refusal it cannot
        # recover from would wedge every later edit behind a failing save.
        self.state.save_session_edits(
            self.name, {"cuts": [{"id": "keep", "start": 0.5, "end": 1.0}]})
        with self.assertRaises(studio_app.CutsRejected) as ctx:
            self.state.save_session_edits(
                self.name, {"cuts": [{"id": "c1", "start": 0.0, "end": 12.0}]})
        current = ctx.exception.current
        self.assertEqual([c["id"] for c in current["cuts"]], ["keep"])
        self.assertIn("rev", current)

    def test_the_guard_sees_the_trim_already_on_disk(self):
        # A patch carrying only `cuts` still has to answer for the trim that
        # is already saved -- the guard runs on the MERGED document.
        self.state.save_session_edits(
            self.name, {"trim": {"start": 1.0, "end": 2.0}})
        with self.assertRaises(studio_app.CutsRejected):
            self.state.save_session_edits(
                self.name, {"cuts": [{"id": "c1", "start": 1.0, "end": 2.0}]})

    def test_a_stale_rev_reports_the_conflict_not_the_budget(self):
        # Ordering: the CAS runs first, so a save that is BOTH stale and
        # over-budget is a 409, not a 400 about a document it never owned.
        self.state.save_session_edits(self.name, {"markers": []})
        with self.assertRaises(studio_app.EditsConflict):
            self.state.save_session_edits(
                self.name, {"cuts": [{"id": "c1", "start": 0.0, "end": 12.0}]},
                base_rev=0)

    def test_too_many_ranges_is_rejected(self):
        # Spaced so the frame-grid snap cannot merge neighbours into
        # fewer ranges -- the cap is about the EMITTED span count.
        many = [{"id": "c{}".format(i), "start": i * 0.15, "end": i * 0.15 + 0.04}
                for i in range(edits.CUT_MAX_RANGES + 1)]
        with self.assertRaises(studio_app.CutsRejected) as ctx:
            self.state.save_session_edits(self.name, {"cuts": many})
        self.assertIn("too many cut ranges", str(ctx.exception))

    def _write_transcript(self, words):
        """A cache the loader will actually accept: keyed on raw.mov's mtime
        and on this machine's engine identity."""
        raw = os.path.join(self.session_dir, "raw.mov")
        doc = {
            "mtime": os.path.getmtime(raw),
            "engine": transcribe.engine_id(model=transcribe.find_model()),
            "status": "ok", "reason": "", "language": "en", "model": "test",
            "words": words,
            "segments": [{"t": words[0]["t"], "dur": 1.0,
                          "text": " ".join(w["text"] for w in words)}],
        }
        with open(os.path.join(self.session_dir,
                               transcribe.TRANSCRIPT_FILENAME), "w") as fh:
            json.dump(doc, fh)

    def test_transcript_serves_the_repaired_word_end_not_t_plus_dur(self):
        # The whole point of the derived field: whisper collapses word
        # timings, so `t + dur` leaves the last selected word in the video.
        # Assert the VALUE, not merely that a key exists.
        self._write_transcript([
            {"t": 1.0, "dur": 0.0, "text": "um", "conf": 0.5},
            {"t": 1.4, "dur": 0.3, "text": "so", "conf": 0.9},
        ])
        try:
            doc = self.state.transcript(self.name)
            self.assertEqual(doc["status"], "ok")
            self.assertAlmostEqual(doc["words"][0]["end"], 1.4)   # repaired
            self.assertAlmostEqual(doc["words"][1]["end"], 1.7)   # t + dur
        finally:
            os.remove(os.path.join(self.session_dir,
                                   transcribe.TRANSCRIPT_FILENAME))

    def test_transcript_serves_the_cut_limits(self):
        doc = self.state.transcript(self.name)
        self.assertEqual(doc["cut_limits"]["max_ranges"], edits.CUT_MAX_RANGES)
        self.assertAlmostEqual(doc["cut_limits"]["min_kept_sec"],
                               edits.CUT_MIN_KEPT_SEC)

    def test_an_already_invalid_document_does_not_lock_out_other_edits(self):
        # The guard is a DELTA check. A document that already violates the
        # contract (hand-written, or left by an older build) must not wedge
        # every later save behind an error about cuts the user did not make.
        import autocine.edits as ed
        bad = ed.normalize_edits(
            {"trim": {"start": 0.0, "end": 12.0},
             "cuts": [{"id": "c1", "start": 0.0, "end": 12.0}]},
            duration=12.0)
        ed.save_edits(self.session_dir, bad, duration=12.0)
        state = studio_app.StudioState(recordings_root=self.root)
        saved = state.save_session_edits(
            self.name, {"markers": [{"id": "m1", "time": 1.0, "label": "x"}]})
        self.assertEqual(len(saved["markers"]), 1)


class TypingSessionDescribe(unittest.TestCase):
    """Key events still get recorded and surfaced (key_count, key_times);
    they just no longer drive the camera in the current model.
    This class replaces the old TypingSessionRenderLayer (whose camera-
    driving assertions no longer apply)."""

    @classmethod
    def setUpClass(cls):
        if FFMPEG is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp()
        cls.session_dir = _make_session(cls.root, name="20240101-000001",
                                        dur=4.0)
        events = [{"t": 0.2, "type": "move", "x": 50, "y": 50},
                  {"t": 0.4, "type": "down", "x": 100, "y": 80}]
        t = 0.7
        while t <= 1.7 + 1e-9:
            events.append({"t": round(t, 1), "type": "key",
                           "x": 100.0, "y": 80.0})
            t += 0.1
        with open(os.path.join(cls.session_dir, "events.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        meta_path = os.path.join(cls.session_dir, "meta.json")
        with open(meta_path) as f:
            meta = json.load(f)
        meta["key_capture"] = "activity"
        with open(meta_path, "w") as f:
            json.dump(meta, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_describe_surfaces_key_fields(self):
        info = render.describe_session(self.session_dir,
                                       include_click_times=True)
        self.assertEqual(info["key_count"], 11)
        self.assertEqual(info["key_capture"], "activity")
        self.assertEqual(len(info["key_times"]), 11)
        self.assertAlmostEqual(info["key_times"][0], 0.7, places=5)
        self.assertAlmostEqual(info["key_times"][-1], 1.7, places=5)

    def test_studio_describe_passes_key_times_through_cache(self):
        state = studio_app.StudioState(recordings_root=self.root)
        full = state._describe(self.session_dir, include_click_times=True)
        self.assertEqual(len(full["key_times"]), 11)
        lite = state._describe(self.session_dir)
        self.assertNotIn("key_times", lite)


class ScrollSessionDescribe(unittest.TestCase):
    """Scroll events still record and surface; the auto-scroll-zoom
    detector is gone (the current model doesn't have one) so the old
    ScrollSessionRenderLayer/VisualAnchorRenderLayer camera assertions
    no longer apply."""

    @classmethod
    def setUpClass(cls):
        if FFMPEG is None:
            raise unittest.SkipTest("ffmpeg not available")
        cls.root = tempfile.mkdtemp()
        cls.session_dir = _make_session(cls.root, name="20240101-000004",
                                        dur=6.0)
        events = [{"t": 0.2, "type": "move", "x": 160, "y": 100}]
        t = 1.0
        while t <= 1.8 + 1e-9:
            events.append({"t": round(t, 1), "type": "scroll",
                           "x": 160.0, "y": 100.0})
            t += 0.1
        t = 2.4
        while t <= 3.0 + 1e-9:
            events.append({"t": round(t, 1), "type": "scroll",
                           "x": 160.0, "y": 100.0})
            t += 0.1
        with open(os.path.join(cls.session_dir, "events.jsonl"), "w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def test_describe_surfaces_scroll_count_and_times(self):
        info = render.describe_session(self.session_dir,
                                       include_click_times=True)
        self.assertEqual(info["scroll_count"], 16)
        self.assertEqual(len(info["scroll_times"]), 16)
        self.assertAlmostEqual(info["scroll_times"][0], 1.0, places=5)
        state = studio_app.StudioState(recordings_root=self.root)
        full = state._describe(self.session_dir, include_click_times=True)
        self.assertEqual(len(full["scroll_times"]), 16)
        lite = state._describe(self.session_dir)
        self.assertNotIn("scroll_times", lite)


class CameraParamsFromRenderOpts(unittest.TestCase):
    """Direct pin on the studio_app camera-params helper (mutation-proven
    gap: the drag_hold branch could be deleted with the suite green)."""

    def test_defaults_produce_none(self):
        self.assertIsNone(studio_app._camera_params_from_render_opts({}))

    def test_drag_hold_off_maps_through(self):
        self.assertEqual(
            studio_app._camera_params_from_render_opts({"drag_hold": False}),
            {"drag_hold": False})

    def test_overview_off_maps_through_and_on_is_omitted(self):
        # overview is default-ON in the camera, so only the OFF state deviates
        # and must reach the camera as an override; the default must stay None.
        self.assertEqual(
            studio_app._camera_params_from_render_opts({"overview": False}),
            {"overview": False})
        self.assertIsNone(
            studio_app._camera_params_from_render_opts({"overview": True}))

    def test_always_zoomed_and_drag_hold_compose(self):
        self.assertEqual(
            studio_app._camera_params_from_render_opts(
                {"always_zoomed": True, "drag_hold": False}),
            {"always_zoomed": True, "drag_hold": False})

    def test_zoom_speed_maps_through_and_normal_is_omitted(self):
        self.assertEqual(
            studio_app._camera_params_from_render_opts({"zoom_speed": "slow"}),
            {"zoom_speed": "slow"})
        self.assertIsNone(
            studio_app._camera_params_from_render_opts({"zoom_speed": "normal"}))


class _FakeChildProc(object):
    """Minimal Popen stand-in for the detached-bar / watchdog handles."""
    def __init__(self, pid=4242, alive=True):
        self.pid = pid
        self.returncode = None if alive else 0
        self.terminated = False
        self.killed = False
        self.signals = []

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        if self.returncode is None:
            self.returncode = 0
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9

    def send_signal(self, sig):
        self.signals.append(sig)


class ServerShutdownReapsWhatItSpawned(unittest.TestCase):
    """`serve()` calls StudioState.shutdown() on the way out.

    Both of these were observed outliving a Ctrl+C'd server for real: the
    native bar is spawned into its OWN session (so the terminal's SIGINT
    never reaches it) and an in-flight recording's ffmpeg is too. A window
    with no backend, and a silent screen recorder, respectively."""

    def _state(self):
        return studio_app.StudioState(recordings_root=tempfile.mkdtemp())

    def test_shutdown_terminates_the_detached_bar(self):
        st = self._state()
        bar = _FakeChildProc(pid=777)
        st._bar_process = bar
        st.shutdown()
        self.assertTrue(bar.terminated)
        self.assertIsNone(st._bar_process)

    def test_shutdown_stands_the_bar_watchdog_down(self):
        # The watchdog must be told to stand down BEFORE we close the bar
        # ourselves, or it races us to SIGKILL a window already exiting.
        st = self._state()
        st._bar_process = _FakeChildProc(pid=777)
        dog = _FakeChildProc(pid=888)
        st._bar_watchdog = dog
        st.shutdown()
        self.assertEqual(dog.signals, [signal.SIGINT])
        self.assertIsNone(st._bar_watchdog)

    def test_shutdown_is_a_noop_with_nothing_spawned(self):
        st = self._state()
        st.shutdown()   # must not raise
        self.assertIsNone(st._bar_process)

    def test_shutdown_does_not_terminate_an_already_dead_bar(self):
        st = self._state()
        bar = _FakeChildProc(pid=777, alive=False)
        st._bar_process = bar
        st.shutdown()
        self.assertFalse(bar.terminated)

    def test_shutdown_stops_an_in_flight_recording(self):
        st = self._state()
        st._record_status = {"status": "recording", "session": "s"}
        calls = []
        def fake_stop():
            calls.append(True)
            st._record_status = {"status": "done", "session": "s"}
        st.stop_record = fake_stop
        st.shutdown()
        self.assertEqual(len(calls), 1)

    def test_shutdown_leaves_an_idle_recorder_alone(self):
        st = self._state()
        st._record_status = {"status": "idle", "session": None}
        calls = []
        st.stop_record = lambda: calls.append(True)
        st.shutdown()
        self.assertEqual(calls, [])

    def test_serve_calls_shutdown_even_on_keyboard_interrupt(self):
        class _FakeHttpd(object):
            def __init__(self):
                self.state = mock.Mock()
                self.closed = False
            def serve_forever(self):
                raise KeyboardInterrupt()
            def server_close(self):
                self.closed = True
        httpd = _FakeHttpd()
        with contextlib.redirect_stdout(io.StringIO()):
            studio_app.serve(httpd)
        httpd.state.shutdown.assert_called_once()
        self.assertTrue(httpd.closed)

    def test_serve_still_closes_the_socket_if_shutdown_raises(self):
        # A failing tidy-up must never leave the port bound.
        class _FakeHttpd(object):
            def __init__(self):
                self.state = mock.Mock()
                self.state.shutdown.side_effect = RuntimeError("boom")
                self.closed = False
            def serve_forever(self):
                raise KeyboardInterrupt()
            def server_close(self):
                self.closed = True
        httpd = _FakeHttpd()
        with contextlib.redirect_stdout(io.StringIO()):
            studio_app.serve(httpd)
        self.assertTrue(httpd.closed)


if __name__ == "__main__":
    unittest.main()


class StartRecordDisplayValidation(unittest.TestCase):
    """An explicitly-picked `display` is a number the CLIENT chose, and the
    bar builds its picker once in loadDevices() and then holds those raw
    avfoundation indices for as long as its window lives. Indices renumber
    underneath it whenever a video device connects or disconnects, so a
    long-lived bar can hand back an index that now names the WEBCAM. Passing
    that straight to ffmpeg records the user's face instead of their screen.
    """

    DEVS = {"video": [(0, "FaceTime HD Camera"), (1, "Capture screen 0")],
            "audio": [(0, "MacBook Air Microphone")]}

    def setUp(self):
        self.td = tempfile.mkdtemp()
        self.state = studio_app.StudioState(recordings_root=self.td)

    def tearDown(self):
        shutil.rmtree(self.td, ignore_errors=True)

    @contextlib.contextmanager
    def _ready(self, devs=None):
        """Everything _start_record touches, stubbed -- nothing records."""
        self.state._record_status = {"status": "idle", "message": "",
                                     "session": None}
        with contextlib.ExitStack() as stack:
            p = stack.enter_context
            p(mock.patch.object(self.state, "permissions",
                                return_value={"missing_required": [],
                                              "checks": {}}))
            p(mock.patch.object(
                studio_app.dev, "list_avf_devices",
                return_value=self.DEVS if devs is None else devs))
            recorder = p(mock.patch.object(studio_app.rec, "Recorder"))
            p(mock.patch.object(studio_app.threading, "Thread"))
            yield recorder

    def _display_passed_to_recorder(self, recorder):
        # Recorder(session_dir, display, ...) -- display is 2nd positional.
        return recorder.call_args[0][1]

    def test_a_valid_pick_is_honoured(self):
        with self._ready() as recorder:
            self.state._start_record({"display": 1})
        self.assertEqual(self._display_passed_to_recorder(recorder), 1)

    def test_a_stale_pick_naming_a_camera_falls_back_to_the_screen(self):
        # THE BUG: index 0 is the webcam. Without validation this recorded
        # the user's face.
        with self._ready() as recorder:
            self.state._start_record({"display": 0})
        self.assertEqual(self._display_passed_to_recorder(recorder), 1)

    def test_an_index_that_no_longer_exists_falls_back_too(self):
        with self._ready() as recorder:
            self.state._start_record({"display": 7})
        self.assertEqual(self._display_passed_to_recorder(recorder), 1)

    def test_auto_still_detects(self):
        with self._ready() as recorder:
            self.state._start_record({"display": ""})
        self.assertEqual(self._display_passed_to_recorder(recorder), 1)

    def test_a_second_screen_pick_is_left_alone(self):
        # Validation must not collapse a legitimate multi-display choice
        # onto the first screen -- it only rejects non-screens.
        devs = {"video": [(0, "FaceTime HD Camera"),
                          (1, "Capture screen 0"),
                          (2, "Capture screen 1")], "audio": []}
        with self._ready(devs) as recorder:
            self.state._start_record({"display": 2})
        self.assertEqual(self._display_passed_to_recorder(recorder), 2)

    def test_no_screen_at_all_is_a_400_not_a_camera(self):
        devs = {"video": [(0, "FaceTime HD Camera")], "audio": []}
        with self._ready(devs) as recorder:
            with self.assertRaises(studio_app.StudioError) as ctx:
                self.state._start_record({"display": 0})
        self.assertEqual(ctx.exception.status, 400)
        recorder.assert_not_called()


class CameraRelease(unittest.TestCase):
    """Who is allowed to keep the webcam open, and what can take it back.

    The device lives in the long-running `studio.py app` server: a GET
    /api/camera/preview opens it, and `_Camera` stays up while its client
    refcount is above zero. That made ONE client's socket the sole authority
    over the camera — with `camera_preview.shutdown()` unreferenced anywhere
    in the codebase, and no bound on a client that stops reading. Observed
    consequence: the webcam ran for ~35 minutes after a finished take, with
    ~1.9 GB of JPEG streamed to an <img> the bar had hidden.
    """

    def _fake_cam(self, mgr, ordinal=0):
        """A fresh _Camera (no frame yet) with its thread machinery stubbed."""
        cam = camera_preview._Camera(ordinal)
        cam.started, cam.stopped = [], []
        cam.start = lambda: cam.started.append(1)
        cam.shutdown = lambda: cam.stopped.append(1)
        mgr.cams[ordinal] = cam
        return cam

    def _pump(self, cam, data=b"jpeg-bytes"):
        """Publish one frame the way _Camera._run does."""
        with cam.cond:
            cam.latest = data
            cam.seq += 1
            cam.cond.notify_all()

    def _read_one(self, gen, cam):
        """next(gen), with a frame arriving just after the reader subscribes.

        Readers subscribe at the camera's CURRENT seq and wait for the next
        frame, so a test can't just pre-set `latest` and expect it back —
        that state means "already delivered".
        """
        t = threading.Timer(0.05, self._pump, args=(cam,))
        t.start()
        try:
            return next(gen)
        finally:
            t.cancel()

    def test_a_reader_holds_the_device_and_closing_releases_it(self):
        mgr = camera_preview._Manager()
        cam = self._fake_cam(mgr)
        gen = mgr.frames(0)
        self.assertEqual(self._read_one(gen, cam), b"jpeg-bytes")
        self.assertEqual(cam.clients, 1)      # device pinned while reading
        self.assertEqual(len(cam.started), 1)
        gen.close()
        self.assertEqual(cam.clients, 0)
        self.assertEqual(len(cam.stopped), 1)  # last reader out turns it off

    def test_a_second_reader_keeps_it_open(self):
        # Refcount, not a boolean: one client leaving must not cut off another.
        mgr = camera_preview._Manager()
        cam = self._fake_cam(mgr)
        a, b = mgr.frames(0), mgr.frames(0)
        self._read_one(a, cam)
        self._read_one(b, cam)
        self.assertEqual(cam.clients, 2)
        a.close()
        self.assertEqual(cam.clients, 1)
        self.assertEqual(cam.stopped, [])      # still wanted
        b.close()
        self.assertEqual(cam.clients, 0)
        self.assertEqual(len(cam.stopped), 1)

    def test_suspended_manager_refuses_to_open(self):
        mgr = camera_preview._Manager()
        self._fake_cam(mgr)
        mgr.suspend()
        with self.assertRaises(RuntimeError):
            next(mgr.frames(0))

    # ---- the server-side release ------------------------------------------
    def test_server_shutdown_releases_the_camera(self):
        # Before this existed, camera_preview.shutdown() had ZERO call sites:
        # quitting the server released the device only as a side effect of the
        # process dying.
        td = tempfile.mkdtemp()
        try:
            state = studio_app.StudioState(recordings_root=td)
            with mock.patch.object(studio_app.camera_preview, "shutdown") as sd:
                state.shutdown()
            sd.assert_called_once()
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_shutdown_survives_a_camera_that_wont_release(self):
        # Best-effort: shutdown must not raise on the way out and leave the
        # listening socket open.
        td = tempfile.mkdtemp()
        try:
            state = studio_app.StudioState(recordings_root=td)
            with mock.patch.object(studio_app.camera_preview, "shutdown",
                                   side_effect=OSError("stuck")):
                state.shutdown()   # must not raise
        finally:
            shutil.rmtree(td, ignore_errors=True)

    def test_stream_bounds_a_client_that_stops_reading(self):
        # A peer that closes is caught by EPIPE on the next write, but one that
        # stays ESTABLISHED and stops draining parks the handler in sendall
        # forever -- and the refcount is dropped nowhere but that loop's exit.
        src = inspect.getsource(
            studio_app.StudioRequestHandler._stream_camera)
        self.assertIn("settimeout(CAMERA_STREAM_TIMEOUT_SEC)", src)
        self.assertIn("socket.timeout", src)
        # a timed-out write must end the stream normally (running the finally
        # that drops the refcount), not escape as an error
        self.assertLess(src.index("socket.timeout"),
                        src.index("stream.close()"))

    def test_stream_timeout_cannot_fire_on_a_healthy_client(self):
        # ~30 KB per frame over loopback at 30fps -- orders of magnitude of
        # headroom, so this only ever trips a genuinely wedged reader.
        self.assertGreaterEqual(studio_app.CAMERA_STREAM_TIMEOUT_SEC, 5.0)


class BarPreviewLifetime(unittest.TestCase):
    """Source pins for the bar's docked-preview teardown.

    There is no JS test harness here, and this invariant is not visible from
    Python — but it is the one that had the webcam running with no UI to stop
    it, so it gets a guard rather than nothing. The docked preview's <img>
    lives inside #face-idle, and hiding an element does not cancel an
    in-flight load: leaving the idle face without dropping the src keeps the
    MJPEG connection (and the camera) alive.
    """

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "bar.js")) as f:
            self.js = f.read()
        with open(os.path.join(root, "studio_web", "bar.html")) as f:
            self.html = f.read()

    def test_the_done_face_releases_the_preview(self):
        i = self.js.index('showFace("done")')
        self.assertIn("dockedPreview(false)", self.js[i:i + 400])

    def test_returning_to_idle_restores_it(self):
        i = self.js.index("ui.again.addEventListener")
        self.assertIn("dockedPreview(true)", self.js[i:i + 500])

    def test_a_take_never_drops_the_stream(self):
        # camera_preview.start_recording tees face.mov off THIS capture and
        # needs clients > 0 -- releasing on the countdown/recording faces
        # would kill the face track. The helper must bail on floating mode
        # only, and be called from neither take branch.
        i = self.js.index("function dockedPreview(")
        body = self.js[i:i + 300]
        self.assertIn('state.faceMode === "floating"', body)
        for face in ('showFace("countdown")', 'showFace("recording")'):
            j = self.js.index(face)
            self.assertNotIn("dockedPreview", self.js[j:j + 250])

    def test_the_camera_controls_really_do_live_inside_the_idle_face(self):
        # The premise of the bug: if these ever move out, the teardown above
        # is no longer needed and this pin should be revisited.
        start = self.html.index('id="face-idle"')
        end = self.html.index('id="face-countdown"')
        block = self.html[start:end]
        for ident in ("bar-cam-img", "bar-camera"):
            self.assertIn('id="{}"'.format(ident), block)

    def test_escape_still_refuses_to_kill_a_live_take(self):
        i = self.js.index('ev.key === "Escape"')
        cond = self.js[i:i + 200]
        self.assertIn('state.face !== "countdown"', cond)
        self.assertIn('state.face !== "recording"', cond)


class CameraReopen(unittest.TestCase):
    """A released camera must be reopenable.

    _Camera objects are cached in _Manager.cams and REUSED, while `seq` keeps
    counting across opens. Subscribing from a hardcoded 0 therefore made a
    reopened camera look like it had already produced a frame, so `frames`
    skipped its wait, read the `latest` that _run's finally had cleared to
    None, and reported end-of-stream at once -- a 503 "camera produced no
    frames" on every open after the first. It stayed latent for as long as the
    bar opened the preview once and held it for the life of the page; the
    moment the preview started being released after a take, every "New
    Recording" would have come back with a dead camera.
    """

    def _cam(self, mgr, ordinal=0, seq=0, latest=b"jpeg"):
        cam = camera_preview._Camera(ordinal)
        cam.start = lambda: None
        cam.shutdown = lambda: None
        cam.seq, cam.latest = seq, latest
        mgr.cams[ordinal] = cam
        return cam

    def test_reopening_a_reused_camera_still_yields_frames(self):
        mgr = camera_preview._Manager()
        # a camera that has already run and been shut down: high seq, and the
        # latest frame cleared by _run's finally
        cam = self._cam(mgr, seq=417, latest=None)

        def produce():
            with cam.cond:
                cam.latest = b"fresh"
                cam.seq += 1
                cam.cond.notify_all()

        gen = mgr.frames(0, timeout=5.0)
        t = threading.Timer(0.05, produce)
        t.start()
        try:
            self.assertEqual(next(gen), b"fresh")
        finally:
            t.cancel()
            gen.close()

    def test_a_new_reader_waits_for_a_live_frame(self):
        # Subscribing at the CURRENT seq also means a reader joining a camera
        # that is already running gets the next real frame, not whatever the
        # previous reader happened to leave in `latest`.
        mgr = camera_preview._Manager()
        cam = self._cam(mgr, seq=9, latest=b"stale")

        def produce():
            with cam.cond:
                cam.latest = b"live"
                cam.seq += 1
                cam.cond.notify_all()

        gen = mgr.frames(0, timeout=5.0)
        t = threading.Timer(0.05, produce)
        t.start()
        try:
            self.assertEqual(next(gen), b"live")
        finally:
            t.cancel()
            gen.close()
        self.assertEqual(cam.clients, 0)

    def test_shutdown_wakes_readers_instead_of_stranding_them(self):
        # _run's finally sets latest=None to signal EOF; without bumping seq
        # the reader re-checked `seq == seen`, saw no change, and blocked to
        # its full timeout rather than returning.
        mgr = camera_preview._Manager()
        cam = self._cam(mgr, seq=3, latest=b"x")
        gen = mgr.frames(0, timeout=30.0)   # long: a hang would blow the test

        def end_of_stream():
            with cam.cond:
                cam.latest = None
                cam.seq += 1
                cam.cond.notify_all()

        t = threading.Timer(0.05, end_of_stream)
        t.start()
        started = time.time()
        try:
            with self.assertRaises(StopIteration):
                next(gen)                    # clean EOF, promptly
        finally:
            t.cancel()
            gen.close()
        self.assertLess(time.time() - started, 5.0)

    def test_run_bumps_seq_when_it_clears_latest(self):
        # The producer half of the contract above.
        src = inspect.getsource(camera_preview._Camera._run)
        tail = src[src.index("finally:"):]
        self.assertIn("self.latest = None", tail)
        self.assertIn("self.seq += 1", tail)


class CameraAuthoritativeRelease(unittest.TestCase):
    """The webcam is released by the SERVER, because the client cannot.

    bar.js dropped the <img>'s src and the code assumed that ended the
    request. It does not in WKWebView: a multipart/x-mixed-replace load keeps
    running, so the socket, the client refcount and the camera all outlive
    the UI that wanted them. Measured on the native bar -- one preview
    connection survived a recording, the done face, AND an explicit
    "No camera" pick, streaming 1.6 GB with the webcam lit the whole time.
    Since frames() only decrements when that loop ends, no client action
    could release the device at all.
    """

    def _cam(self, mgr, ordinal=0, sink=None):
        cam = camera_preview._Camera(ordinal)
        cam.stopped = []
        cam.shutdown = lambda: cam.stopped.append(1)
        cam.sink = sink
        mgr.cams[ordinal] = cam
        return cam

    def test_release_shuts_a_preview_camera_down(self):
        mgr = camera_preview._Manager()
        cam = self._cam(mgr)
        self.assertEqual(mgr.release(), 1)
        self.assertEqual(len(cam.stopped), 1)

    def test_release_spares_a_camera_feeding_a_take(self):
        # That sink is a take's face.mov being teed off this very capture --
        # releasing it would silently truncate the face track.
        mgr = camera_preview._Manager()
        cam = self._cam(mgr, sink=object())
        self.assertEqual(mgr.release(), 0)
        self.assertEqual(cam.stopped, [])

    def test_release_keeps_the_camera_object_for_reuse(self):
        # Unlike shutdown(), which empties self.cams: the next frames() must
        # reuse this _Camera, which is what the seq/latest contract assumes.
        mgr = camera_preview._Manager()
        self._cam(mgr)
        mgr.release()
        self.assertIn(0, mgr.cams)

    def test_release_is_exported_at_module_level(self):
        self.assertTrue(callable(camera_preview.release))

    def test_endpoint_releases_and_reports_the_count(self):
        td = tempfile.mkdtemp()
        try:
            server, base = studio_app.build_server(
                "127.0.0.1", 0, recordings_root=td)
            t = threading.Thread(target=server.serve_forever, daemon=True)
            t.start()
            try:
                with mock.patch.object(studio_app.camera_preview, "release",
                                       return_value=2) as rel:
                    req = urllib.request.Request(
                        base + "/api/camera/release", data=b"{}",
                        headers={
                            "Content-Type": "application/json",
                            studio_app._TOKEN_HEADER: server.security_token,
                        },
                        method="POST")
                    body = json.loads(
                        urllib.request.urlopen(req).read().decode())
                rel.assert_called_once()
                self.assertEqual(body, {"released": 2})
            finally:
                server.shutdown()
                server.server_close()
        finally:
            shutil.rmtree(td, ignore_errors=True)


class BarCameraRelease(unittest.TestCase):
    """Source pins for the bar half of the server-authoritative release."""

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "bar.js")) as f:
            self.js = f.read()

    def test_releasecam_asks_the_server(self):
        i = self.js.index("function releaseCam(")
        self.assertIn("/api/camera/release", self.js[i:i + 400])

    def test_every_off_path_releases_rather_than_only_stopping(self):
        for anchor in ("function dockedPreview(",
                       'an explicit "No camera" IS the intent',
                       "function closeBar(",
                       "async function floatFacecam("):
            i = self.js.index(anchor)
            self.assertIn("releaseCam()", self.js[i:i + 600], anchor)

    def test_startcam_releases_before_it_opens_and_awaits_it(self):
        # Orphaned readers from earlier opens have to go BEFORE a new one
        # starts, and the release must be awaited or it races the open and
        # tears down the stream it just started.
        i = self.js.index("async function startCam(")
        seg = self.js[i:i + 1600]
        self.assertLess(seg.index('await api("/api/camera/release"'),
                        seg.index("ui.camImg.src ="))

    def test_stopcam_is_never_mistaken_for_a_release(self):
        # It tears down the picture only. If it ever calls the release
        # endpoint, startCam's stopCam() would race its own open.
        i = self.js.index("function stopCam(")
        seg = self.js[i:self.js.index("\n  }", i)]   # its body only
        self.assertNotIn("/api/camera/release", seg)
        self.assertIn("onerror = null", seg)   # no spurious "unavailable"


class MultiNativeLivePreview(unittest.TestCase):
    """An occlusion-free multi-window take has N per-window buffers and NO
    raw.mov, so the editor's single `<video>` had nothing to point at and the
    player sat on a silent 404. These pin the three seams that let it play:
    a per-channel media kind, the composite layout payload, and the client
    actually asking for channels rather than raw."""

    JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "studio_web", "editor.js")

    @classmethod
    def setUpClass(cls):
        with open(cls.JS) as f:
            cls.js = f.read()

    def test_channel_media_kind_is_served(self):
        from autocine import studio_app
        src = inspect.getsource(studio_app.StudioState.media_path)
        self.assertIn('kind.startswith("channel")', src)
        # Bounds-checked against the manifest rather than trusting the index.
        self.assertIn("capture_channels", src)

    def test_layout_exposes_channels_cells_and_offsets(self):
        from autocine import render
        src = inspect.getsource(render.multi_native_layout)
        for token in ('"channels"', '"cells"', '"canvas"', '"duration"',
                      '"media"', '"start"'):
            self.assertIn(token, src, token)

    def test_layout_start_offsets_mirror_the_export(self):
        # The export skips each channel to the shared origin (max offset);
        # the preview has to use the same skip or the cards sit apart.
        from autocine import render
        src = inspect.getsource(render.multi_native_layout)
        self.assertIn("origin - offsets[i]", src)
        self.assertIn("min(remaining)", src)   # shared overlap, not the longest

    def test_editor_requests_channel0_not_raw_when_multi_native(self):
        # Window widened past the scene-take branch that now sits between the
        # isNative gate and the src assignment (docs/architecture.md, S1 editor
        # posture: scene takes skip the src entirely and show a message).
        i = self.js.index("ui.video.src = autocineUrl(")
        seg = self.js[i - 1300:i + 300]
        self.assertIn("multi_native", seg)
        self.assertIn('"/channel0"', seg)
        self.assertIn('"/raw"', seg)          # unchanged for every other take

    def test_every_channel_gets_its_own_absolute_start(self):
        # This USED to place followers relative to channel 0 (`- base`),
        # because the playhead was channel 0's FILE time and adding the
        # master's own `start` would have double-counted it.
        #
        # It is OUTPUT time now (`fleetLead`), so the master's skip is applied
        # exactly once, at the element boundary, and every channel -- master
        # included -- is its own absolute `start` past the playhead. Cancelling
        # it here is what left the picture on the session clock while the
        # overlays moved to the composite one. See
        # FlatFleetPlayheadIsOutputTime.
        i = self.js.index("function syncChannels(")
        seg = self.js[i:self.js.index("\n  }", i)]
        self.assertIn("toNum(p.channels[i].start, 0) + t", seg)
        self.assertNotIn("- base", seg)

    def test_each_cell_samples_its_own_channel(self):
        # The whole point: one <video> per card, not N crops of one recording.
        i = self.js.index("function cellSource(")
        seg = self.js[i:self.js.index("\n  }", i)]
        self.assertIn("multi_native", seg)
        self.assertIn("S.chanVideos", seg)
        self.assertIn("readyState", seg)      # skip a follower with no frame

    def test_source_space_tools_are_blocked(self):
        # crop / zoom-pin / window-pick measure against a source that does not
        # exist here; the MCP surface already refuses them. These stay behind
        # spatialToolsBlocked() even after card PLACEMENT relaxes (that split
        # off into placementBlocked()) -- source-space tools are the ones with
        # no per-card mapping in native/scene modes.
        i = self.js.index("function spatialToolsBlocked(")
        self.assertIn("multi_native", self.js[i:i + 600])
        for fn in ("function armPin(", "function armWinPick(", "function armCrop("):
            j = self.js.index(fn)
            self.assertIn("spatialToolsBlocked()", self.js[j:j + 400], fn)


class FlatFleetPlayheadIsOutputTime(unittest.TestCase):
    """The editor's flat-fleet playhead is OUTPUT time, not channel-0 file time.

    `_render_multi_native` grab()-skips every channel to the shared origin, so
    the composite's frame 0 is channel 0's file frame `channels[0].start`. The
    element the editor plays is raw_0.mov served verbatim, while `duration`,
    the ruler's click ticks and every per-card zoom/focus track are on the
    COMPOSITE clock. Reading one as the other put the entire overlay layer
    `start[0]` ahead of the picture -- which is exactly what happened when the
    server-side anchor moved and this side did not follow.

    Source pins, the same discipline the other editor-gesture classes use: the
    behavior was verified live in a browser (playhead 4.0 -> channel 0 at 5.0
    on a fixture whose `start[0]` is 1.0s; 4.0 before the fix), and these keep
    it from being simplified back.
    """

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "editor.js")) as f:
            self.js = f.read()

    def test_the_lead_comes_from_channel_zeros_start(self):
        i = self.js.index("function fleetLead")
        seg = self.js[i:i + 400]
        self.assertIn("p.channels[0].start", seg)
        # Zero for every other take shape, or this moves takes it must not.
        self.assertIn("sceneLive()", seg)
        self.assertIn("p.multi_native", seg)

    def test_seek_adds_the_lead_going_to_the_element(self):
        i = self.js.index("const wantMedia = fleetLead()")
        seg = self.js[i:i + 300]
        self.assertIn("fleetLead() + S.playhead", seg)
        self.assertIn("ui.video.currentTime = wantMedia", seg)

    def test_playback_subtracts_it_coming_back(self):
        i = self.js.index("function playbackTick")
        seg = self.js[i:i + 400]
        self.assertIn("ui.video.currentTime - fleetLead()", seg)

    def test_followers_are_no_longer_relative_to_the_master(self):
        # The bug: `base = start[0]` made every follower relative to channel 0
        # and cancelled the master's own skip, so the picture was internally
        # consistent at SESSION time while the overlays were at COMPOSITE
        # time. Each channel is its own `start` past the OUTPUT time, master
        # included -- which is what `seekFleetTo` already does for scenes.
        i = self.js.index("function syncChannels")
        seg = self.js[i:i + 900]
        self.assertIn("toNum(p.channels[i].start, 0) + t", seg)
        self.assertNotIn("const base =", seg)

    def test_the_scene_player_uses_the_same_rule(self):
        # The template this was made to match: `start[i] + tLocal` for ALL i.
        i = self.js.index("function seekFleetTo")
        seg = self.js[i:i + 700]
        self.assertIn("toNum(entry.channels[i].start, 0) + tLocal", seg)


class CardResizeHandles(unittest.TestCase):
    """The corner-resize gesture on multi-window cards (editor.js). Source
    pins, same discipline as the other editor-gesture classes: the browser
    behavior itself was verified live; these keep its load-bearing choices
    from being silently simplified away."""

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "editor.js")) as f:
            self.js = f.read()

    def test_resize_is_aspect_locked(self):
        # The painter STRETCHES each crop over its placement rect (framing:
        # auto-layouts always place at the crop's own aspect), so a freeform
        # w/h would distort the recording in the export. The gesture must
        # derive h from w through the card's aspect, never track both axes.
        i = self.js.index('d.mode === "resize"')
        seg = self.js[i:i + 1600]
        self.assertIn("const h = w / d.arf", seg)
        self.assertNotIn("d.h = Math.abs", seg)

    def test_resize_scales_about_the_opposite_corner(self):
        i = self.js.index("function armCardResize")
        seg = self.js[i:i + 900]
        # The anchor is the OPPOSITE corner of the grabbed one.
        self.assertIn('grab.corner === "nw" || grab.corner === "sw"', seg)
        self.assertIn('grab.corner === "nw" || grab.corner === "ne"', seg)

    def test_min_size_and_canvas_clamp(self):
        i = self.js.index('d.mode === "resize"')
        seg = self.js[i:i + 1600]
        self.assertIn("CARD_MIN_FRAC", seg)
        self.assertIn("availX", seg)
        self.assertIn("availY", seg)

    def test_corner_grab_wins_over_body_grab(self):
        i = self.js.index("function onCardDown")
        seg = self.js[i:i + 700]
        self.assertLess(seg.index("cardCornerAt"), seg.index("cardAtPoint"))

    def test_corners_only_where_the_write_lands(self):
        # Corners appear only where the placement write lands: gated on
        # placementBlocked() (allows display-crop + multi-native, refuses scene
        # takes P2 / plain / a display-crop with no windows). A display-crop
        # card still needs its windows[i]; a multi-native card is always
        # placeable (native short-circuit).
        i = self.js.index("function cardCornerAt")
        seg = self.js[i:i + 800]
        self.assertIn("placementBlocked()", seg)
        self.assertIn("!native && !wins", seg)

    def test_grab_zone_maps_client_px_through_canvasFrac(self):
        # The tolerance must track the composite scale AND the focus camera;
        # a raw fraction constant would make corners untouchable zoomed in.
        i = self.js.index("function cardCornerAt")
        seg = self.js[i:i + 1200]
        self.assertIn("CORNER_GRAB_PX", seg)
        self.assertEqual(seg.count("canvasFrac"), 2)

    def test_hover_redraws_only_on_change(self):
        i = self.js.index("function onCardMove")
        seg = self.js[i:self.js.index("const p = S.camPath", i)]
        self.assertIn("key !== prev", seg)

    def test_release_writes_the_resized_fractions(self):
        # onCardUp persists d.w/d.h -- the resize branch must keep them
        # current so the same save path serves both gestures.
        i = self.js.index('d.mode === "resize"')
        seg = self.js[i:i + 1600]
        self.assertIn("d.w = w", seg)
        self.assertIn("d.h = h", seg)


@unittest.skipUnless(FFMPEG, "ffmpeg not available")
class ScenePreviewEndpoint(unittest.TestCase):
    """The `/api/preview` -> State.preview() dispatch for a SCENE take. Before
    scene_preview_frame, a scrub on a scene take was a guaranteed 500 (the
    single-file preview_frame opening a raw.mov that does not exist); the
    editor showed a fatal "no live preview" message. This pins that the
    endpoint now returns a real composited JPEG, and that the render-kwargs
    mapping stays wired (a renamed key would 500 again)."""

    @classmethod
    def setUpClass(cls):
        from tests.test_render_scenes import _mk_scene_session
        cls.root = tempfile.mkdtemp()
        cls.name = "20240102-000000"
        _mk_scene_session(os.path.join(cls.root, cls.name),
                          events=[{"t": 100.5, "type": "down", "x": 100.0,
                                   "y": 100.0, "button": "Button.left"}])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _state(self):
        return studio_app.StudioState(recordings_root=self.root)

    def test_describe_flags_it_as_a_scene_take(self):
        info = self._state()._describe(os.path.join(self.root, self.name))
        self.assertTrue(info.get("scene_take"))

    def test_preview_returns_a_jpeg_across_the_seam(self):
        st = self._state()
        for t in (0.3, 1.5):     # scene 0 (1 card), scene 1 (2 cards)
            data = st.preview(self.name, t, {})
            self.assertGreater(data["width"], 0)
            self.assertGreater(data["height"], 0)
            self.assertTrue(data["image_jpeg_base64"])
            # decodes to real JPEG bytes
            raw = base64.b64decode(data["image_jpeg_base64"])
            self.assertEqual(raw[:2], b"\xff\xd8")   # JPEG SOI

    def test_preview_clamps_to_the_full_joined_timeline(self):
        # Trim is ignored on the scene export, so the preview reports the full
        # duration as its range rather than a trim that would never land.
        st = self._state()
        data = st.preview(self.name, 999.0, {})
        self.assertEqual(data["trim"]["start"], 0.0)
        self.assertGreater(data["trim"]["end"], 1.0)
        self.assertLessEqual(data["time"], data["trim"]["end"] + 1e-6)

    def test_resolution_option_shrinks_the_preview(self):
        # The export-resolution control reaches the scene preview: a height cap
        # comes back as a smaller composited frame, aspect-locked.
        st = self._state()
        native = st.preview(self.name, 0.3, {})
        capped = st.preview(self.name, 0.3, {"resolution": "720"})
        self.assertLess(capped["height"], native["height"])
        self.assertEqual(capped["height"], 720)
        self.assertAlmostEqual(
            capped["width"] / float(capped["height"]),
            native["width"] / float(native["height"]), delta=0.03)


class ManualCardLayoutJS(unittest.TestCase):
    """editor.js wiring for hand-placed cards on a multi-native take
    (docs/architecture.md P1). Source pins for the gate split, the save
    routing, and the payload keys that persist / preview the placement."""

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "editor.js")) as f:
            self.js = f.read()

    def test_placement_gate_is_split_from_source_tools(self):
        i = self.js.index("function placementBlocked(")
        seg = self.js[i:i + 340]
        self.assertIn("d.multi_native) return false", seg)   # native placeable
        # scene takes become placeable once the live plan loads (P2)
        self.assertIn("d.scene_take) return !sceneLive()", seg)
        # source-space tools keep their own (unchanged) gate
        self.assertIn("function spatialToolsBlocked(", self.js)

    def test_card_save_routes_to_channel_layouts_with_null_padding(self):
        i = self.js.index("function onCardUp(")
        seg = self.js[i:i + 1300]
        self.assertIn("S.details.multi_native", seg)
        self.assertIn("e.channel_layouts", seg)
        # null-padding, never a compacting push (positional binding)
        self.assertIn("e.channel_layouts.push(null)", seg)
        self.assertIn("e.channel_layouts[d.index] = layout", seg)

    def test_reset_is_mode_aware(self):
        i = self.js.index("function resetCardPlacement(")
        seg = self.js[i:i + 600]
        self.assertIn("e.channel_layouts = []", seg)
        self.assertIn("e.scene_layouts[String(S.activeScene)] = []", seg)

    def test_scene_placement_gate_and_readiness(self):
        # P2: scene takes become placeable once the live plan loads; card-drag
        # arms on sceneLive() (multiReady() is false for scenes).
        i = self.js.index("function placementBlocked(")
        self.assertIn("d.scene_take) return !sceneLive()", self.js[i:i + 320])
        i = self.js.index("function onCardDown(")
        self.assertIn("!multiReady() && !sceneLive()", self.js[i:i + 400])

    def test_scene_save_routes_to_scene_layouts_by_active_scene(self):
        i = self.js.index("function onCardUp(")
        seg = self.js[i:i + 2200]
        self.assertIn("S.details.scene_take", seg)
        self.assertIn("String(S.activeScene)", seg)
        self.assertIn("e.scene_layouts[s].push(null)", seg)   # null-padding
        self.assertIn("e.scene_layouts[s][d.index] = layout", seg)

    def test_hit_test_and_draw_use_in_scene_time(self):
        # The per-scene focus/zoom tracks are sampled from the scene's frame 0,
        # so the hit-test + draw must index by cellTime() (in-scene), not the
        # take-level playhead, or they land on the wrong scene's frame.
        self.assertIn("function cellTime(", self.js)
        i = self.js.index("function cellTime(")
        self.assertIn("S.playhead - S.seamTimes[S.activeScene]",
                      self.js[i:i + 260])
        # drawMulti defaults its time to cellTime()
        i = self.js.index("function drawMulti(")
        self.assertIn("t = cellTime()", self.js[i:i + 120])

    def test_payload_and_options_carry_channel_layouts(self):
        # editsPayload persists it; fullOptions previews it live; cellsSignature
        # forces a re-fetch when it changes.
        i = self.js.index("function editsPayload(")
        seg = self.js[i:i + 1200]
        self.assertIn("channel_layouts: S.edits.channel_layouts", seg)
        self.assertIn("scene_layouts: S.edits.scene_layouts", seg)
        i = self.js.index("function fullOptions(")
        seg = self.js[i:i + 2000]
        self.assertIn("channel_layouts: S.edits.channel_layouts", seg)
        self.assertIn("scene_layouts: S.edits.scene_layouts", seg)
        i = self.js.index("function cellsSignature(")
        seg = self.js[i:i + 560]
        self.assertIn("e.channel_layouts", seg)
        self.assertIn("e.scene_layouts", seg)


class CardSelectionAndEdgeHandles(unittest.TestCase):
    """Click-to-select cards + the design-tool selection frame + edge-handle
    resize (the feature that made the drag stack discoverable). Source pins,
    same discipline as CardResizeHandles: the interaction was verified live on
    real multi-native / display-crop / scene takes; these keep its load-bearing
    choices from being silently simplified away."""

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "editor.js")) as f:
            self.js = f.read()

    def test_selection_state_drives_the_chrome_not_hover_alone(self):
        # A click SELECTS (persistent), and the frame is drawn for the selected
        # card -- not only while the pointer rests on it, as the old corner-dot
        # hover did. drawMulti must key the full frame off S.cardSel.
        self.assertIn("cardSel:", self.js)          # state field exists
        i = self.js.index("function drawMulti(")
        seg = self.js[i:self.js.index("function canvasFrac", i)]
        self.assertIn("S.cardSel === i", seg)
        self.assertIn("drawCardChrome(", seg)

    def test_select_and_deselect_are_wired_into_onCardDown(self):
        i = self.js.index("function onCardDown(")
        seg = self.js[i:i + 900]
        # grab or body selects; empty stage deselects (both reachable).
        self.assertIn("selectCard(", seg)
        self.assertIn("deselectCard()", seg)
        for fn in ("selectCard", "deselectCard"):
            self.assertIn("function " + fn + "(", self.js)

    def test_selecting_drops_the_still_so_the_frame_shows(self):
        # The server still is opaque and ABOVE the composite (windows / scene
        # takes), so a selected card's frame is only visible if the still is
        # suppressed while it is up. requestStill must decline on the selection
        # at BOTH its guards (before and after the async fetch).
        i = self.js.index("async function requestStill(")
        seg = self.js[i:i + 1400]
        self.assertEqual(seg.count("selHoldsStill()"), 2)
        # selectCard reveals the live composite by taking the still down.
        i = self.js.index("function selectCard(")
        self.assertIn("staleStill()", self.js[i:i + 200])

    def test_selection_keeps_the_scene_cold_fleet_still_floor(self):
        # A scene take's still is ALSO the cold-fleet correctness floor (drawMulti
        # paints only the backdrop until the fleet is frame-ready), so the
        # selection suppression must NOT strip it while the fleet is cold --
        # else selecting a card over a warming scene shows an empty canvas.
        i = self.js.index("function selHoldsStill(")
        seg = self.js[i:i + 200]
        self.assertIn("!sceneLive() || sceneFleetReady()", seg)
        self.assertIn("function sceneFleetReady(", self.js)

    def test_pointerleave_clears_hover_on_scene_takes_too(self):
        # multiReady() is false for a scene take, so gating the leave-redraw on
        # it alone strands the faint hover outline; sceneLive() must also arm it.
        i = self.js.index('addEventListener("pointerleave"')
        seg = self.js[i:i + 600]
        self.assertIn("multiReady() || sceneLive()", seg)

    def test_cardCornerAt_returns_edge_handles(self):
        i = self.js.index("function cardCornerAt")
        seg = self.js[i:self.js.index("function ", i + 10)]
        for edge in ('corner: "n"', 'corner: "s"', 'corner: "e"', 'corner: "w"'):
            self.assertIn(edge, seg)

    def test_edge_resize_is_aspect_locked_and_gated_on_d_edge(self):
        # An edge drag is a SEPARATE branch (d.edge) from the corner one, and
        # is just as aspect-locked -- the perpendicular axis is derived through
        # the aspect, never tracked freely, so the recording never distorts.
        i = self.js.index('d.mode === "resize"')
        seg = self.js[i:i + 2600]
        self.assertIn("if (!d.edge) {", seg)     # corner branch guarded
        self.assertIn("h = w / d.arf", seg)      # e/w edge derives height
        self.assertIn("w = h * d.arf", seg)      # n/s edge derives width
        # never a free per-axis edge stretch
        self.assertNotIn("c.h = Math.abs", seg)

    def test_escape_deselects_a_selected_card(self):
        i = self.js.index('case "Escape":')
        seg = self.js[i:i + 400]
        self.assertIn("S.cardSel != null) deselectCard()", seg)

    def test_windows_panel_is_native_aware(self):
        # Native / scene takes get their own copy + a mode-aware Reset, and the
        # source-space "Draw a window" picker is hidden (it is refused there).
        self.assertIn("const nativeCards =", self.js)
        i = self.js.index("const nativeCards =")
        seg = self.js[i:i + 1800]
        self.assertIn("Click a window on the preview to select it", seg)
        self.assertIn("resetCardPlacement", seg)
        self.assertIn("!nativeCards && windows.length < 4", self.js)

    def test_card_handle_overlay_rides_over_the_still(self):
        # A scene take shows the opaque server still while its fleet warms (and
        # if the browser can't decode the channel .movs, drawMulti early-returns
        # to the backdrop and never reaches the chrome) -- so the resize handles
        # were invisible there. drawCardOverlay draws them on an always-on-top,
        # click-through canvas whenever the Windows panel is open and the live
        # composite is NOT up. Keep the overlay + its wiring from being dropped.
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "editor.html")) as f:
            html = f.read()
        with open(os.path.join(root, "studio_web", "editor.css")) as f:
            css = f.read()
        self.assertIn('id="ed-cardchrome"', html)
        self.assertIn(".ed-cardchrome", css)
        self.assertIn("function drawCardOverlay(", self.js)
        self.assertIn('cardChrome: $("ed-cardchrome")', self.js)
        i = self.js.index("function drawCardOverlay(")
        seg = self.js[i:i + 2200]
        # Only while ARRANGING (Windows panel), PAUSED, and the live composite
        # is down. `!S.playing` is load-bearing: a scene take's brief fleet
        # re-buffer blanks the composite mid-play, and outlines painted over
        # that read as "the windows vanished".
        self.assertIn('S.panel === "windows"', seg)
        self.assertIn("!S.playing", seg)
        self.assertIn("multiReady() || (sceneLive() && sceneFleetReady())", seg)
        self.assertIn("drawCardChrome(", seg)
        # Refreshed from the composite draw, the still fetch, and panel switches,
        # so the handles track the current frame whether or not the fleet warms.
        dm = self.js.index("function drawMulti(")
        self.assertIn("drawCardOverlay()",
                      self.js[dm:self.js.index("function canvasFrac", dm)])
        rs = self.js.index("async function requestStill(")
        self.assertIn("drawCardOverlay()", self.js[rs:rs + 1500])

    def test_remove_and_restore_card_are_wired(self):
        # Remove-a-card (reversible hide): the editor writes the selected card's
        # FILE into edits.hidden_channels, clears manual placements (a hide
        # restructures the set), and re-fetches. Restore clears the key. The
        # key rides the live options + the dirty signature so a hide/restore
        # re-fetches the camera path (the card SET changed).
        for fn in ("function removeSelectedCard(", "function restoreHiddenCards(",
                   "function anchorFile("):
            self.assertIn(fn, self.js)
        rem = self.js[self.js.index("function removeSelectedCard("):][:900]
        self.assertIn("hidden_channels", rem)
        self.assertIn("e.channel_layouts = []", rem)   # stale placements dropped
        self.assertIn("e.scene_layouts = {}", rem)
        self.assertIn("anchorFile()", rem)             # anchor guard
        # Threaded so a change re-fetches the camera path and persists on save.
        for site in ("function fullOptions(", "function editsPayload(",
                     "function cellsSignature("):
            seg = self.js[self.js.index(site):][:2400]
            self.assertIn("hidden_channels", seg)
        # Panel offers Remove (scene) + Restore, wired to the handlers.
        self.assertIn("removeSelectedCard", self.js[self.js.index(
            "const nativeCards ="):][:2400])
        self.assertIn("restoreHiddenCards", self.js[self.js.index(
            "const nativeCards ="):][:2400])


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class MultiNativeManualLayoutEndpoint(unittest.TestCase):
    """`/api/camera-path` honors channel_layouts for a multi-native take: the
    override reaches the composite cells the editor draws, and the off-switch
    (absent) is unchanged."""

    @classmethod
    def setUpClass(cls):
        from tests.test_capture_window import _mk_multi_native_session
        cls.root = tempfile.mkdtemp()
        cls.name = "20240103-000001"
        _mk_multi_native_session(
            os.path.join(cls.root, cls.name),
            [{"file": "raw_0.mov", "width": 320, "height": 180,
              "rect": [0, 0, 320, 180], "t0_monotonic": 100.0},
             {"file": "raw_1.mov", "width": 240, "height": 180,
              "rect": [400, 0, 240, 180], "t0_monotonic": 100.0}])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _state(self):
        return studio_app.StudioState(recordings_root=self.root)

    def test_override_moves_a_cell_but_not_the_canvas(self):
        st = self._state()
        base = st.camera_path(self.name, {"stride": 2})
        moved = st.camera_path(self.name, {
            "stride": 2,
            "channel_layouts": [None, {"x": 0.05, "y": 0.05,
                                       "w": 0.30, "h": 0.30}]})
        self.assertTrue(base.get("multi_native"))
        # canvas unchanged (freeze rule), but card 1's cell moved
        self.assertEqual(base["canvas"], moved["canvas"])
        self.assertNotEqual(base["cells"][1], moved["cells"][1])
        # card 0 (null slot) is untouched
        self.assertEqual(base["cells"][0], moved["cells"][0])

    def test_absent_layouts_is_the_baseline(self):
        st = self._state()
        a = st.camera_path(self.name, {"stride": 2})
        b = st.camera_path(self.name, {"stride": 2, "channel_layouts": []})
        self.assertEqual(a["cells"], b["cells"])


@unittest.skipUnless(FFMPEG, "needs ffmpeg")
class SceneCameraPathEndpoint(unittest.TestCase):
    """`/api/camera-path` -> State.camera_path for a SCENE take. The stub that
    said "export to view" is replaced by the per-scene composite plan the S3
    live player consumes (docs/architecture.md). Pins the payload shape, the
    degrade-to-stub floor, and the off-switch: single-raw and multi-native
    payloads carry NO scene keys."""

    @classmethod
    def setUpClass(cls):
        from tests.test_render_scenes import _mk_scene_session
        from tests.test_capture_window import _mk_multi_native_session
        cls.root = tempfile.mkdtemp()
        cls.scene = "20240102-000001"
        cls.plain = "20240102-000002"
        cls.native = "20240102-000003"
        _mk_scene_session(os.path.join(cls.root, cls.scene),
                          events=[{"t": 100.5, "type": "down", "x": 100.0,
                                   "y": 100.0, "button": "Button.left"}])
        _make_session(cls.root, name=cls.plain)
        _mk_multi_native_session(
            os.path.join(cls.root, cls.native),
            [{"file": "raw_0.mov", "width": 320, "height": 180,
              "rect": [0, 0, 320, 180], "t0_monotonic": 100.0},
             {"file": "raw_1.mov", "width": 240, "height": 180,
              "rect": [400, 0, 240, 180], "t0_monotonic": 100.0}])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)

    def _state(self):
        return studio_app.StudioState(recordings_root=self.root)

    def test_scene_take_returns_the_per_scene_plan(self):
        data = self._state().camera_path(self.scene, {"stride": 2})
        self.assertTrue(data.get("scene_take"))
        self.assertTrue(data.get("windows_mode"))
        self.assertEqual(data["seams"], [0, 30])
        self.assertEqual(data["total_frames"], 57)
        self.assertEqual([s["index"] for s in data["scenes"]], [0, 1])
        self.assertEqual([m["media"] for m in data["scenes"][1]["channels"]],
                         ["scene1channel0", "scene1channel1"])
        # canvas/cells nested under scenes[], never top-level -- so the
        # single-scene multiReady() draw gate is never armed for a scene take.
        self.assertNotIn("cells", data)
        self.assertNotIn("canvas", data)
        self.assertIn("canvas", data["scenes"][0])
        # trim still comes back (full joined timeline; trim is export-ignored).
        self.assertEqual(data["trim"]["start"], 0.0)
        self.assertGreater(data["trim"]["end"], 1.0)

    def test_degrades_to_the_still_only_stub_on_emitter_failure(self):
        # A preview nicety must never 500: if the per-scene emitter raises, the
        # editor still gets its still-only stub and falls back to
        # scene_preview_frame (the correctness floor).
        with mock.patch.object(studio_app.ren, "scene_camera_path",
                               side_effect=RuntimeError("boom")):
            data = self._state().camera_path(self.scene, {"stride": 2})
        self.assertTrue(data.get("scene_take"))
        self.assertEqual(data.get("path"), [])
        self.assertNotIn("scenes", data)
        self.assertIn("trim", data)

    def test_off_switch_single_raw_and_multi_native_have_no_scene_keys(self):
        # The scene branch is the ONLY thing that changed; neither the plain
        # nor the multi-native camera-path payload gains scene keys.
        st = self._state()
        plain = st.camera_path(self.plain, {"stride": 2})
        self.assertNotIn("scene_take", plain)
        self.assertNotIn("scenes", plain)
        self.assertNotIn("seams", plain)
        native = st.camera_path(self.native, {"stride": 2})
        self.assertTrue(native.get("multi_native"))
        self.assertNotIn("scenes", native)
        self.assertNotIn("seams", native)

    def test_scene_layouts_moves_a_cell_in_that_scene_only(self):
        # P2: /api/camera-path honors a per-scene override -- scene 1's card 1
        # moves, scene 0 and the canvas are untouched (freeze + isolation).
        st = self._state()
        base = st.camera_path(self.scene, {"stride": 2})
        moved = st.camera_path(self.scene, {
            "stride": 2,
            "scene_layouts": {"1": [None, {"x": 0.5, "y": 0.5,
                                           "w": 0.3, "h": 0.3}]}})
        b0, b1 = base["scenes"][0], base["scenes"][1]
        m0, m1 = moved["scenes"][0], moved["scenes"][1]
        self.assertEqual(b0["cells"], m0["cells"])            # scene 0 untouched
        self.assertEqual(b1["canvas"], m1["canvas"])          # canvas frozen
        self.assertNotEqual(b1["cells"][1], m1["cells"][1])   # scene 1 card 1
        self.assertEqual(b1["cells"][0], m1["cells"][0])      # card 0 (null slot)


class SceneStillPreviewJS(unittest.TestCase):
    """editor.js wiring for the scene-take paused-scrub preview. Source pins:
    the browser behavior was verified live; these keep the load-bearing
    choices (still-only mode, not the old fatal message; scene takes fetch
    /api/preview; Play is blocked with a hint) from being simplified away."""

    def setUp(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "studio_web", "editor.js")) as f:
            self.js = f.read()

    def test_scene_take_is_still_only_not_fatal(self):
        # The isScene branch must set the still-only flag, NOT showFatal the
        # old "live preview isn't built" message.
        i = self.js.index("const isScene =")
        seg = self.js[i:i + 900]
        self.assertIn("S.sceneStillOnly = true", seg)
        self.assertNotIn("Live preview isn't built", seg)

    def test_entrance_cells_carry_a_corner_radius(self):
        # The live-morph cells feed roundRectPath's CLIP via `c.radius`, and
        # entranceCells builds fresh cells from the server's `from_cells` (which
        # carry no radius). Without setting it, `c.radius` is undefined -> the
        # clip radius is NaN -> the clip region is degenerate -> every card's
        # content is clipped to nothing, so the windows go BLANK for the whole
        # join-entrance morph in LIVE playback (export + paused still were fine).
        # Must match framing.paint_at's per-card radius: max(8, 0.02*width).
        i = self.js.index("function entranceCells(")
        seg = self.js[i:i + 1600]
        self.assertIn("radius:", seg)
        self.assertIn("0.02", seg)

    def test_card_shadow_lift_survives_a_null_focus(self):
        # `shadows` rides `moving = entMorph || focus`, so the shadow call is
        # reached with focus === null on an entrance-morph scene that has no
        # clicks (camera.py returns no focus_cells) and during a card drag.
        # An unguarded `focus.lifts` throws inside sceneTick, which has no
        # try/catch: the rAF never re-arms and S.playing stays true, so live
        # playback FREEZES for good rather than degrading.
        # Anchor on the guard, not on `drawCardShadow(ctx, p, c, k,` -- that
        # substring matches the FUNCTION DEFINITION first.
        i = self.js.index("if (shadows) {")
        seg = self.js[i:i + 300]
        self.assertIn("drawCardShadow(", seg)
        self.assertIn("focus && focus.lifts", seg)

    def test_requeststill_no_longer_bails_on_scene_takes(self):
        # The still fetch must fire for scene takes (the server can composite a
        # frame now); only multi-native, which paints a live client canvas,
        # still bails.
        i = self.js.index("async function requestStill")
        seg = self.js[i:i + 1100]
        self.assertIn("S.details.multi_native) return", seg)
        self.assertNotIn("scene_take) return", seg)

    def test_play_runs_the_live_scene_player_or_falls_back(self):
        # Play now drives the per-scene fleet-ring player when a live plan is
        # built (sceneLive()); the sceneStillOnly hint survives only as the
        # fallback for a take whose plan has not loaded / failed.
        i = self.js.index("function play()")
        seg = self.js[i:i + 700]
        self.assertIn("if (sceneLive()) { playScene(); return; }", seg)
        self.assertIn("S.sceneStillOnly", seg)   # the still-only fallback stays

    def test_scene_live_player_wiring(self):
        # The S3 fleet-ring player: a JS-authoritative per-scene clock (each
        # scene's channel 0 is the played master), a fleet keyed by scene index,
        # seam crossing, and the still floor as fallback.
        for token in ("function resolveScene(", "function buildFleet(",
                      "function getFleet(", "function masterVideo(",
                      "function crossToScene(", "function holdAtSeam(",
                      "function sceneTick(", "function adoptScenePayload("):
            self.assertIn(token, self.js, token)

    def test_scene_resolve_uses_the_export_half_open_rule(self):
        # resolveScene must select the scene the EXPORT does: half-open, the
        # boundary frame belongs to the EARLIER scene (render.py / owner()).
        i = self.js.index("function resolveScene(")
        seg = self.js[i:i + 500]
        self.assertIn("kg < cum + n", seg)

    def test_scene_player_is_gated_off_for_other_takes(self):
        # The off-switch: every scene-player fork is behind sceneLive() /
        # S.details.scene_take, so a plain / multi-native / whole-screen take
        # never enters it and keeps hitting ui.video.
        i = self.js.index("function masterVideo(")
        seg = self.js[i:i + 260]
        self.assertIn("if (sceneLive())", seg)
        self.assertIn("return ui.video", seg)      # non-scene master unchanged
        j = self.js.index("function cellSource(")
        self.assertIn("if (sceneLive())", self.js[j:j + 400])

    def test_scene_fleet_ring_is_lru_capped_and_retains_neighbours(self):
        # A backward scrub over a just-crossed seam must hit a retained fleet,
        # not a cold rebuild -- the LRU keeps the active scene and both
        # neighbours.
        self.assertIn("SCENE_FLEET_CAP", self.js)
        i = self.js.index("function evictFleets(")
        seg = self.js[i:i + 500]
        self.assertIn("S.activeScene - 1", seg)
        self.assertIn("S.activeScene + 1", seg)

    def test_resolution_control_is_wired(self):
        # The export-resolution <select> must save to render.resolution and
        # ride in fullOptions (the live-preview + export payload), or the
        # control would be inert.
        self.assertIn("e.render.resolution = ui.resolution.value", self.js)
        self.assertIn("resolution: r.resolution", self.js)
        self.assertIn("applyResolutionSelect", self.js)

    def test_trim_is_ignored_for_scene_takes(self):
        # Trim is export-ignored on scene takes; the editor timeline must
        # mirror that (full duration) or a stray trim clamps the whole scrub.
        i = self.js.index("function trimRange()")
        seg = self.js[i:i + 500]
        self.assertIn("S.details.scene_take", seg)
        self.assertIn("start: 0, end: d", seg)

    def test_trim_drag_is_disabled_for_scene_takes(self):
        # And the trim-handle gesture must not author one (a lying control).
        i = self.js.index("function startTrimDrag")
        seg = self.js[i:i + 400]
        self.assertIn("S.details.scene_take) return", seg)
