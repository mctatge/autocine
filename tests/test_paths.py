"""Portable path and MCP configuration contracts."""

import contextlib
import io
import json
import os
import unittest
from unittest import mock

from autocine import __version__
from autocine import cli
from autocine import mcp_server
from autocine import paths


class PortablePaths(unittest.TestCase):
    def test_cli_reports_the_package_version(self):
        out = io.StringIO()
        with self.assertRaises(SystemExit) as raised, \
                contextlib.redirect_stdout(out):
            cli.main(["--version"])
        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(out.getvalue().strip(), "AutoCine " + __version__)

    def test_default_recordings_root_is_checkout_relative(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                paths.recordings_root(),
                os.path.join(paths.ROOT, "recordings"))

    def test_environment_override_is_expanded_and_absolute(self):
        with mock.patch.dict(
                os.environ,
                {"AUTOCINE_RECORDINGS_ROOT": "~/Movies/AutoCine"},
                clear=True):
            self.assertEqual(
                paths.recordings_root(),
                os.path.abspath(os.path.expanduser("~/Movies/AutoCine")))

    def test_print_config_uses_current_python_and_absolute_paths(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["mcp", "--print-config"])
        self.assertEqual(code, 0)
        config = json.loads(out.getvalue())["mcpServers"]["autocine"]
        self.assertEqual(config["command"], os.path.abspath(os.sys.executable))
        self.assertEqual(config["args"][0], paths.studio_entrypoint())
        self.assertEqual(config["args"][1], "mcp")
        self.assertEqual(config["args"][2], "--recordings-root")
        self.assertTrue(os.path.isabs(config["args"][3]))

    def test_print_tools_is_generated_from_the_live_schema(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["mcp", "--print-tools"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out.getvalue())["tools"],
                         mcp_server.TOOL_DEFS)


if __name__ == "__main__":
    unittest.main()
