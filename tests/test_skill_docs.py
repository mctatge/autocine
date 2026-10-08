"""The edit-recording skill must not drift from the tool surface it documents.

`.claude/skills/edit-recording/SKILL.md` is the playbook an agent follows to
edit a take over MCP, and it spells out exact argument names -- because getting
one wrong costs a round trip and an error ("missing required argument:
zoom_id"). Prose drifting from code is a named hazard in this repo; the docs
themselves say so about `window_layout`, which shipped with five copies of one
sentence. A doc that tells an agent to call `remove_zoom(id=...)` is worse than
no doc, so the signature block is pinned against `mcp_server.TOOL_DEFS`.

Deliberately narrow: it checks the things a rename breaks (tool names, required
argument names), not the prose around them.
"""

import os
import re
import unittest

from autocine import mcp_server

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(_REPO, ".claude", "skills", "edit-recording", "SKILL.md")

# `name(*session, *start, end, level)` -- `*` marks a required argument. NOT
# line-anchored: the signature block is laid out in two columns, so a line can
# carry two signatures (the first version of this regex silently saw only the
# left-hand one, which its own count guard caught).
_SIG = re.compile(r"(\w+)\((\*?session[^)]*)\)")


def _skill_text():
    with open(SKILL) as f:
        return f.read()


def _signature_block():
    """The fenced block in "3. Edit" that lists the tool signatures.

    Scoped to that block deliberately: parsing the whole file would also pick
    up illustrative calls in the prose (`preview_frame(session, time,
    source: true, ...)`), which are examples, not signatures.
    """
    for block in re.findall(r"```\n(.*?)```", _skill_text(), re.S):
        if "set_trim(" in block and "add_zoom(" in block:
            return block
    return ""


class SkillFrontmatterTests(unittest.TestCase):

    def test_skill_exists_with_a_name_and_description(self):
        self.assertTrue(os.path.isfile(SKILL), SKILL + " is missing")
        text = _skill_text()
        m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
        self.assertIsNotNone(m, "SKILL.md needs a YAML frontmatter block")
        block = m.group(1)
        self.assertIn("name: edit-recording", block)
        self.assertRegex(block, r"description:\s*\S")


class SkillSignatureTests(unittest.TestCase):
    """Every signature the skill prints must match the real tool schema."""

    def setUp(self):
        self.defs = dict((t["name"], t) for t in mcp_server.TOOL_DEFS)
        self.sigs = _SIG.findall(_signature_block())

    def test_the_signature_block_was_actually_found(self):
        # Guards the regex itself: if it stops matching, every assertion
        # below would vacuously pass.
        self.assertTrue(_signature_block(),
                        "the signature fence in section 3 was not found")
        self.assertGreaterEqual(len(self.sigs), 12,
                                "parsed only %d signatures -- the block moved "
                                "or the format changed" % len(self.sigs))

    def test_every_documented_tool_exists(self):
        for name, _args in self.sigs:
            self.assertIn(name, self.defs,
                          "SKILL.md documents a tool that does not exist: %s"
                          % name)

    def test_documented_arguments_exist_on_the_tool(self):
        for name, args in self.sigs:
            props = self.defs[name]["inputSchema"].get("properties", {})
            for raw in args.split(","):
                arg = raw.strip().lstrip("*")
                if not arg or arg == "...":
                    continue
                self.assertIn(arg, props,
                              "SKILL.md documents %s(%s), which is not in the "
                              "schema" % (name, arg))

    def test_starred_arguments_are_the_required_ones(self):
        for name, args in self.sigs:
            required = set(self.defs[name]["inputSchema"].get("required", []))
            starred = set()
            for raw in args.split(","):
                a = raw.strip()
                if a.startswith("*"):
                    starred.add(a.lstrip("*"))
            # The block abbreviates optional args with "..."; only check that
            # what it CALLS required really is.
            for a in starred:
                self.assertIn(a, required,
                              "SKILL.md marks %s(%s) required; the schema does "
                              "not" % (name, a))

    def test_the_id_argument_trap_is_still_real(self):
        """The skill warns that the remove tools take `zoom_id`, not `id`.

        If that ever stops being true the warning becomes a lie that costs an
        agent a failed call, so pin the fact rather than the sentence.
        """
        for tool, expected in (("remove_zoom", "zoom_id"),
                               ("remove_speedup", "speedup_id"),
                               ("remove_marker", "marker_id"),
                               ("remove_window", "window_id")):
            props = self.defs[tool]["inputSchema"].get("properties", {})
            self.assertIn(expected, props)
            self.assertNotIn("id", props)
        self.assertIn("zoom_id", _skill_text())


class SkillClaimTests(unittest.TestCase):
    """Claims the skill makes about behavior, pinned to the code."""

    def test_describe_session_really_takes_detail_and_max_beats(self):
        props = dict((t["name"], t) for t in mcp_server.TOOL_DEFS)[
            "describe_session"]["inputSchema"]["properties"]
        for arg in ("session", "detail", "max_beats"):
            self.assertIn(arg, props)

    def test_preview_frame_really_takes_source(self):
        props = dict((t["name"], t) for t in mcp_server.TOOL_DEFS)[
            "preview_frame"]["inputSchema"]["properties"]
        self.assertIn("source", props)
        self.assertIn("max_width", props)

    def test_trim_is_still_endpoints_and_cuts_are_the_ranges_tool(self):
        """The skill says trim is the single in/out endpoints tool and CUTS
        are how mid-take ranges are removed. Pin both halves: set_trim's
        schema staying {session, start, end}, and the three cut tools
        existing with the documented cut_id argument convention -- so the
        skill's cuts section can't silently rot in either direction.
        """
        defs = dict((t["name"], t) for t in mcp_server.TOOL_DEFS)
        props = defs["set_trim"]["inputSchema"]["properties"]
        self.assertEqual(set(props), {"session", "start", "end"})
        for tool in ("add_cut", "remove_cut", "set_cuts"):
            self.assertIn(tool, defs)
        rc = defs["remove_cut"]["inputSchema"]
        self.assertIn("cut_id", rc.get("properties", {}))
        self.assertNotIn("id", rc.get("properties", {}))
        self.assertIn("cut_id", rc.get("required", []))
        # the skill text documents the ripple, the snap echo, and the
        # preview limitation -- pin the load-bearing words
        text = _skill_text()
        self.assertIn("add_cut", text)
        self.assertIn("set_cuts", text)
        self.assertIn("snapped", text)
        self.assertIn("SCRUBBING still shows removed frames", text)


if __name__ == "__main__":
    unittest.main()
