"""Static pins on the browser sources in `studio_web/`.

There is no JS harness here (see docs/architecture.md), so these are source pins --
the only guard available for an invariant that lives entirely in the browser.

`WebScriptsResolve` exists because of a real, shipped bug: a refactor routed
three card-drag call sites through a helper, `canvasFrac`, that was never
written. Dragging a window card in the editor's grid died outright, and
silently -- an uncaught ReferenceError inside a `pointerdown` handler shows
up nowhere the user or the server can see it, so nothing surfaced until
someone tried to move a card. `node --check` does not catch it (the syntax is
valid) and no Python touches these files, so this walk of the sources is what
stands in for a linter.
"""

import inspect
import os
import re
import unittest

from autocine import edits, framing
from tests import arrangement_claims as claims

WEB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "studio_web")

_KEYWORDS = set("""
if for while switch catch function return typeof new delete void do else await
yield in of instanceof case throw with super this try finally var let const
""".split())

# Supplied by the browser rather than by these files. Deliberately a list of
# what is actually used, not a copy of the web platform: a name that belongs
# here but is missing just means one more line the day it is first called.
_BROWSER = set("""
window document console navigator location fetch setTimeout clearTimeout
setInterval clearInterval requestAnimationFrame cancelAnimationFrame
Math JSON Object Array Number String Boolean Date Promise Error Set Map
parseInt parseFloat isNaN isFinite encodeURIComponent decodeURIComponent
encodeURI decodeURI alert confirm prompt Image FormData Blob URL Audio Video
PointerEvent MouseEvent KeyboardEvent CustomEvent Event EventSource WebSocket
Uint8Array Float32Array ArrayBuffer TextDecoder TextEncoder ResizeObserver
IntersectionObserver MutationObserver AbortController structuredClone atob btoa
getComputedStyle matchMedia devicePixelRatio Intl Symbol Proxy Reflect BigInt
queueMicrotask performance localStorage sessionStorage history screen
URLSearchParams DOMParser XMLHttpRequest Notification MediaRecorder
""".split())


def _strip_noise(src):
    """Blank out comments, strings and regex literals, keeping line numbers.

    Prose in a comment ("...one consumer (the cursor)...") and CSS in a
    string ("rgba(...)") both look exactly like a call to the scan below, so
    they have to go first. Blanked rather than deleted so a reported line
    number still points at the real line.
    """
    out = []
    i, n = 0, len(src)
    prev = ""      # last significant char -- tells a regex from a divide
    while i < n:
        ch = src[i]
        two = src[i:i + 2]
        if two == "//":
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
        elif two == "/*":
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join(c if c == "\n" else " " for c in src[i:j]))
            i = j
        elif ch in "\"'`":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == ch:
                    j += 1
                    break
                j += 1
            out.append("".join(c if c == "\n" else " " for c in src[i:j]))
            i = j
        elif ch == "/" and prev in "(,=:[!&|?{};+-*%~^<>":
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == "[":
                    while j < n and src[j] != "]":
                        j += 2 if src[j] == "\\" else 1
                if src[j] == "/":
                    j += 1
                    break
                if src[j] == "\n":
                    break
                j += 1
            out.append(" " * (j - i))
            i = j
        else:
            out.append(ch)
            if not ch.isspace():
                prev = ch
            i += 1
    return "".join(out)


def _bound_names(src):
    """Every name the file introduces: declarations, params, catch vars."""
    out = set()
    out |= set(re.findall(r"\bfunction\s+([A-Za-z_$][\w$]*)", src))
    out |= set(re.findall(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)", src))
    for block in re.findall(r"\b(?:const|let|var)\s*[\{\[]([^\}\]]*)[\}\]]", src):
        out |= set(re.findall(r"[A-Za-z_$][\w$]*", block))
    for params in re.findall(r"function\s*[A-Za-z_$\w]*\s*\(([^)]*)\)", src):
        out |= set(re.findall(r"[A-Za-z_$][\w$]*", params))
    for params in re.findall(r"\(([^()]*)\)\s*=>", src):
        out |= set(re.findall(r"[A-Za-z_$][\w$]*", params))
    out |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*=>", src))
    out |= set(re.findall(r"catch\s*\(\s*([A-Za-z_$][\w$]*)", src))
    out |= set(re.findall(r"([A-Za-z_$][\w$]*)\s*:\s*function", src))
    return out


def _bare_calls(src):
    """`name(` call sites that are not method calls, as {name: line}."""
    hits = {}
    for m in re.finditer(r"([A-Za-z_$][\w$]*)\s*\(", src):
        name = m.group(1)
        before = src[:m.start()].rstrip()
        if before.endswith(".") or before.endswith("?.") \
                or before.endswith("function"):
            continue
        if name in _KEYWORDS:
            continue
        hits.setdefault(name, src[:m.start()].count("\n") + 1)
    return hits


def _read(name):
    with open(os.path.join(WEB, name)) as f:
        return _strip_noise(f.read())


def _raw(name):
    """The file as written, strings intact.

    `_read` blanks string literals, which is right for the call scan and
    wrong for a pin *on* a literal -- the layout names below live in quotes.
    """
    with open(os.path.join(WEB, name)) as f:
        return f.read()


def _scripts():
    return sorted(f for f in os.listdir(WEB) if f.endswith(".js"))


def _call_args(src, open_idx):
    """Top-level arguments of the call whose `(` is at `open_idx`.

    A `split(",")` cannot read these: every one of the calls pinned below
    passes a function literal, whose body has commas and parens of its own.
    Depth-counted instead, over a source `_strip_noise` has already blanked,
    so a comma inside a string cannot end an argument either.
    """
    assert src[open_idx] == "(", src[open_idx:open_idx + 20]
    depth = 0
    args, cur = [], []
    for i in range(open_idx, len(src)):
        ch = src[i]
        if ch in "([{":
            depth += 1
            if depth == 1:
                continue
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                args.append("".join(cur))
                return [a.strip() for a in args]
        elif ch == "," and depth == 1:
            args.append("".join(cur))
            cur = []
            continue
        cur.append(ch)
    raise AssertionError("unbalanced call at offset %d" % open_idx)


def _args_of(src, raw, needle):
    """`_call_args` for the call whose source text starts with `needle`.

    Located in the RAW source (the label that identifies the call is a string
    literal) and read in the stripped one -- legal only because
    `_strip_noise` blanks characters and never removes them, so one offset
    means the same place in both.
    """
    assert len(src) == len(raw), "the stripper changed the file's length"
    idx = raw.index(needle)
    return _call_args(src, src.index("(", idx))


def _fit_like_the_editor(W, H, cells, pad):
    """`fitPlan`'s arithmetic transcribed, then re-applied like the server.

    The editor writes CANVAS FRACTIONS and `framing._apply_card_overrides`
    turns them back into pixels, so a comparison against `fit_placements`
    only means something if it goes through both halves. Kept a transcription
    rather than a call into framing on purpose: the point is to be a second
    implementation that can DISAGREE.
    """
    x0 = min(c["x"] for c in cells)
    y0 = min(c["y"] for c in cells)
    x1 = max(c["x"] + c["w"] for c in cells)
    y1 = max(c["y"] + c["h"] for c in cells)
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    scale = min((W - 2 * pad) / float(bw), (H - 2 * pad) / float(bh))
    off_x, off_y = (W - bw * scale) / 2.0, (H - bh * scale) / 2.0
    out = []
    for c in cells:
        lx = (off_x + (c["x"] - x0) * scale) / float(W)
        ly = (off_y + (c["y"] - y0) * scale) / float(H)
        lw, lh = (c["w"] * scale) / float(W), (c["h"] * scale) / float(H)
        fw = max(2, min(W, int(round(lw * W))))
        fh = max(2, min(H, int(round(lh * H))))
        out.append((max(0, min(W - fw, int(round(lx * W)))),
                    max(0, min(H - fh, int(round(ly * H)))), fw, fh))
    return out


class WebScriptsResolve(unittest.TestCase):
    def test_every_bare_call_resolves(self):
        shared = _bound_names(_read("shared.js"))
        for fn in _scripts():
            src = _read(fn)
            known = _bound_names(src) | _BROWSER | _KEYWORDS
            if fn != "shared.js":
                known |= shared          # every page loads shared.js first
            missing = sorted((line, name)
                             for name, line in _bare_calls(src).items()
                             if name not in known)
            self.assertEqual(missing, [], "%s calls names nothing defines: %s"
                             % (fn, missing))

    def test_the_scan_would_have_caught_it(self):
        """The heuristic above only earns its keep if it still detects one.

        Blanking too much (a stripper bug) would turn the test above into a
        no-op that passes forever, which is exactly the failure mode it was
        written to prevent.
        """
        src = _strip_noise(
            "/* canvasFrac( in a comment */\n"
            "const css = 'translate(1px)';\n"
            "function cardAtPoint(x, y) { return canvasFrac(x, y); }\n")
        missing = set(_bare_calls(src)) - _bound_names(src) - _BROWSER
        self.assertEqual(missing, {"canvasFrac"})


class CardDragPins(unittest.TestCase):
    """The two things that make a card drag visible, both browser-only.

    Neither has a Python surface: the first is a canvas the server never
    sees, the second an <img> stacking order.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def test_drag_falls_back_to_the_base_layout(self):
        # Window focus hands the browser finished per-frame cells, so a drag
        # that moved p.cells could not move THEM -- the card would sit frozen
        # under the pointer. activeFocusFrame is what stands down for a drag.
        body = self.src.split("function activeFocusFrame")[1].split("}")[0]
        self.assertIn("S.cardDrag", body)
        self.assertIn("return null", body)

    def test_the_rendered_still_stands_down_for_a_drag(self):
        # The still is opaque and painted OVER the composite, so one arriving
        # mid-drag hides the card being moved.
        body = self.src.split("async function requestStill")[1].split("\n  }")[0]
        self.assertIn("S.cardDrag", body)


class ArrangementControlPins(unittest.TestCase):
    """The Arrangement control has to offer every layout the server accepts.

    A new arrangement lands in `framing._ARRANGEMENTS`, is let through by
    `edits._WINDOW_LAYOUTS`, and reaches users through a CLI flag, an MCP
    tool and this one array -- and the array is the only one of those a
    person can click. Add "feature" everywhere else and it ships working,
    passes every Python test, and is INVISIBLE in the editor. Nothing
    raises, nothing logs; the preset simply isn't offered.
    """

    def test_the_control_offers_every_accepted_layout(self):
        block = re.search(r"\bWINDOW_LAYOUTS\s*=\s*\[(.*?)\]", _raw("editor.js"),
                          re.S)
        self.assertIsNotNone(block, "editor.js no longer declares WINDOW_LAYOUTS")
        strings = re.findall(r"""["']([^"'\n]*)["']""", block.group(1))
        # Which quoted strings are VALUES is read off the file's own
        # convention -- every seg control in editor.js pairs a lowercase
        # token with a capitalized label -- rather than off the `v` field
        # name, so renaming the field doesn't cost a red test.
        values = [s for s in strings if re.match(r"[a-z][a-z0-9_-]*$", s)]
        self.assertEqual(sorted(values), sorted(edits._WINDOW_LAYOUTS),
                         "the Arrangement buttons and edits._WINDOW_LAYOUTS "
                         "disagree about what layouts exist")

    def test_that_array_is_what_the_control_paints(self):
        # An options array nothing renders pins nothing.
        self.assertRegex(_read("editor.js"), r"rowSeg\w*\(\s*[^()]*WINDOW_LAYOUTS")

    def test_the_lit_segment_is_decided_from_the_same_array(self):
        """Which button lights up used to come from a two-way ternary
        (`=== "desktop" ? "desktop" : "grid"`). Left in place beside five
        buttons, that shows "Grid" selected for feature/row/column: the saved
        value is correct and the UI contradicts it, which is worse than the
        preset being missing.

        This asked only that SOME method be called on `WINDOW_LAYOUTS`
        anywhere in the file, which caught that bug purely because there was
        exactly one such call -- one unrelated `WINDOW_LAYOUTS.map()` for a
        tooltip and it would have passed with the ternary back in place, and
        it never checked that the answer reached the control. So follow the
        value instead: the argument that lights a segment, back to the array.
        """
        raw = _raw("editor.js")
        src = _read("editor.js")
        args = _args_of(src, raw, 'rowSegStack("Arrangement"')
        self.assertGreaterEqual(len(args), 3, "the Arrangement control's call "
                                              "shape changed: %r" % (args,))
        self.assertEqual(args[1], "WINDOW_LAYOUTS",
                         "the control is painted from something other than "
                         "WINDOW_LAYOUTS")
        value = args[2]
        self.assertIn("window_layout", value,
                      "the lit segment ignores the saved arrangement")

        # The array has to be consulted about THIS value. Either inline, or
        # through a local -- follow it rather than accepting a bare mention.
        decision = value if re.search(r"WINDOW_LAYOUTS\s*\.\s*\w+\s*\(", value) \
            else None
        if decision is None:
            for name in sorted(set(re.findall(r"[A-Za-z_$][\w$]*", value))):
                decl = re.search(
                    r"\b(?:const|let|var)\s+%s\s*=\s*([^;]*)" % re.escape(name),
                    src)
                if decl and re.search(r"WINDOW_LAYOUTS\s*\.\s*\w+\s*\(",
                                      decl.group(1)):
                    decision = decl.group(1)
                    break
        self.assertIsNotNone(
            decision,
            "nothing WINDOW_LAYOUTS decides reaches the lit segment; it is "
            "chosen by %r" % (value,))
        self.assertIn("window_layout", decision,
                      "the membership test is not asked about the saved "
                      "arrangement: %r" % (decision,))

    def test_the_third_argument_is_what_lights_a_segment(self):
        """The pin above reads argument 3 as "the selected value", which is
        only true of `rowSegStack`'s own body -- and that body is one file
        over, free to change. Nothing else in these pins would notice."""
        raw = _raw("editor.js")
        src = _read("editor.js")
        params = _args_of(src, raw, "function rowSegStack")
        self.assertGreaterEqual(len(params), 3)
        body = src.split("function rowSegStack", 1)[1]
        # `o.v === value ? "on" : null` -- the option compared against that
        # parameter is what decides the lit class, whatever either is called.
        self.assertRegex(
            body, r"[A-Za-z_$][\w$]*\.\w+\s*===\s*%s\b" % re.escape(params[2]),
            "rowSegStack no longer lights a segment by comparing its third "
            "argument (%r) against an option" % params[2])


class CardPlacementActionPins(unittest.TestCase):
    """"Fit to frame" and "Reset placement" -- both browser-only writers.

    `WebScriptsResolve` above catches a handler that is called but never
    written; these catch the inverse (an action nothing can reach) and the
    two ways the write itself can be wrong while still looking right on
    screen: bypassing `mutate` (which is what calls `pushUndo` and
    `scheduleSave` -- a direct `S.edits` poke repaints the preview, so it
    LOOKS applied, then is neither undoable nor ever saved), and touching
    one card instead of all of them.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def _body(self, fn):
        head = "function " + fn
        self.assertIn(head, self.src, "%s is gone" % fn)
        return self.src.split(head)[1].split("\n  }")[0]

    def test_both_actions_are_reachable(self):
        for fn in ("fitCardsToFrame", "resetCardPlacement"):
            self.assertGreaterEqual(
                self.src.count(fn), 2,
                "%s is defined and nothing references it" % fn)

    def test_fit_writes_every_card_through_mutate(self):
        body = self._body("fitCardsToFrame")
        self.assertIn("mutate(", body)
        before, inside = body.split("mutate(", 1)
        # The only `layout` write in the function is the one inside the
        # callback -- the function ends with that call, so "after mutate("
        # is "inside it".
        self.assertNotRegex(before, r"\.layout\s*=[^=]")
        self.assertRegex(inside, r"\.layout\s*=[^=]")
        # Every card, not the one that happened to be dragged: a fit that
        # rescales card 0 and leaves the rest pulls the arrangement apart.
        self.assertRegex(inside, r"\b(?:forEach|for)\s*\(")
        self.assertNotRegex(body, r"S\.edits[^;]*\.layout\s*=[^=]")

    def test_fit_uses_the_painters_padding_convention(self):
        """The fit arithmetic is duplicated in JS; the margin must not be.

        `framing.MultiFramePainter` takes its pad off the WIDTH and spends
        it on both axes. A fit that padded each axis by its own dimension
        would look fine on its own and then jump the moment the user
        switched arrangement, because the auto-layout underneath it sits in
        a different margin.
        """
        body = self._body("fitPlan")
        # `int(framing.PAD_FRAC * W)` on the Python side -- see the numeric pin
        # for why the truncation is part of the convention and not noise.
        pad_decl = re.search(
            r"(\w+)\s*=\s*Math\.trunc\(\s*(\d*\.\d+)\s*\*\s*(\w+)\s*\)", body)
        self.assertIsNotNone(pad_decl, "no truncated pad fraction in fitPlan")
        pad, frac, w_name = pad_decl.groups()
        for owner in (framing.MultiFramePainter.__init__, framing.fit_placements):
            self.assertEqual(float(frac),
                             inspect.signature(owner).parameters["pad_frac"].default,
                             "the editor's fit pads by a different fraction "
                             "than %s" % owner.__name__)
        # ONE margin: a second `padH = PAD_FRAC * H` beside it is the whole
        # mistake, and it hides from a substring search for "pad".
        self.assertEqual(len(re.findall(r"\d*\.\d+\s*\*\s*\w+", body)), 1,
                         "fitPlan computes more than one margin")
        # taken off the WIDTH...
        self.assertRegex(body, r"\b%s\s*=\s*[\w.]*canvas\s*\[\s*0\s*\]" % w_name)
        h_decl = re.search(r"(\w+)\s*=\s*[\w.]*canvas\s*\[\s*1\s*\]", body)
        self.assertIsNotNone(h_decl, "fitPlan reads no canvas height")
        # ...and subtracted from both, the same one on each.
        for axis in (w_name, h_decl.group(1)):
            self.assertRegex(body, r"\b%s\b\s*-[^)]*\b%s\b" % (axis, pad),
                             "%s is not reduced by %s" % (axis, pad))

    def test_the_padding_convention_includes_throwing_the_remainder_away(self):
        """The pin above used to stop at the fraction, and the NUMBER was
        never the part that differed.

        Python spends `int(PAD_FRAC * W)`; JS `PAD_FRAC * W` keeps the
        remainder,
        and on a real canvas there always is one (1440 px wide -> 79.2). That
        0.2 px is not the error -- it goes through `scale`, which multiplies
        the whole arrangement, so the editor's fit and the `fit_placements`
        the MCP twin calls land in visibly different places. Proved here by
        computing both: the truncated transcription of the editor's
        arithmetic reproduces `fit_placements` exactly, the untruncated one
        does not. Without the second assertion this test would still pass
        against a JS file that never truncated.
        """
        W, H = 1440, 900
        F = framing.PAD_FRAC
        self.assertNotEqual(F * W, int(F * W), "pick a W with a remainder")
        cells = [{"x": 96, "y": 70, "w": 520, "h": 640},
                 {"x": 700, "y": 120, "w": 430, "h": 260}]
        expected = framing.fit_placements(W, H, cells)
        self.assertEqual(_fit_like_the_editor(W, H, cells, int(F * W)),
                         expected)
        self.assertNotEqual(_fit_like_the_editor(W, H, cells, F * W),
                            expected)

    def test_fit_is_written_by_card_id(self):
        # The plan is measured against S.edits as it stands, so naming the
        # card the geometry belongs to costs one lookup and removes the last
        # place an index could land on the wrong window.
        inside = self._body("fitCardsToFrame").split("mutate(", 1)[1]
        self.assertRegex(inside, r"\.id\s*===",
                         "fitCardsToFrame writes cards by position, not id")

    def test_reset_clears_every_card_through_mutate(self):
        body = self._body("resetCardPlacement")
        self.assertIn("mutate(", body)
        inside = body.split("mutate(", 1)[1]
        # A reset that misses a card is the worst of both: half the
        # composition snaps back to the arrangement, half stays where it was
        # dropped, and no auto-layout produces the result.
        self.assertRegex(inside, r"\b(?:forEach|for)\s*\(")
        self.assertRegex(
            inside,
            r"(?:delete\s+[\w.]*\blayout\b|\.layout\s*=\s*(?:null|undefined))")

    # -- and what that clearing is WORTH, which the regex above cannot say --
    #
    # Everything above this line pins the shape of the JS. None of it asserts
    # that a card whose `layout` was cleared comes back where the arrangement
    # wanted it -- which is the entire promise of the button ("Drop every hand
    # placement and go back to the arrangement"). `_apply_card_overrides` is
    # the Python that decides it, so the promise IS executable; it simply had
    # nothing checking it on either side.
    #
    # `delete w.layout` is deliberately NOT the case tested: it leaves the
    # card as the dict it started as, so asserting it lands on the
    # arrangement compares an input with itself and could never fail. What
    # can is a cleared card that still HAS the key -- `w.layout = null` (the
    # other spelling the source pin above accepts) and `w.layout = {}` (a
    # half-written one). Both have to read as "no override" or a reset drops
    # every card in the top-left corner at 20% size.

    RESET_CANVAS = (960, 600)

    def _cells(self, layout, rects):
        W, H = self.RESET_CANVAS
        return framing.make_multi_painter(W, H, rects, layout=layout).cells

    def _dragged(self):
        """The three real windows, every card hand-placed somewhere else."""
        return [dict(r, layout={"x": 0.03 + 0.31 * i, "y": 0.55,
                                "w": 0.28, "h": 0.33})
                for i, r in enumerate(claims.REAL_WINDOWS)]

    def test_clearing_the_overrides_reproduces_the_arrangement(self):
        base = [dict(r) for r in claims.REAL_WINDOWS]
        dragged = self._dragged()
        for layout in edits._WINDOW_LAYOUTS:
            auto = self._cells(layout, base)
            self.assertNotEqual(
                self._cells(layout, dragged), auto,
                "%s: the overrides move nothing, so this proves nothing about "
                "clearing them" % layout)
            for cleared in (None, {}):
                rects = [dict(r, layout=cleared) for r in dragged]
                self.assertEqual(
                    self._cells(layout, rects), auto,
                    "%s: `layout = %r` does not restore the arrangement"
                    % (layout, cleared))

    def test_a_cleared_placement_survives_the_save_as_no_override(self):
        """`resetCardPlacement` writes into `S.edits` and the document is then
        saved, so the reset only holds if `normalize_edits` agrees a cleared
        card has no placement."""
        dragged = edits.normalize_edits(
            {"windows": self._dragged()}, duration=10.0)["windows"]
        self.assertTrue(all("layout" in w for w in dragged),
                        "the fixture's overrides did not survive normalize, "
                        "so the contrast below is empty")
        cleared = edits.normalize_edits(
            {"windows": [dict(w, layout=None) for w in dragged]},
            duration=10.0)["windows"]
        for w in cleared:
            self.assertNotIn("layout", w)
        pristine = edits.normalize_edits(
            {"windows": [dict(r) for r in claims.REAL_WINDOWS]},
            duration=10.0)["windows"]
        for layout in edits._WINDOW_LAYOUTS:
            self.assertEqual(self._cells(layout, cleared),
                             self._cells(layout, pristine),
                             "%s: a saved reset is not the pure arrangement"
                             % layout)


class FitFreshnessPins(unittest.TestCase):
    """"Fit to frame" writes geometry it measured from a payload that LAGS
    the document it writes into.

    `/api/camera-path` answers ~1s after the edit that provoked it (a 320ms
    debounce plus the round trip), and its `cells` bind to `edits.windows`
    by POSITION -- the payload carries no id to check that against. Remove a
    window, or reorder two, and click Fit before the next path lands: index
    i is a different card in each list, so every remaining card is written
    the previous arrangement's cell -- wrong shape, wrong place, no error.
    The first version paired them under
    `Math.min(p.cells.length, wins.length)`, which made even a length
    mismatch proceed silently.

    There is no id in the payload to bind by (that would be a change to
    `studio_app.camera_path`), so the binding is a freshness stamp instead:
    the path records what it was computed from and the fit refuses anything
    else. A button that is dead for one round trip is a fine price; a silent
    write of the wrong geometry is not.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def _body(self, fn):
        head = "function " + fn
        self.assertIn(head, self.src, "%s is gone" % fn)
        return self.src.split(head)[1].split("\n  }")[0]

    def test_the_path_carries_what_it_was_computed_from(self):
        body = self._body("fetchCamPath")
        self.assertIn("cellsSignature(", body,
                      "fetchCamPath no longer stamps the path")
        self.assertRegex(body, r"cells_sig\s*=",
                         "nothing writes the stamp onto the response")
        # Stamped from the edits the REQUEST was built from. Computing it
        # after the await would sign the document the answer arrived into --
        # which is the very thing being guarded against, and would make the
        # gate pass exactly when it must not.
        self.assertLess(body.index("cellsSignature("), body.index("await "),
                        "the stamp is taken after the request comes back")

    def test_the_stamp_covers_everything_a_cell_depends_on(self):
        # `multi_window_layout(windows, background, style, aspect,
        # window_layout)` -- background paints the plate and cannot move a
        # cell, so it is deliberately absent; the other four are what makes
        # a stale path detectable.
        body = self._body("cellsSignature")
        for key in ("windows", "window_layout", "aspect", "style"):
            self.assertIn(key, body, "the stamp ignores %s" % key)

    def test_the_fit_refuses_a_path_it_cannot_bind(self):
        body = self._body("fitPlan")
        self.assertIn("cells_sig", body, "fitPlan does not check freshness")
        self.assertRegex(body, r"cells\.length\s*!==\s*\w+\.length",
                         "fitPlan does not check that the lists correspond")
        self.assertNotRegex(
            body, r"Math\.min\s*\([^)]*length[^)]*length",
            "fitPlan is back to truncating two lists to the shorter one")

    def test_the_button_and_the_write_ask_the_same_question(self):
        """Two copies of "is this safe" is how a disabled button and a
        garbage write coexist: the button reads one, the click runs the
        other, and only one of them gets fixed."""
        self.assertRegex(self._body("fitCardsToFrame"), r"fitPlan\(\s*\)")
        self.assertRegex(self._body("syncFitAction"), r"fitPlan\(\s*\)")
        # ...and the click stands down on exactly what the button greys out
        # for, rather than deciding for itself.
        self.assertRegex(self._body("fitCardsToFrame"),
                         r"if\s*\(\s*!\s*\w+\.cards\s*\)\s*return")

    def test_the_disabled_button_says_why(self):
        """A control that greys out for two unrelated reasons -- the path is
        catching up, or there is genuinely nothing to take out -- and
        explains neither is indistinguishable from a broken one."""
        body = self._body("syncFitAction")
        self.assertRegex(body, r"\.disabled\s*=")
        self.assertIn("reason", body, "syncFitAction shows no reason")
        # The class name is a string literal, which `_read` blanks -- this
        # one has to be read off the file as written, in both places.
        raw = _raw("editor.js")
        raw_sync = raw.split("function syncFitAction", 1)[1].split("\n  }")[0]
        self.assertIn("ed-fit-note", raw_sync,
                      "nothing paints the reason where it can be read")
        self.assertIn("ed-fit-note", raw.split("ed-actions", 1)[1],
                      "the panel builds no note element for it")
        with open(os.path.join(WEB, "editor.css")) as f:
            self.assertIn(".ed-fit-note", f.read(),
                          "the note has no style of its own")

    def test_the_fit_declines_to_do_nothing(self):
        """Fit is a real 2.15x of ink on a hand-dragged arrangement and takes
        nothing off a freshly applied preset -- the four scaled arrangements
        already sit at precisely the scale it computes, to within the pixel
        of re-centring that FIT_EPS_PX exists to absorb. Offering it as if it
        will do something is the same class of untruth as the "fills the
        frame" claim it replaced, so the plan measures whether it moves
        anything and the button follows."""
        body = self._body("fitPlan")
        self.assertRegex(body, r"Math\.abs\([^)]*\)\s*>",
                         "fitPlan never compares its result to what is there")
        self.assertRegex(body, r"return\s*\{\s*reason",
                         "fitPlan cannot report a no-op")


class ArrangementFactsMatchThePainter(unittest.TestCase):
    """`arrangement_claims.placements` copies the painter's pad/gap derivation
    so the copy pins below can measure a 2880x1800 arrangement without paying
    for the shadow bake. Copied arithmetic that nothing checks would leave
    every coverage figure describing a layout nothing renders -- and the copy
    would then be judged against a fiction.
    """

    def test_the_measurement_helper_is_what_the_painter_places(self):
        for layout in edits._WINDOW_LAYOUTS:
            painter = framing.make_multi_painter(1200, 750,
                                                 claims.REAL_WINDOWS,
                                                 layout=layout)
            self.assertEqual(
                [tuple(c) for c in claims.placements(layout, 1200, 750)],
                [(c["x"], c["y"], c["w"], c["h"]) for c in painter.cells],
                "%s: the coverage helper places cards the painter does not"
                % layout)


class ArrangementHintPins(unittest.TestCase):
    """The hint under the Arrangement control is the only place the editor
    says what each arrangement is FOR -- and the place this feature has now
    shipped wrong copy twice.

    First it claimed feature/row/column "fill the frame on their own". They do
    not: all four scaled arrangements go through one uniform
    `min(avail_w/span_w, avail_h/span_h)`, which fills ONE axis and
    letterboxes the other by however the arrangement's bbox aspect differs
    from the canvas -- unavoidable for aspect-locked rectangles.

    Then, after `feature` learned to transpose, the hint went on describing
    the layout it used to be: "for a wide export, a waste of a tall one",
    written when feature covered 26.6% of a 9:16 canvas. It covers 67.1% of
    one now -- within a quarter-point of the best of the five, and best
    outright on 16:10 (67.7%), 16:9 (57.9%) and 1:1 (50.6%). The editor was
    steering users away from the one arrangement that holds up at any export
    shape.

    So the pins here are not "some words exist". They measure the layouts and
    check the prose against the measurement. Their reach is bounded, and the
    bound is written on `arrangement_claims.dismissals`: a write-off in so
    many words is caught, steering by omission is not.
    """

    def _hint_block(self):
        raw = _raw("editor.js")
        self.assertIn('rowSegStack("Arrangement"', raw,
                      "the Arrangement control is gone")
        after = raw.split('rowSegStack("Arrangement"', 1)[1]
        return after.split("if (windows.length > 0)", 1)[0]

    def _hint_prose(self):
        """Just the words.

        The block also carries the control's own JS, and `"grid"` in the
        `known ? ... : "grid"` fallback is not the copy talking about the
        grid -- reading it as prose would attach the next sentence's verdict
        to the wrong arrangement.
        """
        block = self._hint_block()
        self.assertIn('"ed-hint"', block, "the Arrangement control has no hint")
        pieces = re.findall(r'"((?:[^"\\]|\\.)*)"',
                            block.split('"ed-hint"', 1)[1])
        prose = "".join(pieces)
        self.assertTrue(prose.strip(), "the Arrangement hint is empty")
        return prose

    def test_the_hint_quotes_no_coverage_the_code_does_not_produce(self):
        """Every percentage in the hint has to be one the layouts actually
        measure, today, on the windows the figures were taken from.

        This is the pin the round that went wrong needed and did not have:
        the numbers were true when written and became false when `feature`
        transposed, and nothing in the tree read them. Checked against every
        arrangement the sentence names (see `percent_claims` for why
        attribution inside a sentence is not attempted), which is still
        enough -- the stale figures were coverages of nothing at all.
        """
        for names, pct in claims.percent_claims(self._hint_prose()):
            measured = [v for name in names
                        for v in claims.measured_values(name)]
            self.assertTrue(
                any(abs(pct - v) <= claims.PCT_TOL for v in measured),
                "the hint claims %.1f%% for %s; measured today: %s"
                % (pct, "/".join(names),
                   ", ".join("%.1f" % v for v in sorted(measured))))

    def test_the_hint_writes_off_no_arrangement_the_code_rates(self):
        """A layout may be called bad for an export shape only if it is.

        "Feature ... for a wide export, a waste of a tall one" was the exact
        sentence that shipped, against a layout measuring 0.2 points off the
        best on a 9:16 canvas. Writing off `desktop` there (41 points off) is
        fine and stays fine -- the pin is the contradiction, not the tone.
        """
        for name, shape, sentence in claims.dismissals(self._hint_prose()):
            self.assertGreater(
                claims.gap_from_best(name, shape), claims.DISMISSAL_TOL,
                "the hint writes '%s' off for a %s export, but it measures "
                "within %.1f points of the best layout there -- %r"
                % (name, shape, claims.gap_from_best(name, shape), sentence))

    def test_the_hint_speaks_to_every_arrangement(self):
        # Labels off the same array the buttons are painted from, so adding
        # an arrangement and not saying what it is for is a red test rather
        # than a button nobody can choose between.
        block = re.search(r"\bWINDOW_LAYOUTS\s*=\s*\[(.*?)\]", _raw("editor.js"),
                          re.S).group(1)
        labels = [s for s in re.findall(r"""["']([^"'\n]*)["']""", block)
                  if re.match(r"[A-Z][a-z]+$", s)]
        self.assertTrue(labels)
        hint = self._hint_block()
        for label in labels:
            self.assertIn(label, hint,
                          "the Arrangement hint never mentions %s" % label)

    def test_the_hint_does_not_claim_they_fill_the_frame(self):
        self.assertNotRegex(
            self._hint_block(), r"fill(?:s|ing)?\s+the\s+frame",
            "the hint is back to claiming an arrangement fills the frame; it "
            "fills one axis")


class PreviewBoxPins(unittest.TestCase):
    """The stage box has to BE the output's shape, and the composite has to
    survive it not being.

    The shipped bug this pins: `fitCanvas` measured the wrap with
    `clientWidth/clientHeight - 8`, but those include .ed-canvas-wrap's
    `padding: 14px 20px 18px`, while .ed-canvas's `max-width/max-height: 100%`
    resolve against the CONTENT box. So the fit set a correctly-shaped box and
    CSS clamped ONE axis of it -- measured, wrap 1116x616: set 973x608 (1.600),
    served 973x584 (1.666).

    Nothing reported that, because the two things drawn into the box absorb a
    wrong aspect differently: the paused still was `object-fit: fill` and
    stretched to hide it, while the live composite scaled by WIDTH alone and
    anchored at the origin. Pressing play therefore resized and shifted the
    frame -- 24px here, and the other way (smaller, riding up, a band of stage
    grey beneath) on a stage whose width binds first.

    Three source pins because the bug needed all three parts to be invisible.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def _fit_canvas_body(self):
        return self.src.split("function fitCanvas")[1].split("\n  }")[0]

    def test_the_fit_measures_the_wrap_content_box(self):
        body = self._fit_canvas_body()
        self.assertIn("padding", body.lower(),
                      "fitCanvas no longer subtracts .ed-canvas-wrap's "
                      "padding, so .ed-canvas's max-width/max-height clamp one "
                      "axis and the stage stops being the output's shape")
        self.assertNotRegex(
            body, r"client(?:Width|Height)\s*-\s*\d",
            "fitCanvas is back to fitting a constant inset of the PADDED "
            "client box; the padding is 20px/16px a side, not 4")

    def test_the_fit_rounds_away_from_the_clamp(self):
        # Rounding UP puts the box back over the max it was just fitted to,
        # which hands it to the very clamp this function exists to avoid.
        body = self._fit_canvas_body()
        self.assertIn("Math.floor", body)
        self.assertNotIn("Math.round", body)

    def test_the_composite_contains_rather_than_scaling_by_width(self):
        # "function drawMulti(" with the paren: drawMultiCursor is defined
        # first and a bare prefix split lands in it instead.
        body = self.src.split("function drawMulti(")[1].split("\n  }")[0]
        self.assertIn("compositeFit(", body,
                      "drawMulti no longer goes through compositeFit")
        self.assertNotRegex(
            body, r"\bk\s*=\s*bw\s*/",
            "drawMulti is back to a width-only scale: a box whose aspect "
            "isn't the layout's then draws the composite at the wrong size, "
            "anchored at the origin, instead of centred at the right one")
        fit = self.src.split("function compositeFit")[1].split("\n  }")[0]
        self.assertIn("Math.min(", fit,
                      "compositeFit no longer takes the smaller of the two "
                      "axes, which is the whole of 'contain'")

    def test_the_pointer_asks_the_same_question_as_the_painter(self):
        # A card is hit-tested where it was DRAWN. Mapping the pointer against
        # the element instead leaves every card off by the centring margin,
        # which is a card that doesn't follow the cursor.
        body = self.src.split("function canvasFrac")[1].split("\n  }")[0]
        self.assertIn("compositeFit(", body)

    def test_the_still_is_not_stretched_to_fit_the_box(self):
        css = _raw("editor.css")
        block = re.search(r"\.ed-still\s*\{(.*?)\}", css, re.S)
        self.assertIsNotNone(block, "editor.css no longer styles .ed-still")
        self.assertNotRegex(
            block.group(1), r"object-fit\s*:\s*fill",
            "the rendered still is stretched into the stage box again -- it "
            "IS the export, so distorting it to fit is what let a wrong box "
            "stay invisible until playback disagreed with it")


class CutAuthoringPins(unittest.TestCase):
    """Select-to-cut: the browser half, which no Python touches.

    Cuts shipped renderer-first — the export, the retime map, the struck
    overlays and the playback skip all worked while the editor could not
    author one, because `editsPayload()` had no `cuts` key and `merge_edits`
    patches on key MEMBERSHIP. That is the same failure `CropSavePins`
    exists for: the panel looks right, the preview looks right, and the edit
    is gone on the next reload.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def test_the_save_payload_carries_the_cuts(self):
        body = self.src.split("function editsPayload")[1].split("\n  }")[0]
        self.assertIn("cuts", body,
                      "editsPayload() is the client's own allowlist -- a key "
                      "missing here is never sent, and merge_edits keeps "
                      "whatever was on disk")

    def test_there_is_exactly_one_cut_authoring_site(self):
        """`proposeCut` is the funnel every future proposer (a word-aligned
        lane, a drag on the clip lane) is meant to use, so the id policy,
        the sort and the write path are decided once. A second `cuts.push`
        is a second policy."""
        pushes = re.findall(r"\bcuts\.push\b", self.src)
        self.assertEqual(len(pushes), 1,
                         "cuts.push belongs only in proposeCut()")
        body = self.src.split("function proposeCut")[1].split("\n  }")[0]
        self.assertIn("cuts.push", body)

    def test_the_word_end_comes_from_the_server(self):
        """whisper collapses roughly half of all word timings to zero length.
        The repair is one rule and it lives in Python (transcribe
        .word_end_times, served as `w.end`); a second copy here is exactly
        the contract drift the cluster-contract invariant warns about, and
        there is no JS harness to pin it against."""
        body = self.src.split("function txWordEnd")[1].split("\n  }")[0]
        self.assertIn("w.end", body)
        span = self.src.split("function txSpan")[1].split("\n  }")[0]
        self.assertIn("txWordEnd", span)
        self.assertNotIn("dur", span,
                         "txSpan must not recompute an end from t + dur")

    def test_authoring_is_gated_on_the_take_type(self):
        """Scene / multi-native / card-layout takes export UN-cut (render
        prints a note). The editor must not offer a verb it cannot keep."""
        body = self.src.split("function txCutSelection")[1].split("\n  }")[0]
        self.assertIn("cutsApplyToThisTake", body)

    def test_the_rejection_branch_is_gated_on_the_code(self):
        """A generic 400 (bad session name, torn body) must not unwind an
        edit the user never connected to the failure."""
        body = self.src.split("async function saveNow")[1].split("\n  }")[0]
        # _read blanks string literals, so the gate is checked structurally
        # here and the literal is checked against the raw file below.
        self.assertIn("e.data.code ===", body)
        self.assertIn("pushUndo()", body)
        with open(os.path.join(WEB, "editor.js")) as fh:
            self.assertIn('code === "cuts"', fh.read())

    def test_a_transcript_drag_holds_off_the_adopt(self):
        """adoptEdits rebuilds the panel, which replaces the very word spans
        a live drag is hit-testing against. BOTH adopt paths have to know:
        saveNow's, and the 3s external-edit poll -- a word drag mutates
        nothing, so the poll is the one that actually fires during it."""
        body = self.src.split("async function saveNow")[1].split("\n  }")[0]
        self.assertIn("S.txDrag", body)
        # The poll is the one that actually fires mid-drag: a word drag
        # mutates nothing, so mutCount == savedMutCount the whole time and
        # the idle watcher is free to adopt right under the pointer.
        poll = self.src.split("setInterval(async function")[1].split("\n    }")[0]
        self.assertEqual(poll.count("S.txDrag"), 2,
                         "both poll guards (entry and post-fetch re-check) "
                         "must hold off a transcript word drag")
        # The 409/400 recovery branches adopt REGARDLESS of any drag, and
        # should: the save already failed, so there is nothing to protect.

    def test_the_list_stops_the_keys_it_claims(self):
        """The transcript list is a focusable DIV, and the document keydown
        handler only ignores INPUT/SELECT/TEXTAREA. Without stopPropagation,
        Delete cuts the speech AND runs the global deleteSelection(),
        destroying whatever zoom or marker was selected on the timeline --
        in a second undo entry the user never connects to what they pressed.
        Space is deliberately NOT claimed, so it stays play/pause."""
        # Raw read: _read blanks string literals, and the handler is
        # identified by one.
        with open(os.path.join(WEB, "editor.js")) as fh:
            raw = fh.read()
        body = raw.split('list.addEventListener("keydown"')[1].split("\n    }")[0]
        self.assertEqual(body.count("ev.stopPropagation()"), 3,
                         "every key the list claims (arrows, Escape, "
                         "Delete/Backspace) must stop propagation")
        self.assertNotIn('=== " "', body)   # Space stays play/pause

    def test_the_list_scroll_survives_a_rebuild(self):
        """renderPanel() rebuilds the list on every mutate, so cutting one
        filler word at 8:00 would otherwise throw the reader back to 0:00 --
        and cutting filler words IS the feature."""
        # Captured off the OUTGOING node in renderPanel, before the wipe --
        # a scroll listener would miss a rebuild landing between two scroll
        # events, and never fires at all while the page is hidden.
        panel = self.src.split("function renderPanel")[1].split("\n  }")[0]
        self.assertIn("S.txScroll", panel)
        self.assertLess(panel.index("S.txScroll"), panel.index("innerHTML"),
                        "the scroll position must be read BEFORE the wipe")
        body = self.src.split("function paintLines")[1].split("\n    }")[0]
        self.assertIn("scrollTop", body)

    def test_restore_subtracts_rather_than_dropping_whole_cuts(self):
        """"Restore to video" promises the selected speech back. Dropping
        every cut that merely OVERLAPS the selection would silently undo an
        agent-authored half-minute because one word of it was selected."""
        body = self.src.split("function restoreRange")[1].split("\n  }")[0]
        self.assertIn("tempId", body)      # the trailing remainder is a new cut
        self.assertNotIn("filter", body)


if __name__ == "__main__":
    unittest.main()


class CursorErasePins(unittest.TestCase):
    """`render.cursor_erase` clears the Python gates for free -- it is a key
    in `_DEFAULT_RENDER`, so `normalize_edits`, `merge_edits` and both preset
    rebuilders carry it, and `editsPayload()` sends `S.edits.render` whole.

    `fullOptions()` is the one hand-written list it still has to appear in.
    That dict is the UNSAVED override the paused still and the camera path
    are rendered with, so a key missing there means flipping the switch
    changes nothing on screen until the next save lands -- which reads as a
    dead control, the same failure `window_layout` shipped with.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def test_the_live_options_dict_carries_it(self):
        body = self.src.split("function fullOptions")[1].split("\n  }")[0]
        self.assertIn("cursor_erase", body)

    def test_the_panel_offers_it(self):
        self.assertIn("e.render.cursor_erase = on", self.src)

    def test_it_is_disabled_on_a_synthetic_cursor_session(self):
        """Nothing is burned into those pixels, so an enabled switch would be
        a control that cannot do anything -- the same reason the synthetic
        cursor switch is disabled on a system-cursor take."""
        # `_raw`, not `_read`: this pin is on a string LITERAL, and _read
        # blanks those.
        body = _raw("editor.js").split("RECORDED CURSOR")[1][:1400]
        self.assertIn("synthetic", body)

    def test_the_two_switches_move_together_on_a_system_take(self):
        """`render._cursor_fx_draws` made the pair reachable: erasing the
        recorded pointer is what leaves a frame the synthetic one may draw
        into. So on a system take the panel moves both -- turning the drawn
        cursor ON turns the eraser on, and turning the eraser OFF clears the
        drawn one. Either half alone is a switch sitting "on" while nothing
        happens, which is the dead-control failure this file exists to catch.
        """
        src = _raw("editor.js")
        recorded = src.split("RECORDED CURSOR")[1].split("SYNTHETIC CURSOR")[0]
        synth = src.split("SYNTHETIC CURSOR")[1][:1600]
        self.assertIn("e.render.cursor_fx = false", recorded)
        self.assertIn("e.render.cursor_erase = true", synth)

    def test_the_drawn_cursor_switch_is_not_dead_on_a_system_take(self):
        """It used to be hard-disabled there ("record with cursor mode
        Synthetic to enable"). It is now the entry point to the retrofit, so
        nothing may disable it on `!synthetic` any more."""
        synth = _raw("editor.js").split("SYNTHETIC CURSOR")[1][:1600]
        self.assertNotIn("}, !synthetic,", synth)


class CropSavePins(unittest.TestCase):
    """The crop has to clear FOUR separate allowlists to reach disk.

    `edits.crop` is not carried by "just save the doc": three of the four
    gates are hand-written key lists, and the fourth lives in the browser.
    A rect that clears the first three and dies in `editsPayload` looks
    exactly like a working feature in the editor -- the panel reports the
    crop, the preview shows it -- right up until the page is reloaded and
    the crop is gone. That is what shipped during development, so it is
    pinned here.

    The three Python gates are covered by `tests/test_crop.py`; this is the
    browser one, which no Python touches.
    """

    def setUp(self):
        self.src = _read("editor.js")

    def test_the_save_payload_carries_the_crop(self):
        body = self.src.split("function editsPayload")[1].split("\n  }")[0]
        self.assertIn("crop", body,
                      "editsPayload() is the client's own allowlist -- a key "
                      "missing here is never sent, and merge_edits keeps "
                      "whatever was on disk")

    def test_reset_sends_an_explicit_null(self):
        """`merge_edits` patches on key MEMBERSHIP, so a payload that drops
        the key when there is no crop would leave the old rect on disk and
        the editor would silently disagree with the export."""
        body = self.src.split("function editsPayload")[1].split("\n  }")[0]
        self.assertIn("null", body)

    def test_the_stage_clip_uses_the_composed_rect(self):
        """Live playback is a browser compositor, not a server render. It has
        to clip the <video> to camera_path's `stage_crop` (capture crop +
        editor crop); `crop` alone leaves playback uncropped while the paused
        still is cropped -- the two-compositor split this editor already has
        a history of."""
        self.assertIn("stage_crop", self.src)
        # The WHOLE function body, via the dedent terminator `editsPayload`
        # above uses -- not a fixed-length prefix. The old `[:900]` window was
        # a proximity guess with no stated meaning, and it broke the moment a
        # legitimate early-return branch (the scene-take live player, which has
        # no ui.video and refuses source-space tools) was inserted above the
        # crop machinery: the pin failed while the invariant it names was
        # untouched. Slicing to the function's real end pins the property
        # itself -- this transform composes stageCrop(), not a bare crop.
        body = self.src.split("function applyTransformAt")[1].split("\n  }")[0]
        self.assertIn("stageCrop()", body)

    def test_the_crop_tool_stands_down_for_the_composite(self):
        """The rect is dragged on the RAW source at identity scale, so the
        multi-window composite has to come down while it is armed -- the same
        reason armWinPick does it."""
        head, _, tail = self.src.partition("if (S.pinArm || S.winPick")
        self.assertTrue(tail, "the identity-scale branch was renamed")
        self.assertIn("S.cropArm", tail[:80],
                      "the crop tool must share the identity-scale branch")
        # Generous window: _read() blanks comments rather than removing them,
        # so the branch's explanatory block eats most of the first few hundred
        # characters here.
        self.assertIn("showMulti(false)", tail[:900])


class JoinTriggerSurface(unittest.TestCase):
    """Seamless window-JOIN trigger (docs/architecture.md milestone 2): the bar chip
    + endpoint wiring. Source pins so the off-switch DOM/handler can't silently
    drop, mirroring how the repick surface is pinned."""

    def test_bar_html_has_the_chip_container(self):
        self.assertIn("bar-grow-chips", _raw("bar.html"))

    def test_bar_js_wires_the_grow_surface(self):
        src = _raw("bar.js")
        for token in ("/api/record/grow", "grow_supported", "new_windows",
                      "bar-grow-chip", "window_id", "renderGrowChips",
                      "growPending"):
            self.assertIn(token, src, token)

    def test_bar_css_styles_the_chip(self):
        self.assertIn("bar-grow-chip", _raw("bar.css"))


class ShrinkTriggerSurface(unittest.TestCase):
    """Card-shrink hints on the recording face (docs/architecture.md M3.4). Source
    pins in the JoinTriggerSurface idiom: the copy says what HAPPENED
    (recording stopped -- 'restore it' is wrong advice for a closed window),
    and both hints dedupe by VALUE (the grow pattern; the durable keys
    re-serve every poll and would re-fire forever without it)."""

    def test_bar_js_wires_the_departed_hint(self):
        src = _raw("bar.js")
        for token in ("rec.departed", "state.departedKey",
                      "Stopped recording",
                      "bring the window back to re-add it",
                      "rec.departed_app"):
            self.assertIn(token, src, token)
        # The dedupe must be the FULL compare-then-assign pair -- token
        # presence alone let a dropped assignment (hint re-fires every
        # poll, the resume_error pattern the scope forbids) survive
        # mutation testing.
        self.assertIn("rec.departed !== state.departedKey", src)
        self.assertIn("state.departedKey = rec.departed", src)

    def test_bar_js_wires_the_mic_freeze_hint(self):
        src = _raw("bar.js")
        for token in ("rec.mic_anchor_hidden", "state.micHideKey",
                      "recording your mic"):
            self.assertIn(token, src, token)
        self.assertIn("rec.mic_anchor_hidden !== state.micHideKey", src)
        self.assertIn("state.micHideKey = rec.mic_anchor_hidden", src)
        # Absence resets the key so the NEXT hide episode (same app name)
        # surfaces after the server's seen-edge clear.
        self.assertIn("state.micHideKey = null", src)

    def test_bar_js_resets_hint_keys_per_take(self):
        # Worker indexes restart at 0 and the anchor app repeats: keys
        # surviving into the next take would swallow its hints.
        src = _raw("bar.js")
        self.assertIn("rec.session !== state.hintSession", src)
        for token in ("state.growErrKey = null",
                      "state.growJoinedKey = null",
                      "state.departedKey = null"):
            self.assertIn(token, src, token)
