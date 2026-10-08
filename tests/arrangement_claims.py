"""Measured arrangement facts, plus a scanner for the copy that describes them.

Not a test module (unittest's `test*.py` pattern skips it). It is imported by
`test_web_sources.ArrangementHintPins` -- the editor's Arrangement hint -- and
by `test_mcp_server` -- the `window_layout` tool description -- because those
two texts describe ONE feature to two audiences and have already contradicted
each other and the code.

The failure this exists to catch, exactly as it happened: `feature` learned to
transpose (hero on top when the canvas is taller than the arrangement wants),
which took it from 26.6% of a 9:16 canvas to 67.1% -- joint-best of the five.
The copy was written in parallel from the PRE-transpose measurements, so the
editor went on telling users Feature was "a waste of a tall" export and the MCP
description went on ranking `desktop` first on 16:10 when `feature` had passed
it. Nothing raised. Every arrangement test stayed green, because none of them
reads the prose.

So: nothing here is a remembered table. Every figure is measured off
`autocine.framing` at test time, and the prose is checked against THAT. A copy pin
that carried its own numbers would be one more copy to go stale.

What this can and cannot catch is written on `dismissals` -- read it before
trusting a green run.
"""

import re

from autocine import framing

# The names `edits._WINDOW_LAYOUTS` accepts. Spelled out rather than imported
# because the scanners below need them as *words to look for in prose*, and a
# name that lands in edits.py without ever being written about is exactly the
# thing `test_the_hint_speaks_to_every_arrangement` already fails on.
LAYOUTS = ("grid", "desktop", "feature", "row", "column")

# The three windows the arrangements were built for, at the origins
# `test_framing.InkCoverage` pins -- tall, wide, wide, arranged as a real
# desktop had them. The copy's numbers are quoted from this set, so a check
# against any other one would compare against a different feature.
REAL_WINDOWS = [{"x": 0, "y": 0, "w": 1418, "h": 1718},
                {"x": 1462, "y": 0, "w": 1418, "h": 852},
                {"x": 1462, "y": 900, "w": 1418, "h": 854}]

# 16:10 Retina, 16:9, a 9:16 export, and square.
CANVASES = ((2880, 1800), (1920, 1080), (1080, 1920), (1440, 1440))

# Which canvases a sentence is talking about when it says "wide" or "tall".
# 1:1 belongs to neither on purpose: a square canvas is what the copy reaches
# for neither word about, and forcing it into one bucket would let a true
# statement about landscape be judged against it.
SHAPE_CANVASES = {"wide": ((2880, 1800), (1920, 1080)),
                  "tall": ((1080, 1920),)}

# A percentage in the prose is allowed to be the measured value rounded to a
# whole point (0.5) plus a hair, and no more. Wider than that and "desktop
# covers 70%" -- the pre-transpose figure, 2.9 points off -- reads as honest
# rounding.
PCT_TOL = 0.55

# How far off the best a layout has to measure before the copy is allowed to
# write it off for a canvas shape. Generous on purpose: this pin exists to
# catch a claim the code REVERSES (feature was called a waste of a tall export
# while measuring 0.2 points off the best one), not to arbitrate between two
# layouts that are genuinely close.
DISMISSAL_TOL = 5.0


def placements(layout, W, H, rects=None):
    """`MultiFramePainter.__init__`'s placement pass, without the shadow bake.

    The pad/gap derivation is copied from the painter, so
    `test_web_sources` pins this against a real painter -- otherwise every
    number below could be measuring an arrangement nothing renders.
    """
    rects = REAL_WINDOWS if rects is None else rects
    pad = int(framing.PAD_FRAC * W)
    gap = max(0, int(0.02 * W))
    fn = framing._ARRANGEMENTS.get(layout)
    cells = fn(W, H, rects, pad, gap) if fn is not None else None
    if cells is None:
        cells = framing._grid_placements(W, H, rects, pad, gap)
    return framing._apply_card_overrides(W, H, cells, rects)


def coverage(layout, W, H, rects=None):
    """Percentage of the canvas this layout's cards actually cover.

    Summing areas is exact rather than an approximation of a union: no
    arrangement overlaps its cards (pinned in `test_framing`).
    """
    cells = placements(layout, W, H, rects)
    return 100.0 * sum(c[2] * c[3] for c in cells) / float(W * H)


def measured_values(layout):
    """Every coverage figure the copy could legitimately be quoting for one
    layout: the four canvases with all three windows, and -- since two windows
    rank differently from three and the copy says so -- the two canvases it
    quotes a pair on."""
    out = [coverage(layout, W, H) for W, H in CANVASES]
    pair = REAL_WINDOWS[:2]
    out += [coverage(layout, W, H, pair) for W, H in ((2880, 1800), (1080, 1920))]
    return out


def gap_from_best(layout, shape):
    """How far this layout falls short of the best one, on the canvas of that
    shape where it does BEST.

    The minimum rather than the maximum because this decides whether a
    dismissal is a lie: one canvas of that shape where the layout is at the
    top is enough to make "no good for a tall export" untrue.
    """
    gaps = []
    for W, H in SHAPE_CANVASES[shape]:
        best = max(coverage(name, W, H) for name in LAYOUTS)
        gaps.append(best - coverage(layout, W, H))
    return min(gaps)


_NAME_RE = re.compile(r"\b(%s)\b" % "|".join(LAYOUTS), re.I)
_PCT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
# Only unambiguous write-offs. A word that merely ranks ("row wants the widest
# export of the five") is not one: it is true of a layout that is also 38
# points off the best there, and treating it as a dismissal would fail honest
# copy.
_AGAINST_RE = re.compile(
    r"\b(wastes?|wasteful|wasted|poorly|poor|weakest|weaker|weak|worst"
    r"|avoid|never|pointless|useless|struggles?|hopeless"
    r"|no\s+good|not\s+for|wrong\s+for|falls\s+apart)\b", re.I)
_SHAPE_RE = re.compile(
    r"\b(wide|wider|widest|landscape|16:10|16:9"
    r"|tall|taller|tallest|vertical|portrait|9:16)\b", re.I)
_SHAPE_OF = {"wide": "wide", "wider": "wide", "widest": "wide",
             "landscape": "wide", "16:10": "wide", "16:9": "wide",
             "tall": "tall", "taller": "tall", "tallest": "tall",
             "vertical": "tall", "portrait": "tall", "9:16": "tall"}


def _sentences(text):
    return re.split(r"(?<=[.;])\s+", text)


def percent_claims(text):
    """Every coverage percentage the prose asserts, as `(layouts, percent)`.

    `layouts` is every arrangement the sentence names, not one -- and that is
    deliberate, not laziness. Which layout a number belongs to is genuinely
    beyond a regex once a sentence compares two ("Grid ... matches Feature on
    a tall canvas (67.4%)" is Grid's figure, and the nearest name is
    Feature), and a scanner that guessed would fail honest copy, which is how
    a pin gets deleted. Checking the number against every candidate still
    catches the thing that actually shipped: "desktop 70%" and "desktop 27%"
    are not coverages of ANY of the five, on any canvas, with two windows or
    three.

    A sentence carrying numbers and no name inherits the names of the last
    one that had any, because that is how the copy reads: "Feature runs window
    1 large... Turning is why it holds up at any shape: it covers 67.7%...".
    """
    out = []
    carried = ()
    for sentence in _sentences(text):
        names = tuple(sorted(set(m.group(1).lower()
                                 for m in _NAME_RE.finditer(sentence))))
        if names:
            carried = names
        for m in _PCT_RE.finditer(sentence):
            if carried:
                out.append((carried, float(m.group(1))))
    return out


def dismissals(text):
    """Every `(layout, shape, sentence)` the prose writes off.

    HONEST SCOPE, because a pin that overstates its reach is the same defect
    as the copy it guards: this finds a layout being told it is BAD for a
    canvas shape, in the same sentence, in so many words. It does not find
    steering by omission -- "column or grid for a vertical export", written
    when feature is joint-best there, names nothing negative and passes. That
    one needs a human reading the ranking, or a pin on the sentence itself,
    and pinning a sentence is what put four contradictory descriptions in the
    tree in the first place.
    """
    out = []
    for sentence in _sentences(text):
        for cue in _AGAINST_RE.finditer(sentence):
            # "'column' is poor on a wide export" puts the name first;
            # "never pick 'feature' for a vertical one" puts the cue first.
            # Nearest name wins, preferring the one already introduced.
            before = [m for m in _NAME_RE.finditer(sentence)
                      if m.end() <= cue.start()]
            if not before:
                before = [m for m in _NAME_RE.finditer(sentence)
                          if m.start() >= cue.end()][:1]
            if not before:
                continue
            # "...for a wide export, a waste of a tall one": the shape the
            # write-off is ABOUT is the one after it, not the one it is being
            # contrasted with. Fall back to the nearest earlier shape when the
            # sentence puts it the other way round.
            after = [m for m in _SHAPE_RE.finditer(sentence)
                     if m.start() >= cue.end()]
            if after:
                word = after[0].group(1)
            else:
                before_shape = [m for m in _SHAPE_RE.finditer(sentence)
                                if m.end() <= cue.start()]
                if not before_shape:
                    continue
                word = before_shape[-1].group(1)
            out.append((before[-1].group(1).lower(), _SHAPE_OF[word.lower()],
                        sentence.strip()))
    return out
