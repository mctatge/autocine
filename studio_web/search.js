/* search.js — AutoCineSearch: find a recording by its name or by what was said.
 *
 * PURE: no DOM, no fetches, no globals beyond the one it exports, so it runs
 * under node for tests:
 *
 *   const S = require("./studio_web/search.js");
 *   const idx = S.index([{ id: "20260818-223714", name: "Aug 18, 10:37 PM",
 *                          segments: [{ t: 12.5, dur: 3, text: "..." }] }]);
 *   S.search("place order", idx);
 *
 * The library (library.js) owns the UI and the transcript fetches; this file
 * owns matching, ranking and safe highlighting.
 *
 * Ported from the landing page's search (site/main.js) — same tokenizer,
 * stopwords, light stemming, prefix match for the word being typed and
 * unicode folding — with three differences the real library needs:
 *   1. Every project has a NAME entry of its own, so a name matches before
 *      (or without) any transcript. The landing's projects always had
 *      segments; here most recordings have none.
 *   2. Results say what they are: kind "name" (opens the project) or
 *      "moment" (opens it at `t`).
 *   3. Words are indexed once per project (`indexProject`), so the library
 *      re-indexes only the transcript that just arrived, not the whole
 *      library on every keystroke.
 */
(function (root) {
  "use strict";

  // The landing's list, plus the contracted forms of those same words
  // ("it's" is tokenized as "its"), so "where's the part about it's" stays empty.
  var STOPWORDS = Object.create(null); // no prototype: "constructor" is a word
  ("the a an i where part about when what that it my of to in on at for and " +
   "is was did do we you me this how " +
   "its thats whats wheres whens hows im id ive ill youre youve were weve lets")
    .split(" ").forEach(function (w) { STOPWORDS[w] = true; });

  var MAX_RESULTS = 8;
  var PER_PROJECT = 4;       // first pass: at most this many rows per project
  var NAME_WEIGHT = 0.6;     // a name word inside a MOMENT counts less than one that was said
  var NAME_ENTRY_WEIGHT = 1; // ...but a project's own name row is a full-strength match
  var PHRASE_BONUS = 0.2;    // query words found side by side ("place order")
  var COVERAGE_KEEP = 0.6;   // drop results covering much less of the query than the best

  // A word is letters/digits in any script (combining marks and invisible
  // format characters stay inside it); apostrophes inside it are dropped:
  // "here's" -> "heres", "won't" -> "wont" (never a stray "s" or "t").
  // Words are folded for comparison: lower case, compatibility forms and
  // accents removed ("Naïve" -> "naive", fullwidth "ｄｅｍｏ" -> "demo").
  // Folding happens per word, so match positions stay those of the
  // original text (the highlighter needs them).
  var WORD_RE = /[\p{L}\p{N}][\p{L}\p{N}\p{M}\p{Cf}]*(?:['’][\p{L}\p{N}][\p{L}\p{N}\p{M}\p{Cf}]*)*/gu;
  var NONWORD_RE = /[^\p{L}\p{N}]+/gu;
  var TYPING_DONE_RE = /[\s.,!?;:"”)]$/;   // the query ends in a finished word
  // Scripts written without spaces between words (Chinese, Japanese, Thai,
  // Lao, Khmer, Myanmar): a whole run is ONE "word" here, so a query word in
  // one of them also matches INSIDE a word ("演示" in "我们今天演示录屏功能").
  var NOSPACE_RE = /[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}\p{Script=Thai}\p{Script=Lao}\p{Script=Khmer}\p{Script=Myanmar}]/u;

  function norm(w) {
    w = String(w).toLowerCase();
    if (w.normalize) w = w.normalize("NFKD");
    return w.replace(NONWORD_RE, "");
  }

  // Light stemming: clicked/clicking/clicks -> click, pricing/price -> pric,
  // codes/code -> cod. Both sides of every comparison go through it, so the
  // stems only have to agree with each other, not be real words. Where they
  // don't -- a root that itself ends in -ed/-ing ("speed" -> "spe" but
  // "speeds" -> "speed"; "string" -> "str") -- `strength` also compares each
  // side's stem with the other's raw word.
  function stem(word) {
    var w = String(word).toLowerCase();
    if (w.length <= 3 || /^\d+$/.test(w)) return w;
    var cut = false;
    if (w.length > 5 && /ing$/.test(w)) { w = w.slice(0, -3); cut = true; }
    else if (w.length > 4 && /ed$/.test(w)) { w = w.slice(0, -2); cut = true; }
    else if (w.length > 4 && /es$/.test(w)) { w = w.slice(0, -2); }
    else if (/[^s]s$/.test(w)) { w = w.slice(0, -1); }
    // shipping -> shipp -> ship (but keep scroll, miss, buzz)
    if (cut && /([b-df-hj-np-tv-z])\1$/.test(w) && !/(ll|ss|zz)$/.test(w)) w = w.slice(0, -1);
    if (w.length > 3 && /e$/.test(w)) w = w.slice(0, -1);
    return w;
  }

  // Words of a piece of text, with their positions (for highlighting).
  function words(text) {
    var out = [], m;
    text = String(text == null ? "" : text);
    WORD_RE.lastIndex = 0;
    while ((m = WORD_RE.exec(text))) {
      var raw = norm(m[0]);
      if (raw.length < 2 && !/\d/.test(raw)) continue; // "a", "I"
      out.push({ raw: raw, stem: stem(raw), start: m.index, end: m.index + m[0].length });
    }
    return out;
  }

  // Query -> tokens. The LAST token counts as still being typed (prefix
  // match) unless the query ends in a space or punctuation. `keepStopwords`
  // is the names-only fallback (see `search`): a query made ONLY of
  // stopwords ("The Part About Me") still has words to look up in names.
  function tokenize(query, keepStopwords) {
    var q = String(query == null ? "" : query).toLowerCase();
    var typing = q.length > 0 && !TYPING_DONE_RE.test(q);
    WORD_RE.lastIndex = 0;
    var raw = q.match(WORD_RE) || [];
    var out = [], seen = Object.create(null);
    for (var i = 0; i < raw.length; i++) {
      var w = norm(raw[i]), last = typing && i === raw.length - 1;
      if (!w || (STOPWORDS[w] && !keepStopwords)) continue;
      if (w.length < 2 && !/\d/.test(w) && (!last || keepStopwords)) continue;
      var s = stem(w);
      if (seen[s]) continue;
      seen[s] = true;
      out.push({ raw: w, stem: s, typing: last, prefix: last || s.length >= 4,
                 nospace: NOSPACE_RE.test(w) });
    }
    return out;
  }

  // Tokens a query is searched with: its real words, or -- when every word
  // in it is a stopword -- all of its words, flagged `namesOnly`.
  function queryTokens(query) {
    var qts = tokenize(query);
    if (qts.length) return { qts: qts, namesOnly: false };
    return { qts: tokenize(query, true), namesOnly: true };
  }

  // True when a query has anything to search for (the UI's "is the box
  // empty?" -- a stopword-only query like "what i did" still counts).
  function hasTerms(query) {
    return queryTokens(query).qts.length > 0;
  }

  // Real words (3+ letters, a number, or the word being typed). When a
  // query has one, a result must match one: "export as CSV" should not
  // surface a segment whose only hit is "as".
  function isStrong(qt) {
    return qt.raw.length >= 3 || qt.nospace || /\d/.test(qt.raw) || qt.typing;
  }

  function strength(qt, w) {
    if (w.stem === qt.stem || w.raw === qt.raw) return 1;
    // a root ending in -ed/-ing: "speed" vs "speeds"/"speeding" (stem
    // "speed"), "embed" vs "embedded", "string" vs "strings"
    if (w.stem === qt.raw || w.raw === qt.stem) return 1;
    // no spaces between words in this script: a match can sit mid-run
    if (qt.nospace && w.raw.indexOf(qt.raw) >= 0) return 1;
    if (qt.prefix && (w.raw.indexOf(qt.raw) === 0 || w.stem.indexOf(qt.stem) === 0)) {
      return qt.typing ? 0.8 : 0.5;
    }
    return 0;
  }

  function best(qt, ws) {
    var b = 0, at = -1;
    for (var i = 0; i < ws.length && b < 1; i++) {
      var s = strength(qt, ws[i]);
      if (s > b) { b = s; at = i; }
    }
    return { s: b, at: at };
  }

  function finiteNonNeg(x) {
    var n = Number(x);
    return isFinite(n) && n > 0 ? n : 0;
  }

  /**
   * indexProject(project, order) -> entries for ONE project:
   *   one "name" entry, plus one "moment" entry per transcript segment.
   * project = { id, name, segments?: [{ t, dur, text }] }. Malformed input
   * (a null project or segment, segments that are not an array, a missing
   * name) is skipped, never thrown on: transcripts come off disk.
   */
  function indexProject(p, order) {
    if (!p || typeof p !== "object") return [];
    var name = p.name == null ? "" : String(p.name);
    var nameWords = words(name);
    var pi = isFinite(Number(order)) ? Number(order) : 0;
    var entries = [{ p: p, pi: pi, kind: "name", t: 0, dur: 0, text: "", words: [], nameWords: nameWords }];
    var segs = Array.isArray(p.segments) ? p.segments : [];
    segs.forEach(function (seg) {
      if (!seg || typeof seg !== "object") return;
      var text = seg.text == null ? "" : String(seg.text);
      var ws = words(text);
      if (!ws.length) return;
      entries.push({
        p: p, pi: pi, kind: "moment", t: finiteNonNeg(seg.t), dur: finiteNonNeg(seg.dur),
        text: text, words: ws, nameWords: nameWords
      });
    });
    return entries;
  }

  // index(projects) -> { entries }. Order in the array is the tie-break.
  function index(projects) {
    var entries = [];
    (Array.isArray(projects) ? projects : []).forEach(function (p, i) {
      indexProject(p, i).forEach(function (e) { entries.push(e); });
    });
    return { entries: entries };
  }

  function entriesFrom(idx) {
    if (!idx) return [];
    if (Array.isArray(idx.entries)) return idx.entries;
    if (Array.isArray(idx.projects)) return index(idx.projects).entries;
    if (Array.isArray(idx)) return index(idx).entries;
    return [];
  }

  function idOf(p) { return p.id == null ? "" : String(p.id); }

  function order(a, b) {
    return (b.score - a.score) ||
      ((a.e.kind === "name" ? 0 : 1) - (b.e.kind === "name" ? 0 : 1)) ||
      (a.e.pi - b.e.pi) ||
      (a.e.t - b.e.t);
  }

  /**
   * search(query, idx, opts?) ->
   *   [{ projectId, projectName, kind: "name"|"moment", t, dur, text, score }]
   * Best first, at most opts.limit (default 8). `idx` is index(...) output
   * (or { projects } / a projects array, indexed on the spot).
   *
   * Scores are 0..~1.2: the idf-weighted share of the query's words found
   * (in what was said, or in the project's name), plus a bonus when query
   * words sit side by side in what was said. A moment's words and its
   * project's name combine ("demo zoom" finds "zoom" said in the Demo take);
   * a moment matched ONLY by its project's name is dropped, because the
   * project's own name row already stands for it.
   *
   * A query made only of stopwords ("the part about me", "what I did") says
   * nothing about a moment, but it can still be a project's NAME: it is then
   * looked up in names only, and a name must contain every one of its words.
   */
  function search(query, idx, opts) {
    var limit = opts && opts.limit > 0 ? Math.floor(opts.limit) : MAX_RESULTS;
    var tq = queryTokens(query), qts = tq.qts;
    if (!qts.length) return [];
    var entries = entriesFrom(idx);
    if (tq.namesOnly) {
      entries = entries.filter(function (e) {
        return e.kind === "name" && qts.every(function (qt) { return best(qt, e.nameWords).s > 0; });
      });
    }
    if (!entries.length) return [];

    // Document frequency: a segment counts once for what was said in it, a
    // project once for its name (not once per segment it owns, which would
    // make any word in a long take's name look common).
    var N = 0;
    entries.forEach(function (e) { if (e.kind === "name" || e.words.length) N++; });
    N = N || 1;
    var weights = qts.map(function (qt) {
      var df = 0;
      entries.forEach(function (e) {
        var ws = e.kind === "name" ? e.nameWords : e.words;
        if (best(qt, ws).s > 0) df++;
      });
      return Math.log(1 + N / (1 + df));
    });
    var den = weights.reduce(function (a, b) { return a + b; }, 0) || 1;

    // a result must match a real word when the query has one (isStrong)
    var needStrong = qts.some(isStrong);

    var hits = [];
    entries.forEach(function (e) {
      var num = 0, cov = 0, textHit = false, strongHit = false, at = [];
      qts.forEach(function (qt, k) {
        var ts = 0, n;
        if (e.kind === "name") {
          n = best(qt, e.nameWords).s * NAME_ENTRY_WEIGHT;
          at.push(-1);
        } else {
          var t = best(qt, e.words);
          ts = t.s;
          n = best(qt, e.nameWords).s * NAME_WEIGHT;
          if (ts > 0) textHit = true;
          at.push(ts > 0 ? t.at : -1);
        }
        if ((ts > 0 || n > 0) && isStrong(qt)) strongHit = true;
        if (ts > 0 || n > 0) cov += weights[k];
        num += weights[k] * Math.max(ts, n);
      });
      if (num <= 0 || (needStrong && !strongHit)) return;
      if (e.kind === "moment" && !textHit) return;   // the name row stands for it
      var score = num / den;
      if (at.length > 1) {
        var adj = 0;
        for (var k = 1; k < at.length; k++) if (at[k - 1] >= 0 && at[k] === at[k - 1] + 1) adj++;
        score += PHRASE_BONUS * adj / (at.length - 1);
      }
      hits.push({ e: e, score: score, cov: cov / den });
    });

    // Precision: when one result covers the whole query, results that only
    // share its commonest word are noise, not answers.
    var bestCov = 0;
    hits.forEach(function (h) { if (h.cov > bestCov) bestCov = h.cov; });
    hits = hits.filter(function (h) { return h.cov >= COVERAGE_KEEP * bestCov - 1e-9; });
    hits.sort(order);

    // Variety: one long take full of the word must not push every other
    // project off the list. First pass caps rows per project; leftovers
    // fill any room that remains (a single-project answer still shows all).
    var out = [], rest = [], per = Object.create(null);
    hits.forEach(function (h) {
      var id = idOf(h.e.p), c = per[id] || 0;
      if (out.length < limit && c < PER_PROJECT) { out.push(h); per[id] = c + 1; }
      else rest.push(h);
    });
    for (var i = 0; out.length < limit && i < rest.length; i++) out.push(rest[i]);
    out.sort(order);

    return out.map(function (h) {
      return {
        projectId: idOf(h.e.p),
        projectName: h.e.p.name == null ? "" : String(h.e.p.name),
        kind: h.e.kind,
        t: h.e.t,
        dur: h.e.dur,
        text: h.e.text,
        score: Math.round(h.score * 1000) / 1000
      };
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // [start, end, strong] per matched word. A no-space-script match covers
  // just the matched characters, not the whole run it sits in.
  function hitRanges(text, query) {
    var qts = queryTokens(query).qts, ranges = [];
    if (!qts.length) return ranges;
    words(text).forEach(function (w) {
      for (var i = 0; i < qts.length; i++) {
        var qt = qts[i];
        if (strength(qt, w) <= 0) continue;
        var start = w.start, end = w.end;
        if (qt.nospace && w.raw !== qt.raw) {
          var at = text.slice(w.start, w.end).toLowerCase().indexOf(qt.raw);
          if (at >= 0) { start = w.start + at; end = start + qt.raw.length; }
        }
        ranges.push([start, end, isStrong(qt)]);
        return;
      }
    });
    return ranges;
  }

  /**
   * highlightParts(text, query) -> [{ text, hit }] covering `text` exactly.
   * The safe form: the caller puts each part in a text node (a <mark> for
   * hits), so nothing in a recording's name or transcript can become markup.
   * Neighbouring hits separated only by spaces share one part ("place order").
   */
  function highlightParts(text, query) {
    text = String(text == null ? "" : text);
    var merged = [];
    hitRanges(text, query).forEach(function (r) {
      var prev = merged[merged.length - 1];
      if (prev && /^\s+$/.test(text.slice(prev[1], r[0]))) prev[1] = r[1];
      else merged.push([r[0], r[1]]);
    });
    var out = [], last = 0;
    merged.forEach(function (r) {
      if (r[0] > last) out.push({ text: text.slice(last, r[0]), hit: false });
      out.push({ text: text.slice(r[0], r[1]), hit: true });
      last = r[1];
    });
    if (last < text.length || !out.length) out.push({ text: text.slice(last), hit: false });
    return out;
  }

  // HTML-escaped text with the matched words wrapped in <mark> (parity with
  // the landing page; the library itself builds nodes from highlightParts).
  function highlight(text, query) {
    return highlightParts(text, query).map(function (part) {
      return part.hit ? "<mark>" + escapeHtml(part.text) + "</mark>" : escapeHtml(part.text);
    }).join("");
  }

  // A window of `text` starting a few words before the first hit:
  //   "...then click Place order." for "clicked place order"
  // Anchored on the first hit of a REAL word (isStrong) when there is one:
  // a weak two-letter hit ("as" in "export as csv") near the start of a
  // long segment would otherwise push the words that matched out of view.
  function excerpt(text, query, max) {
    text = String(text == null ? "" : text).replace(/\s+/g, " ").trim();
    max = max > 0 ? max : 90;
    var r = hitRanges(text, query), strong = r.filter(function (x) { return x[2]; });
    if (strong.length) r = strong;
    var start = r.length ? r[0][0] : 0;
    if (start > 0) {
      // a little lead-in, cut at a word boundary
      var lead = Math.max(0, start - 24), sp = text.indexOf(" ", lead);
      start = lead === 0 ? 0 : (sp >= 0 && sp < start ? sp + 1 : start);
    }
    var s = text.slice(start);
    if (s.length > max) {
      var cut = s.lastIndexOf(" ", max);
      s = s.slice(0, cut > max * 0.5 ? cut : max).replace(/[\s,;:]+$/, "") + "…";
    }
    return (start > 0 ? "…" : "") + s;
  }

  // True while the word being typed could still become a stopword ("whe"
  // on the way to "where"): the UI waits instead of saying "no results".
  function typingStopword(query) {
    var q = String(query == null ? "" : query).toLowerCase();
    if (!q || TYPING_DONE_RE.test(q)) return false;
    WORD_RE.lastIndex = 0;
    var raw = q.match(WORD_RE) || [];
    var last = raw.length ? norm(raw[raw.length - 1]) : "";
    if (!last) return false;
    for (var w in STOPWORDS) if (w.length > last.length && w.indexOf(last) === 0) return true;
    return false;
  }

  // m:ss (minutes run past 59, like the library's duration chips: "64:06")
  function timecode(t) {
    t = Math.max(0, Math.floor(Number(t) || 0));
    var m = Math.floor(t / 60), s = t % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  var api = {
    search: search,
    index: index,
    indexProject: indexProject,
    highlightParts: highlightParts,
    highlight: highlight,
    excerpt: excerpt,
    timecode: timecode,
    tokenize: tokenize,
    hasTerms: hasTerms,
    typingStopword: typingStopword,
    stem: stem,
    escapeHtml: escapeHtml
  };
  root.AutoCineSearch = api;
  if (typeof module === "object" && module && module.exports) module.exports = api;
})(typeof window !== "undefined" ? window : (typeof globalThis !== "undefined" ? globalThis : this));
