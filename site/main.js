/* main.js - AutoCine landing hero.
 *
 * Two independent halves:
 *   1. window.AutoCineSearch: a pure, DOM-free search over AUTOCINE_DEMO.
 *      It also runs under node, for tests:
 *        global.window = {};
 *        require('./demo-library.js');
 *        const S = require('./main.js');
 *        S.search('promo code', window.AUTOCINE_DEMO);
 *   2. The hero (skipped when there is no document): a sample take playing
 *      behind the copy on the featured project's clock, the virtual camera
 *      that frames it, the real-video takeover, the prompt-bar combobox, the
 *      typewriter placeholder and the pause control.
 *
 * URL params: ?shot=1 freezes all motion on one composed frame (for
 * screenshots); ?q=<text> prefills the search and opens its results.
 */

/* =====================================================================
   1. Search (pure: no DOM, also runs under node for the tests)
   ===================================================================== */
(function (root) {
  'use strict';

  // The required list, plus the contracted forms of those same words
  // ("it's" is tokenized as "its"), so "where's the part about it's" stays empty.
  var STOPWORDS = Object.create(null); // no prototype: "constructor" is a word
  ('the a an i where part about when what that it my of to in on at for and ' +
   'is was did do we you me this how ' +
   'its thats whats wheres whens hows im id ive ill youre youve were weve lets')
    .split(' ').forEach(function (w) { STOPWORDS[w] = true; });

  var MAX_RESULTS = 6;
  var NAME_WEIGHT = 0.6;   // a hit in a project's NAME counts less than one in what was said
  var PHRASE_BONUS = 0.2;  // query words found side by side ("place order")
  var COVERAGE_KEEP = 0.6; // drop results that cover much less of the query than the best one

  // A word is letters/digits in any script (combining marks and invisible
  // format characters stay inside it), apostrophes inside it are dropped:
  // "here's" -> "heres", "won't" -> "wont" (never a stray "s" or "t").
  // Words are folded for comparison: lower case, compatibility forms and
  // accents removed ("Naïve" -> "naive", fullwidth "ｐｒｏｍｏ" -> "promo").
  // Folding happens per word, so match positions stay those of the
  // original text (the highlighter needs them). An engine without \p{...}
  // support falls back to ASCII words instead of failing to parse.
  var WORD_RE, NONWORD_RE;
  try {
    WORD_RE = new RegExp("[\\p{L}\\p{N}][\\p{L}\\p{N}\\p{M}\\p{Cf}]*(?:['’][\\p{L}\\p{N}][\\p{L}\\p{N}\\p{M}\\p{Cf}]*)*", 'gu');
    NONWORD_RE = new RegExp('[^\\p{L}\\p{N}]+', 'gu');
  } catch (e) {
    WORD_RE = /[a-z0-9]+(?:['’][a-z0-9]+)*/gi;
    NONWORD_RE = /[^a-z0-9]+/g;
  }
  function norm(w) {
    w = String(w).toLowerCase();
    if (w.normalize) w = w.normalize('NFKD');
    return w.replace(NONWORD_RE, '');
  }

  // Light stemming: clicked/clicking/clicks -> click, pricing/price -> pric,
  // codes/code -> cod. Both sides of every comparison go through it, so the
  // stems only have to agree with each other, not be real words.
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
    text = String(text == null ? '' : text);
    WORD_RE.lastIndex = 0;
    while ((m = WORD_RE.exec(text))) {
      var raw = norm(m[0]);
      if (raw.length < 2 && !/\d/.test(raw)) continue; // "a", "I"
      out.push({ raw: raw, stem: stem(raw), start: m.index, end: m.index + m[0].length });
    }
    return out;
  }

  // Query -> tokens. The LAST token counts as still being typed (prefix
  // match) unless the query ends in a space or punctuation.
  function tokenize(query) {
    var q = String(query == null ? '' : query).toLowerCase();
    var typing = q.length > 0 && !/[\s.,!?;:"”)]$/.test(q);
    var raw = q.match(WORD_RE) || [];
    var out = [], seen = Object.create(null);
    for (var i = 0; i < raw.length; i++) {
      var w = norm(raw[i]), last = typing && i === raw.length - 1;
      if (!w || STOPWORDS[w]) continue;
      if (w.length < 2 && !/\d/.test(w) && !last) continue;
      var s = stem(w);
      if (seen[s]) continue;
      seen[s] = true;
      out.push({ raw: w, stem: s, typing: last, prefix: last || s.length >= 4 });
    }
    return out;
  }

  function strength(qt, w) {
    if (w.stem === qt.stem || w.raw === qt.raw) return 1;
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

  // The index, rebuilt on every call: the library is a few dozen segments,
  // and a rebuild can never serve stale words after the data is edited in
  // place. Malformed entries (a null project or segment, segments that are
  // not an array) are skipped, never thrown on: the featured transcript
  // gets pasted in by hand.
  function entriesOf(data) {
    var entries = [];
    var projects = data && Array.isArray(data.projects) ? data.projects : [];
    projects.forEach(function (p, pi) {
      if (!p || typeof p !== 'object') return;
      var nameWords = words(p.name);
      var segs = Array.isArray(p.segments) ? p.segments : [];
      segs.forEach(function (seg) {
        if (!seg || typeof seg !== 'object') return;
        var t = Number(seg.t);
        entries.push({
          p: p, pi: pi, t: isFinite(t) && t > 0 ? t : 0,
          text: seg.text == null ? '' : String(seg.text),
          words: words(seg.text), nameWords: nameWords
        });
      });
    });
    return entries;
  }

  /**
   * search(query, data) -> [{ projectId, projectName, t, text, score, isFeatured }]
   * Best first, at most 6. Scores are 0..~1.2: the idf-weighted share of the
   * query's words found (in what was said, or at 0.6 in the project name),
   * plus a bonus when query words sit side by side in the segment.
   */
  function search(query, data) {
    data = data || root.AUTOCINE_DEMO;
    if (!data || !Array.isArray(data.projects)) return [];
    var qts = tokenize(query);
    if (!qts.length) return [];

    var entries = entriesOf(data);
    var N = entries.length || 1;
    var weights = qts.map(function (qt) {
      var df = 0;
      entries.forEach(function (e) {
        if (best(qt, e.words).s > 0 || best(qt, e.nameWords).s > 0) df++;
      });
      return Math.log(1 + N / (1 + df));
    });
    var den = weights.reduce(function (a, b) { return a + b; }, 0) || 1;

    // When the query has a real word in it (3+ letters, or a number), a
    // result must match one: "export as CSV" should not surface a segment
    // whose only hit is "as".
    var isStrong = function (qt) { return qt.raw.length >= 3 || /\d/.test(qt.raw) || qt.typing; };
    var needStrong = qts.some(isStrong);

    var hits = [];
    entries.forEach(function (e) {
      var num = 0, cov = 0, textHit = false, strongHit = false, at = [];
      qts.forEach(function (qt, k) {
        var t = best(qt, e.words);
        var n = best(qt, e.nameWords).s * NAME_WEIGHT;
        if (t.s > 0) textHit = true;
        if ((t.s > 0 || n > 0) && isStrong(qt)) strongHit = true;
        if (t.s > 0 || n > 0) cov += weights[k];
        num += weights[k] * Math.max(t.s, n);
        at.push(t.s > 0 ? t.at : -1);
      });
      if (num <= 0 || (needStrong && !strongHit)) return;
      var score = num / den;
      if (at.length > 1) {
        var adj = 0;
        for (var k = 1; k < at.length; k++) if (at[k - 1] >= 0 && at[k] === at[k - 1] + 1) adj++;
        score += PHRASE_BONUS * adj / (at.length - 1);
      }
      hits.push({ e: e, score: score, cov: cov / den, textHit: textHit });
    });

    // Precision: when one moment covers the whole query, moments that only
    // share its commonest word ("click") are noise, not results.
    var bestCov = 0;
    hits.forEach(function (h) { if (h.cov > bestCov) bestCov = h.cov; });
    hits = hits.filter(function (h) { return h.cov >= COVERAGE_KEEP * bestCov - 1e-9; });

    // A project matched only by its NAME shows once, by its opening segment;
    // if any of its segments matched by what was said, those stand for it.
    var spoken = {}, named = {};
    hits.forEach(function (h) { if (h.textHit) spoken[h.e.p.id] = true; });
    hits = hits.filter(function (h) {
      if (h.textHit) return true;
      if (spoken[h.e.p.id] || named[h.e.p.id]) return false;
      named[h.e.p.id] = true;
      return true;
    });

    var featured = data.featured;
    hits.sort(function (a, b) {
      return (b.score - a.score) ||
        ((b.e.p.id === featured ? 1 : 0) - (a.e.p.id === featured ? 1 : 0)) ||
        (a.e.pi - b.e.pi) ||
        (a.e.t - b.e.t);
    });

    return hits.slice(0, MAX_RESULTS).map(function (h) {
      return {
        projectId: h.e.p.id,
        projectName: h.e.p.name == null ? '' : String(h.e.p.name),
        t: h.e.t,
        text: h.e.text,
        score: Math.round(h.score * 1000) / 1000,
        isFeatured: h.e.p.id === featured
      };
    });
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  function hitRanges(text, query) {
    var qts = tokenize(query), ranges = [];
    if (!qts.length) return ranges;
    words(text).forEach(function (w) {
      for (var i = 0; i < qts.length; i++) {
        if (strength(qts[i], w) > 0) { ranges.push([w.start, w.end]); return; }
      }
    });
    return ranges;
  }

  // HTML-escaped text with the matched words wrapped in <mark>; neighbours
  // separated only by spaces share one mark ("promo code").
  function highlight(text, query) {
    text = String(text == null ? '' : text);
    var ranges = hitRanges(text, query), merged = [];
    ranges.forEach(function (r) {
      var prev = merged[merged.length - 1];
      if (prev && /^\s+$/.test(text.slice(prev[1], r[0]))) prev[1] = r[1];
      else merged.push([r[0], r[1]]);
    });
    var out = '', last = 0;
    merged.forEach(function (r) {
      out += escapeHtml(text.slice(last, r[0])) + '<mark>' + escapeHtml(text.slice(r[0], r[1])) + '</mark>';
      last = r[1];
    });
    return out + escapeHtml(text.slice(last));
  }

  // "Card details, then click Place order." + "clicked place order"
  //   -> "…click Place order."
  function excerpt(text, query, max) {
    text = String(text == null ? '' : text);
    max = max || 64;
    var r = hitRanges(text, query), start = r.length ? r[0][0] : 0;
    var s = text.slice(start);
    if (s.length > max) {
      var cut = s.lastIndexOf(' ', max);
      s = s.slice(0, cut > max * 0.5 ? cut : max).replace(/[\s,;:]+$/, '') + '…';
    }
    return (start > 0 ? '…' : '') + s;
  }

  // True while the word being typed could still become a stopword ("whe"
  // on the way to "where"): the UI waits instead of saying "no results".
  function typingStopword(query) {
    var q = String(query == null ? '' : query).toLowerCase();
    if (!q || /[\s.,!?;:"”)]$/.test(q)) return false;
    var raw = q.match(WORD_RE) || [];
    var last = raw.length ? norm(raw[raw.length - 1]) : '';
    if (!last) return false;
    for (var w in STOPWORDS) if (w.length > last.length && w.indexOf(last) === 0) return true;
    return false;
  }

  function timecode(t) {
    t = Math.max(0, Math.floor(Number(t) || 0));
    var m = Math.floor(t / 60), s = t % 60;
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  var api = {
    search: search,
    highlight: highlight,
    excerpt: excerpt,
    timecode: timecode,
    tokenize: tokenize,
    typingStopword: typingStopword,
    stem: stem,
    escapeHtml: escapeHtml
  };
  root.AutoCineSearch = api;
  if (typeof module === 'object' && module && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));


/* =====================================================================
   2. The hero (browser only)
   ===================================================================== */
(function () {
  'use strict';
  if (typeof document === 'undefined' || typeof window === 'undefined') return;

  var Search = window.AutoCineSearch;
  var DATA = window.AUTOCINE_DEMO || { featured: null, projects: [], examples: [] };
  // demo-library.js is hand-edited: tolerate a malformed entry rather than
  // failing the whole hero (the search itself skips them too)
  function isObj(x) { return !!x && typeof x === 'object'; }
  function segsOf(p) { return p && Array.isArray(p.segments) ? p.segments.filter(isObj) : []; }
  function segT(seg) { var t = Number(seg.t); return isFinite(t) && t > 0 ? t : 0; }
  var PROJECTS = Array.isArray(DATA.projects) ? DATA.projects.filter(isObj) : [];
  var EXAMPLES = (Array.isArray(DATA.examples) ? DATA.examples : []).filter(function (x) {
    return typeof x === 'string' && x.trim();
  });
  if (!EXAMPLES.length) EXAMPLES = ['promo code'];
  var FEATURED = null;
  PROJECTS.forEach(function (p) { if (p.id === DATA.featured) FEATURED = p; });
  var FEATURED_SEGS = segsOf(FEATURED);
  function projectById(id) {
    for (var i = 0; i < PROJECTS.length; i++) if (PROJECTS[i].id === id) return PROJECTS[i];
    return null;
  }
  function durationOf(p) {
    var d = Number(p && p.duration);
    if (isFinite(d) && d > 0) return d;
    return segsOf(p).reduce(function (m, s) { return Math.max(m, segT(s) + (Number(s.dur) || 0)); }, 0);
  }
  var html = document.documentElement;

  /* ---------------------------------------------------------------
     Environment: URL params, motion preferences, the pause control
     --------------------------------------------------------------- */
  var params = {};
  try {
    new URLSearchParams(window.location.search).forEach(function (v, k) { params[k] = v; });
  } catch (e) { /* no URLSearchParams: no params */ }
  var SHOT = params.shot === '1';
  var PREFILL = typeof params.q === 'string' ? params.q : null;

  function mq(q) { return window.matchMedia ? window.matchMedia(q) : null; }
  var motionQuery = mq('(prefers-reduced-motion: reduce)');
  var coarseQuery = mq('(pointer: coarse)');
  var narrowQuery = mq('(max-width: 400px)');
  function reducedMotion() { return !!(motionQuery && motionQuery.matches); }
  function coarse() { return !!(coarseQuery && coarseQuery.matches); }

  // The visitor's pause choice (WCAG 2.2.2) is remembered in this browser
  // only; storage may be missing or throw, and the page works without it.
  var PAUSE_KEY = 'autocine.hero.motion';
  var paused = false;
  if (!SHOT) {
    try { paused = window.localStorage.getItem(PAUSE_KEY) === 'paused'; } catch (e) { paused = false; }
  }
  function still() { return SHOT || paused || reducedMotion(); }

  html.classList.add('js');
  if (SHOT) html.classList.add('is-shot');
  html.classList.toggle('is-paused', paused);

  function $(id) { return document.getElementById(id); }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function minJerk(u) { u = clamp(u, 0, 1); return u * u * u * (10 - 15 * u + 6 * u * u); }
  function lerp2(a, b, u) { return [a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u]; }
  var esc = Search.escapeHtml;

  var frameEl = $('top');
  var heroEl = $('main');
  var stageEl = $('stage');
  var cameraEl = $('camera');
  var video = $('hero-video');
  var ph = $('placeholder');
  var phFx = $('placeholder-fx');
  var cutEl = $('stage-cut');
  var reticle = $('reticle');
  var corners = reticle ? reticle.getElementsByTagName('i') : [];
  var reticleZ = $('reticle-z');
  var navEl = document.querySelector('.nav');
  var slateNow = $('slate-now');
  var slateDur = $('slate-dur');
  var slateSub = $('slate-sub');
  var slateTitle = $('slate-title');
  var slateZ = $('slate-z');
  var laneZooms = $('lane-zooms');
  var laneHead = $('lane-head');
  var motionBtn = $('motion');
  var form = $('search');
  var input = $('search-input');
  var bar = $('bar');
  var submitBtn = form.querySelector('.submit');
  var panel = $('results-panel');
  var list = $('search-results');
  var emptyEl = $('results-empty');
  var tagline = $('tagline');
  var caption = $('caption');
  var announce = $('announce');
  var ghostText = $('ghost-text');
  var phEl = {
    cursor: $('ph-cursor'), ring: $('ph-ring'), ringA: $('ph-ring-a'), ringB: $('ph-ring-b'),
    addr: $('ph-addr'), addrHint: $('ph-addr-hint'), addrText: $('ph-addr-text'),
    promoPh: $('ph-promo-ph'), promoVal: $('ph-promo-val'),
    discount: $('ph-discount'), ship: $('ph-ship'), total: $('ph-total'),
    optStd: $('ph-opt-std'), optExp: $('ph-opt-exp'), dotStd: $('ph-dot-std'), dotExp: $('ph-dot-exp'),
    cardEmpty: $('ph-card-empty'), cardFull: $('ph-card-full'),
    place: $('ph-place'), placeLabel: $('ph-place-label'), placedLabel: $('ph-placed-label'),
    note: $('ph-note'), noteDone: $('ph-note-done')
  };

  // Change-only DOM writes: every frame computes everything, but only
  // values that actually changed reach the DOM.
  var memo = {};
  function put(key, el, prop, val) { if (el && memo[key] !== val) { memo[key] = val; el.style[prop] = val; } }
  function putAttr(key, el, name, val) { if (el && memo[key] !== val) { memo[key] = val; el.setAttribute(name, val); } }
  function putText(key, el, val) { if (el && memo[key] !== val) { memo[key] = val; el.textContent = val; } }
  function putClass(key, el, cls, on) { if (el && memo[key] !== on) { memo[key] = on; el.classList.toggle(cls, on); } }

  /* ---------------------------------------------------------------
     The placeholder desktop. Everything is in the SVG's own units: a
     1600x1000 "screen". The checkout page is drawn in page units and
     placed on the desktop at 75% (the <g transform> in index.html).
     --------------------------------------------------------------- */
  var WORLD_W = 1600, WORLD_H = 1000;
  var PAGE = { s: 0.75, x: 200, y: 28 };
  function pg(x, y, w, h) { return { x: PAGE.x + x * PAGE.s, y: PAGE.y + y * PAGE.s, w: w * PAGE.s, h: h * PAGE.s }; }
  function pgPt(x, y) { return [PAGE.x + x * PAGE.s, PAGE.y + y * PAGE.s]; }
  function rc(x, y, w, h) { return { x: x, y: y, w: w, h: h }; }

  // The parts of the screen a moment can be about.
  var SPOTS = {
    // the checkout page (the featured take)
    title: pg(88, 192, 430, 84),
    header: pg(1216, 122, 296, 50),
    address: pg(90, 386, 756, 198),
    promo: pg(1100, 424, 400, 184),
    express: pg(90, 598, 756, 162),
    card: pg(90, 776, 756, 140),
    confirm: pg(1100, 624, 400, 170),
    place: pg(1100, 670, 400, 76),
    tabPricing: pg(326, 42, 156, 38),
    tabSettings: pg(486, 42, 144, 38),
    // the terminal (the API quickstart take)
    termLogin: rc(40, 274, 186, 46),
    termJson: rc(40, 330, 186, 66),
    termNext: rc(40, 374, 186, 24),
    term401: rc(40, 408, 186, 46),
    termExport: rc(40, 464, 186, 46),
    // the metrics window (the weekly metrics take)
    metSignups: rc(1376, 188, 202, 100),
    metRetention: rc(1376, 296, 202, 92),
    metFunnel: rc(1376, 396, 202, 96),
    metSupport: rc(1376, 500, 202, 78),
    metActions: rc(1376, 594, 202, 64)
  };

  // A picked result from another take frames the matching window: segment
  // i of that take -> spot. The pricing and settings takes are browser tabs.
  var TAKE_SPOTS = {
    'api-quickstart': ['termLogin', 'termLogin', 'termJson', 'term401', 'termNext', 'termExport'],
    'pricing-review': ['tabPricing'],
    'settings-bug': ['tabSettings'],
    'metrics-weekly': ['metSignups', 'metRetention', 'metFunnel', 'metSupport', 'metActions']
  };

  /* ---------------------------------------------------------------
     The sample take: what happens on the placeholder screen, second by
     second, on the FEATURED project's clock, so the slate's subtitles are
     its transcript and a picked result from it seeks to that moment.
     TODO(hero-video): none of this applies to the real video; it only
     has to stay in step with demo-library.js's featured segments.
     --------------------------------------------------------------- */
  var TAKE = durationOf(FEATURED) || 38;
  var START_T = 8.8;    // the page opens just before the first zoom
  var STILL_T = 29.72;  // the composed frame: the instant "Place order" is clicked
  var PROMO = 'SPRING15';

  var CURSOR_HOME = pgPt(760, 440);
  // Cursor travel: minimum-jerk moves between rests (a pure function of
  // time, so seeks and still frames need no simulation).
  var MOVES = [
    { t0: 2.4, t1: 3.7, to: pgPt(470, 252) },     // reads the page title
    { t0: 9.7, t1: 10.75, to: pgPt(150, 441) },   // first address field
    { t0: 15.1, t1: 15.95, to: pgPt(1196, 460) }, // promo code field
    { t0: 20.3, t1: 21.15, to: pgPt(134, 724) },  // Express
    { t0: 25.8, t1: 26.55, to: pgPt(214, 832) },  // card number
    { t0: 28.75, t1: 29.55, to: pgPt(1268, 712) },// Place order
    { t0: 33.6, t1: 35.0, to: CURSOR_HOME }       // back where the loop starts
  ];
  var CLICKS = [
    { t: 10.8, at: MOVES[1].to }, { t: 16.0, at: MOVES[2].to }, { t: 21.2, at: MOVES[3].to },
    { t: 26.6, at: MOVES[4].to }, { t: 29.6, at: MOVES[5].to }
  ];
  // The camera's plan for the take, like the real planner's output: a zoom
  // range per click, starting ~0.3s before it (the plan knows where the
  // click lands). The card click, 3s and a page-width away from "Place
  // order", stays an overview moment instead of a sweep across the page.
  var ZOOMS = [
    { from: 10.45, to: 13.7, spot: 'address', click: 1 },
    { from: 15.65, to: 19.3, spot: 'promo', click: 2 },
    { from: 20.85, to: 24.5, spot: 'express', click: 3 },
    { from: 29.25, to: 34.8, spot: 'confirm', click: 5 }
  ];
  // Picking a result from the featured take seeks the take so the moment
  // plays inside the zoom. `at` is the instant a still frame shows.
  var FEATURED_PICKS = [
    { seek: 0.7, at: 1.2, spot: 'title', hold: 2.8 },
    { seek: 4.6, at: 5.0, spot: 'header', hold: 2.6 },
    { seek: 10.15, at: 11.4, spot: 'address', hold: 3.4 },
    { seek: 15.4, at: 17.5, spot: 'promo', hold: 3.7 },
    { seek: 20.55, at: 21.6, spot: 'express', hold: 3.6 },
    { seek: 28.85, at: 29.9, spot: 'confirm', hold: 4.0 },
    { seek: 31.0, at: 31.6, spot: 'confirm', hold: 3.6 }
  ];
  // TODO(hero-video): once media/hero-loop.* is the real take, list where
  // each featured segment's moment sits in the VIDEO frame, as fractions
  // [x, y, w, h] of the frame (read them off the rendered take at that
  // segment's t). Until then a pick on the real video is a centred push.
  var VIDEO_SPOTS = [];

  function cursorAt(t) {
    var p = CURSOR_HOME;
    for (var i = 0; i < MOVES.length; i++) {
      var m = MOVES[i];
      if (t < m.t0) break;
      if (t >= m.t1) { p = m.to; continue; }
      var u = (t - m.t0) / (m.t1 - m.t0), s = minJerk(u), bow = 0.09 * Math.sin(Math.PI * u);
      var dx = m.to[0] - p[0], dy = m.to[1] - p[1];
      return [p[0] + dx * s - dy * bow, p[1] + dy * s + dx * bow];
    }
    return p;
  }

  function stateAt(t) {
    var typeStart = 16.35, perChar = 0.075;
    return {
      address: t >= 11.0,
      typed: t < typeStart ? 0 : Math.min(PROMO.length, Math.floor((t - typeStart) / perChar) + 1),
      discount: t >= 17.3,
      express: t >= 21.25,
      card: t >= 26.95,
      pressed: t >= 29.6 && t < 29.82,
      placed: t >= 29.82,
      confirmed: t >= 31.3
    };
  }

  function ringAt(t) {
    for (var i = 0; i < CLICKS.length; i++) {
      var d = t - CLICKS[i].t;
      if (d >= 0 && d < 0.62) return { x: CLICKS[i].at[0], y: CLICKS[i].at[1], p: d / 0.62 };
    }
    return null;
  }

  // The loop point: a short dip through black, like a cut in an edit.
  function dipAt(t) {
    if (t > TAKE - 0.6) return 0.92 * clamp((t - (TAKE - 0.6)) / 0.6, 0, 1);
    if (t < 0.5) return 0.92 * (1 - t / 0.5);
    return 0;
  }

  /* ---------------------------------------------------------------
     Geometry: which part of the desktop the card films (the crop), and
     screen units -> card pixels.
     --------------------------------------------------------------- */
  var geo = { W: 1, H: 1, vb: [0, 0, WORLD_W, WORLD_H], k: 1, ox: 0, oy: 0 };
  var band = { top: 0, bottom: 1 };  // the open strip between the nav and the bar
  var homeVb = geo.vb;
  var mode = 'placeholder';          // 'video' once the real footage plays
  var activeZooms = ZOOMS.slice();

  // The crop matches the card's aspect exactly: wide cards film the whole
  // desktop (trimming top and bottom evenly), squarer ones trim the sides,
  // phones film the order-summary column.
  function cropFor(a) {
    var w, h, x, y;
    if (a >= 1.6) { w = WORLD_W; h = WORLD_W / a; x = 0; y = (WORLD_H - h) / 2; }
    else if (a >= 1.1) { h = WORLD_H; w = WORLD_H * a; x = clamp(800 - w / 2, 0, WORLD_W - w); y = 0; }
    else {
      w = Math.min(WORLD_H * a, a < 0.62 ? 470 : 900);
      h = w / a;
      x = clamp(Math.min(1145 - w / 2, 1370 - w), 0, WORLD_W - w);
      y = clamp(470 - h / 2, 0, WORLD_H - h);
    }
    return [x, y, w, h];
  }
  // A crop the same size as `like`, centred on a spot (for cutting to a
  // window the home crop does not show).
  function cropAround(rect, like) {
    var w = like[2], h = like[3];
    return [clamp(rect.x + rect.w / 2 - w / 2, 0, WORLD_W - w), clamp(rect.y + rect.h / 2 - h / 2, 0, WORLD_H - h), w, h];
  }
  function spotInCrop(rect, vb) {
    var m = 4;
    return rect.x >= vb[0] + m && rect.y >= vb[1] + m &&
      rect.x + rect.w <= vb[0] + vb[2] - m && rect.y + rect.h <= vb[1] + vb[3] - m;
  }

  function setVb(vb) {
    geo.vb = vb;
    geo.k = Math.max(geo.W / vb[2], geo.H / vb[3]);
    geo.ox = (geo.W - vb[2] * geo.k) / 2;
    geo.oy = (geo.H - vb[3] * geo.k) / 2;
    if (mode === 'placeholder') {
      var s = vb.map(function (n) { return Math.round(n * 100) / 100; }).join(' ');
      putAttr('vb', ph, 'viewBox', s);
      putAttr('vbfx', phFx, 'viewBox', s);
    }
  }
  function toPx(x, y) {
    return [geo.ox + (x - geo.vb[0]) * geo.k, geo.oy + (y - geo.vb[1]) * geo.k];
  }

  function measure() {
    var r = stageEl.getBoundingClientRect();
    geo.W = Math.max(1, r.width);
    geo.H = Math.max(1, r.height);
    var nb = navEl ? navEl.getBoundingClientRect() : null, bb = bar.getBoundingClientRect();
    band.top = nb ? nb.bottom - r.top : 0;
    band.bottom = bb.top - r.top;
    if (band.bottom - band.top < 120) { band.top = 0; band.bottom = geo.H; }
    if (mode === 'video') {
      var vw = video.videoWidth || 1920, vh = video.videoHeight || 1080;
      setVb([0, 0, vw, vh]);
      activeZooms = [];
    } else {
      homeVb = cropFor(geo.W / geo.H);
      if (!(pick && pick.cropOn)) setVb(homeVb);
      activeZooms = ZOOMS.filter(function (z) {
        var c = CLICKS[z.click - 1].at;
        return spotInCrop(rc(c[0] - 30, c[1] - 30, 60, 60), homeVb);
      });
    }
    buildLane();
  }

  /* ---------------------------------------------------------------
     Critically damped spring, integrated in closed form (exact for any
     dt, so a dropped frame can't make it wobble or overshoot):
       x(t) = to + (d + (v0 + w d) t) e^(-w t),  d = x0 - to
     --------------------------------------------------------------- */
  function Spring(x) { this.x = x; this.v = 0; this.to = x; }
  Spring.prototype.step = function (dt, w) {
    var d = this.x - this.to, e = Math.exp(-w * dt), c = this.v + w * d;
    this.x = this.to + (d + c * dt) * e;
    this.v = (this.v - w * c * dt) * e;
  };
  Spring.prototype.snap = function () { this.x = this.to; this.v = 0; };
  Spring.prototype.rest = function () { return Math.abs(this.x - this.to) < 1e-4 && Math.abs(this.v) < 1e-3; };

  var OMEGA_IN = 5.2;   // rad/s; pushing in is brisk,
  var OMEGA_OUT = 3.3;  // easing back out is slower.

  // The virtual camera: zoom springs in log space (1x->2x feels as even as
  // 2x->4x), the centre in card fractions; clamped at render, so no edge
  // of the footage ever shows.
  var cam = { ls: new Spring(0), cx: new Spring(0.5), cy: new Spring(0.5), w: OMEGA_OUT };
  function snapCamera() { cam.ls.snap(); cam.cx.snap(); cam.cy.snap(); }
  function cameraAtRest() { return cam.ls.rest() && cam.cx.rest() && cam.cy.rest(); }
  function landscape() { return geo.W >= geo.H * 1.1; }

  // A pick frames its moment so it reads: the spot fills ~46% of the
  // card's width (up to 2.4x).
  function zoomFor(rect) {
    var wpx = Math.max(1, rect.w * geo.k), hpx = Math.max(1, rect.h * geo.k);
    if (!landscape()) return clamp(Math.min(0.9 * geo.W / wpx, 0.3 * geo.H / hpx), 1.3, 2.2);
    return clamp(Math.min(0.46 * geo.W / wpx, 0.3 * geo.H / hpx), 1.6, 2.4);
  }
  // Where the framed moment lands in the card (fractions). The real
  // renderer centres it; behind a landing page the centre column is copy,
  // so a pick (title stepped back) lands in the open band between the nav
  // and the prompt bar, never behind the bar.
  function pickAnchor(rect, s) {
    var c = toPx(rect.x + rect.w / 2, rect.y + rect.h / 2);
    var px = c[0] / geo.W;
    var hz = rect.h * geo.k * s;
    var top = band.top + 18 + hz / 2, bot = band.bottom - 22 - hz / 2;
    var y = top > bot ? (band.top + band.bottom) / 2 : clamp(c[1], top, bot);
    return [landscape() ? clamp(px, 0.28, 0.72) : 0.5, y / geo.H];
  }
  function aimPoint(pt, s, a, omega) {
    var c = toPx(pt[0], pt[1]);
    cam.ls.to = Math.log(s);
    cam.cx.to = c[0] / geo.W - (a[0] - 0.5) / s;
    cam.cy.to = c[1] / geo.H - (a[1] - 0.5) / s;
    cam.w = omega;
  }
  function aimAt(rect, omega) {
    var s = zoomFor(rect);
    aimPoint([rect.x + rect.w / 2, rect.y + rect.h / 2], s, pickAnchor(rect, s), omega);
    return s;
  }
  // The ambient take plays behind the copy, so its zooms are gentler (a
  // fixed push toward the click) and the click lands in the clear strip
  // beside the text column on its own side; on portrait cards (all copy)
  // the camera simply pushes in on the click where it is.
  var AMBIENT_ZOOM = 1.6, AMBIENT_ZOOM_PORTRAIT = 1.35;
  function aimAmbient(pt, omega) {
    var c = toPx(pt[0], pt[1]), px = c[0] / geo.W, py = c[1] / geo.H;
    if (landscape()) aimPoint(pt, AMBIENT_ZOOM, [px < 0.5 ? 0.14 : 0.86, clamp(py, 0.42, 0.6)], omega);
    else aimPoint(pt, AMBIENT_ZOOM_PORTRAIT, [clamp(px, 0.25, 0.75), clamp(py, 0.25, 0.75)], omega);
  }
  function aimPush(s, omega) {  // no known spot: a centred push
    cam.ls.to = Math.log(s); cam.cx.to = 0.5; cam.cy.to = 0.5; cam.w = omega;
  }
  function aimOverview(omega) {
    cam.ls.to = 0; cam.cx.to = 0.5; cam.cy.to = 0.5; cam.w = omega;
  }

  /* ---------------------------------------------------------------
     The moving parts' state for this instant
     --------------------------------------------------------------- */
  var tt = START_T;          // the take's clock (placeholder mode)
  var suppressUntil = -1;    // an ambient zoom a pick already played
  var cursorPos = cursorAt(tt);
  var ring = null;           // { x, y, p } click ring
  var ret = null;            // { rect, alpha, z } the camera's framing reticle
  var dip = 0;               // 0..1 dip through black (loop point, cuts)
  var picking = false;       // the title steps back while a pick has the stage
  var pick = null;
  var restTimer = 0;

  function ambientZoom(t) {
    if (t < suppressUntil) return null;
    for (var i = 0; i < activeZooms.length; i++) {
      if (t >= activeZooms[i].from && t < activeZooms[i].to) return activeZooms[i];
    }
    return null;
  }

  // Everything for this instant, from the take clock + the active pick.
  function plan() {
    var amb = mode === 'placeholder';
    ret = null; ring = null; dip = 0; picking = false;

    if (!pick) {
      if (amb) {
        // (no reticle here: it would sit on the headline; the slate's lane
        // and zoom readout carry the camera while the title is showing)
        var az = ambientZoom(tt);
        if (az) aimAmbient(CLICKS[az.click - 1].at, OMEGA_IN);
        else aimOverview(OMEGA_OUT);
        cursorPos = cursorAt(tt);
        ring = ringAt(tt);
        dip = dipAt(tt);
      } else {
        aimOverview(OMEGA_OUT);
      }
      return;
    }

    var p = pick, t = p.t, T = pickTimes(p);
    picking = t < T.outAt;
    // the cut to another part of the desktop (crops that don't show the spot)
    if (p.crop) {
      if (t < T.s0) dip = t / T.s0;
      else if (t < T.s0 + CUT) dip = 1 - (t - T.s0) / CUT;
      if (t >= T.s0 && !p.cropOn) {
        p.cropOn = true; setVb(p.crop); aimOverview(OMEGA_OUT); snapCamera();
        if (p.pendingSeek) { p.pendingSeek = false; tt = p.seek; p.from = cursorAt(tt); }
      }
      if (t >= T.backAt && t < T.backAt + CUT) dip = (t - T.backAt) / CUT;
      if (t >= T.backAt + CUT) {
        if (p.cropOn) { p.cropOn = false; setVb(homeVb); aimOverview(OMEGA_OUT); snapCamera(); }
        dip = 1 - (t - T.backAt - CUT) / CUT;
      }
      dip = clamp(dip, 0, 1);
    }

    if (t >= T.s0 && t < T.outAt) {
      if (p.rect) {
        var z = aimAt(p.rect, OMEGA_IN);
        ret = { rect: p.rect, alpha: clamp(Math.min((t - T.s0) / 0.2, (T.outAt - t) / 0.35), 0, 1), z: z };
      } else {
        aimPush(p.push, OMEGA_IN);
      }
    } else {
      aimOverview(OMEGA_OUT);
    }

    if (amb) {
      if (p.freeze) {
        // another take's moment: the cursor goes to it and clicks
        var click = p.click;
        if (t < T.outAt) cursorPos = lerp2(p.from, click, minJerk((t - T.s0) / 0.55));
        else cursorPos = lerp2(click, cursorAt(tt), minJerk((t - T.outAt) / 0.6));
        var d = t - T.s0 - 0.6;
        if (d >= 0 && d < 0.62) ring = { x: click[0], y: click[1], p: d / 0.62 };
      } else {
        // the featured take was seeked: blend the cursor onto its new path
        cursorPos = lerp2(p.from, cursorAt(tt), minJerk(t / 0.45));
        ring = ringAt(tt);
      }
    }
  }

  var CUT = 0.22;
  function pickTimes(p) {
    var s0 = p.crop ? CUT : 0;
    var outAt = s0 + p.hold;
    var backAt = outAt + 1.3;
    return { s0: s0, outAt: outAt, backAt: backAt, endAt: backAt + (p.crop ? 2 * CUT : 0) };
  }

  function step(dt) {
    if (pick) {
      pick.t += dt;
      if (pick.t >= pickTimes(pick).endAt) endPick();
    }
    if (mode === 'placeholder' && !(pick && pick.freeze)) {
      tt += dt;
      if (tt >= TAKE) { tt -= TAKE; suppressUntil = -1; }
    }
    plan();
    cam.ls.step(dt, cam.w);
    cam.cx.step(dt, cam.w);
    cam.cy.step(dt, cam.w);
  }

  function endPick() {
    if (!pick) return;
    if (pick.cropOn) setVb(homeVb);
    pick = null;
  }

  /* ---------------------------------------------------------------
     Rendering (transform/opacity writes, plus the SVG cursor layer)
     --------------------------------------------------------------- */
  function renderCamera() {
    var s = Math.max(1, Math.exp(cam.ls.x)), half = 0.5 / s;
    var cx = clamp(cam.cx.x, half, 1 - half), cy = clamp(cam.cy.x, half, 1 - half);
    var tx = geo.W * (0.5 - s * cx), ty = geo.H * (0.5 - s * cy);
    put('cam', cameraEl, 'transform', 'translate3d(' + tx.toFixed(2) + 'px,' + ty.toFixed(2) + 'px,0) scale(' + s.toFixed(4) + ')');
    return { s: s, tx: tx, ty: ty };
  }

  // The copy the still frame's reticle must keep clear of, in card pixels
  // (read when the frame is composed: layout only changes on resize).
  var COPY_SEL = ['.badge', '.headline', '.lede', '.hero > .btn-lg', '#bar', '.tagline', '.marks'];
  function copyRects() {
    var sr = stageEl.getBoundingClientRect(), out = [];
    COPY_SEL.forEach(function (sel) {
      var el = document.querySelector(sel);
      if (!el) return;
      var r = el.getBoundingClientRect();
      if (r.width && r.height) out.push({ x0: r.left - sr.left, y0: r.top - sr.top, x1: r.right - sr.left, y1: r.bottom - sr.top });
    });
    return out;
  }
  function collides(x0, y0, x1, y1, rects, gap) {
    for (var i = 0; i < rects.length; i++) {
      var r = rects[i];
      if (x0 < r.x1 + gap && x1 > r.x0 - gap && y0 < r.y1 + gap && y1 > r.y0 - gap) return true;
    }
    return false;
  }

  var CORNER = 18, CLEAR = 16, chipSize = {};
  function renderReticle(v) {
    if (!reticle) return;
    if (!ret || ret.alpha <= 0.001) { put('ret-o', reticle, 'opacity', '0'); return; }
    var a = toPx(ret.rect.x, ret.rect.y), b = toPx(ret.rect.x + ret.rect.w, ret.rect.y + ret.rect.h);
    var sx0 = a[0] * v.s + v.tx, sy0 = a[1] * v.s + v.ty, sx1 = b[0] * v.s + v.tx, sy1 = b[1] * v.s + v.ty;
    var avoid = ret.avoid || null, pad = 12;
    // Over visible copy (the composed still frame), the frame keeps CLEAR
    // pixels off it: its margin tightens first, and when even the tightest
    // one collides the frame is left out rather than drawn over the text.
    if (avoid) {
      while (pad >= 4 && collides(sx0 - pad, sy0 - pad, sx1 + pad, sy1 + pad, avoid, CLEAR)) pad -= 4;
      if (pad < 4) { put('ret-o', reticle, 'opacity', '0'); return; }
    }
    var x0 = sx0 - pad, y0 = sy0 - pad, x1 = sx1 + pad, y1 = sy1 + pad;
    function tr(x, y) { return 'translate3d(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px,0)'; }
    put('c1', corners[0], 'transform', tr(x0, y0));
    put('c2', corners[1], 'transform', tr(x1 - CORNER, y0));
    put('c3', corners[2], 'transform', tr(x1 - CORNER, y1 - CORNER));
    put('c4', corners[3], 'transform', tr(x0, y1 - CORNER));
    // The zoom readout on a dark chip. Live, it is the camera's zoom; on the
    // composed still (camera at 1x, the frame showing where it is about to
    // go) it is the TARGET, so it reads "-> 2.40x", never contradicting the
    // slate's 1.00x.
    var kind = ret.preview ? 'target' : 'live';
    putText('z', reticleZ, (ret.preview ? '→ ' : '') + (still() ? ret.z : v.s).toFixed(2) + '×');
    if (!chipSize[kind]) chipSize[kind] = [reticleZ.offsetWidth || 52, reticleZ.offsetHeight || 20];
    var cw = chipSize[kind][0], ch = chipSize[kind][1];
    // first clear spot: above the frame's left corner, above its right, then
    // under it; never under the nav, never on the copy
    var spots = [[x0, y0 - ch - 10], [x1 - cw, y0 - ch - 10], [x0, y1 + 8], [x1 - cw, y1 + 8]], at = null;
    for (var i = 0; i < spots.length && !at; i++) {
      var cx = clamp(spots[i][0], 8, geo.W - cw - 8), cy = spots[i][1];
      if (cy < band.top + 6 || cy + ch > geo.H - 6) continue;
      if (avoid && collides(cx, cy, cx + cw, cy + ch, avoid, 10)) continue;
      at = [cx, cy];
    }
    if (at) put('chip', reticleZ, 'transform', tr(at[0], at[1]));
    put('chip-o', reticleZ, 'opacity', at ? '1' : '0');
    put('ret-o', reticle, 'opacity', ret.alpha.toFixed(3));
  }

  function renderCursor() {
    putAttr('cur', phEl.cursor, 'transform', 'translate(' + cursorPos[0].toFixed(1) + ' ' + cursorPos[1].toFixed(1) + ')');
    if (ring) {
      var e = 1 - Math.pow(1 - ring.p, 3), r = (7 + 30 * e).toFixed(1);
      putAttr('rx', phEl.ringA, 'cx', ring.x); putAttr('ry', phEl.ringA, 'cy', ring.y); putAttr('rr', phEl.ringA, 'r', r);
      putAttr('rx2', phEl.ringB, 'cx', ring.x); putAttr('ry2', phEl.ringB, 'cy', ring.y); putAttr('rr2', phEl.ringB, 'r', r);
      putAttr('ro', phEl.ring, 'opacity', (0.9 * (1 - ring.p)).toFixed(3));
    } else {
      putAttr('ro', phEl.ring, 'opacity', '0');
    }
  }

  function applyState(st) {
    putAttr('s-addr-f', phEl.addr, 'fill', st.address ? '#faf5e4' : '#fff');
    putAttr('s-addr-s', phEl.addr, 'stroke', st.address ? '#e4dbc1' : '#dcd7cf');
    putAttr('s-addr-h', phEl.addrHint, 'opacity', st.address ? '0' : '1');
    putAttr('s-addr-t', phEl.addrText, 'opacity', st.address ? '1' : '0');
    putText('s-promo', phEl.promoVal, PROMO.slice(0, st.typed));
    putAttr('s-promo-ph', phEl.promoPh, 'opacity', st.typed ? '0' : '1');
    putAttr('s-disc', phEl.discount, 'opacity', st.discount ? '1' : '0');
    // amounts are bars (no prices on this page); they still visibly change
    // when the promo code and the shipping option change the total
    var shipW = st.express ? 50 : 44, totalW = st.discount ? (st.express ? 72 : 66) : (st.express ? 82 : 76);
    putAttr('s-ship-x', phEl.ship, 'x', 1492 - shipW); putAttr('s-ship-w', phEl.ship, 'width', shipW);
    putAttr('s-tot-x', phEl.total, 'x', 1492 - totalW); putAttr('s-tot-w', phEl.total, 'width', totalW);
    putAttr('s-std-s', phEl.optStd, 'stroke', st.express ? '#dcd7cf' : '#2a2926');
    putAttr('s-std-w', phEl.optStd, 'stroke-width', st.express ? '1' : '2');
    putAttr('s-exp-s', phEl.optExp, 'stroke', st.express ? '#2a2926' : '#dcd7cf');
    putAttr('s-exp-w', phEl.optExp, 'stroke-width', st.express ? '2' : '1');
    putAttr('s-dstd', phEl.dotStd, 'opacity', st.express ? '0' : '1');
    putAttr('s-dexp', phEl.dotExp, 'opacity', st.express ? '1' : '0');
    putAttr('s-ce', phEl.cardEmpty, 'opacity', st.card ? '0' : '1');
    putAttr('s-cf', phEl.cardFull, 'opacity', st.card ? '1' : '0');
    putAttr('s-place', phEl.place, 'fill', st.pressed ? '#47443f' : '#2a2926');
    putAttr('s-pl', phEl.placeLabel, 'opacity', st.placed ? '0' : '1');
    putAttr('s-pd', phEl.placedLabel, 'opacity', st.placed ? '1' : '0');
    putAttr('s-n', phEl.note, 'opacity', st.confirmed ? '0' : '1');
    putAttr('s-nd', phEl.noteDone, 'opacity', st.confirmed ? '1' : '0');
  }

  function segmentAt(t) {
    var hit = null;
    for (var i = 0; i < FEATURED_SEGS.length; i++) if (segT(FEATURED_SEGS[i]) <= t + 1e-6) hit = FEATURED_SEGS[i];
    return hit;
  }
  function takeTime() { return mode === 'video' ? (video.currentTime || 0) : tt; }
  function takeDuration() { return mode === 'video' && isFinite(video.duration) && video.duration > 0 ? video.duration : TAKE; }

  // The slate names what the stage shows: the featured take, or, while a
  // pick has cut the placeholder to another take's window, that take (its
  // name, the picked line, its clock; the zoom lane is the featured take's
  // plan, so it steps aside).
  function renderSlate() {
    var other = pick && pick.freeze ? pick : null;
    var t, d, sub;
    if (other) {
      t = other.r.t; d = Math.max(other.dur, t, 1); sub = other.r.text;
    } else {
      t = takeTime(); d = takeDuration();
      var seg = segmentAt(t);
      sub = seg && seg.text != null ? String(seg.text) : '';
    }
    putText('sl-title', slateTitle, other ? other.r.projectName : (FEATURED ? String(FEATURED.name) : ''));
    putClass('sl-other', laneZooms, 'is-off', !!other);
    putText('sl-now', slateNow, Search.timecode(t));
    putText('sl-dur', slateDur, Search.timecode(d));
    putText('sl-sub', slateSub, sub);
    put('sl-head', laneHead, 'transform', 'translateX(' + (100 * clamp(t / d, 0, 1)).toFixed(2) + '%)');
  }

  function buildLane() {
    if (!laneZooms) return;
    var key = activeZooms.map(function (z) { return z.from + '-' + z.to; }).join(',');
    if (memo.lane === key) return;
    memo.lane = key;
    laneZooms.innerHTML = activeZooms.map(function (z) {
      return '<i style="left:' + (100 * z.from / TAKE).toFixed(2) + '%;width:' + (100 * (z.to - z.from) / TAKE).toFixed(2) + '%"></i>';
    }).join('');
  }

  function render() {
    var v = renderCamera();
    putText('sl-z', slateZ, v.s.toFixed(2) + '×');
    putClass('sl-zc', slateZ, 'is-zoomed', v.s > 1.015);
    renderReticle(v);
    if (mode === 'placeholder') {
      renderCursor();
      applyState(stateAt(tt));
      put('dip', cutEl, 'opacity', dip.toFixed(3));
    } else {
      put('dip', cutEl, 'opacity', '0');
    }
    renderSlate();
    putClass('picking', frameEl, 'is-picking', picking);
    putClass('focus', frameEl, 'is-focus', picking && mode === 'placeholder');
  }

  /* ---------------------------------------------------------------
     Time
     --------------------------------------------------------------- */
  var running = false, lastTs = 0;
  var pageVisible = !document.hidden, heroInView = true;

  function wantsLoop() {
    if (still() || !pageVisible || !heroInView) return false;
    return mode === 'placeholder' || !!pick || !cameraAtRest();
  }
  function ensureLoop() {
    if (running || !wantsLoop()) return;
    running = true;
    lastTs = 0;
    window.requestAnimationFrame(tick);
  }
  function tick(ts) {
    if (!wantsLoop()) { running = false; return; }
    var dt = lastTs ? Math.min((ts - lastTs) / 1000, 1 / 24) : 1 / 60;
    lastTs = ts;
    step(dt);
    render();
    window.requestAnimationFrame(tick);
  }

  // The frame reduced-motion visitors and ?shot=1 get: the whole screen at
  // the instant "Place order" is clicked, the reticle showing what the
  // camera is about to frame (landscape only: on portrait cards it would
  // sit on the copy).
  function composeStill() {
    clearTimeout(restTimer);
    endPick();
    picking = false; dip = 0;
    if (mode === 'placeholder') {
      setVb(homeVb);
      tt = STILL_T;
      cursorPos = cursorAt(tt);
      var r = ringAt(tt);
      ring = r ? { x: r.x, y: r.y, p: 0.2 } : null;
      ret = landscape() && spotInCrop(SPOTS.place, homeVb) ?
        { rect: SPOTS.place, alpha: 1, z: zoomFor(SPOTS.place), preview: true, avoid: copyRects() } : null;
    } else {
      ring = null; ret = null;
    }
    aimOverview(OMEGA_OUT);
    snapCamera();
    render();
  }
  // Pause: the take stops where it is and the camera cuts back out.
  function freezeFrame() {
    clearTimeout(restTimer);
    endPick();
    picking = false; ret = null; ring = null; dip = 0;
    aimOverview(OMEGA_OUT);
    snapCamera();
    render();
  }
  function restFrame() {
    if (paused && !SHOT && !reducedMotion()) freezeFrame(); else composeStill();
  }

  /* ---------------------------------------------------------------
     The signature move for a picked result: seek, spring in toward the
     moment, hold, ease back out.
     --------------------------------------------------------------- */
  function segIndex(r) {
    var idx = -1;
    segsOf(projectById(r.projectId)).forEach(function (seg, i) { if (idx < 0 && segT(seg) === r.t) idx = i; });
    return idx;
  }

  function buildPick(r) {
    var i = segIndex(r);
    var p = { r: r, t: 0, hold: 2.6, rect: null, push: 1, freeze: false, crop: null, cropOn: false, from: cursorPos.slice() };
    if (mode === 'video') {
      var vs = r.isFeatured ? VIDEO_SPOTS[i] : null;
      if (vs) p.rect = { x: vs[0] * geo.vb[2], y: vs[1] * geo.vb[3], w: vs[2] * geo.vb[2], h: vs[3] * geo.vb[3] };
      else p.push = 1.35;
      return p;
    }
    if (r.isFeatured) {
      var fp = FEATURED_PICKS[i] || { seek: r.t, at: r.t, spot: 'title', hold: 2.8 };
      p.seek = fp.seek; p.at = fp.at; p.hold = fp.hold;
      p.rect = SPOTS[fp.spot];
    } else {
      var list = TAKE_SPOTS[r.projectId] || ['title'];
      p.rect = SPOTS[list[Math.min(Math.max(i, 0), list.length - 1)]];
      p.freeze = true;
      p.dur = durationOf(projectById(r.projectId));
      p.click = [p.rect.x + Math.min(p.rect.w * 0.35, 60), p.rect.y + Math.min(p.rect.h * 0.5, 24)];
    }
    if (!spotInCrop(p.rect, homeVb)) p.crop = cropAround(p.rect, homeVb);
    return p;
  }

  function applyStillPick() {
    var p = pick;
    if (p.crop) { p.cropOn = true; setVb(p.crop); }
    if (mode === 'placeholder') { cursorPos = p.freeze ? p.click : cursorAt(tt); ring = null; }
    if (p.rect) ret = { rect: p.rect, alpha: 1, z: aimAt(p.rect, OMEGA_IN) };
    else { ret = null; aimPush(p.push, OMEGA_IN); }
    snapCamera();
    picking = true; dip = 0;
    render();
  }

  function playPick(r) {
    clearTimeout(restTimer);
    if (pick && pick.cropOn) { setVb(homeVb); aimOverview(OMEGA_OUT); snapCamera(); }
    pick = null;
    var p = buildPick(r);

    if (still()) {
      // no motion: cut straight to the framed moment, then cut back
      if (mode === 'placeholder' && !p.freeze) tt = p.at;
      p.still = true;
      pick = p;
      applyStillPick();
      restTimer = setTimeout(restFrame, 3600);
      return;
    }

    if (mode === 'placeholder') {
      if (p.freeze) {
        if (tt > TAKE - 0.6 || tt < 0.5) tt = 0.5; // never freeze mid-dip
      } else {
        // a seek is a cut: under a dip when the crop changes too
        if (p.crop) p.pendingSeek = true; else tt = p.seek;
        // the zoom range this moment belongs to has been played by the pick
        var endT = p.seek + p.hold;
        suppressUntil = endT;
        ZOOMS.forEach(function (z) { if (endT >= z.from && endT < z.to) suppressUntil = z.to; });
        for (var k = 0; k < ZOOMS.length - 1; k++) {
          if (ZOOMS[k].to === ZOOMS[k + 1].from && suppressUntil === ZOOMS[k].to) suppressUntil = ZOOMS[k + 1].to;
        }
      }
    }
    pick = p;
    ensureLoop();
  }

  /* ---------------------------------------------------------------
     The real video. The placeholder stays until the footage is actually
     PLAYING (a poster alone, a blocked autoplay or a 404 keep the
     animated placeholder). Still modes (reduced motion, pause, ?shot=1)
     accept the poster instead; ?shot=1 also accepts the first frame.
     --------------------------------------------------------------- */
  var videoFailed = false, posterProbed = false, liveTimer = 0;
  function holdVideo() { try { video.pause(); } catch (e) { /* ignore */ } }
  // Motion is back (unpaused, reduced motion off). A still frame may have
  // put the POSTER up; in motion only footage that actually plays may stay,
  // so a failed, missing, refused or stuck video hands the stage back to
  // the placeholder (the 'playing' event brings the video in again).
  function resumeVideo() {
    if (!video || still()) return;
    clearTimeout(liveTimer);
    if (videoFailed || video.networkState === 3 /* NETWORK_NO_SOURCE */) { leaveVideoMode(); return; }
    if (mode === 'video' && video.paused) {
      liveTimer = setTimeout(function () { if (!still() && video.paused) leaveVideoMode(); }, 1500);
    }
    if (!video.paused) return;
    var p = video.play();
    if (p && p.catch) p.catch(function () { if (!still()) leaveVideoMode(); });
  }
  function onVideoFailed() {
    videoFailed = true;
    if (!still()) leaveVideoMode();
  }
  // Still frames accept the poster. It is probed only once the page is
  // still, so full motion (which never shows it) costs no extra request.
  function probePoster() {
    var src = video && video.getAttribute('poster');
    if (posterProbed || !src) return;
    posterProbed = true;
    var probe = new Image();
    probe.onload = function () { if (still()) enterVideoMode(); };
    probe.src = src;
  }
  function leaveVideoMode() {
    clearTimeout(liveTimer);
    if (mode !== 'video') return;
    endPick();
    mode = 'placeholder';
    frameEl.classList.remove('has-video');
    ret = null; ring = null; dip = 0; picking = false;
    measure();
    if (still()) restFrame(); else { plan(); render(); ensureLoop(); }
  }
  function enterVideoMode() {
    if (mode === 'video') return;
    mode = 'video';
    frameEl.classList.add('has-video');
    endPick();
    ret = null; ring = null; dip = 0; picking = false;
    measure();
    aimOverview(OMEGA_OUT);
    snapCamera();
    render();
    if (still()) holdVideo();
  }
  function seekVideo(t) {
    if (mode !== 'video' || video.readyState < 1 || !isFinite(video.duration)) return false;
    try { video.currentTime = clamp(t, 0, Math.max(0, video.duration - 0.05)); } catch (e) { return false; }
    return true;
  }

  if (video) {
    if (still()) { video.autoplay = false; video.removeAttribute('autoplay'); holdVideo(); }
    var sources = video.getElementsByTagName('source');
    var lastSource = sources.length ? sources[sources.length - 1] : null;
    if (lastSource) lastSource.addEventListener('error', onVideoFailed);
    video.addEventListener('error', onVideoFailed);
    video.addEventListener('playing', function () {
      if (still()) { holdVideo(); return; }
      if (video.videoWidth > 0) enterVideoMode();
    });
    video.addEventListener('play', function () { if (still()) holdVideo(); });
    video.addEventListener('loadeddata', function () { if (SHOT) enterVideoMode(); });
    video.addEventListener('timeupdate', function () { if (mode === 'video' && !running) renderSlate(); });
    video.addEventListener('loadedmetadata', function () { if (mode === 'video') { measure(); render(); } });
    if (SHOT && video.readyState >= 2) enterVideoMode();
    if (!still() && !video.paused && video.readyState >= 3 && video.videoWidth > 0) enterVideoMode();
    if (still()) probePoster();
  }

  /* ---------------------------------------------------------------
     Search UI: ARIA 1.2 combobox + listbox
     --------------------------------------------------------------- */
  var BASE_PLACEHOLDER = input.getAttribute('placeholder');

  var results = [];
  var active = -1;
  var isOpen = false;
  var sayTimer = 0;

  function say(msg, delay) {
    clearTimeout(sayTimer);
    sayTimer = setTimeout(function () { announce.textContent = msg; }, delay || 0);
  }

  // the "now showing" marker: the viewfinder from the logo, no red dot
  // (red means recording)
  var NOW_GLYPH = '<svg viewBox="0 0 24 24" aria-hidden="true"><g fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M3.5 8V5.5a2 2 0 012-2H8"/><path d="M16 3.5h2.5a2 2 0 012 2V8"/><path d="M20.5 16v2.5a2 2 0 01-2 2H16"/><path d="M8 20.5H5.5a2 2 0 01-2-2V16"/></g><circle cx="12" cy="12" r="3.2" fill="currentColor"/></svg>';

  function runSearch() {
    var q = input.value;
    bar.classList.toggle('has-value', q.length > 0);
    if (!Search.tokenize(q).length) { results = []; closePanel(); return; }
    results = Search.search(q, DATA);
    if (!results.length && Search.typingStopword(q)) { closePanel(); return; }
    renderResults(q);
    openPanel();
  }

  function renderResults(q) {
    setActive(-1);
    if (!results.length) {
      list.innerHTML = '';
      list.hidden = true;
      var tries = EXAMPLES.filter(function (x) { return x.toLowerCase() !== q.trim().toLowerCase(); }).slice(-2);
      emptyEl.innerHTML =
        '<p>Nothing in the sample library matches “' + esc(q.trim()) + '”.</p>' +
        '<p>Try ' + tries.map(function (x) {
          return '<button type="button" class="try" data-q="' + esc(x) + '">' + esc(x) + '</button>';
        }).join(' or ') + '</p>';
      emptyEl.hidden = false;
      // aria-expanded mirrors the LISTBOX (hidden here); the message itself
      // is announced through the live region
      input.setAttribute('aria-expanded', 'false');
      say('No matching moments. Try ' + tries.join(' or ') + '.', 500);
      return;
    }
    emptyEl.hidden = true;
    list.hidden = false;
    list.innerHTML = results.map(function (r, i) {
      var tc = Search.timecode(r.t);
      var label = r.projectName + ', ' + tc + (r.isFeatured ? ', now showing in the background' : '') + ': ' + r.text;
      return '<li class="opt" role="option" id="opt-' + i + '" aria-selected="false" data-i="' + i + '" aria-label="' + esc(label) + '">' +
        '<span class="opt-name">' + Search.highlight(r.projectName, q) + '</span>' +
        '<span class="opt-meta">' +
          (r.isFeatured ? '<span class="opt-feat">' + NOW_GLYPH + '<span class="opt-feat-label">Now showing</span></span>' : '') +
          '<span class="tc">' + tc + '</span>' +
        '</span>' +
        '<span class="opt-text">' + Search.highlight(r.text, q) + '</span>' +
      '</li>';
    }).join('');
    input.setAttribute('aria-expanded', 'true');
    say(results.length + (results.length === 1 ? ' moment' : ' moments') + ' found.', 500);
  }

  function setActive(i) {
    var opts = list.children;
    if (active >= 0 && opts[active]) opts[active].setAttribute('aria-selected', 'false');
    active = i;
    if (i >= 0 && opts[i]) {
      opts[i].setAttribute('aria-selected', 'true');
      input.setAttribute('aria-activedescendant', opts[i].id);
      revealOption(opts[i]);
    } else {
      input.removeAttribute('aria-activedescendant');
    }
  }

  // Scrolls the LISTBOX so the active option shows, never the page: the
  // panel is sized to the room it has (placePanel), and ?shot=1 frames must
  // stay aligned to the top.
  function revealOption(el) {
    var pad = 6, top = el.offsetTop, bottom = top + el.offsetHeight;
    if (top - pad < list.scrollTop) list.scrollTop = Math.max(0, top - pad);
    else if (bottom + pad > list.scrollTop + list.clientHeight) list.scrollTop = bottom + pad - list.clientHeight;
  }

  function move(d) {
    if (!isOpen) runSearch();
    if (!results.length) return;
    var n = results.length;
    setActive(active < 0 ? (d > 0 ? 0 : n - 1) : (active + d + n) % n);
  }

  function openPanel() {
    panel.hidden = false;
    isOpen = true;
    heroEl.classList.add('is-searching');
    placePanel();
  }
  function closePanel() {
    panel.hidden = true;
    isOpen = false;
    heroEl.classList.remove('is-searching');
    frameEl.classList.remove('is-searching-up');
    setActive(-1);
    input.setAttribute('aria-expanded', 'false');
  }

  // Desktop: the panel fits the window it opens in. It drops below the bar
  // when everything fits there; otherwise it opens ABOVE the bar (like a
  // composer's suggestions) when that side has more room, and the list
  // scrolls inside whichever room it gets. Phones keep the CSS sizing: the
  // bar is lifted to the top of the screen while typing (touchTyping).
  var PANEL_GAP = 10, EDGE = 12, MAX_LIST = 420, MIN_LIST = 120;
  var footEl = panel.querySelector('.results-foot');
  function placePanel() {
    if (!isOpen) return;
    if (coarse() || window.innerWidth <= 760) {
      panel.classList.remove('is-up');
      frameEl.classList.remove('is-searching-up');
      list.style.maxHeight = '';
      return;
    }
    var br = bar.getBoundingClientRect(), nb = navEl ? navEl.getBoundingClientRect() : null;
    var chrome = (footEl ? footEl.offsetHeight : 0) + (list.hidden ? emptyEl.offsetHeight : 0) + 2;
    var content = list.hidden ? 0 : Math.min(list.scrollHeight, MAX_LIST);
    // below: down to the window's edge or the slate, whichever comes first,
    // so the panel stays inside the card
    var slateEl = $('slate'), floor = window.innerHeight - EDGE;
    if (slateEl) floor = Math.min(floor, slateEl.getBoundingClientRect().top - 8);
    var below = floor - br.bottom - PANEL_GAP - chrome;
    var above = br.top - Math.max(nb ? nb.bottom : 0, 0) - PANEL_GAP - EDGE - chrome;
    var up = below < content && above > below;
    panel.classList.toggle('is-up', up);
    frameEl.classList.toggle('is-searching-up', up);
    list.style.maxHeight = Math.round(clamp(up ? above : below, MIN_LIST, MAX_LIST)) + 'px';
    if (active >= 0 && list.children[active]) revealOption(list.children[active]);
  }
  window.addEventListener('resize', placePanel);
  window.addEventListener('scroll', placePanel, { passive: true });

  var capTimer = 0;
  function showCaption(r) {
    var tc = Search.timecode(r.t);
    // real spaces between the parts (the text is what gets read out)
    caption.innerHTML = '<b>' + esc(r.projectName) + '</b> · <span class="tc-inline">' + tc + '</span>' +
      '<span class="cap-dash"> — </span><span class="cap-q">“' + esc(Search.excerpt(r.text, input.value, 60)) + '”</span>';
    tagline.classList.add('is-caption');
    heroEl.classList.add('has-caption');
    say('Showing ' + r.projectName + ' at ' + tc + ': ' + r.text);
    clearTimeout(capTimer);
    capTimer = setTimeout(function () {
      tagline.classList.remove('is-caption');
      heroEl.classList.remove('has-caption');
    }, 7500);
  }

  function choose(r) {
    if (!r) return;
    closePanel();
    if (r.isFeatured) seekVideo(r.t);
    playPick(r);
    showCaption(r);
    if (coarse()) {
      // drop the phone keyboard and bring the whole card back into view,
      // so the zoom and its caption are visible
      input.blur();
      touchTyping(false);
      window.scrollTo({ top: 0, behavior: still() ? 'auto' : 'smooth' });
    }
  }

  // Phones: the keyboard covers the lower half of the screen, which is where
  // the results open. While typing, give the page room below and lift the
  // bar to the top of the screen.
  var touchLift = 0;
  function touchTyping(on) {
    clearTimeout(touchLift);
    if (!on) { html.classList.remove('is-touch-typing'); return; }
    html.classList.add('is-touch-typing');
    touchLift = setTimeout(function () {
      var top = form.getBoundingClientRect().top;
      if (top > 24) window.scrollTo({ top: window.pageYOffset + top - 12, behavior: still() ? 'auto' : 'smooth' });
    }, 320);
  }

  input.addEventListener('input', runSearch);
  var quietFocus = false;
  input.addEventListener('focus', function () {
    twStop();
    if (coarse()) touchTyping(true);
    if (!quietFocus && Search.tokenize(input.value).length) runSearch();
  });
  input.addEventListener('blur', function () {
    if (!input.value) twLater();
    setTimeout(function () { if (document.activeElement !== input) touchTyping(false); }, 150);
  });
  input.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowDown') { e.preventDefault(); move(1); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
    else if (e.key === 'Escape') {
      if (isOpen) { e.preventDefault(); closePanel(); }
      else if (input.value) { e.preventDefault(); input.value = ''; runSearch(); }
    }
  });
  form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (!input.value.trim()) {
      // an empty bar runs the example the placeholder is showing
      input.value = currentExample();
      twStop();
    }
    if (!isOpen || !results.length) runSearch();
    choose(results[active >= 0 ? active : 0]);
  });
  // Escape from the submit button or a "Try" suggestion (focus is still in
  // the form, so the panel is open): close it and go back to the input.
  form.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape' || e.target === input) return;
    if (isOpen) closePanel();
    e.preventDefault();
    quietFocus = true;   // back to the input WITHOUT reopening the panel
    input.focus();
    quietFocus = false;
  });
  form.addEventListener('focusout', function () {
    setTimeout(function () { if (isOpen && !form.contains(document.activeElement)) closePanel(); }, 0);
  });

  // A press anywhere in the search form except the input itself keeps focus
  // in the input: the panel stays open and the arrowed-to option stays
  // active (Safari and Firefox never focus a clicked <button>, so without
  // this the input's blur closed the panel before the submit click). A
  // press on the bar's padding or its "/" hint focuses the input.
  form.addEventListener('mousedown', function (e) {
    if (e.target === input || e.button !== 0) return;
    e.preventDefault();
    if (document.activeElement !== input && bar.contains(e.target) && !submitBtn.contains(e.target)) input.focus();
  });
  list.addEventListener('click', function (e) {
    var li = e.target.closest ? e.target.closest('.opt') : null;
    if (li) choose(results[+li.getAttribute('data-i')]);
  });
  list.addEventListener('mousemove', function (e) {
    var li = e.target.closest ? e.target.closest('.opt') : null;
    if (li && +li.getAttribute('data-i') !== active) setActive(+li.getAttribute('data-i'));
  });
  emptyEl.addEventListener('click', function (e) {
    var b = e.target.closest ? e.target.closest('[data-q]') : null;
    if (!b) return;
    input.value = b.getAttribute('data-q');
    input.focus();
    runSearch();
  });

  document.addEventListener('pointerdown', function (e) {
    if (isOpen && !form.contains(e.target)) closePanel();
  });
  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
    e.preventDefault();
    input.focus();
    input.select();
  });

  /* ---------------------------------------------------------------
     Typewriter placeholder (empty + unfocused input, motion allowed)
     --------------------------------------------------------------- */
  var tw = { i: 0, timer: 0, on: false };

  function currentExample() { return EXAMPLES[tw.i % EXAMPLES.length]; }
  function twWanted() {
    return !still() && pageVisible && !input.value && document.activeElement !== input;
  }
  function twStart() {
    if (tw.on || !twWanted()) return;
    tw.on = true;
    bar.classList.add('ghosting');
    typeOut(currentExample(), 0);
  }
  function typeOut(str, n) {
    ghostText.textContent = str.slice(0, n);
    if (n < str.length) {
      bar.classList.add('typing');
      tw.timer = setTimeout(typeOut, 46 + ((n * 7) % 5) * 14, str, n + 1);
    } else {
      bar.classList.remove('typing');
      tw.timer = setTimeout(eraseOut, 2300, str, n);
    }
  }
  function eraseOut(str, n) {
    ghostText.textContent = str.slice(0, n);
    if (n > 0) {
      bar.classList.add('typing');
      tw.timer = setTimeout(eraseOut, 22, str, n - 1);
    } else {
      bar.classList.remove('typing');
      tw.i++;
      tw.timer = setTimeout(typeOut, 450, currentExample(), 0);
    }
  }
  function twStop() {
    clearTimeout(tw.timer);
    tw.on = false;
    bar.classList.remove('ghosting', 'typing');
  }
  function twLater() {
    twStop();
    if (!still()) tw.timer = setTimeout(twStart, 900);
  }

  function applyPlaceholderMode() {
    if (SHOT) {
      // frozen: the first example, fully typed, caret showing
      twStop();
      tw.i = 0;
      bar.classList.add('ghosting');
      ghostText.textContent = EXAMPLES[0];
      input.setAttribute('placeholder', BASE_PLACEHOLDER);
    } else if (still()) {
      twStop();
      // phones: the bare example, the "Try “…”" frame does not fit the bar
      input.setAttribute('placeholder', narrowQuery && narrowQuery.matches ? currentExample() : 'Try “' + currentExample() + '”');
    } else {
      input.setAttribute('placeholder', BASE_PLACEHOLDER);
      twStart();
    }
  }

  /* ---------------------------------------------------------------
     The pause control (WCAG 2.2.2): stops the take, the camera, the
     video, the typewriter and the caret blink.
     --------------------------------------------------------------- */
  function syncMotionButton() {
    if (!motionBtn) return;
    motionBtn.setAttribute('aria-pressed', paused ? 'true' : 'false');
    var label = paused ? 'Play background motion' : 'Pause background motion';
    motionBtn.setAttribute('aria-label', label);
    motionBtn.setAttribute('title', label);
  }
  // Motion allowed again: a still pick (and its pending cut back) ends
  // here, so no stale timer can cut the moving camera later.
  function motionResumed() {
    clearTimeout(restTimer);
    endPick();
    plan();          // re-aim: a camera parked on a still pick eases back out
    resumeVideo();
    ensureLoop();
  }
  function setPaused(on) {
    paused = !!on;
    try { window.localStorage.setItem(PAUSE_KEY, paused ? 'paused' : 'playing'); } catch (e) { /* not stored */ }
    html.classList.toggle('is-paused', paused);
    syncMotionButton();
    applyPlaceholderMode();
    if (paused) { holdVideo(); probePoster(); freezeFrame(); }
    else if (!still()) motionResumed();
    else restFrame();
  }
  if (motionBtn) motionBtn.addEventListener('click', function () { setPaused(!paused); });

  /* ---------------------------------------------------------------
     Boot
     --------------------------------------------------------------- */
  syncMotionButton();
  measure();
  if (still()) restFrame(); else { plan(); render(); ensureLoop(); }

  if (PREFILL !== null) {
    input.value = PREFILL;
    runSearch();
    if (SHOT && results.length) setActive(0);
  }
  applyPlaceholderMode();

  function onResize() {
    measure();
    if (still() && !SHOT) applyPlaceholderMode();
    if (still()) { if (pick && pick.still) applyStillPick(); else restFrame(); }
    else render();
  }
  if (window.ResizeObserver) new ResizeObserver(onResize).observe(stageEl);
  else window.addEventListener('resize', onResize);
  // the headline's web font can land after the still frame was composed;
  // recompose so the reticle's clearance is measured against the real copy
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(function () { if (still()) onResize(); placePanel(); });
  }

  if (window.IntersectionObserver) {
    new IntersectionObserver(function (entries) {
      heroInView = entries[entries.length - 1].isIntersecting;
      if (heroInView) ensureLoop();
    }).observe(frameEl);
  }

  document.addEventListener('visibilitychange', function () {
    pageVisible = !document.hidden;
    if (pageVisible) { ensureLoop(); twStart(); } else { twStop(); }
  });

  function onMotionChange() {
    applyPlaceholderMode();
    if (still()) { holdVideo(); probePoster(); restFrame(); }
    else motionResumed();
  }
  if (motionQuery) {
    if (motionQuery.addEventListener) motionQuery.addEventListener('change', onMotionChange);
    else if (motionQuery.addListener) motionQuery.addListener(onMotionChange);
  }
})();
