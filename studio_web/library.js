/* library.js — the projects home: cards for every recording session, and
   "search your recordings" (by name or by what was said). */
"use strict";

(function () {
  const $ = function (id) { return document.getElementById(id); };
  const ui = {
    top: $("lib-top"),
    grid: $("lib-grid"),
    empty: $("lib-empty"),
    refresh: $("lib-refresh"),
    record: $("lib-record"),
    recordEmpty: $("lib-record-empty"),
    status: $("lib-status"),
    statusText: $("lib-status-text"),
    sub: $("lib-sub"),
    perms: $("lib-perms"),
    permsDetail: $("lib-perms-detail"),
    permsRefresh: $("lib-perms-refresh"),
  };

  const state = {
    sessions: [],
    loaded: false,
    menuFor: null,          // session name with an open kebab menu
    lastRecordDone: null,
    lastRenderDone: null,
    renamingFor: null,
  };

  function editorUrl(name, t) {
    let url = "/editor.html?session=" + encodeURIComponent(name);
    // ?t=<seconds>: the editor opens with the playhead there (a transcript hit)
    if (Number.isFinite(t) && t > 0) url += "&t=" + String(Math.round(t * 100) / 100);
    return url;
  }

  // Recorder pill open lives in shared.js (openRecorderBar) so the editor's
  // "New Recording" button shares the exact native-then-popup path.
  const openBar = openRecorderBar;

  function closeMenu() {
    const open = document.querySelector(".menu");
    if (open) open.remove();
    state.menuFor = null;
  }

  function cardMenu(card, s) {
    closeMenu();
    state.menuFor = s.session;
    const menu = el("div", "menu");
    const mkItem = function (label, fn, danger) {
      const b = el("button", danger ? "danger" : null, label);
      b.addEventListener("click", function (ev) {
        ev.stopPropagation();
        closeMenu();
        fn();
      });
      return b;
    };
    menu.appendChild(mkItem("Open", function () { window.location.href = editorUrl(s.session); }));
    menu.appendChild(mkItem("Rename", function () { startRename(s.session); }));
    menu.appendChild(mkItem("Reveal in Finder", function () {
      api("/api/reveal/" + encodeURIComponent(s.session), { body: {} }).catch(function (e) { toast(e.message, "err"); });
    }));
    menu.appendChild(el("div", "sep"));
    menu.appendChild(mkItem("Move to Trash", function () { deleteSession(s.session); }, true));
    card.appendChild(menu);
    menu.style.right = "8px";
    menu.style.top = "8px";
  }

  async function deleteSession(name) {
    if (!window.confirm("Move this recording to the Trash?\n\n" + name)) return;
    try {
      await api("/api/sessions/" + encodeURIComponent(name) + "/delete", { body: {} });
      toast("Moved to Trash");
      loadSessions();
    } catch (e) {
      toast(e.message, "err");
    }
  }

  function startRename(name) {
    state.renamingFor = name;
    renderGrid();
  }

  async function commitRename(name, value) {
    state.renamingFor = null;
    const trimmed = (value || "").trim();
    try {
      await api("/api/project/" + encodeURIComponent(name), { body: { name: trimmed || null } });
      const s = state.sessions.find(function (x) { return x.session === name; });
      if (s) s.name = trimmed || null;
    } catch (e) {
      toast(e.message, "err");
    }
    renderGrid();
  }

  function displayName(s) {
    return s.name || fmtSessionDate(s.session) || s.session;
  }

  function sessionDuration(s) {
    return Number.isFinite(s.trimmed_duration) ? s.trimmed_duration : s.duration;
  }

  function card(s) {
    const c = el("div", "proj-card");
    c.tabIndex = 0;
    c.setAttribute("role", "button");

    const thumb = el("div", "proj-thumb");
    if (s.error) {
      thumb.appendChild(el("div", "proj-thumb-empty", "Unreadable recording"));
    } else {
      const img = document.createElement("img");
      img.loading = "lazy";
      img.alt = "";
      img.src = autocineUrl("/api/thumb/" + encodeURIComponent(s.session));
      img.addEventListener("error", function () {
        img.remove();
        thumb.insertBefore(el("div", "proj-thumb-empty", "No preview"), thumb.firstChild);
      });
      thumb.appendChild(img);
      const dur = sessionDuration(s);
      if (Number.isFinite(dur)) thumb.appendChild(el("span", "proj-duration", fmtTime(dur, false)));
    }
    c.appendChild(thumb);

    const body = el("div", "proj-body");
    const nameRow = el("div", "proj-name-row");
    const nameEl = el("div", "proj-name");
    if (state.renamingFor === s.session) {
      const input = document.createElement("input");
      input.type = "text";
      input.value = s.name || "";
      input.placeholder = fmtSessionDate(s.session) || s.session;
      input.addEventListener("click", function (ev) { ev.stopPropagation(); });
      input.addEventListener("keydown", function (ev) {
        ev.stopPropagation();     // the card's own keydown navigates on Enter
        if (ev.key === "Enter") commitRename(s.session, input.value);
        if (ev.key === "Escape") { state.renamingFor = null; renderGrid(); }
      });
      input.addEventListener("blur", function () { commitRename(s.session, input.value); });
      nameEl.appendChild(input);
      setTimeout(function () { input.focus(); input.select(); }, 0);
    } else {
      nameEl.textContent = displayName(s);
      nameEl.title = displayName(s);
    }
    nameRow.appendChild(nameEl);

    const kebab = el("button", "btn-icon btn-quiet proj-kebab");
    kebab.innerHTML = ICONS.kebab;
    kebab.title = "More";
    kebab.addEventListener("click", function (ev) {
      ev.stopPropagation();
      if (state.menuFor === s.session) closeMenu();
      else cardMenu(c, s);
    });
    nameRow.appendChild(kebab);
    body.appendChild(nameRow);

    body.appendChild(el("div", "proj-meta", metaLine(s)));

    const badges = el("div", "proj-badges");
    if (s.error) badges.appendChild(el("span", "proj-badge err", "Error"));
    if (s.has_edits) badges.appendChild(el("span", "proj-badge edited", "Edited"));
    if (s.has_output_mp4) badges.appendChild(el("span", "proj-badge rendered", "MP4"));
    if (s.has_output_gif) badges.appendChild(el("span", "proj-badge rendered", "GIF"));
    if (s.chapter_count > 0) badges.appendChild(el("span", "proj-badge", s.chapter_count + (s.chapter_count === 1 ? " chapter" : " chapters")));
    body.appendChild(badges);
    c.appendChild(body);

    c.addEventListener("click", function () {
      if (state.renamingFor === s.session || state.menuFor === s.session) return;
      if (s.error) { toast("This recording can't be opened: " + s.error, "err"); return; }
      window.location.href = editorUrl(s.session);
    });
    c.addEventListener("keydown", function (ev) {
      if (ev.target !== c) return;   // ignore keys from the rename input etc.
      if (ev.key === "Enter" && !s.error) window.location.href = editorUrl(s.session);
    });
    return c;
  }

  function metaLine(s) {
    const metaBits = [];
    if (!s.error) {
      if (Number.isFinite(s.fps)) metaBits.push(Math.round(s.fps) + " fps");
      if (s.size && s.size.length === 2) metaBits.push(s.size[0] + "×" + s.size[1]);
      if (Number.isFinite(s.click_count)) metaBits.push(s.click_count + " clicks");
      const when = fmtSessionDate(s.session);
      if (when && s.name) metaBits.push(when);
    } else {
      metaBits.push(s.error);
    }
    return metaBits.join("  /  ");
  }

  function renderGrid() {
    closeMenu();
    ui.grid.innerHTML = "";
    state.sessions.forEach(function (s) { ui.grid.appendChild(card(s)); });
    const empty = state.sessions.length === 0;
    ui.empty.hidden = !empty;
    ui.grid.hidden = empty;
    ui.sub.textContent = empty ? "Projects" :
      state.sessions.length + (state.sessions.length === 1 ? " project" : " projects");
    finder.sessionsChanged();
  }

  async function loadSessions() {
    try {
      const data = await api("/api/sessions");
      state.sessions = data.sessions || [];
      state.loaded = true;
      renderGrid();
    } catch (e) {
      toast("Could not load projects: " + e.message, "err");
    }
  }

  async function loadPermissions() {
    try {
      const report = await api("/api/permissions");
      const missing = report.missing_required || [];
      if (missing.length) {
        const labels = missing.map(function (k) {
          const c = report.checks && report.checks[k];
          return c ? c.label : k;
        });
        ui.perms.hidden = false;
        ui.permsDetail.textContent = labels.join(", ") +
          ". Grant them to your terminal app in System Settings, then fully quit and relaunch it.";
      } else {
        ui.perms.hidden = true;
      }
    } catch (e) { /* leave as-is */ }
  }

  function setStatus(text, tone) {
    if (!text) { ui.status.hidden = true; return; }
    ui.status.hidden = false;
    ui.statusText.textContent = text;
    ui.status.className = "pill" + (tone ? " " + tone : "");
  }

  async function pollState() {
    try {
      const s = await api("/api/state");
      const rec = s.record || {};
      const ren = s.render || {};
      if (rec.status === "recording" || rec.status === "countdown" || rec.status === "stopping") {
        setStatus("Rec active", "rec");
      } else if (ren.status === "running") {
        setStatus("Export running", "ok");
      } else if (ren.status === "error") {
        setStatus("Export failed", "err");
      } else {
        setStatus(null);
      }
      if (rec.status === "done" && rec.session && rec.session !== state.lastRecordDone) {
        state.lastRecordDone = rec.session;
        loadSessions();
      }
      const renKey = (ren.session || "") + "|" + (ren.out_path || "");
      if (ren.status === "done" && renKey !== state.lastRenderDone) {
        state.lastRenderDone = renKey;
        loadSessions();
      }
    } catch (e) { /* server briefly unreachable */ }
  }

  /* ======================= search your recordings =======================
     A combobox over the REAL projects. Names are searchable the moment the
     grid paints; transcripts are fetched lazily (first focus or keystroke),
     TRANSCRIPT_CONCURRENCY at a time, and cached for the life of the page --
     GET /api/transcript/<name> only reads the cache on disk (it never runs
     ASR), and a take with none answers status "none". Name matches never
     wait on those fetches. Matching, ranking and highlighting are pure and
     live in search.js (window.AutoCineSearch, node-testable). */
  const finder = (function () {
    const Search = window.AutoCineSearch;
    const TRANSCRIPT_CONCURRENCY = 4;
    // A search this slow (ms) is a big, transcribed library: from then on a
    // keystroke waits for a pause in typing instead of re-ranking every time.
    // Small libraries search in about a millisecond and stay instant.
    const SLOW_SEARCH_MS = 12;
    const INPUT_DEBOUNCE_MS = 80;
    const RERANK_MS = 150;      // transcript arrivals re-rank once per burst
    const q = {
      wrap: $("lib-search-wrap"), form: $("lib-search"), bar: $("lib-q-bar"),
      input: $("lib-q"), go: $("lib-q-go"), panel: $("lib-q-panel"),
      list: $("lib-q-list"), empty: $("lib-q-empty"), status: $("lib-q-status"),
      live: $("lib-q-live"),
    };
    const sx = {
      started: false,
      transcripts: new Map(),   // session -> [{t, dur, text}]; [] = none/unreadable
      waiting: new Set(),       // queued or in flight
      queue: [],
      inflight: 0,
      perProject: new Map(),    // session -> {key, entries} (search.js indexProject)
      idx: null,                // combined index; null = rebuild on next search
      results: [],
      active: -1,
      open: false,
      liveTimer: 0,
      cost: 0,                  // ms the last search took (drives the debounce)
      inputTimer: 0,            // a debounced keystroke not yet searched
      rerankTimer: 0,           // coalesces transcript arrivals
      stale: false,             // results changed while the pointer rested on the list
    };

    if (!Search || !q.form) {
      return { sessionsChanged: function () {} };
    }

    function searchable() {
      return state.sessions.filter(function (s) { return !s.error; });
    }
    function sessionByName(name) {
      return state.sessions.find(function (s) { return s.session === name; }) || null;
    }

    // --- index: per project, rebuilt only when its name/transcript changes ---
    function currentIndex() {
      if (sx.idx) return sx.idx;
      const entries = [];
      const alive = new Set();
      searchable().forEach(function (s, i) {
        const name = displayName(s);
        const segs = sx.transcripts.has(s.session) ? sx.transcripts.get(s.session) : null;
        const key = i + "\u0000" + name + "\u0000" + (segs ? segs.length : -1);
        let cached = sx.perProject.get(s.session);
        if (!cached || cached.key !== key) {
          cached = { key: key, entries: Search.indexProject({ id: s.session, name: name, segments: segs }, i) };
          sx.perProject.set(s.session, cached);
        }
        alive.add(s.session);
        cached.entries.forEach(function (e) { entries.push(e); });
      });
      sx.perProject.forEach(function (_, k) { if (!alive.has(k)) sx.perProject.delete(k); });
      sx.idx = { entries: entries };
      return sx.idx;
    }

    // --- transcripts: lazy, limited concurrency, cached for the page's life ---
    function startIndexing() {
      sx.started = true;
      searchable().forEach(function (s) {
        if (sx.transcripts.has(s.session) || sx.waiting.has(s.session)) return;
        sx.waiting.add(s.session);
        sx.queue.push(s.session);
      });
      pump();
      renderStatus();
    }

    function pump() {
      while (sx.inflight < TRANSCRIPT_CONCURRENCY && sx.queue.length) {
        fetchTranscript(sx.queue.shift());
      }
    }

    function fetchTranscript(name) {
      sx.inflight++;
      api("/api/transcript/" + encodeURIComponent(name)).then(function (d) {
        const segs = d && Array.isArray(d.segments) ? d.segments : [];
        // keep only what search needs; the payload also carries every word
        sx.transcripts.set(name, segs.map(function (g) {
          return { t: g && g.t, dur: g && g.dur, text: g && g.text };
        }));
      }).catch(function () {
        sx.transcripts.set(name, []);   // unreadable: names still match; not retried this page
      }).then(function () {
        sx.inflight--;
        sx.waiting.delete(name);
        sx.idx = null;
        pump();
        // Re-rank only a panel that is OPEN. A closed one was closed on
        // purpose (Esc keeps focus in the field by design, click-outside),
        // and an arrival must never pop it back; the next keystroke searches
        // the fuller index anyway.
        scheduleRerank();
      });
    }

    // A burst of arrivals costs one re-rank, not one per transcript.
    function scheduleRerank() {
      renderStatus();
      if (sx.rerankTimer || !sx.open) return;
      sx.rerankTimer = setTimeout(function () {
        sx.rerankTimer = 0;
        rerank();
      }, RERANK_MS);
    }

    // Re-rank the open panel, keeping the arrowed-to row. Not while the
    // pointer rests on the list: rebuilding it would slide a different
    // result under the pointer between aiming and clicking (and Enter and
    // click would then name different rows). It waits for the pointer to
    // leave, an arrow key, or the next keystroke.
    function rerank() {
      if (!sx.open) { renderStatus(); return; }
      if (!q.list.hidden && q.list.matches(":hover")) { sx.stale = true; renderStatus(); return; }
      runSearch(true);
    }

    function transcribedCount() {
      let n = 0;
      searchable().forEach(function (s) {
        const segs = sx.transcripts.get(s.session);
        if (segs && segs.length) n++;
      });
      return n;
    }

    function renderStatus() {
      const pending = sx.waiting.size;
      q.status.classList.toggle("is-indexing", pending > 0);
      if (pending > 0) {
        q.status.textContent = "Indexing " + pending + (pending === 1 ? " recording…" : " recordings…");
        return;
      }
      const total = searchable().length;
      const tx = transcribedCount();
      q.status.textContent = total + (total === 1 ? " recording" : " recordings") + " · " +
        (tx ? tx + " transcribed" : "no transcripts yet");
    }

    // --- results ---
    function keyOf(r) { return r ? r.kind + "|" + r.projectId + "|" + r.t : ""; }

    function runSearch(keepActive) {
      const query = q.input.value;
      sx.stale = false;
      q.bar.classList.toggle("has-value", query.length > 0);
      renderStatus();
      if (!Search.hasTerms(query)) { sx.results = []; closePanel(); return; }
      const before = keepActive ? keyOf(sx.results[sx.active]) : "";
      const t0 = performance.now();
      sx.results = Search.search(query, currentIndex());
      sx.cost = performance.now() - t0;
      if (!sx.results.length && Search.typingStopword(query)) { closePanel(); return; }
      renderResults(query);
      openPanel();
      if (before) {
        const i = sx.results.findIndex(function (r) { return keyOf(r) === before; });
        if (i >= 0) setActive(i);
      }
    }

    function appendHighlighted(node, text, query) {
      Search.highlightParts(text, query).forEach(function (part) {
        if (!part.text) return;
        if (part.hit) node.appendChild(el("mark", null, part.text));
        else node.appendChild(document.createTextNode(part.text));
      });
    }

    function optionEl(r, i, query) {
      const s = sessionByName(r.projectId) || {};
      const moment = r.kind === "moment";
      const li = el("li", "lib-q-opt " + (moment ? "is-moment" : "is-name"));
      li.id = "lib-q-opt-" + i;
      li.setAttribute("role", "option");
      li.setAttribute("aria-selected", "false");
      li.dataset.i = String(i);

      const thumb = el("span", "lib-q-thumb");
      const img = document.createElement("img");
      img.alt = "";
      img.decoding = "async";
      img.src = autocineUrl("/api/thumb/" + encodeURIComponent(r.projectId));
      img.addEventListener("error", function () { img.remove(); });
      thumb.appendChild(img);
      li.appendChild(thumb);

      const nameEl = el("span", "lib-q-name");
      appendHighlighted(nameEl, r.projectName, query);
      li.appendChild(nameEl);

      const meta = el("span", "lib-q-meta");
      const tc = Search.timecode(r.t);
      const dur = sessionDuration(s);
      if (moment) meta.appendChild(el("span", "lib-q-tc", tc));
      else if (Number.isFinite(dur)) meta.appendChild(el("span", "lib-q-dur", fmtTime(dur, false)));
      li.appendChild(meta);

      if (moment) {
        const snippet = Search.excerpt(r.text, query, 150);
        const text = el("span", "lib-q-text");
        appendHighlighted(text, snippet, query);
        li.appendChild(text);
        li.setAttribute("aria-label", r.projectName + ", at " + tc + ": " + snippet);
      } else {
        li.appendChild(el("span", "lib-q-sub", "Project  /  " + metaLine(s)));
        li.setAttribute("aria-label", r.projectName + ", open project");
      }
      return li;
    }

    function renderResults(query) {
      setActive(-1);
      q.list.textContent = "";
      if (!sx.results.length) {
        q.list.hidden = true;
        q.empty.textContent = "";
        q.empty.appendChild(el("p", null, "No recordings match “" + query.trim() + "”."));
        const pending = sx.waiting.size;
        let hint = "";
        if (pending) hint = "Still indexing " + pending + (pending === 1 ? " transcript…" : " transcripts…");
        else if (!transcribedCount()) hint = "Transcribe a take in the editor to search what was said in it.";
        if (hint) q.empty.appendChild(el("p", null, hint));
        q.empty.hidden = false;
        // aria-expanded mirrors the LISTBOX (hidden here); the message is
        // announced through the live region instead
        q.input.setAttribute("aria-expanded", "false");
        say("No matching recordings.");
        return;
      }
      q.empty.hidden = true;
      q.list.hidden = false;
      const frag = document.createDocumentFragment();
      sx.results.forEach(function (r, i) { frag.appendChild(optionEl(r, i, query)); });
      q.list.appendChild(frag);
      q.input.setAttribute("aria-expanded", "true");
      say(sx.results.length + (sx.results.length === 1 ? " result." : " results."));
    }

    function say(msg) {
      clearTimeout(sx.liveTimer);
      sx.liveTimer = setTimeout(function () { q.live.textContent = msg; }, 500);
    }

    function setActive(i) {
      const opts = q.list.children;
      if (sx.active >= 0 && opts[sx.active]) opts[sx.active].setAttribute("aria-selected", "false");
      sx.active = i;
      if (i >= 0 && opts[i]) {
        opts[i].setAttribute("aria-selected", "true");
        q.input.setAttribute("aria-activedescendant", opts[i].id);
        revealOption(opts[i]);
      } else {
        q.input.removeAttribute("aria-activedescendant");
      }
    }

    // scrolls the LISTBOX so the active option shows, never the page
    function revealOption(node) {
      const pad = 6, top = node.offsetTop, bottom = top + node.offsetHeight;
      if (top - pad < q.list.scrollTop) q.list.scrollTop = Math.max(0, top - pad);
      else if (bottom + pad > q.list.scrollTop + q.list.clientHeight) q.list.scrollTop = bottom + pad - q.list.clientHeight;
    }

    function move(d) {
      flushInput();
      if (!sx.open || sx.stale) runSearch(true);
      if (!sx.results.length) return;
      const n = sx.results.length;
      setActive(sx.active < 0 ? (d > 0 ? 0 : n - 1) : (sx.active + d + n) % n);
    }

    function openPanel() {
      q.panel.hidden = false;
      sx.open = true;
      placePanel();
    }
    function closePanel() {
      q.panel.hidden = true;
      sx.open = false;
      sx.stale = false;
      setActive(-1);
      q.input.setAttribute("aria-expanded", "false");
    }

    // the list gets the room between the bar and the window's bottom edge
    function placePanel() {
      if (!sx.open) return;
      const br = q.bar.getBoundingClientRect();
      const foot = q.panel.querySelector(".lib-q-foot");
      const chrome = (foot ? foot.offsetHeight : 0) + 10 + 16;
      const room = window.innerHeight - br.bottom - chrome;
      q.list.style.maxHeight = Math.round(clamp(room, 180, 440)) + "px";
    }

    function choose(r) {
      if (!r) return;
      const s = sessionByName(r.projectId);
      closePanel();
      if (!s) { toast("That recording is gone. Refresh the library.", "err"); return; }
      if (s.error) { toast("This recording can't be opened: " + s.error, "err"); return; }
      window.location.href = editorUrl(s.session, r.kind === "moment" ? r.t : null);
    }

    function isTypingTarget(t) {
      const tag = t && t.tagName;
      return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || !!(t && t.isContentEditable);
    }

    // --- wiring ---
    // a keystroke not searched yet (debounced) is searched NOW -- before an
    // arrow key moves or Enter opens, so neither acts on the previous query
    function flushInput() {
      if (!sx.inputTimer) return;
      clearTimeout(sx.inputTimer);
      sx.inputTimer = 0;
      runSearch(false);
    }

    q.input.addEventListener("focus", function () {
      if (!sx.started) startIndexing();
      if (Search.hasTerms(q.input.value)) runSearch(false);
    });
    q.input.addEventListener("input", function () {
      if (!sx.started) startIndexing();
      q.bar.classList.toggle("has-value", q.input.value.length > 0);
      clearTimeout(sx.inputTimer);
      sx.inputTimer = 0;
      if (sx.cost < SLOW_SEARCH_MS) { runSearch(false); return; }
      sx.inputTimer = setTimeout(function () { sx.inputTimer = 0; runSearch(false); }, INPUT_DEBOUNCE_MS);
    });
    q.input.addEventListener("keydown", function (e) {
      if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
      else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
      else if (e.key === "Escape") {
        // a pending (debounced) search must not re-open what Esc closes
        clearTimeout(sx.inputTimer);
        sx.inputTimer = 0;
        if (sx.open) { e.preventDefault(); closePanel(); }
        else if (q.input.value) { e.preventDefault(); q.input.value = ""; runSearch(false); }
        else q.input.blur();
      }
    });
    q.form.addEventListener("submit", function (e) {
      e.preventDefault();
      flushInput();
      if (!sx.open || !sx.results.length) runSearch(false);
      choose(sx.results[sx.active >= 0 ? sx.active : 0]);
    });
    // A press anywhere in the form except the input keeps focus IN the input
    // (so its blur can't close the panel before an option's click lands); a
    // press on the pill's padding, icon or "/" hint focuses the input.
    q.form.addEventListener("mousedown", function (e) {
      if (e.target === q.input || e.button !== 0) return;
      e.preventDefault();
      if (document.activeElement !== q.input && q.bar.contains(e.target) && !q.go.contains(e.target)) q.input.focus();
    });
    q.list.addEventListener("click", function (e) {
      const li = e.target.closest ? e.target.closest(".lib-q-opt") : null;
      if (li) choose(sx.results[+li.dataset.i]);
    });
    q.list.addEventListener("mousemove", function (e) {
      const li = e.target.closest ? e.target.closest(".lib-q-opt") : null;
      if (li && +li.dataset.i !== sx.active) setActive(+li.dataset.i);
    });
    // the pointer left the list: apply a re-rank that waited for it
    q.list.addEventListener("pointerleave", function () {
      if (sx.stale && sx.open) runSearch(true);
    });
    q.form.addEventListener("focusout", function () {
      setTimeout(function () { if (sx.open && !q.form.contains(document.activeElement)) closePanel(); }, 0);
    });
    document.addEventListener("pointerdown", function (e) {
      if (sx.open && !q.form.contains(e.target)) closePanel();
    });
    // "/" focuses the search unless the user is typing in another field
    document.addEventListener("keydown", function (e) {
      if (e.key !== "/" || e.metaKey || e.ctrlKey || e.altKey || e.defaultPrevented) return;
      if (isTypingTarget(e.target) || q.wrap.hidden) return;
      e.preventDefault();
      q.input.focus({ preventScroll: true });
      q.input.select();
      // the bar sits near the top of the page: make sure the sticky nav
      // is not covering it
      const navBottom = ui.top ? ui.top.getBoundingClientRect().bottom : 0;
      const r = q.bar.getBoundingClientRect();
      if (r.top < navBottom || r.bottom > window.innerHeight) window.scrollTo(0, 0);
    });
    window.addEventListener("resize", placePanel);
    window.addEventListener("scroll", placePanel, { passive: true });

    return {
      // the grid changed (load, refresh, rename, a finished take)
      sessionsChanged: function () {
        q.wrap.hidden = !state.loaded || searchable().length === 0;
        sx.idx = null;
        if (sx.started) startIndexing();   // picks up sessions that are new
        rerank();
      },
    };
  })();

  // the glass nav darkens and lifts once content scrolls under it
  function syncTopShade() {
    if (ui.top) ui.top.classList.toggle("is-scrolled", window.scrollY > 2);
  }
  window.addEventListener("scroll", syncTopShade, { passive: true });
  syncTopShade();

  document.addEventListener("click", function (ev) {
    if (!ev.target.closest || !ev.target.closest(".menu")) closeMenu();
  });
  ui.refresh.addEventListener("click", loadSessions);
  ui.record.addEventListener("click", openBar);
  ui.recordEmpty.addEventListener("click", openBar);
  ui.permsRefresh.addEventListener("click", loadPermissions);

  loadSessions();
  loadPermissions();
  setInterval(pollState, 1500);
  setInterval(loadPermissions, 8000);
})();
