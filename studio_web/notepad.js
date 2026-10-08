/* notepad.js — the invisible notes overlay.

   Popped out of the recording bar (bar.js toggleNotes) into its own frameless,
   always-on-top window, where the user parks the script for a demo. It floats
   over the very display being captured, so — exactly like the facecam bubble —
   it counts on being a window the BAR PROCESS owns: the recorder's pid sweep
   enumerates it and hands its id to ScreenCaptureKit's content filter, which
   drops it from the take while it stays on the user's own screen. That only
   holds for the SCK ("Hide bar") backend; with avfoundation the window is
   burned in, which is why the bar warns when Notes is opened on that engine.

   Persistence: in the native shell the text and window geometry go through the
   pywebview bridge to files beside the app (bar_native). In a plain browser
   fallback (no bridge) everything falls back to localStorage — the invisibility
   guarantee doesn't apply there anyway. */
"use strict";

(function () {
  const $ = function (id) { return document.getElementById(id); };
  const card = $("notes-card");
  const head = $("notes-head");
  const ta = $("notes-text");
  const grip = $("notes-grip");
  const savedTag = $("notes-saved");

  const isNative = queryParam("native") === "1";
  if (isNative) document.documentElement.classList.add("native");

  const LS_TEXT = "ssnotes.text";
  const LS_SIZE = "ssnotes.size";
  const SIZE_MIN = 11, SIZE_MAX = 28, SIZE_DEFAULT = 15;
  const WIN_MIN_W = 220, WIN_MIN_H = 150;

  function nativeApi() {
    return (window.pywebview && window.pywebview.api) ? window.pywebview.api : null;
  }

  /* Fire-and-forget bridge call. Returns the Promise (so callers that need the
     result can await it) or null when there's no bridge. Swallows a sync throw
     and a rejection — the window may be going away underneath us. */
  function callBridge(name) {
    const bridge = nativeApi();
    if (!bridge || !bridge[name]) return null;
    const args = [].slice.call(arguments, 1);
    try {
      const p = bridge[name].apply(bridge, args);
      if (p && p.catch) p.catch(function () {});
      return p;
    } catch (e) { return null; }
  }

  /* --- content: seed once, then autosave ---------------------------------- */
  let seeded = false;

  function seed(text) {
    // Never clobber what the user has already started — including a paste that
    // beat the async notepad_load round-trip. The input handler latches
    // `seeded` the instant they touch it, and this also bails if the textarea
    // is non-empty for any other reason.
    if (seeded || (ta.value && ta.value.length)) { seeded = true; return; }
    seeded = true;
    ta.value = text || "";
  }

  function loadText() {
    if (isNative) {
      // File is the source of truth in the native shell. If the bridge isn't
      // attached yet, DON'T seed from localStorage — that would latch `seeded`
      // and shadow the file when pywebviewready fires us again.
      const p = callBridge("notepad_load");
      if (p && p.then) {
        p.then(function (r) { seed(r && r.text); }).catch(function () { seed(""); });
      }
      return;
    }
    try { seed(localStorage.getItem(LS_TEXT) || ""); } catch (e) { seed(""); }
  }

  let saveTimer = null;
  function scheduleSave() {
    if (saveTimer) clearTimeout(saveTimer);
    saveTimer = setTimeout(save, 350);
  }
  function save() {
    if (saveTimer) { clearTimeout(saveTimer); saveTimer = null; }
    const text = ta.value;
    if (isNative && callBridge("notepad_save", text)) { flashSaved(); return; }
    try { localStorage.setItem(LS_TEXT, text); } catch (e) { /* private mode */ }
    flashSaved();
  }

  let savedTimer = null;
  function flashSaved() {
    if (!savedTag) return;
    savedTag.textContent = "Saved";
    savedTag.classList.add("show");
    if (savedTimer) clearTimeout(savedTimer);
    savedTimer = setTimeout(function () { savedTag.classList.remove("show"); }, 1100);
  }

  // Latch `seeded` on the first keystroke/paste so a late-resolving
  // notepad_load can't overwrite it (ta.value = ... in seed() fires no input
  // event, so this can't loop).
  ta.addEventListener("input", function () { seeded = true; scheduleSave(); });
  ta.addEventListener("blur", save);

  /* --- font size: a teleprompter you read while presenting ----------------- */
  function clampSize(n) { return Math.max(SIZE_MIN, Math.min(SIZE_MAX, n)); }
  function loadSize() {
    let n = SIZE_DEFAULT;
    try { const v = parseInt(localStorage.getItem(LS_SIZE), 10); if (!isNaN(v)) n = v; }
    catch (e) { /* ignore */ }
    return clampSize(n);
  }
  let fontSize = loadSize();
  function applySize() { ta.style.fontSize = fontSize + "px"; }
  function bumpSize(delta) {
    fontSize = clampSize(fontSize + delta);
    applySize();
    try { localStorage.setItem(LS_SIZE, String(fontSize)); } catch (e) { /* ignore */ }
  }
  applySize();
  $("notes-smaller").addEventListener("click", function () { bumpSize(-1); });
  $("notes-bigger").addEventListener("click", function () { bumpSize(1); });

  /* --- close --------------------------------------------------------------- */
  function closeNotes() {
    save();
    if (isNative && callBridge("notepad_close")) return;
    try {
      if (window.opener && window.opener.__barNotesClosed) window.opener.__barNotesClosed();
    } catch (e) { /* same-origin, shouldn't happen */ }
    window.close();
  }
  $("notes-close").addEventListener("click", closeNotes);

  /* --- drag + resize (native only) ----------------------------------------
     The header is the sole drag region; the grip drives a native-frame resize.
     BOTH the grip-start and the control guards live in this one capture-phase
     handler, and that is load-bearing: `stopPropagation()` here halts the event
     before it reaches the target, so a separate `grip.addEventListener` would
     never fire (the exact trap bar.js documents for its window <select>). */
  let resizing = false, lastW = 0, lastH = 0;
  document.addEventListener("mousedown", function (ev) {
    if (!isNative) return;
    if (!ev.target.closest) return;
    // The resize grip: begin sizing the window ourselves. Frameless WKWebView
    // windows don't reliably offer native edge-resize; the pointer inside the
    // window IS the new bottom-right corner (frameless == 1:1 CSS px to
    // points), and notepad_resize keeps the top-left anchored.
    if (ev.target.closest(".notes-grip")) {
      ev.preventDefault();           // no text selection while dragging
      resizing = true;
      lastW = lastH = 0;
      document.body.style.cursor = "nwse-resize";
      ev.stopPropagation();
      return;
    }
    // Other controls (the header buttons) must eat the event so pywebview's
    // drag handler — which walks UP from the target — doesn't move the window
    // instead of clicking them.
    if (ev.target.closest("button")) {
      ev.stopPropagation();
      return;
    }
    if (ev.target.closest(".notes-head")) head.classList.add("dragging");
  }, true);
  window.addEventListener("mouseup", function () {
    if (!head.classList.contains("dragging")) return;
    head.classList.remove("dragging");
    callBridge("notepad_moved");     // persist the new position
  });

  window.addEventListener("mousemove", function (ev) {
    if (!resizing) return;
    const w = Math.max(WIN_MIN_W, Math.round(ev.clientX + 3));
    const h = Math.max(WIN_MIN_H, Math.round(ev.clientY + 3));
    if (w === lastW && h === lastH) return;
    lastW = w; lastH = h;
    callBridge("notepad_resize", w, h);
  });
  window.addEventListener("mouseup", function () {
    if (!resizing) return;
    resizing = false;
    document.body.style.cursor = "";
    callBridge("notepad_moved");     // persist final geometry (incl. size)
  });

  // in a browser popup there's no native frame to resize/drag ourselves.
  // Inline display, not the [hidden] attribute: .notes-grip carries an author
  // `display: flex`, which beats the UA `[hidden] { display: none }` rule
  // (same trap the bar's window picker documents).
  if (!isNative) grip.style.display = "none";

  window.addEventListener("pagehide", save);

  /* The pywebview bridge isn't attached until pywewviewready; seed then. Also
     try immediately in case it's already up (a live-reload of this page). */
  if (isNative) window.addEventListener("pywebviewready", loadText);
  loadText();
})();
