/* bar.js — the floating recording pill.
   States: idle -> countdown -> recording -> done (open editor) driven by
   POST /api/record/start|stop plus GET /api/state polling. */
"use strict";

(function () {
  const $ = function (id) { return document.getElementById(id); };
  const faces = {
    idle: $("face-idle"),
    countdown: $("face-countdown"),
    recording: $("face-recording"),
    done: $("face-done"),
  };
  const ui = {
    bar: $("bar"),
    display: $("bar-display"),
    window: $("bar-window"),
    windowBtn: $("bar-window-btn"),
    windowWrap: $("bar-window-wrap"),
    occlusion: $("bar-occlusion"),
    occlusionWrap: $("bar-occlusion-wrap"),
    occlusionLabel: $("bar-occlusion-label"),
    autoadd: $("bar-autoadd"),
    autoaddWrap: $("bar-autoadd-wrap"),
    autoaddLabel: $("bar-autoadd-label"),
    mic: $("bar-mic"),
    cursor: $("bar-cursor"),
    camera: $("bar-camera"),
    camSlot: $("bar-cam-slot"),
    camPreview: $("bar-cam-preview"),
    camDetached: $("bar-cam-detached"),
    camVideo: $("bar-cam-video"),
    camImg: $("bar-cam-img"),
    countdown: $("bar-countdown"),
    backend: $("bar-backend"),
    record: $("bar-record"),
    cancel: $("bar-cancel"),
    stop: $("bar-stop"),
    pause: $("bar-pause"),
    pauseGlyph: $("bar-pause-glyph"),
    resumeGlyph: $("bar-resume-glyph"),
    repick: $("bar-repick"),
    growChips: $("bar-grow-chips"),
    liveDot: $("bar-live-dot"),
    count: $("bar-count"),
    countNote: $("bar-count-note"),
    timer: $("bar-timer"),
    recNote: $("bar-rec-note"),
    doneNote: $("bar-done-note"),
    openEditor: $("bar-open-editor"),
    again: $("bar-again"),
    close: $("bar-close"),
    notes: $("bar-notes"),
    hint: $("bar-hint"),
  };

  const state = {
    face: "idle",
    recordingSince: null,   // client-side elapsed anchor
    doneSession: null,
    pollTimer: null,
    permissionsOk: true,
    busy: false,
    firstPoll: true,        // absorb pre-existing server state without acting on it
    errorKey: null,         // dedupe: last error we already surfaced
    recStatus: "idle",      // last record status seen (drives the pause toggle)
    faceMode: "docked",     // "docked" (bubble in the pill) | "floating"
    faceReleased: false,    // camera handed to the capture ffmpeg
    windowLabel: "",        // picked window(s), named on the countdown/recording faces
    windowIds: [],          // overlay-picker selection, in card order
    windowArrange: false,   // un-overlap them on screen before the take
    occlusionFree: false,   // record just the one window's own buffer (SCK)
    autoAddWindows: false,  // occlusion-free: auto-JOIN every window raised mid-take
    // The PERSISTED preference (/api/settings), tri-state: true / false /
    // null = never chosen. Null is what makes auto-add default ON without
    // overriding someone who deliberately turned it off.
    autoAddPref: null,
    notesOpen: false,       // the invisible notes overlay is up
  };

  // native mode: hosted in a frameless pywebview window (transparent
  // backdrop, drag-anywhere pill, pywebview bridge)
  const isNative = queryParam("native") === "1";
  if (isNative) document.documentElement.classList.add("native");

  function nativeApi() {
    return (window.pywebview && window.pywebview.api) ? window.pywebview.api : null;
  }

  /* pywebview bridge calls return a promise that rejects if the Python side
     raised (or the window is already going away). These are all
     fire-and-forget, so swallow both the sync throw and the rejection. */
  function bridgeCall(name) {
    const bridge = nativeApi();
    if (!bridge || !bridge[name]) return false;
    const args = [].slice.call(arguments, 1);
    try {
      const p = bridge[name].apply(bridge, args);
      if (p && p.catch) p.catch(function () {});
    } catch (e) { return false; }
    return true;
  }

  // fill picker icons from shared ICONS
  document.querySelectorAll(".bar-picker-icon[data-icon]").forEach(function (span) {
    span.innerHTML = ICONS[span.getAttribute("data-icon")] || "";
  });

  function showFace(name) {
    state.face = name;
    Object.keys(faces).forEach(function (k) { faces[k].hidden = k !== name; });
    reportFit();
  }

  /* A take is in flight — the pill is on screen and inside the capture. The
     server's "stopping" status rides the recording face (see applyRecord), so
     the two faces cover countdown/recording/stopping. */
  function recordingInFlight() {
    return state.face === "countdown" || state.face === "recording";
  }

  function setHint(text, isErr) {
    ui.hint.hidden = !text;
    ui.hint.textContent = text || "";
    ui.hint.classList.toggle("err", !!isErr);
    reportFit();
  }

  /* The pill's resting message. Picking a window parks a caveat there, so
     "clear the hint" has to mean "back to the resting message" — every blanking
     path goes through here or the caveat gets wiped (loadPermissions alone
     clears the strip every 5s whenever it's happy). */
  function idleHint() {
    const n = currentWindowIds().length;
    if (!n) return "";
    if (state.occlusionFree) {
      // Occlusion-free now spans 1-4 windows: one window's own buffer, or N
      // buffers composited (no inter-window overlap). Either way the pill
      // stays out of the video.
      return n > 1 ? WINDOW_NOTE_OCCLUSION_FREE_MULTI : WINDOW_NOTE_OCCLUSION_FREE;
    }
    return n > 1 ? WINDOW_NOTE_MULTI : WINDOW_NOTE;
  }
  function clearHint() {
    // loadPermissions owns the strip while it's unhappy — a missing grant
    // outranks anything else we'd say there
    if (!state.permissionsOk) return;
    setHint(idleHint());
  }

  /* --- native window fitting ---
   * The pill's width swings from ~682px (idle) to ~245px (recording). A fixed
   * window would leave the rest as invisible, click-eating dead space over
   * whatever is behind it, so we hand the measured content to the Python side
   * and it resizes the window + masks everything else click-through. */
  let lastFit = "";
  /* Every fit carries this page's id and a counter. pywebview runs each
     bridge call on its own thread, so two fits sent in one tick (showFace +
     setHint) can reach Python in either order; bar_fit serializes them and
     drops one that is older than a fit it already applied, so the window,
     click-through rects and glass all settle on the LAST size sent here.
     The page id makes a reload start a fresh count. */
  const fitPage = Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
  let fitSeq = 0;
  function reportFit() {
    if (!isNative) return;
    const pill = ui.bar.getBoundingClientRect();
    const hint = ui.hint.hidden ? null : ui.hint.getBoundingClientRect();
    const dims = [
      Math.ceil(pill.width), Math.ceil(pill.height),
      hint ? Math.ceil(hint.width) : 0, hint ? Math.ceil(hint.height) : 0,
    ];
    const key = dims.join(",");
    if (key === lastFit) return;   // also breaks any resize feedback loop
    const bridge = nativeApi();
    if (!bridge || !bridge.bar_fit) return;
    let reply;
    try { reply = bridge.bar_fit(dims[0], dims[1], dims[2], dims[3], fitPage, ++fitSeq); }
    catch (e) { return; }
    // only latch once the bridge actually took it, so a pre-ready call gets
    // retried by the observer / pywebviewready instead of being swallowed
    lastFit = key;
    if (reply && reply.then) reply.then(applyGlass, function () {});
  }

  /* Native frosted glass. CSS backdrop-filter can't reach the desktop behind
     a transparent native window, so bar_native puts real NSVisualEffectViews
     under the pill and says so in bar_fit's reply. Only then does the pill go
     translucent (html.glass in bar.css): no reply, a failed install or
     AUTOCINE_NO_VIBRANCY=1 all keep today's opaque pill. A reply only ever
     ADDS the class: replies can land out of order, and a lazy install's
     first reply says "no glass" while the views are still being built
     (bar_native then adds the class itself, GLASS_ON_JS). Taking it away is
     bar_native's job alone (GLASS_LOST_JS), which also sets
     window.autocineGlassLost so a late reply can't put it back. */
  function applyGlass(res) {
    if (!res || !res.ok || !res.glass || window.autocineGlassLost) return;
    document.documentElement.classList.add("glass");
  }
  if (isNative) {
    // selects widen once real device names land, the hint wraps to two lines,
    // faces swap — just watch the boxes rather than chasing every cause.
    // reportFit reads layout synchronously, so no rAF hop (which a hidden or
    // occluded window would throttle).
    if (window.ResizeObserver) {
      const ro = new ResizeObserver(reportFit);
      ro.observe(ui.bar);
      ro.observe(ui.hint);
    }
    window.addEventListener("pywebviewready", function () {
      lastFit = "";
      reportFit();
    });
  }

  /* --- drag the window from anywhere on the pill ---
   * .bar-pill carries pywebview's .pywebview-drag-region, whose handler walks
   * up from the event target — so without this guard a mousedown on a select
   * or button would drag the window instead of operating the control. Killing
   * the event in the capture phase keeps it from ever reaching pywebview's
   * body-level listener; the control's own default action still happens. */
  const NO_DRAG = "button, select, input, textarea, a, [contenteditable]";
  document.addEventListener("mousedown", function (ev) {
    if (!ev.target || !ev.target.closest) return;
    // Refresh the window list as its picker opens. This has to live HERE and
    // not as a listener on the <select>: the guard below calls
    // stopPropagation() in the CAPTURE phase at document, which ends the
    // event's path — including the target's own listeners.
    if (ev.target === ui.window) loadWindows();
    if (!isNative) return;
    if (ev.target.closest(NO_DRAG)) {
      ev.stopPropagation();
      return;
    }
    // only the pill drags — a press on the transparent slack around it isn't
    // one (pywebview's handler ignores it too)
    if (ev.target.closest(".bar-pill")) ui.bar.classList.add("dragging");
  }, true);
  window.addEventListener("mouseup", function () {
    if (!isNative || !ui.bar.classList.contains("dragging")) return;
    ui.bar.classList.remove("dragging");
    bridgeCall("bar_moved");
  });

  async function loadDevices() {
    try {
      const d = await api("/api/devices");
      ui.display.innerHTML = "";
      const auto = el("option", null, "Auto");
      auto.value = "";
      ui.display.appendChild(auto);
      // d.video is the raw avfoundation VIDEO device list — webcams and screens
      // share it ([0] FaceTime HD Camera, [1] Capture screen 0 here), and this
      // picker is displays only. Same test devices.find_screen_device uses, so
      // "Auto" and the explicit picks agree on what counts as a screen.
      (d.video || []).filter(function (v) {
        return String(v.name || "").toLowerCase().indexOf("capture screen") !== -1;
      }).forEach(function (v) {
        const o = el("option", null, v.name);
        o.value = String(v.index);
        ui.display.appendChild(o);
      });
      ui.mic.innerHTML = "";
      const none = el("option", null, "No mic");
      none.value = "none";
      ui.mic.appendChild(none);
      (d.audio || []).forEach(function (a) {
        const o = el("option", null, a.name);
        o.value = String(a.index);
        ui.mic.appendChild(o);
      });
    } catch (e) {
      setHint("Could not list devices: " + e.message, true);
    }
    reportFit();   // real device names widen the selects
  }

  /* --- window capture ---
   * avfoundation cannot target a window, so a pick is a CROP: we record the
   * whole display and render.py crops to the window. Its geometry is polled
   * during the take, so the crop follows a window that is moved or resized
   * (see the geometry track in docs/architecture.md) -- what it cannot follow is
   * occlusion, since the pixels of anything drawn on top are what the display
   * capture saw. Hence the remaining caveat, and hence its own picker rather
   * than extra entries in the Display list. */
  const WINDOW_NOTE =
    "Window capture crops the full-screen recording to this window, following " +
    "it if you move or resize it. Anything drawn over the window, including " +
    "this pill, is still inside the crop.";
  const WINDOW_NOTE_MULTI =
    "Recording several windows: they're composited onto a background keeping " +
    "their on-screen arrangement. Auto-zoom is off in this mode.";
  const WINDOW_NOTE_OCCLUSION_FREE =
    "Occlusion-free: recording just this window's own pixels, so anything in " +
    "front of it, including this pill, stays out of the video. Uses the " +
    "Hide-bar engine.";
  const WINDOW_NOTE_OCCLUSION_FREE_MULTI =
    "Occlusion-free: each window recorded as its own pixels (nothing in front " +
    "of any of them appears), then composited. Uses the Hide-bar engine.";
  const WINDOW_NOTE_MAX = 32;    // the faces are narrow; titles run to ~80 chars

  /* --- the hover picker ---
   * The real UI is a full-screen overlay window (studio_app.pick_windows):
   * hover a window, it lights up, click to toggle, up to four. The <select>
   * is kept as an automatic fallback for runtimes that can't host a second
   * native window -- the browser app-mode bar, or a pywebview that refuses.
   * useOverlay is decided ONCE at boot and never flips, so the pill can't
   * change shape under the user mid-session.
   *
   * state.windowIds is the source of truth in overlay mode; the <select>'s
   * value is the source of truth in fallback mode. currentWindowIds() is the
   * single place that knows which. */
  let useOverlay = false;
  let pickerOpen = false;

  function currentWindowIds() {
    if (useOverlay) return state.windowIds.slice();
    return ui.window.value === "" ? [] : [parseInt(ui.window.value, 10)];
  }

  function windowBtnLabel(ids) {
    if (!ids.length) return "Full screen";
    if (ids.length === 1) {
      const w = windowById(ids[0]);
      return shortLabel(w ? (w.label || w.app || ("Window " + w.id))
                          : "1 window");
    }
    return ids.length + " windows";
  }

  function windowById(id) {
    for (let i = 0; i < windowList.length; i++) {
      if (windowList[i].id === id) return windowList[i];
    }
    return null;
  }

  function syncWindowBtn() {
    const ids = state.windowIds;
    ui.windowBtn.textContent = windowBtnLabel(ids);
    ui.windowBtn.classList.toggle("is-set", ids.length > 0);
    state.windowLabel = ids.length ? windowBtnLabel(ids) : "";
    syncOcclusionToggle();
    setHint(idleHint());
    reportFit();
  }

  /* The occlusion-free toggle makes sense for 1-4 windows on the SCK engine
     (P3.4): one window's own buffer, or N buffers composited. It is shown
     whenever at least one window is picked (and the engine isn't env-locked
     to Standard). Hiding it also CLEARS it, so Full screen can never smuggle
     a stale occlusion_free into the next take. */
  function syncOcclusionToggle() {
    if (!ui.occlusionWrap) return;
    const n = currentWindowIds().length;
    const anyWindow = n >= 1 && n <= 4;
    const lockedOff = !!(ui.backend && ui.backend.disabled &&
                         ui.backend.value !== "sck");
    const show = anyWindow && !lockedOff;
    ui.occlusionWrap.hidden = !show;
    ui.occlusionWrap.style.display = show ? "" : "none";
    if (!show && state.occlusionFree) {
      state.occlusionFree = false;
      if (ui.occlusion) ui.occlusion.checked = false;
    }
    ui.occlusionWrap.classList.toggle("is-on", state.occlusionFree);
    // Label reads singular for one window, plural for a multi-native pick.
    if (ui.occlusionLabel) {
      ui.occlusionLabel.textContent =
        n > 1 ? "Only these windows" : "Only this window";
    }
    // Arrange physically moves the user's real windows to un-overlap them --
    // pointless for a native take (each window is recorded as its own buffer,
    // so on-screen overlap doesn't matter) and the server forces it off. Clear
    // the state so the payload's `arrange` field is false; the server is the
    // authority, but sending a stale true would just be noise.
    if (state.occlusionFree && state.windowArrange) {
      state.windowArrange = false;
    }
    // Auto-add is a sub-option of occlusion-free, so it follows it here (after
    // the block above may have cleared occlusionFree).
    syncAutoAddToggle();
  }

  /* Auto-add is a sub-option of occlusion-free: while it's on, every window
     brought to the front DURING the take is joined automatically (see
     maybeAutoAdd -- "brought to the front", because opening Finder or a
     document usually RAISES an existing window). It is
     shown ONLY while occlusion-free is on, because growing is a fleet-only
     capability; hiding it also CLEARS it, so a plain (occluding) window pick
     can never smuggle a stale auto-add into the next take -- the exact rule
     syncOcclusionToggle applies to occlusion_free itself. */
  function syncAutoAddToggle() {
    if (!ui.autoaddWrap) return;
    const show = !!state.occlusionFree;
    const wasShown = !ui.autoaddWrap.hidden;
    ui.autoaddWrap.hidden = !show;
    ui.autoaddWrap.style.display = show ? "" : "none";
    if (!show && state.autoAddWindows) {
      state.autoAddWindows = false;
      if (ui.autoadd) ui.autoadd.checked = false;
    }
    // Turning occlusion-free ON arms auto-add from the saved preference --
    // and DEFAULTS IT ON when there is none. Recording the windows you bring
    // up is the behavior people expect of occlusion-free mode; making them
    // find a second checkbox for it meant takes that silently missed windows.
    // Only the transition arms it, so an explicit uncheck survives every
    // later re-sync within the same session.
    if (show && !wasShown) {
      state.autoAddWindows = (state.autoAddPref === null)
        ? true : !!state.autoAddPref;
      if (ui.autoadd) ui.autoadd.checked = state.autoAddWindows;
    }
    ui.autoaddWrap.classList.toggle("is-on", state.autoAddWindows);
  }

  function openOverlayPicker() {
    if (pickerOpen) return;
    const bridge = nativeApi();
    if (!bridge || !bridge.pick_windows) return;
    pickerOpen = true;
    Promise.resolve(bridge.pick_windows()).then(function (res) {
      if (res && res.ok === false) {
        // Couldn't open. Fall back permanently rather than leaving a button
        // that does nothing -- a broken overlay must never cost a recording.
        pickerOpen = false;
        useOverlay = false;
        ui.windowBtn.hidden = true;
        ui.window.hidden = false;
        setHint("Couldn't open the window picker. Using the list instead.", true);
        loadWindows();
      }
    }).catch(function () { pickerOpen = false; });
  }

  /* Called by Python when the picker confirms. */
  window.__barWindowsPicked = function (result) {
    pickerOpen = false;
    const r = result || {};
    // Mid-take RE-PICK (scene takes): the picker was opened from the PAUSED
    // face, so the ids are the next scene's window set — send them to
    // resume, and leave the idle-face pick state alone (it belongs to the
    // NEXT take, not this one). An empty confirm is a cancel, not a resume.
    if (state.recStatus === "paused") {
      const ids = (r.ids || []).map(Number).filter(function (n) {
        return !isNaN(n);
      });
      if (ids.length) {
        api("/api/record/resume", { body: { window_ids: ids } })
          .then(pollFast)
          .catch(function (e) { setHint(e.message, true); });
      }
      return;
    }
    state.windowIds = (r.ids || []).map(Number).filter(function (n) {
      return !isNaN(n);
    });
    state.windowArrange = !!r.arrange;
    syncWindowBtn();
  };

  /* Called by Python when the picker is dismissed with no change. */
  window.__barPickerClosed = function () { pickerOpen = false; };

  /* bar.css gives .bar-picker `display: flex`, and an author rule beats the UA
     `[hidden]` rule — so the attribute alone does NOT hide the wrapper. Both
     live here so they can never disagree. */
  function showWindowPicker(on) {
    ui.windowWrap.hidden = !on;
    ui.windowWrap.style.display = on ? "" : "none";
  }

  function shortLabel(text) {
    const s = String(text || "").trim();
    return s.length > WINDOW_NOTE_MAX
      ? s.slice(0, WINDOW_NOTE_MAX - 3).trim() + "..."
      : s;
  }

  function currentWindowLabel() {
    const opt = ui.window.options[ui.window.selectedIndex];
    return ui.window.value === "" || !opt ? "" : opt.textContent;
  }

  /* Best-effort "open the dropdown" after a stale pick was rejected. A native
     popup can't be opened programmatically (showPicker isn't in WKWebView, and
     it needs user activation anyway) — focus at least lands on the control. */
  function openWindowPicker() {
    try {
      ui.window.focus();
      if (ui.window.showPicker) ui.window.showPicker();
    } catch (e) { /* no activation — focus is enough */ }
  }

  let windowList = [];       // last /api/windows payload, front-to-back
  let windowSig = "";        // id+label list we last rendered
  let windowShown = false;
  let windowLoading = false;

  /* Refetched every time the picker is opened, never polled: the list is stale
     within seconds (any move, close, minimize or Space switch), which is also
     why the server keeps it out of /api/devices. */
  async function loadWindows() {
    if (windowLoading) return;
    windowLoading = true;
    let list, available;
    try {
      const r = await api("/api/windows");
      list = (r && r.windows) || [];
      windowList = list;
      available = !!(r && r.available);
    } catch (e) {
      return;                // transient — keep whatever the picker shows
    } finally {
      windowLoading = false;
    }
    if (!available) {
      // no Quartz: the feature can never work in this runtime, so don't offer
      // a dropdown that can only ever say "Full screen"
      windowShown = false;
      windowSig = "";
      ui.window.value = "";
      showWindowPicker(false);
      reportFit();
      return;
    }
    // Nothing pickable yet — stay out of the pill's width. Once shown we keep
    // the control: yanking it out from under the click that opened it would be
    // worse than a lonely "Full screen".
    if (!list.length && !windowShown) return;

    const sig = list.map(function (w) {
      return String(w.id) + ":" + (w.label || "");
    }).join("\n");
    if (!useOverlay && sig !== windowSig) {
      // rebuilding is what can disturb an already-open popup, so only do it
      // when the list actually changed
      windowSig = sig;
      const want = ui.window.value;
      ui.window.innerHTML = "";
      const full = el("option", null, "Full screen");
      full.value = "";
      ui.window.appendChild(full);
      // z-order, front-to-back — the server's order is the one a picker wants
      list.forEach(function (w) {
        const o = el("option", null, w.label || w.app || ("Window " + w.id));
        o.value = String(w.id);
        o.title = o.textContent;      // the select clips at 120px
        ui.window.appendChild(o);
      });
      // Keep the pick only if that window is still up. Assigning a value no
      // option carries leaves selectedIndex at -1 — a blank box, not a
      // fallback to "Full screen" — so decide explicitly.
      const kept = want !== "" && list.some(function (w) {
        return String(w.id) === want;
      });
      ui.window.value = kept ? want : "";
      if (want !== "" && !kept) {
        setHint("That window is no longer on screen. Back to Full screen.", true);
      }
    }
    windowShown = true;
    showWindowPicker(true);
    if (useOverlay) {
      // Drop picks whose window has gone, so Start can never be pressed on
      // a selection the server would reject.
      const before = state.windowIds.length;
      state.windowIds = state.windowIds.filter(function (id) {
        return !!windowById(id);
      });
      if (state.windowIds.length !== before) {
        setHint("A picked window is no longer on screen. It was dropped.", true);
      }
      syncWindowBtn();
    }
    reportFit();   // an unhidden picker (and real labels) widen the pill
  }

  // --- Facecam preview (browser getUserMedia; separate from the backend
  // avfoundation capture). Privacy-first: default "No camera", so we never
  // touch the webcam until the user explicitly picks one. ---
  let camStream = null;
  // The device we last successfully opened. The <select>'s value is not a
  // reliable record of this — its option list gets rebuilt when permission
  // lands and the value can silently reset to "none" (see selectCamera) —
  // so placement decisions read this instead.
  let camDeviceId = null;

  /* Where preview frames come from.
   *
   * The native shell ALWAYS uses Python: device list from /api/devices,
   * frames as MJPEG in an <img>. Two reasons, and the second is the one that
   * bites users:
   *
   * 1. WKWebView exposes no navigator.mediaDevices until the host app holds a
   *    camera grant, and none at all before that — so the browser path can't
   *    be relied on, and used to hide the camera picker outright, silently
   *    removing the feature from the native app.
   * 2. Even once it IS exposed, getUserMedia makes WebKit raise its own
   *    per-origin prompt ("Allow 127.0.0.1 to use your camera?") on every new
   *    web view. The app already holds the OS-level grant it needs to record;
   *    asking again per launch is pure noise.
   *
   * Real browsers keep the getUserMedia path — there Python has no special
   * standing and the prompt is the browser's own, asked once per site. */
  const hasMediaDevices = !isNative &&
                          !!(navigator.mediaDevices &&
                             navigator.mediaDevices.enumerateDevices &&
                             navigator.mediaDevices.getUserMedia);

  async function loadCameras() {
    const prev = ui.camera.value;
    ui.camera.innerHTML = "";
    const none = el("option", null, "No camera");
    none.value = "none";
    ui.camera.appendChild(none);
    try {
      if (hasMediaDevices) {
        const devs = await navigator.mediaDevices.enumerateDevices();
        devs.filter(function (d) { return d.kind === "videoinput"; })
          .forEach(function (c, i) {
            // labels are blank until a getUserMedia grant exists
            const o = el("option", null, c.label || ("Camera " + (i + 1)));
            o.value = c.deviceId || String(i);
            ui.camera.appendChild(o);
          });
      } else {
        const d = await api("/api/devices");
        (d.cameras || []).forEach(function (c) {
          const o = el("option", null, c.name);
          o.value = "cv:" + c.ordinal;    // OpenCV ordinal, not the avf index
          ui.camera.appendChild(o);
        });
      }
    } catch (e) { /* leave just "No camera" */ }
    selectCamera(prev);
    reportFit();   // real camera labels widen the select
  }

  /* Re-select a camera after the option list was rebuilt.
   *
   * This is subtler than it looks and got it wrong once: pre-permission,
   * enumerateDevices() returns blank deviceIds so the options carry synthetic
   * list indices ("0", "1"); once a grant exists they carry real ids. A plain
   * value match across that rebuild therefore always misses, the select
   * silently falls back to its first option ("No camera") while the camera is
   * still streaming, and anything keyed off ui.camera.value quietly does
   * nothing. Match the real id first, then fall back to the list position. */
  function selectCamera(want) {
    if (!want || want === "none") return false;
    const opts = [].slice.call(ui.camera.options);
    let hit = opts.filter(function (o) { return o.value === want; })[0];
    if (!hit && /^\d+$/.test(want)) {
      hit = opts.filter(function (o) { return o.value !== "none"; })[parseInt(want, 10)];
    }
    if (!hit) return false;
    ui.camera.value = hit.value;
    return true;
  }

  /* Tear down the PICTURE only. getUserMedia tracks really do stop here;
     the MJPEG <img> does NOT — see releaseCam for why dropping the src is
     not a release, and never treat this function as one. */
  function stopCam() {
    if (camStream) {
      camStream.getTracks().forEach(function (t) { t.stop(); });
      camStream = null;
    }
    ui.camVideo.srcObject = null;
    if (ui.camImg) {
      // Clear onerror BEFORE ending the load: we are the ones ending it, and
      // the handler would report that to the user as "Camera unavailable".
      ui.camImg.onerror = null;
      ui.camImg.removeAttribute("src");
      ui.camImg.hidden = true;
    }
  }

  /* Actually give the webcam back.

     Dropping the <img>'s src does NOT end a multipart/x-mixed-replace load
     in WKWebView. The socket stays open, so Python's client refcount never
     falls and the camera stays lit with nothing on screen using it. Measured
     on the native bar: one preview connection survived a recording, the done
     face, AND an explicit "No camera" pick — 1.6 GB streamed, camera on the
     whole time. The old comment here claimed the opposite; it was never
     true in the native shell, and every "release" path in this file was
     resting on it.

     So the release is server-side: /api/camera/release shuts the capture
     down, which ends the generator, which closes the socket from the far
     end. Deliberately NOT called from startCam (see the release-then-open
     sequencing there) — a fire-and-forget release racing a fresh open would
     kill the stream it just started. */
  function releaseCam() {
    stopCam();
    return api("/api/camera/release", { body: {} }).catch(function () {});
  }

  /* off | docked (bubble lives in the pill) | floating (its own window) */
  function showCamSlot(mode) {
    ui.camSlot.hidden = mode === "off";
    ui.camPreview.hidden = mode !== "docked";
    ui.camDetached.hidden = mode !== "floating";
    ui.camSlot.classList.toggle("detached", mode === "floating");
    ui.camSlot.title = mode === "floating"
      ? "Facecam is floating on screen. Click to dock it back in the bar."
      : "Facecam preview. Click to pop it out of the bar.";
    reportFit();
  }

  function camFailed(err) {
    stopCam();
    camDeviceId = null;
    ui.camera.value = "none";
    showCamSlot("off");
    setHint("Camera unavailable (" + (err.name || err.message || "denied") + ")", true);
  }

  async function startCam(deviceId) {
    stopCam();
    if (!deviceId || deviceId === "none") return;
    if (!hasMediaDevices || String(deviceId).indexOf("cv:") === 0) {
      // Python owns the camera; point the <img> at its MJPEG stream.
      const ordinal = String(deviceId).replace(/^cv:/, "") || "0";
      // RELEASE THEN OPEN, and await it. A previous <img> load can still be
      // running server-side even though we dropped its src (WKWebView does
      // not abort it — see releaseCam), so without this every re-open would
      // strand another reader on the capture and the refcount could never
      // fall to zero again. Awaiting is what keeps it from racing the open
      // below and killing the stream we are about to start.
      await api("/api/camera/release", { body: {} }).catch(function () {});
      ui.camVideo.hidden = true;
      ui.camImg.hidden = false;
      // a failed stream (no Camera grant, device busy) must say why rather
      // than sit there as a broken image
      ui.camImg.onerror = function () {
        ui.camImg.onerror = null;
        api("/api/camera/preview?ordinal=" + encodeURIComponent(ordinal))
          .then(function () { camFailed(new Error("preview ended")); })
          .catch(function (e) { camFailed(e); });
      };
      ui.camImg.src = autocineUrl(
        "/api/camera/preview?ordinal=" + encodeURIComponent(ordinal) +
        "&t=" + Date.now());         // a fresh URL per open, never a cached one
      camDeviceId = deviceId;
      showCamSlot("docked");
      clearHint();
      return;
    }
    ui.camImg.hidden = true;
    ui.camVideo.hidden = false;
    const wantExact = deviceId && deviceId !== "none";
    try {
      const video = wantExact ? { deviceId: { exact: deviceId } } : true;
      camStream = await navigator.mediaDevices.getUserMedia({ video: video, audio: false });
    } catch (e) {
      if (wantExact && e.name === "OverconstrainedError") {
        // loadCameras() ran pre-permission, so the picker was built from
        // enumerateDevices() entries with blank deviceId -> deviceId fell back
        // to a synthetic list index (see loadCameras) that no real camera
        // matches. Grab an unconstrained stream to obtain the permission
        // grant, then re-enumerate (now with real ids) and retry against the
        // device at that same list position.
        try {
          camStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
          const idx = parseInt(deviceId, 10);
          const devs = await navigator.mediaDevices.enumerateDevices();
          const cams = devs.filter(function (d) { return d.kind === "videoinput"; });
          const real = cams[idx] && cams[idx].deviceId;
          if (real) {
            camStream.getTracks().forEach(function (t) { t.stop(); });
            camStream = await navigator.mediaDevices.getUserMedia({
              video: { deviceId: { exact: real } }, audio: false
            });
          }
        } catch (e2) {
          camFailed(e2);
          return;
        }
      } else {
        camFailed(e);
        return;
      }
    }
    ui.camVideo.srcObject = camStream;
    showCamSlot("docked");
    clearHint();
    // permission now granted -> real device labels/ids are available
    const want = ui.camera.value;
    // the id of the device we actually opened beats any list bookkeeping
    let live = null;
    try {
      const track = camStream.getVideoTracks()[0];
      live = track && track.getSettings ? track.getSettings().deviceId : null;
    } catch (e) { /* getSettings is best-effort */ }
    await loadCameras();
    if (!selectCamera(live)) selectCamera(want);
    camDeviceId = ui.camera.value !== "none" ? ui.camera.value : (live || want);
  }

  /* --- facecam placement: in the pill, or floating in its own window ---
   * Only one process may hold the webcam at a time, so the two placements
   * hand it back and forth: the pill drops its stream before the floating
   * window opens, and re-acquires after it closes. */
  let faceWin = null;          // browser-fallback popup (non-native mode)

  async function floatFacecam() {
    // Guard on whether a camera is actually LIVE, not on the select's value.
    // The value can drift out from under us when the option list is rebuilt
    // (see selectCamera), and keying the pop-out off it made this button a
    // silent no-op while the bubble sat there streaming.
    if (!camStream && !camDeviceId) return;
    // await: the floating window opens its OWN stream right after, and a
    // release still in flight would tear that one down instead.
    await releaseCam();
    showCamSlot("floating");
    state.faceMode = "floating";
    const bridge = nativeApi();
    let ok = false, why = "";
    if (bridge && bridge.facecam_float) {
      try {
        // tell the floating window which camera to stream
        const ordinal = String(camDeviceId || "").replace(/^cv:/, "") || "0";
        const r = await bridge.facecam_float(ordinal);
        ok = !!(r && r.ok);
        if (!ok) why = (r && r.reason) || "";
      } catch (e) { ok = false; why = e.message || ""; }
    } else {
      // no native bridge (Chrome app-mode / plain tab): a small popup window
      const s = 168;
      faceWin = window.open("/facecam.html", "autocine-facecam",
        "popup=yes,width=" + s + ",height=" + s +
        ",left=" + (screen.availWidth - s - 48) +
        ",top=" + (screen.availHeight - s - 160));
      ok = !!faceWin;
      if (!ok) setHint("Popup blocked. Allow popups to float the facecam.", true);
    }
    if (!ok) {                 // couldn't detach — put it back in the pill
      state.faceMode = "docked";
      showCamSlot("docked");
      // say so: silently snapping back reads as "the button does nothing"
      setHint("Could not float the facecam" + (why ? " (" + why + ")" : "") +
        ". Keeping it in the bar.", true);
      startCam(ui.camera.value);
    }
  }

  function dockFacecam() {
    // set the mode first: the native side echoes back through
    // __barFacecamDocked and this makes that a no-op instead of a re-entry
    state.faceMode = "docked";
    const want = camDeviceId || ui.camera.value;
    const off = !want || want === "none";
    showCamSlot(off ? "off" : "docked");
    if (!bridgeCall("facecam_dock") && faceWin && !faceWin.closed) {
      faceWin.close();
    }
    faceWin = null;
    if (off) return;
    // give the floating window a moment to actually release the device
    setTimeout(function () {
      if (state.faceMode === "docked" && !state.faceReleased) {
        startCam(want);
      }
    }, 250);
  }

  // the floating bubble's own dock button comes back through here
  window.__barFacecamDocked = function () {
    if (state.faceMode !== "floating") return;
    dockFacecam();
  };

  /* --- notes overlay ---------------------------------------------------
     A floating scratchpad for the demo script. Like the facecam bubble it's
     its own window, so once opened it persists through the whole take —
     which is the point, since the recording face has no controls of its own.
     It rides the same capture exclusion (a window this process owns is swept
     out of the SCK take), so it stays on screen but out of the video. */
  let notesWin = null;              // browser-fallback popup (non-native mode)

  function setNotesOpen(on) {
    state.notesOpen = on;
    ui.notes.classList.toggle("active", on);
    ui.notes.setAttribute("aria-pressed", on ? "true" : "false");
  }

  async function toggleNotes() {
    if (state.notesOpen) {
      if (!bridgeCall("notepad_close") && notesWin && !notesWin.closed) {
        notesWin.close();
      }
      notesWin = null;
      setNotesOpen(false);
      return;
    }
    const bridge = nativeApi();
    if (bridge && bridge.notepad_open) {
      try {
        const r = await bridge.notepad_open();
        if (r && r.ok) { setNotesOpen(true); notesCaptureCaveat(); }
        else setHint("Could not open notes" +
          (r && r.reason ? " (" + r.reason + ")" : "") + ".", true);
      } catch (e) { setHint("Could not open notes.", true); }
      return;
    }
    // no native bridge (Chrome app-mode / plain tab): a small popup window.
    // It works, but only the native SCK path can keep it out of the video.
    const w = 360, h = 420;
    notesWin = window.open("/notepad.html", "autocine-notes",
      "popup=yes,width=" + w + ",height=" + h +
      ",left=" + (screen.availWidth - w - 48) +
      ",top=" + Math.max(0, (screen.availHeight - h) / 2));
    if (notesWin) setNotesOpen(true);
    else setHint("Popup blocked. Allow popups to open notes.", true);
  }

  /* Be honest about the one condition the invisibility depends on: only the
     SCK ("Hide bar") engine drops our chrome from the take. On avfoundation
     the overlay is burned in, same as the pill — say so, don't imply magic. */
  function notesCaptureCaveat() {
    if (ui.backend && ui.backend.value === "avfoundation") {
      setHint("Notes stay out of the video only with “Hide bar” capture.", false);
    }
  }

  // the overlay's own close button (or its window closing) comes back here
  window.__barNotesClosed = function () {
    notesWin = null;
    setNotesOpen(false);
  };

  /* The cv ordinal behind the current camera pick, or null when the preview
     isn't the Python MJPEG one (a real browser using getUserMedia has no
     ordinal to share, and the backend has to open the device itself). */
  function nativeCamOrdinal() {
    const v = ui.camera.value;
    if (!v || v === "none" || v.indexOf("cv:") !== 0) return null;
    const n = parseInt(v.slice(3), 10);
    return isNaN(n) ? null : n;
  }

  function faceRelease() {
    state.faceReleased = true;
    stopCam();
    if (!bridgeCall("facecam_release") &&
        faceWin && !faceWin.closed && faceWin.__faceRelease) {
      faceWin.__faceRelease();
    }
  }

  function faceResume() {
    if (!state.faceReleased) return;
    state.faceReleased = false;
    if (ui.camera.value === "none") return;
    if (state.faceMode === "floating") {
      if (!bridgeCall("facecam_resume") &&
          faceWin && !faceWin.closed && faceWin.__faceResume) {
        faceWin.__faceResume();
      }
      return;
    }
    startCam(ui.camera.value);
  }

  /* Hold the docked preview's stream open only while it is actually wanted.

     The docked bubble's <img> lives INSIDE #face-idle (bar.html), and hiding
     an element does NOT cancel an in-flight load — so showFace() alone leaves
     the MJPEG connection open, which keeps Python's _Camera client refcount
     above zero, which keeps the WEBCAM LIT with the bubble invisible. That is
     exactly what used to happen after every take: the pill parked on the done
     face and the camera ran until the bar was closed (observed: ~1.9 GB of
     JPEG streamed over half an hour with no recording in flight, and, since
     the camera picker and the close X are inside #face-idle too, nothing on
     screen to turn it off with).

     Deliberately NOT called for countdown/recording/stopping, which also
     leave the idle face: `camera_preview.start_recording` tees face.mov off
     this very capture and needs clients > 0, so dropping the stream there
     would kill the face track. The take is the one time an invisible preview
     must keep running. That leaves the DONE face as the one that releases.

     Floating mode is left alone: the bubble is its own window, the docked
     <img> was already stopped by floatFacecam, and it stays visible past the
     end of a take on purpose. */
  function dockedPreview(on) {
    if (state.faceMode === "floating") return;
    if (on) startCam(ui.camera.value);
    else releaseCam();
  }

  async function loadPermissions() {
    try {
      const report = await api("/api/permissions");
      const missing = report.missing_required || [];
      state.permissionsOk = missing.length === 0;
      ui.record.disabled = !state.permissionsOk;
      if (!state.permissionsOk) {
        // Never paint the warning mid-take: the pill is being captured, and
        // setHint resizes the native window to fit the strip — so a false
        // alarm (this poll has fired during a take that logged clicks and keys
        // just fine) gets burned into the recording, and grows the pill while
        // it's at it. Starting is already hard-blocked server-side when
        // permissions are missing (studio_app._start_record), which leaves
        // this poll as a pre-flight nudge only.
        if (recordingInFlight()) return;
        const labels = missing.map(function (k) {
          const c = report.checks && report.checks[k];
          return c ? c.label : k;
        });
        setHint("Missing macOS permissions: " + labels.join(", ") +
          ". Grant them to this terminal app in System Settings, then fully quit and relaunch it.", true);
      } else if (state.face === "idle") {
        clearHint();     // back to the resting message, caveat and all
      }
    } catch (e) {
      // server down etc — leave record enabled; start will error meaningfully
    }
  }

  async function startRecording() {
    if (state.busy) return;
    state.busy = true;
    ui.record.disabled = true;
    const ids = currentWindowIds();
    const options = {
      display: ui.display.value === "" ? null : parseInt(ui.display.value, 10),
      // a picked window only crops the display capture (see WINDOW_NOTE). The
      // server re-reads the rect and 400s if the window has since gone away —
      // there is no silent full-screen fallback.
      // A single pick CROPS the display capture (see WINDOW_NOTE). Two to
      // four is a different mode: they're composited onto a background. The
      // server re-reads every rect — a stale single id is a 400, while one
      // stale id out of several is just dropped, since throwing the whole
      // take away over it would be the wrong trade.
      window_id: ids.length === 1 ? ids[0] : null,
      window_ids: ids.length > 1 ? ids : null,
      // Arrange only applies to the display-crop MULTI composite; it is
      // forced off for occlusion-free (each window is its own buffer, so the
      // on-screen arrangement is irrelevant -- and moving the user's windows
      // for zero benefit is what the doc says never to do).
      arrange: ids.length > 1 && !!state.windowArrange && !state.occlusionFree,
      mic: ui.mic.value === "none" ? "none" : parseInt(ui.mic.value, 10),
      fps: 60,
      countdown: parseInt(ui.countdown.value, 10) || 0,
      duration: null,
      cursor: ui.cursor.value,
      // Sent explicitly as well as persisted: the saved preference is the
      // fallback, but what the user can SEE in the picker is what must run.
      // Occlusion-free needs the Hide-bar (SCK) engine, so it forces sck here
      // rather than depend on the Capture picker (which stays where the user
      // left it; the hint says the engine switched).
      capture_backend: (state.occlusionFree && ids.length >= 1)
        ? "sck" : (ui.backend ? ui.backend.value : null),
      // Record each picked window's OWN buffer via SCK, so occluders never
      // appear (see WINDOW_NOTE_OCCLUSION_FREE). 1 window is single-file
      // native, 2-4 is multi-native (P3.1). Server also forces sck and 400s
      // if nothing resolves.
      occlusion_free: !!(state.occlusionFree && ids.length >= 1
                         && ids.length <= 4),
      // facecam: capture the webcam too when one is picked. The backend uses
      // the default camera (browser deviceId != avfoundation index).
      face: ui.camera.value !== "none",
      // The OpenCV ordinal the preview is streaming. Handing it over is what
      // lets the backend tee THAT capture into face.mov instead of opening a
      // second one — which is why the bubble keeps updating while recording
      // instead of going dark for the whole take.
      face_ordinal: nativeCamOrdinal(),
    };
    // Only give the webcam up when the backend is going to open it itself.
    // On the shared path releasing would be exactly the bug we just fixed.
    if (options.face && options.face_ordinal === null) faceRelease();
    // The picker is a full-screen window. If it is somehow still up
    // when a take starts it would be the entire recording.
    bridgeCall("close_picker_if_open");
    // named on the countdown/recording faces, so a mis-pick is visible before
    // the take is spent
    state.windowLabel = useOverlay
      ? windowBtnLabel(ids)
      : shortLabel(currentWindowLabel());
    try {
      await api("/api/record/start", { body: { options: options } });
      // blank, NOT clearHint(): the hint strip is on screen and inside the
      // capture from here on — the caveat has done its job
      setHint("");
      pollFast();
    } catch (e) {
      // The start never happened, so nothing is holding the camera — undo the
      // release. Without this the take never reaches applyRecord's done/error
      // branches (the server stays idle and state.face stays "idle"), so the
      // released state is stuck: the docked preview stays dark and, since
      // facecam_release now HIDES the floating bubble's native window, the
      // bubble vanishes with no way to dock or reopen it (facecam_float
      // reports "already open"). A blocked start is the likely path here —
      // _start_record rejects outright when macOS permissions are missing.
      faceResume();
      setHint(e.message, true);
      ui.record.disabled = !state.permissionsOk;
      state.windowLabel = "";
      // The other likely 400 is "that window is no longer on screen" — the
      // list we offered is stale, so refresh it and put the user back on the
      // picker instead of leaving them to work out that they must pick again.
      if (e.status === 400 && ids.length) {
        loadWindows().then(function () {
          if (useOverlay) openOverlayPicker();
          else openWindowPicker();
        });
      }
    }
    state.busy = false;
  }

  async function stopRecording() {
    try {
      await api("/api/record/stop", { body: {} });
      ui.recNote.textContent = "Stopping...";
    } catch (e) {
      setHint(e.message, true);
    }
  }

  // Pause/resume (segmented takes): the paused wall-clock span is deleted from
  // the exported video, so the timer freezes at content time (the server
  // slides started_wall forward on resume -- see studio_app on_record_state).
  function setPausedUi(paused) {
    // toggleAttribute, not the .hidden property: these are SVG elements, and
    // SVGElement has no `hidden` property -- assigning it is a silent no-op.
    ui.pauseGlyph.toggleAttribute("hidden", paused);
    ui.resumeGlyph.toggleAttribute("hidden", !paused);
    ui.pause.title = paused ? "Resume recording" : "Pause recording";
    ui.liveDot.classList.toggle("paused", paused);
  }

  async function togglePauseResume() {
    // Ignore clicks during the pausing/resuming transients -- the server
    // 409s those anyway; waiting a poll tick is calmer than surfacing it.
    if (state.recStatus !== "recording" && state.recStatus !== "paused") return;
    const paused = state.recStatus === "paused";
    try {
      await api(paused ? "/api/record/resume" : "/api/record/pause", { body: {} });
      pollFast();
    } catch (e) {
      setHint(e.message, true);
    }
  }

  /* Seamless window-JOIN chips: one "+ App" per window brought up since the
     take started -- newly opened, OR already open and raised to the front
     (server-fed `new_windows`, already capped + app-name-only). Rebuilt
     ONLY when the candidate id-set changes -- the recording branch runs every
     ~400ms poll, and an unconditional innerHTML rebuild would flicker and drop
     an in-flight chip click. */
  function renderGrowChips(rec) {
    if (!ui.growChips) return;
    const list = (rec && rec.grow_supported && Array.isArray(rec.new_windows))
      ? rec.new_windows : [];
    const key = list.map(function (e) { return e.id; }).join(",");
    if (key !== state.growIdsKey) {
      state.growIdsKey = key;
      ui.growChips.innerHTML = "";
      list.forEach(function (e) {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "bar-grow-chip";
        b.textContent = "+ " + (e.app || "Window");
        b.title = "Add this window to the recording";
        b.dataset.wid = String(e.id);   // opaque id, verbatim round-trip
        ui.growChips.appendChild(b);
      });
    }
    ui.growChips.hidden = list.length === 0;
  }

  /* One join is ever in flight: the auto-add engine and the manual chips BOTH
     `beginGrow()` when they POST a grow, and the latch clears on the poll that
     reports the join's outcome (joined / grow_error). But that outcome is seen
     ONLY inside applyRecord's `recording` branch, and the server re-serves
     grow_error DURABLY with no per-attempt id -- so two failures carrying the
     SAME message (the value-keyed dedupe collapses them), or a resolve that
     lands after the take already left `recording`, can strand the latch and
     wedge BOTH add paths for the rest of the take. `growInFlight` bounds that:
     past GROW_PENDING_CEILING_MS a still-set latch is treated as stale and
     released, self-healing a lost outcome. The ceiling sits ABOVE the
     recorder's PATHOLOGICAL resolve, not just its ~2 s GROW_T0 nominal: a grow
     queued behind an in-progress shrink harvest (SHRINK_HARVEST_DEADLINE_SEC
     5 s + kill grace) can take ~9 s to resolve, so 12 s guarantees the valve
     only ever fires on a genuinely LOST outcome, never on a slow-but-live join
     -- which is what keeps it from re-opening the recorder's pre-existing
     same-wid double-spawn gap (a manual re-click during a worker's
     spawn-not-yet-appended window; auto-add is immune via `autoAddTried`). The
     per-take reset clears it outright across takes; this is the within-take
     backstop. */
  const GROW_PENDING_CEILING_MS = 12000;
  function growInFlight() {
    if (!state.growPending) return false;
    if (state.growPendingSince
        && Date.now() - state.growPendingSince > GROW_PENDING_CEILING_MS) {
      state.growPending = false;   // stale latch -- outcome was lost; let the next grow fire
      return false;
    }
    return true;
  }
  function beginGrow(autoWid) {
    state.growPending = true;
    state.growPendingSince = Date.now();
    // Remember the window of an AUTO-add attempt so a TRANSIENT failure can be
    // retried selectively -- the recorder's grow_error string carries no id.
    // null for a manual chip: a user's own click is never auto-retried.
    state.autoAddPending = (autoWid == null) ? null : String(autoWid);
  }

  /* A just-opened window can deliver no first frame within the recorder's ~2s
     join budget (SCK sends nothing while a window is visually STATIC, measured
     -- docs/architecture.md), so its auto-add join fails with a transient
     "never delivered a frame" while the window is still on screen and will draw
     the moment it changes. Re-arm that ONE window (keyed on the auto-add attempt
     in flight, since the error has no id) for another attempt, BOUNDED by
     AUTO_ADD_MAX_TRIES so a window that never repaints can't loop the whole
     take. A manual chip (autoAddPending null) is never retried here; a PERMANENT
     failure (at cap / paused / vanished) does not carry the marker, so it
     stays a one-shot. Runs on the grow_error poll, which has already cleared
     `growPending`, so the trailing maybeAutoAdd() re-fires it the same tick. */
  const AUTO_ADD_MAX_TRIES = 4;
  function maybeRetryAutoAdd(rec) {
    const wid = state.autoAddPending;
    state.autoAddPending = null;
    if (wid == null || !state.autoAddWindows) return;
    if (!/never delivered a frame/.test(rec.grow_error || "")) return;
    const list = Array.isArray(rec.new_windows) ? rec.new_windows : [];
    if (!list.some(function (e) { return String(e.id) === wid; })) return; // gone -> let it go
    if (!state.autoAddRetries) state.autoAddRetries = {};
    const tries = state.autoAddRetries[wid] || 1;   // the first attempt counts as 1
    if (tries >= AUTO_ADD_MAX_TRIES) return;
    state.autoAddRetries[wid] = tries + 1;
    if (state.autoAddTried) delete state.autoAddTried[wid];   // re-arm; maybeAutoAdd re-fires
  }

  /* Auto-add windows: a client-side convenience over the same seamless-JOIN
     endpoint the chips use. When the toggle is on, fire a grow for the first
     candidate we haven't tried yet THIS take, one at a time. A candidate is
     any window the user BROUGHT TO THE FRONT mid-take (docs/architecture.md M2.4c) --
     the server decides that; this loop just drains whatever it is served. It shares
     `growPending` with the manual chips, so only ONE join is ever in flight
     (the recorder is last-wins; a second would silently drop the first).
     `autoAddTried` (reset per take alongside the hint keys) makes it EXACTLY one
     automatic attempt per window: a window that errors -- vanished mid-join, or
     the take just hit the cap -- is then left for a manual chip click rather
     than retried on every poll forever. At the 4-window maximum the server
     stops serving `new_windows`, so this naturally goes quiet. Effective only
     while the bar is open and polling; a closed or reloaded bar just stops
     auto-adding -- the take keeps recording, and the chips still work. Must run
     AFTER this poll's joined/grow_error handling clears `growPending`. */
  function maybeAutoAdd(rec) {
    if (!state.autoAddWindows || growInFlight()) return;
    if (!rec || rec.status !== "recording" || !rec.grow_supported) return;
    const list = Array.isArray(rec.new_windows) ? rec.new_windows : [];
    if (!state.autoAddTried) state.autoAddTried = {};
    let next = null;
    for (let i = 0; i < list.length; i++) {
      if (!state.autoAddTried[String(list[i].id)]) { next = list[i]; break; }
    }
    if (!next) return;
    const wid = Number(next.id);        // opaque id, verbatim round-trip (== manual chip)
    if (isNaN(wid)) return;
    state.autoAddTried[String(next.id)] = true;   // one automatic attempt (retried on transient fail)
    beginGrow(wid);
    api("/api/record/grow", { body: { window_id: wid } })
      .then(function () { setHint("Adding " + (next.app || "window") + "..."); pollFast(); })
      .catch(function (e) { state.growPending = false; setHint(e.message, true); });
  }

  function applyRecord(rec) {
    const status = rec && rec.status ? rec.status : "idle";
    state.recStatus = status;
    // Chips belong to the RECORDING state only; hide by default so every
    // other status (paused, pausing, stopping, done…) drops them. The
    // recording branch re-shows via renderGrowChips. (Hide, don't empty --
    // emptying every poll would rebuild the DOM and drop a chip mid-click.)
    if (ui.growChips) ui.growChips.hidden = true;
    if (state.firstPoll) {
      state.firstPoll = false;
      // A terminal status from a PREVIOUS recording must not trigger the
      // done face / auto-open when the bar first loads.
      if (status === "done" || status === "error") {
        if (rec.session) state.doneSession = rec.session;
        state.errorKey = status === "error" ? (rec.session || rec.message || "err") : null;
        return;
      }
    }
    if (status === "countdown") {
      state.errorKey = null;
      showFace("countdown");
      const m = /in (\d+)/.exec(rec.message || "");
      ui.count.textContent = m ? m[1] : "...";
      ui.countNote.textContent = state.windowLabel
        ? "Recording " + state.windowLabel + "..."
        : "Recording starts...";
    } else if (status === "recording") {
      state.errorKey = null;
      if (state.face !== "recording") {
        showFace("recording");
        state.recordingSince = Date.now();
        // the window's name, when there is one — a full-screen take has
        // nothing more to say than "Recording"
      }
      ui.recNote.textContent = state.windowLabel || "Recording";
      setPausedUi(false);
      ui.repick.hidden = true;
      // Hide pause on takes the server would 400. The SERVER's
      // pause_supported flag is the authority (whole-screen + occlusion-free
      // fleet takes pause; display-crop captures don't) — a reloaded bar (or
      // a second client) mid-take has no local picker state to gate on. The
      // old window_capture||windowLabel gate stays as the fallback for a
      // server that predates the flag.
      ui.pause.hidden = rec.pause_supported !== undefined
        ? !rec.pause_supported
        : !!(rec.window_capture || state.windowLabel);
      // Seamless window-join: chips for windows brought up since the take
      // started (opened, or raised from behind the cards).
      renderGrowChips(rec);
      // Surface the join outcome the next poll carries. Both keys are durable
      // (re-served every poll), so dedupe by VALUE or they re-fire forever.
      // The hint-dedupe keys are per-TAKE: worker indexes restart at 0 and
      // the mic anchor's app name repeats, so keys surviving into the next
      // take would silently swallow its hints (adversarially caught). The
      // server stamps `session` per take; key the reset on it.
      if (rec.session !== state.hintSession) {
        state.hintSession = rec.session;
        state.growErrKey = null;
        state.growJoinedKey = null;
        state.departedKey = null;
        state.micHideKey = null;
        state.autoAddTried = {};   // one auto-add attempt per window, per TAKE
        state.autoAddRetries = {}; // bounded transient-failure retries, per TAKE
        state.autoAddPending = null;
        // A join left in flight when a PRIOR take stopped (its outcome is only
        // observed while status==='recording', which Stop skips) must not carry
        // its `growPending` latch into this take -- that would wedge BOTH the
        // auto-add engine and the manual chips here. A new take can never
        // legitimately inherit a live join, so clear it outright.
        state.growPending = false;
      }
      if (rec.grow_error && rec.grow_error !== state.growErrKey) {
        setHint(rec.grow_error, true);
        state.growErrKey = rec.grow_error;
        state.growPending = false;
        maybeRetryAutoAdd(rec);   // a transient "never delivered a frame" is retryable
      } else if (rec.joined != null && rec.joined !== state.growJoinedKey) {
        setHint("Window added");
        state.growJoinedKey = rec.joined;
        state.growErrKey = null;
        state.growPending = false;
        state.autoAddPending = null;   // the in-flight auto-add landed; nothing to retry
      } else if (rec.departed != null && rec.departed !== state.departedKey) {
        // Card SHRINK (docs/architecture.md M3.4): say what actually HAPPENED --
        // recording of that window stopped ("restore it" would be wrong
        // advice for a closed window; a reopened one gets a fresh chip).
        // Value-keyed dedupe like joined/grow_error; a fast restore's
        // `joined` supersedes an unread `departed` server-side, on purpose.
        setHint("Stopped recording " + (rec.departed_app || "a window")
                + "; bring the window back to re-add it");
        state.departedKey = rec.departed;
      }
      if (rec.mic_anchor_hidden
          && rec.mic_anchor_hidden !== state.micHideKey) {
        // The mic-carrying card froze (decision 6's exemption): the mic
        // keeps recording, the card's picture doesn't.
        setHint("Keep " + rec.mic_anchor_hidden
                + " visible. It is recording your mic.", true);
        state.micHideKey = rec.mic_anchor_hidden;
      } else if (!rec.mic_anchor_hidden) {
        // The server retired the hint (anchor restored): reset the dedupe
        // key so the NEXT hide episode -- same app name -- surfaces again.
        state.micHideKey = null;
      }
      // Auto-add new windows, if armed. Runs here -- AFTER the joined/grow_error
      // handling above has cleared `growPending` for this poll -- so the next
      // join fires only once the previous one has actually landed or failed.
      maybeAutoAdd(rec);
      // prefer the server's wall-clock start so a reloaded bar shows the true
      // elapsed time of an already-running take (on a resumed take the server
      // has slid started_wall forward, so this stays CONTENT time)
      const elapsed = rec.started_wall
        ? Math.max(0, Date.now() / 1000 - rec.started_wall)
        : (Date.now() - state.recordingSince) / 1000;
      ui.timer.textContent = fmtTime(elapsed, false);
    } else if (status === "pausing" || status === "resuming") {
      if (state.face !== "recording") showFace("recording");
      ui.recNote.textContent = status === "pausing" ? "Pausing..." : "Resuming...";
    } else if (status === "paused") {
      if (state.face !== "recording") showFace("recording");
      ui.recNote.textContent = "Paused";
      setPausedUi(true);
      // A FAILED resume re-enters `paused` with the server carrying the
      // reason (e.g. the picked window closed during spin-up) — surface it,
      // or the pill silently flips "Resuming…" → "Paused" and the user
      // never learns their pick didn't take.
      if (rec.resume_error) setHint(rec.resume_error, true);
      // Scene takes: while paused on an occlusion-free take, offer the
      // window re-pick — the next scene can record a different window set.
      // Needs the native overlay picker; the <select> fallback has no way
      // to express a multi pick mid-take, so it simply doesn't offer this.
      ui.repick.hidden = !(useOverlay && rec.window_capture
                           && rec.pause_supported);
      // freeze the timer at the paused instant's content time
      if (rec.paused_wall && rec.started_wall) {
        ui.timer.textContent = fmtTime(
          Math.max(0, rec.paused_wall - rec.started_wall), false);
      }
    } else if (status === "stopping") {
      if (state.face !== "recording") showFace("recording");
      ui.recNote.textContent = "Stopping...";
    } else if (status === "done") {
      faceResume();   // the capture ffmpeg is finished with the webcam
      if (rec.session && rec.session !== state.doneSession) {
        state.doneSession = rec.session;
        showFace("done");
        // The preview is leaving the screen with #face-idle — let go of the
        // camera instead of streaming it to a hidden <img>. See dockedPreview.
        dockedPreview(false);
        ui.doneNote.textContent = "Recording saved";
        ui.openEditor.hidden = false;
        ui.again.hidden = false;
        // auto-open the editor in the default browser
        api("/api/open", { body: { session: rec.session } }).catch(function () {});
      }
    } else if (status === "error") {
      faceResume();
      const key = rec.session || rec.message || "err";
      if (state.errorKey !== key) {       // act once per distinct failure
        state.errorKey = key;
        if (rec.session) state.doneSession = rec.session;
        showFace("idle");
        ui.record.disabled = !state.permissionsOk;
        setHint(rec.message || "Recording failed", true);
      }
    } else if (state.face === "countdown" || state.face === "recording") {
      // fell back to idle without a done/error we saw
      faceResume();
      showFace("idle");
      ui.record.disabled = !state.permissionsOk;
    }
  }

  async function poll() {
    try {
      const s = await api("/api/state");
      applyRecord(s.record || {});
    } catch (e) {
      /* server briefly unreachable — keep polling */
    }
    scheduleNext();
  }

  function scheduleNext() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(poll, recordingInFlight() ? 400 : 1500);
  }

  function pollFast() {
    if (state.pollTimer) clearTimeout(state.pollTimer);
    state.pollTimer = setTimeout(poll, 250);
  }

  ui.record.addEventListener("click", startRecording);
  ui.stop.addEventListener("click", stopRecording);
  ui.pause.addEventListener("click", togglePauseResume);
  ui.repick.addEventListener("click", function () {
    if (state.recStatus !== "paused") return;
    openOverlayPicker();
  });
  // Seamless window-JOIN: click a chip to add that window to the LIVE take.
  // Delegated (chips are rebuilt as windows open/close). Debounced with
  // `growPending` — the recorder join is last-wins, so a second grow in flight
  // would silently drop the first; the flag clears on the next poll that
  // reports joined/grow_error. A 400 (window vanished / at cap) just shows a
  // hint; the chip disappears on the next poll — no error face.
  if (ui.growChips) {
    ui.growChips.addEventListener("click", function (ev) {
      const chip = ev.target.closest(".bar-grow-chip");
      if (!chip || state.recStatus !== "recording" || growInFlight()) return;
      const wid = Number(chip.dataset.wid);
      if (isNaN(wid)) return;
      beginGrow();
      api("/api/record/grow", { body: { window_id: wid } })
        .then(function () { setHint("Adding window..."); pollFast(); })
        .catch(function (e) { state.growPending = false; setHint(e.message, true); });
    });
  }
  ui.cancel.addEventListener("click", stopRecording);
  // shows the crop caveat while a window is picked, and clears it on the way
  // back to Full screen
  ui.window.addEventListener("change", function () {
    clearHint();
    syncOcclusionToggle();     // single-window state may have changed
    // Park the pill on what was just picked. Only the <select> FALLBACK needs
    // this — the overlay picker's result comes back through Python, which
    // docks there (picker_result). A no-op in a browser, where there is no
    // native window to move.
    bridgeCall("dock_to_windows", currentWindowIds());
  });
  if (ui.occlusion) {
    ui.occlusion.addEventListener("change", function () {
      state.occlusionFree = ui.occlusion.checked;
      if (ui.occlusionWrap) {
        ui.occlusionWrap.classList.toggle("is-on", state.occlusionFree);
      }
      syncAutoAddToggle();     // auto-add appears with / clears against occlusion-free
      setHint(idleHint());     // swap in / out the occlusion-free caveat
      reportFit();
    });
    // The label reads as part of the control; clicking it toggles the box.
    if (ui.occlusionLabel) {
      ui.occlusionLabel.addEventListener("click", function () {
        ui.occlusion.checked = !ui.occlusion.checked;
        ui.occlusion.dispatchEvent(new Event("change"));
      });
    }
  }
  if (ui.autoadd) {
    ui.autoadd.addEventListener("change", function () {
      state.autoAddWindows = ui.autoadd.checked;
      // Remember it: this toggle used to live in memory only, so every bar
      // relaunch reverted it and the next take silently recorded none of the
      // windows you brought up.
      state.autoAddPref = state.autoAddWindows;
      api("/api/settings", { body: { auto_add_windows: state.autoAddWindows } })
        .catch(function () { /* a preference that won't save must not block */ });
      if (ui.autoaddWrap) {
        ui.autoaddWrap.classList.toggle("is-on", state.autoAddWindows);
      }
      reportFit();
    });
    // The label reads as part of the control; clicking it toggles the box.
    if (ui.autoaddLabel) {
      ui.autoaddLabel.addEventListener("click", function () {
        ui.autoadd.checked = !ui.autoadd.checked;
        ui.autoadd.dispatchEvent(new Event("change"));
      });
    }
  }
  ui.camera.addEventListener("change", function () {
    if (ui.camera.value === "none") {
      camDeviceId = null;        // an explicit "No camera" IS the intent
      if (state.faceMode === "floating") dockFacecam();
      releaseCam();
      showCamSlot("off");
      return;
    }
    camDeviceId = ui.camera.value;
    if (state.faceMode === "floating") {
      // the floating window owns the device; hand it the new pick
      if (!bridgeCall("facecam_resume") &&
          faceWin && !faceWin.closed && faceWin.__faceResume) {
        faceWin.__faceResume();
      }
      return;
    }
    startCam(ui.camera.value);
  });
  ui.camSlot.addEventListener("click", function () {
    if (state.faceMode === "floating") dockFacecam();
    else floatFacecam();
  });
  ui.notes.addEventListener("click", toggleNotes);
  // release the webcam (and any floating bubble) if the pill is torn down
  window.addEventListener("pagehide", function () {
    stopCam();
    if (state.faceMode === "floating") dockFacecam();
    // native teardown destroys the overlay via close_window; a browser popup
    // has to be closed explicitly or it outlives its opener.
    if (notesWin && !notesWin.closed) notesWin.close();
  });
  ui.openEditor.addEventListener("click", function () {
    if (state.doneSession) {
      api("/api/open", { body: { session: state.doneSession } }).catch(function (e) {
        setHint(e.message, true);
      });
    }
  });
  ui.again.addEventListener("click", function () {
    // keep state.doneSession — the server's record status stays "done" until a
    // new recording starts, and nulling it here would re-trigger the done face
    // (and another auto-open) on the very next poll
    ui.openEditor.hidden = true;
    ui.again.hidden = true;
    showFace("idle");
    dockedPreview(true);   // the bubble is back on screen — reopen its stream
    ui.record.disabled = !state.permissionsOk;
    clearHint();       // the caveat belongs to the idle face
    loadWindows();     // the previous take's window list is long stale
  });
  function closeBar() {
    releaseCam();
    // native pywebview window: close through the JS bridge (it tears the
    // floating facecam and the notes overlay down with it — see close_window)
    if (window.pywebview && window.pywebview.api && window.pywebview.api.close_window) {
      window.pywebview.api.close_window();
      return;
    }
    if (faceWin && !faceWin.closed) faceWin.close();
    if (notesWin && !notesWin.closed) notesWin.close();
    window.close();
    // window.close() only works for script/app-opened windows; fall back to library
    setTimeout(function () { window.location.href = "/"; }, 150);
  }
  ui.close.addEventListener("click", closeBar);
  document.addEventListener("keydown", function (ev) {
    // Escape closes the bar on any face that ISN'T a live take. The gate used
    // to be `=== "idle"`, which left the done face with no way out at all: the
    // close X lives inside #face-idle (bar.html) and is hidden along with it,
    // so the only exit was New Recording -> back to idle -> X. A take still
    // can't be killed by a stray keypress, which is what the gate was for.
    if (ev.key === "Escape" &&
        state.face !== "countdown" && state.face !== "recording") closeBar();
  });

  // best-effort: shrink an app-mode window to pill size at the bottom of the screen
  try {
    if (!isNative && window.outerHeight > 220) {
      const w = 880, h = 150;
      window.resizeTo(w, h);
      window.moveTo(Math.round((screen.availWidth - w) / 2), screen.availHeight - h - 56);
    }
  } catch (e) { /* not permitted in a normal tab — fine */ }

  /* Which window control is live is decided ONCE, here, and never flips
     afterwards (except the one-way fall back to the list if the overlay
     refuses to open). The pill changing shape mid-session would be worse
     than either control on its own. */
  ui.windowBtn.addEventListener("click", openOverlayPicker);
  function chooseWindowControl() {
    useOverlay = isNative && !!(nativeApi() || {}).pick_windows;
    ui.windowBtn.hidden = !useOverlay;
    ui.window.hidden = useOverlay;
    if (useOverlay) syncWindowBtn();
  }
  if (isNative) {
    // The bridge isn't attached until pywebviewready, so ask again then.
    window.addEventListener("pywebviewready", chooseWindowControl);
  }
  chooseWindowControl();

  /* Capture engine: reflect what a take would ACTUALLY use, and persist any
     change so it survives a relaunch. The whole reason this control exists
     is that the choice used to live in an environment variable with nothing
     on screen to show it — and the difference it makes (this bar burned into
     the video, or not) was only discoverable afterwards, in the finished
     recording. */
  async function loadBackend() {
    if (!ui.backend) return;
    try {
      const s = await api("/api/settings");
      // Same round trip, so arming auto-add costs no extra request. undefined
      // (a server that predates the key) reads as "never chosen" -> default ON.
      state.autoAddPref = (s.auto_add_windows === undefined)
        ? null : s.auto_add_windows;
      ui.backend.value = s.capture_backend_effective || "avfoundation";
      if (s.capture_backend_locked) {
        // AUTOCINE_CAPTURE_BACKEND wins over anything saved, so the
        // picker would lie if it stayed editable.
        ui.backend.disabled = true;
        ui.backend.title = "Set by AUTOCINE_CAPTURE_BACKEND for this "
          + "launch; the saved preference is ignored until it is unset.";
      }
      // A lock to Standard hides the occlusion-free toggle (it needs SCK).
      syncOcclusionToggle();
    } catch (e) { /* leave the markup default */ }
  }
  if (ui.backend) {
    ui.backend.addEventListener("change", function () {
      api("/api/settings", { body: { capture_backend: ui.backend.value } })
        .catch(function () { /* a preference that won't save must not block */ });
    });
  }

  showCamSlot("off");
  loadBackend();
  loadDevices();
  loadWindows();
  loadCameras();
  loadPermissions();
  setInterval(loadPermissions, 5000);
  poll();
  reportFit();
})();
