/* editor.js — the studio editor for one session.
 *
 * Model: S.edits (the edits.json document) is the single source of truth.
 * Every mutation goes through mutate() -> autosave (debounced), camera-path
 * refresh, rendered-still refresh, and targeted UI updates. Undo/redo are
 * JSON snapshots of S.edits.
 *
 * Preview model: while paused, a server-rendered JPEG (`/api/preview`) shows
 * the exact output frame. While playing/scrubbing, the raw <video> is shown
 * with the real camera path applied as a CSS transform (`/api/camera-path`),
 * so zooms play back live at 60fps.
 */
"use strict";

(function () {
  const $ = function (id) { return document.getElementById(id); };
  const enc = encodeURIComponent;

  const ui = {
    title: $("ed-title"), saveState: $("ed-save-state"),
    undo: $("ed-undo"), redo: $("ed-redo"),
    preset: $("ed-preset"), presetDup: $("ed-preset-dup"),
    reveal: $("ed-reveal"), exportOpen: $("ed-export-open"),
    newRecording: $("ed-new-recording"),
    aspect: $("ed-aspect"), resolution: $("ed-resolution"),
    stageNote: $("ed-stage-note"), zoomReadout: $("ed-zoom-readout"),
    canvasWrap: $("ed-canvas-wrap"), canvas: $("ed-canvas"), frameBg: $("ed-frame-bg"),
    viewport: $("ed-viewport"), video: $("ed-video"), still: $("ed-still"),
    face: $("ed-face"),
    multi: $("ed-multi"), cardChrome: $("ed-cardchrome"),
    pinDot: $("ed-pin-dot"), pinHint: $("ed-pin-hint"), canvasMsg: $("ed-canvas-msg"),
    winPickRect: $("ed-winpick-rect"), winPickHint: $("ed-winpick-hint"),
    cropRect: $("ed-crop-rect"), cropHint: $("ed-crop-hint"),
    cropBox: $("ed-crop-box"),
    inspector: $("ed-inspector"), panel: $("ed-panel"), rail: $("ed-rail"),
    time: $("ed-time"), total: $("ed-total"),
    skipBack: $("ed-skip-back"), play: $("ed-play"), skipFwd: $("ed-skip-fwd"),
    addZoom: $("ed-add-zoom"), addSuppress: $("ed-add-suppress"), addMarker: $("ed-add-marker"),
    tl: $("ed-tl"), ruler: $("ed-ruler"), cam: $("ed-cam"),
    clipLane: $("ed-clip-lane"), clipBar: $("ed-clip-bar"), wave: $("ed-wave"),
    clipLabel: $("ed-clip-label"), dimLeft: $("ed-dim-left"), dimRight: $("ed-dim-right"),
    trimLeft: $("ed-trim-left"), trimRight: $("ed-trim-right"),
    zoomLane: $("ed-zoom-lane"), suppressLane: $("ed-suppress-lane"), markerLane: $("ed-marker-lane"),
    playhead: $("ed-playhead"),
    markerPop: $("ed-marker-pop"), mpLabel: $("mp-label"), mpTime: $("mp-time"),
    mpSummary: $("mp-summary"), mpTrim: $("mp-trim"), mpDelete: $("mp-delete"),
    exScrim: $("ed-export-scrim"), exFormat: $("ex-format"), exGifOpts: $("ex-gif-opts"),
    exGifFps: $("ex-gif-fps"), exGifWidth: $("ex-gif-width"),
    exStatus: $("ex-status"), exSpinner: $("ex-spinner"), exMessage: $("ex-message"),
    exReveal: $("ex-reveal"), exCancel: $("ex-cancel"), exStart: $("ex-start"),
  };

  const S = {
    session: queryParam("session"),
    details: null,
    edits: null,
    name: null,
    backgrounds: [],
    waveform: null,
    transcript: null,
    transcriptPoll: 0,
    transcriptQuery: "",
    camPath: null,
    camToken: 0,
    plate: null,           // decoded multi-window background plate (Image)
    plateB64: null,        // source of S.plate, to skip redundant decodes
    bgPlate: null,         // same backdrop WITHOUT baked shadows (window focus)
    bgPlateB64: null,
    multiShown: false,
    mutCount: 0,
    savedMutCount: 0,
    rev: 0,        // last server doc rev we synced with (CAS token; NOT part
                   // of undo snapshots, so undo-after-conflict can still save)
    saveSeq: 0,
    undo: [],
    redo: [],
    playhead: 0,
    playing: false,
    sel: null,             // {kind: "zoom"|"suppress"|"marker", id}
    txSel: null,           // {a, b} inclusive WORD INDICES while the user is
                           // choosing speech to remove. Indices, not times
                           // and not a browser Selection: the panel rebuilds
                           // on every search keystroke, every playhead move
                           // and every external-edit poll, and an index is
                           // the only handle that survives all of them.
    txDrag: false,         // true between pointerdown and pointerup on a word
    txScroll: 0,           // transcript list scrollTop, captured in
                           // renderPanel() just before the wipe and restored
                           // after the rebuild — see the note there
    txFocus: false,        // the word list held focus; restored after those
                           // same rebuilds so the keyboard flow survives
    pinArm: false,
    winPick: null,         // {id: <window id> | null} while the crop-drag tool is armed
    cropArm: false,        // true while the frame-crop tool is armed
    cardDrag: null,        // {index, mode, ...} while a card is moved/resized
    cardSel: null,         // index of the click-selected card (shows the full selection frame), or null
    cardHover: null,       // {index, corner} while the pointer rests on a card or one of its handles
    dragSnap: null,        // {video, face} frozen frame for the duration of that drag
    camPathStale: false,   // a camera path arrived mid-drag and was held back
    drag: null,
    panel: "zoom",
    previewToken: 0,
    stillShown: false,
    exporting: false,
    exportPoll: null,
    lastRenderOut: null,
    scrubbing: false,
    // --- scene take (S3 live player) ---
    sceneTake: false,      // this session's window set changes across pause seams
    sceneStillOnly: false, // no live plan yet -> the server still IS the preview
    scenePlayHinted: false,
    scenePlan: null,       // per-scene composite entries (from /api/camera-path)
    sceneSeams: null,      // output frame index each scene begins at
    seamTimes: null,       // same, in seconds
    sceneTotalFrames: 0,
    sceneFps: 60,
    activeScene: 0,
    fleets: null,          // Map<sceneIndex, {videos:[<video>...]}> (LRU-capped)
    fleetLRU: null,        // scene indices, most-recently-touched last
    pendingResumeScene: null,  // a seam HOLD is waiting for this scene to buffer
  };

  /* ============================ basics ============================ */

  function duration() {
    return S.details && Number.isFinite(S.details.duration) && S.details.duration > 0
      ? S.details.duration : 1;
  }

  function trimRange() {
    const d = duration();
    // Trim is ignored on the scene-take export (v1 scope), so it must not gate
    // the editor timeline either -- otherwise a stray/legacy trim silently
    // clamps the whole scrub (and Export would still emit the full take). Same
    // contract the server's scene preview uses: the full joined timeline.
    if (S.details && S.details.scene_take) return { start: 0, end: d };
    const t = (S.edits && S.edits.trim) || {};
    const start = clamp(toNum(t.start, 0), 0, d);
    let end = t.end === null || t.end === undefined ? d : clamp(toNum(t.end, d), 0, d);
    if (end < start + 0.01) end = Math.min(d, start + 0.01);
    return { start: start, end: end };
  }

  function fullOptions() {
    const r = S.edits.render;
    return {
      zoom: r.zoom, zoom_speed: r.zoom_speed,
      offset: r.offset, style: r.style, background: r.background,
      aspect: r.aspect, resolution: r.resolution,
      click_fx: r.click_fx, click_color: r.click_color,
      spotlight: r.spotlight, cursor_fx: r.cursor_fx, cursor_size: r.cursor_size,
      cursor_erase: r.cursor_erase,
      fade: r.fade, music: r.music, click_sound: r.click_sound,
      key_sound: r.key_sound, sfx_volume: r.sfx_volume,
      always_zoomed: r.always_zoomed, motion_blur: r.motion_blur,
      overview: r.overview, screen_focus: r.screen_focus,
      // window_layout belongs here for the same reason window_follow does:
      // this dict is the ephemeral override the paused still and the camera
      // path are rendered with, so anything the rail can change must be in it
      // or the still lags a save behind the control that changed it.
      window_follow: r.window_follow, window_layout: r.window_layout,
      window_zoom: r.window_zoom, window_focus: r.window_focus,
      facecam: r.facecam, facecam_position: r.facecam_position,
      facecam_size: r.facecam_size, facecam_shape: r.facecam_shape,
      facecam_border: r.facecam_border, facecam_blur: r.facecam_blur,
      gif: r.gif, gif_fps: r.gif_fps, gif_width: r.gif_width,
      trim_start: S.edits.trim.start, trim_end: S.edits.trim.end,
      zooms: S.edits.zooms, suppressed: S.edits.suppressed, markers: S.edits.markers,
      windows: S.edits.windows, focus: S.edits.focus,
      // Manual card placement rides alongside windows so a live (unsaved) drag
      // previews through the camera path immediately: multi-native (array) and
      // scene takes (per-scene dict).
      channel_layouts: S.edits.channel_layouts || [],
      scene_layouts: S.edits.scene_layouts || {},
      // Removed (reversibly hidden) cards: changes the card SET, so it must
      // ride the live camera-path options and force a re-fetch (cellsSignature).
      hidden_channels: S.edits.hidden_channels || [],
    };
  }

  function editsPayload() {
    return {
      trim: S.edits.trim,
      render: S.edits.render,
      // active_preset_id MUST travel with render: the server writes the render
      // patch into whichever preset it believes is active, so an undo that
      // reverts both locally would otherwise corrupt the other preset on disk.
      active_preset_id: S.edits.active_preset_id,
      zooms: S.edits.zooms,
      suppressed: S.edits.suppressed,
      markers: S.edits.markers,
      windows: S.edits.windows,
      // Manual card placement. Always sent so a Reset ([] / {}) persists
      // through merge_edits' membership patch, exactly like crop: multi-native
      // (array) + scene takes (per-scene dict, whole-key replaced).
      channel_layouts: S.edits.channel_layouts || [],
      scene_layouts: S.edits.scene_layouts || {},
      // Removed cards. Always sent so a Restore ([]) persists through
      // merge_edits' membership patch, exactly like channel_layouts / cuts.
      hidden_channels: S.edits.hidden_channels || [],
      // Cuts (ripple delete). Without this key an editor-authored cut is
      // dropped by `merge_edits`'s membership patch and never reaches disk
      // — the feature looks like it worked until the next reload.
      cuts: S.edits.cuts || [],
      // Always sent, including as an explicit null: `merge_edits` patches on
      // key MEMBERSHIP, so omitting it on reset would leave the old crop on
      // disk and the editor would silently disagree with the export.
      crop: S.edits.crop === undefined ? null : S.edits.crop,
    };
  }

  function findItem(kind, id) {
    const arr = kind === "zoom" ? S.edits.zooms :
      kind === "suppress" ? S.edits.suppressed : S.edits.markers;
    for (let i = 0; i < arr.length; i++) if (arr[i].id === id) return arr[i];
    return null;
  }

  /* ============================ undo / save ============================ */

  function setSaveState(text, isErr) {
    ui.saveState.textContent = text || "";
    ui.saveState.classList.toggle("err", !!isErr);
  }

  function pushUndo(snapshot) {
    S.undo.push(snapshot !== undefined ? snapshot : JSON.stringify(S.edits));
    if (S.undo.length > 120) S.undo.shift();
    S.redo.length = 0;
    syncUndoButtons();
  }

  function syncUndoButtons() {
    ui.undo.disabled = S.undo.length === 0;
    ui.redo.disabled = S.redo.length === 0;
  }

  function undo() {
    if (!S.undo.length) return;
    S.redo.push(JSON.stringify(S.edits));
    S.edits = JSON.parse(S.undo.pop());
    S.mutCount++;
    afterMutate({});
    syncUndoButtons();
  }

  function redo() {
    if (!S.redo.length) return;
    S.undo.push(JSON.stringify(S.edits));
    S.edits = JSON.parse(S.redo.pop());
    S.mutCount++;
    afterMutate({});
    syncUndoButtons();
  }

  const scheduleSave = debounce(saveNow, 700);

  async function saveNow(requestOptions) {
    const keepalive = Boolean(requestOptions && requestOptions.keepalive);
    const seq = ++S.saveSeq;
    const mutAtSend = S.mutCount;
    setSaveState("Saving...");
    try {
      const data = await api("/api/edits/" + enc(S.session), {
        body: { edits: editsPayload(), base_rev: S.rev },
        keepalive: keepalive,
      });
      if (seq !== S.saveSeq) return;                 // a newer save is in flight
      S.savedMutCount = Math.max(S.savedMutCount, mutAtSend);
      if (data.edits) S.rev = data.edits.rev || S.rev;
      // S.cardDrag and S.txDrag join S.drag here: adoptEdits rebuilds the
      // timeline and the panel, and a DOM rebuild under a moving pointer is
      // a visible hitch. For the transcript that rebuild also replaces the
      // word spans the drag is hit-testing against.
      if (S.mutCount === mutAtSend && !S.drag && !S.cardDrag && !S.txDrag) {
        adoptEdits(data.edits);
      }
      setSaveState("Saved");
    } catch (e) {
      // A pagehide flush is best-effort. In particular, do not interpret its
      // 409 as an external edit and mark the closing page clean: another save
      // may still be in flight, and this latest payload was not accepted.
      if (keepalive) {
        setSaveState("Save failed. Retrying...", true);
        setTimeout(scheduleSave, 2500);
        return;
      }
      if (e.status === 409 && e.data && e.data.conflict && e.data.edits) {
        // someone else (another tab, the MCP server) saved first — keep our
        // version on the undo stack, adopt theirs, and say so
        pushUndo();
        S.mutCount++;
        S.savedMutCount = S.mutCount;
        adoptEdits(e.data.edits);
        scheduleCamPath();
        staleStill();
        applyFrameStyle();
        applyAspectSelect();
        applyResolutionSelect();
        setSaveState("Saved");
        toast("This project was edited outside the editor. Showing the latest version (Cmd-Z restores yours)");
        return;
      }
      if (e.status === 400 && e.data && e.data.code === "cuts" && e.data.edits) {
        // The cut set would produce an export the renderer refuses. The
        // editor autosaves a WHOLE document, so we cannot just leave this
        // one failing forever — recover exactly as the 409 does: keep the
        // user's version on the undo stack, adopt the one that is really on
        // disk, and say what happened in the server's own words.
        //
        // Gated on `code`: a generic 400 (bad session name, torn body) must
        // NOT unwind an edit the user never connected to the failure.
        pushUndo();
        S.mutCount++;
        S.savedMutCount = S.mutCount;
        adoptEdits(e.data.edits);
        scheduleCamPath();
        staleStill();
        setSaveState("Saved");
        toast(e.data.error || "That cut can't be applied", "err");
        return;
      }
      setSaveState("Save failed. Retrying...", true);
      setTimeout(scheduleSave, 2500);
    }
  }

  /* Debounced saves die with the page — flush pending edits on the way out.
     `fetch(..., keepalive)` survives navigation while still carrying the API
     token header; sendBeacon cannot set that header. saveNow advances the
     saved counter only after the server actually accepts the CAS write. */
  window.addEventListener("pagehide", function () {
    if (!S.edits || S.mutCount === S.savedMutCount) return;
    scheduleSave.cancel();
    saveNow({ keepalive: true });
  });

  /* Adopt the server-normalized doc (clamps, dedup, canonical ids). */
  function adoptEdits(normalized) {
    if (!normalized) return;
    if (typeof normalized.rev === "number") S.rev = normalized.rev;
    if (JSON.stringify(normalized) === JSON.stringify(S.edits)) return;
    S.edits = normalized;
    if (S.sel && !findItem(S.sel.kind, S.sel.id)) S.sel = null;
    renderTimeline();
    renderPanel();
    syncPresetSelect();
  }

  /* mutate(fn) — the single write path. opts:
   *   coalesce: true  -> caller already pushed the undo snapshot (drags)
   *   camera: false   -> edit can't change the camera path (labels etc.)
   *   still: false    -> skip the rendered-still refresh
   */
  function mutate(fn, opts) {
    opts = opts || {};
    if (!opts.coalesce) pushUndo();
    fn(S.edits);
    S.mutCount++;
    afterMutate(opts);
  }

  function afterMutate(opts) {
    opts = opts || {};
    setSaveState("Unsaved");
    scheduleSave();
    if (opts.camera !== false) scheduleCamPath();
    if (opts.still !== false) staleStill(opts.stillDelay || 240);
    renderTimeline();
    if (!opts.skipPanel) renderPanel();
    applyFrameStyle();
    applyAspectSelect();
    applyResolutionSelect();
    syncPresetSelect();
  }

  /* ============================ camera path ============================ */

  const scheduleCamPath = debounce(fetchCamPath, 320);

  /* What a camera path's `cells` are a function of, as one comparable string.
     The server hands `multi_window_layout` exactly this: the window cards
     (identity, order, crop rect, any hand placement) and the three options
     that decide the canvas and the arrangement.

     Nothing else from fullOptions() belongs here, deliberately. A background
     or a click colour repaints the plate and cannot move a cell, so signing
     the whole options dict would mark paths stale that are perfectly good --
     and every consumer of this (today: the "Fit to frame" gate) would blink
     off every time a slider moved. */
  function cellsSignature(e) {
    if (!e) return "";
    const r = e.render || {};
    // channel_layouts joins windows here: a manual card placement on a
    // multi-native take changes the cells the camera-path returns, so a pin
    // change must force a re-fetch (and the mid-drag camPathStale hold).
    return JSON.stringify([e.windows || [], e.channel_layouts || [],
                           e.scene_layouts || {}, e.hidden_channels || [],
                           r.window_layout || null,
                           r.aspect || null, r.style || null]);
  }

  async function fetchCamPath() {
    const token = ++S.camToken;
    // Stamped BEFORE the request leaves and carried back on the response.
    // `cells` are bound to the windows POSITIONALLY -- cell i is
    // edits.windows[i] and nothing in the payload says so -- while this
    // request lands a debounce plus a round trip later, by which time
    // S.edits can have had a card removed, reordered or redrawn. Pairing the
    // two by index across that gap writes one card's geometry onto another.
    const sig = cellsSignature(S.edits);
    try {
      const data = await api("/api/camera-path", {
        body: { session: S.session, options: fullOptions() },
      });
      if (token !== S.camToken) return;
      // A response that lands mid-drag would replace p.cells wholesale --
      // including the cell the pointer is holding, which snaps back to the
      // server's placement until the next pointermove moves it again. Hold
      // it; onCardUp re-requests once the gesture is over.
      if (S.cardDrag) { S.camPathStale = true; return; }
      // Scene take: N per-scene fleets, not one layout -- adopt the per-scene
      // plan (which aliases S.camPath to the active scene itself) and return;
      // the multi-window tail below assumes a single top-level layout.
      if (data.scene_take && data.scenes && data.scenes.length) {
        adoptScenePayload(data);
        return;
      }
      data.cells_sig = sig;
      S.camPath = data;
      // `fleetLead` is 0 until this payload arrives, so a lagged fleet has
      // been sitting `start[0]` off since load. Re-seat the element now that
      // the conversion is known; a no-op for every take whose lead is 0.
      if (!S.playing && fleetLead() > 0) {
        const seat = fleetLead() + S.playhead;
        if (Math.abs(ui.video.currentTime - seat) > 0.004) {
          try { ui.video.currentTime = seat; } catch (e) {}
        }
        syncChannels(S.playhead);
      }
      loadPlate(data.windows_mode ? data.plate_jpeg_base64 : null);
      loadBgPlate(data.windows_mode ? data.bg_jpeg_base64 : null);
      fitCanvas();
      applyTransformAt(S.playhead);
      updateStageNote();
      updateCaptureUI();
      drawCamCurve();
      // The camera path is the only thing "Fit to frame" can measure, so the
      // first one to land is what makes that button live. Open the Windows
      // panel before it arrives and nothing else would ever repaint it --
      // one attribute rather than renderPanel(), which would rebuild the
      // panel under the pointer every time a path came back.
      syncFitAction();
    } catch (e) {
      /* keep last good path */
    }
  }

  /* interpolated camera sample at time t -> {cx, cy, z} */
  function sampleCam(t) {
    const p = S.camPath;
    if (!p || !p.times || p.times.length === 0) {
      const d = S.details || {};
      return { cx: (d.width || 2) / 2, cy: (d.height || 2) / 2, z: 1 };
    }
    const times = p.times;
    const n = times.length;
    if (t <= times[0]) return { cx: p.cx[0], cy: p.cy[0], z: p.z[0] };
    if (t >= times[n - 1]) return { cx: p.cx[n - 1], cy: p.cy[n - 1], z: p.z[n - 1] };
    const step = (times[n - 1] - times[0]) / Math.max(1, n - 1);
    let i = clamp(Math.floor((t - times[0]) / Math.max(1e-9, step)), 0, n - 2);
    while (i < n - 2 && times[i + 1] < t) i++;
    while (i > 0 && times[i] > t) i--;
    const f = clamp((t - times[i]) / Math.max(1e-9, times[i + 1] - times[i]), 0, 1);
    return {
      cx: p.cx[i] + (p.cx[i + 1] - p.cx[i]) * f,
      cy: p.cy[i] + (p.cy[i + 1] - p.cy[i]) * f,
      z: p.z[i] + (p.z[i + 1] - p.z[i]) * f,
    };
  }

  function canvasAspect() {
    if (S.camPath && S.camPath.canvas) return S.camPath.canvas[0] / S.camPath.canvas[1];
    const d = S.details || {};
    return (d.width || 16) / (d.height || 10);
  }

  /* Size .ed-canvas to the OUTPUT aspect inside its wrap.
   *
   * The room has to be measured off the wrap's CONTENT box. `clientWidth` /
   * `clientHeight` include .ed-canvas-wrap's padding (14px 20px 18px), while
   * .ed-canvas's own `max-width/max-height: 100%` resolve against the content
   * box -- so fitting to the padded numbers sets a box that CSS then clamps on
   * ONE axis, and a one-axis clamp is an ASPECT change. Measured here, wrap
   * 1116x616: this set 973x608 (1.600, correct) and CSS served 973x584
   * (1.666). Nothing reports that; the element simply is not the shape the
   * layout was computed for.
   *
   * Which is not cosmetic. The paused still is `object-fit` into this box and
   * drawMulti scales the live composite by it, so a 4% wrong box is exactly
   * how the preview stops matching the export -- and, because the two absorb
   * it differently, how the frame visibly jumps the moment you press play.
   *
   * Floored, not rounded, for the same reason: rounding up puts the box back
   * over the max and hands it straight to the clamp this just avoided.
   */
  function fitCanvas() {
    const wrap = ui.canvasWrap;
    const cs = getComputedStyle(wrap);
    const padX = (parseFloat(cs.paddingLeft) || 0)
               + (parseFloat(cs.paddingRight) || 0);
    const padY = (parseFloat(cs.paddingTop) || 0)
               + (parseFloat(cs.paddingBottom) || 0);
    const maxW = wrap.clientWidth - padX;
    const maxH = wrap.clientHeight - padY;
    if (maxW <= 0 || maxH <= 0) return;
    const ar = canvasAspect();
    let w = maxW, h = w / ar;
    if (h > maxH) { h = maxH; w = h * ar; }
    ui.canvas.style.width = Math.floor(w) + "px";
    ui.canvas.style.height = Math.floor(h) + "px";
    applyFrameStyle();
    applyTransformAt(S.playhead);
  }

  /* framed style: approximate the render-time inset + rounded corners so the
     live (video) preview roughly matches; the paused still is exact. */
  function applyFrameStyle() {
    if (!S.edits) return;
    const r = S.edits.render;
    const framed = r.style === "framed" || !!r.background;
    const cw = ui.canvas.clientWidth, ch = ui.canvas.clientHeight;
    if (framed && cw > 0 && ch > 0) {
      // mirror framing.FramePainter: pad = framing.PAD_FRAC of WIDTH,
      // aspect-preserving inner box (so the width-derived video scale stays
      // exact). Keep in step with autocine/framing.py's PAD_FRAC.
      const pad = 0.03 * cw;
      let iw = cw - 2 * pad;
      let ih = iw * ch / cw;
      if (ih > ch - 2 * pad) {
        ih = ch - 2 * pad;
        iw = ih * cw / ch;
      }
      ui.viewport.style.inset = "";
      ui.viewport.style.left = ((cw - iw) / 2) + "px";
      ui.viewport.style.top = ((ch - ih) / 2) + "px";
      ui.viewport.style.width = iw + "px";
      ui.viewport.style.height = ih + "px";
      ui.viewport.classList.add("framed");
      ui.frameBg.style.background = resolveBgCss(r.background);
    } else {
      ui.viewport.style.inset = "";
      ui.viewport.style.left = "0";
      ui.viewport.style.top = "0";
      ui.viewport.style.width = "100%";
      ui.viewport.style.height = "100%";
      ui.viewport.classList.remove("framed");
      ui.frameBg.style.background = "transparent";
    }
    applyTransformAt(S.playhead);
  }

  function resolveBgCss(spec) {
    if (!spec) return "linear-gradient(135deg, #2b2823, #141210)";
    if (spec.charAt(0) === "#") return spec;
    for (let i = 0; i < S.backgrounds.length; i++) {
      if (S.backgrounds[i].id === spec) return gradientCss(S.backgrounds[i].colors);
    }
    return "linear-gradient(135deg, #2b2823, #141210)";
  }

  /* ==================== live multi-window composite ====================
   * In windows mode there is no camera path to drive a CSS transform with --
   * the output is N static crops of this same recording, laid out on a shared
   * background. A single <video> element can only ever be in one place, so
   * playback draws the grid into a canvas instead: the server-built plate
   * (background + shadows, from framing.MultiFramePainter) blitted once, then
   * one drawImage per cell through a rounded clip.
   *
   * The layout is NOT recomputed here -- `/api/camera-path` hands over the
   * exact cells render() will composite with, so the live preview and the
   * export cannot drift apart.
   */

  /* Decoded ONCE into a canvas rather than kept as the <Image>.
   *
   * The backdrop is drawn on every single composite frame -- every drag
   * event, every playback tick. Measured in a real browser here: blitting
   * this 2880x1800 JPEG straight from an <img> costs ~20 ms a draw (the
   * decode is not cached across draws at this size), the identical pixels
   * from a canvas cost ~1.8 ms. One rasterize at load time buys that back
   * for every frame after it. */
  function rasterize(img) {
    const c = document.createElement("canvas");
    c.width = img.naturalWidth || img.width;
    c.height = img.naturalHeight || img.height;
    try {
      c.getContext("2d").drawImage(img, 0, 0);
    } catch (e) { return img; }   // fall back to the element itself
    return c;
  }

  function loadPlate(b64) {
    if (!b64) { S.plate = null; S.plateB64 = null; return; }
    if (S.plateB64 === b64 && S.plate) return;
    S.plateB64 = b64;
    const img = new Image();
    img.onload = function () {
      if (S.plateB64 !== b64) return;   // a newer layout landed mid-decode
      S.plate = rasterize(img);
      applyTransformAt(S.playhead);
    };
    // Keep the previous plate on failure rather than dropping to a blank
    // canvas; a stale backdrop beats no preview at all.
    img.src = "data:image/jpeg;base64," + b64;
  }

  /* The same backdrop WITHOUT baked shadows, sent only when window focus is
     actually animating -- see drawCardShadows. Kept separate from the plate
     so the static path is untouched and keeps the server's exact shadows. */
  function loadBgPlate(b64) {
    if (!b64) { S.bgPlate = null; S.bgPlateB64 = null; return; }
    if (S.bgPlateB64 === b64 && S.bgPlate) return;
    S.bgPlateB64 = b64;
    const img = new Image();
    img.onload = function () {
      if (S.bgPlateB64 !== b64) return;
      S.bgPlate = rasterize(img);       // same reason as the plate above
      applyTransformAt(S.playhead);
    };
    img.src = "data:image/jpeg;base64," + b64;
  }

  /* Windows mode with everything needed to actually draw it. Falls false while
     the plate decodes, which lands us on the plain <video> for a frame or two
     -- the pre-existing behavior, not a regression. */
  function multiReady() {
    const p = S.camPath;
    return !!(p && p.windows_mode && p.cells && p.cells.length && p.canvas
              && S.plate);
  }

  function showMulti(on) {
    if (S.multiShown === on) return;
    S.multiShown = on;
    ui.multi.hidden = !on;
    // visibility, not display: the <video> must stay laid out and decoding,
    // since it is the source every cell is drawn from.
    ui.viewport.style.visibility = on ? "hidden" : "";
  }

  /* ==================== scene take: live player (S3) ====================
   *
   * A SCENE take (recorded with pause/resume, a DIFFERENT window set per
   * scene) has no single raw.mov AND a channel count that changes mid-
   * timeline, so the multi-native player above -- which builds ONE follower
   * fleet at load and treats ui.video as the master -- cannot host it.
   *
   * The player here is a per-scene "fleet ring": each scene owns a fleet of
   * <video> elements (one per window), and the ACTIVE scene's channel 0 is the
   * master clock (played, so it decode-paces itself -- real smooth motion,
   * not a wall-clock accumulator). `S.camPath` is aliased to the active
   * scene's plan so drawMulti / cellSource / cardSrcRect / focusFrameAt are
   * reused unchanged; only cellSource, drawMulti's master-readiness gate and
   * masterVideo() are forked (on S.details.scene_take) off the srcless
   * ui.video onto the active fleet. Fleets are LRU-capped so a backward scrub
   * over a just-crossed seam hits a retained neighbour instead of a cold
   * rebuild. The server still (`scene_preview_frame`, bit-identical to export)
   * is the always-available correctness floor shown whenever the fleet is not
   * frame-ready. See docs/architecture.md, section 6.
   */
  const SCENE_FLEET_CAP = 3;     // active + both neighbours (<=4 chans each)
  const SCENE_LEAD_SEC = 0.6;    // prebuffer the next fleet this far ahead

  function sceneLive() {
    return !!(S.sceneTake && S.scenePlan && S.scenePlan.length);
  }

  function masterVideo() {
    if (sceneLive()) {
      const fl = S.fleets && S.fleets.get(S.activeScene);
      return (fl && fl.videos[0]) || null;
    }
    return ui.video;
  }

  function sceneStart0(s) {
    const e = S.scenePlan[s];
    return toNum(e && e.channels[0] && e.channels[0].start, 0);
  }

  /* Output time -> {s, kLocal}. The EXACT export rule (render.py: half-open,
     the boundary frame belongs to the EARLIER scene -- SegmentClock.owner),
     so the live composite selects the same scene the export/still does. */
  function resolveScene(t) {
    const fps = S.sceneFps, total = S.sceneTotalFrames;
    let kg = Math.round(Math.max(0, t) * fps);
    kg = Math.max(0, Math.min(kg, total - 1));
    let cum = 0;
    for (let i = 0; i < S.scenePlan.length; i++) {
      const n = S.scenePlan[i].frame_count;
      if (kg < cum + n) return { s: i, kLocal: kg - cum };
      cum += n;
    }
    const last = S.scenePlan.length - 1;
    return { s: last, kLocal: S.scenePlan[last].frame_count - 1 };
  }

  function decodeScenePlates(entry) {
    if (entry._plateB64 !== entry.plate_jpeg_base64) {
      entry._plateB64 = entry.plate_jpeg_base64;
      entry._plate = null;
      if (entry.plate_jpeg_base64) {
        const img = new Image();
        img.onload = function () {
          if (entry._plateB64 !== entry.plate_jpeg_base64) return;
          entry._plate = rasterize(img);
          if (S.scenePlan && S.scenePlan[S.activeScene] === entry) {
            S.plate = entry._plate;
            applyTransformAt(S.playhead);
          }
        };
        img.src = "data:image/jpeg;base64," + entry.plate_jpeg_base64;
      }
    }
    if (entry._bgB64 !== entry.bg_jpeg_base64) {
      entry._bgB64 = entry.bg_jpeg_base64;
      entry._bgPlate = null;
      if (entry.bg_jpeg_base64) {
        const img2 = new Image();
        img2.onload = function () {
          if (entry._bgB64 !== entry.bg_jpeg_base64) return;
          entry._bgPlate = rasterize(img2);
          if (S.scenePlan && S.scenePlan[S.activeScene] === entry) {
            S.bgPlate = entry._bgPlate;
            applyTransformAt(S.playhead);
          }
        };
        img2.src = "data:image/jpeg;base64," + entry.bg_jpeg_base64;
      }
    }
  }

  /* Adopt a fresh /api/camera-path scene payload. Called on load and on every
     re-fetch (a look/zoom toggle changes the plates + tracks); the fleet
     <video>s are keyed by media kind, which is stable, so they are kept. */
  function adoptScenePayload(data) {
    S.sceneTake = true;
    S.sceneFps = toNum(data.fps, 60) || 60;
    S.sceneSeams = data.seams || [];
    S.sceneTotalFrames = toNum(data.total_frames, 0);
    S.seamTimes = S.sceneSeams.map(function (f) { return f / S.sceneFps; });
    S.scenePlan = data.scenes;
    for (let i = 0; i < S.scenePlan.length; i++) decodeScenePlates(S.scenePlan[i]);
    S.sceneStillOnly = false;           // the live player is available now
    if (!S.fleets) { S.fleets = new Map(); S.fleetLRU = []; }
    const r = resolveScene(S.playhead);
    activateScene(r.s);
    getFleet(r.s);
    fitCanvas();
    applyTransformAt(S.playhead);
    updateStageNote();
    updateCaptureUI();
    drawCamCurve();
    syncFitAction();
  }

  function activateScene(s) {
    S.activeScene = s;
    const entry = S.scenePlan[s];
    S.camPath = entry;                  // the alias every compositor fn reads
    S.plate = entry._plate || null;
    S.bgPlate = entry._bgPlate || null;
    touchFleet(s);
  }

  /* --- the fleet ring --- */
  function buildFleet(s) {
    const entry = S.scenePlan[s];
    const videos = [];
    for (let i = 0; i < entry.channels.length; i++) {
      const v = document.createElement("video");
      v.className = "ed-face-src";      // 1x1/opacity:0 -> still decodes
      v.muted = true;                   // scene takes are silent (v1)
      v.playsInline = true;
      v.preload = "auto";
      v.src = autocineUrl(
        "/api/media/" + enc(S.session) + "/" + entry.channels[i].media);
      v.addEventListener("seeked", function () { onFleetProgress(s); });
      v.addEventListener("canplay", function () { onFleetProgress(s); });
      v.addEventListener("loadedmetadata", function () { onFleetProgress(s); });
      document.body.appendChild(v);
      videos[i] = v;
    }
    return { videos: videos, scene: s, prebuffered: false };
  }

  function getFleet(s) {
    let fl = S.fleets.get(s);
    if (!fl) { fl = buildFleet(s); S.fleets.set(s, fl); }
    touchFleet(s);
    evictFleets();
    return fl;
  }

  function touchFleet(s) {
    if (!S.fleetLRU) return;
    const i = S.fleetLRU.indexOf(s);
    if (i >= 0) S.fleetLRU.splice(i, 1);
    S.fleetLRU.push(s);
  }

  function evictFleets() {
    const keep = { };
    keep[S.activeScene] = 1; keep[S.activeScene - 1] = 1; keep[S.activeScene + 1] = 1;
    while (S.fleetLRU.length > SCENE_FLEET_CAP) {
      let idx = -1;
      for (let i = 0; i < S.fleetLRU.length; i++) {
        if (!keep[S.fleetLRU[i]]) { idx = i; break; }
      }
      if (idx < 0) break;               // everything left is a protected neighbour
      destroyFleet(S.fleetLRU.splice(idx, 1)[0]);
    }
  }

  function destroyFleet(s) {
    const fl = S.fleets.get(s);
    if (!fl) return;
    for (let i = 0; i < fl.videos.length; i++) {
      const v = fl.videos[i];
      try { v.pause(); v.removeAttribute("src"); v.load(); v.remove(); } catch (e) {}
    }
    S.fleets.delete(s);
  }

  function fleetReady(fl) {
    if (!fl || !fl.videos.length) return false;
    for (let i = 0; i < fl.videos.length; i++) {
      if (!fl.videos[i] || fl.videos[i].readyState < 2) return false;
    }
    return true;
  }

  /* Seek every channel (incl. the master) to the exact frame for in-scene
     output time `tLocal` -- tight tolerance, the correct paused-scrub use of
     currentTime. Channel i's file time is start[i] + tLocal (start[i] skips it
     to the scene's shared origin), so this reproduces the export frame. */
  function seekFleetTo(fl, s, tLocal) {
    const entry = S.scenePlan[s];
    for (let i = 0; i < fl.videos.length; i++) {
      const v = fl.videos[i];
      if (!v || !v.src) continue;
      let want = toNum(entry.channels[i].start, 0) + tLocal;
      if (want < 0) want = 0;
      if (!v.paused) { try { v.pause(); } catch (e) {} }
      if (Math.abs(v.currentTime - want) > 0.02) {
        try { v.currentTime = want; } catch (e) {}
      }
    }
  }

  /* Hold the followers (channels 1..N) in step with the master while playing.
     Same free-run strategy as syncChannels: loose tolerance so a nudge does
     not fight the decoder. want_i = masterTime + (start[i] - start[0]). */
  function syncSceneFleet(s, masterTime) {
    const fl = S.fleets.get(s);
    const entry = S.scenePlan[s];
    if (!fl || !entry || !entry.channels) return;
    const base = toNum(entry.channels[0] && entry.channels[0].start, 0);
    for (let i = 1; i < entry.channels.length; i++) {
      const v = fl.videos[i];
      if (!v || !v.src) continue;
      let want = masterTime + (toNum(entry.channels[i].start, 0) - base);
      if (!isFinite(want)) continue;
      if (want < 0) want = 0;
      if (S.playing) {
        if (Math.abs(v.currentTime - want) > 0.15) {
          try { v.currentTime = want; } catch (e) {}
        }
        if (v.paused) { const q = v.play(); if (q && q.catch) q.catch(function () {}); }
      } else {
        if (!v.paused) { try { v.pause(); } catch (e) {} }
        if (Math.abs(v.currentTime - want) > 0.02) {
          try { v.currentTime = want; } catch (e) {}
        }
      }
    }
  }

  /* A fleet element fired seeked/canplay/loadedmetadata. If it is the active
     scene and we are paused, swap the still floor out for the live composite
     the moment every channel has a frame. Also un-holds a seam that was
     waiting on a neighbour to buffer. */
  function onFleetProgress(s) {
    if (!S.playing && s === S.activeScene) {
      const fl = S.fleets.get(s);
      if (fl && fleetReady(fl)) {
        S.stillShown = false;
        ui.still.classList.remove("show");
        drawMulti(S.playhead - S.seamTimes[s]);
        drawCardOverlay();   // composite now carries the chrome -> overlay hides
      }
    }
    maybeResumeFromHold();
  }

  /* Warm the next scene's fleet before the seam so the crossing is seamless:
     build it (LRU keeps it as a protected neighbour) and seek every channel to
     its frame-0 target so it is decoding the seam frame, not frame 0 of file. */
  function prebufferScene(s) {
    if (s < 0 || s >= S.scenePlan.length) return;
    const fl = getFleet(s);
    if (fl.prebuffered) return;
    fl.prebuffered = true;
    const entry = S.scenePlan[s];
    for (let i = 0; i < fl.videos.length; i++) {
      const want = toNum(entry.channels[i].start, 0);
      try { fl.videos[i].currentTime = want; } catch (e) {}
    }
  }

  function startSceneMaster(s, tLocal) {
    activateScene(s);
    const fl = getFleet(s);
    const master = fl.videos[0];
    const want = sceneStart0(s) + Math.max(0, tLocal);
    try { master.currentTime = want; } catch (e) {}
    const q = master.play();
    if (q && q.catch) q.catch(function () { S.playing = false; setPlayIcon(); });
  }

  function playScene() {
    const tr = trimRange();
    if (S.playhead >= tr.end - 0.02) { S.playhead = tr.start; }
    closeMarkerPop();
    S.playing = true;
    S.stillShown = false;
    ui.still.classList.remove("show");
    ui.pinDot.hidden = true;
    const r = resolveScene(S.playhead);
    startSceneMaster(r.s, r.kLocal / S.sceneFps);
    setPlayIcon();
    requestAnimationFrame(sceneTick);
  }

  function crossToScene(next) {
    activateScene(next);
    const fl = getFleet(next);
    const master = fl.videos[0];
    try { master.currentTime = sceneStart0(next); } catch (e) {}
    const q = master.play();
    if (q && q.catch) q.catch(function () {});
    S.playhead = S.seamTimes[next];
  }

  /* Prebuffer lost the decode race at a seam: freeze on the seam frame (via
     the server still), keep buffering the next fleet, resume when it is
     ready. Never a torn/wrong frame -- a bounded freeze only, and only when
     the next scene could not be warmed in time. */
  function holdAtSeam(next) {
    const cur = S.fleets.get(S.activeScene);
    if (cur && cur.videos[0]) { try { cur.videos[0].pause(); } catch (e) {} }
    S.playing = false;
    setPlayIcon();
    S.playhead = S.seamTimes[next];
    updatePlayheadUI();
    S.pendingResumeScene = next;
    prebufferScene(next);
    scheduleStill();
    pollResumeFromHold();
  }

  /* The un-hold is normally driven by the next fleet's media events -- but
     `prebufferScene` is idempotent, so a fleet that was already warmed (and
     already fired its seeked/canplay) emits NOTHING new after the hold is
     armed. Then no event ever arrives and the take sits frozen on the seam
     forever (seen live: a gapfree-join seam, whose survivors need a deep seek
     into one continuous file and so routinely miss the 0.6s lead). This poll
     is the floor under that race; it stops the moment the hold clears. */
  function pollResumeFromHold() {
    if (S.pendingResumeScene == null) return;
    maybeResumeFromHold();
    if (S.pendingResumeScene != null) setTimeout(pollResumeFromHold, 100);
  }

  function maybeResumeFromHold() {
    if (S.pendingResumeScene == null) return;
    const s = S.pendingResumeScene;
    const fl = S.fleets.get(s);
    if (fl && fleetReady(fl)) {
      S.pendingResumeScene = null;
      activateScene(s);
      S.playing = true;
      S.stillShown = false;
      ui.still.classList.remove("show");
      setPlayIcon();
      const master = fl.videos[0];
      try { master.currentTime = sceneStart0(s); } catch (e) {}
      const q = master.play();
      if (q && q.catch) q.catch(function () {});
      requestAnimationFrame(sceneTick);
    }
  }

  function sceneTick() {
    if (!S.playing || !sceneLive()) return;
    const s = S.activeScene;
    const fl = S.fleets.get(s);
    if (!fl || !fl.videos[0]) { pause(); return; }
    const master = fl.videos[0];
    const tr = trimRange();
    const sceneDur = S.scenePlan[s].frame_count / S.sceneFps;
    let tLocal = master.currentTime - sceneStart0(s);
    if (tLocal < 0) tLocal = 0;
    // Scene exhausted -> cross the seam, HOLD, or end the take.
    if (master.ended || tLocal >= sceneDur - 0.5 / S.sceneFps) {
      if (s + 1 < S.scenePlan.length) {
        const next = getFleet(s + 1);
        if (fleetReady(next)) {
          crossToScene(s + 1);
          requestAnimationFrame(sceneTick);
        } else {
          holdAtSeam(s + 1);
        }
        return;
      }
      pause();
      seek(tr.end);
      return;
    }
    S.playhead = Math.min(S.seamTimes[s] + tLocal, tr.end);
    syncSceneFleet(s, master.currentTime);
    drawMulti(tLocal);
    updatePlayheadUI();
    if (sceneDur - tLocal < SCENE_LEAD_SEC) prebufferScene(s + 1);
    if (S.playhead >= tr.end - 0.015) { pause(); seek(tr.end); return; }
    requestAnimationFrame(sceneTick);
  }

  /* ctx.roundRect is too new to rely on here (and this also runs in WKWebView),
     so trace the rounded rect by hand. */
  function roundRectPath(ctx, x, y, w, h, r) {
    r = Math.max(0, Math.min(r, w / 2, h / 2));
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  /* The selection frame a click-selected card wears (drawMulti draws it over
     the card): a straight bounding rectangle plus round corner handles and
     pill edge handles -- the familiar design-tool look from the reference.
     `full` draws the handles (a selected / dragging card); a bare hover passes
     false for just the faint outline that signals "selectable". There is NO
     rotation handle by design: framing stretches each window's crop over an
     axis-aligned rect, so a rotated card would have nowhere to render. All
     eight handles resize aspect-locked (onCardMove), so a resized card never
     distorts its recording. Colour is a selection-blue, distinct from the
     teal the camera/zoom system reserves. */
  const CARD_SEL = "#3b82f6";
  function drawCardChrome(ctx, dx, dy, dw, dh, dpr, full) {
    ctx.save();
    ctx.strokeStyle = CARD_SEL;
    ctx.lineWidth = (full ? 1.5 : 1.25) * dpr;
    if (!full) ctx.globalAlpha = 0.55;
    ctx.strokeRect(dx, dy, dw, dh);
    if (!full) { ctx.restore(); return; }
    const cx = dx + dw / 2, cy = dy + dh / 2;
    const pillW = 4 * dpr, pillL = 9 * dpr;   // edge pill: half-thickness, half-length
    const r = 5 * dpr;                         // corner handle radius
    ctx.lineWidth = 1.5 * dpr;
    ctx.fillStyle = "#ffffff";
    // Edge pills first, so a corner handle draws cleanly on top where they meet.
    // [centreX, centreY, halfW, halfH] -- horizontal on the top/bottom edges,
    // vertical on the left/right.
    const edges = [[cx, dy, pillL, pillW], [cx, dy + dh, pillL, pillW],
                   [dx, cy, pillW, pillL], [dx + dw, cy, pillW, pillL]];
    for (let e = 0; e < edges.length; e++) {
      const g = edges[e];
      roundRectPath(ctx, g[0] - g[2], g[1] - g[3], g[2] * 2, g[3] * 2,
                    Math.min(g[2], g[3]));
      ctx.fill();
      ctx.stroke();
    }
    const corners = [[dx, dy], [dx + dw, dy], [dx, dy + dh], [dx + dw, dy + dh]];
    for (let q = 0; q < corners.length; q++) {
      ctx.beginPath();
      ctx.arc(corners[q][0], corners[q][1], r, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  /* Selection chrome mirrored onto an always-on-top overlay, so the frame +
     handles are visible over the opaque server STILL. drawMulti draws the
     chrome onto the live composite -- but a SCENE take shows the still while
     its fleet warms, and if the browser can't decode the channel .movs at all,
     drawMulti early-returns to the backdrop before it ever reaches the chrome
     (render.py's cold-fleet floor), so the resize handles were invisible and
     resizing looked impossible. This overlay decouples the affordance from the
     fleet: while the Windows panel is open on a native/scene take and the still
     covers the composite, every card gets a faint "grab me" outline and the
     selected one wears the full frame + handles. Hidden whenever the live
     composite is up (multi-native, or a warm scene) so nothing is double-drawn.
     Pure canvas geometry, reusing drawMulti's own transform (a layout point c
     lands at fit.o + z*k*(c - c0)); the pointer still lands on .ed-multi below,
     since this canvas is click-through. */
  function drawCardOverlay() {
    const cv = ui.cardChrome;
    if (!cv) return;
    const octx = cv.getContext("2d");
    const p = S.camPath;
    // Arranging is a PAUSED activity: never draw the outline/handles during
    // playback. Otherwise a scene take's brief fleet re-buffer (e.g. right after
    // a backward scrub) drops the composite to its bare backdrop for a frame or
    // two, and the overlay would paint empty boxes over it -- reading as
    // "the windows vanished" when they are merely reseeking.
    const arranging = S.panel === "windows" && !placementBlocked() && !S.playing;
    // Only while the still (or the bare backdrop) is what's showing -- when the
    // live composite is up it carries its own chrome from drawMulti.
    const compositeLive = multiReady() || (sceneLive() && sceneFleetReady());
    if (!arranging || compositeLive || !p || !p.cells || !p.canvas) {
      if (!cv.hidden) {
        if (octx) octx.clearRect(0, 0, cv.width, cv.height);
        cv.hidden = true;
      }
      return;
    }
    const cw = ui.canvas.clientWidth, ch = ui.canvas.clientHeight;
    if (cw <= 0 || ch <= 0 || !octx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const bw = Math.max(1, Math.round(cw * dpr));
    const bh = Math.max(1, Math.round(ch * dpr));
    if (cv.width !== bw || cv.height !== bh) { cv.width = bw; cv.height = bh; }
    const t = cellTime();
    const fit = compositeFit(bw, bh, p.canvas);
    const k = fit.k;
    const focus = activeFocusFrame(p, t);
    const entMorph = entranceCells(p, t);
    const cells = entMorph || (focus ? focus.cells : p.cells);
    const cam = entMorph ? null : focusCameraWindow(p, focus);
    const z = cam ? cam.z : 1;
    octx.setTransform(1, 0, 0, 1, 0, 0);
    octx.clearRect(0, 0, bw, bh);
    for (let i = 0; i < cells.length; i++) {
      const c = cells[i];
      const dx = c.x * k, dy = c.y * k, dw = c.w * k, dh = c.h * k;
      const sx = fit.ox + (cam ? z * (dx - cam.x0 * k) : dx);
      const sy = fit.oy + (cam ? z * (dy - cam.y0 * k) : dy);
      const sel = (S.cardSel === i) || (S.cardDrag && S.cardDrag.index === i);
      drawCardChrome(octx, sx, sy, dw * z, dh * z, dpr, sel);
    }
    cv.hidden = false;
  }

  /* Resize cursor for a grabbed handle (corner or edge); "move" for the body. */
  function cardCursor(corner) {
    switch (corner) {
      case "nw": case "se": return "nwse-resize";
      case "ne": case "sw": return "nesw-resize";
      case "n": case "s": return "ns-resize";
      case "e": case "w": return "ew-resize";
      default: return "move";
    }
  }

  /* --- facecam bubble, live ---
   *
   * The paused still gets this from the server for free (render.py composites
   * it after the grid). Playback used to just drop it, so the bubble blinked
   * out the moment you pressed play -- the single most visible way the live
   * preview disagreed with the export.
   *
   * Unlike the grid, this geometry is NOT handed over by the server: it is
   * cheap, and re-deriving it beats a per-seek round trip. It therefore has
   * to mirror effects.FacecamOverlay.__init__ exactly -- diameter and margin
   * are fractions of output HEIGHT (not width), the source frame is
   * centre-cropped square before scaling, and it is selfie-mirrored. Change
   * one of those there and this drifts.
   */
  function faceMetrics(canvasW, canvasH) {
    const r = S.edits && S.edits.render ? S.edits.render : {};
    const D = Math.min(Math.max(16, Math.round(toNum(r.facecam_size, 0.2) * canvasH)),
                       canvasW, canvasH);
    const margin = Math.round(0.035 * canvasH);       // FACECAM_DEFAULTS.margin_frac
    const pos = String(r.facecam_position || "bottom-left").toLowerCase();
    const px = pos.indexOf("left") >= 0 ? margin : Math.max(0, canvasW - D - margin);
    const py = pos.indexOf("top") >= 0 ? margin : Math.max(0, canvasH - D - margin);
    return { D: D, px: px, py: py, shape: String(r.facecam_shape || "circle"),
             // FacecamOverlay: border_th = round(border_frac * D / 2). Not
             // reachable from the Facecam panel (position/size/shape only),
             // but it IS a normalized, server-honored edits key, so a ring
             // set via the API or by hand must not blink on play.
             border: Math.max(0, Math.round(toNum(r.facecam_border, 0) * D / 2)),
             // Background blur strength, 0..1. Mirrors effects.FacecamOverlay's
             // vignette so the bubble doesn't sharpen the instant you press play.
             blur: Math.max(0, Math.min(1, toNum(r.facecam_blur, 0))) };
  }

  /* Compose the facecam content (sharp centre, blurred rim) into a reusable
     offscreen canvas, mirroring effects.FacecamOverlay's `_vignette`: a blurred
     copy is masked by a radial alpha that is transparent across the inner half
     (core = 0.5 of the radius) and opaque at the rim, then laid over the sharp
     base. Kept module-level so playback doesn't allocate a canvas per frame.
     ctx.filter carries the blur; where it is unsupported the copy stays sharp,
     which merely no-ops the effect in live play (the paused still is exact). */
  var _faceOff = null, _faceTmp = null;
  function _faceCanvas(which, size) {
    var c = which === 0 ? _faceOff : _faceTmp;
    if (!c) { c = document.createElement("canvas"); if (which === 0) _faceOff = c; else _faceTmp = c; }
    if (c.width !== size || c.height !== size) { c.width = size; c.height = size; }
    return c;
  }
  function faceBlurBubble(src, sx, sy, s, dd, blur) {
    var n = Math.max(1, Math.round(dd));
    var off = _faceCanvas(0, n), octx = off.getContext("2d");
    octx.clearRect(0, 0, n, n);
    octx.drawImage(src, sx, sy, s, s, 0, 0, n, n);          // sharp base
    var tmp = _faceCanvas(1, n), tctx = tmp.getContext("2d");
    tctx.clearRect(0, 0, n, n);
    tctx.filter = "blur(" + (blur * n * 0.06) + "px)";      // ~ effects._blur_sigma
    tctx.drawImage(src, sx, sy, s, s, 0, 0, n, n);          // blurred copy
    tctx.filter = "none";
    tctx.globalCompositeOperation = "destination-in";
    var g = tctx.createRadialGradient(n / 2, n / 2, n / 4, n / 2, n / 2, n / 2);
    g.addColorStop(0, "rgba(0,0,0,0)");                     // core: fully sharp
    g.addColorStop(1, "rgba(0,0,0,1)");                     // rim: fully blurred
    tctx.fillStyle = g;
    tctx.fillRect(0, 0, n, n);
    tctx.globalCompositeOperation = "source-over";
    octx.drawImage(tmp, 0, 0);                              // blurred rim over sharp
    return off;
  }

  function drawFaceBubble(ctx, canvasW, canvasH, k) {
    const r = S.edits && S.edits.render ? S.edits.render : {};
    if (r.facecam === false) return;
    const v = ui.face;
    const snap = S.dragSnap && S.dragSnap.face;   // frozen frame, see snapFrame
    if (!snap && (!v || !v.src || v.readyState < 2)) return;
    const m = faceMetrics(canvasW, canvasH);
    if (m.px < 0 || m.py < 0 || m.px + m.D > canvasW || m.py + m.D > canvasH) return;
    const vw = snap ? snap.width : v.videoWidth;
    const vh = snap ? snap.height : v.videoHeight;
    if (!vw || !vh) return;
    // centre-crop to a square, exactly as FacecamOverlay.draw does
    const s = Math.min(vw, vh);
    const sx = (vw - s) / 2, sy = (vh - s) / 2;
    const dx = m.px * k, dy = m.py * k, dd = m.D * k;

    ctx.save();
    if (m.shape === "rounded") roundRectPath(ctx, dx, dy, dd, dd, 0.32 * dd);
    else {
      ctx.beginPath();
      ctx.arc(dx + dd / 2, dy + dd / 2, dd / 2, 0, Math.PI * 2);
      ctx.closePath();
    }
    // The server feathers a drop shadow under the bubble; a plain canvas
    // shadow is the closest cheap equivalent and reads the same at this size.
    ctx.shadowColor = "rgba(0,0,0,0.38)";
    ctx.shadowBlur = 0.1 * dd;
    ctx.shadowOffsetY = Math.max(1, dd / 40);
    ctx.fill();
    ctx.shadowColor = "transparent";
    ctx.clip();
    // selfie-mirror (FACECAM_DEFAULTS.mirror)
    ctx.translate(dx + dd, dy);
    ctx.scale(-1, 1);
    try {
      if (m.blur > 0) {
        // Pre-composite the vignette (radially symmetric, so mirroring it here
        // is harmless) and draw it like the raw frame would have been.
        ctx.drawImage(faceBlurBubble(snap || v, sx, sy, s, dd, m.blur), 0, 0, dd, dd);
      } else {
        ctx.drawImage(snap || v, sx, sy, s, s, 0, 0, dd, dd);
      }
    } catch (e) { /* frame not decodable yet */ }
    ctx.restore();

    if (m.border > 0) {
      // effects.FacecamOverlay: a circle of radius D/2 - border_th/2, stroked
      // border_th wide, in FACECAM_DEFAULTS.border_color (BGR 248,244,240).
      const th = m.border * k;
      ctx.save();
      ctx.beginPath();
      ctx.arc(dx + dd / 2, dy + dd / 2, dd / 2 - th / 2, 0, Math.PI * 2);
      ctx.lineWidth = th;
      ctx.strokeStyle = "rgb(240,244,248)";
      ctx.stroke();
      ctx.restore();
    }
  }

  /* face.mov and raw.mov are two independent media elements on one timeline;
     `face_offset` (screen media time -> face media time) comes off
     /api/session, derived from the record-time monotonic anchors. */
  function syncFace(t) {
    const v = ui.face;
    if (!v || !v.src || !S.details) return;
    // Facecam switched off: nothing consumes these frames, so don't keep the
    // webcam track decoding behind an invisible bubble.
    if (S.edits && S.edits.render && S.edits.render.facecam === false) {
      if (!v.paused) { try { v.pause(); } catch (e) {} }
      return;
    }
    // Clamped, not rejected: a negative offset makes `want` < 0 for the first
    // few frames, and bailing there would leave the bubble on a stale frame
    // at exactly t=0. FacecamOverlay._frame_at clamps the same way.
    let want = t + toNum(S.details.face_offset, 0);
    if (!isFinite(want)) return;
    if (want < 0) want = 0;
    if (S.playing) {
      // Let it run and only nudge on real drift, or every frame's seek would
      // fight the decoder and the bubble would stutter.
      if (Math.abs(v.currentTime - want) > 0.15) {
        try { v.currentTime = want; } catch (e) {}
      }
      if (v.paused) { const q = v.play(); if (q && q.catch) q.catch(function () {}); }
    } else {
      if (!v.paused) { try { v.pause(); } catch (e) {} }
      if (Math.abs(v.currentTime - want) > 0.02) {
        try { v.currentTime = want; } catch (e) {}
      }
    }
  }

  /* --- multi-native channels, live ---
   *
   * An occlusion-free multi-window take has NO raw.mov: every window is its
   * own `raw_i.mov`, so a single <video> has nothing to point at and the
   * player used to sit on a silent 404. The composite is drawn from N
   * elements instead -- channel 0 IS `ui.video` (which keeps play/seek/
   * playbackTick working unchanged, since they are all built on it), and the
   * rest are offscreen followers kept in step exactly the way the facecam is.
   *
   * They live at 1x1/opacity:0 rather than `hidden` for the same reason
   * `.ed-face-src` does: display:none is a state engines may stop decoding in.
   */
  function setupChannelVideos() {
    S.chanVideos = [];
    const chans = (S.details && S.details.capture_channels) || [];
    for (let i = 1; i < chans.length; i++) {
      const v = document.createElement("video");
      v.className = "ed-face-src";      // 1x1, opacity 0, still decoding
      v.muted = true;                   // only ui.video is ever unmuted
      v.playsInline = true;
      v.preload = "auto";
      v.src = autocineUrl(
        "/api/media/" + enc(S.session) + "/channel" + i);
      // A seek is async: repaint when the frame actually lands, or a scrub
      // composites the frame we were showing before it.
      v.addEventListener("seeked", function () {
        if (!S.playing && multiReady()) drawMulti();
      });
      v.addEventListener("loadedmetadata", function () {
        applyTransformAt(S.playhead);
      });
      document.body.appendChild(v);
      S.chanVideos[i] = v;
    }
  }

  /* Seconds of channel 0 that the composite SKIPS -- the ONE conversion
     between the <video> element's clock and the playhead's.

     `_render_multi_native` grab()-skips every channel to the shared origin, so
     the composite's output frame 0 is channel 0's FILE frame
     `origin - frame_offsets[0]`, i.e. `channels[0].start` seconds in. The
     element we play is raw_0.mov served verbatim, so `currentTime` is file
     time -- while `S.playhead`, `details.duration`, the ruler's click ticks
     and every per-card zoom/focus track are OUTPUT time. Reading one as the
     other put the whole overlay layer `start[0]` ahead of the picture.

     Zero for every other take shape, so nothing but a lagged fleet moves. A
     scene take converts in `seekFleetTo`/`sceneStart0`, which add `start[i]`
     for EVERY channel INCLUDING the master -- the same rule; this is the flat
     player finally following it. */
  function fleetLead() {
    if (sceneLive()) return 0;
    const p = S.camPath;
    if (!p || !p.multi_native || !p.channels || !p.channels.length) return 0;
    return toNum(p.channels[0].start, 0);
  }

  /* Hold every follower channel at OUTPUT time `t`. Same strategy as
     syncFace, and for the same reasons: loose drift tolerance while playing
     so seeks don't fight the decoder, tight while paused so a scrub lands
     exactly.

     `start` is per channel and comes off the manifest -- the streams begin at
     slightly different wall-clock instants, and the export skips each one to
     the shared origin. This used to take offsets RELATIVE to channel 0 and
     note that the preview therefore sat up to `start[0]` earlier in the take
     than the export; `fleetLead` closes that gap instead of documenting it. */
  function syncChannels(t) {
    const p = S.camPath;
    if (!p || !p.multi_native || !p.channels || !S.chanVideos) return;
    for (let i = 1; i < p.channels.length; i++) {
      const v = S.chanVideos[i];
      if (!v || !v.src) continue;
      // `t` is OUTPUT time, so every channel -- master included, via
      // `fleetLead` -- is its own `start` past it. Cancelling `start[0]` here
      // is what made the master the odd one out.
      let want = toNum(p.channels[i].start, 0) + t;
      if (!isFinite(want)) continue;
      if (want < 0) want = 0;
      if (S.playing) {
        if (Math.abs(v.currentTime - want) > 0.15) {
          try { v.currentTime = want; } catch (e) {}
        }
        if (v.paused) { const q = v.play(); if (q && q.catch) q.catch(function () {}); }
      } else {
        if (!v.paused) { try { v.pause(); } catch (e) {} }
        if (Math.abs(v.currentTime - want) > 0.02) {
          try { v.currentTime = want; } catch (e) {}
        }
      }
    }
  }

  /* Which element cell `i` samples. One shared recording for every other
   * mode; one video PER CELL for multi-native. Returns null for a follower
   * with no frame yet, so that card is skipped for this paint rather than
   * throwing -- the readyState gate in drawMulti only covers channel 0. */
  function cellSource(p, i, vsrc) {
    // Scene take: every cell (channel 0 included) samples the ACTIVE fleet's
    // own <video>; there is no shared master recording here.
    if (sceneLive()) {
      const fl = S.fleets && S.fleets.get(S.activeScene);
      const fv = fl && fl.videos[i];
      return (fv && fv.readyState >= 2) ? fv : null;
    }
    if (!p || !p.multi_native || i === 0) return vsrc || ui.video;
    const v = S.chanVideos && S.chanVideos[i];
    return (v && v.readyState >= 2) ? v : null;
  }

  /* --- synthetic cursor, live ---
   *
   * Windows mode composites the cursor server-side (render._draw_multi_cursor),
   * so without this it would blink out on play exactly like the facecam did.
   * The GLYPH, its size, the EMA smoothing, the idle fade and the click pulse
   * all stay defined once in effects.CursorFX -- `/api/camera-path` hands over
   * the sampled track (CursorFX.sample_track) and this only replays it, the
   * same division of labour as the grid's cells/plate.
   *
   * The mapping is the cell's own transform, i.e. literally the argument pair
   * drawImage already uses below: src rect -> destination box. Anything the
   * cursor is not inside is skipped, so it can't ghost onto a neighbouring
   * card, and each card clips it.
   */
  /* --- per-card auto-zoom, live ---
   *
   * With render.window_zoom on, each card has its own camera and only the
   * card with the most recent activity is zoomed (see camera.build_card_paths).
   * The composite expresses that by sampling a SMALLER SOURCE RECT for that
   * cell -- drawImage already takes one, so zooming a card costs nothing but
   * the right numbers. `/api/camera-path` ships the sampled (cx, cy, z), and
   * this reproduces render._camera_window's clamp so the browser and the
   * export frame identically.
   */
  function cardSrcRect(p, i, t) {
    const cell = p.cells[i];
    const src = cell.src;
    const cp = p.card_paths;
    const path = cp && cp.cards ? cp.cards[i] : null;
    if (!path) return src;
    const j = Math.max(0, Math.min(path.z.length - 1,
                                   Math.round(t * cp.fps / cp.stride)));
    const z = Math.max(1, path.z[j]);
    // The card's zoom-1.0 window is the WHOLE card rect (never a cell-aspect
    // sub-box): at z == 1 that reproduces the plain stretch the compositor
    // does, so turning zoom on cannot crop content. Mirrors
    // render._card_to_cell.
    const dw = src[2], dh = src[3];
    const cw = dw / z, ch = dh / z;
    // _camera_window's clamp, in card space, then back into raw file px.
    const x0 = Math.min(Math.max(path.cx[j] - cw / 2, 0), dw - cw);
    const y0 = Math.min(Math.max(path.cy[j] - ch / 2, 0), dh - ch);
    return [src[0] + x0, src[1] + y0, cw, ch];
  }

  /* The capture-indicator repair for one card, in destination px.
   * `src` is the source rect drawImage sampled (which is what a per-card zoom
   * changes), so the badge tracks a zoomed card for free -- and lands
   * off-card, harmlessly clipped, when the zoom framed past it. */
  function drawBadgePatch(ctx, p, i, src, dx, dy, dw, dh) {
    const chans = p && p.channels;
    const b = chans && chans[i] && chans[i].badge;
    if (!b || !src || !src[2] || !src[3]) return;
    const fx = dw / src[2], fy = dh / src[3];
    ctx.fillStyle = b.color;
    ctx.fillRect(dx + (b.rect[0] - src[0]) * fx,
                 dy + (b.rect[1] - src[1]) * fy,
                 b.rect[2] * fx, b.rect[3] * fy);
  }

  function drawMultiCursor(ctx, p, k, t, cells) {
    const c = p && p.cursor;
    if (!c || !c.x || !c.x.length) return;
    const i = Math.max(0, Math.min(c.x.length - 1,
                                   Math.round(t * c.fps / c.stride)));
    const alpha = c.a[i];
    if (!(alpha > 0)) return;
    const cx = c.x[i], cy = c.y[i];
    cells = cells || p.cells;
    for (let n = 0; n < cells.length; n++) {
      const cell = cells[n];
      // Containment is decided on the rect the card SHOWS (so the cursor
      // doesn't ghost onto a neighbour), and the mapping uses the same rect
      // the video crop was drawn from -- which is the camera window when that
      // card is zoomed. Same composition render._card_to_cell does.
      const s = cardSrcRect(p, n, t);
      if (!s || cx < s[0] || cx >= s[0] + s[2] || cy < s[1] || cy >= s[1] + s[3]) continue;
      const zx = (cell.w * k) / s[2], zy = (cell.h * k) / s[3];
      const ox = cell.x * k + (cx - s[0]) * zx;
      const oy = cell.y * k + (cy - s[1]) * zy;
      // Sized by the x scale only -- a stretched glyph would read as a bug,
      // matching CursorFX.draw's use of z_eff (never z_eff_y) for size.
      const size = c.size * zx * (1 + c.pulse[i]);
      const ang = c.tilt[i] * Math.PI / 180;
      const cosA = Math.cos(ang), sinA = Math.sin(ang);
      ctx.save();
      roundRectPath(ctx, cell.x * k, cell.y * k, cell.w * k, cell.h * k,
                    cell.radius * k);
      ctx.clip();
      const path = function (dx, dy) {
        ctx.beginPath();
        for (let j = 0; j < c.shape.length; j++) {
          const px = c.shape[j][0] * size, py = c.shape[j][1] * size;
          const rx = px * cosA - py * sinA + ox + dx;
          const ry = px * sinA + py * cosA + oy + dy;
          if (j === 0) ctx.moveTo(rx, ry); else ctx.lineTo(rx, ry);
        }
        ctx.closePath();
      };
      const off = Math.max(1, Math.round(c.size * zx * c.shadow_offset_frac));
      ctx.globalAlpha = c.shadow_alpha * alpha;
      path(off, off);
      ctx.fillStyle = "#000";
      ctx.fill();
      ctx.globalAlpha = alpha;
      path(0, 0);
      ctx.fillStyle = "#fff";
      ctx.fill();
      ctx.lineWidth = Math.max(1, size * 0.05);
      ctx.strokeStyle = "rgb(30,30,30)";
      ctx.stroke();
      ctx.restore();
    }
  }

  /* --- window focus, live ---
   *
   * With render.window_focus on, the LAYOUT re-weights: the card you are
   * working in grows and the others shrink toward the edge they were
   * already nearest (see framing.focus_placements). The browser is handed
   * finished per-frame placements rather than the emphasis track, so
   * `framing` stays the only place a layout is ever decided -- re-deriving
   * it in JS is exactly how the paused server render and the playing canvas
   * would drift apart.
   *
   * `frames[i]` is `{c: [[x, y, w, h, radius], ...], o: [draw order],
   * l: [lift per card]}`. The order matters: a demoted card travelling
   * toward the strip passes BEHIND the growing hero. `l` is how far each
   * card has grown toward its own focused size, which is what its drop
   * shadow is scaled by (see drawCardShadow); absent on frames where every
   * card is at rest, which is most of them.
   */
  function focusFrameAt(p, t) {
    const f = p && p.focus_cells;
    if (!f || !f.frames || !f.frames.length) return null;
    const j = Math.max(0, Math.min(f.frames.length - 1,
                                   Math.round(t * f.fps / f.stride)));
    const fr = f.frames[j];
    if (!fr || !fr.c) return null;
    return {
      cells: fr.c.map(function (c) {
        return { x: c[0], y: c[1], w: c[2], h: c[3], radius: c[4] };
      }),
      order: fr.o || fr.c.map(function (_c, i) { return i; }),
      lifts: fr.l || null,
      cam: fr.z || null,
    };
  }

  /* Stage 2's camera window in layout px, or null below it.
   *
   * Window focus is a two-rung ladder: the card grows (stage 1, expressed in
   * the cells above), then the WHOLE composition zooms in on it (stage 2,
   * this). The export warps its composed canvas; the browser applies the
   * same numbers as one transform over the plate, every cell and the cursor
   * -- so the two cannot frame differently. Mirrors render._camera_window's
   * clamp. */
  function focusCameraWindow(p, focus) {
    if (!focus || !focus.cam) return null;
    const z = Math.max(1, focus.cam[2]);
    if (z <= 1 + 1e-9) return null;
    const lw = p.canvas[0], lh = p.canvas[1];
    const ww = lw / z, wh = lh / z;
    return {
      z: z,
      x0: Math.min(Math.max(focus.cam[0] - ww / 2, 0), lw - ww),
      y0: Math.min(Math.max(focus.cam[1] - wh / 2, 0), lh - wh),
    };
  }

  /* The focus frame governing what is ON SCREEN at `t`, or null.
   *
   * Null for the duration of a card drag, on purpose: a drag edits the BASE
   * layout, and the focus frames are the server's finished placements --
   * moving `p.cells` cannot move them, so the card would sit frozen under
   * the pointer until release. Dropping to the base layout while dragging
   * shows exactly the thing being edited. Both the painter and the pointer
   * mapping go through here, so they can never disagree about which layout
   * is on screen. */
  function activeFocusFrame(p, t) {
    if (S.cardDrag) return null;
    return focusFrameAt(p, t);
  }

  /* The cells to draw at `t` -- the animated layout when focus is running,
     else the static ones the layout endpoint shipped. Every consumer (the
     card crops, the cursor, the drag picker) goes through this, so none of
     them has to know whether the layout is moving. */
  function cellsAt(p, t) {
    const f = activeFocusFrame(p, t);
    return f ? f.cells : p.cells;
  }

  /* The time the compositor indexes per-card tracks (focus/zoom) by: the
     take-level playhead for every mode EXCEPT a scene take, whose per-scene
     tracks are sampled from the scene's own frame 0 -- so the hit-test and the
     draw must both use IN-scene time or they land on the wrong frame. */
  function cellTime() {
    return sceneLive() ? (S.playhead - S.seamTimes[S.activeScene]) : S.playhead;
  }

  /* The layers of ONE card's drop shadow: `[blur, offsetY, alpha]` in layout
     px, mirroring framing.MultiFramePainter._shadow_layers. `shadowBlur` is
     twice the Gaussian sigma by spec, which is why 0.02*W of server sigma is
     0.04*W here.

     `lift` is 0 for a card at its base size and 1 for a fully grown one. At
     rest that is the single soft cast layer the server bakes into the plate.
     Lifted, the card is sitting on its NEIGHBOURS rather than on the
     backdrop, and 0.04*W of blur spread over another window's bright content
     is a gradient, not a boundary -- measured on the server's compositor, a
     grown card over a bright neighbour had 1.6 levels of edge contrast. So
     the cast layer deepens and throws further, and a tight contact layer
     hugs the edge to actually draw the boundary. */
  function cardShadowLayers(p, lift) {
    const blur = 0.04 * p.canvas[0], dy = 0.012 * p.canvas[1];
    if (!(lift > 0)) return [[blur, dy, 0.45]];
    const at = function (rest, full) { return rest + lift * (full - rest); };
    return [[blur * at(1, 1.7), dy * at(1, 2.2), at(0.45, 0.66)],
            [0.016 * p.canvas[0], 0, 0.55 * lift]];
  }

  /* One card's drop shadow, drawn immediately before that card.
   *
   * The static path uses a plate with the shadows baked in by the server;
   * once cells move those shadows belong to a layout that is no longer on
   * screen, so the browser gets the bare backdrop and draws its own.
   *
   * Per card and interleaved with the cards, exactly like the server's
   * `paint_at`, NOT one pass for all of them first: window focus makes the
   * grown card overlap its neighbours, so a shadow drawn under every card
   * up front is painted over by whichever neighbour is drawn after it. The
   * subject then lands on that neighbour with no shadow between them at all
   * -- the one place the composite most needs one.
   *
   * Canvas shadowBlur is not bit-identical to the server's Gaussian, and
   * layers here stack source-over where the server multiplies. Geometry (the
   * offset, the blur, the alpha) matches; the falloff curve does not. */
  function drawCardShadow(ctx, p, c, k, lift) {
    const layers = cardShadowLayers(p, lift);
    ctx.save();
    ctx.fillStyle = "#000";
    for (let n = 0; n < layers.length; n++) {
      const L = layers[n];
      if (!(L[2] > 0.002)) continue;   // < half a level of 255; a no-op
      ctx.shadowColor = "rgba(0,0,0," + L[2].toFixed(3) + ")";
      ctx.shadowBlur = L[0] * k;
      ctx.shadowOffsetY = L[1] * k;
      roundRectPath(ctx, c.x * k, c.y * k, c.w * k, c.h * k, c.radius * k);
      ctx.fill();
    }
    ctx.restore();
  }

  /* --- the frame a drag draws from ---
   *
   * A card drag redraws the whole composite on every pointer event, and each
   * card was drawn straight from the <video>: three uploads of a 2880x1800
   * frame per redraw. Measured in a real browser on this machine, ONE
   * drawImage off that video costs ~10 ms (17 ms full-canvas) and a whole
   * composite ~38 ms -- so a 120 Hz pointer queues work far faster than the
   * browser retires it, and the card falls seconds behind the cursor. That
   * is the whole of "dragging is very delayed"; it is not the network.
   *
   * The frame cannot change during a drag (playback is paused, the playhead
   * is fixed), so it is uploaded ONCE at pointerdown and every redraw blits
   * from that canvas instead -- a plain canvas-to-canvas copy. Copied at
   * NATIVE size, so the pixels the cards sample are the same ones, and
   * `cardSrcRect`'s raw-file coordinates keep meaning what they meant. */
  function snapFrame(v) {
    if (!v || !v.src || v.readyState < 2 || !v.videoWidth) return null;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    try {
      c.getContext("2d").drawImage(v, 0, 0);
    } catch (e) { return null; }   // frame not decodable yet -- draw live
    return c;
  }

  /* Where the composite lands inside the canvas backing store: `{k, ox, oy}`.
   *
   * `k` used to be `bw / p.canvas[0]` -- WIDTH alone -- which is only right
   * while the element's box has exactly the layout's aspect. When it does not,
   * width-only scaling draws the whole composite at the wrong size anchored at
   * the ORIGIN: too tall and clipped off the bottom, or too short with a band
   * of stage grey under it and the cards riding up. fitCanvas is what keeps
   * the box honest, but it cannot be the only thing standing between a
   * mid-resize frame and that -- so contain-and-centre here, where the worst a
   * wrong box can do is an even hairline margin.
   *
   * Pure, and recomputed at each use rather than cached, because the painter
   * (drawMulti) and the pointer mapping (canvasFrac) answering this
   * differently is a card that does not follow the cursor.
   */
  function compositeFit(bw, bh, canvas) {
    const lw = canvas && canvas[0] > 0 ? canvas[0] : 0;
    const lh = canvas && canvas[1] > 0 ? canvas[1] : 0;
    if (!lw || !lh) return { k: 1, ox: 0, oy: 0 };
    const k = Math.min(bw / lw, bh / lh);
    return { k: k, ox: (bw - lw * k) / 2, oy: (bh - lh * k) / 2 };
  }

  /* Seamless-join ENTRANCE, browser side. Ports render._smoothstep +
     blend_placements (single-target, shrink-lead) EXACTLY so the live morph
     equals the exported one: the new card grows in from the centre of its
     final cell and the survivors reflow, over the first `entrance.frames` of a
     join scene. `entranceCells` returns null outside that window (baked cells
     -> a plain scene take is byte-identical). */
  var _JOIN_SHRINK_LEAD = 1.8;
  function _smoothstepJ(x) {
    x = x < 0 ? 0 : (x > 1 ? 1 : x);
    return x * x * (3 - 2 * x);
  }
  function entranceCells(p, t) {
    const e = p && p.entrance;
    if (!e || !S.sceneFps) return null;
    const km = Math.round(t * S.sceneFps);        // in-scene frame, == server k
    if (km < 0 || km >= e.frames) return null;
    const w = _smoothstepJ(e.frames > 1 ? km / (e.frames - 1) : 1);
    const from = e.from_cells, to = p.cells, out = [];
    for (let j = 0; j < to.length; j++) {
      const b = from[j], c = to[j];
      // shrink leads the move (get out of the way, then go) -- matches
      // blend_placements so travelling cards never overlap mid-morph.
      const st = (c.w < b.w) ? Math.min(1, w * _JOIN_SHRINK_LEAD) : w;
      const cw = Math.max(2, Math.round(b.w + st * (c.w - b.w)));
      const chh = Math.max(2, Math.round(b.h + st * (c.h - b.h)));
      out.push({
        x: Math.round(b.x + w * (c.x - b.x)),
        y: Math.round(b.y + w * (c.y - b.y)),
        w: cw,
        h: chh,
        // The server derives each card's corner radius from its CURRENT width
        // (framing.paint_at: max(8, 0.02*fw)); `from_cells` carry none, so
        // without this `c.radius` is undefined -> the roundRect clip gets NaN
        // and clips the card's content to nothing -> the windows go BLANK for
        // the whole morph in live playback (export/still were fine). Match the
        // server; roundRectPath clamps it to the cell's half-size.
        radius: Math.max(8, Math.round(0.02 * cw)),
        src: c.src,
      });
    }
    return out;
  }

  /* `t` is passed in, never read off S.playhead: applyTransformAt is called
     with an explicit time (the video's own currentTime from the 'seeked'
     handler, the tick's time during playback) and those can differ from
     S.playhead mid-seek -- which would sample a card's camera and the cursor
     one instant away from the frame actually being drawn. */
  function drawMulti(t) {
    if (t === undefined) t = cellTime();
    const p = S.camPath;
    const cw = ui.canvas.clientWidth, ch = ui.canvas.clientHeight;
    if (cw <= 0 || ch <= 0) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const bw = Math.max(1, Math.round(cw * dpr));
    const bh = Math.max(1, Math.round(ch * dpr));
    if (ui.multi.width !== bw || ui.multi.height !== bh) {
      ui.multi.width = bw;
      ui.multi.height = bh;
    }
    const ctx = ui.multi.getContext("2d");
    if (!ctx) return;
    const fit = compositeFit(bw, bh, p.canvas);
    const k = fit.k;                   // layout px -> canvas backing px
    // Window focus moves the cells, so the baked plate's shadows belong to a
    // layout that is no longer on screen -- draw the bare backdrop and put
    // fresh shadows under this frame's cells instead.
    const focus = activeFocusFrame(p, t);
    // A join scene's entrance morphs the cells the way focus does; when active
    // it OVERRIDES focus (per-card cameras/focus are suppressed during the
    // ~0.8s morph, matching the export) and, like focus, needs the bare
    // backdrop with fresh per-card shadows drawn over it.
    const entMorph = entranceCells(p, t);
    const cells = entMorph || (focus ? focus.cells : p.cells);
    const moving = !!(entMorph || focus);
    // A selection can outlive the card it framed (the mode changed, a window
    // was removed, a scene has fewer cards): drop it rather than draw a frame
    // around nothing or index past the end of `cells`.
    if (S.cardSel != null && (placementBlocked() || S.cardSel >= cells.length)) {
      S.cardSel = null;
    }
    const backdrop = (moving && S.bgPlate) ? S.bgPlate : S.plate;
    // Stage 2: one transform over the whole scene -- plate, cards, cursor.
    // Layered ON TOP of the k scaling every draw below already does, so
    // those keep speaking layout px: a layout point c lands at z*k*(c - x0),
    // which is the camera window filling the backing canvas. That is
    // render._warp, expressed as a matrix. The facecam is deliberately
    // outside it, matching render()'s order.
    //
    // The ox/oy in both branches is compositeFit's centring: carrying it in
    // the matrix is what lets every draw below keep speaking plain layout px.
    const cam = entMorph ? null : focusCameraWindow(p, focus);
    if (cam) {
      ctx.setTransform(cam.z, 0, 0, cam.z,
                       fit.ox - cam.x0 * cam.z * k,
                       fit.oy - cam.y0 * cam.z * k);
    } else {
      ctx.setTransform(1, 0, 0, 1, fit.ox, fit.oy);
    }
    // Offset back out of the centring so the plate still covers the WHOLE
    // store: nothing clears this canvas, and a rounding remainder left
    // uncovered would read as a seam of stage grey along one edge. (A scene
    // take's plate may not be decoded for the first frame or two; the server
    // still floor covers the gap, so skip the blit rather than throw on null.)
    if (backdrop) ctx.drawImage(backdrop, -fit.ox, -fit.oy, bw, bh);
    // Each card's shadow is drawn with the card, below in the loop -- see
    // drawCardShadow for why it cannot be one pass up here.
    const shadows = !!(moving && S.bgPlate);
    // readyState < HAVE_CURRENT_DATA means drawImage would paint nothing (or
    // throw on some engines) -- the plate alone is the right thing to show.
    // The bubble still goes on: it rides a SEPARATE media element, so there
    // is no reason to blank it while the screen track buffers. The "master"
    // is ui.video normally, or the ACTIVE fleet's channel 0 for a scene take.
    const vsrc = (S.dragSnap && S.dragSnap.video) || null;   // see snapFrame
    const master = masterVideo();
    if (!vsrc && (!master || master.readyState < 2)) {
      ctx.setTransform(1, 0, 0, 1, fit.ox, fit.oy);
      drawFaceBubble(ctx, p.canvas[0], p.canvas[1], k);
      drawCardOverlay();   // cold fleet: composite is only the backdrop -> handles ride the overlay
      return;
    }
    // Least-emphasized first: a demoted card travelling toward the strip
    // passes BEHIND the growing hero, and painting it after would punch a
    // hole in the subject.
    const order = (focus && !entMorph) ? focus.order
                        : cells.map(function (_c, i) { return i; });
    for (let n = 0; n < order.length; n++) {
      const i = order[n];
      const c = cells[i], s = cardSrcRect(p, i, t);
      const dx = c.x * k, dy = c.y * k, dw = c.w * k, dh = c.h * k;
      // Onto whatever is already composited: the backdrop for a card
      // standing alone, a NEIGHBOUR for the grown subject.
      if (shadows) {
        // `focus` is null on an entrance-morph scene with no clicks (and
        // during a card drag) -- `shadows` rides `entMorph || focus`.
        drawCardShadow(ctx, p, c, k,
                       (focus && focus.lifts && focus.lifts[i]) || 0);
      }
      ctx.save();
      roundRectPath(ctx, dx, dy, dw, dh, c.radius * k);
      ctx.clip();
      const srcEl = cellSource(p, i, vsrc);
      if (srcEl) {
        try {
          ctx.drawImage(srcEl, s[0], s[1], s[2], s[3], dx, dy, dw, dh);
        } catch (e) { /* frame not decodable yet */ }
        // macOS's capture indicator, painted out of the LIVE composite too.
        // The server erases it in the export and in the paused scrub; without
        // this, pressing play would put the badge back, and "preview ==
        // export" would hold everywhere except while the video is moving.
        // Server-supplied box + flat colour (render._badge_patch), in this
        // channel's buffer px, mapped through the same src->dest transform
        // drawImage just used. Inside the card's rounded clip, so a badge on
        // a card cropped past its corner cannot spill onto the backdrop.
        drawBadgePatch(ctx, p, i, s, dx, dy, dw, dh);
      }
      ctx.restore();
      // Selection chrome. The click-selected card (or the one being dragged)
      // wears the full frame -- border + corner + edge handles, the design-tool
      // look from the reference. A merely hovered card gets a faint outline so
      // it reads as selectable without cluttering the stage. Handles are drawn
      // in UNTRANSFORMED backing px (the stage-2 camera is undone first) so
      // their thickness stays a constant screen size at any window-focus zoom.
      const selected = (S.cardSel === i) || (S.cardDrag && S.cardDrag.index === i);
      const hovering = (S.cardHover && S.cardHover.index === i && !S.cardDrag);
      if (selected || hovering) {
        // Draw the frame in UNTRANSFORMED backing px so its stroke and handles
        // keep a constant screen size at any window-focus zoom. Map this card's
        // rect through the same stage-2 camera the cards were drawn under
        // (a layout point c lands at fit.o + z*k*(c - c0)), then re-apply that
        // camera so the next card in the loop draws as before.
        const z = cam ? cam.z : 1;
        const sx = fit.ox + (cam ? z * (dx - cam.x0 * k) : dx);
        const sy = fit.oy + (cam ? z * (dy - cam.y0 * k) : dy);
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        drawCardChrome(ctx, sx, sy, dw * z, dh * z, dpr, selected);
        if (cam) {
          ctx.setTransform(z, 0, 0, z,
                           fit.ox - cam.x0 * z * k, fit.oy - cam.y0 * z * k);
        } else {
          ctx.setTransform(1, 0, 0, 1, fit.ox, fit.oy);
        }
      }
    }
    // Cursor sits on the cards (clipped to them); the bubble goes on top of
    // everything -- the same order render.py composites in.
    drawMultiCursor(ctx, p, k, t, cells);
    ctx.setTransform(1, 0, 0, 1, fit.ox, fit.oy);
    drawFaceBubble(ctx, p.canvas[0], p.canvas[1], k);
    drawCardOverlay();   // live composite carries its own chrome -> overlay clears
  }

  /* --- drag a card to place it by hand ---
   *
   * Note the frame of reference: every other picker in this editor (the zoom
   * pin, the window-crop rect) works in SOURCE px and goes through
   * viewportPxToSourcePx. A card is not in source space -- it is a position
   * in the OUTPUT -- so this maps through the canvas element's own box and
   * stores 0-1 CANVAS fractions, which is also what keeps a hand-placed card
   * where you put it when the export aspect changes. */

  /* Client px -> 0-1 LAYOUT fractions of the composite, or null before the
     canvas has a box. Two mappings in one: the canvas element's own rect,
     then the inverse of stage 2's camera window -- because with window
     focus zoomed in, the layout point under the pointer is NOT the element
     fraction the pointer is at (drawMulti puts layout point c at
     z*(c - x0), so this divides that back out). Undoing it here is what
     lets every consumer below -- hit-test, grab offset, move -- speak plain
     layout fractions without knowing the camera exists. */
  function canvasFrac(clientX, clientY) {
    const box = ui.multi.getBoundingClientRect();
    if (!box.width || !box.height) return null;
    let fx = (clientX - box.left) / box.width;
    let fy = (clientY - box.top) / box.height;
    const p = S.camPath;
    // Element fraction -> backing px -> the composite's own box inside it. The
    // element and the composite are the same rect only while the box has the
    // layout's aspect; skipping this measures against the ELEMENT and every
    // card is off by the margin drawMulti actually drew it at.
    if (p && p.canvas && ui.multi.width && ui.multi.height) {
      const fit = compositeFit(ui.multi.width, ui.multi.height, p.canvas);
      fx = (fx * ui.multi.width - fit.ox) / (p.canvas[0] * fit.k);
      fy = (fy * ui.multi.height - fit.oy) / (p.canvas[1] * fit.k);
    }
    const cam = p && p.canvas
      ? focusCameraWindow(p, activeFocusFrame(p, S.playhead)) : null;
    if (cam) {
      fx = fx / cam.z + cam.x0 / p.canvas[0];
      fy = fy / cam.z + cam.y0 / p.canvas[1];
    }
    return { fx: fx, fy: fy };
  }

  function cardAtPoint(clientX, clientY) {
    const p = S.camPath;
    if (!p || !p.cells) return -1;
    const f = canvasFrac(clientX, clientY);
    if (!f) return -1;
    const fx = f.fx, fy = f.fy;
    const cells = cellsAt(p, cellTime());
    // Topmost first: later cells are painted over earlier ones.
    for (let i = cells.length - 1; i >= 0; i--) {
      const c = cells[i];
      const x = c.x / p.canvas[0], y = c.y / p.canvas[1];
      const w = c.w / p.canvas[0], h = c.h / p.canvas[1];
      if (fx >= x && fx <= x + w && fy >= y && fy <= y + h) return i;
    }
    return -1;
  }

  /* --- resize a card from a corner ---
   *
   * ASPECT-LOCKED, deliberately: every auto-layout places a card at its
   * crop's own aspect and the painter stretches the crop straight over the
   * placement rect (framing.MultiFramePainter -- there is no per-cell
   * letterbox), so a freeform w/h would DISTORT the recording in the
   * export. Scaling about the opposite corner keeps the layout rect on the
   * crop's aspect, which is the same invariant `fit_windows` preserves. */

  // Smallest card, as a fraction of the canvas's shorter dimension of work
  // -- below this a card is an unclickable sliver and the corner zones of
  // its own corners start overlapping.
  const CARD_MIN_FRAC = 0.05;
  // Corner grab zone, CLIENT px (converted per-event into layout fractions,
  // so it stays a constant finger-size under stage fit and window focus).
  const CORNER_GRAB_PX = 12;

  /* {index, corner: "nw"|"ne"|"sw"|"se"} for a pointer within the grab zone
     of a card corner, or null. Topmost card first, corners before body --
     the same z-order cardAtPoint resolves. Only offered where the write
     would land: spatial tools blocked (multi-native / scene takes) or a
     session with no `windows` array has no per-card layout to save. */
  function cardCornerAt(clientX, clientY) {
    if (placementBlocked()) return null;
    const p = S.camPath;
    if (!p || !p.cells) return null;
    // Multi-native / scene cards are ALL placeable (channel_layouts /
    // scene_layouts has a slot per channel, created lazily); display-crop only
    // offers a corner where the windows[i] it would write to actually exists.
    const native = !!(S.details && (S.details.multi_native
                                    || S.details.scene_take));
    const wins = (S.edits && S.edits.windows) || [];
    if (!native && !wins.length) return null;
    const f = canvasFrac(clientX, clientY);
    if (!f) return null;
    // Tolerance in layout fractions: map a point CORNER_GRAB_PX away and
    // take the delta, so the zone tracks the composite scale and the focus
    // camera without duplicating either transform.
    const fx2 = canvasFrac(clientX + CORNER_GRAB_PX, clientY + CORNER_GRAB_PX);
    if (!fx2) return null;
    const tolX = Math.abs(fx2.fx - f.fx), tolY = Math.abs(fx2.fy - f.fy);
    const cells = cellsAt(p, cellTime());
    for (let i = cells.length - 1; i >= 0; i--) {
      if (!native && !wins[i]) continue;
      const c = cells[i];
      const x = c.x / p.canvas[0], y = c.y / p.canvas[1];
      const w = c.w / p.canvas[0], h = c.h / p.canvas[1];
      const corners = [["nw", x, y], ["ne", x + w, y],
                       ["sw", x, y + h], ["se", x + w, y + h]];
      for (let k = 0; k < 4; k++) {
        if (Math.abs(f.fx - corners[k][1]) <= tolX
            && Math.abs(f.fy - corners[k][2]) <= tolY) {
          return { index: i, corner: corners[k][0] };
        }
      }
      // Edge handles, checked AFTER the corners (their zones overlap at the
      // card ends, and the corner is the more precise grab). An edge is
      // grabbable along its span between the two corner zones.
      const nearL = Math.abs(f.fx - x) <= tolX, nearR = Math.abs(f.fx - (x + w)) <= tolX;
      const nearT = Math.abs(f.fy - y) <= tolY, nearB = Math.abs(f.fy - (y + h)) <= tolY;
      const spanX = f.fx > x + tolX && f.fx < x + w - tolX;
      const spanY = f.fy > y + tolY && f.fy < y + h - tolY;
      if (spanX && nearT) return { index: i, corner: "n" };
      if (spanX && nearB) return { index: i, corner: "s" };
      if (spanY && nearL) return { index: i, corner: "w" };
      if (spanY && nearR) return { index: i, corner: "e" };
    }
    return null;
  }

  /* Click-selection of a card. The selected card wears the full frame; a
     move or resize arms with it selected. Selecting drops any server still so
     the live composite (which carries the frame) shows -- requestStill then
     declines to put one back over it until the card is deselected. */
  function selectCard(i) {
    if (S.cardSel === i) return;
    S.cardSel = i;
    staleStill();          // reveal the live composite + its selection frame
    drawCardOverlay();     // ...and the still-independent handle overlay
  }
  function deselectCard() {
    if (S.cardSel == null) return;
    S.cardSel = null;
    if (multiReady() || sceneLive()) drawMulti();
    drawCardOverlay();     // clears the overlay now that nothing is selected
    scheduleStill();       // bring the crisp server still back (windows/scene)
  }

  function onCardDown(ev) {
    // The crop picker drags rects on the RAW source; the composite is down
    // while it's armed, so the two drag tools can never both be live.
    if ((!multiReady() && !sceneLive()) || S.winPick || S.cropArm) return;
    // No per-card store to save into (scene take P2 / plain take) -> no drag.
    if (placementBlocked()) return;
    // A corner grab wins over a body grab -- the zones overlap wherever a
    // corner sits inside the card. Both also select it; empty stage clears.
    const grab = cardCornerAt(ev.clientX, ev.clientY);
    if (grab) { selectCard(grab.index); armCardResize(ev, grab); return; }
    const idx = cardAtPoint(ev.clientX, ev.clientY);
    if (idx < 0) { deselectCard(); return; }
    selectCard(idx);
    // Hit-tested against what is on SCREEN (so you grab the card you see,
    // grown or travelling), but the drag itself is expressed against the
    // BASE cell -- that is what moves, what gets drawn once the drag starts,
    // and what `layout` is saved from. Identical rects whenever focus isn't
    // mid-animation, which is every session with window focus off.
    const p = S.camPath, c = p.cells[idx];
    // Armed BEFORE the grab offset is measured, deliberately: arming is what
    // drops the composite to the base, un-zoomed layout, and a grab measured
    // in the emphasized frame of reference would make the card leap on the
    // first move. One frame of reference for the whole gesture.
    S.cardDrag = {
      index: idx,
      mode: "move",
      grabX: 0, grabY: 0,
      w: c.w / p.canvas[0],
      h: c.h / p.canvas[1],
      moved: false,
    };
    const f = canvasFrac(ev.clientX, ev.clientY);
    if (!f) { S.cardDrag = null; return; }
    ev.preventDefault();
    // One video upload for the whole gesture instead of three per pointer
    // event -- the difference between ~38 ms and ~3 ms a redraw.
    S.dragSnap = { video: snapFrame(ui.video), face: snapFrame(ui.face) };
    // Grab offset in LAYOUT fractions, so the card doesn't jump to centre
    // itself under the pointer on the first move.
    S.cardDrag.grabX = f.fx - c.x / p.canvas[0];
    S.cardDrag.grabY = f.fy - c.y / p.canvas[1];
    // The rendered still is opaque and sits ON TOP of the composite: leave it
    // up and the card looks frozen while the pointer drags the live canvas
    // underneath it. requestStill declines to put one back until the drop.
    staleStill();
    drawMulti();
    ui.multi.setPointerCapture && ui.multi.setPointerCapture(ev.pointerId);
  }

  /* Arm a corner-resize gesture: same snapshot/still/pointer-capture ritual
     as the move arm, but the drag state carries the ANCHOR (the opposite
     corner, which stays put) and the card's aspect in layout fractions. */
  function armCardResize(ev, grab) {
    const p = S.camPath, c = p.cells[grab.index];
    const x = c.x / p.canvas[0], y = c.y / p.canvas[1];
    const w = c.w / p.canvas[0], h = c.h / p.canvas[1];
    S.cardDrag = {
      index: grab.index,
      mode: "resize",
      corner: grab.corner,
      // A one-letter corner ("n"/"e"/"s"/"w") is an EDGE handle: same
      // aspect-locked scale, but it pins the opposite EDGE and keeps the
      // perpendicular axis centred, so it needs the full starting frame below.
      edge: grab.corner.length === 1,
      // The opposite corner anchors a corner scale.
      ax: grab.corner === "nw" || grab.corner === "sw" ? x + w : x,
      ay: grab.corner === "nw" || grab.corner === "ne" ? y + h : y,
      // Original frame + centre, for an edge scale's pinned edge / centred axis.
      ox: x, oy: y, ow: w, oh: h, mx: x + w / 2, my: y + h / 2,
      // Aspect in FRACTION space (canvas w/h cancel differently per axis).
      arf: h > 0 ? w / h : 1,
      w: w, h: h,
      moved: false,
    };
    ev.preventDefault();
    S.cardHover = null;    // the selection frame takes over from hover
    S.dragSnap = { video: snapFrame(ui.video), face: snapFrame(ui.face) };
    staleStill();
    drawMulti();
    ui.multi.setPointerCapture && ui.multi.setPointerCapture(ev.pointerId);
  }

  function onCardMove(ev) {
    const d = S.cardDrag;
    if (!d) {
      // No gesture: hover feedback. Redraw only on a CHANGE -- a drawMulti per
      // pointermove would be a full composite per event.
      if ((!multiReady() && !sceneLive()) || S.winPick || S.cropArm) return;
      const over = cardCornerAt(ev.clientX, ev.clientY);   // a handle, or null
      const idx = over ? over.index : cardAtPoint(ev.clientX, ev.clientY);
      // A handle keys by index+corner; a bare body hover keys by index alone,
      // so moving between the two on one card still counts as a change.
      const key = over ? over.index + over.corner : (idx >= 0 ? "b" + idx : null);
      const ph = S.cardHover;
      const prev = ph ? (ph.corner ? ph.index + ph.corner : "b" + ph.index) : null;
      ui.multi.style.cursor = over ? cardCursor(over.corner)
                                   : (idx >= 0 ? "move" : "");
      if (key !== prev) {
        S.cardHover = over || (idx >= 0 ? { index: idx, corner: null } : null);
        drawMulti();
      }
      return;
    }
    const p = S.camPath;
    const f = canvasFrac(ev.clientX, ev.clientY);
    if (!f) return;
    const c = p.cells[d.index];
    if (d.mode === "resize") {
      if (!d.edge) {
        // Aspect-locked scale about the anchored opposite corner: the larger
        // of the two axis-implied widths wins, so the card chases whichever
        // direction the pointer is really pulling.
        const fx = Math.max(0, Math.min(1, f.fx));
        const fy = Math.max(0, Math.min(1, f.fy));
        let w = Math.max(Math.abs(fx - d.ax), Math.abs(fy - d.ay) * d.arf);
        // Floors: never below the minimum on either axis...
        w = Math.max(w, CARD_MIN_FRAC, CARD_MIN_FRAC * d.arf);
        // ...and never past the canvas edge in the drag direction (the anchor
        // is pinned, so available room is what's on the pointer's side).
        const availX = d.corner === "ne" || d.corner === "se"
          ? 1 - d.ax : d.ax;
        const availY = d.corner === "sw" || d.corner === "se"
          ? 1 - d.ay : d.ay;
        w = Math.min(w, availX, availY * d.arf);
        const h = w / d.arf;
        d.w = w;
        d.h = h;
        d.moved = true;
        c.w = w * p.canvas[0];
        c.h = h * p.canvas[1];
        c.x = (d.corner === "nw" || d.corner === "sw" ? d.ax - w : d.ax)
          * p.canvas[0];
        c.y = (d.corner === "nw" || d.corner === "ne" ? d.ay - h : d.ay)
          * p.canvas[1];
        drawMulti();
        return;
      }
      // Edge handle: same aspect lock, but ONE axis is driven by the pointer
      // and the perpendicular one follows through the aspect, growing about
      // the card's centre so it never drifts sideways. The opposite edge is
      // pinned. availPerp caps the perpendicular growth so the card stays on
      // the canvas; availDrive caps the driven axis against the pinned edge.
      const fx = Math.max(0, Math.min(1, f.fx));
      const fy = Math.max(0, Math.min(1, f.fy));
      let w, h;
      if (d.corner === "e" || d.corner === "w") {
        const availDrive = d.corner === "e" ? 1 - d.ox : d.ox + d.ow;
        const availPerp = 2 * Math.min(d.my, 1 - d.my);   // height about my
        w = d.corner === "e" ? fx - d.ox : d.ox + d.ow - fx;
        w = Math.max(w, CARD_MIN_FRAC, CARD_MIN_FRAC * d.arf);
        w = Math.min(w, availDrive, availPerp * d.arf);
        h = w / d.arf;
      } else {
        const availDrive = d.corner === "s" ? 1 - d.oy : d.oy + d.oh;
        const availPerp = 2 * Math.min(d.mx, 1 - d.mx);   // width about mx
        h = d.corner === "s" ? fy - d.oy : d.oy + d.oh - fy;
        h = Math.max(h, CARD_MIN_FRAC, CARD_MIN_FRAC / d.arf);
        h = Math.min(h, availDrive, availPerp / d.arf);
        w = h * d.arf;
      }
      d.w = w;
      d.h = h;
      d.moved = true;
      c.w = w * p.canvas[0];
      c.h = h * p.canvas[1];
      c.x = (d.corner === "e" ? d.ox
             : d.corner === "w" ? d.ox + d.ow - w
             : d.mx - w / 2) * p.canvas[0];
      c.y = (d.corner === "s" ? d.oy
             : d.corner === "n" ? d.oy + d.oh - h
             : d.my - h / 2) * p.canvas[1];
      drawMulti();
      return;
    }
    let x = f.fx - d.grabX;
    let y = f.fy - d.grabY;
    x = Math.max(0, Math.min(1 - d.w, x));
    y = Math.max(0, Math.min(1 - d.h, y));
    d.moved = true;
    // Move the local copy so the preview tracks the pointer; the save (and
    // the authoritative re-layout) happens once, on release.
    c.x = x * p.canvas[0];
    c.y = y * p.canvas[1];
    drawMulti();
  }

  function onCardUp(ev) {
    const d = S.cardDrag;
    S.cardDrag = null;
    S.dragSnap = null;
    // Camera-path/save responses that were held back rather than applied
    // over a moving card (they replace p.cells wholesale) get their chance
    // now that the gesture is over.
    if (d && S.camPathStale) { S.camPathStale = false; scheduleCamPath(); }
    // A click that never moved still took the still down to arm the drag, so
    // it has to put one back; a real drag gets one from mutate's staleStill.
    if (!d || !d.moved) { drawMulti(); if (d) scheduleStill(); return; }
    const p = S.camPath, c = p.cells[d.index];
    const layout = {
      x: c.x / p.canvas[0], y: c.y / p.canvas[1],
      w: d.w, h: d.h,
    };
    mutate(function (e) {
      if (S.details && S.details.multi_native) {
        // Positional per-channel store; pad with nulls so an override on card
        // i never index-shifts a sibling (the null-preserving contract the
        // server normalizer also holds -- docs/architecture.md).
        if (!Array.isArray(e.channel_layouts)) e.channel_layouts = [];
        while (e.channel_layouts.length <= d.index) e.channel_layouts.push(null);
        e.channel_layouts[d.index] = layout;
      } else if (S.details && S.details.scene_take) {
        // Per-scene store, keyed by the ACTIVE scene (correct during the paused
        // drag -- S.activeScene tracks the playhead). Positional + null-padded
        // within the scene, exactly like the multi-native store.
        const s = String(S.activeScene);
        if (!e.scene_layouts || typeof e.scene_layouts !== "object") {
          e.scene_layouts = {};
        }
        if (!Array.isArray(e.scene_layouts[s])) e.scene_layouts[s] = [];
        while (e.scene_layouts[s].length <= d.index) e.scene_layouts[s].push(null);
        e.scene_layouts[s][d.index] = layout;
      } else {
        const win = e.windows[d.index];
        if (win) win.layout = layout;
      }
    });
  }

  /* A placement this close to where it already is counts as "nowhere".
     Both sides of the comparison have been rounded to whole canvas pixels
     twice (framing rounds each placement, then `_apply_card_overrides`
     rounds the fraction this writes back), so a sub-pixel disagreement is
     that rounding, not slack worth taking out. Measured against both real
     multi-window sessions, every arrangement: re-fitting one the server
     just laid out moves an edge by 0.5px (three cards) or 1.0px (two), and
     the scale it computes is 1.00000. A hand-dragged arrangement measured
     alongside them moved edges by ~190px and grew each card by ~390px.

     Read the first number as "on those two sessions", not as a law: `grid`
     is not a fixed point of this fit anywhere (it aspect-fits each crop
     inside its own cell, so the bounding box can sit off-centre) and on
     other canvases it moves far more than the epsilon -- 78px measured for
     two windows on 1080x1920, still at scale 1.00000. That case is meant to
     pass this test and enable the button; the tooltip is what has to stay
     honest about it being a re-centre rather than a growth. */
  const FIT_EPS_PX = 1;

  /* The write "Fit to frame" would perform: `{cards: [...]}` when it would do
     something, `{reason}` when it would not. ONE function, because the
     button's enabled state and the write have to agree about whether acting
     is safe -- a second copy of the test is exactly how a disabled-looking
     button and a garbage write coexist.

     The camera path is the only place the current placement exists (the
     auto-layouts run on the server; a hand-dragged card is a `layout` in
     edits.json), and its `cells` bind to the windows POSITIONALLY with
     nothing in the payload to check that against. So the freshness stamp IS
     the binding: same cards, same order, same arrangement, or this refuses
     for the one round trip it takes to catch up. It used to pair them under
     a `Math.min(cells.length, wins.length)`, which made a mismatch proceed
     silently on data that does not correspond -- remove or reorder a window
     and every remaining card was written the previous arrangement's cell.

     `pad` comes off the WIDTH on both axes, TRUNCATED, because that is what
     framing.MultiFramePainter and framing.fit_placements both spend
     (`int(framing.PAD_FRAC * W)`); that product is not a whole number of
     pixels on a real canvas, and keeping the remainder here put the editor's
     fit and the export's arrangements in slightly different margins.
     The fraction must equal autocine/framing.py's PAD_FRAC --
     test_web_sources.py reads this literal back out and pins the two. */
  function fitPlan() {
    const p = S.camPath;
    const wins = (S.edits && S.edits.windows) || [];
    if (!p || !p.windows_mode || !p.cells || !p.canvas || !wins.length) {
      return { reason: "" };
    }
    if (p.cells_sig !== cellsSignature(S.edits) || p.cells.length !== wins.length) {
      return { reason: "Catching up with that change..." };
    }
    // A card under the pointer is mid-move: p.cells holds a position that is
    // still changing and is not in edits yet, so a fit computed off it would
    // bake in wherever the drag happened to be. The pointer cannot press this
    // button, but a focused one answers to Enter and Space.
    if (S.cardDrag) return { reason: "Finish placing that card first" };
    const W = p.canvas[0], H = p.canvas[1];
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    p.cells.forEach(function (c) {
      x0 = Math.min(x0, c.x); y0 = Math.min(y0, c.y);
      x1 = Math.max(x1, c.x + c.w); y1 = Math.max(y1, c.y + c.h);
    });
    const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0);
    const pad = Math.trunc(0.03 * W);
    const scale = Math.min((W - 2 * pad) / bw, (H - 2 * pad) / bh);
    const offX = (W - bw * scale) / 2, offY = (H - bh * scale) / 2;
    let moves = false;
    const cards = p.cells.map(function (c, i) {
      const x = offX + (c.x - x0) * scale, y = offY + (c.y - y0) * scale;
      const w = c.w * scale, h = c.h * scale;
      if (Math.abs(x - c.x) > FIT_EPS_PX || Math.abs(y - c.y) > FIT_EPS_PX
          || Math.abs(w - c.w) > FIT_EPS_PX || Math.abs(h - c.h) > FIT_EPS_PX) {
        moves = true;
      }
      return { id: wins[i].id, layout: { x: x / W, y: y / H, w: w / W, h: h / H } };
    });
    // The four scaled arrangements already sit at exactly this scale, so a
    // fit straight after applying one is a no-op. Saying so is the whole
    // difference between a control that does nothing and one that lies.
    if (!moves) return { reason: "Already as large as one scale can make it." };
    return { cards: cards };
  }

  /* Rescale whatever is on screen -- auto-layout, hand-dragged, or both -- as
     a `layout` override on every card.

     Deliberately an explicit action and not something a drop triggers: the
     whole point of dragging is that the card stays where it was dropped. The
     server re-applies these verbatim through `_apply_card_overrides`, so
     this decides data, not layout. Written by card id rather than by index:
     the plan was measured against `S.edits` as it is right now, and naming
     the card it belongs to costs nothing and cannot land on the wrong one. */
  function fitCardsToFrame() {
    const plan = fitPlan();
    if (!plan.cards) return;
    mutate(function (e) {
      plan.cards.forEach(function (card) {
        const win = e.windows.find(function (w) { return w.id === card.id; });
        if (win) win.layout = card.layout;
      });
    });
  }

  /* Repaint the fit control from the current plan -- enabled state, tooltip
     and the one line under it that says why, when it is off. Called both as
     the panel is built and whenever a camera path lands: one attribute and
     one string rather than renderPanel(), which would rebuild the panel
     under the pointer every time a path came back. */
  function syncFitAction() {
    if (S.panel !== "windows") return;
    const fit = ui.panel.querySelector(".ed-fit");
    if (!fit) return;
    const plan = fitPlan();
    fit.disabled = !plan.cards;
    // "Scale up until it runs out of room" was a promise the click does not
    // always keep: a Grid is already at the scale this computes, and fitting
    // it only slides the cards back to centre (measured, two windows on a
    // 1080x1920 canvas: 78px of re-centring, scale exactly 1.0). The plan
    // only knows that SOMETHING moves, not which of the two it is, so the
    // title has to be true of both.
    fit.title = plan.cards
      ? "Scale this arrangement out to the frame margin, or re-centre it if it is already that large"
      : (plan.reason || "Nothing on the canvas to scale up");
    const note = ui.panel.querySelector(".ed-fit-note");
    if (note) {
      note.textContent = plan.reason || "";
      note.hidden = !plan.reason;
    }
  }

  function resetCardPlacement() {
    // Same reason fitPlan refuses: clearing the override the drag is about to
    // write leaves the card drawn where the pointer left it and the edits
    // saying otherwise, and the drop would then quietly reinstate it.
    if (S.cardDrag) return;
    mutate(function (e) {
      if (S.details && S.details.multi_native) {
        e.channel_layouts = [];           // back to the preset arrangement
      } else if (S.details && S.details.scene_take) {
        if (e.scene_layouts) e.scene_layouts[String(S.activeScene)] = [];
      } else {
        e.windows.forEach(function (w) { delete w.layout; });
      }
    });
  }

  /* The session ANCHOR file (scene 0, channel 0) -- never removable: it carries
     the clock (and, on a mic take, the audio). The render refuses it too. */
  function anchorFile() {
    const s0 = S.scenePlan && S.scenePlan[0];
    const c0 = s0 && s0.channels && s0.channels[0];
    return (c0 && c0.file) || null;
  }

  /* Remove the SELECTED window from the video -- a REVERSIBLE hide. The
     recording is untouched; edits.hidden_channels drops that channel's card
     from every scene at render (keyed by FILE, stable across scenes). The
     camera path re-fetches (cellsSignature carries hidden_channels), so the
     preview loses the card immediately, exactly as the export will -- and if
     that was the only added card at a join seam, the seam simply disappears.
     A hide restructures the card set, so it also clears manual card placements
     (positional, authored for the old arrangement). Scene takes only in v1. */
  function removeSelectedCard() {
    if (S.cardDrag || S.cardSel == null || !sceneLive()) return;
    const ch = (S.camPath && S.camPath.channels || [])[S.cardSel];
    const file = ch && ch.file;
    if (!file) return;
    // The clock anchor (scene 0, channel 0) never leaves -- the render refuses
    // it too. The panel disables Remove while it is the selection.
    if (file === anchorFile()) return;
    deselectCard();
    mutate(function (e) {
      const hid = Array.isArray(e.hidden_channels)
        ? e.hidden_channels.slice() : [];
      if (hid.indexOf(file) < 0) hid.push(file);
      e.hidden_channels = hid;
      e.channel_layouts = [];   // stale positional placements
      e.scene_layouts = {};
    });
  }

  function restoreHiddenCards() {
    if (S.cardDrag) return;
    mutate(function (e) { e.hidden_channels = []; });
  }

  function applyTransformAt(t) {
    const d = S.details;
    if (!d) return;
    const vpW = ui.viewport.clientWidth, vpH = ui.viewport.clientHeight;
    if (vpW <= 0 || vpH <= 0) return;
    // Scene take (live player): draw the active scene's composite, never the
    // srcless ui.video. Spatial tools are blocked here, so the pin/crop/pick
    // branch below is unreachable and this can sit first. While paused, resolve
    // the scene under `t`, seek its fleet to the exact frame, and draw; the
    // server still floor (scheduled by seek) covers a cold fleet until its
    // channels are frame-ready (onFleetProgress then swaps to live). While
    // playing, sceneTick owns the fleet + clock and just asks us to repaint.
    if (sceneLive()) {
      showMulti(true);
      ui.zoomReadout.textContent = "";
      ui.pinDot.hidden = true;
      if (!S.playing) {
        const r = resolveScene(t);
        activateScene(r.s);
        const fl = getFleet(r.s);
        const tLocal = r.kLocal / S.sceneFps;
        seekFleetTo(fl, r.s, tLocal);
        if (fleetReady(fl)) { S.stillShown = false; ui.still.classList.remove("show"); }
        drawMulti(tLocal);
      } else {
        drawMulti(t - S.seamTimes[S.activeScene]);
      }
      return;
    }
    // The <video> element always plays the WHOLE recording, so it is sized in
    // RAW file px while every coordinate below is in window space (identical
    // for an ordinary session, where crop is null and raw == source). The
    // element is clipped to the window and shifted by the crop origin;
    // #ed-video has transform-origin 0 0, so a post-scale translate of
    // -crop*scale is exactly the right correction.
    const crop = stageCrop();
    const rawW = d.raw_width || d.width, rawH = d.raw_height || d.height;
    ui.video.style.width = rawW + "px";
    ui.video.style.height = rawH + "px";
    // Without the clip, the rest of the screen bleeds into the letterbox
    // around a contain-fit window. inset() is in the element's own,
    // untransformed px — i.e. literally the crop rect.
    ui.video.style.clipPath = crop
      ? "inset(" + crop[1] + "px " + (rawW - crop[0] - crop[2]) + "px "
        + (rawH - crop[1] - crop[3]) + "px " + crop[0] + "px)"
      : "";
    // The shift is `+ 0` — a literal, not `0 * s` — with no crop, so the
    // strings below stay byte-identical to the pre-feature ones and a
    // degenerate scale can never turn a finite translate into NaN.
    if (S.pinArm || S.winPick || S.cropArm) {
      // Identity/contain-fit: the raw, unzoomed source at true scale. The
      // zoom-pin tool needs this to place a target in source pixels; the
      // window-crop tool needs it for the same reason (dragging a rect
      // directly on the real recording, not a composited/zoomed view) --
      // which is also why the composite has to come down while either is
      // armed, even in windows mode.
      showMulti(false);
      const s = Math.min(vpW / d.width, vpH / d.height);
      const tx = (vpW - d.width * s) / 2, ty = (vpH - d.height * s) / 2;
      ui.video.style.transform = "translate(" + (tx + (crop ? -crop[0] * s : 0)) + "px," +
        (ty + (crop ? -crop[1] * s : 0)) + "px) scale(" + s + ")";
      ui.zoomReadout.textContent = "";
      return;
    }
    if (multiReady()) {
      showMulti(true);
      syncFace(t);
      syncChannels(t);
      drawMulti(t);
      ui.zoomReadout.textContent = "";
      ui.pinDot.hidden = true;
      return;
    }
    // Outside the composite nothing samples face.mov, so don't leave it
    // running in the background burning decode on a bubble no one can see.
    if (ui.face && !ui.face.paused) { try { ui.face.pause(); } catch (e) {} }
    showMulti(false);
    const cam = sampleCam(t);
    const win = (S.camPath && S.camPath.win) || [d.width, d.height];
    const s = (vpW * cam.z) / win[0];
    // tx/ty stay in WINDOW space — updatePinDot maps pins with them; only the
    // element's own transform carries the crop shift.
    const tx = vpW / 2 - cam.cx * s;
    const ty = vpH / 2 - cam.cy * s;
    ui.video.style.transform = "translate(" + (tx + (crop ? -crop[0] * s : 0)).toFixed(2) + "px," +
      (ty + (crop ? -crop[1] * s : 0)).toFixed(2) + "px) scale(" + s.toFixed(5) + ")";
    ui.zoomReadout.textContent = cam.z > 1.004 ? cam.z.toFixed(2) + "×" : "";
    updatePinDot(cam, s, tx, ty);
  }

  function updatePinDot(cam, s, tx, ty) {
    const sel = S.sel && S.sel.kind === "zoom" ? findItem("zoom", S.sel.id) : null;
    const show = sel && sel.x !== null && sel.x !== undefined && !S.stillShown;
    if (!show) { ui.pinDot.hidden = true; return; }
    const vpRect = ui.viewport.getBoundingClientRect();
    const cRect = ui.canvas.getBoundingClientRect();
    const px = sel.x * s + tx + (vpRect.left - cRect.left);
    const py = sel.y * s + ty + (vpRect.top - cRect.top);
    if (px < -10 || py < -10 || px > cRect.width + 10 || py > cRect.height + 10) {
      ui.pinDot.hidden = true;
      return;
    }
    ui.pinDot.style.left = px + "px";
    ui.pinDot.style.top = py + "px";
    ui.pinDot.hidden = false;
  }

  function updateStageNote() {
    if (S.camPath && S.camPath.canvas) {
      ui.stageNote.textContent = S.camPath.canvas[0] + " × " + S.camPath.canvas[1];
    } else if (S.details) {
      ui.stageNote.textContent = S.details.width + " × " + S.details.height;
    }
  }

  /* ==================== record-time window capture ====================
   * avfoundation cannot target a window, so recording one captures the whole
   * display and meta.json carries the window's rect; the crop happens at
   * RENDER time. What that means here:
   *   - d.width/d.height are already the WINDOW's size, and they are THE
   *     source coordinate space — so viewportPxToSourcePx, the zoom-pin
   *     clamp, the window-rect clamp and sampleCam's fallback are correct
   *     as-is, with no crop arithmetic of their own;
   *   - the paused still is server-rendered, so it is already cropped;
   *   - only the raw <video> element needs help (see applyTransformAt), and
   *     only it knows about crop origins.
   */

  /* The crop rect in RAW file px, [x, y, w, h], or null. The camera path is
     the authority: the server derives it from meta.json at the real per-axis
     file scale, and reports null both for an ordinary full-screen session and
     when it REJECTED the recorded rect (malformed, non-point units, a window
     that was on another display) — in which case the export is full-frame and
     so is this preview. Null too until the first path lands. */
  function captureCrop() {
    const c = S.camPath && S.camPath.crop;
    return c && c.length === 4 ? c : null;
  }

  /* What the <video> element is clipped to on the stage: the capture crop and
     the editor's crop composed into one raw-file rect (camera_path's
     `stage_crop`). While the crop tool is armed we deliberately drop back to
     the capture crop alone -- you cannot drag a new rect on a frame that has
     already had the old one cut out of it. Falls back to the capture crop on
     an older server that predates the field. */
  function stageCrop() {
    if (S.cropArm) return captureCrop();
    const c = S.camPath && S.camPath.stage_crop;
    return c && c.length === 4 ? c : captureCrop();
  }

  /* Did a crop actually take effect? `capture_window` being non-null does NOT
     answer that -- the block survives a rejected rect (non-point units, a rect
     on a secondary display) and such a session renders full-frame.

     describe_session states the fact as `capture_window.applied`; prefer it.
     The two fallbacks are for an older server that predates that field:
     camPath.crop once a path has loaded, then a raw-vs-window dim compare
     (which is wrong for a window exactly covering the display, but that crop
     is a no-op anyway and the camera path corrects it on arrival). */
  function captureCropApplied() {
    const d = S.details;
    const cw = d && d.capture_window;
    if (cw && cw.applied !== undefined) return !!cw.applied;
    if (S.camPath && S.camPath.crop !== undefined) return !!captureCrop();
    return !!(d && d.raw_width &&
      (d.raw_width !== d.width || d.raw_height !== d.height));
  }

  function captureWindowLabel(cw) {
    const app = String(cw.app || "").trim();
    const title = String(cw.title || "").trim();
    if (app && title && title !== app) return app + " / " + title;
    return app || title || "Window";
  }

  let winChip = null, winChipText = null;
  let winBanner = null, winBannerDismissed = false;

  function updateCaptureUI() {
    const d = S.details;
    const cw = (d && d.capture_window) || null;
    const on = !!cw && captureCropApplied();

    if (on && !winChip) {
      winChip = el("span", "ed-window-chip");
      winChip.appendChild(iconEl("frame", 12));
      winChip.appendChild(el("span", "ed-window-chip-key", "Window"));
      winChipText = el("span", "ed-window-chip-text");
      winChip.appendChild(winChipText);
      ui.stageNote.insertAdjacentElement("afterend", winChip);
    }
    if (winChip) {
      winChip.hidden = !on;
      if (on) {
        const label = captureWindowLabel(cw);
        winChipText.textContent = label;
        winChip.title = "Recorded as a window capture. Every frame is cropped to \""
          + label + "\" (" + d.width + " x " + d.height + " of the "
          + (d.raw_width || d.width) + " x " + (d.raw_height || d.height)
          + " screen recording)."
          + (cw.tracked
             ? " Its geometry was tracked, so the crop follows the window"
               + " wherever it was moved or resized."
             : "");
      }
    }

    // A tracked session followed the movement, so there is nothing to warn
    // about — the banner is only for a fixed crop that got left behind.
    const warn = on && !!cw.moved && !cw.tracked && !winBannerDismissed;
    if (warn && !winBanner) buildWindowBanner();
    if (winBanner) winBanner.hidden = !warn;
  }

  /* The payoff for snapshotting the window rect again at stop: without a
     geometry track the crop is fixed to where the window STARTED, so a window
     that was dragged or resized mid-take leaves part of the action outside the
     frame. Advisory, never blocking — so it's dismissable. */
  function buildWindowBanner() {
    winBanner = el("div", "ed-stage-banner");
    const body = el("div", "ed-stage-banner-body");
    body.appendChild(el("strong", null, "This window moved during recording"));
    body.appendChild(el("span", "t-label",
      "The crop is fixed to where it started, so part of the take may be off-frame."));
    winBanner.appendChild(body);
    const x = el("button", "btn-icon btn-quiet ed-stage-banner-x");
    x.innerHTML = ICONS.close;
    x.title = "Dismiss";
    x.addEventListener("click", function () {
      winBannerDismissed = true;
      winBanner.hidden = true;
    });
    winBanner.appendChild(x);
    ui.canvasWrap.parentNode.insertBefore(winBanner, ui.canvasWrap);
  }

  /* ============================ rendered still ============================ */

  const scheduleStill = debounce(requestStill, 200);

  let stillHideTimer = null;

  /* Mark the rendered still stale and ask for a fresh one.

     `delay` is a grace period, and it is the whole point of the two-argument
     form. Blanking the still the instant an edit lands is right when the
     caller is about to reveal something else underneath it (the live
     composite, a selection frame) -- that is every no-argument call site, and
     they still hide immediately. It is WRONG for a look change: on a scene
     take there is no playable video, so the server-rendered still IS the
     picture, and dropping it here while the replacement is still rendering
     reads as "the background stopped working and the windows disappeared".
     Holding the stale frame for `delay` ms shows the old colour a beat longer
     instead, which is strictly better than showing nothing. If the
     replacement is slower than that we blank as before; if it lands first it
     sets `S.stillShown` and this no-ops. */
  function staleStill(delay) {
    S.stillShown = false;
    ui.pinDot.hidden = true;
    if (stillHideTimer) { clearTimeout(stillHideTimer); stillHideTimer = null; }
    if (delay > 0) {
      stillHideTimer = setTimeout(function () {
        stillHideTimer = null;
        if (!S.stillShown) ui.still.classList.remove("show");
      }, delay);
    } else {
      ui.still.classList.remove("show");
    }
    if (!S.playing && !S.pinArm) {
      scheduleStill.cancel();
      setTimeout(scheduleStill, 0);
    }
  }

  function sceneFleetReady() {
    if (!sceneLive()) return false;
    return fleetReady(S.fleets && S.fleets.get(S.activeScene));
  }

  /* A selected card suppresses the server still so its selection frame is
     visible on the live composite. But a SCENE take's still is ALSO the
     cold-fleet correctness floor -- so don't suppress it while the active
     fleet is not frame-ready (drawMulti would paint only the backdrop then).
     onFleetProgress swaps in the live composite -- with the frame, cardSel is
     still set -- the moment the fleet is ready. Multi-native never has a still;
     display-crop's master is always ready, so both simply hold. */
  function selHoldsStill() {
    return S.cardSel != null && (!sceneLive() || sceneFleetReady());
  }

  async function requestStill() {
    // S.cardDrag / the selection-hold join playing/pinArm here for the same
    // reason they are here: the still is opaque and on top, so putting one up
    // mid-drag -- or over a selected card's frame -- hides what is being edited.
    if (S.playing || S.pinArm || S.cardDrag || selHoldsStill() || !S.edits) return;
    // Multi-native (fixed window set) shows a LIVE client-side canvas
    // composite from its channel videos, so it never needs a server still and
    // `preview_frame` has no branch for it -- keep it out. Scene takes are the
    // opposite: no playable video, so the server-composited still IS the
    // preview (`scene_preview_frame`), and this is the path that fetches it.
    if (S.details && S.details.multi_native) return;
    const token = ++S.previewToken;
    const t = S.playhead;
    try {
      const data = await api("/api/preview", {
        body: { session: S.session, time: t, options: fullOptions() },
      });
      if (token !== S.previewToken || S.playing || S.pinArm || S.cardDrag
          || selHoldsStill()) return;
      if (Math.abs(t - S.playhead) > 0.001) return;   // playhead moved on
      ui.still.src = "data:image/jpeg;base64," + data.image_jpeg_base64;
      ui.still.classList.add("show");
      S.stillShown = true;
      ui.pinDot.hidden = true;
      drawCardOverlay();   // keep the handle overlay on top of the fresh still
    } catch (e) {
      /* leave live view up */
    }
  }

  /* ============================ playback ============================ */

  function setPlayIcon() {
    ui.play.innerHTML = S.playing ? ICONS.pause : ICONS.play;
  }

  function seek(t, opts) {
    opts = opts || {};
    const tr = trimRange();
    S.playhead = clamp(t, tr.start, tr.end);
    // Scene take: no shared ui.video to drive -- applyTransformAt resolves the
    // scene, seeks its fleet to the exact frame, and draws the live composite;
    // the still floor (scheduled below) covers a cold fleet until it is ready.
    if (sceneLive()) {
      updatePlayheadUI();
      if (!S.playing) {
        S.stillShown = false;
        ui.still.classList.remove("show");
        applyTransformAt(S.playhead);
        if (!opts.noStill) scheduleStill();
      }
      return;
    }
    const wantMedia = fleetLead() + S.playhead;
    if (Math.abs(ui.video.currentTime - wantMedia) > 0.004) {
      try { ui.video.currentTime = wantMedia; } catch (e) { /* not ready yet */ }
    }
    updatePlayheadUI();
    if (!S.playing) {
      S.stillShown = false;
      ui.still.classList.remove("show");
      applyTransformAt(S.playhead);
      if (!opts.noStill) scheduleStill();
    }
  }

  function togglePlay() {
    if (S.playing) pause();
    else play();
  }

  function play() {
    if (!S.details) return;
    // Scene take with a built plan: the per-scene fleet-ring player (the active
    // scene's channel 0 is the played master clock; sceneTick crosses seams).
    if (sceneLive()) { playScene(); return; }
    if (S.sceneStillOnly) {
      // A scene take whose live plan has not loaded (or failed): the server
      // still IS the preview. Scrubbing previews every frame; say so once
      // rather than flickering the icon against an empty element.
      if (!S.scenePlayHinted) {
        S.scenePlayHinted = true;
        toast("Scrub the timeline to preview. Smooth playback for "
          + "multi-window scene takes is coming; use Export for video.");
      }
      return;
    }
    const tr = trimRange();
    if (S.playhead >= tr.end - 0.02) seek(tr.start, { noStill: true });
    disarmPin();
    closeMarkerPop();
    S.playing = true;
    S.stillShown = false;
    ui.still.classList.remove("show");
    ui.pinDot.hidden = true;
    ui.video.muted = false;
    const p = ui.video.play();
    if (p && p.catch) p.catch(function () { S.playing = false; setPlayIcon(); });
    setPlayIcon();
    requestAnimationFrame(playbackTick);
  }

  function pause() {
    S.playing = false;
    // Scene take: pause the active fleet's master and recompute the playhead
    // from its clock (never ui.video, which is srcless here -- reading its
    // currentTime would snap the playhead to 0).
    if (sceneLive()) {
      const fl = S.fleets.get(S.activeScene);
      const master = fl && fl.videos[0];
      if (master) { try { master.pause(); } catch (e) {} }
      setPlayIcon();
      if (master) {
        const tLocal = Math.max(0, master.currentTime - sceneStart0(S.activeScene));
        S.playhead = Math.min(S.seamTimes[S.activeScene] + tLocal, trimRange().end);
      }
      updatePlayheadUI();
      drawMulti(S.playhead - S.seamTimes[S.activeScene]);
      scheduleStill();
      return;
    }
    try { ui.video.pause(); } catch (e) {}
    setPlayIcon();
    const stopped = ui.video.currentTime - fleetLead();
    S.playhead = isFinite(stopped) && stopped > 0 ? stopped : S.playhead;
    updatePlayheadUI();
    syncFace(S.playhead);
    syncChannels(S.playhead);
    scheduleStill();
  }

  function cutsApplyToThisTake() {
    // Mirrors the render's v1 refusal set (scene / multi-native /
    // card-layout takes print a note and export UN-cut): the editor must
    // not promise an "Export" length -- or skip playback -- for a take
    // whose export will not ripple.
    if (S.details && (S.details.multi_native || S.details.scene_take)) return false;
    return !((S.edits && S.edits.windows) || []).length;
  }

  function cutRanges() {
    // Union-merged, sorted, clipped cut ranges (ripple delete) -- the
    // client-side mirror of the render's union. Frame-grid snapping stays
    // server-side; a sub-frame sliver is invisible at timeline scale.
    const d = duration();
    const raw = ((S.edits && S.edits.cuts) || [])
      .map(function (c) {
        return { start: clamp(+c.start || 0, 0, d),
                 end: clamp(+c.end || 0, 0, d) };
      })
      .filter(function (c) { return c.end > c.start; })
      .sort(function (a, b) { return a.start - b.start; });
    const out = [];
    raw.forEach(function (c) {
      if (out.length && c.start <= out[out.length - 1].end) {
        out[out.length - 1].end = Math.max(out[out.length - 1].end, c.end);
      } else {
        out.push({ start: c.start, end: c.end });
      }
    });
    return out;
  }

  function playbackTick() {
    if (!S.playing) return;
    const tr = trimRange();
    const t = ui.video.currentTime - fleetLead();
    // Skip over cut ranges: the cheapest honest preview of the ripple.
    // Scrubbing still lands inside a cut on purpose (the user is choosing
    // where the cut ends); only PLAYBACK jumps -- and only on takes whose
    // export actually ripples.
    const cuts = cutsApplyToThisTake() ? cutRanges() : [];
    for (let i = 0; i < cuts.length; i++) {
      if (t >= cuts[i].start && t < cuts[i].end - 0.015) {
        const target = Math.min(cuts[i].end, tr.end);
        if (target >= tr.end - 0.015) {
          // The cut runs to (or past) the clip end: this playback is
          // over. Without this, seek() clamps INSIDE the same cut and
          // the skip re-fires every frame -- a stuck play state.
          pause();
          seek(tr.end);
          return;
        }
        seek(target);
        requestAnimationFrame(playbackTick);
        return;
      }
    }
    S.playhead = t;
    applyTransformAt(t);
    updatePlayheadUI();
    if (t >= tr.end - 0.015 || ui.video.ended) {
      pause();
      seek(tr.end);
      return;
    }
    requestAnimationFrame(playbackTick);
  }

  /* ============================ timeline ============================ */

  function tlWidth() { return ui.tl.clientWidth; }

  function xToTime(clientX) {
    const rect = ui.tl.getBoundingClientRect();
    return clamp((clientX - rect.left) / Math.max(1, rect.width), 0, 1) * duration();
  }

  function pct(t) { return (clamp(t, 0, duration()) / duration() * 100); }

  function updatePlayheadUI() {
    ui.playhead.style.left = pct(S.playhead) + "%";
    ui.time.textContent = fmtTime(S.playhead);
    highlightTranscript();
  }

  function renderTimeline() {
    if (!S.edits) return;
    const d = duration();
    const tr = trimRange();

    drawRuler();
    drawWave();
    drawCamCurve();

    // trim
    ui.dimLeft.style.left = "0";
    ui.dimLeft.style.width = pct(tr.start) + "%";
    ui.dimRight.style.left = pct(tr.end) + "%";
    ui.dimRight.style.width = (100 - pct(tr.end)) + "%";
    ui.clipBar.style.left = pct(tr.start) + "%";
    ui.clipBar.style.width = Math.max(0.3, pct(tr.end) - pct(tr.start)) + "%";

    // cuts (ripple delete): read-only v1 -- MCP/CLI-authored ranges drawn
    // as struck regions, playback skips them (playbackTick), and the clip
    // label reports the EXPORT length. Authoring UI (drag/delete) arrives
    // with the transcript lane; deletion today is remove_cut via MCP.
    ui.clipLane.querySelectorAll(".ed-cut-dim").forEach(function (n) { n.remove(); });
    const cutsApply = cutsApplyToThisTake();
    let cutInClip = 0;
    cutRanges().forEach(function (c) {
      const a = Math.max(c.start, tr.start);
      const b = Math.min(c.end, tr.end);
      if (b > a) cutInClip += b - a;
      const el = document.createElement("div");
      el.className = "ed-cut-dim";
      el.title = cutsApply
        ? "Cut " + fmtTime(c.start) + "-" + fmtTime(c.end) +
          " / removed from the export"
        : "Cut " + fmtTime(c.start) + "-" + fmtTime(c.end) +
          " / saved, but not applied on this take type yet";
      el.style.left = pct(c.start) + "%";
      el.style.width = Math.max(0.2, pct(c.end) - pct(c.start)) + "%";
      ui.clipLane.appendChild(el);
    });
    ui.clipLabel.textContent = (cutsApply && cutInClip > 0.0005)
      ? "Export / " + fmtDuration(tr.end - tr.start - cutInClip) +
        " (" + fmtDuration(cutInClip) + " cut)"
      : "Clip / " + fmtDuration(tr.end - tr.start);
    ui.total.textContent = "/ " + fmtTime(tr.end);

    // zoom blocks
    ui.zoomLane.querySelectorAll(".ed-block").forEach(function (n) { n.remove(); });
    S.edits.zooms.forEach(function (z) {
      ui.zoomLane.appendChild(buildBlock(z, "zoom"));
    });

    // suppress blocks
    ui.suppressLane.querySelectorAll(".ed-block").forEach(function (n) { n.remove(); });
    S.edits.suppressed.forEach(function (r) {
      ui.suppressLane.appendChild(buildBlock(r, "suppress"));
    });

    // markers
    ui.markerLane.querySelectorAll(".ed-marker").forEach(function (n) { n.remove(); });
    S.edits.markers.forEach(function (m, i) {
      ui.markerLane.appendChild(buildMarker(m, i));
    });

    updatePlayheadUI();
  }

  function buildBlock(item, kind) {
    const b = el("div", "ed-block ed-block-" + kind);
    b.style.left = pct(item.start) + "%";
    b.style.width = Math.max(0.35, pct(item.end) - pct(item.start)) + "%";
    if (S.sel && S.sel.kind === kind && S.sel.id === item.id) b.classList.add("selected");
    if (kind === "zoom") {
      const pinned = item.x !== null && item.x !== undefined;
      b.appendChild(el("span", "ed-block-label",
        (toNum(item.level, 2)).toFixed(1) + "× " + (pinned ? "Pinned" : "Auto")));
      b.title = "Manual zoom " + fmtTime(item.start) + " - " + fmtTime(item.end);
    } else {
      b.title = "Auto-zoom suppressed " + fmtTime(item.start) + " - " + fmtTime(item.end);
    }
    const hl = el("div", "ed-block-handle l");
    const hr = el("div", "ed-block-handle r");
    b.appendChild(hl); b.appendChild(hr);
    hl.addEventListener("pointerdown", function (ev) { startBlockDrag(ev, kind, item.id, "start"); });
    hr.addEventListener("pointerdown", function (ev) { startBlockDrag(ev, kind, item.id, "end"); });
    b.addEventListener("pointerdown", function (ev) {
      if (ev.target === hl || ev.target === hr) return;
      startBlockDrag(ev, kind, item.id, "move");
    });
    return b;
  }

  function buildMarker(m, index) {
    const node = el("div", "ed-marker", m.label || ("Marker " + (index + 1)));
    node.style.left = pct(m.time) + "%";
    if (S.sel && S.sel.kind === "marker" && S.sel.id === m.id) node.classList.add("selected");
    node.title = fmtTime(m.time);
    node.addEventListener("pointerdown", function (ev) { startMarkerDrag(ev, m.id); });
    return node;
  }

  /* ---- drags ---- */

  function beginDrag(descriptor) {
    // snapshot now, but only PUSH it once real movement happens — a plain
    // click must not disturb the undo/redo stacks at all
    descriptor.snapshot = JSON.stringify(S.edits);
    descriptor.moved = false;
    S.drag = descriptor;
    window.addEventListener("pointermove", onDragMove);
    window.addEventListener("pointerup", onDragUp, { once: true });
  }

  function onDragMove(ev) {
    const dstate = S.drag;
    if (!dstate) return;
    const dxPx = ev.clientX - dstate.startX;
    if (!dstate.moved && Math.abs(dxPx) < 3) return;
    if (!dstate.moved) {
      dstate.moved = true;
      pushUndo(dstate.snapshot);
    }
    const dt = (dxPx / Math.max(1, tlWidth())) * duration();
    dstate.apply(dt, ev);
    S.mutCount++;
    renderTimeline();
    if (dstate.livePanel) renderPanel();
  }

  function onDragUp(ev) {
    window.removeEventListener("pointermove", onDragMove);
    const dstate = S.drag;
    S.drag = null;
    if (!dstate) return;
    if (!dstate.moved) {
      if (dstate.onClick) dstate.onClick(ev);
      return;
    }
    afterMutate({ coalesce: true });
  }

  function startBlockDrag(ev, kind, id, mode) {
    ev.preventDefault();
    ev.stopPropagation();
    const item = findItem(kind, id);
    if (!item) return;
    const d = duration();
    const MIN = 0.15;
    beginDrag({
      startX: ev.clientX,
      livePanel: true,
      apply: function (dt) {
        const orig = this._orig || (this._orig = { start: item.start, end: item.end });
        if (mode === "move") {
          const span = orig.end - orig.start;
          const ns = clamp(orig.start + dt, 0, d - span);
          item.start = ns;
          item.end = ns + span;
        } else if (mode === "start") {
          item.start = clamp(orig.start + dt, 0, orig.end - MIN);
        } else {
          item.end = clamp(orig.end + dt, orig.start + MIN, d);
        }
      },
      onClick: function () {
        selectItem(kind, id);
      },
    });
  }

  function startMarkerDrag(ev, id) {
    ev.preventDefault();
    ev.stopPropagation();
    const m = findItem("marker", id);
    if (!m) return;
    const d = duration();
    beginDrag({
      startX: ev.clientX,
      apply: function (dt) {
        const orig = this._orig !== undefined ? this._orig : (this._orig = m.time);
        m.time = clamp(orig + dt, 0, d);
        if (S.sel && S.sel.kind === "marker" && S.sel.id === id) positionMarkerPop();
      },
      onClick: function () {
        selectItem("marker", id);
      },
    });
  }

  function startTrimDrag(ev, which) {
    ev.preventDefault();
    ev.stopPropagation();
    // Scene takes ignore trim on export, so authoring one here would be a
    // lying gesture that only corrupts the timeline (see trimRange).
    if (S.details && S.details.scene_take) return;
    const d = duration();
    beginDrag({
      startX: ev.clientX,
      apply: function (dt) {
        const orig = this._orig || (this._orig = trimRange());
        if (which === "start") {
          S.edits.trim.start = clamp(orig.start + dt, 0, orig.end - 0.05);
        } else {
          const ne = clamp(orig.end + dt, orig.start + 0.05, d);
          S.edits.trim.end = ne >= d - 0.02 ? null : ne;
        }
        const tr = trimRange();
        S.playhead = clamp(S.playhead, tr.start, tr.end);
      },
    });
  }

  /* scrubbing on the ruler / lane backgrounds */
  function startScrub(ev) {
    if (ev.button !== 0) return;
    disarmPin();
    if (S.playing) pause();
    S.scrubbing = true;
    seek(xToTime(ev.clientX), { noStill: true });
    const move = function (e) { seek(xToTime(e.clientX), { noStill: true }); };
    const up = function () {
      S.scrubbing = false;
      window.removeEventListener("pointermove", move);
      seek(S.playhead);           // triggers the still
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up, { once: true });
  }

  ui.ruler.addEventListener("pointerdown", startScrub);
  [ui.clipLane, ui.zoomLane, ui.suppressLane, ui.markerLane].forEach(function (lane) {
    lane.addEventListener("pointerdown", function (ev) {
      if (ev.target !== lane && !ev.target.classList.contains("ed-wave") &&
          !ev.target.classList.contains("ed-clip-bar") &&
          !ev.target.classList.contains("ed-trim-dim")) return;
      startScrub(ev);
    });
  });
  ui.trimLeft.addEventListener("pointerdown", function (ev) { startTrimDrag(ev, "start"); });
  ui.trimRight.addEventListener("pointerdown", function (ev) { startTrimDrag(ev, "end"); });

  ui.zoomLane.addEventListener("dblclick", function (ev) {
    addZoomAt(xToTime(ev.clientX));
  });
  ui.suppressLane.addEventListener("dblclick", function (ev) {
    addSuppressAt(xToTime(ev.clientX));
  });
  ui.markerLane.addEventListener("dblclick", function (ev) {
    addMarkerAt(xToTime(ev.clientX));
  });

  /* ---- ruler + waveform drawing ---- */

  function setupCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  function niceTickStep(d, width) {
    const target = 80;                          // px between labeled ticks
    const steps = [0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 30, 60, 120];
    for (let i = 0; i < steps.length; i++) {
      if ((steps[i] / d) * width >= target) return steps[i];
    }
    return 120;
  }

  function drawRuler() {
    const ctx = setupCanvas(ui.ruler);
    const w = ui.ruler.clientWidth, h = ui.ruler.clientHeight;
    const d = duration();
    const step = niceTickStep(d, w);
    ctx.font = "500 9px ui-monospace, 'SF Mono', Menlo, monospace";
    ctx.textBaseline = "top";
    for (let t = 0; t <= d + 1e-6; t += step / 4) {
      const x = Math.round((t / d) * w) + 0.5;
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      ctx.strokeStyle = major ? "rgba(255,250,240,0.28)" : "rgba(255,250,240,0.10)";
      ctx.beginPath();
      ctx.moveTo(x, major ? 11 : 17);
      ctx.lineTo(x, 24);
      ctx.stroke();
      if (major) {
        ctx.fillStyle = "rgba(244,241,236,0.40)";
        ctx.fillText(fmtTime(t, false), x + 4, 2);
      }
    }
    // click ticks — scope teal like the camera line they trigger
    const clicks = (S.details && S.details.click_times) || [];
    ctx.fillStyle = "rgba(92,217,204,0.8)";
    clicks.forEach(function (t) {
      const x = (t / d) * w;
      ctx.beginPath();
      ctx.arc(x, 22, 1.6, 0, Math.PI * 2);
      ctx.fill();
    });
    // typing activity — teal micro-ticks above the click dots, deduped to
    // pixel columns (long recordings can hold thousands of key ticks)
    const keys = (S.details && S.details.key_times) || [];
    if (keys.length) {
      const cols = {};
      for (let i = 0; i < keys.length; i++) {
        cols[Math.round((keys[i] / d) * w)] = true;
      }
      ctx.fillStyle = "rgba(92,217,204,0.55)";
      Object.keys(cols).forEach(function (x) {
        ctx.fillRect(Number(x) - 0.5, 14, 1, 4);
      });
    }
    // scroll activity — dimmer teal micro-ticks between typing and clicks,
    // so scroll-triggered camera holds are explainable from the ruler
    const scrolls = (S.details && S.details.scroll_times) || [];
    if (scrolls.length) {
      const cols = {};
      for (let i = 0; i < scrolls.length; i++) {
        cols[Math.round((scrolls[i] / d) * w)] = true;
      }
      ctx.fillStyle = "rgba(92,217,204,0.30)";
      Object.keys(cols).forEach(function (x) {
        ctx.fillRect(Number(x) - 0.5, 19, 1, 2);
      });
    }
  }

  /* the camera line: the planned zoom level z(t) over the whole clip, drawn
     into the zoom lane behind the manual blocks. zoom 1.0 hugs the floor;
     the plan's peak zoom touches the top. This is the auto-zoom made visible
     BEFORE playback — cause (click dots above) and effect (the glide). */
  function drawCamCurve() {
    if (!ui.cam) return;
    const ctx = setupCanvas(ui.cam);
    const w = ui.cam.clientWidth, h = ui.cam.clientHeight;
    const p = S.camPath;
    if (!p || !p.times || p.times.length < 2 || w <= 0 || h <= 0) return;
    const d = duration();
    let zMax = 1;
    for (let i = 0; i < p.z.length; i++) if (p.z[i] > zMax) zMax = p.z[i];
    const rz = S.edits && S.edits.render ? toNum(S.edits.render.zoom, 2) : 2;
    zMax = Math.max(zMax, rz, 1.15);
    const pad = 3.5;
    const yFor = function (z) {
      return h - pad - ((z - 1) / (zMax - 1)) * (h - pad * 2);
    };
    // decimate to ~2 samples per px so drags stay cheap
    const n = p.times.length;
    const stride = Math.max(1, Math.floor(n / (w * 2)));
    const xs = [], ys = [];
    for (let i = 0; i < n; i += stride) {
      xs.push((p.times[i] / d) * w);
      ys.push(yFor(p.z[i]));
    }
    xs.push((p.times[n - 1] / d) * w);
    ys.push(yFor(p.z[n - 1]));

    const tracePath = function () {
      ctx.beginPath();
      ctx.moveTo(xs[0], ys[0]);
      for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], ys[i]);
    };
    // soft teal wash under the line
    tracePath();
    ctx.lineTo(xs[xs.length - 1], h + 2);
    ctx.lineTo(xs[0], h + 2);
    ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, "rgba(63, 191, 178, 0.26)");
    g.addColorStop(1, "rgba(63, 191, 178, 0.03)");
    ctx.fillStyle = g;
    ctx.fill();
    // the line itself
    tracePath();
    ctx.strokeStyle = "rgba(92, 217, 204, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.stroke();
  }

  function drawWave() {
    const ctx = setupCanvas(ui.wave);
    const w = ui.wave.clientWidth, h = ui.wave.clientHeight;
    const peaks = S.waveform && S.waveform.peaks;
    if (!peaks || !peaks.length || w <= 0) return;
    const tr = trimRange();
    const d = duration();
    // the wave canvas covers only the clip bar (trim range)
    ctx.fillStyle = "rgba(255, 206, 150, 0.5)";
    const n = peaks.length;
    for (let x = 0; x < w; x += 2) {
      const t = tr.start + (x / w) * (tr.end - tr.start);
      const idx = clamp(Math.floor((t / d) * n), 0, n - 1);
      const amp = clamp(peaks[idx], 0, 1);
      const bh = Math.max(1.5, amp * (h - 8));
      ctx.fillRect(x, (h - bh) / 2, 1.4, bh);
    }
  }

  /* ============================ create / select / delete ============================ */

  function selectItem(kind, id) {
    S.sel = { kind: kind, id: id };
    if (kind === "marker") {
      renderTimeline();
      openMarkerPop();
    } else {
      closeMarkerPop();
      if (kind === "zoom" || kind === "suppress") setPanel("zoom");
      renderTimeline();
      renderPanel();
    }
    applyTransformAt(S.playhead);
  }

  function deselect() {
    S.sel = null;
    closeMarkerPop();
    renderTimeline();
    renderPanel();
    ui.pinDot.hidden = true;
  }

  function addZoomAt(t) {
    const d = duration();
    const span = Math.min(1.2, d);
    const start = clamp(t - span / 2, 0, d - span);
    const id = tempId("zoom");
    mutate(function (e) {
      e.zooms.push({ id: id, start: start, end: start + span, x: null, y: null,
        level: toNum(e.render.zoom, 2.2) });
      e.zooms.sort(function (a, b) { return a.start - b.start; });
    });
    selectItem("zoom", id);
  }

  function addSuppressAt(t) {
    const d = duration();
    const span = Math.min(1.0, d);
    const start = clamp(t - span / 2, 0, d - span);
    const id = tempId("suppress");
    mutate(function (e) {
      e.suppressed.push({ id: id, start: start, end: start + span });
      e.suppressed.sort(function (a, b) { return a.start - b.start; });
    });
    selectItem("suppress", id);
  }

  function addMarkerAt(t) {
    const id = tempId("marker");
    mutate(function (e) {
      e.markers.push({ id: id, time: clamp(t, 0, duration()), label: null });
      e.markers.sort(function (a, b) { return a.time - b.time; });
    }, { camera: false, still: false });
    selectItem("marker", id);
  }

  function deleteSelection() {
    if (!S.sel) return;
    const sel = S.sel;
    mutate(function (e) {
      if (sel.kind === "zoom") e.zooms = e.zooms.filter(function (z) { return z.id !== sel.id; });
      else if (sel.kind === "suppress") e.suppressed = e.suppressed.filter(function (r) { return r.id !== sel.id; });
      else e.markers = e.markers.filter(function (m) { return m.id !== sel.id; });
    }, { camera: sel.kind !== "marker", still: sel.kind !== "marker" });
    deselect();
  }

  /* ============================ pin placement ============================ */

  /* Screen px (e.g. ev.clientX/Y) -> source-video px, via the viewport's
     current contain-fit rect. Valid whenever the video is shown at
     identity scale (pinArm / winPick) -- NOT during normal camera-zoomed
     playback, where the video is panned/scaled to a moving crop window. */
  function viewportPxToSourcePx(clientX, clientY) {
    const d = S.details;
    const vpRect = ui.viewport.getBoundingClientRect();
    const s = Math.min(vpRect.width / d.width, vpRect.height / d.height);
    const ox = (vpRect.width - d.width * s) / 2;
    const oy = (vpRect.height - d.height * s) / 2;
    return { x: (clientX - vpRect.left - ox) / s, y: (clientY - vpRect.top - oy) / s };
  }

  /* Source-space tools need ONE source to point at. A multi-native take has
   * N window buffers and no shared coordinate space, which is exactly what
   * `mcp_server._refuse_manual_spatial_edit` already refuses on the MCP side;
   * without this the picker would happily write rects measured against the
   * 1920x1080 COMPOSITE canvas into edits.crop / edits.windows, where they
   * mean nothing. */
  function spatialToolsBlocked() {
    // SOURCE-space tools (crop, zoom-pin, window-pick) — refused for
    // multi-native / scene takes, which surface no single source space to map
    // a viewport rect back into a window's pixels. Card PLACEMENT is NOT one
    // of these (it is pure canvas geometry); see placementBlocked().
    return !!(S.details && (S.details.multi_native || S.details.scene_take));
  }

  /* Card MOVE / RESIZE — pure canvas-space placement, so it relaxes where the
     source-space tools above do not. Allowed only where there is a per-card
     store to write into: display-crop (edits.windows) and multi-native
     (channel_layouts, docs/architecture.md P1). Scene takes are P2 (no
     scene_layouts store wired yet); plain takes have no cards. */
  function placementBlocked() {
    const d = S.details;
    if (!d) return true;
    if (d.multi_native) return false;
    if (d.scene_take) return !sceneLive();   // P2: allowed once the plan loads
    return !((S.edits && S.edits.windows) || []).length;
  }

  function armPin() {
    if (!S.sel || S.sel.kind !== "zoom") return;
    if (spatialToolsBlocked()) return;
    disarmWinPick();
    if (S.playing) pause();
    S.pinArm = true;
    ui.canvas.classList.add("pin-arm");
    ui.pinHint.hidden = false;
    ui.still.classList.remove("show");
    S.stillShown = false;
    ui.pinDot.hidden = true;
    const item = findItem("zoom", S.sel.id);
    if (item) seek(clamp(S.playhead, item.start, item.end), { noStill: true });
    applyTransformAt(S.playhead);
  }

  function disarmPin() {
    if (!S.pinArm) return;
    S.pinArm = false;
    ui.canvas.classList.remove("pin-arm");
    ui.pinHint.hidden = true;
    applyTransformAt(S.playhead);
    scheduleStill();
  }

  ui.canvas.addEventListener("click", function (ev) {
    if (!S.pinArm || !S.sel || S.sel.kind !== "zoom") return;
    const d = S.details;
    const p = viewportPxToSourcePx(ev.clientX, ev.clientY);
    if (p.x < 0 || p.y < 0 || p.x > d.width || p.y > d.height) return;
    const selId = S.sel.id;
    mutate(function (e) {
      const item = e.zooms.find(function (z) { return z.id === selId; });
      if (item) { item.x = Math.round(p.x); item.y = Math.round(p.y); }
    });
    disarmPin();
  });

  /* ============================ window crop picker ============================
   * Multi-window mode composites 1-4 static crops of this recording as
   * separate framed cards; see edits.windows / autocine/framing.py's
   * MultiFramePainter. This tool lets the user drag out each crop rect
   * directly on the raw (identity-transform) source, the same trick armPin
   * uses -- the source is always fully visible at true scale here,
   * regardless of the composited multi-window render.
   */

  let winRectDrag = null;

  function armWinPick(windowId) {
    if (!S.details) return;
    if (spatialToolsBlocked()) return;
    disarmPin();
    if (S.playing) pause();
    S.winPick = { id: windowId || null };
    ui.canvas.classList.add("winpick-arm");
    ui.winPickHint.hidden = false;
    ui.still.classList.remove("show");
    S.stillShown = false;
    ui.pinDot.hidden = true;
    applyTransformAt(S.playhead);
    renderPanel();
  }

  function disarmWinPick() {
    if (!S.winPick) return;
    S.winPick = null;
    ui.canvas.classList.remove("winpick-arm");
    ui.winPickHint.hidden = true;
    ui.winPickRect.hidden = true;
    applyTransformAt(S.playhead);
    scheduleStill();
    renderPanel();
  }

  /* ============================== frame crop ==============================
   *
   * `edits.crop` is a spatial trim: one static rect, in the same source px
   * `edits.windows` and the zoom pins use, that render.py slices out of every
   * frame before the camera runs (render._user_crop_px). Dragging it needs the
   * identity-scale stage for exactly the reason the window picker does -- the
   * rect is authored against the real recording, not against whatever the
   * camera happens to be showing.
   *
   * Deliberately NOT a render option: it changes what the source *is*, so it
   * sits beside `trim` rather than under `render`.
   */

  let cropDrag = null;

  function armCrop() {
    if (!S.details) return;
    if (spatialToolsBlocked()) return;
    disarmPin();
    disarmWinPick();
    if (S.playing) pause();
    S.cropArm = true;
    ui.canvas.classList.add("winpick-arm");
    ui.cropHint.hidden = false;
    ui.still.classList.remove("show");
    S.stillShown = false;
    ui.pinDot.hidden = true;
    applyTransformAt(S.playhead);
    renderPanel();
  }

  function disarmCrop() {
    if (!S.cropArm) return;
    S.cropArm = false;
    ui.canvas.classList.remove("winpick-arm");
    ui.cropHint.hidden = true;
    ui.cropRect.hidden = true;
    applyTransformAt(S.playhead);
    scheduleStill();
    renderPanel();
  }

  /* Source px -> canvas px: the inverse of viewportPxToSourcePx, expressed
     against ui.canvas because that is what the overlay is positioned in. */
  function sourceRectToCanvasRect(rect) {
    const d = S.details;
    if (!d || !rect) return null;
    const vpRect = ui.viewport.getBoundingClientRect();
    const cRect = ui.canvas.getBoundingClientRect();
    const s = Math.min(vpRect.width / d.width, vpRect.height / d.height);
    if (!(s > 0)) return null;
    const ox = vpRect.left - cRect.left + (vpRect.width - d.width * s) / 2;
    const oy = vpRect.top - cRect.top + (vpRect.height - d.height * s) / 2;
    return { left: ox + rect.x * s, top: oy + rect.y * s,
             width: rect.w * s, height: rect.h * s };
  }

  /* The saved crop, drawn over the stage. Only while the tool is armed: the
     rest of the time the preview already SHOWS the cropped render, so
     painting the box on top of it would double-report the crop. */
  function paintCropBox() {
    const crop = S.edits && S.edits.crop;
    if (!S.cropArm || !crop) { ui.cropBox.hidden = true; return; }
    const r = sourceRectToCanvasRect(crop);
    if (!r) { ui.cropBox.hidden = true; return; }
    ui.cropBox.style.left = r.left + "px";
    ui.cropBox.style.top = r.top + "px";
    ui.cropBox.style.width = Math.max(1, r.width) + "px";
    ui.cropBox.style.height = Math.max(1, r.height) + "px";
    ui.cropBox.hidden = false;
  }

  ui.canvas.addEventListener("pointerdown", function (ev) {
    if (!S.cropArm) return;
    ev.preventDefault();
    cropDrag = {
      startClientX: ev.clientX, startClientY: ev.clientY,
      start: viewportPxToSourcePx(ev.clientX, ev.clientY),
    };
    ui.cropBox.hidden = true;
    ui.cropRect.hidden = false;
    window.addEventListener("pointermove", onCropMove);
    window.addEventListener("pointerup", onCropUp, { once: true });
  });

  function onCropMove(ev) {
    if (!cropDrag) return;
    const cRect = ui.canvas.getBoundingClientRect();
    const x0 = Math.min(cropDrag.startClientX, ev.clientX);
    const y0 = Math.min(cropDrag.startClientY, ev.clientY);
    const x1 = Math.max(cropDrag.startClientX, ev.clientX);
    const y1 = Math.max(cropDrag.startClientY, ev.clientY);
    ui.cropRect.style.left = (x0 - cRect.left) + "px";
    ui.cropRect.style.top = (y0 - cRect.top) + "px";
    ui.cropRect.style.width = Math.max(1, x1 - x0) + "px";
    ui.cropRect.style.height = Math.max(1, y1 - y0) + "px";
  }

  function onCropUp(ev) {
    window.removeEventListener("pointermove", onCropMove);
    const drag = cropDrag;
    cropDrag = null;
    ui.cropRect.hidden = true;
    if (!drag) return;
    const d = S.details;
    const end = viewportPxToSourcePx(ev.clientX, ev.clientY);
    const x0 = clamp(Math.min(drag.start.x, end.x), 0, d.width);
    const y0 = clamp(Math.min(drag.start.y, end.y), 0, d.height);
    const x1 = clamp(Math.max(drag.start.x, end.x), 0, d.width);
    const y1 = clamp(Math.max(drag.start.y, end.y), 0, d.height);
    // A zero-sized viewport (the stage caught mid-relayout) makes the
    // contain-fit scale 0 and every coordinate above comes back NaN -- which
    // would sail through the size check below as a silently dropped crop.
    if (![x0, y0, x1, y1].every(Number.isFinite)) { disarmCrop(); return; }
    // Even sides, rounded INWARD, because libx264 -pix_fmt yuv420p rejects an
    // odd dimension and this rect becomes the output's. render._user_crop_px
    // even-izes too; matching here keeps the number the panel reports equal
    // to the number that gets rendered.
    const w = Math.floor(x1 - x0) & ~1;
    const h = Math.floor(y1 - y0) & ~1;
    // Below the render stage's own floor there is nothing to save: it would
    // drop the rect and the panel would claim a crop that never happened.
    if (w < 16 || h < 16) { disarmCrop(); return; }
    const rect = { x: Math.round(x0), y: Math.round(y0), w: w, h: h };
    mutate(function (e) { e.crop = rect; });
    disarmCrop();
  }

  /* Card dragging lives on the composite canvas itself, not on ui.canvas:
     the composite is the only thing showing cards, and binding here means
     the two drag tools can't fight over the same element. */
  ui.multi.addEventListener("pointerdown", onCardDown);
  ui.multi.addEventListener("pointermove", onCardMove);
  ui.multi.addEventListener("pointerup", onCardUp);
  ui.multi.addEventListener("pointercancel", onCardUp);
  ui.multi.addEventListener("pointerleave", function () {
    ui.multi.style.cursor = "";
    // `|| sceneLive()`: a scene take draws its composite but multiReady() is
    // false (no top-level windows_mode), so gating the redraw on multiReady
    // alone would strand the faint hover outline on the composite after the
    // pointer left. Same gate deselectCard() uses.
    if (S.cardHover) {
      S.cardHover = null;
      if (multiReady() || sceneLive()) drawMulti();
    }
  });

  ui.canvas.addEventListener("pointerdown", function (ev) {
    if (!S.winPick) return;
    ev.preventDefault();
    winRectDrag = {
      startClientX: ev.clientX, startClientY: ev.clientY,
      start: viewportPxToSourcePx(ev.clientX, ev.clientY),
    };
    ui.winPickRect.hidden = false;
    window.addEventListener("pointermove", onWinRectMove);
    window.addEventListener("pointerup", onWinRectUp, { once: true });
  });

  function onWinRectMove(ev) {
    if (!winRectDrag) return;
    const cRect = ui.canvas.getBoundingClientRect();
    const x0 = Math.min(winRectDrag.startClientX, ev.clientX);
    const y0 = Math.min(winRectDrag.startClientY, ev.clientY);
    const x1 = Math.max(winRectDrag.startClientX, ev.clientX);
    const y1 = Math.max(winRectDrag.startClientY, ev.clientY);
    ui.winPickRect.style.left = (x0 - cRect.left) + "px";
    ui.winPickRect.style.top = (y0 - cRect.top) + "px";
    ui.winPickRect.style.width = Math.max(1, x1 - x0) + "px";
    ui.winPickRect.style.height = Math.max(1, y1 - y0) + "px";
  }

  function onWinRectUp(ev) {
    window.removeEventListener("pointermove", onWinRectMove);
    const drag = winRectDrag;
    winRectDrag = null;
    ui.winPickRect.hidden = true;
    if (!drag) return;
    const d = S.details;
    const end = viewportPxToSourcePx(ev.clientX, ev.clientY);
    const x0 = clamp(Math.min(drag.start.x, end.x), 0, d.width);
    const y0 = clamp(Math.min(drag.start.y, end.y), 0, d.height);
    const x1 = clamp(Math.max(drag.start.x, end.x), 0, d.width);
    const y1 = clamp(Math.max(drag.start.y, end.y), 0, d.height);
    const w = x1 - x0, h = y1 - y0;
    if (w < 8 || h < 8) { disarmWinPick(); return; }  // too small to be intentional
    const rect = { x: Math.round(x0), y: Math.round(y0), w: Math.round(w), h: Math.round(h) };
    const pickId = S.winPick ? S.winPick.id : null;
    mutate(function (e) {
      if (!e.windows) e.windows = [];
      if (pickId) {
        const it = e.windows.find(function (item) { return item.id === pickId; });
        if (it) { it.x = rect.x; it.y = rect.y; it.w = rect.w; it.h = rect.h; }
      } else if (e.windows.length < 4) {
        e.windows.push(Object.assign({ id: tempId("window") }, rect));
      }
    });
    disarmWinPick();
  }

  /* ============================ inspector panels ============================ */

  const RAIL_ICONS = { zoom: "zoom", frame: "frame", windows: "windows", cursor: "cursor", facecam: "camera", audio: "audio", transcript: "transcript" };

  /* The arrangements the server knows (`edits._WINDOW_LAYOUTS`), and the two
     `span`s the segmented control lays them out on -- see `rowSegStack`. Named
     here rather than inline so the "is the saved value still one of these?"
     check below can't drift from the buttons it paints. */
  const WINDOW_LAYOUTS = [
    { v: "grid", label: "Grid", span: 3 },
    { v: "desktop", label: "Desktop", span: 3 },
    { v: "feature", label: "Feature", span: 2 },
    { v: "row", label: "Row", span: 2 },
    { v: "column", label: "Column", span: 2 },
  ];

  function setPanel(name) {
    S.panel = name;
    // The selection frame is a Windows-panel affordance; leaving the panel
    // clears it (and restores the crisp still it was suppressing).
    if (name !== "windows") deselectCard();
    ui.inspector.hidden = !name;
    ui.rail.querySelectorAll(".ed-rail-btn").forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-panel") === name);
    });
    renderPanel();
  }

  ui.rail.querySelectorAll(".ed-rail-btn").forEach(function (b) {
    b.innerHTML = ICONS[RAIL_ICONS[b.getAttribute("data-panel")]] || "";
    b.addEventListener("click", function () {
      const name = b.getAttribute("data-panel");
      setPanel(S.panel === name ? null : name);
    });
  });

  /* small builders */
  function rowSlider(label, min, max, step, value, fmt, onInput) {
    const wrap = el("div", "ed-slider-row");
    const head = el("div", "ed-slider-head");
    head.appendChild(el("span", "t-label", label));
    const val = el("span", "ed-slider-val", fmt(value));
    head.appendChild(val);
    wrap.appendChild(head);
    const input = document.createElement("input");
    input.type = "range";
    input.className = "slider";
    input.min = min; input.max = max; input.step = step; input.value = value;
    const paint = function () {
      const f = ((parseFloat(input.value) - min) / (max - min)) * 100;
      input.style.setProperty("--fill", f + "%");
    };
    paint();
    input.addEventListener("input", function () {
      val.textContent = fmt(parseFloat(input.value));
      paint();
      onInput(parseFloat(input.value));
    });
    wrap.appendChild(input);
    return wrap;
  }

  function rowSwitch(label, checked, onChange, disabled, hint) {
    const row = el("div", "field-row");
    row.appendChild(el("span", "t-label", label));
    const sw = el("label", "switch");
    const input = document.createElement("input");
    input.type = "checkbox";
    input.checked = !!checked;
    input.disabled = !!disabled;
    input.addEventListener("change", function () { onChange(input.checked); });
    sw.appendChild(input);
    sw.appendChild(el("span", "knob"));
    row.appendChild(sw);
    const frag = document.createDocumentFragment();
    frag.appendChild(row);
    if (hint) frag.appendChild(el("div", "ed-hint", hint));
    return frag;
  }

  function rowSeg(label, options, value, onChange) {
    const row = el("div", "field-row");
    if (label) row.appendChild(el("span", "t-label", label));
    const seg = el("div", "seg");
    options.forEach(function (o) {
      const b = el("button", o.v === value ? "on" : null, o.label);
      b.addEventListener("click", function () {
        seg.querySelectorAll("button").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        onChange(o.v);
      });
      seg.appendChild(b);
    });
    row.appendChild(seg);
    return row;
  }

  /* rowSeg, but the options sit on their own full-width rows under the label.
     Measured: five arrangement names in one `.seg` want 299px next to a 75px
     label, and the inspector gives the panel 229. Each option carries the
     number of `.seg-grid` columns (of six) it spans, so where the row breaks
     is a decision here rather than whatever the labels happened to measure. */
  function rowSegStack(label, options, value, onChange) {
    const wrap = el("div", "ed-slider-row");
    wrap.appendChild(el("span", "t-label", label));
    const seg = el("div", "seg seg-grid");
    options.forEach(function (o) {
      const b = el("button", o.v === value ? "on" : null, o.label);
      b.style.gridColumn = "span " + (o.span || 2);
      b.addEventListener("click", function () {
        seg.querySelectorAll("button").forEach(function (x) { x.classList.remove("on"); });
        b.classList.add("on");
        onChange(o.v);
      });
      seg.appendChild(b);
    });
    wrap.appendChild(seg);
    return wrap;
  }

  // Mirrors sfx.resolve() on the Python side: null / "" / "auto" all mean
  // the BUILT-IN sound (event sounds are on by default), and only an
  // explicit off-word silences a kind. Keep the two in step -- a switch
  // that disagrees with the renderer is worse than no switch at all.
  const SFX_OFF_WORDS = ["off", "none", "no", "false", "0"];

  function isOff(value) {
    if (value === null || value === undefined) return false;
    return SFX_OFF_WORDS.indexOf(String(value).trim().toLowerCase()) >= 0;
  }

  // Toggling a sound kind off writes "off" into the SAME field that holds a
  // custom file path, so a naive switch would destroy the user's path and
  // turn it back on as the built-in. Remember it for the session so the
  // obvious off/on fidget is not destructive. (Session-scoped: a reload
  // still loses a path the user turned off and never turned back on --
  // storing it properly would need a second edits field.)
  const sfxPathMemo = { click_sound: "", key_sound: "" };

  function rememberSfxPath(field, value) {
    const path = customPath(value);
    if (path) sfxPathMemo[field] = path;
    return path;
  }

  function sfxToggleValue(field, on) {
    if (!on) return "off";
    return sfxPathMemo[field] || null;
  }

  function customPath(value) {
    if (value === null || value === undefined || isOff(value)) return "";
    const text = String(value).trim();
    return (text === "" || text.toLowerCase() === "auto") ? "" : text;
  }

  function rowText(label, value, placeholder, onCommit) {
    const wrap = el("div", "ed-slider-row");
    wrap.appendChild(el("span", "t-label", label));
    const input = document.createElement("input");
    input.type = "text";
    input.value = value || "";
    input.placeholder = placeholder || "";
    input.addEventListener("change", function () { onCommit(input.value.trim() || null); });
    wrap.appendChild(input);
    return wrap;
  }

  function rowNumber(label, value, step, width, onCommit) {
    const row = el("div", "field-row");
    row.appendChild(el("span", "t-label", label));
    const input = document.createElement("input");
    input.type = "number";
    input.step = step;
    input.value = value;
    input.style.width = (width || 76) + "px";
    input.addEventListener("change", function () { onCommit(parseFloat(input.value)); });
    row.appendChild(input);
    return row;
  }

  function renderPanel() {
    // Cards become draggable only in the panel that is about them, so the
    // composite doesn't quietly swallow clicks meant for the stage.
    ui.multi.classList.toggle("is-draggable", S.panel === "windows");
    drawCardOverlay();   // show/hide the handle overlay as the Windows panel opens/closes
    paintCropBox();
    if (!S.panel || !S.edits) { ui.inspector.hidden = true; return; }
    ui.inspector.hidden = false;
    const p = ui.panel;
    // Remember where the transcript was scrolled to BEFORE the wipe. Read
    // synchronously off the outgoing node rather than tracked by a scroll
    // listener: this runs on every mutate and every adopted server doc, and
    // a listener would miss a rebuild that lands between two scroll events
    // (and never fires at all while the page is hidden). Without it,
    // cutting one filler word at 8:00 throws the reader back to 0:00 --
    // and cutting filler words IS the feature.
    const outgoing = p.querySelector(".ed-tx-list");
    if (outgoing) S.txScroll = outgoing.scrollTop;
    p.innerHTML = "";
    const r = S.edits.render;

    if (S.panel === "zoom") {
      p.appendChild(el("h3", null, "Zoom"));
      p.appendChild(rowSlider("Max auto zoom", 1.0, 4.0, 0.1, toNum(r.zoom, 2.2),
        function (v) { return v.toFixed(1) + "×"; },
        function (v) { mutate(function (e) { e.render.zoom = v; }, { skipPanel: true }); }));
      p.appendChild(rowSeg("Zoom speed",
        [{ v: "slow", label: "Slow" }, { v: "normal", label: "Normal" },
         { v: "fast", label: "Fast" }],
        r.zoom_speed || "normal", function (v) {
          mutate(function (e) { e.render.zoom_speed = v; }, { skipPanel: true });
        }));
      p.appendChild(rowSwitch("Always keep zoomed", r.always_zoomed, function (on) {
        mutate(function (e) { e.render.always_zoomed = on; }, { skipPanel: true });
      }, false, "Hold the final zoom level instead of settling back out at the end."));
      p.appendChild(rowSwitch("Overview framing", r.overview !== false, function (on) {
        mutate(function (e) { e.render.overview = on; }, { skipPanel: true });
      }, false, "When clicks are spread too wide to hold zoomed, pull back to a still overview locked on them instead of whip-panning between clicks."));
      p.appendChild(rowSwitch("Grow active window", r.screen_focus !== false, function (on) {
        mutate(function (e) { e.render.screen_focus = on; }, { skipPanel: true });
      }, false, "Whole-screen takes with several windows: when you click in one window, grow it a bit to overlap its neighbours while the frame eases in, instead of zooming into a crop that cuts across windows. On by default."));
      p.appendChild(rowSwitch("Motion blur", r.motion_blur !== false, function (on) {
        mutate(function (e) { e.render.motion_blur = on; }, { camera: false, skipPanel: true });
      }, false, "Blur fast camera pans and zooms in the export, like real camera shutter."));
      p.appendChild(rowNumber("Sync offset (s)", toNum(r.offset, 0), "0.01", 76, function (v) {
        if (Number.isFinite(v)) mutate(function (e) { e.render.offset = v; }, { skipPanel: true });
      }));

      const selZoom = S.sel && S.sel.kind === "zoom" ? findItem("zoom", S.sel.id) : null;
      const selSup = S.sel && S.sel.kind === "suppress" ? findItem("suppress", S.sel.id) : null;

      if (selZoom) {
        const sect = el("div", "sect");
        sect.appendChild(el("span", "t-cap sect-cap", "SELECTED ZOOM"));
        sect.appendChild(rowSlider("Level", 1.0, 6.0, 0.1, toNum(selZoom.level, 2),
          function (v) { return v.toFixed(1) + "×"; },
          function (v) {
            mutate(function (e) {
              const it = e.zooms.find(function (z) { return z.id === selZoom.id; });
              if (it) it.level = v;
            }, { skipPanel: true });
          }));
        const pinned = selZoom.x !== null && selZoom.x !== undefined;
        sect.appendChild(rowSeg("Target",
          [{ v: "auto", label: "Follow cursor" }, { v: "pin", label: "Pinned" }],
          pinned ? "pin" : "auto",
          function (v) {
            if (v === "pin") armPin();
            else {
              disarmPin();
              mutate(function (e) {
                const it = e.zooms.find(function (z) { return z.id === selZoom.id; });
                if (it) { it.x = null; it.y = null; }
              });
            }
          }));
        if (pinned) {
          const reBtn = el("button", "btn-quiet", "Re-place target...");
          reBtn.addEventListener("click", armPin);
          sect.appendChild(reBtn);
          sect.appendChild(el("div", "ed-hint",
            "Pinned to " + Math.round(selZoom.x) + ", " + Math.round(selZoom.y) + " (source px)."));
        }
        sect.appendChild(rowNumber("Start (s)", Math.round(selZoom.start * 100) / 100, "0.01", 76, function (v) {
          if (!Number.isFinite(v)) return;
          mutate(function (e) {
            const it = e.zooms.find(function (z) { return z.id === selZoom.id; });
            if (it) it.start = clamp(v, 0, it.end - 0.15);
          }, { skipPanel: true });
          renderPanel();
        }));
        sect.appendChild(rowNumber("End (s)", Math.round(selZoom.end * 100) / 100, "0.01", 76, function (v) {
          if (!Number.isFinite(v)) return;
          mutate(function (e) {
            const it = e.zooms.find(function (z) { return z.id === selZoom.id; });
            if (it) it.end = clamp(v, it.start + 0.15, duration());
          }, { skipPanel: true });
          renderPanel();
        }));
        const del = el("button", "btn-quiet btn-danger", "Remove zoom");
        del.style.marginTop = "8px";
        del.addEventListener("click", deleteSelection);
        sect.appendChild(del);
        p.appendChild(sect);
      } else if (selSup) {
        const sect = el("div", "sect");
        sect.appendChild(el("span", "t-cap sect-cap", "SELECTED NO-ZOOM RANGE"));
        sect.appendChild(el("div", "ed-hint",
          "Auto-zoom ignores clicks between " + fmtTime(selSup.start) + " and " + fmtTime(selSup.end) + "."));
        const del = el("button", "btn-quiet btn-danger", "Remove range");
        del.style.marginTop = "8px";
        del.addEventListener("click", deleteSelection);
        sect.appendChild(del);
        p.appendChild(sect);
      } else {
        p.appendChild(el("div", "ed-hint",
          "Zooms are planned automatically from your clicks. Add a manual zoom with the + Zoom button (Z), or select a block on the timeline to adjust it."));
      }
    }

    if (S.panel === "frame") {
      p.appendChild(el("h3", null, "Background & frame"));
      const framed = r.style === "framed" || !!r.background;
      p.appendChild(rowSeg("Style",
        [{ v: "clean", label: "Clean" }, { v: "framed", label: "Framed" }],
        framed ? "framed" : "clean",
        function (v) {
          mutate(function (e) {
            e.render.style = v;
            if (v === "clean") e.render.background = null;
          });
        }));

      if (framed) {
        const sect = el("div", "sect");
        sect.appendChild(el("span", "t-cap sect-cap", "WALLPAPER"));
        const grid = el("div", "swatch-grid");
        S.backgrounds.forEach(function (bg) {
          const sw = el("button", "swatch" + (r.background === bg.id ? " on" : ""));
          sw.style.background = gradientCss(bg.colors);
          sw.title = bg.id;
          sw.addEventListener("click", function () {
            mutate(function (e) { e.render.background = bg.id; e.render.style = "framed"; });
          });
          grid.appendChild(sw);
        });
        sect.appendChild(grid);

        const colorRow = el("div", "ed-color-row");
        colorRow.style.marginTop = "12px";
        const well = document.createElement("input");
        well.type = "color";
        well.value = r.background && r.background.charAt(0) === "#" ? r.background : "#131110";
        well.addEventListener("change", function () {
          mutate(function (e) { e.render.background = well.value; e.render.style = "framed"; });
        });
        colorRow.appendChild(well);
        colorRow.appendChild(el("span", "t-label", "Solid color"));
        sect.appendChild(colorRow);

        sect.appendChild(rowText("Wallpaper image (file path)",
          r.background && r.background.charAt(0) !== "#" && !isPresetId(r.background) ? r.background : "",
          "/path/to/wallpaper.jpg",
          function (v) {
            mutate(function (e) {
              e.render.background = v;
              if (v) e.render.style = "framed";
            });
          }));
        p.appendChild(sect);
      }

      const cropSect = el("div", "sect");
      cropSect.appendChild(el("span", "t-cap sect-cap", "CROP"));
      const crop = S.edits.crop;
      const d = S.details;
      cropSect.appendChild(el("div", "ed-hint", crop
        ? Math.round(crop.w) + " x " + Math.round(crop.h) + " px at ("
          + Math.round(crop.x) + ", " + Math.round(crop.y) + ")"
        : "Full frame" + (d ? " / " + d.width + " x " + d.height + " px" : "")));
      const cropRow = el("div", "field-row");
      if (S.cropArm) {
        cropSect.appendChild(el("div", "ed-hint",
          "Drag the part of the frame you want to keep."));
        const cancel = el("button", "btn-quiet", "Cancel");
        cancel.addEventListener("click", disarmCrop);
        cropRow.appendChild(cancel);
      } else {
        const draw = el("button", "btn-quiet", crop ? "Redraw..." : "Crop...");
        draw.addEventListener("click", armCrop);
        cropRow.appendChild(draw);
      }
      if (crop) {
        const reset = el("button", "btn-quiet btn-danger", "Reset");
        reset.title = "Back to the full frame";
        reset.addEventListener("click", function () {
          mutate(function (e) { e.crop = null; });
        });
        cropRow.appendChild(reset);
      }
      cropSect.appendChild(cropRow);
      p.appendChild(cropSect);

      const fadeSect = el("div", "sect");
      fadeSect.appendChild(el("span", "t-cap sect-cap", "TRANSITIONS"));
      fadeSect.appendChild(rowSlider("Fade in/out", 0, 2, 0.05, toNum(r.fade, 0),
        function (v) { return v <= 0 ? "Off" : v.toFixed(2) + "s"; },
        function (v) { mutate(function (e) { e.render.fade = v; }, { camera: false, skipPanel: true }); }));
      p.appendChild(fadeSect);
    }

    if (S.panel === "windows") {
      p.appendChild(el("h3", null, "Windows"));
      const windows = S.edits.windows || [];
      // Multi-native / scene takes: each card IS a separately-captured window
      // buffer (no edits.windows). The per-window crop controls below are for
      // display-crop takes only; native/scene get their own copy + a Reset.
      const nativeCards = !!(S.details && (S.details.multi_native
                                           || S.details.scene_take));
      if (nativeCards) {
        const scene = !!S.details.scene_take;
        p.appendChild(el("div", "ed-hint",
          (scene
            ? "Each scene was recorded with its own set of windows, each "
              + "captured to its own buffer. "
            : "Each window was captured to its own buffer and composited onto "
              + "the background. ")
          + "Click a window on the preview to select it, then drag it to move, "
          + "or grab a handle to resize. Resizing keeps the window's aspect "
          + "ratio, so the recording never stretches"
          + (scene ? ". Every scene keeps its own arrangement." : ".")));
        const overridden = scene
          ? !!(S.edits.scene_layouts
               && (S.edits.scene_layouts[String(S.activeScene)] || [])
                    .some(Boolean))
          : (S.edits.channel_layouts || []).some(Boolean);
        const acts = el("div", "ed-actions");
        const reset = el("button", "btn-quiet", "Reset placement");
        reset.title = "Drop every hand placement and go back to the arrangement";
        reset.disabled = !overridden;
        reset.addEventListener("click", resetCardPlacement);
        acts.appendChild(reset);
        // Remove a window from the video (reversible). Scene/join takes only in
        // v1 -- the live multi-native preview doesn't yet filter, so offering it
        // there would show a card the export drops.
        if (scene) {
          const remove = el("button", "btn-quiet", "Remove selected window");
          remove.title = "Click a window on the preview to select it, then "
            + "remove it from the video. The recording is kept, so you can "
            + "restore it.";
          remove.addEventListener("click", removeSelectedCard);
          acts.appendChild(remove);
        }
        const hiddenN = (S.edits.hidden_channels || []).length;
        if (hiddenN) {
          const restore = el("button", "btn-quiet",
            "Restore removed (" + hiddenN + ")");
          restore.title = "Bring every removed window back into the video";
          restore.addEventListener("click", restoreHiddenCards);
          acts.appendChild(restore);
        }
        p.appendChild(acts);
      } else {
        p.appendChild(el("div", "ed-hint",
          "Crop 1-4 regions from this recording and arrange them as separate "
          + "framed cards on a shared background. This gives a polished "
          + "window-capture look, generalized to multiple windows. The "
          + "whole-screen camera is off here; the two switches below are how "
          + "these cards move."));
      }

      windows.forEach(function (w, i) {
        const sect = el("div", "sect");
        sect.appendChild(el("span", "t-cap sect-cap", "WINDOW " + (i + 1)));
        sect.appendChild(el("div", "ed-hint",
          Math.round(w.w) + " × " + Math.round(w.h) + " px at ("
          + Math.round(w.x) + ", " + Math.round(w.y) + ")"));
        const row = el("div", "field-row");
        const redraw = el("button", "btn-quiet", "Redraw...");
        redraw.addEventListener("click", function () { armWinPick(w.id); });
        row.appendChild(redraw);
        if (i > 0) {
          const up = el("button", "btn-icon btn-quiet", "↑");
          up.title = "Move earlier in the arrangement";
          up.addEventListener("click", function () {
            mutate(function (e) {
              const idx = e.windows.findIndex(function (x) { return x.id === w.id; });
              if (idx > 0) {
                const t = e.windows[idx - 1];
                e.windows[idx - 1] = e.windows[idx];
                e.windows[idx] = t;
              }
            });
          });
          row.appendChild(up);
        }
        if (i < windows.length - 1) {
          const down = el("button", "btn-icon btn-quiet", "↓");
          down.title = "Move later in the arrangement";
          down.addEventListener("click", function () {
            mutate(function (e) {
              const idx = e.windows.findIndex(function (x) { return x.id === w.id; });
              if (idx >= 0 && idx < e.windows.length - 1) {
                const t = e.windows[idx + 1];
                e.windows[idx + 1] = e.windows[idx];
                e.windows[idx] = t;
              }
            });
          });
          row.appendChild(down);
        }
        const del = el("button", "btn-quiet btn-danger", "Remove");
        del.addEventListener("click", function () {
          mutate(function (e) {
            e.windows = e.windows.filter(function (x) { return x.id !== w.id; });
          });
        });
        row.appendChild(del);
        sect.appendChild(row);
        p.appendChild(sect);
      });

      if (windows.length > 1) {
        const known = WINDOW_LAYOUTS.some(function (o) { return o.v === r.window_layout; });
        p.appendChild(rowSegStack("Arrangement", WINDOW_LAYOUTS,
          known ? r.window_layout : "grid",
          function (v) {
            mutate(function (e) { e.render.window_layout = v; });
          }));
        p.appendChild(el("div", "ed-hint",
          "Feature runs window 1 large with the others beside it, and turns "
          + "that on its head for a tall export: window 1 on top, the rest "
          + "in a line beneath. Turning is why it is the safe pick at any "
          + "shape (feature covers 81.8% of a 16:10 canvas, 74.9% at 9:16). "
          + "Grid gives every card an equal cell, chosen against the export "
          + "aspect: the one to beat on a tall canvas at 73.2%. Desktop "
          + "keeps the windows where they sat on screen: 80.7% wide, a weak "
          + "29.1% tall. Column is the tall specialist at 67.5%; Row is that "
          + "line sideways, and wants the widest export of the five. Neither "
          + "of those turns. None of the five fills both axes: the cards "
          + "scale as one block, so one axis meets the margin and the other "
          + "keeps more. Use ↑ ↓ to choose which window leads, or click a "
          + "card on the preview to move or resize it by hand."));
      }

      if (windows.length > 0) {
        const acts = el("div", "ed-actions");
        const fit = el("button", "btn-quiet ed-fit", "Fit to frame");
        fit.addEventListener("click", fitCardsToFrame);
        acts.appendChild(fit);
        const reset = el("button", "btn-quiet", "Reset placement");
        reset.title = "Drop every hand placement and go back to the arrangement";
        reset.disabled = !windows.some(function (w) { return !!w.layout; });
        reset.addEventListener("click", resetCardPlacement);
        acts.appendChild(reset);
        p.appendChild(acts);
        p.appendChild(el("div", "ed-hint ed-fit-note"));
        p.appendChild(el("div", "ed-hint",
          "Fit to frame takes the slack out of a hand-placed arrangement: it "
          + "scales what is on screen up around the same margin the "
          + "arrangements use, keeping every card's aspect and their relative "
          + "sizes. An arrangement you just applied is already at that scale, "
          + "so there is nothing to grow. Under Grid it can still slide the "
          + "cards back to centre, because Grid places each card inside its "
          + "own cell. Reset clears every hand placement."));
        // Enabled state, tooltip and the note above all decided in one place,
        // so the button and the write can never disagree about what it does.
        syncFitAction();
      }

      if (windows.length > 0) {
        p.appendChild(rowSwitch("Focus active window", r.window_focus === true,
          function (on) {
            mutate(function (e) { e.render.window_focus = on; });
          }));
        p.appendChild(el("div", "ed-hint",
          "Focus the window you're working in. Clicking in a window grows "
          + "its card so it sits over the others; clicking again zooms the "
          + "whole screen in until that window fills the frame. The other "
          + "cards stay put until the zoom, and nothing is ever cropped. It "
          + "unwinds when you move on. Only one card is ever the subject."));
        p.appendChild(rowSwitch("Zoom inside cards", r.window_zoom === true,
          function (on) {
            mutate(function (e) { e.render.window_zoom = on; });
          }));
        p.appendChild(el("div", "ed-hint",
          "Give each card its own auto-zoom, planned from the clicks that "
          + "landed in that window. Only the card with the most recent "
          + "activity zooms. The others hold their full framing, so the "
          + "composition stays readable. Off keeps every card a steady "
          + "static crop."));
        p.appendChild(rowSwitch("Follow windows", r.window_follow !== false,
          function (on) {
            mutate(function (e) { e.render.window_follow = on; },
                   { skipPanel: true });
          }));
        p.appendChild(el("div", "ed-hint",
          "Each card sticks to the window it was drawn around, so a window "
          + "moved or resized mid-take stays framed. Needs a recording with "
          + "window geometry; older takes stay static either way."));
      }

      // "Draw a window" arms the source-space window picker, which is refused
      // for native/scene takes (their cards are fixed capture buffers, not
      // crops of a shared source) -- so it only belongs on a display-crop take.
      if (!nativeCards && windows.length < 4) {
        const add = el("button", "btn-quiet",
          windows.length === 0 ? "+ Draw a window" : "+ Add another window");
        add.style.marginTop = "8px";
        add.addEventListener("click", function () { armWinPick(null); });
        p.appendChild(add);
      } else if (!nativeCards) {
        p.appendChild(el("div", "ed-hint", "Maximum of 4 windows."));
      }

      if (S.winPick) {
        const pickSect = el("div", "sect");
        pickSect.appendChild(el("div", "ed-hint",
          "Drag a rectangle on the frame to "
          + (S.winPick.id ? "redraw this window." : "define the new window.")));
        const cancel = el("button", "btn-quiet", "Cancel");
        cancel.addEventListener("click", disarmWinPick);
        pickSect.appendChild(cancel);
        p.appendChild(pickSect);
      } else if (windows.length > 0) {
        p.appendChild(el("div", "ed-hint",
          "Playback composites this layout live, using the same arrangement "
          + "the export does. The paused frame is still exact, with "
          + "the facecam bubble and fade on top."));
      }
    }

    if (S.panel === "cursor") {
      p.appendChild(el("h3", null, "Cursor & clicks"));
      p.appendChild(rowSwitch("Click ripples", r.click_fx, function (on) {
        mutate(function (e) { e.render.click_fx = on; }, { camera: false, skipPanel: true });
      }));

      const colors = [null, "#ffffff", "#3fbfb2", "#e08a3c", "#e8c34a", "#e04b41"];
      const grid = el("div", "swatch-grid");
      grid.style.marginBottom = "10px";
      colors.forEach(function (cval) {
        const on = (r.click_color || null) === cval;
        const sw = el("button", "swatch" + (on ? " on" : ""));
        sw.style.background = cval || "linear-gradient(135deg,#3fbfb2,#e08a3c)";
        sw.title = cval || "Default";
        sw.addEventListener("click", function () {
          mutate(function (e) { e.render.click_color = cval; }, { camera: false });
        });
        grid.appendChild(sw);
      });
      p.appendChild(grid);

      p.appendChild(rowSwitch("Cursor spotlight", r.spotlight, function (on) {
        mutate(function (e) { e.render.spotlight = on; }, { camera: false, skipPanel: true });
      }, false, "Dims everything except a circle around the cursor while zoomed."));

      const synthetic = S.details && S.details.cursor_mode === "synthetic";

      // The eraser is the mirror image of the synthetic cursor below: that
      // one DRAWS a pointer, this one takes the recorded one out. Which one
      // your take NEEDS is decided at capture time -- but on a system-cursor
      // take they are also a pair, because erasing is exactly what leaves a
      // frame the synthetic cursor is allowed to draw into
      // (render._cursor_fx_draws). So the two switches move together, rather
      // than one of them sitting "on" while it silently does nothing.
      const erase = el("div", "sect");
      erase.appendChild(el("span", "t-cap sect-cap", "RECORDED CURSOR"));
      erase.appendChild(rowSwitch("Erase recorded cursor", r.cursor_erase, function (on) {
        // Turning the eraser back off puts the recorded pointer back in the
        // picture, which strands the drawn one -- clear it in the same edit.
        const alsoFx = !on && r.cursor_fx && !synthetic;
        mutate(function (e) {
          e.render.cursor_erase = on;
          if (alsoFx) e.render.cursor_fx = false;
        }, { camera: false, skipPanel: !alsoFx });
      }, synthetic, synthetic ?
        "This take was recorded with the cursor hidden. There is nothing burned into the picture." :
        "Repaints the pointer away using the real pixels from before it arrived and after it left."));
      if (!synthetic && r.cursor_erase) {
        erase.appendChild(el("div", "ed-hint",
          "Export scans the whole recording for clean pixels, so it takes " +
          "noticeably longer. Where the pointer sat still over something " +
          "that changed underneath it, those pixels were never recorded and " +
          "get a local repair instead. The export prints how many frames " +
          "needed one. This still previews from a half-second window either " +
          "side, so it can look worse here than in the export."));
      }
      p.appendChild(erase);

      const sect = el("div", "sect");
      sect.appendChild(el("span", "t-cap sect-cap", "SYNTHETIC CURSOR"));
      sect.appendChild(rowSwitch("Smooth synthetic cursor", r.cursor_fx, function (on) {
        // On a take recorded WITH the cursor, drawing one means lifting the
        // recorded one out first -- so this switch turns the eraser on too,
        // visibly, instead of being a control that quietly does nothing.
        const alsoErase = on && !synthetic && !r.cursor_erase;
        mutate(function (e) {
          e.render.cursor_fx = on;
          if (alsoErase) e.render.cursor_erase = true;
        }, { camera: false, skipPanel: !alsoErase });
      }, false, synthetic ?
        "Draws a smoothed, enlarged cursor from the recorded events." :
        "Replaces the recorded pointer with a smoothed, enlarged one. Turns “Erase recorded cursor” on, because the real one has to come out of the picture first."));
      if (synthetic || r.cursor_fx) {
        sect.appendChild(rowSlider("Cursor size", 0.5, 3.0, 0.1, toNum(r.cursor_size, 1),
          function (v) { return v.toFixed(1) + "×"; },
          function (v) { mutate(function (e) { e.render.cursor_size = v; }, { camera: false, skipPanel: true }); }));
      }
      p.appendChild(sect);
    }

    if (S.panel === "audio") {
      p.appendChild(el("h3", null, "Audio"));
      p.appendChild(el("div", "ed-hint",
        (S.details && S.details.has_audio) ?
          "This recording has a mic track. It is kept in the export." :
          "This recording has no mic track."));
      const sect = el("div", "sect");
      sect.appendChild(el("span", "t-cap sect-cap", "MUSIC"));
      sect.appendChild(rowText("Background music (file path)", r.music, "/path/to/track.mp3",
        function (v) { mutate(function (e) { e.render.music = v; }, { camera: false }); }));
      sect.appendChild(el("div", "ed-hint", "Ducked under the mic automatically; trimmed to the video length."));
      p.appendChild(sect);

      // Event SFX. The stored fields are tri-state (null = the built-in
      // sound, "off" = silent, anything else = a file path), so the switch
      // reads "not off" and writes null/"off" -- a custom path is a
      // separate, secondary control that only appears once the kind is on.
      const sfx = el("div", "sect");
      sfx.appendChild(el("span", "t-cap sect-cap", "SOUND EFFECTS"));
      const clickOn = !isOff(r.click_sound);
      const keyOn = !isOff(r.key_sound);
      const keys = S.details ? (S.details.key_count || 0) : 0;
      const clickPath = rememberSfxPath("click_sound", r.click_sound);
      sfx.appendChild(rowSwitch("Click sounds", clickOn, function (on) {
        const next = sfxToggleValue("click_sound", on);
        mutate(function (e) { e.render.click_sound = next; },
               { camera: false });
      }, false, "A click at every recorded mouse press, and a softer one when it comes back up."));
      if (clickOn) {
        sfx.appendChild(rowText("Custom click sound (file path)", clickPath, "built-in",
          function (v) {
            sfxPathMemo.click_sound = v || "";
            mutate(function (e) { e.render.click_sound = v; }, { camera: false });
          }));
      }
      const keyPath = rememberSfxPath("key_sound", r.key_sound);
      sfx.appendChild(rowSwitch("Keyboard sounds", keyOn && keys > 0, function (on) {
        const next = sfxToggleValue("key_sound", on);
        mutate(function (e) { e.render.key_sound = next; },
               { camera: false });
      }, keys === 0, keys > 0 ?
        "A keystroke at every recorded typing tick (" + keys + " in this take)." :
        "This recording has no key activity, so there is nothing to sound out."));
      if (keyOn && keys > 0) {
        sfx.appendChild(rowText("Custom key sound (file path)", keyPath, "built-in",
          function (v) {
            sfxPathMemo.key_sound = v || "";
            mutate(function (e) { e.render.key_sound = v; }, { camera: false });
          }));
      }
      if (clickOn || (keyOn && keys > 0)) {
        sfx.appendChild(rowSlider("Volume", 0, 2, 0.05, toNum(r.sfx_volume, 1),
          function (v) { return Math.round(v * 100) + "%"; },
          function (v) { mutate(function (e) { e.render.sfx_volume = v; }, { camera: false, skipPanel: true }); }));
      }
      p.appendChild(sfx);
    }

    if (S.panel === "transcript") {
      p.appendChild(el("h3", null, "Transcript"));
      renderTranscriptPanel(p);
    }

    if (S.panel === "facecam") {
      p.appendChild(el("h3", null, "Facecam"));
      const hasFace = !!(S.details && S.details.has_face);
      if (!hasFace) {
        p.appendChild(el("div", "ed-hint",
          "This recording has no webcam track. Pick a camera in the recording "
          + "bar before recording (or run record --face) to add a facecam bubble."));
      } else {
        p.appendChild(el("div", "ed-hint",
          "A webcam bubble composited over the video."));
        p.appendChild(rowSwitch("Show facecam", r.facecam !== false, function (on) {
          mutate(function (e) { e.render.facecam = on; }, { camera: false, skipPanel: true });
        }));
        // corner arrows map to the screen corner the bubble sits in
        p.appendChild(rowSeg("Position", [
          { v: "top-left", label: "↖" }, { v: "top-right", label: "↗" },
          { v: "bottom-left", label: "↙" }, { v: "bottom-right", label: "↘" }],
          r.facecam_position || "bottom-left", function (v) {
            mutate(function (e) { e.render.facecam_position = v; }, { camera: false, skipPanel: true });
          }));
        p.appendChild(rowSlider("Size", 0.08, 0.5, 0.01, toNum(r.facecam_size, 0.20),
          function (v) { return Math.round(v * 100) + "%"; },
          function (v) { mutate(function (e) { e.render.facecam_size = v; }, { camera: false, skipPanel: true }); }));
        p.appendChild(rowSeg("Shape", [
          { v: "circle", label: "Circle" }, { v: "rounded", label: "Rounded" }],
          r.facecam_shape || "circle", function (v) {
            mutate(function (e) { e.render.facecam_shape = v; }, { camera: false, skipPanel: true });
          }));
        // Background blur: keeps the face sharp, softens toward the rim.
        p.appendChild(rowSlider("Background blur", 0, 1, 0.05, toNum(r.facecam_blur, 0),
          function (v) { return v <= 0 ? "Off" : Math.round(v * 100) + "%"; },
          function (v) { mutate(function (e) { e.render.facecam_blur = v; }, { camera: false, skipPanel: true }); }));
      }
    }
  }

  /* ============================ transcript ============================ */

  /* The transcript is a READ MODEL — no edit is ever stored in it. Selecting
     words WRITES A CUT, and it writes it to `edits.cuts` like any other edit,
     so it round-trips through undo/redo, the rev CAS and the export exactly
     as a zoom does. What the transcript contributes is only the MAPPING from
     words to a time range; the range is the edit.

     Speech-to-text runs on the server as a job (a real ASR pass over the
     whole take), so this polls while it runs rather than blocking the panel.

     Word ends come from the SERVER (`w.end`, transcribe.word_end_times) —
     whisper collapses roughly half of all word timings to zero length, and
     the repair for that is one rule that lives in Python. Never recompute a
     word's end here; `t + dur` is wrong often enough to be the common case,
     and it silently leaves the last selected word in the video. */

  function loadTranscript() {
    if (!S.session) return Promise.resolve();
    return api("/api/transcript/" + encodeURIComponent(S.session)).then(
      function (data) { S.transcript = data; },
      function (err) {
        S.transcript = { status: "error", reason: err.message, words: [],
                         segments: [], job: {} };
      }
    ).then(function () {
      if (S.panel === "transcript") renderPanel();
      const job = (S.transcript && S.transcript.job) || {};
      clearTimeout(S.transcriptPoll);
      if (job.status === "running") {
        S.transcriptPoll = setTimeout(loadTranscript, 1500);
      }
    });
  }

  function startTranscribe(force) {
    S.transcript = Object.assign({}, S.transcript || {},
      { job: { status: "running", message: "transcribing" } });
    renderPanel();
    return api("/api/transcript/" + encodeURIComponent(S.session),
               { body: { force: !!force } })
      .then(function () { return loadTranscript(); },
            function (err) {
              S.transcript = { status: "error", reason: err.message, words: [],
                               segments: [], job: {} };
              renderPanel();
            });
  }

  /* ---- word selection -> a time range ---- */

  function txWords() {
    return (S.transcript && S.transcript.words) || [];
  }

  /* One entry per transcript line: the segment plus the GLOBAL word index
     range it covers. Words are matched to lines by time (neither list
     carries the other's index), and a word that falls in no segment joins
     the last line that started before it — so every word is reachable and
     the indices stay contiguous. */
  function txLines() {
    const words = txWords();
    const segs = (S.transcript && S.transcript.segments) || [];
    if (!segs.length || !words.length) return [];
    const lines = segs.map(function (seg) {
      return { seg: seg, first: -1, last: -1 };
    });
    let li = 0;
    for (let i = 0; i < words.length; i++) {
      const t = +words[i].t || 0;
      while (li + 1 < lines.length && t >= (+lines[li + 1].seg.t || 0) - 1e-6) li++;
      if (lines[li].first < 0) lines[li].first = i;
      lines[li].last = i;
    }
    return lines.filter(function (l) { return l.first >= 0; });
  }

  function txWordEnd(w) {
    // The server's repaired end. The fallback is only for a transcript
    // cached by an older server; it is deliberately the naive rule so the
    // difference shows up as a short cut rather than a wrong one.
    if (w && w.end !== undefined && w.end !== null) return +w.end;
    return (+((w || {}).t) || 0) + (+((w || {}).dur) || 0);
  }

  /* The time range a word selection removes. */
  function txSpan(sel) {
    const words = txWords();
    if (!sel) return null;
    const a = Math.min(sel.a, sel.b);
    const b = Math.max(sel.a, sel.b);
    if (!words[a] || !words[b]) return null;
    const start = +words[a].t || 0;
    const end = txWordEnd(words[b]);
    return end > start ? { start: start, end: end } : null;
  }

  /* Every word index the span actually removes — which can be MORE than the
     user highlighted, because whisper gives a run of words one identical
     timestamp and no rule can separate them. The panel paints these, so the
     over-reach is visible BEFORE the click rather than surprising after. */
  function txEffective(span) {
    const words = txWords();
    const out = [];
    if (!span) return out;
    for (let i = 0; i < words.length; i++) {
      const t = +words[i].t || 0;
      if (t >= span.start - 1e-6 && t < span.end - 1e-6) out.push(i);
    }
    return out;
  }

  /* Whisper sometimes stamps a long run of words — occasionally a whole
     hallucinated repetition — onto ONE instant. Measured on a real take:
     46% of words have an end that reaches past the next word's start, and
     one such run piles 90 words onto a 0.02s span.

     There is no rule that recovers the missing timing, so a cut built from
     such a selection removes two hundredths of a second and leaves every
     one of those words audible. That reads as the feature being broken.
     Detect it instead and say what is actually wrong: a span far too short
     to hold the words it covers is timing we cannot use.

     Deliberately narrow — a single well-timed short word has effCount 1 and
     is never degenerate, whatever its length. */
  const TX_DEGENERATE_SEC = 0.12;
  const TX_DEGENERATE_WORDS = 3;

  function txIsDegenerate(span, effCount) {
    if (!span) return false;
    return effCount >= TX_DEGENERATE_WORDS &&
      (span.end - span.start) < TX_DEGENERATE_SEC;
  }

  /* Takes the union list rather than calling cutRanges() itself: this runs
     once per word per repaint, and a repaint happens on every pointermove
     of a drag.

     Struck means "this leaves the export", so on a take whose export
     ignores cuts nothing is struck — it would be a lie, and the action
     bar says why instead. */
  function txTimeIsCut(t, cuts) {
    if (!cutsApplyToThisTake()) return false;
    for (let k = 0; k < cuts.length; k++) {
      if (t >= cuts[k].start - 1e-6 && t < cuts[k].end - 1e-6) return true;
    }
    return false;
  }

  /* THE single client-side cut authoring site. Every future proposer (a
     word-aligned lane, a drag on the clip lane) funnels through here, so
     the id policy, the sort and the write path are decided once. */
  function proposeCut(start, end) {
    if (!(end > start)) return null;
    const id = tempId("cut");
    mutate(function (e) {
      if (!e.cuts) e.cuts = [];
      e.cuts.push({ id: id, start: start, end: end });
      e.cuts.sort(function (a, b) { return a.start - b.start; });
    });
    return id;
  }

  /* Put back exactly this range — SUBTRACT it from the cuts rather than
     dropping every cut it touches.

     The difference is the whole promise of the button: a phrase selected
     inside a 30s cut authored by an agent must restore that phrase, not
     silently undo the agent's half-minute. A partially covered cut keeps
     its leading remainder (and its id, so `remove_cut` still addresses it)
     and grows a new trailing entry; only fully covered cuts disappear. */
  function restoreRange(start, end) {
    mutate(function (e) {
      const out = [];
      (e.cuts || []).forEach(function (c) {
        const cs = +c.start || 0;
        const ce = +c.end || 0;
        if (!(ce > start + 1e-6 && cs < end - 1e-6)) { out.push(c); return; }
        if (cs < start - 1e-6) {
          out.push(Object.assign({}, c, { start: cs, end: start }));
        }
        if (ce > end + 1e-6) {
          out.push({ id: tempId("cut"), start: end, end: ce });
        }
      });
      out.sort(function (a, b) { return a.start - b.start; });
      e.cuts = out;
    });
  }

  function txSetSel(a, b) {
    S.txSel = { a: a, b: b };
    paintTxSelection();
  }

  function txClearSel() {
    if (!S.txSel) return;
    S.txSel = null;
    paintTxSelection();
  }

  /* Repaint selection state WITHOUT rebuilding the list — the same reason
     highlightTranscript() exists. A drag across a long take would otherwise
     rebuild hundreds of rows on every pointermove. */
  function paintTxSelection() {
    const list = ui.panel.querySelector(".ed-tx-list");
    const bar = ui.panel.querySelector(".ed-tx-act");
    if (!list) return;
    const sel = S.txSel;
    const span = txSpan(sel);
    const a = sel ? Math.min(sel.a, sel.b) : -1;
    const b = sel ? Math.max(sel.a, sel.b) : -1;
    const eff = {};
    const effIdx = txEffective(span);
    effIdx.forEach(function (i) { eff[i] = true; });

    // Both hoisted out of the per-word loop: this runs on every
    // pointermove of a drag, over every word on screen.
    const words = txWords();
    const cuts = cutRanges();
    const nodes = list.querySelectorAll(".ed-tx-w");
    for (let k = 0; k < nodes.length; k++) {
      const i = +nodes[k].dataset.i;
      const chosen = sel && i >= a && i <= b;
      nodes[k].classList.toggle("sel", !!chosen);
      nodes[k].classList.toggle("pending", !!eff[i] && !chosen);
      nodes[k].classList.toggle(
        "cut", words[i] ? txTimeIsCut(+words[i].t || 0, cuts) : false);
    }
    if (bar) paintTxActions(bar, span, effIdx.length, b - a + 1);
  }

  function paintTxActions(bar, span, effCount, selCount) {
    bar.innerHTML = "";
    if (!span) {
      bar.classList.remove("on");
      return;
    }
    bar.classList.add("on");
    const applies = cutsApplyToThisTake();
    const cuts = cutRanges();
    let covered = 0;
    cuts.forEach(function (c) {
      const lo = Math.max(c.start, span.start);
      const hi = Math.min(c.end, span.end);
      if (hi > lo) covered += hi - lo;
    });
    const isCut = covered >= (span.end - span.start) - 1e-3;

    const degenerate = txIsDegenerate(span, effCount);

    const btn = el("button", isCut ? "btn-quiet" : "btn-danger",
                   isCut ? "Restore to video" : "Remove from video");
    btn.type = "button";
    btn.disabled = !applies || (degenerate && !isCut);
    btn.title = isCut
      ? "Puts this speech, and the video under it, back in the export"
      : "Removes this speech, and the video under it, from the export";
    btn.addEventListener("click", function () {
      if (isCut) restoreRange(span.start, span.end);
      else txCutSelection();
    });
    bar.appendChild(btn);

    const info = el("div", "ed-tx-act-info");
    info.appendChild(el("span", null,
      fmtDuration(span.end - span.start) + " / " + effCount +
      (effCount === 1 ? " word" : " words")));
    if (!applies) {
      info.appendChild(el("span", "warn",
        (S.edits.windows || []).length
          ? "Cuts don't apply while this take is laid out as window cards "
            + "until you remove the cards."
          : "Cuts don't apply to this take type yet. It exports uncut."));
    } else if (degenerate) {
      // Refusing rather than performing a cut that cannot do what the user
      // is asking. The alternative — remove the 0.02s the recogniser
      // actually timed — leaves every one of these words audible, which
      // reads as the feature being broken rather than the timing being bad.
      info.appendChild(el("span", "warn",
        "The transcript gives these " + effCount + " words one shared "
        + "timestamp, so there is no way to tell where their audio is. "
        + "They can't be removed from here."));
    } else if (effCount > selCount) {
      // Not a rounding artefact and not our choice: these words carry one
      // identical timestamp from the recogniser, so they can only leave
      // together. Say it in words, because a tint alone asks the user to
      // infer a data limitation.
      info.appendChild(el("span", "warn",
        "The highlighted words share a timestamp with " +
        (effCount - selCount) + " more, so they can only be removed together."));
    }
    bar.appendChild(info);
  }

  function txCutSelection() {
    const span = txSpan(S.txSel);
    if (!span || !cutsApplyToThisTake()) return;
    // Also guarded here, not just on the button: Delete/Backspace reaches
    // this without passing the disabled control.
    if (txIsDegenerate(span, txEffective(span).length)) return;
    // Report what the EXPORT loses, not what was selected. A selection
    // reaching outside the trim window, or overlapping a cut that already
    // exists, removes less than its own length — and this number is the
    // one the user acts on, so it must not be the optimistic one.
    const before = trimmedLength();
    proposeCut(span.start, span.end);
    // The selection deliberately SURVIVES the cut: the action bar flips to
    // "Restore to video", so the mistake anyone makes on their first try is
    // one click to undo without knowing what ⌘Z is.
    const after = trimmedLength();
    if (before - after < 0.02) {
      toast("Those words are already outside the clip. The export is unchanged.");
      return;
    }
    toast("Removed " + fmtDuration(before - after) +
          ". Export is now " + fmtDuration(after));
  }

  /* Export length as the clip label computes it, so the toast and the label
     can never disagree. */
  function trimmedLength() {
    const tr = trimRange();
    let cut = 0;
    if (cutsApplyToThisTake()) {
      cutRanges().forEach(function (c) {
        const lo = Math.max(c.start, tr.start);
        const hi = Math.min(c.end, tr.end);
        if (hi > lo) cut += hi - lo;
      });
    }
    return Math.max(0, tr.end - tr.start - cut);
  }

  /* Registered ONCE, not per panel render (renderPanel rebuilds the panel on
     every mutate, and a listener per render is a leak). On window rather
     than on the list, so a drag that ends outside the panel still ends —
     otherwise the next hover keeps extending the selection. */
  window.addEventListener("pointerup", function () { S.txDrag = false; });
  window.addEventListener("pointercancel", function () { S.txDrag = false; });

  function txMoveSel(delta, extend) {
    const words = txWords();
    if (!words.length) return;
    const sel = S.txSel;
    if (!sel) {
      txSetSel(0, 0);
      seek(+words[0].t || 0);
      return;
    }
    const next = clamp(sel.b + delta, 0, words.length - 1);
    txSetSel(extend ? sel.a : next, next);
    seek(+words[next].t || 0);
  }

  function renderTranscriptPanel(p) {
    const tx = S.transcript;
    if (!tx) {
      p.appendChild(el("div", "ed-hint", "Loading..."));
      loadTranscript();
      return;
    }
    if (!tx.has_audio) {
      p.appendChild(el("div", "ed-hint",
        "This recording has no mic track, so there is nothing to transcribe. "
        + "Record with a microphone to get a transcript."));
      return;
    }
    const job = tx.job || {};
    const running = job.status === "running";
    const lines = tx.segments || [];
    const words = tx.words || [];

    if (running) {
      p.appendChild(el("div", "ed-hint",
        "Transcribing... this runs once per recording and can take a few "
        + "minutes on a long take."));
    } else if (tx.asr && tx.asr.available === false) {
      p.appendChild(el("div", "ed-hint", tx.asr.reason));
      return;
    } else if (!lines.length) {
      p.appendChild(el("div", "ed-hint",
        tx.reason || "No transcript yet. Speech-to-text runs locally and is "
        + "cached beside the recording."));
    }

    if (!running) {
      const btnRow = el("div", "ed-tx-actions");
      const btn = el("button", "btn-quiet",
                     lines.length ? "Re-transcribe" : "Transcribe");
      btn.addEventListener("click", function () {
        startTranscribe(lines.length > 0);
      });
      btnRow.appendChild(btn);
      p.appendChild(btnRow);
    }
    // Both lists are needed: segments give the lines, words give the
    // selectable units. A transcript with sentences but no word timings
    // (an older cache, a model run without -dtw) can still be read, so
    // fall back to seek-only lines rather than showing nothing.
    if (!lines.length) return;

    const searchRow = el("div", "ed-tx-search");
    const search = document.createElement("input");
    search.type = "search";
    search.placeholder = "Find a phrase...";
    search.value = S.transcriptQuery || "";
    search.addEventListener("input", function () {
      S.transcriptQuery = search.value;
      paintLines();
    });
    searchRow.appendChild(search);
    p.appendChild(searchRow);

    // ABOVE the list, not below it. The list is as tall as the panel allows
    // and the panel is clipped by the stage, so a bar underneath sits off
    // screen on any take long enough to scroll — which is every take worth
    // cutting. Here it appears directly under the search box, where the eye
    // already is, and cannot be pushed anywhere.
    const actions = el("div", "ed-tx-act");
    p.appendChild(actions);

    const list = el("div", "ed-tx-list");
    // The noun is a word range, so the browser's own text selection is
    // switched off — two selections fighting over the same words is worse
    // than one. tabindex makes the list a real focus target so the whole
    // gesture is reachable from the keyboard, and `touch-action` (CSS) keeps
    // a vertical flick scrolling instead of selecting.
    list.tabIndex = 0;
    p.appendChild(list);

    // One row per segment, each carrying its own word index range. Empty
    // when the transcript has no word timings — see the fallback below.
    const rows = txLines();
    const selectable = rows.length > 0;

    function paintLines() {
      const q = (S.transcriptQuery || "").trim().toLowerCase();
      list.innerHTML = "";
      let shown = 0;
      // Without word timings there is nothing to select, but the transcript
      // is still worth reading and seeking by — degrade to the pre-feature
      // behavior rather than to a blank panel.
      const src = selectable
        ? rows
        : lines.map(function (seg) { return { seg: seg, first: -1, last: -2 }; });
      src.forEach(function (line) {
        const seg = line.seg;
        if (q && (seg.text || "").toLowerCase().indexOf(q) < 0) return;
        shown++;
        const row = el("div", "ed-tx-line");
        row.dataset.t = seg.t;
        row.dataset.end = seg.t + (seg.dur || 0);
        row.appendChild(el("span", "ed-tx-t t-mono", fmtTime(seg.t, false)));
        const text = el("span", "ed-tx-text");
        if (!selectable) {
          text.textContent = seg.text || "";
          row.addEventListener("click", function () { seek(seg.t); });
        }
        for (let i = line.first; i <= line.last; i++) {
          const w = words[i];
          if (!w) continue;
          const span = el("span", "ed-tx-w", w.text || "");
          span.dataset.i = i;
          text.appendChild(span);
          if (i < line.last) text.appendChild(document.createTextNode(" "));
        }
        row.appendChild(text);
        list.appendChild(row);
      });
      if (!shown) list.appendChild(el("div", "ed-hint", "No match."));
      // renderPanel() rebuilds this list on EVERY mutate and every adopted
      // server doc, so a fresh list would start at 0:00 each time. Cutting
      // one filler word at 8:00 would then scroll the take back to the top
      // — and cutting filler words is the whole feature.
      //
      // The offsetHeight read is load-bearing, not a tic: the list was
      // just rebuilt, and a scrollTop written before layout has run clamps
      // straight back to 0. Reading a layout property flushes it, and does
      // so synchronously, so the list never paints at the top and jumps.
      void list.offsetHeight;
      list.scrollTop = S.txScroll || 0;
      highlightTranscript();
      paintTxSelection();
    }

    // Keyboard flow across a rebuild: Delete cuts, the panel is rebuilt,
    // and the node that had focus is gone — so the next arrow key would
    // reach nothing. Restore focus only if this list is what was focused.
    list.addEventListener("focus", function () { S.txFocus = true; });
    list.addEventListener("blur", function () { S.txFocus = false; });
    if (S.txFocus) list.focus({ preventScroll: true });

    function wordAt(target) {
      const node = target && target.closest ? target.closest(".ed-tx-w") : null;
      return node ? +node.dataset.i : -1;
    }

    list.addEventListener("pointerdown", function (ev) {
      const i = wordAt(ev.target);
      if (i < 0) return;
      // Own the gesture: without this the pointer drag also starts a native
      // selection, and the user sees two disagreeing highlights.
      ev.preventDefault();
      // preventScroll: focusing a scrollable element otherwise scrolls it
      // into view, so the first click on a word jumps the panel out from
      // under the pointer.
      list.focus({ preventScroll: true });
      if (ev.shiftKey && S.txSel) txSetSel(S.txSel.a, i);
      else txSetSel(i, i);
      S.txDrag = true;
      seek(+(words[i] || {}).t || 0);
    });

    list.addEventListener("pointermove", function (ev) {
      if (!S.txDrag) return;
      const i = wordAt(ev.target);
      if (i >= 0 && S.txSel && i !== S.txSel.b) txSetSel(S.txSel.a, i);
    });

    list.addEventListener("keydown", function (ev) {
      // Every key this list claims must ALSO stopPropagation. The document
      // handler ignores INPUT/SELECT/TEXTAREA, and this is a focusable
      // `div` — so without it Delete cuts the speech AND runs the global
      // deleteSelection(), destroying whatever zoom or marker happened to
      // be selected on the timeline, in a second undo entry the user never
      // connects to what they pressed. Arrows double-handle the same way
      // and walk the playhead a full second off the word.
      //
      // Space is play/pause everywhere in this editor and is deliberately
      // NOT claimed here, so it keeps bubbling.
      if (ev.key === "ArrowRight" || ev.key === "ArrowLeft") {
        ev.preventDefault();
        ev.stopPropagation();
        txMoveSel(ev.key === "ArrowRight" ? 1 : -1, ev.shiftKey);
      } else if (ev.key === "Escape") {
        ev.stopPropagation();
        txClearSel();
      } else if (ev.key === "Backspace" || ev.key === "Delete") {
        // Claimed whether or not there is a selection: the list has focus,
        // so these keys are ours, and letting an empty selection fall
        // through would delete a timeline item instead.
        ev.preventDefault();
        ev.stopPropagation();
        if (S.txSel) txCutSelection();
      }
    });

    paintLines();
    p.appendChild(el("div", "ed-hint",
      (selectable
        ? "Drag across words to select them, then remove them from the video. "
        : "Click a line to jump there. ")
      + (tx.model ? "Model: " + tx.model : "")));
  }

  /* Which line the playhead is inside -- updated on every seek, without
     rebuilding the list (it can be hundreds of rows on a long take). */
  function highlightTranscript() {
    if (S.panel !== "transcript") return;
    const rows = ui.panel.querySelectorAll(".ed-tx-line");
    for (let i = 0; i < rows.length; i++) {
      const t0 = parseFloat(rows[i].dataset.t);
      const t1 = parseFloat(rows[i].dataset.end);
      rows[i].classList.toggle("on", S.playhead >= t0 && S.playhead < t1);
    }
  }

  function isPresetId(spec) {
    for (let i = 0; i < S.backgrounds.length; i++) {
      if (S.backgrounds[i].id === spec) return true;
    }
    return false;
  }

  /* ============================ marker popover ============================ */

  function markerChapter(m) {
    const tr = trimRange();
    if (m.time > tr.end - 1e-6) return null;
    const start = Math.max(tr.start, m.time);
    let end = tr.end;
    S.edits.markers.forEach(function (o) {
      if (o.time > m.time + 1e-6 && o.time < end) end = o.time;
    });
    end = Math.min(end, tr.end);
    if (end - start < 0.01) return null;
    return { start: start, end: end };
  }

  function openMarkerPop() {
    const m = S.sel && S.sel.kind === "marker" ? findItem("marker", S.sel.id) : null;
    if (!m) return;
    ui.mpLabel.value = m.label || "";
    ui.mpTime.value = Math.round(m.time * 100) / 100;
    const ch = markerChapter(m);
    ui.mpSummary.textContent = ch ?
      "Chapter: " + fmtTime(ch.start) + " - " + fmtTime(ch.end) + " (" + fmtDuration(ch.end - ch.start) + ")" :
      "Marker is outside the current trim window.";
    ui.mpTrim.disabled = !ch;
    ui.markerPop.hidden = false;
    positionMarkerPop();
    seek(clamp(m.time, trimRange().start, trimRange().end));
  }

  function positionMarkerPop() {
    const m = S.sel && S.sel.kind === "marker" ? findItem("marker", S.sel.id) : null;
    if (!m || ui.markerPop.hidden) return;
    const laneRect = ui.markerLane.getBoundingClientRect();
    const x = laneRect.left + (m.time / duration()) * laneRect.width;
    const popW = 250;
    ui.markerPop.style.left = clamp(x - popW / 2, 8, window.innerWidth - popW - 8) + "px";
    ui.markerPop.style.top = (laneRect.top - ui.markerPop.offsetHeight - 10) + "px";
  }

  function closeMarkerPop() {
    ui.markerPop.hidden = true;
  }

  ui.mpLabel.addEventListener("change", function () {
    const sel = S.sel;
    if (!sel || sel.kind !== "marker") return;
    mutate(function (e) {
      const m = e.markers.find(function (x) { return x.id === sel.id; });
      if (m) m.label = ui.mpLabel.value.trim() || null;
    }, { camera: false, still: false });
  });
  ui.mpTime.addEventListener("change", function () {
    const sel = S.sel;
    const v = parseFloat(ui.mpTime.value);
    if (!sel || sel.kind !== "marker" || !Number.isFinite(v)) return;
    mutate(function (e) {
      const m = e.markers.find(function (x) { return x.id === sel.id; });
      if (m) m.time = clamp(v, 0, duration());
    }, { camera: false, still: false });
    positionMarkerPop();
  });
  ui.mpTrim.addEventListener("click", function () {
    const m = S.sel && S.sel.kind === "marker" ? findItem("marker", S.sel.id) : null;
    if (!m) return;
    const ch = markerChapter(m);
    if (!ch) return;
    mutate(function (e) {
      e.trim.start = ch.start;
      e.trim.end = ch.end >= duration() - 0.02 ? null : ch.end;
    });
    seek(ch.start);
    openMarkerPop();
    toast("Trimmed to chapter");
  });
  ui.mpDelete.addEventListener("click", deleteSelection);

  document.addEventListener("pointerdown", function (ev) {
    if (ui.markerPop.hidden) return;
    if (ev.target.closest && (ev.target.closest("#ed-marker-pop") || ev.target.closest(".ed-marker"))) return;
    closeMarkerPop();
    if (S.sel && S.sel.kind === "marker") { S.sel = null; renderTimeline(); }
  });

  /* ============================ presets / title / top bar ============================ */

  function renderPresets() {
    ui.preset.innerHTML = "";
    const presets = S.edits.presets || [];
    presets.forEach(function (pr) {
      const o = el("option", null, pr.name);
      o.value = pr.id;
      ui.preset.appendChild(o);
    });
    ui.preset.value = S.edits.active_preset_id || (presets[0] && presets[0].id) || "";
    ui.preset._sig = presets.map(function (p) { return p.id + ":" + p.name; }).join("|");
  }

  /* Keep the preset dropdown tracking S.edits (undo/redo can change the
     active preset without going through the select's own handler). */
  function syncPresetSelect() {
    if (!S.edits) return;
    const presets = S.edits.presets || [];
    const sig = presets.map(function (p) { return p.id + ":" + p.name; }).join("|");
    if (ui.preset._sig !== sig) renderPresets();
    else if (ui.preset.value !== (S.edits.active_preset_id || "")) {
      ui.preset.value = S.edits.active_preset_id || "";
    }
  }

  ui.preset.addEventListener("change", async function () {
    const id = ui.preset.value;
    scheduleSave.cancel();
    try {
      await saveNow();
      const data = await api("/api/presets/" + enc(S.session) + "/activate", { body: { preset_id: id } });
      pushUndo();
      S.edits = data.edits;
      S.mutCount++;
      afterMutate({});
      renderPresets();
      toast("Preset applied");
    } catch (e) {
      toast(e.message, "err");
      renderPresets();
    }
  });

  ui.presetDup.addEventListener("click", async function () {
    const active = (S.edits.presets || []).find(function (p) { return p.id === S.edits.active_preset_id; });
    const name = window.prompt("New preset name:", (active ? active.name : "Preset") + " Copy");
    if (!name) return;
    try {
      const data = await api("/api/presets/" + enc(S.session) + "/duplicate", {
        body: { name: name, source_preset_id: S.edits.active_preset_id, source_render: S.edits.render },
      });
      S.edits = data.edits;
      S.mutCount++;
      renderPresets();
      renderPanel();
      toast("Preset saved");
    } catch (e) {
      toast(e.message, "err");
    }
  });

  function renderTitle() {
    ui.title.textContent = S.name || fmtSessionDate(S.session) || S.session;
    document.title = (S.name || S.session) + " - AutoCine";
  }

  ui.title.addEventListener("click", function () {
    if (ui.title.querySelector("input")) return;
    const input = document.createElement("input");
    input.type = "text";
    input.value = S.name || "";
    input.placeholder = fmtSessionDate(S.session) || S.session;
    ui.title.textContent = "";
    ui.title.appendChild(input);
    input.focus();
    input.select();
    const commit = async function () {
      const v = input.value.trim() || null;
      try {
        await api("/api/project/" + enc(S.session), { body: { name: v } });
        S.name = v;
      } catch (e) { toast(e.message, "err"); }
      renderTitle();
    };
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter") input.blur();
      if (ev.key === "Escape") { input.value = S.name || ""; input.blur(); }
      ev.stopPropagation();
    });
    input.addEventListener("blur", commit);
  });

  ui.undo.innerHTML = ICONS.undo;
  ui.redo.innerHTML = ICONS.redo;
  ui.undo.addEventListener("click", undo);
  ui.redo.addEventListener("click", redo);
  ui.reveal.addEventListener("click", function () {
    api("/api/reveal/" + enc(S.session), { body: {} }).catch(function (e) { toast(e.message, "err"); });
  });

  // "New Recording" reopens the floating pill (native, popup fallback) so the
  // bar <-> studio loop closes without going back to the Library first.
  ui.newRecording.addEventListener("click", function () { openRecorderBar(); });

  ui.aspect.addEventListener("change", function () {
    mutate(function (e) { e.render.aspect = ui.aspect.value; });
  });

  function applyAspectSelect() {
    if (!S.edits) return;
    const v = (S.edits.render.aspect || "auto");
    if (ui.aspect.value !== v) {
      const known = Array.prototype.some.call(ui.aspect.options, function (o) { return o.value === v; });
      if (!known) {
        const o = el("option", null, v);
        o.value = v;
        ui.aspect.appendChild(o);
      }
      ui.aspect.value = v;
    }
  }

  ui.resolution.addEventListener("change", function () {
    mutate(function (e) { e.render.resolution = ui.resolution.value; });
  });

  function applyResolutionSelect() {
    if (!S.edits) return;
    const v = (S.edits.render.resolution || "auto");
    const known = Array.prototype.some.call(ui.resolution.options,
      function (o) { return o.value === v; });
    ui.resolution.value = known ? v : "auto";
  }

  /* ============================ export modal ============================ */

  let exFormat = "mp4";

  function openExport() {
    exFormat = S.edits.render.gif ? "mp4gif" : "mp4";
    ui.exFormat.querySelectorAll("button").forEach(function (b) {
      b.classList.toggle("on", b.getAttribute("data-v") === exFormat);
    });
    ui.exGifOpts.hidden = exFormat !== "mp4gif";
    ui.exGifFps.value = toNum(S.edits.render.gif_fps, 15);
    ui.exGifWidth.value = toNum(S.edits.render.gif_width, 1000);
    ui.exStatus.hidden = true;
    ui.exReveal.hidden = true;
    ui.exStart.disabled = false;
    ui.exScrim.hidden = false;
    pollExportState();
  }

  function closeExport() {
    ui.exScrim.hidden = true;
    if (S.exportPoll) { clearTimeout(S.exportPoll); S.exportPoll = null; }
  }

  ui.exFormat.querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () {
      exFormat = b.getAttribute("data-v");
      ui.exFormat.querySelectorAll("button").forEach(function (x) { x.classList.toggle("on", x === b); });
      ui.exGifOpts.hidden = exFormat !== "mp4gif";
    });
  });

  ui.exportOpen.addEventListener("click", openExport);
  ui.exCancel.addEventListener("click", closeExport);
  ui.exScrim.addEventListener("click", function (ev) {
    if (ev.target === ui.exScrim && !S.exporting) closeExport();
  });

  ui.exStart.addEventListener("click", async function () {
    mutate(function (e) {
      e.render.gif = exFormat === "mp4gif";
      e.render.gif_fps = clamp(Math.round(toNum(ui.exGifFps.value, 15)), 1, 50);
      e.render.gif_width = clamp(Math.round(toNum(ui.exGifWidth.value, 1000)), 64, 4000);
    }, { camera: false, still: false });
    scheduleSave.cancel();
    try {
      await saveNow();
      await api("/api/render", { body: { session: S.session, options: fullOptions() } });
      S.exporting = true;
      ui.exStart.disabled = true;
      ui.exStatus.hidden = false;
      ui.exSpinner.className = "ex-spinner";
      ui.exMessage.textContent = "Rendering...";
      ui.exReveal.hidden = true;
      pollExportState();
    } catch (e) {
      ui.exStatus.hidden = false;
      ui.exSpinner.className = "ex-spinner err";
      ui.exMessage.textContent = e.message;
    }
  });

  ui.exReveal.addEventListener("click", function () {
    api("/api/reveal/" + enc(S.session), { body: {} }).catch(function (e) { toast(e.message, "err"); });
  });

  async function pollExportState() {
    if (S.exportPoll) clearTimeout(S.exportPoll);
    if (ui.exScrim.hidden) return;
    try {
      const s = await api("/api/state");
      const ren = s.render || {};
      // the server has one render slot — only claim status that is OURS
      const mine = ren.session === S.session;
      if (ren.status === "running" && mine) {
        S.exporting = true;
        ui.exStart.disabled = true;
        ui.exStatus.hidden = false;
        ui.exSpinner.className = "ex-spinner";
        ui.exMessage.textContent = ren.message || "Rendering...";
      } else if (ren.status === "running" && !mine) {
        ui.exStart.disabled = true;
        ui.exStatus.hidden = false;
        ui.exSpinner.className = "ex-spinner";
        ui.exMessage.textContent = "Another project is rendering. Waiting for it to finish...";
      } else if (ren.status === "done" && mine && S.exporting) {
        S.exporting = false;
        ui.exStart.disabled = false;
        ui.exSpinner.className = "ex-spinner done";
        ui.exMessage.textContent = "Done / " + (ren.out_path || "output.mp4");
        ui.exReveal.hidden = false;
      } else if (ren.status === "error" && mine && S.exporting) {
        S.exporting = false;
        ui.exStart.disabled = false;
        ui.exSpinner.className = "ex-spinner err";
        ui.exMessage.textContent = ren.message || "Render failed";
      } else if (!S.exporting && ren.status !== "running") {
        ui.exStart.disabled = false;
        if (ui.exSpinner.className === "ex-spinner" && !ui.exStatus.hidden &&
            ui.exMessage.textContent.indexOf("Another project") === 0) {
          ui.exStatus.hidden = true;   // foreign render finished; clear the note
        }
      }
    } catch (e) { /* transient */ }
    S.exportPoll = setTimeout(pollExportState, 1000);
  }

  /* ============================ transport & keyboard ============================ */

  ui.play.addEventListener("click", togglePlay);
  ui.skipBack.innerHTML = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 3.2h1.8v9.6H4zM13 3.6v8.8c0 .7-.8 1.1-1.4.7L6.4 8.9a.85.85 0 010-1.4l5.2-4.2c.6-.5 1.4 0 1.4.7z"/></svg>';
  ui.skipFwd.innerHTML = '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 3.2h1.8v9.6H4zM13 3.6v8.8c0 .7-.8 1.1-1.4.7L6.4 8.9a.85.85 0 010-1.4l5.2-4.2c.6-.5 1.4 0 1.4.7z" transform="scale(-1,1) translate(-17,0)"/></svg>';
  ui.skipBack.addEventListener("click", function () { if (S.playing) pause(); seek(trimRange().start); });
  ui.skipFwd.addEventListener("click", function () { if (S.playing) pause(); seek(trimRange().end); });
  ui.addZoom.addEventListener("click", function () { addZoomAt(S.playhead); });
  ui.addSuppress.addEventListener("click", function () { addSuppressAt(S.playhead); });
  ui.addMarker.addEventListener("click", function () { addMarkerAt(S.playhead); });
  setPlayIcon();

  document.addEventListener("keydown", function (ev) {
    const target = ev.target;
    if (target && (target.tagName === "INPUT" || target.tagName === "SELECT" ||
        target.tagName === "TEXTAREA" || target.isContentEditable)) return;
    const meta = ev.metaKey || ev.ctrlKey;

    if (meta && ev.key.toLowerCase() === "z") {
      ev.preventDefault();
      if (ev.shiftKey) redo(); else undo();
      return;
    }
    if (meta && ev.key.toLowerCase() === "e") {
      ev.preventDefault();
      openExport();
      return;
    }
    if (meta) return;

    switch (ev.key) {
      case " ":
        ev.preventDefault();
        togglePlay();
        break;
      case "ArrowLeft":
      case "ArrowRight": {
        ev.preventDefault();
        if (S.playing) pause();
        const fps = (S.details && S.details.fps) || 60;
        const step = ev.shiftKey ? 1 : 1 / fps;
        seek(S.playhead + (ev.key === "ArrowLeft" ? -step : step));
        break;
      }
      case "z": case "Z": addZoomAt(S.playhead); break;
      case "m": case "M": addMarkerAt(S.playhead); break;
      case "i": case "I":
        mutate(function (e) {
          e.trim.start = clamp(S.playhead, 0, trimRange().end - 0.05);
        });
        break;
      case "o": case "O":
        mutate(function (e) {
          const v = clamp(S.playhead, trimRange().start + 0.05, duration());
          e.trim.end = v >= duration() - 0.02 ? null : v;
        });
        break;
      case "Backspace":
      case "Delete":
        ev.preventDefault();
        deleteSelection();
        break;
      case "Escape":
        if (!ui.exScrim.hidden) { if (!S.exporting) closeExport(); }
        else if (S.pinArm) disarmPin();
        else if (S.winPick) disarmWinPick();
        else if (S.cropArm) disarmCrop();
        else if (S.cardSel != null) deselectCard();
        else if (S.sel) deselect();
        break;
      default: break;
    }
  });

  /* ============================ boot ============================ */

  function showFatal(msg) {
    ui.canvasMsg.hidden = false;
    ui.canvasMsg.textContent = msg;
  }

  async function boot() {
    if (!S.session) {
      showFatal("No session specified. Open a project from the library.");
      return;
    }
    let detail;
    try {
      const data = await api("/api/sessions/" + enc(S.session));
      detail = data.session;
    } catch (e) {
      showFatal("Could not open this project: " + e.message);
      return;
    }
    S.details = detail;
    S.edits = detail.edits;
    S.rev = (detail.edits && detail.edits.rev) || 0;
    S.name = detail.name || null;

    renderTitle();
    renderPresets();
    applyAspectSelect();
    applyResolutionSelect();
    updateStageNote();
    updateCaptureUI();

    // A multi-native take has NO raw.mov -- each window is its own buffer, so
    // asking for one is a guaranteed 404 and a dead player. Channel 0 stands
    // in as the master clock (play/seek/playbackTick all read ui.video) and
    // the remaining channels follow it; see setupChannelVideos.
    const isNative = !!(S.details && S.details.multi_native);
    const isScene = !!(S.details && S.details.scene_take);
    if (isScene) {
      // Scene take (the window set changes mid-recording): there is no single
      // raw.mov to play and the channel COUNT changes mid-timeline. ui.video
      // stays srcless; the live fleet-ring player is wired up once the first
      // /api/camera-path scene plan lands (adoptScenePayload). Until then --
      // and if that ever fails -- `sceneStillOnly` keeps the server-composited
      // still (`scene_preview_frame`) as the preview and `play()` shows a hint.
      S.sceneTake = true;
      S.sceneStillOnly = true;
      if (!S.fleets) { S.fleets = new Map(); S.fleetLRU = []; }
    } else {
      ui.video.src = autocineUrl(
        "/api/media/" + enc(S.session)
        + (isNative ? "/channel0" : "/raw"));
      if (isNative) setupChannelVideos();
    }
    if (S.details && S.details.has_face) {
      // Muted: face.mov is a picture source here. The mic audio the viewer
      // should hear rides raw.mov, and a second unmuted element would either
      // double it or (with no audio track) do nothing but risk autoplay
      // blocking the element we need frames from.
      ui.face.muted = true;
      ui.face.src = autocineUrl(
        "/api/media/" + enc(S.session) + "/face");
      ui.face.addEventListener("loadedmetadata", function () {
        syncFace(S.playhead);
        applyTransformAt(S.playhead);
      });
      // A seek is asynchronous, but syncFace sets currentTime and drawMulti
      // paints in the SAME tick -- so a scrub would composite the frame the
      // element had *before* the seek and hold it until something else
      // redrew. Repaint when the new frame is actually there. (While playing
      // this is moot: the element is running, not seeking. Same reason
      // #ed-video has its own 'seeked' handler.)
      ui.face.addEventListener("seeked", function () {
        if (!S.playing && multiReady()) drawMulti();
      });
    }
    ui.video.addEventListener("loadedmetadata", function () {
      fitCanvas();
      applyTransformAt(S.playhead);
    });
    ui.video.addEventListener("seeked", function () {
      if (!S.playing) applyTransformAt(ui.video.currentTime - fleetLead());
    });
    // stay honest if the browser/OS pauses the media element out from under us
    ui.video.addEventListener("pause", function () {
      if (S.playing && !ui.video.ended) pause();
    });

    // parallel extras — each optional
    api("/api/backgrounds").then(function (d) {
      S.backgrounds = (d && d.presets) || [];
      renderPanel();
      applyFrameStyle();
    }).catch(function () {});
    api("/api/waveform/" + enc(S.session)).then(function (d) {
      S.waveform = d;
      drawWave();
    }).catch(function () {});

    fetchCamPath();

    const tr = trimRange();
    S.playhead = tr.start;
    setPanel("zoom");
    renderTimeline();
    fitCanvas();
    applyFrameStyle();
    // ?t=<seconds> -- the library's transcript search deep link: open AT that
    // moment, through the same seek() a transcript-line click uses (it also
    // clamps into the trim range, so a too-large t lands on the clip's end).
    // Missing, non-numeric or negative t is ignored: the take opens at its start.
    // Strictly a plain decimal number of seconds -- parseFloat alone would
    // read "12abc" as 12.
    const rawT = queryParam("t");
    const deepT = rawT !== null && /^\d+(\.\d+)?$/.test(rawT) ? Number(rawT) : NaN;
    seek(Number.isFinite(deepT) ? deepT : tr.start);

    new ResizeObserver(function () { fitCanvas(); }).observe(ui.canvasWrap);
    new ResizeObserver(function () { renderTimeline(); }).observe(ui.tl);
    setSaveState("Saved");

    // While idle (nothing unsaved, not dragging/playing), watch for external
    // saves — e.g. the MCP server editing this project — and adopt them live.
    setInterval(async function () {
      if (!S.edits || S.drag || S.cardDrag || S.txDrag || S.playing || S.exporting || S.pinArm) return;
      if (S.mutCount !== S.savedMutCount) return;
      try {
        const data = await api("/api/edits/" + enc(S.session));
        if (!data.edits || typeof data.edits.rev !== "number") return;
        // changed (or a card picked up) while fetching
        if (S.mutCount !== S.savedMutCount || S.drag || S.cardDrag || S.txDrag) return;
        if (data.edits.rev !== S.rev) {
          adoptEdits(data.edits);
          scheduleCamPath();
          staleStill();
          applyFrameStyle();
          applyAspectSelect();
          applyResolutionSelect();
          toast("Edits updated outside the editor");
        }
      } catch (e) { /* transient */ }
    }, 3000);
  }

  boot().catch(function (e) {
    showFatal("Editor failed to start: " + e.message);
  });
})();
