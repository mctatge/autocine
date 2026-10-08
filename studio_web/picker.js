/* Full-screen window picker: hover a window, it lights up; click to toggle.
 *
 * Runs in its own native window (studio_app._NativePickerApi). Everything it
 * knows about windows comes from GET /api/windows — the same front-to-back,
 * points-space list the recorder resolves picks against, so what lights up
 * here and what gets recorded cannot disagree.
 */
(function () {
  "use strict";

  var MAX = 4;
  var REFRESH_MS = 500;   // windows move and close while you're picking

  var canvas = document.getElementById("pk-canvas");
  var ctx = canvas.getContext("2d");
  var badges = document.getElementById("pk-badges");
  var hud = document.getElementById("pk-hud");
  var elTitle = document.getElementById("pk-title");
  var elCount = document.getElementById("pk-count");
  var elWarn = document.getElementById("pk-warn");
  var elWarnText = document.getElementById("pk-warn-text");
  var elArrangeRow = document.getElementById("pk-arrange-row");
  var elArrange = document.getElementById("pk-arrange");
  var elDone = document.getElementById("pk-done");
  var elClear = document.getElementById("pk-clear");

  var windows = [];          // front-to-back /api/windows entries
  var picked = [];           // window ids, in pick order == card order
  var hoverId = null;
  var scale = 1;             // CSS px per point; asserted to be 1 at boot
  var closed = false;

  function requestHeaders(initial) {
    var headers = Object.assign({}, initial || {});
    var node = document.querySelector('meta[name="autocine-token"]');
    var token = node ? node.getAttribute("content") || "" : "";
    if (token) headers["X-AutoCine-Token"] = token;
    return headers;
  }

  function bridge() {
    return (window.pywebview && window.pywebview.api)
      ? window.pywebview.api : null;
  }

  /* --- coordinate sanity -------------------------------------------------
   * A frameless pywebview window reports CSS px == points, and we're placed
   * at the whole screen frame, so a window rect in global top-left points is
   * a CSS position directly. That's an assumption about someone else's
   * window manager, so check it rather than trust it: a picker that
   * highlights the wrong window is worse than no picker, because you don't
   * find out until the recording is done. */
  function calibrate(displayW) {
    if (!displayW || !window.innerWidth) return true;
    scale = window.innerWidth / displayW;
    if (Math.abs(scale - 1) < 0.02) { scale = 1; return true; }
    // Non-1:1 is survivable — we just scale — but a wild value means we're
    // not covering the display we think we are.
    return scale > 0.25 && scale < 4;
  }

  function px(v) { return v * scale; }

  /* --- geometry ---------------------------------------------------------- */

  function hitTest(x, y) {
    // windows is FRONT-TO-BACK, so the first containing rect is the one the
    // user is actually looking at — the same window a click would land in.
    for (var i = 0; i < windows.length; i++) {
      var w = windows[i];
      if (x >= px(w.x) && x <= px(w.x + w.w) &&
          y >= px(w.y) && y <= px(w.y + w.h)) return w;
    }
    return null;
  }

  function byId(id) {
    for (var i = 0; i < windows.length; i++) {
      if (windows[i].id === id) return windows[i];
    }
    return null;
  }

  function intersect(a, b) {
    var x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y);
    var x1 = Math.min(a.x + a.w, b.x + b.w);
    var y1 = Math.min(a.y + a.h, b.y + b.h);
    if (x1 <= x0 || y1 <= y0) return null;
    return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  }

  function overlaps() {
    var out = [];
    for (var i = 0; i < picked.length; i++) {
      for (var j = i + 1; j < picked.length; j++) {
        var a = byId(picked[i]), b = byId(picked[j]);
        if (!a || !b) continue;
        var hit = intersect(a, b);
        if (hit) out.push(hit);
      }
    }
    return out;
  }

  /* --- painting ---------------------------------------------------------- */

  function roundRect(c, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    if (c.roundRect) { c.beginPath(); c.roundRect(x, y, w, h, r); return; }
    c.beginPath();
    c.moveTo(x + r, y);
    c.arcTo(x + w, y, x + w, y + h, r);
    c.arcTo(x + w, y + h, x, y + h, r);
    c.arcTo(x, y + h, x, y, r);
    c.arcTo(x, y, x + w, y, r);
    c.closePath();
  }

  function rectOf(w) {
    return { x: px(w.x), y: px(w.y), w: px(w.w), h: px(w.h) };
  }

  function paint() {
    var dpr = window.devicePixelRatio || 1;
    var vw = window.innerWidth, vh = window.innerHeight;
    if (canvas.width !== Math.round(vw * dpr)) {
      canvas.width = Math.round(vw * dpr);
      canvas.height = Math.round(vh * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, vw, vh);

    // 1. the scrim.
    ctx.fillStyle = "rgba(9, 8, 7, 0.58)";
    ctx.fillRect(0, 0, vw, vh);

    // 2. punch it out wherever a window is lit, so "lit" means "you can see
    //    it" rather than "it has a border" — that's what makes the picker
    //    read as a preview of the recording.
    var lit = picked.slice();
    if (hoverId !== null && lit.indexOf(hoverId) === -1) lit.push(hoverId);
    ctx.globalCompositeOperation = "destination-out";
    lit.forEach(function (id) {
      var w = byId(id);
      if (!w) return;
      var r = rectOf(w);
      roundRect(ctx, r.x, r.y, r.w, r.h, 10);
      ctx.fill();
    });
    ctx.globalCompositeOperation = "source-over";

    // 3. rings. Selected windows get a solid accent ring and a faint wash;
    //    the hovered-but-unselected one gets a thinner ring only, so the two
    //    states never look alike.
    picked.forEach(function (id) {
      var w = byId(id);
      if (!w) return;
      var r = rectOf(w);
      roundRect(ctx, r.x, r.y, r.w, r.h, 10);
      ctx.fillStyle = "rgba(63, 191, 178, 0.10)";
      ctx.fill();
      ctx.strokeStyle = "#5cd9cc";
      ctx.lineWidth = 3;
      ctx.stroke();
    });
    if (hoverId !== null && picked.indexOf(hoverId) === -1) {
      var hw = byId(hoverId);
      if (hw) {
        var hr = rectOf(hw);
        roundRect(ctx, hr.x, hr.y, hr.w, hr.h, 10);
        ctx.strokeStyle = "rgba(92, 217, 204, 0.85)";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }

    // 4. overlaps between SELECTED windows, hatched in the signal colour.
    //    This is not decoration: it is the region where the recording will
    //    contain the wrong window's pixels.
    overlaps().forEach(function (o) {
      var r = { x: px(o.x), y: px(o.y), w: px(o.w), h: px(o.h) };
      ctx.save();
      roundRect(ctx, r.x, r.y, r.w, r.h, 4);
      ctx.clip();
      ctx.fillStyle = "rgba(224, 75, 65, 0.20)";
      ctx.fillRect(r.x, r.y, r.w, r.h);
      ctx.strokeStyle = "rgba(224, 75, 65, 0.55)";
      ctx.lineWidth = 2;
      for (var d = -r.h; d < r.w; d += 12) {
        ctx.beginPath();
        ctx.moveTo(r.x + d, r.y);
        ctx.lineTo(r.x + d + r.h, r.y + r.h);
        ctx.stroke();
      }
      ctx.restore();
    });

    paintBadges();
  }

  function paintBadges() {
    badges.textContent = "";
    picked.forEach(function (id, i) {
      var w = byId(id);
      if (!w) return;
      var el = document.createElement("div");
      el.className = "pk-badge";
      el.textContent = String(i + 1);
      el.style.left = (px(w.x) + 12) + "px";
      el.style.top = (px(w.y) + 12) + "px";
      badges.appendChild(el);
    });
  }

  /* --- HUD --------------------------------------------------------------- */

  function syncHud() {
    var n = picked.length;
    elCount.textContent = n + " / " + MAX;
    elDone.disabled = n === 0;
    elTitle.textContent = n === 0
      ? "Click windows to record"
      : (n === 1 ? "1 window / fills the frame"
                 : n + " windows / arranged like your desktop");

    var missing = picked.filter(function (id) { return !byId(id); }).length;
    var laps = overlaps().length;
    if (missing) {
      elWarnText.textContent = missing === 1
        ? "One picked window has gone. It will be dropped."
        : missing + " picked windows have gone. They will be dropped.";
      elWarn.hidden = false;
    } else if (laps) {
      elWarnText.textContent = laps === 1
        ? "Two of these overlap. Whatever's on top gets recorded into the one underneath."
        : "Some of these overlap. Whatever's on top gets recorded into the ones underneath.";
      elWarn.hidden = false;
    } else {
      elWarn.hidden = true;
    }
    elArrangeRow.hidden = laps === 0;
  }

  function toggle(w) {
    if (!w) return;
    var at = picked.indexOf(w.id);
    if (at >= 0) picked.splice(at, 1);
    else if (picked.length < MAX) picked.push(w.id);
    else return;             // silently capped; the count says 4 / 4
    syncHud();
    paint();
  }

  /* --- data -------------------------------------------------------------- */

  function load() {
    return fetch("/api/windows", {
      headers: requestHeaders({ "Accept": "application/json" })
    })
      .then(function (r) { return r.json(); })
      .then(function (data) {
        windows = (data && data.windows) || [];
        // Drop picks whose window has gone; keeping a ghost selected would
        // let you hit Done on something that can't be recorded.
        picked = picked.filter(function (id) { return !!byId(id); });
        syncHud();
        paint();
      })
      .catch(function () { /* transient; the next tick retries */ });
  }

  /* --- exit -------------------------------------------------------------- */

  function finish(ids) {
    if (closed) return;
    closed = true;
    var api = bridge();
    if (!api || !api.picker_done) { return; }
    try {
      api.picker_done(ids, ids.length > 1 && !!elArrange.checked);
    } catch (e) { /* the window is going away regardless */ }
  }

  function cancel() {
    if (closed) return;
    closed = true;
    var api = bridge();
    if (api && api.picker_cancel) {
      try { api.picker_cancel(); } catch (e) {}
    }
  }

  /* --- wiring ------------------------------------------------------------ */

  window.addEventListener("mousemove", function (ev) {
    if (hud.contains(ev.target)) {
      if (hoverId !== null) { hoverId = null; paint(); }
      return;
    }
    var w = hitTest(ev.clientX, ev.clientY);
    var id = w ? w.id : null;
    if (id !== hoverId) { hoverId = id; paint(); }
  });

  window.addEventListener("mousedown", function (ev) {
    if (hud.contains(ev.target)) return;
    ev.preventDefault();
    toggle(hitTest(ev.clientX, ev.clientY));
  });

  window.addEventListener("keydown", function (ev) {
    if (ev.key === "Escape") { ev.preventDefault(); cancel(); }
    else if (ev.key === "Enter" && picked.length) { ev.preventDefault(); finish(picked.slice()); }
  });

  elDone.addEventListener("click", function () { finish(picked.slice()); });
  elClear.addEventListener("click", function () { finish([]); });
  window.addEventListener("resize", paint);

  // The bar hands us the display size so we can check our own scale.
  window.__pickerInit = function (displayW) {
    if (!calibrate(displayW)) {
      // We are not where we think we are. Bail rather than mislead.
      cancel();
      return false;
    }
    return true;
  };

  load();
  setInterval(function () { if (!closed) load(); }, REFRESH_MS);
  syncHud();
  paint();
})();
