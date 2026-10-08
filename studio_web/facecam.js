/* facecam.js — the floating webcam bubble.
   Popped out of the recording pill (bar.js floatFacecam) into its own
   frameless always-on-top window. It owns the camera while it's open: the
   pill drops its stream first, and hands it back when this docks.

   The bar drives it through two globals the native bridge evaluates in this
   page — __faceRelease() before a recording (the capture ffmpeg needs the
   device) and __faceResume() after. The native side also hides this whole
   window around the take, since it floats over the display being captured. */
"use strict";

(function () {
  const $ = function (id) { return document.getElementById(id); };
  const bubble = $("face-bubble");
  const video = $("face-video");
  const img = $("face-img");
  const empty = $("face-empty");
  const dock = $("face-dock");

  const isNative = queryParam("native") === "1";
  if (isNative) document.documentElement.classList.add("native");

  let stream = null;

  function nativeApi() {
    return (window.pywebview && window.pywebview.api) ? window.pywebview.api : null;
  }

  /* bridge calls return a promise that rejects if Python raised or the window
     is already going away — these are fire-and-forget, so swallow both */
  function bridgeCall(name) {
    const bridge = nativeApi();
    if (!bridge || !bridge[name]) return false;
    try {
      const p = bridge[name]();
      if (p && p.catch) p.catch(function () {});
    } catch (e) { return false; }
    return true;
  }

  function setLive(live) {
    video.hidden = !live;
    empty.hidden = live;
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
    video.srcObject = null;
    // clearing src is what closes the MJPEG connection, which is what makes
    // Python release the device
    if (img) { img.removeAttribute("src"); img.hidden = true; }
    setLive(false);
  }

  /* The native shell takes frames from Python, never from getUserMedia — the
   * web API makes WebKit raise its own per-origin camera prompt on every new
   * web view, and this window is created fresh each time the bubble pops out,
   * so it would ask every single time. Same reasoning as bar.js. */
  const useNativeStream = isNative ||
    !(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);

  async function start() {
    stop();
    if (useNativeStream) {
      const ordinal = queryParam("ordinal") || "0";
      video.hidden = true;
      img.hidden = false;
      img.onerror = function () {
        img.onerror = null;
        img.hidden = true;
        setLive(false);
        empty.textContent = "Camera unavailable";
      };
      img.src = autocineUrl(
        "/api/camera/preview?ordinal=" + encodeURIComponent(ordinal) +
        "&t=" + Date.now());
      empty.hidden = true;
      return;
    }
    img.hidden = true;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      video.srcObject = stream;
      setLive(true);
    } catch (e) {
      setLive(false);
      empty.textContent = "Camera unavailable";
    }
  }

  /* --- bridge the bar calls into --- */
  /* Release paints NOTHING. This window sits on top of the display being
     captured, so any placeholder it shows is burned into raw.mov and into the
     thumbnail cut from it — "RECORDING" ended up in the corner of every take.
     The native side hides the window outright (facecam_release); the browser
     popup fallback at least falls back to the neutral empty state. */
  window.__faceRelease = function () { stop(); };
  window.__faceResume = function () { empty.textContent = "Camera off"; start(); };

  function dockBack() {
    stop();
    if (bridgeCall("face_dock")) return;
    // browser-fallback popup: tell the opener, then close ourselves
    try {
      if (window.opener && window.opener.__barFacecamDocked) {
        window.opener.__barFacecamDocked();
      }
    } catch (e) { /* cross-origin — shouldn't happen, same server */ }
    window.close();
  }

  dock.addEventListener("click", dockBack);

  /* Drag the window from anywhere on the bubble. pywebview's drag handler
     walks up from the event target, so the dock button has to kill the event
     in the capture phase or clicking it would drag instead. */
  document.addEventListener("mousedown", function (ev) {
    if (!isNative) return;
    if (ev.target.closest && ev.target.closest("button")) {
      ev.stopPropagation();
      return;
    }
    bubble.classList.add("dragging");
  }, true);
  window.addEventListener("mouseup", function () {
    if (!bubble.classList.contains("dragging")) return;
    bubble.classList.remove("dragging");
    bridgeCall("face_moved");
  });

  window.addEventListener("pagehide", stop);
  start();
})();
