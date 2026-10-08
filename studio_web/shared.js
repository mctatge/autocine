/* shared.js — tiny helpers shared by the library, editor, and recording bar. */
"use strict";

function autocineToken() {
  const node = document.querySelector('meta[name="autocine-token"]');
  return node ? node.getAttribute("content") || "" : "";
}

function autocineHeaders(initial) {
  const headers = Object.assign({}, initial || {});
  const token = autocineToken();
  if (token) headers["X-AutoCine-Token"] = token;
  return headers;
}

function autocineUrl(path) {
  const token = autocineToken();
  if (!token) return path;
  const sep = path.indexOf("?") >= 0 ? "&" : "?";
  return path + sep + "autocine_token=" + encodeURIComponent(token);
}

async function api(path, options) {
  const opts = {
    method: (options && options.method) || "GET",
    headers: autocineHeaders(options && options.headers)
  };
  if (options && options.body !== undefined) {
    opts.method = options.method || "POST";
    opts.headers["Content-Type"] = "application/json";
    opts.body = JSON.stringify(options.body);
  }
  if (options && options.keepalive) opts.keepalive = true;
  const res = await fetch(path, opts);
  let data = null;
  try { data = await res.json(); } catch (_) { /* non-JSON */ }
  if (!res.ok) {
    const msg = data && data.error ? data.error : "HTTP " + res.status;
    const err = new Error(msg);
    err.status = res.status;
    err.data = data;
    throw err;
  }
  return data;
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}

function toNum(value, fallback) {
  const n = parseFloat(value);
  return Number.isFinite(n) ? n : fallback;
}

/* 0:07.42 style timecode */
function fmtTime(sec, showCs) {
  if (!Number.isFinite(sec) || sec < 0) sec = 0;
  const m = Math.floor(sec / 60);
  const s = sec - m * 60;
  if (showCs === false) {
    return m + ":" + String(Math.floor(s)).padStart(2, "0");
  }
  const whole = Math.floor(s);
  const cs = Math.floor((s - whole) * 100);
  return m + ":" + String(whole).padStart(2, "0") + "." + String(cs).padStart(2, "0");
}

function fmtDuration(sec) {
  if (!Number.isFinite(sec)) return "-";
  if (sec >= 60) {
    const m = Math.floor(sec / 60);
    const s = Math.round(sec - m * 60);
    return m + "m " + s + "s";
  }
  return (Math.round(sec * 10) / 10) + "s";
}

/* "20260724-171021" -> "Jul 24, 5:10 PM" (best effort) */
function fmtSessionDate(name) {
  const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$/.exec(name || "");
  if (!m) return null;
  const d = new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6]);
  if (isNaN(d.getTime())) return null;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    ", " + d.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}

function debounce(fn, waitMs) {
  let t = null;
  const wrapped = function () {
    const args = arguments;
    if (t) clearTimeout(t);
    t = setTimeout(function () { t = null; fn.apply(null, args); }, waitMs);
  };
  wrapped.cancel = function () { if (t) { clearTimeout(t); t = null; } };
  wrapped.flush = function () { if (t) { clearTimeout(t); t = null; fn(); } };
  return wrapped;
}

function tempId(prefix) {
  return prefix + "-" + Math.random().toString(36).slice(2, 10);
}

function queryParam(name) {
  return new URLSearchParams(window.location.search).get(name);
}

/* --- toasts --- */
let _toastWrap = null;
function toast(message, kind) {
  if (!_toastWrap) {
    _toastWrap = el("div", "toast-wrap");
    document.body.appendChild(_toastWrap);
  }
  const node = el("div", "toast" + (kind === "err" ? " err" : ""), message);
  _toastWrap.appendChild(node);
  setTimeout(function () {
    node.style.transition = "opacity 200ms ease";
    node.style.opacity = "0";
    setTimeout(function () { node.remove(); }, 220);
  }, kind === "err" ? 4200 : 2400);
}

/* --- open the floating recorder pill ---
 * Prefers the native frameless window (POST /api/bar spawns `studio.py bar`
 * as a detached child that reuses this server and opens the pill on its own
 * Cocoa main thread). Falls back to a chromeless browser popup if pywebview
 * isn't available. Shared by the Library and the editor "New Recording". */
function openRecorderBar() {
  api("/api/bar", { body: {} }).then(function (j) {
    if (j && j.native) return;
    openRecorderBarPopup();
  }).catch(function () {
    openRecorderBarPopup();
  });
}

function openRecorderBarPopup() {
  const w = 880, h = 150;
  const left = Math.round((screen.availWidth - w) / 2);
  const top = screen.availHeight - h - 56;
  const win = window.open("/bar.html", "autocine-bar",
    "popup=yes,width=" + w + ",height=" + h + ",left=" + left + ",top=" + top);
  if (!win) toast("Popup blocked. Allow popups for this local AutoCine page and try again.", "err");
}

/* --- gradient CSS from a list of hex stops (background preset swatches) --- */
function gradientCss(colors) {
  if (!colors || !colors.length) return "#2b2823";
  if (colors.length === 1) return colors[0];
  return "linear-gradient(135deg, " + colors.join(", ") + ")";
}

/* --- shared inline icons (stroke follows currentColor) --- */
const ICONS = {
  record: '<svg viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="8" r="6"/></svg>',
  stop: '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="4" y="4" width="8" height="8" rx="1.5"/></svg>',
  play: '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M5 3.2v9.6c0 .8.9 1.3 1.6.9l7-4.8c.6-.4.6-1.4 0-1.8l-7-4.8C5.9 1.9 5 2.4 5 3.2z" transform="translate(-1,0)"/></svg>',
  pause: '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3.5" y="3" width="3.2" height="10" rx="1"/><rect x="9.3" y="3" width="3.2" height="10" rx="1"/></svg>',
  close: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M4 4l8 8M12 4l-8 8"/></svg>',
  chevronL: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M10 3L5 8l5 5"/></svg>',
  undo: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h7a3.5 3.5 0 010 7H8"/><path d="M6 4L3 7l3 3"/></svg>',
  redo: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M13 7H6a3.5 3.5 0 000 7h2"/><path d="M10 4l3 3-3 3"/></svg>',
  zoom: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><circle cx="7" cy="7" r="4.2"/><path d="M10.2 10.2L13.5 13.5"/><path d="M7 5.2v3.6M5.2 7h3.6"/></svg>',
  frame: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><rect x="2.2" y="3.2" width="11.6" height="9.6" rx="2"/><rect x="4.6" y="5.6" width="6.8" height="4.8" rx="1"/></svg>',
  windows: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="1.6" y="3.6" width="5.6" height="8.8" rx="1.3"/><rect x="8.8" y="3.6" width="5.6" height="8.8" rx="1.3"/></svg>',
  cursor: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M4 2.5l8.5 5.2-3.9 1-2.2 3.4L4 2.5z"/></svg>',
  audio: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6v4h2.5L9 13V3L5.5 6H3z"/><path d="M11 5.5a3.6 3.6 0 010 5"/></svg>',
  transcript: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2.4 3.6h11.2M2.4 6.6h11.2M2.4 9.6h7.4M2.4 12.6h4.6"/></svg>',
  marker: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4 14V2.8h7.5L9.5 5.9l2 3.1H4"/></svg>',
  folder: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><path d="M2.2 4.5c0-.7.6-1.3 1.3-1.3h2.8l1.4 1.6h4.8c.7 0 1.3.6 1.3 1.3v5.6c0 .7-.6 1.3-1.3 1.3H3.5c-.7 0-1.3-.6-1.3-1.3V4.5z"/></svg>',
  kebab: '<svg viewBox="0 0 16 16" fill="currentColor"><circle cx="8" cy="3.5" r="1.3"/><circle cx="8" cy="8" r="1.3"/><circle cx="8" cy="12.5" r="1.3"/></svg>',
  export: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M8 10V2.5M5 5l3-2.8L11 5"/><path d="M3 9.5v3c0 .8.7 1.5 1.5 1.5h7c.8 0 1.5-.7 1.5-1.5v-3"/></svg>',
  camera: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"><rect x="1.8" y="4.2" width="8.4" height="7.6" rx="1.6"/><path d="M10.2 7.2l4-2.2v6l-4-2.2"/></svg>',
  display: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="1.8" y="2.8" width="12.4" height="8.4" rx="1.4"/><path d="M6 13.6h4"/></svg>',
  mic: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"><rect x="6" y="1.8" width="4" height="7.4" rx="2"/><path d="M3.6 7.5a4.4 4.4 0 008.8 0"/><path d="M8 12v2.2"/></svg>',
};

function iconEl(name, size) {
  const span = el("span", "icon");
  span.style.display = "inline-flex";
  span.style.width = (size || 16) + "px";
  span.style.height = (size || 16) + "px";
  span.innerHTML = ICONS[name] || "";
  const svg = span.querySelector("svg");
  if (svg) { svg.style.width = "100%"; svg.style.height = "100%"; }
  return span;
}
