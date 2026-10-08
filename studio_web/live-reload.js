// Auto-reload on server restart during `studio app --reload`.
//
// Polls /api/rev; the server's boot id changes on every restart, and the
// `reload` flag tells us whether we're even in dev mode. In non-reload
// mode the first response makes us stop polling — this script is safe to
// include in production HTML.
(function () {
  var POLL_MS = 1000;
  var firstRev = null;
  var errCount = 0;

  function schedule(ms) { setTimeout(tick, ms); }

  function tick() {
    fetch("/api/rev", {
      cache: "no-store",
      headers: autocineHeaders()
    }).then(function (r) {
      if (!r.ok) throw new Error("bad status");
      return r.json();
    }).then(function (j) {
      errCount = 0;
      if (!j || !j.reload) return; // not in dev mode; stop polling silently
      if (firstRev === null) {
        firstRev = j.rev;
        schedule(POLL_MS);
        return;
      }
      if (j.rev !== firstRev) {
        location.reload();
        return;
      }
      schedule(POLL_MS);
    }).catch(function () {
      // Server is probably restarting — back off and keep trying. When it
      // comes back up the boot id will have changed and we'll reload above.
      errCount++;
      schedule(Math.min(300 + errCount * 200, 2000));
    });
  }

  tick();
})();
