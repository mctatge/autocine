# site/: the public landing page

A static, dependency-free page for AutoCine: an interactive hero, an MCP
workflow section, and a minimal release-status footer. Behind the hero copy, a
sample take plays on a drawn desktop and a virtual camera springs toward each
click. The prompt bar is a working search over a sample library of takes and
their transcripts. Picking a result seeks the take, and the camera plays the
auto-zoom toward that moment. The page is dark only and has no build step, no
framework and no fetch/XHR, so it also runs from `file://`.

## Preview

```bash
open site/index.html                              # straight from disk works
python3 -m http.server --directory site 5288     # or http://localhost:5288
```

URL params for deterministic screenshots:

- `?shot=1` freezes all motion on one composed frame.
- `?q=promo%20code` prefills the search and opens its results.

Both can be combined.

Search check (every example must hit):

```bash
node -e "global.window={};require('./site/demo-library.js');const S=require('./site/main.js');for(const q of window.AUTOCINE_DEMO.examples)console.log(q,'->',S.search(q,window.AUTOCINE_DEMO).length)"
```

## Files

| File | What it holds |
|---|---|
| `index.html` | The markup, the inline SVG placeholder desktop, the `<video>` wiring, and the MCP workflow/privacy boundary. |
| `styles.css` | Tokens copied from `studio_web/ui.css`. Teal means the camera, orange the footage, red the recording (logo dot only). It also carries the responsive MCP workflow grid below the hero. |
| `main.js` | Part 1 is `window.AutoCineSearch.search(query, data)`, pure and also runnable under node. Part 2 is the hero: the sample take, the spring camera, the video takeover, the combobox, the typewriter and the pause control. |
| `demo-library.js` | `window.AUTOCINE_DEMO`, the invented sample library. It ships publicly. |
| `_headers` | Cloudflare Pages response-security headers for the public preview. |
| `media/` | Empty on purpose (`.gitkeep`). The hero video goes here. |

## Behavior worth knowing before editing

- **Search** matches words in any script and folds case, accents and compatibility forms ("naïve" finds "naive"). Matching is per word, so highlights stay on the original text.
- **Results panel.** On desktop it opens below the bar when everything fits there. Otherwise it opens above the bar and the title steps back while it is open. The list scrolls inside whatever room it gets, and the page itself never scrolls. Phones keep it below the bar and lift the bar to the top of the screen while typing.
- **Still frame** (`?shot=1`, reduced motion). The reticle previews the "Place order" zoom and its chip reads `→ 2.40×`, the target, while the camera is at 1.00×. The frame keeps 16 px clear of the copy: its margin tightens first, and it is left out where even that collides. The chip does the same with its four candidate spots.
- **Large displays** (from 2000x1100). `--u` in `styles.css` scales the whole 1920x1080 layout proportionally, up to 1.45×.
- **MCP section.** The four-stage copy mirrors the public guide in
  `docs/mcp.md`: orient, locate, edit, verify. Keep the privacy boundary
  explicit whenever tool results include transcript text or preview images.
  Do not imply that a connected model runs locally merely because AutoCine
  does.

## TODO: hero video

The page already contains
`<video autoplay muted loop playsinline poster="media/hero-poster.jpg">` with
`media/hero-loop.webm` and `media/hero-loop.mp4` sources. All three files are
missing today, so the animated placeholder shows.

Once the files exist, the real video takes over with no code change:

- Full motion: the video replaces the placeholder once it is actually playing. A 404, blocked autoplay or a poster on its own keeps the placeholder.
- Reduced motion and the pause state: the poster is shown. The page requests it only once it is still, so full motion costs no extra request.
- Motion coming back (unpausing, or reduced motion turned off) with a video that cannot play (missing, failed, refused, or not playing within 1.5 s) returns to the animated placeholder.

To make the files:

1. **Record** a clean take on a demo machine with nothing personal on screen: a short checkout-style walkthrough with 4 to 6 deliberate clicks, narrated.
2. **Render** it with AutoCine so the auto-zoom is baked in. The page adds no ambient zoom on top of real video; a pick adds only a gentle push.
3. **Export** these files:

   | Property | Value |
   |---|---|
   | Length | About 30 to 40 s, a seamless loop (end on the frame it starts on) |
   | Resolution | 1920x1080, or 2560x1440 |
   | Audio | None (muted, no audio track) |
   | Size | Under 6 MB per file |

   - `media/hero-loop.mp4`: H.264, yuv420p, `-movflags +faststart`
   - `media/hero-loop.webm`: VP9 or AV1
   - `media/hero-poster.jpg`: one good frame, the same size as the video
4. **Enable the media.** Add `poster="media/hero-poster.jpg"` to
   `#hero-video`, then add WebM and MP4 `<source>` children. The public default
   intentionally has no source URLs, so an incomplete checkout does not issue
   failed media requests.
5. **Transcript.** Replace the featured project's `segments` in `demo-library.js` with that take's `transcript.json` segments (`{t, dur, text}`, seconds on the video's clock). Also update its `name` and `duration`, and keep `examples` hitting. A malformed entry (a null segment, `segments` that is not an array) is skipped rather than breaking the page, so rerun the search check above to catch one. Never paste a personal take's transcript: this file is public.
6. **In `main.js`:**
   - Optionally fill `VIDEO_SPOTS`: per featured segment, where its moment sits in the video frame, as fractions `[x, y, w, h]`. Until then, a pick on the video is a centred push.
   - `FEATURED_PICKS`, `MOVES` and `ZOOMS` describe only the placeholder, so leave them alone.
7. **Share image.** Add an absolute `og:image` once the site has a domain (see the TODO in `index.html`).

## Launch status

- **Public URL.** <https://autocine.pages.dev/> is the production URL. The
  `autocine` Cloudflare Pages project was created by Direct Upload on
  2026-10-05 and is not Git-connected.
- **Release status.** The page presents AutoCine as a free, experimental source
  release and links to <https://github.com/mctatge/autocine>. It does not offer
  a packaged Mac download. Replace the Mac-download language only after a
  signed and notarized build exists. The source-versus-binary boundary is also
  documented in `../docs/publishing.md`.
- **Typography.** The public preview uses the system Bodoni/Didot stack. It makes no third-party font request.

## Product name locations (for a rename)

The product name appears in user-facing metadata, the hero, the MCP section,
and the footer in `index.html`. Search the file rather than relying on a
fixed count. The principal locations are:

- `<title>`
- the meta description
- `og:site_name`
- `og:title`
- the brand link's `aria-label`
- the nav wordmark
- the bold first line of the lede
- the MCP section's privacy boundary
- the footer line

It also appears in code identifiers and comments:

- `window.AutoCineSearch` in `main.js`. This is the search API name the page contract asks for.
- `window.AUTOCINE_DEMO` in `demo-library.js` and `main.js`.
- The `localStorage` key `autocine.hero.motion` in `main.js`.
- One comment in `styles.css`.
