# Oral Cancer Classifier — Frontend

React 18 + Vite + TypeScript single-page app for the
[oral cancer classification API](../backend/README.md). Upload an intraoral
photograph, get a Cancer / Non-Cancer classification with the probability,
threshold, and raw model output.

> **Research prototype — not a diagnostic device.** The disclaimer banner is
> persistent and non-dismissible by design.

---

## Quick start

Requires Node 18+ (built and tested on Node 22).

```bash
cd frontend

npm install
cp .env.example .env              # defaults to http://localhost:8000

npm run dev                       # http://localhost:5173
```

The backend must be running separately — see
[`../backend/README.md`](../backend/README.md).

### Scripts

| Command | Does |
| --- | --- |
| `npm run dev` | Dev server with HMR on port 5173 |
| `npm run build` | Type-check (`tsc -b`) then production build to `dist/` |
| `npm run preview` | Serve the production build locally |
| `npm run typecheck` | `tsc --noEmit`, no build output |

---

## Configuration

One variable, in `.env`:

```
VITE_API_BASE_URL=http://localhost:8000
```

No trailing slash. Two things to know:

1. **Vite inlines this at build time**, not runtime. Changing it requires a
   rebuild, not just a restart. On Vercel, set it in
   *Project → Settings → Environment Variables* and redeploy.
2. **The value must also appear in the backend's `ALLOWED_ORIGINS`** — or rather,
   this app's own origin must. Otherwise the browser blocks every response and
   the app reports it cannot reach the server.

The app detects the two misconfigurations that only surface once deployed — a
deployed page still pointing at `localhost`, and an `https://` page calling an
`http://` API (blocked as mixed content) — and names the actual fix instead of
showing a generic connection error.

---

## Architecture

```
src/
├── App.tsx                  layout + composition; owns only file selection
├── main.tsx                 entry point
├── index.css                Tailwind v4 + the dark theme tokens
├── types.ts                 mirrors the backend's Pydantic models
├── lib/
│   ├── api.ts               axios client, ApiError normalisation
│   └── fileValidation.ts    client-side type/size pre-checks
├── hooks/
│   ├── usePrediction.ts     idle → loading → success | error
│   └── useObjectUrl.ts      object URL lifecycle + revocation
└── components/
    ├── DisclaimerBanner.tsx persistent, non-dismissible
    ├── UploadZone.tsx       drag-drop, click-to-browse, mobile camera
    ├── ImagePreview.tsx     selected image + "model sees 224×224" note
    ├── AnalyzeButton.tsx    disabled until a file is chosen; spinner
    ├── ResultCard.tsx       verdict, probability bar, technical details
    ├── ProbabilityBar.tsx   P(Cancer) on a fixed 0–1 scale with threshold
    └── ErrorAlert.tsx       friendly errors with retry
```

**All request state lives in `usePrediction`.** Components render it; none hold
request state of their own. An in-flight request is aborted when a new one
starts or the component unmounts, so a slow response can never overwrite a newer
result.

### No health polling — deliberately

There is no backend readiness probe on page load. It cost a request on every
visit, raced the user's first action, and reported "offline" for transient
conditions that would have cleared by the time they pressed Analyze.

Connection problems surface where they matter instead: as an error on the
request the user actually made, with a retry attached. The page makes **zero API
requests until you click Analyze**.

### Types mirror the backend by hand

[`types.ts`](src/types.ts) is written to match `backend/app/schemas.py`. There is
no code generation, so **changing a response model means editing both**. This is
the main argument for keeping both apps in one repo — split repos let these
drift silently.

---

## Design

Dark clinical theme built on semantic tokens in
[`index.css`](src/index.css), so component classes say what a colour is *for*
rather than how dark it is:

| Token | Value | Role |
| --- | --- | --- |
| `canvas` | `#0b0f14` | page |
| `surface` | `#131a22` | panels |
| `raised` | `#1b232d` | raised panels |
| `line` | `#263140` | hairlines |
| `ink` | `#e9eef4` | primary text |
| `muted` | `#97a5b6` | secondary text |
| `faint` | `#8593a6` | captions, labels |
| `accent` | `#f2f6fa` | the primary action |
| `alert` | `#ee6d5f` | **reserved for a positive (Cancer) finding** |

On a dark ground the primary action inverts: the Analyze button is near-white
with dark text, making it the brightest block on the page so the main action
still reads first. `alert` is the only saturated tone in the layout and means
exactly one thing.

Retheming is a single block at the top of `index.css`.

**Contrast is verified, not eyeballed.** Every text/background pair clears WCAG
AA for normal text (4.5:1), the weakest at 5.07:1.

### Reading the result

The probability bar shows **P(Cancer) on a fixed 0–1 track** with the decision
threshold marked. The scale never rescales to the value — that would make every
result look equally extreme. The threshold caption anchors to the edge and drops
the colliding endpoint when the threshold sits near 0 or 1.

The verdict is styled from `probability >= threshold`, not by string-matching
the label, so renaming a class in the checkpoint cannot break the colour coding.

`ImagePreview` states that the model only sees 224 × 224 — explaining up front
why fine detail cannot have influenced the result.

---

## Accessibility

- Upload zone is a keyboard-operable `role="button"` with `tabIndex=0`;
  **Enter and Space** both open the file picker (verified in a real browser).
- The verdict carries `aria-live="polite"` so it is announced when it replaces
  the placeholder.
- The probability bar exposes `role="img"` with an `aria-label` stating both the
  probability and the threshold, since the visual is meaningless to a screen
  reader.
- Errors use `role="alert"`; the disclaimer uses `role="note"`.
- Visible `focus-visible` rings throughout, offset against the dark ground.
- `prefers-reduced-motion` disables the spinner and panel transitions.

---

## Behaviour notes

- **Object URLs are revoked** on file change and unmount (`useObjectUrl`), so
  selecting many images doesn't leak them for the page's lifetime.
- **Client-side validation is a convenience, not a boundary.** Type and size are
  pre-checked to avoid a pointless round trip — an oversized file makes **zero**
  requests — but the server re-validates by actually decoding the bytes.
- **Selecting a new image clears the previous verdict**, so a stale result never
  sits beside a new photograph.
- **Mobile camera capture** uses a separate input with
  `accept="image/*" capture="environment"`, shown only below the `sm` breakpoint
  where it isn't a duplicate of the file picker.
- **Responsive**: single column on mobile, two columns (preview | result) from
  `lg` up.

---

## Deployment (Vercel)

> **Before your first deploy:** this directory currently contains **both**
> `package-lock.json` (npm) and `pnpm-lock.yaml` (pnpm). Vercel picks its package
> manager from the lockfile it finds, so keep exactly one — delete the other and
> commit that removal. Both are currently in sync with `package.json`, so either
> choice works; just be consistent.

1. Import the repo; set **Root Directory** to `frontend`.
2. Framework preset **Vite** is detected; [`vercel.json`](vercel.json) pins the
   build command, output directory, SPA rewrites, asset caching, and security
   headers.
3. Set `VITE_API_BASE_URL` to your backend's public **https** URL.
4. Add that Vercel URL to the backend's `ALLOWED_ORIGINS`. For preview
   deployments — a new hostname per commit — set `ALLOWED_ORIGIN_REGEX` on the
   backend instead:

   ```
   ALLOWED_ORIGIN_REGEX=https://your-app-.*-yourteam\.vercel\.app
   ```

Production build is ~215 kB JS (~72 kB gzipped) and ~25 kB CSS (~6 kB gzipped).

---

## Troubleshooting

**"Cannot reach the server" while the backend is running.**
Almost always CORS, not connectivity. The browser refuses to hand a
cross-origin response to JavaScript when the origin isn't approved, so axios
sees *no response* — indistinguishable from a dead server.

Check the address bar port. Vite silently falls back to **5174**, then **5175**,
when 5173 is already taken (a second `npm run dev`, or a server left running).
The backend allows all three by default; if you're on some other port, add it to
`ALLOWED_ORIGINS`. The browser Console confirms it with `blocked by CORS policy`.

**Deployed app can't reach the API.** The app should tell you why. If it says the
deployment is calling `localhost`, `VITE_API_BASE_URL` was never set in the
hosting project — Vite inlined the default at build time. Set it and
**redeploy** (a restart won't do it).

**Type errors after changing the API.** `types.ts` mirrors
`backend/app/schemas.py` by hand; update both.

**Styles missing after editing `index.css`.** Tailwind v4 is configured via the
`@tailwindcss/vite` plugin and the `@theme` block — there is no
`tailwind.config.js`. Utilities are generated from the `--color-*` custom
properties, so a token must be defined there before `bg-…` / `text-…` will work.
An unknown utility is silently dropped rather than erroring.

---

## Stack

React 18 · Vite 8 · TypeScript 7 (strict, no `any`) · Tailwind CSS v4 · axios

`tsconfig.json` enables `strict`, `noUncheckedIndexedAccess`,
`exactOptionalPropertyTypes`, `noUnusedLocals`, and `verbatimModuleSyntax`.
