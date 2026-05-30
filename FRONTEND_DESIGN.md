# DataLens AI — Frontend Design System

A reference for the visual language of the React frontend: colors, typography, CSS architecture, chart styling, and motion. Tagline: **"Read the signal in your data."**

The stack is **React 19 + Vite 6 + TypeScript + Tailwind CSS 4 + Highcharts 12 + Framer Motion**. Fonts are loaded from Google Fonts in `index.html`.

There are intentionally **two coexisting palettes**:
1. **v3 Editorial** (current direction) — warm cream type on near-black with a single teal accent. Driven by CSS variables in `globals.css`.
2. **Legacy "cockpit/space"** (being retired) — dark slate with cyan + violet neon. Still used by some dashboard panels.

---

## 1. Color System

### 1.1 v3 Editorial palette (primary)
Defined as CSS custom properties in `src/styles/globals.css` and mapped to Tailwind tokens in `tailwind.config.ts`.

**Surfaces (backgrounds)**
| Token | Value | Tailwind | Use |
|-------|-------|----------|-----|
| `--bg-0` | `#0a0a0a` | `bg-bg-0` | Page background (near-black) |
| `--bg-1` | `#111111` | `bg-bg-1` | Cards / panels |
| `--bg-2` | `#161616` | `bg-bg-2` | Raised / hover surfaces, chips |
| `--bg-3` | `#1c1c1c` | `bg-bg-3` | Highest elevation |

**Lines / borders**
| Token | Value | Use |
|-------|-------|-----|
| `--line` | `rgba(255,255,255,0.08)` | Default border |
| `--line-strong` | `rgba(255,255,255,0.16)` | Hover / emphasis border |
| `--line-accent` | `rgba(94,234,212,0.32)` | Accent border (glow) |

**Ink (text), warm cream ramp**
| Token | Value | Use |
|-------|-------|-----|
| `--ink-1` | `#f5efe6` | Primary text / headings (cream) |
| `--ink-2` | `#c9c1b4` | Body / secondary text |
| `--ink-3` | `#8a8478` | Muted labels, axis titles |
| `--ink-4` | `#5a5650` | Disabled / faintest |

**Single accent — teal**
| Token | Value | Use |
|-------|-------|-----|
| `--accent` | `#5eead4` | The one accent: links, highlights, primary chart series |
| `--accent-soft` | `rgba(94,234,212,0.14)` | Accent fills, chip backgrounds |
| `--accent-line` | `rgba(94,234,212,0.28)` | Accent borders |
| `--accent-strong` | `rgba(94,234,212,0.85)` | Strong accent / underlines |
| `--accent-glow` | `rgba(94,234,212,0.18)` | Hover glow shadows |

**Status colors**
| Token | Value | Meaning |
|-------|-------|---------|
| `--ok` | `#84d8b8` | Good / pass (green) |
| `--warn` | `#f5b14a` | Warning (amber) |
| `--bad` | `#f08080` | Error / bad (rose) |

### 1.2 Legacy "cockpit/space" palette (being retired)
Still consumed by some dashboard panels (TimeSeries, Telemetry, Upload, Nav, inputs).

| Group | Shades |
|-------|--------|
| `space` (slate bg) | `950 #030712`, `900 #0f172a`, `800 #111827`, `700 #1e293b`, `600 #334155` |
| `supernova` (cyan) | `100 #ecfeff`, `200 #cffafe`, `300 #a5f3fc`, `400 #22d3ee`, `500 #06b6d4`, `600 #0891b2` |
| `pulsar` (violet) | `300 #c4b5fd`, `400 #a78bfa`, `500 #8b5cf6`, `600 #7c3aed` |
| `cockpit` | `line rgba(255,255,255,0.05)`, `panel rgba(255,255,255,0.03)`, `glow rgba(6,182,212,0.18)` |

Slate text tones used inline: `text-slate-100/200/400/500`, `text-cyan-100`.

---

## 2. Typography

Fonts loaded in `index.html` via Google Fonts:
- **Inter** — weights 400–900
- **JetBrains Mono** — weights 500–700

| Family | Stack | Role |
|--------|-------|------|
| `--font-display` | `Inter, system-ui, sans-serif` | All display + body |
| `--font-mono` | `JetBrains Mono, ui-monospace, monospace` | Eyebrows, labels, code, table headers, stats |
| `--font-serif` | `Georgia, "Times New Roman", serif` | Defined, rarely used |

**Weights:** body `600`, strong `800`, display `900`.

**Type scale (component classes in `globals.css`):**
| Class | Size | Notes |
|-------|------|-------|
| `.display-1` | `clamp(56px, 7.5vw, 108px)` | Weight 900, tight `-0.035em`, line-height 0.96 |
| `.display-2` | `clamp(34px, 4.5vw, 58px)` | Weight 850 |
| `.lede` | `clamp(15px, 1.2vw, 18px)` | Weight 700, max-width 64ch |
| `.eyebrow` | 11px mono | Teal, uppercase, letter-spacing 0.32em |
| `.label-mono` | 11px mono | Muted ink-3, uppercase, 0.28em |
| `.stat-num` | — | Tabular figures (`font-feature-settings: "tnum"`) |

Body uses antialiased smoothing, `optimizeLegibility`, `overflow-x: hidden`.

---

## 3. CSS Architecture

- **Tailwind 4** via `@tailwindcss/vite`, imported with `@import "tailwindcss"` at the top of `src/styles/globals.css` (loaded once in `main.tsx`).
- **Three Tailwind layers** organize custom CSS:
  - `@layer base` — `:root` tokens, resets, body, focus, scrollbars, background effects.
  - `@layer components` — reusable classes (buttons, cards, chips, tables, prose).
  - `@layer utilities` — helpers (`.tnum`, `.text-balance`, `.text-pretty`, `.shimmer`).
- **Token strategy:** all editorial colors are CSS variables → mapped to Tailwind tokens in `tailwind.config.ts`, so a theme swap is a one-line change. `darkMode: ["class"]`, and `<html class="dark">` with `color-scheme: dark`.

**Global background effects** (fixed, pointer-events none, z-index -1):
- `body::before` — two soft teal radial glows (top-right + bottom-left), ~4.5% / 2.5% opacity.
- `body::after` — faint 96px×96px grid (`rgba(255,255,255,0.018)`), masked to fade at top/bottom edges.

**Misc base styling:**
- Selection: teal-soft background, cream text.
- Focus-visible: 2px solid teal outline, 2px offset.
- Scrollbars: 8px, translucent white thumb, brightens on hover.
- `prefers-reduced-motion`: collapses all animations/transitions to ~0ms.

**Radii:** `--r-sm 4px`, `--r-md 8px`, `--r-lg 12px` (cards often use larger `18px` / `rounded-2xl/3xl`).

---

## 4. Component Classes (`@layer components`)

| Class | Description |
|-------|-------------|
| `.btn-primary` | Pill, cream (`--ink-1`) fill on dark text; hover → white, lift `-1px`, soft shadow |
| `.btn-ghost` | Pill, transparent with `--line-strong` border; hover border `--ink-2`, bg `--bg-2` |
| `.btn-accent` | Pill, teal fill on dark text; hover lift + teal glow shadow |
| `.card` | `--bg-1`, 1px `--line`, radius 18px; hover → `--line-strong` |
| `.card-glow` | Hover → accent border + ring + drop shadow |
| `.chip` / `.chip-accent` | Pill tag, mono 11px; accent variant uses teal-soft bg + teal text |
| `.hr-rule` | Divider with centered mono label, gradient lines each side |
| `.table-editorial` | Mono uppercase headers (10px, ink-3), row hover bg `--bg-2`, line borders |
| `.prose-editorial` | Long-form text: ink-2 body 15px/1.75; headings cream weight 600; teal dotted-underline links; mono inline `code` on `--bg-2` |
| `.stat-num` | Tabular-figure numerics for metrics |

---

## 5. Chart Styling (Highcharts)

### 5.1 Global theme — `src/charts/theme.ts`
Applied once on import via `applyHighchartsTheme()` (idempotent).
- **Background:** transparent (inherits page).
- **Font:** Inter; titles weight 800, subtitles 600.
- **Animation:** chart 350ms, series 400ms.
- **Default series color sequence:**
  1. `#5eead4` accent (teal)
  2. `#c9c1b4` ink-2
  3. `#f5efe6` ink-1
  4. `#8a8478` ink-3
  5. `#5a5650` ink-4
  6. `#84d8b8` ok
  7. `#f5b14a` warn
  8. `#f08080` bad
- **Axes:** lines/ticks `rgba(255,255,255,0.16)`; gridlines `~0.04–0.06`; labels ink-2 11px; titles ink-3.
- **Legend:** ink-2 items, hover ink-1, hidden ink-4.
- **Tooltip:** bg `rgba(17,17,17,0.95)`, border `rgba(255,255,255,0.16)`, radius 4, cream text 12px.
- **Credits + accessibility module:** disabled.

### 5.2 Per-chart colors (editorial charts)
| Chart | Colors |
|-------|--------|
| `DistributionHistogram` | bars `var(--accent)` |
| `MissingnessBars` | bars `var(--accent)` |
| `FeatureImportanceBars` | bars `var(--accent)` |
| `ShapGlobalBars` | theme defaults |
| `HealthRadar` | `colorByPoint` (cycles theme palette) |
| `LeaderboardBars` | winner bar `#5eead4`, others `#c9c1b4`; data labels ink-2 |
| `CorrelationHeatmap` | diverging stops: `#f08080` (−1) → `#161616` (0) → `#5eead4` (+1); data labels `#0a0a0a` |
| `AnomalyHeatmap` | stops: `#161616` (0) → `#5eead4` (0.5) → `#f08080` (1); data labels `#0a0a0a` |

### 5.3 Legacy dashboard charts (`HC_DARK_THEME`)
Used by `TimeSeriesPanel`, `TelemetryCard`, etc. Slate/neon scheme:
- Series: cyan `#22d3ee`, violet `#a78bfa`, slate `#94a3b8`.
- Axis labels `#94a3b8`, axis titles `#64748b`, legend `#cbd5e1`, gridlines `rgba(255,255,255,0.04)`.
- `TelemetryCard` single series `#06b6d4`.
- `AnomalyEnsemblePanel` score gradient: rose `rgba(244,63,94,.85)` ≥0.85 → orange `rgba(251,146,60,.85)` ≥0.6 → amber `rgba(245,158,11,.7)` ≥0.4 → sky `rgba(56,189,248,.55)` ≥0.2 → slate `rgba(148,163,184,.35)`.

---

## 6. Motion & Animation

**Tokens (`globals.css`):** ease-out `cubic-bezier(0.22,1,0.36,1)`; durations fast 180ms, base 250ms, slow 350ms.

**Library:** Framer Motion. `Motion.tsx` has a cursor-following spotlight: `radial-gradient(420px circle at cursor, rgba(94,234,212,0.08), transparent 40%)`.

**Hooks:** `useCountUp` (number tweening, respects reduced motion), `useInView`, `useReducedMotion`.

**CSS animations:**
- `.shimmer` — skeleton loader, sweeping white gradient, 1.6s infinite.

**Tailwind keyframes/animations (legacy):**
| Name | Behavior |
|------|----------|
| `warp` | 1.5s pulse with translate + cyan drop-shadow (upload hangar) |
| `scan` | 1.55s background sweep |
| `orbit` | 32s full rotation |
| `trace` | 700ms SVG stroke-dashoffset draw-in |

---

## 7. Shadows, Gradients & Effects (Tailwind extend)

**Box shadows:**
- `panel` — `0 1px 0 rgba(255,255,255,0.05), 0 24px 60px rgba(0,0,0,0.45)`
- `halo` — cyan ring + 28px glow
- `warp` — cyan ring + 42px glow

**Background images:**
- `radial-cockpit` — cyan (top-left) + violet (top-right) radial glows
- `void-grid` — slate 1px grid lines

**Inline effects:** upload zone drag drop-shadow `0 0 10px #06b6d4`; radial cyan glow `rgba(6,182,212,0.16)`; nav active state cyan ring + glow.

---

## 8. Quick Reference — "Use This"

- **Page bg:** `#0a0a0a` · **Cards:** `#111111`
- **Text:** cream `#f5efe6` (primary), `#c9c1b4` (body), `#8a8478` (muted)
- **Accent (everything highlighted):** teal `#5eead4`
- **Status:** ok `#84d8b8`, warn `#f5b14a`, bad `#f08080`
- **Fonts:** Inter (display/body), JetBrains Mono (labels/code)
- **Borders:** white at 8% / 16% opacity · **Radii:** 4 / 8 / 12px, cards 18px
- **Charts:** teal-first sequence, transparent bg, dark tooltips, no credits
