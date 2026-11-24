# Agon UI full redesign — spec

Status: approved direction (Dyl, 2026-11-24): **Hearth & Hide wins the chrome call, but a
reskin is not the ask.** Both candidates were token overrides on the same structure; the
structure is what's weak. This spec is the full redesign: information architecture, new
components, depth where the sim deserves it, UX testing, and implementation requirements.
Supersedes the chrome-pick section of DESIGN-revamp.md (A "Field Notes" is dead; its one
good part — the stress-test method — moves into the testing strategy below).

## 1. Direction

**Material language: Hearth & Hide v2.** The UI is made of what the world is made of —
hide panels, bone ink, carved pictographs, ember accents. Not a reskin of the current
panels: a rebuilt structure wearing that skin. The current chrome's real failures are
structural, and no token swap touches them:

- **Dev chrome and player chrome are interleaved.** Generator button, sandbox toggles,
  Signs/Names/Shadows/Quiet sit next to Resume and the clock. The observer's instrument
  panel is mixed with the workshop's.
- **Flat information everywhere, depth nowhere.** Six bottom tabs + four taskbar panels +
  floating MAP card ≈ eleven surfaces, every one a shallow list. The sim underneath has
  warmth graphs (r55), grudges and defections (r57), treaties (r58), work identities (r54),
  genetics, wounds-by-body-part, per-species ecology — none of it reachable.
- **Alerts are disconnected.** Top-right toasts name a subject you then hunt for by eye.
- **The story is invisible.** A 5-entry log window is the only narrative surface in a game
  whose entire product is the narrative.

**Principles** (every component gets judged against these):
1. **Observer-first.** No affordance may read as a command. Surfaces answer "what is
   happening and why", never "what should I order".
2. **The world is the protagonist.** Chrome holds the edges; the center stays world. Panels
   are furniture at the hearth, not windows over it.
3. **Depth on demand.** Glance → study → trace: every surface has a 1-second read, a
   10-second read, and a click-through to the underlying history.
4. **Everything shown is measured.** A UI element may only display state the sim actually
   has; no decorative fake meters. (Tech panel shows discovered techs only, relationship
   view shows real relc values, etc.)
5. **Diegetic, with restraint.** Hide and bone set the mood; data stays in high-contrast
   ink on quiet grounds. Richness in the frame, never in the reading surface.

## 2. Information architecture

Three layers replace the current eleven-surface sprawl:

- **Ambient (the world).** The map, its overlays, and in-world labels/bubbles. Everything
  here must be legible at a glance and honest to sim state.
- **Glance (the HUD).** One top vitals bar, one alert stack, one roster strip, one
  time/season block. Total fixed chrome ≤ 3 edges; the fourth (left) stays clear.
- **Study (the panels).** Exactly one study panel open at a time, docked right:
  Inspector · Chronicle · Knowledge · Structures · Ecology · Colony. Bottom tabs die;
  their content moves into these six. The MAP card folds into the time block.
- **Workshop (dev drawer).** Generator, sandbox toggles, render toggles, LLM settings —
  behind a wrench, visually distinct (plain dark, no hide), collapsed by default.

## 3. Component inventory

### 3.1 HUD
| Component | Replaces | Content | Depth hook |
|---|---|---|---|
| **Vitals bar** (top) | legend card | colonists · meals · water · warmth risk · mood median, each a pictograph + number + 24h spark | click → Colony panel |
| **Alert stack** (top-right) | current toasts | severity-ramped (note/warn/crisis), grouped by subject, each with a **jump-to** that pans camera to the subject and pulses it | click → subject's Inspector |
| **Roster strip** (top) | portrait row | chip = portrait + need micro-bars + mood ring + status glyph (asleep/hurt/fighting/exiled) | click → Inspector; hover → name + current act |
| **Time block** (bottom-right) | MAP card | season dial (the existing bar, kept), clock, weather, speed controls, minimap | unchanged behaviors |

### 3.2 Study panels (the depth)
- **Inspector v2** — tabs: Body (existing x-ray stays, it's good) · Mind (pers PLEX
  position, mood drivers, beliefs) · **Bonds** (warmth graph from `relc`: who they sort
  with, grudges in red, kin links; this data has existed since r55 with zero UI) ·
  Work (stScore tallies, xp, tool provenance) · **Story** (that colonist's own event
  history: wounds, defections, births, kills — filtered chronicle).
- **Chronicle** — the colony's story as a timeline: era markers (treaty struck, defection,
  first fire, deaths), filter by kind/colonist/camp, written in the log's voice. Backed by
  a persistent event store (the current `s.log` 5-entry window is a render buffer, not a
  history — a real ring buffer of notable events, capped ~500, ships with this panel).
- **Knowledge** — emergent tech support: discovered techs as carved tokens on a hide map,
  **undiscovered techs do not render** (fog of knowledge — the tree's shape is not leaked);
  per-tech: who had the insight, at what tick, what it unlocked, tools crafted since.
  Grows with the four-pillars/tech rounds; the panel ships reading `store.techs` + insight
  events as they exist today.
- **Structures** — built things as the colony's ledger: walls/doors/roofs by material,
  rooms detected (enclosed + roofed regions), builder credits (a.work.build), build
  timeline. Grows into the EB1 need-driven-building rounds' observer surface.
- **Ecology** — per-species population curves (eco data exists), predation feed (X kills Y
  events), territory notes. The lion-elephant moment belongs here as a feed entry.
- **Colony** — stocks (current STOCK), goals, work priorities, rivals/treaty state
  (rel score + truce history).

### 3.3 Ambient
- **Overlay picker** (small, bottom-left by taskbar): fertility · shelter/roof · warmth
  of ground (fire radius) · territory. Fertility overlay finally makes rich/grav patches
  *interpretable* (the dark-circles confusion dies for good: toggle shows green=fertile).
- **In-world labels**: current names kept; add act captions option (exists) and
  relationship beams on Inspector-Bonds hover (draw warm/cold lines to bonded colonists).
- **Event grammar** (continues this round's work): every sim event that matters gets a
  legible motion — attack lunge/brace (done), build hammer+chips (done), repelled-attack
  recoil, death fall (exists), defection walk-out with pause-and-look-back (queued).

### 3.4 Visual language (Hearth & Hide v2 tokens)
- Ground: `#2a1d14`; panel hide: `#4a3626` with 1.5px `#241709` hewn border (corner cuts,
  not radii); raised: `#57402c`.
- Ink: bone `#ece2cf`; muted `#b5a48c`; carved-line icon set, 2px stroke, 20×20 grid.
- Accent: ember `#d8a43b` (story beats, headers, season dial); blood `#b5543b` (crisis
  only); moss `#7fa06a` (ok/positive only). Semantic colors never decorate.
- Type: serif display for panel titles/chronicle voice (Georgia stack), humanist sans for
  data rows; min 11px data, 13px body — bigger than today, fewer words.
- Texture discipline: hide grain on panel frames only; reading surfaces are flat quiet
  color. (B's stated risk — "richness needs restraint to avoid mud" — becomes a rule.)
- Motion: panels settle (120ms ease-out, 4px rise); ember pulse on new chronicle beats;
  nothing loops in chrome except the fire.

## 4. UX testing strategy

1. **Legibility matrix (automated, exists in embryo).** Playwright grid: {4 biomes:
   darkest/brightest/wettest/snowiest} × {day, night, rain, storm} × {each HUD component,
   each panel over world}. Screenshot set + sampled text/background contrast ≥ 4.5:1
   (WCAG AA) for data text, ≥ 3:1 for glyphs. The chrome_A_snow stress test, generalized
   and kept as a regression suite (scratchpad/chromecmp harness grows into this).
2. **Observer task tests (the 10x UX measure).** Scripted scenarios on known seeds, a
   human (Dyl) answers with a stopwatch: "why did A07 die?" · "who defected and why?" ·
   "which tech did the colony get this year and from whom?" · "is the treaty holding?" ·
   "which colonists are bonded?" Targets: every answer reachable in ≤ 3 clicks and ≤ 20s,
   from cold. Today most of these are *unanswerable at any depth* — that's the 10x.
3. **First-contact comprehension.** 5 minutes, no guidance, fresh person: list what they
   believe is happening. Score against ground truth (the sim log). Repeat per phase.
4. **Performance budget.** Full HUD + one panel at 4x speed: render ≤ 8ms/frame on the
   dev box, no layout thrash (Chrome tracing), DOM node count budget per panel (≤ 400).
5. **Screenshot regression.** Per-component fixture states (alert stack at 1/3/8 alerts,
   roster at 4/13/30 colonists, chronicle empty/full) diffed per commit.
6. **Honesty audit (Agon-specific).** Every number in every new panel traced to the sim
   field it reads; anything unverifiable gets cut. No invented meters. Checked per phase.

## 5. Implementation requirements

- **Single-file discipline**: components are template partials + token CSS inside
  Main.dc.html; no frameworks, no build-step additions.
- **Render-only**: panels read state; zero sim writes from UI code. Panel hashes vs
  ab_treaty arm B stay byte-exact. The one exception each phase must declare: the
  Chronicle's event ring buffer is new *state* (not behavior) — it writes a bounded
  `s.events` list that nothing in the sim reads. Verified inert by 3-world hash check
  (hash excludes it: `h(agents+store)` unchanged) + a no-behavior-reads grep.
- **One phase per round**, each committed with screenshots in the legibility matrix and
  the TODO entry. No phase starts until the previous one's regression set is green.
- **Dev/player split is Phase 0** and ships before any beauty work — it's the cheapest
  structural win and unblocks honest screenshots.

### Phases
| Phase | Scope | Rounds |
|---|---|---|
| 0 | Token foundation (H&H v2), dev drawer, kill bottom-tab/taskbar sprawl (move, don't rebuild) | 1 |
| 1 | HUD: vitals bar, alert stack w/ jump-to, roster chips, time block merge | 1–2 |
| 2 | Inspector v2 (Bonds tab first — data is waiting), event ring buffer + Chronicle | 2 |
| 3 | Knowledge + Structures (co-ships with four-pillars/tech sim rounds) | 2 |
| 4 | Ecology panel, overlay picker, relationship beams, remaining event grammar | 1–2 |

### Queued sim-side dependencies (flagged, panel-tested, NOT render-only)
- **Build duration**: walls currently appear in one tick (that's why construction read as
  broken even before the missing animation). Segments should take N ticks with visible
  progress states (frame → infill → done). Behavior change: exposure windows, workwarm,
  build rates → `nobuildt` opt-out + 24-world panel.
- Repelled-attack marker (prey drives predator off — currently invisible in state),
  defection walk-out beat, four pillars audit, tech emergence rounds: per DESIGN-revamp.md §E.
