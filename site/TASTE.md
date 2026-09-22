# Taste — what it means here, and how this site tries to earn it

Derived from reading Chinmay Karkar's RL-journey blog and gwern.net side by side,
plus what "fable-grade" generated sites get right when they restrain themselves.

## 1. Restraint is the whole game
- One column of text that respects the eye (~66–70ch), one accent color, one radius, one shadow.
- No hero gradients, no card soup, no marketing chrome. The research is the interface.
- Every element answers: does this help the reader trust the next sentence more?

## 2. Typography carries the argument
- Serif display for ideas, quiet sans for UI labels, mono only for numbers/code that must align.
- Type scale with a single ratio; line-height 1.65–1.75 for prose; tabular numerals in every table.
- Captions are sentences, not labels. A figure without a one-line reading is unfinished.

## 3. Informal first, technical on demand (progressive disclosure)
- Chinmay pattern: TL;DR → story → math in `<details>` → appendix ladder.
- Gwern pattern: abstract → essay → sidenotes → versioned appendices.
- Rule used here: each section opens with a picture or a feeling; proofs, configs,
  and shas live one click deeper. Nothing load-bearing hides, nothing casual blocks.

## 4. Visualization before numbers
- Show the shape (overlap, churn, surface) before the point estimate.
- Every % travels with its denominator; every delta with its paired test.
- Nulls are drawn, not apologized for: overlapping intervals are the figure, not a footnote.

## 5. One coherent diagram language
- Deep/source is always rust (`—`), shallow/destination always blue (`—`),
  rescued always green, regressed always clay. Same strokes, same dots, same words
  in SVG demo, PNG figures, heatmap, and tables.
- If a color means something in §3, it means the same thing in §7.

## 6. Navigation is a promise
- Sticky contents with scroll-spy, anchor on every heading, reading progress,
  prev/next rhythm, back-to-top. Gwern's lesson: long pages must be addressable.
- Mobile is not a degraded desktop: TOC collapses, sidenotes inline, tables scroll.

## 7. Both modes are first-class
- Light and dark are token swaps, not separate designs. Diagrams get
  `filter` treatment or redrawn tokens so rust/blue/green stay legible at 4.5:1.
- `prefers-color-scheme` respected, choice persisted, no flash on load.

## 8. Provenance is part of the prose
- Every figure names its generator (`analysis/gsm8k_phase2.py`, frozen `reports/data/`),
  every run names its manifest. Reproducibility is a link, not a claim.

## Self-critique log (passes)
- Pass 1 (v1): check — single accent? tables share tabular-nums? figures all have readings?
  Risk: too many PNG styles (matplotlib defaults) clash with CSS system.
  Fix: wrap PNGs in uniform `.figure` frames, add coherent HTML heatmap + SVG demo
  as the primary language, PNGs as provenance.
- Pass 2 (v2): check — dark-mode contrast on rust/blue? progress bar distraction?
  motion respect? print? Fix: brighten deep/shallow tokens in dark, thin 2px progress,
  `prefers-reduced-motion` kills animation, `@media print` flattens to ink-on-paper.
- Pass 3 (editorial): read aloud — does §1 make sense with zero ML? does §8 earn its asks?
  Cut any sentence that performs curiosity instead of practicing it.
- Pass 4 (rebuild): retire static matplotlib PNG debt entirely from main flow.
  Replaced with hand-authored, theme-aware SVGs in `figures.html` (Fig 0 Staircase,
  Fig A Overlap, Fig B Symmetric Churn, Fig C Lengths, Fig D/G Living Ground +
  Bonferroni badge, Fig H4 Schedule Indifference) mounted via local client JS.
  Restyled demo with CSS tokens; fixed dark-mode `--ink-faint` contrast to 5.2:1;
  added collapsible mobile TOC for 390px screens; ensured `@media print` expands
  all details; expanded prose ~1.8× with lived analogies, guided demo tours, and
  falsifiable experiment protocols while strictly preserving all 9 section IDs.
