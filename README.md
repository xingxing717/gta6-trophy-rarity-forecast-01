[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xingxing717/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/xingxing717/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/c/gta6-trophy-rarity-forecast-01)
Built for the [make.fun competition](https://make.fun/c/gta6-trophy-rarity-forecast-01).

# GTA VI Trophy Rarity Forecast

Entry for make.fun competition `gta6-trophy-rarity-forecast-01`: forecast,
for each of GTA VI's 55 PS5 trophies, the percent of Exophase-tracked players
who will have earned it at the case's target, as five quantiles per trophy
(levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by pinball loss on the
logit scale. The final case forecasts 2026-11-26T00:00:00Z from everything
recorded before the 2026-11-21T11:30:00Z deadline — a 6,510-minute horizon.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

The grader runs `python3 -P -s /bot/main.py` (Python 3.13, standard library
only, no network, 8 s per case). `main.py` reads every `<id>.in` in the
input directory and writes `<id>.out` to the output directory:
`{"trophies": [{"id": 1, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Method

Everything is fitted per case, from that case's history alone — nothing is
hard-coded to the practice cases, and no answer is ever read or assumed.

### Five views, blended (the median)

Per trophy, on the logit scale:

- **M0 — persistence.** The last read's rate. The honest default when the
  history says nothing.
- **M1 — damped logit-linear trend.** OLS slope of logit(rate) on time over
  the recent 1.5 days, anchored at the last read. The move saturates: the
  effective time is `1.5 * H / (H + 1.5 days)`, so a 4.5-day extrapolation
  cannot run away on a steep launch-night slope. Capped at 2.5 logit
  units of total move.
- **M2 — saturating adoption curve, on the rate scale.**
  `rate(t) = A - B * exp(-t/tau)` in time since release, tau chosen by grid
  search over 0.25–2 days and accepted only when the data actually shows
  curvature (small tau, sane asymptote). Fitted on the rate scale because
  a saturating rate is S-shaped on the logit scale, so a logit-scale
  saturating fit misreads it. The rise flattens toward an asymptote
  instead of running away over a week.
- **M3 — decomposition.** Earners `k(t)` and players tracked `N(t)` are
  each fitted with the same saturating form and the rate is `k / N`. This
  is the mechanism behind Exophase refreshes: new players dilute rates
  while old players keep earning, so the ratio moves whichever way wins.
- **M4 — log-time trend.** logit(rate) linear in ln(time since release)
  over the recent window, anchored at the last read. Adoption is roughly
  power-law in time, so this decelerates on its own over a long horizon
  where a straight line in time would overshoot.

The blend weights are **not** fixed: they are chosen per horizon by
leave-future-out backtests inside the case's own history (see below).
Medians then pass through the trophy chain's known nesting —
Chapter 1 >= Chapter 2 >= Chapter 3 >= Chapter 4 >= final mission >=
100% completion >= platinum (earners of a later step are a subset of
earners of the earlier one) — enforced by pool-adjacent-violators on the
logit medians. This only binds when a model violates a known ordering.
A trophy with no earners observed yet is left out of the chain, so a
zero-signal trophy cannot drag a well-identified neighbour down with it.

### Frozen-page branch (launch-night practice shape)

When **every** read in the history is byte-identical (same players
tracked, same rates — Exophase had not refreshed the page at all) and the
horizon is short (<= 6 h), a refresh before the target is expected: the
rules state Exophase refreshes on its own schedule, hours apart, while
the game is live. A refresh adds a wave of brand-new players who have
earned almost nothing yet, so every rate but a trivial one dilutes toward
`old * N_old / N_new`. The median central-cases the tracked base roughly
tripling at the refresh (factor 0.30) with a wide spread (sigma 0.5
logit), and the 100% trophy stays at 100%. The branch only fires when
the whole history is frozen, so it can never fire on the final case,
whose two days of reads move.

### Spread (the quantiles)

Spread is calibrated, not assumed — but calibrated on the *right* regime.
The leave-future-out backtests that pick the blend weights also yield
residuals per horizon (1.4826 * MAD of the blended model's residuals).
Long-horizon backtests only exist from mid-history cuts, when adoption
was exploding, and their residuals wildly overstate the uncertainty at
the deadline (in simulation: sigma 2.5 at 24 h, while the true 4.5-day
median error is ~0.36 logit). So the global scale is taken from the
short-horizon backtests (1 h / 6 h), whose cuts reach the end of the
history and measure the regime the forecast extrapolates from, grown
sublinearly with horizon (exponent 1/4: adoption decelerates), with a
floor of `0.20 * sqrt(horizon in days)` (min 0.12) for quantisation,
staleness and refresh-jump risk, and a cap of 0.9. Each trophy then adds
its own model disagreement at the target — the standard deviation of
the five views' logit predictions, halved — as a second, regime-current
uncertainty signal, plus extra variance when its rate has not moved
lately. A trophy at ~100% gets a narrow band (it cannot rise).

### Backtests

Cuts are spread over the second half of the history (up to 5 cuts) — the
regime the final prediction extrapolates from, not the launch explosion.
At each cut, every model is fitted on reads up to the cut and scored
against the first read at/after cut + horizon, for horizons 1 h / 6 h /
12 h / 24 h as far as the history allows. Blend weights per horizon
minimise the pooled mean absolute logit error of the blended median over
a 0.25 weight grid; weights for the case's horizon interpolate in
log-horizon between the nearest tested horizons (beyond the longest
tested horizon, the longest horizon's weights are used).

## Why this beats the house baselines

- The persistence baseline (starter) assumes rates never move and spreads
  `0.5 * sqrt(days)` — on the practice cases it scores ~1.51, because the
  refresh moved every rate outside its narrow band.
- The curve baseline fits one straight line per trophy; on flat history
  it is persistence byte for byte.
- This entry expects the refresh when the page is provably stale,
  extrapolates real adoption curves (not straight lines) when the history
  moves, models dilution and accumulation separately, and sizes its bands
  from how wrong the same models were at the same horizons in the recent
  regime — instead of from a fixed constant.

## Verification

- Runs on both practice cases (`p00`, `p01`); output is valid JSON,
  55 rows, monotone quantiles in (0, 100], deterministic across runs
  (byte-identical on re-run, including under `python3 -P -s` with
  `PYTHONHASHSEED=0`), ~0.4 s for both practice cases and ~0.9 s for a
  332-read final-like case — well under the 8 s cap.
- **True practice pinball loss cannot be computed locally**: the kit ships
  no answers and `makefun grade-local` for this competition is
  server-only ("its scorer and reference data are not published in the
  kit"). The practice history is perfectly flat (7–8 identical reads over
  159–189 minutes), so in-history backtests are uninformative there by
  construction: only the 1 h horizon is even testable (6 h / 24 h need a
  longer history than the practice cases have), and every model scores
  ~0 against later flat reads. The honest checks are the two scenarios
  below.
- Synthetic refresh scenario (built from the rules' own published
  figures: 638 -> 3,868 players tracked, 943 -> 4,276 trophies earned,
  every rate but the 100% one down; new-cohort trophies allocated
  proportionally to old counts): the persistence baseline scores
  1.404 loss / 0.68-point typical miss on it — matching the published
  1.51 / 0.68 — while this entry's frozen-page branch scores **0.164
  loss / 0.07-point typical miss, 90% coverage 0.98**.
- Synthetic final-like cases (332 simulated reads over two days with
  refresh jumps and logistic adoption, scored at +4.5 days):
  - Scenario 1 (fast adoption, 4 h refreshes): this entry **0.301 loss /
    4.16-point typical miss / 0.96 coverage** vs baseline 0.653 / 14.03 /
    0.85. Backtest weights at 6 h: (0.75, 0.25, 0, 0, 0); at 24 h:
    (0.5, 0.5, 0, 0, 0); backtest sigma 0.161 at 6 h, 0.448 at 12 h,
    2.507 at 24 h (mid-launch regime — see Spread; not used directly).
    The entry's weights are near-oracle for this scenario (best grid
    weights score 0.296 vs 0.297).
  - Scenario 2 (slower adoption, 6 h refreshes, slower player growth):
    this entry **0.512 / 6.52 / 0.85** vs baseline 1.070 / 15.84 / 0.67.
- Edge cases verified without crashing: single-read history, two reads,
  `null` rates, a trophy with all-`null` rates, zero rates, 30-day
  horizon.

## Known limits

- A trophy with zero earners in all of history (e.g. 100% completion if
  nobody has it by the deadline) cannot be forecast from data; the entry
  predicts ~0 and keeps it out of the chain so it cannot drag the
  platinum down with it.
- The practice cases' ~0 losses on the dev board are not reproducible by
  any honest forecaster from the flat history: the answers are the
  refreshed page, whose per-trophy rates are not derivable from the case
  data (the rules publish only the totals). This entry's frozen-page
  branch is the honest version of that insight — it expects the refresh,
  dilutes, and stays wide — and scores ~0.16 on the synthetic refresh
  answer above, vs the baseline's ~1.4.

## Files

- `main.py` — the whole entry (standard library only).
- `README.md` — this file.
- `LICENSE` — MIT.
