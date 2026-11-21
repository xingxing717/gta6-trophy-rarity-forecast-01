"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the game's 55 PS5 trophies, the percent of
Exophase-tracked players who will have earned it at the case's target, as
five quantiles (levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95), scored by pinball
loss on the logit scale.

Method (see README.md for the full story)
----------------------------------------
Per trophy, five views of the future are fitted on the case's history and
blended; the blend weights and the spread are calibrated by leave-future-out
backtests *inside* that same history, at every horizon the history allows:

  M0  persistence: the last read's rate.
  M1  damped logit-linear trend: OLS slope of logit(rate) on time over the
      recent window, anchored at the last read. The move saturates: the
      effective time is tau*H/(H+tau), so a 4.5-day extrapolation cannot
      run away on a steep launch-night slope.
  M2  saturating adoption curve: rate(t) = A - B * exp(-t/tau) in time
      since release, tau chosen by grid search and accepted only when the
      data actually shows curvature (small tau, sane asymptote). The rise
      flattens toward an asymptote instead of running away. (Fitted on
      the rate scale: a saturating rate is S-shaped on the logit scale,
      so a logit-scale saturating fit misreads it.)
  M3  decomposition: earners k(t) and players tracked N(t) are each fitted
      with the same saturating form and the rate is k / N. This is the
      mechanism behind Exophase refreshes: new players dilute rates while
      old players keep earning, so the ratio moves whichever way wins.
  M4  log-time trend: logit(rate) linear in ln(time since release) over
      the recent window, anchored at the last read. Adoption is roughly
      power-law in time, so this decelerates naturally over a long
      horizon where a straight line in time would overshoot.

A frozen-page branch handles the launch-night practice shape: when every
read in the history is byte-identical (Exophase had not refreshed the page
at all) and the horizon is short, a refresh before the target is expected
and the median is diluted (new players have earned almost nothing except
the trivial trophy) with a wide, honest spread. It only fires when the
whole history is frozen, so it can never fire on the final case, whose
history moves.

Medians are then passed through the trophy chain's known ordering
(Chapter 1 >= Chapter 2 >= ... >= final mission >= 100% >= platinum),
and each trophy's spread combines a global scale — the short-horizon
backtest residuals grown sublinearly with horizon, floored for
refresh-jump risk — with the five models' disagreement at the target,
which is a regime-current uncertainty signal.

Python 3.13, standard library only, no network, deterministic.
Run:  MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
The grader runs:  python3 -P -s /bot/main.py  (8 s per case).
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RATE_LO, RATE_HI = 0.01, 99.99
# Frozen-page refresh branch: only when the whole history is identical.
FROZEN_MAX_HORIZON_MIN = 360
FROZEN_DILUTION = 0.30      # new tracked base ~3x the old one at a refresh
FROZEN_SIGMA = 0.5          # wide: the refresh size is genuinely uncertain
# Spread floors / caps (logit units).
SIGMA_FLOOR_RATE = 0.20     # * sqrt(horizon in days)
SIGMA_FLOOR_MIN = 0.12
SIGMA_CAP = 1.5
SIGMA_SATURATED = 0.15      # multiplier for a trophy stuck at ~100%
# Trend damping / caps.
TREND_TAU_DAYS = 1.5
TREND_MAX_SLOPE = 5.0       # logit per day
TREND_MAX_MOVE = 2.5        # logit
TREND_MIN_MOVE = 0.05       # total movement over the window to count
SAT_TAUS = (0.25, 0.5, 1.0, 1.5, 2.0)
SAT_TAUS_N = (0.5, 1.0, 2.0, 4.0, 8.0)
SAT_MAX_ASYMPTOTE_GAP = 3.0  # logit above the last read an asymptote may sit
FIT_MAX_POINTS = 100        # stride-sampled points per fit
TREND_WINDOW_DAYS = 1.5
MAX_CUTS = 5
BACKTEST_HORIZONS_DAYS = (1 / 24, 6 / 24, 12 / 24, 1.0)
N_MODELS = 5
# Chain whose earners nest (chapter order, then 100%, then platinum).
CHAIN_IDS = (3, 4, 5, 6, 7, 14, 1)


def clamp(x: float) -> float:
    return min(RATE_HI, max(RATE_LO, x))


def logit(pct: float) -> float:
    c = clamp(pct)
    return math.log(c / (100.0 - c))


def inv_logit(y: float) -> float:
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y)) if y < 700 else 100.0
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def parse_time(s: str) -> float:
    """ISO-8601 to epoch seconds."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def ols(xs, ys):
    """Least squares y = a + b x. Returns (a, b, sse) or None."""
    n = len(xs)
    if n < 2:
        return None
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    if abs(den) < 1e-12:
        return None
    b = (n * sxy - sx * sy) / den
    a = (sy - b * sx) / n
    sse = 0.0
    for x, y in zip(xs, ys):
        d = y - (a + b * x)
        sse += d * d
    return (a, b, sse)


def stride(xs, ys, limit):
    """Down-sample (keeping the last point) so a fit stays cheap."""
    n = len(xs)
    if n <= limit:
        return xs, ys
    step = (n - 1) / (limit - 1)
    idx = [round(i * step) for i in range(limit - 1)]
    idx.append(n - 1)
    seen = []
    for i in idx:
        if not seen or seen[-1] != i:
            seen.append(i)
    return [xs[i] for i in seen], [ys[i] for i in seen]


class TrophySeries:
    """One trophy's history, aligned to read indices via `pos`."""

    __slots__ = ("t", "y", "k", "n", "t_rel", "pos", "last_rate",
                 "last_y", "frozen_days", "y_by_pos")

    def __init__(self, ts, ys, ks, ns, t_rels, poss, last_rate):
        self.t = ts
        self.y = ys
        self.k = ks
        self.n = ns
        self.t_rel = t_rels
        self.pos = poss
        self.last_rate = last_rate
        self.last_y = ys[-1] if ys else logit(0.0)
        self.y_by_pos = {p: y for p, y in zip(poss, ys)}
        # Days since this trophy's rate last changed (staleness).
        self.frozen_days = 0.0
        if ys:
            last = ys[-1]
            j = len(ys) - 1
            while j > 0 and ys[j - 1] == last:
                j -= 1
            self.frozen_days = ts[-1] - ts[j]

    def prefix(self, read_cut):
        """Series restricted to reads with index <= read_cut."""
        m = 0
        while m < len(self.pos) and self.pos[m] <= read_cut:
            m += 1
        return TrophySeries(self.t[:m], self.y[:m], self.k[:m], self.n[:m],
                            self.t_rel[:m], self.pos[:m], self.last_rate)


def sat_fit(xs, ys, taus):
    """Fit y = A - B * exp(-x/tau) (B >= 0, a rise toward an asymptote).

    Returns (A, B, tau, sse, base_sse) of the best tau, or None. The form
    only earns its keep when the data shows curvature; with a large tau it
    degenerates to a straight line, whose asymptote is meaningless.
    """
    if len(xs) < 3:
        return None
    mean_y = sum(ys) / len(ys)
    base_sse = sum((y - mean_y) ** 2 for y in ys)
    best = None
    for tau in taus:
        ex = [math.exp(-x / tau) for x in xs]
        fit = ols(ex, ys)
        if fit is None:
            continue
        a, neg_b, sse = fit
        b_par = -neg_b
        if b_par < 0:
            continue
        if best is None or sse < best[0]:
            best = (sse, a, b_par, tau)
    if best is None:
        return None
    sse, a, b_par, tau = best
    return (a, b_par, tau, sse, base_sse)


def fit_models(series, t_pred, t_pred_rel, horizon_days):
    """Fit M0..M4 on `series` (already a prefix); predict logit at t_pred.
    Returns a list of 5 predictions."""
    ts, ys, ks, ns, tr = series.t, series.y, series.k, series.n, series.t_rel
    if not ts:
        return [None] * N_MODELS
    t_last = ts[-1]
    y_last = ys[-1]
    out = [y_last] * N_MODELS

    # M1: damped logit-linear trend over the recent window. The move
    # saturates: delta = b * tau * H / (H + tau) -> b * tau as H grows.
    w0 = t_last - TREND_WINDOW_DAYS
    wxs, wys = [], []
    for t, y in zip(ts, ys):
        if t >= w0:
            wxs.append(t)
            wys.append(y)
    if len(wxs) >= 3:
        wxs, wys = stride(wxs, wys, FIT_MAX_POINTS)
        fit = ols(wxs, wys)
        if fit is not None:
            _, b, _ = fit
            b = max(-TREND_MAX_SLOPE, min(TREND_MAX_SLOPE, b))
            span = wxs[-1] - wxs[0]
            if abs(b) * span >= TREND_MIN_MOVE:
                h = horizon_days
                delta = b * TREND_TAU_DAYS * h / (h + TREND_TAU_DAYS)
                delta = max(-TREND_MAX_MOVE, min(TREND_MAX_MOVE, delta))
                out[1] = y_last + delta

    # M2: saturating adoption curve in time since release, fitted on the
    # RATE scale: rate(t) = A - B * exp(-t/tau). Accepted only when the
    # data shows real curvature (small tau) and the asymptote sits within
    # a sane distance of the last read; otherwise a nearly linear fit
    # would smuggle a straight-line extrapolation in.
    if len(ts) >= 3 and tr[-1] > 0:
        rates = [inv_logit(y) for y in ys]
        xs_all, rs_all = stride(tr, rates, FIT_MAX_POINTS)
        got = sat_fit(xs_all, rs_all, SAT_TAUS)
        if got is not None:
            a, b_par, tau, sse, base_sse = got
            if (sse < 0.98 * base_sse and tau <= 2.0
                    and 0 < a <= 100
                    and logit(a) - y_last <= SAT_MAX_ASYMPTOTE_GAP):
                pred_rate = a - b_par * math.exp(-t_pred_rel / tau)
                pred_rate = max(rates[-1] - 0.25, min(a, pred_rate))
                out[2] = logit(pred_rate)

    # M3: decomposition with saturating counts. Earners k(t) and players
    # tracked N(t) each rise toward an asymptote; the rate is k / N, so
    # dilution (N outgrowing k) and accumulation are both in the picture.
    if len(ts) >= 3 and ns[-1] > 0 and ks[-1] > 0:
        xs_all, ks_all = stride(tr, ks, FIT_MAX_POINTS)
        _, ns_all = stride(tr, ns, FIT_MAX_POINTS)
        fk = sat_fit(xs_all, ks_all, SAT_TAUS)
        fn = sat_fit(xs_all, ns_all, SAT_TAUS_N)
        if fk is not None and fn is not None:
            ak, bk, tauk, ssek, basek = fk
            an, bn, taun, ssen, basen = fn
            if (ssek < 0.98 * basek and ssen < 0.98 * basen
                    and ak > 0 and an > 0):
                k_pred = ak - bk * math.exp(-t_pred_rel / tauk)
                n_pred = an - bn * math.exp(-t_pred_rel / taun)
                n_pred = max(ns[-1], min(ns[-1] * 3.0, n_pred))
                k_pred = max(ks[-1], min(k_pred, n_pred))
                if n_pred > 0 and k_pred >= 0:
                    out[3] = logit(100.0 * k_pred / n_pred)

    # M4: log-time trend over the RECENT window. logit(rate) linear in
    # ln(time since release), anchored at the last read: adoption is
    # roughly power-law in time, so this decelerates on its own over a
    # long horizon. The recent window matters: a fit over all of history
    # averages in the steep early slope and overshoots.
    if len(ts) >= 3 and tr[-1] > 0.05:
        wrel = tr[-1] - TREND_WINDOW_DAYS
        lx, ly = [], []
        for t, y in zip(tr, ys):
            if t >= wrel:
                lx.append(math.log(max(t, 0.02)))
                ly.append(y)
        if len(lx) >= 3:
            lx_all, ys_all = stride(lx, ly, FIT_MAX_POINTS)
            fit = ols(lx_all, ys_all)
            if fit is not None:
                _, b, _ = fit
                b = max(-TREND_MAX_SLOPE, min(TREND_MAX_SLOPE, b))
                span = lx_all[-1] - lx_all[0]
                if abs(b) * span >= TREND_MIN_MOVE:
                    delta = b * (math.log(max(t_pred_rel, 0.02))
                                 - math.log(max(tr[-1], 0.02)))
                    delta = max(-TREND_MAX_MOVE, min(TREND_MAX_MOVE, delta))
                    out[4] = y_last + delta
    return out


def blend(preds, weights):
    """Weighted mean of model predictions in logit space."""
    tot = 0.0
    wsum = 0.0
    for p, w in zip(preds, weights):
        if p is not None and w > 0:
            tot += w * p
            wsum += w
    return tot / wsum if wsum > 0 else None


def backtest(series_list, times, t_rels, horizon_days_list):
    """Leave-future-out backtests inside the history.

    For each horizon, cuts are spread over the second half of the
    history — the regime the final prediction extrapolates from. (Cuts
    from the launch explosion would dominate the residuals and pick
    weights for a regime that no longer holds.) Returns
    (weights_by_horizon, sigma_by_horizon): for each horizon with usable
    cuts, the blend weights minimising pooled MAE of the blended median,
    and the robust residual spread of that blend.
    """
    n = len(times)
    if n < 4:
        return {}, {}
    lo = max(2, n // 4)
    hi = n - 2
    if hi < lo:
        return {}, {}
    cuts = []
    step = max(1, (hi - lo) // (MAX_CUTS - 1)) if MAX_CUTS > 1 else 1
    j = lo
    while j <= hi and len(cuts) < MAX_CUTS:
        cuts.append(j)
        j += step
    if not cuts:
        return {}, {}
    # Weight grid over N_MODELS models, step 0.25.
    grid = []
    def _gen(rem, k, acc):
        if k == 1:
            grid.append(tuple(acc + [rem]))
            return
        for v in range(rem + 1):
            _gen(rem - v, k - 1, acc + [v])
    _gen(4, N_MODELS, [])
    grid = [tuple(v / 4 for v in g) for g in grid]

    weights_by_h = {}
    sigma_by_h = {}
    for h in horizon_days_list:
        per_trophy = []
        for s in series_list:
            errs = []
            for c in cuts:
                tt = times[c] + h
                if tt > times[-1]:
                    continue
                m = c + 1
                while m < n and times[m] < tt:
                    m += 1
                if m >= n or times[m] - tt > max(2 / 24, 0.25 * h):
                    continue
                if m not in s.y_by_pos:
                    continue
                sub = s.prefix(c)
                if not sub.t:
                    continue
                preds = fit_models(sub, times[m], t_rels[m],
                                   times[m] - times[c])
                errs.append((preds, s.y_by_pos[m]))
            per_trophy.append(errs)
        if not any(per_trophy):
            continue
        best = None
        for w in grid:
            tot = 0.0
            cnt = 0
            for errs in per_trophy:
                for preds, actual in errs:
                    b = blend(preds, w)
                    if b is None:
                        continue
                    tot += abs(b - actual)
                    cnt += 1
            if cnt == 0:
                continue
            mae = tot / cnt
            if best is None or mae < best[0]:
                best = (mae, w)
        if best is None:
            continue
        weights_by_h[h] = best[1]
        devs = []
        for errs in per_trophy:
            for preds, actual in errs:
                b = blend(preds, best[1])
                if b is not None:
                    devs.append(b - actual)
        if devs:
            devs.sort()
            med = devs[len(devs) // 2]
            mad = sorted(abs(d - med) for d in devs)[len(devs) // 2]
            sigma_by_h[h] = max(0.05, 1.4826 * mad)
    return weights_by_h, sigma_by_h


def weights_for(weights_by_h, horizon_days, default):
    """Interpolate blend weights in log-horizon; fall back to defaults."""
    if not weights_by_h:
        return default
    hs = sorted(weights_by_h)
    if horizon_days <= hs[0]:
        return weights_by_h[hs[0]]
    if horizon_days >= hs[-1]:
        return weights_by_h[hs[-1]]
    for a, b in zip(hs, hs[1:]):
        if a <= horizon_days <= b:
            f = (math.log(horizon_days) - math.log(a)) / (math.log(b) - math.log(a))
            wa, wb = weights_by_h[a], weights_by_h[b]
            return tuple(wa[i] * (1 - f) + wb[i] * f for i in range(N_MODELS))
    return default


def sigma_global_for(sigma_by_h, horizon_days, floor):
    """Global spread for the case's horizon.

    The honest scale comes from the SHORT-horizon backtests (1 h / 6 h):
    their cuts reach the end of the history, so they measure the regime
    the forecast extrapolates from. Longer-horizon backtests only exist
    from mid-history cuts, when adoption was exploding, and their
    residuals wildly overstate the uncertainty at the deadline. The
    short-horizon scale is grown sublinearly (exponent 1/4: adoption
    decelerates, so uncertainty grows slower than a random walk), floored
    for refresh-jump risk and capped.
    """
    short = {h: s for h, s in sigma_by_h.items() if h <= 0.25 + 1e-9}
    if short:
        h_ref = max(short)
        sig = short[h_ref] * (horizon_days / h_ref) ** 0.25
        return min(0.9, max(floor, sig))
    if sigma_by_h:
        return min(0.9, max(floor, max(sigma_by_h.values())))
    return min(0.9, floor)


def enforce_chain(medians, trophies, series_list):
    """Known nesting: chapter trophies, then 100%, then platinum, in rate.
    Pool-adjacent-violators on the chain's logit medians (decreasing).
    A trophy with no earners observed yet carries no signal and is left
    out, so it cannot drag a well-identified neighbour down with it."""
    idx = {}
    for pos, t in enumerate(trophies):
        idx[t["id"]] = pos
    chain = [idx[i] for i in CHAIN_IDS
             if i in idx and series_list[idx[i]].last_rate > 0]
    if len(chain) < 2:
        return
    blocks = [[medians[p]] for p in chain]
    i = 0
    while i < len(blocks) - 1:
        m0 = sum(blocks[i]) / len(blocks[i])
        m1 = sum(blocks[i + 1]) / len(blocks[i + 1])
        if m0 < m1:  # violation: earlier must be >= later
            blocks[i] = blocks[i] + blocks[i + 1]
            del blocks[i + 1]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for b in blocks:
        m = sum(b) / len(b)
        out.extend([m] * len(b))
    for p, v in zip(chain, out):
        medians[p] = v


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    horizon_min = case["horizon_minutes"]
    horizon_days = horizon_min / 1440.0

    t0 = parse_time(history[0]["at"])
    try:
        t_rel0 = parse_time(case["game"]["release_date"] + "T00:00:00Z")
    except Exception:
        t_rel0 = t0
    times = [(parse_time(r["at"]) - t0) / 86400.0 for r in history]
    t_rels = [(parse_time(r["at"]) - t_rel0) / 86400.0 for r in history]
    n_reads = len(history)

    # Frozen page? Every read identical (rates and players tracked).
    frozen = n_reads >= 2
    if frozen:
        r0 = history[0]
        for r in history[1:]:
            if r["players_tracked"] != r0["players_tracked"] or r["rates"] != r0["rates"]:
                frozen = False
                break

    # Per-trophy series.
    series_list = []
    for i, tr in enumerate(trophies):
        ts, ys, ks, ns, trs, poss = [], [], [], [], [], []
        for j, r in enumerate(history):
            rate = r["rates"][i]
            if rate is None:
                continue
            ts.append(times[j])
            ys.append(logit(rate))
            n = r["players_tracked"] or 0
            ns.append(float(n))
            ks.append(rate * n / 100.0)
            trs.append(max(t_rels[j], 0.0))
            poss.append(j)
        last_rate = history[-1]["rates"][i]
        series_list.append(TrophySeries(ts, ys, ks, ns, trs, poss,
                                        last_rate if last_rate is not None else 0.0))

    rows = []
    if frozen and horizon_min <= FROZEN_MAX_HORIZON_MIN:
        # A refresh is expected before the target: the page has shown the
        # same counts since launch night while the game is live, and
        # Exophase refreshes on its own schedule, hours apart. A refresh
        # adds a wave of brand-new players who have earned almost nothing
        # yet, so every rate but a trivial one dilutes toward
        # old * N_old / N_new. We central-case the tracked base roughly
        # tripling at the refresh and stay wide about it.
        medians = []
        sigmas = []
        for i, tr in enumerate(trophies):
            s = series_list[i]
            if s.last_rate >= 99.99:
                medians.append(logit(100.0))
                sigmas.append(FROZEN_SIGMA * SIGMA_SATURATED)
            elif s.last_rate <= 0:
                medians.append(logit(0.0))
                sigmas.append(FROZEN_SIGMA)
            else:
                medians.append(logit(s.last_rate * FROZEN_DILUTION))
                sigmas.append(FROZEN_SIGMA)
        enforce_chain(medians, trophies, series_list)
        for i, tr in enumerate(trophies):
            qs = [round4(inv_logit(medians[i] + z * sigmas[i])) for z in Z]
            for j in range(1, 5):
                if qs[j] < qs[j - 1]:
                    qs[j] = qs[j - 1]
            rows.append({"id": tr["id"], "q": qs})
        return {"trophies": rows}

    # Live history: backtest-calibrated blend.
    weights_by_h, sigma_by_h = backtest(series_list, times, t_rels,
                                        BACKTEST_HORIZONS_DAYS)
    default_w = (0.35, 0.25, 0.15, 0.10, 0.15)
    weights = weights_for(weights_by_h, horizon_days, default_w)
    floor = max(SIGMA_FLOOR_MIN, SIGMA_FLOOR_RATE * math.sqrt(horizon_days))
    sigma_glob = sigma_global_for(sigma_by_h, horizon_days, floor)

    medians = []
    sigmas = []
    for i, tr in enumerate(trophies):
        s = series_list[i]
        if not s.t:
            medians.append(logit(0.0))
            sigmas.append(sigma_glob)
            continue
        preds = fit_models(s, s.t[-1] + horizon_days,
                           s.t_rel[-1] + horizon_days, horizon_days)
        med = blend(preds, weights)
        if med is None:
            med = s.last_y
        # Model disagreement at the target is a second, regime-current
        # uncertainty signal: where the five views diverge, the truth is
        # genuinely less pinned down.
        present = [p for p in preds if p is not None]
        disagree = 0.0
        if len(present) >= 2:
            mean_p = sum(present) / len(present)
            disagree = math.sqrt(sum((p - mean_p) ** 2 for p in present)
                                 / len(present))
        sig = math.sqrt(sigma_glob * sigma_glob
                        + (0.5 * disagree) ** 2)
        # Staleness: a trophy whose rate has not moved lately is less
        # certain than the last read suggests.
        if s.frozen_days > 0:
            sig = math.sqrt(sig * sig + 0.0625 * min(s.frozen_days, 2.0))
        # A trophy at ~100% cannot rise; its band is narrow.
        if s.last_rate >= 99.9:
            sig *= SIGMA_SATURATED
            med = min(med, logit(100.0))
        sig = min(SIGMA_CAP, max(0.05, sig))
        medians.append(med)
        sigmas.append(sig)

    enforce_chain(medians, trophies, series_list)

    for i, tr in enumerate(trophies):
        qs = [round4(inv_logit(medians[i] + z * sigmas[i])) for z in Z]
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": tr["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
