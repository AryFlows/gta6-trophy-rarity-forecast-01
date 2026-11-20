// Frost Rarity Forecast v3 — GTA VI trophy earn-rate forecaster (make.fun).
//
// Model: forecast COUNTS, not rates. A trophy's earn rate is
//   rate = 100 * (players who earned it) / (players tracked),
// so the program forecasts the two counts separately and divides them.
// That split matters because the two counts move for different reasons:
// tracked players grow with launch-week sales, while each trophy's earned
// count grows as players progress through the game. Fitting the rate
// directly confuses the two.
//
// Two regimes, picked by how much the history actually moved:
//
//  1. RICH history (>= 3 distinct reads): log-time trend fit. For each
//     trophy, ln(earned + 0.5) is regressed on ln(1 + hours since release)
//     over the last 48h of distinct reads; the tracked-players series gets
//     the same fit. Each trophy's slope is shrunk toward the players
//     slope (a trophy with no signal keeps a flat rate), with the shrink
//     weight growing with the trophy's earned count, so rare trophies
//     borrow strength instead of chasing noise. Forecasts anchor on the
//     last observed value plus slope * distance, and the spread combines
//     the fits' residuals (leveraged for extrapolation) with a horizon
//     floor. Exophase refreshes in jumps, so the fit uses only distinct
//     reads: repeated identical reads carry no new information.
//
//  2. FLAT history (<= 2 distinct reads): the page has not refreshed in
//     view, so no trend can be estimated. The forecast then uses the
//     refresh mechanism the kit documents: Exophase refreshes a game's
//     counts every few hours, and a refresh dilutes every rate, because
//     the newly tracked players hold fewer trophies. When a full refresh
//     interval elapses between the page going stale and the target, the
//     central scenario is one refresh: the median shifts down by
//     ln(refresh growth), where the growth factor comes from refresh
//     jumps seen in the history when there are any, else from the
//     kit-documented launch-night refresh (638 -> 3,868 players, x6.06)
//     decayed as the launch ages. Trophies near 100% are exempt: every
//     new player earns those immediately, so dilution does not touch
//     them. The spread stays honest about whether the refresh lands.
//
// Scoring is pinball loss on the logit scale, so every step above works
// in logits and the quantiles are symmetric normal bands around the
// logit median unless the regime says otherwise.
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const Z = [-1.6448536269514722, -0.6744897501960817, 0, 0.6744897501960817, 1.6448536269514722];
const HOUR_MS = 3600000;

// --- small math helpers (logit scale throughout) ---

function clampRate(x: number): number {
  if (!Number.isFinite(x)) return 1;
  return Math.min(99.99, Math.max(0.01, x));
}

function logit(pct: number): number {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
}

function invLogit(l: number): number {
  if (l > 20) return 99.99;
  if (l < -20) return 0.01;
  return clampRate(100 / (1 + Math.exp(-l)));
}

function round4(x: number): number {
  return Math.round(x * 10000) / 10000;
}

// Least squares of y on x. Returns slope, intercept, residual std,
// mean of x and sum of squared x-deviations (for the leverage term).
function ols(xs: number[], ys: number[]): { slope: number; intercept: number; resid: number; xbar: number; sxx: number } {
  const n = xs.length;
  let sx = 0, sy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; }
  const xbar = sx / n, ybar = sy / n;
  let sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sxx += (xs[i] - xbar) ** 2; sxy += (xs[i] - xbar) * (ys[i] - ybar); }
  const slope = sxx > 1e-12 ? sxy / sxx : 0;
  const intercept = ybar - slope * xbar;
  let ss = 0;
  for (let i = 0; i < n; i++) { const r = ys[i] - (intercept + slope * xs[i]); ss += r * r; }
  const resid = n > 2 ? Math.sqrt(ss / (n - 2)) : 0.3;
  return { slope, intercept, resid, xbar, sxx };
}

// --- case handling ---

type Read = { tMs: number; x: number; players: number; rates: (number | null)[] };

function logTime(tMs: number, releaseMs: number): number {
  const hours = (tMs - releaseMs) / HOUR_MS;
  return Math.log(Math.max(0.05, 1 + hours));
}

// Collapse runs of identical reads (same players, same rates) to the
// first read of each run: Exophase shows a stale page between refreshes,
// and the repeats carry no new information for a trend fit.
function distinctReads(history: any[], releaseMs: number): Read[] {
  const out: Read[] = [];
  let prevKey = "";
  for (const h of history) {
    const n = Number(h.players_tracked);
    if (!Number.isFinite(n) || n <= 0) continue;
    const key = n + "|" + JSON.stringify(h.rates);
    if (key === prevKey) continue;
    prevKey = key;
    const tMs = Date.parse(h.at);
    out.push({ tMs, x: logTime(tMs, releaseMs), players: n, rates: h.rates as (number | null)[] });
  }
  return out;
}

function quantiles(medianPct: number, sigma: number): number[] {
  const m = logit(medianPct);
  const q = LEVELS.map((_, i) => round4(invLogit(m + Z[i] * sigma)));
  for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  return q;
}

// --- regime 2: flat history, refresh-dilution prior ---

function flatForecast(
  lastRates: (number | null)[],
  firstReadMs: number,
  checkpointMs: number,
  horizonHours: number,
  releaseMs: number,
  observedJumps: number[],
): number[][] {
  // How stale is the displayed page? At least the span of the history.
  const staleHours = Math.max(0, (checkpointMs - firstReadMs) / HOUR_MS) + 0.25;
  // Refreshes land "hours apart"; the launch-night interval the kit
  // documents ran longer than 4.2h, so 4.5h is the working prior.
  const intervalHours = 4.5;
  const refreshesDue = (staleHours + horizonHours) / intervalHours;
  const weight = Math.min(1, Math.max(0, refreshesDue));

  // Growth factor of one refresh: prefer jumps actually seen in this
  // case's history; else the kit's launch-night refresh (x6.06),
  // decayed as the launch ages (early refreshes are the huge ones).
  let growth: number;
  if (observedJumps.length > 0) {
    const sorted = [...observedJumps].sort((a, b) => a - b);
    growth = sorted[Math.floor(sorted.length / 2)];
  } else {
    const ageHours = Math.max(0, (checkpointMs - releaseMs) / HOUR_MS);
    growth = 1 + 5.062 * Math.exp(-ageHours / 36);
  }
  growth = Math.min(8, Math.max(1.05, growth));

  // 0.9 tempers the shift for the chance the refresh does not land
  // before the target and for uncertainty in the growth factor.
  const shift = -Math.log(growth) * weight * 0.9;
  const days = horizonHours / 24;
  const sigma = Math.min(0.9, Math.max(0.5, 0.5 + 0.1 * Math.sqrt(days)));

  return lastRates.map((r) => {
    const rate = r === null || !Number.isFinite(r) ? 1 : r;
    // Near-100% trophies are earned by every new player too, so
    // dilution leaves them alone; the (1 - r) factor encodes that.
    const shifted = invLogit(logit(rate) + shift * (1 - rate / 100));
    return quantiles(shifted, sigma);
  });
}

// --- regime 1: rich history, counts decomposition + log-time trends ---

function trendForecast(
  reads: Read[],
  checkpointMs: number,
  targetMs: number,
  releaseMs: number,
  horizonHours: number,
  trophyCount: number,
): number[][] {
  const xTarget = logTime(targetMs, releaseMs);
  // Fit on the recent regime only: launch-night growth is explosive
  // and would dominate a full-history fit; the last 48h carries the
  // slope that extrapolates, while dropping the wildest launch-night
  // reads (the flat 638-player page and the first refresh). A shorter
  // window starves the fit (24h tested worse on synthetic finals);
  // the full history lets the launch spike distort the slope.
  // Fall back to all reads if too few.
  const windowMs = 48 * HOUR_MS;
  let fit = reads.filter((r) => checkpointMs - r.tMs <= windowMs);
  if (fit.length < 3) fit = reads;
  const last = reads[reads.length - 1];

  // Tracked-players trend: ln(N) on log-time.
  const nFit = ols(fit.map((r) => r.x), fit.map((r) => Math.log(r.players)));
  const logNTarget = Math.log(last.players) + nFit.slope * (xTarget - last.x);
  const nLeverage = 1 + (nFit.sxx > 1e-12 ? (xTarget - nFit.xbar) ** 2 / nFit.sxx : 1);
  const nVar = Math.min(0.45, nFit.resid) ** 2 * Math.min(4, nLeverage);

  const days = horizonHours / 24;
  const floorVar = (0.22 * Math.sqrt(days)) ** 2;

  const rows: number[][] = [];
  for (let i = 0; i < trophyCount; i++) {
    // Earned counts for this trophy over the fit window.
    const xs: number[] = [];
    const ys: number[] = [];
    let countSum = 0, countN = 0;
    for (const r of fit) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined || !Number.isFinite(rate)) continue;
      const c = (rate / 100) * r.players;
      xs.push(r.x);
      ys.push(Math.log(c + 0.5));
      countSum += c; countN++;
    }
    const lastRate = last.rates[i];
    const lastCount = lastRate === null || lastRate === undefined
      ? 0 : (lastRate / 100) * last.players;
    const meanCount = countN > 0 ? countSum / countN : 0;

    let logCountTarget: number;
    let residVar = 0.3 * 0.3;
    if (xs.length >= 3) {
      const f = ols(xs, ys);
      // Shrink the trophy's slope toward the players slope: with no
      // signal a trophy's rate is flat (its count grows with the
      // player base). Weight grows with the trophy's earned count.
      const w = meanCount / (meanCount + 10);
      let slope = nFit.slope + w * (f.slope - nFit.slope);
      slope = Math.min(nFit.slope + 1.5, Math.max(nFit.slope - 1.5, slope));
      logCountTarget = Math.log(lastCount + 0.5) + slope * (xTarget - last.x);
      const lev = 1 + (f.sxx > 1e-12 ? (xTarget - f.xbar) ** 2 / f.sxx : 1);
      residVar = Math.min(0.45, f.resid) ** 2 * Math.min(4, lev);
    } else if (xs.length >= 1) {
      // Too few points for a slope: hold the count's share of players.
      logCountTarget = Math.log(lastCount + 0.5) + nFit.slope * (xTarget - last.x);
    } else {
      // Trophy never seen in the window: flat rate, wide spread.
      rows.push(quantiles(lastRate ?? 1, 1.0));
      continue;
    }

    let countTarget = Math.exp(logCountTarget) - 0.5;
    if (!Number.isFinite(countTarget) || countTarget < 0.02) countTarget = 0.02;
    let rateTarget = 100 * (countTarget / Math.exp(logNTarget));
    // Sanity cap: no trophy's median moves more than ~20x in rate.
    const lastLogit = logit(lastRate ?? 1);
    rateTarget = invLogit(Math.min(lastLogit + 3, Math.max(lastLogit - 3, logit(rateTarget))));

    const sigma = Math.min(1.4, Math.max(0.35, Math.sqrt(residVar + nVar + floorVar)));
    rows.push(quantiles(rateTarget, sigma));
  }
  return rows;
}

// --- per-case driver ---

function forecastCase(c: any): { id: number; q: number[] }[] {
  const releaseMs = Date.parse((c.game?.release_date ?? "2026-11-19") + "T00:00:00Z");
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonHours = (Number(c.horizon_minutes) || 0) / 60;
  const trophies = c.trophies as { id: number }[];
  const history = (c.history ?? []) as any[];

  const reads = distinctReads(history, releaseMs);
  const lastRead = history.length > 0 ? history[history.length - 1] : null;
  const lastRates: (number | null)[] = lastRead
    ? (lastRead.rates as (number | null)[])
    : trophies.map(() => null);

  let rows: number[][];
  if (reads.length <= 2) {
    // Flat history: refresh-dilution prior. Any jumps seen between
    // the (at most two) distinct reads inform the growth factor.
    const jumps: number[] = [];
    for (let i = 1; i < reads.length; i++) {
      const ratio = reads[i].players / reads[i - 1].players;
      if (ratio > 1.02) jumps.push(ratio);
    }
    const firstMs = history.length > 0 ? Date.parse(history[0].at) : checkpointMs;
    rows = flatForecast(lastRates, firstMs, checkpointMs, horizonHours, releaseMs, jumps);
  } else {
    rows = trendForecast(reads, checkpointMs, targetMs, releaseMs, horizonHours, trophies.length);
  }

  return trophies.map((t, i) => ({ id: t.id, q: rows[i] }));
}

function main(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  mkdirSync(outputDir, { recursive: true });
  const files = readdirSync(inputDir).filter((f) => f.endsWith(".in")).sort();
  for (const file of files) {
    const raw = readFileSync(join(inputDir, file), "utf8");
    const c = JSON.parse(raw);
    const rows = forecastCase(c);
    const out = JSON.stringify({ trophies: rows });
    writeFileSync(join(outputDir, file.replace(/\.in$/, ".out")), out, "utf8");
  }
}

main();
