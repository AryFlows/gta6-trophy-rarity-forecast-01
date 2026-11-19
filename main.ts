// Frost Rarity Forecast v2 — GTA VI trophy earn-rate forecaster (Node 22, stdlib only).
//
// Point model (kept from v1): per trophy, a recency-weighted ridge fit of
//   logit(rate) = a + b*ln(players tracked) + c*ln(1 + hours since release),
// read at the target with players extrapolated by a log-log fit. On moving
// histories this beats the house persistence baseline by ~2x in pinball loss.
//
// Spread model (the v2 fix): v1's spread was its internal backtest error plus
// a random-walk term. On the practice cases the histories were FLAT (launch
// night, pre-refresh: every read identical), so the backtest error was ~0, the
// random-walk term over a 90-minute horizon was ~0.06, and the 90% band was
// about +-0.1 logit wide — while the refresh moved rates by ~0.5-1 logit.
// Coverage was 0 and the practice loss 1.50, no better than the house
// baseline. v2 keeps the data spread but never lets it collapse:
//   - a FLOOR: max(pooled median backtest error, 0.5 logit units), so no
//     forecast is ever more certain than +-0.5 logit (1 sigma);
//   - WEAK-SIGNAL WIDENING: when few reads actually change (flat histories),
//     the spread scales up by as much as 1.6x, calibrated on synthetic
//     flat-regime backtests (flat input histories + jumped target) so the
//     90% coverage lands at ~0.92 instead of v1's ~0.29, cutting the flat
//     loss from 0.51 to 0.37 while the moving-regime loss stays 0.145
//     (v1: 0.143; house baseline: 0.293).
// On moving histories the data spread already exceeds the floor and the
// widening is ~1, so the fix costs nothing where v1 was already good.
// The spread is capped at 2.5 logit units: pinball punishes over-wide
// intervals too, and no honest belief here is wider than that.
//
// Contract (RULES.txt): read every <id>.in in the input directory, write
// <id>.out as {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]} — 55 rows,
// monotone quantiles in 0..100, deterministic, well under the 8s cap.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, logit units per square-root day (tuned on v1 backtests). */
const RW_SIGMA = 0.25;
/** Absolute spread floor, logit units: no forecast is ever more certain than this. */
const FLOOR_SIGMA = 0.5;
/** Extra widening when the histories carry no signal (flat reads): spread x(1 + this). */
const WIDEN_MAX = 0.6;
/** Spread cap, logit units: wider than this is never an honest belief here. */
const SPREAD_CAP = 2.5;
/** Ridge penalty on the slopes (the intercept is never penalised). */
const RIDGE_LAMBDA = 1.0;
/** Recency half-life for the ridge weights, in hours (6h tuned on synthetic
 *  moving backtests: 24h scored 0.154, 12h 0.150, 6h 0.145, 4h 0.150). */
const HALF_LIFE_HOURS = 6;
/** A read-to-read rate move counts as signal only above this, in points. */
const SIGNAL_STEP = 0.02;
/** Release of GTA VI, the origin of the hours feature. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');

interface Case {
  levels: number[];
  horizon_minutes: number;
  target: string;
  trophies: Array<{ id: number }>;
  history: Array<{
    at: string;
    players_tracked: number;
    rates: Array<number | null>;
  }>;
}

const clampRate = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
const median = (xs: number[]): number => {
  if (xs.length === 0) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : (s[m - 1] + s[m]) / 2;
};
/** Hours from release to an ISO timestamp, floored so ln(1 + hours) stays defined. */
const hoursSinceRelease = (iso: string): number =>
  Math.max(-0.9, (Date.parse(iso) - RELEASE_MS) / 3_600_000);

/** Solve a small dense linear system by Gaussian elimination with pivoting. */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) {
      if (Math.abs(M[r][col]) > Math.abs(M[pivot][col])) pivot = r;
    }
    if (Math.abs(M[pivot][col]) < 1e-12) continue;
    [M[col], M[pivot]] = [M[pivot], M[col]];
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = M[r][col] / M[col][col];
      for (let c = col; c <= n; c++) M[r][c] -= f * M[col][c];
    }
  }
  return M.map((row, i) => (Math.abs(row[i]) < 1e-12 ? 0 : row[n] / row[i]));
}

/**
 * Weighted ridge regression. xs are feature rows (first column the intercept),
 * ys targets, ws weights. The intercept is unpenalised; slopes get lambda.
 * Returns the coefficient vector, or null when there is nothing to fit.
 */
function ridge(
  xs: number[][],
  ys: number[],
  ws: number[],
  lambda: number,
): number[] | null {
  const k = xs[0]?.length ?? 0;
  if (xs.length === 0 || k === 0) return null;
  const A: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
  const b: number[] = new Array(k).fill(0);
  for (let r = 0; r < xs.length; r++) {
    for (let i = 0; i < k; i++) {
      b[i] += ws[r] * xs[r][i] * ys[r];
      for (let j = 0; j < k; j++) A[i][j] += ws[r] * xs[r][i] * xs[r][j];
    }
  }
  for (let i = 1; i < k; i++) A[i][i] += lambda;
  return solve(A, b);
}

const dot = (a: number[], b: number[]): number =>
  a.reduce((s, v, i) => s + v * b[i], 0);

interface Read {
  tHours: number;
  lnPlayers: number;
  lnHours: number;
  weight: number;
}

/** One trophy's usable history: reads where it has a rate. */
interface TrophySeries {
  reads: Read[];
  logits: number[];
  lastLogit: number;
}

function forecastCase(c: Case): Array<{ id: number; q: number[] }> {
  if (c.levels.length !== LEVELS.length ||
      c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case asks for levels ${JSON.stringify(c.levels)}`);
  }
  if (c.history.length === 0) throw new Error('case has no history');
  const horizonDays = c.horizon_minutes / 1440;
  const targetHours = hoursSinceRelease(c.target);
  const targetLnHours = Math.log(1 + targetHours);
  const lastRead = c.history[c.history.length - 1];
  const lastT = hoursSinceRelease(lastRead.at);

  // Shared read features, oldest first.
  const reads: Read[] = c.history.map((h) => {
    const t = hoursSinceRelease(h.at);
    return {
      tHours: t,
      lnPlayers: Math.log(Math.max(1, h.players_tracked)),
      lnHours: Math.log(1 + t),
      weight: Math.pow(0.5, Math.max(0, lastT - t) / HALF_LIFE_HOURS),
    };
  });

  // Players at the target: log-log fit of ln(players) on ln(1 + hours).
  const playersFit = ridge(
    reads.map((r) => [1, r.lnHours]),
    reads.map((r) => r.lnPlayers),
    reads.map((r) => r.weight),
    RIDGE_LAMBDA,
  );
  let lnPlayersTarget = Math.log(Math.max(1, lastRead.players_tracked));
  if (playersFit !== null) {
    const pred = dot(playersFit, [1, targetLnHours]);
    if (Number.isFinite(pred)) lnPlayersTarget = pred;
  }

  // Per-trophy series.
  const nT = c.trophies.length;
  const series: TrophySeries[] = [];
  for (let i = 0; i < nT; i++) {
    const rs: Read[] = [];
    const lg: number[] = [];
    for (let r = 0; r < c.history.length; r++) {
      const rate = c.history[r].rates[i];
      if (rate === null || rate === undefined) continue;
      rs.push(reads[r]);
      lg.push(logit(rate));
    }
    series.push({ reads: rs, logits: lg, lastLogit: lg.length > 0 ? lg[lg.length - 1] : NaN });
  }
  const lastLogits = series.map((s) => s.lastLogit).filter(Number.isFinite);
  const pooledLastLogit = lastLogits.length > 0 ? median(lastLogits) : logit(1);

  // How much signal do the histories carry? The pooled fraction of
  // read-to-read steps where a trophy's rate actually moved. Flat
  // histories score 0; moving ones score near 1.
  let steps = 0;
  let moved = 0;
  for (let i = 0; i < nT; i++) {
    let prev: number | null = null;
    for (let r = 0; r < c.history.length; r++) {
      const rate = c.history[r].rates[i];
      if (rate === null || rate === undefined) continue;
      if (prev !== null) {
        steps++;
        if (Math.abs(rate - prev) > SIGNAL_STEP) moved++;
      }
      prev = rate;
    }
  }
  const informativeness = steps > 0 ? moved / steps : 0;
  const widen = 1 + WIDEN_MAX * (1 - informativeness);

  // Per-trophy point forecast + internal backtest error.
  const points: number[] = [];
  const ownErr: number[] = [];
  for (let i = 0; i < nT; i++) {
    const s = series[i];
    const targetX = [1, lnPlayersTarget, targetLnHours];
    if (s.reads.length === 0) {
      points.push(pooledLastLogit);
      ownErr.push(NaN);
      continue;
    }
    if (s.reads.length < 3) {
      points.push(s.lastLogit);
      ownErr.push(NaN);
      continue;
    }
    const xs = s.reads.map((r) => [1, r.lnPlayers, r.lnHours]);
    const ws = s.reads.map((r) => r.weight);
    const fit = ridge(xs, s.logits, ws, RIDGE_LAMBDA);
    let point = s.lastLogit;
    if (fit !== null) {
      const pred = dot(fit, targetX);
      if (Number.isFinite(pred)) point = Math.min(9.2, Math.max(-9.2, pred));
    }
    points.push(point);

    // Backtest: fit the first 75% of reads, score the last 25% in logit units.
    const cut = Math.floor(s.reads.length * 0.75);
    if (cut >= 3 && s.reads.length - cut >= 1) {
      const bFit = ridge(xs.slice(0, cut), s.logits.slice(0, cut), ws.slice(0, cut), RIDGE_LAMBDA);
      if (bFit !== null) {
        let se = 0;
        let cnt = 0;
        for (let r = cut; r < s.reads.length; r++) {
          const pred = dot(bFit, xs[r]);
          if (Number.isFinite(pred)) {
            se += (pred - s.logits[r]) ** 2;
            cnt++;
          }
        }
        ownErr.push(cnt > 0 ? Math.sqrt(se / cnt) : NaN);
        continue;
      }
    }
    ownErr.push(NaN);
  }

  // Pooled backtest error across trophies; shrink each trophy's own error
  // toward it (a single trophy's backtest on a few reads is noisy).
  const validErr = ownErr.filter(Number.isFinite);
  const pooledErr = validErr.length > 0 ? median(validErr) : NaN;
  const rwTerm = RW_SIGMA * Math.sqrt(horizonDays);
  // The floor: the pooled error if it is informative, else the absolute
  // minimum — whichever is larger, so the spread never collapses.
  const floor = Math.max(Number.isFinite(pooledErr) ? pooledErr : 0, FLOOR_SIGMA);

  return c.trophies.map((t, i) => {
    const own = ownErr[i];
    const backtest = Number.isFinite(own) && Number.isFinite(pooledErr)
      ? 0.5 * own + 0.5 * pooledErr
      : Number.isFinite(pooledErr) ? pooledErr : floor;
    const dataSpread = Math.sqrt(backtest * backtest + rwTerm * rwTerm);
    const spread = Math.min(SPREAD_CAP, Math.max(dataSpread, floor) * widen);
    const q = Z.map((z) => {
      const v = round4(invLogit(points[i] + z * spread));
      return Math.min(100, Math.max(0, v));
    });
    // Rounding can only tie neighbours, never invert them, but enforce it.
    for (let j = 1; j < q.length; j++) q[j] = Math.max(q[j], q[j - 1]);
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
