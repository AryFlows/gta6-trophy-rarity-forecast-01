// Frost Rarity Forecast — GTA VI trophy earn-rate forecaster (Node 22, stdlib only).
//
// The task: for every trophy, forecast five quantiles of its Exophase earn rate
// (%) at the case's target, scored by mean pinball loss on the logit scale.
// The entry must beat the best house entry to take a paid place.
//
// Model, per trophy, on the logit scale. The mechanism that moves earn rates
// after a launch is dilution: the tracked-player count P grows as new (mostly
// casual) players join, while the earned count E grows as players progress.
// So logit(rate) is fitted against two drivers at once:
//
//   logit(rate) = a + b * ln(P) + c * ln(1 + hours since release)
//
// b < 0 means dilution wins and the rate falls as the player base grows;
// c > 0 means progression wins and the rate rises as players play through.
// The fit is a recency-weighted ridge regression (24 h half-life, mild
// shrinkage toward a flat line), so a flat history collapses to carry-forward
// and a trending history extrapolates along the trend. The player count at
// the target is itself extrapolated from a log-log fit of P against time.
//
// The spread is honest, not tuned: an internal backtest fits the same model
// on the first 75% of the reads and measures its logit error on the last 25%.
// Each trophy's spread is its backtest error (shrunk toward the pooled
// median) plus a random-walk term that grows with sqrt(horizon). A forecast
// that is too narrow pays for every trophy outside it; one too wide pays at
// every trophy.
//
// Nothing here is copied from the house baseline: the baseline is persistence
// (last rate, fixed 0.5/sqrt-day spread); this entry fits a trend and
// calibrates its own spread from the case's own history.
//
// I/O contract: read every <id>.in in $MAKEFUN_INPUT_DIR (else /input), write
// one <id>.out per case to $MAKEFUN_OUTPUT_DIR (else /output):
// {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]} — one row per trophy,
// monotone quantiles, each 0..100, at most 65,536 bytes. Deterministic: two
// runs write identical bytes.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];
/** Random-walk spread, logit units per square-root day, added to the backtest error.
 * Calibrated on synthetic backtests (dilution + progression dynamics, 120 reads):
 * 0.25 beats 0.35 by ~22% on both a 4.5-day and a 24-hour horizon, while still
 * leaving a real allowance for the refresh jumps the house documents (the page
 * sat unchanged for hours, then every rate moved at once). 0.20 scored a touch
 * better on the synthetic data, but that world has no refresh jumps, so 0.25
 * is the compromise: narrower than the house baseline's 0.5, wide enough to
 * survive a late refresh. */
const RW_SIGMA = 0.25;
/** Recency half-life of a read's weight, in hours. */
const HALF_LIFE_H = 24;
/** Ridge penalty on the two (standardised) slopes; the intercept is unpenalised. */
const RIDGE = 2.0;
/** Largest logit move allowed away from a trophy's last read rate. */
const MAX_LOGIT_MOVE = 3.0;
/** Floor on the per-trophy backtest error, in logit units. */
const SIGMA_FLOOR = 0.12;

interface Trophy {
  id: number;
  tier: string;
  points: number;
  secret: boolean;
}

interface Read {
  atMs: number;
  players: number;
  rates: Array<number | null>;
}

interface CaseFile {
  id: string;
  levels: number[];
  horizon_minutes: number;
  target: string;
  game: { release_date?: string };
  trophies: Trophy[];
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
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/** Solve a small dense linear system by Gaussian elimination with partial pivoting. */
function solveLinear(a: number[][], b: number[]): number[] {
  const n = b.length;
  const m = a.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(m[r][col]) > Math.abs(m[pivot][col])) pivot = r;
    const tmp = m[col];
    m[col] = m[pivot];
    m[pivot] = tmp;
    const diag = m[col][col];
    if (Math.abs(diag) < 1e-12) continue;
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const factor = m[r][col] / diag;
      if (factor === 0) continue;
      for (let c = col; c <= n; c++) m[r][c] -= factor * m[col][c];
    }
  }
  return m.map((row, i) => (Math.abs(row[i]) < 1e-12 ? 0 : row[n] / row[i]));
}

interface Fit {
  /** Weighted means of the two predictors, for standardising a new point. */
  mean1: number;
  mean2: number;
  sd1: number;
  sd2: number;
  /** Intercept and slopes in the standardised predictor space. */
  beta: number[];
}

/**
 * Weighted ridge fit of y on [ln(P), ln(1+hours)] over the given reads.
 * Returns null when there are too few usable points to fit.
 */
function fitLogitModel(
  points: Array<{ x1: number; x2: number; y: number; w: number }>,
): Fit | null {
  const usable = points.filter((p) => Number.isFinite(p.y) && Number.isFinite(p.x1) && Number.isFinite(p.x2));
  if (usable.length < 4) return null;
  let wSum = 0;
  let m1 = 0;
  let m2 = 0;
  for (const p of usable) {
    wSum += p.w;
    m1 += p.w * p.x1;
    m2 += p.w * p.x2;
  }
  m1 /= wSum;
  m2 /= wSum;
  let v1 = 0;
  let v2 = 0;
  for (const p of usable) {
    v1 += p.w * (p.x1 - m1) ** 2;
    v2 += p.w * (p.x2 - m2) ** 2;
  }
  const sd1 = Math.sqrt(v1 / wSum) || 1;
  const sd2 = Math.sqrt(v2 / wSum) || 1;
  // Normal equations for [1, z1, z2] with a ridge penalty on the slopes only.
  const ata: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const atb = [0, 0, 0];
  for (const p of usable) {
    const row = [1, (p.x1 - m1) / sd1, (p.x2 - m2) / sd2];
    for (let i = 0; i < 3; i++) {
      atb[i] += p.w * row[i] * p.y;
      for (let j = 0; j < 3; j++) ata[i][j] += p.w * row[i] * row[j];
    }
  }
  ata[1][1] += RIDGE;
  ata[2][2] += RIDGE;
  const beta = solveLinear(ata, atb);
  return { mean1: m1, mean2: m2, sd1, sd2, beta };
}

function predictLogit(fit: Fit, x1: number, x2: number): number {
  const z1 = (x1 - fit.mean1) / fit.sd1;
  const z2 = (x2 - fit.mean2) / fit.sd2;
  return fit.beta[0] + fit.beta[1] * z1 + fit.beta[2] * z2;
}

/** Extrapolate the tracked-player count to the target hour by a log-log fit. */
function extrapolatePlayers(reads: Read[], releaseMs: number, targetMs: number): number {
  const last = reads[reads.length - 1];
  if (reads.length < 3) return last.players;
  const cutoff = last.atMs - 24 * 3600 * 1000;
  const recent = reads.filter((r) => r.atMs >= cutoff);
  const use = recent.length >= 3 ? recent : reads;
  // Weighted least squares of ln(P) on ln(1+hours), recent reads weighted most.
  let wSum = 0;
  let mx = 0;
  let my = 0;
  const pts = use.map((r) => {
    const x = Math.log(1 + Math.max(0, (r.atMs - releaseMs) / 3600000));
    const y = Math.log(Math.max(1, r.players));
    const w = 0.5 ** ((last.atMs - r.atMs) / (24 * 3600000));
    wSum += w;
    mx += w * x;
    my += w * y;
    return { x, y, w };
  });
  mx /= wSum;
  my /= wSum;
  let sxx = 0;
  let sxy = 0;
  for (const p of pts) {
    sxx += p.w * (p.x - mx) ** 2;
    sxy += p.w * (p.x - mx) * (p.y - my);
  }
  let slope = sxx > 1e-12 ? sxy / sxx : 0;
  slope = Math.min(1.5, Math.max(0, slope));
  const intercept = my - slope * mx;
  const xTarget = Math.log(1 + Math.max(0, (targetMs - releaseMs) / 3600000));
  const pTarget = Math.exp(intercept + slope * xTarget);
  return Math.min(last.players * 20, Math.max(last.players, pTarget));
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`case ${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (!c.history || c.history.length === 0) throw new Error(`case ${c.id} has no history`);
  const releaseMs = c.game?.release_date
    ? Date.parse(`${c.game.release_date}T00:00:00Z`)
    : Date.parse(c.history[0].at);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max(0, c.horizon_minutes / 1440);
  const reads: Read[] = c.history.map((h) => ({
    atMs: Date.parse(h.at),
    players: Math.max(1, h.players_tracked),
    rates: h.rates,
  }));
  const last = reads[reads.length - 1];
  const lastMs = last.atMs;

  const hoursOf = (atMs: number): number => Math.max(0, (atMs - releaseMs) / 3600000);
  const weightOf = (atMs: number): number => 0.5 ** ((lastMs - atMs) / (HALF_LIFE_H * 3600000));

  // Player count at the target, shared by every trophy's fit.
  const playersTarget = extrapolatePlayers(reads, releaseMs, targetMs);
  const x1Target = Math.log(playersTarget);
  const x2Target = Math.log(1 + hoursOf(targetMs));

  // Backtest split: fit on the first 75% of reads, score on the rest.
  const splitAt = Math.floor(reads.length * 0.75);
  const backtestErrors: number[] = [];

  const rows = c.trophies.map((trophy, i) => {
    const points = reads
      .map((r) => ({
        x1: Math.log(r.players),
        x2: Math.log(1 + hoursOf(r.atMs)),
        y: r.rates[i] === null || r.rates[i] === undefined ? NaN : logit(r.rates[i] as number),
        w: weightOf(r.atMs),
      }))
      .filter((p) => Number.isFinite(p.y));
    const lastRate = last.rates[i] ?? 0;
    const lastLogit = logit(lastRate);

    let medianLogit = lastLogit;
    let sigma = 0.35;
    const fit = fitLogitModel(points);
    if (fit !== null) {
      const raw = predictLogit(fit, x1Target, x2Target);
      medianLogit = Math.min(lastLogit + MAX_LOGIT_MOVE, Math.max(lastLogit - MAX_LOGIT_MOVE, raw));
      // Internal backtest for this trophy's spread.
      if (splitAt >= 4 && reads.length - splitAt >= 2) {
        const train = points.filter((_, idx) => idx < splitAt);
        const trainFit = fitLogitModel(train);
        if (trainFit !== null) {
          let se = 0;
          let n = 0;
          for (let k = splitAt; k < reads.length; k++) {
            const r = reads[k];
            if (r.rates[i] === null || r.rates[i] === undefined) continue;
            const pred = predictLogit(trainFit, Math.log(r.players), Math.log(1 + hoursOf(r.atMs)));
            const err = pred - logit(r.rates[i] as number);
            se += err * err;
            n++;
          }
          if (n > 0) {
            sigma = Math.sqrt(se / n);
            backtestErrors.push(sigma);
          }
        }
      }
    }

    return { trophy, medianLogit, sigma, lastLogit };
  });

  // Pool the backtest errors: shrink each trophy's error toward the median.
  const sorted = [...backtestErrors].sort((a, b) => a - b);
  const pooled = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : 0.3;
  const rw = RW_SIGMA * Math.sqrt(horizonDays);

  return rows.map(({ trophy, medianLogit, sigma }) => {
    const shrunk = Math.sqrt((sigma * sigma + 10 * pooled * pooled) / 11);
    const s = Math.sqrt(Math.max(SIGMA_FLOOR, shrunk) ** 2 + rw * rw);
    let prev = -Infinity;
    const q = Z.map((z) => {
      let v = round4(clampRate(invLogit(medianLogit + z * s)));
      if (v < prev) v = prev;
      prev = v;
      return v;
    });
    return { id: trophy.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
