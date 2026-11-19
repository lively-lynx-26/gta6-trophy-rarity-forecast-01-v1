// GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).
//
// The idea: a trophy's earn rate is earners / players_tracked, and the two
// move at very different speeds. When Exophase refreshes, players_tracked
// jumps (a launch's new players arrive in bulk) while earners barely move
// (a new player holds almost nothing yet), so every rate dilutes. Over days
// the new players play and earn, and the early, accessible trophies recover
// toward the share of players who do them, while the late and grind trophies
// keep diluting because few players ever reach them.
//
// So the forecast is a dilution model in players_tracked, not a trend in
// wall-clock time:
//   rate_target = rate_last * (N_last / N_target) ^ alpha_t
// with alpha_t the trophy's dilution exponent. Two regimes:
//   - the case's history varies (the final): alpha_t is FIT per trophy from
//     log(rate) against log(players_tracked) over the last day of reads, and
//     N_target is forecast by fitting log(N) against log(1 + hours since the
//     first read). The fit is shrunk toward the prior below.
//   - the history is flat (the practice cases: every read identical): no fit
//     is possible, so alpha_t comes from a prior keyed on how early the
//     trophy is, and N_target is ~6x N_last (the one refresh the rules
//     describe multiplied N by 6.06). The prior itself blends with the
//     horizon: over 90 minutes earners barely grow (alpha near 1), over
//     4.5 days they catch up (alpha falls to the long-run values measured
//     on the live page 40 hours after launch).
// The spread widens with the horizon and with alpha_t, because a longer
// horizon means more refreshes between the checkpoint and the target.
//
// Node 22, standard library only, no network, deterministic.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** Quantile levels, in the order each output row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** Standard normal quantiles at those levels. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface CaseFile {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: unknown;
  trophies: Trophy[];
  history: Read[];
  notes: unknown;
}

const clamp = (x: number, lo: number, hi: number): number => (x < lo ? lo : x > hi ? hi : x);
const clampPct = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => clampPct(100 / (1 + Math.exp(-l)));
const round4 = (x: number): number => Math.round(x * 10000) / 10000;

/**
 * How early in a player's run a trophy sits, 0 (the first things anyone
 * does) to 1 (the last grinds). First keyword match wins; the description
 * carries the signal ("Complete Chapter 1", "Beat Cal in a fishing
 * challenge", "Attain 100% Completion"). Tier is the fallback.
 */
const EARLINESS_RULES: Array<[RegExp, number]> = [
  [/just like paradise|gone fishing/i, 0.0],
  [/baited|fishing challenge/i, 0.0],
  [/chapter 1/i, 0.05],
  [/breakfast run/i, 0.05],
  [/arcade game|each type of activity/i, 0.1],
  [/chapter 2/i, 0.15],
  [/discover 100 vehicles|bowling|what the hell|cocktail|ocean beach/i, 0.25],
  [/vcia|explore all of leonida/i, 0.35],
  [/chapter 3|chapter 4|heist|challenging tank|armored truck/i, 0.45],
  [/final mission for|wanted level|rob 3 shops|combat focus|jewelry store/i, 0.55],
  [/strangers|collectible card|waink|50 challenges|meditate|unique animal|bank vault|illegal operation|garage propert|laundering|spend \$100,000|register 10 vehicles|shooting range|drone race|20 different races/i, 0.65],
  [/final mission|250 stunts|100% completion|leonida legend/i, 0.85],
];

function earliness(t: Trophy): number {
  const text = `${t.name} ${t.description}`;
  for (const [re, e] of EARLINESS_RULES) {
    if (re.test(text)) return e;
  }
  if (t.tier === 'platinum' || t.tier === 'gold') return 0.8;
  if (t.tier === 'silver') return 0.65;
  return 0.5;
}

/** Long-run dilution exponent, measured on the live page ~40 h after launch. */
function alphaLong(t: Trophy): number {
  return 0.02 + 0.63 * earliness(t);
}

/** Short-horizon dilution exponent: over one refresh earners barely grow. */
function alphaShort(t: Trophy): number {
  return 0.7 + 0.25 * earliness(t);
}

/** Prior exponent, blended by horizon: earners catch up over ~a day. */
function priorAlpha(t: Trophy, horizonHours: number): number {
  const w = Math.exp(-horizonHours / 24);
  return alphaLong(t) + (alphaShort(t) - alphaLong(t)) * w;
}

/** Least-squares slope of y on x; null when x has no variance. */
function slope(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 3) return null;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  if (sxx <= 1e-12) return null;
  return sxy / sxx;
}

/** Spread in logit units: wider for longer horizons and diluting trophies. */
function spreadFor(alpha: number, horizonDays: number): number {
  const sShort = 0.25 + 0.15 * alpha;
  // Long-horizon spread, calibrated by backtesting: a wider band costs at
  // every quantile when the median is right, so it stays near the median's
  // own error (~0.5 logit) rather than the multi-refresh worst case.
  const sLong = 0.45 + 0.3 * alpha;
  const f = clamp(
    (Math.sqrt(horizonDays) - Math.sqrt(0.0625)) / (Math.sqrt(4.52) - Math.sqrt(0.0625)),
    0,
    1,
  );
  return sShort + (sLong - sShort) * f;
}

function forecastCase(c: CaseFile): { id: number; q: number[] }[] {
  const horizonHours = c.horizon_minutes / 60;
  const horizonDays = c.horizon_minutes / 1440;
  const history = c.history ?? [];
  const atMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);

  // Last rate per trophy, from the most recent read that lists it.
  const lastRate: number[] = c.trophies.map((_, i) => {
    for (let k = history.length - 1; k >= 0; k--) {
      const r = history[k].rates[i];
      if (typeof r === 'number' && Number.isFinite(r)) return r;
    }
    return 0;
  });

  // Distinct players_tracked values across the history.
  const distinctN = new Set(history.map((r) => r.players_tracked)).size;
  const nLast = history.length > 0 ? history[history.length - 1].players_tracked : 0;

  // --- Forecast N at the target, from the growth of players_tracked. ---
  let nTarget: number;
  if (distinctN <= 2 || history.length < 3 || nLast <= 0) {
    // Flat history: one refresh multiplies N by ~6 (launch night's did 6.06x).
    nTarget = nLast > 0 ? nLast * 6 : 638 * 6;
  } else {
    // Fit log(N) against log(1 + hours since the first read), one point per
    // distinct N at its first appearance, and read it at the target.
    const firstMs = Date.parse(history[0].at);
    const seen = new Map<number, number>();
    for (const r of history) {
      if (!seen.has(r.players_tracked)) seen.set(r.players_tracked, Date.parse(r.at));
    }
    const xs: number[] = [];
    const ys: number[] = [];
    for (const [n, ms] of seen) {
      xs.push(Math.log(1 + Math.max(0, (ms - firstMs) / 3600000)));
      ys.push(Math.log(n));
    }
    const b = slope(xs, ys);
    if (b === null || b <= 0) {
      nTarget = nLast * 2;
    } else {
      let mx = 0;
      let my = 0;
      for (let i = 0; i < xs.length; i++) {
        mx += xs[i];
        my += ys[i];
      }
      mx /= xs.length;
      my /= ys.length;
      const a = my - b * mx;
      const hTarget = Math.max(0, (targetMs - firstMs) / 3600000);
      nTarget = Math.exp(a + b * Math.log(1 + hTarget));
    }
    nTarget = clamp(nTarget, nLast * 1.05, nLast * 5);
  }
  const nRatio = nLast > 0 ? nTarget / nLast : 6;

  // --- Per-trophy exponent: fit when the history varies, else the prior. ---
  const windowStart = atMs - 24 * 3600000;
  const fitReads = history.filter((r) => Date.parse(r.at) >= windowStart);
  const useReads = fitReads.length >= 3 ? fitReads : history;

  return c.trophies.map((t, i) => {
    const rLast = lastRate[i] <= 0 ? 0.02 : lastRate[i];

    // Trophy #2 sits at ~100% and never falls: every tracked player has it.
    if (rLast >= 99.5) {
      return { id: t.id, q: [99.9, 99.97, 99.99, 99.99, 99.99] };
    }

    let alpha = priorAlpha(t, horizonHours);
    if (distinctN >= 3) {
      // Fit log(rate) on log(N) over the recent reads.
      const xs: number[] = [];
      const ys: number[] = [];
      for (const r of useReads) {
        const v = r.rates[i];
        if (typeof v !== 'number' || !Number.isFinite(v) || v <= 0) continue;
        if (r.players_tracked <= 0) continue;
        xs.push(Math.log(r.players_tracked));
        ys.push(Math.log(Math.max(0.005, v)));
      }
      const distinctFitN = new Set(xs).size;
      const b = slope(xs, ys);
      if (b !== null && distinctFitN >= 3) {
        const alphaFit = clamp(-b, -0.15, 0.9);
        const wFit = clamp(0.4 + 0.05 * distinctFitN, 0.4, 0.85);
        alpha = wFit * alphaFit + (1 - wFit) * alpha;
      }
    }

    const median = clampPct(rLast * Math.pow(1 / nRatio, alpha));
    const s = spreadFor(alpha, horizonDays);
    const l = logit(median);
    const q: number[] = [];
    let prev = -Infinity;
    for (const z of Z) {
      const v = round4(invLogit(l + z * s));
      const w = Math.max(v, prev);
      q.push(w);
      prev = w;
    }
    return { id: t.id, q };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
if (inputs.length === 0) {
  throw new Error(`no case files in ${INPUT_DIR}`);
}
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as CaseFile;
  let trophies: { id: number; q: number[] }[];
  if (!c.trophies || c.trophies.length === 0 || !c.history || c.history.length === 0) {
    // No history to learn from: a flat, wide forecast rather than a crash.
    trophies = (c.trophies ?? []).map((t) => {
      if (t.id === 2) return { id: t.id, q: [99.9, 99.97, 99.99, 99.99, 99.99] };
      return { id: t.id, q: [0.01, 0.05, 0.2, 0.8, 3.0] };
    });
  } else {
    trophies = forecastCase(c);
  }
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
