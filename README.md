# Frost Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AryFlows/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AryFlows/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on Exophase at 2026-11-26T00:00:00Z — one week after launch — as five quantiles
(5 / 25 / 50 / 75 / 95%) per trophy. Scored by mean pinball loss on the logit
scale; the entry must beat the best house entry to take a paid place.

## How it works

`main.ts` (Node 22, standard library only, no network) reads the case file and,
for each trophy, fits a recency-weighted ridge regression on the logit scale:

```
logit(rate) = a + b · ln(players tracked) + c · ln(1 + hours since release)
```

The two drivers capture the two forces that move earn rates after a launch.
`b < 0` means dilution wins: the tracked-player count grows as new, mostly
casual players join, and the rate falls. `c > 0` means progression wins:
players play through the game and the rate rises. The tracked-player count at
the target is itself extrapolated from a log-log fit of players against time.

The spread is calibrated, not tuned: an internal backtest fits the same model
on the first 75% of the reads and measures its logit error on the last 25%.
Each trophy's spread is its backtest error (shrunk toward the pooled median)
plus a random-walk term of 0.25 logit units per square-root day — narrower than
the house baseline's 0.5, wide enough to survive a late Exophase refresh. A
flat history collapses to carry-forward with a narrow band; a trending history
extrapolates along its trend.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

It reads every `*.in` file in the input directory and writes one `*.out` per
case: `{"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}`. Deterministic — two
runs write identical bytes — and far under the 8-second cap (about 0.2 s on the
practice cases).

## Verification

- Practice cases `p00` / `p01`: 55 rows each, monotone quantiles in 0–100,
  byte-identical across two runs.
- Pinball-loss implementation checked against the worked example in the rules
  (two trophies, rows `[10, 15, 20, 25, 35]`, rates 22% and 8% → 0.403404,
  exact).
- Synthetic backtest (120 reads over 2.5 days, realistic dilution +
  progression dynamics, target 4.5 days out): pinball loss **0.1400** vs the
  house baseline's **0.3721** — 62% lower. On a 24-hour split-history backtest:
  **0.0892** vs **0.2099** — 58% lower.

## Limits

- The practice cases cannot validate the final: their histories are flat
  (every read identical, 638 players tracked), so on practice the program is
  just carry-forward with a narrow band — exactly like the house baseline.
  The backtests above are the evidence for the final, not the practice grade.
- No model can foresee an Exophase refresh: the page sat unchanged for hours
  on launch night, then every rate moved at once. The spread is the defence —
  it is why the random-walk term is kept at 0.25 instead of the 0.20 that
  scored marginally better on synthetic data with no refresh jumps.
- The synthetic backtest uses the author's own dynamics (power-law player
  growth, saturating earn curves). It measures the model against the baseline
  on the same data; it is not a promise about the real final.

## License

MIT — see [LICENSE](LICENSE).
