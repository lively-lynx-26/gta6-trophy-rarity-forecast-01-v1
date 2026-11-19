# GTA VI trophy rarity forecast

Forecasts each Grand Theft Auto VI PlayStation 5 trophy's earn rate on
Exophase at 2026-11-26T00:00:00Z, as five quantiles (5/25/50/75/95%), from
the house's reads of the trophy page up to the case's checkpoint.

## Model

A trophy's earn rate is `earners / players_tracked`, and the two move at
very different speeds: when Exophase refreshes, `players_tracked` jumps
while earners barely move, so every rate dilutes; over days the new players
play and earn, and the early trophies recover while the late and grind
trophies keep diluting. So the forecast is a dilution model in
`players_tracked`, not a trend in wall-clock time:

```
rate_target = rate_last * (N_last / N_target) ^ alpha_t
```

- **Final case (history varies):** `alpha_t` is fit per trophy from
  `log(rate)` against `log(players_tracked)` over the last day of reads,
  shrunk toward the prior; `N_target` is forecast by fitting `log(N)`
  against `log(1 + hours since the first read)`.
- **Practice cases (history flat):** no fit is possible, so `alpha_t` comes
  from a prior keyed on how early the trophy is, blended by horizon
  (earners barely grow over 90 minutes, they catch up over days), and
  `N_target` is ~6x `N_last` (the one observed refresh multiplied N by 6.06).
- The spread widens with the horizon and with `alpha_t`. Trophy #2 sits at
  ~100% and never falls.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

Node 22, standard library only, no network, deterministic: two runs on the
same case write byte-identical output, well inside the 8-second cap.

## Results (local)

Against the study's estimated practice answers (proportional allocation of
the first refresh's +103 earners): loss **0.098**, typical miss 0.024 pp,
90% coverage 0.98 — versus the persistence baseline's 1.40. On a synthetic
final-like case (358 reads, N 638 → 31k → 122k): loss **0.154** versus the
baseline's 0.354.
