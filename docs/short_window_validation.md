# Short-Window Validation Summary

This repository was validated on a strict A-share short-window experiment using JQData.

## Setup

- Universe: `120` A-share stocks
- Window: `2025-02-05` to `2026-02-06`
- Neutralization: `industry`
- Cost assumption: `10 bps` one-way
- Filters:
  - `ST` exclusion
  - paused-day exclusion
  - open-at-limit execution constraint
  - minimum traded amount
  - minimum close price

## Result Snapshot

- `101 / 101` alphas executed successfully
- `87 / 101` alphas had positive `IC mean`
- `61 / 101` alphas had positive gross return
- `8 / 101` alphas remained positive after transaction costs

Representative net-positive alphas in this short window:

- `alpha_040`
- `alpha_026`
- `alpha_006`

## Interpretation

This validation is strong enough to demonstrate that:

- the factor formulas are implemented and runnable
- the data pipeline works on real A-share market data
- the backtest engine handles basic China-specific trading constraints

It is not sufficient to claim a full academic replication, because the accessible data horizon is short.
