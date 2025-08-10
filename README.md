[![Lint](https://github.com/shawnjoshi/vwap-reclaim/actions/workflows/lint.yml/badge.svg)](https://github.com/shawnjoshi/vwap-reclaim/actions/workflows/lint.yml) ![Coverage](coverage.svg) [![Docker Pulls](https://img.shields.io/docker/pulls/quantive/vwap_reclaim.svg)](https://hub.docker.com/r/quantive/vwap_reclaim) [![Image Size](https://img.shields.io/docker/image-size/quantive/vwap_reclaim/latest)](https://hub.docker.com/r/quantive/vwap_reclaim/tags) ![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

# VWAP Reclaim Strategy (SPY Options)

Single-shot intraday strategy that buys 0DTE SPY options on structured VWAP stretch-and-reclaim signals. 

Fully cacheable, sweep-ready, and designed for reproducible backtests and optimization.

### Folder guide

| Path | What’s inside |
|---|---|
| `strategy/` | Core logic: parameters, data loading, signals, option selection, exits, backtest, reporting. See [strategy/README.md](strategy/). |
| `optimize/` | Parameter sweeps with Optuna and Postgres-backed storage. Prunes low-trade trials, deduplicates param combos, and reports best trials. See [optimize/README.md](optimize/). |
| `quickstart/` | One-command run with bundled synthetic data. See [quickstart/README.md](quickstart/). |
| `tests/` | Basic correctness and regression tests against synthetic cache. |

### Run with your own Polygon API key

1) Provide `API_KEY` in `config.py` or as environment variable.
2) Populate `polygon_cache/` (the strategy will cache API pulls automatically).
3) Execute:

```powershell
python -m strategy.main
```

Notes:
- Data are aligned to 1-second resolution and validated. Missing/misaligned data hard-fail with clear logs.
- Strategy only buys options (no selling). Option selection happens exactly at the signal timestamp to prevent look-ahead.

### Strategy at a glance

- Entry: Detects VWAP stretch beyond a threshold, then a partial reclaim within a cooldown window.
- Direction: Below VWAP → buy call; Above VWAP → buy put. Same-day expiry enforced by default.
- Exits: Take-profit, stop-loss, max-duration, end-of-day, and an emergency failsafe time.
- Frictions: Latency, slippage, commissions/fees.

Key parameters (subset):

| Parameter | Example | Description |
|---|---|---|
| `stretch_threshold` | `0.003` | Min percent move away from VWAP to mark stretch. |
| `reclaim_threshold` | `0.0021` | Percent move back toward VWAP to consider reclaim. |
| `cooldown_period_seconds` | `120` | Throttle multiple signals per side. |
| `option_selection_mode` | `itm` | `itm`/`otm`/`atm` selection. |
| `strikes_depth` | `1` | 1 = closest depth to ATM within the chosen mode. |
| `take_profit_percent` | `80` | Exit on gain. |
| `stop_loss_percent` | `-25` | Exit on loss. |
| `max_trade_duration_seconds` | `600` | Time-based exit. |

Full details in [strategy/README.md](strategy/).
