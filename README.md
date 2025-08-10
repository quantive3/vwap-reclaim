[![Lint](https://github.com/shawnjoshi/vwap-reclaim/actions/workflows/lint.yml/badge.svg)](https://github.com/shawnjoshi/vwap-reclaim/actions/workflows/lint.yml) ![Coverage](coverage.svg) [![Docker Pulls](https://img.shields.io/docker/pulls/quantive/vwap_reclaim.svg)](https://hub.docker.com/r/quantive/vwap_reclaim) [![Image Size](https://img.shields.io/docker/image-size/quantive/vwap_reclaim/latest)](https://hub.docker.com/r/quantive/vwap_reclaim/tags) ![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

### VWAP Reclaim Strategy (SPY Options)

Concise, single-shot intraday strategy that buys short-dated SPY options on structured VWAP stretch-and-reclaim signals. Fully cacheable, sweep-ready, and designed for reproducible backtests and optimization.

### Quickstart (no API required)

Run with bundled synthetic data in minutes.

```powershell
# From repo root on Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python quickstart/run_strategy.py
```

What this does:
- Copies synthetic data into `polygon_cache/`
- Sets dummy credentials to avoid prompts
- Runs the strategy across Jan 3–5, 2023 using fixed params

### Folder guide

| Path | What’s inside |
|---|---|
| `strategy/` | Core logic: parameters, data loading, signals, option selection, exits, backtest, reporting. See `strategy/README.md`. |
| `optimize/` | Optuna-powered parameter search with Postgres persistence (focus: `smart.py`). See `optimize/README.md`. |
| `quickstart/` | One-command run with bundled synthetic data. See `quickstart/README.md`. |
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

Full details in `strategy/README.md`.

### Optimization (optional)

Reproducible parameter search with Optuna and Postgres-backed storage. Prunes low-trade trials, deduplicates param combos, and reports best trials.

```powershell
python -m optimize.smart
```

See `optimize/README.md` for setup and tips.

### Docker

An example `Dockerfile` is provided for containerized runs. Pull the prebuilt image or build locally.

### License

MIT

