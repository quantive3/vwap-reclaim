[![Lint](https://github.com/shawnjoshi/vwap-reclaim/actions/workflows/lint.yml/badge.svg)](https://github.com/shawnjoshi/vwap-reclaim/actions/workflows/lint.yml) ![Coverage](coverage.svg) [![Image Size](https://img.shields.io/docker/image-size/quantive/vwap_reclaim/latest)](https://hub.docker.com/r/quantive/vwap_reclaim/tags) ![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)

# VWAP Reclaim Lab

## Introduction

An intraday options backtesting framework built around VWAP stretch-and-reclaim entries. Built with clean boundaries for data, signals, option selection, exits, and reporting.

Includes simulation for contract scaling and real-world trading friction (latency, slippage, fees).

Fully cacheable, sweep-ready, and designed for reproducible backtests and optimization.

## Strategy at a Glance

**Signals:** Detects VWAP stretch beyond a threshold, then a partial reclaim within a specified window.

**Option Selection:** Sweepable parameters for ITM/ATM/OTM and strike depth. Buys calls and puts based on stretch/reclaim direction. Same-day expiry enforced by default.

**Exits:** Take-profit, stop-loss, max-duration, end-of-day, and an emergency failsafe exits.

**Trading Friction:** Latency, slippage, commissions/fees.

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

## Folder Guide

| Path | What’s inside |
|---|---|
| `strategy/` | Core logic: parameters, data loading, signals, option selection, exits, backtest, reporting. See [strategy/README.md](strategy/). |
| `optimize/` | Parameter sweeps with Optuna and Postgres-backed storage. Prunes low-trade trials, deduplicates param combos, and reports best trials. See [optimize/README.md](optimize/). |
| `quickstart/` | One-command run with bundled synthetic data. See [quickstart/README.md](quickstart/). |
| `tests/` | Basic correctness and regression tests against synthetic cache. |

## Key Dependencies

| Dependency | Description |
|---|---|
| Python | Runtime for the backtester and optimizer; install packages via `requirements.txt`. |
| Polygon.io API key | Stocks and Options Developer plans (or higher) are required to pull historical data at one second aggregates. **Note:** Synthetic data has been provided in the quickstart/ section  |
| Optuna | Beysian optimization engine used by `optimize/smart.py` to sweep parameters. |
| PostgreSQL | Persists Optuna studies and deduplicates parameter combos; configure `PG_*` env vars. Used for large-scale paramater sweeps. |
| Docker | Reproducible environment and fast builds for virtual/ephemeral compute. |
