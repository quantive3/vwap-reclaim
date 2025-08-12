# Strategy

[Module Breakdown](#module-breakdown) | [Parameter Guide](#parameter-guide) | [Data Integrity & Auditability](#data-integrity-and-auditability) | [Run It](#run-it)

This backtesting framework is built around a modular VWAP stretch-and-reclaim strategy.

It specifically focuses on trading same-day expiry (0DTE) options for SPY.

## What It Does

1. Detects when SPY stretches away from VWAP, then partially reclaims towards it within a specified window.

2. On a valid reclaim:

- Price below VWAP → Buy a call
- Price above VWAP → Buy a put

3. Option contract is selected at the exact signal timestamp to avoid look-ahead bias.

4. Exits via take-profit, stop-loss, max-duration, end-of-day cutoff, and an emergency failsafe time.

5. Simulates trade frictions: latency, slippage, fees.


## Module Breakdown

| File | Purpose |
|---|---|
| `params.py` | Centralized parameters and issue-tracker initialization. |
| `data.py` | Underlying, chain, and option loaders with caching and alignment. |
| `data_loader.py` | Thin wrapper class bundling the `data.py` loaders. |
| `signals.py` | Stretch and partial reclaim detection with cooldown and time-window filtering. |
| `option_select.py` | Contract selection (`itm`/`otm`/`atm`, `strikes_depth`) at signal time. |
| `exits.py` | Exit rules, emergency failsafe, latency application, stale-price handling. |
| `backtest.py` | Single entry point `run_backtest(...)` orchestrates the full loop and returns metrics/logs. |
| `reporting.py` | Readable test summaries. |
| `main.py` | Script entry for running a backtest with parameters from `params.py`. |


## Parameter Guide

All behavior is controlled via `initialize_parameters()` in `params.py`.

| Name | Example | Notes |
|---|---|---|
| `start_date`, `end_date` | `2025-01-01` - `2025-01-31` | Backtest date range (business days). |
| `entry_start_time`, `entry_end_time` | `09:30` - `13:00` | Only consider signals within this window. |
| `stretch_threshold` | `0.003` | Stretch beyond VWAP required to mark a signal. |
| `reclaim_threshold` | `0.0021` | Reclaim back toward VWAP required for entry intent. |
| `cooldown_period_seconds` | `120` | Throttle multiple signals per side. |
| `option_selection_mode` | `itm` | One of `itm`/`otm`/`atm`. |
| `strikes_depth` | `1` | Depth from ATM within chosen mode. |
| `contracts_per_trade` | `1` | Position size in contracts. |
| `take_profit_percent` | `80` | Exit on profit. |
| `stop_loss_percent` | `-25` | Exit on loss. |
| `max_trade_duration_seconds` | `600` | Time-based exit. |
| `late_entry_cutoff_time` | `15:30` | Blocks new entries late in session. |
| `end_of_day_exit_time` | `15:45` | Regular EOD exit. |
| `emergency_exit_time` | `15:55` | Failsafe exit, overrides all else. |
| `slippage_amount` | `0.02` | Fixed per-share slippage model. |
| `latency_seconds` | `1` | Entry latency after signal. |
| `brokerage_fee_per_contract` | `0.65` | One-way commission. |
| `exchange_fee_per_contract` | `0.65` | One-way exchange fee (bundled). |

Data quality thresholds are also in `params.py` (e.g. `min_spy_data_rows`, `timestamp_mismatch_threshold`, `price_staleness_threshold_seconds`).


## Data Integrity and Auditability

There is significant emphasis on ensuring data is aligned and reproducible.

**Aligned Dataframes:** Underlying and option prices are aligned on a 1-second grid. Misalignments are detected, reported, and can be set to block entries.

**Option Price Freshness:** Each option price carries “seconds since last update” column. Stale quotes are flagged so you can analyze or filter them later.

**Deterministic Caching:** Data pulls are cached locally and guarded with file locks so reruns are reproducible and race-free.

**Reproducibility:** Optional dataframe hashes help confirm that inputs and alignments haven’t changed between runs.

**Run Summaries:** A structured issue tracker aggregates warnings/errors (e.g., stale prices, missing rows, late entries, emergency exits) into a concise end-of-run report.

**Debug + Silent Modes:** Prints are controlled via `debug_mode` and `silent_mode` in `params.py`. Enable `debug_mode` for step-level diagnostics, hashing, and more. Use `silent_mode` for optimization runs to suppress all remaining prints.

*Note: `debug_mode` significantly increases time-to-completion for backtests. It's only recommended for short date ranges.*


## Run It

```bash
python -m strategy.main
```

Provide `API_KEY` in `config.py` or via env var when not using synthetic data. For a zero-dependency run, see [quickstart/README.md](../quickstart/).
