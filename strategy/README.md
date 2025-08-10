### Strategy

Clear, modular VWAP stretch-and-reclaim strategy for intraday SPY options. Stateless by design, parameter-driven, and backtest-ready.

### What it does

- Detects when SPY stretches away from VWAP by a configurable threshold, then partially reclaims toward VWAP within a cooldown window.
- On a valid reclaim:
  - Below VWAP → buy a call
  - Above VWAP → buy a put
- Option contract is selected at the exact signal timestamp to avoid look-ahead bias, using same-day expiry by default.
- Exits via take-profit, stop-loss, max-duration, end-of-day cutoff, and an emergency failsafe time.

### Key modules

| File | Purpose |
|---|---|
| `params.py` | Centralized parameters and issue-tracker initialization. |
| `data.py` | SPY, chain, and option loaders with caching and 1-second alignment; data validation/hard-fail. |
| `data_loader.py` | Thin wrapper class bundling the `data.py` loaders. |
| `signals.py` | Stretch and partial reclaim detection with cooldown and time-window filtering. |
| `option_select.py` | Contract picking (`itm`/`otm`/`atm`, `strikes_depth`) at signal time only. |
| `exits.py` | Exit rules, emergency failsafe, latency application, stale-price handling. |
| `backtest.py` | Single entry point `run_backtest(...)` orchestrates the full loop and returns metrics/logs. |
| `reporting.py` | Human-readable summaries. |
| `main.py` | Script entry for running a backtest with params from `params.py`. |

### Parameters (selected)

All behavior is controlled via `initialize_parameters()` in `params.py`.

| Name | Type | Example | Notes |
|---|---|---|---|
| `start_date`, `end_date` | str | `"2023-01-03"` | Backtest date range (business days). |
| `entry_start_time`, `entry_end_time` | time | `09:30`–`15:45` | Only consider signals within this window. |
| `stretch_threshold` | float | `0.003` | Stretch beyond VWAP required to mark a signal. |
| `reclaim_threshold` | float | `0.0021` | Reclaim back toward VWAP required for entry intent. |
| `cooldown_period_seconds` | int | `120` | Throttle multiple signals per side. |
| `option_selection_mode` | str | `itm` | One of `itm`/`otm`/`atm`. |
| `strikes_depth` | int | `1` | Depth from ATM within chosen mode. |
| `contracts_per_trade` | int | `1` | Position size in contracts. |
| `take_profit_percent` | int | `80` | Exit on profit. |
| `stop_loss_percent` | int | `-25` | Exit on loss. |
| `max_trade_duration_seconds` | int | `600` | Time-based exit. |
| `late_entry_cutoff_time` | time | `15:54` | Blocks new entries late in session. |
| `end_of_day_exit_time` | time | `15:54` | Regular EOD exit. |
| `emergency_exit_time` | time | `15:55` | Failsafe exit, overrides all else. |
| `slippage_amount` | float | `0.02` | Fixed per-share slippage model. |
| `latency_seconds` | int | `1` | Entry latency after signal. |
| `brokerage_fee_per_contract` | float | `0.65` | One-way commission. |
| `exchange_fee_per_contract` | float | `0.65` | One-way exchange fee. |

Data quality thresholds (warnings or hard-fail depending on context) are also in `params.py`, e.g., `min_spy_data_rows`, `timestamp_mismatch_threshold`, `price_staleness_threshold_seconds`.

### Single entry point

The backtest runs through the single function:

```python
from strategy.backtest import run_backtest

results = run_backtest(params, data_loader, issue_tracker)
# returns dict with: trades (via all_contracts), log (issue_tracker), stats (metrics)
```

It enforces:
- 1-second alignment across SPY, VWAP, and options
- Option selection at signal time
- No entries after `late_entry_cutoff_time`
- Hard-fail on missing or misaligned data

### Run directly

```bash
python -m strategy.main
```

Provide `API_KEY` in `config.py` or via env var when not using synthetic data. For a zero-dependency run, see `../quickstart/README.md`.


