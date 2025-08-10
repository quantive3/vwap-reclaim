### Optimization

Parameter search powered by Optuna with Postgres-backed persistence. Focus file: `smart.py`.

### What it optimizes

Objective: maximize average return on risk (`return_on_risk_percent`).

Suggested dimensions (from `smart.py`):

| Name | Values |
|---|---|
| `entry_window` | 0: 9:30–10:30, 1: 9:30–12:00, 2: 12:00–15:00, 3: 15:00–15:45, 4: 9:30–15:45 |
| `stretch_threshold` | 0.001–0.02 (discrete set) |
| `reclaim_percentage` | 0.4–0.8 of stretch |
| `cooldown_period_seconds` | 60, 90, 120, 180, 300, 600 |
| `take_profit_percent` | 25, 35, 50, 60, 70, 80, 90, 100 |
| `stop_loss_percent` | -25, -35, -50, -60, -70, -80, -90, -100 |
| `max_trade_duration_seconds` | 60, 120, 180, 240, 300, 600 |
| `strikes_depth` | 1 |
| `option_selection_mode` | `itm`, `otm` |

Additional behavior:
- Overrides `start_date`/`end_date` via `BACKTEST_START_DATE`/`BACKTEST_END_DATE` in `smart.py`.
- Forces `silent_mode=True` during trials.
- Prunes trials with too few trades (`MIN_TRADE_THRESHOLD`).
- Deduplicates parameter combos in-memory and in Postgres (`seen_combos` table).

### Requirements

- Postgres accessible via environment variables or `config.py`:
  - `PG_HOST`, `PG_PORT`, `PG_DATABASE`, `PG_USER`, `PG_PASSWORD`
- Python requirements installed (`optuna`, `sqlalchemy`, etc.).
- Cached data in `polygon_cache/` for the chosen backtest period.

### Run

```bash
python -m optimize.smart
```

Useful knobs inside `smart.py`:

| Setting | Purpose |
|---|---|
| `STUDY_NAME` | Name of the Optuna study (change to start fresh). |
| `N_TRIALS` | Target number of completed trials. |
| `ENABLE_PERSISTENCE` | If false, deletes the existing study and clears `seen_combos`. |
| `N_STARTUP_TRIALS` | Random start before TPE. |
| `N_EI_CANDIDATES` | Candidates per TPE trial. |
| `MIN_TRADE_THRESHOLD` | Minimum trades required to count a trial. |
| `MAX_ATTEMPT_LIMIT` | Cap on total attempts (completed + pruned). |

### Outputs

- Best trial value and parameters
- Stored metrics per trial (as user attributes)
- Summary of completed/pruned trials and duplicates by source

### Notes

- The optimization calls `strategy.backtest.run_backtest(...)` with parameters injected per trial to maintain stateless execution.
- Option selection occurs at the signal timestamp; no look-ahead is introduced by the optimizer.


