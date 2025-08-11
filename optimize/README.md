# Strategy Optimization

Parameter sweeps, powered by Optuna, with Postgres-backed persistence.


## File Breakdown

| File | Purpose |
|---|---|
| `smart.py` | Core optimizer. Runs Optuna's TPE Sampler against `strategy/`. Includes Postgres persistence and duplicate checks/pruning. |
| `launch.py` | Runs N optimizer workers in parallel (each runs `smart.py`). Running `python -m optimize.launch 5` starts 5 workers and waits for all to finish. |
| `check_study.py` | Study inspector. Loads the Postgres-backed study, reports trial counts and duplicates, and saves `db_opt_history.png` showing objective over completed trials. |
| `export_top_trials.py` | CSV exporter. Dumps the top-N completed trials (objective, parameters, and user attributes) to `top_trials.csv`. |


## Objective and Dimensions

**Objective:** Maximize average return on risk (`return_on_risk_percent`).

### Dimensions:

| Name | Values |
|---|---|
| `entry_window` | 0: 9:30–10:30, 1: 9:30–12:00, 2: 12:00–15:00, 3: 15:00–15:45, 4: 9:30–15:45 |
| `stretch_threshold` | 0.001–0.02 (discrete set) |
| `reclaim_percentage` | 0.4–0.8 (percentage of stretch) |
| `cooldown_period_seconds` | 60, 90, 120, 180, 300, 600 |
| `take_profit_percent` | 25, 35, 50, 60, 70, 80, 90, 100 |
| `stop_loss_percent` | -25, -35, -50, -60, -70, -80, -90, -100 |
| `max_trade_duration_seconds` | 60, 120, 180, 240, 300, 600 |
| `strikes_depth` | 1 |
| `option_selection_mode` | `itm`, `otm` |


## Run It

```bash
python -m optimize.smart
```

### Useful Settings Inside `smart.py`:

| Setting | Purpose |
|---|---|
| `STUDY_NAME` | Name of the Optuna study. Use different names to test different strategy variants. |
| `N_TRIALS` | Target number of completed trials. Does not include pruned trials (e.g. failures, not enough trades, duplicates). |
| `ENABLE_PERSISTENCE` | If true, continues adding more trials to existing study. If false, deletes the existing study and starts fresh. |
| `N_STARTUP_TRIALS` | N randomized trials before TPE Sampler kicks in. |
| `N_EI_CANDIDATES` | Candidates per TPE trial. |
| `MIN_TRADE_THRESHOLD` | Minimum trades required to count as a completed trial. Trials are pruned if they don't meet this threshold. |
| `MAX_ATTEMPT_LIMIT` | Cap on total attempts (completed + pruned). Prevents `smart.py` from running indefinitely. |

## Output Examples [NOT COMPLETE]

Need to add artifacts.

- Best trial value and parameters
- Stored metrics per trial (as user attributes)
- Summary of completed/pruned trials and duplicates by source