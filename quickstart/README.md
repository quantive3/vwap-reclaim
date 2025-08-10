### Quickstart

Run the strategy with bundled synthetic data. No API key or network calls required.

### One-time setup

```powershell
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run

```powershell
python quickstart/run_strategy.py
```

What this script does:
- Copies `quickstart/synthetic_data/` into `polygon_cache/` (merge-copy)
- Sets dummy credentials to avoid prompts
- Builds a local `params` dict pinned to Jan 3–5, 2023
- Verifies required cache files exist
- Temporarily injects those params and runs `strategy.main`

### Parameters used

| Name | Value |
|---|---|
| Dates | 2023-01-03 to 2023-01-05 |
| Entry Window | 09:30–15:45 |
| Stretch / Reclaim | 0.003 / 0.0021 |
| Cooldown | 120s |
| Exits | TP 80%, SL -25%, Max 600s, EOD 15:54, Emergency 15:55 |
| Option Selection | `SPY`, same-day expiry, `itm`, `strikes_depth=1` |
| Frictions | Slippage $0.02, Latency 1s, Fees $0.65 + $0.65 one-way |

### Common issues

- Missing cache files: the script will list missing paths and stop. Ensure `quickstart/synthetic_data/spy/` and `chain/` are present for the selected dates.
- Modified working directory: the script `chdir`s to repo root to ensure imports work.

For full strategy details, see [strategy/README.md](../strategy/).


