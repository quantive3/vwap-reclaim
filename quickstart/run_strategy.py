import os
import shutil
from pathlib import Path
import sys
from typing import Dict, Any

def ensure_cache(repo_root: Path) -> Path:
    """
    Ensure the synthetic data is available at polygon_cache/ by copying
    from quickstart/synthetic_data/ via merge-copy.
    Returns the destination path.
    """
    # Support either directory name
    candidates = ["synthetic_data"]
    src = None
    for name in candidates:
        cand = repo_root / "quickstart" / name
        if cand.exists():
            src = cand
            break
    if src is None:
        raise FileNotFoundError(
            f"Missing synthetic data: {repo_root / 'quickstart' / 'synthetic_data'}"
        )
    dst = repo_root / "polygon_cache"

    # Create destination root
    dst.mkdir(exist_ok=True)

    # Merge-copy known subfolders if present
    for sub in ("spy", "chain", "option"):
        s = src / sub
        d = dst / sub
        if s.exists():
            shutil.copytree(s, d, dirs_exist_ok=True)

    return dst


def set_dummy_credentials() -> None:
    """
    Set environment variables to avoid any credential prompts during quickstart.
    """
    os.environ.setdefault("API_KEY", "DUMMY")
    # Optional (used only for optimization workflows)
    os.environ.setdefault("PG_HOST", "DUMMY")
    os.environ.setdefault("PG_PORT", "5432")
    os.environ.setdefault("PG_DATABASE", "DUMMY")
    os.environ.setdefault("PG_USER", "DUMMY")
    os.environ.setdefault("PG_PASSWORD", "DUMMY")


def build_quickstart_params() -> Dict[str, Any]:
    """
    Build a self-contained parameter set that only uses bundled synthetic data.
    Dates are pinned to match the synthetic dataset.
    """
    from datetime import time

    return {
        # Output controls
        "debug_mode": False,
        "silent_mode": False,
        "enable_profiling": False,

        # Backtest period - must match synthetic dataset (Jan 3-5, 2023)
        "start_date": "2023-01-03",
        "end_date": "2023-01-05",

        # Time windows
        "entry_start_time": time(9, 30),
        "entry_end_time": time(15, 45),

        # Entry parameters
        "stretch_threshold": 0.003,
        "reclaim_threshold": 0.0021,
        "cooldown_period_seconds": 120,

        # Exit conditions
        "take_profit_percent": 80,
        "stop_loss_percent": -25,
        "max_trade_duration_seconds": 600,
        "late_entry_cutoff_time": time(15, 54),
        "end_of_day_exit_time": time(15, 54),
        "emergency_exit_time": time(15, 55),

        # Option selection
        "ticker": "SPY",
        "require_same_day_expiry": True,
        "strikes_depth": 1,
        "option_selection_mode": "itm",

        # Position sizing
        "contracts_per_trade": 1,

        # Frictions
        "slippage_amount": 0.02,
        "latency_seconds": 1,
        "brokerage_fee_per_contract": 0.65,
        "exchange_fee_per_contract": 0.65,

        # Data quality thresholds (warnings only if below; no network calls)
        "min_spy_data_rows": 10000,
        "min_option_chain_rows": 10,
        "min_option_price_rows": 800,
        "timestamp_mismatch_threshold": 0,
        "price_staleness_threshold_seconds": 10,
        "report_stale_prices": True,
    }


def verify_cache_for_params(cache_dir: Path, params: Dict[str, Any]) -> None:
    """
    Verify required cache files exist for the configured dates. Fails fast to avoid API calls.
    """
    import pandas as pd

    spy_dir = cache_dir / "spy"
    chain_dir = cache_dir / "chain"
    option_dir = cache_dir / "option"

    for d in (spy_dir, chain_dir, option_dir):
        if not d.exists():
            raise FileNotFoundError(f"Expected cache subdirectory missing: {d}")

    business_days = pd.date_range(start=params["start_date"], end=params["end_date"], freq="B")
    ticker = params["ticker"]

    missing = []
    for day in business_days:
        date_str = day.strftime("%Y-%m-%d")
        spy_file = spy_dir / f"{ticker}_{date_str}.pkl"
        chain_file = chain_dir / f"{ticker}_chain_{date_str}.pkl"
        if not spy_file.exists():
            missing.append(str(spy_file))
        if not chain_file.exists():
            missing.append(str(chain_file))

    if missing:
        details = "\n  - " + "\n  - ".join(missing)
        raise FileNotFoundError(
            "Synthetic cache is incomplete for selected dates. Missing files:" + details
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Ensure environment is set to avoid prompts
    set_dummy_credentials()

    # Ensure cache is in the expected location
    cache_dir = ensure_cache(repo_root)

    # Run from repo root so relative paths inside strategy work as intended
    os.chdir(repo_root)
    # Ensure repo root is importable (so 'strategy' package resolves)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print(f"✅ Using synthetic data at: {cache_dir}")
    print("🚀 Running strategy with synthetic data (no API calls)...\n")

    # Build params locally (self-contained quickstart)
    params = build_quickstart_params()

    # Verify cache completeness for the chosen dates to avoid any API attempts
    verify_cache_for_params(cache_dir, params)

    # Run the main script with our hardwired params to reproduce identical logging/output
    import importlib
    import runpy
    params_module = importlib.import_module("strategy.params")

    original_init = params_module.initialize_parameters
    try:
        def fake_init():
            return params
        # Monkeypatch initialize_parameters so strategy.main uses our hardwired params
        params_module.initialize_parameters = fake_init  # type: ignore
        runpy.run_module("strategy.main", run_name="__main__")
    finally:
        # Restore original function
        params_module.initialize_parameters = original_init  # type: ignore


if __name__ == "__main__":
    main()
