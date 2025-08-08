import runpy
import os
import shutil
from pathlib import Path
from datetime import time
import strategy.params as params_module

def test_jan35_regression(capsys, monkeypatch, tmp_path):
    # 1) Ensure we never get prompted for an API key
    monkeypatch.setenv("API_KEY", "DUMMY")

    # Point the runtime to use our synthetic cache by running from a temp CWD
    synthetic = Path(__file__).parent / "synthetic_cache"
    dest = tmp_path / "polygon_cache"
    shutil.copytree(synthetic, dest)
    monkeypatch.chdir(tmp_path)

    # 2) Completely override initialize_parameters() with the exact dict we want
    def fake_init():
        return {
            "debug_mode": False,
            "silent_mode": False,
            "enable_profiling": False,
            "start_date": "2023-01-03",
            "end_date":   "2023-01-05",
            "entry_start_time": time(9, 30),
            "entry_end_time":   time(15, 45),
            "stretch_threshold":           0.003,
            "reclaim_threshold":           0.0021,
            "cooldown_period_seconds":     120,
            "take_profit_percent":         80,
            "stop_loss_percent":          -25,
            "max_trade_duration_seconds": 600,
            "late_entry_cutoff_time": time(15, 54),
            "end_of_day_exit_time":    time(15, 54),
            "emergency_exit_time":     time(15, 55),
            "ticker": "SPY",
            "require_same_day_expiry": True,
            "strikes_depth": 1,
            "option_selection_mode": "itm",
            "contracts_per_trade": 1,
            "slippage_amount": 0.02,
            "latency_seconds": 1,
            "brokerage_fee_per_contract": 0.65,
            "exchange_fee_per_contract": 0.65,
            "min_spy_data_rows":          10000,
            "min_option_chain_rows":      10,
            "min_option_price_rows":      10000,
            "timestamp_mismatch_threshold":          0,
            "price_staleness_threshold_seconds":     10,
            "report_stale_prices":                 True,
        }
    monkeypatch.setattr(params_module, "initialize_parameters", fake_init)

    # 3) Run the entire strategy as a script
    runpy.run_module("strategy.main", run_name="__main__")

    # 4) Capture stdout and assert the baseline is byte-for-byte present
    out = capsys.readouterr().out
    baseline = Path(__file__).parent / "fixtures" / "jan35_baseline.txt"
    expected = baseline.read_text(encoding="utf-8")
    assert expected in out, "🚨 Summary output has diverged from the baseline!"
