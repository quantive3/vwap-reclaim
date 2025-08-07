import pandas as pd
from datetime import datetime, time, timedelta
import pytest
from strategy.exits import (
    evaluate_exit_conditions,
    check_emergency_exit_time,
    process_emergency_exit,
    process_exits_for_contract,
    set_track_issue_function,
    set_issue_tracker
)

# stub tracker
tracker = {"risk_management": {"emergency_exits": 0, "emergency_exit_dates": set()}}
set_track_issue_function(lambda *args, **kwargs: None)
set_issue_tracker(tracker)

def make_future_df(entry_time, prices, vwap=None):
    times = [entry_time + timedelta(seconds=i) for i in range(len(prices))]
    df = pd.DataFrame({"ts_raw": times, "close": prices})
    if vwap is not None:
        df["vwap"] = vwap
    return df


def test_evaluate_exit_and_emergency():
    entry = datetime(2023,1,4,9,30)
    df = make_future_df(entry, prices=[10, 18], vwap=[None, None])
    contract = {"entry_time": entry, "entry_option_price": 10, "ticker": "T"}
    # emergency exit check
    assert not check_emergency_exit_time(entry, {"emergency_exit_time": time(15,58)})
    assert check_emergency_exit_time(datetime(2023,1,4,15,58), {})
    out = process_emergency_exit(contract.copy(), datetime(2023,1,4,15,58), df, {"silent_mode": True})
    assert out["exit_reason"] == "emergency_exit"
    assert out["is_closed"]


def test_process_exits_for_contract_take_profit():
    entry = datetime(2023,1,4,9,30)
    future = make_future_df(entry, prices=[10, 11, 20], vwap=[None, None, None])
    c = {
        "ticker": "X",
        "entry_time": entry,
        "entry_option_price": 10,
        "df_option_aligned": future
    }
    params = {
        "take_profit_percent": 25,
        "stop_loss_percent": -50,
        "end_of_day_exit_time": time(16,0),
        "max_trade_duration_seconds": 9999,
        "silent_mode": True,
        "price_staleness_threshold_seconds": 999,
        "debug_mode": False
    }
    out = process_exits_for_contract(c, params)
    assert out["exit_reason"] == "take_profit"
    assert abs(out["pnl_percent"] - 100) < 1e-6
    assert out["is_closed"]
