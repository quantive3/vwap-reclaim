import pandas as pd
import pytest
from strategy.option_select import (
    select_option_contract,
    set_track_issue_function,
    set_debug_mode
)

# no-op tracker so we don’t explode
set_track_issue_function(lambda *args, **kwargs: None)
set_debug_mode(False)

@pytest.fixture
def base_chain():
    today = "2023-01-04"
    data = []
    for opt in ["call", "put"]:
        for strike in [99, 100, 101]:
            data.append({
                "option_type": opt,
                "expiration_date": today,
                "strike_price": strike,
                "ticker": f"O:{opt}:{strike}",
            })
    return pd.DataFrame(data)


def test_itm_call_and_put_selection(base_chain):
    params = {
        "option_selection_mode": "itm",
        "strikes_depth": 1,
        "require_same_day_expiry": True,
        "debug_mode": False,
        "silent_mode": True
    }
    # use a timestamp that matches fixture expiration_date
    sig = {"stretch_label": "below", "ts_raw": pd.Timestamp("2023-01-04 09:30:00")}
    c = select_option_contract(sig, base_chain, spy_price=100, params=params)
    assert c["option_type"] == "call"
    assert c["strike_price"] == 99

    sig["stretch_label"] = "above"
    c = select_option_contract(sig, base_chain, spy_price=100, params=params)
    assert c["option_type"] == "put"
    assert c["strike_price"] == 101


def test_otm_and_atm_and_invalid_mode(base_chain):
    params = {
        "option_selection_mode": "otm",
        "strikes_depth": 2,
        "require_same_day_expiry": True,
        "debug_mode": False,
        "silent_mode": True
    }
    # use a timestamp that matches fixture expiration_date
    sig = {"stretch_label": "below", "ts_raw": pd.Timestamp("2023-01-04 09:30:00")}
    c = select_option_contract(sig, base_chain, spy_price=100, params=params)
    assert c["option_type"] == "call" and c["strike_price"] == 101

    params["option_selection_mode"] = "atm"
    c = select_option_contract(sig, base_chain, spy_price=100, params=params)
    assert c["strike_price"] == 100

    c = select_option_contract(sig, base_chain, spy_price=105, params=params)
    assert c is None

    params["option_selection_mode"] = "bogus"
    c = select_option_contract(sig, base_chain, spy_price=100, params=params)
    assert c["selection_mode_used"].startswith("itm")
