import pytest
from strategy.data import load_spy_data, set_track_issue_function
from strategy.params import initialize_parameters

class DummyResponse:
    status_code = 200
    def __init__(self, payload):
        self._payload = payload
    def json(self):
        return self._payload

@pytest.fixture(autouse=True)
def patch_requests(monkeypatch):
    # 1672835400000 ms == 2023-01-04 09:30:00 UTC → 09:30 US/Eastern
    ms = 1672835400000
    payload = {"results": [{"t": ms, "o": 100, "h": 101, "l": 99, "c": 100, "v": 10, "vw": 100.5, "n": 5} ]}
    monkeypatch.setattr("strategy.data.requests.get", lambda *args, **kwargs: DummyResponse(payload))

def test_load_spy_data(tmp_path):
    # stub issue tracker to avoid NoneType errors
    set_track_issue_function(lambda *args, **kwargs: None)
    cache_dir = tmp_path / "cache"
    (cache_dir / "spy").mkdir(parents=True)
    params = initialize_parameters()
    # silence prints and warnings
    params["silent_mode"] = True
    # allow minimal data rows
    params["min_spy_data_rows"] = 1

    # Expect a ValueError due to NaNs in vwap_running
    with pytest.raises(ValueError) as excinfo:
        load_spy_data("2023-01-04", str(cache_dir), "DUMMY_KEY", params, debug_mode=False)
    # It should mention NaNs detected in vwap_running
    assert "NaNs detected in vwap_running" in str(excinfo.value)
