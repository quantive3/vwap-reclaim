# === Data Loading Module ===
import os
import pandas as pd
import requests
import hashlib
import numpy as np
from datetime import time
from filelock import FileLock

# === Placeholder/Injection Plumbing ===
# These variables will be set by external code
_issue_tracker = None
_track_issue_function = None
_hash_generation_function = None

# Setter functions for injection
def set_issue_tracker(tracker):
    """
    Set the issue tracker object.
    
    Args:
        tracker: Issue tracker dictionary
    """
    global _issue_tracker
    _issue_tracker = tracker

def set_track_issue_function(func):
    """
    Set the track_issue function.
    
    Args:
        func: Function for tracking issues
    """
    global _track_issue_function
    _track_issue_function = func

def set_hash_generation_function(func):
    """
    Set the hash generation function.
    
    Args:
        func: Function for generating dataframe hashes
    """
    global _hash_generation_function
    _hash_generation_function = func

# === Cache Directory Setup ===
def setup_cache_directories(cache_dir):
    """
    Set up the cache directory structure for storing data.
    
    Args:
        cache_dir (str): Base cache directory path
        
    Returns:
        tuple: Paths to SPY, chain, and option cache directories
    """
    spy_dir = os.path.join(cache_dir, "spy")
    chain_dir = os.path.join(cache_dir, "chain")
    option_dir = os.path.join(cache_dir, "option")

    for d in [spy_dir, chain_dir, option_dir]:
        os.makedirs(d, exist_ok=True)
        
    return spy_dir, chain_dir, option_dir

# === Cache Lock Wrapper ===
def load_with_cache_lock(cache_path, fetch_func, *args, lock_timeout=60, **kwargs):
    """
    Load data from cache_path using a file lock. If cache file doesn't exist,
    fetch data by calling fetch_func(*args, **kwargs), save to cache_path, and return it.
    Only one process will fetch and write data at a time.
    """
    # Check for existing cache without locking (allow concurrent reads)
    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)
    lock_path = f"{cache_path}.lock"
    lock = FileLock(lock_path, timeout=lock_timeout)
    with lock:
        # Re-check inside lock in case another process created it
        if os.path.exists(cache_path):
            return pd.read_pickle(cache_path)
        data = fetch_func(*args, **kwargs)
        # Only write to pickle if data is not None and has at least one row
        if data is not None and len(data) > 0:
            data.to_pickle(cache_path)
        return data

# === Internal Fetch Functions ===
def _fetch_spy(date, api_key, params, debug_mode=False):
    """
    Fetch SPY price data from Polygon API and process it.
    
    Args:
        date (str): Date string in format 'YYYY-MM-DD'
        api_key (str): Polygon API key
        params (dict): Strategy parameters
        debug_mode (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: DataFrame with SPY price and VWAP data for the specified date
    """
    ticker = params['ticker']
    base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/second/{date}/{date}"
    headers = {"Authorization": f"Bearer {api_key}"}

    all_results = []
    cursor = None
    while True:
        url = f"{base_url}?adjusted=true&limit=50000"
        if cursor:
            url += f"&cursor={cursor}"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            error_msg = f"SPY price request failed: {response.status_code}"
            _track_issue_function("errors", "api_connection_failures", error_msg, level="error", date=date)
            raise Exception(error_msg)

        json_data = response.json()
        results = json_data.get("results", [])
        all_results.extend(results)

        if "next_url" in json_data:
            cursor = json_data["next_url"].split("cursor=")[-1]
        else:
            break

    if not all_results:
        no_data_msg = f"No {ticker} data for {date} — skipping."
        if not params.get('silent_mode', False):
            print(f"⚠️ {no_data_msg}")
        _track_issue_function("warnings", f"no_{ticker}_data", no_data_msg, date=date)
        return None

    df_raw = pd.DataFrame(all_results)
    df_raw["timestamp"] = pd.to_datetime(df_raw["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
    df_raw.rename(columns={
        "o": "open", "h": "high", "l": "low", "c": "close",
        "v": "volume", "vw": "vw", "n": "trades"
    }, inplace=True)
    df_raw = df_raw[["timestamp", "open", "high", "low", "close", "volume", "vw", "trades"]]

    df_rth = df_raw[
        (df_raw["timestamp"].dt.time >= time(9, 30)) &
        (df_raw["timestamp"].dt.time <= time(16, 0))
    ].sort_values("timestamp").reset_index(drop=True)

    start_time = pd.Timestamp(f"{date} 09:30:00", tz="US/Eastern")
    end_time = pd.Timestamp(f"{date} 16:00:00", tz="US/Eastern")
    full_index = pd.date_range(start=start_time, end=end_time, freq="1s", tz="US/Eastern")

    df_rth_filled = df_rth.set_index("timestamp").reindex(full_index).ffill().reset_index()
    df_rth_filled.rename(columns={"index": "timestamp"}, inplace=True)
    df_rth_filled["ts_raw"] = df_rth_filled["timestamp"]
    df_rth_filled["timestamp"] = df_rth_filled["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    # === Updated VWAP using volume-weighted price (vw) ===
    df_rth_filled["cum_pv"] = (df_rth_filled["vw"] * df_rth_filled["volume"]).cumsum()
    df_rth_filled["cum_vol"] = df_rth_filled["volume"].cumsum()
    df_rth_filled["vwap_running"] = df_rth_filled["cum_pv"] / df_rth_filled["cum_vol"]

    if df_rth_filled["vwap_running"].isna().any():
        error_msg = "NaNs detected in vwap_running — check data or ffill logic"
        _track_issue_function("errors", "other", error_msg, level="error", date=date)
        raise ValueError(f"❌ {error_msg}")

    if not df_rth_filled["vwap_running"].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
        error_msg = "Non-finite values (inf/-inf) in vwap_running"
        _track_issue_function("errors", "other", error_msg, level="error", date=date)
        raise ValueError(f"❌ {error_msg}")

    # Check for NaNs and non-finite values in critical columns
    critical_columns = ["open", "high", "low", "close", "volume", "vw"]
    for column in critical_columns:
        if df_rth_filled[column].isna().any():
            error_msg = f"NaNs detected in {column} — check data integrity"
            _track_issue_function("errors", "other", error_msg, level="error", date=date)
            raise ValueError(f"❌ {error_msg}")
        if not df_rth_filled[column].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
            error_msg = f"Non-finite values (inf/-inf) in {column} — check data integrity"
            _track_issue_function("errors", "other", error_msg, level="error", date=date)
            raise ValueError(f"❌ {error_msg}")

    if len(df_rth_filled) < params['min_spy_data_rows']:
        short_data_msg = f"SPY data for {date} is unusually short with only {len(df_rth_filled)} rows after pulling from API. This may indicate incomplete data."
        if not params.get('silent_mode', False):
            print(f"⚠️ {short_data_msg}")
        _track_issue_function("warnings", "short_data_warnings", short_data_msg, date=date)
        
    return df_rth_filled

# === Loader Functions ===
def load_spy_data(date, cache_dir, api_key, params, debug_mode=False):
    """
    Load SPY price data for a given date, either from cache or from Polygon API.
    
    Args:
        date (str): Date string in format 'YYYY-MM-DD'
        cache_dir (str): Base cache directory
        api_key (str): Polygon API key
        params (dict): Strategy parameters
        debug_mode (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: DataFrame with SPY price and VWAP data for the specified date
    """
    ticker = params['ticker']
    spy_dir = os.path.join(cache_dir, "spy")
    spy_path = os.path.join(spy_dir, f"{ticker}_{date}.pkl")
    
    # Track whether the fetch function was called
    was_fetched = [False]
    
    def fetch_wrapper():
        was_fetched[0] = True
        return _fetch_spy(date, api_key, params, debug_mode)
    
    # Use the file-lock wrapper to handle caching
    df_rth_filled = load_with_cache_lock(
        spy_path,
        fetch_wrapper,
        lock_timeout=60
    )
    
    # Check if df_rth_filled is None before proceeding
    if df_rth_filled is None:
        return None
    
    # Re-attach debug prints, hash generation, and warnings
    if debug_mode:
        if was_fetched[0]:
            print("💾 SPY data pulled and cached.")
        else:
            print("📂 SPY data loaded from cache.")
        # Generate and log hash for SPY data
        _hash_generation_function(df_rth_filled, f"SPY {date}")
    
    if len(df_rth_filled) < params['min_spy_data_rows']:
        short_data_msg = f"SPY data for {date} is unusually short with only {len(df_rth_filled)} rows. This may indicate incomplete data."
        if not params.get('silent_mode', False):
            print(f"⚠️ {short_data_msg}")
        _track_issue_function("warnings", "short_data_warnings", short_data_msg, date=date)
    
    return df_rth_filled

def load_chain_data(date, cache_dir, api_key, params, debug_mode=False):
    """
    Load option chain data for a given date, either from cache or from Polygon API.
    
    Args:
        date (str): Date string in format 'YYYY-MM-DD'
        cache_dir (str): Base cache directory
        api_key (str): Polygon API key
        params (dict): Strategy parameters
        debug_mode (bool): Whether to print debug information
        
    Returns:
        pd.DataFrame: DataFrame with option chain data for the specified date
    """
    ticker = params['ticker']
    chain_dir = os.path.join(cache_dir, "chain")
    chain_path = os.path.join(chain_dir, f"{ticker}_chain_{date}.pkl")
    
    # Track whether the fetch function was called
    was_fetched = [False]
    
    def fetch_chain():
        was_fetched[0] = True
        
        def fetch_contract_type(contract_type):
            url = (
                f"https://api.polygon.io/v3/reference/options/contracts"
                f"?underlying_ticker={ticker}"
                f"&contract_type={contract_type}"
                f"&expiration_date.gte={date}"  # Greater than or equal to current date
                f"&expiration_date.lte={date}"  # Less than or equal to current date (same day)
                f"&as_of={date}"
                f"&order=asc"
                f"&limit=1000"
                f"&sort=ticker"
                f"&apiKey={api_key}"
            )
            resp = requests.get(url)
            if resp.status_code != 200:
                error_msg = f"{contract_type.upper()} request failed: {resp.status_code}"
                _track_issue_function("errors", "api_connection_failures", error_msg, level="error", date=date)
                raise Exception(error_msg)
            df = pd.DataFrame(resp.json().get("results", []))
            df["option_type"] = contract_type
            return df

        df_calls = fetch_contract_type("call")
        df_puts = fetch_contract_type("put")
        df_chain = pd.concat([df_calls, df_puts], ignore_index=True)
        df_chain["ticker_clean"] = df_chain["ticker"].str.replace("O:", "", regex=False)

        return df_chain
    
    # Use the file-lock wrapper to handle caching
    df_chain = load_with_cache_lock(
        chain_path,
        fetch_chain,
        lock_timeout=60
    )
    
    # Re-attach debug prints, hash generation, and warnings
    if debug_mode:
        if was_fetched[0]:
            print("💾 Option chain pulled and cached.")
            # Generate and log hash for option chain data
            _hash_generation_function(df_chain, f"Chain {date}")
        else:
            print("📂 Option chain loaded from cache.")
            # Generate and log hash for option chain data
            _hash_generation_function(df_chain, f"Chain {date}")
    
    # Check for unusually short data
    if len(df_chain) < params['min_option_chain_rows']:
        short_data_msg = f"Option chain data for {date} is unusually short with only {len(df_chain)} rows. This may indicate incomplete data."
        if not params.get('silent_mode', False):
            print(f"⚠️ {short_data_msg}")
        _track_issue_function("warnings", "short_data_warnings", short_data_msg, date=date)

    if df_chain.empty:
        no_data_msg = f"No option chain data for {date} — skipping."
        if not params.get('silent_mode', False):
            print(f"⚠️ {no_data_msg}")
        _track_issue_function("warnings", "other", no_data_msg, date=date)
        return None
        
    return df_chain

def load_option_data(option_ticker, date, cache_dir, df_rth_filled, api_key, params, signal_idx=None):
    """
    Load and process option price data for a given option ticker and date.
    
    Parameters:
    - option_ticker: The option ticker symbol to load data for
    - date: Date string in format 'YYYY-MM-DD'
    - cache_dir: Base cache directory path
    - df_rth_filled: DataFrame with SPY price data for timestamp alignment
    - api_key: Polygon API key
    - params: Strategy parameters
    - signal_idx: Optional signal index for debug output
    
    Returns:
    - df_option_aligned: DataFrame with aligned option price data
    - option_entry_price: The entry price for the option (or None if no valid price found)
    - status: Dictionary with status information including:
        - success: Boolean indicating if loading was successful
        - error_message: Error message if loading failed
        - mismatch_count: Number of timestamp mismatches
    """
    # Initialize return status
    status = {
        'success': False,
        'error_message': None,
        'mismatch_count': 0
    }
    
    # Setup cache path
    option_dir = os.path.join(cache_dir, "option")
    option_path = os.path.join(option_dir, f"{date}_{option_ticker.replace(':', '')}.pkl")
    
    # Initialize return values
    df_option_aligned = None
    option_entry_price = None

    # === Load or pull option price data ===
    try:
        # Track whether the fetch function was called
        was_fetched = [False]
        
        def fetch_option():
            was_fetched[0] = True
            
            option_url = (
                f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/second/"
                f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={api_key}"
            )
            resp = requests.get(option_url)
            option_results = resp.json().get("results", [])
            df_option = pd.DataFrame(option_results)

            if df_option.empty:
                missing_data_msg = f"No option price data for {option_ticker} on {date} — skipping this entry."
                if not params.get('silent_mode', False):
                    print(f"⚠️ {missing_data_msg}")
                _track_issue_function("errors", "missing_option_price_data", missing_data_msg, level="error", date=date)
                _issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                status['error_message'] = missing_data_msg
                return None

            df_option["timestamp"] = pd.to_datetime(df_option["t"], unit="ms", utc=True).dt.tz_convert("US/Eastern")
            df_option.rename(columns={
                "o": "open", "h": "high", "l": "low", "c": "close",
                "v": "volume", "vw": "vwap", "n": "trades"
            }, inplace=True)
            df_option = df_option[["timestamp", "open", "high", "low", "close", "volume", "vwap", "trades"]]

            df_option_rth = df_option[
                (df_option["timestamp"].dt.time >= time(9, 30)) &
                (df_option["timestamp"].dt.time <= time(16, 0))
            ].sort_values("timestamp").reset_index(drop=True)
            
            # Mark rows with actual data (before forward filling)
            df_option_rth['is_actual_data'] = True
                
            return df_option_rth
        
        # Use the file-lock wrapper to handle caching
        df_option_rth = load_with_cache_lock(
            option_path,
            fetch_option,
            lock_timeout=60
        )
        
        # Handle case where fetch_option returns None due to empty data
        if df_option_rth is None:
            return df_option_aligned, option_entry_price, status
        
        # Re-attach debug prints, hash generation, and warnings
        if params['debug_mode']:
            if was_fetched[0]:
                print(f"💾 Option price data for {option_ticker} pulled and cached.")
                # Generate and log hash for option price data
                _hash_generation_function(df_option_rth, f"Option {option_ticker} {date}")
            else:
                print(f"📂 Option price data for {option_ticker} loaded from cache.")
                # Generate and log hash for option price data
                _hash_generation_function(df_option_rth, f"Option {option_ticker} {date}")
                
        if len(df_option_rth) < params['min_option_price_rows']:
            source_context = "after pulling from API" if was_fetched[0] else "from cache"
            short_data_msg = f"Option price data for {option_ticker} on {date} is unusually short with only {len(df_option_rth)} rows {source_context}. This may indicate incomplete data."
            if not params.get('silent_mode', False):
                print(f"⚠️ {short_data_msg}")
            _track_issue_function("warnings", "short_data_warnings", short_data_msg, date=date)
        
        # === Timestamp alignment check ===
        # Add is_actual_data column if loading from cache and column doesn't exist
        if 'is_actual_data' not in df_option_rth.columns:
            df_option_rth['is_actual_data'] = True
        
        # Create a copy of the actual data flags before alignment
        actual_data_timestamps = df_option_rth[df_option_rth['is_actual_data']]['timestamp'].copy()
        
        # Align and forward fill the option data using the recommended pattern
        df_option_aligned = df_option_rth.set_index("timestamp").reindex(df_rth_filled["ts_raw"])
        df_option_aligned = df_option_aligned.ffill()
        df_option_aligned = df_option_aligned.infer_objects(copy=False)
        df_option_aligned = df_option_aligned.reset_index()
        df_option_aligned.rename(columns={"index": "ts_raw"}, inplace=True)
        
        # Initialize staleness tracking
        df_option_aligned['is_actual_data'] = df_option_aligned['ts_raw'].isin(actual_data_timestamps)
        # Initialize as float64 type to properly handle inf values
        df_option_aligned['seconds_since_update'] = pd.Series(0.0, index=df_option_aligned.index, dtype='float64')
        
        # Calculate staleness (seconds since last actual data point)
        # --- Vectorized staleness calculation ---
        # 1. Build a Series of actual timestamps (NaT where False)
        actual_ts = df_option_aligned['ts_raw'].where(df_option_aligned['is_actual_data'])

        # 2. Forward-fill to carry the last actual timestamp to ensuing rows
        last_actual = actual_ts.ffill()

        # 3. Compute time differences in seconds
        staleness = (df_option_aligned['ts_raw'] - last_actual).dt.total_seconds()

        # 4. Fill NaN (before first actual) with infinity
        df_option_aligned['seconds_since_update'] = staleness.fillna(float('inf'))
        
        # Define a threshold for allowable mismatches
        mismatch_threshold = params['timestamp_mismatch_threshold']
        
        # Check for timestamp mismatches
        mismatch_count = (~df_option_aligned["ts_raw"].eq(df_rth_filled["ts_raw"])).sum()
        status['mismatch_count'] = mismatch_count
        
        # Track timestamp mismatches
        if mismatch_count > 0:
            if mismatch_count > mismatch_threshold:
                mismatch_msg = f"Timestamp mismatch in {mismatch_count} rows exceeds threshold of {mismatch_threshold}"
                _track_issue_function("data_integrity", "timestamp_mismatches", mismatch_msg, date=date)
            else:
                mismatch_msg = f"Timestamp mismatch in {mismatch_count} rows (below threshold of {mismatch_threshold})"
                _track_issue_function("warnings", "timestamp_mismatches_below_threshold", mismatch_msg, date=date)
        
        # Hash-based timestamp verification as additional sanity check
        def hash_timestamps(df):
            return hashlib.md5("".join(df["ts_raw"].astype(str)).encode()).hexdigest()

        if params['debug_mode']:
            spy_hash = hash_timestamps(df_rth_filled)
            opt_hash = hash_timestamps(df_option_aligned)
            hash_match = spy_hash == opt_hash

            # Track hash mismatches in debug mode
            if not hash_match:
                hash_mismatch_msg = f"Hash mismatch between SPY and option data"
                _track_issue_function("data_integrity", "hash_mismatches", hash_mismatch_msg, date=date)
        
        # Only print the debug information if debug mode is on
        if params['debug_mode']:
            if signal_idx is not None:
                print(f"🧪 Signal #{signal_idx+1}: Timestamp mismatches for {option_ticker}: {mismatch_count}")
            else:
                print(f"🧪 Timestamp mismatches for {option_ticker}: {mismatch_count}")
            print(f"⏱️ SPY rows: {len(df_rth_filled)}")
            print(f"⏱️ OPT rows: {len(df_option_aligned)}")
            print(f"🔐 SPY hash:  {spy_hash}")
            print(f"🔐 OPT hash:  {opt_hash}")
            print(f"🔍 Hash match: {hash_match}")
            
            # Generate and log hash for the aligned option data for consistency checks
            _hash_generation_function(df_option_aligned, f"Aligned Option {option_ticker} {date}")
        
        # Check if mismatches exceed the threshold
        if mismatch_count > mismatch_threshold:
            status['error_message'] = f"Timestamp mismatch in {mismatch_count} rows exceeds threshold of {mismatch_threshold} — skipping this entry."
            if not params.get('silent_mode', False):
                print(f"⚠️ {status['error_message']}")
            _issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
            return df_option_aligned, option_entry_price, status
        
        # Success! Option data loaded and aligned successfully
        status['success'] = True
        return df_option_aligned, option_entry_price, status
        
    except Exception as e:
        error_msg = f"Error loading option data for {option_ticker} on {date}: {str(e)}"
        if not params.get('silent_mode', False):
            print(f"❌ {error_msg}")
        status['error_message'] = error_msg
        _track_issue_function("errors", "other", error_msg, level="error", date=date)
        return df_option_aligned, option_entry_price, status