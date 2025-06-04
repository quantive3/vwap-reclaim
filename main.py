# === STEP 1: Mount Google Drive for caching ===
from google.colab import drive
drive.mount('/content/drive')

# === STEP 2: Manual API Key Input ===
API_KEY = input("🔑 Enter your Polygon API key: ").strip()

# === STEP 3: Imports and setup ===
import pandas as pd
# Set pandas option to use future behavior for fill operations
pd.set_option('future.no_silent_downcasting', True)
import requests
import os
from datetime import time
import hashlib
import numpy as np

# === STEP 6: Define Parameters ===
PARAMS = {
    # Backtest period
    'start_date': "2023-01-23",
    'end_date': "2023-01-27",
    
    # Strategy parameters
    'stretch_threshold': 0.003,  # 0.3%
    'reclaim_threshold': 0.002,  # 0.2% - should always be less than stretch threshold
    'cooldown_period_seconds': 60,  # Cooldown period in seconds
    
    # Time windows
    'entry_start_time': time(9, 30),
    'entry_end_time': time(16, 0),
    
    # Instrument selection
    'ticker': 'SPY',
    'require_same_day_expiry': True,  # Whether to strictly require same-day expiry options
    'strikes_depth': 1,  # Number of strikes from ATM to target (1 = closest, 2 = second closest, etc.). Always use 1 or greater.
    'option_selection_mode': 'itm',  # Options: 'itm', 'otm', or 'atm' - determines whether to select in-the-money, out-of-money, or at-the-money options
    
    # Data quality thresholds - for error checking
    'min_spy_data_rows': 10000,  # Minimum acceptable rows for SPY data
    'min_option_chain_rows': 10,  # Minimum acceptable rows for option chain data
    'min_option_price_rows': 10000,  # Minimum acceptable rows for option price data
    'timestamp_mismatch_threshold': 0,  # Maximum allowable timestamp mismatches
    'price_staleness_threshold_seconds': 10,  # Maximum allowable staleness in seconds for option prices
    'report_stale_prices': True,  # Enable/disable reporting of stale prices
    
    # Debug settings
    'debug_mode': True,  # Enable/disable debug outputs
}

# Use params for variables that were previously global
DEBUG_MODE = PARAMS['debug_mode']
start_date = PARAMS['start_date']
end_date = PARAMS['end_date']
business_days = pd.date_range(start=start_date, end=end_date, freq="B")
ticker = PARAMS['ticker']

# === STEP 4: Caching paths ===
CACHE_DIR = "/content/drive/MyDrive/polygon_cache"
SPY_DIR = os.path.join(CACHE_DIR, "spy")
CHAIN_DIR = os.path.join(CACHE_DIR, "chain")
OPTION_DIR = os.path.join(CACHE_DIR, "option")

for d in [SPY_DIR, CHAIN_DIR, OPTION_DIR]:
    os.makedirs(d, exist_ok=True)

# === STEP 7: Stretch Signal Detection ===
def detect_stretch_signal(df_rth_filled, params):
    """
    Detects stretch signals when SPY price moves beyond VWAP by ±0.3%.

    Parameters:
    - df_rth_filled: DataFrame containing SPY price and VWAP data.
    - params: Dictionary of parameters including stretch threshold.

    Returns:
    - signals: DataFrame with stretch signals.
    """
    stretch_threshold = params['stretch_threshold']
    df_rth_filled['stretch_signal'] = (
        (df_rth_filled['close'] > df_rth_filled['vwap_running'] * (1 + stretch_threshold)) |
        (df_rth_filled['close'] < df_rth_filled['vwap_running'] * (1 - stretch_threshold))
    )
    df_rth_filled['percentage_stretch'] = (
        (df_rth_filled['close'] - df_rth_filled['vwap_running']) / df_rth_filled['vwap_running']
    ) * 100
    signals = df_rth_filled[df_rth_filled['stretch_signal']].copy()
    
    # Log the first 5 stretch signals
#    if DEBUG_MODE:
#        print("\nFirst 5 Stretch Signals:")
#        print(signals[['timestamp', 'close', 'vwap_running', 'percentage_stretch']].head())
    
    # Tag each stretch signal with "above" or "below"
    signals['stretch_label'] = np.where(
        signals['close'] > signals['vwap_running'], 'above', 'below'
    )
    
    # === Filter signals to within sweepable time range ===
    entry_start = params['entry_start_time']
    entry_end = params['entry_end_time']

    # Count signals before time filtering for diagnostic purposes
    signals_before_time_filter = len(signals)

    try:
        # Convert timestamp to time objects for filtering
        signals['ts_obj'] = signals['ts_raw'].dt.time
        
        # Check for NaN or invalid timestamps before filtering
        invalid_timestamps = signals['ts_raw'].isna().sum()
        if invalid_timestamps > 0 and DEBUG_MODE:
            print(f"⚠️ Found {invalid_timestamps} signals with invalid timestamps before time filtering")
            
        # Apply the time filter - core logic unchanged
        signals = signals[(signals['ts_obj'] >= entry_start) & (signals['ts_obj'] <= entry_end)]
        signals.drop(columns=['ts_obj'], inplace=True)
        signals = signals[signals['ts_raw'].notna()].sort_values("ts_raw").reset_index(drop=True)
        
    except Exception as e:
        # Capture errors in timestamp conversion or filtering
        error_msg = f"❌ Error during time filtering of signals: {str(e)}"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Count signals after time filtering for diagnostic purposes
    signals_after_time_filter = len(signals)
    signals_dropped = signals_before_time_filter - signals_after_time_filter

    # Log time filtering results without affecting logic
    if DEBUG_MODE and signals_dropped > 0:
        print(f"ℹ️ Time filtering: {signals_dropped} signals were outside the {entry_start}-{entry_end} trading window")
        print(f"ℹ️ Signals before time filter: {signals_before_time_filter}, after: {signals_after_time_filter}")

#    if DEBUG_MODE:
#        print("\n🧹 Post-filter signal integrity check:")
#        print(f"  NaT timestamps: {signals['ts_raw'].isna().sum()}")
#        print(f"  Time ordered: {signals['ts_raw'].is_monotonic_increasing}")

        # Log the first 5 stretch labels for "above" and "below"
#    if DEBUG_MODE:
#        print("\nFirst 5 'Above' Stretch Signals:")
#        print(signals[signals['stretch_label'] == 'above'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
#        print("\nFirst 5 'Below' Stretch Signals:")
#        print(signals[signals['stretch_label'] == 'below'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
    
    # Implement cooldown logic
    last_above_signal_time = None
    last_below_signal_time = None
    cooldown_period = pd.Timedelta(seconds=params['cooldown_period_seconds'])

    filtered_signals = []
    for _, row in signals.iterrows():
        current_time = row['ts_raw']
        if row['stretch_label'] == 'above':
            if last_above_signal_time is None or (current_time - last_above_signal_time) >= cooldown_period:
                filtered_signals.append(row)
                last_above_signal_time = current_time
        elif row['stretch_label'] == 'below':
            if last_below_signal_time is None or (current_time - last_below_signal_time) >= cooldown_period:
                filtered_signals.append(row)
                last_below_signal_time = current_time

    # Convert filtered signals to DataFrame
    processed_signals_df = pd.DataFrame(filtered_signals)
    
    # Additional diagnostic for cooldown filtering
#    if DEBUG_MODE:
#        signals_after_cooldown = len(processed_signals_df)
#        signals_dropped_by_cooldown = len(signals) - signals_after_cooldown
#        if signals_dropped_by_cooldown > 0:
#            print(f"ℹ️ Cooldown filtering: {signals_dropped_by_cooldown} signals were dropped due to cooldown period")
#            print(f"ℹ️ Signals before cooldown: {len(signals)}, after: {signals_after_cooldown}")

#    if DEBUG_MODE:
        # Log the first 5 processed 'above' stretch signals
#        print("\nFirst 5 Processed 'Above' Stretch Signals:")
#        print(processed_signals_df[processed_signals_df['stretch_label'] == 'above'][['ts_raw', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
        
        # Log the first 5 processed 'below' stretch signals
#        print("\nFirst 5 Processed 'Below' Stretch Signals:")
#        print(processed_signals_df[processed_signals_df['stretch_label'] == 'below'][['ts_raw', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())

    return processed_signals_df

# === STEP 7b: Detect Partial Reclaims ===
def detect_partial_reclaims(df_rth_filled, stretch_signals, params):
    """
    For each stretch signal, detect if a partial reclaim toward VWAP occurs within the cooldown window.
    Returns stretch signals with reclaim metadata.
    """
    reclaim_threshold = params['reclaim_threshold']
    cooldown_seconds = params['cooldown_period_seconds']
    enriched_signals = []

    for _, row in stretch_signals.iterrows():
        stretch_time = row['ts_raw']
        label = row['stretch_label']
        vwap_at_stretch = row['vwap_running']

        # Extract the reclaim window (up to 60 seconds ahead)
        reclaim_window = df_rth_filled[
            (df_rth_filled['ts_raw'] > stretch_time) &
            (df_rth_filled['ts_raw'] <= stretch_time + pd.Timedelta(seconds=cooldown_seconds))
        ].copy()

        # Define reclaim zone based on stretch direction
        if label == 'below':
            valid_reclaims = reclaim_window[
                reclaim_window['close'] >= reclaim_window['vwap_running'] * (1 - reclaim_threshold)
            ]
        elif label == 'above':
            valid_reclaims = reclaim_window[
                reclaim_window['close'] <= reclaim_window['vwap_running'] * (1 + reclaim_threshold)
            ]
        else:
            continue  # skip malformed

        # Filter reclaim window based on direction-specific threshold
        if not valid_reclaims.empty:
            first_reclaim = valid_reclaims.iloc[0]
            row['entry_intent'] = True
            row['reclaim_ts'] = first_reclaim['ts_raw']
            row['reclaim_price'] = first_reclaim['close']
            row['vwap_at_reclaim'] = first_reclaim['vwap_running']
        else:
            row['entry_intent'] = False
            row['reclaim_ts'] = pd.NaT
            row['reclaim_price'] = np.nan
            row['vwap_at_reclaim'] = np.nan

        enriched_signals.append(row)

    return pd.DataFrame(enriched_signals)

# === STEP 7c: Select Option Contract ===
def select_option_contract(entry_signal, df_chain, spy_price, params):
    """
    Select the appropriate option contract based on stretch direction and selection mode logic.
    
    Parameters:
    - entry_signal: DataFrame row with entry signal information
    - df_chain: DataFrame with option chain data
    - spy_price: Current SPY price at entry
    - params: Strategy parameters including option_selection_mode and strikes_depth
    
    Returns:
    - selected_contract: Dictionary with selected contract details
    """
    # Determine option type based on stretch direction
    stretch_direction = entry_signal['stretch_label']
    
    # Below VWAP stretch → buy call option
    # Above VWAP stretch → buy put option
    option_type = 'call' if stretch_direction == 'below' else 'put'
    
    # Get current date for filtering same-day expiry options
    current_date = entry_signal['ts_raw'].strftime('%Y-%m-%d')
    
    # Filter chain to only include the desired option type AND same-day expiry
    filtered_chain = df_chain[
        (df_chain['option_type'] == option_type) & 
        (df_chain['expiration_date'] == current_date)
    ].copy()
    
    if filtered_chain.empty:
        if params['debug_mode']:
            print(f"⚠️ No same-day expiry {option_type} options available in the chain")
            
        # If no same-day expiry options, check if we allow non-same-day expiry as fallback
        if not params.get('require_same_day_expiry', True):
            filtered_chain = df_chain[df_chain['option_type'] == option_type].copy()
            
            if filtered_chain.empty:
                if params['debug_mode']:
                    print(f"⚠️ No {option_type} options available in the chain at all")
                return None
            else:
                if params['debug_mode']:
                    print(f"ℹ️ Using non-same-day expiry options as fallback")
        else:
            if params['debug_mode']:
                print(f"⚠️ Same-day expiry required but none available - skipping")
            return None
    
    # Calculate absolute difference between each strike and current price for ATM selection
    filtered_chain['abs_diff'] = (filtered_chain['strike_price'] - spy_price).abs()
    
    # Sort by absolute difference to find closest to ATM
    atm_chain = filtered_chain.sort_values('abs_diff')
    
    # Get option selection mode (itm, otm, or atm)
    option_selection_mode = params.get('option_selection_mode', 'itm').lower()
    
    # Get the target strike depth (how many strikes from ATM to go)
    strikes_depth = params.get('strikes_depth', 1)  # Default to 1 if not specified
    
    # Use selection mode logic paths
    if option_selection_mode == 'itm':
        # ===== ITM SELECTION LOGIC =====
        if option_type == 'call':
            # For calls, ITM means strike < price
            itm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not itm_contracts.empty:
                # Sort by strike price in descending order (highest strike first)
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=False)
                
                # Get the nth ITM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm'
            else:
                # If no ITM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
        else:  # put
            # For puts, ITM means strike > price
            itm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not itm_contracts.empty:
                # Sort by strike price in ascending order (lowest strike first)
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=True)
                
                # Get the nth ITM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm'
            else:
                # If no ITM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
    
    elif option_selection_mode == 'otm':
        # ===== OTM SELECTION LOGIC =====
        if option_type == 'call':
            # For calls, OTM means strike > price
            otm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not otm_contracts.empty:
                # Sort by strike price in ascending order (lowest strike first)
                otm_sorted = otm_contracts.sort_values('strike_price', ascending=True)
                
                # Get the nth OTM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(otm_sorted) - 1)
                selected_contract = otm_sorted.iloc[target_idx]
                selection_mode_used = 'otm'
            else:
                # If no OTM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
        else:  # put
            # For puts, OTM means strike < price
            otm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not otm_contracts.empty:
                # Sort by strike price in descending order (highest strike first)
                otm_sorted = otm_contracts.sort_values('strike_price', ascending=False)
                
                # Get the nth OTM strike based on depth parameter
                target_idx = min(strikes_depth - 1, len(otm_sorted) - 1)
                selected_contract = otm_sorted.iloc[target_idx]
                selection_mode_used = 'otm'
            else:
                # If no OTM contracts, fall back to ATM
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
    
    elif option_selection_mode == 'atm':
        # ===== ATM SELECTION LOGIC =====
        # Only select exact price match for ATM (strict definition)
        exact_match = atm_chain[atm_chain['strike_price'] == spy_price]
        if not exact_match.empty:
            selected_contract = exact_match.iloc[0]
            selection_mode_used = 'atm_exact'
        else:
            # No exact ATM match found, return None to skip this contract
            if params['debug_mode']:
                print(f"⚠️ ATM mode requested but no exact match to {spy_price} found - skipping")
            return None
    
    else:
        # Invalid selection mode, default to ITM
        if params['debug_mode']:
            print(f"⚠️ Invalid option_selection_mode: {option_selection_mode}. Defaulting to 'itm'")
        
        # Reuse ITM logic as default case
        if option_type == 'call':
            itm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not itm_contracts.empty:
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=False)
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm_default'
            else:
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
        else:  # put
            itm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not itm_contracts.empty:
                itm_sorted = itm_contracts.sort_values('strike_price', ascending=True)
                target_idx = min(strikes_depth - 1, len(itm_sorted) - 1)
                selected_contract = itm_sorted.iloc[target_idx]
                selection_mode_used = 'itm_default'
            else:
                selected_contract = atm_chain.iloc[0]
                selection_mode_used = 'atm_fallback'
    
    # Define moneyness states for the contract
    is_atm = selected_contract['strike_price'] == spy_price
    is_itm = ((option_type == 'call' and selected_contract['strike_price'] < spy_price) or
             (option_type == 'put' and selected_contract['strike_price'] > spy_price))
    is_otm = not (is_atm or is_itm)
    
    # Create a dictionary with only the necessary contract details
    contract_details = {
        'ticker': selected_contract['ticker'],
        'option_type': option_type,
        'strike_price': selected_contract['strike_price'],
        'expiration_date': selected_contract.get('expiration_date', None),
        'abs_diff': selected_contract['abs_diff'],
        'is_atm': is_atm,
        'is_itm': is_itm,
        'is_otm': is_otm,
        'is_same_day_expiry': selected_contract.get('expiration_date', '') == current_date,
        'selection_mode': option_selection_mode,
        'selection_mode_used': selection_mode_used,
        'strikes_depth': strikes_depth
    }
    
    if params['debug_mode']:
        # Determine actual moneyness status for display
        if is_atm:
            moneyness_status = "ATM"
        elif is_itm:
            moneyness_status = "ITM"
        else:
            moneyness_status = "OTM"
            
        expiry_status = "Same-day expiry" if contract_details['is_same_day_expiry'] else "Future expiry"
        
        print(f"✅ Selected {option_type.upper()} option: {contract_details['ticker']} with strike {contract_details['strike_price']} ({moneyness_status}, {expiry_status})")
        print(f"   Underlying price: {spy_price}, Strike diff: {contract_details['abs_diff']:.4f}")
        print(f"   Entry timestamp: {entry_signal['reclaim_ts']}, Expiration date: {contract_details['expiration_date']}")
    
    if DEBUG_MODE:
#        print(f"DEBUG: Stretch direction: {entry_signal['stretch_label']}")
#        print(f"DEBUG: Selected option type: {option_type}")
#        print(f"DEBUG: SPY price: {spy_price}, Strike: {contract_details['strike_price']}")
#        print(f"DEBUG: Is ATM: {contract_details['is_atm']}, Is ITM: {contract_details['is_itm']}")
        print(f"   Selection mode: {option_selection_mode.upper()}, Actual mode used: {selection_mode_used}")
        print(f"   Strike depth: {strikes_depth} strikes from ATM")
    
    return contract_details

# Initialize a counter for total entry intent signals
total_entry_intent_signals = 0
days_processed = 0
all_contracts = []  # Master list to store all contract data

# === STEP 8: Backtest loop ===
for date_obj in business_days:
    date = date_obj.strftime("%Y-%m-%d")
    print(f"\n📅 Processing {date}...")

    try:
        # === STEP 5a: Load or pull SPY OHLCV ===
        spy_path = os.path.join(SPY_DIR, f"{ticker}_{date}.pkl")
        if os.path.exists(spy_path):
            df_rth_filled = pd.read_pickle(spy_path)
            print("📂 SPY data loaded from cache.")
            if len(df_rth_filled) < PARAMS['min_spy_data_rows']:
                print(f"⚠️ SPY data for {date} is unusually short with only {len(df_rth_filled)} rows. This may indicate incomplete data.")
        else:
            base_url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/second/{date}/{date}"
            headers = {"Authorization": f"Bearer {API_KEY}"}

            all_results = []
            cursor = None
            while True:
                url = f"{base_url}?adjusted=true&limit=50000"
                if cursor:
                    url += f"&cursor={cursor}"
                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    raise Exception(f"SPY price request failed: {response.status_code}")

                json_data = response.json()
                results = json_data.get("results", [])
                all_results.extend(results)

                if "next_url" in json_data:
                    cursor = json_data["next_url"].split("cursor=")[-1]
                else:
                    break

            if not all_results:
                print(f"⚠️ No SPY data for {date} — skipping.")
                continue

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
                raise ValueError("❌ NaNs detected in vwap_running — check data or ffill logic")

            if not df_rth_filled["vwap_running"].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
                raise ValueError("❌ Non-finite values (inf/-inf) in vwap_running")

            # Check for NaNs and non-finite values in critical columns
            critical_columns = ["open", "high", "low", "close", "volume", "vw"]
            for column in critical_columns:
                if df_rth_filled[column].isna().any():
                    raise ValueError(f"❌ NaNs detected in {column} — check data integrity")
                if not df_rth_filled[column].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
                    raise ValueError(f"❌ Non-finite values (inf/-inf) in {column} — check data integrity")

            if len(df_rth_filled) < PARAMS['min_spy_data_rows']:
                print(f"⚠️ SPY data for {date} is unusually short with only {len(df_rth_filled)} rows after pulling from API. This may indicate incomplete data.")

            df_rth_filled.to_pickle(spy_path)
            print("💾 SPY data pulled and cached.")

        # === STEP 5b: Load or pull option chain ===
        chain_path = os.path.join(CHAIN_DIR, f"{ticker}_chain_{date}.pkl")
        if os.path.exists(chain_path):
            df_chain = pd.read_pickle(chain_path)
            print("📂 Option chain loaded from cache.")
            if len(df_chain) < PARAMS['min_option_chain_rows']:
                print(f"⚠️ Option chain data for {date} is unusually short with only {len(df_chain)} rows. This may indicate incomplete data.")
        else:
            def fetch_chain(contract_type):
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
                    f"&apiKey={API_KEY}"
                )
                resp = requests.get(url)
                if resp.status_code != 200:
                    raise Exception(f"{contract_type.upper()} request failed: {resp.status_code}")
                df = pd.DataFrame(resp.json().get("results", []))
                df["option_type"] = contract_type
                return df

            df_calls = fetch_chain("call")
            df_puts = fetch_chain("put")
            df_chain = pd.concat([df_calls, df_puts], ignore_index=True)
            df_chain["ticker_clean"] = df_chain["ticker"].str.replace("O:", "", regex=False)

            # Check for unusually short data before caching
            if len(df_chain) < PARAMS['min_option_chain_rows']:
                print(f"⚠️ Option chain data for {date} is unusually short with only {len(df_chain)} rows after pulling from API. This may indicate incomplete data.")

            df_chain.to_pickle(chain_path)
            print("💾 Option chain pulled and cached.")

        if df_chain.empty:
            print(f"⚠️ No option chain data for {date} — skipping.")
            continue

        # === Insert strategy logic here ===
        stretch_signals = detect_stretch_signal(df_rth_filled, PARAMS)

        # Log if no stretch signals are detected
        if stretch_signals.empty:
            print(f"⚠️ No stretch signals detected for {date} — skipping.")
            continue

        stretch_signals = detect_partial_reclaims(df_rth_filled, stretch_signals, PARAMS)
        
        # Ensure 'entry_intent' column exists
        if 'entry_intent' not in stretch_signals.columns:
            print(f"⚠️ 'entry_intent' column missing for {date} — skipping.")
            continue

        # Filter for valid entry signals
        valid_entries = stretch_signals[stretch_signals['entry_intent'] == True]
        
        # Initialize container for daily contracts
        daily_contracts = []
        
        # MODIFIED: Process ALL valid entries instead of just the first one
        if not valid_entries.empty:
            print(f"✅ Found {len(valid_entries)} valid entry signals for {date}")
            
            # Process each valid entry signal
            for idx, entry_signal in valid_entries.iterrows():
                entry_time = entry_signal['reclaim_ts']
                entry_price = entry_signal['reclaim_price']
                
                # Get SPY price at entry
                spy_price_at_entry = df_rth_filled[df_rth_filled['ts_raw'] == entry_time]['close'].iloc[0]
                
                # Select appropriate option contract
                selected_contract = select_option_contract(entry_signal, df_chain, spy_price_at_entry, PARAMS)
                
                if selected_contract:
                    # Now we need to load the option price data for the selected contract
                    option_ticker = selected_contract['ticker']
                    
                    # === STEP 5d: Load or pull option price data ===
                    option_path = os.path.join(OPTION_DIR, f"{date}_{option_ticker.replace(':', '')}.pkl")
                    if os.path.exists(option_path):
                        df_option_rth = pd.read_pickle(option_path)
                        print(f"📂 Option price data for {option_ticker} loaded from cache.")
                        if len(df_option_rth) < PARAMS['min_option_price_rows']:
                            print(f"⚠️ Option price data for {option_ticker} on {date} is unusually short with only {len(df_option_rth)} rows. This may indicate incomplete data.")
                    else:
                        option_url = (
                            f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/second/"
                            f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
                        )
                        resp = requests.get(option_url)
                        option_results = resp.json().get("results", [])
                        df_option = pd.DataFrame(option_results)

                        if df_option.empty:
                            print(f"⚠️ No option price data for {option_ticker} on {date} — skipping this entry.")
                            continue

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

                        if len(df_option_rth) < PARAMS['min_option_price_rows']:
                            print(f"⚠️ Option price data for {option_ticker} on {date} is unusually short with only {len(df_option_rth)} rows after pulling from API. This may indicate incomplete data.")

                        df_option_rth.to_pickle(option_path)
                        print(f"💾 Option price data for {option_ticker} pulled and cached.")
                    
                    # === STEP 5e: Timestamp alignment check ===
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
                    last_actual_ts = None
                    for idx_opt, row_opt in df_option_aligned.iterrows():
                        if row_opt['is_actual_data']:
                            last_actual_ts = row_opt['ts_raw']
                            # No need to set 0.0 since already initialized
                        elif last_actual_ts is not None:
                            seconds_diff = (row_opt['ts_raw'] - last_actual_ts).total_seconds()
                            df_option_aligned.at[idx_opt, 'seconds_since_update'] = seconds_diff
                        else:
                            # Edge case: No actual data points before this timestamp
                            df_option_aligned.at[idx_opt, 'seconds_since_update'] = float('inf')  # Mark as infinitely stale
                    
                    # Define a threshold for allowable mismatches
                    mismatch_threshold = PARAMS['timestamp_mismatch_threshold']
                    
                    # Check for timestamp mismatches
                    mismatch_count = (~df_option_aligned["ts_raw"].eq(df_rth_filled["ts_raw"])).sum()
                    print(f"🧪 Signal #{idx+1}: Timestamp mismatches for {option_ticker}: {mismatch_count}")
                    
                    # Hash-based timestamp verification as additional sanity check
                    if DEBUG_MODE:
                        print(f"⏱️ SPY rows: {len(df_rth_filled)}")
                        print(f"⏱️ OPT rows: {len(df_option_aligned)}")

                        def hash_timestamps(df):
                            return hashlib.md5("".join(df["ts_raw"].astype(str)).encode()).hexdigest()

                        spy_hash = hash_timestamps(df_rth_filled)
                        opt_hash = hash_timestamps(df_option_aligned)
                        hash_match = spy_hash == opt_hash
                        
                        print(f"🔐 SPY hash:  {spy_hash}")
                        print(f"🔐 OPT hash:  {opt_hash}")
                        print(f"🔍 Hash match: {hash_match}")
                    
                    # Check if mismatches exceed the threshold
                    if mismatch_count > mismatch_threshold:
                        print(f"⚠️ Timestamp mismatch in {mismatch_count} rows exceeds threshold of {mismatch_threshold} — skipping this entry.")
                        continue
                    
                    # Lookup option price at entry time
                    option_row = df_option_aligned[df_option_aligned['ts_raw'] == entry_time]
                    
                    if option_row.empty:
                        print(f"⚠️ Could not find option price for entry at {entry_time} - skipping this entry")
                        continue
                    
                    # Extract entry price for the option
                    option_entry_price = option_row['vwap'].iloc[0]
                    
                    # Check price staleness at entry
                    staleness_threshold = PARAMS['price_staleness_threshold_seconds']
                    price_staleness = option_row['seconds_since_update'].iloc[0]
                    is_price_stale = price_staleness > staleness_threshold
                    
                    if is_price_stale and PARAMS['report_stale_prices']:
                        print(f"⚠️ Signal #{idx+1}: Using stale option price at entry - {price_staleness:.1f} seconds old")
                    
                    # Store contract with complete entry details
                    contract_with_entry = {
                        **selected_contract,
                        'entry_time': entry_time,
                        'entry_spy_price': spy_price_at_entry,
                        'entry_option_price': option_entry_price,
                        'price_staleness_seconds': price_staleness,
                        'is_price_stale': is_price_stale,
                        'signal_number': idx + 1,  # Add signal sequence number for reference
                        'df_option_aligned': df_option_aligned,  # Save aligned option data for later use
                        'entry_signal': entry_signal.to_dict()
                    }
                    daily_contracts.append(contract_with_entry)
                    
                    # Memory optimization: Clear the large DataFrame after use
                    del df_option_aligned
        
        # Count the number of entry intent signals for the day
        daily_entry_intent_signals = stretch_signals['entry_intent'].sum()
        total_entry_intent_signals += daily_entry_intent_signals
        days_processed += 1

        if DEBUG_MODE:
            print(f"🎯 Entry intent signals (valid reclaims): {daily_entry_intent_signals}")
            # UPDATED: Report successful entries vs attempted entries
            successful_entries = len(daily_contracts)
            print(f"💰 Successful contract entries: {successful_entries}/{daily_entry_intent_signals} ({(successful_entries/daily_entry_intent_signals*100):.1f}% success rate)" if daily_entry_intent_signals > 0 else "💰 No valid entry signals to process")
            
            # UPDATED: Show contract details for all entries instead of just the first one
            if daily_contracts:
                print("\n   Contract details for each entry:")
                for i, contract in enumerate(daily_contracts):
                    print(f"   [{i+1}] {contract['option_type'].upper()} option: {contract['ticker']}")
                    print(f"       Strike: {contract['strike_price']}, Entry price: ${contract['entry_option_price']:.2f}")
                    print(f"       Entry time: {contract['entry_time'].strftime('%H:%M:%S')}")
            
            # Log the daily breakdown of stretch signals
            above_count = len(stretch_signals[stretch_signals['stretch_label'] == 'above'])
            below_count = len(stretch_signals[stretch_signals['stretch_label'] == 'below'])
            total_count = len(stretch_signals)
            print(f"🔍 Detected {total_count} stretch signals on {date} (Above: {above_count}, Below: {below_count}).")

        # Add daily contracts to the master list
        if daily_contracts:
            # For each contract in the daily list, clean up memory and add to master list
            for contract in daily_contracts:
                # Make a clean copy without the large DataFrame to save memory
                clean_contract = contract.copy()
                
                # Remove the full DataFrame before adding to the master list to save memory
                if 'df_option_aligned' in clean_contract:
                    del clean_contract['df_option_aligned']
                    
                # Add to our master list of all contracts
                all_contracts.append(clean_contract)

    except Exception as e:
        print(f"❌ {date} — Error: {str(e)}")
        continue

# Calculate and log the average number of daily entry intent signals
if days_processed > 0:
    average_entry_intent_signals = total_entry_intent_signals / days_processed
    print(f"📊 Average daily entry intent signals over the period: {average_entry_intent_signals:.2f}")
    
    # Calculate average successful entries per day
    if all_contracts:
        average_entries_per_day = len(all_contracts) / days_processed
        success_rate = len(all_contracts) / total_entry_intent_signals * 100 if total_entry_intent_signals > 0 else 0
        print(f"📊 Average successful trades per day: {average_entries_per_day:.2f}")
        print(f"📊 Overall success rate: {success_rate:.1f}% ({len(all_contracts)}/{total_entry_intent_signals} signals)")
else:
    print("⚠️ No days processed, cannot calculate average entry intent signals.")

# Summary of contract selections
if all_contracts:
    print(f"\n📈 Total option contracts selected: {len(all_contracts)}")
    
    # Create a DataFrame for easier analysis
    contracts_df = pd.DataFrame(all_contracts)
    
    # Count by option type
    call_count = len(contracts_df[contracts_df['option_type'] == 'call'])
    put_count = len(contracts_df[contracts_df['option_type'] == 'put'])
    
    print(f"  Call options: {call_count} ({call_count/len(contracts_df)*100:.1f}%)")
    print(f"  Put options: {put_count} ({put_count/len(contracts_df)*100:.1f}%)")
    
    # Count by positioning
    atm_count = len(contracts_df[contracts_df['is_atm'] == True])
    itm_count = len(contracts_df[contracts_df['is_itm'] == True])
    otm_count = len(contracts_df[(contracts_df['is_atm'] == False) & (contracts_df['is_itm'] == False)])
    
    print(f"  ATM contracts: {atm_count} ({atm_count/len(contracts_df)*100:.1f}%)")
    print(f"  ITM contracts: {itm_count} ({itm_count/len(contracts_df)*100:.1f}%)")
    print(f"  OTM contracts: {otm_count} ({otm_count/len(contracts_df)*100:.1f}%)")
    
    # Get trading day distribution
    print("\n📆 Trades by Day:")
    date_counts = contracts_df['entry_time'].dt.date.value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"  {date}: {count} trade(s)")
    
    # Average strike distance from price
    avg_diff = contracts_df['abs_diff'].mean()
    print(f"\n  Average distance from ATM: {avg_diff:.4f}")
    
    # Add price staleness statistics if enabled
    if PARAMS['report_stale_prices'] and 'is_price_stale' in contracts_df.columns:
        stale_count = len(contracts_df[contracts_df['is_price_stale'] == True])
        fresh_count = len(contracts_df) - stale_count
        avg_staleness = contracts_df['price_staleness_seconds'].mean()
        max_staleness = contracts_df['price_staleness_seconds'].max()
        
        print("\n🔍 Price Staleness Statistics:")
        print(f"  Fresh price entries: {fresh_count} ({(fresh_count/len(contracts_df))*100:.1f}%)")
        print(f"  Stale price entries: {stale_count} ({(stale_count/len(contracts_df))*100:.1f}%)")
        print(f"  Average staleness: {avg_staleness:.2f} seconds")
        print(f"  Maximum staleness: {max_staleness:.2f} seconds")
    
    # Sample of contracts
    print("\n🔍 Sample of selected option contracts:")
    if PARAMS['report_stale_prices'] and 'is_price_stale' in contracts_df.columns:
        print(contracts_df[['entry_time', 'option_type', 'strike_price', 'entry_spy_price', 
                          'entry_option_price', 'price_staleness_seconds', 'is_price_stale']].head(10))
    else:
        print(contracts_df[['entry_time', 'option_type', 'strike_price', 'entry_spy_price', 'entry_option_price']].head(10))
