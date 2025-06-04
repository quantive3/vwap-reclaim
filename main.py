# === STEP 1: Mount Google Drive for caching ===
from google.colab import drive
drive.mount('/content/drive')

# === STEP 2: Manual API Key Input ===
API_KEY = input("🔑 Enter your Polygon API key: ").strip()

# === STEP 3: Imports and setup ===
import pandas as pd
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
    'option_type': 'call',  # 'call' or 'put'
    
    # Data quality thresholds - for error checking
    'min_spy_data_rows': 1000,  # Minimum acceptable rows for SPY data
    'min_option_chain_rows': 10,  # Minimum acceptable rows for option chain data
    'min_option_price_rows': 100,  # Minimum acceptable rows for option price data
    'timestamp_mismatch_threshold': 0,  # Maximum allowable timestamp mismatches
    
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
    Select the appropriate option contract based on stretch direction and ATM/ITM logic.
    
    Parameters:
    - entry_signal: DataFrame row with entry signal information
    - df_chain: DataFrame with option chain data
    - spy_price: Current SPY price at entry
    - params: Strategy parameters
    
    Returns:
    - selected_contract: Dictionary with selected contract details
    """
    # Determine option type based on stretch direction
    stretch_direction = entry_signal['stretch_label']
    
    # Below VWAP stretch → buy call option
    # Above VWAP stretch → buy put option
    option_type = 'call' if stretch_direction == 'below' else 'put'
    
    # Filter chain to only include the desired option type
    filtered_chain = df_chain[df_chain['option_type'] == option_type].copy()
    
    if filtered_chain.empty:
        if params['debug_mode']:
            print(f"⚠️ No {option_type} options available in the chain")
        return None
    
    # Calculate absolute difference between each strike and current price for ATM selection
    filtered_chain['abs_diff'] = (filtered_chain['strike_price'] - spy_price).abs()
    
    # Sort by absolute difference to find closest to ATM
    atm_chain = filtered_chain.sort_values('abs_diff')
    
    # If there's an exact match (unlikely but possible), use it
    exact_match = atm_chain[atm_chain['strike_price'] == spy_price]
    
    if not exact_match.empty:
        selected_contract = exact_match.iloc[0]
    else:
        # For calls, ITM means strike < price
        # For puts, ITM means strike > price
        if option_type == 'call':
            itm_contracts = atm_chain[atm_chain['strike_price'] < spy_price]
            if not itm_contracts.empty:
                # Get the highest strike that's still ITM (closest to ATM)
                selected_contract = itm_contracts.sort_values('strike_price', ascending=False).iloc[0]
            else:
                # If no ITM contracts, just get the closest ATM
                selected_contract = atm_chain.iloc[0]
        else:  # put
            itm_contracts = atm_chain[atm_chain['strike_price'] > spy_price]
            if not itm_contracts.empty:
                # Get the lowest strike that's still ITM (closest to ATM)
                selected_contract = itm_contracts.sort_values('strike_price', ascending=True).iloc[0]
            else:
                # If no ITM contracts, just get the closest ATM
                selected_contract = atm_chain.iloc[0]
    
    # Create a dictionary with only the necessary contract details
    contract_details = {
        'ticker': selected_contract['ticker'],
        'option_type': option_type,
        'strike_price': selected_contract['strike_price'],
        'expiration_date': selected_contract.get('expiration_date', None),
        'abs_diff': selected_contract['abs_diff'],
        'is_atm': selected_contract['strike_price'] == spy_price,
        'is_itm': (option_type == 'call' and selected_contract['strike_price'] < spy_price) or
                  (option_type == 'put' and selected_contract['strike_price'] > spy_price)
    }
    
    if params['debug_mode']:
        itm_status = "ITM" if contract_details['is_itm'] else ("ATM" if contract_details['is_atm'] else "OTM")
        print(f"✅ Selected {option_type.upper()} option: {contract_details['ticker']} with strike {contract_details['strike_price']} ({itm_status})")
        print(f"   Underlying price: {spy_price}, Strike diff: {contract_details['abs_diff']:.4f}")
    
#    if DEBUG_MODE:
#        print(f"DEBUG: Stretch direction: {entry_signal['stretch_label']}")
#        print(f"DEBUG: Selected option type: {option_type}")
#        print(f"DEBUG: SPY price: {spy_price}, Strike: {contract_details['strike_price']}")
#        print(f"DEBUG: Is ATM: {contract_details['is_atm']}, Is ITM: {contract_details['is_itm']}")
    
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
                    f"&expiration_date={date}"
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
        
        # Process the first valid entry (maximum 1 contract per day)
        if not valid_entries.empty:
            entry_signal = valid_entries.iloc[0]
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
                    print("📂 Option price data loaded from cache.")
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
                        print(f"⚠️ No option price data for {option_ticker} on {date} — skipping.")
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

                    if len(df_option_rth) < PARAMS['min_option_price_rows']:
                        print(f"⚠️ Option price data for {option_ticker} on {date} is unusually short with only {len(df_option_rth)} rows after pulling from API. This may indicate incomplete data.")

                    df_option_rth.to_pickle(option_path)
                    print("💾 Option price data pulled and cached.")
                
                # === STEP 5e: Timestamp alignment check ===
                df_option_aligned = df_option_rth.set_index("timestamp").reindex(df_rth_filled["ts_raw"]).ffill().reset_index()
                df_option_aligned.rename(columns={"index": "ts_raw"}, inplace=True)
                
                # Define a threshold for allowable mismatches
                mismatch_threshold = PARAMS['timestamp_mismatch_threshold']
                
                # Check for timestamp mismatches
                mismatch_count = (~df_option_aligned["ts_raw"].eq(df_rth_filled["ts_raw"])).sum()
                print(f"🧪 Timestamp mismatches: {mismatch_count}")
                
                # Check if mismatches exceed the threshold
                if mismatch_count > mismatch_threshold:
                    print(f"⚠️ Timestamp mismatch in {mismatch_count} rows exceeds threshold of {mismatch_threshold} — skipping.")
                    continue
                
                # Lookup option price at entry time
                option_row = df_option_aligned[df_option_aligned['ts_raw'] == entry_time]
                
                if option_row.empty:
                    print(f"⚠️ Could not find option price for entry at {entry_time} - skipping")
                    continue
                
                # Extract entry price for the option
                option_entry_price = option_row['close'].iloc[0]
                
                # Store contract with complete entry details
                contract_with_entry = {
                    **selected_contract,
                    'entry_time': entry_time,
                    'entry_spy_price': spy_price_at_entry,
                    'entry_option_price': option_entry_price,
                    'df_option_aligned': df_option_aligned,  # Save aligned option data for later use
                    'entry_signal': entry_signal.to_dict()
                }
                daily_contracts.append(contract_with_entry)
        
        # Count the number of entry intent signals for the day
        daily_entry_intent_signals = stretch_signals['entry_intent'].sum()
        total_entry_intent_signals += daily_entry_intent_signals
        days_processed += 1

        if DEBUG_MODE:
            print(f"🎯 Entry intent signals (valid reclaims): {daily_entry_intent_signals}")
            print(f"💰 Selected contracts: {len(daily_contracts)}")
            
            if daily_contracts:
                contract = daily_contracts[0]
                print(f"   Selected {contract['option_type'].upper()} option: {contract['ticker']}")
                print(f"   Strike: {contract['strike_price']}, Entry price: ${contract['entry_option_price']:.2f}")
            
            # Log the daily breakdown of stretch signals
            above_count = len(stretch_signals[stretch_signals['stretch_label'] == 'above'])
            below_count = len(stretch_signals[stretch_signals['stretch_label'] == 'below'])
            total_count = len(stretch_signals)
            print(f"🔍 Detected {total_count} stretch signals on {date} (Above: {above_count}, Below: {below_count}).")

        if daily_contracts:
            contract = daily_contracts[0]
            
            # Remove the full DataFrame before adding to the master list to save memory
            # (we'll only keep the essential trade data for analysis)
            if 'df_option_aligned' in contract:
                del contract['df_option_aligned']
                
            # Add to our master list of all contracts
            all_contracts.append(contract)

    except Exception as e:
        print(f"❌ {date} — Error: {str(e)}")
        continue

# Calculate and log the average number of daily entry intent signals
if days_processed > 0:
    average_entry_intent_signals = total_entry_intent_signals / days_processed
    print(f"📊 Average daily entry intent signals over the period: {average_entry_intent_signals:.2f}")
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
    
    print(f"  Call options: {call_count}")
    print(f"  Put options: {put_count}")
    
    # Count by positioning
    atm_count = len(contracts_df[contracts_df['is_atm'] == True])
    itm_count = len(contracts_df[contracts_df['is_itm'] == True])
    otm_count = len(contracts_df[(contracts_df['is_atm'] == False) & (contracts_df['is_itm'] == False)])
    
    print(f"  ATM contracts: {atm_count}")
    print(f"  ITM contracts: {itm_count}")
    print(f"  OTM contracts: {otm_count}")
    
    # Average strike distance from price
    avg_diff = contracts_df['abs_diff'].mean()
    print(f"  Average distance from ATM: {avg_diff:.4f}")
    
    # Sample of contracts
    print("\n🔍 Sample of selected option contracts:")
    print(contracts_df[['entry_time', 'option_type', 'strike_price', 'entry_spy_price', 'entry_option_price']].head())
