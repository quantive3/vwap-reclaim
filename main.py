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
    'start_date': "2023-01-01",
    'end_date': "2023-01-31",
    
    # Strategy parameters
    'stretch_threshold': 0.003,  # 0.3%
    'reclaim_threshold': 0.002,  # 0.2% - should always be less than stretch threshold
    'cooldown_period_seconds': 60,  # Cooldown period in seconds
    
    # Time windows
    'entry_start_time': time(9, 30),
    'entry_end_time': time(16, 0),
    
    # Exit conditions
    'take_profit_percent': 25,     # Take profit at 25% gain
    'stop_loss_percent': -50,      # Stop loss at 50% loss
    'max_trade_duration_seconds': 300,  # Exit after 300 seconds (5 minutes)
    'end_of_day_exit_time': time(15, 54),  # trade exit cutoff
    'emergency_exit_time': time(15, 55),   # absolute failsafe exit (overrides all other logic)
    
    # Risk management failsafes
    'late_entry_cutoff_time': time(15, 54),  # No new entries after this time
    
    # Latency simulation
    'latency_seconds': 3,    # Seconds delay between signal and execution (0 = disabled)
    
    # Instrument selection
    'ticker': 'SPY',
    'require_same_day_expiry': True,  # Whether to strictly require same-day expiry options
    'strikes_depth': 1,  # Number of strikes from ATM to target (1 = closest, 2 = second closest, etc.). Always use 1 or greater.
    'option_selection_mode': 'itm',  # Options: 'itm', 'otm', or 'atm' - determines whether to select in-the-money, out-of-money, or at-the-money options
    
    # Position sizing
    'contracts_per_trade': 1,  # Number of contracts to trade per signal (for P&L calculations)
    
    # Transaction costs
    'brokerage_fee_per_contract': 0.65,  # Brokerage fee per contract per direction (entry or exit)
    'exchange_fee_per_contract': 0.65,   # Exchange and other fees per contract per direction
    
    # Data quality thresholds - for error checking
    'min_spy_data_rows': 10000,  # Minimum acceptable rows for SPY data
    'min_option_chain_rows': 10,  # Minimum acceptable rows for option chain data
    'min_option_price_rows': 10000,  # Minimum acceptable rows for option price data
    'timestamp_mismatch_threshold': 0,  # Maximum allowable timestamp mismatches
    'price_staleness_threshold_seconds': 10,  # Maximum allowable staleness in seconds for option prices
    'report_stale_prices': True,  # Enable/disable reporting of stale prices
    
    # Slippage settings
    'slippage_percent': 0.01,  # 1% slippage
    
    # Debug settings
    'debug_mode': True,  # Enable/disable debug outputs
}

# === STEP 6b: Initialize Issue Tracker ===
# Initialize issue tracker for summary reporting
issue_tracker = {
    "days": {
        "attempted": 0,
        "processed": 0,
        "skipped_errors": 0,
        "skipped_warnings": 0,
        "dates_with_issues": set()  # Track specific dates with issues
    },
    "data_integrity": {
        "hash_mismatches": 0,
        "timestamp_mismatches": 0,
        "days_with_mismatches": set()  # Track days with timestamp mismatches
    },
    "warnings": {
        f"no_{PARAMS['ticker']}_data": 0,  # Dynamically use the ticker name
        "price_staleness": 0,
        "short_data_warnings": 0,
        "timestamp_mismatches_below_threshold": 0,
        "shares_per_contract_missing": 0,  # Warning when default shares_per_contract is used
        "non_standard_contract_size": 0,   # Warning when shares_per_contract is not 100
        "vwap_fallback_to_close": 0,       # Added for tracking fallbacks from VWAP to close price
        "emergency_exit_triggered": 0,     # Added for tracking emergency exit activations
        "other": 0,
        "details": []  # Store details for uncommon warnings
    },
    "errors": {
        "missing_option_price_data": 0,
        "api_connection_failures": 0,
        "missing_exit_data": 0,     # Added for tracking missing option data for exits
        "no_future_price_data": 0,  # Added for tracking missing future price data
        "forced_exit_end_of_data": 0,  # Added for tracking forced exits
        "exit_evaluation_error": 0,  # Added for tracking errors during exit evaluation
        "forced_exit_error": 0,     # Added for tracking errors during forced exit
        "latency_entry_failures": 0, # Added for tracking entry failures due to latency
        "latency_exit_failures": 0,  # Added for tracking exit failures due to latency
        "other": 0,
        "details": []  # Store details for uncommon errors
    },
    "opportunities": {
        "total_stretch_signals": 0,
        "valid_entry_opportunities": 0,
        "failed_entries_data_issues": 0,
        "total_options_contracts": 0
    },
    "risk_management": {
        "emergency_exits": 0,       # Total number of emergency exits triggered
        "emergency_exit_dates": set(),  # Dates when emergency exits were triggered
        "late_entries_blocked": 0,  # Total number of late entry attempts blocked
        "late_entry_dates": set()   # Dates when late entries were blocked
    }
}

# Function to track issues
def track_issue(category, subcategory, message, level="warning", date=None):
    """
    Track an issue in the issue tracker.
    
    Parameters:
    - category: Main category (e.g., "warnings", "errors")
    - subcategory: Specific type of issue (e.g., "price_staleness")
    - message: Description of the issue
    - level: Severity level ("warning" or "error")
    - date: Date when the issue occurred (for day-specific tracking)
    """
    # Always track the date with issues if provided
    if date and category in ["warnings", "errors", "data_integrity"]:
        issue_tracker["days"]["dates_with_issues"].add(date)
    
    # Track based on category and subcategory
    if category == "warnings":
        if subcategory in issue_tracker["warnings"]:
            issue_tracker["warnings"][subcategory] += 1
        else:
            issue_tracker["warnings"]["other"] += 1
            issue_tracker["warnings"]["details"].append(f"{date}: {message}")
            
    elif category == "errors":
        if subcategory in issue_tracker["errors"]:
            issue_tracker["errors"][subcategory] += 1
        else:
            issue_tracker["errors"]["other"] += 1
            issue_tracker["errors"]["details"].append(f"{date}: {message}")
            
    elif category == "data_integrity":
        if subcategory in issue_tracker["data_integrity"]:
            issue_tracker["data_integrity"][subcategory] += 1
            # Track days with timestamp mismatches
            if subcategory == "timestamp_mismatches" and date:
                issue_tracker["data_integrity"]["days_with_mismatches"].add(date)
                
    elif category == "opportunities":
        if subcategory in issue_tracker["opportunities"]:
            issue_tracker["opportunities"][subcategory] += 1

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

    if DEBUG_MODE:
        print("\n🧹 Post-filter signal integrity check:")
        print(f"  NaT timestamps: {signals['ts_raw'].isna().sum()}")
        print(f"  Time ordered: {signals['ts_raw'].is_monotonic_increasing}")

        # Log the first 5 stretch labels for "above" and "below"
    if DEBUG_MODE:
        print("\nFirst 5 'Above' Stretch Signals:")
        print(signals[signals['stretch_label'] == 'above'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
        print("\nFirst 5 'Below' Stretch Signals:")
        print(signals[signals['stretch_label'] == 'below'][['timestamp', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
    
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
    if DEBUG_MODE:
        signals_after_cooldown = len(processed_signals_df)
        signals_dropped_by_cooldown = len(signals) - signals_after_cooldown
        if signals_dropped_by_cooldown > 0:
            print(f"ℹ️ Cooldown filtering: {signals_dropped_by_cooldown} signals were dropped due to cooldown period")
            print(f"ℹ️ Signals before cooldown: {len(signals)}, after: {signals_after_cooldown}")

    if DEBUG_MODE:
        # Log the first 5 processed 'above' stretch signals
        print("\nFirst 5 Processed 'Above' Stretch Signals:")
        print(processed_signals_df[processed_signals_df['stretch_label'] == 'above'][['ts_raw', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())
        
        # Log the first 5 processed 'below' stretch signals
        print("\nFirst 5 Processed 'Below' Stretch Signals:")
        print(processed_signals_df[processed_signals_df['stretch_label'] == 'below'][['ts_raw', 'close', 'vwap_running', 'percentage_stretch', 'stretch_label']].head())

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

# === STEP 7b-2: Apply Latency to Timestamps ===
def apply_latency(original_timestamp, df_data, latency_seconds):
    """
    Apply execution latency to a timestamp and return the delayed timestamp and price data.
    
    Parameters:
    - original_timestamp: The original signal timestamp
    - df_data: DataFrame containing price data with 'ts_raw' timestamps column
    - latency_seconds: Number of seconds to delay execution
    
    Returns:
    - Dictionary containing:
        - delayed_timestamp: Timestamp after adding latency
        - delayed_row: DataFrame row at the delayed timestamp (or None if not found)
        - delayed_price: Price at the delayed timestamp (or None if not found)
        - is_valid: Boolean indicating if valid price data was found at delayed timestamp
    """
    # Calculate the delayed timestamp
    delayed_timestamp = original_timestamp + pd.Timedelta(seconds=latency_seconds)
    
    # Find the row at the delayed timestamp
    delayed_rows = df_data[df_data['ts_raw'] == delayed_timestamp]
    
    # Initialize return values
    result = {
        'original_timestamp': original_timestamp,
        'delayed_timestamp': delayed_timestamp,
        'delayed_row': None,
        'delayed_price': None,
        'is_valid': False
    }
    
    # If we found a row at the delayed timestamp
    if not delayed_rows.empty:
        delayed_row = delayed_rows.iloc[0]
        result['delayed_row'] = delayed_row
        
        # Get price (prefer VWAP if available, fallback to close)
        if 'vwap' in delayed_row and pd.notna(delayed_row['vwap']):
            result['delayed_price'] = delayed_row['vwap']
            result['is_valid'] = True
            result['used_fallback'] = False
        elif pd.notna(delayed_row['close']):
            result['delayed_price'] = delayed_row['close']
            result['is_valid'] = True
            result['used_fallback'] = True
            # Note: we don't track the issue here because this is a utility function
            # The calling code will handle tracking based on the 'used_fallback' flag
    
    return result

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
        'strikes_depth': strikes_depth,
        'shares_per_contract': selected_contract.get('shares_per_contract', 100)  # Default to 100 if not available
    }
    
    # Add warnings for shares_per_contract
    # Warning for missing shares_per_contract
    if 'shares_per_contract' not in selected_contract:
        missing_shares_msg = f"Missing shares_per_contract for {contract_details['ticker']} - using default of 100"
        if params['debug_mode']:
            print(f"⚠️ {missing_shares_msg}")
        track_issue("warnings", "shares_per_contract_missing", missing_shares_msg, date=current_date)
    
    # Warning for non-standard contract size
    if contract_details['shares_per_contract'] != 100:
        non_standard_msg = f"Non-standard contract size detected: {contract_details['shares_per_contract']} shares for {contract_details['ticker']}"
        if params['debug_mode']:
            print(f"⚠️ {non_standard_msg}")
        track_issue("warnings", "non_standard_contract_size", non_standard_msg, date=current_date)
    
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
        print(f"DEBUG: Stretch direction: {entry_signal['stretch_label']}")
        print(f"DEBUG: Selected option type: {option_type}")
        print(f"DEBUG: SPY price: {spy_price}, Strike: {contract_details['strike_price']}")
        print(f"DEBUG: Is ATM: {contract_details['is_atm']}, Is ITM: {contract_details['is_itm']}")
        print(f"   Selection mode: {option_selection_mode.upper()}, Actual mode used: {selection_mode_used}")
        print(f"   Strike depth: {strikes_depth} strikes from ATM")
    
    return contract_details

# === STEP 7d: Exit Logic ===
def evaluate_exit_conditions(contract, current_time, current_price, params):
    """
    Evaluate if exit conditions are met for a given contract.
    
    Parameters:
    - contract: The contract data dictionary
    - current_time: Current timestamp being evaluated
    - current_price: Current option price
    - params: Strategy parameters
    
    Returns:
    - (should_exit, exit_reason) tuple
    """
    entry_time = contract['entry_time']
    entry_price = contract['entry_option_price']
    
    # Calculate P&L percentage
    pnl_percent = (current_price - entry_price) / entry_price * 100
    
    # Calculate time elapsed
    elapsed_seconds = (current_time - entry_time).total_seconds()
    
    # Check end of day cutoff time
    end_of_day_exit_time = params.get('end_of_day_exit_time', time(15, 55))
    if current_time.time() >= end_of_day_exit_time:
        return True, "end_of_day"
    
    # Check take profit
    take_profit_level = params.get('take_profit_percent', 25)
    if pnl_percent >= take_profit_level:
        return True, "take_profit"
        
    # Check stop loss
    stop_loss_level = params.get('stop_loss_percent', -50)
    if pnl_percent <= stop_loss_level:
        return True, "stop_loss"
        
    # Check time-based exit
    max_duration = params.get('max_trade_duration_seconds', 300)
    if elapsed_seconds >= max_duration:
        return True, "time_exit"
        
    return False, None

# === STEP 7e: Emergency Failsafe Exit System ===
def check_emergency_exit_time(current_time, params):
    """
    Check if the current time has reached the emergency failsafe exit time.
    This function provides the ultimate protection against overnight positions.
    
    Parameters:
    - current_time: Current timestamp to evaluate
    - params: Strategy parameters
    
    Returns:
    - Boolean indicating if emergency exit should be triggered
    """
    # Get the emergency failsafe exit time (default 3:58 PM - even later than regular EOD exit)
    # This should be later than your end_of_day_exit_time but before market close
    emergency_exit_time = params.get('emergency_exit_time', time(15, 58))
    
    # Extract time component if current_time is a datetime
    if hasattr(current_time, 'time'):
        current_time = current_time.time()
    
    # Return True if emergency exit should be triggered
    return current_time >= emergency_exit_time

def process_emergency_exit(contract, current_time, df_option, params):
    """
    Process an emergency failsafe exit for a contract.
    This will force close a position regardless of other exit conditions.
    
    Parameters:
    - contract: The contract data dictionary
    - current_time: Current timestamp
    - df_option: DataFrame with option price data
    - params: Strategy parameters
    
    Returns:
    - Updated contract with emergency exit details
    """
    # Find the current price data
    current_rows = df_option[df_option['ts_raw'] == current_time]
    
    if current_rows.empty:
        # If we can't find the exact timestamp, use the last available price
        # This ensures we still exit even if there's a data issue
        exit_row = df_option.iloc[-1]
        exit_time = exit_row['ts_raw']
        print(f"⚠️ EMERGENCY EXIT: Using last available price for {contract['ticker']} at {exit_time}")
    else:
        exit_row = current_rows.iloc[0]
        exit_time = current_time
    
    # Get the exit price (VWAP preferred, fall back to close)
    if 'vwap' in exit_row and pd.notna(exit_row['vwap']):
        exit_price = exit_row['vwap']
    else:
        exit_price = exit_row['close']
        # Log the fallback
        fallback_msg = f"EMERGENCY EXIT: Falling back to close price for {contract['ticker']} at {exit_time}"
        print(f"⚠️ {fallback_msg}")
        entry_date = contract['entry_time'].strftime("%Y-%m-%d")
        track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
    
    # Calculate P&L
    entry_price = contract['entry_option_price']
    pnl_percent = (exit_price - entry_price) / entry_price * 100
    
    # Calculate duration
    entry_time = contract['entry_time']
    trade_duration = (exit_time - entry_time).total_seconds()
    
    # Update contract with exit details
    contract['exit_time'] = exit_time
    contract['exit_price'] = exit_price
    contract['exit_reason'] = "emergency_exit"
    contract['pnl_percent'] = pnl_percent
    contract['trade_duration_seconds'] = trade_duration
    contract['is_closed'] = True
    
    # Add emergency exit flag for tracking
    contract['emergency_exit_triggered'] = True
    
    # Log the emergency exit
    print(f"🚨 EMERGENCY EXIT TRIGGERED for {contract['ticker']} at {exit_time.strftime('%H:%M:%S')}")
    print(f"   P&L: {pnl_percent:.2f}%, Duration: {trade_duration:.0f}s")
    
    # Track this issue in warnings
    entry_date = entry_time.strftime("%Y-%m-%d")
    track_issue("warnings", "emergency_exit_triggered", 
               f"Emergency exit for {contract['ticker']} entered at {entry_time}", 
               level="warning", date=entry_date)
    
    # Also track in risk_management stats
    issue_tracker["risk_management"]["emergency_exits"] += 1
    issue_tracker["risk_management"]["emergency_exit_dates"].add(entry_date)
    
    return contract

# === STEP 7f: Late Entry Blocker Failsafe ===
def check_late_entry_cutoff(entry_time, params):
    """
    Check if the entry time is past the late entry cutoff time.
    This function prevents new positions from being opened too late in the trading day.
    
    Parameters:
    - entry_time: Timestamp of the potential entry signal
    - params: Strategy parameters
    
    Returns:
    - Tuple of (is_blocked, message):
      - is_blocked: Boolean indicating if entry should be blocked
      - message: String with blocking reason (if blocked) or None
    """
    # Get the late entry cutoff time (default 3:30 PM)
    cutoff_time = params.get('late_entry_cutoff_time', time(15, 30))
    
    # Extract time component if entry_time is a datetime
    entry_time_only = entry_time.time() if hasattr(entry_time, 'time') else entry_time
    
    # Check if entry time is past cutoff
    if entry_time_only >= cutoff_time:
        cutoff_str = cutoff_time.strftime('%H:%M:%S')
        entry_str = entry_time_only.strftime('%H:%M:%S')
        message = f"ENTRY BLOCKED: Time {entry_str} is past the late entry cutoff ({cutoff_str})"
        return True, message
    
    return False, None

def process_exits_for_contract(contract, params):
    """
    Process exit conditions for a single contract.
    
    Parameters:
    - contract: Contract data dictionary including df_option_aligned
    - params: Strategy parameters
    
    Returns:
    - Updated contract with exit details
    """
    if 'df_option_aligned' not in contract:
        # Always show critical warning and track the issue
        print(f"⚠️ No option data available for exit processing for {contract['ticker']}")
        
        # Get entry date for tracking if available
        entry_date = contract['entry_time'].strftime("%Y-%m-%d") if 'entry_time' in contract else None
        
        # Track this issue
        track_issue("errors", "missing_exit_data", 
                   f"No option data for exit processing: {contract['ticker']}", 
                   level="error", date=entry_date)
        
        return contract
    
    # Get option data aligned with timestamps
    df_option = contract['df_option_aligned']
    entry_time = contract['entry_time']
    
    # Initialize exit details if not already there
    contract['exit_time'] = None
    contract['exit_price'] = None
    contract['exit_reason'] = None
    contract['pnl_percent'] = None
    contract['trade_duration_seconds'] = None
    contract['is_closed'] = False
    
    # Initialize exit staleness fields
    contract['exit_price_staleness_seconds'] = None
    contract['is_exit_price_stale'] = None
    
    # Get future prices after entry
    future_prices = df_option[df_option['ts_raw'] > entry_time].copy()
    
    if future_prices.empty:
        # Always show critical warning and track the issue
        print(f"⚠️ No future price data available after entry at {entry_time} for {contract['ticker']}")
        
        # Get entry date for tracking
        entry_date = entry_time.strftime("%Y-%m-%d")
        
        # Track this issue
        track_issue("errors", "no_future_price_data", 
                   f"No future price data after entry: {contract['ticker']} at {entry_time}", 
                   level="error", date=entry_date)
        
        contract['exit_reason'] = "no_future_data"
        return contract
    
    # Check each future timestamp for exit conditions
    for idx, row in future_prices.iterrows():
        current_time = row['ts_raw']
        # Use VWAP for consistent pricing with entry
        current_price = row['vwap'] if 'vwap' in row and pd.notna(row['vwap']) else row['close']
        
        # Skip if price is NaN
        if pd.isna(current_price):
            continue
            
        # Check if we had to use close price instead of VWAP
        if 'vwap' not in row or pd.isna(row['vwap']):
            # Log the fallback to close price
            fallback_msg = f"Falling back to close price for exit check at {current_time} - VWAP not available"
            if params['debug_mode']:
                print(f"⚠️ {fallback_msg}")
            entry_date = contract['entry_time'].strftime("%Y-%m-%d")
            track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
        
        try:
            # FAILSAFE: Check for emergency exit time first, overriding all other conditions
            if check_emergency_exit_time(current_time, params):
                return process_emergency_exit(contract, current_time, future_prices, params)
            
            # Check for price staleness at this potential exit point
            staleness_threshold = params['price_staleness_threshold_seconds']
            price_staleness = row['seconds_since_update'] if 'seconds_since_update' in row else 0.0
            is_price_stale = price_staleness > staleness_threshold
            
            # Evaluate exit conditions
            should_exit, reason = evaluate_exit_conditions(
                contract, current_time, current_price, params
            )
            
            if should_exit:
                # Store the original exit signal time (before latency)
                original_exit_time = current_time
                exit_time = current_time
                exit_price = current_price
                exit_staleness = price_staleness
                exit_is_stale = is_price_stale
                
                # Apply latency to exit if configured
                latency_seconds = params.get('latency_seconds', 0)
                if latency_seconds > 0:
                    # Apply exit latency
                    latency_result = apply_latency(original_exit_time, future_prices, latency_seconds)
                    
                    if latency_result['is_valid']:
                        # Use delayed time and price
                        exit_time = latency_result['delayed_timestamp']
                        exit_price = latency_result['delayed_price']
                        
                        # Update staleness for the delayed price
                        if 'seconds_since_update' in latency_result['delayed_row']:
                            exit_staleness = latency_result['delayed_row']['seconds_since_update']
                            exit_is_stale = exit_staleness > staleness_threshold
                        
                        # Check if we had to use close price instead of VWAP
                        if latency_result.get('used_fallback', False):
                            # Log the fallback to close price
                            fallback_msg = f"Falling back to close price for latency-adjusted exit at {exit_time} - VWAP not available"
                            if params['debug_mode']:
                                print(f"⚠️ {fallback_msg}")
                            entry_date = contract['entry_time'].strftime("%Y-%m-%d")
                            track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
                            
                        if params['debug_mode']:
                            print(f"🕒 Exit latency applied: {latency_seconds}s")
                            print(f"   Original signal: {original_exit_time.strftime('%H:%M:%S')}")
                            print(f"   Execution time: {exit_time.strftime('%H:%M:%S')}")
                            print(f"   Price difference: ${exit_price - current_price:.4f}")
                    else:
                        # If we can't find data at the delayed timestamp, continue looking
                        # This is different from entry - we want to keep trying to exit
                        exit_latency_msg = f"No price data at exit latency of {latency_seconds}s from {original_exit_time} - continuing to next check"
                        if params['debug_mode']:
                            print(f"⚠️ {exit_latency_msg}")
                        
                        # Track this issue but continue processing
                        entry_date = contract['entry_time'].strftime("%Y-%m-%d")
                        track_issue("errors", "latency_exit_failures", exit_latency_msg, level="error", date=entry_date)
                        continue
                
                contract['exit_time'] = exit_time
                contract['original_exit_time'] = original_exit_time
                contract['latency_seconds'] = params.get('latency_seconds', 0)
                contract['exit_price'] = exit_price
                contract['exit_reason'] = reason
                contract['pnl_percent'] = (exit_price - contract['entry_option_price']) / contract['entry_option_price'] * 100
                contract['trade_duration_seconds'] = (exit_time - contract['entry_time']).total_seconds()
                contract['original_trade_duration_seconds'] = (original_exit_time - contract['original_signal_time']).total_seconds() if 'original_signal_time' in contract else None
                contract['is_closed'] = True
                
                # Store exit price staleness info
                contract['exit_price_staleness_seconds'] = exit_staleness
                contract['is_exit_price_stale'] = exit_is_stale
                
                # Log staleness warning if needed
                if exit_is_stale and params['report_stale_prices']:
                    entry_date = contract['entry_time'].strftime("%Y-%m-%d")
                    staleness_msg = f"Using stale option price at exit for {contract['ticker']} - {exit_staleness:.1f} seconds old"
                    print(f"⚠️ {staleness_msg}")
                    track_issue("warnings", "price_staleness", staleness_msg, date=entry_date)
                
                if params['debug_mode']:
                    pnl_str = f"{contract['pnl_percent']:.2f}%" if contract['pnl_percent'] is not None else "N/A"
                    duration_str = f"{contract['trade_duration_seconds']:.0f}s" if contract['trade_duration_seconds'] is not None else "N/A"
                    print(f"🚪 Exit: {contract['ticker']} at {exit_time.strftime('%H:%M:%S')} - Reason: {reason}")
                    print(f"   Entry: ${contract['entry_option_price']:.2f}, Exit: ${exit_price:.2f}, P&L: {pnl_str}, Duration: {duration_str}")
                    if exit_is_stale:
                        print(f"   ⚠️ Exit price is stale: {exit_staleness:.1f} seconds old")
                
                break
        except Exception as e:
            # Log the error and track it
            error_msg = f"Error evaluating exit for {contract['ticker']}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Get entry date for tracking
            entry_date = entry_time.strftime("%Y-%m-%d")
            
            # Track the error
            track_issue("errors", "exit_evaluation_error", 
                      error_msg, level="error", date=entry_date)
            
            # Mark this trade with an error, but still allow processing to continue
            contract['exit_reason'] = "exit_evaluation_error"
            continue
    
    # Force exit for any trades not closed (data missing or other issue)
    if not contract['is_closed']:
        try:
            # Get the last available row for forced exit
            last_row = future_prices.iloc[-1]
            original_exit_time = last_row['ts_raw']
            exit_time = original_exit_time
            
            # Get price with fallback from VWAP to close
            if 'vwap' in last_row and pd.notna(last_row['vwap']):
                exit_price = last_row['vwap']
            else:
                # Log the fallback to close price
                fallback_msg = f"Falling back to close price for forced exit at {original_exit_time} - VWAP not available"
                if params['debug_mode']:
                    print(f"⚠️ {fallback_msg}")
                entry_date = contract['entry_time'].strftime("%Y-%m-%d")
                track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
                exit_price = last_row['close']
            
            # Check staleness for forced exit
            staleness_threshold = params['price_staleness_threshold_seconds']
            exit_staleness = last_row['seconds_since_update'] if 'seconds_since_update' in last_row else 0.0
            exit_is_stale = exit_staleness > staleness_threshold
            
            # Forced exits, we don't apply latency since we're already at the end of data
            # But we still track both original and execution times for consistency
            
            # Store both original and execution times (which are the same for forced exits)
            contract['exit_time'] = exit_time
            contract['original_exit_time'] = original_exit_time
            # Latency value is already stored during entry
            contract['exit_price'] = exit_price
            contract['exit_reason'] = "end_of_data"
            if pd.notna(exit_price):
                contract['pnl_percent'] = (exit_price - contract['entry_option_price']) / contract['entry_option_price'] * 100
            contract['trade_duration_seconds'] = (exit_time - contract['entry_time']).total_seconds()
            contract['original_trade_duration_seconds'] = (original_exit_time - contract['original_signal_time']).total_seconds() if 'original_signal_time' in contract else None
            contract['is_closed'] = True
            
            # Store exit price staleness info
            contract['exit_price_staleness_seconds'] = exit_staleness
            contract['is_exit_price_stale'] = exit_is_stale
            
            # Log staleness warning if needed
            if exit_is_stale and params['report_stale_prices']:
                entry_date = contract['entry_time'].strftime("%Y-%m-%d")
                staleness_msg = f"Using stale option price at forced exit for {contract['ticker']} - {exit_staleness:.1f} seconds old"
                print(f"⚠️ {staleness_msg}")
                track_issue("warnings", "price_staleness", staleness_msg, date=entry_date)
            
            # Always show critical warning and track the issue
            print(f"❌ Forced exit at end of available data: {contract['ticker']}")
            
            # Get entry date for tracking
            entry_date = entry_time.strftime("%Y-%m-%d")
            
            # Track this issue as an error, not a warning
            track_issue("errors", "forced_exit_end_of_data", 
                      f"Forced exit at end of data: {contract['ticker']} entered at {entry_time}", 
                      level="error", date=entry_date)
        except Exception as e:
            # Log the error and track it
            error_msg = f"Error during forced exit for {contract['ticker']}: {str(e)}"
            print(f"❌ {error_msg}")
            
            # Get entry date for tracking
            entry_date = entry_time.strftime("%Y-%m-%d")
            
            # Track the error
            track_issue("errors", "forced_exit_error", 
                      error_msg, level="error", date=entry_date)
            
            # Mark this trade with an error
            contract['exit_reason'] = "forced_exit_error"
            contract['is_closed'] = True  # Mark as closed so we don't keep trying to process it
    
    return contract

# Initialize a counter for total entry intent signals
total_entry_intent_signals = 0
days_processed = 0
all_contracts = []  # Master list to store all contract data

# === STEP 8: Backtest loop ===
for date_obj in business_days:
    date = date_obj.strftime("%Y-%m-%d")
    if DEBUG_MODE:
        print(f"\n📅 Processing {date}...")
    
    # Track day attempted
    issue_tracker["days"]["attempted"] += 1

    try:
        # === STEP 5a: Load or pull SPY OHLCV ===
        spy_path = os.path.join(SPY_DIR, f"{ticker}_{date}.pkl")
        if os.path.exists(spy_path):
            df_rth_filled = pd.read_pickle(spy_path)
            if DEBUG_MODE:
                print("📂 SPY data loaded from cache.")
            if len(df_rth_filled) < PARAMS['min_spy_data_rows']:
                short_data_msg = f"SPY data for {date} is unusually short with only {len(df_rth_filled)} rows. This may indicate incomplete data."
                print(f"⚠️ {short_data_msg}")
                track_issue("warnings", "short_data_warnings", short_data_msg, date=date)
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
                    error_msg = f"SPY price request failed: {response.status_code}"
                    track_issue("errors", "api_connection_failures", error_msg, level="error", date=date)
                    raise Exception(error_msg)

                json_data = response.json()
                results = json_data.get("results", [])
                all_results.extend(results)

                if "next_url" in json_data:
                    cursor = json_data["next_url"].split("cursor=")[-1]
                else:
                    break

            if not all_results:
                no_data_msg = f"No SPY data for {date} — skipping."
                print(f"⚠️ {no_data_msg}")
                track_issue("warnings", f"no_{ticker}_data", no_data_msg, date=date)
                issue_tracker["days"]["skipped_warnings"] += 1
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
                error_msg = "NaNs detected in vwap_running — check data or ffill logic"
                track_issue("errors", "other", error_msg, level="error", date=date)
                raise ValueError(f"❌ {error_msg}")

            if not df_rth_filled["vwap_running"].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
                error_msg = "Non-finite values (inf/-inf) in vwap_running"
                track_issue("errors", "other", error_msg, level="error", date=date)
                raise ValueError(f"❌ {error_msg}")

            # Check for NaNs and non-finite values in critical columns
            critical_columns = ["open", "high", "low", "close", "volume", "vw"]
            for column in critical_columns:
                if df_rth_filled[column].isna().any():
                    error_msg = f"NaNs detected in {column} — check data integrity"
                    track_issue("errors", "other", error_msg, level="error", date=date)
                    raise ValueError(f"❌ {error_msg}")
                if not df_rth_filled[column].apply(lambda x: pd.notna(x) and np.isfinite(x)).all():
                    error_msg = f"Non-finite values (inf/-inf) in {column} — check data integrity"
                    track_issue("errors", "other", error_msg, level="error", date=date)
                    raise ValueError(f"❌ {error_msg}")

            if len(df_rth_filled) < PARAMS['min_spy_data_rows']:
                short_data_msg = f"SPY data for {date} is unusually short with only {len(df_rth_filled)} rows after pulling from API. This may indicate incomplete data."
                print(f"⚠️ {short_data_msg}")
                track_issue("warnings", "short_data_warnings", short_data_msg, date=date)

            df_rth_filled.to_pickle(spy_path)
            if DEBUG_MODE:
                print("💾 SPY data pulled and cached.")

        # === STEP 5b: Load or pull option chain ===
        chain_path = os.path.join(CHAIN_DIR, f"{ticker}_chain_{date}.pkl")
        if os.path.exists(chain_path):
            df_chain = pd.read_pickle(chain_path)
            if DEBUG_MODE:
                print("📂 Option chain loaded from cache.")
            if len(df_chain) < PARAMS['min_option_chain_rows']:
                short_data_msg = f"Option chain data for {date} is unusually short with only {len(df_chain)} rows. This may indicate incomplete data."
                print(f"⚠️ {short_data_msg}")
                track_issue("warnings", "short_data_warnings", short_data_msg, date=date)
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
                    error_msg = f"{contract_type.upper()} request failed: {resp.status_code}"
                    track_issue("errors", "api_connection_failures", error_msg, level="error", date=date)
                    raise Exception(error_msg)
                df = pd.DataFrame(resp.json().get("results", []))
                df["option_type"] = contract_type
                return df

            df_calls = fetch_chain("call")
            df_puts = fetch_chain("put")
            df_chain = pd.concat([df_calls, df_puts], ignore_index=True)
            df_chain["ticker_clean"] = df_chain["ticker"].str.replace("O:", "", regex=False)

            # Check for unusually short data before caching
            if len(df_chain) < PARAMS['min_option_chain_rows']:
                short_data_msg = f"Option chain data for {date} is unusually short with only {len(df_chain)} rows after pulling from API. This may indicate incomplete data."
                print(f"⚠️ {short_data_msg}")
                track_issue("warnings", "short_data_warnings", short_data_msg, date=date)

            df_chain.to_pickle(chain_path)
            if DEBUG_MODE:
                print("💾 Option chain pulled and cached.")

        if df_chain.empty:
            no_data_msg = f"No option chain data for {date} — skipping."
            print(f"⚠️ {no_data_msg}")
            track_issue("warnings", "other", no_data_msg, date=date)
            issue_tracker["days"]["skipped_warnings"] += 1
            continue

        # === Insert strategy logic here ===
        stretch_signals = detect_stretch_signal(df_rth_filled, PARAMS)

        # Log if no stretch signals are detected
        if stretch_signals.empty:
            no_signals_msg = f"No stretch signals detected for {date} — skipping."
            print(f"⚠️ {no_signals_msg}")
            # This is normal behavior, not a warning
            continue

        stretch_signals = detect_partial_reclaims(df_rth_filled, stretch_signals, PARAMS)
        
        # Ensure 'entry_intent' column exists
        if 'entry_intent' not in stretch_signals.columns:
            error_msg = f"'entry_intent' column missing for {date} — skipping."
            print(f"⚠️ {error_msg}")
            track_issue("errors", "other", error_msg, level="error", date=date)
            issue_tracker["days"]["skipped_errors"] += 1
            continue

        # Filter for valid entry signals
        valid_entries = stretch_signals[stretch_signals['entry_intent'] == True]
        
        # Track opportunity stats
        total_signals = len(stretch_signals)
        total_valid_entries = len(valid_entries)
        issue_tracker["opportunities"]["total_stretch_signals"] += total_signals
        issue_tracker["opportunities"]["valid_entry_opportunities"] += total_valid_entries
        
        # Initialize container for daily contracts
        daily_contracts = []
        
        # MODIFIED: Process ALL valid entries instead of just the first one
        if not valid_entries.empty:
            if DEBUG_MODE:
                print(f"✅ Found {len(valid_entries)} valid entry signals for {date}")
            
            # Process each valid entry signal
            for idx, entry_signal in valid_entries.iterrows():
                entry_time = entry_signal['reclaim_ts']
                
                # FAILSAFE: Check for late entry cutoff first (blocks entries that are too late in the day)
                is_blocked, block_message = check_late_entry_cutoff(entry_time, PARAMS)
                if is_blocked:
                    print(f"⛔ {block_message}")
                    
                    # Track this in risk management stats
                    current_date = entry_time.strftime("%Y-%m-%d")
                    issue_tracker["risk_management"]["late_entries_blocked"] += 1
                    issue_tracker["risk_management"]["late_entry_dates"].add(current_date)
                    
                    # Skip this entry and continue to next signal
                    continue
                
                entry_price = entry_signal['reclaim_price']
                
                # IMPORTANT: Store original signal time and SPY price at signal time (before latency)
                original_signal_time = entry_time  # entry_time is reclaim_ts at this point
                spy_price_at_signal = df_rth_filled[df_rth_filled['ts_raw'] == original_signal_time]['close'].iloc[0]
                
                # Select appropriate option contract using SPY price at SIGNAL time
                # CHANGED: Using SPY price at signal time (not execution time) for contract selection
                # This prevents look-ahead bias by ensuring we only use information available at signal time
                # to decide which option contract to trade
                selected_contract = select_option_contract(entry_signal, df_chain, spy_price_at_signal, PARAMS)
                
                # Get SPY price at entry for logging/reporting purposes (after latency will be applied)
                spy_price_at_entry = df_rth_filled[df_rth_filled['ts_raw'] == entry_time]['close'].iloc[0]
                
                # We'll add the look-ahead bias verification log after latency is applied
                
                if selected_contract:
                    # Now we need to load the option price data for the selected contract
                    option_ticker = selected_contract['ticker']
                    
                    # === STEP 5d: Load or pull option price data ===
                    option_path = os.path.join(OPTION_DIR, f"{date}_{option_ticker.replace(':', '')}.pkl")
                    if os.path.exists(option_path):
                        df_option_rth = pd.read_pickle(option_path)
                        if DEBUG_MODE:
                            print(f"📂 Option price data for {option_ticker} loaded from cache.")
                        if len(df_option_rth) < PARAMS['min_option_price_rows']:
                            short_data_msg = f"Option price data for {option_ticker} on {date} is unusually short with only {len(df_option_rth)} rows. This may indicate incomplete data."
                            print(f"⚠️ {short_data_msg}")
                            track_issue("warnings", "short_data_warnings", short_data_msg, date=date)
                    else:
                        option_url = (
                            f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/range/1/second/"
                            f"{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={API_KEY}"
                        )
                        resp = requests.get(option_url)
                        option_results = resp.json().get("results", [])
                        df_option = pd.DataFrame(option_results)

                        if df_option.empty:
                            missing_data_msg = f"No option price data for {option_ticker} on {date} — skipping this entry."
                            print(f"⚠️ {missing_data_msg}")
                            track_issue("errors", "missing_option_price_data", missing_data_msg, level="error", date=date)
                            issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
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
                            short_data_msg = f"Option price data for {option_ticker} on {date} is unusually short with only {len(df_option_rth)} rows after pulling from API. This may indicate incomplete data."
                            print(f"⚠️ {short_data_msg}")
                            track_issue("warnings", "short_data_warnings", short_data_msg, date=date)

                        df_option_rth.to_pickle(option_path)
                        if DEBUG_MODE:
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
                    if DEBUG_MODE:
                        print(f"🧪 Signal #{idx+1}: Timestamp mismatches for {option_ticker}: {mismatch_count}")
                    
                    # Track timestamp mismatches
                    if mismatch_count > 0:
                        if mismatch_count > mismatch_threshold:
                            mismatch_msg = f"Timestamp mismatch in {mismatch_count} rows exceeds threshold of {mismatch_threshold}"
                            track_issue("data_integrity", "timestamp_mismatches", mismatch_msg, date=date)
                        else:
                            mismatch_msg = f"Timestamp mismatch in {mismatch_count} rows (below threshold of {mismatch_threshold})"
                            track_issue("warnings", "timestamp_mismatches_below_threshold", mismatch_msg, date=date)
                    
                    # Hash-based timestamp verification as additional sanity check
                    def hash_timestamps(df):
                        return hashlib.md5("".join(df["ts_raw"].astype(str)).encode()).hexdigest()

                    spy_hash = hash_timestamps(df_rth_filled)
                    opt_hash = hash_timestamps(df_option_aligned)
                    hash_match = spy_hash == opt_hash
                    
                    # Track hash mismatches - this happens regardless of debug mode
                    if not hash_match:
                        hash_mismatch_msg = f"Hash mismatch between SPY and option data"
                        track_issue("data_integrity", "hash_mismatches", hash_mismatch_msg, date=date)
                    
                    # Only print the debug information if debug mode is on
                    if DEBUG_MODE:
                        print(f"⏱️ SPY rows: {len(df_rth_filled)}")
                        print(f"⏱️ OPT rows: {len(df_option_aligned)}")
                        print(f"🔐 SPY hash:  {spy_hash}")
                        print(f"🔐 OPT hash:  {opt_hash}")
                        print(f"🔍 Hash match: {hash_match}")
                    
                    # Check if mismatches exceed the threshold
                    if mismatch_count > mismatch_threshold:
                        print(f"⚠️ Timestamp mismatch in {mismatch_count} rows exceeds threshold of {mismatch_threshold} — skipping this entry.")
                        issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                        continue
                    
                    # Capture the original price at original timestamp before applying latency
                    original_option_row = df_option_aligned[df_option_aligned['ts_raw'] == original_signal_time]
                    original_price = None
                    if not original_option_row.empty:
                        if pd.notna(original_option_row['vwap'].iloc[0]):
                            original_price = original_option_row['vwap'].iloc[0]
                        elif pd.notna(original_option_row['close'].iloc[0]):
                            original_price = original_option_row['close'].iloc[0]

                    # Apply latency to entry if configured
                    latency_seconds = PARAMS.get('latency_seconds', 0)
                    if latency_seconds > 0:
                        latency_result = apply_latency(original_signal_time, df_option_aligned, latency_seconds)
                        
                        if latency_result['is_valid']:
                            # Use delayed time and price
                            entry_time = latency_result['delayed_timestamp']
                            option_entry_price = latency_result['delayed_price']
                            option_row = df_option_aligned[df_option_aligned['ts_raw'] == entry_time]
                            
                            # Update spy_price_at_entry after latency applied
                            spy_price_at_entry = df_rth_filled[df_rth_filled['ts_raw'] == entry_time]['close'].iloc[0]
                            
                            # Check if we had to use close price instead of VWAP in the latency result
                            if 'vwap' in latency_result['delayed_row'] and pd.isna(latency_result['delayed_row']['vwap']):
                                # Log the fallback to close price
                                fallback_msg = f"Falling back to close price for latency-adjusted entry at {entry_time} - VWAP not available"
                                if DEBUG_MODE:
                                    print(f"⚠️ {fallback_msg}")
                                track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=date)
                            
                            if DEBUG_MODE:
                                # Log look-ahead bias fix verification (after latency applied)
                                print(f"🔍 Look-ahead bias fix verification:")
                                print(f"   Signal time: {original_signal_time.strftime('%H:%M:%S')}, SPY price: ${spy_price_at_signal:.4f}")
                                print(f"   Entry time:  {entry_time.strftime('%H:%M:%S')}, SPY price: ${spy_price_at_entry:.4f}")
                                print(f"   Using signal time price for option contract selection")
                                
                                print(f"🕒 Entry latency applied: {latency_seconds}s")
                                print(f"   Original signal: {original_signal_time.strftime('%H:%M:%S')}")
                                print(f"   Execution time: {entry_time.strftime('%H:%M:%S')}")
                                if original_price is not None:
                                    # Now correctly comparing delayed price with original price
                                    print(f"   Price difference: ${latency_result['delayed_price'] - original_price:.4f}")
                                else:
                                    print(f"   Price difference: Unable to calculate - original price data not found")
                        else:
                            # If we can't find data at the delayed timestamp, skip this entry
                            latency_error_msg = f"No option price data after applying entry latency of {latency_seconds}s from {original_signal_time}"
                            print(f"⚠️ {latency_error_msg}")
                            track_issue("errors", "latency_entry_failures", latency_error_msg, level="error", date=date)
                            issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                            continue
                    else:
                        # Original logic without latency
                        option_row = df_option_aligned[df_option_aligned['ts_raw'] == entry_time]
                        
                        if option_row.empty:
                            missing_price_msg = f"Could not find option price for entry at {entry_time} - skipping this entry"
                            print(f"⚠️ {missing_price_msg}")
                            track_issue("errors", "missing_option_price_data", missing_price_msg, level="error", date=date)
                            issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                            continue
                        
                        # Extract entry price for the option (with fallback to close)
                        if pd.notna(option_row['vwap'].iloc[0]):
                            option_entry_price = option_row['vwap'].iloc[0]
                        elif pd.notna(option_row['close'].iloc[0]):
                            # Log the fallback to close price
                            fallback_msg = f"Falling back to close price for entry at {entry_time} - VWAP not available"
                            if DEBUG_MODE:
                                print(f"⚠️ {fallback_msg}")
                            track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=date)
                            option_entry_price = option_row['close'].iloc[0]
                        else:
                            # Neither VWAP nor close is valid - skip this entry
                            missing_price_msg = f"Both VWAP and close prices missing for entry at {entry_time} - skipping this entry"
                            print(f"⚠️ {missing_price_msg}")
                            track_issue("errors", "missing_option_price_data", missing_price_msg, level="error", date=date)
                            issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                            continue
                    
                    # Check price staleness at entry
                    staleness_threshold = PARAMS['price_staleness_threshold_seconds']
                    price_staleness = option_row['seconds_since_update'].iloc[0]
                    is_price_stale = price_staleness > staleness_threshold
                    
                    if is_price_stale and PARAMS['report_stale_prices']:
                        staleness_msg = f"Signal #{idx+1}: Using stale option price at entry - {price_staleness:.1f} seconds old"
                        print(f"⚠️ {staleness_msg}")
                        track_issue("warnings", "price_staleness", staleness_msg, date=date)
                    
                    # Store contract with complete entry details
                    contract_with_entry = {
                        **selected_contract,
                        'entry_time': entry_time,
                        'original_signal_time': original_signal_time,
                        'latency_seconds': PARAMS.get('latency_seconds', 0),
                        'latency_applied': PARAMS.get('latency_seconds', 0) > 0,
                        'entry_spy_price': spy_price_at_entry,
                        'spy_price_at_signal': spy_price_at_signal,  # Add this one new field 
                        'entry_option_price': option_entry_price,
                        'price_staleness_seconds': price_staleness,
                        'is_price_stale': is_price_stale,
                        'signal_number': idx + 1,  # Add signal sequence number for reference
                        'df_option_aligned': df_option_aligned,  # Save aligned option data for later use
                        'entry_signal': entry_signal.to_dict()
                    }
                    
                    # Process exits for this contract
                    contract_with_entry = process_exits_for_contract(contract_with_entry, PARAMS)
                    
                    # Memory optimization: Clean up df_option_aligned after exit processing is complete
                    # This prevents memory accumulation when processing multiple trades within a day
                    if 'df_option_aligned' in contract_with_entry:
                        del contract_with_entry['df_option_aligned']
                    
                    daily_contracts.append(contract_with_entry)
                    
                    # Track option contract selection
                    issue_tracker["opportunities"]["total_options_contracts"] += 1
                    
                    # Note: No longer deleting df_option_aligned here, will clean up after all processing
        
        # Count the number of entry intent signals for the day
        daily_entry_intent_signals = stretch_signals['entry_intent'].sum()
        total_entry_intent_signals += daily_entry_intent_signals
        days_processed += 1
        
        # Track successfully processed day
        issue_tracker["days"]["processed"] += 1

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
                # Add to our master list of all contracts
                all_contracts.append(contract)

    except Exception as e:
        error_msg = f"{date} — Error: {str(e)}"
        print(f"❌ {error_msg}")
        track_issue("errors", "other", error_msg, level="error", date=date)
        issue_tracker["days"]["skipped_errors"] += 1
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
    
    # === SLIPPAGE ADJUSTMENT (Post-Processing) ===
    # Apply slippage without modifying original trade decisions
    SLIPPAGE_PERCENT = PARAMS['slippage_percent']
    
    # Create slippage-adjusted entry and exit prices (worse entries, worse exits)
    contracts_df['entry_option_price_slipped'] = contracts_df['entry_option_price'] * (1 + SLIPPAGE_PERCENT)
    contracts_df['exit_price_slipped'] = contracts_df['exit_price'] * (1 - SLIPPAGE_PERCENT)
    
    # Recalculate P&L with slippage
    contracts_df['pnl_percent_slipped'] = ((contracts_df['exit_price_slipped'] - contracts_df['entry_option_price_slipped']) / 
                                           contracts_df['entry_option_price_slipped'] * 100)
    
    # Get contracts per trade from PARAMS
    CONTRACTS_PER_TRADE = PARAMS['contracts_per_trade']
    
    # Calculate per-share and dollar P&L with slippage
    contracts_df['pnl_per_share_slipped'] = contracts_df['exit_price_slipped'] - contracts_df['entry_option_price_slipped']
    contracts_df['pnl_dollars_slipped'] = contracts_df['pnl_per_share_slipped'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE
    
    # Calculate percentage impact (this one is fine to calculate here)
    contracts_df['slippage_impact_pct'] = contracts_df['pnl_percent_slipped'] - contracts_df['pnl_percent']
    
    # Dollar impact will be calculated later after pnl_dollars is created
    
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
    
    # Display contracts per trade info
    print(f"  Contracts per trade: {CONTRACTS_PER_TRADE}")
    
    # Get trading day distribution
    print("\n📆 Trades by Day:")
    date_counts = contracts_df['entry_time'].dt.date.value_counts().sort_index()
    for date, count in date_counts.items():
        print(f"  {date}: {count} trade(s)")
    
    # Average strike distance from price
    avg_diff = contracts_df['abs_diff'].mean()
    print(f"\n  Average distance from ATM: {avg_diff:.4f}")
    
    # Show contract multiplier stats
    print("\n📊 Contract Multiplier Statistics:")
    shares_counts = contracts_df['shares_per_contract'].value_counts()
    for shares, count in shares_counts.items():
        print(f"  {shares} shares per contract: {count} trade(s) ({count/len(contracts_df)*100:.1f}%)")
    
    # Add price staleness statistics if enabled
    if PARAMS['report_stale_prices'] and 'is_price_stale' in contracts_df.columns:
        stale_count = len(contracts_df[contracts_df['is_price_stale'] == True])
        fresh_count = len(contracts_df) - stale_count
        avg_staleness = contracts_df['price_staleness_seconds'].mean()
        max_staleness = contracts_df['price_staleness_seconds'].max()
        
        print("\n🔍 Entry Price Staleness Statistics:")
        print(f"  Fresh price entries: {fresh_count} ({(fresh_count/len(contracts_df))*100:.1f}%)")
        print(f"  Stale price entries: {stale_count} ({(stale_count/len(contracts_df))*100:.1f}%)")
        print(f"  Average staleness: {avg_staleness:.2f} seconds")
        print(f"  Maximum staleness: {max_staleness:.2f} seconds")
        
        # Add exit price staleness statistics if those fields exist
        if 'is_exit_price_stale' in contracts_df.columns and 'exit_price_staleness_seconds' in contracts_df.columns:
            # Filter to only include rows with valid exit staleness data
            valid_exit_data = contracts_df.dropna(subset=['is_exit_price_stale', 'exit_price_staleness_seconds'])
            
            if not valid_exit_data.empty:
                exit_stale_count = len(valid_exit_data[valid_exit_data['is_exit_price_stale'] == True])
                exit_fresh_count = len(valid_exit_data) - exit_stale_count
                exit_avg_staleness = valid_exit_data['exit_price_staleness_seconds'].mean()
                exit_max_staleness = valid_exit_data['exit_price_staleness_seconds'].max()
                
                print("\n🔍 Exit Price Staleness Statistics:")
                print(f"  Fresh price exits: {exit_fresh_count} ({(exit_fresh_count/len(valid_exit_data))*100:.1f}%)")
                print(f"  Stale price exits: {exit_stale_count} ({(exit_stale_count/len(valid_exit_data))*100:.1f}%)")
                print(f"  Average exit staleness: {exit_avg_staleness:.2f} seconds")
                print(f"  Maximum exit staleness: {exit_max_staleness:.2f} seconds")
                
                # Compare entry vs exit staleness
                if stale_count > 0 or exit_stale_count > 0:
                    print("\n🔍 Entry vs Exit Staleness Comparison:")
                    print(f"  Stale entries: {stale_count}/{len(contracts_df)} ({(stale_count/len(contracts_df))*100:.1f}%)")
                    print(f"  Stale exits: {exit_stale_count}/{len(valid_exit_data)} ({(exit_stale_count/len(valid_exit_data))*100:.1f}%)")
                    print(f"  Average entry staleness: {avg_staleness:.2f} seconds")
                    print(f"  Average exit staleness: {exit_avg_staleness:.2f} seconds")
    
    # Add exit reason distribution
    if 'exit_reason' in contracts_df.columns:
        print("\n🚪 Exit Reason Distribution:")
        exit_counts = contracts_df['exit_reason'].value_counts()
        for reason, count in exit_counts.items():
            print(f"  {reason}: {count} ({count/len(contracts_df)*100:.1f}%)")
    
    # Add P&L statistics
    if 'pnl_percent' in contracts_df.columns:
        # Count total trades vs trades with P&L
        total_trades = len(contracts_df)
        trades_with_pnl = contracts_df['pnl_percent'].notna().sum()
        trades_missing_pnl = total_trades - trades_with_pnl
        
        # Calculate percentage of missing P&L
        missing_pnl_percent = (trades_missing_pnl / total_trades) * 100 if total_trades > 0 else 0
        
        # Report the missing P&L percentage
        print(f"\n📊 P&L Data Completeness:")
        print(f"  Trades with P&L data: {trades_with_pnl}/{total_trades} ({100-missing_pnl_percent:.1f}%)")
        print(f"  Trades missing P&L data: {trades_missing_pnl}/{total_trades} ({missing_pnl_percent:.1f}%)")
        
        # Filter out None/NaN values
        pnl_data = contracts_df['pnl_percent'].dropna()
        pnl_slipped_data = contracts_df['pnl_percent_slipped'].dropna()
        
        if not pnl_data.empty:
            # Calculate dollar P&L statistics using contract multiplier
            contracts_df['pnl_per_share'] = contracts_df['exit_price'] - contracts_df['entry_option_price']
            # Apply contract multiplier for true dollar P&L
            contracts_df['pnl_dollars'] = contracts_df['pnl_per_share'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE
            
            # Now calculate dollar slippage impact
            contracts_df['slippage_impact_dollars'] = contracts_df['pnl_dollars_slipped'] - contracts_df['pnl_dollars']
            
            # Calculate transaction costs
            # Get fees from parameters
            brokerage_fee = PARAMS.get('brokerage_fee_per_contract', 0.65)
            exchange_fee = PARAMS.get('exchange_fee_per_contract', 0.65)
            fee_per_contract_per_direction = brokerage_fee + exchange_fee
            
            # Calculate fees for entry and exit (per trade)
            contracts_df['transaction_cost_entry'] = fee_per_contract_per_direction * CONTRACTS_PER_TRADE
            contracts_df['transaction_cost_exit'] = fee_per_contract_per_direction * CONTRACTS_PER_TRADE
            contracts_df['transaction_cost_total'] = contracts_df['transaction_cost_entry'] + contracts_df['transaction_cost_exit']
            
            # Calculate P&L with transaction costs
            contracts_df['pnl_dollars_with_fees'] = contracts_df['pnl_dollars'] - contracts_df['transaction_cost_total']
            contracts_df['pnl_dollars_slipped_with_fees'] = contracts_df['pnl_dollars_slipped'] - contracts_df['transaction_cost_total']
            
            # Prepare datasets for reporting
            dollar_pnl_data = contracts_df['pnl_dollars'].dropna()
            dollar_pnl_slipped_data = contracts_df['pnl_dollars_slipped'].dropna()
            dollar_pnl_with_fees_data = contracts_df['pnl_dollars_with_fees'].dropna()
            dollar_pnl_slipped_with_fees_data = contracts_df['pnl_dollars_slipped_with_fees'].dropna()
            
            if not dollar_pnl_data.empty:
                # Dollar P&L Statistics section removed as requested
                
                # Slippage-adjusted dollar P&L statistics section removed as requested
                
                # Transaction cost statistics section removed as requested
                
                # Fully adjusted P&L section removed as requested
                
                # Calculate total capital risked and total P&L with contract multiplier
                total_pnl = dollar_pnl_data.sum()
                total_pnl_slipped = dollar_pnl_slipped_data.sum()
                total_pnl_with_fees = dollar_pnl_with_fees_data.sum()
                total_pnl_slipped_with_fees = dollar_pnl_slipped_with_fees_data.sum()
                total_risked = (contracts_df['entry_option_price'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE).sum()
                total_risked_slipped = (contracts_df['entry_option_price_slipped'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE).sum()
                # All calculations for total P&L, win rates, etc. have been removed
                # since they're not used anywhere else in the code
            
            # Average P&L by exit reason
            print("\n  P&L by Exit Reason:")
            exit_pnl = contracts_df.groupby('exit_reason')['pnl_percent'].agg(['mean', 'count'])
            
            # Also calculate slippage-adjusted P&L by exit reason
            exit_pnl_slipped = contracts_df.groupby('exit_reason')['pnl_percent_slipped'].agg(['mean', 'count'])
            
            for reason, stats in exit_pnl.iterrows():
                slipped_mean = exit_pnl_slipped.loc[reason, 'mean'] if reason in exit_pnl_slipped.index else float('nan')
                print(f"    {reason}: {stats['mean']:.2f}% avg | With slippage: {slipped_mean:.2f}% ({stats['count']} trades)")
    
                # Add trade duration statistics
    if 'trade_duration_seconds' in contracts_df.columns:
        duration_data = contracts_df['trade_duration_seconds'].dropna()
        
        if not duration_data.empty:
            print("\n⏱️ Trade Duration Statistics:")
            print(f"  Average duration: {duration_data.mean():.1f} seconds")
            print(f"  Median duration: {duration_data.median():.1f} seconds")
            print(f"  Min duration: {duration_data.min():.1f} seconds")
            print(f"  Max duration: {duration_data.max():.1f} seconds")
    
    # Add latency statistics if latency was applied
    if 'latency_applied' in contracts_df.columns and contracts_df['latency_applied'].any():
        print("\n🕒 Latency Simulation Statistics:")
        
        # Latency stats
        if 'latency_seconds' in contracts_df.columns:
            latency_avg = contracts_df['latency_seconds'].mean()
            print(f"  Latency setting: {latency_avg:.1f} seconds")
        
        # Note: For accurate latency impact analysis, run the algorithm twice:
        # 1. Once with PARAMS['latency_seconds'] = 0
        # 2. Once with PARAMS['latency_seconds'] = 3 (or your desired latency)
        # Then compare the results between the two runs.
    
    # Sample of contracts with entry and exit details
    if DEBUG_MODE:
        print("\n🔍 Sample of trades with P&L (including latency, slippage, and transaction costs):")
        display_columns = ['original_signal_time', 'entry_time', 'option_type', 'strike_price', 
                       'entry_option_price', 'entry_option_price_slipped',
                       'original_exit_time', 'exit_time', 'exit_price', 'exit_price_slipped',
                       'pnl_percent', 'pnl_percent_slipped',
                       'transaction_cost_total', 'pnl_dollars', 'pnl_dollars_with_fees', 'pnl_dollars_slipped_with_fees',
                       'exit_reason', 'trade_duration_seconds']
    
        # Only include columns that exist
        existing_columns = [col for col in display_columns if col in contracts_df.columns]
        print(contracts_df[existing_columns].head(10))

# ==================== GENERATE SUMMARY REPORT ====================
print("\n" + "=" * 20 + " SUMMARY OF ERRORS + WARNINGS " + "=" * 20)

# Processing Stats Section
# Doesn't account for days skipped due to no valid entry signals.
print("\n📊 PROCESSING STATS:")
print(f"  - Days attempted: {issue_tracker['days']['attempted']}")
print(f"  - Days successfully processed: {issue_tracker['days']['processed']}")
print(f"  - Days skipped due to errors: {issue_tracker['days']['skipped_errors']}")
print(f"  - Days skipped due to warnings: {issue_tracker['days']['skipped_warnings']}")

# Data Integrity Section
print("\n🔍 DATA INTEGRITY:")
print(f"  - Hash mismatches: {issue_tracker['data_integrity']['hash_mismatches']}")
mismatch_count = issue_tracker['data_integrity']['timestamp_mismatches']
days_with_mismatches = issue_tracker['data_integrity']['days_with_mismatches']
if mismatch_count > 0:
    days_str = ", ".join(d for d in sorted(days_with_mismatches))
    print(f"  - Timestamp mismatches: {mismatch_count} (on {days_str})")
else:
    print(f"  - Timestamp mismatches: {mismatch_count}")

# Warning Summary
print("\n⚠️ WARNING SUMMARY:")
ticker_warnings = issue_tracker['warnings'][f"no_{ticker}_data"]
print(f"  - No {ticker} Data: {ticker_warnings}")
print(f"  - Price staleness: {issue_tracker['warnings']['price_staleness']}")
print(f"  - Short data warnings: {issue_tracker['warnings']['short_data_warnings']}")
print(f"  - Timestamp mismatches below threshold: {issue_tracker['warnings']['timestamp_mismatches_below_threshold']}")
print(f"  - Missing shares_per_contract: {issue_tracker['warnings']['shares_per_contract_missing']}")
print(f"  - Non-standard contract size: {issue_tracker['warnings']['non_standard_contract_size']}")
print(f"  - VWAP fallbacks to close price: {issue_tracker['warnings']['vwap_fallback_to_close']}")
print(f"  - Emergency exit activations: {issue_tracker['warnings']['emergency_exit_triggered']}")
print(f"  - Other warnings: {issue_tracker['warnings']['other']}")

# Show details of 'other' warnings if any exist
if issue_tracker['warnings']['other'] > 0 and issue_tracker['warnings']['details']:
    print("    Details:")
    for detail in issue_tracker['warnings']['details'][:5]:  # Show first 5 to avoid clutter
        print(f"    - {detail}")
    if len(issue_tracker['warnings']['details']) > 5:
        print(f"    ... and {len(issue_tracker['warnings']['details']) - 5} more")

# Error Summary
print("\n❌ ERROR SUMMARY:")
print(f"  - Missing option price data: {issue_tracker['errors']['missing_option_price_data']}")
print(f"  - API connection failures: {issue_tracker['errors']['api_connection_failures']}")
print(f"  - Missing exit data: {issue_tracker['errors']['missing_exit_data']}")
print(f"  - No future price data: {issue_tracker['errors']['no_future_price_data']}")
print(f"  - Forced exit at end of data: {issue_tracker['errors']['forced_exit_end_of_data']}")
print(f"  - Exit evaluation error: {issue_tracker['errors']['exit_evaluation_error']}")
print(f"  - Forced exit error: {issue_tracker['errors']['forced_exit_error']}")
print(f"  - Latency entry failures: {issue_tracker['errors']['latency_entry_failures']}")
print(f"  - Latency exit failures: {issue_tracker['errors']['latency_exit_failures']}")
print(f"  - Other errors: {issue_tracker['errors']['other']}")

# Show details of 'other' errors if any exist
if issue_tracker['errors']['other'] > 0 and issue_tracker['errors']['details']:
    print("    Details:")
    for detail in issue_tracker['errors']['details'][:5]:  # Show first 5 to avoid clutter
        print(f"    - {detail}")
    if len(issue_tracker['errors']['details']) > 5:
        print(f"    ... and {len(issue_tracker['errors']['details']) - 5} more")

# Opportunity Analysis
print("\n🎯 OPPORTUNITY ANALYSIS:")
print(f"  - Total stretch signals: {issue_tracker['opportunities']['total_stretch_signals']}")
print(f"  - Valid entry opportunities: {issue_tracker['opportunities']['valid_entry_opportunities']}")
print(f"  - Failed entries due to data issues: {issue_tracker['opportunities']['failed_entries_data_issues']}")
print(f"  - Total options contracts selected: {issue_tracker['opportunities']['total_options_contracts']}")

# Risk Management Statistics
print("\n🛡️ RISK MANAGEMENT STATISTICS:")
print(f"  - Emergency exits triggered: {issue_tracker['risk_management']['emergency_exits']}")
if issue_tracker['risk_management']['emergency_exits'] > 0:
    emergency_dates = sorted(issue_tracker['risk_management']['emergency_exit_dates'])
    dates_str = ", ".join(emergency_dates)
    print(f"  - Dates with emergency exits: {dates_str}")
    
print(f"  - Late entries blocked: {issue_tracker['risk_management']['late_entries_blocked']}")
if issue_tracker['risk_management']['late_entries_blocked'] > 0:
    late_entry_dates = sorted(issue_tracker['risk_management']['late_entry_dates'])
    dates_str = ", ".join(late_entry_dates)
    print(f"  - Dates with blocked late entries: {dates_str}")
    
print(f"  - Regular end-of-day exits: {sum(1 for c in all_contracts if c.get('exit_reason') == 'end_of_day')}")
print(f"  - Total trades with defined exit reason: {sum(1 for c in all_contracts if c.get('exit_reason') is not None)}")

print("\n" + "=" * 20 + " END OF REPORT " + "=" * 20)

# ==================== PERFORMANCE SUMMARY ====================
if all_contracts:
    # Create DataFrame if not already created
    if 'contracts_df' not in locals():
        contracts_df = pd.DataFrame(all_contracts)
    
    print("\n" + "=" * 20 + " PERFORMANCE SUMMARY " + "=" * 20)
    
    # 1. Total number of trades
    total_trades = len(contracts_df)
    print(f"\n💼 TOTAL TRADES: {total_trades}")
    
    # Filter for trades with valid fully-adjusted P&L data (slippage + fees)
    valid_pnl_contracts = contracts_df.dropna(subset=['pnl_dollars_slipped_with_fees'])
    
    if not valid_pnl_contracts.empty:
        # Win Rate
        winning_trades = valid_pnl_contracts[valid_pnl_contracts['pnl_dollars_slipped_with_fees'] > 0]
        win_rate = len(winning_trades) / len(valid_pnl_contracts) * 100
        print(f"\n🎯 WIN RATE: {win_rate:.2f}%")
        
        # Expectancy
        if not winning_trades.empty:
            avg_win = winning_trades['pnl_dollars_slipped_with_fees'].mean()
            losing_trades = valid_pnl_contracts[valid_pnl_contracts['pnl_dollars_slipped_with_fees'] < 0]
            
            if not losing_trades.empty:
                avg_loss = abs(losing_trades['pnl_dollars_slipped_with_fees'].mean())
                loss_rate = 1 - (len(winning_trades) / len(valid_pnl_contracts))
                
                expectancy = (win_rate/100 * avg_win) - (loss_rate * avg_loss)
                print(f"\n💡 EXPECTANCY: ${expectancy:.2f} per trade")
            else:
                print("\n💡 EXPECTANCY: ∞ (no losing trades)")
                # Set expectancy to a high value for risk-adjusted calculations if needed
                expectancy = float('inf')
        
        # Risk calculation (independent of expectancy)
        # Set up components for risk calculation
        contract_fees = PARAMS['brokerage_fee_per_contract'] + PARAMS['exchange_fee_per_contract']
        round_trip_fees = contract_fees * 2 * PARAMS['contracts_per_trade']  # Both entry and exit
        stop_loss_decimal = abs(PARAMS['stop_loss_percent']) / 100  # Convert to positive decimal
        
        # Calculate capital at risk (total position value)
        valid_pnl_contracts['capital_at_risk'] = (
            valid_pnl_contracts['entry_option_price_slipped'] * 
            valid_pnl_contracts['shares_per_contract'] * 
            PARAMS['contracts_per_trade']
        )
        
        # Calculate max loss per trade (stop loss + fees)
        valid_pnl_contracts['max_loss_per_trade'] = (
            valid_pnl_contracts['capital_at_risk'] * stop_loss_decimal
        ) + round_trip_fees
        
        # Calculate average risk per trade
        avg_risk_per_trade = valid_pnl_contracts['max_loss_per_trade'].mean()
        print(f"\n💵 AVERAGE RISK PER TRADE: ${avg_risk_per_trade:.2f}")
        
        # Risk-adjusted expectancy
        if 'expectancy' in locals() and avg_risk_per_trade > 0:
            if expectancy != float('inf'):
                risk_adjusted_expectancy = expectancy / avg_risk_per_trade
                return_on_risk_percent = risk_adjusted_expectancy * 100
                print(f"\n📊 AVERAGE RETURN ON RISK: {return_on_risk_percent:.2f}%")
            else:
                print(f"\n📊 AVERAGE RETURN ON RISK: ∞% (no losing trades)")
        
        # Sharpe Ratio (using daily returns, fully adjusted)
        # Group by date to get daily returns
        daily_returns = valid_pnl_contracts.groupby(valid_pnl_contracts['entry_time'].dt.date)['pnl_dollars_slipped_with_fees'].sum()
        
        if len(daily_returns) > 1:  # Need at least 2 days to calculate Sharpe
            # Unannualized Sharpe: Mean daily return / StdDev of daily returns
            mean_daily_return = daily_returns.mean()
            std_daily_return = daily_returns.std()
            
            if std_daily_return > 0:  # Prevent division by zero
                sharpe_ratio = mean_daily_return / std_daily_return
                print(f"\n📈 UNANNUALIZED SHARPE RATIO: {sharpe_ratio:.2f}")
            else:
                print("\n📈 UNANNUALIZED SHARPE RATIO: N/A (insufficient volatility)")
        else:
            print("\n📈 SHARPE RATIO: N/A (need data from at least two days)")
    else:
        print("\n⚠️ No valid P&L data available for performance metrics")

print("\n" + "=" * 20 + " END OF PERFORMANCE SUMMARY " + "=" * 20)
