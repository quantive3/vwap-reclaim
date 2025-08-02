# === STEP 1: Local Cache Setup ===
import os

# Create local cache directories
CACHE_DIR = "./polygon_cache"  # Local cache directory

# === STEP 2: API Key from Secret File ===
try:
    from config import API_KEY
    API_KEY_LOADED_FROM_SECRET = True
except ImportError:
    # Fallback to manual input if secret.py is not available
    API_KEY = input("🔑 Enter your Polygon API key (or create a secret.py file): ").strip()
    API_KEY_LOADED_FROM_SECRET = False

# === STEP 3: Imports and setup ===
import pandas as pd
# Set pandas option to use future behavior for fill operations
pd.set_option('future.no_silent_downcasting', True)
import requests
from datetime import time
import hashlib
import numpy as np
import cProfile
import pstats

from filelock import FileLock
from strategy.params import initialize_parameters, initialize_issue_tracker
# Import data module functions
from strategy.data import (
    setup_cache_directories, 
    load_with_cache_lock,
    load_spy_data, 
    load_chain_data, 
    load_option_data,
    set_issue_tracker,
    set_track_issue_function,
    set_hash_generation_function
)
# Import signal detection functions
from strategy.signals import detect_stretch_signal as detect_stretch_signal_from_signals
from strategy.signals import detect_partial_reclaims as detect_partial_reclaims_from_signals

# Functions moved to params.py

# The following functions have been moved to data.py:
# - setup_cache_directories
# - load_with_cache_lock
# - _fetch_spy (internal)
# - load_spy_data
# - load_chain_data

# === STEP 6: Define Parameters ===
# Get parameters from initialization function
PARAMS = initialize_parameters()

# === DataLoader Class ===
class DataLoader:
    """
    Data loading class that bundles SPY, option chain, and option price data loading.
    This is a pure wrapper around existing functions to maintain exact functional equivalence.
    """
    
    def __init__(self, api_key, cache_dir, params, debug_mode=False, silent_mode=False):
        """
        Initialize the DataLoader with configuration parameters.
        
        Args:
            api_key (str): Polygon API key
            cache_dir (str): Base cache directory path
            params (dict): Strategy parameters dictionary
            debug_mode (bool): Whether to enable debug output
            silent_mode (bool): Whether to suppress non-debug output
        """
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.params = params
        self.debug_mode = debug_mode
        self.silent_mode = silent_mode
    
    def load_spy(self, date):
        """
        Load SPY price data for a given date.
        
        Args:
            date (str): Date string in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with SPY price and VWAP data for the specified date
        """
        return load_spy_data(date, self.cache_dir, self.api_key, self.params, debug_mode=self.debug_mode)
    
    def load_chain(self, date):
        """
        Load option chain data for a given date.
        
        Args:
            date (str): Date string in format 'YYYY-MM-DD'
            
        Returns:
            pd.DataFrame: DataFrame with option chain data for the specified date
        """
        return load_chain_data(date, self.cache_dir, self.api_key, self.params, debug_mode=self.debug_mode)
    
    def load_option(self, ticker, date, df_rth_filled, signal_idx=None):
        """
        Load and process option price data for a given option ticker and date.
        
        Args:
            ticker (str): The option ticker symbol to load data for
            date (str): Date string in format 'YYYY-MM-DD'
            df_rth_filled (pd.DataFrame): DataFrame with SPY price data for timestamp alignment
            signal_idx (int, optional): Optional signal index for debug output
            
        Returns:
            tuple: (df_option_aligned, option_entry_price, status) where:
                - df_option_aligned: DataFrame with aligned option price data
                - option_entry_price: The entry price for the option (or None if no valid price found)
                - status: Dictionary with status information
        """
        return load_option_data(ticker, date, self.cache_dir, df_rth_filled, self.api_key, self.params, signal_idx=signal_idx)

# Print API key loading status now that PARAMS is available
if API_KEY_LOADED_FROM_SECRET and not PARAMS.get('silent_mode', False):
    print("✅ API key loaded from secret.py")

# === STEP 6b: Initialize Issue Tracker ===
issue_tracker = initialize_issue_tracker(PARAMS)

# Function to generate hash for dataframe verification
def generate_dataframe_hash(df, name):
    """
    Generate a deterministic hash for a dataframe to verify consistency between runs.
    Only runs when debug mode is enabled to avoid impacting performance.
    
    Args:
        df (pd.DataFrame): The dataframe to hash
        name (str): Name of the dataframe for logging
        
    Returns:
        str: Hash string if debug mode is on, empty string otherwise
    """
    # Skip entire process if debug mode is off
    if not DEBUG_MODE:
        return ""
        
    # Convert dataframe to string representation
    # Sort by all columns to ensure consistent ordering
    if not df.empty:
        # Try to sort if possible (some dataframes may not be sortable)
        try:
            df_sorted = df.sort_values(by=list(df.columns))
            df_str = df_sorted.to_string()
        except:
            # Fall back to unsorted string if sorting fails
            df_str = df.to_string()
    else:
        df_str = "EMPTY_DATAFRAME"
    
    # Generate hash
    hash_obj = hashlib.md5(df_str.encode())
    hash_str = hash_obj.hexdigest()
    
    # Print hash info
    print(f"🔐 {name} Hash: {hash_str}")
    print(f"📊 {name} Shape: {df.shape}")
    
    return hash_str

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
            
# Inject objects into data.py
set_issue_tracker(issue_tracker)
set_track_issue_function(track_issue)
set_hash_generation_function(generate_dataframe_hash)

# Use params for variables that were previously global
DEBUG_MODE = PARAMS['debug_mode']
start_date = PARAMS['start_date']
end_date = PARAMS['end_date']
business_days = pd.date_range(start=start_date, end=end_date, freq="B")
ticker = PARAMS['ticker']

# Import the extracted select_option_contract function and setup dependencies
from strategy.option_select import select_option_contract, set_track_issue_function, set_debug_mode
# Inject dependencies
set_track_issue_function(track_issue)
set_debug_mode(DEBUG_MODE)

# === STEP 4: Caching paths ===
# CACHE_DIR is already defined in Step 1
# Cache directories are now set up in data.py

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
    # Delegate to the implementation in signals.py
    return detect_stretch_signal_from_signals(df_rth_filled, params, DEBUG_MODE, PARAMS.get('silent_mode', False))

# === STEP 7b: Detect Partial Reclaims ===
def detect_partial_reclaims(df_rth_filled, stretch_signals, params):
    """
    For each stretch signal, detect if a partial reclaim toward VWAP occurs within the cooldown window.
    Returns stretch signals with reclaim metadata.
    """
    # Delegate to the implementation in signals.py
    return detect_partial_reclaims_from_signals(df_rth_filled, stretch_signals, params, DEBUG_MODE, PARAMS.get('silent_mode', False))

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
# Function moved to strategy/option_select.py

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
        if not params.get('silent_mode', False):
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
        if not params.get('silent_mode', False):
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
    if not params.get('silent_mode', False):
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

# The load_option_data function has been moved to data.py

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
        if not params.get('silent_mode', False):
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
        if not params.get('silent_mode', False):
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
                if not params.get('silent_mode', False):
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
                    if not params.get('silent_mode', False):
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
            if not params.get('silent_mode', False):
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
                if not params.get('silent_mode', False):
                    print(f"⚠️ {staleness_msg}")
                track_issue("warnings", "price_staleness", staleness_msg, date=entry_date)
            
            # Always show critical warning and track the issue
            if not params.get('silent_mode', False):
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
            if not params.get('silent_mode', False):
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

# === STEP 8: Define the backtest function ===
def run_backtest(params, data_loader, issue_tracker):
    """
    Run a backtest over a date range using the specified parameters.
    
    Args:
        params (dict): Dictionary of strategy parameters
        data_loader (DataLoader): Initialized DataLoader instance
        issue_tracker (dict): The issue tracking dictionary to use
        
    Returns:
        dict: Summary metrics from the backtest
    """
    # Get parameters
    debug_mode = params['debug_mode']
    start_date = params['start_date']
    end_date = params['end_date']
    business_days = pd.date_range(start=start_date, end=end_date, freq="B")
    ticker = params['ticker']
    
    # Initialize tracking variables
    total_entry_intent_signals = 0
    days_processed = 0
    all_contracts = []  # Master list to store all contract data
    
    # === Backtest loop ===
    for date_obj in business_days:
        date = date_obj.strftime("%Y-%m-%d")
        if debug_mode:
            print(f"\n📅 Processing {date}...")
        
        # Track day attempted
        issue_tracker["days"]["attempted"] += 1

        try:
            # === STEP 5a: Load or pull SPY OHLCV ===
            df_rth_filled = data_loader.load_spy(date)
            
            # If data loading failed, skip this day
            if df_rth_filled is None:
                issue_tracker["days"]["skipped_warnings"] += 1
                continue

            # === STEP 5b: Load or pull option chain ===
            df_chain = data_loader.load_chain(date)
            
            # If data loading failed, skip this day
            if df_chain is None:
                issue_tracker["days"]["skipped_warnings"] += 1
                continue

            # === Insert strategy logic here ===
            stretch_signals = detect_stretch_signal(df_rth_filled, params)

            # Log if no stretch signals are detected
            if stretch_signals.empty:
                no_signals_msg = f"No stretch signals detected for {date} — skipping."
                if not params.get('silent_mode', False):
                    print(f"⚠️ {no_signals_msg}")
                # This is normal behavior, not a warning
                continue

            stretch_signals = detect_partial_reclaims(df_rth_filled, stretch_signals, params)
            
            # Ensure 'entry_intent' column exists
            if 'entry_intent' not in stretch_signals.columns:
                error_msg = f"'entry_intent' column missing for {date} — skipping."
                if not params.get('silent_mode', False):
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
                if debug_mode:
                    print(f"✅ Found {len(valid_entries)} valid entry signals for {date}")
                
                # Process each valid entry signal
                for idx, entry_signal in valid_entries.iterrows():
                    entry_time = entry_signal['reclaim_ts']
                    
                    # FAILSAFE: Check for late entry cutoff first (blocks entries that are too late in the day)
                    is_blocked, block_message = check_late_entry_cutoff(entry_time, params)
                    if is_blocked:
                        if not params.get('silent_mode', False):
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
                    selected_contract = select_option_contract(entry_signal, df_chain, spy_price_at_signal, params)
                    
                    # Get SPY price at entry for logging/reporting purposes (after latency will be applied)
                    spy_price_at_entry = df_rth_filled[df_rth_filled['ts_raw'] == entry_time]['close'].iloc[0]
                    
                    # We'll add the look-ahead bias verification log after latency is applied
                    
                    if selected_contract:
                        # Now we need to load the option price data for the selected contract
                        option_ticker = selected_contract['ticker']
                        
                        # === STEP 5d: Load or pull option price data ===
                        df_option_aligned, option_entry_price, option_load_status = data_loader.load_option(
                            ticker=option_ticker,
                            date=date,
                            df_rth_filled=df_rth_filled,
                            signal_idx=idx
                        )
                        
                        # Check if option data loading was successful
                        if not option_load_status['success']:
                            # If we had an error loading the option data, skip this entry
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
                        latency_seconds = params.get('latency_seconds', 0)
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
                                    if debug_mode:
                                        print(f"⚠️ {fallback_msg}")
                                    track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=date)
                                
                                if debug_mode:
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
                                if debug_mode:
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
                        staleness_threshold = params['price_staleness_threshold_seconds']
                        price_staleness = option_row['seconds_since_update'].iloc[0]
                        is_price_stale = price_staleness > staleness_threshold
                        
                        if is_price_stale and params['report_stale_prices']:
                            staleness_msg = f"Signal #{idx+1}: Using stale option price at entry - {price_staleness:.1f} seconds old"
                            if not params.get('silent_mode', False):
                                print(f"⚠️ {staleness_msg}")
                            track_issue("warnings", "price_staleness", staleness_msg, date=date)
                        
                        # Store contract with complete entry details
                        contract_with_entry = {
                            **selected_contract,
                            'entry_time': entry_time,
                            'original_signal_time': original_signal_time,
                            'latency_seconds': params.get('latency_seconds', 0),
                            'latency_applied': params.get('latency_seconds', 0) > 0,
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
                        contract_with_entry = process_exits_for_contract(contract_with_entry, params)
                        
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

            if debug_mode:
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
            if not params.get('silent_mode', False):
                print(f"❌ {error_msg}")
            track_issue("errors", "other", error_msg, level="error", date=date)
            issue_tracker["days"]["skipped_errors"] += 1
            continue
    
    # Calculate performance metrics within the backtest function
    if all_contracts:
        # Create DataFrame for metrics calculation
        contracts_df = pd.DataFrame(all_contracts)
        
        # Apply slippage adjustments (same logic as in main script)
        SLIPPAGE_AMOUNT = params['slippage_amount']
        contracts_df['entry_option_price_slipped'] = contracts_df['entry_option_price'] + SLIPPAGE_AMOUNT
        contracts_df['exit_price_slipped'] = contracts_df['exit_price'] - SLIPPAGE_AMOUNT
        
        # Calculate P&L with slippage
        contracts_df['pnl_percent_slipped'] = ((contracts_df['exit_price_slipped'] - contracts_df['entry_option_price_slipped']) / 
                                               contracts_df['entry_option_price_slipped'] * 100)
        
        # Calculate per-share and dollar P&L with slippage
        CONTRACTS_PER_TRADE = params['contracts_per_trade']
        contracts_df['pnl_per_share_slipped'] = contracts_df['exit_price_slipped'] - contracts_df['entry_option_price_slipped']
        contracts_df['pnl_dollars_slipped'] = contracts_df['pnl_per_share_slipped'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE
        
        # Calculate transaction costs
        brokerage_fee = params.get('brokerage_fee_per_contract', 0.65)
        exchange_fee = params.get('exchange_fee_per_contract', 0.65)
        fee_per_contract_per_direction = brokerage_fee + exchange_fee
        contracts_df['transaction_cost_total'] = fee_per_contract_per_direction * 2 * CONTRACTS_PER_TRADE
        
        # Calculate P&L with transaction costs
        contracts_df['pnl_dollars_slipped_with_fees'] = contracts_df['pnl_dollars_slipped'] - contracts_df['transaction_cost_total']
        
        # Calculate the 6 performance metrics
        total_trades = len(contracts_df)
        
        # Initialize metrics with default values
        win_rate = None
        expectancy = None
        avg_risk_per_trade = None
        return_on_risk_percent = None
        sharpe_ratio = None
        
        # Filter for trades with valid fully-adjusted P&L data (slippage + fees)
        valid_pnl_contracts = contracts_df.dropna(subset=['pnl_dollars_slipped_with_fees'])
        
        if not valid_pnl_contracts.empty:
            # Win Rate
            winning_trades = valid_pnl_contracts[valid_pnl_contracts['pnl_dollars_slipped_with_fees'] > 0]
            win_rate = len(winning_trades) / len(valid_pnl_contracts) * 100
            
            # Expectancy
            if not winning_trades.empty:
                avg_win = winning_trades['pnl_dollars_slipped_with_fees'].mean()
                losing_trades = valid_pnl_contracts[valid_pnl_contracts['pnl_dollars_slipped_with_fees'] < 0]
                
                if not losing_trades.empty:
                    avg_loss = abs(losing_trades['pnl_dollars_slipped_with_fees'].mean())
                    loss_rate = 1 - (len(winning_trades) / len(valid_pnl_contracts))
                    
                    expectancy = (win_rate/100 * avg_win) - (loss_rate * avg_loss)
                else:
                    expectancy = float('inf')
            
            # Risk calculation
            contract_fees = params['brokerage_fee_per_contract'] + params['exchange_fee_per_contract']
            round_trip_fees = contract_fees * 2 * params['contracts_per_trade']
            stop_loss_decimal = abs(params['stop_loss_percent']) / 100
            
            # Calculate capital at risk and max loss per trade
            valid_pnl_contracts['capital_at_risk'] = (
                valid_pnl_contracts['entry_option_price_slipped'] * 
                valid_pnl_contracts['shares_per_contract'] * 
                params['contracts_per_trade']
            )
            valid_pnl_contracts['max_loss_per_trade'] = (
                valid_pnl_contracts['capital_at_risk'] * stop_loss_decimal
            ) + round_trip_fees
            
            avg_risk_per_trade = valid_pnl_contracts['max_loss_per_trade'].mean()
            
            # Risk-adjusted expectancy
            if expectancy is not None and avg_risk_per_trade > 0:
                if expectancy != float('inf'):
                    risk_adjusted_expectancy = expectancy / avg_risk_per_trade
                    return_on_risk_percent = risk_adjusted_expectancy * 100
                else:
                    return_on_risk_percent = float('inf')
            
            # Sharpe Ratio
            daily_returns = valid_pnl_contracts.groupby(valid_pnl_contracts['entry_time'].dt.date)['pnl_dollars_slipped_with_fees'].sum()
            
            if len(daily_returns) > 1:
                mean_daily_return = daily_returns.mean()
                std_daily_return = daily_returns.std()
                
                if std_daily_return > 0:
                    sharpe_ratio = mean_daily_return / std_daily_return
        
        # Create metrics dictionary
        metrics = {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'expectancy': expectancy,
            'avg_risk_per_trade': avg_risk_per_trade,
            'return_on_risk_percent': return_on_risk_percent,
            'sharpe_ratio': sharpe_ratio
        }
    else:
        # Create empty metrics dictionary if no contracts
        metrics = {
            'total_trades': 0,
            'win_rate': None,
            'expectancy': None,
            'avg_risk_per_trade': None,
            'return_on_risk_percent': None,
            'sharpe_ratio': None
        }
    
    # Prepare summary metrics
    summary_metrics = {
        'total_days_processed': days_processed,
        'total_entry_intent_signals': total_entry_intent_signals,
        'total_contracts': len(all_contracts),
        'all_contracts': all_contracts,
        'metrics': metrics
    }
    
    # Calculate and add average entry intent signals
    if days_processed > 0:
        summary_metrics['average_entry_intent_signals'] = total_entry_intent_signals / days_processed
        
        # Calculate average successful entries per day
        if all_contracts:
            summary_metrics['average_entries_per_day'] = len(all_contracts) / days_processed
            if total_entry_intent_signals > 0:
                summary_metrics['success_rate'] = len(all_contracts) / total_entry_intent_signals * 100
            else:
                summary_metrics['success_rate'] = 0
    
    return summary_metrics

if __name__ == "__main__":
    # === STEP 8: Run backtest ===
    # Setup cache directories
    spy_dir, chain_dir, option_dir = setup_cache_directories(CACHE_DIR)
    
    # Initialize DataLoader
    data_loader = DataLoader(API_KEY, CACHE_DIR, PARAMS, debug_mode=PARAMS['debug_mode'], silent_mode=PARAMS.get('silent_mode', False))
    
    # === PROFILED BACKTEST ===
    if PARAMS.get('enable_profiling', False):
        profiler = cProfile.Profile()
        profiler.enable()

    backtest_results = run_backtest(PARAMS, data_loader, issue_tracker)

    if PARAMS.get('enable_profiling', False):
        profiler.disable()
        profiler.dump_stats('backtest.prof')
        
        stats = pstats.Stats('backtest.prof')
        stats.strip_dirs().sort_stats('cumtime').print_stats(20)

    # Extract results
    total_entry_intent_signals = backtest_results['total_entry_intent_signals']
    days_processed = backtest_results['total_days_processed']
    all_contracts = backtest_results['all_contracts']
    metrics = backtest_results['metrics']

    # Debug: Show extracted metrics dictionary
    if PARAMS['debug_mode']:
        print(f"\n🔍 DEBUG - EXTRACTED METRICS DICTIONARY:")
        for key, value in metrics.items():
            if value is None:
                print(f"  {key}: None")
            elif isinstance(value, float):
                if value == float('inf'):
                    print(f"  {key}: ∞")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    # Calculate and log the average number of daily entry intent signals
    if days_processed > 0:
        average_entry_intent_signals = total_entry_intent_signals / days_processed
        if not PARAMS.get('silent_mode', False):
            print(f"📊 Average daily entry intent signals over the period: {average_entry_intent_signals:.2f}")
        
        # Calculate average successful entries per day
        if all_contracts:
            average_entries_per_day = len(all_contracts) / days_processed
            success_rate = len(all_contracts) / total_entry_intent_signals * 100 if total_entry_intent_signals > 0 else 0
            if not PARAMS.get('silent_mode', False):
                print(f"📊 Average successful trades per day: {average_entries_per_day:.2f}")
                print(f"📊 Overall success rate: {success_rate:.1f}% ({len(all_contracts)}/{total_entry_intent_signals} signals)")
    else:
        if not PARAMS.get('silent_mode', False):
            print("⚠️ No days processed, cannot calculate average entry intent signals.")

    # Summary of contract selections
    if all_contracts:
        if not PARAMS.get('silent_mode', False):
            print(f"\n📈 Total option contracts selected: {len(all_contracts)}")
        
        # Create a DataFrame for easier analysis
        contracts_df = pd.DataFrame(all_contracts)
        
        # === SLIPPAGE ADJUSTMENT (Post-Processing) ===
        # Apply flat slippage without modifying original trade decisions
        SLIPPAGE_AMOUNT = PARAMS['slippage_amount']
        contracts_df['entry_option_price_slipped'] = contracts_df['entry_option_price'] + SLIPPAGE_AMOUNT
        contracts_df['exit_price_slipped'] = contracts_df['exit_price'] - SLIPPAGE_AMOUNT
        
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
        
        if not PARAMS.get('silent_mode', False):
            print(f"  Call options: {call_count} ({call_count/len(contracts_df)*100:.1f}%)")
            print(f"  Put options: {put_count} ({put_count/len(contracts_df)*100:.1f}%)")
        
        # Count by positioning
        atm_count = len(contracts_df[contracts_df['is_atm'] == True])
        itm_count = len(contracts_df[contracts_df['is_itm'] == True])
        otm_count = len(contracts_df[(contracts_df['is_atm'] == False) & (contracts_df['is_itm'] == False)])
        
        if not PARAMS.get('silent_mode', False):
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
            
            if not PARAMS.get('silent_mode', False):
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
                    
                    if not PARAMS.get('silent_mode', False):
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
            if not PARAMS.get('silent_mode', False):
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
            if not PARAMS.get('silent_mode', False):
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
                if not PARAMS.get('silent_mode', False):
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
                if not PARAMS.get('silent_mode', False):
                    print("\n⏱️ Trade Duration Statistics:")
                    print(f"  Average duration: {duration_data.mean():.1f} seconds")
                    print(f"  Median duration: {duration_data.median():.1f} seconds")
                    print(f"  Min duration: {duration_data.min():.1f} seconds")
                    print(f"  Max duration: {duration_data.max():.1f} seconds")
        
        # Add latency statistics if latency was applied
        if 'latency_applied' in contracts_df.columns and contracts_df['latency_applied'].any():
            if not PARAMS.get('silent_mode', False):
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
                           'transaction_cost_total', 'pnl_dollars', 'pnl_dollars_with_fees', 'pnl_dollars_slipped_with_fees',
                           'exit_reason', 'trade_duration_seconds']
        
            # Only include columns that exist
            existing_columns = [col for col in display_columns if col in contracts_df.columns]
            print(contracts_df[existing_columns].head(10))
            # Also export the sample to CSV for easier review when in debug mode
            contracts_df[existing_columns].head(20).to_csv("sample_trades.csv", index=False)
            print("🔄 Sample of trades also written to sample_trades.csv")

    # ==================== GENERATE SUMMARY REPORT ====================
    if not PARAMS.get('silent_mode', False):
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

        # Check if there are any warnings (excluding the 'details' list)
        total_warnings = sum(count for key, count in issue_tracker['warnings'].items() if key != 'details')
        if total_warnings > 0:
            print("⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️")  # Print 10 warning symbols if any warnings exist

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

        # Check if there are any errors (excluding the 'details' list)
        total_errors = sum(count for key, count in issue_tracker['errors'].items() if key != 'details')
        if total_errors > 0:
            print("❌❌❌❌❌❌❌❌❌❌")  # Print 10 error symbols if any errors exist

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
    if not PARAMS.get('silent_mode', False):
        if all_contracts:
            # Create DataFrame if not already created
            if 'contracts_df' not in locals():
                contracts_df = pd.DataFrame(all_contracts)
            
            print("\n" + "=" * 20 + " PERFORMANCE SUMMARY " + "=" * 20)
            
            # 1. Total number of trades
            total_trades = len(contracts_df)
            print(f"\n💼 TOTAL TRADES: {total_trades}")
            
            # Initialize metrics with default values
            win_rate = None
            expectancy = None
            avg_risk_per_trade = None
            return_on_risk_percent = None
            sharpe_ratio = None
            
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
                        return_on_risk_percent = float('inf')
                
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
                        sharpe_ratio = None
                else:
                    print("\n📈 SHARPE RATIO: N/A (need data from at least two days)")
                    sharpe_ratio = None
            else:
                print("\n⚠️ No valid P&L data available for performance metrics")

        print("\n" + "=" * 20 + " END OF PERFORMANCE SUMMARY " + "=" * 20)
