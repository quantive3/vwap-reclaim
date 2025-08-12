# === Exit Logic Module ===
# Contains all exit-related logic
import pandas as pd
import numpy as np  # noqa: F401
from datetime import time

# Module-level placeholders for injected dependencies
_track_issue = None
_issue_tracker = None

# Setter functions for dependency injection
def set_track_issue_function(func):
    """
    Set the track_issue function.
    
    Args:
        func: Function for tracking issues
    """
    global _track_issue
    _track_issue = func

def set_issue_tracker(tracker):
    """
    Set the issue tracker object.
    
    Args:
        tracker: Issue tracker dictionary
    """
    global _issue_tracker
    _issue_tracker = tracker

# === Apply Latency to Timestamps ===
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

# === Exit Logic ===
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

# === Emergency Failsafe Exit System ===
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
        _track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
    
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
    _track_issue("warnings", "emergency_exit_triggered", 
               f"Emergency exit for {contract['ticker']} entered at {entry_time}", 
               level="warning", date=entry_date)
    
    # Also track in risk_management stats
    _issue_tracker["risk_management"]["emergency_exits"] += 1
    _issue_tracker["risk_management"]["emergency_exit_dates"].add(entry_date)
    
    return contract

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
        _track_issue("errors", "missing_exit_data", 
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
        _track_issue("errors", "no_future_price_data", 
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
            _track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
        
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
                            _track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
                            
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
                        _track_issue("errors", "latency_exit_failures", exit_latency_msg, level="error", date=entry_date)
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
                    _track_issue("warnings", "price_staleness", staleness_msg, date=entry_date)
                
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
            _track_issue("errors", "exit_evaluation_error", 
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
                _track_issue("warnings", "vwap_fallback_to_close", fallback_msg, date=entry_date)
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
                _track_issue("warnings", "price_staleness", staleness_msg, date=entry_date)
            
            # Always show critical warning and track the issue
            if not params.get('silent_mode', False):
                print(f"❌ Forced exit at end of available data: {contract['ticker']}")
            
            # Get entry date for tracking
            entry_date = entry_time.strftime("%Y-%m-%d")
            
            # Track this issue as an error, not a warning
            _track_issue("errors", "forced_exit_end_of_data", 
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
            _track_issue("errors", "forced_exit_error", 
                      error_msg, level="error", date=entry_date)
            
            # Mark this trade with an error
            contract['exit_reason'] = "forced_exit_error"
            contract['is_closed'] = True  # Mark as closed so we don't keep trying to process it
    
    return contract