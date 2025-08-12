# === Backtest Module ===
# Contains the run_backtest function
import pandas as pd
import numpy as np  # noqa: F401
from datetime import time

# === Stretch Signal Detection ===
def detect_stretch_signal(df_rth_filled, params):
    """
    Detects stretch signals when SPY price moves beyond VWAP by X%.

    Parameters:
    - df_rth_filled: DataFrame containing SPY price and VWAP data.
    - params: Dictionary of parameters including stretch threshold.

    Returns:
    - signals: DataFrame with stretch signals.
    """
    # Delegate to the implementation in signals.py
    from strategy.signals import detect_stretch_signal as detect_stretch_signal_from_signals
    return detect_stretch_signal_from_signals(df_rth_filled, params, params['debug_mode'], params.get('silent_mode', False))

# === Detect Partial Reclaims ===
def detect_partial_reclaims(df_rth_filled, stretch_signals, params):
    """
    For each stretch signal, detect if a partial reclaim toward VWAP occurs within the cooldown window.
    Returns stretch signals with reclaim metadata.
    """
    # Delegate to the implementation in signals.py
    from strategy.signals import detect_partial_reclaims as detect_partial_reclaims_from_signals
    return detect_partial_reclaims_from_signals(df_rth_filled, stretch_signals, params, params['debug_mode'], params.get('silent_mode', False))

# === Late Entry Blocker Failsafe ===
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

# === Define the backtest function ===
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
    ticker = params['ticker']  # noqa: F841
    
    # Import required functions
    from strategy.option_select import select_option_contract
    from strategy.exits import (  # noqa: F401
        apply_latency,
        evaluate_exit_conditions,
        check_emergency_exit_time,
        process_emergency_exit,
        process_exits_for_contract
    )
    from datetime import time  # noqa: F401
    
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
            # === Load or pull SPY OHLCV ===
            df_rth_filled = data_loader.load_spy(date)
            
            # If data loading failed, skip this day
            if df_rth_filled is None:
                issue_tracker["days"]["skipped_warnings"] += 1
                continue

            # === Load or pull option chain ===
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
                track_issue(issue_tracker, "errors", "other", error_msg, level="error", date=date)
                issue_tracker["days"]["skipped_errors"] += 1
                continue

            # Filter for valid entry signals
            valid_entries = stretch_signals[stretch_signals['entry_intent'] == True]  # noqa: E712
            
            # Track opportunity stats
            total_signals = len(stretch_signals)
            total_valid_entries = len(valid_entries)
            issue_tracker["opportunities"]["total_stretch_signals"] += total_signals
            issue_tracker["opportunities"]["valid_entry_opportunities"] += total_valid_entries
            
            # Initialize container for daily contracts
            daily_contracts = []
            
            # Process ALL valid entries instead of just the first one
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
                    
                    entry_price = entry_signal['reclaim_price']  # noqa: F841
                    
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
                        # We need to load the option price data for the selected contract
                        option_ticker = selected_contract['ticker']
                        
                        # === Load or pull option price data ===
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
                        latency_result = None
                        option_row = None
                        
                        if latency_seconds > 0:
                            latency_result = apply_latency(original_signal_time, df_option_aligned, latency_seconds)
                            
                            if latency_result['is_valid']:
                                # Use delayed time and price
                                entry_time = latency_result['delayed_timestamp']
                                option_entry_price = latency_result['delayed_price']
                                
                                # Get the option row for staleness checking
                                option_row = df_option_aligned[df_option_aligned['ts_raw'] == entry_time]
                                
                                # Update spy_price_at_entry after latency applied
                                spy_price_at_entry = df_rth_filled[df_rth_filled['ts_raw'] == entry_time]['close'].iloc[0]
                                
                                # Check if we had to use close price instead of VWAP in the latency result
                                if 'vwap' in latency_result['delayed_row'] and pd.isna(latency_result['delayed_row']['vwap']):
                                    # Log the fallback to close price
                                    fallback_msg = f"Falling back to close price for latency-adjusted entry at {entry_time} - VWAP not available"
                                    if debug_mode:
                                        print(f"⚠️ {fallback_msg}")
                                    track_issue(issue_tracker, "warnings", "vwap_fallback_to_close", fallback_msg, date=date)
                                
                                if debug_mode:
                                    # Log look-ahead bias fix verification (after latency applied)
                                    print("🔍 Look-ahead bias fix verification:")
                                    print(f"   Signal time: {original_signal_time.strftime('%H:%M:%S')}, SPY price: ${spy_price_at_signal:.4f}")
                                    print(f"   Entry time:  {entry_time.strftime('%H:%M:%S')}, SPY price: ${spy_price_at_entry:.4f}")
                                    print("   Using signal time price for option contract selection")
                                    
                                    print(f"🕒 Entry latency applied: {latency_seconds}s")
                                    print(f"   Original signal: {original_signal_time.strftime('%H:%M:%S')}")
                                    print(f"   Execution time: {entry_time.strftime('%H:%M:%S')}")
                                    if original_price is not None:
                                        # Now correctly comparing delayed price with original price
                                        print(f"   Price difference: ${latency_result['delayed_price'] - original_price:.4f}")
                                    else:
                                        print("   Price difference: Unable to calculate - original price data not found")  # noqa: F541
                            else:
                                # If we can't find data at the delayed timestamp, skip this entry
                                latency_error_msg = f"No option price data after applying entry latency of {latency_seconds}s from {original_signal_time}"
                                print(f"⚠️ {latency_error_msg}")
                                track_issue(issue_tracker, "errors", "latency_entry_failures", latency_error_msg, level="error", date=date)
                                issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                                continue
                        else:
                            # Original logic without latency
                            option_row = df_option_aligned[df_option_aligned['ts_raw'] == entry_time]
                            
                            if option_row.empty:
                                missing_price_msg = f"Could not find option price for entry at {entry_time} - skipping this entry"
                                print(f"⚠️ {missing_price_msg}")
                                track_issue(issue_tracker, "errors", "missing_option_price_data", missing_price_msg, level="error", date=date)
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
                                track_issue(issue_tracker, "warnings", "vwap_fallback_to_close", fallback_msg, date=date)
                                option_entry_price = option_row['close'].iloc[0]
                            else:
                                # Neither VWAP nor close is valid - skip this entry
                                missing_price_msg = f"Both VWAP and close prices missing for entry at {entry_time} - skipping this entry"
                                print(f"⚠️ {missing_price_msg}")
                                track_issue(issue_tracker, "errors", "missing_option_price_data", missing_price_msg, level="error", date=date)
                                issue_tracker["opportunities"]["failed_entries_data_issues"] += 1
                                continue
                        
                        # Verify we have a valid option_row for staleness checking
                        if option_row is None or option_row.empty:
                            missing_row_msg = f"Missing option row for staleness check at {entry_time} - skipping this entry"
                            print(f"⚠️ {missing_row_msg}")
                            track_issue(issue_tracker, "errors", "missing_option_price_data", missing_row_msg, level="error", date=date)
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
                            track_issue(issue_tracker, "warnings", "price_staleness", staleness_msg, date=date)
                        
                        # Store contract with complete entry details
                        contract_with_entry = {
                            **selected_contract,
                            'entry_time': entry_time,
                            'original_signal_time': original_signal_time,
                            'latency_seconds': params.get('latency_seconds', 0),
                            'latency_applied': params.get('latency_seconds', 0) > 0,
                            'entry_spy_price': spy_price_at_entry,
                            'spy_price_at_signal': spy_price_at_signal,  
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
                        
            # Count the number of entry intent signals for the day
            daily_entry_intent_signals = stretch_signals['entry_intent'].sum()
            total_entry_intent_signals += daily_entry_intent_signals
            days_processed += 1
            
            # Track successfully processed day
            issue_tracker["days"]["processed"] += 1

            if debug_mode:
                print(f"🎯 Entry intent signals (valid reclaims): {daily_entry_intent_signals}")
                # Report successful entries vs attempted entries
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
            track_issue(issue_tracker, "errors", "other", error_msg, level="error", date=date)
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

# Helper function for tracking issues
def track_issue(issue_tracker, category, subcategory, message, level="warning", date=None):
    """
    Track an issue in the issue tracker.
    
    Parameters:
    - issue_tracker: The issue tracker dictionary
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