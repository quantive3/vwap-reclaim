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
    set_issue_tracker,
    set_track_issue_function,
    set_hash_generation_function
)
from strategy.data_loader import DataLoader
# Import signal detection functions
from strategy.signals import detect_stretch_signal as detect_stretch_signal_from_signals
from strategy.signals import detect_partial_reclaims as detect_partial_reclaims_from_signals

# Import exit-related functions
from strategy.exits import (
    apply_latency,
    evaluate_exit_conditions,
    check_emergency_exit_time,
    process_emergency_exit,
    process_exits_for_contract,
    set_track_issue_function as set_exits_track_issue_function,
    set_issue_tracker as set_exits_issue_tracker
)

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

# DataLoader class has been moved to strategy/data_loader.py

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

# Function to track issues - this is now also defined in backtest.py
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

# Inject objects into exits.py
set_exits_issue_tracker(issue_tracker)
set_exits_track_issue_function(track_issue)

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

# The following functions have been moved to strategy/backtest.py:
# - detect_stretch_signal (wrapper)
# - detect_partial_reclaims (wrapper)
# - check_late_entry_cutoff
# - run_backtest

# Initialize a counter for total entry intent signals
total_entry_intent_signals = 0
days_processed = 0
all_contracts = []  # Master list to store all contract data

if __name__ == "__main__":
    # === STEP 8: Run backtest ===
    # Setup cache directories
    spy_dir, chain_dir, option_dir = setup_cache_directories(CACHE_DIR)
    
    # Initialize DataLoader from the imported class
    data_loader = DataLoader(API_KEY, CACHE_DIR, PARAMS, debug_mode=PARAMS['debug_mode'], silent_mode=PARAMS.get('silent_mode', False))
    
    # Import run_backtest from backtest module
    from strategy.backtest import run_backtest
    
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
