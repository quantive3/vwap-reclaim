# === Standard Library Imports ===
import os  # noqa: F401
import pandas as pd
import requests  # noqa: F401
from datetime import time  # noqa: F401
import hashlib
import numpy as np  # noqa: F401
import cProfile
import pstats
from filelock import FileLock  # noqa: F401

# === Strategy Module Imports ===
from strategy.params import initialize_parameters, initialize_issue_tracker
from strategy.data import (
    setup_cache_directories,
    set_issue_tracker,
    set_track_issue_function,
    set_hash_generation_function
)
from strategy.data_loader import DataLoader
from strategy.signals import detect_stretch_signal as detect_stretch_signal_from_signals  # noqa: F401
from strategy.signals import detect_partial_reclaims as detect_partial_reclaims_from_signals  # noqa: F401
from strategy.exits import (  # noqa: F401
    apply_latency,
    evaluate_exit_conditions,
    check_emergency_exit_time,
    process_emergency_exit,
    process_exits_for_contract,
    set_track_issue_function as set_exits_track_issue_function,
    set_issue_tracker as set_exits_issue_tracker
)

# === Local Cache Setup ===
# Create local cache directories
CACHE_DIR = "./polygon_cache"  # Local cache directory

# === API Key from Secret File ===
try:
    from config import API_KEY
    API_KEY_LOADED_FROM_SECRET = True
except ImportError:
    # Fallback to manual input if secret.py is not available
    API_KEY = input("🔑 Enter your Polygon API key (or create a secret.py file): ").strip()
    API_KEY_LOADED_FROM_SECRET = False

# Set pandas option to use future behavior for fill operations
pd.set_option('future.no_silent_downcasting', True)

# === Define Parameters ===
# Get parameters from initialization function
PARAMS = initialize_parameters()

# DataLoader class has been moved to strategy/data_loader.py

# Print API key loading status now that PARAMS is available
if API_KEY_LOADED_FROM_SECRET and not PARAMS.get('silent_mode', False):
    print("✅ API key loaded from secret.py")

# === Initialize Issue Tracker ===
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
        except:  # noqa: E722
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
from strategy.option_select import select_option_contract, set_track_issue_function, set_debug_mode  # noqa: E402, F401
# Inject dependencies
set_track_issue_function(track_issue)
set_debug_mode(DEBUG_MODE)

# Import reporting functions
from strategy.reporting import print_summary_report, print_performance_summary  # noqa: E402

# Initialize a counter for total entry intent signals
total_entry_intent_signals = 0
days_processed = 0
all_contracts = []  # Master list to store all contract data

if __name__ == "__main__":
    # === Run backtest ===
    # Setup cache directories
    spy_dir, chain_dir, option_dir = setup_cache_directories(CACHE_DIR)
    
    # Initialize DataLoader from the imported class
    data_loader = DataLoader(API_KEY, CACHE_DIR, PARAMS, debug_mode=PARAMS['debug_mode'], silent_mode=PARAMS.get('silent_mode', False))
    
    # Import run_backtest from backtest module
    from strategy.backtest import run_backtest  # noqa: E402
    
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
        print("\n🔍 DEBUG - EXTRACTED METRICS DICTIONARY:")
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
        atm_count = len(contracts_df[contracts_df['is_atm'] == True])  # noqa: E712
        itm_count = len(contracts_df[contracts_df['is_itm'] == True])  # noqa: E712
        otm_count = len(contracts_df[(contracts_df['is_atm'] == False) & (contracts_df['is_itm'] == False)])  # noqa: E712
        
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
            stale_count = len(contracts_df[contracts_df['is_price_stale'] == True])  # noqa: E712
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
                    exit_stale_count = len(valid_exit_data[valid_exit_data['is_exit_price_stale'] == True])  # noqa: E712
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
                print("\n📊 P&L Data Completeness:")
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
                    
                    # Calculate total capital risked and total P&L with contract multiplier
                    total_pnl = dollar_pnl_data.sum()
                    total_pnl_slipped = dollar_pnl_slipped_data.sum()
                    total_pnl_with_fees = dollar_pnl_with_fees_data.sum()
                    total_pnl_slipped_with_fees = dollar_pnl_slipped_with_fees_data.sum()
                    total_risked = (contracts_df['entry_option_price'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE).sum()
                    total_risked_slipped = (contracts_df['entry_option_price_slipped'] * contracts_df['shares_per_contract'] * CONTRACTS_PER_TRADE).sum()
                
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
            # Export the sample to CSV for easier review when in debug mode
            contracts_df[existing_columns].head(20).to_csv("sample_trades.csv", index=False)
            print("🔄 Sample of trades also written to sample_trades.csv")

    # Replace inline summary and performance printing with reporting module calls
    if not PARAMS.get('silent_mode', False):
        print_summary_report(issue_tracker, all_contracts)
        print_performance_summary(contracts_df)
