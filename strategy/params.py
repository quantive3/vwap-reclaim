# === Parameter and Issue Tracker Initialization ===
from datetime import time

def initialize_parameters():
    """
    Initialize and return the strategy parameters dictionary.
    This function centralizes all parameter definitions.
    
    Returns:
        dict: Dictionary containing all strategy parameters
    """
    return {
        # Print/info gates
        'debug_mode': True,  # Enable/disable debug outputs. Significantly slows execution.
        'silent_mode': False,  # Enable/disable all non-debug print outputs
        'enable_profiling': False,  # Enable/disable cProfile profiling

        # Backtest period
        'start_date': "2023-01-04",
        'end_date': "2023-01-04",

        # Time windows
        'entry_start_time': time(9, 30),
        'entry_end_time': time(15, 45),
        
        # Entry parameters
        'stretch_threshold': 0.003,  # 0.3%
        'reclaim_threshold': 0.0021,  # 0.2% - should always be less than stretch threshold
        'cooldown_period_seconds': 120,  # Cooldown period in seconds
        
        # Exit conditions
        'take_profit_percent': 80,     # Take profit at 25% gain
        'stop_loss_percent': -25,      # Stop loss at 50% loss
        'max_trade_duration_seconds': 600,  # Exit after 300 seconds (5 minutes)
        'late_entry_cutoff_time': time(15, 54),  # No new entries after this time
        'end_of_day_exit_time': time(15, 54),  # trade exit cutoff
        'emergency_exit_time': time(15, 55),   # absolute failsafe exit (overrides all other logic)
        
        # Option selection
        'ticker': 'SPY',
        'require_same_day_expiry': True,  # Whether to strictly require same-day expiry options
        'strikes_depth': 1,  # Number of strikes from ATM to target (1 = closest, 2 = second closest, etc.). Always use 1 or greater.
        'option_selection_mode': 'itm',  # Options: 'itm', 'otm', or 'atm' - determines whether to select in-the-money, out-of-money, or at-the-money options
        
        # Position sizing
        'contracts_per_trade': 1,  # Number of contracts to trade per signal (for P&L calculations)
        
        # Real-world trading friction
        'slippage_amount': 0.02,   # fixed slippage per share
        'latency_seconds': 1,    # Seconds delay between signal and execution (0 = disabled)
        'brokerage_fee_per_contract': 0.65,  # Brokerage fee per contract per direction (entry or exit)
        'exchange_fee_per_contract': 0.65,   # Exchange and other fees per contract per direction
        
        # Data quality thresholds - for error checking
        'min_spy_data_rows': 10000,  # Minimum acceptable rows for SPY data
        'min_option_chain_rows': 10,  # Minimum acceptable rows for option chain data
        'min_option_price_rows': 10000,  # Minimum acceptable rows for option price data
        'timestamp_mismatch_threshold': 0,  # Maximum allowable timestamp mismatches
        'price_staleness_threshold_seconds': 10,  # Maximum allowable staleness in seconds for option prices
        'report_stale_prices': True,  # Enable/disable reporting of stale prices
    }

def initialize_issue_tracker(params):
    """
    Initialize and return the issue tracking dictionary.
    
    Args:
        params (dict): Strategy parameters
        
    Returns:
        dict: Dictionary for tracking issues during strategy execution
    """
    ticker = params['ticker']
    return {
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
            f"no_{ticker}_data": 0,  # Dynamically use the ticker name
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