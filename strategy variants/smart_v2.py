# === SMART GRID SEARCH FOR VWAP BOUNCE STRATEGY ===
import optuna
import pandas as pd
from datetime import time
import datetime   # added so eval() can resolve datetime.time
import copy
import sys
import os
from optuna.storages import RDBStorage
import sqlalchemy
from sqlalchemy.exc import IntegrityError

# Import credentials from config file
from config import PG_HOST, PG_PORT, PG_DATABASE, PG_USER, PG_PASSWORD

# PostgreSQL connection info (still allows override via env vars)
PG_HOST     = os.getenv("PG_HOST",     PG_HOST)
PG_PORT     = os.getenv("PG_PORT",     PG_PORT)
PG_DATABASE = os.getenv("PG_DATABASE", PG_DATABASE)
PG_USER     = os.getenv("PG_USER",     PG_USER)
PG_PASSWORD = os.getenv("PG_PASSWORD", PG_PASSWORD)

POSTGRES_URL = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}"
    f"@{PG_HOST}:{PG_PORT}/{PG_DATABASE}"
)

# Import the main strategy components
from strategy.main import (
    initialize_parameters, 
    initialize_issue_tracker,
    DataLoader,
    run_backtest,
    API_KEY,
    CACHE_DIR
)
# Import data module functions
from strategy.data import setup_cache_directories

# Track all param‐combos we've already tried
seen = set()

# Override backtest dates (edit these to change the backtest window)
BACKTEST_START_DATE = "2023-01-01"
BACKTEST_END_DATE   = "2024-05-31"

# Name of the Optuna study (edit this to start a fresh study)
STUDY_NAME = "vwap_bounce_optimization_v2"

# Configuration flags
ENABLE_PERSISTENCE = True  # Set to True to accumulate trials across runs
OPTIMIZATION_SEED = None     # Set to a number for reproducible results, or None for random
N_TRIALS = 150  # Adjust based on your computational budget

# Database connection pool settings
DB_POOL_SIZE    = 3   # Number of persistent connections to the Postgres DB
DB_MAX_OVERFLOW = 5  # Additional "burst" connections above DB_POOL_SIZE
DB_POOL_TIMEOUT = 30  # Seconds to wait for a connection from the pool

# TPE Sampler configuration
N_STARTUP_TRIALS = 50      # Number of random trials before TPE optimization starts
N_EI_CANDIDATES = 48       # Number of candidates evaluated per TPE trial

# Pruning configuration
MIN_TRADE_THRESHOLD = 170     # Minimum trades required for valid trial
MAX_ATTEMPT_LIMIT = 600      # Maximum total attempts (including pruned trials)

# Entry windows mapping - used throughout the optimization
ENTRY_WINDOWS = {
    0: (time(9, 30), time(10, 30)),   # 9:30 to 10:30
    1: (time(9, 30), time(12, 0)),    # 9:30 to 12:00
    2: (time(12, 0), time(15, 0)),    # 12:00 to 15:00
    3: (time(15, 0), time(15, 45)),   # 15:00 to 15:45
    4: (time(9, 30), time(15, 45))    # 9:30 to 15:45
}

from optuna.samplers import TPESampler
from optuna.trial import TrialState

# Global variable to store the SQLAlchemy engine
db_engine = None

def ensure_seen_combos_table(engine):
    """
    Ensure the seen_combos table exists in the database.
    
    Args:
        engine: SQLAlchemy engine connected to the database
    """
    try:
        # Create the seen_combos table if it doesn't exist
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text("""
                CREATE TABLE IF NOT EXISTS seen_combos (
                    study_name TEXT NOT NULL,
                    combo_key TEXT NOT NULL,
                    PRIMARY KEY (study_name, combo_key)
                )
            """))
            connection.commit()
        print("✅ Verified seen_combos table exists")
    except Exception as e:
        print(f"⚠️ Error ensuring seen_combos table: {e}")
        raise

class ValidCountTPESampler(TPESampler):
    def __init__(self, n_valid_startup_trials, **kwargs):
        # Force parent's n_startup_trials to 0; we'll control startup ourselves
        super().__init__(n_startup_trials=0, **kwargs)
        self._n_valid_startup = n_valid_startup_trials

    def _is_startup(self, study):
        # Count only COMPLETE trials
        valid = sum(1 for t in study.trials if t.state is TrialState.COMPLETE)
        return valid < self._n_valid_startup

def create_sampler():
    """
    Create Optuna sampler based on configuration.
    
    Returns:
        optuna.samplers.BaseSampler: Configured sampler
    """
    sampler_kwargs = dict(
        seed=OPTIMIZATION_SEED,
        n_ei_candidates=N_EI_CANDIDATES
    )
    return ValidCountTPESampler(
        n_valid_startup_trials=N_STARTUP_TRIALS,
        **sampler_kwargs
    )

def validate_loaded_study(study):
    """
    Basic safety check for loaded study - fails fast on obvious problems.
    
    Args:
        study: Loaded Optuna study object
        
    Raises:
        ValueError: If study has critical issues
    """
    if study.direction.name != 'MAXIMIZE':
        raise ValueError("⛔ Loaded study direction mismatch - expected MAXIMIZE")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"✅ Study validation passed - {len(completed_trials)} valid trials")
    return len(completed_trials)

def create_optimized_params(trial):
    """
    Create parameter dictionary with Optuna trial suggestions.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        dict: Optimized parameters dictionary
    """
    # Start with base parameters
    base_params = initialize_parameters()
    
    # Override default backtest date range if desired
    if BACKTEST_START_DATE is not None:
        base_params['start_date'] = BACKTEST_START_DATE
    if BACKTEST_END_DATE is not None:
        base_params['end_date'] = BACKTEST_END_DATE
    
    # Suggest parameters from the specified ranges
    entry_window_idx = trial.suggest_categorical('entry_window', [0, 1, 2, 3, 4])
    stretch_threshold = trial.suggest_categorical('stretch_threshold', 
        [0.001, 0.002, 0.003, 0.004, 0.005])
    reclaim_percentage = trial.suggest_categorical('reclaim_percentage', [0.4, 0.5, 0.6, 0.7, 0.8])
    cooldown_period = trial.suggest_categorical('cooldown_period_seconds', [60, 90, 120, 180, 300, 600])
    take_profit = trial.suggest_categorical('take_profit_percent', [25, 35, 50, 60, 70, 80, 90, 100])
    stop_loss = trial.suggest_categorical('stop_loss_percent', [-25, -35, -50, -60, -70, -80, -90, -100])
    max_duration = trial.suggest_categorical('max_trade_duration_seconds', [600, 900, 1200, 1500, 1800, 2700, 3600])
    strikes_depth = trial.suggest_categorical('strikes_depth', [1])
    option_mode = trial.suggest_categorical('option_selection_mode', ['itm', 'otm'])
    
    # Apply the optimized parameters
    optimized_params = copy.deepcopy(base_params)
    
    # Entry window
    start_time, end_time = ENTRY_WINDOWS[entry_window_idx]
    optimized_params['entry_start_time'] = start_time
    optimized_params['entry_end_time'] = end_time
    
    # Stretch and reclaim thresholds
    optimized_params['stretch_threshold'] = stretch_threshold
    optimized_params['reclaim_threshold'] = stretch_threshold * reclaim_percentage
    
    # Other parameters
    optimized_params['cooldown_period_seconds'] = cooldown_period
    optimized_params['take_profit_percent'] = take_profit
    optimized_params['stop_loss_percent'] = stop_loss
    optimized_params['max_trade_duration_seconds'] = max_duration
    optimized_params['strikes_depth'] = strikes_depth
    optimized_params['option_selection_mode'] = option_mode
    
    # Enable silent mode for optimization (reduce output noise)
    optimized_params['silent_mode'] = True
    optimized_params['debug_mode'] = False
    
    return optimized_params

def objective(trial):
    """
    Objective function for Optuna optimization.
    Maximizes average return on risk.
    
    Args:
        trial: Optuna trial object
        
    Returns:
        float: Average return on risk percentage (to be maximized)
    """
    global seen, db_engine
    
    try:
        # Create optimized parameters for this trial
        params = create_optimized_params(trial)
        
        # Skip if this combo was already tried (in-memory check)
        key = tuple(sorted(params.items()))
        if key in seen:
            trial.set_user_attr('duplicate', True)
            trial.set_user_attr('duplicate_source', 'memory')
            raise optuna.TrialPruned("duplicate parameter combo (memory)")
        
        # Skip if this combo exists in the database (persistent check)
        study_name = trial.study.study_name
        combo_key = str(key)  # Convert tuple to string for storage
        
        try:
            # Try to insert into seen_combos table
            with db_engine.connect() as connection:
                connection.execute(
                    sqlalchemy.text(
                        "INSERT INTO seen_combos (study_name, combo_key) VALUES (:study_name, :combo_key)"
                    ),
                    {"study_name": study_name, "combo_key": combo_key}
                )
                connection.commit()
        except IntegrityError:
            # If insert fails due to primary key violation, it's a duplicate
            trial.set_user_attr('duplicate', True)
            trial.set_user_attr('duplicate_source', 'database')
            raise optuna.TrialPruned("duplicate parameter combo (database)")
        except Exception as e:
            # Log other database errors but continue with the trial
            print(f"⚠️ Database error in duplicate check: {e}")
        
        # Add to in-memory set
        seen.add(key)
        
        # Initialize issue tracker for this trial
        issue_tracker = initialize_issue_tracker(params)
        
        # Setup data loader
        data_loader = DataLoader(
            API_KEY, 
            CACHE_DIR, 
            params, 
            debug_mode=False, 
            silent_mode=True
        )
        
        # Run backtest with optimized parameters
        backtest_results = run_backtest(params, data_loader, issue_tracker)
        
        # Extract metrics
        metrics = backtest_results.get('metrics', {})
        
        # Pruning logic - Check minimum trade requirement
        total_trades = metrics.get('total_trades', 0)
        trial.set_user_attr('total_trades', total_trades)  # Store before potential pruning
        
        if total_trades < MIN_TRADE_THRESHOLD:
            trial.set_user_attr('pruned_reason', 'insufficient_trades')
            raise optuna.TrialPruned(f"Insufficient trades: {total_trades} < {MIN_TRADE_THRESHOLD}")
        
        # Store ALL metrics as user attributes dynamically
        for metric_name, metric_value in metrics.items():
            if metric_value is not None:
                trial.set_user_attr(metric_name, metric_value)
        
        return_on_risk = metrics.get('return_on_risk_percent')
        
        # Handle edge cases
        if return_on_risk is None:
            # No valid trades or unable to calculate
            return -1000.0  # Large negative value to discourage this parameter set
        elif return_on_risk == float('inf'):
            # All winning trades - cap at a high but finite value
            return 1000.0
        elif return_on_risk == float('-inf'):
            # All losing trades
            return -1000.0
        else:
            return return_on_risk
            
    except optuna.TrialPruned:
        # Re-raise pruned trials (don't catch them)
        raise
    except Exception as e:
        print(f"Error in trial: {str(e)}")
        # Store error info
        trial.set_user_attr('error', str(e))
        # Return large negative value for failed trials
        return -1000.0

def run_optimization(n_trials=100, study_name=STUDY_NAME, max_attempts=None):
    """
    Run the Optuna optimization study.
    
    Args:
        n_trials (int): Number of trials to run
        study_name (str): Name for the study
        
    Returns:
        optuna.Study: Completed study object
    """
    global db_engine
    
    print(f"🚀 Starting VWAP Bounce Strategy Optimization")
    print(f"📊 Target: Maximize Average Return on Risk")
    print(f"🔄 Target completed trials: {n_trials}")
    print(f"🚫 Maximum total attempts: {max_attempts or MAX_ATTEMPT_LIMIT}")
    print(f"📊 Minimum trades per trial: {MIN_TRADE_THRESHOLD}")
    print(f"📈 Parameters: 9 dimensions")
    print("-" * 50)
    
    # PostgreSQL-based Optuna storage using RDBStorage
    storage = RDBStorage(
        url=POSTGRES_URL,
        engine_kwargs={
            "pool_size": DB_POOL_SIZE,
            "max_overflow": DB_MAX_OVERFLOW,
            "pool_timeout": DB_POOL_TIMEOUT,
        }
    )
    
    # Get the SQLAlchemy engine from the RDBStorage
    db_engine = storage.engine
    
    # Ensure the seen_combos table exists
    ensure_seen_combos_table(db_engine)
    
    # Handle persistence flag: optionally delete existing study in Postgres
    if not ENABLE_PERSISTENCE:
        try:
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"🔄 Persistence disabled – deleted existing study '{study_name}' in Postgres")
            
            # Also clear the seen_combos table for this study
            with db_engine.connect() as connection:
                connection.execute(
                    sqlalchemy.text("DELETE FROM seen_combos WHERE study_name = :study_name"),
                    {"study_name": study_name}
                )
                connection.commit()
                print(f"🔄 Cleared seen_combos for study '{study_name}'")
                
        except KeyError:
            print(f"🆕 No existing study '{study_name}' to delete in Postgres")
        except Exception as e:
            print(f"⚠️ Failed to delete study '{study_name}': {e}")
    else:
        print(f"📂 Persistence enabled – using existing study '{study_name}' in Postgres (or creating new if none)")
    
    # Create or load study with PostgreSQL storage
    try:
        # Always try to load existing study first
        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction='maximize',
            sampler=create_sampler(),
            load_if_exists=True
        )
        
        # Count completed trials
        completed_in_study = validate_loaded_study(study)
        
        if completed_in_study > 0:
            print(f"📂 Loaded existing study with {completed_in_study} valid trials")
            print(f"🔄 Will add {n_trials} more valid trials (total valid will be {completed_in_study + n_trials})")
        else:
            print("🆕 Created new study (no previous trials found)")
    
    except Exception as e:
        print(f"⚠️ Error with study: {e}")
        print("⛔ Unable to create or load study - exiting")
        raise
    
    # Track all past parameter combinations
    global seen
    seen = { tuple(sorted(t.params.items())) for t in study.trials }
    
    # Also load parameter combinations from the seen_combos table
    try:
        with db_engine.connect() as connection:
            result = connection.execute(
                sqlalchemy.text("SELECT combo_key FROM seen_combos WHERE study_name = :study_name"),
                {"study_name": study_name}
            )
            # Note: This is a simplification - proper deserialization would be needed
            # for the actual tuple format, but this demonstrates the concept
            db_seen = {eval(row[0]) for row in result}
            seen.update(db_seen)
            print(f"📂 Loaded {len(db_seen)} parameter combinations from database")
    except Exception as e:
        print(f"⚠️ Error loading seen combinations from database: {e}")
    
    # Determine attempt limit
    attempt_limit = max_attempts or MAX_ATTEMPT_LIMIT
    
    # Capture baseline completed trials (for persistence support)
    baseline_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    
    # Run optimization - try to get n_trials completed, but don't exceed attempt_limit
    attempts_made = 0
    target_reached = False
    
    while attempts_made < attempt_limit:
        # Check current progress
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if len(completed_trials) - baseline_completed >= n_trials:
            target_reached = True
            break
        
        # Run one more trial
        remaining_attempts = attempt_limit - attempts_made
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        attempts_made += 1
        
        # Show progress
        completed_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        new_completed = completed_count - baseline_completed
        print(f"Progress: {new_completed}/{n_trials} new completed, {pruned_count} total pruned, {attempts_made}/{attempt_limit} attempts")
    
    # Final results
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    
    print(f"\n✅ Total completed trials in study: {len(completed_trials)}")
    print(f"✂️ Total pruned trials in study: {len(pruned_trials)}")
    print(f"🔄 Total attempts made in this run: {attempts_made}")
    
    if target_reached:
        print(f"🎯 Successfully reached target of {n_trials} completed trials!")
    else:
        print(f"⚠️ Only found {len(completed_trials)} valid trials out of {attempts_made} attempts")
        print(f"💡 Consider lowering MIN_TRADE_THRESHOLD or expanding parameter ranges")
    
    return study

def print_optimization_results(study):
    """
    Print detailed optimization results.
    
    Args:
        study: Completed Optuna study
    """
    print("\n" + "=" * 60)
    print("🎯 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    # Best trial info
    best_trial = study.best_trial
    print(f"\n🏆 BEST TRIAL:")
    print(f"   Trial Number: {best_trial.number}")
    print(f"   Best Return on Risk: {best_trial.value:.2f}%")
    
    # Best parameters
    print(f"\n⚙️ OPTIMAL PARAMETERS:")
    
    # Entry window mapping for display
    entry_window_names = {
        0: "9:30-10:30",
        1: "9:30-12:00", 
        2: "12:00-15:00",
        3: "15:00-15:45",
        4: "9:30-15:45"
    }
    
    params = best_trial.params
    
    print(f"   Entry Window: {entry_window_names[params['entry_window']]}")
    print(f"   Stretch Threshold: {params['stretch_threshold']:.3f} ({params['stretch_threshold']*100:.1f}%)")
    
    # Calculate actual reclaim threshold
    reclaim_actual = params['stretch_threshold'] * params['reclaim_percentage']
    print(f"   Reclaim Percentage: {params['reclaim_percentage']*100:.0f}% of stretch")
    print(f"   Reclaim Threshold: {reclaim_actual:.4f} ({reclaim_actual*100:.2f}%)")
    
    print(f"   Cooldown Period: {params['cooldown_period_seconds']} seconds")
    print(f"   Take Profit: {params['take_profit_percent']}%")
    print(f"   Stop Loss: {params['stop_loss_percent']}%")
    print(f"   Max Trade Duration: {params['max_trade_duration_seconds']} seconds")
    print(f"   Strikes Depth: {params['strikes_depth']}")
    print(f"   Option Selection: {params['option_selection_mode'].upper()}")
    
    # Study statistics
    print(f"\n📊 STUDY STATISTICS:")
    print(f"   Total Trials: {len(study.trials)}")
    print(f"   Completed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"   Failed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    # Count duplicates by source
    memory_dupes = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED 
                       and t.user_attrs.get('duplicate_source') == 'memory'])
    db_dupes = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED 
                   and t.user_attrs.get('duplicate_source') == 'database'])
    print(f"   Duplicate Trials: {memory_dupes} (memory) + {db_dupes} (database)")
    
    # Top 5 trials
    print(f"\n🏅 TOP 5 TRIALS:")
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else -float('inf'), reverse=True)
    
    for i, trial in enumerate(sorted_trials[:5]):
        if trial.value is not None:
            entry_window = entry_window_names[trial.params['entry_window']]
            print(f"   #{i+1}: {trial.value:.2f}% (Trial {trial.number}) - Window: {entry_window}, Stretch: {trial.params['stretch_threshold']:.3f}")

def run_best_trial_detailed(study):
    """
    Run the best trial with detailed output for analysis.
    
    Args:
        study: Completed Optuna study
    """
    print("\n" + "=" * 60)
    print("🔍 DETAILED ANALYSIS OF BEST PARAMETERS")
    print("=" * 60)
    
    # Get best parameters
    best_params_raw = study.best_trial.params
    
    # Create full parameter set
    base_params = initialize_parameters()
    
    # Apply best parameters
    best_params = copy.deepcopy(base_params)
    
    # Entry window
    start_time, end_time = ENTRY_WINDOWS[best_params_raw['entry_window']]
    best_params['entry_start_time'] = start_time
    best_params['entry_end_time'] = end_time
    
    # Other parameters
    best_params['stretch_threshold'] = best_params_raw['stretch_threshold']
    best_params['reclaim_threshold'] = best_params_raw['stretch_threshold'] * best_params_raw['reclaim_percentage']
    best_params['cooldown_period_seconds'] = best_params_raw['cooldown_period_seconds']
    best_params['take_profit_percent'] = best_params_raw['take_profit_percent']
    best_params['stop_loss_percent'] = best_params_raw['stop_loss_percent']
    best_params['max_trade_duration_seconds'] = best_params_raw['max_trade_duration_seconds']
    best_params['strikes_depth'] = best_params_raw['strikes_depth']
    best_params['option_selection_mode'] = best_params_raw['option_selection_mode']
    
    # Get stored metrics from the best trial (no re-run needed)
    best_trial = study.best_trial
    print("Retrieving stored metrics from optimization...")
    
    # Get all stored metrics dynamically
    metrics = dict(best_trial.user_attrs)
    # Add the objective value
    metrics['return_on_risk_percent'] = best_trial.value
    
    # Create a mock detailed_results for compatibility
    detailed_results = {'metrics': metrics}
    
    print(f"\n📈 DETAILED PERFORMANCE METRICS:")
    for metric_name, metric_value in metrics.items():
        if metric_name not in ['error', 'duplicate', 'duplicate_source']:  # Skip non-metric attributes
            formatted_name = metric_name.replace('_', ' ').title()
            # Format the value appropriately
            if metric_value == float('inf'):
                print(f"   {formatted_name}: ∞")
            elif metric_value == float('-inf'):
                print(f"   {formatted_name}: -∞")
            elif isinstance(metric_value, float):
                # Special formatting for percentages and dollar amounts
                if 'percent' in metric_name or 'rate' in metric_name:
                    print(f"   {formatted_name}: {metric_value:.2f}%")
                elif 'trade' in metric_name and 'avg' in metric_name:
                    print(f"   {formatted_name}: ${metric_value:.2f}")
                elif metric_name == 'expectancy':
                    print(f"   {formatted_name}: ${metric_value:.2f} per trade")
                else:
                    print(f"   {formatted_name}: {metric_value:.2f}")
            else:
                print(f"   {formatted_name}: {metric_value}")
    
    return best_params, detailed_results

if __name__ == "__main__":
    print("🤖 VWAP Bounce Strategy - Smart Grid Search")
    print("=" * 50)
    
    # Run optimization
    study = run_optimization(n_trials=N_TRIALS, study_name=STUDY_NAME, max_attempts=MAX_ATTEMPT_LIMIT)
    
    # Print results
    print_optimization_results(study)
    
    # Run detailed analysis of best parameters
    best_params, detailed_results = run_best_trial_detailed(study)
    
    # No need to save study as it's already persisted in PostgreSQL
    print("\n💾 Study saved to PostgreSQL database")
    print(f"💾 Parameter combinations stored in seen_combos table for future runs")
    
    print("\n✅ Optimization complete!") 