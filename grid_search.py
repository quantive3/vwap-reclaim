# === RANDOM SEARCH FOR VWAP BOUNCE STRATEGY ===
import csv
import os
import random
import numpy as np
from datetime import time
from main import run_backtest, DataLoader, initialize_parameters, initialize_issue_tracker, API_KEY, CACHE_DIR

# ── pre-define the only windows you'd ever trade ──────────────
ENTRY_WINDOWS = [
    (time(9, 30), time(11, 0)),   # 09:30-11:00
    (time(9, 30), time(12, 0)),   # 09:30-12:00
    (time(10, 0), time(15, 0)),   # 10:00-15:00
    (time(9, 30), time(15, 30))   # 09:30-15:30
]

def random_combo():
    """
    Generate a random parameter combination with smart relationships.
    """
    stretch  = round(random.uniform(0.001, 0.02), 4)
    reclaim  = round(stretch * random.uniform(0.3, 0.9), 4)      # 30–90 % ratio
    tp, sl   = random.choice([(25, -25), (35, -25), (50, -30), (50, -50), (50, -100)])  # paired TP/SL
    start, end = random.choice(ENTRY_WINDOWS)                    # pick one window

    return {
        "stretch_threshold":        stretch,
        "reclaim_threshold":        reclaim,
        "cooldown_period_seconds":  random.choice([30,45,60,90,120,180,300]),
        "entry_start_time":         start,
        "entry_end_time":           end,
        "take_profit_percent":      tp,
        "stop_loss_percent":        sl,
        "max_trade_duration_seconds": random.choice([60,120,180,240,300,600]),
        "strikes_depth":            random.choice([1,2,3]),
        "option_selection_mode":    random.choice(["itm", "otm"])
    }

def run_random_search(num_iterations=100, seed=None):
    """
    Run a random parameter search over the VWAP bounce strategy.
    Results are saved to a CSV file for analysis.
    
    Args:
        num_iterations (int): Number of random parameter combinations to test
        seed (int, optional): Random seed for reproducible results. If None, uses truly random values.
    """
    
    # Set random seeds for reproducible results
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        print(f"🎯 Using random seed: {seed} (results will be reproducible)")
    else:
        print(f"🎲 Using truly random parameters (results will vary between runs)")
    
    # Get base parameters
    base_params = initialize_parameters()
    
    # Force silent mode and disable debug for random search
    base_params['silent_mode'] = True
    base_params['debug_mode'] = False
    
    # Define CSV output file
    csv_filename = "random_search_results.csv"
    
    # Define CSV headers (parameter names + metric names)
    param_headers = [
        'stretch_threshold',
        'reclaim_threshold', 
        'cooldown_period_seconds',
        'entry_start_time',
        'entry_end_time',
        'take_profit_percent',
        'stop_loss_percent',
        'max_trade_duration_seconds',
        'strikes_depth',
        'option_selection_mode'
    ]
    metric_headers = [
        'total_trades',
        'win_rate', 
        'expectancy',
        'avg_risk_per_trade',
        'return_on_risk_percent',
        'sharpe_ratio'
    ]
    csv_headers = param_headers + metric_headers
    
    # Open CSV file and write header
    print(f"🎲 Starting random search...")
    print(f"🔄 Testing {num_iterations} random parameter combinations")
    print(f"💾 Results will be saved to: {csv_filename}")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        
        # Loop over random parameter combinations
        for i in range(1, num_iterations + 1):
            # Generate random parameter combination
            combo = random_combo()
            
            print(f"🧪 Testing combination {i}/{num_iterations}")
            print(f"   📊 Stretch: {combo['stretch_threshold']:.4f}, Reclaim: {combo['reclaim_threshold']:.4f}")
            print(f"   ⏰ Window: {combo['entry_start_time']}-{combo['entry_end_time']}, TP/SL: {combo['take_profit_percent']}/{combo['stop_loss_percent']}")
            
            # Create parameters for this combination
            params = base_params.copy()
            params.update(combo)
            
            # Initialize fresh issue tracker for each run
            issue_tracker = initialize_issue_tracker(params)
            
            # Create DataLoader with silent mode
            dl = DataLoader(API_KEY, CACHE_DIR, params, debug_mode=False, silent_mode=True)
            
            try:
                # Run backtest
                result = run_backtest(params, dl, issue_tracker)
                
                # Extract metrics
                metrics = result['metrics']
                
                # Prepare row data (parameters + metrics)
                row_data = []
                
                # Add parameter values
                for param_name in param_headers:
                    param_value = combo.get(param_name, '')
                    # Convert time objects to string for CSV
                    if isinstance(param_value, time):
                        param_value = param_value.strftime('%H:%M')
                    row_data.append(param_value)
                
                # Add metric values
                for metric_name in metric_headers:
                    metric_value = metrics.get(metric_name, None)
                    if metric_value is None:
                        row_data.append('')
                    elif metric_value == float('inf'):
                        row_data.append('inf')
                    elif isinstance(metric_value, float):
                        row_data.append(f"{metric_value:.6f}")
                    else:
                        row_data.append(metric_value)
                
                # Write row to CSV
                writer.writerow(row_data)
                
                # Print brief summary
                total_trades = metrics.get('total_trades', 0)
                win_rate = metrics.get('win_rate', None)
                expectancy = metrics.get('expectancy', None)
                
                print(f"   ✅ Completed: {total_trades} trades", end="")
                if win_rate is not None:
                    print(f", {win_rate:.1f}% win rate", end="")
                if expectancy is not None:
                    if expectancy == float('inf'):
                        print(f", ∞ expectancy")
                    else:
                        print(f", ${expectancy:.2f} expectancy")
                else:
                    print()
                
            except Exception as e:
                print(f"   ❌ Error in combination {i}: {str(e)}")
                
                # Write error row
                error_row = []
                for param_name in param_headers:
                    param_value = combo.get(param_name, '')
                    if isinstance(param_value, time):
                        param_value = param_value.strftime('%H:%M')
                    error_row.append(param_value)
                for _ in metric_headers:
                    error_row.append('ERROR')
                
                writer.writerow(error_row)
    
    print(f"✅ Random search completed! Results saved to {csv_filename}")
    print(f"📈 You can now analyze the results using Excel, pandas, or your preferred tool.")



if __name__ == "__main__":
    run_random_search(100)  # Test 100 random combinations by default
    # run_random_search(100, seed=42)  # For reproducible results (testing/debugging)
    # run_random_search(2000)  # For production runs (2000 combinations)
    # run_random_search(2000, seed=123)  # For reproducible production runs 