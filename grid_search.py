# === GRID SEARCH FOR VWAP BOUNCE STRATEGY ===
import csv
import os
from sklearn.model_selection import ParameterGrid
from main import run_backtest, DataLoader, initialize_parameters, initialize_issue_tracker, API_KEY, CACHE_DIR

def run_grid_search():
    """
    Run a parameter grid search over the VWAP bounce strategy.
    Results are saved to a CSV file for analysis.
    """
    
    # === Parameter grid (YOU fill in the lists later) ===
    param_grid = {
        "stretch_threshold": [],      # e.g. [0.002, 0.003, 0.005]
        "reclaim_threshold": [],      # e.g. [0.001, 0.002, 0.003]
        "take_profit_percent": [],    # e.g. [20, 25, 30]
        "stop_loss_percent": [],      # e.g. [-40, -50, -60]
        # …add any other knobs you want to sweep
    }
    
    # Check if parameter grid is empty
    if not any(param_grid.values()):
        print("⚠️ Parameter grid is empty. Please populate the parameter lists before running.")
        print("Example:")
        print('param_grid["stretch_threshold"] = [0.002, 0.003, 0.005]')
        print('param_grid["reclaim_threshold"] = [0.001, 0.002, 0.003]')
        return
    
    # Get base parameters
    base_params = initialize_parameters()
    
    # Force silent mode and disable debug for grid search
    base_params['silent_mode'] = True
    base_params['debug_mode'] = False
    
    # Define CSV output file
    csv_filename = "grid_search_results.csv"
    
    # Define CSV headers (parameter names + metric names)
    param_headers = list(param_grid.keys())
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
    print(f"📊 Starting grid search...")
    print(f"💾 Results will be saved to: {csv_filename}")
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        total_combinations = len(param_combinations)
        
        print(f"🔄 Total parameter combinations to test: {total_combinations}")
        
        # Loop over parameter grid combinations
        for i, combo in enumerate(param_combinations, 1):
            print(f"🧪 Testing combination {i}/{total_combinations}: {combo}")
            
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
                    row_data.append(combo.get(param_name, ''))
                
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
                    error_row.append(combo.get(param_name, ''))
                for _ in metric_headers:
                    error_row.append('ERROR')
                
                writer.writerow(error_row)
    
    print(f"✅ Grid search completed! Results saved to {csv_filename}")
    print(f"📈 You can now analyze the results using Excel, pandas, or your preferred tool.")

if __name__ == "__main__":
    run_grid_search() 