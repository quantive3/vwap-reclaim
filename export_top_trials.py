import optuna
import pandas as pd
import os
from optuna.trial import TrialState
from optuna.storages import RDBStorage

# Import credentials from secret file
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

def export_top_trials_to_csv(study_name="vwap_bounce_optimization_v2", top_n=250, output_file="top_trials.csv"):
    """
    Export the top N trials from an Optuna study to a CSV file.
    
    Args:
        study_name (str): Name of the study in the database
        top_n (int): Number of top trials to export
        output_file (str): Path to the output CSV file
    """
    print(f"🔍 Loading study '{study_name}' from PostgreSQL...")
    
    # Create storage object
    storage = RDBStorage(url=POSTGRES_URL)
    
    # Load the study
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    print(f"✅ Study loaded with {len(study.trials)} total trials")
    
    # Get completed trials
    completed_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    print(f"✅ Found {len(completed_trials)} completed trials")
    
    # Sort by objective value (descending)
    sorted_trials = sorted(
        completed_trials, 
        key=lambda t: t.value if t.value is not None else float('-inf'),
        reverse=True
    )
    
    # Take top N trials
    top_trials = sorted_trials[:top_n]
    actual_count = len(top_trials)
    print(f"✅ Selected top {actual_count} trials")
    
    # Collect all possible parameter names and user attribute names
    all_param_names = set()
    all_user_attr_names = set()
    
    for trial in top_trials:
        all_param_names.update(trial.params.keys())
        all_user_attr_names.update(trial.user_attrs.keys())
    
    # Create a list to store trial data
    trials_data = []
    
    # Extract data from each trial
    for i, trial in enumerate(top_trials):
        # Start with basic trial info
        trial_data = {
            'rank': i + 1,
            'trial_number': trial.number,
            'objective_value': trial.value,
        }
        
        # Add all parameters
        for param_name in all_param_names:
            trial_data[f'param_{param_name}'] = trial.params.get(param_name, None)
        
        # Add all user attributes
        for attr_name in all_user_attr_names:
            trial_data[f'attr_{attr_name}'] = trial.user_attrs.get(attr_name, None)
        
        trials_data.append(trial_data)
    
    # Create DataFrame
    df = pd.DataFrame(trials_data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Exported {len(trials_data)} trials to {output_file}")
    
    # Print column names for verification
    columns = list(df.columns)
    print(f"✅ CSV contains {len(columns)} columns:")
    
    # Group columns by type for better readability
    basic_cols = [col for col in columns if not (col.startswith('param_') or col.startswith('attr_'))]
    param_cols = [col for col in columns if col.startswith('param_')]
    attr_cols = [col for col in columns if col.startswith('attr_')]
    
    print(f"   Basic columns: {', '.join(basic_cols)}")
    print(f"   Parameter columns: {', '.join(param_cols)}")
    print(f"   Attribute columns: {', '.join(attr_cols)}")

if __name__ == "__main__":
    export_top_trials_to_csv() 