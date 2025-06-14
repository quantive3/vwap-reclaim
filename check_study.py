import joblib

# Load the study file
study = joblib.load('vwap_optimization_study.pkl')

print(f"Study has {len(study.trials)} trials")
print("\nChecking user attributes for each trial:")
print("-" * 50)

# Check what's stored
for trial in study.trials:
    print(f"Trial {trial.number}:")
    print(f"  Value: {trial.value}")
    print(f"  State: {trial.state}")
    print(f"  User attrs: {trial.user_attrs}")
    print() 