import optuna
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from smart import ValidCountTPESampler

# Load the study from SQLite database
study = optuna.load_study(
    study_name="vwap_bounce_optimization", 
    storage="sqlite:///vwap.db"
)

from collections import defaultdict
from optuna.trial import TrialState

# Group only completed trials by their sorted param tuples
param_groups = defaultdict(list)
for trial in study.trials:
    if trial.state != TrialState.COMPLETE:
        continue
    key = tuple(sorted(trial.params.items()))
    param_groups[key].append(trial.number)

# Find any parameter tuples that appear more than once
dupes = {params: nums for params, nums in param_groups.items() if len(nums) > 1}

if dupes:
    print("⚠️ Duplicate parameter combinations detected among COMPLETED trials:")
    for params, nums in dupes.items():
        print(f"  Trials {nums}: {dict(params)}")
else:
    print("✅ No duplicate parameter combos found among completed trials.")

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

# ——— Plot completed trials with sequential X-axis ———
completed = [t for t in study.trials if t.state == TrialState.COMPLETE]

# 1…N where N = number of completed trials
xs = list(range(1, len(completed) + 1))
ys = [t.value for t in completed]

plt.plot(xs, ys, marker="o")
plt.xlabel("Completed Trial #")
plt.ylabel("Return on Risk (%)")
plt.title("Optimization History")
plt.tight_layout()
plt.savefig("db_opt_history.png")
print(f"✅ Optimization history saved to db_opt_history.png ({len(completed)} points)")

# Additional analysis - best trial details
if completed:
    best_trial = study.best_trial
    print("\n" + "=" * 50)
    print("🏆 BEST TRIAL DETAILS:")
    print("=" * 50)
    print(f"Trial Number: {best_trial.number}")
    print(f"Return on Risk: {best_trial.value:.2f}%")
    print("\nParameters:")
    for name, value in best_trial.params.items():
        print(f"  {name}: {value}")
    
    print("\nMetrics:")
    for name, value in best_trial.user_attrs.items():
        if name != 'error':  # Skip error messages
            if isinstance(value, float):
                print(f"  {name}: {value:.2f}")
            else:
                print(f"  {name}: {value}")
else:
    print("\n⚠️ No completed trials found in the study.") 