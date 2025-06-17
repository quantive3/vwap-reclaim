import joblib
import matplotlib.pyplot as plt
from optuna.trial import TrialState
from smart import ValidCountTPESampler

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
plt.savefig("opt_history.png")
print(f"✅ Optimization history saved to opt_history.png ({len(completed)} points)") 