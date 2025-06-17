"""
Unit test for ValidCountTPESampler to verify it correctly counts only COMPLETE trials.
"""
import optuna
from optuna.trial import TrialState
import unittest
from smart import ValidCountTPESampler

class TestValidCountTPESampler(unittest.TestCase):
    def test_startup_phase_counts_only_complete_trials(self):
        """Test that _is_startup counts only COMPLETE trials, not pruned or failed ones."""
        # Create a sampler that requires 10 valid trials before leaving startup
        sampler = ValidCountTPESampler(n_valid_startup_trials=10, seed=42)
        
        # Create a study with this sampler
        study = optuna.create_study(sampler=sampler)
        
        # Helper to add fake trials with specific states
        def add_trial(state):
            trial = optuna.trial.create_trial(
                state=state,
                value=0.0 if state == TrialState.COMPLETE else None,
                params={},
                distributions={},
                intermediate_values={}
            )
            study.add_trial(trial)
        
        # Add 5 COMPLETE trials - should still be in startup
        for _ in range(5):
            add_trial(TrialState.COMPLETE)
        
        # Add 10 PRUNED trials - should still be in startup
        for _ in range(10):
            add_trial(TrialState.PRUNED)
            
        # Add 5 FAIL trials - should still be in startup
        for _ in range(5):
            add_trial(TrialState.FAIL)
        
        # Verify we're still in startup (only 5 COMPLETE trials so far)
        self.assertTrue(sampler._is_startup(study), 
                       "Should still be in startup with only 5 COMPLETE trials")
        
        # Add 5 more COMPLETE trials to reach 10 total
        for _ in range(5):
            add_trial(TrialState.COMPLETE)
        
        # Verify we're now out of startup (10 COMPLETE trials)
        self.assertFalse(sampler._is_startup(study), 
                        "Should be out of startup with 10 COMPLETE trials")
        
        # Add 5 more PRUNED trials - should still be out of startup
        for _ in range(5):
            add_trial(TrialState.PRUNED)
        
        # Verify we're still out of startup
        self.assertFalse(sampler._is_startup(study), 
                        "Should remain out of startup after adding more PRUNED trials")
        
        # Print summary
        complete_count = len([t for t in study.trials if t.state == TrialState.COMPLETE])
        pruned_count = len([t for t in study.trials if t.state == TrialState.PRUNED])
        failed_count = len([t for t in study.trials if t.state == TrialState.FAIL])
        
        print(f"Test passed with {complete_count} COMPLETE, {pruned_count} PRUNED, {failed_count} FAILED trials")
        print(f"Total trials: {len(study.trials)}")

if __name__ == "__main__":
    unittest.main() 