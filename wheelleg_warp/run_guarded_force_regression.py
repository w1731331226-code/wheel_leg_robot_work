"""Same registered33 force-prior cases with a shared experimental nominal guard."""
from pathlib import Path
import execution_history_evaluation as evaluation
import run_fixed_force_regression as driver
import joint_state_guard as guard

OUT=Path(__file__).resolve().parent/'results/paper_recovery_20261004/joint_reference_candidate_v1/guarded_force_dynamic_regression_v1'


def run():
    old=evaluation.make_env;previous=driver.OUT
    def make_env(cases,arm,normalization,directory):
        return guard.instrument(lambda:old(cases,arm,normalization,directory),cases,directory)
    evaluation.make_env=make_env;driver.OUT=OUT
    try:driver.run()
    finally:evaluation.make_env=old;driver.OUT=previous


if __name__=='__main__':run()
