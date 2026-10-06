"""Small independent file/stream check for the frozen diagnostic runner."""
import tempfile
import numpy as np
from pathlib import Path
import run_parking_withdrawal as runner


def run():
    p=runner.verify();assert len(runner.jobs(p))==6
    for condition in p['conditions']:
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp);log=np.zeros((3,30));log[:,:3]=[[1,1,0],[1,2,-.5],[1,3,0]]
            log[:,3]=[0,1,1];log[:,5:11]=.5;log[:,11:17]=.5;log[:,17:23]=np.array([0,.01,.02])[:,None]
            expected=[.01,.02,.03]
            if condition!='original':log[2,4]=1;log[2,11:17]=0;expected[2]=.01
            log[:,23:29]=np.array(expected)[:,None]
            np.savez_compressed(d/'parking.npz',trace=log,columns=runner.probe.COL)
            phase=np.zeros((3,6));phase[:,:3]=log[:,:3];np.savez_compressed(d/'phase.npz',trace=phase)
            full=np.zeros((3,112));full[:,:2]=log[:,:2];full[:,47:53]=log[:,23:29];np.savez_compressed(d/'full.npz',trace=full)
            runner.write(d/'result.json',dict(runs=[dict(physical_steps=3,parking_trace=dict(path='parking.npz',sha256=runner.sha(d/'parking.npz'),rows=3),
                phase_trace=dict(path='phase.npz'),complete_trace=dict(path='full.npz'))]))
            assert runner.parking_logs(d,condition)['parking_trace_files']==1
            full[1,47]+=1;np.savez_compressed(d/'full.npz',trace=full)
            try:runner.parking_logs(d,condition)
            except AssertionError:pass
            else:raise AssertionError('Cross-stream filtered-state mismatch was accepted')
    runner.write(runner.OUT/'round212_runner_unit.json',dict(verified=True,conditions=2,independent_synthetic_samples=6,
        actual_controller_memory_columns47_to52_checked=True,cross_stream_corruption_rejected=True,
        fixed_jobs=6,evaluation_budget=78,new_training_or_evaluations=0,
        runner_source_sha256=runner.sha(runner.__file__),runner_contract_sha256=runner.sha(runner.OUT/'runner_contract.json'),
        test_source_sha256=runner.sha(__file__),limits='Synthetic parser check only;no actual episodes or pair equivalence inferred.'))
    print('PASS frozen78 matrix and independent parking/phase/filter stream parser;0 evaluations',flush=True)


if __name__=='__main__':run()
