"""Only the five unexecuted candidate episodes; never reruns saved oldB0."""
import json
import time
import mujoco
from run_joint_reference_zero_pair import OUT, evaluate, witness, ROOT, sha, atomic_json


def run():
    assert not any((OUT/n).exists() for n in ('continuation_started.json','continuation_completion.json','continuation_failure.json'))
    p=json.loads((OUT/'continuation_proposal.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items())
    old=json.loads((OUT/'old_B0/result.json').read_text())
    checked=witness.check_files(OUT/'old_B0',p['cases'],{r['seed']:r for r in old['runs']})
    failure=json.loads((OUT/'failure.json').read_text());calls=[];fd=mujoco.mjd_transitionFD
    def counted(*args,**kwargs):
        calls.append(float(args[2]));assert len(calls)+len(failure['FD_calls'])<=p['total_FD_budget']
        return fd(*args,**kwargs)
    mujoco.mjd_transitionFD=counted;start=time.perf_counter()
    atomic_json(OUT/'continuation_started.json',dict(proposal_sha256=sha(OUT/'continuation_proposal.json'),new_evaluations=5,old_evaluations_replayed=0))
    try:
        directory=OUT/'joint_zero';directory.mkdir(exist_ok=False)
        new,new_checked=evaluate(p['cases'],'joint_zero',directory,p['total_graph_budget']-old['actual_graph_world_steps'])
        differences=[{k:[a[k],b[k]] for k in ('success','reason','physical_safety_passed','design_joint_passed','terrain_passed','terrain_exit_passed') if a[k]!=b[k]} for a,b in zip(old['runs'],new['runs'])]
        passed=not any(differences) and old['physical']==old['design']==new['physical']==new['design']==5
        atomic_json(OUT/'continuation_completion.json',dict(verified=True,round=304,evaluations=10,new_evaluations=5,old_evaluations_replayed=0,
            label_differences=differences,engineering_pair_gate_passed=passed,old_checked=checked,new_checked=new_checked,
            cumulative_first_episode_world_steps=old['first_episode_world_steps']+new['first_episode_world_steps'],
            cumulative_actual_graph_world_steps=old['actual_graph_world_steps']+new['actual_graph_world_steps'],
            old_constructor_FD_calls=failure['FD_calls'],new_constructor_FD_calls=calls,
            original_metadata_failure_preserved=True,wall_seconds_new_arm=time.perf_counter()-start,
            new_training_samples=0,formal_PPO_admitted=False,method_advantage_established=False))
        print('DONE304 continuation5 only; zeroengineering gate',passed,flush=True)
    except BaseException as error:
        atomic_json(OUT/'continuation_failure.json',dict(error=repr(error),new_FD_calls=calls,implicit_retry=False))
        raise
    finally:
        mujoco.mjd_transitionFD=fd


if __name__=='__main__':
    run()
