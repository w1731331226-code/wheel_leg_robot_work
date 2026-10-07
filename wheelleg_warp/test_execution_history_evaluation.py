"""Finite dynamic source qualification;not scientific model evaluation."""
import json
import sys
from pathlib import Path
import numpy as np
import warp as wp
from stable_baselines3 import PPO
from review_yaw_sector import ROOT,sha
import execution_history_evaluation as evaluation
from test_nom_yaw_filter_probe import control_args
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'


def classical_mapping(case):
    raw3=evaluation.parking.phase.instrument([case],'diff3')
    raw6=evaluation.parking.phase.instrument([case],'virtual6')
    count=0
    try:
        for speed in (0.,70.):
            for request in (-.8,.0,.8):
                outputs=[]
                for raw in (raw3,raw6):
                    raw.reset();raw.command.fill_(.7)
                    v=raw.data.qvel.numpy();ids=raw.ids.numpy();v[:,ids[8:10]]=speed;raw.data.qvel.assign(v)
                    target=np.zeros((1,raw.action_dim),np.float32)
                    if raw.action_dim==3:target[:,2]=request
                    else:target[:,4]=request;target[:,5]=-request
                    raw.targets.assign(target)
                    evaluation.parking.phase.prepare_step(raw,0,'phase_support')
                    args=control_args(raw);args[12]=raw._role_buffers[0]
                    wp.launch(evaluation.parking.phase.roles.experimental.control_physical_nominal,1,args,block_dim=32)
                    outputs.append((raw.data.ctrl.numpy(),raw.diag.numpy()))
                for a,b in zip(*outputs):np.testing.assert_array_equal(a,b)
                count+=1
        return dict(static_nonzero_clipped_queries=count,ctrl_and_diag_exact=True,physics_steps=0)
    finally:raw3.close();raw6.close()


class StopAfterTwo:
    def __init__(self,model):self.model=model;self.calls=0
    def __getattr__(self,name):return getattr(self.model,name)
    def predict(self,*args,**kwargs):
        self.calls+=1
        if self.calls==3:raise RuntimeError('deliberate-source-prefix-stop')
        return self.model.predict(*args,**kwargs)


def run():
    continuation=sys.argv[1:] == ['continue']
    assert (OUT/'evaluation_unit_registration.json').exists()==continuation
    p=json.loads((OUT/'study_proposal.json').read_text());base=OUT/'evaluation_unit';base.mkdir(exist_ok=continuation)
    jobs=[dict(arm='H1',case=p['controlled'][0],model=True),
          dict(arm='B0',case=next(c for c in p['legacy'] if c['scenario']['solver_iterations']==50),model=False),
          dict(arm='B1-route',case=p['controlled'][2],model=False)]
    if continuation:
        registered=json.loads((OUT/'evaluation_unit_registration.json').read_text());assert registered['jobs']==jobs
        atomic_json(OUT/'evaluation_unit_explicit_continuation.json',dict(reason='Replay exact original1world CUDAcallshape;reuse completedfirstepisode,no new budget or tolerance.',
            prior_failure_sha256=sha(OUT/'evaluation_attempt0_batched_replay/failure.json'),source_sha256=sha(__file__)))
    else:atomic_json(OUT/'evaluation_unit_registration.json',dict(proposal_sha256=sha(OUT/'study_proposal.json'),
        jobs=jobs,complete_episodes=3,deliberate_prefix_actor_calls=2,training_updates=0,
        scope='Engineeringmodel/sourceinterfaceonly;no scientific scoreselection or budgetconsumption. Threefirstepisodes+80step deliberateprefix;static6queries0integration.'))
    prefix=OUT/'engineering/H1/step_12000';model=PPO.load(prefix.with_suffix('.zip'),device='cuda');reports=[]
    try:
        mapping=classical_mapping(p['controlled'][0])
        for i,job in enumerate(jobs):
            directory=base/f'complete_{i}';directory.mkdir(exist_ok=continuation)
            if (directory/'result.json').exists():
                result=json.loads((directory/'result.json').read_text())
                checked=evaluation.recorder.prior.prior.check_files(directory,[job['case']],{r['seed']:r for r in result['runs']})
                checked.pop('original_label_differences');assert evaluation.recorder.logs(directory,[job['case']])==checked['physics_steps']
            else:
                result,checked=evaluation.evaluate([job['case']],job['arm'],model if job['model'] else None,
                    prefix.with_suffix('.pkl') if job['model'] else None,directory,p['classical'].get(job['arm']))
            row=result['runs'][0];trace=np.load(directory/row['actor_trace']['path'])['trace']
            assert trace.shape==(int(np.ceil(row['physical_steps']/40)),968)
            if job['model']:
                replay=np.concatenate([model.predict(row[481:962][None],deterministic=True)[0] for row in trace])
                np.testing.assert_array_equal(replay,trace[:,962:])
            else:np.testing.assert_array_equal(trace[:,:481],trace[:,481:962])
            reports.append(dict(arm=job['arm'],case=job['case']['seed'],physics_steps=row['physical_steps'],
                actor_rows=len(trace),six_streams_plus_actor_checked=True,result_sha256=sha(directory/'result.json'),checked=checked))
            print('UNIT COMPLETE',job['arm'],row['physical_steps'],flush=True)
        directory=base/'deliberate_prefix';directory.mkdir()
        try:evaluation.evaluate([p['controlled'][0]],'H1',StopAfterTwo(model),prefix.with_suffix('.pkl'),directory)
        except RuntimeError as e:assert str(e)=='deliberate-source-prefix-stop'
        else:raise AssertionError('Deliberate prefix did not stop')
        actor=np.load(directory/'interrupted_actor_0.npz')['trace'];assert actor.shape==(2,968)
        assert (directory/'interrupted_history.npz').exists() and (directory/'interrupted_command.npz').exists()
        files={str(f.relative_to(OUT)):sha(f) for f in base.rglob('*') if f.is_file()}
        atomic_json(OUT/'evaluation_admission.json',dict(verified=True,round=234,evaluation_source_admitted=True,
            proposal_sha256=sha(OUT/'study_proposal.json'),registration_sha256=sha(OUT/'evaluation_unit_registration.json'),
            source_sha256={str(Path(m.__file__).resolve().relative_to(ROOT)):sha(m.__file__) for m in
                (evaluation,evaluation.collector,evaluation.parking)},mapping=mapping,reports=reports,
            prefix_actor_rows=2,prefix_physics_steps=80,artifact_sha256=files,scientific_evaluations_consumed=0,
            training_updates=0,limits='Only3sourceepisodes andoldengineeringmodel;notlearnedmethodperformance,newindependenttest or fullGPUtrajectoryidentity. Staticclassicalctrl/diag equality not closed-loopbitwiseclaim.'))
        print('PASS234 dynamicdense/legacy50/Actorlog/classicmapping/prefix;0scientificconsumed',flush=True)
    except BaseException as e:
        atomic_json(OUT/'evaluation_unit_failure.json',dict(error=repr(e),reports=reports,implicit_retry=False));raise


if __name__=='__main__':run()
