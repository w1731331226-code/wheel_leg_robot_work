"""One bounded actualruntime/initial-actuation admission; no learning."""
import json
from pathlib import Path
import numpy as np
import mujoco
import warp as wp
from reference_learning_engineering import OUT,make_env,agent,equal
from qualify_joint_reference_source import prepare
import joint_reference_adapter as adapter
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    assert not any((OUT/n).exists() for n in ('runtime_source_started.json','source_admission.json','runtime_source_failure.json'))
    config=json.loads((OUT/'runtime_config.json').read_text());p=json.loads((OUT/'runtime_source_proposal.json').read_text())
    assert p['runtime_config_sha256']==sha(OUT/'runtime_config.json')
    r=np.random.default_rng(p['random_seed']).normal(0,.25,(64,10,6)).clip(-1,1).astype(np.float32)
    fd,graph=mujoco.mjd_transitionFD,wp.capture_launch;fd_calls=[];queries=0;reports={};shared=None;world=None
    def counted(*args,**kwargs):
        fd_calls.append(float(args[2]));assert len(fd_calls)<=25
        return fd(*args,**kwargs)
    def reject(*args,**kwargs):raise AssertionError('No physicsgraph source check')
    mujoco.mjd_transitionFD,wp.capture_launch=counted,reject
    atomic_json(OUT/'runtime_source_started.json',dict(proposal_sha256=sha(OUT/'runtime_source_proposal.json'),runner_sha256=sha(__file__),budget_queries=1300))
    try:
        for arm in p['arms']:
            directory=OUT/f'source_{arm}';directory.mkdir(exist_ok=False)
            raw,history,norm=make_env(config,arm,directory);model,shared=agent(config,norm,shared)
            try:
                obs=norm.reset();assert obs.shape==(10,481) and raw.data.qpos.device.is_cuda
                current=[b.numpy().copy() for b in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
                if world is None:world=current
                else:equal(world,current)
                norm_before=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
                outputs=[];zero=None
                for index in range(-1,64):
                    raw.reset();raw.diag.zero_()
                    actions=np.zeros((10,3 if arm=='M_ref3' else 6),np.float32)
                    if index>=0:
                        actions=r[index] if arm=='U_ref6' else r[index][:,[0,1,4]]
                    canonical,motor=adapter.decode(actions,'M3' if arm=='M_ref3' else 'U6',10)
                    raw._joint_requested.assign(canonical);raw.targets.assign(motor)
                    before=raw.k['state'].numpy().copy();args=prepare(raw)
                    queries+=10;wp.launch(adapter.candidate.control_physical_nominal,10,args,block_dim=32)
                    ctrl=raw.data.ctrl.numpy().copy();diag=raw.diag.numpy().copy();after=raw.k['state'].numpy().copy()
                    assert np.isfinite(ctrl).all() and np.isfinite(diag).all() and np.isfinite(after).all()
                    assert not diag[:,14].any()
                    np.testing.assert_array_equal(raw.data.time.numpy(),0)
                    if index==-1:
                        zero=(ctrl,diag,after)
                        np.savez_compressed(directory/'zero.npz',ctrl=ctrl,diag=diag,memory_before=before,memory_after=after)
                    else:outputs.append((actions.copy(),canonical.copy(),ctrl,diag[:,12].copy()))
                equal(norm_before,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count))
                actions,canonical,control,lambdas=map(np.stack,zip(*outputs))
                np.savez_compressed(directory/'initial_actuation.npz',actions=actions,canonical=canonical,ctrl=control,lambda_values=lambdas,
                    zero_ctrl=zero[0],world_q=current[0],param=current[1],gains=current[2],reference=current[3])
                delta=(control-zero[0]).reshape(-1,6)
                report=dict(verified=True,arm=arm,queries=650,normalized_samples=64,firstsubstep_only=True,
                    actual_ctrl_RMS_Nm=np.sqrt(np.mean(control**2,axis=(0,1))).tolist(),delta_ctrl_RMS_Nm=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                    delta_ctrl_covariance_Nm2=np.cov(delta,rowvar=False).tolist(),mean_lambda=float(lambdas.mean()),
                    zero_lambda_fraction=float((lambdas==0).mean()),policy_parameter_count=sum(t.numel() for t in model.policy.parameters()),
                    policy_device=str(model.device),physics_device=str(raw.data.qpos.device),same_nominal_world=True,
                    normalized_RMS_unchanged_by_queries=True,actuation_sha256=sha(directory/'initial_actuation.npz'))
                atomic_json(directory/'report.json',report);reports[arm]=report
                print('SOURCE317',arm,'queries',queries,flush=True)
            finally:norm.close()
        assert queries==1300
        with np.load(OUT/'source_M_ref3/zero.npz') as a,np.load(OUT/'source_U_ref6/zero.npz') as b:
            for n in ('ctrl','diag','memory_after'):np.testing.assert_array_equal(a[n],b[n])
        sources=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1/study_contract.json').read_text())['source_sha256']
        for name in ('reference_learning_engineering.py','test_reference_learning_policy.py','admit_reference_learning_source.py','joint_reference_adapter.py','joint_reference_control.py','joint_reference_prepare.py','qualify_joint_reference_source.py'):
            sources['wheelleg_warp/'+name]=sha(ROOT/'wheelleg_warp'/name)
        atomic_json(OUT/'source_admission.json',dict(verified=True,round=317,runtime_config_sha256=sha(OUT/'runtime_config.json'),
            runtime_proposal_sha256=sha(OUT/'runtime_source_proposal.json'),source_sha256=sources,reports=reports,
            actual_constructor_FD_calls=fd_calls,controller_queries=queries,physics_graph_launches=0,new_policy_samples=0,
            physical_exploration_exactly_matched=False,matched_zero_output_function=True,
            shortengineering_steps_admitted=8000,formal_PPO_admitted=False,
            limits='Sharednominal/zero function andrecorded startup-firstsubstep actualdistributions only;notcomplete initialrollout distributionequality orlearnedadvantage. Only2x4000 engineering lifecycle mayrun,notformal5/sciencepromotion.'))
        print('PASS317 short8000 sourceadmitted;0learn/physicsgraph',flush=True)
    except BaseException as error:
        atomic_json(OUT/'runtime_source_failure.json',dict(error=repr(error),controller_queries=queries,FD_calls=fd_calls,reports=reports,implicit_retry=False))
        raise
    finally:mujoco.mjd_transitionFD,wp.capture_launch=fd,graph


if __name__=='__main__':run()
