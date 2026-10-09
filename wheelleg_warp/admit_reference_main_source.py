"""Bounded100-world initial actuation audit; no physical graph or learning."""
import json
import numpy as np
import mujoco
import warp as wp
from reference_learning_runtime import make_env
from reference_learning_engineering import agent,equal
from qualify_joint_reference_source import prepare
import joint_reference_adapter as adapter
from train_reference_prequalification import OUT
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json


def run():
    out=OUT/'initial_distribution';q=json.loads((out/'proposal.json').read_text())
    assert not any((out/n).exists() for n in ('started.json','completion.json','failure.json'))
    assert all(sha(ROOT/f)==h for f,h in q['source_sha256'].items())
    p=json.loads((OUT/'proposal.json').read_text());seed=p['seeds'][0];config={**p,'engineering_seed':seed}
    draws=np.random.default_rng(q['random_seed']).normal(0,.25,(64,100,6)).clip(-1,1).astype(np.float32)
    fd,graph=mujoco.mjd_transitionFD,wp.capture_launch;calls=[];queries=0;shared=None;world=None;reports={};zero=None
    def counted(*args,**kwargs):
        calls.append(float(args[2]));assert len(calls)<=q['constructor_FD_budget']
        return fd(*args,**kwargs)
    def reject(*args,**kwargs):raise AssertionError('Static source audit forbids physical graphs')
    mujoco.mjd_transitionFD=counted;wp.capture_launch=reject
    atomic_json(out/'started.json',dict(proposal_sha256=sha(out/'proposal.json')))
    try:
        for arm in p['arms']:
            directory=out/arm;directory.mkdir(exist_ok=False)
            curriculum,raw,history,norm=make_env(p,arm,seed);model,shared=agent(config,norm,shared)
            try:
                obs=norm.reset();assert obs.shape==(100,481)
                state=[b.numpy().copy() for b in (raw.q0,raw.param,raw.k['gains'],raw.k['reference'])]
                if world is None:world=state
                else:equal(world,state)
                stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
                outputs=[];baseline=None
                for draw in range(-1,64):
                    raw.reset();raw.diag.zero_()
                    actions=np.zeros((100,3 if arm=='M_ref3' else 6),np.float32)
                    if draw>=0:actions=draws[draw] if arm=='U_ref6' else draws[draw][:,[0,1,4]]
                    canonical,motor=adapter.decode(actions,'M3' if arm=='M_ref3' else 'U6',100)
                    raw._joint_requested.assign(canonical);raw.targets.assign(motor)
                    args=prepare(raw);wp.launch(adapter.candidate.control_physical_nominal,100,args,block_dim=32);queries+=100
                    assert queries<=q['controller_query_budget']
                    ctrl=raw.data.ctrl.numpy().copy();diag=raw.diag.numpy().copy();memory=raw.k['state'].numpy().copy()
                    assert all(np.isfinite(x).all() for x in (ctrl,diag,memory)) and not diag[:,14].any()
                    np.testing.assert_array_equal(raw.data.time.numpy(),0)
                    if draw==-1:
                        baseline=(ctrl,diag,memory)
                        np.savez_compressed(directory/'zero.npz',ctrl=ctrl,diag=diag,memory=memory)
                    else:outputs.append((actions.copy(),canonical.copy(),raw._joint_buffers[1].numpy(),raw._joint_buffers[4].numpy(),memory[:,16:22],ctrl,diag[:,12]))
                equal(stats,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count))
                if zero is None:zero=baseline
                else:equal(zero,baseline)
                names=('actions','canonical','effective','filtered_reference','filtered_motor','ctrl','lambda_values')
                values={k:np.stack(v) for k,v in zip(names,zip(*outputs))}
                np.savez_compressed(directory/'actuation.npz',**values,zero_ctrl=baseline[0],q=state[0],param=state[1],gains=state[2],reference=state[3])
                delta=(values['ctrl']-baseline[0]).reshape(-1,6);lam=values['lambda_values']
                reports[arm]=dict(verified=True,worlds=100,draws_per_world=64,delta_ctrl_RMS_Nm=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
                    delta_ctrl_covariance_Nm2=np.cov(delta,rowvar=False).tolist(),mean_lambda=float(lam.mean()),zero_lambda_fraction=float((lam==0).mean()),
                    parameters=sum(t.numel() for t in model.policy.parameters()),policy_device=str(model.device),physics_device=str(raw.data.qpos.device),actuation_sha256=sha(directory/'actuation.npz'))
                atomic_json(directory/'report.json',reports[arm])
            finally:norm.close()
        assert queries==13000 and all(sha(ROOT/f)==h for f,h in q['source_sha256'].items())
        atomic_json(out/'completion.json',dict(verified=True,reports=reports,controller_queries=queries,constructor_FD_calls=calls,physics_graph_launches=0,learning_samples=0,
            exact_exploration_matching=False,scope='seed32031 stage1 startup first0.5ms only; notallseeds/stages or40substeps/initialrollout matching;no effect-based scale tuning'))
    except BaseException as error:
        atomic_json(out/'failure.json',dict(error=repr(error),queries=queries,FD_calls=calls,implicit_retry=False));raise
    finally:mujoco.mjd_transitionFD=fd;wp.capture_launch=graph


if __name__=='__main__':run()
