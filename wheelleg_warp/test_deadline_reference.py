"""Public bounded path-heading intervention; reuse the frozen round92 runner."""
from pathlib import Path
import argparse,json
import numpy as np
import test_lateral_intervention as base

def reference(position,direction):
    # Public step front 1.8-.25, minus the 50mm tire radius; no fitted gain.
    remaining=np.maximum(1.8-.25-.05-direction*position[:,0],.05)
    return np.clip(np.arctan2(-direction*position[:,1],remaining),-np.deg2rad(5),np.deg2rad(5))

def run(out):
    saved=(base.actions,base.atomic_json,base.raw_env)
    env=None;pending=set();records=[]
    def factory(cases,mode):
        nonlocal env
        env=saved[2](cases,mode);pending.update(range(len(cases)));records.extend([[] for _ in cases])
        wait=env.step_wait
        def step_wait():
            answer=wait();pending.difference_update(np.flatnonzero(answer[2]));return answer
        env.step_wait=step_wait
        return env
    def actions(obs,y,direction,arm,candidate,unused_gain):
        assert candidate['roll_gain']==0
        position=env.data.qpos.numpy()[:,:2] if env is not None else np.column_stack([np.zeros(len(y)),y])
        desired=arm*reference(position,direction)*(abs(obs[:,9])>.05)
        torque=-(candidate['kp']*obs[:,2]+candidate['kd']*obs[:,5])+(.4+candidate['kp'])*desired
        result=np.zeros((len(y),3),dtype=np.float32);result[:,2]=np.clip(torque/.3,-1,1)
        assert np.isfinite(result).all() and np.max(abs(result))<=1 and np.max(abs(desired))<=np.deg2rad(5)
        if env is not None:
            times=env.state.numpy()[:,0]*.0005
            for w in pending:records[w].append([times[w],*position[w],obs[w,2],obs[w,5],obs[w,9],desired[w],result[w,2]])
        return result
    def write(path,value):
        value=dict(value)
        value.pop('gain_Nm_per_m',None);value.pop('gain_rule',None)
        value['verifier_sha256']=base.digest(__file__);value['harness_sha256']=base.digest(Path(base.__file__))
        if path.name=='preregistration.json':
            value.update(formula='B1unclippedPD+(nominalyawKp0.4+B1kp)*arm*clip(atan2(-sign(cruise)*y,max(1.5-sign(cruise)*x,0.05)),+-5deg) when abs(command)>0.05; total clipped to original0.3Nm',
                derivation='Aim at centerline by public step front minus tire radius (1.8-.25-.05=1.5m); remaining-distance floor is tire radius0.05m; original5deg yaw threshold also bounds reference, not guaranteed actual yaw.',
                arms={'0':'unchangedB1','1':'correct bounded deadline reference','-1':'wrong-sign same-reference control'},
                new_information='Ideal current root x/y odometry plus public step-front task geometry; no private mass/friction/drive/contact truth',
                case_role='Same 36 public diagnostic conditions; new control intervention, not independent generalization test',
                rejection='No gain or reference-bound sweep after outcome; original full task/physical/design/attitude thresholds retained')
        if path.name=='result.json':value['conclusion_scope']='Constructed bounded deadline-reference intervention with ideal odometry/public task geometry; no learned-policy or novelty claim'
        return saved[1](path,value)
    probe=reference(np.array([[0.,-.12],[0.,0.],[0.,.12]]),np.ones(3))
    assert probe[0]>0 and probe[1]==0 and probe[2]<0
    base.actions=actions;base.atomic_json=write;base.raw_env=factory
    try:base.run(out)
    finally:base.actions,base.atomic_json,base.raw_env=saved
    rows=json.loads((out/'result.json').read_text())['runs'];arrays=[np.asarray(r) for r in records]
    for row,a in zip(rows,arrays):assert len(a)==row['episode']['l']
    np.savez_compressed(out/'reference_trace.npz',columns=np.array(['start_s','body_x_m','body_y_m','observed_yaw_rad','observed_yaw_rate_rad_s','observed_command_m_s','desired_yaw_rad','bounded_action_wheel_diff']),
        offsets=np.cumsum([0]+[len(a) for a in arrays]),trace=np.concatenate(arrays))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
