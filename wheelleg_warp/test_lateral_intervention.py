"""Public matched B1 lateral-feedback/sign-control intervention, no learning."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,sys
import numpy as np
import mujoco_warp as mjw

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from native.terrain import HeightTerrainScenario
from train_height_comparison import protocol,raw_env,summary
from training_contract import digest
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'

def actions(obs,y,direction,arm,candidate,gain):
    scale=1+candidate['roll_gain']*np.minimum(abs(obs[:,0])/np.deg2rad(2),1)
    torque=-scale*(candidate['kp']*obs[:,2]+candidate['kd']*obs[:,5])
    moving=abs(obs[:,9])>.05
    torque=torque-arm*direction*gain*y*moving
    result=np.zeros((len(y),3),dtype=np.float32);result[:,2]=np.clip(torque/.3,-1,1)
    assert np.isfinite(result).all() and np.max(abs(result))<=1
    return result

def run(out):
    protocol(P);candidate=json.loads((P/'classical_selection.json').read_text())['selected']['candidate']
    gain=(.4+candidate['kp'])*np.deg2rad(3)/.12
    specs=[];cases=[]
    for arm in [0,1,-1]:
        for direction in [1,-1]:
            for offset in [-.12,0.,.12]:
                for repeat in range(2):
                    specs.append(dict(arm=arm,direction=direction,offset_m=offset,repeat=repeat))
                    scene=HeightTerrainScenario(speed=direction*.7,mass=7.,height_l=0.,height_r=0.,center=1.8,
                        offset=0.,mu_l=.8,mu_r=.8,drive_difference=0.,delay_ms=0.,terrain='step',step_height_m=.01,
                        terrain_seed=4002000,relative_attitude=True,stand_height_m=.3)
                    cases.append(dict(seed=4002000+len(cases),scenario=asdict(scene)))
    arms=np.asarray([s['arm'] for s in specs]);directions=np.asarray([s['direction'] for s in specs]);offsets=np.asarray([s['offset_m'] for s in specs])
    probe=np.zeros((3,38));probe[:,9]=.7;pred=actions(probe,np.array([-.12,0,.12]),np.ones(3),np.ones(3),candidate,gain)
    assert pred[0,2]>0 and pred[1,2]==0 and pred[2,2]<0
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'preregistration.json',dict(cases=cases,conditions=specs,candidate=candidate,gain_Nm_per_m=gain,
        gain_rule='(nominalyawKp0.4+B1yawKp)*3deg/maxoffset0.12; no outcome-based gain search',
        formula='torque=B1unclippedPD-arm*sign(cruise)*gain*ideal_public_odometry_y when abs(command)>0.05; total clipped to original0.3Nm',
        arms={'0':'unchangedB1','1':'correct lateral feedback','-1':'wrong-sign equal-gain control'},
        new_information='Ideal simulated lateral odometry only; no terrain/friction/mass truth',
        training_updates=0,old_gate_or_final_replayed=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))
    raw=raw_env(cases,'diff3');pending=set(range(len(cases)));rows=[None]*len(cases);traces=[[] for _ in cases]
    try:
        obs=raw.reset();q=raw.data.qpos.numpy();q[:,1]=offsets;raw.data.qpos.assign(q);mjw.forward(raw.model,raw.data);ids=raw.ids.numpy()
        deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            position=raw.data.qpos.numpy();action=actions(obs,position[:,1],directions,arms,candidate,gain)
            obs,_,done,infos=raw.step(action);position=raw.data.qpos.numpy();stopped=raw.stopped_q.numpy();state=raw.state.numpy()
            for w in list(pending):
                info=infos[w] if done[w] else None;q=stopped[w] if info else position[w]
                time=info['duration_s'] if info else float(state[w,0]*.0005)
                traces[w].append([time,float(q[0]),float(q[1]),float(action[w,2])])
                if not info:continue
                mean_fk=float(np.mean([sim.fk_joints(float(q[ids[2*j]]),float(q[ids[2*j+1]]))['leg_len'] for j in range(2)]))
                rows[w]=dict(**cases[w],**{k:v for k,v in info.items() if k!='terminal_observation'},
                    final_mean_fk_leg_m=mean_fk,condition=specs[w],final_cross_track_m=float(q[1]))
                pending.remove(w)
            if not pending:break
        assert not pending
        groups={str(arm):summary([r for r in rows if r['condition']['arm']==arm]) for arm in [0,1,-1]}
        offsets_trace=np.cumsum([0]+[len(t) for t in traces])
        np.savez_compressed(out/'trace.npz',columns=np.array(['time_s','body_x_m','body_y_m','bounded_action_wheel_diff']),offsets=offsets_trace,trace=np.concatenate([np.asarray(t) for t in traces]))
        atomic_json(out/'result.json',dict(groups=groups,runs=rows,gain_Nm_per_m=gain,
            additional_public_diagnostic_episodes=len(rows),training_updates=0,old_gate_or_final_replayed=False,
            conclusion_scope='Constructed ideal-odometry and fixed-law intervention; not general learned-policy performance or a novelty proof',
            protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))
        for arm,stats in groups.items():print('ARM',arm,stats,'physical/design',sum(r['physical_safety_passed'] for r in rows if str(r['condition']['arm'])==arm),sum(r['design_joint_passed'] for r in rows if str(r['condition']['arm'])==arm),flush=True)
    finally:raw.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
