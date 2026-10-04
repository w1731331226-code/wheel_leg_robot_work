"""Controlled public lateral translations with original observation and dynamics."""
from pathlib import Path
from dataclasses import asdict
import argparse,json,pickle,sys
import numpy as np
import mujoco_warp as mjw
from stable_baselines3 import PPO

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
from native.terrain import HeightTerrainScenario
from train_height_comparison import protocol,raw_env,b1_action,summary
from training_contract import digest,verify_checkpoint
from dashboard.live_env import atomic_json
import wheelleg_sim as sim

P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'

def run(out):
    p=protocol(P);s=HeightTerrainScenario(speed=.7,mass=7.,height_l=0.,height_r=0.,
        center=1.8,offset=0.,mu_l=.8,mu_r=.8,drive_difference=0.,delay_ms=0.,
        terrain='step',step_height_m=.01,terrain_seed=4001000,relative_attitude=True,stand_height_m=.3)
    offsets=np.array([-.12,0.,.12]);cases=[dict(seed=4001000+i,scenario=asdict(s)) for i in range(3)]
    classic=json.loads((P/'classical_selection.json').read_text())['selected']['candidate']
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'preregistration.json',dict(cases=cases,lateral_offsets_m=offsets.tolist(),
        controls='Original B1, same model/constraints; only initial root lateral position changes',
        observation_tolerance=1e-5,control_tolerance_Nm=1e-5,training=False,
        old_gate_or_final_replayed=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))
    raw=raw_env(cases,'diff3')
    try:
        raw.reset();q=raw.data.qpos.numpy();q[:,1]=offsets;raw.data.qpos.assign(q);mjw.forward(raw.model,raw.data)
        obs,_,done,_=raw.step(np.zeros((3,3),dtype=np.float32));assert not done.any()
        state=raw.state.numpy();assert np.all(state[:,14]==0),'Unexpected pre-road obstacle contact'
        control=raw.data.ctrl.numpy();pose=raw.data.qpos.numpy()
        obs_error=float(np.max(abs(obs-obs[1])));control_error=float(np.max(abs(control-control[1])))
        b1=np.stack([b1_action(o,classic) for o in obs]);b1_error=float(np.max(abs(b1-b1[1])))
        predictions={}
        for method in p['methods']:
            r=json.loads((P/'runs'/method/'1610/step_2000000.json').read_text());verify_checkpoint(r['path'],r)
            agent=PPO.load(r['path']+'.zip',device='cpu')
            with open(r['path']+'.pkl','rb') as f:norm=pickle.load(f)
            output=agent.predict(norm.normalize_obs(obs),deterministic=True)[0]
            predictions[method]=dict(maximum_action_difference=float(np.max(abs(output-output[1]))),actions=output.tolist())
        # This is the runnable check: distinct lateral states have the same old packet/control.
        atomic_json(out/'pre_road_measurements.json',dict(lateral_spread_m=float(np.ptp(pose[:,1])),
            observation_max_difference=obs_error,control_max_difference_Nm=control_error,b1_action_max_difference=b1_error,
            observation_differences_by_column=np.max(abs(obs-obs[1]),axis=0).tolist(),predictions=predictions))
        np.savez_compressed(out/'pre_road_arrays.npz',qpos=pose,observation=obs,control=control)
        print('PRE_ROAD_MEASUREMENTS',float(np.ptp(pose[:,1])),obs_error,control_error,b1_error,flush=True)
        assert np.ptp(pose[:,1])>.23 and obs_error<=1e-5 and control_error<=1e-5 and b1_error<=1e-5
        assert all(x['maximum_action_difference']<=1e-5 for x in predictions.values())
        np.savez_compressed(out/'alias_state.npz',qpos=pose,observation=obs,control=control)
        rows=[None]*3;pending=set(range(3));ids=raw.ids.numpy();deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
        for _ in range(deadline):
            action=np.stack([b1_action(o,classic) for o in obs]);obs,_,done,infos=raw.step(action)
            stopped=raw.stopped_q.numpy()
            for w in list(pending):
                if not done[w]:continue
                info={k:v for k,v in infos[w].items() if k!='terminal_observation'}
                mean_fk=float(np.mean([sim.fk_joints(float(stopped[w,ids[2*j]]),float(stopped[w,ids[2*j+1]]))['leg_len'] for j in range(2)]))
                rows[w]=dict(**cases[w],**info,final_mean_fk_leg_m=mean_fk,lateral_offset_m=float(offsets[w]));pending.remove(w)
            if not pending:break
        assert not pending
        result=summary(rows)
        atomic_json(out/'result.json',dict(passed=True,pre_road_observation_max_difference=obs_error,
            pre_road_control_max_difference_Nm=control_error,b1_action_max_difference=b1_error,
            learned_policy_predictions=predictions,summary=result,runs=rows,
            observations_alias_distinct_lateral_states=True,
            path_state_outcome_relevance_observed=len({(r['success'],r['touched_terrain_contact_mask']) for r in rows})>1,
            inference='Constructed alias counterexample and initial-position intervention, not proof of cause for all trained failures or optimal corrective policy',
            additional_public_diagnostic_episodes=3,training_updates=0,old_gate_or_final_replayed=False,
            protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))
        print('PASS alias observation/control/action differences',obs_error,control_error,b1_error,flush=True)
        for row in rows:print('OFFSET',row['lateral_offset_m'],'success',row['success'],'terrain',row['touched_terrain_contact_mask'],'required',row['required_terrain_contact_mask'],'physical/design',row['physical_safety_passed'],row['design_joint_passed'],flush=True)
    finally:raw.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
