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
    offsets=np.repeat([-.12,0.,.12],2);cases=[dict(seed=4001000+i,scenario=asdict(s)) for i in range(6)]
    classic=json.loads((P/'classical_selection.json').read_text())['selected']['candidate']
    out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'preregistration.json',dict(cases=cases,lateral_offsets_m=offsets.tolist(),
        controls='Original B1, same model/constraints; only initial root lateral position changes',
        test='Exact initial packet/source-input equality; same-offset duplicates describe post-integration numerical variation, not a relaxed pass tolerance',training=False,
        old_gate_or_final_replayed=False,protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))
    raw=raw_env(cases,'diff3')
    try:
        obs=raw.reset();q=raw.data.qpos.numpy();q[:,1]=offsets;raw.data.qpos.assign(q);mjw.forward(raw.model,raw.data)
        initial_q=raw.data.qpos.numpy();initial_v=raw.data.qvel.numpy();sensor=raw.data.sensordata.numpy();ids=raw.ids.numpy()
        # All physical packet inputs are unchanged; absolute root y is omitted.
        np.testing.assert_array_equal(initial_q[:,0],np.repeat(initial_q[2,0],6))
        np.testing.assert_array_equal(initial_q[:,2:],np.repeat(initial_q[2:3,2:],6,axis=0))
        np.testing.assert_array_equal(initial_v,np.repeat(initial_v[2:3],6,axis=0))
        np.testing.assert_array_equal(sensor[:,int(ids[10]):int(ids[10])+3],np.repeat(sensor[2:3,int(ids[10]):int(ids[10])+3],6,axis=0))
        np.testing.assert_array_equal(obs,np.repeat(obs[2:3],6,axis=0))
        initial_obs=obs.copy()
        obs,_,done,_=raw.step(np.zeros((6,3),dtype=np.float32));assert not done.any()
        state=raw.state.numpy();assert np.all(state[:,14]==0),'Unexpected pre-road obstacle contact'
        control=raw.data.ctrl.numpy();pose=raw.data.qpos.numpy()
        obs_error=float(np.max(abs(obs-obs[2])));control_error=float(np.max(abs(control-control[2])))
        b1=np.stack([b1_action(o,classic) for o in obs]);b1_error=float(np.max(abs(b1-b1[2])))
        predictions={}
        for method in p['methods']:
            r=json.loads((P/'runs'/method/'1610/step_2000000.json').read_text());verify_checkpoint(r['path'],r)
            agent=PPO.load(r['path']+'.zip',device='cpu')
            with open(r['path']+'.pkl','rb') as f:norm=pickle.load(f)
            normalized=norm.normalize_obs(initial_obs)
            np.testing.assert_array_equal(normalized,np.repeat(normalized[2:3],6,axis=0))
            batched=agent.predict(normalized,deterministic=True)[0]
            output=np.concatenate([agent.predict(normalized[i:i+1],deterministic=True)[0] for i in range(6)])
            predictions[method]=dict(maximum_action_difference=float(np.max(abs(output-output[2]))),
                batched_numerical_difference=float(np.max(abs(batched-batched[2]))),actions=output.tolist())
        # Exact aliasing is checked before integration, not by accepting replay noise.
        atomic_json(out/'pre_road_measurements.json',dict(lateral_spread_m=float(np.ptp(pose[:,1])),
            observation_max_difference=obs_error,control_max_difference_Nm=control_error,b1_action_max_difference=b1_error,
            observation_differences_by_column=np.max(abs(obs-obs[2]),axis=0).tolist(),predictions=predictions,
            same_offset_duplicate_obs_difference=max(float(np.max(abs(obs[i]-obs[i+1]))) for i in [0,2,4]),
            same_offset_duplicate_ctrl_difference=max(float(np.max(abs(control[i]-control[i+1]))) for i in [0,2,4])))
        np.savez_compressed(out/'pre_road_arrays.npz',qpos=pose,initial_qpos=initial_q,initial_observation=initial_obs,observation=obs,control=control)
        print('PRE_ROAD_MEASUREMENTS',float(np.ptp(pose[:,1])),obs_error,control_error,b1_error,flush=True)
        assert np.ptp(initial_q[:,1])>.23
        assert all(x['maximum_action_difference']==0 for x in predictions.values())
        np.savez_compressed(out/'alias_state.npz',qpos=pose,observation=obs,control=control)
        rows=[None]*6;pending=set(range(6));ids=raw.ids.numpy();deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2)/.02))+2
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
            initial_observations_exactly_alias_distinct_lateral_states=True,
            path_state_outcome_relevance_observed=len({(r['success'],r['touched_terrain_contact_mask']) for r in rows})>1,
            inference='Exact initial packet alias; post-integration replay equality not claimed. Constructed initial-position intervention, not proof of cause for all trained failures or optimal corrective policy',
            additional_public_diagnostic_episodes=6,training_updates=0,old_gate_or_final_replayed=False,
            protocol_sha256=digest(P/'protocol.json'),verifier_sha256=digest(__file__)))
        print('PASS alias observation/control/action differences',obs_error,control_error,b1_error,flush=True)
        for row in rows:print('OFFSET',row['lateral_offset_m'],'success',row['success'],'terrain',row['touched_terrain_contact_mask'],'required',row['required_terrain_contact_mask'],'physical/design',row['physical_safety_passed'],row['design_joint_passed'],flush=True)
    finally:raw.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
