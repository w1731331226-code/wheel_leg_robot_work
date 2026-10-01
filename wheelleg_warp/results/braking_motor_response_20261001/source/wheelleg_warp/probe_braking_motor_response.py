"""Fixed six-motor0.1Nm axis hull and held-out common mixes at18 archived braking states."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp
import wheelleg_sim as sim
from native.models import batch
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN as LIMIT
from probe_height_115_contact_action_pair import basis,margins
from probe_height_115_passive import geometry,JOINTS
from probe_height_115_action_predict_loow import sha


def features(m,q,v):
    geo=geometry(m,q);middle=np.array([(m.body_pos[m.body('leg'+s).id]+m.body_pos[m.body('leg'+s+'_D').id])/2 for s in ('L','R')])
    lengths=np.c_[geo[1],np.linalg.norm(geo[5]-middle,axis=-1)]
    positions=q[:,[m.joint(n).qposadr[0] for n in JOINTS]]
    return np.c_[lengths,positions,v]


def run(source,output,reanalyze=False):
    source=source.resolve();output=output.resolve()
    if not reanalyze:assert not output.exists();output.mkdir(parents=True)
    else:assert output.exists() and not (output/'verification.json').exists()
    metadata=json.loads((source/'verification.json').read_text());columns=metadata['columns']
    scenes=[HeightTerrainScenario(**s) for s in metadata['scenarios']];cpu=[model(s) for s in scenes]
    arms=['zero','zero_twin']+[f'motor{j}{sign}' for j in range(6) for sign in ('+','-')]
    arms += [f'{name}{sign}' for name in ('F','H','W','FH','HW','FW') for sign in ('+','-')]
    samples=[];starts=[];deltas=[];bounds=[]
    with np.load(source/'windows.npz',allow_pickle=False) as archive:
        for w,m in enumerate(cpu):
            pre=archive[f'w{w}_braking_pre'];rad=archive[f'w{w}_braking_radial'];arrival=float(rad[:,3].max())
            for elapsed in (.1,.5,1.):
                index=int(np.argmin(abs(rad[:,0]-arrival-elapsed)));assert abs(rad[index,0]-arrival-elapsed)<1e-9
                start=pre[index];q=start[1:columns['qvel_start']];v=start[columns['qvel_start']:columns['warmstart_start']]
                delta=np.zeros((26,6));axis=np.eye(6)*.1
                for j in range(6):delta[2+2*j]=axis[j];delta[3+2*j]=-axis[j]
                common=basis(m,q,v);common=common/np.sum(abs(common),axis=0)*.1
                mixes=[common[:,j] for j in range(3)]+[(common[:,0]+common[:,1])/2,(common[:,1]+common[:,2])/2,(common[:,0]+common[:,2])/2]
                for j,change in enumerate(mixes):delta[14+2*j]=change;delta[15+2*j]=-change
                assert np.max(np.sum(abs(delta),axis=1))<=.10000000001
                maximum=[]
                for j,name in enumerate(('alphaL','betaL','alphaR','betaR','wheel1','wheel2')):
                    speed=v[m.joint(name).dofadr[0]]
                    maximum.append(sim.hw.torque_limit(1e6,speed,j<4,0.,.0005)[0]/(1.05 if j>=4 else 1.))
                command=start[columns['ctrl_start']:]
                assert np.all(abs(command+delta)<=np.array(maximum)+1e-6)
                starts.append(start);deltas.append(delta);bounds.append(maximum);samples.append(dict(world=w,elapsed_s=elapsed,source_index=index))
    starts=np.array(starts);deltas=np.array(deltas);bounds=np.array(bounds)
    n=len(samples);assert n==18
    q0=np.repeat(starts[:,1:columns['qvel_start']],26,axis=0).astype(np.float32)
    v0=np.repeat(starts[:,columns['qvel_start']:columns['warmstart_start']],26,axis=0).astype(np.float32)
    warm=np.repeat(starts[:,columns['warmstart_start']:columns['ctrl_start']],26,axis=0).astype(np.float32)
    commands=(starts[:,None,columns['ctrl_start']:]+deltas).astype(np.float32)
    effective=commands.astype(float)-commands[:,0:1].astype(float)
    if reanalyze:
        raw=np.load(output/'native_response.npz',allow_pickle=False)
        assert np.array_equal(raw['start'],starts) and np.array_equal(raw['commands'],commands) and np.array_equal(raw['effective_delta'],effective)
        q=raw['qpos'];v=raw['qvel'];force=raw['actuator_force']
    else:
        wp.init();wp.set_device('cuda:0')
        models=[cpu[s['world']] for s in samples for _ in arms];scenario=[scenes[s['world']] for s in samples for _ in arms]
        _,wm,data,_=batch(models,scenario)
        data.qpos.assign(q0);data.qvel.assign(v0);data.qacc_warmstart.assign(warm);data.ctrl.assign(commands.reshape(-1,6))
        assert np.array_equal(data.qpos.numpy(),q0) and np.array_equal(data.qvel.numpy(),v0)
        mjw.step(wm,data)
        q=data.qpos.numpy().astype(float).reshape(n,26,-1);v=data.qvel.numpy().astype(float).reshape(n,26,-1)
        force=data.actuator_force.numpy().astype(float).reshape(n,26,6)
        np.savez_compressed(output/'native_response.npz',start=starts,commands=commands,effective_delta=effective,qpos=q,qvel=v,actuator_force=force)
    rows=[];calibration=[];predictions=[];nominal_predictions=[];all_features=[]
    for i,sample in enumerate(samples):
        m=cpu[sample['world']];f=features(m,q[i],v[i]);all_features.append(f)
        G=np.column_stack([(f[2+2*j]-f[3+2*j])/(effective[i,2+2*j,j]-effective[i,3+2*j,j]) for j in range(6)])
        prediction=f[0]+effective[i,14:]@G.T;error=prediction-f[14:];calibration.append(G);predictions.append(prediction)
        # Fixed public plant at CURRENT state; full q/v is offline simulation evidence.
        d=mujoco.MjData(m);d.qpos[:]=q0[26*i];d.qvel[:]=v0[26*i];d.qacc_warmstart[:]=warm[26*i];d.ctrl[:]=commands[i,0]
        linear=np.empty((2*m.nv,6));mujoco.mjd_transitionFD(m,d,1e-6,True,None,linear,None,None)
        nominal_v=v[i,0]+effective[i,14:]@linear[m.nv:].T;nominal_predictions.append(nominal_v)
        nominal_error=nominal_v-v[i,14:]
        joint_margin=margins(m,q[i]);angles=q[i,:,3:7]
        pitch=np.arcsin(np.clip(2*(angles[:,0]*angles[:,2]-angles[:,3]*angles[:,1]),-1,1))
        roll=np.arctan2(2*(angles[:,0]*angles[:,1]+angles[:,2]*angles[:,3]),1-2*(angles[:,1]**2+angles[:,2]**2))
        yaw=np.arctan2(2*(angles[:,0]*angles[:,3]+angles[:,1]*angles[:,2]),1-2*(angles[:,2]**2+angles[:,3]**2))
        attitude=np.max(abs(np.c_[roll,pitch,yaw]),axis=1)
        safe=(f[:,:4].min(axis=1)>=LIMIT)&(joint_margin.min(axis=1)>=0)&(attitude<=np.deg2rad(5))&(np.max(abs(force[i])-bounds[i],axis=1)<=1e-6)
        active=[m.joint(name).dofadr[0] for name in ('alphaL','betaL','alphaR','betaR')]
        rows.append(dict(**sample,physical_safe_arms=int(safe.sum()),
            min_actual_A_B_m=float(f[:,:4].min()),min_eight_joint_margin_rad=float(joint_margin.min()),
            zero_twin_qpos_max_error=float(abs(q[i,0]-q[i,1]).max()),zero_twin_qvel_max_error=float(abs(v[i,0]-v[i,1]).max()),
            heldout_native_length_error_m=float(abs(error[:,:4]).max()),heldout_native_joint_error_rad=float(abs(error[:,4:12]).max()),
            heldout_native_vx_error_m_s=float(abs(error[:,12]).max()),heldout_native_active_velocity_error_rad_s=float(abs(error[:,12+np.array(active)]).max()),
            current_nominal_vx_delta_error_m_s=float(abs(nominal_error[:,0]).max()),current_nominal_active_velocity_delta_error_rad_s=float(abs(nominal_error[:,active]).max())))
    arrays=dict(features=all_features,G6=calibration,heldout_prediction=predictions,current_nominal_velocity_prediction=nominal_predictions)
    if reanalyze:
        previous=np.load(output/'prediction.npz',allow_pickle=False)
        for key,value in arrays.items():np.testing.assert_allclose(value,previous[key],atol=1e-12,rtol=0)
    else:np.savez_compressed(output/'prediction.npz',**arrays)
    result=dict(role='six_motor_native_local_braking_response',samples=rows,arms=arms,scenarios=[asdict(s) for s in scenes],
        sample_count=n,physical_step_s=.0005,requested_axis_increment_Nm=.1,mix_requested_L1_limit_Nm=.1,
        calibration_axes=12,heldout_mixes=12,physical_safe_arms=sum(r['physical_safe_arms'] for r in rows),total_arms=n*26,
        feature_columns=['A_L','A_R','B_L','B_R']+list(JOINTS)+[f'qvel{j}' for j in range(cpu[0].nv)],
        features_source='actual passive A/B chains and eight joints, full post-step velocities',
        current_nominal_response='mjd_transitionFD of fixed nominal plant at archived CURRENT q/v/known issued command. Predict delta from the same Warp zero baseline, not a future-free absolute state forecast.',
        limitations='Native G6 and zero baseline are offline future response calibration, never online inputs. Current nominal FD also uses full simulation q/v and known public terrain layout; not deployable sensor or real-time evidence.18 finite states, one0.5ms step,0.1Nm axis hull only; no multi-step, recursive uncertainty bound, full task or default promotion. Archive uses preceding nominal table, not the newer current-J trial.',
        input_sha256={str((source/f).relative_to(ROOT)):sha(source/f) for f in ('verification.json','windows.npz')},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/native/models.py',ROOT/'wheelleg_warp/native/terrain.py',ROOT/'wheelleg_warp/probe_height_115_passive.py',ROOT/'wheelleg_warp/probe_height_115_contact_action_pair.py')})
    result['reanalyzed_without_new_Warp_collection']=reanalyze
    if reanalyze:result['collection_entry_sha256']=sha(output/'source_at_collection.py')
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('COMPLETED safe',result['physical_safe_arms'],'/',result['total_arms'],flush=True)
    for key in ('heldout_native_length_error_m','heldout_native_joint_error_rad','heldout_native_vx_error_m_s','heldout_native_active_velocity_error_rad_s','current_nominal_vx_delta_error_m_s','current_nominal_active_velocity_delta_error_rad_s'):
        print(key,max(r[key] for r in rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--reanalyze',action='store_true')
    args=parser.parse_args();run(args.source,args.output,args.reanalyze)
