"""Rebuild CPU diagnostic branches; quantify, do not certify, model uncertainty."""
from pathlib import Path
import argparse,json,math
import numpy as np
import mujoco
from native.terrain import HeightTerrainScenario,model
from review_yaw_sector import sha,success,ROOT
import sys
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from state_estimation import leg_kinematics
import wheelleg_sim as sim


def run(out):
    p=json.loads((out/'registration.json').read_text());c=json.loads((out/'completion.json').read_text());ledger=json.loads((out/'completed_jobs.json').read_text())
    assert c['completed_episodes']==ledger['completed_episodes']==p['budget_episodes']==81 and c['training_updates']==p['training_updates']==0
    assert c['records']==ledger['records'] and c['registration_sha256']==sha(out/'registration.json')
    assert all(sha(ROOT/n)==v for n,v in p['source_sha256'].items()) and not p['old_gate_or_final_used']
    for m in p['models']:
        q=Path(m['prefix']);assert sha(q.with_suffix('.zip'))==m['checkpoint']['checkpoint_sha256'] and sha(q.with_suffix('.pkl'))==m['checkpoint']['normalization_sha256']
    summary={};events=[];snapshots=0;steps=0
    for rec in c['records']:
        label=rec['label'];sp=out/(label+'_states.npz');ep=out/(label+'_episodes.json');rp=out/(label+'_response.json')
        assert sha(sp)==rec['states_sha256'] and sha(ep)==rec['episodes_sha256'] and sha(rp)==rec['response_sha256']
        z=np.load(sp,allow_pickle=False);d=json.loads(ep.read_text());responses=json.loads(rp.read_text());state=z['state'];ids=z['ids'];worlds=z['world'];kinds=z['event']
        assert state.shape==(rec['snapshots'],101) and np.isfinite(state).all() and len(responses['outcomes'])==len(state)
        assert responses['shadow_steps']==len(state)*6 and len(d['runs'])==27
        for row,case in zip(d['runs'],p['cases']):
            assert row['seed']==case['seed'] and row['scenario']==case['scenario'] and row['success']==success(row)
            assert row['physical_evidence_steps']==row['physical_steps']>0
        maxq=0.;maxv=0.;maxmid=0.;event_res=[];models={}
        for i,s in enumerate(state):
            w=int(worlds[i]);kind=int(kinds[i]);r=responses['outcomes'][i]
            assert r['index']==i and r['world']==w and r['event']==kind and r['case']==p['cases'][w]['seed']
            scene=HeightTerrainScenario(**p['cases'][w]['scenario'])
            if w not in models:models[w]=model(scene)
            m=models[w];assert m.na==0 and m.nmocap==0 and m.opt.timestep==.0005
            q=s[:17];v=s[17:33];nom=s[56:62];actor=s[62:68];actual=s[50:56]
            np.testing.assert_allclose(actual,(nom+actor).astype(np.float32),rtol=0,atol=1e-6)
            matrix=np.vstack([leg_kinematics(q[ids[:2]],np.zeros(2))[3]*[3.4335,1],-leg_kinematics(q[ids[2:4]],np.zeros(2))[3]*[3.4335,1]])
            u=np.linalg.lstsq(matrix,actor[:4],rcond=None)[0];assert abs(u).max()<=1+1e-8
            np.testing.assert_allclose(matrix@u,actor[:4],rtol=0,atol=1e-8)
            qout={}
            for name,branch in r['branches'].items():
                command=np.asarray(branch['ctrl']);assert len(command)==6
                if name=='actual_command':np.testing.assert_array_equal(command,actual)
                elif name=='nominal_only':np.testing.assert_array_equal(command,nom)
                else:
                    coord,sign=name.rsplit('_',1);j=0 if coord=='Fdiff' else 1
                    perturbed=np.clip(u+int(sign)*.01*np.eye(2)[j],-1,1);inc=np.r_[matrix@perturbed,0.,0.];lam=1.
                    for a in range(6):
                        bound=sim.hw.torque_limit(float('inf'),float(v[ids[4+a]]),a<4,0.,.0005)[0]/(1 if a<4 else 1.05)
                        if inc[a]>0:lam=min(lam,(bound-nom[a])/inc[a])
                        elif inc[a]<0:lam=min(lam,(-bound-nom[a])/inc[a])
                    np.testing.assert_allclose(command,nom+np.clip(lam,0,1)*inc,rtol=0,atol=1e-12)
                data=mujoco.MjData(m);data.qpos[:]=q;data.qvel[:]=v;data.qacc_warmstart[:]=s[33:49];data.ctrl[:]=command;data.time=s[49]
                mujoco.mj_step(m,data)
                np.testing.assert_allclose(data.qpos[ids[:4]],branch['active_q'],rtol=0,atol=1e-12)
                np.testing.assert_allclose(data.qvel[ids[4:8]],branch['active_v'],rtol=0,atol=1e-12)
                np.testing.assert_allclose((data.qvel[ids[4:8]]-v[ids[4:8]])/.0005,branch['effective_acceleration'],rtol=0,atol=1e-9)
                qout[name]=data.qpos[ids[:4]].copy();steps+=1
            assert set(qout)=={'actual_command','nominal_only','Fdiff_-1','Fdiff_1','Hdiff_-1','Hdiff_1'}
            gpuq=s[68:85][ids[:4]];gpuv=s[85:101][ids[4:8]];cpuq=qout['actual_command'];cpuv=np.asarray(r['branches']['actual_command']['active_v'])
            np.testing.assert_allclose(cpuq-gpuq,r['cpu_gpu_active_q_error'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(cpuv-gpuv,r['cpu_gpu_active_v_error'],rtol=0,atol=1e-12)
            maxq=max(maxq,float(abs(cpuq-gpuq).max()));maxv=max(maxv,float(abs(cpuv-gpuv).max()))
            sensitivity={}
            for coord in ['Fdiff','Hdiff']:
                slope=(qout[coord+'_1']-qout[coord+'_-1'])/.02
                midpoint=(qout[coord+'_1']+qout[coord+'_-1'])/2-cpuq
                sensitivity[coord]=dict(active_q_slope_per_normalized_coordinate=slope.tolist(),midpoint_residual_rad=midpoint.tolist())
                maxmid=max(maxmid,float(abs(midpoint).max()))
            entry=dict(case=r['case'],event=kind,time_s=r['time_s'],gpu_margin_rad=float(1.4-abs(gpuq).max()),
                cpu_actual_margin_rad=float(1.4-abs(cpuq).max()),cpu_nominal_margin_rad=float(1.4-abs(qout['nominal_only']).max()),
                cpu_gpu_q_error_rad=float(abs(cpuq-gpuq).max()),local_response=sensitivity)
            event_res.append(entry);events.append(dict(label=label,**entry))
        snapshots+=len(state);summary[label]=dict(snapshots=len(state),maximum_cpu_gpu_q_error_rad=maxq,maximum_cpu_gpu_v_error_rad_s=maxv,
            maximum_local_midpoint_residual_rad=maxmid,recorded_crossing_events=sum(e['event']==2 for e in event_res))
        print('VERIFIED',label,summary[label],flush=True)
    assert snapshots==27 and steps==162 and snapshots<=p['maximum_snapshots'] and steps<=p['maximum_cpu_shadow_steps']
    report=dict(verified=True,rollout_episodes=81,snapshots=snapshots,cpu_shadow_steps=steps,summary=summary,events=events,
        registration_sha256=sha(out/'registration.json'),completion_sha256=sha(out/'completion.json'),reviewer_sha256=sha(__file__),training_updates=0,
        interpretation='CPU shadow reproduces its own branches with exact same recorded state and bounded differential commands. GPU/CPU error and local response are empirical diagnostics, not an uncertainty certificate or deployed predictor.',
        limits='Actual simulated terrain/parameters used only offline. Derived time is harmless only for this static/no-mocap/no-actuator-activation diagnostic; no general full-simulator-state resume claim. Snapshot count differs from prior crossing count due freshGPU boundary variation; never replaced old failures.',
        next='Validate a practical nominal joint response model and prospective error assumptions separately before formulating state-admissible Nom repair/Actor constraints. Do not relabel empirical extrema as robust bounds.')
    (out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('PASS27 snapshots,162CPU shadow steps, no learning',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
