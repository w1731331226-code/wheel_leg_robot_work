"""Independent all-sample algebraic and original-task gate review of cone probe."""
from pathlib import Path
import argparse,json,math
import numpy as np
from review_yaw_sector import sha,success,ROOT
import sys
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim
from state_estimation import leg_kinematics


def maps(q):
    result=np.empty((len(q),4,2))
    for side in range(2):
        p1=sim.PHI1_STAND-q[:,2*side];p4=sim.PHI4_STAND-q[:,2*side+1]
        bx=.15*np.cos(p1);bz=.15*np.sin(p1);dx=.15+.15*np.cos(p4);dz=.15*np.sin(p4)
        aa=.54*(dx-bx);bb=.54*(dz-bz);cc=(dx-bx)**2+(dz-bz)**2
        p2=2*np.arctan2(bb-np.sqrt(np.maximum(0,aa*aa+bb*bb-cc*cc)),aa+cc)
        cx=bx+.27*np.cos(p2);cz=bz+.27*np.sin(p2);x=cx-.075;r=np.sqrt(x*x+cz*cz)
        cbx=cx-bx;cbz=cz-bz;cdx=cx-dx;cdz=cz-dz;det=cbx*cdz-cbz*cdx
        k0=cbx*.15*np.sin(p1)-cbz*.15*np.cos(p1);k1=cdx*.15*np.sin(p4)-cdz*.15*np.cos(p4)
        j0x=cdz*k0/det;j0z=-cdx*k0/det;j1x=-cbz*k1/det;j1z=cbx*k1/det
        sign=1 if side==0 else -1
        result[:,2*side,0]=sign*(x*j0x+cz*j0z)/r*3.4335
        result[:,2*side,1]=sign*(-cz*j0x+x*j0z)/(r*r)
        result[:,2*side+1,0]=sign*(x*j1x+cz*j1z)/r*3.4335
        result[:,2*side+1,1]=sign*(-cz*j1x+x*j1z)/(r*r)
    return result


def run(out):
    reg=json.loads((out/'registration.json').read_text());c=json.loads((out/'completion.json').read_text());ledger=json.loads((out/'completed_jobs.json').read_text())
    assert c['completed_episodes']==ledger['completed_episodes']==reg['budget_episodes']==243 and c['training_updates']==0
    assert c['registration_sha256']==sha(out/'registration.json') and c['records']==ledger['records'] and not reg['old_gate_or_final_used']
    assert all(sha(ROOT/n)==v for n,v in reg['source_sha256'].items())
    stats={};rows={};samples=0;algebra={}
    for record in ledger['records']:
        label=record['label'];mode=label.split('_',1)[1];rp=out/(label+'.json');tp=out/(label+'_trace.npz')
        assert sha(rp)==record['result_sha256'] and sha(tp)==record['trace_sha256']
        d=json.loads(rp.read_text());z=np.load(tp,allow_pickle=False);scores=[];modified=0;guarded=0;max_cone=0.;max_bound=0.
        assert len(d['runs'])==len(reg['cases'])==27 and len(z['offsets'])==28 and np.isfinite(z['trace']).all()
        for i,(r,case) in enumerate(zip(d['runs'],reg['cases'])):
            assert r['seed']==case['seed'] and r['scenario']==case['scenario'] and r['success']==success(r)
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]];assert len(t)==r['physical_steps']==r['physical_evidence_steps']
            np.testing.assert_allclose(t[:,1],np.arange(1,len(t)+1)*.0005,rtol=0,atol=1e-9)
            np.testing.assert_array_equal(t[1:,2:6],t[:-1,10:14]);np.testing.assert_array_equal(t[1:,6:10],t[:-1,14:18])
            margin=1.4-abs(t[:,10:14]).max(axis=1);np.testing.assert_allclose(np.minimum.accumulate(margin),t[:,40],rtol=0,atol=1e-12)
            assert r['design_joint_passed']==bool(margin.min()>=0)
            np.testing.assert_allclose((t[:,18:24]+t[:,24:30]).astype(np.float32),t[:,30:36].astype(np.float32),rtol=0,atol=1e-6)
            matrix=maps(t[:,2:6]);raw=t[:,36,None]*t[:,41:43];requested=np.einsum('nij,nj->ni',matrix,raw)
            # Validate vectorized geometry independently at each trajectory end.
            for k in [0,len(t)-1]:
                independent=np.vstack([leg_kinematics(t[k,2:4],np.zeros(2))[3]*[3.4335,1],-leg_kinematics(t[k,4:6],np.zeros(2))[3]*[3.4335,1]])
                np.testing.assert_allclose(matrix[k],independent,rtol=0,atol=1e-9)
            effect=t[:,24:28];change=np.linalg.norm(effect-requested,axis=1);modified+=int(np.sum(change>1e-8))
            if mode=='original':np.testing.assert_allclose(effect,requested,rtol=0,atol=1e-9)
            elif mode=='zero_after_filter':assert np.all(effect==0)
            else:
                q=t[:,2:6];v=t[:,6:10];nom=t[:,18:22]
                guard=(abs(q)+np.maximum(0,np.sign(q)*v)*.02>=1.4)&(np.sign(q)*nom<0)
                guarded+=int(np.sum(np.any(guard,axis=1)))
                if guard.any():max_cone=max(max_cone,float((np.sign(q)*effect)[guard].max()));assert max_cone<=1e-8
                aa=np.einsum('ni,ni->n',matrix[:,:,0],matrix[:,:,0]);ab=np.einsum('ni,ni->n',matrix[:,:,0],matrix[:,:,1]);bb=np.einsum('ni,ni->n',matrix[:,:,1],matrix[:,:,1])
                b0=np.einsum('ni,ni->n',matrix[:,:,0],effect);b1=np.einsum('ni,ni->n',matrix[:,:,1],effect);det=aa*bb-ab*ab
                u=np.c_[(bb*b0-ab*b1)/det,(aa*b1-ab*b0)/det]
                np.testing.assert_allclose(np.einsum('nij,nj->ni',matrix,u),effect,rtol=0,atol=1e-9);assert abs(u).max()<=1+1e-8
                distance=np.sum((effect-requested)**2,axis=1);zero_cost=np.sum(requested**2,axis=1);assert np.all(distance<=zero_cost+1e-8)
                np.testing.assert_allclose(effect[~guard.any(axis=1)],requested[~guard.any(axis=1)],rtol=0,atol=1e-9)
            speed=abs(t[:,6:10])*60/(2*np.pi);bounds=np.where(speed>175,40*np.maximum(0,(280-speed)/105),40)
            max_bound=max(max_bound,float((abs(t[:,30:34])-bounds).max()));assert max_bound<=1e-6
            assert np.all(t[:,28:30]==0) and np.all(t[:,43]==0) and np.all(t[:,46]==0)
            if r['reason']=='completed':
                assert r['duration_s']>=r['arrival_s']+2-1e-8;scores.append(r['rms_deg'][2]*math.sqrt(r['duration_s']/(3.5+1.5*r['task_goal_progress_m']/abs(r['scenario']['speed']))))
            else:scores.append(None)
        complete=all(x is not None for x in scores);assert d['summary']['complete']==complete
        if complete:assert math.isclose(sum(scores)/27,d['summary']['mean_yaw_score_deg'],abs_tol=1e-12)
        else:assert d['summary']['mean_yaw_score_deg'] is None
        assert d['summary']['success_count']==sum(r['success'] for r in d['runs']) and d['physical']==sum(r['physical_safety_passed'] for r in d['runs']) and d['design']==sum(r['design_joint_passed'] for r in d['runs'])
        samples+=len(z['trace']);assert record['stats']['physics_samples']==len(z['trace']);rows[label]=d['runs'];stats[label]={k:v for k,v in d.items() if k!='runs'}
        algebra[label]=dict(projected_samples=modified,guarded_samples=guarded,max_selected_direction_violation_Nm=max_cone,max_command_bound_excess_Nm=max_bound)
        print('VERIFIED',label,stats[label],algebra[label],flush=True)
    preserve={};gain=[];clean=True
    for m in reg['models']:
        seed=str(m['seed']);cone=rows[seed+'_cone'];original=rows[seed+'_original'];zero=rows[seed+'_zero_after_filter']
        preserve[seed]=all((not b['success'] or a['success']) and (not b['physical_safety_passed'] or a['physical_safety_passed']) and (not b['design_joint_passed'] or a['design_joint_passed']) for ref in [original,zero] for a,b in zip(cone,ref))
        clean=clean and all(r['design_joint_passed'] for r in cone);gain.append(sum(r['success'] for r in cone)-sum(r['success'] for r in zero))
    gate=all(preserve.values()) and clean and sum(gain)>0
    assert c['preservation']==preserve and c['design_clean']==clean and c['cone_minus_zero_success_counts']==gain and c['continue_gate']==gate
    result=dict(verified=True,episodes=243,physics_samples=samples,stats=stats,algebra=algebra,preservation=preserve,design_clean=clean,cone_minus_zero_success_counts=gain,continue_gate=gate,
        registration_sha256=sha(out/'registration.json'),completion_sha256=sha(out/'completion.json'),reviewer_sha256=sha(__file__),
        scope='All original task rows, trajectory q/v/time/decomposition and cone-coordinate-torque bounds verified. Unit QP optimality supports solver; finite direction conditions are not joint acceleration/state-domain invariance or learned method superiority.')
    (out/'review.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print('PASS',samples,'samples, continue gate',gate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
