"""Summarize 2 kHz contact and posture divergence from a paired public run."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np


def attitude(q):
    w,x,y,z=np.moveaxis(q[...,3:7],-1,0)
    roll=np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
    pitch=np.degrees(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))
    yaw=np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    return np.stack((roll,pitch,yaw),axis=-1)


def first_time(time,condition):
    indexes=np.flatnonzero(condition)
    return float(time[indexes[0]]) if len(indexes) else None


def analyze(directory):
    summary=json.loads((directory/'summary.json').read_text())
    with np.load(directory/'initial.npz',allow_pickle=False) as initial:
        nq=initial['a_qpos'].shape[1];nv=initial['a_qvel'].shape[1];nu=initial['a_ctrl'].shape[1]
    with np.load(directory/'first_20ms.npz',allow_pickle=False) as first:
        first_a=first['state_a'];first_b=first['state_b']
    with np.load(directory/'selected_2khz.npz',allow_pickle=False) as selected:
        a=selected['a'];b=selected['b'];seeds=selected['selected_seeds'].tolist()
        steps=selected['policy_steps']
    assert a.shape==b.shape and a.shape[1]==len(seeds)==4
    time=(np.repeat(steps-1,40)*.02+np.tile(np.arange(1,41)*.0005,len(steps)))
    assert len(time)==len(a) and np.allclose(np.diff(time),.0005,atol=1e-9)
    mask_col=1+nq+nv+nu;results=[]
    for j,seed in enumerate(seeds):
        i=next(k for k,row in enumerate(summary['outcomes']['a']) if row['seed']==seed)
        changed=np.any(first_a[:,i,1:1+nq+nv]!=first_b[:,i,1:1+nq+nv],axis=1)
        first_step=int(np.flatnonzero(changed)[0]+1) if changed.any() else None
        action=next((dict(policy_step=row['step'],max_abs=row['action_max_delta'][j])
                     for row in summary['selected_policy_trace'] if row['action_max_delta'][j]>0),None)
        pair=[]
        for record in (a,b):
            mask=record[:,j,mask_col].astype(int)
            left=first_time(time,(mask&4)!=0);right=first_time(time,(mask&8)!=0)
            q=record[:,j,1:1+nq];angles=attitude(q)
            breach=first_time(time,np.max(abs(angles),axis=1)>5.)
            pair.append(dict(first_left_contact_s=left,first_right_contact_s=right,
                contact_gap_ms=abs(right-left)*1000 if left is not None and right is not None else None,
                first_world_attitude_5deg_s=breach,peak_attitude_deg=float(np.max(abs(angles)))))
        contact=min(x['first_left_contact_s'] for x in pair if x['first_left_contact_s'] is not None)
        growth=[]
        for offset in (-.1,0.,.1,.3):
            k=int(np.argmin(abs(time-(contact+offset))))
            qa=a[k,j,1:1+nq];qb=b[k,j,1:1+nq]
            growth.append(dict(relative_to_first_contact_s=offset,time_s=float(time[k]),
                base_position_gap_m=float(np.linalg.norm(qa[:3]-qb[:3])),
                yaw_a_deg=float(attitude(qa)[2]),yaw_b_deg=float(attitude(qb)[2]),
                qvel_max_abs=float(np.max(abs(a[k,j,1+nq:1+nq+nv]-b[k,j,1+nq:1+nq+nv])))))
        results.append(dict(seed=seed,first_state_difference_substep=first_step,
            first_policy_action_difference=action,a=pair[0],b=pair[1],growth=growth,
            outcome_a=summary['outcomes']['a'][i],outcome_b=summary['outcomes']['b'][i]))
    output=directory/'analysis.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(dict(cases=results,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Task contract v2 step terrains have zero local attitude reference. Observed divergence does not identify the unique solver cause.'),
        ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print('analyzed',len(results),'selected public worlds')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path)
    parser.add_argument('--check',action='store_true');args=parser.parse_args()
    if args.check:
        np.testing.assert_array_equal(attitude(np.array([0.,0.,0.,1.,0.,0.,0.])),[0.,0.,0.])
        print('attitude check passed')
    elif args.input is None:parser.error('--input required unless --check')
    else:analyze(args.input)
