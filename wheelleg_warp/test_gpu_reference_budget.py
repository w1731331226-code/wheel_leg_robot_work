"""GPU/static NumPy equivalence and whole-array mutation-scope regression."""
from pathlib import Path
import json,sys,gc,weakref
import numpy as np
import warp as wp
from gpu_reference_budget import apply_budget,WIDTH,D
from reference_residual_budget import reference,budget,mix,ROOT
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/reference_budget_v1'


def check():
    base=OUT.parent;states=[];heights=[];ids=None
    for folder in ['contact_state_holdout_v2','joint_response_v1']:
        p=base/folder;reg=json.loads((p/'registration.json').read_text());done=json.loads((p/('collection_completion.json' if folder=='contact_state_holdout_v2' else 'completion.json')).read_text())
        for job in done['records']:
            path=p/'collection'/f'{job["label"]}_states.npz' if folder=='contact_state_holdout_v2' else p/f'{job["label"]}_states.npz'
            with np.load(path,allow_pickle=False) as z:
                if ids is None:ids=z['ids'].copy()
                else:np.testing.assert_array_equal(ids,z['ids'])
                for w,s in zip(z['world'],z['state']):states.append(s[:101]);heights.append(reg['cases'][int(w)]['scenario']['stand_height_m'])
    s=np.array(states);n=len(s);assert n==603
    refs=np.zeros((n,14));refs[:,:2]=np.array([reference(h)[:2] for h in heights]);refs[:,2]=heights
    q=wp.array(s[:,:17].astype(np.float32));qid=wp.array(ids,dtype=int);ref=wp.array(refs,dtype=D);active=wp.array(np.ones(n,np.int32));tracking=wp.array(np.ones(n,np.int32))
    raw_diag=np.zeros((n,38));raw_diag[:,:6]=s[:,56:62];raw_diag[:,6:12]=s[:,62:68];raw_diag[:,12]=.625
    original=s[:,50:56].astype(np.float32);proof={}
    for mode in [0,1,2]:
        diag=wp.array(raw_diag,dtype=D);ctrl=wp.array(original);st=np.zeros((n,WIDTH));st[:,2]=1;stats=wp.array(st,dtype=D)
        wp.launch(apply_budget,n,[mode,q,qid,ref,active,diag,ctrl,tracking,stats]);out=ctrl.numpy();a=diag.numpy();b=stats.numpy()
        for i in range(n):
            rho=budget(s[i,:17][ids[:4]],reference(heights[i]));applied=rho if mode==1 else (0. if mode==2 else 1.)
            expected=mix(raw_diag[i,:6].astype(np.float32),original[i],applied).astype(np.float32)
            np.testing.assert_array_equal(out[i],expected);assert abs(b[i,1]-applied)<1e-12
        np.testing.assert_array_equal(a[:,:6],raw_diag[:,:6]);np.testing.assert_array_equal(a[:,10:],raw_diag[:,10:]);np.testing.assert_array_equal(out[:,4:],original[:,4:])
        np.testing.assert_array_equal(q.numpy(),s[:,:17].astype(np.float32));np.testing.assert_array_equal(ref.numpy(),refs)
        assert np.all(b[:,0]==1) and np.all(b[:,9]==0) and np.all(b[:,6]==0) and np.all(b[:,8]==0)
        if mode==0:np.testing.assert_array_equal(out,original);np.testing.assert_array_equal(a,raw_diag)
        proof[str(mode)]=dict(rows=n,convex_excess=float(b[:,6].max()),original_lambda_unchanged=True,nominal_wheel_state_reference_identity=True)
    # Explicit zero-leg endpoint and boundary cases are absent from some saved fixtures.
    edgeq=np.zeros((4,17),np.float32);edgeq[0,ids[0]]=1.4;edgeq[1,ids[0]]=1.5;edgeq[2,ids[0]]=np.nan
    edged=np.zeros((4,38));edged[:,:6]=1.;edgec=np.full((4,6),1.,np.float32);edgec[:3,:4]=2.;edger=np.zeros((4,14));edger[3,0]=1.4
    q2=wp.array(edgeq);d2=wp.array(edged,dtype=D);c2=wp.array(edgec);t2=wp.zeros((4,WIDTH),dtype=D);on=wp.array(np.ones(4,np.int32))
    wp.launch(apply_budget,4,[1,q2,qid,wp.array(edger,dtype=D),on,d2,c2,on,t2]);a=c2.numpy();t=t2.numpy()
    np.testing.assert_array_equal(a[1,:4],1);assert t[2,9]==t[3,9]==1;np.testing.assert_array_equal(a[2:],edgec[2:])
    buffers=(q2,d2,c2,t2);refs_weak=[weakref.ref(x) for x in buffers];gc.collect();assert all(x() is not None for x in refs_weak)
    atomic_json(OUT/'gpu_unit.json',dict(verified=True,static_states=603,modes=proof,invalid_reference_and_state_rejected=True,buffer_refs_checked=True,physical_rollout_steps=0,training_updates=0))
    print('PASS GPU603x3 NumPy equivalence/identity/convex/edge checks')


if __name__=='__main__':check()
