"""Recorded first-crossing fixtures: WP projection vs independent scipy QP."""
from pathlib import Path
import argparse,json
import numpy as np
import warp as wp
from scipy.optimize import minimize
from leg_residual_cone import supervise
from native.controller import D
from review_yaw_sector import ROOT,sha
import sys
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from state_estimation import leg_kinematics


def run(out):
    source=ROOT/'wheelleg_warp/results/paper_recovery_20261004/design_boundary_v1'
    review=json.loads((source/'review.json').read_text());assert review['verified']
    fixture=[]
    for label in ['1609','1610','1611']:
        z=np.load(source/(label+'_trace.npz'));cases=review['first_crossing_cases'][label]
        for i,r in enumerate(cases):
            if 'first_crossing' not in r:continue
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]];fixture.append(t[np.flatnonzero(t[:,47]<0)[0]])
    t=np.asarray(fixture);n=len(t);assert n==15
    q=t[:,2:6].astype(np.float32);v=t[:,6:10].astype(np.float32)
    ids=np.array([0,1,2,3,0,1,2,3],np.int32);memory=np.zeros((n,19));memory[:,16:19]=t[:,41:44]
    diag=np.zeros((n,38));diag[:,:6]=t[:,18:24];diag[:,6:12]=t[:,24:30];diag[:,12]=t[:,36]
    buffers=[wp.array(q),wp.array(v),wp.array(ids),wp.ones(n,dtype=int),wp.array(memory,dtype=D),wp.array(diag,dtype=D),wp.array(t[:,30:36].astype(np.float32)),wp.ones((n,6),dtype=D),wp.zeros((n,7),dtype=D)]
    wp.launch(supervise,n,[1,*buffers]);accepted=buffers[5].numpy();u=buffers[8].numpy()[:,5:7];records=[]
    for i in range(n):
        matrix=np.vstack([leg_kinematics(q[i,:2],np.zeros(2))[3]*[3.4335,1],-leg_kinematics(q[i,2:],np.zeros(2))[3]*[3.4335,1]])
        nominal=diag[i,:4];raw=t[i,36]*t[i,41:43];near=abs(q[i])+np.maximum(0,np.sign(q[i])*v[i])*.02>=1.4
        guarded=near&(np.sign(q[i])*nominal<0);a=matrix[guarded]*np.sign(q[i,guarded,None])
        metric=matrix.T@matrix
        constraints=[dict(type='ineq',fun=lambda x,a=a:-a@x),dict(type='ineq',fun=lambda x,m=matrix,b=nominal:40-abs(b+m@x))]
        solved=minimize(lambda x:(x-raw)@metric@(x-raw),np.zeros(2),jac=lambda x:2*metric@(x-raw),bounds=[(-1,1)]*2,
            constraints=constraints,method='SLSQP',options=dict(ftol=1e-12,maxiter=200))
        assert solved.success,solved.message
        np.testing.assert_allclose(u[i],solved.x,rtol=0,atol=2e-6)
        np.testing.assert_allclose(accepted[i,6:10],matrix@u[i],rtol=0,atol=1e-9)
        assert np.max(a@u[i])<=1e-9 and np.max(abs(u[i]))<=1+1e-9
        assert np.max(abs(nominal+matrix@u[i]))<=40+1e-8
        assert (u[i]-raw)@metric@(u[i]-raw)<=raw@metric@raw+1e-9
        np.testing.assert_array_equal(accepted[i,:6],diag[i,:6])
        records.append(dict(index=i,guarded=guarded.tolist(),accepted=u[i].tolist(),independent_qp=solved.x.tolist(),cost=float((u[i]-raw)@metric@(u[i]-raw))))
    out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(dict(verified=True,recorded_fixtures=15,records=records,
        maximum_guard_violation=float(buffers[8].numpy()[:,3].max()),training_updates=0,physics_episodes=0,
        source_sha256={'wheelleg_warp/leg_residual_cone.py':sha(ROOT/'wheelleg_warp/leg_residual_cone.py'),'wheelleg_warp/test_leg_residual_cone.py':sha(__file__)},
        boundary_review_sha256=sha(source/'review.json'),scope='2D metric-optimal projection/box bounds and nominal identity on recorded guarded fixtures; not a joint acceleration/state invariant or robot rollout.'),indent=2)+'\n')
    print('PASS15 recorded fixtures vs independent QP, box/cone/nominal identity',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
