"""Candidate nominal-reference joint-room budget, not a state safety barrier.

Only current active joints and the public nominal standing reference are used.
Convex command contraction preserves the accepted endpoint command box at the
same state; coupled next-state invariance and recovery do not follow.
"""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import wheelleg_sim as sim

CAP=1.4


def reference(height):
    if not np.isfinite(height) or not .115<=height<=.38:raise ValueError('Height outside fixed paper domain')
    pair=(0.,0.) if height==sim.L_STAND else sim.ik(float(height))
    return np.tile(pair,2).astype(np.float64)


def budget(q,qref):
    q=np.asarray(q,np.float64);qref=np.asarray(qref,np.float64)
    if q.shape!=(4,) or qref.shape!=(4,) or not np.isfinite(q).all() or not np.isfinite(qref).all():raise ValueError('Expected finite four-joint state/reference')
    reserve=CAP-np.abs(qref)
    if np.any(reserve<=0):raise ValueError('Nominal reference has no declared working-domain room')
    return float(np.clip(np.min((CAP-np.abs(q))/reserve),0.,1.))


def mix(nominal_command,accepted_command,rho):
    n=np.asarray(nominal_command,np.float64);t=np.asarray(accepted_command,np.float64)
    if n.shape!=(6,) or t.shape!=(6,) or not np.isfinite(n).all() or not np.isfinite(t).all() or not np.isfinite(rho) or not 0<=rho<=1:raise ValueError('Invalid accepted commands/budget')
    out=t.copy()
    if rho==0:out[:4]=n[:4]
    elif rho!=1:out[:4]=n[:4]+rho*(t[:4]-n[:4])
    if not np.isfinite(out).all():raise ValueError('Nonfinite mixed command')
    return out


def self_check():
    for h in np.linspace(.115,.38,266):
        qref=reference(float(h));assert np.all(np.abs(qref)<CAP) and budget(qref,qref)==1.
        assert budget(np.array([CAP,0,0,0]),qref)==0 and budget(np.array([CAP+.01,0,0,0]),qref)==0
        assert budget(-qref,-qref)==1.
        rho=[budget(np.r_[x,qref[1:]],qref) for x in np.linspace(abs(qref[0]),CAP,50)]
        assert np.all(np.diff(rho)<=1e-14)
    n=np.array([1.,-2.,3.,-4.,.2,-.3]);t=np.array([-2.,4.,-1.,5.,-.4,.5])
    np.testing.assert_array_equal(mix(n,t,1),t);np.testing.assert_array_equal(mix(n,t,0),np.r_[n[:4],t[4:]])
    for rho in np.linspace(0,1,101):
        out=mix(n,t,float(rho));assert np.all(out[:4]>=np.minimum(n[:4],t[:4])-1e-12) and np.all(out[:4]<=np.maximum(n[:4],t[:4])+1e-12)
        np.testing.assert_array_equal(out[4:],t[4:]);np.testing.assert_array_equal(mix(n,n,float(rho)),n)
    for q,qref in [([np.nan]*4,[0]*4),([0]*4,[CAP]*4)]:
        try:budget(q,qref)
        except ValueError:pass
        else:raise AssertionError('Invalid domain accepted')
    print('PASS reference-domain, monotonicity, nominal/zero identity, wheel identity and convex-box algebra')


def fixtures(output):
    self_check();base=ROOT/'wheelleg_warp/results/paper_recovery_20261004';rows=[]
    for cohort,folder in [('new_saved_states','contact_state_holdout_v2'),('old_static_fixtures','joint_response_v1')]:
        out=base/folder;reg=json.loads((out/'registration.json').read_text());done=json.loads((out/('collection_completion.json' if folder=='contact_state_holdout_v2' else 'completion.json')).read_text())
        for job in done['records']:
            path=(out/'collection'/f'{job["label"]}_states.npz') if folder=='contact_state_holdout_v2' else out/f'{job["label"]}_states.npz'
            with np.load(path,allow_pickle=False) as z:
                for i,state in enumerate(z['state']):
                    case=reg['cases'][int(z['world'][i])];h=case['scenario']['stand_height_m'];q=state[:17][z['ids'][:4]];rho=budget(q,reference(h))
                    # Nominal-only executed command uses the existing float32 actuator interface.
                    nominal=state[56:62].astype(np.float32).astype(np.float64);accepted=state[50:56];new=mix(nominal,accepted,rho)
                    assert np.all(new[:4]>=np.minimum(nominal[:4],accepted[:4])-1e-12) and np.all(new[:4]<=np.maximum(nominal[:4],accepted[:4])+1e-12)
                    np.testing.assert_array_equal(new[4:],accepted[4:])
                    rows.append(dict(cohort=cohort,label=job['label'],index=i,case=case['seed'],event=int(z['event'][i]),height_m=h,rho=rho))
    assert len(rows)==603;summary={}
    for cohort in ['new_saved_states','old_static_fixtures']:
        a=[r['rho'] for r in rows if r['cohort']==cohort];summary[cohort]=dict(states=len(a),min_rho=min(a),mean_rho=float(np.mean(a)),max_rho=max(a),zero=sum(x==0 for x in a),unchanged=sum(x==1 for x in a))
    output.write_text(json.dumps(dict(algebra_verified=True,states=603,summary=summary,records=rows,new_physics_steps=0,training_updates=0,
        scope='Static algebra only; old cohort is numerical fixture. No rollout, model-error/safety/learned superiority or nondegenerate trajectory qualification. Float32 final execution retains existing actuator rounding; virtual-coordinate identity is only before existing mapping/rounding.'),indent=2)+'\n');print(summary)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);args=p.parse_args();fixtures(args.output) if args.output else self_check()
