"""Reference-coordinate position/rate feasibility only; never controls a robot."""
import itertools
import json
import math
import numpy as np
from derive_joint_reference_authority import LOW,HIGH
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/reference_correction_feasibility_v1'
EPS=.02
CAP=.035


def intervals(h,d0,previous,step):
    em,ed=previous;sm,sd=step
    if not all(math.isfinite(x) for x in (h,d0,em,ed,sm,sd)) or min(sm,sd)<0:
        raise ValueError('Finite coordinates and nonnegative correction steps required')
    if not .115<=h<=HIGH or abs(d0)>min(CAP,h-LOW,HIGH-h):
        raise ValueError('Baseline must satisfy the original fixed-mean geometry')
    ml=max(LOW,h-EPS,h+em-sm);mh=min(HIGH,h+EPS,h+em+sm)
    K=min(CAP,(HIGH-LOW)/2,mh-LOW,HIGH-ml)
    dl=max(d0+ed-sd,-K);dh=min(d0+ed+sd,K)
    return (ml,mh),(dl,dh),ml<=mh and dl<=dh


def inequalities(h,d0,previous,step):
    em,ed=previous;sm,sd=step
    A=np.array([[1,1],[-1,-1],[1,-1],[-1,1],[1,0],[-1,0],
                [1,0],[-1,0],[0,1],[0,-1],[0,1],[0,-1]],float)
    b=np.array([HIGH,-LOW,HIGH,-LOW,h+EPS,-h+EPS,h+em+sm,-h-em+sm,
                d0+ed+sd,-d0-ed+sd,CAP,CAP])
    return A,b


def vertex_feasible(A,b):
    # Independent bounded polygon oracle: test every pair of boundary intersections.
    for i,j in itertools.combinations(range(len(b)),2):
        det=A[i,0]*A[j,1]-A[i,1]*A[j,0]
        if det==0:continue
        point=np.array([(b[i]*A[j,1]-A[i,1]*b[j])/det,
                        (A[i,0]*b[j]-b[i]*A[j,0])/det])
        if np.all(A@point<=b+1e-12):return True
    return False


def run():
    assert not (OUT/'derivation.json').exists()
    p=json.loads((OUT/'proposal.json').read_text());assert p['new_physics_budget']==p['new_training_budget']==0
    count=feasible=zero=0
    for h,fraction,em,ed,sm,sd in itertools.product(
            (.115,.16,.24,.3,.38),(-1.,-.5,0.,.5,1.),(-.02,0.,.02),(-.035,0.,.035),(0.,.0002,.02),(0.,.00035,.035)):
        d0=fraction*min(CAP,h-LOW,HIGH-h)
        M,D,ok=intervals(h,d0,(em,ed),(sm,sd));A,b=inequalities(h,d0,(em,ed),(sm,sd))
        assert ok==vertex_feasible(A,b)
        if ok:
            d=(D[0]+D[1])/2;mlo=max(M[0],LOW+abs(d));mhi=min(M[1],HIGH-abs(d))
            assert mlo<=mhi+1e-15;m=(mlo+mhi)/2
            assert np.all(A@np.array([m,d])<=b+1e-12)
            feasible+=1
        count+=1
    for h,fraction,sm,sd in itertools.product((.115,.16,.24,.3,.38),(-1.,0.,1.),(0.,.0002),(0.,.00035)):
        d0=fraction*min(CAP,h-LOW,HIGH-h)
        M,D,ok=intervals(h,d0,(0,0),(sm,sd));assert ok
        A,b=inequalities(h,d0,(0,0),(sm,sd));assert np.all(A@np.array([h,d0])<=b+1e-12)
        zero+=1
    M,D,ok=intervals(.24,.01,(0,.035),(.0002,.00035));assert not ok
    # Zero-rate correction can become incompatible after a baseline change.
    _,_,before=intervals(.24,-.01,(0,.035),(0,0));assert before
    _,_,after=intervals(.24,.01,(0,.035),(0,0));assert not after
    for args in ((.24,0,(0,0),(-1,0)),(.114,0,(0,0),(0,0)),(.38,.01,(0,0),(0,0)),(.24,float('nan'),(0,0),(0,0))):
        try:intervals(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid feasibility input accepted')
    atomic_json(OUT/'derivation.json',dict(verified=True,round=311,grid_checks=count,feasible_points=feasible,
        independent_vertex_oracle_agrees=True,zero_baseline_checks=zero,source_sha256=sha(__file__),
        proposal_sha256=sha(OUT/'proposal.json'),
        equations=dict(mean_interval='M=[max(Lmin,h-eps,h+em_prev-sm),min(Lmax,h+eps,h+em_prev+sm)]',
          projected_difference_cap='K=min(originalcap,(Lmax-Lmin)/2,M_hi-Lmin,Lmax-M_lo)',
          difference_interval='J=[max(d0+ed_prev-sd,-K),min(d0+ed_prev+sd,K)]',
          feasibility='M andJ nonempty; foranyd inJ choosem in[max(Mlo,Lmin+absd),min(Mhi,Lmax-absd)]'),
        counterexample=dict(height=.24,current_baseline_d0=.01,previous_correction=[0,.035],allowed_correction_step=[.0002,.00035],mean_interval=M,difference_interval=D,feasible=ok),
        position_and_rate_simultaneous_unconditionally_admitted=False,
        baseline_zero_from_zero_correction_feasible=True,nonzero_to_zero_instantly_not_guaranteed=True,
        newly_implemented_controller=False,new_physics_steps=0,new_controller_queries=0,new_training_samples=0,new_optimizer_calls=0,
        formal_PPO_admitted=False,novelty_qualified=False,
        limits='Bounded reference-coordinate polygon only; ignores actual closedchain geometry,forces/contacts/dynamics. The algebra counterexample neednot be a reachable0.5ms physical jump. Ordinary interval projection isnotnew theory ortaskbenefit.',
        next='312independent condition/evidence review andfinite close/go-no-go. No newcontroller/filter/gain orPPO basedonlyonthese interval checks.315deepreview/cleanup.'))
    print('PASS311',count,'interval/independentvertex checks,',zero,'zero-baseline checks; emptyintersection preserved;0physics',flush=True)


if __name__=='__main__':run()
