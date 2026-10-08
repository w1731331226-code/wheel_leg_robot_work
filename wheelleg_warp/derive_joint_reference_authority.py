"""Length-coordinate feasibility and explicitly quasi-steady wheel authority."""
import json
import math
import sys
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT,sha
from dashboard.live_env import atomic_json
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
import hardware_profile as hw

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/continuous_nominal_task_pair_v1'
LOW=.1147044660616607
HIGH=.38


def mean_interval(height,difference,tolerance=.02):
    if not all(math.isfinite(x) for x in (height,difference,tolerance)) or tolerance<0:
        raise ValueError('Finite height/difference and nonnegative tolerance required')
    return max(LOW+abs(difference),height-tolerance),min(HIGH-abs(difference),height+tolerance)


def max_difference(height,tolerance=.02):
    if not math.isfinite(height) or not math.isfinite(tolerance) or tolerance<0:
        raise ValueError('Finite height and nonnegative tolerance required')
    return min((HIGH-LOW)/2,height+tolerance-LOW,HIGH-height+tolerance)


def sustained_force_cap(speed,normal_lower,mu_lower=.4):
    if not all(math.isfinite(x) for x in (speed,normal_lower,mu_lower)) or min(normal_lower,mu_lower)<0:
        raise ValueError('Finite speed and nonnegative lower normal/friction bounds required')
    torque=hw.torque_limit(float('inf'),speed,False,0.,.0005)[0]
    # Only quasi-steady powered rolling: inertia/braking contact impulses excluded.
    return min(torque/hw.WHEEL_RADIUS,mu_lower*normal_lower)


def run():
    assert not (OUT/'round289_joint_constraint_derivation.json').exists()
    # Equivalent interval and closed-form feasibility conditions, including endpoints.
    for height in np.linspace(.115,.38,19):
        for delta in np.linspace(-.14,.14,41):
            lo,hi=mean_interval(float(height),float(delta))
            bound=max_difference(float(height))
            assert (lo<=hi)==(abs(delta)<=bound)
            if lo<=hi:
                m=(lo+hi)/2
                assert LOW-1e-15<=m+delta<=HIGH+1e-15 and LOW-1e-15<=m-delta<=HIGH+1e-15
                assert abs(m-height)<=.02+1e-15
    for h in (.115,.38):
        lo,hi=mean_interval(h,.035);assert lo>hi
        lo,hi=mean_interval(h,.01);assert lo<=hi
    assert max_difference(.38)==.02
    for speed in (0.,14.,49.,68.,72.,75.):
        assert sustained_force_cap(speed,0.)==0.
    assert sustained_force_cap(72.,30.)<sustained_force_cap(14.,30.)
    assert sustained_force_cap(75.,30.)==0.
    for args in ((float('nan'),.01,.02),(.2,.01,-1.)):
        try:mean_interval(*args)
        except ValueError:pass
        else:raise AssertionError('Invalid interval input accepted')
    points=[]
    for h in (.115,.16,.25,.3,.38):
        points.append(dict(height=h,max_delta_m=max_difference(h),
            requested035_mean_interval=mean_interval(h,.035),
            requested010_mean_interval=mean_interval(h,.01)))
    authority=[]
    for speed in (0.,14.,49.,68.,72.,75.):
        cap=hw.torque_limit(float('inf'),speed,False,0.,.0005)[0]
        authority.append(dict(speed_rad_s=speed,command_torque_cap_Nm=cap,
            quasi_steady_force_cap_at_normal_lower0_N=sustained_force_cap(speed,0.),
            quasi_steady_force_cap_at_normal_lower30_N=sustained_force_cap(speed,30.)))
    atomic_json(OUT/'round289_joint_constraint_derivation.json',dict(
        verified=True,round=289,source_sha256=sha(__file__),
        hardware_profile_sha256=sha(ROOT/'wheelleg_ppo/tools/hardware_profile.py'),
        length_bounds=[LOW,HIGH],height_tracking_tolerance=.02,
        equations=dict(coordinates='m=(L_left+L_right)/2,d=(L_left-L_right)/2',
            mean_interval='[max(Lmin+abs(d),h-eps),min(Lmax-abs(d),h+eps)]',
            differential_feasibility='abs(d)<=min((Lmax-Lmin)/2,h+eps-Lmin,Lmax-h+eps)',
            wheel_balance='R*F_tangent=tau_motor-J*omega_dot-b*omega-otherlosses',
            yaw_moment='Mz=(trackwidth/2)*(F_right-F_left),straight contact frame only',
            quasi_steady_capacity='C_i=min(tau_speed_i/R,mu_lower*N_lower_i);ifN_lower_i=0,C_i=0'),
        length_examples=points,authority_examples=authority,algebra_grid_checks=779,
        track_width_m=hw.TRACK_WIDTH,wheel_radius_m=hw.WHEEL_RADIUS,
        policy_data_boundary='Currentraw packet contains measuredangles/gyro/bodyvx-vy-vz/activejointq-v/wheelomega,publiccommands andacceptedinputhistory. Diagnosticnormal/friction/truecaseparameters are notactor observations. No certifiedpositiveN_lower exists inthis study.',
        assumptions='Lengthrectangle isnecessaryonly:ignoreclosed-chainangle feasibility/dynamics. Capacity bound isquasi-steady poweredrolling only,notinstantaneousbraking/impactfriction bound;dynamicJomegadot/coupling mustbe included forrealcontrol. Yawmoment expression assumes straight matchedcontact directions.',
        conclusion='Feasibleheightdifference alonecannotguaranteeheading traction. Unknownnormal lower0 givestrivial robustgroundforceauthority;anynonzero guarantee requires justifiedcontact/loadinterval orweaker explicitlystatedproperty. Genericobserver/history/QP/governor+PPO isnot establishednovelty.',
        new_simulation=0,new_optimizer=0,new_training_samples=0,controller_admitted=False,
        next='290 deepreview whethera distinguishablepublic-observation coupledreference/wrench mechanism canbe specified;do notimplementsingleaxisclip/fixeddamping orpromise robustheading withoutload/dynamicproof.'))
    print('PASS289779lengthfeasibility checks/endpoints/conditionalmotorcapacity; no control/simulation/PPO',flush=True)


if __name__=='__main__':
    run()
