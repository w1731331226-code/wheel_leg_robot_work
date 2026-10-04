"""Round125 diagnostic LP and held-state CPU validation, never a controller."""
from pathlib import Path
import json,sys
import numpy as np
from scipy.optimize import linprog
import mujoco
from review_yaw_sector import ROOT,sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from native.terrain import HeightTerrainScenario,model
from state_estimation import leg_kinematics
import wheelleg_sim as sim

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/joint_response_v1'


def run():
    review=json.loads((OUT/'review.json').read_text());reg=json.loads((OUT/'registration.json').read_text());complete=json.loads((OUT/'completion.json').read_text())
    assert review['verified'] and all(sha(ROOT/n)==v for n,v in reg['source_sha256'].items())
    results=[];validated=0;predicted=0;states_count=0
    for record in complete['records']:
        label=record['label'];z=np.load(OUT/(label+'_states.npz'),allow_pickle=False);responses=json.loads((OUT/(label+'_response.json')).read_text())['outcomes']
        assert sha(OUT/(label+'_states.npz'))==record['states_sha256'] and sha(OUT/(label+'_response.json'))==record['response_sha256']
        models={};ids=z['ids']
        for i,s in enumerate(z['state']):
            states_count+=1;w=int(z['world'][i]);q=s[:17];v=s[17:33];nom=s[56:62];actor=s[62:68];r=responses[i]
            if w not in models:models[w]=model(HeightTerrainScenario(**reg['cases'][w]['scenario']))
            m=models[w]
            matrix=np.vstack([leg_kinematics(q[ids[:2]],np.zeros(2))[3]*[3.4335,1],-leg_kinematics(q[ids[2:4]],np.zeros(2))[3]*[3.4335,1]])
            u0=np.linalg.lstsq(matrix,actor[:4],rcond=None)[0]
            x=[];y=[];base=np.asarray(r['branches']['actual_command']['active_q'])
            for name,b in r['branches'].items():
                if name in ('actual_command','nominal_only'):continue
                ub=np.linalg.lstsq(matrix,np.asarray(b['ctrl'])[:4]-nom[:4],rcond=None)[0]
                x.append(ub-u0);y.append(np.asarray(b['active_q'])-base)
            x=np.asarray(x);y=np.asarray(y);slope=np.linalg.lstsq(x,y,rcond=None)[0].T
            fit_error=float(abs(x@slope.T-y).max());offset=base-slope@u0
            # Scale angular inequalities to microradians to avoid LP default
            # tolerances swallowing the original1.4rad boundary. No gate buffer.
            angular=np.vstack([np.c_[slope*1e6,np.ones(4)],np.c_[-slope*1e6,np.ones(4)]])
            rhs=np.r_[(1.4-offset)*1e6,(1.4+offset)*1e6]
            bounds=np.array([sim.hw.torque_limit(float('inf'),float(v[ids[4+j]]),True,0.,.0005)[0] for j in range(4)])
            torque=np.vstack([np.c_[matrix,np.zeros(4)],np.c_[-matrix,np.zeros(4)]])
            torque_rhs=np.r_[bounds-nom[:4],bounds+nom[:4]]
            solved=linprog([0,0,-1],A_ub=np.vstack([angular,torque]),b_ub=np.r_[rhs,torque_rhs],bounds=[(-1,1),(-1,1),(None,None)],
                method='highs',options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
            assert solved.success,solved.message
            u=solved.x[:2];qpred=offset+slope@u;command=np.r_[nom[:4]+matrix@u,nom[4:6]]
            assert abs(u).max()<=1+1e-9 and np.all(abs(command[:4])<=bounds+1e-7)
            data=mujoco.MjData(m);data.qpos[:]=q;data.qvel[:]=v;data.qacc_warmstart[:]=s[33:49];data.ctrl[:]=command;data.time=s[49]
            mujoco.mj_step(m,data);qnext=data.qpos[ids[:4]]
            margin=float(1.4-abs(qnext).max());model_margin=float(1.4-abs(qpred).max());predicted+=int(model_margin>=0);validated+=int(margin>=0)
            results.append(dict(label=label,case=r['case'],event=int(z['event'][i]),time_s=r['time_s'],
                original_gpu_margin_rad=float(1.4-abs(s[68:85][ids[:4]]).max()),original_cpu_margin_rad=float(1.4-abs(base).max()),
                candidate_coordinates=u.tolist(),command=command.tolist(),predicted_margin_rad=model_margin,validated_cpu_margin_rad=margin,
                local_fit_residual_rad=fit_error,extrapolation_validation_error_rad=float(abs(qnext-qpred).max()),
                maximum_change_from_center=float(abs(u-u0).max()),local_probe_radius=.01,
                note='LP may extrapolate beyond the0.01 measurement radius; this row reports newCPU oracle validation, not a prospective bound or deployed decision.'))
        print('VERIFIED reachability',label,len(z['state']),flush=True)
    assert states_count==27
    report=dict(verified=True,states=27,new_cpu_validation_steps=27,training_updates=0,new_gpu_rollouts=0,
        predicted_cpu_safe=predicted,validated_cpu_safe=validated,records=results,joint_response_review_sha256=sha(OUT/'review.json'),reviewer_sha256=sha(__file__),
        method='Local input-response least squares from four bounded perturbations; LP maximizes minimum next-joint margin under original2D coordinates and actuator boxes; exact original1.4rad threshold, angular rows scaled1e6 only for arithmetic.',
        limits='Actual simulated model and geometry used offline. One-step existence in27 captured states does not prove recursive feasibility, unknown-contact robustness, CPU-GPU transfer or task-controller benefits. Empirical errors not certified disturbance bounds.',
        next='Evaluate practical fixed-design/causal-state predictor and independent held-out error before any deployment or new learning; retain all stopped candidates and original gates.')
    (OUT/'reachability_audit_125.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('PASS states27, predicted/CPU validated',predicted,validated,flush=True)


if __name__=='__main__':run()
