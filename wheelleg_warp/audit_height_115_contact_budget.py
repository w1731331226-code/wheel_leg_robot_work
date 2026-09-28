"""新切换轨迹零臂的接触余量和局部制动预算；不外推真实可达性。"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from probe_height_115_action_predict_loow import ROOT, DT, sha
from probe_height_115_live_braking import solve_current
from probe_height_115_live_common_local import common_basis
from probe_height_115_radial_authority import torque_box
from probe_height_115_contact_action_pair import margins
from probe_height_115_passive import geometry
from probe_height_115_braking_budget import ACTIVE
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN


def run():
    folder=ROOT/'wheelleg_warp/results'
    source=folder/'height_115_switching_error_20260929'
    output=folder/'height_115_contact_budget_20260929'
    assert not output.exists()
    report=json.loads((source/'verification.json').read_text())
    fitpath=folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    fit=json.loads(fitpath.read_text())
    archivepath=folder/'height_115_local_states_20260928/verification.json'
    archive=json.loads(archivepath.read_text());rows=[]
    for world in (1,2,3):
        trace=source/f'world_{world}.npz'
        assert sha(trace)==report['worlds'][world]['trace_sha256']
        z=np.load(trace);event=archive['events'][fit['selected_event_ids'][world]]
        m=model(HeightTerrainScenario(**event['scenario']))
        va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
        q=z['pre'][0,:,:m.nq];v=z['pre'][0,:,m.nq:m.nq+m.nv]
        post=z['post'][0]
        before=geometry(m,q)[1];after=geometry(m,post)[1]
        joint=margins(m,post).min(axis=1)
        bad=np.flatnonzero((after.min(axis=1)<HEIGHT_115_GEOMETRIC_MIN)|(joint<0))
        terrain={i for i in range(m.ngeom) if m.geom(i).name.startswith('terrain_')}
        contact=[any(int(r[0])==0 and terrain.intersection(map(int,r[1:])) for r in frame) for frame in z['contact_raw']]
        hits=np.flatnonzero(contact);assert len(hits) and hits[0]>0 and len(bad)
        t=int(hits[0]);failure=int(bad[0]);assert failure>t
        assert np.array_equal(q[t+1],post[t])
        reserve=fit['folds'][world]['reserves'];gain=np.array(fit['folds'][world]['gain'])
        budget=[]
        for delay in (0,1,4):
            k=t+1+delay;nominal=z['pre'][0,k,m.nq+m.nv:m.nq+m.nv+6]
            action,bs,a0,error=solve_current(m,q[k],v[k],v[k-1,va],np.zeros(3),gain,
                (reserve['actual_A_length_m'],reserve['eight_joint_margin_rad']),nominal)
            B=common_basis(m,q[k],v[k]);limit=torque_box(m,v[k])
            A=np.concatenate([B,-B,B,-B]);rhs=np.r_[np.ones(12),limit-nominal,limit+nominal]
            scalar=[]
            for b in bs:
                if b['rate']>=0:continue
                sol=linprog(-b['g'],A_ub=A,b_ub=rhs,bounds=[(None,None)]*3,method='highs')
                assert sol.success
                assert np.max(A@sol.x-rhs)<1e-6
                best=float(b['a0']+b['g']@sol.x)
                required=b['rate']**2/(2*b['h']) if b['h']>0 else None
                scalar.append(dict(name=b['name'],tightened_distance=b['h'],rate=b['rate'],
                    required_constant_braking_acceleration=required,
                    maximum_frozen_model_acceleration=best,
                    scalar_condition_possible=bool(required is not None and best>=required)))
            budget.append(dict(wait_after_first_contact_observation_ms=delay*.5,
                zero_arm_true_A_margin_m=(before[k]-HEIGHT_115_GEOMETRIC_MIN).tolist(),
                remaining_zero_arm_time_to_first_geometry_failure_ms=(failure+1-k)*.5,
                frozen_model_coupled_feasible=error is None,solver_error=error,
                action=None if action is None else action.tolist(),closing_barriers=scalar))
        rows.append(dict(world=world,first_contact_step=t,first_geometry_failure_step=failure,
            contact_to_failure_ms=(failure-t)*.5,
            true_A_pre_contact_margin_m=(before[t]-HEIGHT_115_GEOMETRIC_MIN).tolist(),
            true_A_post_contact_margin_m=(after[t]-HEIGHT_115_GEOMETRIC_MIN).tolist(),
            true_A_single_contact_step_loss_m=(before[t]-after[t]).tolist(),
            true_A_post_contact_step_average_rate_m_s=((after[t]-before[t])/DT).tolist(),
            true_A_following_step_average_rate_m_s=((after[t+1]-before[t+1])/DT).tolist(),
            eight_joint_post_contact_margin_rad=float(joint[t]),budgets=budget,
            trace_sha256=sha(trace)))
    result=dict(role='zero_arm_first_contact_geometry_and_frozen_model_budget',
        nominal_target_m=.115,proxy_lower_m=HEIGHT_115_GEOMETRIC_MIN,
        nominal_headroom_m=.115-HEIGHT_115_GEOMETRIC_MIN,rows=rows,
        limitations=['Post-contact delays follow recorded zero-action dynamics, not a new closed-loop rollout.',
            'Frozen G is already known to fail cross-state error coverage. LP is diagnostic, not certified authority.',
            '1Nm increment trust region plus absolute motor box; not the full hardware action space.',
            'Constant acceleration stopping condition does not include a validated acceleration uncertainty margin.',
            'Finite A-chain differences are interval average rates, not instantaneous measurements.'],
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'verification.json',fitpath,archivepath)},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),
            ROOT/'wheelleg_warp/probe_height_115_live_braking.py',ROOT/'wheelleg_warp/probe_height_115_braking_budget.py',
            ROOT/'wheelleg_warp/probe_height_115_passive.py')})
    output.mkdir();(output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for row in rows:
        print(row['world'],'contact->failure ms',row['contact_to_failure_ms'],
            'before/after um',np.array(row['true_A_pre_contact_margin_m'])*1e6,np.array(row['true_A_post_contact_margin_m'])*1e6,
            'model feasible',[b['frozen_model_coupled_feasible'] for b in row['budgets']],flush=True)


if __name__=='__main__':
    run()
