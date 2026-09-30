"""Fixed newly feasible joint-HOCBF candidate: cold CPU/Warp one-step check."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from probe_height_115_action_predict_loow import sha,DT,ACTIVE
from probe_height_115_late_response import paired_step
from probe_height_115_braking_budget import barriers
from probe_height_115_live_braking import joint_hocbf_requirement
from probe_height_115_contact_action_pair import basis,margins
from probe_height_115_passive import geometry
from probe_height_115_radial_authority import torque_box,pose
from native.environment import PUBLIC_ACTUATOR_GAIN_UPPER
from native.terrain import model,HeightTerrainScenario,HEIGHT_115_GEOMETRIC_MIN


def run(output,archive_warmstart=False):
    assert not output.exists(),output
    folder=ROOT/'wheelleg_warp/results'
    source=folder/'height_joint_hocbf_heldout_20261001/verification.json'
    report=json.loads(source.read_text())
    for p,h in report['source_sha256'].items():assert sha(ROOT/p)==h,p
    paths=[source,folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json',
        folder/'height_115_local_states_20260928/verification.json']
    fitted,archive=[json.loads(p.read_text()) for p in paths[1:]]
    # First additional feasible, nonzero candidate; selected before physical testing.
    row=next(r for r in report['six_world_rows'] if r['world']==0 and r['state']=='15ms_before_failure')
    assert not row['immediate_public']['feasible'] and row['joint_hocbf_public']['feasible']
    c=np.array(row['joint_hocbf_public']['action']);assert np.any(c)
    event_id=fitted['selected_event_ids'][0];event=archive['events'][event_id]
    m=model(HeightTerrainScenario(**event['scenario']));scenario=HeightTerrainScenario(**event['scenario'])
    windows=paths[2].parent/'windows.npz';assert sha(windows)==archive['windows_sha256']
    with np.load(windows) as z:pre=z[f'event_{event_id}_pre'].copy()
    cols=archive['columns'];i=int(np.argmin(abs(pre[:,0]-row['time_s'])));assert i>0 and abs(pre[i,0]-row['time_s'])<1e-6
    q=pre[i,cols['qpos_start']:cols['qvel_start']];v=pre[i,cols['qvel_start']:cols['warmstart_start']]
    nom=pre[i,cols['ctrl_start']:cols['ctrl_start']+6];B=basis(m,q,v)
    warm=pre[i,cols['warmstart_start']:cols['ctrl_start']] if archive_warmstart else None
    bound=torque_box(m,v)/np.array(PUBLIC_ACTUATOR_GAIN_UPPER)
    actions=np.array([np.zeros(3),c,np.zeros(3)]);request=nom+actions@B.T;issued=request.astype(np.float32)
    assert np.max(abs(issued)-bound)<=1e-6 and np.max(abs(issued-request))<=1e-6
    cq,cv,gq,gv,cp,gp,cf,gf=paired_step(m,scenario,q,v,issued,warmstart=warm)
    pairing=dict(q_error=float(abs(cq-gq).max()),v_error=float(abs(cv-gv).max()),contacts_equal=cp==gp)
    paired=pairing['q_error']<=2e-6 and pairing['v_error']<=1e-3 and pairing['contacts_equal']
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    r=fitted['folds'][0]['reserves'];reserve=(r['actual_A_length_m'],r['eight_joint_margin_rad'])
    initial,_=barriers(q[qa],v[va],np.zeros(4),np.zeros((4,3)),reserve);rows=[]
    middle=np.array([(m.body_pos[m.body('leg'+s).id]+m.body_pos[m.body('leg'+s+'_D').id])/2 for s in ('L','R')])
    wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')}
    for backend,Q,V,contacts,forces in [('cpu',cq,cv,cp,cf),('warp',gq,gv,gp,gf)]:
        for arm in (0,1):
            acceleration=(V[arm,va]-v[va])/DT
            nxt,_=barriers(Q[arm,qa],V[arm,va],acceleration,np.zeros((4,3)),reserve)
            radial=np.array([(nxt[j]['rate']-initial[j]['rate'])/DT for j in range(2)])
            joint_acc=np.stack([-acceleration,acceleration],axis=1).ravel()
            joint_slack=[a-joint_hocbf_requirement(b['h'],b['rate'])[1] for a,b in zip(joint_acc,initial[2:])]
            leg_slack=[radial[j]-initial[j]['rate']**2/(2*initial[j]['h']) for j in range(2) if initial[j]['rate']<0]
            lengths=geometry(m,Q[arm]);joint=float(margins(m,Q[arm]).min());att=max(pose(Q[arm],scenario)['world_abs_deg'])
            nonwheel=any(not wheels.intersection(p) for p in contacts[arm])
            physical=bool(min(lengths[1].min(),np.linalg.norm(lengths[5]-middle,axis=-1).min())>=HEIGHT_115_GEOMETRIC_MIN and joint>=0 and att<=5 and not nonwheel)
            # Motor output checked against the true nominal envelope; command against the public robust box.
            motor=bool(np.max(abs(forces[arm])-torque_box(m,v))<=1e-6 and np.max(abs(issued[arm]-nom))<=1+1e-6)
            rows.append(dict(backend=backend,arm=arm,physical_gate=physical,motor_gate=motor,
                joint_hocbf_proxy_gate=bool(min(joint_slack)>=-1e-5),leg_stopping_proxy_gate=bool(not leg_slack or min(leg_slack)>=-1e-5),
                minimum_joint_proxy_slack=float(min(joint_slack)),minimum_leg_proxy_slack=float(min(leg_slack)) if leg_slack else None,
                actual_A_margin_m=(lengths[1]-HEIGHT_115_GEOMETRIC_MIN).tolist(),min_eight_joint_margin_rad=joint,attitude_deg=att))
    candidate=[r for r in rows if r['arm']==1]
    passed=paired and all(r[k] for r in candidate for k in ('physical_gate','motor_gate','joint_hocbf_proxy_gate','leg_stopping_proxy_gate'))
    output.mkdir(parents=True)
    np.savez_compressed(output/'trace.npz',initial_q=q,initial_v=v,nominal=nom,actions=actions,issued=issued,
        initial_warmstart=np.zeros(m.nv) if warm is None else warm,
        cpu_q=cq,cpu_v=cv,warp_q=gq,warp_v=gv,cpu_force=cf,warp_force=gf)
    result=dict(role='fixed_joint_hocbf_candidate_cold_single_step',status='single_step_proxy_pass' if passed else 'failed_single_step_proxy',
        world=0,source_time_s=row['time_s'],action=c.tolist(),pairing=pairing,rows=rows,
        numerical_warmstart='archived on both backends' if archive_warmstart else 'zero on both backends',
        limitations='Archived state and onset remain offline. Numerical warmstart is initialization only, never an optimizer observation. Frozen nominal torque, one0.5ms step only. Discrete average acceleration proxies do not prove continuous or recursive safety; no default control or full-height claim.',
        trace_sha256=sha(output/'trace.npz'),input_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths+[windows]},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_late_response.py',ROOT/'wheelleg_warp/probe_height_115_live_braking.py')})
    (output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(result['status'],pairing,c.tolist(),candidate,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--archive-warmstart',action='store_true')
    args=parser.parse_args();run(args.output,args.archive_warmstart)
