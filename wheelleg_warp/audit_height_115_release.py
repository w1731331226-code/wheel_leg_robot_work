"""保留严格复现失败，用既有数值配对门独立审计已保存的撤动作轨迹。"""
import argparse
import json
from pathlib import Path
import numpy as np
from probe_height_115_action_predict_loow import ROOT, sha, START_LIMITS
from probe_height_115_live_common_local import common_basis
from probe_height_115_radial_authority import torque_box, pose
from probe_height_115_passive import geometry
from probe_height_115_contact_action_pair import margins
from probe_height_115_braking_budget import ACTIVE, barriers
from native.terrain import model, HeightTerrainScenario, HEIGHT_115_GEOMETRIC_MIN


def run(one_second=False):
    folder=ROOT/'wheelleg_warp/results'
    source=folder/('height_115_contact_release_1s_20260929' if one_second else 'height_115_contact_release_20260929')
    prior=folder/'height_115_contact_hold_v2_20260929'
    output=folder/('height_115_contact_release_1s_audit_20260929' if one_second else 'height_115_contact_release_audit_20260929')
    assert not output.exists()
    failed=json.loads((source/'verification.json').read_text());previous=json.loads((prior/'verification.json').read_text())
    assert sha(source/'trace.npz')==failed['trace_sha256'] and sha(prior/'trace.npz')==previous['trace_sha256']
    fitpath=folder/'height_115_action_predict_1nm_single_graph_20260929/verification.json'
    arcpath=folder/'height_115_local_states_20260928/verification.json'
    fit=json.loads(fitpath.read_text());archive=json.loads(arcpath.read_text())
    scenario=HeightTerrainScenario(**archive['events'][fit['selected_event_ids'][3]]['scenario']);m=model(scenario)
    # Read each compressed array once, instead of decompressing it inside every step.
    with np.load(source/'trace.npz') as data:z={k:data[k] for k in data.files}
    with np.load(prior/'trace.npz') as data:old={k:data[k] for k in data.files}
    steps=2040 if one_second else 240
    assert z['pre'].shape[1]==steps
    difference=dict(qpos=float(max(abs(z['pre'][:,:40,:m.nq]-old['pre'][:,:,:m.nq]).max(),abs(z['post'][:,:40]-old['post']).max())),
        qvel=float(max(abs(z['pre'][:,:40,m.nq:m.nq+m.nv]-old['pre'][:,:,m.nq:m.nq+m.nv]).max(),abs(z['post_velocity'][:,:40]-old['post_velocity']).max())),
        ctrl=float(max(abs(z['pre'][:,:40,m.nq+m.nv:m.nq+m.nv+6]-old['pre'][:,:,m.nq+m.nv:m.nq+m.nv+6]).max(),abs(z['applied'][:,:40]-old['applied']).max())))
    assert all(difference[k]<=START_LIMITS[k] for k in difference)
    assert np.array_equal(z['pre'][:,:40,-1],old['pre'][:,:,-1])
    def contacts(trace,t,arm):return {tuple(sorted((int(a),int(b)))) for w,a,b in trace['contact_raw'][t] if int(w)==arm}
    assert all(contacts(z,t,a)==contacts(old,t,a) for t in range(40) for a in range(7))
    qa=np.array([m.joint(n).qposadr[0] for n in ACTIVE]);va=np.array([m.joint(n).dofadr[0] for n in ACTIVE])
    wheels={m.geom('wheel_collide_'+s).id for s in ('L','R')};summaries=[];allrows=[]
    for arm in (0,1):
        rows=[];last_leg=geometry(m,z['pre'][arm,0,:m.nq])[1]
        for t in range(steps):
            pre=z['pre'][arm,t];q=pre[:m.nq];v=pre[m.nq:m.nq+m.nv];nom=pre[m.nq+m.nv:m.nq+m.nv+6]
            ctrl=z['applied'][arm,t];nxt=z['post'][arm,t];vn=z['post_velocity'][arm,t]
            action=z['schedule'][t,arm] if t<40 else np.zeros(3)
            leg=geometry(m,nxt)[1];joint=float(margins(m,nxt).min());att=max(pose(nxt,scenario)['world_abs_deg'])
            nonwheel=any(not wheels.intersection(pair) for pair in contacts(z,t,arm))
            match=float(abs(ctrl-nom-common_basis(m,q,v)@action).max());peak=float(abs(ctrl-nom).max())
            box=float((abs(ctrl)-torque_box(m,v)).max())
            if t>=40:assert np.array_equal(ctrl,nom)
            b,_=barriers(nxt[qa],vn[va],np.zeros(4),np.zeros((4,3)),(0,0))
            gates=dict(geometry=bool(leg.min()>=HEIGHT_115_GEOMETRIC_MIN and joint>=0),pose=att<=5,
                nonwheel=not nonwheel,motor=bool(peak<=2+1e-5 and box<=1e-6 and match<=1e-5),active=bool(pre[-1]==1))
            rows.append(dict(step=t,time_ms=(t+1)*.5,gates=gates,all_gates=all(gates.values()),
                true_A_margin_m=(leg-HEIGHT_115_GEOMETRIC_MIN).tolist(),target_error_m=(leg-.115).tolist(),
                interval_A_rate_m_s=((leg-last_leg)/.0005).tolist(),instantaneous_FK_rate_m_s=[x['rate'] for x in b[:2]],
                joint_min_margin_rad=joint,world_attitude_max_deg=att,extra_motor_peak_Nm=peak,
                motor_box_excess_Nm=box,command_error_Nm=match))
            last_leg=leg
        summaries.append(dict(arm=('zero','hold20ms_then_release')[arm],
            total_duration_ms=steps*.5,recovery_duration_ms=(steps-40)*.5,
            all_window_safe=all(r['all_gates'] for r in rows),recovery_safe=all(r['all_gates'] for r in rows[40:]),
            first_failure=next((r for r in rows if not r['all_gates']),None),
            minimum_A_margin_m=min(min(r['true_A_margin_m']) for r in rows),
            recovery_target_RMSE_m=float(np.sqrt(np.mean(np.array([r['target_error_m'] for r in rows[40:]])**2))),
            recovery_max_abs_target_error_m=float(np.max(abs(np.array([r['target_error_m'] for r in rows[40:]])))),
            last100ms_max_abs_target_error_m=float(np.max(abs(np.array([r['target_error_m'] for r in rows[-200:]])))),
            last100ms_max_abs_A_rate_m_s=float(np.max(abs(np.array([r['interval_A_rate_m_s'] for r in rows[-200:]])))),
            terminal=rows[-1]))
        allrows.append(rows)
    result=dict(role='independent_numeric_pairing_release_audit',strict_prefix_pass=all(v==0 for v in difference.values()),
        previous_audit_source_git_revision='053d8e8',
        historical_numeric_pairing_pass=True,prefix_errors=difference,
        prefix_limits={k:START_LIMITS[k] for k in difference},prefix_contacts_all_equal=True,
        extra_action_removed_exactly_after20ms=True,summaries=summaries,rows=allrows,
        limitations='Uses pre-existing numeric tolerances and unchanged physical gates; reports bitwise equality separately. Earlier strict-prefix failure remains archived. No full-task/online claim.',
        input_sha256={str(p.relative_to(ROOT)):sha(p) for p in (source/'verification.json',source/'trace.npz',prior/'verification.json',prior/'trace.npz',fitpath,arcpath)},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__),ROOT/'wheelleg_warp/probe_height_115_action_predict_loow.py',ROOT/'wheelleg_warp/probe_height_115_live_common_local.py')})
    output.mkdir();(output/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=None if one_second else 2)+'\n')
    print(summaries,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--one-second',action='store_true')
    run(parser.parse_args().one_second)
