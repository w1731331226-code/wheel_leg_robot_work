"""Independent receipt of the registered 18 fixed route-reference episodes."""
import json
import numpy as np
import mujoco
from run_classical_route_reference import OUT, inputs
from review_yaw_sector import ROOT, sha
from score_reference_learning import load_rows
from complete_contact_witness import check_files
from run_phase_support_qualification import logs
from parking_withdrawal_probe import check_log
from review_fixed_force_evaluation import chain_minima, physical_metrics
from analyze_reward_failures import flags
from probe_route_feedback import actions
from dashboard.live_env import atomic_json
from native.terrain import HEIGHT_115_GEOMETRIC_MIN
import wheelleg_sim as sim


def check_actions(actor, route, target, classical):
    assert actor.ndim==2 and actor.shape[1]==968 and route.shape==(len(actor),43)
    assert np.isfinite(actor).all() and np.isfinite(route).all()
    np.testing.assert_array_equal(actor[:,:481],actor[:,481:962])
    raw=actor[:,351:390].astype(np.float64)
    np.testing.assert_array_equal(route[:,:39],raw)
    np.testing.assert_array_equal(route[:,39],target)
    raw[:,38]-=target
    np.testing.assert_array_equal(route[:,40],raw[:,38])
    command,reference=actions(raw,classical['candidate'],classical['arm'])
    np.testing.assert_array_equal(route[:,41],reference)
    np.testing.assert_array_equal(route[:,42],command[:,2])
    expected=np.zeros((len(actor),6),np.float32);expected[:,4]=command[:,2];expected[:,5]=-command[:,2]
    np.testing.assert_array_equal(actor[:,962:],expected)


def self_check():
    classical=dict(candidate=dict(name='B1_3',kp=.4,kd=.3,roll_gain=0),arm=1)
    actor=np.zeros((3,968),np.float32);actor[:,360]=[.7,-.7,0];actor[:,389]=[.01,-.02,.03]
    actor[:,481:962]=actor[:,:481]
    raw=actor[:,351:390].astype(float);shift=raw.copy();shift[:,38]-=.05
    command,reference=actions(shift,classical['candidate'],1)
    route=np.column_stack((raw,np.full(3,.05),shift[:,38],reference,command[:,2]))
    actor[:,966]=command[:,2];actor[:,967]=-command[:,2]
    check_actions(actor,route,.05,classical)
    for kind in ('offset','raw','leg','wheel','reference'):
        x,y=actor.copy(),route.copy();target=.05
        if kind=='offset':target=-.05
        elif kind=='raw':y[0,38]+=.01
        elif kind=='leg':x[0,962]=.1
        elif kind=='wheel':x[0,966]*=-1
        else:y[0,41]+=.01
        try:check_actions(x,y,target,classical)
        except AssertionError:pass
        else:raise AssertionError('Corrupted '+kind+' accepted')
    print('PASS357 wrong offset/raw/reference/leg/wheel rejected; 0 physics',flush=True)


def run():
    self_check();p=inputs();assert not (OUT/'review.json').exists()
    c=json.loads((OUT/'completion.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert c['verified'] and c['evaluations']==len(c['records'])==18
    assert c['new_model_forward_rows']==c['new_learning_samples']==0
    assert c['proposal_sha256']==a['proposal_sha256']==sha(OUT/'proposal.json')
    assert c['source_admission_sha256']==sha(OUT/'source_admission.json') and a['verified']
    assert all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    assert [r['condition'] for r in c['records']]==[r['label'] for r in p['conditions']]
    spec=mujoco.MjSpec.from_file(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'));sim.hw.configure_spec(spec);m=spec.compile()
    qids=np.array([m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    vids=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    initial={};details={};hashes={};steps=actor_rows=contacts=0;reproduction=[]
    for job,entry in zip(p['conditions'],c['records']):
        directory=OUT/job['label'];path=directory/'result.json';assert sha(path)==entry['sha256']
        assert entry['case']==job['case']['seed'] and entry['controller']==job['controller'] and entry['target_y_m']==job['target_y_m']
        row=load_rows(path,[job['case']])['runs'][0]
        assert row['success']==entry['success']==(not any(flags(row).values()))
        checked=check_files(directory,[job['case']],{row['seed']:row});checked.pop('original_label_differences')
        assert checked==entry['checked'] and logs(directory,[job['case']])==row['physical_steps'];contacts+=checked['contacts']
        for name in ('initial','terminal','route_requests'):assert sha(directory/f'{name}.npz')==entry[name+'_sha256']
        with np.load(directory/'initial.npz') as z:snapshot={k:z[k].copy() for k in z.files}
        assert {'ctrl','time','warm','sensors','q','v','param','reference','controller_memory','guard_memory','frames','inputs','elapsed','valid','geom_xpos','geom_xmat'}==set(snapshot)
        key=job['case']['seed']
        if key not in initial:initial[key]=snapshot
        else:
            for k in initial[key]:np.testing.assert_array_equal(snapshot[k],initial[key][k])
        for field in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace'):
            f=directory/row[field]['path'];assert sha(f)==row[field]['sha256'];hashes[str(f.relative_to(ROOT))]=sha(f)
        with np.load(directory/row['complete_trace']['path']) as z:t,pre,post=z['trace'],z['pre'],z['post']
        n=len(t);steps+=n;assert n==entry['first_episode_world_steps']<=p['per_job_first_episode_steps_cap']
        np.testing.assert_array_equal(snapshot['q'][0],pre[0,:17]);np.testing.assert_array_equal(snapshot['v'][0],pre[0,17:33])
        margin=1.4-abs(post[:,qids]).max(axis=1);np.testing.assert_array_equal(margin,t[:,24]);assert margin.min()==row['min_active_design_margin_rad']>=0
        lengths,loop=chain_minima(post[:,:17],m)
        np.testing.assert_allclose(lengths,[row['min_actual_A_leg_m'],row['min_actual_B_leg_m']],rtol=0,atol=1e-12)
        assert min(lengths)>=HEIGHT_115_GEOMETRIC_MIN
        np.testing.assert_allclose(loop,row['max_loop_error_m'],rtol=0,atol=1e-12)
        physical=physical_metrics(pre,post,m)
        np.testing.assert_allclose(physical,[row['min_eight_joint_margin_rad'],row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm']],rtol=0,atol=1e-12)
        assert physical[0]>=0 and max(physical[1:])<=1e-6
        with np.load(directory/'terminal.npz') as z:
            np.testing.assert_array_equal(z['terminal_q'][0],post[-1,:17]);np.testing.assert_array_equal(z['terminal_v'][0],post[-1,17:33])
        with np.load(directory/row['joint_guard_trace']['path']) as z:g=z['trace']
        with np.load(directory/'guard_prefix.npz') as z:np.testing.assert_array_equal(g,z['trace'])
        assert g.shape==(n,62);np.testing.assert_array_equal(g[:,50:54],pre[:,qids]);np.testing.assert_array_equal(g[:,54:58],pre[:,17+vids]);np.testing.assert_array_equal(g[:,20:26],pre[:,33:39])
        with np.load(directory/row['parking_trace']['path']) as z:parking=z['trace']
        check_log(parking,'original')
        with np.load(directory/row['actor_trace']['path']) as z:actor=z['trace']
        with np.load(directory/'route_requests.npz') as z:route=z['trace']
        assert actor.shape==(int(np.ceil(n/40)),968) and len(actor)==entry['classical_action_rows'];actor_rows+=len(actor)
        check_actions(actor,route,job['target_y_m'],p['classical_config'])
        np.testing.assert_array_equal(actor[0,:390],snapshot['frames'].reshape(-1))
        np.testing.assert_array_equal(parking[:,5:11],actor[np.arange(n)//40,962:].astype(float))
        assert row['physical_safety_passed'] and row['design_joint_passed'] and entry['physical']==entry['design']==1
        details[job['label']]=dict(case=key,controller=job['controller'],target_y_m=job['target_y_m'],success=row['success'],flags=flags(row),
            peak_deg=row['peak_deg'],body_y_peak_m=float(abs(t[:,5]).max()),velocity_rmse=row['velocity_rmse'],arrival_s=row['arrival_s'],
            height_rmse_m=row['height_rmse_m'],stop_distance_m=row['stop_distance_m'],tail_speed_m_s=row['tail_speed_m_s'],
            minimum_design_margin_rad=float(margin.min()),guard_nominal_corrected=int((g[:,60]>1e-12).sum()),guard_residual_reduced=int((g[:,27]<1-1e-9).sum()))
        if job['controller']=='zero':
            historical=next(x for x in p['historical_zero_results'] if x['seed']==key);file=ROOT/historical['result']
            assert sha(file)==p['evidence_sha256'][str(file.relative_to(ROOT))]
            prior=json.loads(file.read_text())['runs']
            prior=next(r for r in prior if r['seed']==key)
            assert prior['scenario']==row['scenario']
            reproduction.append(dict(case=key,flags_reproduced=flags(row)==flags(prior),success_reproduced=row['success']==prior['success'],historical_result_sha256=sha(file)))
        for f in (path,directory/'initial.npz',directory/'terminal.npz',directory/'guard_prefix.npz',directory/'route_requests.npz',directory/'geometry.json'):
            hashes[str(f.relative_to(ROOT))]=sha(f)
        print('RECEIVED357',job['label'],n,'steps',flush=True)
    assert len(initial)==len(reproduction)==6 and steps==c['first_episode_world_steps']
    assert sum(r['actual_graph_world_steps'] for r in c['records'])==c['actual_graph_world_steps']<=p['actual_graph_world_step_budget']
    assert len(c['FD_calls'])<=p['constructor_FD_budget']
    assert actor_rows==c['classical_action_rows']<=p['classical_action_row_budget']
    reproduced=all(r['flags_reproduced'] and r['success_reproduced'] for r in reproduction)
    counts={name:sum(r['success'] for r in details.values() if r['controller']==name) for name in ('zero','plus_radius','minus_radius')}
    closed=reproduced and counts['plus_radius']==counts['minus_radius']==0
    atomic_json(OUT/'review.json',dict(round=357,verified=True,evaluations=18,unique_seen_cases=6,initial_state_exact_per_case=True,
        first_episode_world_steps=steps,contacts_checked=contacts,classical_action_rows_replayed=actor_rows,action_replay_exact=True,
        physical_passed=18,design_passed=18,reproduction_gate_passed=reproduced,reproduction=reproduction,success_counts=counts,details=details,raw_sha256=hashes,
        completion_sha256=sha(OUT/'completion.json'),source_admission_sha256=sha(OUT/'source_admission.json'),proposal_sha256=sha(OUT/'proposal.json'),reviewer_sha256=sha(__file__),
        new_physics_steps=0,new_model_forward_rows=0,new_learning_samples=0,formal5_admitted=False))
    atomic_json(OUT/'closure.json',dict(round=357,review_sha256=sha(OUT/'review.json'),fixed_offset_branch_closed=closed,
        reason='All six zero-reference task flags reproduced; both fixed offset controllers failed all six full tasks.' if closed else 'Closure not established; inspect reproduction and outcomes.',
        prohibited_expansion=['offset magnitude/gain sweep to rescue this branch','casewise sign oracle','new training or formal5 from these diagnostics'],
        scope='Only registered +/- one wheel radius with this fixed B1-route configuration on six seen cases. No claim that all classical paths fail or RL is necessary.',
        next='358 synthesize remaining learning-method requirements and a concrete discriminating proposal; no additional generic source-only audits.360 direction/cleanup/framework.',goal_complete=False))
    print('DONE357',counts,'reproduced',reproduced,'closed',closed,flush=True)


if __name__=='__main__':run()
