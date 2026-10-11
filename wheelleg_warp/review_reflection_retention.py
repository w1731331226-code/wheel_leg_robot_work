"""Independent16-episode selected-gain retention receipt and branch stopping decision."""
import json
import pickle
import numpy as np
import mujoco
import torch
from stable_baselines3 import PPO
from run_reflection_retention import OUT,STUDY,inputs
from run_wheel_interference import OUT as PRIOR
from audit_policy_reflection import reflect_raw,ACTION
from review_yaw_sector import ROOT,sha
from score_reference_learning import load_rows
from complete_contact_witness import check_files
from run_phase_support_qualification import logs
from parking_withdrawal_probe import check_log
from fixed_reference_force import embed
from review_fixed_force_evaluation import chain_minima,physical_metrics
from analyze_reward_failures import flags
from dashboard.live_env import atomic_json
import wheelleg_sim as sim
from review_frozen_reflection import check_operator,self_check
from analyze_task_mode_witness import centre_metrics
from path_qualification_draft import conditions


def run():
    self_check();p=inputs();assert not (OUT/'review.json').exists()
    c=json.loads((OUT/'completion.json').read_text());a=json.loads((OUT/'source_admission.json').read_text())
    assert c['verified'] and c['evaluations']==16 and len(c['records'])==16 and c['new_learning_samples']==0
    assert c['proposal_sha256']==sha(OUT/'proposal.json') and c['source_admission_sha256']==sha(OUT/'source_admission.json')
    assert a['verified'] and all(sha(ROOT/f)==h for f,h in a['source_sha256'].items())
    assert [r['condition'] for r in c['records']]==[r['label'] for r in p['conditions']]
    spec=mujoco.MjSpec.from_file(str(ROOT/'wheelleg_ppo/xml/wheelleg.xml'));sim.hw.configure_spec(spec);m=spec.compile()
    qids=np.array([m.jnt_qposadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    vids=np.array([m.jnt_dofadr[m.joint(n).id] for n in ('alphaL','betaL','alphaR','betaR')])
    initial={};rows={};details={};hashes={};steps=actor_rows=query_rows=0;torch.set_num_threads(1)
    for job,entry in zip(p['conditions'],c['records']):
        directory=OUT/job['label'];path=directory/'result.json';assert sha(path)==entry['sha256']
        assert entry['case']==job['case']['seed'] and entry['pair']==job['pair'] and entry['model']==job['model'] and entry['operator']==job['operator']
        row=load_rows(path,[job['case']])['runs'][0];assert row['success']==(not any(flags(row).values()))
        checked=check_files(directory,[job['case']],{row['seed']:row});checked.pop('original_label_differences')
        assert checked==entry['checked'] and logs(directory,[job['case']])==row['physical_steps']
        for name in ('initial','terminal'):assert sha(directory/f'{name}.npz')==entry[name+'_sha256']
        with np.load(directory/'initial.npz') as z:snapshot={k:z[k].copy() for k in z.files}
        assert {'ctrl','time','warm','sensors'}<=set(snapshot);key=job['case']['seed']
        if key not in initial:initial[key]=snapshot
        else:
            assert set(snapshot)==set(initial[key])
            for k in initial[key]:np.testing.assert_array_equal(snapshot[k],initial[key][k])
        for field in ('complete_trace','gyro_trace','role_trace','phase_trace','parking_trace','joint_guard_trace','actor_trace'):
            f=directory/row[field]['path'];assert sha(f)==row[field]['sha256'];hashes[str(f.relative_to(ROOT))]=sha(f)
        with np.load(directory/row['complete_trace']['path']) as z:t,pre,post=z['trace'],z['pre'],z['post']
        n=len(t);steps+=n;assert n==entry['first_episode_world_steps']<=p['per_job_first_episode_steps_cap']
        margin=1.4-abs(post[:,qids]).max(axis=1);np.testing.assert_array_equal(margin,t[:,24]);assert margin.min()==row['min_active_design_margin_rad']
        lengths,loop=chain_minima(post[:,:17],m)
        np.testing.assert_allclose(lengths,[row['min_actual_A_leg_m'],row['min_actual_B_leg_m']],rtol=0,atol=1e-12)
        np.testing.assert_allclose(loop,row['max_loop_error_m'],rtol=0,atol=1e-12)
        np.testing.assert_allclose(physical_metrics(pre,post,m),[row['min_eight_joint_margin_rad'],row['max_actual_torque_excess_Nm'],row['max_command_torque_excess_Nm']],rtol=0,atol=1e-12)
        with np.load(directory/'terminal.npz') as z:
            np.testing.assert_array_equal(z['terminal_q'][0],post[-1,:17]);np.testing.assert_array_equal(z['terminal_v'][0],post[-1,17:33])
        with np.load(directory/row['joint_guard_trace']['path']) as z:g=z['trace']
        with np.load(directory/'guard_prefix.npz') as z:np.testing.assert_array_equal(g,z['trace'])
        assert g.shape==(n,62);np.testing.assert_array_equal(g[:,50:54],pre[:,qids]);np.testing.assert_array_equal(g[:,54:58],pre[:,17+vids]);np.testing.assert_array_equal(g[:,20:26],pre[:,33:39])
        with np.load(directory/row['parking_trace']['path']) as z:parking=z['trace']
        check_log(parking,'parking_request_withdrawal')
        with np.load(directory/row['actor_trace']['path']) as z:actor=z['trace']
        assert actor.shape==(int(np.ceil(n/40)),968);actor_rows+=len(actor);error=secondary_error=0.;secondary=np.empty((0,6))
        if job['model'] is None:
            original=np.zeros_like(actor[:,962:]);np.testing.assert_array_equal(actor[:,:481],actor[:,481:962])
        else:
            assert sha(directory/'unmasked_requests.npz')==entry['unmasked_sha256']
            with np.load(directory/'unmasked_requests.npz') as z:original=z['trace']
            config=next(x for x in p['models'] if x['label']==job['model']);model=PPO.load(ROOT/config['model'],device='cpu')
            assert model.num_timesteps==200000 and model._n_updates==400
            with (ROOT/config['normalization']).open('rb') as f:norm=pickle.load(f)
            np.testing.assert_array_equal(norm.normalize_obs(actor[:,:481].copy()),actor[:,481:962])
            prediction=embed(model.predict(actor[:,481:962],deterministic=True)[0],config['arm'],len(actor));error=float(abs(prediction-original).max());assert error<=1e-6
            file=directory/'operator_queries.npz';assert sha(file)==entry['operator_queries_sha256'];hashes[str(file.relative_to(ROOT))]=sha(file)
            with np.load(file) as z:
                np.testing.assert_array_equal(z['original'],original);np.testing.assert_array_equal(z['submitted'],actor[:,962:]);secondary=z['secondary'];secondary_obs=z['secondary_normalized']
            queries=len(actor)
            if job['operator']=='reflection':
                mirrored=reflect_raw(actor[:,:481]);np.testing.assert_array_equal(reflect_raw(mirrored),actor[:,:481])
                np.testing.assert_array_equal(norm.normalize_obs(mirrored),secondary_obs)
                predicted=embed(model.predict(secondary_obs,deterministic=True)[0],config['arm'],len(actor))
                secondary_error=float(abs(predicted-secondary).max());assert secondary_error<=1e-6;queries+=len(actor)
                if config['arm']=='D3':np.testing.assert_array_equal(actor[0,962:],0)
            else:assert secondary.shape==(0,6) and secondary_obs.shape==(0,481)
            assert entry['model_forward_rows']==queries;query_rows+=queries
        if job['model'] is None:assert entry['model_forward_rows']==0
        check_operator(original,secondary,actor[:,962:],job['operator'])
        np.testing.assert_array_equal(parking[:,5:11],actor[np.arange(n)//40,962:].astype(float))
        assert row['physical_safety_passed'] and row['design_joint_passed']
        rows[job['label']]=row;details[job['label']]=dict(success=row['success'],flags=flags(row),yaw_peak_deg=row['peak_deg'][2],body_y_peak_m=float(abs(t[:,5]).max()),minimum_design_margin_rad=float(margin.min()),
            actor_replay_max_error=error,secondary_replay_max_error=secondary_error,velocity_rmse=row['velocity_rmse'],arrival_s=row['arrival_s'],height_rmse_m=row['height_rmse_m'],stop_distance_m=row['stop_distance_m'],tail_speed_m_s=row['tail_speed_m_s'],roll_pitch_peak_deg=row['peak_deg'][:2],guard_nominal_corrected=int((g[:,60]>1e-12).sum()),guard_residual_reduced=int((g[:,27]<1-1e-9).sum()))
        geometry=json.loads((directory/'geometry.json').read_text());assert len(geometry['boxes'][0])==1
        side='left' if row['scenario']['height_l'] else 'right';box=geometry['boxes'][0][0]
        details[job['label']]['centre_lane']=centre_metrics(t,box,side)
        details[job['label']]['sampled_geometry']=conditions(t,box,side,1 if row['scenario']['speed']>0 else -1)
        for f in (path,directory/'initial.npz',directory/'terminal.npz',directory/'guard_prefix.npz'):
            hashes[str(f.relative_to(ROOT))]=sha(f)
        if job['model']:hashes[str((directory/'unmasked_requests.npz').relative_to(ROOT))]=sha(directory/'unmasked_requests.npz')
        print('RECEIVED352',job['label'],n,'steps',flush=True)
    assert steps==c['first_episode_world_steps'] and sum(r['actual_graph_world_steps'] for r in c['records'])==c['actual_graph_world_steps']<=p['actual_graph_world_step_budget'] and len(c['FD_calls'])<=p['constructor_FD_budget']
    assert query_rows==c['model_forward_rows']==sum(r['model_forward_rows'] for r in c['records'])<=p['model_forward_row_budget']
    prior_review=json.loads((STUDY/'evaluation_review.json').read_text());assert prior_review['completion_sha256']==sha(STUDY/'models/completion.json')
    old=json.loads((STUDY/'models/completion.json').read_text());accepted={str((STUDY/'models'/e['path']).relative_to(ROOT)):e['sha256'] for e in old['records']};pairs=[]
    for pair in p['pairs']:
        file=ROOT/pair['original_result'];assert sha(file)==pair['original_result_sha256']==accepted[pair['original_result']]
        prior=next(r for r in json.loads(file.read_text())['runs'] if r['seed']==pair['case']['seed'])
        jobs=[j for j in p['conditions'] if j['pair']==pair['index']];original=rows[jobs[0]['label']];reflected=rows[jobs[1]['label']]
        reproduced=flags(original)==flags(prior)
        pairs.append(dict(pair=pair['index'],model=pair['model'],case=pair['case']['seed'],original_flags_reproduced=reproduced,
            original_success=original['success'],reflection_success=reflected['success'],original_peak_deg=original['peak_deg'],reflection_peak_deg=reflected['peak_deg'],
            reflection_failure_flags=[k for k,v in flags(reflected).items() if v],retention_counterexample=bool(reproduced and original['success'] and not reflected['success']),historical_result_sha256=sha(file)))
    reproduced=all(r['original_flags_reproduced'] and r['original_success'] for r in pairs)
    counterexamples=[r['pair'] for r in pairs if r['retention_counterexample']]
    assert len(initial)==6
    atomic_json(OUT/'review.json',dict(round=352,verified=True,evaluations=16,unique_seen_cases=6,initial_state_exact_per_case=True,first_episode_world_steps=steps,actor_rows_replayed=actor_rows,model_forward_rows_replayed=query_rows,
        reproduction_gate_passed=reproduced,retention_counterexample_pairs=counterexamples,uniform_reflection_expansion_closed=bool(counterexamples),details=details,pairs=pairs,raw_sha256=hashes,
        completion_sha256=sha(OUT/'completion.json'),source_admission_sha256=sha(OUT/'source_admission.json'),proposal_sha256=sha(OUT/'proposal.json'),reviewer_sha256=sha(__file__),
        new_physics_steps=0,new_learning_samples=0,original_candidate_benefit_branch_closed=True,formal5_admitted=False,
        interpretation='Posthoc selected-gain falsification,notgeneralization rate. A reproducedoriginal win lostunderreflection refutes universalpreservation. Originalcontact-task scores remainvalid;centre/vertical-witness descriptors do notreplace gates orcertify fulltyre passage. No model ornewtraining promotion.',
        next='353 synthesize accepted retention/contact-path tradeoff andclose thisuniformreflection branch;do notrefit tofailedcases.355direction/cleanup,360wholeframework.'))
    print('DONE35216independent receipts;reproduction',reproduced,'retentioncounterexamples',len(counterexamples),flush=True)


if __name__=='__main__':run()
