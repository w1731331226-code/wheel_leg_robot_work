"""Static final-evaluator/zero-Nom/reference-row reuse qualification, no rollout."""
import json
from pathlib import Path
import numpy as np
import warp as wp
from stable_baselines3 import PPO
import nominal_mean_training as training
import nominal_mean_evaluation as evaluation
from test_nom_yaw_filter_probe import control_args
from score_reference_learning import load_rows
from smoke_reward_training import weight_digest
from dashboard.live_env import atomic_json

OUT,ROOT,sha = training.OUT,training.ROOT,training.sha


def run():
    p,_ = training.verify_engineering(); proof=json.loads((OUT/'round203_engineering_review.json').read_text());assert proof['verified']
    reference_inputs={};reference_count=0
    for ref in p['references']['phase_result_refs']+p['references']['aux_result_refs']:
        file=ROOT/ref['path'];assert sha(file)==ref['sha256'];rows=json.loads(file.read_text())['runs']
        cases=[dict(seed=r['seed'],scenario=r['scenario']) for r in rows]
        load_rows(file,cases);reference_inputs[ref['path']]=ref['sha256'];reference_count+=len(rows)
    assert reference_count==418
    prefix=OUT/'engineering'/'anchored'/'step_12000';model=PPO.load(prefix.with_suffix('.zip'),device='cuda')
    before=(model.num_timesteps,model._n_updates,weight_digest(model));owners=[];comparisons=0
    import nominal_mean_main as main
    for job in main.eval_jobs(p):
        cases=job['cases'];panel=job['panel']
        raw6,cache,norm,env=evaluation.make_env(cases,panel,prefix.with_suffix('.pkl'))
        if panel=='auxiliary':raw3=evaluation.force.instrument(cases,'diff3')
        else:raw3=evaluation.phase.instrument(cases,'diff3')
        try:
            stats=(norm.obs_rms.mean.copy(),norm.obs_rms.var.copy(),norm.obs_rms.count)
            obs=env.reset();raw3.reset()
            assert obs.shape==(len(cases),78) and raw6.action_space.shape==(6,)
            np.testing.assert_array_equal(obs[:,:39],norm.normalize_obs(np.c_[raw6.obs.numpy(),np.zeros(len(cases),np.float32)]))
            action=model.predict(obs,deterministic=True)[0];assert action.shape==(len(cases),6) and np.isfinite(action).all()
            for command in (0.,-.8,.8):
                outputs=[]
                for raw in (raw3,raw6):
                    raw.reset();raw.command.fill_(command);raw.targets.zero_()
                    evaluation.phase.prepare_step(raw,0,'phase_support');args=control_args(raw);args[12]=raw._role_buffers[0]
                    wp.launch(evaluation.phase.roles.experimental.control_physical_nominal,raw.num_envs,args,block_dim=32)
                    outputs.append((raw.data.ctrl.numpy(),raw.diag.numpy()))
                training.equal(outputs[0],outputs[1]);comparisons+=len(cases)
            training.equal(stats,(norm.obs_rms.mean,norm.obs_rms.var,norm.obs_rms.count))
            owners.append(dict(panel=panel,indices=job['indices'],worlds=len(cases),paired_storage=78,
                current_half_exact_original39=True,zero_diff3_virtual6_command_diag_exact=True,
                core=raw6._complete_topology,phase=raw6._phase_topology,force=getattr(raw6,'_force_topology',None)))
        finally:env.close();raw3.close()
        print('EVAL SOURCE CHECKED',panel,job['indices'],flush=True)
    assert comparisons==209*3 and before==(model.num_timesteps,model._n_updates,weight_digest(model))
    atomic_json(OUT/'round204_evaluation_unit.json',dict(verified=True,static_registered_evaluation_cases=209,
        diff3_virtual6_static_worlds=418,zero_command_comparisons=comparisons,owners=owners,
        reference_rows=reference_count,reference_input_sha256=reference_inputs,model_RMS_immutable=True,
        saved78_policy_39RMS_interface_inference=True,new_scientific_evaluations=0,training_updates=0,
        limits='Constructors/staticzero-control andsavedengineering-model inference only. Source-qualified reuse,notbitwise replay;'
            'allnewlearner1254 evaluations still required with source/model/RMS/complete logs checks.'))
    sources=json.loads((OUT/'policy_source_contract.json').read_text())['source_sha256']
    assert all(sha(ROOT/n)==s for n,s in sources.items())
    for name in ['nominal_mean_training.py','nominal_mean_evaluation.py','nominal_mean_main.py','test_nominal_mean_evaluation.py']:
        sources['wheelleg_warp/'+name]=sha(ROOT/'wheelleg_warp'/name)
    atomic_json(OUT/'evaluation_admission.json',dict(verified=True,round=204,evaluation_source_admitted=True,
        proposal_sha256=sha(OUT/'proposal.json'),engineering_review_sha256=sha(OUT/'round203_engineering_review.json'),
        unit_sha256=sha(OUT/'round204_evaluation_unit.json'),source_sha256=sources,reference_input_sha256=reference_inputs,
        conditional_reference_reuse_rows=418,new_final_evaluation_budget=1254,new_scientific_evaluations=0,
        main_trainer_frozen=False,training_updates=0,
        limits='Compatible wrapper andzeroNom sources only; references aredevelopment/regression/aux notfresh independenttest. '
            'No scientificscores used toselect model,checkpoint orseed inthis admission.'))
    print('PASS final78 evaluator/627zero command comparisons/418reference rows;0 mainlearning/evals',flush=True)


if __name__=='__main__':run()
