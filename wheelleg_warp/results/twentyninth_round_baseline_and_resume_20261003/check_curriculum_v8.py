"""Real zero-action episode ends, masked stage fields and independent CPU models."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from train_height_comparison import protocol,CurriculumEnv
from native.terrain import HeightTerrainScenario,model
HERE=Path(__file__).resolve().parent
p=protocol(HERE/'protocol_v8');env=CurriculumEnv(p,'diff3',1609,4,milestones=[1,1000000])
rows=[];seen2=False;seen3=False;delayed_checked=False
try:
    env.reset();gains=env.venv.k['gains'].numpy().copy()
    for step in range(2000):
        before=env.stages.copy();obs,_,done,infos=env.step(np.zeros((4,3),np.float32))
        changed=np.flatnonzero(env.stages!=before)
        assert all(done[w] for w in changed)
        assert np.all(env.stages[~done]==before[~done])
        for w in changed:
            stage=int(env.stages[w]);s=HeightTerrainScenario(**env.banks[stage][w]['scenario']);cpu=model(s);raw=env.venv
            for name in ('body_mass','body_inertia','body_ipos','geom_pos','geom_size','geom_friction','actuator_gainprm'):
                np.testing.assert_allclose(getattr(raw.model,name).numpy()[w],getattr(cpu,name),atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(raw.model.stat.meaninertia.numpy()[w],cpu.stat.meaninertia,atol=1e-6,rtol=1e-6)
            np.testing.assert_allclose(raw.data.qpos.numpy()[w],raw.q0.numpy()[w],atol=0,rtol=0)
            assert raw.state.numpy()[w,0]==0 and np.all(raw.k['state'].numpy()[w,16:]==0)
            assert abs(float(obs[w,11])-(s.stand_height_m-.3))<1e-7 and np.all(obs[w,32:]==0)
            assert infos[w]['curriculum_stage']==int(before[w]) and infos[w]['physical_steps']==infos[w]['physical_evidence_steps']
            rows.append(dict(step=step,world=int(w),old_stage=int(before[w]),new_stage=stage,source_episode_reason=infos[w]['reason']))
            seen2|=stage==2;seen3|=stage==3
        np.testing.assert_array_equal(gains,env.venv.k['gains'].numpy())
        raw=env.venv
        assert raw.history.shape[1]==41
        for w in range(4):
            stage=int(env.stages[w]);cache=env.cache[stage]
            for name,values in cache['info'].items():assert getattr(raw,name)[w]==values[w]
            delay=int(raw.param.numpy()[w,4]);count=int(raw.state.numpy()[w,0])
            if stage==3 and delay>0 and count>40 and not done[w]:
                index=(count-delay+41)%41
                np.testing.assert_array_equal(obs[w,:32],raw.history.numpy()[w,index,:])
                assert np.any(obs[w,:32]!=raw.history.numpy()[w,count%41,:]);delayed_checked=True
        if np.all(env.stages==2) and env.milestones[1]==1000000:env.milestones=[1,env.policy_steps+4]
        if seen2 and seen3 and np.all(env.stages==3) and delayed_checked:break
    assert seen2 and seen3 and np.all(env.stages==3) and delayed_checked
    (HERE/'curriculum_check_v8.json').write_text(json.dumps(dict(passed=True,zero_actor=True,transitions=rows,
        switch_only_after_actual_termination=True,independent_cpu_model_fields_match=True,nominal_gains_unchanged=True,
        new_target_requests_and_physical_state_reset=True,stage3_physical_delay_effective=True,all_reporting_contract_fields_match=True,final_holdout_opened=False),indent=2)+'\n')
    print('PASS actual curriculum: stage2/3 only on true ends; CPU mass/inertia/friction/gain/geometry match; Nom fixed and reset correct')
finally:env.close()
