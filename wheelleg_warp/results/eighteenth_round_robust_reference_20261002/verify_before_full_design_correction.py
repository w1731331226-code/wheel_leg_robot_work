"""Recalculate solver selection and independent actual-episode constraints."""
import json
import sys
from pathlib import Path
import numpy as np
from evaluate import OUT,ROOT,SOURCE,SCALE,schedules,sha,model,HeightTerrainScenario
from select_braking_common_action import batch_constraint_margins,task_forecast


def run(directions=(0,1)):
    assert directions and all(d in (0,1) for d in directions)
    nom=model(HeightTerrainScenario(stand_height_m=.115));summaries=[]
    for direction in directions:
        folder=OUT/f'solve_world{direction}'
        report=json.loads((folder/'summary.json').read_text())
        log=[json.loads(line) for line in (folder/'evaluations.jsonl').read_text().splitlines()]
        assert len(log)==report['evaluations']<=120
        selected=max(log,key=lambda r:(r['all_registered_feasible'],-r['scaled_violation'],-r['nominal_cost']))
        saved=np.load(folder/'best.npz',allow_pickle=False)
        np.testing.assert_array_equal(saved['parameters'],selected['parameters'])
        assert report['best_feasible']==selected['all_registered_feasible']
        plan=schedules(saved['parameters'][None,:],saved['common'],400)
        np.testing.assert_allclose(plan[:,0],saved['schedule'],rtol=0,atol=1e-15)
        assert abs(plan).max()<=1 and abs(np.diff(np.concatenate([np.zeros_like(plan[:1]),plan]),axis=0)).sum(axis=2).max()<=.1+1e-12
        independent=json.loads((folder/'independent.json').read_text())
        z=np.load(folder/'independent.npz',allow_pickle=False)
        assert sha(folder/'independent.npz')==independent['trace_sha256']
        np.testing.assert_array_equal(z['schedule'],plan)
        recomputed=[]
        for w,count in enumerate(z['lengths']):
            a=z['trace'][:int(count),w:w+1,:-2];task=z['trace'][:int(count),w:w+1,-2:]
            q=a[:,:,:nom.nq];v=a[:,:,nom.nq:nom.nq+nom.nv]
            physical=batch_constraint_margins(nom,q,v,a[:,:,-12:-6],a[:,:,-6:],z['initial_v'][w:w+1])
            budget=task_forecast(q,v,z['initial_q'][w:w+1],z['initial_v'][w:w+1],z['initial_task'][w:w+1])
            state=z['initial_task'][w]
            rmse=np.sqrt((state[28]+np.sum((task[:,:,0]-.115)**2)*.0005)/(state[29]+int(count)*.0005))
            margins=np.r_[physical[0],.6-budget['peak_distance_m'][0],.03-budget['peak_tail_speed_m_s'][0],
                .02-rmse,.02-abs(task[-1,0,0]-.115),-task[:,:,1].max()]
            np.testing.assert_allclose(margins,independent['margins'][w],rtol=0,atol=1e-12)
            e=independent['episodes'][w]
            assert e['physical_steps']==e['physical_evidence_steps']
            recomputed.append(bool(np.all(margins>=0) and e['success']))
        assert recomputed==independent['valid']
        for r in (report,independent):
            for name,digest in r['source_sha256'].items():assert sha(ROOT/name)==digest,name
        summaries.append(dict(direction=direction,evaluations=len(log),solver=report['solver'],
            selected_evaluation=selected['evaluation'],batch_feasible=report['best_feasible'],
            independent_valid=recomputed,independent_stop_m=[e['stop_distance_m'] for e in independent['episodes']],
            independent_tail_m_s=[e['tail_speed_m_s'] for e in independent['episodes']],
            minimum_independent_joint_design_margin_rad=float(np.min(np.array(independent['margins'])[:,:8])),
            nominal_cost_change_vs_initial_fraction=report['best_nominal_cost']/log[0]['nominal_cost']-1))
    result=dict(directions=summaries,learning=False,default_promoted=False,full_admission=False,
        limitations='Registered ten high-speed cases only; no other four normal tasks, continuous height/parameter domain, nonzero Actor, final holdout or paper advantage claim.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/('verification.json' if directions==(0,1) else f'verification_world{directions[0]}.json')
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('CHECKED registered budget, original-domain plans, cost selection and independent original constraints:',summaries,flush=True)


if __name__=='__main__':run((int(sys.argv[1]),) if len(sys.argv)>1 else (0,1))
