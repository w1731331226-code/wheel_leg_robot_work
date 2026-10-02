"""Recalculate solver selection and independent actual-episode constraints."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
from evaluate import OUT,ROOT,SOURCE,SCALE,schedules,sha,model,HeightTerrainScenario
from select_braking_common_action import batch_constraint_margins,task_forecast


def recalculate(record,z,nom,full=False):
    valid=[]
    for w,count in enumerate(z['lengths']):
        a=z['trace'][:int(count),w:w+1,:-2];task=z['trace'][:int(count),w:w+1,-2:]
        q=a[:,:,:nom.nq];v=a[:,:,nom.nq:nom.nq+nom.nv]
        physical=batch_constraint_margins(nom,q,v,a[:,:,-12:-6],a[:,:,-6:],z['initial_v'][w:w+1])
        budget=task_forecast(q,v,z['initial_q'][w:w+1],z['initial_v'][w:w+1],z['initial_task'][w:w+1])
        state=z['initial_task'][w]
        rmse=np.sqrt((state[28]+np.sum((task[:,:,0]-.115)**2)*.0005)/(state[29]+int(count)*.0005))
        margins=np.r_[physical[0],.6-budget['peak_distance_m'][0],.03-budget['peak_tail_speed_m_s'][0],
            .02-rmse,.02-abs(task[-1,0,0]-.115),-task[:,:,1].max()]
        e=record['episodes'][w]
        if full:
            history=z['full_active_q'][:e['physical_steps'],w]
            design=np.min(1.4-abs(history),axis=0)
            prefix=np.min(1.4-abs(history[:record['start_steps'][w]]),axis=0)
            np.testing.assert_array_equal(design,z['full_design'][w])
            np.testing.assert_array_equal(prefix,z['pre_design'][w])
            assert bool(np.all(prefix>=0))==record['pre_reference_design_pass'][w]
            hit=np.flatnonzero(abs(history).max(axis=1)>1.4)
            assert (int(hit[0])+1 if len(hit) else -1)==record['first_design_violation_step'][w]
            margins[[0,1,4,5]]=np.minimum(margins[[0,1,4,5]],design)
        np.testing.assert_allclose(margins,record['margins'][w],rtol=0,atol=1e-12)
        assert e['physical_steps']==e['physical_evidence_steps']
        valid.append(bool(np.all(margins>=0) and e['success']))
    assert valid==record['valid']
    return valid


def prefix_rejection_check():
    import solve
    fixture=json.loads((OUT/'full_design_world0.json').read_text())
    assert not fixture['pre_reference_design_pass'][-1]
    class RecordedPrefix:
        def __init__(self,*args,**kwargs):pass
        def evaluate(self,plan):return fixture
        def close(self):pass
    original_out,original_factory=solve.OUT,solve.Episodes
    try:
        with tempfile.TemporaryDirectory(prefix='wheelleg-prefix-check-') as folder:
            solve.OUT=Path(folder);solve.Episodes=RecordedPrefix
            solve.run_direction(0)
            result=json.loads((Path(folder)/'solve_world0/summary.json').read_text())
            assert result['pre_reference_rejected'] and result['evaluations']==1 and not result['best_feasible']
            assert not (Path(folder)/'solve_world0/independent.json').exists()
    finally:solve.OUT,solve.Episodes=original_out,original_factory


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
        braking=recalculate(independent,z,nom)
        complete=json.loads((OUT/f'full_design_world{direction}.json').read_text())
        full=np.load(OUT/f'full_design_world{direction}.npz',allow_pickle=False)
        assert sha(OUT/f'full_design_world{direction}.npz')==complete['trace_sha256']
        recomputed=recalculate(complete,full,nom,full=True)
        for r in (report,independent,complete):
            for name,digest in r['source_sha256'].items():
                path=ROOT/name
                if sha(path)!=digest:
                    path=OUT/{'evaluate.py':'evaluate_at_solver.py','solve.py':'solve_at_run.py'}[path.name]
                assert sha(path)==digest,name
        summaries.append(dict(direction=direction,evaluations=len(log),solver=report['solver'],
            selected_evaluation=selected['evaluation'],braking_batch_feasible=report['best_feasible'],
            braking_only_independent_valid=braking,full_episode_valid=recomputed,
            first_design_violation_step=complete['first_design_violation_step'],
            independent_stop_m=[e['stop_distance_m'] for e in complete['episodes']],
            independent_tail_m_s=[e['tail_speed_m_s'] for e in complete['episodes']],
            minimum_full_active_design_margin_rad=float(np.min(full['full_design'])),
            nominal_cost_change_vs_initial_fraction=report['best_nominal_cost']/log[0]['nominal_cost']-1))
    prefix_rejection_check()
    panel=json.loads((OUT/'full_panel/verification.json').read_text())
    result=dict(directions=summaries,full_panel_groups=panel['groups'],pre_reference_rejection_guard_checked=True,
        learning=False,default_promoted=False,full_admission=False,
        limitations='Supersedes the earlier braking-only interpretation. Full design coverage rejects the positive training-combination prefix. No continuous domain, nonzero Actor, final holdout or paper advantage claim.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/('full_verification.json' if directions==(0,1) else f'full_verification_world{directions[0]}.json')
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('CHECKED registered budget, original-domain plans, cost selection and independent original constraints:',summaries,flush=True)


if __name__=='__main__':run((int(sys.argv[1]),) if len(sys.argv)>1 else (0,1))
