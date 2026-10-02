"""Independent cost/registration/domain checks for complete-episode preflight."""
import json
from pathlib import Path
import numpy as np
from evaluate import OUT,ROOT,SOURCE,SCALE,VARIATIONS,model,HeightTerrainScenario,lqr_cost,sagittal_basis,sha
from select_braking_common_action import trajectory_cost


def run():
    old=json.loads((OUT.parent/'sixteenth_round_parameter_isolation_20261002/verification.json').read_text())
    combined=json.loads((OUT.parent/'eleventh_round_nominal_panel_checked_20261002/verification.json').read_text())
    nom=model(HeightTerrainScenario(stand_height_m=.115));ref,Q,R,P=lqr_cost(nom)
    b,u=sagittal_basis(nom);project=np.linalg.pinv(b);input_project=np.linalg.pinv(u)
    comparisons=[];schedule_errors=[]
    for direction in range(2):
        r=json.loads((OUT/f'preflight_world{direction}.json').read_text());z=np.load(OUT/f'preflight_world{direction}.npz',allow_pickle=False)
        assert sha(OUT/f'preflight_world{direction}.npz')==r['trace_sha256']
        for name,digest in r['source_sha256'].items():
            path=OUT/'evaluate_at_preflight.py' if name==str((OUT/'evaluate.py').relative_to(ROOT)) else ROOT/name
            assert sha(path)==digest,name
        plan=z['schedule'];archived=np.load(SOURCE/f'world{direction}_best.npz')['schedule']
        schedule_errors.append(float(abs(plan[:,0]-archived).max()))
        np.testing.assert_allclose(plan[:,0],archived,rtol=0,atol=1e-15)
        assert abs(plan).max()<=1 and abs(np.diff(np.concatenate([np.zeros_like(plan[:1]),plan]),axis=0)).sum(axis=2).max()<=.1+1e-12
        for group in range(5):
            source=old if group<4 else combined;index=6*group+4+direction if group<4 else 10+direction
            assert r['scenarios'][group]==source['registration']['scenarios'][index]
            assert r['valid'][group]==source['original_full_gates'][index]
            e=r['episodes'][group];previous=source['episodes'][index]
            assert e['reason']=='completed' and e['success']==previous['success'] and e['physical_safety_passed']
            assert e['physical_steps']==e['physical_evidence_steps']
            comparisons.append(dict(direction=direction,group=VARIATIONS[group][0],
                stop_difference_m=abs(e['stop_distance_m']-previous['stop_distance_m']),
                tail_difference_m_s=abs(e['tail_speed_m_s']-previous['tail_speed_m_s'])))
        count=int(z['lengths'][0]);a=z['trace'][:count,0,:-2]
        cost=trajectory_cost(nom,a[:,:nom.nq],a[:,nom.nq:nom.nq+nom.nv],a[:,-12:-6],
            z['initial_q'][0],z['initial_v'][0],z['initial_controller'][0],ref,Q,R,P,project,input_project)
        np.testing.assert_allclose(cost,r['costs'][0],rtol=1e-10,atol=1e-8)
    result=dict(preflight_passed=True,independent_full_episode_gate_labels_match=True,
        scalar_original_cost_matches=True,original_schedule_and_domain_checked=True,schedule_reconstruction_peak_Nm=max(schedule_errors),comparisons=comparisons,
        limitations='Matches full-episode gate conclusions and reports continuous differences; not a bitwise independent rollout or online predictor parity claim.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(Path(__file__))})
    target=OUT/'checks.json'
    if target.exists():assert json.loads(target.read_text())==result
    else:target.write_text(json.dumps(result,indent=2)+'\n')
    print('PASS full-episode preflight: original cost, schedule/domain, registration and ten independent gate labels.')


if __name__=='__main__':run()
