"""One fixed-policy guarded projection experiment; no learning or gain sweep."""
from pathlib import Path
import argparse,json
import numpy as np
import warp as wp
from stable_baselines3 import PPO
from native.controller import D,control_physical_nominal
from leg_residual_cone import supervise
from trace_design_boundary import measured
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'wheelleg_warp/results/paper_recovery_20261004/design_boundary_v1'


def collect(cases,model,normalization,mode,out,label):
    original=wp.launch;stats=wp.zeros((len(cases),7),dtype=D)
    def launch(kernel,dim,inputs=None,**kwargs):
        answer=original(kernel,dim,**kwargs) if inputs is None else original(kernel,dim,inputs,**kwargs)
        if kernel is control_physical_nominal:
            original(supervise,len(cases),[mode,inputs[0],inputs[1],inputs[7],inputs[5],inputs[6],inputs[15],inputs[14],inputs[18],stats])
        return answer
    wp.launch=launch
    try:return measured(cases,model,normalization,None,out,label)
    finally:wp.launch=original


def run(out):
    p=json.loads((BASE/'registration.json').read_text());boundary=json.loads((BASE/'review.json').read_text())
    unit_path=ROOT/'wheelleg_warp/results/paper_recovery_20261004/leg_cone_unit_v2/verification.json'
    unit=json.loads(unit_path.read_text());assert unit['verified'] and unit['unguarded_and_zero_identity_exact']
    assert all(digest(ROOT/n)==v for n,v in unit['source_sha256'].items())
    assert boundary['verified'] and all(digest(ROOT/n)==v for n,v in p['source_sha256'].items())
    out.mkdir(parents=True,exist_ok=False)
    reg=dict(version='leg-cone-fixed-policy-v1',cases=p['cases'],models=p['models'],modes={'original':0,'cone':1,'zero_after_filter':2},
        budget_episodes=243,training_updates=0,physics_hz=2000,actor_hz=50,
        guard='abs(q_j)+max(0,sign(q_j)*qdot_j)*0.02>=1.4 and sign(q_j)*accepted_nominal_j<0',
        projection='Minimize four-leg-motor squared command change in2D F/H coordinates, constraints sign(q_j)*(M_j*u)<=0 on guarded joints, original actuator torque boxes and normalized coordinate boxes; zero feasible. Face/vertex enumeration; selected after original filtering/physical projection, before physics.',
        invariants='Nom and wheel outputs unchanged; nonbinding projection preserves original commands exactly; original lambda remains pre-cone physical scalar, not the cone acceptance factor. Filtered request context remains requested, physical packet records actual projected motor increments.',
        guarantee='Selected outward-moving guarded joint residual command power nonpositive within numeric tolerance, input/torque feasibility only; no acceleration/contact/joint-domain invariance or global-safety assertion.',
        controls='Three fixed L2 final models x original/cone/zero-after-filter x all27 low-height cases; no model/case/gain selection. Previous B0/B1 are reused descriptive development references, not new independent controls.',
        continue_gate='All three cone models preserve every original and zero-control task/physical/design pass; remove all cone design failures; cone mean task success exceeds zero-after-filter mean; otherwise stop this fixed guard/projection without tuning.',
        unit_verification_sha256=digest(unit_path),boundary_review_sha256=digest(BASE/'review.json'),
        source_sha256={**p['source_sha256'],'wheelleg_warp/leg_residual_cone.py':digest(ROOT/'wheelleg_warp/leg_residual_cone.py'),
            'wheelleg_warp/probe_leg_cone.py':digest(__file__)},old_gate_or_final_used=False,development_only=True)
    atomic_json(out/'registration.json',reg);records=[];completed=0
    try:
        for m in p['models']:
            prefix=Path(m['prefix']);c=m['checkpoint']
            assert digest(prefix.with_suffix('.zip'))==c['checkpoint_sha256'] and digest(prefix.with_suffix('.pkl'))==c['normalization_sha256']
            model=PPO.load(str(prefix)+'.zip',device='cuda')
            for name,mode in reg['modes'].items():
                label=str(m['seed'])+'_'+name
                atomic_json(out/'progress.json',dict(status='running',completed_episodes=completed,pending_job=label))
                assert all(digest(ROOT/n)==v for n,v in reg['source_sha256'].items())
                stats=collect(p['cases'],model,str(prefix)+'.pkl',mode,out,label)
                records.append(dict(label=label,stats=stats,result_sha256=digest(out/(label+'.json')),trace_sha256=digest(out/(label+'_trace.npz'))))
                completed+=len(p['cases']);atomic_json(out/'completed_jobs.json',dict(completed_episodes=completed,records=records));print('COMPLETED',completed,label,stats,flush=True)
        panels={r['label']:json.loads((out/(r['label']+'.json')).read_text()) for r in records};preserved={};design_clean=True;gains=[]
        for m in p['models']:
            seed=str(m['seed']);c=panels[seed+'_cone'];z=panels[seed+'_zero_after_filter'];o=panels[seed+'_original']
            preserved[seed]=all((not b['success'] or a['success']) and (not b['physical_safety_passed'] or a['physical_safety_passed']) and (not b['design_joint_passed'] or a['design_joint_passed']) for ref in [o,z] for a,b in zip(c['runs'],ref['runs']))
            design_clean=design_clean and c['design']==27;gains.append(c['summary']['success_count']-z['summary']['success_count'])
        gate=all(preserved.values()) and design_clean and sum(gains)>0
        assert completed==243
        atomic_json(out/'completion.json',dict(completed_episodes=completed,training_updates=0,records=records,preservation=preserved,cone_minus_zero_success_counts=gains,
            design_clean=design_clean,continue_gate=gate,registration_sha256=digest(out/'registration.json')))
        atomic_json(out/'progress.json',dict(status='complete',completed_episodes=completed))
    except BaseException as e:
        atomic_json(out/'interruption.json',dict(completed_episodes=completed,error=str(e),pending_job_consumption_unknown=True,silently_resumable=False));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);run(parser.parse_args().output)
