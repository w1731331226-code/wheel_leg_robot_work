"""Register a bounded new reward-timing pilot and check fixed classical feasibility."""
from pathlib import Path
import argparse,json
from train_height_comparison import protocol,entry,evaluate,summary
from native.terrain import sample_height_terrain_115
from training_contract import digest
from dashboard.live_env import atomic_json

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3'

def run(out):
    config=protocol(P);out.mkdir(parents=True,exist_ok=False)
    candidates=json.loads((P/'classical_selection.json').read_text())
    b1=candidates['selected']['candidate'];b0=dict(name='B0',kp=0.,kd=0.,roll_gain=0.)
    development=[entry(i,sample_height_terrain_115(i,3,'development')) for i in range(4300000,4300096)]
    study=dict(version='reward-timing-pilot-v1',status='Registered proposal; trainer integration and feasibility required before launch',
        question='Does known potential shaping improve complete-task learning of constrained M3 PPO while preserving full-episode discounted objective under the same38 observations?',
        contribution_scope='Candidate reproducible empirical audit of reward timing/value approximation and cumulative constraints; not a new PBRS algorithm, not differential superiority or guaranteed improvement',
        arms=['original','potential'],method='M3',training_seeds=[1609,1610,1611],
        environments=100,policy_steps_per_run=200000,total_maximum_pilot_policy_steps=1200000,
        initialization='Fresh matched weights/std per pair from original new_agent; no smoke or old2M checkpoint promotion',
        training_banks=config['training_banks'],curriculum_milestones=config['curriculum_milestones'],
        ppo=config['ppo'],normalization=config['normalization'],device='cuda',policy_network='Original new_agent standardSB3 2x64, PPO loss unchanged',
        physical_baseline=config['baseline_version'],observation_contract='Original38 for both Actor and Critic, same delayed packet/current requests; Phi only in reward bookkeeping',
        potential=dict(gamma=.99,beta=10.,initial_and_terminal=0.,intermediate='-10 if historical irreversible constraint breach',source_sha256=digest(ROOT/'wheelleg_warp/reward_potential.py')),
        development=development,development_role='New public96, fixed before any baseline results or learning; primary pilot evaluation uses final200k only, no best-checkpoint selection',
        checkpoint_every=20000,checkpoint_role='Diagnostics/recovery artifacts only; no budget/model selection using intermediate development scores',
        fixed_classical={'B0':b0,'B1':b1},
        classical_admission='All96 complete; B1 physical96/96, design>=92/96 and success>=92/96; retain every failure if rejected, no panel trimming',
        pilot_primary='Seed-paired mean complete-success fraction: potential minus original on fixed public96 at final200k',
        continuation_gate=dict(mean_success_improvement_at_least=.05,positive_seed_pairs_at_least=2,
            no_increase_in_mean_physical_or_design_failure=True,mean_yaw_score_no_worse_than_raw_plus_deg=.05,
            all_runs_finite_and_budget_matched=True),
        statistic_units='3 paired training seeds; repeated96 cases are paired by scenario, not288 independent scenarios; pilot admission is not a significance/publication claim',
        learning_reward_monitor='Preserve original episode.r and original success/gates, track shaped reward separately',
        run_lifecycle='Each200k run continuous, no planned physical reset/resume during run; checkpoints do not reset physics/Phi/RNG',
        interruption='Confirm terminal/missing process first. Preserve interrupted artifacts and budget; incomplete run not admitted as200k or resumed with silent physical reset. Register recovery/revised budget explicitly before using it for paired inference.',
        final_cut='Record ongoing episode/Phi and gamma^L*Phi boundary; completed-episode equality does not assert full equality of approximate PPO bootstrap or unfinished prefixes',
        scope='Original training domain115..380mm and nine regular terrains; single-side ramp/jump/extreme OOD not covered',
        independent_evaluation='No independent gate or final run in this pilot. After pilot direction review, freeze fresh gate/pressure/nonregression before any independent evaluation; old64 gate and final3000 stay protected',
        fail_rule='Failure keeps this pilot negative; no gain/Phi-amplitude/budget sweep to force admission, no use of old gate/final for tuning',
        original_protocol_sha256=digest(P/'protocol.json'),source_sha256=digest(Path(__file__)))
    atomic_json(out/'proposal.json',study)
    results={}
    for label,candidate in [('B0',b0),('B1',b1)]:
        rows=evaluate(development,'diff3',candidate=candidate)
        stats=summary(rows)
        results[label]=dict(summary=stats,physical_passed=sum(r['physical_safety_passed'] for r in rows),
            design_passed=sum(r['design_joint_passed'] for r in rows),runs=rows)
        atomic_json(out/(label+'.json'),results[label]);print(label,stats,'physical/design',results[label]['physical_passed'],results[label]['design_passed'],flush=True)
    b=results['B1'];admitted=all(r['summary']['complete'] for r in results.values()) and b['physical_passed']==96 and b['design_passed']>=92 and b['summary']['success_count']>=92
    atomic_json(out/'feasibility.json',dict(verified=True,admitted=admitted,classical_episode_count=192,
        results={k:{n:v for n,v in r.items() if n!='runs'} for k,r in results.items()},training_steps=0,
        proposal_sha256=digest(out/'proposal.json'),source_sha256=digest(Path(__file__)),old_gate_or_final_replayed=False,
        next='Finish continuous-run trainer/final-cut auditing then launch bounded six-run pilot after source freeze' if admitted else 'Do not start pilot; diagnose retained classical failures without trimming or weakening baseline'))
    print('CLASSICAL ADMISSION',admitted,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
