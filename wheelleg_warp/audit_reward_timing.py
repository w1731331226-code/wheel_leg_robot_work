"""Offline reward-timing audit, using all saved round90 public trajectories."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/public_path_reward_diagnostic'
PROTOCOL=ROOT/'wheelleg_warp/results/thirtieth_round_admission_review_20261003/protocol_gpu_v3/protocol.json'
METHODS=['B1','M3','B2-V','B2']
COL=['time_s','body_x_m','body_y_m','yaw_rad','return','yaw_cost','dense_return',
     'contact_mask','design_margin_rad','min_actual_chain_m','irreversible_constraint_breach']
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def run(out):
    protocol=json.loads(PROTOCOL.read_text());gamma=protocol['ppo']['gamma'];lam=protocol['ppo']['gae_lambda'];horizon=protocol['ppo']['n_steps']
    assert gamma==.99 and lam==.95 and horizon==50
    pre=json.loads((INPUT/'preregistration.json').read_text());old=json.loads((INPUT/'summary.json').read_text())
    assert pre['protocol_sha256']==old['protocol_sha256']==sha(PROTOCOL)
    assert pre['verifier_sha256']==old['verifier_sha256']==sha(ROOT/'wheelleg_warp/diagnose_paper_recovery.py')
    assert all(sha(ROOT/name)==value for name,value in protocol['source_sha256'].items())
    names=['preregistration.json','summary.json']+[m+s for m in METHODS for s in ['.json','_trace.npz']]
    out.mkdir(parents=True,exist_ok=False)
    write(out/'analysis_registration.json',dict(role='Exploratory offline analysis after public diagnostics; no prospective learning claim',
        input_sha256={name:sha(INPUT/name) for name in names},gamma=gamma,gae_lambda=lam,rollout_steps=horizon,
        beta=10.,beta_rule='Use original terminal penalty amplitude; no scan',formula='Phi(initial)=Phi(terminal)=0; intermediate Phi=-10 if historical irreversible breach; r_shaped=r+gamma*Phi_next-Phi_current',
        naive_comparator='Move -10 terminal penalty to first sampled breach without discount compensation',
        all_128_trajectories_used=True,training_updates=0,physics_replays=0,source_sha256=sha(Path(__file__))))
    details=[];groups={};arrays=[];lengths=[];max_return_error=0.;max_td_error=0.
    for method in METHODS:
        rows=json.loads((INPUT/(method+'.json')).read_text())['runs'];z=np.load(INPUT/(method+'_trace.npz'),allow_pickle=False)
        assert len(rows)==32 and list(z['columns'])==COL and len(z['offsets'])==33
        assert z['offsets'][0]==0 and z['offsets'][-1]==len(z['trace'])
        for i,row in enumerate(rows):
            assert row['seed']==pre['cases'][i]['seed'] and row['scenario']==pre['cases'][i]['scenario']
            t=z['trace'][z['offsets'][i]:z['offsets'][i+1]];assert np.isfinite(t).all()
            assert np.all(np.diff(t[:,0])>0) and len(t)==row['episode']['l']
            assert abs(t[-1,4]-row['episode']['r'])<1e-12 and abs(t[-1,6]-row['exact_dense_return'])<1e-12
            assert abs(t[-1,4]-t[-1,6]-(10 if row['success'] else -10))<1e-12
            rewards=np.diff(np.r_[0.,t[:,4]]);weights=gamma**np.arange(len(t))
            np.testing.assert_allclose(rewards.sum(),row['episode']['r'],atol=1e-11,rtol=0)
            breach=t[:,10].astype(bool);assert np.all(np.diff(t[:,10])>=0)
            potential=np.r_[0.,-10.*breach.astype(float)];potential[-1]=0.
            shaped=rewards+gamma*potential[1:]-potential[:-1]
            return_error=abs(float(weights@(shaped-rewards)));max_return_error=max(max_return_error,return_error)
            assert return_error<1e-10
            # If V'(state)=V(state)-Phi(state), TD residuals also remain identical.
            delta_td=shaped+gamma*(-potential[1:])-(-potential[:-1])-rewards
            td_error=float(abs(delta_td).max());max_td_error=max(max_td_error,td_error);assert td_error<1e-12
            hit=np.flatnonzero(breach);lag=None;naive=rewards.copy();naive_change=0.;dense_after=None;delay=None
            if len(hit):
                j=int(hit[0]);assert not row['success']
                lag=len(t)-1-j;assert t[j,0]==row['first_irreversible_constraint_breach_policy_s']
                delay=float(t[-1,0]-t[j,0]);dense_after=float(t[-1,6]-t[j,6])
                assert abs(dense_after-row['snapshot_return_after_first_breach'])<1e-12
                naive[j]-=10;naive[-1]+=10
                np.testing.assert_allclose(naive.sum(),rewards.sum(),atol=1e-11,rtol=0)
                naive_change=float(weights@(naive-rewards))
                np.testing.assert_allclose(naive_change,10*(gamma**(len(t)-1)-gamma**j),atol=1e-12,rtol=0)
            else:
                assert row['first_irreversible_constraint_breach_policy_s'] is None
                np.testing.assert_array_equal(shaped,rewards)
            details.append(dict(method=method,seed=row['seed'],success=row['success'],sampled_breach=bool(len(hit)),
                lag_policy_steps=lag,feedback_delay_s=delay,dense_return_after_first_sampled_breach=dense_after,
                terminal_discount_factor_from_breach=gamma**lag if lag is not None else None,
                whole_episode_GAE_reward_path_factor=(gamma*lam)**lag if lag is not None else None,
                raw_terminal_reward_can_share_rollout_with_breach=(lag<horizon) if lag is not None else None,
                naive_move_discounted_return_change=naive_change,PBRS_discounted_return_error=return_error))
            arrays.append(np.column_stack([t[:,0],rewards,shaped,naive,potential[1:]]));lengths.append(len(t))
        group=[d for d in details if d['method']==method];hits=[d for d in group if d['sampled_breach']]
        groups[method]=dict(episodes=32,failures=sum(not d['success'] for d in group),sampled_breach_cases=len(hits),
            positive_dense_after_breach=sum(d['dense_return_after_first_sampled_breach']>0 for d in hits))
    hits=[d for d in details if d['sampled_breach']];fails=sum(not d['success'] for d in details)
    assert len(details)==128 and fails==49 and len(hits)==8
    np.savez_compressed(out/'rescore.npz',columns=np.array(['time_s','original_reward','potential_shaped_reward','naive_shift_reward','potential_next']),
        offsets=np.cumsum([0]+lengths),trace=np.concatenate(arrays))
    report=dict(verified=True,episodes=128,failed_episodes=fails,sampled_breach_failures=len(hits),failure_without_sampled_breach=fails-len(hits),
        groups=groups,details=details,maximum_PBRS_discounted_return_error=max_return_error,maximum_value_shift_TD_error=max_td_error,
        inference='All8 flagged failures have positive dense return after sampled breach and terminal delay72..226steps; all exceed50-step rollout. Raw terminal reward cannot directly share the breach rollout, but learned value bootstrap can transmit information.',
        decision='Naive penalty advance changes the discounted objective despite conserving undiscounted reward. Potential shaping telescopes to unchanged complete-episode discounted return; use as a known-method controlled mechanism candidate, not a novel algorithm or proven learning benefit.',
        scope='Fixed action/state replay only, first breach bracketed at50Hz, beta/gamma finite episode with Phi(initial/terminal)=0. PPO critic values/actual rollout phases not recorded; whole-episode GAE factors are not actual PPO advantages. Applies to8/49 sampled failures, not a general explanation of all failures.',
        next='Verify online potential bookkeeping, reset/terminal handling and per-rollout bootstrap on a public matched no-update replay before registering new learning seeds. Keep Actor/critic observations and physical/control/evaluation gates unchanged.',
        training_updates=0,physics_replays=0,source_sha256=sha(Path(__file__)),rescore_sha256=sha(out/'rescore.npz'))
    write(out/'result.json',report)
    print(json.dumps({k:report[k] for k in ['episodes','failed_episodes','sampled_breach_failures','failure_without_sampled_breach','maximum_PBRS_discounted_return_error','maximum_value_shift_TD_error','groups']},ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args().output)
