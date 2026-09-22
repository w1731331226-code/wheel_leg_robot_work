"""Paired interventions on fresh development cases; no learning or holdout access."""
from pathlib import Path
import argparse,json,sys
from dataclasses import asdict,replace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from native.environment import NativeEnv
from native.models import batch
from native.terrain import model as terrain_model,sample_terrain_v4
from dashboard.live_env import atomic_json
from terrain_eval import evaluate_terrain
from ppo_env import sample_scenario
from native.terrain import TerrainScenario

def run(cases,checkpoint,margin):
    cases=[replace(s,lateral_margin_m=margin) for s in cases]
    def factory(n,stage,seed,scenario):
        return batch([terrain_model(s) for s in cases],cases)
    raw=NativeEnv(len(cases),bank_factory=factory)
    env=VecNormalize.load(str(checkpoint)+'.pkl',raw);env.training=False;env.norm_reward=False
    policy=PPO.load(str(checkpoint)+'.zip',device='cpu');obs=env.reset()
    pending=set(range(len(cases)));traces=[[] for _ in cases];found={};central_seen=np.zeros(len(cases),dtype=int)
    wheel_dofs=raw.k['ids'].numpy()[8:10];central=raw.cpu.geom('terrain_00').id
    try:
        while pending:
            # Before step_wait auto-reset: sample state from the just-executed graph.
            env.step_async(policy.predict(obs,deterministic=True)[0])
            q=raw.data.qpos.numpy();g=raw.data.geom_xpos.numpy();state=raw.state.numpy();diag=raw.diag.numpy();cs=raw.k['state'].numpy();ctrl=raw.data.ctrl.numpy();vel=raw.data.qvel.numpy()
            count=int(raw.data.nacon.numpy()[0]);contacts=raw.data.contact.geom.numpy()[:count];worlds=raw.data.contact.worldid.numpy()[:count];distance=raw.data.contact.dist.numpy()[:count]
            for j in np.flatnonzero(distance<=0.):
                w=int(worlds[j])
                if w not in pending:continue
                main=(central,) if cases[w].terrain=='cross_slope' else (central,central+1)
                a,b=contacts[j]
                for side,wheel_id in enumerate(raw.wheel_geom_ids):
                    if (a==wheel_id and b in main) or (b==wheel_id and a in main):central_seen[w]|=1<<side
            for i in pending:
                qw,qx,qy,qz=q[i,3:7];yaw=np.degrees(np.arctan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz)))
                wheels=g[i,raw.wheel_geom_ids]
                traces[i].append([float(state[i,0])*.0005,float(yaw),float(state[i,1]),*wheels.ravel().tolist(),float(diag[i,12]),float(diag[i,13]),float(cs[i,8]),float(cs[i,9]),*diag[i,4:6].tolist(),*diag[i,10:12].tolist(),*ctrl[i,4:6].tolist(),*vel[i,wheel_dofs].tolist()])
            obs,_,done,infos=env.step_wait()
            for i in list(pending):
                if done[i]:found[i]={k:v for k,v in infos[i].items() if k!='terminal_observation'};pending.remove(i)
    finally:env.close()
    rows=[]
    for i,s in enumerate(cases):
        trace=np.asarray(traces[i]);wheel=trace[:,3:9].reshape(-1,2,3)
        in_x=(np.abs(np.sign(s.speed)*wheel[:,:,0]-s.center)<.65).all(axis=1)
        width=(.18 if s.terrain=='split_level' else .16)+margin
        beyond=in_x & (np.max(np.abs(wheel[:,:,1]),axis=1)>width)
        exceed=np.abs(trace[:,1])>5
        first=lambda mask:float(trace[np.flatnonzero(mask)[0],0]) if mask.any() else None
        rows.append(dict(seed=s.terrain_seed,scenario=asdict(s),**found[i],central_surface_wheel_contact_mask=int(central_seen[i]),
            first_wheel_center_off_edge_s=first(beyond),first_yaw_over_5_s=first(exceed),
            max_wheel_abs_y_on_terrain=float(np.max(np.abs(wheel[in_x,:,1]))) if in_x.any() else None,
            trace_columns=['time','yaw_deg','arrival_s','left_x','left_y','left_z','right_x','right_y','right_z','lambda','base_infeasible','filtered_yaw_rate','heading_target','base_left','base_right','residual_left','residual_right','ctrl_left','ctrl_right','wheel_speed_left','wheel_speed_right'],trace=traces[i]))
    return rows

def plot(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
    for row,seed in enumerate((470044,470054)):
        for name,label in [('original','Abrupt entry / narrow'),('transition_wide','Ramped entry / lateral margin')]:
            records=json.loads((directory/(name+'.json')).read_text())['runs'];r=next(x for x in records if x['seed']==seed);a=np.asarray(r['trace'])
            axes[row,0].plot(a[:,0],a[:,1],label=label)
            axes[row,1].plot(a[:,0],np.sign(r['scenario']['speed'])*(a[:,3]+a[:,6])/2-r['scenario']['center'],label=label)
        axes[row,0].axhline(5,color='gray',ls='--');axes[row,0].axhline(-5,color='gray',ls='--')
        axes[row,1].axhline(-.65,color='gray',ls='--',label='Central terrain entrance')
        axes[row,0].set_ylabel('Yaw (deg)');axes[row,1].set_ylabel('Wheel-center progress from terrain center (m)')
        for ax in axes[row]:ax.set_xlabel('Actual simulation time (s)');ax.grid(alpha=.25);ax.legend(fontsize=8);ax.set_title(f'Seed {seed}: '+('cross slope' if row==0 else 'split level'))
    fig.suptitle('Same policy and control: geometry intervention isolates entry impact')
    fig.savefig(directory/'root_cause.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--validation',action='store_true');p.add_argument('--plot-only',action='store_true');a=p.parse_args()
    if a.plot_only:plot(a.output);sys.exit(0)
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[];remaining={'cross_slope':24,'split_level':24};seed=520000 if a.validation else 470000
    while any(remaining.values()):
        s=sample_terrain_v4(seed,'ood');seed+=1
        if remaining.get(s.terrain,0):cases.append(replace(s,transition_run_m=0.,lateral_margin_m=0.));remaining[s.terrain]-=1
    checkpoint=Path(json.loads((ROOT/'wheelleg_warp/results/terrain_v3_1024_20260922/selection.json').read_text())['best']['path'])
    if a.validation:
        for i in range(16):
            s=sample_scenario('test_iid',530000+i,3)
            cases.append(TerrainScenario(**asdict(s),terrain_seed=530000+i))
        for hi,h in enumerate((.015,.018,.021)):
            for si,speed in enumerate((.5,.7,.9)):
                for rep in range(4):
                    seed=540000+hi*100+si*10+rep
                    s=sample_scenario('train',seed,3)
                    cases.append(TerrainScenario(**{**asdict(s),'speed':speed*(-1 if rep<2 else 1),'height_l':0.,'height_r':0.},terrain='step',step_height_m=h,terrain_seed=seed,relative_attitude=True))
        policy=PPO.load(str(checkpoint)+'.zip',device='cpu')
        for name,fix in [('original',False),('fixed',True)]:
            selected=[replace(s,transition_run_m=.4,lateral_margin_m=.35) if fix and s.terrain in remaining else s for s in cases]
            rows=evaluate_terrain(policy,str(checkpoint)+'.pkl',selected)
            summary={kind:dict(total=len(rs),success=sum(r['success'] for r in rs),complete=sum(r['reason']=='completed' for r in rs),exit=sum(r['terrain_evidence_passed'] for r in rs)) for kind in ('cross_slope','split_level','legacy','step') for rs in [[r for r in rows if r['scenario']['terrain']==kind]]}
            atomic_json(a.output/(name+'.json'),dict(summary=summary,runs=rows));print(name,summary,flush=True)
        sys.exit(0)
    for name,margin,transition in [('original',0.,0.),('wide',.35,0.),('transition',0.,.4),('transition_wide',.35,.4)]:
        rows=run([replace(s,transition_run_m=transition) for s in cases],checkpoint,margin)
        summary={kind:dict(total=len(rs),success=sum(r['success'] for r in rs),complete=sum(r['reason']=='completed' for r in rs),off_edge=sum(r['first_wheel_center_off_edge_s'] is not None for r in rs)) for kind in remaining for rs in [[r for r in rows if r['scenario']['terrain']==kind]]}
        atomic_json(a.output/(name+'.json'),dict(checkpoint=str(checkpoint),lateral_margin=margin,summary=summary,runs=rows))
        print(name,json.dumps(summary),flush=True)
    plot(a.output)
