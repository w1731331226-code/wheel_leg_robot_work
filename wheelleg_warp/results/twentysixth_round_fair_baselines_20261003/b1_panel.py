"""Reuse the eight frozen B1 candidates on the same native v6 engineering panel."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'wheelleg_warp'),str(ROOT/'wheelleg_ppo/tools')]
import numpy as np
from native.environment import NativeEnv
from native.terrain import HeightTerrainScenario
from pretrain_yaw import b1_action
from terrain_eval import summarize_terrain
from training_contract import source_hashes,digest
OUT=Path(__file__).resolve().parent
FROZEN=ROOT/'wheelleg_ppo/tools/results/yaw_precision_v1_2026-09-20/training_config.json'
PANEL=OUT.parent/'twentyfourth_round_operating_regions_20261002/registered_v2/registered_cases.json'

def run():
    output=OUT/'b1_engineering_checked';output.mkdir()
    candidates=json.loads(FROZEN.read_text())['b1_candidates']
    scenes=json.loads(PANEL.read_text())['scenarios']
    assert len(candidates)==8 and len(scenes)==18
    protocol=dict(role='engineering_reuse_not_selection_or_gate',candidates=candidates,scenarios=scenes,
        source_sha256=source_hashes(__file__),inputs_sha256={str(p.relative_to(ROOT)):digest(p) for p in (FROZEN,PANEL)},
        final_holdout_opened=False,parameters_retuned=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    raw=NativeEnv.height115_candidate(n=144,scenario=[HeightTerrainScenario(**s) for _ in candidates for s in scenes],shared_reference=True)
    try:
        obs=raw.reset();rows=[None]*144
        deadline=int(np.ceil((raw.param.numpy()[:,3].max()+2.)/.02))+2
        for _ in range(deadline):
            actions=np.stack([b1_action(obs[w],candidates[w//18]) for w in range(144)])
            assert actions.shape==(144,3) and np.isfinite(actions).all() and abs(actions).max()<=1
            obs,_,done,infos=raw.step(actions)
            for w in np.flatnonzero(done):
                if rows[w] is None:rows[w]=dict(seed=int(w%18),scenario=scenes[w%18],**{k:v for k,v in infos[w].items() if k!='terminal_observation'})
            if all(r is not None for r in rows):break
        assert all(r is not None for r in rows)
        summaries=[]
        for i,c in enumerate(candidates):
            part=rows[i*18:(i+1)*18]
            assert all(r['physical_steps']==r['physical_evidence_steps'] for r in part)
            summary=dict(candidate=c,**summarize_terrain(part,list(range(18))),physical=sum(r['physical_safety_passed'] for r in part),design=sum(r['design_joint_passed'] for r in part))
            summaries.append(summary);print('B1 ENGINEERING',c['name'],summary,flush=True)
        assert [r['success'] for r in rows[:18]]==[r['success'] for r in json.loads((OUT.parent/'twentyfifth_round_native_admission_20261003/registered_v2/verification.json').read_text())['episodes']]
        (output/'verification.json').write_text(json.dumps(dict(passed=True,summaries=summaries,runs=rows,selected_candidate=None,formal_selection=False),indent=2)+'\n')
    finally:raw.close()

if __name__=='__main__':run()
