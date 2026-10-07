"""Frozen learning-necessity protocol;runner admission remains separate."""
import ast
import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path
import numpy as np
from review_yaw_sector import ROOT, sha
sys.path.insert(0,str(ROOT/'wheelleg_ppo/tools'))
from native.terrain import sample_height_terrain_115
from floor_broad_adapter import batches
from dashboard.live_env import atomic_json

OUT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/execution_input_history_v1'
PARENT=ROOT/'wheelleg_warp/results/paper_recovery_20261004/nominal_mean_matched_learning_v1/proposal.json'


def compatibility():
    name='wheelleg_warp/execution_input_collector.py'
    old=subprocess.check_output(['git','show','5a99dc7d:'+name])
    contract=json.loads((OUT/'engineering_contract.json').read_text())
    assert hashlib.sha256(old).hexdigest()==contract['source_sha256'][name]
    before,after=ast.parse(old),ast.parse((ROOT/name).read_text())
    a=next(x for x in before.body if isinstance(x,ast.FunctionDef) and x.name=='instrument')
    b=next(x for x in after.body if isinstance(x,ast.FunctionDef) and x.name=='instrument')
    assert b.args.args[-1].arg=='expected_captures' and b.args.defaults[-1].value==1
    b.args.args.pop();b.args.defaults.pop()
    assertions=[i for i,x in enumerate(a.body) if isinstance(x,ast.Assert)]
    assert len(assertions)==1
    i=assertions[0]
    assert '40 * expected_captures' in ast.unparse(b.body[i])
    b.body[i]=copy.deepcopy(a.body[i])
    assert ast.dump(before)==ast.dump(after)
    return dict(verified=True,old_source_sha256=hashlib.sha256(old).hexdigest(),new_source_sha256=sha(ROOT/name),
        default_single_capture_AST_unchanged_except_equivalent_count_assert=True,
        record_clear_interval_host_lifecycle_unchanged=True,dense_expected_captures=2,
        limits='Historical engineering checksum remains historical;new dense source still needs actual interface qualification.')


def run():
    assert not (OUT/'study_proposal.json').exists()
    engineering=json.loads((OUT/'engineering_completion.json').read_text());assert engineering['verified'] and engineering['total_policy_samples']==24000
    delivery=json.loads((OUT/'round232_delivery.json').read_text());assert delivery['verified']
    assert delivery['completion_sha256']==sha(OUT/'engineering_completion.json')
    for n,v in delivery['files_sha256'].items():assert sha(OUT/n)==v
    p=json.loads(PARENT.read_text());seeds=[23301,23302,23303]
    assert set(seeds).isdisjoint(p['seeds'])
    banks={}
    for seed in seeds:
        banks[str(seed)]={}
        for stage in (1,2,3):
            rows=[]
            for i in range(100):
                number=23300000+(seed-23301)*1000+i;s=sample_height_terrain_115(number,stage,'train')
                if stage<3:s=replace(s,mass=7.,delay_ms=0.)
                if stage<2:s=replace(s,mu_l=.8,mu_r=.8,drive_difference=0.)
                rows.append(dict(seed=number,scenario=asdict(s)))
            banks[str(seed)][str(stage)]=rows
    refs={r['path']:r['sha256'] for r in p['references']['phase_result_refs']}
    cpu=p['references']['CPU_legacy'];refs[cpu['path']]=cpu['sha256']
    assert all(sha(ROOT/n)==v for n,v in refs.items())
    classical=json.loads((ROOT/'wheelleg_warp/results/paper_recovery_20261004/phase_support_qualification_v1/proposal.json').read_text())['classical']
    evaljobs=[dict(panel=panel,batch=i,indices=[j for j,_ in group],cases=[c for _,c in group])
              for panel in ('regular','controlled','legacy') for i,group in enumerate(batches(p[panel]))]
    assert sum(len(x['cases']) for x in evaljobs)==164
    comp=compatibility();atomic_json(OUT/'round233_source_compatibility.json',comp)
    proposal=dict(version='execution-input-necessity-v1',round=233,status='protocol_frozen_runner_not_admitted',
        question='With identical sensorhistory/publictimestamps/architecture/Nom/authority/reward,does adding finalexecutedcommandhistory improve newtrainedPPO? H0 is an informationablation,H1 is genericPPO;not a novelalgorithm or sameinformation superiorityclaim.',
        seeds=seeds,arms=['H0','H1'],training_banks=banks,training_bank_rule='Fresh scenario IDs/fresh model initialization;same-seedarms share exact100worlds perstage;sample distribution unchanged. No archived/engineering modelwarmstart.',
        ppo=p['ppo'],normalization=p['normalization'],initial_log_std=p['initial_log_std'],device='cuda',worlds=100,
        policy_samples_each=200000,total_policy_samples=1200000,curriculum_milestones=[20000,100000],checkpoint_interval=20000,
        jobs=[dict(seed=s,arm=a) for s in seeds for a in ('H0','H1')],eval_jobs=evaljobs,
        regular=p['regular'],controlled=p['controlled'],legacy=p['legacy'],references=p['references'],classical=classical,
        new_classical=['B0','B1-route'],evaluation_budget=1312,evaluation_rule='Two contemporary classicalreferences164each+sixfinal200kmodels164each. All development136+legacy28;no auxiliaryor sealedoldgate/final3000. Solverhomogeneous <=20batches,firstepisodeallcase trace/Actor inputs/actions kept.',
        common_information='Both same10raw39frames/9sensorclockage intervals/validmasks,481parameters;H0 only54meancommand slotszero,H1retains. Proxy slotsbothzero. Publicsensor timestamp is additionalideal assumptionand may reveal latencyage;truth force/geometry/gain/mass/friction/arrival/unknownparameter labels excluded.',
        control='SamephaseNom+virtual6+commonpublicseen_nonzero/currentcmd0parkingwithdrawal/originalslew/bounds/task/reward. ContemporaryB1retainsoriginalparkingPD;convert itswheel-onlydiff3 action tovirtual6 [0,0,0,0,a,-a],must sourcequalify;oldB0/B1 neverweakened.',
        selection='Final200000 checkpoint only,evaluateall3seeds;no engineering/earlycheckpointscore,bestseed,partialwinner,retry,gain/std/historylength/budgetrescue.',
        gates=dict(integrity='Allsixfresh continuousruns/fullall164evaluations/source/models/RMS/wholeflag/physics/design/Actor traces correct;incomplete study refuses conclusion.',
          primary='H1 eachseed must haveallphysics/design andpreserveeachsuccess+eachcomponentof contemporaneousANDoldstrongB0/B1;CPU28 fulltask,velocity<=old*1.05+.005,rollpitch<=old+.1. Each controlled>=34/40. Regular96/96 baselinecannot requirepositive successgain.',
          advantage='Original literal everypanel regular/controlled/legacy J15%AND.05deg thresholds versus contemporaryANDoldB0/B1 for eachseed. Do not replace withtwo-panels gate.',
          mechanism='Eachseed H1vsH0 no lostsuccessfulcase/newcomponent/physicaldesign failure;eachpanelJ lowerfor eachseed AND3seedmean15%AND.05deg. Failing mechanismcloses explicitcommandbenefit evenif some scoregood.',
          no_promotion='Informationeffect not proofnovelmethod;no automaticformal5/mainbaseline replacement. Allsixresults retained.'),
        interrupted='Reserve/completed counters and entirefailure/model/Adam/RMS/route/history/command/parking/firstepisode traces preserved. No silentrestart or exacttrajectoryresume claim.',
        after='234 qualify dense two-capture collector/parking/model481RMS/Actorlogger/legacyID mapping+classical mapping and partial ownership,then freeze runner before execution.235 deepreview/cleanup,otherwise no training. Positivecontrast still requires distinguishable contribution,matchedstronggenericRL, freshformal5/ID-OOD/statistics/PPOcost/manuscript.',
        parent_sha256=sha(PARENT),engineering_sha256=sha(OUT/'engineering_completion.json'),reference_input_sha256=refs,
        source_compatibility_sha256=sha(OUT/'round233_source_compatibility.json'),scientific_samples_consumed=0,scientific_evaluations_consumed=0,
        trainer_admitted=False,formal5_admitted=False,entire_goal_complete=False)
    atomic_json(OUT/'study_proposal.json',proposal)
    print('FROZEN233 protocol6freshmodels1.2M/1312finalandclassicalevaluations;runnernotadmitted;0consumed',flush=True)


if __name__=='__main__':run()
