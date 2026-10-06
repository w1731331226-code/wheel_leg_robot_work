"""Describe recorded centre lanes/contact witnesses; never replace old success."""
import json
from pathlib import Path

import numpy as np
from task_mode_recorder import OUT, COL
from review_yaw_sector import ROOT, sha
from score_reference_learning import load_rows
from dashboard.live_env import atomic_json

WITNESSES={6301001,6301003,6301005,6301007,6301009,6301011,6301013,6301015}
I={name:i for i,name in enumerate(COL)}


def local(points,box): return (points-np.asarray(box['position']))@np.asarray(box['rotation'])


def centre_metrics(trace,box,side):
    after=local(trace[:,[I[f'post_{side}_{a}'] for a in 'xyz']],box)
    before=local(trace[:,[I[f'pre_{side}_{a}'] for a in 'xyz']],box)
    half=np.asarray(box['size']); tolerance=1e-7
    span=np.abs(after[:,0])<=half[0]+tolerance
    inside=span&(np.abs(after[:,1])<=half[1]+tolerance)
    pre_inside=(np.abs(before[:,0])<=half[0]+tolerance)&(np.abs(before[:,1])<=half[1]+tolerance)
    witness=(trace[:,I[side+'_vertical_target_geom']]==box['geom'])&(trace[:,I[side+'_max_target_vertical_normal_N']]>1e-6)
    central=np.abs(before[:,0])<=half[0]/2
    point=local(trace[:,[I[side+'_vertical_point_'+a] for a in 'xyz']],box)
    point_xy=(np.abs(point[:,0])<=half[0]+tolerance)&(np.abs(point[:,1])<=half[1]+tolerance)
    n=int(span.sum()); proportion=float(inside.sum()/n) if n else None
    mode='insufficient_longitudinal_samples'
    if n: mode='centre_lateral_for_entire_span' if not inside.any() else ('centre_inside_entire_sampled_span' if inside.sum()==n else 'partial_centre_lane')
    return dict(descriptor=mode,longitudinal_samples=n,centre_inside_samples=int(inside.sum()),centre_inside_fraction=proportion,
        min_abs_lateral_offset_m=float(abs(after[span,1]).min()) if n else None,
        max_abs_lateral_offset_m=float(abs(after[span,1]).max()) if n else None,
        half_lane_m=float(half[1]),max_target_vertical_normal_N=float(trace[:,I[side+'_target_vertical_normal_N']].max()),
        positive_vertical_target_witness_rows=int(witness.sum()),witness_with_pre_centre_inside_rows=int((witness&pre_inside).sum()),
        central_half_longitudinal_rows=int(central.sum()),central_half_positive_vertical_witness_rows=int((central&witness).sum()),
        witness_point_inside_box_xy_rows=int((witness&point_xy).sum()),
        both_vertical_normals_zero_in_span_rows=int((span&(trace[:,I['left_vertical_normal_N']]+trace[:,I['right_vertical_normal_N']]<=1e-6)).sum()),
        interpretation='Centre footprint descriptor, not whole tyre footprint or automatic bypass/loaded-top/flight verdict. Contact witness is solver-time, pre-centre matched separately; other smaller simultaneous contacts may not be the retained maximum witness.')


def unit():
    box=dict(position=[0,0,0],rotation=np.eye(3).tolist(),size=[1,.035,.01],geom=3)
    a=np.zeros((4,len(COL))); a[:,I['post_left_x']]=[-.5,0,.5,1.5]; a[:,I['pre_left_x']]=a[:,I['post_left_x']]
    assert centre_metrics(a,box,'left')['descriptor']=='centre_inside_entire_sampled_span'
    a[:,I['post_left_y']]=a[:,I['pre_left_y']]=.06
    a[:,I['left_target_vertical_normal_N']]=10; a[:,I['left_max_target_vertical_normal_N']]=10; a[:,I['left_vertical_target_geom']]=3
    assert centre_metrics(a,box,'left')['descriptor']=='centre_lateral_for_entire_span'
    assert centre_metrics(a,box,'left')['positive_vertical_target_witness_rows']==4
    a[1,I['post_left_y']]=0; assert centre_metrics(a,box,'left')['descriptor']=='partial_centre_lane'


def run():
    unit(); p=json.loads((OUT/'proposal.json').read_text()); c=json.loads((OUT/'completion.json').read_text())
    prior=json.loads((OUT/'round162_data_review.json').read_text()); assert prior['verified'] and c['completed_evaluations']==205
    result=[]; sources={}; success_witnesses=[]
    for job in c['records']:
        file=OUT/job['path']; assert sha(file)==job['sha256']; rows=load_rows(file,p['cases'])['runs']
        gfile=file.parent/'geometry.json'; assert sha(gfile)==job['geometry_sha256']; geometry=json.loads(gfile.read_text())
        sources[str(file.relative_to(OUT))]=sha(file); sources[str(gfile.relative_to(OUT))]=sha(gfile)
        for w,r in enumerate(rows):
            tfile=file.parent/r['task_mode_trace']['path']; assert sha(tfile)==r['task_mode_trace']['sha256']
            sources[str(tfile.relative_to(OUT))]=sha(tfile)
            with np.load(tfile,allow_pickle=False) as data: assert data['columns'].tolist()==COL; t=data['trace']
            rawdiff=(t[:,I['raw_nom_left']]-t[:,I['raw_nom_right']])/2
            accepted=(t[:,I['nom_left']]-t[:,I['nom_right']])/2
            correction=(t[:,I['nom_correction_left']]-t[:,I['nom_correction_right']])/2
            entry=dict(controller=job['label'],case=r['seed'],success=r['success'],height_m=r['target_leg_m'],terrain=r['scenario']['terrain'],
                original_peak_deg=r['peak_deg'],max_abs_body_y_m=float(abs(t[:,I['body_y']]).max()),
                sampled_reference_clip_fraction=float(np.mean(abs(t[:,I['roll_offset_requested']]-t[:,I['roll_offset_selected']])>1e-9)),
                sampled_preclip_diff_demand_RMS_Nm=float(np.sqrt(np.mean(rawdiff**2))),sampled_accepted_nom_diff_RMS_Nm=float(np.sqrt(np.mean(accepted**2))),
                sampled_nom_correction_diff_RMS_Nm=float(np.sqrt(np.mean(correction**2))),
                limits='Sampled distributions are50Hz/2kHz mixed, not uniformly time-weighted. Raw Nom is before nominal correction; differences are not solely clipping loss.')
            if r['scenario']['terrain']=='legacy':
                assert len(geometry['boxes'][w])==1
                side='left' if r['scenario']['height_l'] else 'right'
                entry['target_side']=side; entry['centre_lane']=centre_metrics(t,geometry['boxes'][w][0],side)
                if r['seed'] in WITNESSES and job['label'].startswith('V6-') and r['success']: success_witnesses.append(entry)
            else: entry['mode']='mixed_multiple_geometry_not_forced_into_legacy_classifier'
            result.append(entry)
    assert len(result)==205 and len(success_witnesses)==13
    counts={name:sum(x['centre_lane']['descriptor']==name for x in success_witnesses) for name in ['centre_lateral_for_entire_span','partial_centre_lane','centre_inside_entire_sampled_span']}
    atomic_json(OUT/'physical_mode_analysis.json',dict(verified=True,evaluations=205,unique_development_cases=41,low_success_records=13,low_unique_success_cases=8,low_success_centre_descriptors=counts,low_success_records_detail=success_witnesses,results=result,
        input_sha256=sources,source_sha256=sha(__file__),completion_sha256=sha(OUT/'completion.json'),data_review_sha256=sha(OUT/'round162_data_review.json'),new_evaluations=0,training_updates=0,
        conclusion='Observed low-height wins often use lateral/edge centre paths while having positive target normal load. Original contact-task success preserved. No complete centred loaded-traversal or channel-learning superiority claim from these witnesses; no cheating/complete bypass accusation from centre alone.',
        next='164 separately define qualification for the intended obstacle/path task before new control/learning. Original gates/results untouched;165 deep review/cleanup.'))
    print('PASS205 physical descriptions;13 low successes over8 cases:',counts,flush=True)


if __name__=='__main__': run()
