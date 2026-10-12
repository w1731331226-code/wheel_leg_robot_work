"""Static dense-evidence resource rejection; never constructs a physical environment."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'wheelleg_warp/results/paper_recovery_20261004/joint_reference_candidate_v1/cartesian_classical_candidate_v1'
GIB = 1024 ** 3
# Preserve the frozen dense stream layouts, including repeated physical witnesses.
STREAMS = {
    'complete_trace': (112, 8), 'physical_pre': (39, 8),
    'physical_post': (39, 8), 'contact_meta': (6, 8),
    'gyro_trace': (8, 8), 'role_trace': (11, 8),
    'phase_trace': (6, 8), 'parking_trace': (30, 8),
    'joint_guard_trace': (62, 8), 'cartesian_trace': (38, 8),
    'cartesian_clock': (3, 8),
}
ACTOR_BYTES = 968 * 4
CONTACT_BYTES = 29 * 8 + 8  # Every contact plus its original physical-step index.


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def amounts(steps, packets):
    if type(steps) is not int or type(packets) is not int or steps < 0 or packets < 0:
        raise ValueError('Nonnegative integer world-step and world-packet caps required')
    return {name: steps * width * itemsize for name, (width, itemsize) in STREAMS.items()} | {
        'actor_raw481_raw481_submitted6': packets * ACTOR_BYTES}


def fits(peak_bytes, free_bytes, cap_bytes=4*GIB, reserve_bytes=2*GIB):
    return peak_bytes <= cap_bytes and peak_bytes + reserve_bytes <= free_bytes


def report():
    proposal = OUT / 'proposal.json'; job_file = OUT / 'job_budget_review.json'
    p = json.loads(proposal.read_text()); jobs = json.loads(job_file.read_text())
    assert jobs['proposal_sha256'] == sha(proposal)
    budget = p['prospective_budget']; total = jobs['totals']
    steps, packets = total['graph_worldstep_cap'], total['world_packet_cap']
    assert steps == budget['task_graph_world_step_cap'] == 12839760
    assert packets == budget['world_packet_cap'] == 320994 and steps == 40*packets
    assert total['constructors'] == 68 and total['episodes'] == 584
    assert budget['new_evidence_transient_git_peak_cap_B'] == 4*GIB
    assert budget['free_reserve_B'] == 2*GIB
    fields = amounts(steps, packets); dense_bytes = sum(fields.values())
    # This bound requires binding the installed put_data capacity resolver as well.
    io_file = ROOT / '.venv/lib/python3.10/site-packages/mujoco_warp/_src/io.py'
    models = ROOT / 'wheelleg_warp/native/models.py'
    assert 'nworld=n,nconmax=64,njmax=256' in models.read_text()
    assert 'return n * nworld' in io_file.read_text()
    contact_rows = 64*steps; contact_bytes = CONTACT_BYTES*contact_rows
    raw_bytes = dense_bytes + contact_bytes
    pack = budget['git_pack_cap_B']
    # Charge archive + additional Git representation + one pack/temp allowance.
    # ZIP/Git compressibility, model/metadata overhead and failure copies are not assumed.
    no_contact_peak = 2*dense_bytes + 2*pack
    complete_peak = 2*raw_bytes + 2*pack
    largest = max(jobs['jobs'], key=lambda j: j['graph_worldstep_cap_per_constructor'])
    job_steps = largest['graph_worldstep_cap_per_constructor']
    job_raw = sum(amounts(job_steps, job_steps//40).values()) + 64*job_steps*CONTACT_BYTES
    free = shutil.disk_usage(ROOT).free
    assert not fits(no_contact_peak, free)
    return dict(
        round=379, verified=False, status='resource_rejected_before_constructor_or_FD',
        verified_static_layout_arithmetic=True, resource_admitted=False,
        proposal_sha256=sha(proposal), job_budget_review_sha256=sha(job_file),
        durable_root=None, workspace_peak_B=None, free_reserve_B=2*GIB,
        required_durable_free_B=None, archive_strategy_frozen=False,
        input_sha256={str(f.relative_to(ROOT)): sha(f) for f in (proposal, job_file, models)},
        installed_capacity_resolver_sha256={str(io_file): sha(io_file)},
        worldstep_cap=steps, worldpacket_cap=packets,
        stream_layouts={n: dict(columns=w, dtype_bytes=b, per_worldstep_bytes=w*b,
                               workload_payload_bytes=fields[n]) for n,(w,b) in STREAMS.items()},
        actor_layout='Original raw481 + second identical raw481 (no normalizer) + submitted6 float32',
        actor_payload_bytes=fields['actor_raw481_raw481_submitted6'],
        dense_without_contacts_payload_bytes=dense_bytes,
        contacts=dict(columns=29, dtype_bytes=8, step_index_bytes=8,
            per_contact_bytes=CONTACT_BYTES, capacity_per_allocated_world=64,
            allocation='naconmax=64*num_envs shared across worlds; not a per-world contact censor',
            workload_capacity_rows=contact_rows, workload_payload_bytes=contact_bytes,
            actual_contact_rows_unknown=True, require_model_capacity_validation_before_launch=True),
        dense_and_capacity_contacts_payload_bytes=raw_bytes,
        archive_and_Git_without_contacts_plan_bytes=no_contact_peak,
        archive_and_Git_with_capacity_contacts_plan_bytes=complete_peak,
        workspace_free_bytes=free, required_free_reserve_bytes=2*GIB,
        allowed_peak_bytes=4*GIB, current_peak_guarantee=False,
        excluded_extra_bytes=['models/geometry/metadata/initial/terminal/private snapshots',
                              'Additional classical latent3 and diagnostic arrays',
                              'NPY/ZIP headers and worst-case compression expansion',
                              'Git object headers/pack deltas and interruption copies'],
        CPU_memory=dict(largest_job_graph_worldsteps=job_steps,
            complete_job_payload_bytes=job_raw,
            old_chunk_lists_plus_concatenated_job_allowance_bytes=2*job_raw,
            allocator_model_numpy_copy_overhead_not_yet_bounded=True,
            CPU_memory_cap_registered=False,
            note='The frozen4GiB cap applies to evidence/temp/Git disk peak, not CPU memory. Host recorder overhead and release between jobs still need a separate bound.'),
        possible_exact_redundancy=['No-NN actor second raw481 duplicates the first; actual bit equality must be checked',
            'pre/post39 originate float32, so storing their exact original precision removes f64 promotion',
            'Adjacent pre q/v equal preceding post q/v only after explicit bitwise equality; retain mismatches',
            'History rolling frames can be shared only after verifying exact reconstruction of all raw481'],
        compact_codec_implemented=False,
        claim='This is a conservative byte budget, not measured compression or an irreducible information bound.',
        decision='No physical/FD launch. Preserve complete physical/contact/control evidence; obtain durable storage and bound CPU streaming/Git transients before admission.',
        new_physics_steps=0, new_FD_calls=0, new_model_forward_rows=0)


def self_check():
    assert sum(w*b for w,b in STREAMS.values()) == 2832
    assert sum(amounts(12839760, 320994).values()) == 37605089088
    assert fits(4*GIB, 6*GIB) and not fits(4*GIB+1, 6*GIB+1)
    assert not fits(4*GIB, 6*GIB-1)
    for values in ((-1, 0), (True, 0), (0, .5)):
        try: amounts(*values)
        except ValueError: pass
        else: raise AssertionError('Invalid resource cap accepted')


if __name__ == '__main__':
    self_check()
    print(json.dumps(report(), ensure_ascii=False, indent=2))
