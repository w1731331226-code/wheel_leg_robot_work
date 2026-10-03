"""Authoritative registered queue table; partial runs never count as completed evidence."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_queue as q
HERE=Path(__file__).resolve().parent

def current_json(path):
    for _ in range(3):
        try:return q.read(path)
        except FileNotFoundError:return None
        except json.JSONDecodeError:time.sleep(.1)
    return None

p,registration=q.protocol();rows=[]
for job in registration['queue']:
    method,seed=job['method'],job['seed'];directory=q.P/'runs'/method/str(seed);state=q.snapshot(q.unit(method,seed));progress=current_json(directory/'progress.json');selection=current_json(directory/'selection.json')
    status='not_started';complete=False
    if state.get('ActiveState') in ('active','activating','deactivating'):status='running'
    elif directory.exists():
        try:q.complete(directory,p);complete=True;status='completed'
        except (AssertionError,FileNotFoundError,KeyError,json.JSONDecodeError):status='needs_audit'
    best=selection.get('best') if selection else None
    rows.append(dict(method=method,seed=seed,status=status,active_state=state.get('ActiveState'),main_pid=int(state.get('MainPID','0')),
        consumed_policy_steps=progress['executed_policy_steps'] if progress else None,evaluation_count=selection['evaluation_count'] if selection else 0,
        selected_policy_steps=best['policy_steps'] if best else None,development_success=best['summary']['success_count'] if best else None,
        development_yaw_score_deg=best['summary']['mean_yaw_score_deg'] if best else None,
        completed_registered_budget=complete,development_eligible_for_full_budget_comparison=complete))
assert sum(r['status']=='running' for r in rows)<=1
report=dict(created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=q.sha(q.P/'protocol.json'),registered_runs=9,
 completed_runs=sum(r['completed_registered_budget'] for r in rows),rows=rows,
 status_scope='Live process and mutable-file snapshot; refresh before use. Development scores are not independent gate performance.',
 research_gate_ready=all(r['completed_registered_budget'] for r in rows),gate_opened=(q.P/'gate_opened.json').exists())
path=HERE/'registered_results.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(path)
with (HERE/'registered_results.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
print('REGISTERED table:',report['completed_runs'],'/9 complete;',[(r['method'],r['seed'],r['status'],r['consumed_policy_steps']) for r in rows if r['status']!='not_started'])
