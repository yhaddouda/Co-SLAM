"""Read-only interpretation of frozen v4 results under the author's current scope."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parent
V4=ROOT.parent/'orin_continuous_v4'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):
    p=Path(p);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def project(raw,post=None):
    fields=['id','system','dataset','scene','condition','seed','hash_size','status','output','reason']
    jobs=[{k:j[k] for k in fields if k in j} for j in raw['jobs']]
    by_id={j['id']:j for j in jobs}
    for j in jobs:
        if j.get('system')=='EC-SLAM' and j.get('condition')=='ENG':
            j.update(status='excluded',required_for_completion=False,reason='Author decision 19 October 2026: engineering restricted to Co-SLAM; EC BASE/MC unchanged')
        else:j['required_for_completion']=True
    for seed in [0,1]:
        control=dict(id=f'Co_RTXREF_office0_BASE_CONTROL_seed{seed}',system='Co-SLAM',dataset='Replica',scene='office0',condition='BASE_CONTROL',seed=seed,hash_size=16,status='conditional',required_for_completion=True,reason='Additional matched BASE control for validated isolated Co ENG runtime')
        jobs.append(control);by_id[control['id']]=control
    if post:
        for source in post['jobs']:
            key=source['id']
            if source.get('engineering'):
                key=f'Co-SLAM_office0_ENG_H16_seed{source["seed"]}'
            if key in by_id:
                target=by_id[key]
                for field in ['status','output','reason']:
                    if field in source:target[field]=source[field]
                target['current_execution_id']=source['id']
    counts=Counter(j['status'] for j in jobs);required=[j for j in jobs if j['required_for_completion']]
    native=[{k:j.get(k) for k in ['id','system','dataset','scene','condition','seed','hash_size','datadir']} for j in raw['jobs'] if j.get('condition') in ['BASE','MC']]
    return dict(updated_utc=datetime.now(timezone.utc).isoformat(),scope='Co-SLAM engineering only; EC-SLAM BASE/MC unchanged',counts=dict(counts),required_slots=len(required),excluded_slots=len(jobs)-len(required),ALL_REQUIRED_MAXN_COMPLETE=bool(required) and all(j['status'] in ['done','reused'] for j in required),v4_raw_status_preserved=True,native_BASE_MC_identity_sha256=hashlib.sha256(json.dumps(native,sort_keys=True).encode()).hexdigest(),v4_phase=raw['phase'],v4_parent_pid=raw.get('parent_pid'),v4_active_child_pid=raw.get('active_child_pid'),post_phase=post.get('phase') if post else 'WAITING_FOR_NORMAL_V4_EXHAUSTION',jobs=jobs)
def write_view():
    raw=read(V4/'CONTINUOUS_STATUS.json');post=read(ROOT/'POST_V4_STATUS.json') if (ROOT/'POST_V4_STATUS.json').exists() else None
    result=project(raw,post);save(ROOT/'CURRENT_SCOPE_STATUS.json',result);save(V4/'CURRENT_SCOPE_STATUS.json',result)
    text='# Current Orin MAXN scope\n\nUpdated '+result['updated_utc']+' UTC. **Engineering is restricted to Co-SLAM. EC ENG is excluded from scope and completion requirements. EC BASE/MC remains unchanged.**\n\n'
    text+='Counts: '+str(result['counts'])+'. Required slots: '+str(result['required_slots'])+' (includes two explicitly added Co BASE controls); excluded historical EC ENG slots: 2. **ALL_REQUIRED_MAXN_COMPLETE='+str(result['ALL_REQUIRED_MAXN_COMPLETE']).lower()+'**.\n\n'
    text+='This view applies the latest author scope to the preserved v4 execution records and post-v4 results. The original live v4 agenda/status retains its frozen campaign definition. The follower still waits for normal completion of all 168 original ready/reused identities, no active child, and exclusive shared workload ownership; neither clock time nor ALL_REQUIRED is a trigger.\n\n'
    text+='| System | Scene | H | Condition | Seed | Status | Output / reason |\n|---|---|---:|---|---:|---|---|\n'
    for j in result['jobs']:text+='| '+' | '.join(str(j.get(k,'')) for k in ['system','scene','hash_size','condition','seed','status'])+' | '+str(j.get('output',j.get('reason',''))).replace('\n',' ')+' |\n'
    for p in [ROOT/'ORIN_RUN_AGENDA_CURRENT_SCOPE.md',V4/'ORIN_RUN_AGENDA_CURRENT_SCOPE.md']:p.write_text(text)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');args=p.parse_args()
    while True:
        result=write_view()
        if args.once or result['post_phase'] in ['COMPLETE_ADMITTED_WORK','STOPPED']:return
        time.sleep(45)

if __name__=='__main__':main()
