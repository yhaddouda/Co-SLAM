"""CPU-only read snapshot for transfer while an admitted full job is running."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parent
def main():
    state=json.loads((ROOT/'CONTINUOUS_STATUS.json').read_text())
    assert state['phase']=='FULL_RUNS' and state['counts']['running']==1
    target=ROOT.parents[1]/'ORIN_CURRENT_MODE_RESULTS.zip';tmp=target.with_suffix('.snapshot.tmp')
    entries=[('continuous_v4/'+p.relative_to(ROOT).as_posix(),p) for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.tmp','.lock']]
    for job in state['jobs']:
        if job['status']!='reused':continue
        folder=Path(job['output'])
        for p in folder.rglob('*'):
            if p.is_file():entries.append(('reused/'+folder.name+'/'+p.relative_to(folder).as_posix(),p))
        for suffix in ['.done.json','.launch.json','.log']:
            p=folder.with_name(folder.name+suffix)
            if p.is_file():entries.append(('reused/'+folder.name+'/'+p.name,p))
    manifest={}
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,p in entries:
            data=p.read_bytes();manifest[name]=hashlib.sha256(data).hexdigest();z.writestr(name,data)
        z.writestr('RETURN_MANIFEST.json',json.dumps(dict(snapshot_utc=datetime.now(timezone.utc).isoformat(),final_snapshot=False,ALL_REQUIRED_MAXN_COMPLETE=False,counts=state['counts'],live_files_note='Active log/frame progress is a read snapshot, not a completed result; all archived byte hashes are verified',files=manifest),indent=2)+'\n')
    with zipfile.ZipFile(tmp) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    tmp.replace(target);target.with_suffix('.zip.sha256').write_text(hashlib.sha256(target.read_bytes()).hexdigest()+'  '+target.name+'\n')
    print('Verified non-final transfer snapshot',len(manifest),'files',state['counts'])

if __name__=='__main__':main()
