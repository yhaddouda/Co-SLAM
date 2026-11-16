"""Deferred ScanNet acquisition/extraction. Called only under exclusive ownership."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import urllib.request

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'prepared_data/scans'
def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require_gate():
    assert (ROOT/'NORMAL_EXHAUSTION_TRANSITION.json').is_file(),'No normal-exhaustion authorization'
    assert os.environ.get('DASIP_POST_V4_EXCLUSIVE')=='1','Exclusive worker required'
def reader():
    """Fetch and pin the official historical reader only in the deferred idle window."""
    require_gate();target=ROOT/'data_utilities';target.mkdir(exist_ok=True)
    if (target/'READER_PROVENANCE.json').exists():return target
    revision=subprocess.check_output(['git','ls-remote','https://github.com/ScanNet/ScanNet.git','refs/heads/master'],text=True,timeout=90,env=dict(os.environ,GIT_TERMINAL_PROMPT='0')).split()[0]
    assert len(revision)==40 and all(c in '0123456789abcdef' for c in revision)
    provenance={'repository':'https://github.com/ScanNet/ScanNet','revision':revision,'files':{},'edits':[]}
    from lib2to3.refactor import RefactoringTool,get_fixers_from_package
    tool=RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
    import ast
    for name in ['SensorData.py','reader.py']:
        url=f'https://raw.githubusercontent.com/ScanNet/ScanNet/{revision}/SensReader/python/{name}'
        with urllib.request.urlopen(url,timeout=90) as response:raw=response.read(500001)
        assert len(raw)<500000,'Unexpected reader source size'
        (target/(name+'.original')).write_bytes(raw);source=raw.decode()
        converted=str(tool.refactor_string(source if source.endswith('\n') else source+'\n',name))
        converted=converted.replace("''.join(struct.unpack", "b''.join(struct.unpack").replace('"".join(struct.unpack','b"".join(struct.unpack')
        tree=ast.parse(converted)
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['eval','exec','compile','__import__'] for n in ast.walk(tree))
        assert not any(isinstance(n,ast.Import) and any(x.name in ['subprocess','socket','requests'] for x in n.names) for n in ast.walk(tree))
        if name=='SensorData.py':
            names={n.name for n in ast.walk(tree) if isinstance(n,ast.FunctionDef)}
            assert {'load','export_color_images','export_depth_images','export_poses','export_intrinsics'}<=names
            assert 'version == 4' in converted or 'version==4' in converted,'Unexpected .sens format version'
        (target/name).write_text(converted)
        provenance['files'][name]=dict(url=url,original_sha256=hashlib.sha256(raw).hexdigest(),adapted_sha256=sha(target/name))
    # pypng compatibility is confined to this data utility; 16-bit depth is losslessly encoded.
    if importlib.util.find_spec('png') is None:
        (target/'png.py').write_text('import cv2\nimport numpy as np\nclass Writer:\n def __init__(self,width,height,bitdepth=16,greyscale=True,**kwargs):\n  assert bitdepth==16 and greyscale\n  self.width,self.height=width,height\n def write(self,stream,rows):\n  a=np.asarray(list(rows),dtype=np.uint16).reshape(self.height,self.width)\n  ok,b=cv2.imencode(".png",a);assert ok\n  stream.write(b.tobytes())\n')
        provenance['png_adapter_sha256']=sha(target/'png.py')
    provenance['edits']=['lib2to3 Python-3 syntax conversion','Join struct c fields as bytes; preserve decompression/pixels/frame IDs/poses/intrinsics','Optional local pypng Writer adapter uses lossless OpenCV uint16 PNG, no resizing/filtering']
    save(target/'READER_PROVENANCE.json',provenance);return target
def validate_scene(scene,path):
    require_gate()
    import numpy as np
    import cv2
    n=scene['frames'];ids=set(range(n))
    def indices(folder,suffix):
        items=list((path/folder).glob('*'+suffix));return {int(p.stem) for p in items},len(items)
    for folder,suffix in [('color','.jpg'),('depth','.png'),('pose','.txt')]:
        present,count=indices(folder,suffix);assert present==ids and count==n,(scene['scene'],folder,count,n)
    invalid=[]
    for i in range(n):
        p=np.loadtxt(path/'pose'/f'{i}.txt');assert p.shape==(4,4)
        if not np.isfinite(p).all():invalid.append(i)
    intrinsics={p.name:np.loadtxt(p).tolist() for p in (path/'intrinsic').glob('*.txt')}
    assert intrinsics and 'intrinsic_color.txt' in intrinsics and 'intrinsic_depth.txt' in intrinsics
    probes=[]
    for i in sorted({0,n//2,n-1}):
        color=cv2.imread(str(path/'color'/f'{i}.jpg'));depth=cv2.imread(str(path/'depth'/f'{i}.png'),-1)
        assert color is not None and depth is not None and depth.dtype==np.uint16
        probes.append(dict(frame=i,color_shape=list(color.shape),depth_shape=list(depth.shape),depth_dtype=str(depth.dtype),pose_finite=i not in invalid))
    return dict(status='ready',scene=scene['scene'],path=str(path.resolve()),frames=n,frame_ids_complete=True,invalid_pose_indices_preserved=invalid,intrinsics=intrinsics,probes=probes,pose_axes_preflipped=False,frame_stride=1)
def download(scene):
    require_gate();authorization=read(ROOT/'SCANNET_ACCESS_AUTHORIZATION.json');assert authorization['terms_already_accepted'] is True
    out=ROOT/'downloaded_data';out.mkdir(exist_ok=True)
    destination=out/'scans'/scene['raw_scan_id']/(scene['raw_scan_id']+'.sens')
    if destination.is_file():return destination
    assert shutil.disk_usage(ROOT).free>8*1024**3,'Less than 8 GiB free before acquiring a scene; storage preparation blocked'
    # The inherited prompts are answered only from the author's recorded existing terms acceptance.
    # HTTPS transport fallback changes no selected release/scan/filetype or access requirement.
    for protocol in ['http','https']:
        code="import importlib.util,sys,socket;socket.setdefaulttimeout(120);p=sys.argv[1];spec=importlib.util.spec_from_file_location('approved_downloader',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.BASE_URL=sys.argv[2];m.TOS_URL=m.BASE_URL+'ScanNet_TOS.pdf';sys.argv=[p,*sys.argv[3:]];m.main()"
        argv=[sys.executable,'-u','-c',code,str(ROOT/'download/download-scannet.py'),protocol+'://kaldir.vc.cit.tum.de/scannet/','-o',str(out),'--id',scene['raw_scan_id'],'--type','.sens']
        with (ROOT/'logs'/(scene['scene']+'_download_'+protocol+'.log')).open('x') as log:
            result=subprocess.run(argv,input='\n\n',text=True,stdout=log,stderr=subprocess.STDOUT,timeout=6*3600)
        if result.returncode==0 and destination.is_file():return destination
    raise RuntimeError('Inherited ScanNet acquisition failed for both transports; logs preserved, no credentials or access fabricated')
def prepare_all():
    require_gate();DATA.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True);results=[]
    scenes=read(ROOT/'SCANNET_SCENES.json')['scenes']
    utilities=None
    for scene in scenes:
        try:
            candidates=[DATA/scene['raw_scan_id'],Path(scene['datadir']),Path('/home/yh279050/work/data/scans')/scene['raw_scan_id'],Path('/home/yh279050/work/data/scannet')/scene['raw_scan_id']]
            existing=next((p for p in candidates if (p/'color').is_dir() and (p/'depth').is_dir() and (p/'pose').is_dir()),None)
            if existing:
                record=validate_scene(scene,existing);record['preparation']='existing full extraction reused'
            else:
                sens=download(scene)
                if utilities is None:utilities=reader()
                target=DATA/scene['raw_scan_id'];assert not target.exists(),'Partial extraction preserved; no silent overwrite: '+str(target)
                argv=[sys.executable,str(utilities/'reader.py'),'--filename',str(sens),'--output_path',str(target),'--export_color_images','--export_depth_images','--export_poses','--export_intrinsics']
                with (ROOT/'logs'/(scene['scene']+'_extract.log')).open('x') as log:
                    process=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,cwd=utilities,timeout=6*3600)
                assert process.returncode==0,'Reader failed: '+str(ROOT/'logs'/(scene['scene']+'_extract.log'))
                record=validate_scene(scene,target);record.update(preparation='approved single .sens download and pinned full official-reader export',raw_archive=str(sens),raw_sha256=sha(sens),reader_provenance=str(utilities/'READER_PROVENANCE.json'))
        except Exception as e:record=dict(status='blocked',scene=scene['scene'],reason=repr(e),traceback=traceback.format_exc())
        results.append(record);save(ROOT/'SCANNET_PREPARATION_STATUS.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),phase='PREPARING_ALL_SCENES',scenes=results))
    save(ROOT/'SCANNET_PREPARATION_STATUS.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),phase='ALL_PREPARATION_ATTEMPTS_FINISHED_BEFORE_MEASUREMENTS',scenes=results,no_background_preparation=True))
    return results
