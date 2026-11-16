"""Prepare this machine's native Morton cache and hash/precision probes before NCU."""
from pathlib import Path
import json,os,sys,hashlib
ROOT=Path(__file__).resolve().parent
import torch
from torch.utils.cpp_extension import load
major,minor=torch.cuda.get_device_capability();os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
source=ROOT/'runtime/co/model/round2_morton/morton_cuda.cu';name=f'dasip_round2_morton_{major}{minor}'
ext=load(name=name,sources=[str(source)],extra_cuda_cflags=['-O3'],verbose=True)
points=torch.tensor([[0.,0.,0.],[.5,.5,.5],[.999,.999,.999]],device='cuda',dtype=torch.float32);keys=ext.morton3d_keys(points.contiguous(),128);assert keys.numel()==3 and bool(torch.all(keys[1:]>=keys[:-1]))
sys.path.insert(0,str(ROOT/'runtime/co'))
from model.encodings import get_encoder
records=[]
for h in [16,19]:
 for selector in ['CoherentPrime','Morton']:
  encoding,dims=get_encoder('HashGrid',n_levels=2,level_dim=2,base_resolution=16,desired_resolution=128,log2_hashmap_size=h,hash=selector);v=encoding(points);assert v.dtype==torch.float32 and torch.isfinite(v).all();records.append(dict(hash=selector,hash_size=h,output_dtype=str(v.dtype),parameter_dtype=str(encoding.params.dtype),shape=list(v.shape)));del encoding,v
encoding,dims=get_encoder('OneBlob',n_bins=16);v=encoding(points);assert v.dtype==torch.float32 and torch.isfinite(v).all();torch.cuda.synchronize()
info=dict(status='PASS',device=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,capability=[major,minor],extension=str(ext.__file__),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),probes=records,oneblob_output_dtype=str(v.dtype),preparation_only=True,no_NCU_results_claimed=True)
print('PREPARATION_PASS',json.dumps(info),flush=True)
path=Path(os.environ['TORCH_EXTENSIONS_DIR'])/'GPU_PREPARATION_INFO.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(info,indent=2)+'\n')
