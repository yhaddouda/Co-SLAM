"""Diagnostic-only tensor/RNG evidence, not reconstruction quality scoring."""
import hashlib
import json
import random
import threading
import torch

class Probe:
    def __init__(self):self.calls={};self.lock=threading.Lock()
    def install(self,cls,context):
        original=cls.forward;probe=self
        def forward(model,*args,**kwargs):
            result=original(model,*args,**kwargs)
            key=context(model)
            if key is None:return result
            obs=hashlib.sha256()
            for a in args[2:4]:obs.update(a.detach().cpu().contiguous().numpy().tobytes())
            loss={k:v.detach().cpu().clone() for k,v in result.items() if k.endswith('loss')} if isinstance(result,dict) else {'total':result.detach().cpu().clone()}
            with probe.lock:
                if key in probe.calls:raise RuntimeError('Repeated diagnostic call identity: '+key)
                probe.calls[key]={'observations_sha256':obs.hexdigest(),'losses':loss,'ray_count':args[0].shape[0]}
            return result
        cls.forward=forward
    def save(self,path,model,poses,relative):
        torch.save({'calls':self.calls,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'poses':poses,'relative':relative,'python_rng':repr(random.getstate()),'torch_cpu_rng':torch.get_rng_state(),'torch_cuda_rng':torch.cuda.get_rng_state()},path)

def compare(a,b):
    """Predeclared conservative tolerance; never loosened after seeing gains."""
    a=torch.load(a,map_location='cpu');b=torch.load(b,map_location='cpu')
    details={'atol':1e-5,'rtol':1e-4,'calls':len(a['calls']),'checks':[]}
    assert set(a['calls'])==set(b['calls']),'Different iteration/call identity'
    for key in a['calls']:
        x=a['calls'][key];y=b['calls'][key]
        assert x['observations_sha256']==y['observations_sha256'] and x['ray_count']==y['ray_count'],'Ray observations differ: '+key
        assert x['losses'].keys()==y['losses'].keys()
        for k in x['losses']:assert torch.allclose(x['losses'][k],y['losses'][k],atol=1e-5,rtol=1e-4),'Loss difference: '+key+'/'+k
    for category in ['model','poses','relative']:
        x=a[category];y=b[category]
        if isinstance(x,dict):
            assert x.keys()==y.keys()
            for k in x:assert torch.allclose(x[k],y[k],atol=1e-5,rtol=1e-4),category+' '+str(k)
        else:assert torch.allclose(x,y,atol=1e-5,rtol=1e-4),category
        details['checks'].append(category+' numerical equivalence')
    assert a['python_rng']==b['python_rng'],'Python RNG differs'
    assert torch.equal(a['torch_cpu_rng'],b['torch_cpu_rng']),'CPU Torch RNG differs'
    assert torch.equal(a['torch_cuda_rng'],b['torch_cuda_rng']),'CUDA Torch RNG differs'
    details['checks']+=['identical observation identities/counts','all diagnostic scalar losses within fixed tolerance','Python/CPU/CUDA RNG states identical']
    return details
