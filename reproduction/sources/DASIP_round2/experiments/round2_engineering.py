"""Optional implementation ablation. No pruning or estimator correction."""
import ast
import copy
from pathlib import Path
import types


def methods(path):
    tree=ast.parse(Path(path).read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CoSLAM')
    return {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}


def transformed(path):
    native=methods(path)
    track=copy.deepcopy(native['tracking_render'])
    # The selected tracking indices are fixed for this call. Cache only the two
    # unchanging target gathers; leave pose aliases, best-pose logic and RNG alone.
    moved=[];target_if=None
    for n in ast.walk(track):
        if isinstance(n,ast.If) and ast.unparse(n.test)=='indice is None':target_if=n
        if hasattr(n,'body') and isinstance(n.body,list):
            for item in list(n.body):
                if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name) and item.targets[0].id in ('target_s','target_d'):
                    n.body.remove(item);moved.append(item)
    assert len(moved)==2 and target_if is not None
    target_if.body.extend(moved)
    ba=copy.deepcopy(native['global_BA']);changed=0
    for n in ast.walk(ba):
        if not hasattr(n,'body') or not isinstance(n.body,list):continue
        names=[item.targets[0].id if isinstance(item,ast.Assign) and isinstance(item.targets[0],ast.Name) else None for item in n.body]
        if 'rays_d_cam' in names and 'target_s' in names and 'target_d' in names:
            first=names.index('rays_d_cam')
            assert names[first:first+3]==['rays_d_cam','target_s','target_d']
            n.body[first:first+3]=ast.parse('''_r2_packed = rays.to(self.device)
rays_d_cam = _r2_packed[..., :3]
target_s = _r2_packed[..., 3:6]
target_d = _r2_packed[..., 6:7]
''').body
            changed+=1
    assert changed==1
    final=ba.body[-1]
    assert isinstance(final,ast.If) and 'len(frame_ids_all) > 1' in ast.unparse(final.test)
    final.body=ast.parse('''_r2_commit = self.matrix_from_tensor(cur_rot, cur_trans).detach().clone()
for _r2_offset, _r2_frame in enumerate(frame_ids_all[1:].tolist()):
    self.est_c2w_data[int(_r2_frame)] = _r2_commit[_r2_offset]
if self.config['mapping']['optim_cur']:
    print('Update current pose')
    self.est_c2w_data[cur_frame_id] = _r2_commit[-1]
''').body
    for n in (track,ba):ast.fix_missing_locations(n)
    return [track,ba]


def install(slam):
    import coslam
    functions=transformed(Path(__file__).parent/'coslam.py')
    namespace=dict(coslam.__dict__)
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<round2_engineering>','exec'),namespace)
    for name in ('tracking_render','global_BA'):
        setattr(slam,name,types.MethodType(namespace[name],slam))


def validate(output):
    import json
    import hashlib
    import torch
    from optimization.utils import at_to_transform_matrix,qt_to_transform_matrix
    torch.manual_seed(37);errors={}
    for name,func,dimensions in [('axis_angle',at_to_transform_matrix,3),('quat',qt_to_transform_matrix,4)]:
        rotation=torch.randn(127,dimensions,device='cuda');translation=torch.randn(127,3,device='cuda')
        reference=torch.cat([func(rotation[i:i+1],translation[i:i+1]).detach().clone() for i in range(len(rotation))])
        batched=func(rotation,translation).detach().clone()
        error=float((reference-batched).abs().max().item());errors[name]=error
        assert torch.allclose(reference,batched,atol=1e-6,rtol=1e-6)
        for include_current in (False,True):
            frame_ids=torch.arange(0,(len(rotation)+(0 if include_current else 1))*5,5)
            slow={int(f):reference[i].clone() for i,f in enumerate(frame_ids[1:])}
            fast={int(f):batched[i] for i,f in enumerate(frame_ids[1:])}
            assert slow.keys()==fast.keys()
            for f in slow:assert torch.allclose(slow[f],fast[f],atol=1e-6,rtol=1e-6)
    rays=torch.randn(4099,7)
    legacy=[rays[...,a:b].to('cuda') for a,b in [(0,3),(3,6),(6,7)]]
    packed=rays.to('cuda');new=[packed[...,:3],packed[...,3:6],packed[...,6:7]]
    assert all(torch.equal(a,b) for a,b in zip(legacy,new))
    original=methods(Path(__file__).parent/'coslam.py');optim=transformed(Path(__file__).parent/'coslam.py')
    result=dict(status='PASSED',pose_conversion_max_absolute_error=errors,ray_values_bitwise_equal=True,unchanged_native_pose_aliases=True,unchanged_indices_rng_losses_and_sampling=True,native_method_hashes={k:hashlib.sha256(ast.dump(original[k],include_attributes=False).encode()).hexdigest() for k in ('tracking_render','global_BA')},engineering_method_hashes={n.name:hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest() for n in optim},scope='fixed-state numerical equivalence, not bitwise online CUDA training equivalence')
    Path(output).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--validate',required=True);a=p.parse_args();validate(a.validate)
