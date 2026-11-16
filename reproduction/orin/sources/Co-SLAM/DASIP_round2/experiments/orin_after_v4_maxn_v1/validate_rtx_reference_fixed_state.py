"""Bounded identical-input CUDA numerical case; not an online-trajectory test."""
import argparse
import json
import os
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('DASIP_POST_V4_EXCLUSIVE')=='1'
    os.chdir(a.runtime);sys.path.insert(0,str(a.runtime))
    import torch
    import round2_engineering
    from optimization.utils import at_to_transform_matrix,qt_to_transform_matrix
    # Preserve exactly the supplied 1e-6 validation, including exact grouped rays.
    inherited=a.output.with_name(a.output.stem+'_SUPPLIED.json');round2_engineering.validate(inherited)
    torch.manual_seed(37);checks=[]
    for name,func,dim in [('axis_angle',at_to_transform_matrix,3),('quat',qt_to_transform_matrix,4)]:
        rotation=torch.randn(31,dim,device='cuda');translation=torch.randn(31,3,device='cuda');weights=torch.randn(31,4,4,device='cuda')
        r1=rotation.clone().requires_grad_();t1=translation.clone().requires_grad_();r2=rotation.clone().requires_grad_();t2=translation.clone().requires_grad_()
        slow=torch.cat([func(r1[i:i+1],t1[i:i+1]) for i in range(31)]);fast=func(r2,t2)
        assert torch.allclose(slow,fast,atol=1e-6,rtol=1e-6)
        loss1=((slow*weights)**2).sum();loss2=((fast*weights)**2).sum()
        loss1.backward();loss2.backward()
        gradient_error=max(float((r1.grad-r2.grad).abs().max()),float((t1.grad-t2.grad).abs().max()))
        assert torch.allclose(r1.grad,r2.grad,atol=1e-6,rtol=1e-6) and torch.allclose(t1.grad,t2.grad,atol=1e-6,rtol=1e-6)
        op1=torch.optim.Adam([r1,t1],lr=.001);op2=torch.optim.Adam([r2,t2],lr=.001);op1.step();op2.step()
        assert torch.allclose(r1,r2,atol=1e-6,rtol=1e-6) and torch.allclose(t1,t2,atol=1e-6,rtol=1e-6)
        poses=func(r2,t2).detach().clone();views=[poses[i] for i in range(31)];before=views[1].clone();views[0][:3,3].add_(.125)
        assert torch.equal(views[1],before),'Sibling pose storage overlap'
        parameter=torch.nn.Parameter(views[0][:3,3]);parameter.data.add_(.125);assert torch.equal(parameter,views[0][:3,3]),'Native translation-view alias lost'
        checks.append(dict(rotation=name,input_dtype=str(rotation.dtype),pose_max_abs_error=float((slow-fast).abs().max()),gradient_max_abs_error=gradient_error,one_Adam_update_pass=True,nonoverlapping_commit_views=True,native_translation_alias_preserved=True))
    result=dict(status='PASS',atol=1e-6,rtol=1e-6,checks=checks,supplied_validation=json.loads(inherited.read_text()),scope='Identical inputs, batched/per-pose transforms, gradients, one optimizer update and logical storage aliases; exact ray transfer values from supplied validator. Not full online CUDA identity or reconstruction quality.')
    a.output.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
