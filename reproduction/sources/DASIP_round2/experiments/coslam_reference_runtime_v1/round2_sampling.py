"""Optional along-ray support pruning and Morton ordering on the public model.

The public dense forward and geometry losses remain the reference. Rendering
is approximated on pruned updates; no opacity/convergence guarantee is claimed.
"""
from pathlib import Path
import torch


def depth_lattice(model, rays_o, target_d):
    cfg=model.config; n=target_d.shape[0]
    z=torch.linspace(-cfg['training']['range_d'],cfg['training']['range_d'],steps=cfg['training']['n_range_d']).to(target_d)
    z=z[None,:].repeat(n,1)+target_d
    z[target_d.squeeze()<=0]=torch.linspace(cfg['cam']['near'],cfg['cam']['far'],steps=cfg['training']['n_range_d']).to(target_d)
    if cfg['training']['n_samples_d']>0:
        uniform=torch.linspace(cfg['cam']['near'],cfg['cam']['far'],cfg['training']['n_samples_d'])[None,:].repeat(n,1).to(rays_o)
        z=torch.sort(torch.cat([uniform,z],-1),-1).values
    lower=z
    if cfg['training']['perturb']>0:
        mids=.5*(z[...,1:]+z[...,:-1]);upper=torch.cat([mids,z[...,-1:]],-1)
        lower=torch.cat([z[...,:1],mids],-1)
        z=lower+(upper-lower)*torch.rand(z.shape).to(rays_o)
    return z,lower


def supported_outputs(model, raw, z, keep):
    cfg=model.config;sdf=raw[...,3]
    weights=torch.sigmoid(sdf/cfg['training']['trunc'])*torch.sigmoid(-sdf/cfg['training']['trunc'])
    signs=sdf[:,1:]*sdf[:,:-1]
    crossings=(signs<0)&keep[:,1:]&keep[:,:-1]
    inds=torch.argmax(crossings.to(raw.dtype),axis=1)[...,None]
    first=torch.gather(z,1,inds)
    mask=torch.where(z<first+cfg['data']['sc_factor']*cfg['training']['trunc'],torch.ones_like(z),torch.zeros_like(z))
    weights=weights*mask*keep
    weights=weights/(torch.sum(weights,axis=-1,keepdims=True)+1e-8)
    rgb=torch.sum(weights[...,None]*torch.sigmoid(raw[...,:3]),-2)
    depth=torch.sum(weights*z,-1)
    variance=torch.sum(weights*torch.square(z-depth.unsqueeze(-1)),dim=-1)
    disp=1./torch.max(1e-10*torch.ones_like(depth),depth/torch.sum(weights,-1))
    acc=torch.sum(weights,-1)
    if cfg['training']['white_bkgd']:rgb=rgb+(1.-acc[...,None])
    return dict(rgb=rgb,depth=depth,disp_map=disp,acc_map=acc,depth_var=variance,z_vals=z,raw=raw)


def supported_render(model, rays_o, rays_d, target_d, margin):
    z,lower=depth_lattice(model,rays_o,target_d)
    # Interval membership depends on the fixed depth/ray batch, never predicted SDF.
    cached=getattr(model,'_r2_support',None) if model._r2_stage=='tracking' else None
    if cached is None:
        tau=model.config['training']['trunc']*model.config['data']['sc_factor']
        valid=torch.isfinite(target_d)&(target_d>0)
        keep=(~valid)|(lower<=target_d+tau+margin)
        keep[:,:2]=True
        indices=torch.nonzero(keep.flatten(),as_tuple=False).flatten()
        if model._r2_stage=='tracking':model._r2_support=(keep,indices)
    else:keep,indices=cached
    points=rays_o[...,None,:]+rays_d[...,None,:]*z[...,:,None]
    selected=model.run_network(points.reshape(-1,3)[indices])
    raw=selected.new_zeros((z.numel(),4)).index_copy(0,indices,selected).reshape(*z.shape,4)
    return supported_outputs(model,raw,z,keep)


def install(model, prune=False, morton=False, margin=.15, every=3):
    if model.config['training']['n_importance']!=0:raise ValueError('This support policy requires native n_importance=0')
    model._r2_stage=None;model._r2_iteration=0;model._r2_support=None
    model._r2_actual=0;model._r2_queries=[]
    native_run=model.run_network;native_render=model.render_rays;native_forward=model.forward
    key_function=None
    if morton:
        from torch.utils.cpp_extension import load
        # Build for the actual machine; never copy an RTX binary to the Jetson.
        major,minor=torch.cuda.get_device_capability()
        import os
        os.environ['TORCH_CUDA_ARCH_LIST']=str(major)+'.'+str(minor)
        extension=load(name='dasip_round2_morton_'+str(major)+str(minor),
            sources=[str(Path(__file__).parent/'model/round2_morton/morton_cuda.cu')],
            extra_cuda_cflags=['-O3'],verbose=False)
        key_function=extension.morton3d_keys

    def network(inputs):
        model._r2_actual+=inputs.numel()//3
        if not morton:return native_run(inputs)
        flat=inputs.reshape(-1,3)
        if not model.config['grid']['tcnn_encoding']:return native_run(inputs)
        normalized=(flat-model.bounding_box[:,0])/(model.bounding_box[:,1]-model.bounding_box[:,0])
        with torch.no_grad():perm=torch.argsort(key_function(normalized.float().contiguous(),128),stable=False).long()
        values=model.query_color_sdf(normalized[perm])
        output=torch.empty_like(values);output[perm]=values
        return output.reshape(*inputs.shape[:-1],values.shape[-1])

    def render(rays_o,rays_d,target_d=None):
        stage=model._r2_stage
        total=model.config['tracking']['iter'] if stage=='tracking' else model.config['mapping']['iters']
        i=model._r2_iteration
        full=i==0 or i==total-1 or i%every==0
        if not prune or stage not in ('tracking','bundle_adjustment') or full or target_d is None:
            return native_render(rays_o,rays_d,target_d)
        return supported_render(model,rays_o,rays_d,target_d,margin)

    def forward(rays_o,rays_d,target_rgb,target_d,global_step=0):
        before=model._r2_actual
        result=native_forward(rays_o,rays_d,target_rgb,target_d,global_step)
        if model._r2_stage in ('tracking','bundle_adjustment'):
            model._r2_queries.append(dict(frame_id=model._r2_frame,stage=model._r2_stage,
                iteration=model._r2_iteration,rays=len(rays_o),
                dense_queries=len(rays_o)*(model.config['training']['n_samples_d']+model.config['training']['n_range_d']),
                actual_queries=model._r2_actual-before))
        return result

    model.run_network=network;model.render_rays=render;model.forward=forward
