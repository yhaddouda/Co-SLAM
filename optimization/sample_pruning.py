"""Scientific sample and constraint pruning, separate from execution changes.

The pre-existing point policies select from the original jittered lattice.
Ray cohorts prune historical constraints with population-calibrated losses;
complementary tracking batches use native selected-batch losses and periodic
dense pose-selection checkpoints. Neither mechanism implies unbiased online
optimizer trajectories or unchanged reconstruction quality.
"""
import math
import random

import numpy as np
import torch

from optimization.ba_experiments import baseline_depth_samples
from model.utils import compute_loss, get_masks, get_sdf_loss, masked_mse_loss, mse2psnr

POLICIES = ('dense', 'tail', 'ht', 'adaptive', 'refine', 'refine_dense', 'support', 'support_tv', 'ray_cohort')
STAGES = ('Tracking', 'Bundle_Adjustment')


def validate_pruning(config):
    cfg = config.get('sample_pruning', {})
    # Reject a removed, unvalidated output transform instead of silently
    # accepting a nonzero setting from an older experimental configuration.
    if cfg.get('tracking_pose_average', 0) != 0:
        raise ValueError('tracking_pose_average is not supported by this implementation')
    selection = cfg.get('tracking_pose_selection', 'native')
    if selection not in ('native', 'dense_anchors'):
        raise ValueError('Unknown sample_pruning.tracking_pose_selection')
    if selection == 'dense_anchors':
        every = cfg.get('tracking_anchor_every', 0)
        if (isinstance(every, bool) or not math.isfinite(float(every))
                or int(every) != every or every <= 0):
            raise ValueError('Dense-anchor pose selection requires a positive tracking_anchor_every')
        if not (cfg.get('policy') == 'dense' or (cfg.get('policy') == 'ray_cohort'
                and cfg.get('cohort_tracking', 'support') == 'ray_cycle')):
            raise ValueError('Dense-anchor pose selection requires ray_cycle or a dense-policy control')
    if not cfg.get('enabled', False):
        return
    if cfg.get('policy', 'tail') not in POLICIES:
        raise ValueError(f'Unknown sample_pruning.policy: {cfg.get("policy")}')
    if config['training'].get('n_importance', 0):
        raise ValueError('Sample pruning requires n_importance=0')
    if config.get('ba_bucket', {}).get('enabled', False):
        raise ValueError('Use mixed BA ray batches: disable ba_bucket for sample_pruning')
    if config.get('tracking_bucket', {}).get('enabled', False) or config.get('tracking_naive_pruning', {}).get('enabled', False):
        raise ValueError('Disable legacy tracking pruning for sample_pruning')
    if config.get('tracking_depth_spatial', {}).get('enabled', False) and config['tracking_depth_spatial'].get('mode') != 'balanced_dense':
        raise ValueError('Sample pruning must not use shallow-biased pixel selection')
    if cfg.get('policy') in ('support','support_tv'):
        every = cfg.get('anchor_every', 0)
        if isinstance(every, bool) or int(every) != every or every < 0:
            raise ValueError('sample_pruning.anchor_every must be a nonnegative integer')
    if cfg.get('policy') == 'support_tv':
        fraction=float(cfg.get('tv_fraction',.125))
        if not math.isfinite(fraction) or not 0<fraction<=1:
            raise ValueError('sample_pruning.tv_fraction must lie in (0,1]')
        every=cfg.get('tracking_anchor_every',3)
        if isinstance(every,bool) or int(every)!=every or every<0:
            raise ValueError('sample_pruning.tracking_anchor_every must be a nonnegative integer')
    if cfg.get('policy') == 'ray_cohort':
        fraction=float(cfg.get('tv_fraction',1.))
        if not math.isfinite(fraction) or not 0<fraction<=1:
            raise ValueError('sample_pruning.tv_fraction must lie in (0,1]')
        if config.get('ba_experiment',{}).get('variant','baseline') != 'baseline':
            raise ValueError('Cohort pruning requires the native BA path')
        if config.get('tracking_depth_spatial',{}).get('enabled',False):
            raise ValueError('Cohort pruning requires native tracking ray batches')
        if cfg.get('cohort_weighting','population') not in ('population','native'):
            raise ValueError('Unknown cohort weighting')
        if not isinstance(cfg.get('cohort_keep_empty_pose_blocks',False),bool):
            raise ValueError('cohort_keep_empty_pose_blocks must be boolean')
        if cfg.get('cohort_tracking','support') not in ('support','ray_cycle'):
            raise ValueError('sample_pruning.cohort_tracking must be support or ray_cycle')
        cycle=cfg.get('tracking_cycle_groups',4)
        if isinstance(cycle,bool) or int(cycle)!=cycle or cycle<2:
            raise ValueError('sample_pruning.tracking_cycle_groups must be an integer >= 2')
        every=cfg.get('tracking_anchor_every',0)
        if isinstance(every,bool) or int(every)!=every or every<0:
            raise ValueError('sample_pruning.tracking_anchor_every must be a nonnegative integer')
        for name,minimum,default in (('cohort_count',1,4),('recent_keyframes',0,4),
                                     ('cohort_transition_keyframes',0,0)):
            value=cfg.get(name,default)
            if isinstance(value,bool) or int(value)!=value or value<minimum:
                raise ValueError(f'sample_pruning.{name} must be an integer >= {minimum}')
        period=cfg.get('cohort_transition_period',2)
        if isinstance(period,bool) or period!=2:
            raise ValueError('sample_pruning.cohort_transition_period is fixed at two')
        if cfg.get('cohort_transition_keyframes',0) and (cfg.get('cohort_count',4)<4
                or cfg.get('cohort_count',4)%period):
            raise ValueError('A maturation transition requires an even cohort_count >= 4')
    for name in ('margin', 'residual_threshold'):
        value = float(cfg.get(name, .1 if name == 'margin' else .35))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f'sample_pruning.{name} must be finite and nonnegative')
    stride = cfg.get('free_stride', 2)
    if isinstance(stride, bool) or int(stride) != stride or stride < 1:
        raise ValueError('sample_pruning.free_stride must be a positive integer')
    if not set(cfg.get('stages', STAGES)).issubset(STAGES):
        raise ValueError('Unknown sample_pruning stage')


def pruning_policy(model):
    cfg = model.config.get('sample_pruning', {})
    if not cfg.get('enabled', False) or getattr(model, '_pruning_stage', '') not in cfg.get('stages', STAGES):
        return None
    policy = cfg.get('policy', 'tail')
    stage = getattr(model, '_pruning_stage', '')
    cohort_support = (policy == 'ray_cohort' and stage == 'Tracking'
                      and cfg.get('cohort_tracking','support') == 'support')
    # Cyclic complements have no dense anchors by default. A positive anchor
    # interval explicitly requests first/last and periodic full-ray updates.
    cohort_cycle_anchor = (policy == 'ray_cohort' and stage == 'Tracking'
        and cfg.get('cohort_tracking','support') == 'ray_cycle'
        and int(cfg.get('tracking_anchor_every',0)) > 0)
    if policy in ('support','support_tv') or cohort_support or cohort_cycle_anchor:
        iteration = getattr(model, '_pruning_iteration', 0)
        total = model.config['tracking']['iter'] if stage == 'Tracking' else model.config['mapping']['iters']
        if cohort_support or cohort_cycle_anchor:
            every = int(cfg.get('tracking_anchor_every',0))
        else:
            every = int(cfg.get('tracking_anchor_every',3) if policy=='support_tv' and stage=='Tracking' else cfg.get('anchor_every', 0))
        if iteration == 0:
            model._support_selection = None
        if iteration == 0 or iteration == total-1 or (every > 0 and iteration % every == 0):
            model._pruning_iteration = iteration+1
            return None
    return None if policy == 'dense' else policy


def lattice_selection(z, depth, truncation, margin, stride=1, seed=0):
    """Keep surface support and one uniform representative per free-space block.

    A private CPU generator leaves the model's perturbation RNG untouched. For
    each ray the phase is uniform in [0,stride); each eligible point has inclusion
    probability 1/stride, including partial blocks. Invalid-depth rays stay dense.
    """
    valid = torch.isfinite(depth) & (depth > 0)
    keep = ~valid | (z <= depth + truncation + margin)
    probability = torch.ones_like(z)
    if stride > 1:
        eligible = valid & (z < depth - truncation - margin)
        # Dense first two points preserve the no-crossing renderer's near support.
        eligible[:, :2] = False
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        phase = torch.randint(stride, (z.shape[0], 1), generator=gen).to(z.device)
        index = torch.arange(z.shape[1], device=z.device)[None, :]
        selected = index.remainder(stride) == phase
        keep = keep & (~eligible | selected)
        probability = torch.where(eligible, 1. / stride, probability)
    keep[:, :2] = True
    return keep, probability


def refine_surface_lattice(z, depth, truncation, extra=4):
    """Split endpoint quadrature mass to insert midpoints near sensor depth.

    Each chosen surface interval changes endpoint masses by -1/4 each and adds
    a midpoint of mass 1/2. Zeroth and first depth moments are preserved. Missing
    or insufficient-depth support produces zero-mass slots, never queried.
    """
    count=min(int(extra), z.shape[1]-1)
    middle=.5*(z[:,1:]+z[:,:-1])
    eligible=((z[:,1:]-depth).abs()<=truncation) & ((z[:,:-1]-depth).abs()<=truncation) & (depth>0)
    score=torch.where(eligible,(middle-depth).abs(),torch.full_like(middle,float('inf')))
    chosen=score.topk(count,dim=1,largest=False).indices
    valid=eligible.gather(1,chosen)
    endpoint_mass=torch.ones_like(z)
    decrease=-.25*valid.to(z.dtype)
    endpoint_mass=endpoint_mass.scatter_add(1,chosen,decrease).scatter_add(1,chosen+1,decrease)
    midpoint=middle.gather(1,chosen)
    mass=torch.cat([endpoint_mass,.5*valid.to(z.dtype)],dim=1)
    values,order=torch.cat([z,midpoint],dim=1).sort(dim=1)
    return values,mass.gather(1,order)


def weighted_geometry(z, depth, sdf, keep, probability, truncation, reference_z=None):
    """Retain dense mask balancing/denominator; HT-correct sampled free space."""
    front, surface, fs_weight, sdf_weight = get_masks(z, depth, truncation)
    inverse = keep.to(sdf.dtype) / probability
    if reference_z is None:
        fs = (((sdf - 1.) * front).square() * inverse).mean() * fs_weight
        sd = (((z + sdf * truncation - depth) * surface).square() * inverse).mean() * sdf_weight
    else:
        _,_,fs_weight,sdf_weight=get_masks(reference_z,depth,truncation)
        fs=(((sdf-1.)*front).square()*inverse).sum()/reference_z.numel()*fs_weight
        sd=(((z+sdf*truncation-depth)*surface).square()*inverse).sum()/reference_z.numel()*sdf_weight
    return fs, sd


def render_weights(model, raw, z, keep, probability):
    # Find successive retained samples even across thinned free-space gaps.
    index = torch.arange(z.shape[1], device=z.device).expand_as(z)
    previous = torch.where(keep, index, 0).cummax(dim=1).values
    previous = torch.cat([previous[:, :1], previous[:, :-1]], 1)
    sdf = raw[..., 3]
    crossings = keep & (sdf * sdf.gather(1, previous) < 0)
    first_right = crossings.long().argmax(dim=1, keepdim=True)
    first_left = previous.gather(1, first_right)
    first_z = z.gather(1, first_left)
    truncation = model.config['training']['trunc'] * model.config['data']['sc_factor']
    support = z < first_z + truncation
    w = torch.sigmoid(sdf / model.config['training']['trunc']) * torch.sigmoid(-sdf / model.config['training']['trunc'])
    w = w * support * keep / probability
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
    return w, crossings, support


def forward_pruned(model, rays_o, rays_d, target_rgb, target_d):
    cfg = model.config['sample_pruning']
    policy = cfg['policy']
    if policy == 'ray_cohort':
        if model._pruning_stage=='Tracking':
            if cfg.get('cohort_tracking','support')=='support':
                return forward_supported(model,rays_o,rays_d,target_rgb,target_d)
            return forward_ray_cycle(model,rays_o,rays_d,target_rgb,target_d)
        return forward_cohort_ba(model,rays_o,rays_d,target_rgb,target_d)
    if policy in ('support','support_tv'):
        return forward_supported(model, rays_o, rays_d, target_rgb, target_d)
    z = baseline_depth_samples(model.config, target_d)
    dense_reference = z
    tau = model.config['training']['trunc'] * model.config['data']['sc_factor']
    if policy in ('refine', 'refine_dense'):
        z, mass = refine_surface_lattice(z,target_d,tau,extra=4)
    else:
        mass = torch.ones_like(z)
    stride = int(cfg.get('free_stride', 2)) if policy in ('ht', 'adaptive') else 1
    frame = getattr(model, '_pruning_frame', 0)
    iteration = getattr(model, '_pruning_iteration', 0)
    # Fixed subset across tracking iterations; independent subsets across BA updates.
    seed = int(cfg.get('seed', 0)) * 1000003 + frame * 1009
    if model._pruning_stage == 'Bundle_Adjustment':
        seed += iteration * 17 + 5
    keep, probability = lattice_selection(z, target_d, tau, float(cfg.get('margin', .1)), stride, seed)
    if policy == 'refine_dense':
        keep = torch.ones_like(keep)
    keep = keep & (mass > 0)
    # For refinement this becomes the inverse quadrature mass, rather than an
    # inclusion probability. Quarter/half masses preserve the reference measure.
    probability = probability / mass.clamp_min(1e-8)
    points = (rays_o[:, None, :] + rays_d[:, None, :] * z[..., None]).reshape(-1, 3)
    indices = torch.nonzero(keep.flatten(), as_tuple=False).flatten()
    values = model.run_network(points[indices])
    raw_flat = values.new_zeros((points.shape[0], 4)).index_copy(0, indices, values)
    raw = raw_flat.reshape(*z.shape, 4)
    with torch.no_grad():
        w, crossing, support = render_weights(model, raw, z, keep, probability)
        # Always recover no-crossing rays and a surface whose support reaches the
        # pruned tail. These are empirical guards, not an SDF error-bound proof.
        missing_tail = (~keep) & (z > target_d + tau + float(cfg.get('margin', .1)))
        recover = ~crossing.any(dim=1) | (support & missing_tail).any(dim=1)
        if policy == 'adaptive':
            nearest = (z - target_d).abs().argmin(dim=1, keepdim=True)
            mismatch = raw[..., 3].gather(1, nearest).abs().squeeze(1)
            estimate = (w * z).sum(dim=1)
            recover |= (mismatch > float(cfg.get('residual_threshold', .35)))
            recover |= (estimate - target_d.squeeze(-1)).abs() > 2. * tau
        extra = recover[:, None] & ~keep & (mass > 0)
        extra_indices = torch.nonzero(extra.flatten(), as_tuple=False).flatten()
    if extra_indices.numel():
        extra_raw = model.run_network(points[extra_indices])
        raw_flat = raw_flat.index_copy(0, extra_indices, extra_raw)
        raw = raw_flat.reshape(*z.shape, 4)
        keep = keep | extra
        # Recovery is adaptive and changes inclusion. This branch uses the full
        # dense loss for recovered rays; unbiased HT applies to fixed thinning.
        probability = torch.where(recover[:, None], 1./mass.clamp_min(1e-8), probability)
    w, _, _ = render_weights(model, raw, z, keep, probability)
    rgb = (w[..., None] * torch.sigmoid(raw[..., :3])).sum(dim=1)
    depth = (w * z).sum(dim=1)
    if model.config['training']['white_bkgd']:
        rgb = rgb + (1 - w.sum(dim=1, keepdim=True))
    valid = (target_d.squeeze(-1) > 0) & (target_d.squeeze(-1) < model.config['cam']['depth_trunc'])
    rgb_weight = torch.where(valid[:, None], 1., model.config['training']['rgb_missing']).to(target_rgb)
    rgb_loss = compute_loss(rgb * rgb_weight, target_rgb * rgb_weight)
    depth_loss = masked_mse_loss(depth, target_d.squeeze(-1), valid)
    fs_loss, sdf_loss = weighted_geometry(z, target_d, raw[..., 3], keep, probability, tau,
        reference_z=dense_reference if policy in ('refine', 'refine_dense') else None)
    model._pruning_iteration = iteration + 1
    if not hasattr(model, 'pruning_diagnostics'):
        model.pruning_diagnostics = []
    model.pruning_diagnostics.append(dict(stage=model._pruning_stage, frame=frame,
        dense_samples=dense_reference.numel(), queried_samples=indices.numel()+extra_indices.numel(),
        recovery_samples=extra_indices.numel(), recovered_rays=int(recover.sum()), rays=z.shape[0]))
    return dict(rgb=rgb, depth=depth, rgb_loss=rgb_loss, depth_loss=depth_loss,
                fs_loss=fs_loss, sdf_loss=sdf_loss, psnr=mse2psnr(rgb_loss))


def feature_uniform_pixels(batch, iH, iW, samples, fraction=.25):
    """EC-SLAM-inspired corners plus grid-uniform rays (not its full system)."""
    import cv2
    image = batch['rgb'].squeeze(0).cpu().numpy()
    image = image[iH:image.shape[0]-iH if iH else None, iW:image.shape[1]-iW if iW else None]
    H, W = image.shape[:2]
    gray = cv2.cvtColor(np.uint8(np.clip(image, 0, 1)*255), cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max(1, int(samples*fraction)),
                                    qualityLevel=.01, minDistance=5, blockSize=3)
    selected = set()
    if corners is not None and fraction > 0:
        for x, y in corners[:, 0]:
            selected.add(int(x) * H + int(y))
    # Stratified grid coverage. Randomized cell order avoids a top-left bias if
    # the number of remaining samples is not a multiple of the cell count.
    cells = [(a, b) for a in range(8) for b in range(8)]
    while len(selected) < samples:
        random.shuffle(cells)
        for a, b in cells:
            if len(selected) == samples:
                break
            y = random.randrange(a*H//8, (a+1)*H//8)
            x = random.randrange(b*W//8, (b+1)*W//8)
            selected.add(x*H+y)
    return torch.tensor(sorted(selected), dtype=torch.long)


def forward_supported(model, rays_o, rays_d, target_rgb, target_d):
    """Prune complete jitter intervals; restore full support at anchor updates.

    All reference geometry-supervised samples are kept. Intermediate updates
    truncate the tail without a field-dependent recovery pass; dense anchors
    restore full rendering support. Retained points keep their original jitter.
    """
    cfg=model.config['sample_pruning']
    z,lower=baseline_depth_samples(model.config,target_d,return_bounds=True)
    tau=model.config['training']['trunc']*model.config['data']['sc_factor']
    margin=float(cfg.get('margin',.15))
    key=(getattr(model,'_pruning_frame',0),target_d.data_ptr(),z.shape,tau,margin)
    cached=getattr(model,'_support_selection',None) if model._pruning_stage=='Tracking' else None
    if cached is not None and cached[0]==key:
        keep,indices=cached[1:]
    else:
        valid=torch.isfinite(target_d) & (target_d>0)
        keep=~valid | (lower<=target_d+tau+margin)
        keep[:,:2]=True
        indices=torch.nonzero(keep.flatten(),as_tuple=False).flatten()
        if model._pruning_stage=='Tracking':
            model._support_selection=(key,keep,indices)
    points=rays_o[:,None,:]+rays_d[:,None,:]*z[...,None]
    values=model.run_network(points.reshape(-1,3)[indices])
    raw=values.new_zeros((z.numel(),4)).index_copy(0,indices,values).reshape(*z.shape,4)
    rgb,_,_,_,depth,_=model.raw2outputs(raw,z,model.config['training']['white_bkgd'],sample_mask=keep)
    fs_loss,sdf_loss=get_sdf_loss(z,target_d,raw[...,3],tau,'l2',grad=None)
    good=(target_d.squeeze(-1)>0) & (target_d.squeeze(-1)<model.config['cam']['depth_trunc'])
    rgb_weight=torch.where(good[:,None],1.,model.config['training']['rgb_missing']).to(target_rgb)
    rgb_loss=compute_loss(rgb*rgb_weight,target_rgb*rgb_weight)
    depth_loss=masked_mse_loss(depth,target_d.squeeze(-1),good)
    iteration=getattr(model,'_pruning_iteration',0)
    model._pruning_iteration=iteration+1
    if not hasattr(model,'pruning_diagnostics'):
        model.pruning_diagnostics=[]
    model.pruning_diagnostics.append(dict(stage=model._pruning_stage,frame=getattr(model,'_pruning_frame',0),
        iteration=iteration,dense_samples=z.numel(),queried_samples=indices.numel(),
        recovery_samples=0,recovered_rays=0,rays=z.shape[0],selected_rays=z.shape[0],
        weight_sum=float(z.shape[0]),cohort_old_rays=0,anchor_rays=z.shape[0],
        cohort_transition_rays=0,cohort_mature_rays=0))
    return dict(rgb=rgb,depth=depth,rgb_loss=rgb_loss,depth_loss=depth_loss,
        fs_loss=fs_loss,sdf_loss=sdf_loss,psnr=mse2psnr(rgb_loss))


def sampled_tv_loss(slam,sample_points,voxel_size,margin,edges_per_axis):
    """Axis-stratified Monte Carlo estimator of the ORIGINAL grid TV energy.

    Draw native global offset/jitter unchanged, then sample finite-difference
    edges with replacement from each axis using a private generator. Scale by
    edge population / sample count, retaining the native s**3 denominator.
    """
    volume=slam.bounding_box[:,1]-slam.bounding_box[:,0]
    grid_size=(sample_points-1)*voxel_size
    offset_max=volume-grid_size-2*margin
    offset=torch.rand(3).to(volume)*offset_max+margin
    jitter=torch.rand((1,1,1,3)).to(volume).reshape(3)
    n=sample_points-1;population=(n-1)*n*n
    cfg=slam.config['sample_pruning'];model=slam.model
    frame=getattr(model,'_pruning_frame',0);iteration=getattr(model,'_pruning_iteration',0)-1
    seed=int(cfg.get('seed',0))*1000003+frame*1009+iteration*17+53
    gen=np.random.default_rng(seed)
    flat=gen.integers(population,size=(3,edges_per_axis),dtype=np.int64)
    a=flat//(n*n);b=(flat//n)%n;c=flat%n
    starts=np.stack([np.stack([a[0],b[0],c[0]],-1),np.stack([b[1],a[1],c[1]],-1),np.stack([b[2],c[2],a[2]],-1)])
    ends=starts+np.eye(3,dtype=np.int64)[:,None,:]
    integer=torch.from_numpy(np.stack([starts,ends],axis=2)).to(volume)
    points=(integer+jitter)*voxel_size+slam.bounding_box[:,0]+offset
    normalized=(points-slam.bounding_box[:,0])/volume
    features=model.query_sdf(normalized,embed=True)
    energy=(features[:,:,1,:]-features[:,:,0,:]).square().sum(dtype=torch.float32)
    loss=energy*(population/edges_per_axis)/(sample_points**3)
    if not hasattr(model,'smoothness_diagnostics'):model.smoothness_diagnostics=[]
    model.smoothness_diagnostics.append(dict(frame=frame,iteration=iteration,dense_samples=n**3,
        queried_samples=6*edges_per_axis,sampled_edges=3*edges_per_axis))
    return loss


def forward_ray_cycle(model,rays_o,rays_d,target_rgb,target_d):
    """Evaluate spatial/depth complementary ray groups with native batch losses.

    Every selected ray retains its entire native jittered lattice. Depth
    normalization and geometry balancing use the selected batch, so these are
    conditional ratio estimates, not unbiased full-batch gradients. Dense
    anchors are dispatched by pruning_policy before this function is called.
    """
    from optimization.ray_selection import cyclic_ray_subsets
    cfg=model.config['sample_pruning'];stage=model._pruning_stage
    frame=getattr(model,'_pruning_frame',0);iteration=getattr(model,'_pruning_iteration',0)
    population=len(target_d)
    seed=int(cfg.get('seed',0))*1000003+frame*1009+71
    count=int(cfg.get('tracking_cycle_groups',4))
    cycle_key=(frame,population,count,seed)
    stored=getattr(model,'_ray_cycle_selection',None)
    if stored is None or stored[0]!=cycle_key:
        metadata=getattr(model,'_ray_sampling_metadata',None)
        if isinstance(metadata,torch.Tensor):metadata=metadata.detach().numpy()
        subsets=cyclic_ray_subsets(population,seed,metadata=metadata,groups=count)
        stored=(cycle_key,tuple(torch.from_numpy(ids).to(target_d.device) for ids in subsets))
        model._ray_cycle_selection=stored
    indices=stored[1][iteration % len(stored[1])]
    tau=model.config['training']['trunc']*model.config['data']['sc_factor']
    depth_target=target_d[indices];rgb_target=target_rgb[indices]
    z=baseline_depth_samples(model.config,depth_target)
    good=(depth_target.squeeze(-1)>0)&(depth_target.squeeze(-1)<model.config['cam']['depth_trunc'])
    points=rays_o[indices,None,:]+rays_d[indices,None,:]*z[...,None]
    raw=model.run_network(points.reshape(-1,3)).reshape(len(indices),z.shape[1],4)
    rgb,_,_,_,depth,_=model.raw2outputs(raw,z,model.config['training']['white_bkgd'])
    color_weight=torch.where(good[:,None],1.,model.config['training']['rgb_missing']).to(target_rgb)
    sdf=raw[...,3]
    rgb_loss=compute_loss(rgb*color_weight,rgb_target*color_weight)
    depth_loss=masked_mse_loss(depth,depth_target.squeeze(-1),good)
    fs_loss,sdf_loss=get_sdf_loss(z,depth_target,sdf,tau,'l2',grad=None)
    model._pruning_iteration=iteration+1
    if not hasattr(model,'pruning_diagnostics'):model.pruning_diagnostics=[]
    model.pruning_diagnostics.append(dict(stage=stage,frame=frame,iteration=iteration,
        dense_samples=population*z.shape[1],queried_samples=z.numel(),recovery_samples=0,
        recovered_rays=0,rays=population,selected_rays=len(indices),weight_sum=float(population),
        cohort_old_rays=0,anchor_rays=len(indices),
        cohort_transition_rays=0,cohort_mature_rays=0))
    return dict(rgb=rgb,depth=depth,rgb_loss=rgb_loss,depth_loss=depth_loss,
                fs_loss=fs_loss,sdf_loss=sdf_loss,psnr=mse2psnr(rgb_loss))


def forward_cohort_ba(model,rays_o,rays_d,target_rgb,target_d):
    """Keep current/recent constraints and rotating historical ray groups.

    Population weighting compensates historical inclusion at fixed field/poses,
    reference batch, group membership and anchor classification. It does not
    imply unbiased online gradients. Native weighting uses the original loss
    on the retained batch: a deliberately local-focused objective with recurring
    historical replay, not an unbiased estimate of the dense BA objective.
    Both preserve the complete native jitter draw for paired retained points.
    """
    group_ids=model._ray_group_ids.detach().cpu().numpy()
    active=np.asarray(model._cohort_active_ids,dtype=np.int64)
    probabilities=np.asarray(model._cohort_probabilities)
    chosen=(group_ids<0)|np.isin(group_ids,active)
    selected=np.flatnonzero(chosen)
    p=np.ones(len(group_ids),dtype=np.float32)
    historical=group_ids>=0
    p[historical]=probabilities[group_ids[historical]]
    weighting=model.config['sample_pruning'].get('cohort_weighting','population')
    mass=(1./p[selected]).astype(np.float32) if weighting=='population' else np.ones(len(selected),dtype=np.float32)
    indices=torch.from_numpy(selected).to(target_d.device)
    weights=torch.from_numpy(mass).to(target_d)
    z_full=baseline_depth_samples(model.config,target_d)
    tau=model.config['training']['trunc']*model.config['data']['sc_factor']
    front,surface,fs_weight,sdf_weight=get_masks(z_full,target_d,tau)
    valid=(target_d.squeeze(-1)>0)&(target_d.squeeze(-1)<model.config['cam']['depth_trunc'])
    z=z_full[indices];depth_target=target_d[indices];rgb_target=target_rgb[indices]
    points=rays_o[indices,None,:]+rays_d[indices,None,:]*z[...,None]
    raw=model.run_network(points.reshape(-1,3)).reshape(len(indices),z.shape[1],4)
    rgb,_,_,_,depth,_=model.raw2outputs(raw,z,model.config['training']['white_bkgd'])
    good=valid[indices]
    color_weight=torch.where(good[:,None],1.,model.config['training']['rgb_missing']).to(target_rgb)
    sdf=raw[...,3]
    if weighting=='native':
        rgb_loss=compute_loss(rgb*color_weight,rgb_target*color_weight)
        depth_loss=masked_mse_loss(depth,depth_target.squeeze(-1),good)
        fs_loss,sdf_loss=get_sdf_loss(z,depth_target,sdf,tau,'l2',grad=None)
    else:
        rgb_loss=(((rgb-rgb_target)*color_weight).square()*weights[:,None]).sum()/(3*len(target_d))
        depth_error=torch.where(good,depth-depth_target.squeeze(-1),0.)
        depth_loss=(depth_error.square()*weights).sum()/valid.sum().clamp_min(1)
        fs_loss=(((sdf-1.)*front[indices]).square()*weights[:,None]).sum()/z_full.numel()*fs_weight
        sdf_loss=(((z+sdf*tau-depth_target)*surface[indices]).square()*weights[:,None]).sum()/z_full.numel()*sdf_weight
    iteration=getattr(model,'_pruning_iteration',0);model._pruning_iteration=iteration+1
    if not hasattr(model,'pruning_diagnostics'):model.pruning_diagnostics=[]
    transition_count=(int((p[selected]==.5).sum())
        if model.config['sample_pruning'].get('cohort_transition_keyframes',0)>0 else 0)
    old_count=int((p[selected]<1).sum())
    model.pruning_diagnostics.append(dict(stage=model._pruning_stage,
        frame=getattr(model,'_pruning_frame',0),iteration=iteration,
        dense_samples=z_full.numel(),queried_samples=z.numel(),recovery_samples=0,
        recovered_rays=0,rays=len(target_d),selected_rays=len(indices),weight_sum=float(mass.sum()),
        cohort_old_rays=old_count,anchor_rays=int((p[selected]==1).sum()),
        cohort_transition_rays=transition_count,cohort_mature_rays=old_count-transition_count))
    return dict(rgb=rgb,depth=depth,rgb_loss=rgb_loss,depth_loss=depth_loss,
        fs_loss=fs_loss,sdf_loss=sdf_loss,psnr=mse2psnr(rgb_loss))
