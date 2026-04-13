# package imports
import atexit
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn

# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v2
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss

#nvtx 
import nvtx

# Morton code imports
from .fast_morton import morton3d_keys_cuda
from .layer_timing import DeferredCudaTimer


@torch.no_grad()
def morton_permutation_from_points01(pts01: torch.Tensor, R: int = 128) -> torch.Tensor:
    # pts01: [N,3], float32, CUDA
    with torch.cuda.nvtx.range("morton_keys"):
        keys = morton3d_keys_cuda(pts01, R)          # int32 keys on GPU
    with torch.cuda.nvtx.range("argsort"):
        perm = torch.argsort(keys, stable=False)     # fast radix on int32
    return perm.to(torch.long)


class UniformSamplingStatsWriter:
    def __init__(self, enabled=False, output_csv='./profiling/uniform_samples_until_depth_stats.csv', flush_every_batches=16):
        self.enabled = bool(enabled)
        self.output_csv = Path(output_csv)
        self.flush_every_batches = max(1, int(flush_every_batches))
        self.current_frame_id = -1
        self.current_outer_stage = ''
        self.next_batch_id = 0
        self.pending = []
        self.header_written = False

        if not self.enabled:
            return

        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = self.output_csv.exists() and self.output_csv.stat().st_size > 0
        atexit.register(self.flush)

    def set_context(self, frame_id: int, outer_stage: str):
        if not self.enabled:
            return
        self.current_frame_id = int(frame_id)
        self.current_outer_stage = str(outer_stage)

    def record_batch(
        self,
        ray_counts_before_depth: torch.Tensor,
        valid_depth: torch.Tensor,
        selected_uniform_samples: int,
        requested_uniform_samples: int,
    ):
        if not self.enabled:
            return

        counts_cpu = ray_counts_before_depth.detach().to(dtype=torch.int32, device='cpu').tolist()
        valid_cpu = valid_depth.detach().to(dtype=torch.bool, device='cpu').tolist()
        per_ray_counts = [
            int(count) if is_valid else None
            for count, is_valid in zip(counts_cpu, valid_cpu)
        ]

        self.pending.append([
            self.current_frame_id,
            self.current_outer_stage,
            self.next_batch_id,
            int(selected_uniform_samples),
            int(requested_uniform_samples),
            int(sum(valid_cpu)),
            len(valid_cpu),
            json.dumps(per_ray_counts, separators=(',', ':')),
        ])
        self.next_batch_id += 1

        if len(self.pending) >= self.flush_every_batches:
            self.flush()

    def flush(self):
        if not self.enabled or not self.pending:
            return

        write_header = not self.output_csv.exists() or not self.header_written
        with open(self.output_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    'frame_id',
                    'outer_stage',
                    'batch_id',
                    'selected_uniform_samples',
                    'requested_uniform_samples',
                    'valid_ray_count',
                    'total_ray_count',
                    'ray_samples_before_depth',
                ])
                self.header_written = True
            writer.writerows(self.pending)
        self.pending.clear()


class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)
        timing_cfg = config.get('timing', {})
        mode = timing_cfg.get('mode', 'none')
        self.layer_timing = DeferredCudaTimer(
            enabled=(mode == 'scene_rep'),
            output_csv=timing_cfg.get('scene_output_csv', './profiling/pass1/office0/scene_rep_timing.csv'),
            warmup_frames=timing_cfg.get('warmup_frames', 0),
        )
        training_cfg = config.get('training', {})
        default_sampling_stats_csv = (
            Path(config['data']['output'])
            / config['data']['exp_name']
            / 'uniform_samples_until_depth_stats.csv'
        )
        self.uniform_sampling_stats = UniformSamplingStatsWriter(
            enabled=training_cfg.get('collect_uniform_samples_until_depth_stats', False),
            output_csv=training_cfg.get(
                'uniform_samples_until_depth_stats_csv',
                default_sampling_stats_csv,
            ),
        )
        if self.uniform_sampling_stats.enabled:
            print(f"[sampling-stats] Collecting cropped-sampling stats in {self.uniform_sampling_stats.output_csv}")
            if not training_cfg.get('uniform_samples_until_depth', False):
                print(
                    "[sampling-stats] uniform_samples_until_depth is disabled, "
                    "so no cropped-sampling stats will be recorded."
                )
        

    def set_timing_context(self, frame_id: int, outer_stage: str):
        self.layer_timing.set_context(frame_id, outer_stage)
        self.uniform_sampling_stats.set_context(frame_id, outer_stage)

    def flush_scene_timing(self):
        self.layer_timing.flush()
        self.uniform_sampling_stats.flush()

    def get_resolution(self):
        '''
        Get the resolution of the grid
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()
        if self.config['grid']['voxel_sdf'] > 10:
            self.resolution_sdf = self.config['grid']['voxel_sdf']
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        print('SDF resolution:', self.resolution_sdf)

    def get_encoding(self, config):
        '''
        Get the encoding of the scene representation
        '''
        # Coordinate encoding
        with torch.cuda.nvtx.range("coord_encoder"):
            self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'], hash=config['grid']['hash'])

        # Sparse parametric encoding (SDF)
        with torch.cuda.nvtx.range("parametric_encoder_sdf"):
            self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf, hash=config['grid']['hash'])

        # Sparse parametric encoding (Color)
        if not self.config['grid']['oneGrid']:
            print('Color resolution:', self.resolution_color)
            with torch.cuda.nvtx.range("parametric_encoder_color"):
                self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color, hash=config['grid']['hash'])

    def get_decoder(self, config):
        '''
        Get the decoder of the scene representation
        '''
        if not self.config['grid']['oneGrid']:
            with torch.cuda.nvtx.range("ColorSDFNeT"):
                self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            with torch.cuda.nvtx.range("ColorSDFNet_v2"):
                self.decoder = ColorSDFNet_v2(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        
        with torch.cuda.nvtx.range("Color_Net"):
            self.color_net = batchify(self.decoder.color_net, None)

        with torch.cuda.nvtx.range("SDF_Net"):
            self.sdf_net = batchify(self.decoder.sdf_net, None)

    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    def raw2outputs(self, raw, z_vals, white_bkgd=False):
        '''
        Perform volume rendering using weights computed from sdf.

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)
        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1)
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
        acc_map = torch.sum(weights, -1)

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        Get the SDF value of the query points
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        with torch.cuda.nvtx.range("reshape_tensor"):
            inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        with torch.cuda.nvtx.range("Hashgrid_encoding"):
            embedded = self.embed_fn(inputs_flat)

        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])
        
        # grid interpolation
        with torch.cuda.nvtx.range("Oneblob_encoding"):
            embedded_pos = self.embedpos_fn(inputs_flat)

        # sdf mlp
        with torch.cuda.nvtx.range("MLP"):
            out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))

        with torch.cuda.nvtx.range("tensor_split"):
            sdf, geo_feat = out[..., :1], out[..., 1:]

        with torch.cuda.nvtx.range("reshape_tensor"):
            sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    def query_color(self, query_points):
        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])
      
    def query_color_sdf(self, query_points):
        '''
        Query the color and sdf at query_points.

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        # should be output of sdf mlp
        with self.layer_timing.stage('TCNN_hashgrid_encoding'):
            with torch.cuda.nvtx.range("TCNN_hashgrid_encoding"):
                embed = self.embed_fn(inputs_flat)

        # one blob 
        with self.layer_timing.stage('TCNN_oneblob_encoding'):
            with torch.cuda.nvtx.range("TCNN_oneblob_encoding"):
                embe_pos = self.embedpos_fn(inputs_flat)

        if not self.config['grid']['oneGrid']:
            with self.layer_timing.stage('TCNN_fn_color'):
                with torch.cuda.nvtx.range("TCNN_fn_color"):
                    embed_color = self.embed_fn_color(inputs_flat)
            with self.layer_timing.stage('Decoder'):
                return self.decoder(embed, embe_pos, embed_color)
        
        with self.layer_timing.stage('Decoder'):
            with torch.cuda.nvtx.range("Decoder"):
                out = self.decoder(embed, embe_pos)
        return out

    def run_network(self, inputs):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        with self.layer_timing.stage('run_network'):
            with torch.cuda.nvtx.range("reshape_inputs"):
                inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])  # [N,3]

            use_tcnn = self.config['grid']['tcnn_encoding']
            do_morton = bool(self.config['grid'].get('morton_sort', True))  # enable/disable via config
            R = int(self.config['grid'].get('morton_R', 128))               # quantization per axis

            if use_tcnn:
                # Normalize to [0,1]^3 (Instant-NGP / TCNN convention). This is already
                # what the original code does; we keep the same map.
                bb0 = self.bounding_box[:, 0]  # [3]
                bb1 = self.bounding_box[:, 1]  # [3]
                inputs01 = (inputs_flat - bb0) / (bb1 - bb0)

                if do_morton:
                    # ---- Morton sort on CUDA (GPU) ----
                    # Build permutation that clusters spatial neighbors. Then call the
                    # decoder on the permuted array and scatter back to original order.
                    with self.layer_timing.stage('morton_permutation'):
                        with torch.cuda.nvtx.range("morton_permutation"):
                            perm = morton_permutation_from_points01(inputs01, R=R)  # [N] int32, CUDA
                    inputs01_sorted = inputs01[perm]                        # [N,3]
                    with self.layer_timing.stage('query_color_sdf'):
                        with torch.cuda.nvtx.range("query_color_sdf"):
                            outputs_sorted = batchify(self.query_color_sdf, None)(inputs01_sorted)

                    # Unpermute back to original order (scatter is a cheap inverse):
                    with self.layer_timing.stage('unpermute'):
                        with torch.cuda.nvtx.range("unpermute"):
                            outputs_flat = torch.empty_like(outputs_sorted)
                            outputs_flat[perm] = outputs_sorted
                else:
                    # Original behavior: just run on normalized inputs
                    with self.layer_timing.stage('query_color_sdf'):
                        with torch.cuda.nvtx.range("query_color_sdf"):
                            outputs_flat = batchify(self.query_color_sdf, None)(inputs01)
            else:
                # Non-TCNN path: keep original behavior (world coords, no sorting).
                with self.layer_timing.stage('query_color_sdf'):
                    with torch.cuda.nvtx.range("query_color_sdf"):
                        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)

            outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
            return outputs

    
    def render_surface_color(self, rays_o, normal):
        '''
        Render the surface color of the points.
        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        '''
        n_rays = rays_o.shape[0]
        trunc = self.config['training']['trunc']
        z_vals = torch.linspace(-trunc, trunc, steps=self.config['training']['n_range_d']).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline
        
        pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])
        return rgb
    
    def render_rays(self, rays_o, rays_d, target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]

        '''
        n_rays = rays_o.shape[0]

        # Sample depth
        with self.layer_timing.stage('depth_sampling'):
            with torch.cuda.nvtx.range("depth_sampling"):
                if target_d is not None:
                    z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
                    z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
                    z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

                    if self.config['training']['n_samples_d'] > 0:
                        # Optional mode: keep depth-centered samples, and reduce the number of
                        # uniform samples so they stop at depth (instead of always near->far).
                        # Falls back to classic near->far uniform sampling when disabled.
                        if self.config['training'].get('uniform_samples_until_depth', False):
                            near = self.config['cam']['near']
                            far = self.config['cam']['far']

                            depth_end = target_d.squeeze(-1).clamp(min=near, max=far)
                            valid_depth = (target_d.squeeze(-1) > 0)
                            n_samples_d = self.config['training']['n_samples_d']

                            # Compute an adaptive count from valid depths using the original near->far bins.
                            # This makes z_vals length shrink when scene depth is closer than far.
                            base_uniform = torch.linspace(near, far, n_samples_d).to(rays_o)
                            counts = (base_uniform[None, :] <= depth_end[:, None]).sum(dim=1)
                            counts = torch.where(valid_depth, counts, torch.zeros_like(counts))
                            if valid_depth.any():
                                n_uniform = max(1, int(counts[valid_depth].max().item()))
                            else:
                                n_uniform = n_samples_d

                            self.uniform_sampling_stats.record_batch(
                                ray_counts_before_depth=counts,
                                valid_depth=valid_depth,
                                selected_uniform_samples=n_uniform,
                                requested_uniform_samples=n_samples_d,
                            )

                            if self.config['training'].get('debug_uniform_samples_until_depth', False):
                                print(f"[sampling] uniform_until_depth=True n_uniform={n_uniform} n_range_d={self.config['training']['n_range_d']}")

                            t_vals = torch.linspace(0., 1., steps=n_uniform).to(rays_o)
                            z_uniform_depth = near * (1. - t_vals)[None, :] + depth_end[:, None] * t_vals[None, :]
                            z_uniform_full = torch.linspace(near, far, n_uniform)[None, :].repeat(n_rays, 1).to(rays_o)
                            z_uniform = torch.where(valid_depth[:, None], z_uniform_depth, z_uniform_full)
                        else:
                            z_uniform = torch.linspace(
                                self.config['cam']['near'],
                                self.config['cam']['far'],
                                self.config['training']['n_samples_d']
                            )[None, :].repeat(n_rays, 1).to(rays_o)

                        z_vals, _ = torch.sort(torch.cat([z_uniform, z_samples], -1), -1)
                    else:
                        z_vals = z_samples
                else:
                    z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
                    z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

            # Perturb sampling depths
                if self.config['training']['perturb'] > 0.:
                    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                    upper = torch.cat([mids, z_vals[...,-1:]], -1)
                    lower = torch.cat([z_vals[...,:1], mids], -1)
                    z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        with self.layer_timing.stage('compute_points'):
            with torch.cuda.nvtx.range("compute_points"):
                # Check the flag
                do_morton = self.config['grid'].get('morton2D', False)

                if do_morton:
                    # --- OPTIMIZED PATH: Transposed Layout [Samples, Rays, 3] ---
                    # 1. Transpose z_vals from [Rays, Samples] to [Samples, Rays]
                    z_vals_T = z_vals.permute(1, 0).contiguous()
                    
                    # 2. Broadcast: 
                    # rays_o/d: [Rays, 3] -> [1, Rays, 3]
                    # z_vals_T: [Samples, Rays] -> [Samples, Rays, 1]
                    # Result pts: [Samples, Rays, 3]
                    pts = rays_o[None, ...] + rays_d[None, ...] * z_vals_T[..., None]
                    
                    # 3. Flatten. In memory, this is now:
                    # Sample0_Ray0, Sample0_Ray1, Sample0_Ray2...
                    # Since we sorted Rays 0,1,2 to be neighbors, this is a linear memory read!
                    pts_flat = pts.reshape(-1, 3)
                    
                else:
                    # --- ORIGINAL PATH: Standard Layout [Rays, Samples, 3] ---
                    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] 
                    pts_flat = pts.reshape(-1, 3)
                
        with self.layer_timing.stage('run_network'):
            with torch.cuda.nvtx.range("run_network"):
                # Pass the flattened points (layout agnostic)
                raw_flat = self.run_network(pts_flat)
                
                # Reshape back based on which layout we used
                if do_morton:
                    # Output was [S*R, 4] -> reshape [S, R, 4]
                    raw = raw_flat.reshape(z_vals.shape[1], z_vals.shape[0], 4)
                    # Permute back to [R, S, 4] for volume rendering accumulation
                    raw = raw.permute(1, 0, 2).contiguous()
                else:
                    # Output was [R*S, 4] -> reshape [R, S, 4]
                    raw = raw_flat.reshape(z_vals.shape[0], z_vals.shape[1], 4)

        with self.layer_timing.stage('volume_rendering'):
            with torch.cuda.nvtx.range("volume_rendering"):
                rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Importance sampling
        if self.config['training']['n_importance'] > 0:
            with torch.cuda.nvtx.range("importance_sampling"):
                rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            with torch.cuda.nvtx.range("sample_pdf"):
                z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
                z_samples = z_samples.detach()

            with torch.cuda.nvtx.range("sort_importance_samples"):
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            
            with torch.cuda.nvtx.range("compute_importance_points"):
                pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            with torch.cuda.nvtx.range("run_network_importance"):
                raw = self.run_network(pts)

            with torch.cuda.nvtx.range("raw2outputs_importance"):
                rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Return rendering outputs
        with torch.cuda.nvtx.range("prepare_outputs"):
            ret = {'rgb' : rgb_map, 'depth' :depth_map, 
                'disp_map' : disp_map, 'acc_map' : acc_map, 
                'depth_var':depth_var,}
            ret = {**ret, 'z_vals': z_vals}

            ret['raw'] = raw

            if self.config['training']['n_importance'] > 0:
                ret['rgb0'] = rgb_map_0
                ret['disp0'] = disp_map_0
                ret['acc0'] = acc_map_0
                ret['depth0'] = depth_map_0
                ret['depth_var0'] = depth_var_0
                ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, global_step=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # Get render results
        with self.layer_timing.stage('render_rays'):
            with torch.cuda.nvtx.range("render_rays"):
                rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        if not self.training:
            return rend_dict
        
        # Get depth and rgb weights for loss
        with torch.cuda.nvtx.range("prepare_weights"):
            valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
            rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
            rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # Get render loss
        with self.layer_timing.stage('compute_render_losses'):
            with torch.cuda.nvtx.range("compute_render_losses"):
                rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
                psnr = mse2psnr(rgb_loss)
                depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

                if 'rgb0' in rend_dict:
                    rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
                    depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # Get sdf loss
        with self.layer_timing.stage('compute_sdf_losses'):
            with torch.cuda.nvtx.range("compute_sdf_losses"):
                z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
                sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
                truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
                fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)         
        
        with torch.cuda.nvtx.range("prepare_return_dict"):
            ret = {
                "rgb": rend_dict["rgb"],
                "depth": rend_dict["depth"],
                "rgb_loss": rgb_loss,
                "depth_loss": depth_loss,
                "sdf_loss": sdf_loss,
                "fs_loss": fs_loss,
                "psnr": psnr,
            }

        return ret
