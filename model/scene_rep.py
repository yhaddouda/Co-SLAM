# package imports
import torch
import torch.nn as nn

# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v2
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss

#nvtx 
import nvtx

class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()
        self.config = config
        self.bounding_box = bound_box
        self.get_resolution()
        self.get_encoding(config)
        self.get_decoder(config)
        

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
            self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])

        # Sparse parametric encoding (SDF)
        with torch.cuda.nvtx.range("parametric_encoder_sdf"):
            self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)

        # Sparse parametric encoding (Color)
        if not self.config['grid']['oneGrid']:
            print('Color resolution:', self.resolution_color)
            with torch.cuda.nvtx.range("parametric_encoder_color"):
                self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color)

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

        with torch.cuda.nvtx.range("TCNN_hashgrid_encoding"):
            embedded = self.embed_fn(inputs_flat)

        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])
        
        # grid interpolation
        with torch.cuda.nvtx.range("TCNN_oneblob_encoding"):
            embedded_pos = self.embedpos_fn(inputs_flat)

        # sdf mlp
        with torch.cuda.nvtx.range("sdf_mlp"):
            out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))

        sdf, geo_feat = out[..., :1], out[..., 1:]

        with torch.cuda.nvtx.range("reshape_tensor"):
            sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        with torch.cuda.nvtx.range("reshape_tensor"):
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
        with torch.cuda.nvtx.range("TCNN_hashgrid_encoding"):
            embed = self.embed_fn(inputs_flat)

        # one blob 
        with torch.cuda.nvtx.range("TCNN_oneblob_encoding"):
            embe_pos = self.embedpos_fn(inputs_flat)

        if not self.config['grid']['oneGrid']:
            with torch.cuda.nvtx.range("TCNN_fn_color"):
                embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        
        # decoder is called at return very interesting
        return self.decoder(embed, embe_pos)
    
    def run_network(self, inputs):
        """
        Run the network on a batch of inputs.

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        with torch.cuda.nvtx.range("reshape_inputs"):
            inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
        
        # Normalize the input to [0, 1] (TCNN convention)
        if self.config['grid']['tcnn_encoding']:
            with torch.cuda.nvtx.range("normalize_inputs"):
                inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

        with torch.cuda.nvtx.range("query_color_sdf"):
            outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)

        with torch.cuda.nvtx.range("reshape_outputs"):
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
        with torch.cuda.nvtx.range("depth_sampling"):
            if target_d is not None:
                z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
                z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
                z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

                if self.config['training']['n_samples_d'] > 0:
                    z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                    z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
                else:
                    z_vals = z_samples
            else:
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
                z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]

        # Perturb sampling depths
        with torch.cuda.nvtx.range("perturb_depths"):
            if self.config['training']['perturb'] > 0.:
                mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
                upper = torch.cat([mids, z_vals[...,-1:]], -1)
                lower = torch.cat([z_vals[...,:1], mids], -1)
                z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # Run rendering pipeline
        with torch.cuda.nvtx.range("compute_points"):
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        with torch.cuda.nvtx.range("run_network"):
            raw = self.run_network(pts)
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
        with torch.cuda.nvtx.range("compute_render_losses"):
            rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
            psnr = mse2psnr(rgb_loss)
            depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

            if 'rgb0' in rend_dict:
                rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
                depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # Get sdf loss
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
