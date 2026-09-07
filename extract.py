import pandas as pd

# Load the full data
df = pd.read_csv('office0_scene_rep_timing_rtx.csv')

def compute_rtx_percentages(stage_name):
    sdf = df[df['outer_stage'] == stage_name]
    if sdf.empty:
        print(f"No data found for {stage_name}")
        return
    
    # Calculate max per frame (handles duplicate logs), then mean across all frames
    avg_times = sdf.groupby(['frame_id', 'stage'])['cuda_ms'].max().groupby('stage').mean()
    
    render_rays = avg_times.get('render_rays', 0)
    loss_render = avg_times.get('compute_render_losses', 0)
    loss_sdf = avg_times.get('compute_sdf_losses', 0)
    
    total_forward = render_rays + loss_render + loss_sdf
    
    print(f"--- {stage_name.upper()} STAGE ---")
    
    # Level 2
    pct_rr = (render_rays / total_forward) * 100
    pct_loss = ((loss_render + loss_sdf) / total_forward) * 100
    print(f"Level 2 | Render rays: {pct_rr:.1f}% | SDF/Render Loss: {pct_loss:.1f}%")
    
    # Level 3
    pct_depth = (avg_times.get('depth_sampling', 0) / total_forward) * 100
    pct_net = (avg_times.get('run_network', 0) / total_forward) * 100 
    pct_vr = (avg_times.get('volume_rendering', 0) / total_forward) * 100
    pct_rr_other = pct_rr - pct_depth - pct_net - pct_vr
    print(f"Level 3 | Depth: {pct_depth:.1f}% | Network: {pct_net:.1f}% | VR: {pct_vr:.1f}% | Other: {pct_rr_other:.1f}%")
    
    # Level 4
    pct_hash = (avg_times.get('TCNN_hashgrid_encoding', 0) / total_forward) * 100
    pct_blob = (avg_times.get('TCNN_oneblob_encoding', 0) / total_forward) * 100
    pct_mlp = (avg_times.get('Decoder', 0) / total_forward) * 100
    pct_net_other = pct_net - pct_hash - pct_blob - pct_mlp
    print(f"Level 4 | Hash: {pct_hash:.1f}% | Blob: {pct_blob:.1f}% | MLP: {pct_mlp:.1f}% | Reshape tensors: {pct_net_other:.1f}%\n")

compute_rtx_percentages('Tracking')
compute_rtx_percentages('Bundle_Adjustment')