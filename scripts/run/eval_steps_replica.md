# Run
python coslam.py --config "./configs/Replica/office0.yaml"

--> after the run is completed, a file named mesh_track1999.ply is produced in ./output/Replica/office0/table_size_val/mesh_track1999.ply

# cull the mesh
INPUT_MESH=output/Replica/office0/table_size_val/mesh_track1999.ply
VIRT_CAM_PATH=../datasets/vv_data/Replica/office0/virtual_cameras/
python neural_slam_eval/cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose  

--> In the same folder where mesh_track1999.ply was created, a new culled mesh gets created after the execution : the name is mesh_track1999_cull_virt_cams.ply

# Evaluation 
REC_MESH=output/Replica/office0/table_size_val/mesh_track1999_cull_virt_cams.ply
GT_MESH=../datasets/vv_data/Replica/office0/gt_mesh_cull_virt_cams.ply
python ./neural_slam_eval/eval_recon_headless.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -2d -3d

--> The results get logged in the terminal : accuracy:  1.5266074039565112
completion:  1.5456740065082095
completion ratio:  96.18499875068665
Depth L1:  0.0007932754670036957
The ideal case is editing the file to log these metrics in a file