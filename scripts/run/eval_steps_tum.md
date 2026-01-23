# Run
python coslam.py --config "./configs/Tum/fr1_desk.yaml"

--> after the run is completed, a file named mesh_track1999.ply is produced in ./output/Replica/office0/table_size_val/mesh_track1999.ply

# cull the mesh
INPUT_MESH=output/TUM/fr_desk/table_size_val/mesh_track591.ply
python neural_slam_eval/cull_mesh.py --config configs/Tum/fr1_desk.yaml --input_mesh $INPUT_MESH --gt_pose

--> In the same folder where mesh_track591.ply was created, a new culled mesh gets created after the execution : the name is mesh_track591_cull_frustum.ply

# Evaluation 
REC_MESH=output/Synthetic/wr/19_base/mesh_track1675_cull_virt_cams.ply
GT_MESH=../datasets/vv_data/neural_rgbd_data/whiteroom/gt_mesh_cull_virt_cams.ply
python ./neural_slam_eval/eval_recon_headless.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type RGBD -2d -3d

--> The results get logged in the terminal : accuracy:  1.5266074039565112
completion:  1.5456740065082095
completion ratio:  96.18499875068665
Depth L1:  0.0007932754670036957
The ideal case is editing the file to log these metrics in a file