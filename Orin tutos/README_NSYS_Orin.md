
# Annotated code with nvtx, then the following command : 
sudo nsys profile   --trace cuda,osrt,nvtx,cudnn,cublas   --gpu-metrics-devices all   --cuda-memory-usage true   --force-overwrite true --output profile_20_nvtx_complete   /bin/bash /home/yh279050/work/Co-SLAM/profile_coslam.sh --config configs/Replica/office0.yaml

--> profile_coslam.sh  : important for running cuda morton optimised code, because it tells nsight where to look for the libs and sets the envs variables in conda and not sudo

# After generating the .nsys-rep file with the previous command, you can generate statistics for a nvtx range (function) with this command :
nsys stats --report nvtx_sum --format csv --output Orin_Morton_R128_T13.csv Orin_Morton_R128_T13.nsys-rep

