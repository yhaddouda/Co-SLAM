
# Annotated code with nvtx, then the following command : 
sudo nsys profile   --trace cuda,osrt,nvtx,cudnn,cublas   --gpu-metrics-devices all   --cuda-memory-usage true   --force-overwrite true --output profile_20_nvtx_complete   /home/yh279050/miniforge3/envs/coslam/bin/python coslam.py --config configs/Replica/office0.yaml

# After generating the .nsys-rep file with the previous command, you can generate statistics for a nvtx range (function) with this command :
nsys stats --report nvtx_sum --format csv --output Orin_Morton_R128_T13.csv Orin_Morton_R128_T13.nsys-rep

# For the cuda morton 
For the moment I don't use sudo and --gpu-metrics-devices all 