# For the cuda morton 
sudo nsys profile   --trace cuda,osrt,nvtx,cudnn,cublas   --gpu-metrics-devices all   --cuda-memory-usage true   --force-overwrite true --output profile_20_T20_cfgA   /bin/bash /home/yh279050/work/Co-SLAM/Orin\ tutos/profile_coslam_rtx.sh  --config configs/Replica/office0.yaml

# After generating the .nsys-rep file with the previous command, you can generate statistics for a nvtx range (function) with this command :
nsys stats --report nvtx_sum --format csv --output Orin_Morton_R128_T13.csv Orin_Morton_R128_T13.nsys-rep