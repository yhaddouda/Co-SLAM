
# Annotated code with nvtx, then the following command : 
sudo nsys profile   --trace cuda,osrt,nvtx,cudnn,cublas   --gpu-metrics-devices all   --cuda-memory-usage true   --force-overwrite true --output profile_20_nvtx_complete   /home/yh279050/miniforge3/envs/coslam/bin/python coslam.py --config configs/Replica/office0.yaml