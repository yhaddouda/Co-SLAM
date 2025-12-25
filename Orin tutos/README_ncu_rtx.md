*** get the report with ncu ***
sudo /usr/local/NVIDIA-Nsight-Compute-2025.2/ncu     --target-processes all     --section SpeedOfLight     --section MemoryWorkloadAnalysis     --section SourceCounters     --kernel-name-base function     --kernel-name "regex:.*(grid|blob|sgemm).*"     --launch-count 500     -f     --export coslam_kernels_final_cfgC     /bin/bash /home/yh279050/work/Co-SLAM/Orin\ tutos/profile_coslam_rtx.sh --config configs/Replica/office0.yaml


--> --kernel-name  filters by kernel name which is much simpler than nvtx range which doesn't seem to work on rtx (probably because the gpu is so fast it causes mismatch in the time of the nvtx range and gpu exec)
--> --section MemoryWorkloadAnalysis  contains memory stats that are important
--> /usr/local/NVIDIA-Nsight-Compute-2025.2/ncu different from orin because here it is not the system one, but it was installed manually
--> profile_coslam_rtx.sh  is important when using cuda code for morton key generation, it tells ncu where to look for the ibrairies and not use the sudo ones, the file is different between Orin and RTX because it also sets the compute capability before building


*** print average statistics of kernels after generating the report ***
sudo /usr/local/NVIDIA-Nsight-Compute-2025.2/ncu --import coslam_nvtx_profile_cfgD.ncu-rep --print-summary per-kernel