*** get the report with ncu ***
sudo /usr/local/cuda/bin/ncu --target-processes all    --kernel-name-base function  --kernel-name "regex:.*(grid|oneblob|sgemm).*" --launch-count 500  --section MemoryWorkloadAnalysis     --section SourceCounters     --section SpeedOfLight     -f     --export coslam_nvtx_profile_cfgD     /bin/bash /home/yh279050/work/Co-SLAM/profile_coslam.sh --config configs/Replica/office0.yaml


--> --nvtx-include filters by nvtx range
--> --section MemoryWorkloadAnalysis  contains memory stats that are important
--> its possible to do by kernel name which is even simpler, (cf RTX)


*** print average statistics of kernels after generating the report ***
sudo /usr/local/cuda/bin/ncu --import coslam_nvtx_profile_cfgD.ncu-rep --print-summary per-kernel