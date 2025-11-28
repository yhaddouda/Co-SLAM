*** get the report with ncu ***
sudo /usr/local/cuda/bin/ncu --target-processes all     --nvtx     --nvtx-include "TCNN_hashgrid_encoding/"     --nvtx-include "TCNN_oneblob_encoding/"     --nvtx-include "Decoder/"     --section MemoryWorkloadAnalysis     --section SourceCounters     --section SpeedOfLight     -f     --export coslam_nvtx_profile_cfgD     /bin/bash /home/yh279050/work/Co-SLAM/profile_coslam.sh --config configs/Replica/office0.yaml


--> --nvtx-include filters by nvtx range
--> --section MemoryWorkloadAnalysis  contains memory stats that are important



*** print average statistics of kernels after generating the report ***
sudo /usr/local/cuda/bin/ncu --import coslam_nvtx_profile_cfgD.ncu-rep --print-summary per-kernel