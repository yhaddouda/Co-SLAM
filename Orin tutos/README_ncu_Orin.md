*** 1- get the report with ncu, kernel wise ***
sudo /usr/local/cuda/bin/ncu --target-processes all    --kernel-name-base function  --kernel-name "regex:.*(grid|oneblob|sgemm).*" --launch-count 500  --section MemoryWorkloadAnalysis     --section SourceCounters     --section SpeedOfLight     -f     --export coslam_nvtx_profile_cfgD     /bin/bash /home/yh279050/work/Co-SLAM/profile_coslam.sh --config configs/Replica/office0.yaml


--> --nvtx-include filters by nvtx range
--> --section MemoryWorkloadAnalysis  contains memory stats that are important
--> its possible to do by kernel name which is even simpler, (cf RTX)


*** print average statistics of kernels after generating the report ***
sudo /usr/local/cuda/bin/ncu --import coslam_nvtx_profile_cfgD.ncu-rep --print-summary per-kernel


*** 2- get the report with ncu, nvtx wise ***
sudo /usr/local/cuda/bin/ncu     --target-processes all     --nvtx     --nvtx-include "BA_ITER_PROFILE_TOTAL/" --range-filter ::5 --metrics memory_l2_theoretical_sectors_global     --clock-control none     -f     --export coslam_BA_iter5_office0_T19_cfgD     /bin/bash /home/yh279050/work/Co-SLAM/Orin\ tutos/profile_coslam.sh --config configs/Replica/office0.yaml

--> metrics can be edited as you wish, here i extract the trafic that goes through L2 cache
--> --range-filter allows you to profile only the fifth appearance of the nvtx in this case (the fifth iteration of ba frame in this case)
--> if you want to only extract hashgrid inside ba for nested nvtx ranges you should do : --nvtx-include "BA_ITER_PROFILE_TOTAL/*/TCNN_hashgrid_encoding"

*** print average statistics of kernels by nvtx ranges after generating the report ***
ncu -i coslam_BA_iter5_office0_T19_cfgD.ncu-rep   --csv   --print-summary per-kernel   --print-metric-name name   --print-units base   --metrics memory_l2_theoretical_sectors_global,derived_memory_l2_theoretical_sectors_global_excessive   > coslam_BA_iter5_office0_T19_cfgD.csv


*** 3- get the report with ncu, nvtx wise, muti-thread (backward kernels included) ***
sudo /usr/local/cuda/bin/ncu     --target-processes all     --nvtx     --nvtx-include "TR_ITER_PROFILE_TOTAL"  --metrics memory_l2_theoretical_sectors_global     --clock-control none     -f     --export coslam_TR_office0_T19_cfgA_frame1_all     /bin/bash /home/yh279050/work/Co-SLAM/Orin\ tutos/profile_coslam.sh --config configs/Replica/office0.yaml

--> You have to have nvtx ranges that are start/end and not push/pop 
--> It doesn't work with --range-filter for some reason (so you have to profile the whole iteration)
