#!/bin/bash

configs=(
    "./configs/Replica/office0.yaml"
)

logfile="log_Replica_14.txt"
echo "Run started: $(date)" > $logfile

for cfg in "${configs[@]}"; do
    echo "Running config: $cfg"

    start=$(date +%s)

    # Start nvidia-smi logging (every 1s)
    tmp_log=$(mktemp)
    (
        while true; do
            nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
            sleep 1
        done
    ) > "$tmp_log" &
    nsmi_pid=$!

    # Run program
    python coslam.py --config "$cfg"

    # Stop nvidia-smi logging
    kill $nsmi_pid 2>/dev/null || true
    sleep 0.5

    end=$(date +%s)
    runtime=$((end - start))

    # Extract peak GPU memory usage
    peak_gpu=$(sort -nr "$tmp_log" | head -1)

    echo "$cfg: time=${runtime}s, peak_GPU=${peak_gpu} MB" | tee -a $logfile

    rm "$tmp_log"
done

echo "Run finished: $(date)" >> $logfile

