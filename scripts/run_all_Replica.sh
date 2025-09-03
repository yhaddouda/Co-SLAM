#!/bin/bash

configs=(
    "./configs/Replica/office0.yaml"
    """
    "./configs/Replica/office1.yaml"
    "./configs/Replica/office2.yaml"
    "./configs/Replica/office3.yaml"
    "./configs/Replica/office4.yaml"
    "./configs/Replica/room0.yaml"
    "./configs/Replica/room1.yaml"
    "./configs/Replica/room2.yaml"    
    """
)

logfile="log_Replica_13_half.txt"
echo "Run started: $(date)" > $logfile

for cfg in "${configs[@]}"; do
    echo "Running config: $cfg"

    start=$(date +%s)

    # Start tegrastats logging (1s interval)
    tmp_log=$(mktemp)
    tegrastats --interval 1000 > "$tmp_log" &
    ts_pid=$!

    # Run program
    python coslam.py --config "$cfg"

    # Stop tegrastats
    kill $ts_pid 2>/dev/null || true
    sleep 0.5

    end=$(date +%s)
    runtime=$((end - start))

    # Extract peak RAM usage from tegrastats
    peak_ram=$(grep -oP "RAM \K[0-9]+(?=/)" "$tmp_log" | sort -nr | head -1)

    echo "$cfg: time=${runtime}s, peak_RAM=${peak_ram} MB" | tee -a $logfile

    rm "$tmp_log"
done

echo "Run finished: $(date)" >> $logfile

