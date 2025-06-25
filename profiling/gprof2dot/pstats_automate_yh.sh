#!/bin/bash

# Set variables for file names and paths
PROFILE_OUTPUT="100_iter.profile"
FILTER_PATH="/home/nvidia/work/source/nerfstudio/"
TEXT_OUTPUT="100_iter.txt"

# Run the profiler
echo "Running nerfacto with cProfile..."
python -m cProfile -o "$PROFILE_OUTPUT" $(which ns-train) nerfacto --data /home/nvidia/work/data/nerfstudio/poster

# Check if the profiling was successful
if [ ! -f "$PROFILE_OUTPUT" ]; then
    echo "Error: Profiling failed. $PROFILE_OUTPUT not created."
    exit 1
fi

# Process the profile data
echo "Processing profile data..."
python pstats_to_txt_yh.py "$PROFILE_OUTPUT" "$FILTER_PATH" "$TEXT_OUTPUT"

# Check if the processing was successful
if [ ! -f "$TEXT_OUTPUT" ]; then
    echo "Error: Profile data processing failed. $TEXT_OUTPUT not created."
    exit 1
fi

echo "Profiling and processing complete. Results saved in $TEXT_OUTPUT"