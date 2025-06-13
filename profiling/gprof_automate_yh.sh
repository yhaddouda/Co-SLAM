#!/bin/bash

# Set variables for file names
PROFILE_OUTPUT="whole_pipeline.profile"
DOT_OUTPUT="intermediate.dot"
PNG_OUTPUT="whole_pipeline.png"

# Store the current directory (profiling directory)
PROFILING_DIR=$(pwd)

# Change to the parent directory where coslam.py is located
cd ..

# Run the profiler from the correct directory
echo "Running Co-slam with cProfile..."
python -m cProfile -o "$PROFILING_DIR/$PROFILE_OUTPUT" coslam.py --config 'configs/Replica/office2.yaml'

# Return to profiling directory for processing
cd "$PROFILING_DIR"

# Check if the profiling was successful
if [ ! -f "$PROFILE_OUTPUT" ]; then
    echo "Error: Profiling failed. $PROFILE_OUTPUT not created."
    exit 1
fi

# Convert pstats to dot format
echo "Converting profile data to dot format..."
gprof2dot -f pstats "$PROFILE_OUTPUT" > "$DOT_OUTPUT"

# Check if the conversion was successful
if [ ! -f "$DOT_OUTPUT" ]; then
    echo "Error: Conversion to dot format failed. $DOT_OUTPUT not created."
    exit 1
fi

# Generate PNG from dot file
echo "Generating PNG visualization..."
dot -Tpng "$DOT_OUTPUT" -o "$PNG_OUTPUT"

# Check if the PNG generation was successful
if [ ! -f "$PNG_OUTPUT" ]; then
    echo "Error: PNG generation failed. $PNG_OUTPUT not created."
    exit 1
fi

echo "Profile visualization complete. Output saved as $PNG_OUTPUT"