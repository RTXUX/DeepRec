#!/bin/bash

# Default environment variables
export CACHE_PROFILER_BUCKET_SIZE="${CACHE_PROFILER_BUCKET_SIZE:-10}"
export CACHE_PROFILER_MAX_REUSE_DIST="${CACHE_PROFILER_MAX_REUSE_DIST:-100000}"
export CACHE_TUNING_INTERVAL="${CACHE_TUNING_INTERVAL:-100000}"
export CACHE_TOTAL_SIZE="${CACHE_TOTAL_SIZE:-$((32 * 1024 * 1024))}"
export CACHE_MIN_SIZE="${CACHE_MIN_SIZE:-$((2048 * 128 * 8))}"
export CACHE_TUNING_UNIT="${CACHE_TUNING_UNIT:-$((8 * 128))}"
export CACHE_TUNING_STRATEGY="${CACHE_TUNING_STRATEGY:-min_mc_random_greedy}"
export CACHE_PROFLER_CLEAR="${CACHE_PROFLER_CLEAR:-true}"
export CACHE_REPORT_INTERVAL="${CACHE_REPORT_INTERVAL:-10000}"
export TF_SSDHASH_ASYNC_COMPACTION="${TF_SSDHASH_ASYNC_COMPACTION:-false}"

# Output file is first argument
OUTPUT_FILE="$1"
shift

# If output file exists, check if user wants to append
if [ -e "$OUTPUT_FILE" ]; then
    echo "$OUTPUT_FILE exists. Do you want to append to it? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Exiting..."
        exit 1
    fi
fi

# Print the date and environment variables to console and output file
date | tee -a "$OUTPUT_FILE"
env | tee -a "$OUTPUT_FILE"
CMD="./run.py $@"
echo "${CMD}" | tee -a "${OUTPUT_FILE}"
# Run the program with remaining arguments, print stdout and stderr to console and append to output file
./run.py "$@" 2>&1 | tee -a "$OUTPUT_FILE"