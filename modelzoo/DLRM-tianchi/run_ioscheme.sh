#!/bin/bash

IOSCHEME=("mmap_and_madvise" "mmap" "directio")

set -x

CACHE_SIZE=("8192" "4096" "2048" "1024" "512")

for size in "${CACHE_SIZE[@]}"; do
    for ioscheme in "${IOSCHEME[@]}"; do
        TF_SSDHASH_IO_SCHEME="${ioscheme}" ./run.sh "logs/ioscheme/lru_ev_${size}m_${ioscheme}.log" "--cache_size=${size}" --ev=tiered;
        rm -rf ./result/*
        rm -rf /opt/ev/*
    done 
done