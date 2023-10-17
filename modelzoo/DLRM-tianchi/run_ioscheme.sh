#!/bin/bash

IOSCHEME=("mmap_and_madvise" "mmap" "directio")

set -x

for ioscheme in "${IOSCHEME[@]}"; do
    TF_SSDHASH_IO_SCHEME="${ioscheme}" ./run.sh "logs/ioscheme/lru_ev_512m_${ioscheme}.log" --cache_size=512 --ev=tiered;
    rm -rf ./result/*
    rm -rf /opt/ev/*
done 

for ioscheme in "${IOSCHEME[@]}"; do
    TF_SSDHASH_IO_SCHEME="${ioscheme}" ./run.sh "logs/ioscheme/lru_ev_256m_${ioscheme}.log" --cache_size=256 --ev=tiered;
    rm -rf ./result/*
    rm -rf /opt/ev/*
done 
