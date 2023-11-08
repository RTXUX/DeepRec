#!/bin/bash

set -xu

SHIFTS=("14" "15" "16" "17")

for shi in "${SHIFTS[@]}"; do
    CACHE_SHARD_SHIFT="${shi}" ./run.sh "logs/sharded2/lru_ev_sharded_256m_$((1 << shi)).log" "--cache_size=256" "--ev=sharded" "--no_eval";
    rm -rf ./result/*
    rm -rf /opt/ev/*
done
