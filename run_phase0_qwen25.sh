#!/usr/bin/env bash
# Phase 0 on Qwen2.5 3B with default settings (no --swa-full).
cd ~/kvsnap
./run_server_qwen25.sh > server_qwen25.log 2>&1 &
until curl -s localhost:8089/health | grep -q ok; do sleep 3; done
python3 phase0.py
cp phase0_results.json phase0_results_qwen25_3b.json
kill $(pgrep -x llama-server)
