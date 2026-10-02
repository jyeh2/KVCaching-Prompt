#!/usr/bin/env bash
# mistral.rs server on :8090 with Gemma4 E4B Q4_K_M (unsloth GGUF + mmproj).
# MISTRALRS_GEMMA4_PLE_CPU=1 (local patch) keeps the 1.8 GiB per-layer embedding table in host RAM,
#   as llama.cpp does, which leaves room for the paged-attention KV pool (needed for chunked prefill).
# --max-seq-len only sizes the device-map activation estimate (8 heads x len^2 would claim 17 GB at 32k);
#   the KV cache is sized by --pa-context-len (env CTX, default 18432: larger pools leave no room for the ~480 MB prefill workspace).
# -n 0:42 keeps all layers on the GPU.
# usage: run_mistralrs.sh [extra serve flags...]   e.g. --prefix-cache-n 0
# Stop Ollama's model first (`ollama stop gemma4:e4b`): both won't fit in 8 GB.
M=$HOME/models/gemma4-e4b
export MISTRALRS_GEMMA4_PLE_CPU=1
exec $HOME/src/mistral.rs/target/release/mistralrs serve -p 8090 --max-seq-len 4096 -n "0:42" \
  --pa-context-len ${CTX:-18432} "$@" \
  -m $M -f $M/gemma-4-E4B-it-Q4_K_M.gguf --mmproj $M/mmproj-F16.gguf
