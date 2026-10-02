#!/usr/bin/env bash
# Standalone llama-server (Ollama's bundled build) with slot save/restore enabled.
# --swa-full keeps full history in sliding-window layers. Built-in prompt cache and context checkpoints are off so tests measure save/restore alone.
export LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:$LD_LIBRARY_PATH
export GGML_BACKEND_PATH=/usr/local/lib/ollama/cuda_v12/libggml-cuda.so
exec /usr/local/lib/ollama/llama-server --model /usr/share/ollama/.ollama/models/blobs/sha256-4c27e0f5b5adf02ac956c7322bd2ee7636fe3f45a8512c9aba5385242cb6e09a \
  --host 127.0.0.1 --port 8089 -c 32768 -np 1 -ngl 99 --flash-attn auto -b 512 -ub 512 \
  --slot-save-path "$(cd "$(dirname "$0")" && pwd)/slots" --cache-ram 0 --ctx-checkpoints 0 --no-webui --log-verbosity 4 --swa-full
