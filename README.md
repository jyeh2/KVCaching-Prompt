# kvsnap: reusing the KV cache of a long shared prompt

Agent frameworks resend the same long system prompt with every request. On one 8 GB
secondhand GPU (NVIDIA CMP 40HX), processing a 15.6k-token prompt took 54-85 s before the
first token. This repo measures two ways to skip that work by reusing the KV cache of the
shared prefix:

- **llama.cpp** (`llama-server` slot save/restore): restoring a 264 MiB snapshot brings time
  to first token to 0.29-0.34 s on Gemma 4 E4B, but only with `--swa-full`. In default mode
  the restored cache is silently discarded.
- **mistral.rs** (built-in prefix cache): 54 s cold to **0.27 s** warm on Gemma 4 E4B, with a
  token-level check that 8 of 10 cached answers are identical to recomputing (the other 2
  split at near-ties). Getting Gemma 4 to fit on 8 GB took a 17-line patch to mistral.rs,
  included in [`patches/`](patches/).

Write-up with charts: [kvsnap: mistral.rs vs llama.cpp](https://jyeh2.github.io/projects/posts/2026-09-kvsnap-kv-cache-reuse.html)

## About the data

The original prefix was ~15.6k tokens of private agent prompts and templates, which aren't
published. Model answers in the result files are redacted (`[redacted]`), questions are
replaced by `Q1`-`Q10`, and token strings and IDs in the logprob files are replaced by
opaque labels that keep identical tokens identical. All timings, token counts and
logprobs are unchanged, and `compare_logprobs.py` reproduces the published comparison from
the redacted files. The scripts read prefix documents from `./prefix/` (or `PREFIX_GLOB`)
and ask generic questions, so you can rerun everything on your own documents.

## Running

- llama.cpp: `./run_server.sh` (or `run_server_swa_full.sh`), then `python3 phase0.py`.
- mistral.rs: build mistral.rs with `patches/0001-gemma4-ple-cpu-mistralrs.patch` applied
  (`git am`), start `./run_mistralrs.sh`, then `python3 phase0_mistralrs.py cold|warm LABEL`.
  Restart the server with `--prefix-cache-n 0` for the cold runs.
- Token-level check: `python3 phase0_logprobs.py cold|cold2|warm`, then
  `python3 compare_logprobs.py cold warm`.

## Phase 0 results (2026-09-25)

Setup: Gemma4 e4b (Ollama's blob) on one CMP 40HX, Ollama's bundled llama-server,
32k context, prompt cache and context checkpoints off. Prefix = 15,600 tokens of
private agent prompts and templates. Script: `phase0.py`. Raw numbers: `phase0_results_*.json`.

| | default | `--swa-full` |
|---|---|---|
| Snapshot save | 210 ms | 505 ms |
| Snapshot restore | 169 ms | 169 ms |
| Snapshot on disk | 264 MiB | 264 MiB |
| Cold time to first token | 85 s | 119 s |
| Warm time to first token (restore + new tokens) | 85.6 s (restored cache discarded) | **0.29-0.34 s** |
| VRAM at 32k context | 3.6 GB | 4.9 GB |

Findings:
1. Save/restore works mechanically in both modes.
2. Default mode: after a restore the server logs "forcing full prompt re-processing
   due to lack of cache data (likely due to SWA...)" and reprocesses everything.
   Reuse within one session does work (20 new tokens in 0.2 s, `exp_noreload.py`);
   only the restore path fails.
3. `--swa-full` fixes it: only the new tokens are processed.
4. Costs of `--swa-full`: +1.3 GB VRAM at 32k, and cold prefill was ~40% slower
   (the GPU clock also read 1470 MHz instead of 1935 at the start of that run, so
   part of the gap may be the clock; not yet separated).
5. Output check: 1 of 2 warm answers matched cold word for word; the other diverged
   after ~25 tokens. Likely small numeric differences from processing the question
   in its own batch, not a broken cache, but not proven. Compare logits instead of
   exact text in Phase 1.

### Qwen2.5 3B, default settings (full attention, no `--swa-full`)

Script: `run_phase0_qwen25.sh`. Raw numbers: `phase0_results_qwen25_3b.json`.

| | Qwen2.5 3B |
|---|---|
| Snapshot save / restore | 1007 ms / 280-300 ms |
| Snapshot on disk | 549 MiB (estimate from model config was ~550 MiB) |
| Cold time to first token | 83-84 s |
| Warm time to first token | **0.44-0.45 s** (only the 15-19 new tokens processed) |
| KV cache in VRAM at 32k | 1152 MiB |

Restore works out of the box: no forced reprocessing. Same output pattern as Gemma:
one answer identical, the other identical for its first ~200 characters and then
different, so the drift is not specific to sliding-window models. This confirms
Qwen2.5 3B as the Phase 1-2 development model.

### Gemma4 E4B on mistral.rs 0.9.4 (2026-09-27), default settings, no --swa-full

Scripts: `run_mistralrs.sh` (server on :8090), `phase0_mistralrs.py cold|warm LABEL`.
Raw numbers: `phase0_mistralrs_gemma4_e4b_{cold,warm}.json`. mistral.rs has no slot save/restore,
so this measures its built-in in-memory prefix cache (block-level, paged attention, 32-token blocks).

| | Gemma4 E4B (unsloth Q4_K_M + mmproj) |
|---|---|
| Cold time to first token (`--prefix-cache-n 0`) | 54.3-54.5 s |
| Warm (prefix sent once, then prefix + question) | **0.27 s**, 15,584 of ~15,620 tokens cached |
| Extension (prefix + question + answer + follow-up) | 0.36-0.52 s |
| VRAM | ~6.5 GB incl. 18k-token KV pool |

Restore of the sliding-window state works, unlike llama.cpp default mode. Output check: same pattern as
before (one answer identical, the other diverges after 13 words).

Token-level check (`phase0_logprobs.py`, `compare_logprobs.py`; 10 questions, 32 greedy tokens, top-5 logprobs):
cold vs cold is bit-identical. Cold vs warm: 8/10 token-identical; 2 diverged after 15-16 tokens at near-ties
(cold top-2 margin 0.023 and 0.041 nats). On matching tokens, the chosen-token |dlogprob| is median 0.0016,
p95 0.074, max 0.12; the probability shift over the top-5 is median 0.2%, max 6.6%. So the cached state is
correct but not bit-exact: the question tokens are attended over stored K/V in a separate chunk, so fp16 sums
run in a different order.

Needed on the 8 GB sm_75 card:
- `--max-seq-len 4096`: it only sizes the device-map activation estimate (heads x len^2 claimed 17 GB at 32k).
- `-n 0:42`: auto-mapping put layers 23-41 on the CPU.
- Patch `MISTRALRS_GEMMA4_PLE_CPU=1` ([`patches/`](patches/), mistralrs-core/src/vision_models/gemma4/text.rs, 17 lines):
  keeps the 1.8 GiB per-layer embedding table in host RAM like llama.cpp; frees 1.85 GB for the paged KV pool.
  Paged attention is required: the non-paged path does not chunk prompts and OOMs on 15.6k tokens.
- `--max-num-batched-tokens 512` and an 18k KV pool: without flash attention (sm_75) prefill uses a
  gather workspace that grows with chunk size (973 MB at 4096).

Not done: the cache is in GPU memory only and is lost on restart. Next: a disk tier inside mistral.rs keyed by
its content block hashes (see upstream #2308 / #2364 connector seam) instead of a proxy in front of llama-server.

## Files
- `run_server.sh` / `run_server_swa_full.sh`: launch llama-server on :8089.
  Stop Ollama's model first (`ollama stop gemma4:e4b`): both won't fit in 8 GB.
- `phase0.py`: cold vs save/restore benchmark.
- `exp_noreload.py`, `exp_restore.py`: the two diagnostic experiments.
- `patches/0001-gemma4-ple-cpu-mistralrs.patch`: the mistral.rs patch, as a `git am` file.
- `slots/`: saved snapshots (not committed).

## Open questions for Phase 1
- Can the default (small SWA cache) mode be made to work by taking a llama.cpp
  context checkpoint at the prefix boundary and saving it with the snapshot?
  That would avoid the VRAM and prefill cost of `--swa-full`.
- Why is the snapshot the same size in both modes?
- Separate the prefill slowdown from the GPU clock.
