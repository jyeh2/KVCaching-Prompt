#!/usr/bin/env python3
"""Is the cached prefix numerically the same state as recomputing it? Compare token log-probabilities.

For each question Q after the shared prefix P, generate greedily with top-5 logprobs at every step.
  cold : server started with --prefix-cache-n 0 (run twice: cold vs cold is the engine's own noise floor)
  warm : prefix cache on; P is sent once, then each P+Q reuses it
compare_logprobs.py then lines the runs up token by token.

usage: phase0_logprobs.py cold|cold2|warm      (server from run_mistralrs.sh on :8090)
"""
import json, os, sys, time
from phase0_mistralrs import URL, post, model_id, build_prefix

N_PREDICT = 32
TOP = 5
QUESTIONS = [
    "\n\nQuestion: In two sentences, what does the first document describe?\nAnswer:",
    "\n\nQuestion: List three rules stated in the reference material.\nAnswer:",
    "\n\nQuestion: Which document is the longest?\nAnswer:",
    "\n\nQuestion: What file names appear in the reference material?\nAnswer:",
    "\n\nQuestion: Name two things the material says to avoid.\nAnswer:",
    "\n\nQuestion: What is one hard constraint stated in the material?\nAnswer:",
    "\n\nQuestion: How many documents are included?\nAnswer:",
    "\n\nQuestion: Which document gives formatting rules?\nAnswer:",
    "\n\nQuestion: Summarize the purpose of the reference material in one sentence.\nAnswer:",
    "\n\nQuestion: Which documents are templates?\nAnswer:",
]


def generate(model, prompt):
    r = post("/v1/completions", {"model": model, "prompt": prompt, "max_tokens": N_PREDICT,
                                 "temperature": 0, "top_k": 1, "seed": 0, "logprobs": TOP})
    u = r["usage"]
    return {"text": r["choices"][0]["text"], "logprobs": r["choices"][0]["logprobs"],
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "prompt_tokens": u["prompt_tokens"]}


def main():
    mode = sys.argv[1]
    model = model_id()
    prefix, _ = build_prefix(model)
    out = {"mode": mode, "runs": []}
    if mode == "warm":
        post("/v1/completions", {"model": model, "prompt": prefix, "max_tokens": 1, "temperature": 0})
    for q in QUESTIONS:
        r = generate(model, prefix + q)
        out["runs"].append({"question": q.strip(), **r})
        print(f"{mode}: cached {r['cached_tokens']}/{r['prompt_tokens']} | {r['text'][:70]!r}", flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"logprobs_{mode}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
