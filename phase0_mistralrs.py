#!/usr/bin/env python3
"""Phase 0 on mistral.rs: does its built-in prefix cache give fast reuse of a long shared prefix?

mistral.rs has no slot save/restore, so this measures its automatic in-memory prefix cache:
  cold      : server started with --prefix-cache-n 0, process P+Q from scratch
  warm      : server with prefix cache on; process P once, then P+Q
  extension : P+Q+answer+follow-up after P+Q (the multi-turn chat case)
Run once per mode against a server on :8090 (see run_mistralrs.sh), then compare the JSON files.

usage: phase0_mistralrs.py cold|warm LABEL
"""
import glob, json, os, sys, time, urllib.request

URL = "http://127.0.0.1:8090"
TARGET_PREFIX_TOKENS = 15600          # about the size of OpenClaw's system prompt
N_PREDICT = 64

QUESTIONS = [
    "\n\nQuestion: In two sentences, what does the first document describe?\nAnswer:",
    "\n\nQuestion: List three rules stated in the reference material.\nAnswer:",
]
FOLLOW_UP = "\n\nQuestion: Now say the same thing in one sentence.\nAnswer:"


def post(path, body, timeout=1800):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def model_id():
    return json.load(urllib.request.urlopen(URL + "/v1/models"))["data"][0]["id"]


def complete(model, prompt, n):
    t = time.time()
    r = post("/v1/completions", {"model": model, "prompt": prompt, "max_tokens": n,
                                 "temperature": 0, "top_k": 1, "seed": 0})
    wall = time.time() - t
    u = r["usage"]
    return {"text": r["choices"][0]["text"], "prompt_tokens": u["prompt_tokens"],
            "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "prompt_s": u["total_prompt_time_sec"], "wall_s": wall}


def count_tokens(model, text):
    return post("/v1/messages/count_tokens", {"model": model, "messages": [{"role": "user", "content": text}]})["input_tokens"]


def build_prefix(model):
    # Any set of long documents works; the original runs used ~15.6k tokens of private agent prompts and templates.
    files = sorted(glob.glob(os.path.expanduser(os.environ.get("PREFIX_GLOB", "prefix/*"))))
    assert files, "no prefix documents: put files in ./prefix or set PREFIX_GLOB"
    text = "You are an assistant. Reference material follows.\n"
    for f in files:
        text += f"\n=== {os.path.basename(f)} ===\n" + open(f, encoding="utf-8").read()
    # No raw-token API: binary-search a character cut that lands near the target token count.
    # count_tokens wraps the text in the chat template, so subtract that overhead.
    overhead = count_tokens(model, "")
    lo, hi = 0, len(text)
    while hi - lo > 16:
        mid = (lo + hi) // 2
        if count_tokens(model, text[:mid]) - overhead <= TARGET_PREFIX_TOKENS:
            lo = mid
        else:
            hi = mid
    return text[:lo], len(files)


def main():
    mode, label = sys.argv[1], sys.argv[2]
    model = model_id()
    prefix, nfiles = build_prefix(model)
    results = {"mode": mode, "label": label, "model": model, "runs": []}

    if mode == "warm":
        b = complete(model, prefix, 1)
        results["build"] = b
        print(f"build: {b['prompt_tokens']} tok in {b['prompt_s']:.1f}s (cached {b['cached_tokens']})")

    for q in QUESTIONS:
        r = complete(model, prefix + q, N_PREDICT)
        run = {"question": q.strip(), "main": r}
        print(f"\nQ: {q.strip()[:60]}")
        print(f"  {mode}: {r['prompt_tokens']} prompt tok, {r['cached_tokens']} cached, "
              f"prompt time {r['prompt_s']:.2f}s, wall {r['wall_s']:.1f}s")
        if mode == "warm":
            e = complete(model, prefix + q + r["text"] + FOLLOW_UP, N_PREDICT)
            run["extension"] = e
            print(f"  extension: {e['prompt_tokens']} prompt tok, {e['cached_tokens']} cached, "
                  f"prompt time {e['prompt_s']:.2f}s")
        results["runs"].append(run)
        print("  text:", repr(r["text"][:200]))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"phase0_mistralrs_{label}_{mode}.json")
    json.dump(results, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
