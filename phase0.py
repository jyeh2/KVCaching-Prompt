#!/usr/bin/env python3
"""Phase 0: does llama-server slot save/restore give correct, fast prefix reuse on Gemma4?

For each question Q after a long shared prefix P:
  cold : empty slot, process P+Q from scratch
  warm : restore the saved snapshot of P, then process only Q
and check the greedy outputs match. Requires run_server.sh on :8089.
"""
import glob, json, os, re, time, urllib.request

URL = "http://127.0.0.1:8089"
SLOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "slots")
SNAP = "prefix.bin"
TARGET_PREFIX_TOKENS = 15600          # about the size of OpenClaw's system prompt
N_PREDICT = 64

QUESTIONS = [
    "\n\nQuestion: In two sentences, what does the first document describe?\nAnswer:",
    "\n\nQuestion: List three rules stated in the reference material.\nAnswer:",
]


def post(path, body, timeout=900):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def tokenize(text, special):
    return post("/tokenize", {"content": text, "add_special": special})["tokens"]


def build_prefix():
    # Any set of long documents works; the original runs used ~15.6k tokens of private agent prompts and templates.
    files = sorted(glob.glob(os.path.expanduser(os.environ.get("PREFIX_GLOB", "prefix/*"))))
    assert files, "no prefix documents: put files in ./prefix or set PREFIX_GLOB"
    text = "You are an assistant. Reference material follows.\n"
    for f in files:
        text += f"\n=== {os.path.basename(f)} ===\n" + open(f, encoding="utf-8").read()
    toks = tokenize(text, True)
    return toks[:TARGET_PREFIX_TOKENS], len(files)


def complete(tokens, cache_prompt):
    t = time.time()
    r = post("/completion", {"prompt": tokens, "n_predict": N_PREDICT, "temperature": 0,
                             "top_k": 1, "seed": 0, "cache_prompt": cache_prompt, "id_slot": 0})
    return r, time.time() - t


def slot(action, **body):
    t = time.time()
    r = post(f"/slots/0?action={action}", body)
    return r, time.time() - t


def main():
    prefix, nfiles = build_prefix()
    print(f"prefix: {len(prefix)} tokens from {nfiles} files")
    results = {"prefix_tokens": len(prefix), "runs": []}

    # Build the snapshot once: process P only, save it.
    slot("erase")
    t = time.time()
    r = post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
    build_s = time.time() - t
    s, save_wall = slot("save", filename=SNAP)
    size = os.path.getsize(os.path.join(SLOTS, SNAP))
    print(f"build: {r['timings']['prompt_n']} tok in {build_s:.1f}s | save: {s.get('n_saved')} tok, "
          f"{s['timings']['save_ms']:.0f} ms, {size/2**20:.0f} MiB on disk")
    results.update(build_s=build_s, save_ms=s["timings"]["save_ms"], snapshot_mib=size / 2**20)

    for q in QUESTIONS:
        qt = tokenize(q, False)
        full = prefix + qt

        slot("erase")
        cold, cold_wall = complete(full, False)

        slot("erase")
        rs, _ = slot("restore", filename=SNAP)
        warm, warm_wall = complete(full, True)

        same = cold["content"] == warm["content"]
        run = {"question_tokens": len(qt),
               "cold": {"prompt_n": cold["timings"]["prompt_n"], "prompt_ms": cold["timings"]["prompt_ms"],
                        "ttft_s": cold["timings"]["prompt_ms"] / 1000, "wall_s": cold_wall},
               "warm": {"restore_ms": rs["timings"]["restore_ms"], "n_restored": rs.get("n_restored"),
                        "prompt_n": warm["timings"]["prompt_n"], "prompt_ms": warm["timings"]["prompt_ms"],
                        "ttft_s": (rs["timings"]["restore_ms"] + warm["timings"]["prompt_ms"]) / 1000,
                        "wall_s": warm_wall},
               "outputs_match": same, "cold_text": cold["content"], "warm_text": warm["content"]}
        results["runs"].append(run)
        c, w = run["cold"], run["warm"]
        print(f"\nQ ({len(qt)} tok): {q.strip()[:60]}")
        print(f"  cold: processed {c['prompt_n']} tok, time to first token {c['ttft_s']:.1f}s")
        print(f"  warm: restore {w['restore_ms']:.0f} ms ({w['n_restored']} tok) + processed {w['prompt_n']} tok "
              f"in {w['prompt_ms']:.0f} ms -> time to first token {w['ttft_s']:.2f}s")
        print(f"  outputs match: {same}")
        if not same:
            print("   cold:", repr(cold["content"][:200]))
            print("   warm:", repr(warm["content"][:200]))

    json.dump(results, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase0_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
