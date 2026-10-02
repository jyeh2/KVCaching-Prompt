# Control: is the cache discarded even without save/restore?
import phase0 as p
prefix, _ = p.build_prefix()
q = p.tokenize(p.QUESTIONS[0], False)
p.slot("erase")
p.post("/completion", {"prompt": prefix, "n_predict": 0, "cache_prompt": True, "id_slot": 0})
r, wall = p.complete(prefix + q, True)
print(f"same-session follow-up: processed {r['timings']['prompt_n']} of {len(prefix)+len(q)} tok, {r['timings']['prompt_ms']/1000:.1f}s")
