# Restore path alone: the snapshot from phase0 is still on disk.
import phase0 as p
prefix, _ = p.build_prefix()
q = p.tokenize(p.QUESTIONS[0], False)
p.slot("erase")
rs, _ = p.slot("restore", filename=p.SNAP)
print("restore:", {k: rs[k] for k in rs if k != "timings"}, rs["timings"])
r, wall = p.complete(prefix + q, True)
print(f"after restore: processed {r['timings']['prompt_n']} of {len(prefix)+len(q)} tok, {r['timings']['prompt_ms']/1000:.1f}s")
