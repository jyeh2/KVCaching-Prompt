#!/usr/bin/env python3
"""Line up two logprob runs token by token (see phase0_logprobs.py).

For each question, walk the generated tokens while both runs chose the same token and record the
largest |difference| in log-probability over the shared top-5 alternatives. At the first position
where the runs choose different tokens, report the cold run's margin between its top two tokens:
a flip at a near-tie is rounding noise; a flip where cold was confident would mean a wrong cache state.

usage: compare_logprobs.py A B      e.g. cold warm, cold cold2
"""
import json, sys


def steps(run):
    return [{t["token"]: t["logprob"] for t in s["top_logprobs"]} for s in run["logprobs"]["content"]]


def top_id(d):
    return max(d, key=d.get)


def compare(a, b):
    sa, sb = steps(a), steps(b)
    max_diff, agree = 0.0, 0
    for da, db in zip(sa, sb):
        if top_id(da) != top_id(db):
            two = sorted(da.values(), reverse=True)[:2]
            return {"agree": agree, "max_diff": max_diff, "diverged": True, "margin_at_flip": two[0] - two[1]}
        for t in set(da) & set(db):
            max_diff = max(max_diff, abs(da[t] - db[t]))
        agree += 1
    return {"agree": agree, "max_diff": max_diff, "diverged": False, "margin_at_flip": None}


def main():
    A, B = (json.load(open(f"logprobs_{m}.json")) for m in sys.argv[1:3])
    print(f"{sys.argv[1]} vs {sys.argv[2]}")
    all_diffs = []
    for ra, rb in zip(A["runs"], B["runs"]):
        c = compare(ra, rb)
        all_diffs.append(c["max_diff"])
        flip = (f"diverged after {c['agree']} tokens, cold top-2 margin {c['margin_at_flip']:.3f} nats"
                if c["diverged"] else f"all {c['agree']} tokens identical")
        print(f"  max |dlogprob| {c['max_diff']:.4f} | {flip} | cached {rb['cached_tokens']} | {ra['question'][:48]}")
    print(f"  worst max |dlogprob| over all questions: {max(all_diffs):.4f}")


if __name__ == "__main__":
    main()
