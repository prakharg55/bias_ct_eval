"""Build the training set that every method trains on.

Takes suggested-answer rows 100 to 499 of MMLU, HellaSwag, LogiQA and TruthfulQA from the dumps of
Chua et al. (2024), drops the pairs whose cue the loader removed (1,213 questions remain), and places
every cue before the question ("<cue>\\n<question>"). Cues that name the correct answer are kept.
Also writes the inputs of the AttCT repository's BCT target generator and a 1:1 sample of its
instruction data.

Usage: python -m ctcue.data.build_training_set
Output: $CT_ROOT/shared/shared_train.jsonl and $CT_ROOT/shared/bct_src/
"""

import difflib, json, os, random, hashlib

from ctcue.common.dumps import load_bct_source
from ctcue.paths import CT_ROOT, BCT_DUMPS, ATTCT_REPO

DUMPS = f"{BCT_DUMPS}/test"
OUT = f"{CT_ROOT}/shared"
REPO = ATTCT_REPO
DATASETS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]
ROWS = (100, 500)


def main():
    os.makedirs(f"{OUT}/bct_src", exist_ok=True)
    recs = []
    n_front_ok = n_front = n_multiline = 0
    seen_clean = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/suggested_answer/{ds}_suggested_answer.jsonl")
        for r, p in enumerate(pairs[ROWS[0] : ROWS[1]], start=ROWS[0]):
            b, u = p["biased_messages"], p["unbiased_messages"]
            # cue deleted by the loader: no cue to train on
            if b == u:
                continue
            assert len(b) == len(u) == 1 and b[0]["role"] == u[0]["role"] == "user"
            bt, ut = b[0]["content"], u[0]["content"]
            ops = [
                o
                for o in difflib.SequenceMatcher(None, ut, bt, autojunk=False).get_opcodes()
                if o[0] != "equal"
            ]
            assert len(ops) == 1 and ops[0][0] == "insert", (ds, r, ops)  # exactly one inserted cue
            cue = bt[ops[0][3] : ops[0][4]].strip()
            n_multiline += "\n" in cue
            biased = cue + "\n" + ut
            # already front layout: must be reproduced exactly
            if bt.endswith(ut):
                n_front += 1
                assert biased == bt, (ds, r, repr(bt[:80]))
                n_front_ok += 1
            assert ut not in seen_clean, (ds, r, "duplicate clean prompt")
            seen_clean.add(ut)
            assert biased.count(ut) == 1 and biased.endswith(ut)
            recs.append(
                dict(
                    i=len(recs),
                    ds=ds,
                    row=r,
                    hash=p["original_question_hash"],
                    X=p["bias_target"],
                    T=p["correct_letter"],
                    cue=cue,
                    clean=ut,
                    biased=biased,
                    orig_layout=(
                        "front"
                        if bt.endswith(ut)
                        else ("pre" if bt.index("Answer choices:") > ut.index("Answer choices:") else "post")
                    ),
                )
            )
    with open(f"{OUT}/shared_train.jsonl", "w") as f:
        for x in recs:
            f.write(json.dumps(x) + "\n")
    with open(f"{OUT}/bct_src/control_non_cot.jsonl", "w") as fc, open(
        f"{OUT}/bct_src/bct_non_cot.jsonl", "w"
    ) as fb:
        for x in recs:
            fc.write(json.dumps({"messages": [{"role": "user", "content": x["clean"]}]}) + "\n")
            fb.write(json.dumps({"messages": [{"role": "user", "content": x["biased"]}]}) + "\n")
    inst = [l for l in open(f"{REPO}/datasets/sycophancy_bct/instruct_samples.jsonl") if l.strip()]
    rng = random.Random(0)
    sub = rng.sample(inst, len(recs))
    with open(f"{OUT}/bct_src/instruct_samples.jsonl", "w") as f:
        f.writelines(sub)
    lay = {k: sum(x["orig_layout"] == k for x in recs) for k in ("front", "pre", "post")}
    print(
        f"{len(recs)} questions ({sum(x['X'] == x['T'] for x in recs)} with a cue naming the correct answer); original layouts {lay}; "
        f"front rows reproduced exactly {n_front_ok}/{n_front}; multi-line cues {n_multiline}; instruct subsample {len(sub)}/{len(inst)}"
    )
    print(
        "sha256(shared_train.jsonl) =",
        hashlib.sha256(open(f"{OUT}/shared_train.jsonl", "rb").read()).hexdigest()[:16],
    )
    print("example:", json.dumps({k: recs[0][k] for k in ("ds", "row", "X", "T", "cue")}))


if __name__ == "__main__":
    main()
