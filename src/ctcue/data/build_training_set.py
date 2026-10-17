#!/usr/bin/env python3
"""
SHARED TRAINING SET for the authors'-code comparison (26 Sep 2026; design decision: identical train/test data for every
method, only the training method differs). CPU only, seconds.
Pool: suggested_answer rows 100:500 of mmlu/hellaswag/logiqa/truthfulqa (Chua et al. test dumps, shared loader), the same
questions grid2 trained on; cue-less pairs (the loader's deletion bug) dropped; NO model-knew filter and cues pointing to
the correct answer KEPT (Chua and the AttCT repo randomise the cued letter; each method treats them per its algorithm).
Cue placed as a PREFIX ("<cue>\n<question>"), the layout the dumps already use for 425 of these rows and the one layout all
four codebases support (AttCT's content window needs the clean text contiguous in the cued prompt).
Outputs (OUT):
  shared_train.jsonl         one record per question: clean / biased (prefix) user text, cue, X, T, ds, row, hash
  bct_src/control_non_cot.jsonl, bct_src/bct_non_cot.jsonl   row-aligned inputs for the AttCT repo's own BCT-target
                             generator (scripts/generate_fresh_bct_data.py --use-vllm)
  bct_src/instruct_samples.jsonl  1:1 instruction mix (seeded subsample of the AttCT repo's instruct file, as many
                             samples as BCT pairs; Chua et al. and the RMCT paper both mix 1:1; their loader adds the whole file)
"""
import difflib, json, os, random, sys, hashlib
sys.path.insert(0, "/data/code")
from utils import load_bct_source
DUMPS = "/data/dataset_dumps/test"; OUT = "/data/ct_root/shared"
REPO = "/data/AttCT"; DATASETS = ["mmlu", "hellaswag", "logiqa", "truthfulqa"]; ROWS = (100, 500)


def main():
    os.makedirs(f"{OUT}/bct_src", exist_ok=True); recs = []; n_front_ok = n_front = n_multiline = 0; seen_clean = set()
    for ds in DATASETS:
        pairs, _ = load_bct_source(f"{DUMPS}/suggested_answer/{ds}_suggested_answer.jsonl")
        for r, p in enumerate(pairs[ROWS[0]:ROWS[1]], start=ROWS[0]):
            b, u = p["biased_messages"], p["unbiased_messages"]
            if b == u: continue                                         # cue deleted by the loader: no cue to train on
            assert len(b) == len(u) == 1 and b[0]["role"] == u[0]["role"] == "user"
            bt, ut = b[0]["content"], u[0]["content"]
            ops = [o for o in difflib.SequenceMatcher(None, ut, bt, autojunk=False).get_opcodes() if o[0] != "equal"]
            assert len(ops) == 1 and ops[0][0] == "insert", (ds, r, ops)   # exactly one inserted cue
            cue = bt[ops[0][3]:ops[0][4]].strip(); n_multiline += "\n" in cue
            biased = cue + "\n" + ut
            if bt.endswith(ut):                                          # already front layout: must be reproduced exactly
                n_front += 1; assert biased == bt, (ds, r, repr(bt[:80])); n_front_ok += 1
            assert ut not in seen_clean, (ds, r, "duplicate clean prompt"); seen_clean.add(ut)
            assert biased.count(ut) == 1 and biased.endswith(ut)
            recs.append(dict(i=len(recs), ds=ds, row=r, hash=p["original_question_hash"], X=p["bias_target"], T=p["correct_letter"],
                             cue=cue, clean=ut, biased=biased, orig_layout="front" if bt.endswith(ut) else ("pre" if bt.index("Answer choices:") > ut.index("Answer choices:") else "post")))
    with open(f"{OUT}/shared_train.jsonl", "w") as f:
        for x in recs: f.write(json.dumps(x) + "\n")
    with open(f"{OUT}/bct_src/control_non_cot.jsonl", "w") as fc, open(f"{OUT}/bct_src/bct_non_cot.jsonl", "w") as fb:
        for x in recs:
            fc.write(json.dumps({"messages": [{"role": "user", "content": x["clean"]}]}) + "\n")
            fb.write(json.dumps({"messages": [{"role": "user", "content": x["biased"]}]}) + "\n")
    inst = [l for l in open(f"{REPO}/datasets/sycophancy_bct/instruct_samples.jsonl") if l.strip()]
    rng = random.Random(0); sub = rng.sample(inst, len(recs))
    with open(f"{OUT}/bct_src/instruct_samples.jsonl", "w") as f: f.writelines(sub)
    lay = {k: sum(x["orig_layout"] == k for x in recs) for k in ("front", "pre", "post")}
    print(f"{len(recs)} questions ({sum(x['X'] == x['T'] for x in recs)} with a cue naming the correct answer); original layouts {lay}; "
          f"front rows reproduced exactly {n_front_ok}/{n_front}; multi-line cues {n_multiline}; instruct subsample {len(sub)}/{len(inst)}")
    print("sha256(shared_train.jsonl) =", hashlib.sha256(open(f"{OUT}/shared_train.jsonl", "rb").read()).hexdigest()[:16])
    print("example:", json.dumps({k: recs[0][k] for k in ("ds", "row", "X", "T", "cue")}))


if __name__ == "__main__":
    main()
