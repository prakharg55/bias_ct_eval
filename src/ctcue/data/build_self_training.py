"""Build the training data of the self-training control of Chua et al. (2024).

The same rows, targets and instruction mix as the model's BCT data, with the cue removed from every
prompt.

Usage: python -m ctcue.data.build_self_training --model gemma2b
Output: $CT_ROOT/shared/bct_selftrain_<model>/
"""

import argparse, json, os, shutil
from ctcue.paths import CT_ROOT

SH = f"{CT_ROOT}/shared"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gemma2b", "llama8b"])
    a = ap.parse_args()
    sh = [json.loads(l) for l in open(f"{SH}/shared_train.jsonl")]
    bct = [json.loads(l) for l in open(f"{SH}/bct_{a.model}/bct_non_cot.jsonl")]
    assert len(sh) == len(bct) == 1213 and all(
        b["messages"][0]["content"] == s["biased"] for b, s in zip(bct, sh)
    ), "BCT pairs out of order"
    assert all(s["biased"] == s["cue"] + "\n" + s["clean"] for s in sh), "biased != cue + newline + clean"
    out_dir = f"{SH}/bct_selftrain_{a.model}"
    assert not os.path.exists(out_dir), out_dir
    os.makedirs(out_dir)
    rows = []
    for b, s in zip(bct, sh):
        r = json.loads(json.dumps(b))
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"], r["messages"]
        r["messages"][0]["content"] = s["clean"]
        rows.append(r)
    with open(f"{out_dir}/bct_non_cot.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    shutil.copyfile(f"{SH}/bct_{a.model}/instruct_samples.jsonl", f"{out_dir}/instruct_samples.jsonl")
    stats = {
        "n": len(rows),
        "user_turn": "shared clean prompt (cue removed)",
        "targets": "unchanged (own BCT targets)",
        "instruct": "copied from bct_" + a.model,
        "n_changed_user_turns": sum(
            r["messages"][0]["content"] != b["messages"][0]["content"] for r, b in zip(rows, bct)
        ),
        "n_changed_targets": sum(
            r["messages"][1]["content"] != b["messages"][1]["content"] for r, b in zip(rows, bct)
        ),
    }
    json.dump(stats, open(f"{out_dir}/build_stats.json", "w"), indent=1)
    print(stats)


if __name__ == "__main__":
    main()
