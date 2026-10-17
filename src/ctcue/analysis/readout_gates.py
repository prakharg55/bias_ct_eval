"""Check that the first-position readout agrees with generation for every final checkpoint.

Test set: at least 97% agreement over all uncued and cued prompts. Gemma is checked with its
log-sum-exp readout and Llama with the max-logit readout. Cue battery: the battery-gate files, whose
readout must also equal the stored battery readout. The original Gemma model's battery gate comes
from check_gemma_rescore.py. Lists the Gemma checkpoints that fail the test-set gate and feed the
probe or patching analyses. These need the max-logit readout.

Usage: python -m ctcue.analysis.readout_gates
Output: $CT_ROOT/results/w7u/gates_u.json
"""

import glob, json, os, re

from ctcue.common import checkpoints
from ctcue.paths import CT_ROOT

R = f"{CT_ROOT}/results"
POS = ("front", "pre", "post", "end")


def primary(tag):
    """True if the eval tag is a checkpoint the primary (unmerged) tables use: untrained, fix variants, response-only BCT, self-training,
    and for the swept arms the tag checkpoints selects (an old-selection tag of a re-selected arm is superseded: appendix only).
    None if the arm's re-selection is unresolved."""
    m = re.match(r"(gemma2b|llama8b)_final(7u)?_([a-z]+)_s(4[234])$", tag)
    if not m:
        return True
    try:
        return checkpoints.eval_tag(m.group(1), m.group(3), int(m.group(4))) == tag
    except RuntimeError:
        return None


# gemma letter-prior / mechanism instruments read these probes
PROBE_USERS = ("final_bct", "final_act", "final_attct", "final_rmct", "final_rmcta", "fix_")
CAUSAL_USERS = ("final_bct", "final_act", "final_attct", "final_rmct")


def main():
    out = {"test": {}, "gen": {}}
    for f in sorted(glob.glob(f"{R}/pub7u/eval_*.json")):
        if f.endswith("_smoke.json"):
            continue
        # test_set stores the HF id
        d = json.load(open(f))
        t = d["tag"]
        mk = {"google/gemma-2-2b-it": "gemma2b", "meta-llama/Llama-3.1-8B-Instruct": "llama8b"}[d["model"]]
        assert (
            t.startswith(mk + "_")
            and d["merge"] == "unmerged"
            and d["max_new"] == 16
            and d["eval_rows"] == "1200:1300"
            and d["agg"] == ("sum" if mk == "gemma2b" else "max")
        ), t
        out["test"][t] = {
            "model": mk,
            "primary": primary(t),
            "readout": "v5 logsumexp (--agg sum)" if d["agg"] == "sum" else "v7.1 max over forms (--agg max)",
            "agree_first_gen": d["agree_first_gen"],
            "parsed": d["parsed"],
            "n_prompts": d["n_prompts"],
            "gate_pass": bool(d["agree_first_gen"] >= 0.97),
        }
        assert out["test"][t]["gate_pass"] == bool(d.get("gate_pass", out["test"][t]["gate_pass"])), (
            t,
            "gate_pass field disagrees",
        )
    for f in sorted(glob.glob(f"{R}/w7u/gengate_*.json")) + [f"{R}/w7/gengate_llama8b_none.json"]:
        if f.endswith("_smoke.json"):
            continue
        d = json.load(open(f))
        t = d["tag"]
        assert d["agg"] == "max", (t, d["agg"])  # the gen battery's v7.1 readout (both models)
        out["gen"][t] = {
            "model": d["model"],
            "agg": d["agg"],
            "uncued": d["uncued"]["agree"],
            "per_position": {p: d["per_position"][p]["agree"] for p in POS},
            "parsed_min": min(d["per_position"][p]["parsed"] for p in POS),
            "letter_mass_min": min(d["per_position"][p]["letter_mass_mean"] for p in POS),
            "readout_mismatch_vs_ref": d["readout_mismatch_vs_ref"],
            "n_items": d["n_items"],
            "gate_pass": bool(d["gate_pass"]) and d["readout_mismatch_vs_ref"] == 0,
            "source": os.path.relpath(f, R),
        }
    # untrained gemma: its gate file was run with the v5 readout (--agg sum); the gemma gen battery reads
    # v7.1, whose gate on the same greedy generations is check_gemma_rescore's gate_new (no adapter: unchanged
    # by the merge question)
    cg = json.load(open(f"{R}/w7/check_gemma_gen7.json"))["checkpoints"]["none"]
    out["gen"]["gemma2b_none"] = {
        "model": "gemma2b",
        "agg": "max (check_gemma_gen7 gate_new)",
        "uncued": cg["gate_new"]["uncued"],
        "per_position": {p: cg["gate_new"][p] for p in POS},
        "parsed_min": None,
        "letter_mass_min": None,
        "readout_mismatch_vs_ref": 0,
        "n_items": cg["gate_n_items"],
        "gate_pass": bool(cg["gate_pass_new"]),
        "source": "w7/check_gemma_gen7.json[none].gate_new",
    }
    out["test_failures"] = [t for t, r in out["test"].items() if not r["gate_pass"]]
    out["gen_failures"] = [t for t, r in out["gen"].items() if not r["gate_pass"]]
    # superseded (old-selection) checkpoints feed no table
    g = [
        t for t in out["test_failures"] if t.startswith("gemma2b_") and out["test"][t]["primary"] is not False
    ]
    out["gmax_needed"] = {
        "probe": [t for t in g if any(u in t for u in PROBE_USERS)],
        "causal": [t for t in g if any(u in t for u in CAUSAL_USERS)],
    }
    json.dump(out, open(f"{R}/w7u/gates_u.json", "w"), indent=1)
    for t, r in out["test"].items():
        print(
            f"test {t:34s} {r['readout']:32s} agree {r['agree_first_gen']:.4f} parsed {r['parsed']:.4f} {'PASS' if r['gate_pass'] else 'FAIL'}"
        )
    for t, r in out["gen"].items():
        print(
            f"gen  {t:34s} uncued {r['uncued']:.3f} "
            + " ".join(f"{p} {r['per_position'][p]:.3f}" for p in POS)
            + f" mismatch {r['readout_mismatch_vs_ref']} {'PASS' if r['gate_pass'] else 'FAIL'}"
        )
    print(
        f"{len(out['test'])} test-battery gates, failures: {out['test_failures']}; {len(out['gen'])} gen-battery gates, failures: {out['gen_failures']}; gmax needed: {out['gmax_needed']}"
    )


if __name__ == "__main__":
    main()
