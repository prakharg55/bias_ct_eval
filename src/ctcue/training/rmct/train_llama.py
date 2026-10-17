#!/usr/bin/env python3
"""
RMCT launcher v4 (27 Sep 2026, ~14:30 EDT) = rmct_local3.py + --model {gemma2b, llama8b} on tinker_local3: for gemma2b
exactly v3 (GemmaRenderer, fb_chunk 32, tag gemma2b_...); for llama8b the model's own HF chat template with its
generation-config stop ids (HFChatRenderer; the same template the other methods and every evaluation use) and
--fb_chunk (default 8 for llama8b: 32 datums x ~420 tokens of an 8B model do not fit one backward pass).
RMCT launcher v3 (27 Sep 2026, ~02:00 EDT) = rmct_local2.py + --anchor_weight (default 0.0 = v2 exactly). For the EXPLORATORY
arm RMCT-anchor (declared in PREREGISTERED_PREDICTIONS.md before any run): their RLTrainer's own anchor term with its code
default anchor_weight 0.5, anchor_model "base", n_rollouts_for_anchor 128 (their example_rlct.yaml); the tag gets "_aw<w>".
v2 text follows.
RMCT launcher v2 (27 Sep 2026). rmct_local.py's configuration and declared adaptations are UNCHANGED (their RLTrainer,
unmodified; their paper config; shared data; bare-letter parse fallback; max_new_tokens 16; gemma renderer). Two execution
changes, because v1 measured 12.3-13.2 s/step (~63 min for the 304-step epoch) and the cluster's GPU queue only starts jobs
under 1 h promptly:
  1. tinker_local2: the prompt is prefilled once per sampling call and its KV cache shared by the 128 rollouts.
  2. The epoch runs as consecutive SEGMENTS of their RLTrainer; LoRA weights + AdamW state pass from one segment to the next
     through their own resume_from / resume_with_optimizer. Data order: a seeded permutation of the datapoints, cut into
     blocks of --seg_size (a multiple of batch_size 4, the last block takes the remainder, so every step has 4 datapoints
     except the epoch's final one, as in one uninterrupted run); their loop shuffles within each block. One process per
     segment: a killed job loses at most the segment in progress. Segment k runs only after segment k-1's marker exists
     and checks on load that the LoRA weights (sha256) and the Adam step equal what segment k-1 recorded.
Usage: rmct_local2.py --seed S --lr LR --out DIR --segment k  (k = 0 .. n_seg-1, in order), then --finalize, which puts the
final adapter at DIR/gemma2b_rmct_seed{S}/rmct_shared_gemma2b_rmct_seed{S} and writes run_info.json there.
"""
import argparse, asyncio, hashlib, importlib, importlib.util, json, os, random, re, sys, time, types as pytypes
import numpy as np, torch
SRC = "/data/rmct_src"; LIB = "/data/pylibs/rmct"
SHARED = "/data/ct_root/shared/shared_train.jsonl"
BATCH = 4
MODEL_IDS = {"gemma2b": "google/gemma-2-2b-it", "llama8b": "meta-llama/Llama-3.1-8B-Instruct"}
sys.path.insert(0, LIB); sys.path.insert(0, "/data/code")


def _import_theirs():
    for name, path in [("cot_transparency", f"{SRC}/cot_transparency"), ("cot_transparency.apis", f"{SRC}/cot_transparency/apis"),
                       ("cot_transparency.apis.tinker", f"{SRC}/cot_transparency/apis/tinker")]:
        m = pytypes.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m     # skip their heavy package __init__s
    common = importlib.import_module("cot_transparency.apis.tinker.common")
    rl = importlib.import_module("cot_transparency.apis.tinker.rl_training")
    spec = importlib.util.spec_from_file_location("rmct_answer_parser", f"{SRC}/sycophancy_eval_inspect/mcq/answer_parser.py")
    ap = importlib.util.module_from_spec(spec); spec.loader.exec_module(ap)
    return common, rl, ap


BARE = re.compile(r"^\s*\**\(?([A-P])\)?\**[\s\.\)\*]*$")


def make_parse(ap):
    def parse(response):
        a = ap.fallback_answer_parser(response)                                        # theirs, verbatim
        if a is None:
            m = BARE.match(response); a = m.group(1) if m else None                 # declared: bare-letter answers
        return a
    return parse


def unbiased_perturbation(dp): return {"messages": dp["unbiased_question"]}             # their make_perturbation_fns("cot")
def biased_perturbation(dp): return {"messages": dp["biased_question"]}


def load_datapoints(n=None):
    recs = [json.loads(l) for l in open(SHARED)]
    dps = [dict(unbiased_question=[{"role": "user", "content": r["clean"]}], biased_question=[{"role": "user", "content": r["biased"]}],
                biased_option=r["X"], ground_truth=r["T"], shared_i=r["i"]) for r in recs]
    return dps[:n] if n else dps


def segments(n, seed, seg_size):
    assert seg_size % BATCH == 0 and seg_size > 0
    perm = np.random.RandomState(seed).permutation(n).tolist(); k = max(1, n // seg_size)
    return [perm[i * seg_size:(i + 1) * seg_size] for i in range(k - 1)] + [perm[(k - 1) * seg_size:]]


def lora_sha(core):
    h = hashlib.sha256()
    for name, p in zip(core.param_names, core.params): h.update(name.encode()); h.update(p.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def adam_step(core):
    if core.opt is None: return 0
    steps = {int(core.opt.state[p]["step"]) for p in core.params if p in core.opt.state}
    assert len(steps) == 1, steps
    return steps.pop()


def write_json_atomic(path, obj):
    tmp = path + ".tmp"; json.dump(obj, open(tmp, "w"), indent=1, default=str); os.replace(tmp, path)


def main():
    a_ = argparse.ArgumentParser(); a_.add_argument("--seed", type=int, required=True); a_.add_argument("--lr", type=float, required=True)
    a_.add_argument("--out", required=True); a_.add_argument("--segment", type=int, default=None); a_.add_argument("--finalize", action="store_true")
    a_.add_argument("--seg_size", type=int, default=200); a_.add_argument("--n_datapoints", type=int, default=None); a_.add_argument("--rollouts", type=int, default=128)
    a_.add_argument("--anchor_weight", type=float, default=0.0)
    a_.add_argument("--model", default="gemma2b", choices=["gemma2b", "llama8b"]); a_.add_argument("--fb_chunk", type=int, default=None)
    a = a_.parse_args(); assert (a.segment is not None) ^ a.finalize, "give exactly one of --segment k / --finalize"
    tag = f"{a.model}_rmct{f'_aw{a.anchor_weight:g}' if a.anchor_weight > 0 else ''}_seed{a.seed}"; out_dir = os.path.join(a.out, tag); rdir = os.path.join(out_dir, "resume"); os.makedirs(rdir, exist_ok=True)
    marker = lambda k: os.path.join(rdir, f"seg{k:02d}.done.json")
    dps = load_datapoints(a.n_datapoints); segs = segments(len(dps), a.seed, a.seg_size); K = len(segs)
    final_dir = os.path.join(out_dir, f"rmct_shared_{tag}")

    if a.finalize:
        missing = [k for k in range(K) if not os.path.exists(marker(k))]
        if missing: print(f"[rmct_local2] finalize: segments {missing} not done; nothing to do", flush=True); return
        ms = [json.load(open(marker(k))) for k in range(K)]
        assert sorted(i for m in ms for i in m["datapoint_positions"]) == list(range(len(dps))), "segments do not cover the epoch exactly once"
        for k in range(1, K): assert ms[k]["lora_sha_start"] == ms[k - 1]["lora_sha_end"] and ms[k]["adam_step_start"] == ms[k - 1]["adam_step_end"], k
        if not os.path.exists(final_dir): os.replace(ms[-1]["sampler_path"], final_dir)
        from safetensors.torch import load_file
        sd = load_file(os.path.join(final_dir, "adapter_model.safetensors")); h = hashlib.sha256()
        for name in ms[-1]["param_names"]: h.update(name.encode()); h.update(sd[name.replace(".default", "")].float().numpy().tobytes())
        assert h.hexdigest() == ms[-1]["lora_sha_end"], "final adapter != the weights the last segment ended with"
        import shutil                     # intermediate sampler copies (1.2 GB each, embedding included) duplicate the LoRA
        for m in ms[:-1]:                 # weights kept in each segment's *_state dir (itself a loadable PEFT adapter)
            p = m["sampler_path"]
            if os.path.isdir(p):
                assert os.path.dirname(p) == out_dir and os.path.basename(p) == f"rmct_shared_{tag}_seg{m['segment']:02d}", p
                assert os.path.exists(os.path.join(m["state_path"], "adapter_model.safetensors")), m["state_path"]
                shutil.rmtree(p)
        write_json_atomic(os.path.join(out_dir, "run_info.json"), {
            "tag": tag, "seed": a.seed, "lr": a.lr, "n_datapoints": len(dps), "rollouts": a.rollouts, "seg_size": a.seg_size, "n_segments": K,
            "final": final_dir, "adam_steps": ms[-1]["adam_step_end"], "loop_steps": sum(m["n_steps"] for m in ms),
            "seconds_train": sum(m["seconds"] for m in ms), "segments": [{k: v for k, v in m.items() if k not in ("param_names", "config")} for m in ms],
            "config": ms[0]["config"]})
        print(f"[rmct_local2] finalized {final_dir}: {K} segments, {sum(m['n_steps'] for m in ms)} loop steps, {ms[-1]['adam_step_end']} Adam steps, "
              f"{sum(m['seconds'] for m in ms):.0f}s training", flush=True)
        return

    k = a.segment; assert 0 <= k < K, (k, K)
    if os.path.exists(marker(k)): print(f"[rmct_local2] segment {k} already done", flush=True); return
    if k > 0 and not os.path.exists(marker(k - 1)): print(f"[rmct_local2] segment {k - 1} not done; segment {k} waits", flush=True); return
    prev = json.load(open(marker(k - 1))) if k > 0 else None
    seg_seed = a.seed * 100 + k; random.seed(seg_seed); np.random.seed(seg_seed); torch.manual_seed(seg_seed)
    common, rl, ap = _import_theirs(); parse = make_parse(ap)
    def trait_classifier(response, datapoint): return 1.0 if parse(response) == datapoint.get("biased_option", "") else 0.0
    import tinker
    from tinker_local3 import LocalServiceClient, GemmaRenderer, HFChatRenderer
    HFID = MODEL_IDS[a.model]; snap = f"{os.environ['HF_HOME']}/hub/models--{HFID.replace('/', '--')}/snapshots"
    GP = [os.path.join(snap, x) for x in os.listdir(snap)]; assert len(GP) == 1, GP; GP = GP[0]
    LocalServiceClient.MODEL_PATH, LocalServiceClient.OUT_DIR = GP, out_dir
    LocalServiceClient.FB_CHUNK = a.fb_chunk or (32 if a.model == "gemma2b" else 8)
    tinker.ServiceClient = LocalServiceClient                                           # the only service swap
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GP)
    if a.model == "gemma2b": renderer = GemmaRenderer(tok)
    else:
        eos = json.load(open(os.path.join(GP, "generation_config.json")))["eos_token_id"]; renderer = HFChatRenderer(tok, eos if isinstance(eos, list) else [eos])
    rl.get_renderer_and_tokenizer = lambda model: (renderer, tok)
    run_name = f"{tag}_seg{k:02d}"; log_base = os.path.join(out_dir, "logs")
    old = os.path.join(log_base, "rmct_shared", run_name)
    if os.path.exists(old): os.rename(old, f"{old}.killed{int(time.time())}")          # logs of an attempt killed mid-segment
    R = a.rollouts
    cfg = rl.RLConfig(
        experiment_name="rmct_shared", run_name=run_name, model=HFID,
        lora=common.LoRAConfig(rank=8, seed=a.seed),
        optimizer=common.AdamConfig(learning_rate=a.lr, lr_schedule="constant"),
        reference_rate=rl.RateEstimationConfig(perturbation_indices=[0], n_rollouts=R),
        training=rl.TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=R, n_rollouts_for_consistency=R, n_rollouts_for_anchor=R),
        loop=rl.TrainingLoopConfig(batch_size=BATCH, gradient_accumulation_steps=1, refresh_policy_every_n_steps=1, n_epochs=1),
        generation=rl.GenerationConfig(max_new_tokens=16, temperature=1.0),
        checkpoint=common.CheckpointConfig(save_every_n_steps=None, save_state=True),
        kl_coef=0.05, loss_fn="ppo", anchor_weight=a.anchor_weight, anchor_model="base", log_base_dir=log_base)
    assert cfg.loop.refresh_policy_every_n_steps == 1, "tinker_local's live-weights sampler equals Tinker's snapshot only with refresh every step"
    seg = [dps[i] for i in segs[k]]
    print(f"[rmct_local2] {tag} segment {k + 1}/{K}: {len(seg)} datapoints, rollouts {R}, lr {a.lr}, seg_seed {seg_seed}, "
          f"resume_from {prev['state_path'] if prev else None}", flush=True)
    t0 = time.time()
    trainer = rl.RLTrainer(config=cfg, resume_from=prev["state_path"] if prev else None, resume_with_optimizer=True); trainer.setup()
    core = LocalServiceClient.CORE; sha0, step0 = lora_sha(core), adam_step(core)
    if prev:
        assert sha0 == prev["lora_sha_end"], "resumed LoRA weights != segment k-1's final weights"
        assert step0 == prev["adam_step_end"], (step0, prev["adam_step_end"])
        assert core.param_names == prev["param_names"]
    else:
        assert step0 == 0
    torch.manual_seed(seg_seed); random.seed(seg_seed)                                   # sampling stream + their in-block shuffle
    final = asyncio.run(trainer.train(datapoints=seg, perturbation_fns=[unbiased_perturbation, biased_perturbation],
                                      trait_classifier=trait_classifier, answer_parser=parse))
    state = final + "_state"; assert os.path.exists(os.path.join(state, "optimizer.pt")), state
    n_steps = (len(seg) + BATCH - 1) // BATCH; sha1, step1 = lora_sha(core), adam_step(core)
    assert step0 <= step1 <= step0 + n_steps, (step0, step1, n_steps)
    write_json_atomic(marker(k), {
        "segment": k, "n_segments": K, "seg_seed": seg_seed, "datapoint_positions": segs[k], "shared_i": [d["shared_i"] for d in seg],
        "n_datapoints": len(seg), "n_steps": n_steps, "sampler_path": final, "state_path": state,
        "lora_sha_start": sha0, "lora_sha_end": sha1, "adam_step_start": step0, "adam_step_end": step1,
        "param_names": core.param_names, "shim_stats": core.stats, "seconds": time.time() - t0, "config": cfg.model_dump()})
    print(f"[rmct_local2] segment {k + 1}/{K} done in {time.time() - t0:.0f}s; Adam steps {step0}->{step1} of {n_steps} loop steps; shim {core.stats}", flush=True)


if __name__ == "__main__":
    main()
