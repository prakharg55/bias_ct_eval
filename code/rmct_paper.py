#!/usr/bin/env python3
"""
RMCT PAPER-REGIME REPRODUCTION (28 Sep 2026; design decision after the RMCT fidelity audit, log section 82.17).
Runs the authors' RLTrainer (rmct_src = sohaibimran7/cot-transparency @ 4392d15, unmodified) on the local Tinker
stand-in (tinker_local4) with the PAPER's configuration, scripts/tinker_training/experiment_configs/
rlct_da_aw0_llama_r128b4_lr2_86e4.yaml (and its lr1e4 / lr5e4 siblings), whose values are asserted below:
  data     their train_rl.load_datapoints (verbatim, from the pinned mirror rmct_upstream_4392d15): distractor_argument_g4
           on logiqa + hellaswag, n_datapoints 64 (the first 32 lines of each file); prompt_style "cot" via their
           make_perturbation_fns; trait = their trait_classifier; answer_parser = their fallback_answer_parser (NO
           bare-letter fallback); renderer + tokenizer = their common.get_renderer_and_tokenizer (cookbook "llama3",
           no system block, stop <|eot_id|>).
  training RLConfig built exactly as their train_rl.main builds it from those args: LoRA rank 8 seed 42; lr (constant);
           kl_coef 0.05; anchor_weight 0.0; anchor_model base; loss ppo; 128 ref / 128 train / 128 consistency / 128 anchor
           rollouts; temperature 1.0; max_new_tokens 20480; 1 epoch; batch 4; grad-accum 1; refresh every step.
  service  tinker_local4 with LONGGEN (explicit decode loop, finished rows pruned), ATTN_IMPL sdpa, GRAD_CKPT, token-
           budgeted micro-batches: execution changes only (same distribution / loss / gradient; numerics differ).
Declared differences from the paper run: (1) the service is our local stand-in (audited: Tinker semantics); (2) the epoch
runs as --n_seg segments chained by their own resume_from / resume_with_optimizer (weights + AdamW state; sha256 and Adam
step checked at every boundary), data order = a seeded permutation cut into blocks (their loop shuffles within each
block), where the paper used one unseeded shuffle; (3) no midway (step 8) checkpoint (it was not evaluated in the paper).
Usage: rmct_paper.py --lr LR --out DIR --segment k (k = 0..n_seg-1 in order), then --finalize.
"""
import argparse, asyncio, hashlib, importlib, importlib.util, json, os, random, sys, time, types as pytypes
import numpy as np, torch, yaml
SRC = "/data/rmct_src"; LIB = "/data/pylibs/rmct"; UP = "/data/rmct_upstream_4392d15"
YAMLS = {"2.86e-4": "rlct_da_aw0_llama_r128b4_lr2_86e4.yaml", "1e-4": "rlct_da_aw0_llama_r128b4_lr1e4.yaml", "5e-4": "rlct_da_aw0_llama_r128b4_lr5e4.yaml"}
HFID = "meta-llama/Llama-3.1-8B-Instruct"; BATCH = 4
sys.path.insert(0, LIB); sys.path.insert(0, "/data/code")
from rmct_local4 import segments, lora_sha, adam_step, write_json_atomic                   # the audited segment machinery


def import_theirs():
    """Their modules, verbatim: rl_training / common (rmct_src), answer_parser + dataset.py + train_rl.py (pinned mirror).
    Stub package objects skip heavy package __init__s; inspect_ai (not installed, only used by dataset.py's eval loader, not
    by strip_cot_from_message or the training path) gets inert stub classes."""
    for name, path in [("cot_transparency", f"{SRC}/cot_transparency"), ("cot_transparency.apis", f"{SRC}/cot_transparency/apis"),
                       ("cot_transparency.apis.tinker", f"{SRC}/cot_transparency/apis/tinker"),
                       ("sycophancy_eval_inspect", f"{UP}/sycophancy_eval_inspect"), ("sycophancy_eval_inspect.mcq", f"{UP}/sycophancy_eval_inspect/mcq")]:
        m = pytypes.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m
    ia = pytypes.ModuleType("inspect_ai"); iad = pytypes.ModuleType("inspect_ai.dataset"); iam = pytypes.ModuleType("inspect_ai.model")
    class _Inert:                                                                            # never instantiated on the training path
        def __init__(self, *a, **k): raise RuntimeError("inspect_ai stub used")
    iad.Sample = iad.MemoryDataset = _Inert; iam.ChatMessageUser = iam.ChatMessageAssistant = _Inert
    sys.modules.update({"inspect_ai": ia, "inspect_ai.dataset": iad, "inspect_ai.model": iam})
    common = importlib.import_module("cot_transparency.apis.tinker.common")
    rl = importlib.import_module("cot_transparency.apis.tinker.rl_training")
    importlib.import_module("sycophancy_eval_inspect.mcq.answer_parser"); importlib.import_module("sycophancy_eval_inspect.mcq.dataset")
    spec = importlib.util.spec_from_file_location("their_train_rl", f"{UP}/scripts/tinker_training/train_rl.py")
    tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)
    assert open(f"{UP}/sycophancy_eval_inspect/mcq/answer_parser.py").read() == open(f"{SRC}/sycophancy_eval_inspect/mcq/answer_parser.py").read()
    return common, rl, tr


def paper_args(lr_str):
    lr = float(lr_str); y = yaml.safe_load(open(f"{UP}/scripts/tinker_training/experiment_configs/{YAMLS[lr_str]}"))
    a = y["training"]["args"]; assert y["model"] == HFID and y["training"]["seeds"] == [42]
    want = dict(bias_types="distractor_argument_g4", datasets="logiqa,hellaswag", n_datapoints=64, prompt_style="cot", lr=lr, lr_schedule="constant",
                lora_rank=8, kl_coef=0.05, anchor_weight=0.0, anchor_model="base", loss_fn="ppo", n_ref_rollouts=128, n_train_rollouts=128,
                n_consistency_rollouts=128, n_anchor_rollouts=128, temperature=1.0, max_new_tokens=20480, n_epochs=1, batch_size=4,
                gradient_accumulation_steps=1, refresh_every=1)
    for k, v in want.items(): assert a[k] == v, (k, a[k], v)
    return a


def main():
    ap_ = argparse.ArgumentParser(); ap_.add_argument("--lr", required=True, choices=sorted(YAMLS)); ap_.add_argument("--out", required=True)
    ap_.add_argument("--segment", type=int, default=None); ap_.add_argument("--finalize", action="store_true"); ap_.add_argument("--seg_size", type=int, default=16); ap_.add_argument("--seed", type=int, default=42, help="LoRA-init + data-permutation seed (paper: 42)")
    ap_.add_argument("--max_new_tokens", type=int, default=None, help="smoke only: override 20480"); ap_.add_argument("--rollouts", type=int, default=None, help="smoke only")
    ap_.add_argument("--n_datapoints", type=int, default=None, help="smoke only")
    a = ap_.parse_args(); assert (a.segment is not None) ^ a.finalize
    pa = paper_args(a.lr); seed = a.seed
    smoke = any(x is not None for x in (a.max_new_tokens, a.rollouts, a.n_datapoints))
    tag = f"llama8b_rmctpaper_lr{a.lr}_seed{seed}" + ("_SMOKE" if smoke else ""); out_dir = os.path.join(a.out, tag); rdir = os.path.join(out_dir, "resume"); os.makedirs(rdir, exist_ok=True)
    marker = lambda k: os.path.join(rdir, f"seg{k:02d}.done.json")
    common, rl, tr = import_theirs()
    dps = tr.load_datapoints(pa["bias_types"].split(","), pa["datasets"].split(","), a.n_datapoints or pa["n_datapoints"], __import__("pathlib").Path(f"{UP}/dataset_dumps/test"))
    assert len(dps) == (a.n_datapoints or 64), len(dps)
    for i, d in enumerate(dps): d["_pos"] = i
    segs = segments(len(dps), seed, a.seg_size); K = len(segs); final_dir = os.path.join(out_dir, f"rmct_shared_{tag}")

    if a.finalize:
        missing = [k for k in range(K) if not os.path.exists(marker(k))]
        if missing: print(f"[rmct_paper] finalize: segments {missing} not done", flush=True); return
        ms = [json.load(open(marker(k))) for k in range(K)]
        assert sorted(i for m in ms for i in m["datapoint_positions"]) == list(range(len(dps)))
        for k in range(1, K): assert ms[k]["lora_sha_start"] == ms[k - 1]["lora_sha_end"] and ms[k]["adam_step_start"] == ms[k - 1]["adam_step_end"], k
        if not os.path.exists(final_dir): os.replace(ms[-1]["sampler_path"], final_dir)
        from safetensors.torch import load_file
        sd = load_file(os.path.join(final_dir, "adapter_model.safetensors")); h = hashlib.sha256()
        for name in ms[-1]["param_names"]: h.update(name.encode()); h.update(sd[name.replace(".default", "")].float().numpy().tobytes())
        assert h.hexdigest() == ms[-1]["lora_sha_end"], "final adapter != the weights the last segment ended with"
        write_json_atomic(os.path.join(out_dir, "run_info.json"), {"tag": tag, "lr": a.lr, "paper_args": pa, "n_datapoints": len(dps), "n_segments": K,
                          "final": final_dir, "adam_steps": ms[-1]["adam_step_end"], "loop_steps": sum(m["n_steps"] for m in ms),
                          "seconds_train": sum(m["seconds"] for m in ms), "segments": [{k: v for k, v in m.items() if k not in ("param_names", "config")} for m in ms],
                          "config": ms[0]["config"]})
        print(f"[rmct_paper] finalized {final_dir}: {K} segments, {sum(m['n_steps'] for m in ms)} loop steps, {ms[-1]['adam_step_end']} Adam steps", flush=True)
        return

    k = a.segment; assert 0 <= k < K, (k, K)
    if os.path.exists(marker(k)): print(f"[rmct_paper] segment {k} already done", flush=True); return
    if k > 0 and not os.path.exists(marker(k - 1)): print(f"[rmct_paper] segment {k - 1} not done; segment {k} waits", flush=True); return
    prev = json.load(open(marker(k - 1))) if k > 0 else None
    seg_seed = seed * 100 + k; random.seed(seg_seed); np.random.seed(seg_seed); torch.manual_seed(seg_seed)
    import tinker
    from tinker_local4 import LocalServiceClient
    snap = f"{os.environ['HF_HOME']}/hub/models--{HFID.replace('/', '--')}/snapshots"; GP = [os.path.join(snap, x) for x in os.listdir(snap)]; assert len(GP) == 1; GP = GP[0]
    LocalServiceClient.MODEL_PATH, LocalServiceClient.OUT_DIR = GP, out_dir
    LocalServiceClient.LONGGEN, LocalServiceClient.ATTN_IMPL, LocalServiceClient.GRAD_CKPT = True, "sdpa", True
    LocalServiceClient.FB_CHUNK, LocalServiceClient.LP_CHUNK, LocalServiceClient.FB_TOKENS, LocalServiceClient.LP_TOKENS = 8, 4, 16384, 16384
    tinker.ServiceClient = LocalServiceClient                                               # the only service swap
    renderer, tok = common.get_renderer_and_tokenizer(HFID)                                 # THEIR renderer + tokenizer
    assert type(renderer).__name__ == "Llama3Renderer" and renderer.get_stop_sequences() == [128009], (type(renderer), renderer.get_stop_sequences())
    probe = renderer.build_generation_prompt(dps[0]["biased_question"]).to_ints()
    assert probe[0] == 128000 and 9125 not in probe[:6], "unexpected system block / BOS layout"
    run_name = f"{tag}_seg{k:02d}"; log_base = os.path.join(out_dir, "logs"); old = os.path.join(log_base, "rmct_paper", run_name)
    if os.path.exists(old): os.rename(old, f"{old}.killed{int(time.time())}")
    R = a.rollouts or 128
    cfg = rl.RLConfig(                                                                      # = their train_rl.main with the YAML args
        experiment_name="rmct_paper", run_name=run_name, model=HFID,
        lora=common.LoRAConfig(rank=pa["lora_rank"], seed=seed),
        optimizer=common.AdamConfig(learning_rate=pa["lr"], lr_schedule=pa["lr_schedule"]),
        reference_rate=rl.RateEstimationConfig(perturbation_indices=[0], n_rollouts=R if a.rollouts else pa["n_ref_rollouts"]),
        training=rl.TrainingSamplingConfig(perturbation_indices=[1], n_rollouts_for_rate=R if a.rollouts else pa["n_train_rollouts"],
                                           n_rollouts_for_consistency=R if a.rollouts else pa["n_consistency_rollouts"],
                                           n_rollouts_for_anchor=R if a.rollouts else pa["n_anchor_rollouts"]),
        loop=rl.TrainingLoopConfig(batch_size=pa["batch_size"], gradient_accumulation_steps=pa["gradient_accumulation_steps"],
                                   refresh_policy_every_n_steps=pa["refresh_every"], n_epochs=pa["n_epochs"]),
        generation=rl.GenerationConfig(max_new_tokens=a.max_new_tokens or pa["max_new_tokens"], temperature=pa["temperature"]),
        checkpoint=common.CheckpointConfig(save_every_n_steps=None, save_state=True),
        kl_coef=pa["kl_coef"], loss_fn=pa["loss_fn"], anchor_weight=pa["anchor_weight"], anchor_model=pa["anchor_model"], log_base_dir=log_base)
    assert cfg.loop.refresh_policy_every_n_steps == 1
    unb, bia = tr.make_perturbation_fns(pa["prompt_style"])
    seg = [dps[i] for i in segs[k]]
    print(f"[rmct_paper] {tag} segment {k + 1}/{K}: {len(seg)} datapoints, lr {a.lr}, rollouts {R}, max_new_tokens {cfg.generation.max_new_tokens}, "
          f"resume_from {prev['state_path'] if prev else None}", flush=True)
    t0 = time.time()
    trainer = rl.RLTrainer(config=cfg, resume_from=prev["state_path"] if prev else None, resume_with_optimizer=True); trainer.setup()
    core = LocalServiceClient.CORE; sha0, step0 = lora_sha(core), adam_step(core)
    if prev: assert sha0 == prev["lora_sha_end"] and step0 == prev["adam_step_end"] and core.param_names == prev["param_names"]
    else: assert step0 == 0
    torch.manual_seed(seg_seed); random.seed(seg_seed)
    final = asyncio.run(trainer.train(datapoints=seg, perturbation_fns=[unb, bia], trait_classifier=tr.trait_classifier,
                                      answer_parser=tr.fallback_answer_parser))
    state = final + "_state"; assert os.path.exists(os.path.join(state, "optimizer.pt")), state
    n_steps = (len(seg) + BATCH - 1) // BATCH; sha1, step1 = lora_sha(core), adam_step(core)
    assert step0 <= step1 <= step0 + n_steps, (step0, step1, n_steps)
    write_json_atomic(marker(k), {"segment": k, "n_segments": K, "seg_seed": seg_seed, "datapoint_positions": segs[k], "n_datapoints": len(seg),
                                  "n_steps": n_steps, "sampler_path": final, "state_path": state, "lora_sha_start": sha0, "lora_sha_end": sha1,
                                  "adam_step_start": step0, "adam_step_end": step1, "param_names": core.param_names, "shim_stats": core.stats,
                                  "seconds": time.time() - t0, "config": cfg.model_dump()})
    print(f"[rmct_paper] segment {k + 1}/{K} done in {time.time() - t0:.0f}s; Adam steps {step0}->{step1}; shim {core.stats}", flush=True)


if __name__ == "__main__":
    main()
