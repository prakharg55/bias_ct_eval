#!/usr/bin/env python3
"""
v3 (27 Sep 2026) = tinker_local2.py + (a) class-level FB_CHUNK / LP_CHUNK (defaults 32 / 4 = v2) so an 8B model can use
smaller backward chunks, and (b) HFChatRenderer: the model's own HF chat template + its generation-config stop ids
(GemmaRenderer is HFChatRenderer with gemma's stop ids). Nothing else changed.
LOCAL STAND-IN FOR THE TINKER SERVICE (26 Sep 2026). Lets the RMCT authors' trainer (sohaibimran7/cot-transparency,
cot_transparency/apis/tinker/rl_training.py @ 4392d15, UNMODIFIED) run on gemma-2-2b-it, which Tinker does not serve.
Only the remote service is replaced: their loop, reward, batch-level advantage normalisation, and tinker_cookbook 0.4.1's
trajectory_to_data / incorporate_kl_penalty / _remove_mask all run as published. Service semantics reproduced here:
  * LoRA (create_lora_training_client): rank r on attention (q,k,v,o_proj) if train_attn, MLP (gate,up,down_proj) if
    train_mlp, unembedding (lm_head) if train_unembed; lora_alpha 32 (Tinker's PEFT export uses alpha 32, scaling 32/r);
    dropout 0; standard PEFT init (B = 0, A kaiming-uniform) under torch.manual_seed(seed).
  * sample_async: temperature T, top_k -1 / top_p 1 (off: SamplingParams defaults), max_tokens, stop token ids; returns the
    sampled tokens INCLUDING the stop token and each token's log-prob under the sampling distribution (float32 log_softmax
    of the processed logits), stop_reason "stop" | "length".
  * compute_logprobs_async(seq): [None] + log p(seq[i] | seq[<i]) at EVERY position under that client's weights (full
    sequence, exact); concurrent calls are batched by sequence length (no padding).
  * forward_backward_async(loss_fn="ppo"): Tinker's reference implementation (docs, losses/ppo):
        prob_ratio = exp(target_logprobs - sampling_logprobs); clipped = clamp(prob_ratio, 0.8, 1.2)
        loss = -(min(prob_ratio * A, clipped * A)).sum()   summed over tokens AND datums (no averaging).
    Prompt positions carry A = 0 (cookbook datum construction), so they contribute exactly 0 to loss and gradient;
    log-probs are therefore computed only over each datum's trailing window that contains every A != 0 position
    (asserted), which is exact. Datums that are EXACTLY identical (tokens, targets, sampling log-probs, advantages) are
    computed once and weighted by their count, which leaves the summed loss and its gradient unchanged; the same exact
    de-duplication is applied to compute_logprobs (identical sequences have identical log-probs).
  * optim_step_async(AdamParams): decoupled-weight-decay Adam with (lr, beta1, beta2, eps, weight_decay); global grad-norm
    clipping only if grad_clip_norm > 0 (SDK default 0.0; their RL loop does not set it).
  * save_weights_and_get_sampling_client: a sampler bound to the LIVE weights. Equivalent to Tinker's snapshot here because
    their loop re-creates the sampler after every optimizer step (refresh_policy_every_n_steps = 1) and cancels any
    prefetched sampling first; asserted at construction by the launcher (rmct_local.py).
v2 (27 Sep 2026; tinker_local.py is unchanged and still what the v1 launcher uses):
  * sample_async computes the prompt ONCE and shares its KV cache across the num_samples continuations (HF generate
    expanded the prompt to num_samples copies and prefilled every copy: 128 x ~350 tokens per call, ~2/3 of each RMCT
    step). Same sampling distribution (temperature, no top-k/top-p, stop ids) and the same returned fields; only bf16
    batch-shape numerics differ. Checked by test_tinker_local2.py.
  * load_state / load_state_with_optimizer (their RLTrainer's resume_from path) restore the LoRA weights (+ AdamW state)
    written by save_state_async, which now omits the frozen embedding copy PEFT adds when lm_head is a target.
"""
import asyncio, contextlib, os, time
from dataclasses import dataclass
import numpy as np, torch, torch.nn.functional as F
from tinker import types
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]; MLP = ["gate_proj", "up_proj", "down_proj"]; UNEMBED = ["lm_head"]


class _Done:
    """Resolved future: Tinker calls return futures exposing .result() and async .result_async()."""
    def __init__(self, value): self.value = value
    def result(self, timeout=None): return self.value
    async def result_async(self, timeout=None): return self.value


@dataclass
class _Path:
    path: str


class _Core:
    """Owns the single PEFT model (base weights + LoRA) shared by the training client and both sampling clients."""
    def __init__(self, model_path, rank, seed, train_mlp, train_attn, train_unembed, out_dir, fb_chunk=32, lp_chunk=4, dtype=torch.bfloat16):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        base = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype, attn_implementation="eager").cuda()   # bf16 in runs; float32 in the exactness test
        targets = (ATTN if train_attn else []) + (MLP if train_mlp else []) + (UNEMBED if train_unembed else [])
        assert targets, "no LoRA targets"
        if seed is not None: torch.manual_seed(seed)
        self.model = get_peft_model(base, LoraConfig(r=rank, lora_alpha=32, lora_dropout=0.0, target_modules=targets, bias="none",
                                                    init_lora_weights=True, task_type="CAUSAL_LM"))
        self.params = [p for p in self.model.parameters() if p.requires_grad]
        self.param_names = [n for n, p in self.model.named_parameters() if p.requires_grad]; assert len(self.param_names) == len(self.params)
        assert all(float(m.lora_B["default"].weight.abs().max()) == 0.0 for m in self.model.modules() if hasattr(m, "lora_B") and "default" in m.lora_B)
        self.model.print_trainable_parameters(); self.targets = targets
        self.out_dir = out_dir; os.makedirs(out_dir, exist_ok=True); self.fb_chunk = fb_chunk; self.lp_chunk = lp_chunk; self.opt = None
        self.stats = {"fb_calls": 0, "fb_datums": 0, "sample_calls": 0, "lp_calls": 0, "t_sample": 0.0, "t_fb": 0.0, "t_lp": 0.0}

    def ctx(self, use_adapter):
        return contextlib.nullcontext() if use_adapter else self.model.disable_adapter()


class LocalSamplingClient:
    def __init__(self, core: _Core, use_adapter: bool):
        self.core, self.use_adapter = core, use_adapter; self._pending = []; self._flush_task = None

    async def sample_async(self, prompt, sampling_params, num_samples, **kw):
        c = self.core; t0 = time.time(); c.stats["sample_calls"] += 1
        stop = sampling_params.stop or []
        assert all(isinstance(s, int) for s in stop), "renderer must give stop TOKEN ids"
        assert (sampling_params.top_k in (-1, 0, None)) and sampling_params.top_p in (1, 1.0, None)
        ids = torch.tensor([prompt.to_ints()], device="cuda"); n, L = int(num_samples), ids.shape[1]
        assert L >= 2
        with torch.no_grad(), c.ctx(self.use_adapter):
            c.model.eval()
            # prompt[:-1] prefilled ONCE (batch 1); its cache is repeated to n rows; generate then runs the last prompt token
            # and the continuations for all n rows exactly as before (logits processors, stopping, scores unchanged)
            cache = c.model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
            cache.batch_repeat_interleave(n); assert cache.get_seq_length() == L - 1
            idn = ids.expand(n, L).contiguous()
            # gemma-2's generation_config says cache_implementation="hybrid"; transformers 4.57 already drops that for the
            # model's own config (dynamic sliding-window cache, which v1 therefore used too). Passed as None explicitly
            # because generate rejects a supplied cache whenever cache_implementation is set.
            out = c.model.generate(input_ids=idn, attention_mask=torch.ones_like(idn), past_key_values=cache, cache_implementation=None, do_sample=True,
                                   temperature=float(sampling_params.temperature), top_k=0, top_p=1.0, max_new_tokens=int(sampling_params.max_tokens),
                                   eos_token_id=list(stop), pad_token_id=c.tok.pad_token_id, output_scores=True, return_dict_in_generate=True)
            del cache
        assert out.sequences.shape[0] == n and torch.equal(out.sequences[:, :L], idn)
        gen = out.sequences[:, L:]                                                    # [n, T]
        assert gen.shape[1] == len(out.scores), (gen.shape, len(out.scores))
        tok_lp = torch.stack([F.log_softmax(sc.float(), -1).gather(1, gen[:, t:t + 1])[:, 0] for t, sc in enumerate(out.scores)], 1).cpu()
        seqs = []
        for i in range(gen.shape[0]):
            g = gen[i].tolist(); cut = next((j for j, t in enumerate(g) if t in stop), None)
            n = len(g) if cut is None else cut + 1                                  # keep the stop token
            seqs.append(types.SampledSequence(stop_reason="length" if cut is None else "stop", tokens_np=np.array(g[:n], dtype=np.int64),
                                              logprobs_np=tok_lp[i, :n].numpy().astype(np.float64)))
        c.stats["t_sample"] += time.time() - t0
        return types.SampleResponse(sequences=seqs)

    async def compute_logprobs_async(self, model_input, **kw):
        """Batched: concurrent calls (asyncio.gather) are collected and computed together."""
        fut = asyncio.get_running_loop().create_future(); self._pending.append((model_input.to_ints(), fut))
        if self._flush_task is None:
            self._flush_task = asyncio.ensure_future(self._flush())
        return await fut

    async def _flush(self):
        await asyncio.sleep(0)                                                        # let every concurrent caller enqueue
        pending, self._pending, self._flush_task = self._pending, [], None
        c = self.core; t0 = time.time(); c.stats["lp_calls"] += len(pending)
        try:
            # exact full-sequence log-probs, computed per group of identical length (no padding), last W positions only
            first = {}; dup_of = []
            for k, (seq, fut) in enumerate(pending):
                t = tuple(seq); dup_of.append(first.setdefault(t, k))
            by_len = {}
            for k in sorted(set(dup_of)): by_len.setdefault(len(pending[k][0]), []).append(k)
            results = [None] * len(pending); c.stats["lp_unique"] = c.stats.get("lp_unique", 0) + len(set(dup_of))
            with torch.no_grad(), c.ctx(self.use_adapter):
                c.model.eval()
                for L, ks in by_len.items():
                    for a in range(0, len(ks), c.lp_chunk):                         # full [b, L, V] logits: keep b small
                        kk = ks[a:a + c.lp_chunk]; x = torch.tensor([pending[k][0] for k in kk], device="cuda")
                        logits = c.model(input_ids=x, use_cache=False).logits.float()   # [b, L, V]
                        lp = F.log_softmax(logits[:, :-1], -1).gather(2, x[:, 1:, None])[..., 0].cpu()
                        for j, k in enumerate(kk): results[k] = [None] + lp[j].tolist()
            for k, (seq, fut) in enumerate(pending): fut.set_result(list(results[dup_of[k]]))
        except Exception as e:
            for seq, fut in pending:
                if not fut.done(): fut.set_exception(e)
        c.stats["t_lp"] += time.time() - t0


class LocalTrainingClient:
    def __init__(self, core: _Core):
        self.core = core

    # --- sampling clients ---
    def save_weights_and_get_sampling_client(self, name=None, **kw):
        return LocalSamplingClient(self.core, use_adapter=True)

    async def save_weights_and_get_sampling_client_async(self, name=None, **kw):
        return LocalSamplingClient(self.core, use_adapter=True)

    # --- training ---
    async def forward_backward_async(self, data, loss_fn="ppo", loss_fn_config=None, **kw):
        assert loss_fn == "ppo", f"only Tinker's 'ppo' loss is reproduced, got {loss_fn}"
        lo = (loss_fn_config or {}).get("clip_low_threshold", 0.8); hi = (loss_fn_config or {}).get("clip_high_threshold", 1.2)
        c = self.core; t0 = time.time(); c.stats["fb_calls"] += 1; c.stats["fb_datums"] += len(data); c.model.train()
        items = []; uniq = {}; umap = []
        for d in data:
            x = d.model_input.to_ints(); li = d.loss_fn_inputs
            assert set(li) >= {"target_tokens", "logprobs", "advantages"} and "mask" not in li, set(li)
            tgt = li["target_tokens"].to_torch().long(); samp = li["logprobs"].to_torch().float(); adv = li["advantages"].to_torch().float()
            assert len(x) == len(tgt) == len(samp) == len(adv)
            nz = torch.nonzero(adv != 0)
            w = (len(x) - int(nz[0])) if len(nz) else 1                               # trailing window holding every A != 0
            key = (tuple(x), tuple(tgt.tolist()), samp.numpy().tobytes(), adv.numpy().tobytes())   # EXACT identity
            if key not in uniq: uniq[key] = len(items); items.append([x, tgt, samp, adv, w, 0])
            items[uniq[key]][5] += 1; umap.append(uniq[key])
        by_key = {}
        for k, it in enumerate(items): by_key.setdefault((len(it[0]), it[4]), []).append(k)
        out_lp = [None] * len(items); total = 0.0; c.stats["fb_unique"] = c.stats.get("fb_unique", 0) + len(items)
        for (L, W), ks in by_key.items():
            for a in range(0, len(ks), c.fb_chunk):
                kk = ks[a:a + c.fb_chunk]
                x = torch.tensor([items[k][0] for k in kk], device="cuda")
                logits = c.model(input_ids=x, use_cache=False, logits_to_keep=W).logits.float()      # [b, W, V]
                tgt = torch.stack([items[k][1][-W:] for k in kk]).cuda(); samp = torch.stack([items[k][2][-W:] for k in kk]).cuda()
                adv = torch.stack([items[k][3][-W:] for k in kk]).cuda()
                tlp = F.log_softmax(logits, -1).gather(2, tgt[..., None])[..., 0]                  # target_logprobs
                ratio = torch.exp(tlp - samp); clipped = torch.clamp(ratio, lo, hi)
                mult = torch.tensor([float(items[k][5]) for k in kk], device="cuda")[:, None]   # multiplicity of identical datums
                loss = -(mult * torch.min(ratio * adv, clipped * adv)).sum()
                loss.backward(); total += float(loss.detach())
                for j, k in enumerate(kk):
                    full = torch.zeros(len(items[k][0])); full[-W:] = tlp[j].detach().float().cpu(); out_lp[k] = full
        c.stats["t_fb"] += time.time() - t0
        return _Done(types.ForwardBackwardOutput(loss_fn_output_type="ppo", metrics={"loss:sum": total},
                                                 loss_fn_outputs=[{"logprobs": types.TensorData.from_torch(out_lp[u].clone())} for u in umap]))

    async def optim_step_async(self, adam_params, **kw):
        c = self.core
        if c.opt is None:
            c.opt = torch.optim.AdamW(c.params, lr=adam_params.learning_rate, betas=(adam_params.beta1, adam_params.beta2),
                                      eps=adam_params.eps, weight_decay=adam_params.weight_decay)
        for g in c.opt.param_groups:
            g["lr"] = adam_params.learning_rate; g["betas"] = (adam_params.beta1, adam_params.beta2); g["eps"] = adam_params.eps
            g["weight_decay"] = adam_params.weight_decay
        if getattr(adam_params, "grad_clip_norm", 0.0) and adam_params.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(c.params, adam_params.grad_clip_norm)
        c.opt.step(); c.opt.zero_grad(set_to_none=True)
        return _Done({"ok": True})

    # --- checkpoints (cookbook save_checkpoint_async calls these) ---
    async def save_weights_for_sampler_async(self, name, ttl_seconds=None, **kw):
        p = os.path.join(self.core.out_dir, name); self.core.model.save_pretrained(p); return _Done(_Path(p))

    async def save_state_async(self, name, ttl_seconds=None, **kw):
        p = os.path.join(self.core.out_dir, name + "_state"); self.core.model.save_pretrained(p, save_embedding_layers=False)
        if self.core.opt is not None:
            torch.save({"optimizer": self.core.opt.state_dict(), "param_names": self.core.param_names}, os.path.join(p, "optimizer.pt"))
        return _Done(_Path(p))

    # --- resume (their RLTrainer.setup: resume_from + resume_with_optimizer) ---
    def _load(self, path, with_optimizer):
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        c = self.core; sd = load_file(os.path.join(path, "adapter_model.safetensors"))
        lora = {k: v for k, v in sd.items() if "lora_" in k}
        assert lora and len(lora) == len(sd), f"state must hold exactly the LoRA tensors: {sorted(set(sd) - set(lora))[:5]}"
        n_trainable = sum(1 for p in c.model.parameters() if p.requires_grad)
        assert len(lora) == n_trainable, (len(lora), n_trainable)
        res = set_peft_model_state_dict(c.model, lora)
        assert not res.unexpected_keys, res.unexpected_keys[:5]
        missing = [k for k in res.missing_keys if "lora_" in k]; assert not missing, missing[:5]
        own = {k.replace(".default", ""): v for k, v in c.model.state_dict().items() if "lora_" in k}
        for k, v in lora.items(): assert torch.equal(own[k].cpu(), v.to(own[k].dtype)), f"LoRA tensor {k} not restored exactly"
        if with_optimizer:
            blob = torch.load(os.path.join(path, "optimizer.pt"), map_location="cpu", weights_only=False)   # load_state_dict moves moments to the params' device; "step" stays on CPU as saved
            assert blob["param_names"] == c.param_names, "optimizer state saved for a different parameter order"
            c.opt = torch.optim.AdamW(c.params); c.opt.load_state_dict(blob["optimizer"])
            assert len(c.opt.state) == len(c.params), (len(c.opt.state), len(c.params))
            for p in c.params:
                st = c.opt.state[p]; assert st["exp_avg"].shape == p.shape and st["exp_avg_sq"].shape == p.shape
        print(f"[tinker_local2] restored {len(lora)} LoRA tensors{' + AdamW state' if with_optimizer else ''} from {path}", flush=True)
        return _Done({"ok": True})

    def load_state(self, path, **kw): return self._load(path, with_optimizer=False)
    def load_state_with_optimizer(self, path, **kw): return self._load(path, with_optimizer=True)
    async def load_state_async(self, path, **kw): return self._load(path, with_optimizer=False)
    async def load_state_with_optimizer_async(self, path, **kw): return self._load(path, with_optimizer=True)


class LocalServiceClient:
    """tinker.ServiceClient stand-in; one base model per process (the launcher passes the local snapshot path)."""
    MODEL_PATH = None; OUT_DIR = None; CORE = None; DTYPE = torch.bfloat16; FB_CHUNK = 32; LP_CHUNK = 4

    def __init__(self, *a, **kw):
        assert LocalServiceClient.MODEL_PATH and LocalServiceClient.OUT_DIR, "launcher must set MODEL_PATH/OUT_DIR"

    def create_lora_training_client(self, base_model, rank=32, seed=None, train_mlp=True, train_attn=True, train_unembed=True, **kw):
        assert LocalServiceClient.CORE is None, "one training client per process"
        LocalServiceClient.CORE = _Core(LocalServiceClient.MODEL_PATH, rank, seed, train_mlp, train_attn, train_unembed, LocalServiceClient.OUT_DIR,
                                        fb_chunk=LocalServiceClient.FB_CHUNK, lp_chunk=LocalServiceClient.LP_CHUNK, dtype=LocalServiceClient.DTYPE)
        print(f"[tinker_local] LoRA rank {rank} alpha 32 on {LocalServiceClient.CORE.targets} (seed {seed})", flush=True)
        return LocalTrainingClient(LocalServiceClient.CORE)

    def create_sampling_client(self, base_model=None, model_path=None, **kw):
        assert LocalServiceClient.CORE is not None and model_path is None
        return LocalSamplingClient(LocalServiceClient.CORE, use_adapter=False)            # frozen base model


class GemmaRenderer:
    """Minimal renderer for gemma-2 chat (tinker_cookbook has none): same template the eval scripts use."""
    def __init__(self, tok):
        self.tok = tok; self.stop_ids = [tok.convert_tokens_to_ids("<end_of_turn>"), tok.eos_token_id]
        assert all(isinstance(t, int) and t >= 0 for t in self.stop_ids), self.stop_ids

    def build_generation_prompt(self, messages):
        return types.ModelInput.from_ints(tokens=self.tok.apply_chat_template(messages, add_generation_prompt=True))

    def get_stop_sequences(self):
        return list(self.stop_ids)

    def parse_response(self, tokens):
        cut = next((j for j, t in enumerate(tokens) if t in self.stop_ids), None)
        text = self.tok.decode(tokens if cut is None else tokens[:cut], skip_special_tokens=False)
        return {"role": "assistant", "content": text}, cut is not None


class HFChatRenderer(GemmaRenderer):
    """Any HF chat model: the tokenizer's own chat template (as the evaluation uses) and the stop ids given (for Llama-3.1:
    its generation-config eos ids <|end_of_text|>, <|eom_id|>, <|eot_id|>)."""
    def __init__(self, tok, stop_ids):
        self.tok = tok; self.stop_ids = [int(t) for t in stop_ids]
        assert self.stop_ids and all(t >= 0 for t in self.stop_ids), self.stop_ids
