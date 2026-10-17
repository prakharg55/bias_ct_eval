"""Sample many long continuations of one prompt without storing per-step scores.

Same sampling distribution and returned fields as the sampler of local_tinker_hf.py, written as an
explicit decoding loop that shares the prompt's KV cache and drops finished rows. It supports Llama
only, because Gemma-2's sliding-window cache is not handled. Used by local_tinker_long.py.
"""

import torch, torch.nn.functional as F


@torch.no_grad()
def sample_shared_prefix(model, prompt_ids, n, max_new_tokens, temperature, stop_ids, prune_every=32):
    """Returns (tokens, logprobs, stop_reasons): per row the sampled ids (including the stop id if one was sampled),
    their float32 log-probs under softmax(logits / T), and "stop" | "length"."""
    assert temperature > 0 and n >= 1 and max_new_tokens >= 1 and len(prompt_ids) >= 2
    dev = next(model.parameters()).device
    ids = torch.tensor([list(prompt_ids)], device=dev)
    cache = model(input_ids=ids[:, :-1], use_cache=True, logits_to_keep=1).past_key_values
    cache.batch_repeat_interleave(n)
    assert cache.get_seq_length() == ids.shape[1] - 1
    # every row next feeds the last prompt token
    cur = ids[:, -1:].expand(n, 1).contiguous()
    # original row id of each live batch row
    rows = list(range(n))
    stop_t = torch.tensor(sorted(set(int(s) for s in stop_ids)), device=dev)
    toks = [[] for _ in range(n)]
    lps = [[] for _ in range(n)]
    done = [False] * n
    for t in range(max_new_tokens):
        logits = model(input_ids=cur, past_key_values=cache, use_cache=True).logits[:, -1, :].float()
        logp = F.log_softmax(logits / temperature, dim=-1)
        nxt = torch.multinomial(logp.exp(), 1)  # [m, 1]
        lp = logp.gather(1, nxt)[:, 0]
        is_stop = torch.isin(nxt[:, 0], stop_t)
        nx, lpl, st = nxt[:, 0].tolist(), lp.tolist(), is_stop.tolist()
        for j, r in enumerate(rows):
            # finished rows still in the batch: ignored
            if done[r]:
                continue
            toks[r].append(nx[j])
            lps[r].append(lpl[j])
            if st[j]:
                done[r] = True
        live = [not done[r] for r in rows]
        if not any(live):
            break
        if not all(live) and ((t + 1) % prune_every == 0):
            keep = torch.tensor([j for j, l in enumerate(live) if l], device=dev)
            cache.batch_select_indices(keep)
            nxt = nxt[keep]
            rows = [rows[j] for j in keep.tolist()]
        cur = nxt
    return toks, lps, ["stop" if done[r] else "length" for r in range(n)]
