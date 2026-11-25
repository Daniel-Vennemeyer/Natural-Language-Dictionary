"""Packed concept scoring: many descriptions against one encoded text prefix in a single forward.

A concept score reads the judge's digit distribution after the prompt

    [instruction + text]  +  [description + rating question]
         prefix                      block

The prefix does not depend on the description, so it is encoded once per text. All blocks of a
group are then packed into ONE suffix sequence with a block-diagonal attention mask: a token
in block k attends to the prefix and causally to block k, never to other blocks, and each
block's positions restart at the prefix length. Every block is therefore computed exactly as
if it were the only suffix, but K descriptions cost one forward instead of K, and only the
digit rows of the output layer are evaluated.

Soft-prompt gradients use a two-pass gradient cache: pass 1 scores every (text, description)
pair without gradient; the loss is backpropagated to the score matrix only; pass 2 recomputes
one (text chunk, description group) unit at a time with gradient and backpropagates its slice
of dLoss/dScores. Peak memory is one unit, whatever the batch size or K. The encoded prefixes
from pass 1 are reused in pass 2, and can optionally be kept in host memory across steps
(texts recur across training steps).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import torch

BlockFn = Callable[[int], torch.Tensor]          # description index -> block embeddings [L_k, E]


@dataclass
class Prefix:
    kv: list                                     # per layer (keys, values), each [B, heads, P, head_dim]
    mask: torch.Tensor                           # [B, P], 1 = real token (right padded)
    real_len: torch.Tensor                       # [B]


class HostPrefixCache:
    """Encoded text prefixes kept in CPU memory across calls, keyed by their token ids. Inserts
    stop at max_gb; texts beyond that are re-encoded. Qwen3-4B needs about 147 KB per token."""

    def __init__(self, max_gb: float):
        self.max_bytes, self.bytes, self.store = int(max_gb * 2**30), 0, {}

    def get(self, key):
        return self.store.get(key)

    def put(self, key, t: torch.Tensor):
        nb = t.numel() * t.element_size()
        if key in self.store or self.bytes + nb > self.max_bytes:
            return
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=torch.cuda.is_available())
        host.copy_(t)
        self.store[key] = host
        self.bytes += nb


class PackedScorer:
    """Scores blocks of description embeddings against text prefixes for a FrozenJudge-style model.

    digit_groups[r] lists the token ids whose logits are pooled (logsumexp) into rating r; a token
    may belong to several groups. The concept score is E[r] / 9 under softmax(pooled / temp)."""

    def __init__(self, model, tokenizer, device, digit_groups, prefix_ids_fn,
                 text_batch: int = 32, token_budget: int = 32768, grad_token_budget: int = 4096,
                 prefix_cache_gb: float = 0.0):
        self.model, self.tok, self.device = model, tokenizer, torch.device(device)
        self.base = model.get_decoder() if hasattr(model, "get_decoder") else model.model
        self.prefix_ids_fn = prefix_ids_fn
        self.text_batch, self.token_budget, self.grad_token_budget = text_batch, token_budget, grad_token_budget
        self.attn_impl = getattr(model.config, "_attn_implementation", "sdpa")
        self.softcap = getattr(model.config, "final_logit_softcapping", None)
        self.host_cache = HostPrefixCache(prefix_cache_gb) if prefix_cache_gb > 0 else None
        union = sorted({t for g in digit_groups for t in g})
        self.digit_ids = torch.tensor(union, device=self.device)
        self.group_mask = torch.tensor([[t in g for t in union] for g in digit_groups], device=self.device)
        self.digit_W = model.get_output_embeddings().weight[self.digit_ids].detach()   # [G, E]
        self.temp = 1.0

    # ---------------------------------------------------------------- prefixes
    @torch.no_grad()
    def _encode(self, ids: list[list[int]]) -> Prefix:
        from transformers import DynamicCache
        B, P = len(ids), max(len(p) for p in ids)
        pid = torch.full((B, P), self.tok.pad_token_id, dtype=torch.long, device=self.device)
        pm = torch.zeros((B, P), dtype=torch.long, device=self.device)
        for r, p in enumerate(ids):
            pid[r, :len(p)] = torch.tensor(p, device=self.device)
            pm[r, :len(p)] = 1
        cache = DynamicCache()
        self.base(input_ids=pid, attention_mask=pm, past_key_values=cache, use_cache=True)
        if getattr(cache, "layers", None) is not None:
            kv = [(l.keys, l.values) for l in cache.layers]
        else:
            kv = list(zip(cache.key_cache, cache.value_cache))
        return Prefix(kv=kv, mask=pm, real_len=pm.sum(1))

    @torch.no_grad()
    def encode_prefix(self, ids: list[list[int]]) -> Prefix:
        if self.host_cache is None:
            return self._encode(ids)
        keys = [tuple(p) for p in ids]
        hit = [self.host_cache.get(k) for k in keys]
        miss = [r for r, t in enumerate(hit) if t is None]
        enc = self._encode([ids[r] for r in miss]) if miss else None
        B, P = len(ids), max(len(p) for p in ids)
        ref = enc.kv[0][0] if enc is not None else None
        h, hd = (ref.shape[1], ref.shape[3]) if ref is not None else (hit[0].shape[2], hit[0].shape[4])
        dtype = ref.dtype if ref is not None else hit[0].dtype
        L = len(enc.kv) if enc is not None else hit[0].shape[0]
        big = torch.zeros(L, 2, B, h, P, hd, dtype=dtype, device=self.device)
        pm = torch.zeros((B, P), dtype=torch.long, device=self.device)
        for r, p in enumerate(ids):
            pm[r, :len(p)] = 1
            if hit[r] is not None:
                big[:, :, r, :, :len(p)] = hit[r].to(self.device, non_blocking=True)
        for m, r in enumerate(miss):
            n = len(ids[r])
            t = torch.stack([torch.stack([k[m, :, :n], v[m, :, :n]]) for k, v in enc.kv])   # [L, 2, h, n, hd]
            big[:, :, r, :, :n] = t
            self.host_cache.put(keys[r], t)
        return Prefix(kv=[(big[l, 0], big[l, 1]) for l in range(L)], mask=pm, real_len=pm.sum(1))

    # ---------------------------------------------------------------- packed forward
    def _packed(self, pre: Prefix, blocks: list[torch.Tensor]) -> torch.Tensor:
        """Concept scores [len(blocks), B] for description blocks (each [L_k, E]) against a prefix."""
        from transformers import DynamicCache
        B, P = pre.mask.shape
        dtype = self.model.get_input_embeddings().weight.dtype
        lens = [b.shape[0] for b in blocks]
        Lt = sum(lens)
        x = torch.cat([b.to(dtype) for b in blocks], 0)[None].expand(B, -1, -1)
        offs = torch.cat([torch.arange(L, device=self.device) for L in lens])
        pos = pre.real_len[:, None] + offs[None]                  # each block restarts after the prefix
        blk = torch.zeros(Lt, Lt, dtype=torch.bool, device=self.device)
        o = 0
        for L in lens:
            blk[o:o + L, o:o + L] = torch.tril(torch.ones(L, L, dtype=torch.bool, device=self.device))
            o += L
        m = torch.cat([pre.mask.bool()[:, None, :].expand(B, Lt, P), blk[None].expand(B, Lt, Lt)], -1)[:, None]
        if self.attn_impl != "sdpa":                              # eager attention takes an additive mask
            m = torch.zeros(m.shape, dtype=x.dtype, device=self.device).masked_fill(~m, torch.finfo(x.dtype).min)
        cache = DynamicCache()
        for i, (k, v) in enumerate(pre.kv):
            cache.update(k, v, i)                                 # shares the prefix; suffix keys are concatenated
        h = self.base(inputs_embeds=x, attention_mask=m, position_ids=pos, past_key_values=cache,
                      cache_position=torch.arange(P, P + Lt, device=self.device), use_cache=True
                      ).last_hidden_state
        ends = torch.tensor(np.cumsum(lens) - 1, device=self.device)
        lg = h[:, ends].float() @ self.digit_W.float().T          # [B, K, G] digit logits only
        if self.softcap:
            lg = torch.tanh(lg / self.softcap) * self.softcap
        cols = lg[..., None, :].masked_fill(~self.group_mask, -torch.inf).logsumexp(-1)   # [B, K, 10]
        p = torch.softmax(cols / self.temp, -1)
        return ((p * torch.arange(10, device=self.device)).sum(-1) / 9.0).T              # [K, B]

    def _units(self, lens: list[int], texts: Sequence[str], budget: int):
        """(first text index, prefix ids, description groups) per text chunk; a group fits B * tokens <= budget."""
        ids = self.prefix_ids_fn(texts)
        for t0 in range(0, len(texts), self.text_batch):
            chunk = ids[t0:t0 + self.text_batch]
            B, groups, cur, tot = len(chunk), [], [], 0
            for k, L in enumerate(lens):
                if cur and B * (tot + L) > budget:
                    groups.append(cur)
                    cur, tot = [], 0
                cur.append(k)
                tot += L
            yield t0, chunk, groups + ([cur] if cur else [])

    # ---------------------------------------------------------------- public API
    @torch.no_grad()
    def scores(self, block_fn: BlockFn, K: int, texts: Sequence[str], keep_prefix: bool = False):
        """Score matrix [K, M] without gradient; with keep_prefix also the encoded prefixes."""
        lens = [block_fn(k).shape[0] for k in range(K)]
        out = torch.zeros(K, len(texts), device=self.device)
        kept = []
        for t0, ids, groups in self._units(lens, texts, self.token_budget):
            pre = self.encode_prefix(ids)
            for g in groups:
                out[g, t0:t0 + len(ids)] = self._packed(pre, [block_fn(k) for k in g])
            if keep_prefix:
                kept.append(pre)
        return (out, kept) if keep_prefix else out

    def backward_through(self, block_fn: BlockFn, K: int, texts: Sequence[str], grad_scores: torch.Tensor,
                         prefixes: list[Prefix] | None = None):
        """Pass 2: recompute each unit with gradient and backpropagate <grad_scores, scores>."""
        lens = [block_fn(k).shape[0] for k in range(K)]
        units = list(self._units(lens, texts, self.grad_token_budget))
        # pass-1 prefixes are per text_batch chunk; pass 2 uses the same chunking
        for c, (t0, ids, groups) in enumerate(units):
            pre = prefixes[c] if prefixes is not None else self.encode_prefix(ids)
            for g in groups:
                gs = grad_scores[g, t0:t0 + len(ids)]
                if gs.abs().sum() == 0:
                    continue
                with torch.enable_grad():
                    (self._packed(pre, [block_fn(k) for k in g]) * gs).sum().backward()


class SoftScores(torch.autograd.Function):
    """Differentiable packed scores [K, M] of soft prompts E [K, T, E] followed by a fixed question."""

    @staticmethod
    def forward(ctx, E, scorer, suffix, texts, temp):
        scorer.temp = temp
        keep = ctx.needs_input_grad[0]
        res = scorer.scores(lambda k: torch.cat([E[k], suffix.to(E.dtype)], 0), E.shape[0], texts, keep_prefix=keep)
        S, ctx.prefixes = res if keep else (res, None)
        ctx.scorer, ctx.suffix, ctx.texts, ctx.temp = scorer, suffix, texts, temp
        ctx.save_for_backward(E)
        return S

    @staticmethod
    def backward(ctx, grad_S):
        (E,) = ctx.saved_tensors
        Ed = E.detach().requires_grad_(True)
        ctx.scorer.temp = ctx.temp
        with torch.enable_grad():
            ctx.scorer.backward_through(lambda k: torch.cat([Ed[k], ctx.suffix.to(Ed.dtype)], 0), E.shape[0],
                                        ctx.texts, grad_S.detach().float(), prefixes=ctx.prefixes)
        ctx.prefixes = None
        return (Ed.grad if Ed.grad is not None else torch.zeros_like(E)), None, None, None, None
