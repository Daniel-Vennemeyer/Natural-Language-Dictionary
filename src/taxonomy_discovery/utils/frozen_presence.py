"""Presence via the FROZEN target model itself + a yes/no prompt (cross-encoder judge).

p(concept, text) = P("yes") under the frozen target model. Differentiable in the concept when the
concept is passed as soft-prompt embeddings, so it can ground a continuous learner. No distillation.

The text is the expensive part, so the prompt is laid out text-first and `presence_grid` KV-CACHES
each text once, then queries many concepts (cheap suffix forwards) against that cache -- the
~text_len/concept_len speedup that makes the K*N presence matrix affordable. `presence` (naive,
full forward) is kept as the correctness reference.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.utils.checkpoint

# Manual chat layout, split so the [instruction + TEXT] prefix is cacheable and only the
# [concept + question] suffix varies. PRE+text+MID is the prefix; concept+SUF is the suffix.
# The ChatML wrappers below are the QWEN layout; __init__ rebuilds the instance-level PRE/SUF/
# SUF_SCALE/HEAD_JOINT from the tokenizer's own chat template for other families (e.g. Gemma) --
# Qwen keeps these exact strings so all existing numbers stay byte-identical.
U_OPEN_QWEN = '<|im_start|>user\n'
A_OPEN_QWEN = '<|im_end|>\n<|im_start|>assistant\n'
PRE = U_OPEN_QWEN + 'Text:\n"""\n'
MID = '\n"""\n\nDoes this text express or contain the following concept: "'
# coherence-aware variant: incoherent/gibberish concepts -> "no" -> ~0 presence (anti-drift judge)
MID_COH = ('\n"""\n\nIs the following concept COHERENT and clearly expressed in the text '
           '(answer "no" if it is incoherent or absent): "')
# token-level: the text is a sentence with one token marked <<...>>; judge presence AT that token
MID_TOK = ('\n"""\n\nIs the following concept COHERENT and expressed specifically AT the marked '
           'token (the one wrapped in <<...>>) above (answer "no" if incoherent or not at that token): "')
SUF = '"?\nAnswer with only "yes" or "no".<|im_end|>\n<|im_start|>assistant\n'
# graded variant: elicit a single-digit STRENGTH rating instead of yes/no (less saturated -> richer ranks)
MID_SCALE = '\n"""\n\nHow strongly does the text above express or contain the following concept: "'
SUF_SCALE = ('"?\nRate on a scale from 0 to 9, where 0 = not at all present and 9 = strongly present. '
             'Answer with a single digit.<|im_end|>\n<|im_start|>assistant\n')
# JOINT variant: the whole K-concept taxonomy in ONE prompt, one rating per concept. The judge can
# do explaining-away ("this is a case of C, not E") instead of K independent absolute ratings, and
# one suffix forward scores all K concepts. Letters label concepts so list numbering cannot bleed
# into the digit ratings. Deliberately NEUTRAL about the behavior domain (grounding-as-read-off).
MID_JOINT = ('\n"""\n\nI am labeling a taxonomy of assistant response behaviors. For EACH behavior '
             'below, rate from 0 to 9 how strongly the text above displays it (0 = not at all, '
             '9 = strongly). Rate each behavior on its own evidence.\n\n')
HEAD_JOINT = '\nRatings (one digit each):<|im_end|>\n<|im_start|>assistant\n'
JOINT_LETTERS = "ABCDEFGHIJKLMNOP"


class FrozenJudge:
    def __init__(self, model_name: str, device="cuda", dtype: str = "bfloat16",
                 max_text_chars: int = 800, max_len: int = 1024):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        import os
        # FROZEN_JUDGE_DEVICE_MAP=auto shards the judge across all visible GPUs -- required
        # for ~70GB subjects whose soft-training graphs don't fit beside the weights on one
        # card. Inputs still go to the first device; accelerate moves activations between.
        dm = os.environ.get("FROZEN_JUDGE_DEVICE_MAP", str(device))
        try:
            # transformers v5 path: `torch_dtype` is gone (silently ignored at best -> fp32
            # OOM at 35B); device_map streams shards straight to the GPU (no CPU staging)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=td, device_map=dm, low_cpu_mem_usage=True).eval()
        except TypeError:                                        # very old transformers: no `dtype` kw
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td).to(device).eval()
        if dm == "auto":
            device = torch.device("cuda:0")                      # input device for a sharded judge
        self.model.requires_grad_(False)
        self.device = device; self.max_text_chars = max_text_chars; self.max_len = max_len
        self.embed = self.model.get_input_embeddings()
        # decoder blocks, wrapper-tolerant (multimodal wrappers put the text stack under
        # language_model/text_model; plain decoders under model.layers)
        self.layers = None
        for path in ("model.layers", "layers", "language_model.model.layers",
                     "model.language_model.layers", "model.text_model.layers", "transformer.h"):
            cur = self.model
            for a in path.split("."):
                cur = getattr(cur, a, None)
                if cur is None:
                    break
            if cur is not None:
                self.layers = cur
                break
        if self.layers is None:
            raise AttributeError(f"cannot locate decoder layers on {type(self.model).__name__}")
        # chat wrappers: Qwen keeps the historical hardcoded ChatML (byte-identical numbers);
        # other families (Gemma, Llama, ...) derive user-open / assistant-open from their own
        # chat template so the judge prompt is well-formed for them too.
        if "im_start" in (self.tok.chat_template or ""):
            self.u_open, self.a_open = U_OPEN_QWEN, A_OPEN_QWEN
        else:
            mark = "XQZCONTENTZQX"
            s = self.tok.apply_chat_template([{"role": "user", "content": mark}],
                                             tokenize=False, add_generation_prompt=True)
            self.u_open, self.a_open = s.split(mark)
        self.PRE = self.u_open + 'Text:\n"""\n'
        self.SUF = '"?\nAnswer with only "yes" or "no".' + self.a_open
        self.SUF_SCALE = ('"?\nRate on a scale from 0 to 9, where 0 = not at all present and '
                          '9 = strongly present. Answer with a single digit.' + self.a_open)
        self.HEAD_JOINT = '\nRatings (one digit each):' + self.a_open
        self.yes_ids = self._first_tokens(["yes", "Yes", " yes", " Yes", "YES"])
        self.no_ids = self._first_tokens(["no", "No", " no", " No", "NO"])
        self.scale_groups = [self._first_tokens([str(i), " " + str(i)]) for i in range(10)]   # digit ids 0..9
        self.digit_ids = [self._ids(str(i))[0] for i in range(10)]                             # canonical digit token per rating
        # only the LAST position's logits are read in the grid paths; telling the model so
        # skips materializing [B, L, vocab] logits (+ fp32 softcap copy) -- ~2.4GB/forward on
        # gemma's 262k vocab. Kwarg name differs across transformers versions.
        import inspect
        try:
            _fp = inspect.signature(self.model.forward).parameters
            self._lk = ({"logits_to_keep": 1} if "logits_to_keep" in _fp else
                        {"num_logits_to_keep": 1} if "num_logits_to_keep" in _fp else {})
            self._cp = "cache_position" in _fp
        except (TypeError, ValueError):
            self._lk = {}; self._cp = False
        self._coh_cache: dict = {}                                # concept -> P(coherent), text-independent
        self._cohscale_cache: dict = {}                           # concept -> graded coherence E[0-9]/9

    def _first_tokens(self, words):
        ids = set()
        for w in words:
            t = self.tok(w, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        return sorted(ids)

    def _ids(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def _extract_kv(self, cache):
        """Per-layer (key, value) tensors from a prefix cache, version-robustly."""
        if getattr(cache, "key_cache", None):
            return list(cache.key_cache), list(cache.value_cache)
        if getattr(cache, "layers", None) is not None:
            return [l.keys for l in cache.layers], [l.values for l in cache.layers]
        pk, pv = [], []
        for i in range(len(cache)):
            kv = cache[i]; pk.append(kv[0]); pv.append(kv[1])
        return pk, pv

    def _prefix_handle(self, cache):
        """Reusable prefix handle. Standard KV caches -> (pk, pv) tensor lists shared across
        queries (zero-copy). Hybrid caches (e.g. linear-attention layers with recurrent state,
        no .keys/.values) -> the cache object itself; _fresh_cache deep-copies it per query."""
        try:
            return self._extract_kv(cache)
        except AttributeError:
            return cache

    def _fresh_cache(self, handle):
        if isinstance(handle, tuple):
            return self._cache_from_kv(*handle)
        return copy.deepcopy(handle)                             # hybrid: copy states + KV

    def _cache_from_kv(self, pk, pv):
        """Fresh cache SHARING the prefix tensors (no copy). The model's forward appends suffix KV
        via cat -> new tensors, so the shared prefix tensors are never mutated -> reusable across
        all concepts with one prefix cache in memory (not one per concept)."""
        from transformers import DynamicCache
        c = DynamicCache()
        for i in range(len(pk)):
            c.update(pk[i], pv[i], i)
        return c

    def _score(self, last_logits, temp: float = 1.0):
        py = last_logits[:, self.yes_ids].logsumexp(-1)
        pn = last_logits[:, self.no_ids].logsumexp(-1)
        return torch.sigmoid((py - pn) / temp)                     # temp>1 softens -> non-saturated gradient

    def _score_scale(self, last_logits, temp: float = 1.0):        # graded strength E[i*p_i]/9 over digits 0..9
        cols = torch.stack([last_logits[:, g].logsumexp(-1) for g in self.scale_groups], -1)   # [B, 10]
        p = torch.softmax(cols / temp, -1)
        idx = torch.arange(10, device=last_logits.device, dtype=p.dtype)
        return (p * idx).sum(-1) / 9.0                             # in [0, 1]

    @torch.no_grad()
    def coherent(self, concepts, batch: int = 64):
        """P(coherent) per concept (text-independent, cached). A cheap pre-filter: gate the
        expensive presence calls on whether the concept is a ratable passage-level concept."""
        self.tok.padding_side = "right"
        new = [c for c in dict.fromkeys(concepts) if c not in self._coh_cache]
        for s in range(0, len(new), batch):
            grp = new[s:s + batch]
            prompts = [self.u_open + 'Is "' + (c or "") + '" a coherent, specific, MEANINGFUL concept '
                       'that a human could rate from 0 to 100 for how strongly it is present in a passage '
                       'of text? Answer "no" if it is vacuous, tautological, self-contradictory, or not a '
                       'real concept. Answer only "yes" or "no".' + self.a_open for c in grp]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=128, add_special_tokens=False).to(self.device)
            logits = self.model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            lg = logits[torch.arange(len(grp), device=self.device), last]
            for c, pv in zip(grp, self._score(lg).float().cpu().tolist()):
                self._coh_cache[c] = pv
        return np.array([self._coh_cache[c] for c in concepts], dtype=np.float32)

    @torch.no_grad()
    def coherence_scale(self, concepts, batch: int = 64):
        """Graded 0-9 coherence E[digit]/9 per concept (text-independent, cached). A clarity/usefulness
        rating: high = clear, specific, ratable taxonomy label; low/0 = vague, redundant, non-behavioral."""
        self.tok.padding_side = "right"
        body = (
            "You grade a CONCEPT that labels ONE behavior in a taxonomy of how an AI assistant responds. "
            "9 = a single specific, observable behavior you could reliably detect; 5 = a real but overly "
            "BROAD behavior; 0 = a generic catch-all, vague, or non-behavioral category. Do NOT reward "
            "breadth -- a generic or always-present concept is BAD. Reply with ONLY one digit 0-9.\n\n"
            "Concept: \"validating the user's feelings even when their actions were wrong\"\nScore: 9\n"
            "Concept: \"telling the user to end a relationship\"\nScore: 9\n"
            "Concept: \"softening a criticism by praising the user first\"\nScore: 8\n"
            "Concept: \"giving advice\"\nScore: 5\n"
            "Concept: \"a personal value that guides decisions\"\nScore: 3\n"
            "Concept: \"a feeling or emotion that arises in response to a situation\"\nScore: 0\n"
            "Concept: \"a process or action that influences something\"\nScore: 0\n"
            "Concept: \"a central idea or theme\"\nScore: 0\n"
            "Concept: \"{c}\"\nScore:")
        new = [c for c in dict.fromkeys(concepts) if c not in self._cohscale_cache]
        for s in range(0, len(new), batch):
            grp = new[s:s + batch]
            prompts = [self.u_open + body.format(c=(c or "")) + self.a_open
                       for c in grp]
            enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True,
                           max_length=640, add_special_tokens=False).to(self.device)
            logits = self.model(**enc).logits
            last = enc["attention_mask"].sum(1) - 1
            lg = logits[torch.arange(len(grp), device=self.device), last]
            for c, v in zip(grp, self._score_scale(lg).float().cpu().tolist()):
                self._cohscale_cache[c] = v
        return np.array([self._cohscale_cache[c] for c in concepts], dtype=np.float32)

    # --- naive full-forward (correctness reference; parallel concept/text pairs) ----------
    @torch.no_grad()
    def presence(self, concepts, texts, batch_size: int = 32, verbose: bool = False):
        self.tok.padding_side = "right"
        probs = []
        for i in range(0, len(concepts), batch_size):
            cs = concepts[i:i + batch_size]; xs = texts[i:i + batch_size]
            strs = [self.PRE + (x or "")[:self.max_text_chars] + MID + c + SUF for c, x in zip(cs, xs)]
            enc = self.tok(strs, return_tensors="pt", padding=True, truncation=True,
                           max_length=self.max_len, add_special_tokens=False).to(self.device)
            logits = self.model(**enc).logits
            last = enc.attention_mask.sum(1) - 1                  # right-pad: last real token
            lg = logits[torch.arange(len(cs), device=self.device), last]
            probs.append(self._score(lg).float().cpu())
            if verbose and (i // batch_size) % 20 == 0:
                print(f"[frozen-judge] {i+len(cs)}/{len(concepts)} pairs", flush=True)
        return torch.cat(probs).numpy().astype(np.float32)

    # --- KV-cached grid: every concept x every text, text processed once -----------------
    @torch.no_grad()
    def presence_grid(self, concepts, texts, text_batch: int = 48, verbose: bool = False,
                      coherence: bool = False, grain: str = "response", mode: str = "yesno"):
        """Returns P [K, M] = p(concept_k, text_m). Caches each text prefix, queries all concepts.
        grain='token' -> the text is a marked sentence, judge presence AT the <<...>> token;
        else coherence=True uses the anti-drift prompt.
        mode='scale' -> graded single-digit strength rating E[i*p_i]/9 instead of P(yes) (less saturated)."""
        if mode == "scale":
            mid, suf, scorer = MID_SCALE, self.SUF_SCALE, self._score_scale
        else:
            mid = MID_TOK if grain == "token" else (MID_COH if coherence else MID)
            suf, scorer = self.SUF, self._score
        K, M = len(concepts), len(texts)
        suffix_ids = [self._ids(c + suf) for c in concepts]        # variable length per concept
        out = np.zeros((K, M), dtype=np.float32)
        for tb in range(0, M, text_batch):
            xs = texts[tb:tb + text_batch]; Mb = len(xs)
            pre = [self._ids(self.PRE + (x or "")[:self.max_text_chars] + mid)[: self.max_len - 32] for x in xs]
            P = max(len(p) for p in pre)
            pre_ids = torch.full((Mb, P), self.tok.pad_token_id, dtype=torch.long, device=self.device)
            pre_mask = torch.zeros((Mb, P), dtype=torch.long, device=self.device)
            for r, p in enumerate(pre):                            # RIGHT-pad the prefix
                pre_ids[r, :len(p)] = torch.tensor(p, device=self.device); pre_mask[r, :len(p)] = 1
            cache = self.model(input_ids=pre_ids, attention_mask=pre_mask, use_cache=True).past_key_values
            ph = self._prefix_handle(cache)
            real_len = pre_mask.sum(1)                             # [Mb] real prefix length per row
            for k in range(K):
                sfx = torch.tensor(suffix_ids[k], device=self.device)
                Ls = sfx.shape[0]
                sfx_b = sfx[None].expand(Mb, -1)
                mask = torch.cat([pre_mask, torch.ones(Mb, Ls, dtype=torch.long, device=self.device)], 1)
                pos = real_len[:, None] + torch.arange(Ls, device=self.device)[None, :]   # continue positions
                ck = self._fresh_cache(ph)                         # zero-copy for KV caches; deepcopy for hybrid
                # cache_position = PHYSICAL slots [P, P+Ls) (right-padded prefix fills the
                # cache to P) -- sliding-window/hybrid layers (gemma) mis-slot the suffix
                # without it; full-attention DynamicCache infers the same thing (no-op)
                lg = self.model(input_ids=sfx_b, attention_mask=mask, position_ids=pos,
                                past_key_values=ck,
                                **dict(self._lk, **({"cache_position": torch.arange(pre_mask.shape[1], pre_mask.shape[1] + Ls, device=self.device)} if self._cp else {}))).logits[:, -1]
                out[k, tb:tb + Mb] = scorer(lg).float().cpu().numpy()
            if verbose:
                print(f"[frozen-judge] grid texts {tb+Mb}/{M} (x{K} concepts)", flush=True)
        return out

    # --- differentiable grid for soft-prompt concepts (for the continuous learner) -------
    def presence_grid_soft(self, concept_embeds, texts, text_batch: int = 48, temp: float = 1.0,
                           mode: str = "yesno"):
        """concept_embeds: [K, T, E] learnable. Returns p [K, M] differentiable in concept_embeds
        (text prefix cached under no_grad; gradient flows only through the suffix concept tokens).
        mode='scale' -> graded digit rating (matches presence_grid(mode='scale'), less saturated)."""
        mid, suf, scorer = (MID_SCALE, self.SUF_SCALE, self._score_scale) if mode == "scale" else (MID, self.SUF, self._score)
        K = concept_embeds.shape[0]; M = len(texts)
        suf_ids = torch.tensor(self._ids(suf), device=self.device)
        suf_emb = self.embed(suf_ids).detach()                    # [Lq, E] fixed question tokens
        cols = []
        for tb in range(0, M, text_batch):
            xs = texts[tb:tb + text_batch]; Mb = len(xs)
            pre = [self._ids(self.PRE + (x or "")[:self.max_text_chars] + mid)[: self.max_len - 32] for x in xs]
            P = max(len(p) for p in pre)
            pre_ids = torch.full((Mb, P), self.tok.pad_token_id, dtype=torch.long, device=self.device)
            pre_mask = torch.zeros((Mb, P), dtype=torch.long, device=self.device)
            for r, p in enumerate(pre):
                pre_ids[r, :len(p)] = torch.tensor(p, device=self.device); pre_mask[r, :len(p)] = 1
            with torch.no_grad():
                cache = self.model(input_ids=pre_ids, attention_mask=pre_mask, use_cache=True).past_key_values
            ph = self._prefix_handle(cache)
            real_len = pre_mask.sum(1)
            block = []
            T = concept_embeds.shape[1]; Lq = suf_emb.shape[0]; Ls = T + Lq
            suf_e = suf_emb.to(concept_embeds.dtype)
            mask = torch.cat([pre_mask, torch.ones(Mb, Ls, dtype=torch.long, device=self.device)], 1)
            pos = real_len[:, None] + torch.arange(Ls, device=self.device)[None, :]

            # checkpointed: activations recomputed in BACKWARD, i.e. after this loop has moved on -- the
            # per-chunk tensors MUST be bound as default args (def-time), not read from the enclosing
            # scope (late binding would recompute every chunk with the LAST chunk's prefix cache).
            Pfx = pre_mask.shape[1]

            def _one(emb_k, ph=ph, mask=mask, pos=pos, Mb=Mb, Pfx=Pfx, Ls=Ls):
                emb_b = torch.cat([emb_k, suf_e], 0)[None].expand(Mb, -1, -1).to(self.embed.weight.dtype)
                ck = self._fresh_cache(ph)                         # zero-copy for KV caches; deepcopy for hybrid
                lg = self.model(inputs_embeds=emb_b, attention_mask=mask, position_ids=pos,
                                past_key_values=ck,
                                **dict(self._lk, **({"cache_position": torch.arange(Pfx, Pfx + Ls, device=self.device)} if self._cp else {}))).logits[:, -1]
                return scorer(lg, temp)
            for k in range(K):
                block.append(torch.utils.checkpoint.checkpoint(_one, concept_embeds[k], use_reentrant=False))
            block = torch.stack(block, 0)                          # [K, Mb]
            cols.append(block)
        return torch.cat(cols, dim=1)                              # [K, M], differentiable in concept_embeds

    # --- JOINT differentiable grid: all K concepts in ONE prompt (explaining-away) --------
    def presence_grid_soft_joint(self, concept_embeds, texts, text_batch: int = 48,
                                 temp: float = 1.0, order=None):
        """Two-pass self-conditioned joint scoring, differentiable in concept_embeds [K, T, E].

        Pass 1 (no grad): greedily decode the judge's OWN digit ratings slot by slot, so each
        later rating conditions on the model's real assessments of earlier concepts -- genuine
        explaining-away context, not a teacher-forced placeholder fiction. Pass 2 (grad): one
        forward teacher-forcing those self-ratings; the logits at each rating position are
        IDENTICAL to pass 1's (same conditioning) but now differentiable through every slot in
        the listing. `order` permutes slot positions in the prompt (shuffle per step to
        symmetrize position effects); scores return in ORIGINAL slot order. Returns [K, M]."""
        K = concept_embeds.shape[0]; M = len(texts)
        assert K <= len(JOINT_LETTERS), f"joint scoring supports K <= {len(JOINT_LETTERS)}"
        order = list(range(K)) if order is None else [int(i) for i in order]
        emb = self.embed; dtW = emb.weight.dtype

        def _seg(s):                                             # fixed scaffold segment -> embeds
            return emb(torch.tensor(self._ids(s), device=self.device)).detach()
        lab = [_seg(f'{JOINT_LETTERS[i]}. "') for i in range(K)]
        endq = _seg('"\n')
        head0 = torch.cat([_seg(self.HEAD_JOINT), _seg(f'{JOINT_LETTERS[0]}: ')], 0)
        rat = [None] + [_seg(f'\n{JOINT_LETTERS[i]}: ') for i in range(1, K)]
        dig_emb = emb(torch.tensor(self.digit_ids, device=self.device)).detach()   # [10, E]

        def _listing(cemb):                                      # slots in permuted order
            parts = []
            for pos_i, j in enumerate(order):
                parts += [lab[pos_i], cemb[j].to(dtW), endq]
            return torch.cat(parts, 0)

        cols = []
        for tb in range(0, M, text_batch):
            xs = texts[tb:tb + text_batch]; Mb = len(xs)
            pre = [self._ids(self.PRE + (x or "")[:self.max_text_chars] + MID_JOINT)[: self.max_len - 32] for x in xs]
            Pmax = max(len(p) for p in pre)
            pre_ids = torch.full((Mb, Pmax), self.tok.pad_token_id, dtype=torch.long, device=self.device)
            pre_mask = torch.zeros((Mb, Pmax), dtype=torch.long, device=self.device)
            for r, p in enumerate(pre):
                pre_ids[r, :len(p)] = torch.tensor(p, device=self.device); pre_mask[r, :len(p)] = 1
            with torch.no_grad():
                cache = self.model(input_ids=pre_ids, attention_mask=pre_mask, use_cache=True).past_key_values
            ph = self._prefix_handle(cache)
            real_len = pre_mask.sum(1)

            # ---- pass 1 (no grad): sequential greedy self-ratings ----
            with torch.no_grad():
                s0 = torch.cat([_listing(concept_embeds.detach()), head0], 0)
                cur = s0[None].expand(Mb, -1, -1)
                mask = torch.cat([pre_mask, torch.ones(Mb, cur.shape[1], dtype=torch.long, device=self.device)], 1)
                pos = real_len[:, None] + torch.arange(cur.shape[1], device=self.device)[None, :]
                out = self.model(inputs_embeds=cur, attention_mask=mask, position_ids=pos,
                                 past_key_values=self._fresh_cache(ph), use_cache=True)
                cur_len = real_len + cur.shape[1]
                digits = []                                      # [K] tensors of [Mb], in ORDER positions
                lg = out.logits[:, -1]
                for k in range(K):
                    c10 = torch.stack([lg[:, g].logsumexp(-1) for g in self.scale_groups], -1)
                    d = c10.argmax(-1)
                    digits.append(d)
                    if k < K - 1:
                        step = torch.cat([dig_emb[d][:, None], rat[k + 1][None].expand(Mb, -1, -1)], 1)
                        Lst = step.shape[1]
                        mask = torch.cat([mask, torch.ones(Mb, Lst, dtype=torch.long, device=self.device)], 1)
                        npos = cur_len[:, None] + torch.arange(Lst, device=self.device)[None, :]
                        out = self.model(inputs_embeds=step, attention_mask=mask, position_ids=npos,
                                         past_key_values=out.past_key_values, use_cache=True)
                        cur_len = cur_len + Lst
                        lg = out.logits[:, -1]

            # ---- pass 2 (grad): teacher-force the self-ratings, read all K rating logits ----
            n_list = sum(x.shape[0] for x in lab) + sum(endq.shape[0] for _ in range(K)) \
                + concept_embeds.shape[1] * K
            read_pos = []                                        # suffix index whose LOGIT rates slot k
            off = n_list + head0.shape[0]
            read_pos.append(off - 1)
            for k in range(1, K):
                off += 1 + rat[k].shape[0]                       # digit token + next rating stub
                read_pos.append(off - 1)
            Ls = off                                             # total suffix length (no digit after last)
            maskF = torch.cat([pre_mask, torch.ones(Mb, Ls, dtype=torch.long, device=self.device)], 1)
            posF = real_len[:, None] + torch.arange(Ls, device=self.device)[None, :]

            def _fwd(cemb, ph=ph, digits=digits, maskF=maskF, posF=posF, Mb=Mb):
                rows = [torch.cat([_listing(cemb), head0], 0)[None].expand(Mb, -1, -1)]
                for k in range(1, K):
                    rows.append(dig_emb[digits[k - 1]][:, None])
                    rows.append(rat[k][None].expand(Mb, -1, -1))
                full = torch.cat(rows, 1).to(dtW)
                lgs = self.model(inputs_embeds=full, attention_mask=maskF, position_ids=posF,
                                 past_key_values=self._fresh_cache(ph)).logits
                return torch.stack([self._score_scale(lgs[:, p], temp) for p in read_pos], 0)
            blk = torch.utils.checkpoint.checkpoint(_fwd, concept_embeds, use_reentrant=False)
            unperm = torch.empty_like(blk)                       # ORDER positions -> original slots
            for pos_i, j in enumerate(order):
                unperm[j] = blk[pos_i]
            cols.append(unperm)
        return torch.cat(cols, dim=1)                              # [K, M], differentiable
