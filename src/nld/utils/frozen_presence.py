"""Concept scores from the frozen subject model acting as its own judge.

The concept score of a description c on a text x is the judge's expected 0-9 rating,
E[digit] / 9, read from the digit-token logits after a rating prompt. When c is given as
soft-prompt embeddings the score is differentiable in c, which is what trains the dictionary.

The prompt is laid out text-first: [instruction + text] is a prefix that is encoded once per
text, and every description is a short [description + question] block that attends to it.
Blocks are scored many at a time in one packed forward (see packed.py), which makes the K x N
score matrix affordable during training.
"""
from __future__ import annotations

import numpy as np
import torch

from .packed import PackedScorer, SoftScores

# Qwen ChatML wrappers. For other model families, __init__ derives the equivalent strings
# from the tokenizer's chat template.
U_OPEN_QWEN = '<|im_start|>user\n'
A_OPEN_QWEN = '<|im_end|>\n<|im_start|>assistant\n'
# The prefix is user_open + PRE_BODY + text + MID_SCALE; the suffix is description + SUF_SCALE.
PRE_BODY = 'Text:\n"""\n'
MID_SCALE = '\n"""\n\nHow strongly does the text above express or contain the following concept: "'
SUF_SCALE_BODY = ('"?\nRate on a scale from 0 to 9, where 0 = not at all present and '
                  '9 = strongly present. Answer with a single digit.')


class FrozenJudge:
    def __init__(self, model_name: str, device="cuda", dtype: str = "bfloat16",
                 max_text_chars: int = 800, max_len: int = 1024, prefix_cache_gb: float = 0.0):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
        self.tok = AutoTokenizer.from_pretrained(model_name)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=td, device_map=str(device), low_cpu_mem_usage=True).eval()
        except TypeError:                       # transformers versions without the `dtype` keyword
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=td).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device; self.max_text_chars = max_text_chars; self.max_len = max_len
        self.embed = self.model.get_input_embeddings()
        # decoder blocks (used by callers to hook the residual stream)
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
        # chat wrappers: ChatML tokenizers use the Qwen strings above; any other family
        # takes its user-open / assistant-open strings from its own chat template
        if "im_start" in (self.tok.chat_template or ""):
            self.u_open, self.a_open = U_OPEN_QWEN, A_OPEN_QWEN
        else:
            mark = "XQZCONTENTZQX"
            s = self.tok.apply_chat_template([{"role": "user", "content": mark}],
                                             tokenize=False, add_generation_prompt=True)
            self.u_open, self.a_open = s.split(mark)
        self.PRE = self.u_open + PRE_BODY
        self.SUF_SCALE = SUF_SCALE_BODY + self.a_open
        # token ids for each rating 0..9, with and without a leading space
        self.scale_groups = [self._first_tokens([str(i), " " + str(i)]) for i in range(10)]
        # packed scorer; prefixes are cut at max_len - 32 tokens
        self.packed = PackedScorer(self.model, self.tok, device, self.scale_groups,
                                   lambda texts: [self._ids(self.PRE + (x or "")[:self.max_text_chars] + MID_SCALE)
                                                  [: self.max_len - 32] for x in texts],
                                   prefix_cache_gb=prefix_cache_gb)
        self._suffix_emb = self.embed(torch.tensor(self._ids(self.SUF_SCALE), device=device)).detach()
        self._cohscale_cache: dict = {}                           # description -> rating, text-independent

    def _first_tokens(self, words):
        ids = set()
        for w in words:
            t = self.tok(w, add_special_tokens=False).input_ids
            if t:
                ids.add(t[0])
        return sorted(ids)

    def _ids(self, s):
        return self.tok(s, add_special_tokens=False).input_ids

    def _score_scale(self, last_logits, temp: float = 1.0):
        """Concept score E[i * p_i] / 9 over the digit ratings 0..9, in [0, 1]."""
        cols = torch.stack([last_logits[:, g].logsumexp(-1) for g in self.scale_groups], -1)   # [B, 10]
        p = torch.softmax(cols / temp, -1)
        idx = torch.arange(10, device=last_logits.device, dtype=p.dtype)
        return (p * idx).sum(-1) / 9.0

    @staticmethod
    def _check_mode(grain: str, mode: str):
        if mode != "scale" or grain != "response":
            raise ValueError(f"only grain='response', mode='scale' is supported (got {grain!r}, {mode!r})")

    @torch.no_grad()
    def coherence_scale(self, concepts, batch: int = 64):
        """Text-independent 0-9 rating of how specific and observable each description is,
        as E[digit] / 9 (cached). High = one detectable behavior; low = vague or generic."""
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

    @torch.no_grad()
    def presence_grid(self, concepts, texts, text_batch: int = 32,
                      grain: str = "response", mode: str = "scale"):
        """Concept scores P [K, M] = score(description_k, text_m) for text descriptions. Each
        description is tokenized together with the rating question that follows it."""
        self._check_mode(grain, mode)
        if not len(concepts):
            return np.zeros((0, len(texts)), dtype=np.float32)
        blocks = [self.embed(torch.tensor(self._ids(c + self.SUF_SCALE), device=self.device)).detach()
                  for c in concepts]
        self.packed.text_batch, self.packed.temp = text_batch, 1.0
        return self.packed.scores(lambda k: blocks[k], len(concepts), list(texts)).float().cpu().numpy()

    def presence_grid_soft(self, concept_embeds, texts, text_batch: int = 32, temp: float = 1.0,
                           mode: str = "scale"):
        """Concept scores [K, M] for soft-prompt descriptions concept_embeds [K, T, E],
        differentiable in concept_embeds; temp divides the digit logits."""
        self._check_mode("response", mode)
        self.packed.text_batch = text_batch
        return SoftScores.apply(concept_embeds, self.packed, self._suffix_emb, list(texts), float(temp))
