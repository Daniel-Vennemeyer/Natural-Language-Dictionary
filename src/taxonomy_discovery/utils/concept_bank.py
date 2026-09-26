"""The pipeline's candidate concept bank, in one place.

~1024 concept phrases clustered from the extraction cache (falling back to the built-in
sycophancy list when no cache exists). Two callers need it and must provably get the SAME bank,
or the comparison between them is not a comparison:

  bank_topk.py      -- no-method ablation: rank the bank against the behavior LABEL, take top-K
  dir_baseline.py   -- --cand-bank: snap each DIRECTION to its nearest bank concept

Kept as a function rather than a cached constant because the extraction-cache bank depends on the
examples passed in.
"""
from __future__ import annotations

import hashlib
import warnings
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Built-in fallback bank, used when no extraction cache or bank file is available.
# ---------------------------------------------------------------------------
DEFAULT_SYCOPHANCY_CONCEPTS: tuple[str, ...] = (
    "praise of the user's competence or intelligence",
    "emotional validation of the user's feelings",
    "agreeing with the user to avoid conflict",
    "excessive apologizing or self-deprecation",
    "flattering the user's question as insightful",
    "softening criticism with indirect language",
    "accepting the user's framing without pushback",
    "reassuring the user that they are right",
    "hedging to avoid stating an unwelcome truth",
    "deferring to the user's stated opinion",
    "complimenting the user's writing or work",
    "expressing enthusiasm to please the user",
    "avoiding direct disagreement with a claim",
    "telling the user what they want to hear",
    "minimizing the user's mistakes",
    "encouraging the user regardless of merit",
    "mirroring the user's emotional tone",
    "offering unconditional support",
    "qualifying honest feedback with reassurance",
    "validating a morally questionable choice",
    "agreeing with a factually incorrect statement",
    "prioritizing politeness over accuracy",
    "gratitude or thanks directed at the user",
    "deflecting blame away from the user",
    "framing the user's behavior charitably",
    "reluctance to deliver bad news",
    "downplaying risks the user faces",
    "affirming the user's self-image",
    "concession to social pressure in the prompt",
    "indirect refusal that still pleases the user",
    "endorsing the user's plan without caveats",
    "neutral factual explanation (non-sycophantic)",
)


# ---------------------------------------------------------------------------
# Frozen text embedder e(c_j) in R^m, used only to cluster the extraction-cache
# vocabulary into `target_size` representative phrases.
# ---------------------------------------------------------------------------
class TextEmbedder:
    """Frozen text embedder mapping a phrase to ``R^m``.

    Backends:
      - ``hf``:   mean-pooled hidden states of a small HF encoder
                  (default ``sentence-transformers/all-MiniLM-L6-v2``, m=384).
      - ``hash``: deterministic char-3gram + word hashing into ``m`` dims.
                  Requires no network or model; used as a fallback when the HF
                  model cannot be loaded.
      - ``auto``: try ``hf``, fall back to ``hash``.
    """

    def __init__(
        self,
        backend: str = "auto",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hash_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        self.requested_backend = backend
        self.model_name = model_name
        self.hash_dim = int(hash_dim)
        self.device = device

        self.backend: str = "hash"
        self._tok = None
        self._model = None
        self.dim: int = self.hash_dim

        if backend in ("auto", "hf"):
            try:
                self._load_hf()
                self.backend = "hf"
                self.dim = int(self._model.config.hidden_size)
            except Exception as exc:  # pragma: no cover - depends on env
                if backend == "hf":
                    raise
                warnings.warn(
                    f"TextEmbedder: HF backend unavailable ({exc!r}); "
                    "falling back to deterministic hash embeddings.",
                    RuntimeWarning,
                )
                self.backend = "hash"
                self.dim = self.hash_dim
        elif backend == "hash":
            self.backend = "hash"
            self.dim = self.hash_dim
        else:
            raise ValueError(f"Unknown embedder backend: {backend!r}")

    def _load_hf(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()
        self._model.to(self.device)

    def embed(self, phrases: list[str]) -> np.ndarray:
        import torch

        with torch.no_grad():
            if self.backend == "hf":
                vecs = self._embed_hf(phrases)
            else:
                vecs = np.stack([self._embed_hash_one(p) for p in phrases], axis=0)
        # L2-normalize so cosine == dot and so concept magnitudes don't bias downstream use.
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
        return (vecs / norms).astype(np.float32)

    def _embed_hf(self, phrases: list[str]) -> np.ndarray:
        out: list[np.ndarray] = []
        bs = 64
        for i in range(0, len(phrases), bs):
            chunk = phrases[i : i + bs]
            enc = self._tok(
                chunk, padding=True, truncation=True, max_length=64, return_tensors="pt"
            ).to(self.device)
            hidden = self._model(**enc).last_hidden_state  # [b, t, m]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            out.append(pooled.float().cpu().numpy())
        return np.concatenate(out, axis=0)

    def _embed_hash_one(self, phrase: str) -> np.ndarray:
        v = np.zeros(self.hash_dim, dtype=np.float32)
        text = phrase.lower().strip()
        tokens = text.split()
        grams = tokens + [text[i : i + 3] for i in range(max(0, len(text) - 2))]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
            idx = h % self.hash_dim
            sign = 1.0 if (h // self.hash_dim) % 2 == 0 else -1.0
            v[idx] += sign
        return v


def _bank_from_extraction(examples: list, cfg: dict, target_size: int) -> list[str] | None:
    """Build a large named concept bank from the cached `extract` vocabulary.

    Loads the per-response phrases that an earlier extraction pass cached, then
    clusters the unique phrases into `target_size` representatives (named by
    their most frequent member). Returns None if no cache exists for this dataset.
    """
    from pathlib import Path
    from collections import Counter
    import json
    from taxonomy_discovery.utils.hashing import stable_hash

    max_ex = int(cfg.get("extract_max_examples", len(examples)))
    # Prefer the exact-key cache (same examples); else fall back to ANY cached
    # extraction in the dir -- the bank is just a global vocabulary of concept names,
    # so it doesn't need to row-match the current example set.
    texts = [(ex.response or ex.prompt or "")[:2000] for ex in examples[:max_ex]]
    cache_dir = Path(cfg.get("extraction_cache_dir", "outputs/caches/concept_extraction"))
    key = stable_hash({"texts": texts, "n": max_ex}, length=16)
    cache_file = cache_dir / f"{key}.json"
    if cache_file.exists():
        files = [cache_file]
    else:
        files = sorted(cache_dir.glob("*.json")) if cache_dir.exists() else []
        if files:
            print(f"[bank] exact extraction cache missed; using "
                  f"{len(files)} cached extraction file(s) in {cache_dir}", flush=True)
    if not files:
        return None
    all_phrases: list[str] = []
    for f in files:
        try:
            for ph in json.loads(f.read_text(encoding="utf-8")):
                all_phrases.extend(ph)
        except Exception:
            continue
    uniq = sorted(set(all_phrases))
    if len(uniq) < 2:
        return None
    if len(uniq) <= target_size:
        return uniq
    from sklearn.cluster import KMeans
    emb = TextEmbedder(
        backend=str(cfg.get("bank_cluster_backend", cfg.get("embedder_backend", "auto"))
                    if str(cfg.get("embedder_backend")) != "target_model" else "auto"),
        model_name=str(cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")),
        hash_dim=int(cfg.get("embedder_hash_dim", 256)),
    )
    E = emb.embed(uniq)
    labels = KMeans(n_clusters=target_size, random_state=int(cfg.get("seed", 0)), n_init=10).fit(E).labels_
    freq = Counter(all_phrases)
    names = []
    for c in range(target_size):
        members = [uniq[i] for i in range(len(uniq)) if labels[i] == c]
        names.append(max(members, key=lambda p: freq[p]) if members else f"cluster_{c}")
    return names


def build_bank(cfg: dict, examples: list, size: int) -> tuple[list[str], str]:
    """Return (concepts, source_description). Never raises: falls back to the built-in list.

    Priority: BANK_FILE env / cfg method.bank_file (one phrase per line -- e.g. a
    gen_bank.py domain bank) > extraction-cache clustering > built-in sycophancy list."""
    import os

    bf = os.environ.get("BANK_FILE") or (cfg.get("method", {}) or {}).get("bank_file")
    if bf:
        phrases = [ln.strip() for ln in open(bf, encoding="utf-8")
                   if ln.strip() and not ln.startswith("#")]
        if phrases:
            return phrases[:size], f"bank file {bf} ({len(phrases)} phrases)"
        print(f"[bank] bank file {bf} empty -- falling through", flush=True)
    mcfg: dict[str, Any] = dict(cfg.get("method", {}) or {})
    mcfg.setdefault("vocab_size", size)
    try:
        b = _bank_from_extraction(examples, mcfg, size)
    except Exception as e:  # noqa: BLE001
        print(f"[bank] extraction-cache bank unavailable ({e})", flush=True)
        b = None
    if b:
        return list(b), "extraction cache"
    return list(DEFAULT_SYCOPHANCY_CONCEPTS), "built-in default list"
