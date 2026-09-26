from __future__ import annotations

"""Language Sparse Autoencoder (Language SAE).

A sparse autoencoder whose dictionary atoms are *constrained to be the image of
a natural-language phrase*. Where a standard SAE learns free decoder vectors
``d_j in R^d``, the Language SAE parameterizes each atom as

    d_j = g_theta(c_j) = W e(c_j)        with  c_j a short phrase,
                                                e(.) a frozen text embedder (R^m),
                                                W in R^{d x m} learned.

For a (pooled) residual-stream activation ``h`` at layer ``l`` an encoder
``E_phi : R^d -> R_{>=0}^K`` produces sparse nonnegative concept activations
``a = E_phi(h)``, and the reconstruction is

    h_hat = b + sum_j a_j d_j = b + sum_j a_j W e(c_j).

Training minimizes

    sum_{x,t} || h_{x,t} - h_hat_{x,t} ||^2
      + lambda * sum_{x,t} || a_{x,t} ||_1
      + Omega(C),

with the optional language-dictionary regularizer

    Omega(C) = alpha * sum_j len(c_j) + beta * sum_{i != j} cos(e(c_i), e(c_j)).

Because the concept strings ``C`` are discrete, the inner optimization (over the
continuous parameters phi, W, b) holds ``C`` fixed; ``Omega(C)`` is then a
property of the dictionary and is used to *seed / prune / refresh* concepts in
an optional outer loop rather than entering the gradient. With ``C`` fixed,
gradient training optimizes reconstruction + L1 exactly as written.

After training, the dictionary is **not** assumed to be the taxonomy. We run a
behavior-specific analysis: for each concept ``j`` we compute its mean activation
over positive (``mu_j^+``) and negative (``mu_j^-``) examples, the class
separation ``Delta_j = mu_j^+ - mu_j^-``, and select the top concepts as the
candidate subbehaviors. The selected ``d_j`` are emitted as a ``BasisResult`` so
the existing intervention / selectivity / fuzzing harness performs the causal
test unchanged (steer ``h <- h + alpha d_j``).

The harness supplies one pooled activation per example (``[n, d]``), so the
per-token sum collapses to one term per example (``T_x = 1``). The code is
written to also accept a precomputed ``[n, K]``-agnostic token axis if a
token-level store is ever wired in (each row is treated as an independent
activation), so nothing here assumes ``T_x = 1`` beyond the input shape.
"""

import hashlib
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from taxonomy_discovery.core.base_method import BaseMethod
from taxonomy_discovery.core.types import ActivationBatch, BasisResult, BehaviorExample


# ---------------------------------------------------------------------------
# Default concept bank
# ---------------------------------------------------------------------------
# Used when no `concepts` are given in the config and LLM auto-seeding is off,
# so the method runs out-of-the-box on the sycophancy family. These are seed
# phrases only; the behavior analysis decides which actually mediate behavior.
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
# Text embedder  e(c_j) in R^m
# ---------------------------------------------------------------------------
class TextEmbedder:
    """Frozen text embedder mapping a phrase to ``R^m``.

    Backends:
      - ``hf``:   mean-pooled hidden states of a small HF encoder
                  (default ``sentence-transformers/all-MiniLM-L6-v2``, m=384).
      - ``hash``: deterministic char-3gram + word hashing into ``m`` dims.
                  Requires no network or model; used for tests/offline and as a
                  fallback when the HF model cannot be loaded.
      - ``auto``: try ``hf``, fall back to ``hash``.

    A real (semantic) embedder matters: with a one-hot-per-concept embedding the
    map ``W e(c_j)`` would degenerate to a free decoder column (a standard SAE).
    Sharing a semantic ``e(.)`` across concepts is what makes the phrase
    constrain the atom and lets ``W`` generalize across phrases.
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

    @torch.no_grad()
    def embed(self, phrases: list[str]) -> np.ndarray:
        if self.backend == "hf":
            vecs = self._embed_hf(phrases)
        else:
            vecs = np.stack([self._embed_hash_one(p) for p in phrases], axis=0)
        # L2-normalize so cosine == dot and so concept magnitudes don't bias W.
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

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "requested_backend": self.requested_backend,
            "model_name": self.model_name if self.backend == "hf" else None,
            "dim": int(self.dim),
        }


# ---------------------------------------------------------------------------
# Language SAE module
# ---------------------------------------------------------------------------
class LanguageSAE(nn.Module):
    """Encoder ``E_phi`` + language-parameterized decoder ``d_j = W e(c_j)``.

        a      = relu(W_enc h + b_enc)            in R_{>=0}^K   (optional Top-K)
        D      = E W^T                            in R^{K x d}   (decoder atoms)
        h_hat  = b + a @ D
    """

    def __init__(
        self,
        input_dim: int,
        concept_embeddings: np.ndarray,  # [K, m], frozen
        encoder_hidden: int = 0,
        top_k: int = 0,
        free_decoder: bool = False,
        freeze_W: bool = False,
        w_rank: int = 0,
        projection_encoder: bool = False,
        fixed_atoms: bool = False,
        adapter: str = "none",
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_concepts, self.embed_dim = concept_embeddings.shape
        self.top_k = int(top_k)
        self.free_decoder = bool(free_decoder)
        self.fixed_atoms = bool(fixed_atoms)
        self.w_rank = int(w_rank)
        self.projection_encoder = bool(projection_encoder)

        if self.fixed_atoms and self.embed_dim != self.input_dim:
            raise ValueError(
                "fixed_atoms requires concept atoms in activation space "
                f"(embed_dim {self.embed_dim} != input_dim {self.input_dim}). "
                "Use embedder_backend='target_model'."
            )

        # Frozen concept embeddings e(c_j). For fixed_atoms these ARE the dictionary
        # atoms (the target model's own layer-l representation of each phrase).
        self.register_buffer(
            "concept_embeddings",
            torch.tensor(np.asarray(concept_embeddings), dtype=torch.float32),
        )

        # --- decoder parameterization -----------------------------------------
        # fixed       : d_j = h^(l)(c_j), the target model's own rep of the phrase;
        #               NOTHING in the decoder is learned (the strongest, most
        #               NLA-like form). Optional per-atom scale adapter.
        # free        : atoms are unconstrained learned vectors (no language).
        # full W      : d_j = W e(c_j), W in R^{d x m} (default).
        # low-rank W  : W = U V; constrains W so atoms inherit embedding geometry.
        # freeze_W    : W (or U,V) is fixed at init and never trained.
        self.W = None
        self.U = None
        self.V = None
        self.free_atoms = None
        self.atom_scale = None
        if self.fixed_atoms:
            if str(adapter) == "scale":
                # small learned adapter: a positive per-atom scale.
                self.atom_scale = nn.Parameter(torch.ones(self.num_concepts))
        elif self.free_decoder:
            self.free_atoms = nn.Parameter(
                torch.randn(self.num_concepts, self.input_dim) / (self.input_dim ** 0.5)
            )
        elif self.w_rank and self.w_rank > 0:
            r = int(self.w_rank)
            self.U = nn.Parameter(torch.empty(self.input_dim, r))
            self.V = nn.Parameter(torch.empty(r, self.embed_dim))
        else:
            self.W = nn.Linear(self.embed_dim, self.input_dim, bias=False)

        # --- encoder E_phi -----------------------------------------------------
        # projection encoder ties coefficients to the decoded directions:
        #   a_j = relu( scale_j * <h - center, dhat_j> + bias_j )
        # so a concept only fires when h points along ITS atom. Otherwise a free
        # linear/MLP encoder is used.
        if self.projection_encoder:
            self.enc_center = nn.Parameter(torch.zeros(self.input_dim))
            self.enc_scale = nn.Parameter(torch.ones(self.num_concepts))
            self.enc_bias = nn.Parameter(torch.zeros(self.num_concepts))
            self.encoder = None
        elif encoder_hidden and encoder_hidden > 0:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, encoder_hidden),
                nn.GELU(),
                nn.Linear(encoder_hidden, self.num_concepts),
            )
        else:
            self.encoder = nn.Linear(self.input_dim, self.num_concepts)

        # Reconstruction bias b in R^d.
        self.b = nn.Parameter(torch.zeros(self.input_dim))

        self.reset_parameters()

        if freeze_W and not self.free_decoder and not self.fixed_atoms:
            for p in self._decoder_params():
                p.requires_grad_(False)

    def _decoder_params(self) -> list[nn.Parameter]:
        if self.fixed_atoms:
            return [self.atom_scale] if self.atom_scale is not None else []
        if self.free_decoder:
            return [self.free_atoms]
        if self.W is not None:
            return [self.W.weight]
        return [self.U, self.V]

    def reset_parameters(self) -> None:
        if self.W is not None:
            nn.init.xavier_uniform_(self.W.weight)
        if self.U is not None:
            nn.init.xavier_uniform_(self.U)
            nn.init.xavier_uniform_(self.V)

    def effective_W(self) -> torch.Tensor | None:
        """The d x m map (full or low-rank). None for free / fixed-atom decoders."""
        if self.free_decoder or self.fixed_atoms:
            return None
        if self.W is not None:
            return self.W.weight
        return self.U @ self.V

    def decoder_atoms(self) -> torch.Tensor:
        """Decoder atoms [K, d].

        fixed : d_j = unit(h^(l)(c_j)), the model's own rep (optional learned scale).
        free  : unit-norm learned vectors.
        else  : d_j = W e(c_j).
        """
        if self.fixed_atoms:
            D = self.concept_embeddings
            Dn = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
            return self.atom_scale.unsqueeze(1) * Dn if self.atom_scale is not None else Dn
        if self.free_decoder:
            D = self.free_atoms
            return D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.concept_embeddings @ self.effective_W().t()

    def _apply_topk(self, a: torch.Tensor) -> torch.Tensor:
        if self.top_k <= 0 or self.top_k >= a.shape[-1]:
            return a
        vals, idx = torch.topk(a, k=self.top_k, dim=-1)
        out = torch.zeros_like(a)
        out.scatter_(-1, idx, vals)
        return out

    def encode(self, h: torch.Tensor) -> torch.Tensor:
        if self.projection_encoder:
            D = self.decoder_atoms()
            Dhat = D / D.norm(dim=1, keepdim=True).clamp_min(1e-8)
            proj = (h - self.enc_center) @ Dhat.t()  # [., K] alignment of h with each atom
            a = torch.relu(self.enc_scale * proj + self.enc_bias)
        else:
            a = torch.relu(self.encoder(h))  # nonnegative
        return self._apply_topk(a)

    def decode(self, a: torch.Tensor, atoms: torch.Tensor | None = None) -> torch.Tensor:
        D = self.decoder_atoms() if atoms is None else atoms
        return self.b + a @ D

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = self.encode(h)
        atoms = self.decoder_atoms()
        h_hat = self.b + a @ atoms
        return h_hat, a


@dataclass
class _FitStats:
    best_val_loss: float
    epochs_trained: int
    train_size: int
    val_size: int
    final_recon: float
    final_l1: float


# ---------------------------------------------------------------------------
# Method
# ---------------------------------------------------------------------------
class LanguageSAEMethod(BaseMethod):
    """Language SAE basis-discovery method.

    Config fields (with defaults):
        name: language_sae
        dictionary_size: 32          # K (number of concepts) when auto-seeding
        k: 4                         # number of behavior-mediating concepts to select
        concepts: [..]               # optional explicit phrases (overrides K)
        auto_seed_concepts: false    # if true and no `concepts`, seed via judge LLM
        selection_metric: delta      # delta | effect_size | classifier_weight
        lambda_l1: 0.001             # sparsity coefficient
        alpha_len: 0.0               # Omega: brevity penalty (per-word)
        beta_cos: 0.0                # Omega: pairwise concept-embedding cosine
        decoder_orth_weight: 0.0     # optional differentiable atom-orthogonality
        encoder_hidden: 0            # 0 => linear encoder; >0 => MLP width
        top_k: 0                     # 0 => pure-L1 sparsity (no Top-K gating)
        embedder_backend: auto       # auto | hf | hash
        embedder_model: sentence-transformers/all-MiniLM-L6-v2
        embedder_hash_dim: 256
        lr: 0.001
        max_epochs: 300
        patience: 15
        batch_size: 512
        val_fraction: 0.1
        normalize_directions: true   # unit-norm emitted directions (for steering)
        device: cuda
        seed: 0
    """

    def fit(
        self,
        examples: list[BehaviorExample],
        activations: ActivationBatch,
    ) -> BasisResult:
        X_np = np.asarray(activations.activations)
        if X_np.ndim != 2:
            raise ValueError(f"Expected activations [n, d], got {X_np.shape}")
        n_examples, input_dim = X_np.shape
        if n_examples < 10:
            raise ValueError("LanguageSAE needs more than a handful of examples.")

        cfg = self.config
        self.verbose = bool(cfg.get("verbose", True))
        seed = int(cfg.get("seed", 0))
        device = self._resolve_device(cfg.get("device", "cuda"))
        torch.manual_seed(seed)
        np.random.seed(seed)

        self._log("=" * 72)
        self._log("Language SAE  —  h_hat = b + sum_j a_j W e(c_j)")
        self._log("=" * 72)
        self._log(
            f"activations : n={n_examples}  d={input_dim}  "
            f"(layer {activations.metadata.get('layer')}, "
            f"model {activations.metadata.get('model_name')})"
        )

        # Per-example L2 norm BEFORE any normalization (used for the confound check).
        raw_norms = np.linalg.norm(X_np, axis=1)
        X_raw = X_np.copy()  # truly-raw activations for the probe ceiling (pre-normalize)

        # Optionally unit-normalize each activation. Concept activations a_j scale
        # with ||h||, so without this the behavior analysis (Delta_j) is confounded
        # by per-example magnitude (length/topic) rather than direction. Matches the
        # NLA convention of normalizing activations to unit L2 norm.
        normalize_input = bool(cfg.get("normalize_input", True))
        if normalize_input:
            X_np = (X_np / (raw_norms[:, None] + 1e-8)).astype(np.float32)

        # Total variance (sum of per-dim variance) = mean ||h - mean||^2.
        # Used as the reconstruction baseline / denominator for FVE.
        self._total_var = float(np.sum((X_np - X_np.mean(axis=0)) ** 2, axis=1).mean())
        self._log(
            f"input norm  : mean||h||={raw_norms.mean():.2f}  "
            f"normalize_input={normalize_input}"
            + ("  -> unit-norm activations" if normalize_input else "  -> RAW (magnitude may confound Delta)")
        )
        self._log(
            f"baseline    : total_var(mean ||h-mean||^2)={self._total_var:.4f} "
            "(recon must beat this)"
        )

        # --- align labels to activation row order -------------------------------
        y, has_labels = self._aligned_labels(examples, activations)
        n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
        self._log(
            f"labels      : has_labels={has_labels}  n_pos={n_pos}  n_neg={n_neg}"
            + ("" if has_labels else "  -> selecting by mean activation (unlabeled fallback)")
        )
        if has_labels and n_pos and n_neg:
            # If raw norms differ by class, an un-normalized run will show a global
            # Delta sign driven by magnitude, not behavior.
            self._log(
                f"by class    : mean||h_raw|| pos={raw_norms[y == 1].mean():.2f} "
                f"neg={raw_norms[y == 0].mean():.2f} "
                f"(large gap + normalize_input=false => magnitude confound)"
            )

        # --- raw-probe ceiling -------------------------------------------------
        # Held-out AUC of a logistic probe on the full d-dim activations. This is
        # the upper bound on how much behavior any SAE built on these activations
        # can recover: if this is ~0.5 the behavior is not linearly present here
        # (layer/pooling/label problem) and no dictionary will separate it; if it
        # is high but the SAE's selected_auc is low, the SAE is discarding signal.
        # Probe the TRULY-raw activations (pre-normalize) so a magnitude-borne
        # signal counts toward the ceiling.
        self._raw_probe_auc = self._raw_probe_auc_fn(X_raw, y, has_labels, seed)
        if has_labels:
            self._log(
                f"probe ceil  : held-out AUC from raw {input_dim}-dim activations = "
                f"{self._raw_probe_auc:.3f}  (upper bound for any SAE on this site)"
            )

        # --- ablation mode -----------------------------------------------------
        # none           : Language SAE (atoms = W e(c_j), real phrases, real names)
        # random_phrase  : atoms = W e(c_j) for semantically irrelevant phrases
        # shuffled_label : trained exactly like Language SAE (real phrases) but the
        #                  emitted names are a random permutation -> faithfulness control
        # free           : atoms are freely learned (no language bottleneck), same K
        ablation = str(cfg.get("ablation", "none")).lower()
        if ablation not in {"none", "random_phrase", "shuffled_label", "free"}:
            raise ValueError(f"Unknown ablation: {ablation!r}")

        # --- resolve concept dictionary C --------------------------------------
        real_concepts = self._resolve_concepts(examples, cfg)
        K = len(real_concepts)
        n_select = int(cfg.get("k", min(4, K)))
        n_select = max(1, min(n_select, K))

        # Concepts actually embedded for the decoder.
        if ablation == "random_phrase":
            concepts = self._random_phrases(K, seed)
        else:
            concepts = list(real_concepts)  # none / shuffled_label / free

        self._log(f"ABLATION    : {ablation}")
        src = (
            "explicit cfg" if cfg.get("concepts")
            else ("LLM-seeded" if cfg.get("auto_seed_concepts") else "default bank")
        )
        self._log(f"dictionary  : K={K} concepts ({src}); selecting k={n_select} by "
                  f"'{cfg.get('selection_metric', 'delta')}'")
        if self.verbose:
            for j, c in enumerate(concepts):
                print(f"[language_sae]   c[{j:>2}] = {c}", flush=True)

        # --- embed concepts  e(c_j) --------------------------------------------
        backend = str(cfg.get("embedder_backend", "auto"))
        # target_model: the atom for concept c_j IS the target model's own layer-l
        # representation of the phrase, d_j = h^(l)(c_j), living in the SAME space as
        # the activations we reconstruct. No projection W is learned -> the decoder is
        # anchored to the model's understanding of the phrase (the strongest, most
        # NLA-like form). The free ablation never uses target atoms (it learns them).
        use_target = backend == "target_model" and ablation != "free"
        if use_target:
            E = self._embed_concepts_via_target_model(concepts, activations, cfg, device)
            embedder_info = {
                "backend": "target_model",
                "model_name": activations.metadata["model_name"],
                "layer": activations.metadata.get("layer"),
                "token_selector": activations.metadata.get("token_selector"),
                "dim": int(E.shape[1]),
            }
            embed_dim = int(E.shape[1])
        else:
            embedder = TextEmbedder(
                backend=backend if backend != "target_model" else "auto",
                model_name=str(
                    cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")
                ),
                hash_dim=int(cfg.get("embedder_hash_dim", 256)),
                device="cuda" if (device.type == "cuda") else "cpu",
            )
            E = embedder.embed(list(concepts))  # [K, m], L2-normalized
            embedder_info = embedder.info()
            embed_dim = int(embedder.dim)

        # Mean pairwise cosine quantifies how distinct the atoms/embeddings are.
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        G = En @ En.T
        mean_cos = float((G.sum() - np.trace(G)) / max(K * (K - 1), 1))
        self._log(
            f"embedder    : backend={embedder_info['backend']} m={embed_dim}  "
            f"mean_pairwise_cos={mean_cos:.3f}"
            + ("  (atoms = target-model concept reps)" if use_target else "")
        )

        # --- build + train the SAE ---------------------------------------------
        w_rank = int(cfg.get("w_rank", 0))
        # target_model atoms are FIXED (d_j = h^(l)(c_j) verbatim) UNLESS a learned
        # low-rank bridge is requested (w_rank>0): then d_j = W e(c_j) with W a learned
        # rank-r map applied to the model-rep embeddings. A small r bridges the
        # phrase->transcript distribution gap while staying too rank-limited to place
        # atoms independently (so real vs gibberish concepts stay distinguishable).
        target_learned_bridge = use_target and w_rank > 0
        fixed_atoms = (use_target and not target_learned_bridge) or bool(cfg.get("fixed_atoms", False))
        freeze_W = bool(cfg.get("freeze_W", False))
        projection_encoder = bool(cfg.get("projection_encoder", False))
        adapter = str(cfg.get("adapter", "none"))
        model = LanguageSAE(
            input_dim=input_dim,
            concept_embeddings=E,
            encoder_hidden=int(cfg.get("encoder_hidden", 0)),
            top_k=int(cfg.get("top_k", 0)),
            free_decoder=(ablation == "free"),
            freeze_W=freeze_W,
            w_rank=w_rank,
            projection_encoder=projection_encoder,
            fixed_atoms=fixed_atoms,
            adapter=adapter,
        ).to(device)
        # Initialize reconstruction bias b to the data mean (stabilizes early steps).
        with torch.no_grad():
            mean_vec = torch.tensor(X_np.mean(axis=0), dtype=torch.float32, device=device)
            model.b.copy_(mean_vec)
            if projection_encoder:
                model.enc_center.copy_(mean_vec)
        if projection_encoder:
            enc_kind = "projection (a_j = relu(s_j<h,dhat_j>+t_j))"
        elif int(cfg.get("encoder_hidden", 0)):
            enc_kind = f"MLP(h={cfg.get('encoder_hidden')})"
        else:
            enc_kind = "linear"
        if ablation == "free":
            decoder_desc = "free atoms (no language)"
        elif fixed_atoms:
            decoder_desc = "FIXED atoms = h^(l)(c_j), no W" + (
                f" + {adapter} adapter" if adapter != "none" else " (coeffs only)"
            )
        elif w_rank > 0:
            decoder_desc = f"W=U V rank={w_rank} [{input_dim}x{embed_dim}]"
        else:
            decoder_desc = f"W:[{input_dim}x{embed_dim}]"
        if freeze_W and ablation != "free" and not fixed_atoms:
            decoder_desc += " FROZEN"
        self._log(
            f"model       : encoder={enc_kind}  decoder={decoder_desc}  "
            f"top_k={int(cfg.get('top_k', 0)) or 'off (L1)'}  "
            f"lambda_l1={float(cfg.get('lambda_l1', 1e-3))}"
        )
        self._log("-" * 72)
        self._log("training (recon = mean sum-sq error; FVE = 1 - recon/total_var):")

        stats = self._train(model, X_np, cfg, device, seed)
        self._log(
            f"trained     : best_val_recon={stats.best_val_loss:.4f} "
            f"FVE={1.0 - stats.best_val_loss / max(self._total_var, 1e-8):.3f} "
            f"@ epoch {stats.epochs_trained}  (final L1/example={stats.final_l1:.3f})"
        )

        # --- behavior-specific analysis ----------------------------------------
        model.eval()
        with torch.inference_mode():
            X = torch.tensor(X_np, dtype=torch.float32, device=device)
            A = model.encode(X).detach().cpu().numpy()  # [n, K], graded activations
            atoms = model.decoder_atoms().detach().cpu().numpy()  # [K, d]
            W_eff = model.effective_W()
            W = W_eff.detach().cpu().to(torch.float32).numpy() if W_eff is not None else None
            b = model.b.detach().cpu().to(torch.float32).numpy()

        analysis = self._behavior_analysis(A, y, has_labels, concepts)
        if has_labels and n_pos and n_neg:
            sum_a_pos = float(A[y == 1].sum(axis=1).mean())
            sum_a_neg = float(A[y == 0].sum(axis=1).mean())
            n_dead = int((analysis["firing_rate"] < float(cfg.get("min_firing", 0.02))).sum())
            self._log(
                f"activity    : mean sum_j a_j  pos={sum_a_pos:.2f} neg={sum_a_neg:.2f}  "
                f"dead_concepts={n_dead}/{K} (fire<{float(cfg.get('min_firing', 0.02))})"
            )
        selected = self._select_concepts(analysis, n_select, cfg, has_labels)

        # Held-out predictivity of the SELECTED subbehaviors: do the k chosen
        # concept activations compose to predict the behavior? Directly comparable
        # across ablations (language / random / free).
        selected_auc = self._selected_auc(A, y, selected, has_labels, seed)

        # --- per-mode names ----------------------------------------------------
        # The atoms are identical for none/shuffled_label (same training); only the
        # labels differ. This makes shuffled_label a pure faithfulness control.
        label_permutation: list[int] | None = None
        if ablation == "free":
            names_full = [f"free_atom_{j}" for j in range(K)]
        elif ablation == "shuffled_label":
            perm = np.random.default_rng(seed + 1).permutation(K)
            label_permutation = [int(p) for p in perm]
            names_full = [real_concepts[perm[j]] for j in range(K)]
        else:  # none -> real concepts; random_phrase -> random phrases
            names_full = list(concepts)

        self._log_concept_table(names_full, analysis, selected, has_labels)
        if has_labels:
            self._log(f"selected_auc: held-out AUC from k selected concepts = {selected_auc:.3f}")

        # --- emit directions for the causal-test harness -----------------------
        directions = atoms[selected].astype(np.float32)  # [k, d]
        if bool(cfg.get("normalize_directions", True)):
            dn = np.linalg.norm(directions, axis=1, keepdims=True) + 1e-8
            directions = directions / dn
        direction_names = [names_full[j] for j in selected]

        omega = self._omega(concepts, E, cfg)
        self._log(
            f"omega       : length={omega['length']:.3f}  "
            f"cosine_redundancy={omega['cosine']:.3f}"
        )
        self._log(
            f"emitted     : {len(selected)} directions [{directions.shape}] "
            f"(unit-norm={bool(cfg.get('normalize_directions', True))}) "
            "-> BasisResult for the intervention/selectivity harness"
        )
        self._log("=" * 72)

        return BasisResult(
            method_name="language_sae",
            behavior_family=examples[0].behavior_family,
            train_dataset=examples[0].dataset_name,
            model_name=activations.metadata["model_name"],
            layer=int(activations.metadata["layer"]),
            directions=directions,
            direction_names=direction_names,
            metadata={
                "n_examples": int(n_examples),
                "input_dim": int(input_dim),
                "num_concepts": int(K),
                "n_directions": int(len(selected)),
                "ablation": ablation,
                "label_permutation": label_permutation,
                "selected_concept_indices": [int(i) for i in selected],
                "selected_concepts": direction_names,
                "concepts": list(concepts),
                "selection_metric": str(cfg.get("selection_metric", "delta")),
                "has_labels": bool(has_labels),
                # held-out behavior predictivity of the selected subbehaviors
                "selected_auc": float(selected_auc),
                "raw_probe_auc": float(getattr(self, "_raw_probe_auc", float("nan"))),
                "freeze_W": bool(cfg.get("freeze_W", False)),
                "w_rank": int(cfg.get("w_rank", 0)),
                "projection_encoder": bool(cfg.get("projection_encoder", False)),
                "fixed_atoms": bool(fixed_atoms),
                "atom_source": "target_model" if use_target else "text_embedder",
                "adapter": str(cfg.get("adapter", "none")),
                "total_var": float(self._total_var),
                "fve": float(1.0 - stats.best_val_loss / max(self._total_var, 1e-8)),
                # per-concept behavior statistics (length K, original concept order)
                "mu_pos": analysis["mu_pos"].tolist(),
                "mu_neg": analysis["mu_neg"].tolist(),
                "delta": analysis["delta"].tolist(),
                "effect_size": analysis["effect_size"].tolist(),
                "classifier_weight": analysis["classifier_weight"].tolist(),
                "mean_activation": analysis["mean_activation"].tolist(),
                "firing_rate": analysis["firing_rate"].tolist(),
                # dictionary regularizer terms (reported; see Omega docstring)
                "omega_length": float(omega["length"]),
                "omega_cosine_redundancy": float(omega["cosine"]),
                # training diagnostics
                "best_val_loss": float(stats.best_val_loss),
                "epochs_trained": int(stats.epochs_trained),
                "train_size": int(stats.train_size),
                "val_size": int(stats.val_size),
                "final_recon_mse": float(stats.final_recon),
                "final_l1": float(stats.final_l1),
                "embedder": embedder_info,
                # parameters for reproducibility / downstream re-encoding
                "concept_embeddings": E.tolist(),
                "W": (W.tolist() if W is not None else None),
                "decoder_bias": b.tolist(),
            },
            fit_config={**cfg, "resolved_k": int(n_select), "resolved_K": int(K)},
        )

    # ------------------------------------------------------------------ training
    def _train(
        self,
        model: LanguageSAE,
        X_np: np.ndarray,
        cfg: dict,
        device: torch.device,
        seed: int,
    ) -> _FitStats:
        lambda_l1 = float(cfg.get("lambda_l1", 1e-3))
        orth_w = float(cfg.get("decoder_orth_weight", 0.0))
        lr = float(cfg.get("lr", 1e-3))
        max_epochs = int(cfg.get("max_epochs", 300))
        patience = int(cfg.get("patience", 15))
        batch_size = int(cfg.get("batch_size", 512))
        val_fraction = float(cfg.get("val_fraction", 0.1))

        X = torch.tensor(X_np, dtype=torch.float32)
        dataset = TensorDataset(X)
        val_size = max(1, int(round(len(dataset) * val_fraction)))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("val_fraction too large for dataset size")
        gen = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=gen)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        # Only optimize trainable params (a frozen W has requires_grad=False).
        opt = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad], lr=lr
        )
        print_every = int(cfg.get("print_every", max(1, max_epochs // 15)))
        total_var = float(getattr(self, "_total_var", 1.0))

        best_val = float("inf")
        best_state = None
        best_epoch = -1
        stale = 0
        last_recon = float("nan")
        last_l1 = float("nan")

        for epoch in range(max_epochs):
            model.train()
            for (xb,) in train_loader:
                xb = xb.to(device)
                opt.zero_grad(set_to_none=True)
                h_hat, a = model(xb)
                recon = torch.mean(torch.sum((xb - h_hat) ** 2, dim=-1))
                l1 = torch.mean(torch.sum(a.abs(), dim=-1))
                loss = recon + lambda_l1 * l1
                if orth_w > 0.0:
                    loss = loss + orth_w * self._atom_orth_penalty(model.decoder_atoms())
                loss.backward()
                opt.step()

            # validation (reconstruction MSE only -> comparable across lambda)
            model.eval()
            v_sum, v_cnt, recon_sum, l1_sum, active_sum = 0.0, 0, 0.0, 0.0, 0.0
            with torch.inference_mode():
                for (xb,) in val_loader:
                    xb = xb.to(device)
                    h_hat, a = model(xb)
                    recon = torch.mean(torch.sum((xb - h_hat) ** 2, dim=-1))
                    l1 = torch.mean(torch.sum(a.abs(), dim=-1))
                    bs = xb.shape[0]
                    v_sum += float(recon.item()) * bs
                    recon_sum += float(recon.item()) * bs
                    l1_sum += float(l1.item()) * bs
                    active_sum += float((a > 0).float().sum(-1).mean().item()) * bs
                    v_cnt += bs
            val_loss = v_sum / max(v_cnt, 1)
            last_recon = recon_sum / max(v_cnt, 1)
            last_l1 = l1_sum / max(v_cnt, 1)
            active = active_sum / max(v_cnt, 1)

            improved = val_loss < best_val - 1e-8
            if improved:
                best_val = val_loss
                best_epoch = epoch
                stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1

            if self.verbose and (epoch % print_every == 0 or improved and epoch < 5):
                fve = 1.0 - val_loss / max(total_var, 1e-8)
                print(
                    f"[language_sae]   epoch {epoch:>4}  "
                    f"val_recon={val_loss:8.4f}  FVE={fve:6.3f}  "
                    f"L1/ex={last_l1:7.3f}  active/ex={active:5.2f}"
                    f"{'  *best' if improved else ''}",
                    flush=True,
                )

            if stale >= patience:
                if self.verbose:
                    print(f"[language_sae]   early stop at epoch {epoch} "
                          f"(no val improvement for {patience} epochs)", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)

        return _FitStats(
            best_val_loss=float(best_val),
            epochs_trained=int(best_epoch + 1),
            train_size=int(train_size),
            val_size=int(val_size),
            final_recon=float(last_recon),
            final_l1=float(last_l1),
        )

    @staticmethod
    def _atom_orth_penalty(atoms: torch.Tensor) -> torch.Tensor:
        """Mean squared off-diagonal cosine between decoder atoms (differentiable in W)."""
        D = atoms / (atoms.norm(dim=1, keepdim=True) + 1e-8)
        G = D @ D.t()
        K = G.shape[0]
        off = G - torch.diag(torch.diag(G))
        return (off ** 2).sum() / max(K * (K - 1), 1)

    # ----------------------------------------------------------------- analysis
    @staticmethod
    def _behavior_analysis(
        A: np.ndarray,
        y: np.ndarray,
        has_labels: bool,
        concepts: list[str],
    ) -> dict[str, np.ndarray]:
        K = A.shape[1]
        mean_activation = A.mean(axis=0)
        firing_rate = (A > 0).mean(axis=0)

        if has_labels:
            pos = A[y == 1]
            neg = A[y == 0]
            mu_pos = pos.mean(axis=0) if len(pos) else np.zeros(K)
            mu_neg = neg.mean(axis=0) if len(neg) else np.zeros(K)
            delta = mu_pos - mu_neg
            pooled_std = np.sqrt(
                0.5 * (A[y == 1].var(axis=0, ddof=0) + A[y == 0].var(axis=0, ddof=0))
            ) + 1e-8
            effect_size = delta / pooled_std
            classifier_weight = LanguageSAEMethod._logreg_weights(A, y)
        else:
            mu_pos = mean_activation.copy()
            mu_neg = np.zeros(K)
            delta = mean_activation.copy()
            effect_size = np.zeros(K)
            classifier_weight = np.zeros(K)

        return {
            "mu_pos": mu_pos,
            "mu_neg": mu_neg,
            "delta": delta,
            "effect_size": effect_size,
            "classifier_weight": classifier_weight,
            "mean_activation": mean_activation,
            "firing_rate": firing_rate,
        }

    @staticmethod
    def _logreg_weights(A: np.ndarray, y: np.ndarray) -> np.ndarray:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler

            Az = StandardScaler().fit_transform(A)
            clf = LogisticRegression(max_iter=1000, C=1.0)
            clf.fit(Az, y)
            return clf.coef_.reshape(-1).astype(np.float32)
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"classifier_weight unavailable ({exc!r}); zeros used.", RuntimeWarning)
            return np.zeros(A.shape[1], dtype=np.float32)

    @staticmethod
    def _select_concepts(
        analysis: dict[str, np.ndarray],
        n_select: int,
        cfg: dict,
        has_labels: bool,
    ) -> list[int]:
        """Rank concepts by the chosen separation metric and take the top ``k``.

        Dead concepts (firing rate below ``min_firing``) are deprioritized: the
        encoder never activates them, so their decoder atom does not reflect any
        observed behavior and selecting them is meaningless. A behavior can be
        signaled by *elevated* or *suppressed* activation, so the ``abs_*``
        metrics rank by separation magnitude regardless of sign.
        """
        metric = str(cfg.get("selection_metric", "delta"))
        delta = analysis["delta"]
        eff = analysis["effect_size"]
        cw = analysis["classifier_weight"]
        table = {
            "delta": delta,
            "effect_size": eff,
            "classifier_weight": cw,
            "abs_delta": np.abs(delta),
            "abs_effect_size": np.abs(eff),
            "abs_classifier_weight": np.abs(cw),
        }
        scores = analysis["mean_activation"] if not has_labels else table.get(metric, delta)

        # Deprioritize dead concepts: rank live ones first, then dead as backfill.
        min_firing = float(cfg.get("min_firing", 0.02))
        live = analysis["firing_rate"] >= min_firing
        order = list(np.argsort(scores)[::-1])
        live_order = [int(i) for i in order if live[i]]
        dead_order = [int(i) for i in order if not live[i]]
        ranked = live_order + dead_order
        return ranked[:n_select]

    # Neutral vocabulary for the random-phrase ablation: real, fluent English that
    # is unrelated to any target behavior. This isolates concept *meaning* (vs. the
    # mere fact of being text) from the W e(.) bottleneck structure.
    _RANDOM_VOCAB: tuple[str, ...] = (
        "copper", "valley", "lantern", "quiet", "harbor", "gravel", "maple", "drifting",
        "bicycle", "marble", "thunder", "orchard", "velvet", "compass", "meadow", "kettle",
        "glacier", "ribbon", "cinnamon", "pebble", "willow", "anchor", "ceramic", "twilight",
        "saffron", "boulder", "lantern", "nimble", "ember", "trellis", "wander", "amber",
        "canyon", "satchel", "linen", "murmur", "thistle", "beacon", "cobalt", "driftwood",
    )

    def _embed_concepts_via_target_model(
        self,
        concepts: list[str],
        activations: ActivationBatch,
        cfg: dict,
        device: torch.device,
    ) -> np.ndarray:
        """d_j = h^(l)(c_j): the target model's own representation of each phrase.

        Loads the target model and extracts the layer-l activation of each concept
        phrase at the SAME (model, layer, stream, token_selector) as the activations
        being reconstructed, so atoms and activations share a representational space.
        The atom dictionary is then anchored to the model's understanding of the
        phrase -- no projection W is learned.
        """
        from taxonomy_discovery.activations.extractor import (
            ActivationExtractor,
            HookSpec,
            ModelSpec,
        )

        md = activations.metadata
        template = str(cfg.get("concept_template", "{concept}"))
        self._log(
            f"target embed: running {len(concepts)} concepts through "
            f"{md['model_name']} @ layer {md.get('layer')} "
            f"(selector={md.get('token_selector')}, template={template!r})"
        )
        model_spec = ModelSpec(
            model_name=str(md["model_name"]),
            device=str(cfg.get("device", "cuda")),
            dtype=str(cfg.get("embedder_dtype", "bfloat16")),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            batch_size=int(cfg.get("embedder_batch_size", 16)),
            max_length=int(cfg.get("concept_max_length", 64)),
        )
        hook_spec = HookSpec(
            layer=int(md["layer"]),
            stream=str(md.get("stream", "resid_post")),
            token_selector=str(md.get("token_selector", "last_non_padding_token")),
            normalize=False,
        )
        concept_examples = [
            BehaviorExample(
                example_id=f"concept_{j}",
                behavior_family="concept",
                dataset_name="concepts",
                split="concept",
                prompt=template.format(concept=c),
                response=None,
            )
            for j, c in enumerate(concepts)
        ]
        extractor = ActivationExtractor(model_spec=model_spec, hook_spec=hook_spec)
        try:
            batch = extractor.extract(concept_examples)
        finally:
            try:
                extractor.unload()
            except Exception:
                pass
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        E = np.asarray(batch.activations, dtype=np.float32)  # [K, d]
        if E.shape[1] != int(md.get("hidden_size", E.shape[1])) and self.verbose:
            pass  # dimension comes straight from the model; no assertion needed
        return E

    @classmethod
    def _random_phrases(cls, K: int, seed: int) -> list[str]:
        rng = np.random.default_rng(seed + 7)
        vocab = list(cls._RANDOM_VOCAB)
        phrases: list[str] = []
        for _ in range(K):
            n_words = int(rng.integers(3, 6))
            phrases.append(" ".join(rng.choice(vocab, size=n_words, replace=True)))
        return phrases

    @staticmethod
    def _raw_probe_auc_fn(X: np.ndarray, y: np.ndarray, has_labels: bool, seed: int) -> float:
        """Held-out ROC-AUC of a logistic probe on the full activations (the ceiling)."""
        if not has_labels:
            return float("nan")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            if len(np.unique(y)) < 2:
                return float("nan")
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=0.3, random_state=seed, stratify=y
            )
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                return float("nan")
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(Xtr), ytr)
            prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
            return float(roc_auc_score(yte, prob))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"raw_probe_auc unavailable ({exc!r})", RuntimeWarning)
            return float("nan")

    @staticmethod
    def _selected_auc(
        A: np.ndarray, y: np.ndarray, selected: list[int], has_labels: bool, seed: int
    ) -> float:
        """Held-out ROC-AUC of a logistic readout over the SELECTED concept activations.

        Answers: do the chosen subbehaviors *compose* to predict the behavior?
        Directly comparable across ablations. Returns NaN without usable labels.
        """
        if not has_labels or not selected:
            return float("nan")
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import roc_auc_score
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler

            Xs = A[:, selected]
            if len(np.unique(y)) < 2:
                return float("nan")
            Xtr, Xte, ytr, yte = train_test_split(
                Xs, y, test_size=0.3, random_state=seed, stratify=y
            )
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                return float("nan")
            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=1000).fit(scaler.transform(Xtr), ytr)
            prob = clf.predict_proba(scaler.transform(Xte))[:, 1]
            return float(roc_auc_score(yte, prob))
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"selected_auc unavailable ({exc!r})", RuntimeWarning)
            return float("nan")

    @staticmethod
    def _omega(concepts: list[str], E: np.ndarray, cfg: dict) -> dict[str, float]:
        alpha = float(cfg.get("alpha_len", 0.0))
        beta = float(cfg.get("beta_cos", 0.0))
        length = alpha * sum(len(c.split()) for c in concepts)
        # E is L2-normalized => cosine == dot. Sum over i != j.
        G = E @ E.T
        cosine = beta * float(G.sum() - np.trace(G))
        return {"length": length, "cosine": cosine}

    # ---------------------------------------------------------------- concepts
    def _resolve_concepts(self, examples: list[BehaviorExample], cfg: dict) -> list[str]:
        explicit = cfg.get("concepts")
        if explicit:
            concepts = [str(c).strip() for c in explicit if str(c).strip()]
            if len(concepts) < 2:
                raise ValueError("Provide at least 2 concepts.")
            return concepts

        K = int(cfg.get("dictionary_size", 32))
        if bool(cfg.get("auto_seed_concepts", False)):
            seeded = self._seed_concepts_via_llm(examples, cfg, K)
            if seeded:
                return seeded
            warnings.warn(
                "auto_seed_concepts requested but LLM seeding failed; "
                "using the default sycophancy concept bank.",
                RuntimeWarning,
            )

        family = examples[0].behavior_family if examples else ""
        if family != "sycophancy":
            warnings.warn(
                f"No concepts provided for behavior family {family!r}; using the "
                "sycophancy default bank. Pass `concepts:` in the config.",
                RuntimeWarning,
            )
        bank = list(DEFAULT_SYCOPHANCY_CONCEPTS)
        return bank[: min(K, len(bank))]

    def _seed_concepts_via_llm(
        self, examples: list[BehaviorExample], cfg: dict, K: int
    ) -> list[str] | None:
        """Best-effort: ask the judge LLM for K candidate subbehavior phrases.

        Uses the shared judge config block. Returns None on any failure so the
        caller can fall back gracefully.
        """
        try:
            from taxonomy_discovery.utils.judge_client import build_judge_client, chat_completion

            judge = build_judge_client(cfg)
            family = examples[0].behavior_family if examples else "the behavior"
            positives = [
                (ex.response or ex.prompt)
                for ex in examples
                if self._get_label(ex) == 1
            ][:20]
            sample = "\n\n".join(f"- {p[:300]}" for p in positives if p)
            prompt = (
                f"You are analyzing the behavior '{family}'. Below are example "
                f"responses that exhibit it.\n\n{sample}\n\n"
                f"List exactly {K} short noun phrases (3-8 words each), one per line, "
                f"naming distinct sub-behaviors that compose '{family}'. Include at "
                f"least one phrase describing a neutral / non-{family} pattern. "
                f"Output only the phrases, no numbering."
            )
            resp = chat_completion(
                judge, [{"role": "user", "content": prompt}], max_tokens=600, temperature=0.7
            )
            text = resp.choices[0].message.content or ""
            lines = [ln.strip(" -*0123456789.").strip() for ln in text.splitlines()]
            concepts = [ln for ln in lines if len(ln.split()) >= 2]
            return concepts[:K] if len(concepts) >= 2 else None
        except Exception as exc:  # pragma: no cover - network/LLM dependent
            warnings.warn(f"LLM concept seeding failed: {exc!r}", RuntimeWarning)
            return None

    # ------------------------------------------------------------------- labels
    def _aligned_labels(
        self, examples: list[BehaviorExample], activations: ActivationBatch
    ) -> tuple[np.ndarray, bool]:
        id_to_label = {ex.example_id: self._get_label(ex) for ex in examples}
        ids = activations.example_ids
        if ids and all(i in id_to_label for i in ids):
            labels = [id_to_label[i] for i in ids]
        else:
            labels = [self._get_label(ex) for ex in examples]  # positional fallback
        arr = np.array([(-1 if v is None else int(v)) for v in labels], dtype=int)
        classes = set(int(v) for v in arr if v in (0, 1))
        has_labels = {0, 1}.issubset(classes)
        # map any unknown (-1) to 0 for safety; has_labels gate controls usage
        arr = np.where(arr == 1, 1, 0)
        return arr, has_labels

    @staticmethod
    def _get_label(ex: BehaviorExample) -> int | None:
        if ex.metadata and "discovery_label" in ex.metadata:
            try:
                return int(ex.metadata["discovery_label"])
            except Exception:
                return None
        if ex.label is None:
            return None
        try:
            return int(ex.label)
        except Exception:
            return None

    # --------------------------------------------------------------- diagnostics
    def _log(self, msg: str) -> None:
        if getattr(self, "verbose", True):
            print(f"[language_sae] {msg}", flush=True)

    def _log_concept_table(
        self,
        concepts: list[str],
        analysis: dict[str, np.ndarray],
        selected: list[int],
        has_labels: bool,
    ) -> None:
        if not getattr(self, "verbose", True):
            return
        delta = analysis["delta"]
        mu_pos = analysis["mu_pos"]
        mu_neg = analysis["mu_neg"]
        fire = analysis["firing_rate"]
        eff = analysis["effect_size"]
        order = list(np.argsort(delta)[::-1])
        sel = set(int(i) for i in selected)
        print("[language_sae] " + "-" * 72, flush=True)
        print(
            "[language_sae] behavior analysis (concepts ranked by Delta = mu+ - mu-):",
            flush=True,
        )
        header = f"   {'sel':>3} {'j':>3} {'mu+':>8} {'mu-':>8} {'Delta':>8} {'eff':>7} {'fire':>6}  concept"
        print("[language_sae]" + header, flush=True)
        for j in order:
            mark = " * " if j in sel else "   "
            label = "" if has_labels else "  (no labels: ranked by mean act)"
            print(
                f"[language_sae]   {mark} {j:>3} {mu_pos[j]:8.3f} {mu_neg[j]:8.3f} "
                f"{delta[j]:8.3f} {eff[j]:7.2f} {fire[j]:6.2f}  {concepts[j]}{label}",
                flush=True,
            )
        print(
            "[language_sae] selected subbehaviors -> "
            + " | ".join(concepts[j] for j in selected),
            flush=True,
        )

    # ------------------------------------------------------------------- device
    @staticmethod
    def _resolve_device(device_str: str) -> torch.device:
        if str(device_str).startswith("cuda") and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
