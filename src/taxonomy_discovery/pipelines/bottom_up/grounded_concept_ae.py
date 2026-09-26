from __future__ import annotations

"""Grounded Concept Autoencoder (two-part encoder + learnable decoder).

The architecture requested: a learnable dictionary of natural-language concepts whose
*coefficients are grounded by a frozen LM reading the response*, and a learnable decoder
that reconstructs the activation. This resolves the two failure modes from earlier:

  * learnable decoder atoms  -> atoms live in the activation distribution, so
    reconstruction actually works (the frozen single-token AR did not);
  * frozen-LM grounding gating the coefficients -> language is load-bearing, so it
    cannot collapse to the free-W "language is decorative" degeneracy.

ENCODER (two parts, two views of the example):
  Part A (activations only): a learnable concept dictionary Z = {z_j} in a frozen text
     space. From the activation h, a learnable map produces a query q(h); the activation
     score for concept j is  p_j = <q(h), z_j>.
  Part B (frozen LM on the response): the frozen LM embeds the response once -> r_x; the
     grounding score for concept j is  s_{x,j} = sigmoid(<z_j, r_x> / tau). This is the
     bi-encoder approximation of "frozen LM given concept + response -> score": both are
     embedded by the frozen LM and scored by similarity. Differentiable in z_j, and cheap
     (response embeddings are precomputed/cached; no per-step LM forward).

  The concept code is  a_{x,j} = relu(p_j(h_x)) * s_{x,j}  -- a concept fires only when the
  activation indicates it AND the frozen LM confirms it is in the response. Grounding is
  baked into the coefficient, so the objective is just reconstruction + sparsity.

DECODER (learnable): h_hat = b + sum_j a_{x,j} d_j, with learnable atoms d_j in R^d.

LOSS:  || h - h_hat ||^2  +  lambda * || a ||_1.

Learnable: Z (concepts), the activation encoder, decoder atoms D, bias b. Frozen: the LM
that produces response embeddings r_x and the phrase bank used to read concepts out as text.
"""

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from taxonomy_discovery.core.base_method import BaseMethod
from taxonomy_discovery.core.types import ActivationBatch, BasisResult, BehaviorExample
from taxonomy_discovery.pipelines.bottom_up.language_sae import (
    DEFAULT_SYCOPHANCY_CONCEPTS,
    TextEmbedder,
)


class GroundedConceptAE(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_concepts: int,
                 init_Z: np.ndarray | None = None, init_D: np.ndarray | None = None,
                 temperature: float = 0.1, activation: str = "softplus",
                 k_active: int = 0, unit_norm_decoder: bool = False) -> None:
        super().__init__()
        self.input_dim, self.embed_dim, self.num_concepts = int(input_dim), int(embed_dim), int(num_concepts)
        self.tau = float(temperature)
        self.activation = str(activation)  # softplus (soft) | relu (hard)
        self.k_active = int(k_active)      # >0 => Gated Top-K (grounding selects); 0 => multiplicative
        self.unit_norm_decoder = bool(unit_norm_decoder)
        if init_Z is not None:
            self.Z = nn.Parameter(torch.tensor(np.asarray(init_Z), dtype=torch.float32))
        else:
            self.Z = nn.Parameter(torch.randn(num_concepts, embed_dim) * 0.05)
        self.act_enc = nn.Linear(input_dim, embed_dim)          # part A: h -> concept query
        # Decoder atoms warm-started to top-K activation PCA directions.
        if init_D is not None:
            self.D = nn.Parameter(torch.tensor(np.asarray(init_D), dtype=torch.float32))
        else:
            self.D = nn.Parameter(torch.randn(num_concepts, input_dim) / (input_dim ** 0.5))
        self.b = nn.Parameter(torch.zeros(input_dim))

    def atoms(self) -> torch.Tensor:
        # unit-norm atoms => a constant gate can't be absorbed by rescaling the decoder
        if self.unit_norm_decoder:
            return self.D / self.D.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.D

    def code(self, h: torch.Tensor, r: torch.Tensor):
        p = self.act_enc(h) @ self.Z.t()                        # [., K] activation score (part A)
        s = torch.sigmoid((r @ self.Z.t()) / self.tau)          # [., K] frozen-LM grounding (part B)
        mag = torch.relu(p) if self.activation == "relu" else torch.nn.functional.softplus(p)
        if self.k_active > 0 and self.k_active < self.num_concepts:
            # Gated Top-K: GROUNDING selects which concepts are on (top-k by s), the
            # activation only sets magnitude. The gate is the sole selector, so it can't
            # be a no-op; exactly k concepts fire per example, so the code can't die.
            topi = s.topk(self.k_active, dim=-1).indices
            mask = torch.zeros_like(s).scatter(-1, topi, 1.0)
            a = mask * s * mag                                  # select by grounding, scale by s*mag
        else:
            a = mag * s                                         # multiplicative (legacy)
        return a, p, s

    def forward(self, h: torch.Tensor, r: torch.Tensor):
        a, p, s = self.code(h, r)
        return self.b + a @ self.atoms(), a, s


@dataclass
class _Fit:
    best_val: float
    epochs: int
    final_recon: float
    final_l1: float
    final_active: float


class GroundedConceptAEMethod(BaseMethod):
    """Two-part-encoder grounded concept autoencoder; see module docstring.

    Config (defaults):
        name: grounded_concept_ae
        num_concepts: 32
        concepts: [...]             # phrase bank to init/decode concepts (else default bank)
        embedder_backend: auto      # auto | hf | hash  (embeds responses + phrase bank)
        embedder_hash_dim: 256
        temperature: 0.1            # grounding sigmoid temperature
        lambda_l1: 0.001
        normalize_input: true
        lr: 0.005
        max_epochs: 300
        patience: 25
        batch_size: 256
        seed: 0
    """

    def fit(self, examples: list[BehaviorExample], activations: ActivationBatch) -> BasisResult:
        cfg = self.config
        self.verbose = bool(cfg.get("verbose", True))
        seed = int(cfg.get("seed", 0))
        torch.manual_seed(seed); np.random.seed(seed)
        device = self._device(cfg.get("device", "cuda"))

        H = np.asarray(activations.activations, dtype=np.float32)
        n, d = H.shape
        if bool(cfg.get("normalize_input", True)):
            H = (H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        total_var = float(np.sum((H - H.mean(0)) ** 2, axis=1).mean())

        self._log("=" * 72)
        self._log("Grounded Concept AE  —  a = relu(encoder(h)·Z) * frozenLM(Z, response)")
        self._log("=" * 72)
        self._log(f"activations : n={n} d={d}")

        # responses aligned to activation row order. With per-token activations, ids are
        # REPEATED per token, so embed each UNIQUE response once and broadcast to its
        # tokens (avoids embedding ~100k duplicate responses through the model).
        id2resp = {ex.example_id: (ex.response or ex.prompt or "") for ex in examples}
        ids = activations.example_ids or [ex.example_id for ex in examples]
        per_token = bool(activations.metadata.get("per_token", False))
        uniq_ids = list(dict.fromkeys(ids))
        uniq_resp = [id2resp.get(i, "") for i in uniq_ids]
        id_pos = {i: k for k, i in enumerate(uniq_ids)}
        broadcast = np.array([id_pos[i] for i in ids])         # token-row -> unique-response idx
        if per_token:
            self._log(f"per-token   : {len(ids)} tokens from {len(uniq_ids)} responses "
                      f"({len(ids) / max(len(uniq_ids), 1):.1f} tok/response)")
        explicit = [str(c).strip() for c in (cfg.get("concepts") or []) if str(c).strip()]
        if explicit:
            bank = explicit
        elif str(cfg.get("concept_bank_source", "default")) == "extract":
            bank = self._bank_from_extraction(examples, cfg, int(cfg.get("num_concepts", 32)))
            if not bank:
                warnings.warn(
                    "concept_bank_source=extract but no extraction cache was found; using the "
                    "default bank. Run grounded_concept_sae with grounding_source: extract first.",
                    RuntimeWarning,
                )
                bank = list(DEFAULT_SYCOPHANCY_CONCEPTS)
            else:
                self._log(f"bank        : {len(bank)} concepts from extraction-cache vocabulary")
        else:
            bank = list(DEFAULT_SYCOPHANCY_CONCEPTS)

        # --- frozen LM that embeds responses (part B) + the phrase bank --------------
        # target_model: grounding happens in the TARGET model's own representation space
        # (Qwen mean-pooled hidden states), so "is concept c in response r" is judged by
        # the model we're interpreting, not a foreign embedder.
        backend = str(cfg.get("embedder_backend", "auto"))
        if backend == "target_model":
            Ru, bank_emb, embed_info = self._embed_via_target_model(uniq_resp, bank, activations, cfg)
            m = Ru.shape[1]
        else:
            embedder = TextEmbedder(
                backend=backend,
                model_name=str(cfg.get("embedder_model", "sentence-transformers/all-MiniLM-L6-v2")),
                hash_dim=int(cfg.get("embedder_hash_dim", 256)),
                device="cuda" if device.type == "cuda" else "cpu",
            )
            Ru = embedder.embed(uniq_resp)                      # [n_responses, m] (unique)
            bank_emb = embedder.embed(bank)
            m = int(embedder.dim)
            embed_info = embedder.info()
        R = Ru[broadcast]                                       # [N_rows, m] broadcast to tokens
        self._log(f"frozen LM   : backend={embed_info['backend']} m={m}  (response embeddings precomputed)")

        # Control: permute response embeddings vs activations to BREAK grounding. If FVE
        # barely drops vs the real run, the frozen-LM gate is not load-bearing (it's a
        # free SAE with a PCA warm-start); if FVE drops a lot, grounding is doing work.
        if bool(cfg.get("shuffle_grounding", False)):
            R = R[np.random.default_rng(seed + 1).permutation(len(R))]
            self._log("CONTROL     : shuffle_grounding=ON -> response<->activation grounding broken")

        K = int(cfg.get("num_concepts", 32))
        init_idx = [i % len(bank) for i in range(K)]
        init_Z = bank_emb[init_idx]
        self._log(f"dictionary  : K={K} concepts initialized from a {len(bank)}-phrase bank")

        k_active = int(cfg.get("k_active", 0))
        Hc = H - H.mean(0)
        warmstart_fve = float("nan")
        # Warm-start the decoder so reconstruction starts POSITIVE under the grounded
        # selection. For Gated Top-K, the code selects by grounding, so PCA atoms (which
        # assume a reconstruction-optimal selection) are mismatched and the optimizer
        # kills the magnitude. Instead, fit the atoms by ridge regression on the INITIAL
        # grounded selection -- i.e. initialize from the closed-form grounded_concept_sae
        # solution -- so training refines upward instead of collapsing.
        try:
            if k_active > 0:
                s0 = 1.0 / (1.0 + np.exp(-(R @ init_Z.T) / float(cfg.get("temperature", 0.1))))
                G0 = np.zeros_like(s0)
                topi = np.argpartition(-s0, kth=k_active - 1, axis=1)[:, :k_active]
                np.put_along_axis(G0, topi, 1.0, axis=1)            # binary grounded selection
                alpha = float(cfg.get("warmstart_ridge_alpha", 1.0))
                gram = G0.T @ G0 + alpha * np.eye(K)
                init_D = np.linalg.solve(gram, G0.T @ Hc).astype(np.float32)   # [K, d]
                # closed-form warm-start FVE = the baseline the gradient learning must beat
                warmstart_fve = float(1.0 - ((Hc - G0 @ init_D) ** 2).sum() / max((Hc ** 2).sum(), 1e-8))
            else:
                _, _, Vt = np.linalg.svd(Hc, full_matrices=False)
                init_D = Vt[: min(K, Vt.shape[0])].astype(np.float32)
                if init_D.shape[0] < K:
                    init_D = np.concatenate([init_D, np.random.default_rng(seed).normal(
                        0, 1 / np.sqrt(d), (K - init_D.shape[0], d)).astype(np.float32)])
        except Exception:
            init_D = None

        unit_norm = bool(cfg.get("unit_norm_decoder", k_active > 0))
        model = GroundedConceptAE(d, m, K, init_Z=init_Z, init_D=init_D,
                                  temperature=float(cfg.get("temperature", 0.1)),
                                  activation=str(cfg.get("activation", "softplus")),
                                  k_active=k_active, unit_norm_decoder=unit_norm).to(device)
        with torch.no_grad():
            model.b.copy_(torch.tensor(H.mean(0), dtype=torch.float32, device=device))
        mode = f"Gated Top-K (k_active={k_active}, grounding selects)" if k_active > 0 else "multiplicative gate"
        self._log(f"mode        : {mode}  unit_norm_decoder={unit_norm}")

        # snapshot init params to quantify how much each component actually learns
        Z_init = model.Z.detach().cpu().numpy().copy()
        enc_init = model.act_enc.weight.detach().cpu().numpy().copy()
        Dinit_norm = init_D / (np.linalg.norm(init_D, axis=1, keepdims=True) + 1e-8) if init_D is not None else None

        stats = self._train(model, H, R, bank_emb, cfg, device, seed, total_var)

        # --- outputs -----------------------------------------------------------
        model.eval()
        with torch.inference_mode():
            D = model.atoms().detach().cpu().numpy()
            Z = model.Z.detach().cpu().numpy()
            # gate diagnostic: is s discriminative, or ~constant (a no-op gate)?
            Rt = torch.tensor(R, dtype=torch.float32, device=device)
            s_all = torch.sigmoid((Rt @ model.Z.t()) / model.tau).cpu().numpy()
        s_mean, s_std = float(s_all.mean()), float(s_all.std())
        # spread of per-concept mean presence — if all concepts gate identically, ~0
        gate_spread = float(s_all.mean(axis=0).std())
        # within-example spread across concepts — what Top-K-by-grounding needs to be
        # meaningful (does s differentiate concepts for a GIVEN response?)
        within_spread = float(s_all.std(axis=1).mean())
        names = self._decode(Z, bank, bank_emb)

        # --- how much did each component actually learn? -----------------------
        def rel_drift(a, b):
            return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))
        z_drift = rel_drift(Z, Z_init)
        z_cos = float(np.mean(np.sum(
            (Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)) *
            (Z_init / (np.linalg.norm(Z_init, axis=1, keepdims=True) + 1e-8)), axis=1)))
        enc_drift = rel_drift(model.act_enc.weight.detach().cpu().numpy(), enc_init)
        d_drift = rel_drift(D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8), Dinit_norm) \
            if Dinit_norm is not None else float("nan")
        names_init = self._decode(Z_init, bank, bank_emb)
        decode_changed = int(sum(a != b for a, b in zip(names, names_init)))
        # decode faithfulness: cos(z_j, its decoded/nearest phrase). Low => Z drifted off
        # the language manifold and the names don't describe what z_j encodes (a renamed
        # free SAE), even if reconstruction is good.
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        Bn = bank_emb / (np.linalg.norm(bank_emb, axis=1, keepdims=True) + 1e-8)
        nn_cos = (Zn @ Bn.T).max(axis=1)
        decode_faithfulness = float(nn_cos.mean())
        frac_offmanifold = float((nn_cos < 0.5).mean())
        fve_trained = 1.0 - stats.best_val / max(total_var, 1e-8)
        self._log(
            f"learning    : warm-start FVE={warmstart_fve:.3f} -> trained FVE={fve_trained:.3f}  "
            f"gain={fve_trained - warmstart_fve:+.3f}  (how much gradient added over the closed-form init)"
        )
        self._log(
            f"param drift : Z={z_drift:.2f} (cos {z_cos:.2f}, {decode_changed}/{len(names)} decode changed)  "
            f"encoder={enc_drift:.2f}  decoder={d_drift:.2f}  (rel. ||Δ||/||init||)"
        )
        self._log(
            f"decode faith: mean cos(z, decoded phrase)={decode_faithfulness:.3f}  "
            f"off-manifold(<0.5)={frac_offmanifold:.0%}  "
            "(low => names don't describe z; a renamed free SAE)"
        )
        self._log(f"gate within-example concept-spread={within_spread:.3f} "
                  "(Top-K selection needs this > 0 to differentiate concepts per response)")
        self._log(
            f"gate (s)    : mean={s_mean:.3f} std={s_std:.3f} per-concept-spread={gate_spread:.3f}  "
            "(low std/spread => gate ~constant => not load-bearing; run shuffle_grounding to confirm)"
        )
        directions = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-8)
        fve = 1.0 - stats.best_val / max(total_var, 1e-8)
        self._log(f"trained     : val recon={stats.best_val:.4f} FVE={fve:.3f} @ epoch {stats.epochs} "
                  f"(active/ex={stats.final_active:.2f}, L1/ex={stats.final_l1:.3f})")
        self._log("concepts    : " + " | ".join(names[:8]) + (" ..." if len(names) > 8 else ""))

        return BasisResult(
            method_name="grounded_concept_ae",
            behavior_family=examples[0].behavior_family,
            train_dataset=examples[0].dataset_name,
            model_name=activations.metadata["model_name"],
            layer=int(activations.metadata["layer"]),
            directions=directions.astype(np.float32),
            direction_names=names,
            metadata={
                "n_examples": int(n), "input_dim": int(d), "num_concepts": int(K),
                "embed_dim": int(m), "temperature": float(cfg.get("temperature", 0.1)),
                "fve": float(fve), "best_val_recon": float(stats.best_val),
                "final_recon": float(stats.final_recon), "final_l1": float(stats.final_l1),
                "final_active_per_example": float(stats.final_active),
                "gate_s_mean": s_mean, "gate_s_std": s_std, "gate_per_concept_spread": gate_spread,
                "gate_within_example_spread": within_spread,
                "k_active": k_active, "unit_norm_decoder": unit_norm,
                "warmstart_fve": warmstart_fve, "learning_gain": fve_trained - warmstart_fve,
                "z_drift": z_drift, "z_cos": z_cos, "decode_changed": decode_changed,
                "decode_faithfulness": decode_faithfulness, "frac_offmanifold": frac_offmanifold,
                "encoder_drift": enc_drift, "decoder_drift": d_drift,
                "shuffle_grounding": bool(cfg.get("shuffle_grounding", False)),
                "total_var": float(total_var), "embedder": embed_info,
            },
            fit_config={**cfg, "resolved_K": int(K)},
        )

    def _train(self, model, H, R, bank_emb, cfg, device, seed, total_var) -> _Fit:
        """GPU-resident training. The model is tiny (a few matmuls), so the only way to
        keep an A100 busy is to (1) hold ALL data on the GPU, (2) use large batches, and
        (3) avoid per-batch host syncs. We index GPU tensors directly (no DataLoader, no
        per-batch .to(device)/.item()) and sync once per epoch on the validation pass.
        """
        lam = float(cfg.get("lambda_l1", 1e-3))
        lr = float(cfg.get("lr", 5e-3))
        max_epochs, patience = int(cfg.get("max_epochs", 300)), int(cfg.get("patience", 25))
        bs = int(cfg.get("batch_size", 8192))           # big: the A100 wants large matmuls
        val_frac = float(cfg.get("val_fraction", 0.1))

        # Language anchors: keep Z on the real-phrase manifold DURING training, else the
        # gradient drifts Z off-manifold and the names become fiction (decode_faith ~ 0).
        lam_dec = float(cfg.get("lambda_decode", 0.0))      # pull Z toward nearest phrase
        reproject_every = int(cfg.get("reproject_every", 0))  # snap Z to nearest phrase every N
        Braw = torch.as_tensor(bank_emb, dtype=torch.float32, device=device)        # [V, m]
        Bn = Braw / Braw.norm(dim=1, keepdim=True).clamp_min(1e-8)

        # all data resident on the GPU once (per-token H/R are ~hundreds of MB -> fine)
        X = torch.as_tensor(H, dtype=torch.float32, device=device)
        Rt = torch.as_tensor(R, dtype=torch.float32, device=device)
        N = X.shape[0]
        perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))  # cpu, reproducible
        vs = max(1, int(N * val_frac))
        val_idx = perm[:vs].to(device)
        tr_idx = perm[vs:].to(device)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
        print_every = max(1, max_epochs // 20)

        def run_eval():
            model.eval()
            rec = torch.zeros((), device=device)        # total per-example sum-sq error
            l1 = torch.zeros((), device=device)         # total L1 over rows
            act = torch.zeros((), device=device)        # total active count over rows
            with torch.inference_mode():
                for s0 in range(0, val_idx.shape[0], bs):
                    ib = val_idx[s0:s0 + bs]
                    hhat, a, _ = model(X[ib], Rt[ib])
                    rec = rec + torch.sum((X[ib] - hhat) ** 2)
                    l1 = l1 + a.abs().sum()
                    act = act + (a > 1e-4).float().sum()
            cnt = max(val_idx.shape[0], 1)
            return (rec / cnt).item(), (l1 / cnt).item(), (act / cnt).item()

        best, best_state, best_ep, stale = float("inf"), None, -1, 0
        last_recon = last_l1 = last_active = float("nan")
        for ep in range(max_epochs):
            model.train()
            shuf = tr_idx[torch.randperm(tr_idx.shape[0], device=device)]
            for s0 in range(0, shuf.shape[0], bs):
                ib = shuf[s0:s0 + bs]
                opt.zero_grad(set_to_none=True)
                hhat, a, _ = model(X[ib], Rt[ib])
                recon = torch.mean(torch.sum((X[ib] - hhat) ** 2, dim=-1))
                loss = recon + lam * torch.mean(torch.sum(a.abs(), dim=-1))
                if lam_dec > 0:
                    Zn = model.Z / model.Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    anchor = (1.0 - (Zn @ Bn.t()).max(dim=1).values).mean()  # dist to nearest phrase
                    loss = loss + lam_dec * anchor
                loss.backward()
                opt.step()

            # periodic hard re-projection: snap each concept to its nearest real phrase
            if reproject_every > 0 and (ep + 1) % reproject_every == 0:
                with torch.no_grad():
                    Zn = model.Z / model.Z.norm(dim=1, keepdim=True).clamp_min(1e-8)
                    nn = (Zn @ Bn.t()).argmax(dim=1)
                    model.Z.copy_(Braw[nn])

            val, last_l1, last_active = run_eval()    # one host sync per epoch
            last_recon = val
            improved = val < best - 1e-9
            if improved:
                best, best_ep, stale = val, ep, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if self.verbose and (ep % print_every == 0 or (improved and ep < 4)):
                print(f"[grounded_ae]   epoch {ep:>4} val_recon={val:8.4f} "
                      f"FVE={1 - val/max(total_var,1e-8):6.3f} active/ex={last_active:5.2f}"
                      f"{'  *best' if improved else ''}", flush=True)
            if stale >= patience:
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.to(device)
        return _Fit(best, best_ep + 1, last_recon, last_l1, last_active)

    def _embed_via_target_model(self, responses, bank, activations, cfg):
        """Embed responses + phrase bank in the TARGET model's own space.

        Mean-pools hidden states over the text at a chosen layer (default = the
        activations' layer), unit-normalized. Grounding similarity <z_j, r_x> is then
        computed in the model-we-interpret's representation space, not a foreign one.
        """
        import gc
        from taxonomy_discovery.activations.extractor import (
            ActivationExtractor, HookSpec, ModelSpec,
        )

        md = activations.metadata
        layer = int(cfg.get("embedder_layer", md.get("layer", 0)))
        model_spec = ModelSpec(
            model_name=str(md["model_name"]),
            device=str(cfg.get("device", "cuda")),
            dtype=str(cfg.get("embedder_dtype", "bfloat16")),
            trust_remote_code=bool(cfg.get("trust_remote_code", False)),
            batch_size=int(cfg.get("embedder_batch_size", 16)),
            max_length=int(cfg.get("embedder_max_length", 512)),
        )
        # normalize=False here: raw LLM hidden states are anisotropic (cluster in a cone,
        # high pairwise cosine), which saturates the grounding sigmoid. We de-anisotropize
        # below (center + remove top directions) THEN unit-normalize.
        hook_spec = HookSpec(layer=layer, stream=str(md.get("stream", "resid_post")),
                             token_selector="mean_all_tokens", normalize=False)
        self._log(f"target embed: {md['model_name']} @ layer {layer} (mean-pool)")

        def mk(texts):
            return [
                BehaviorExample(example_id=f"emb_{i}", behavior_family="emb", dataset_name="emb",
                                split="emb", prompt=(t or " "), response=None)
                for i, t in enumerate(texts)
            ]

        ext = ActivationExtractor(model_spec=model_spec, hook_spec=hook_spec)
        try:
            R = np.asarray(ext.extract(mk(responses)).activations, dtype=np.float32)
            bank_emb = np.asarray(ext.extract(mk(bank)).activations, dtype=np.float32)
        finally:
            try:
                ext.unload()
            except Exception:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # De-anisotropize ("all-but-the-top"): raw LLM states share a dominant common
        # direction that makes every pairwise cosine ~1 and saturates the gate. Center by
        # the response-embedding mean and remove the top-k principal directions, then
        # unit-normalize, so cosine reflects discriminative content rather than the cone.
        def cos_offdiag(M):
            Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
            G = Mn @ Mn.T
            K_ = G.shape[0]
            return float((G.sum() - np.trace(G)) / max(K_ * (K_ - 1), 1))

        before = cos_offdiag(R[: min(500, len(R))])
        mu = R.mean(axis=0, keepdims=True)
        Rc, Bc = R - mu, bank_emb - mu
        topk = int(cfg.get("embedder_remove_topk", 1))
        if topk > 0:
            _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
            P = Vt[:topk]                                   # [k, d] dominant directions
            Rc = Rc - (Rc @ P.T) @ P
            Bc = Bc - (Bc @ P.T) @ P
        R = (Rc / (np.linalg.norm(Rc, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        bank_emb = (Bc / (np.linalg.norm(Bc, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        after = cos_offdiag(R[: min(500, len(R))])
        self._log(f"de-anisotropy: mean pairwise cos {before:.3f} -> {after:.3f} "
                  f"(centered, removed top-{topk}); high before => gate would saturate")

        info = {"backend": "target_model", "model_name": str(md["model_name"]),
                "layer": layer, "dim": int(R.shape[1]), "pooling": "mean_all_tokens",
                "remove_topk": topk, "cos_before": before, "cos_after": after}
        return R, bank_emb, info

    def _bank_from_extraction(self, examples, cfg, target_size: int) -> list[str] | None:
        """Build a large named concept bank from the cached `extract` vocabulary.

        Loads the per-response phrases that grounded_concept_sae (grounding_source:
        extract) cached, then clusters the unique phrases into `target_size`
        representatives (named by their most frequent member). Returns None if no cache
        exists for this dataset.
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
                self._log(f"bank        : exact extraction cache missed; using "
                          f"{len(files)} cached extraction file(s) in {cache_dir}")
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

    @staticmethod
    def _decode(Z: np.ndarray, bank: list[str], bank_emb: np.ndarray) -> list[str]:
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
        Bn = bank_emb / (np.linalg.norm(bank_emb, axis=1, keepdims=True) + 1e-8)
        nn_idx = (Zn @ Bn.T).argmax(axis=1)
        return [bank[i] for i in nn_idx]

    @staticmethod
    def _device(s: str) -> torch.device:
        return torch.device("cuda") if str(s).startswith("cuda") and torch.cuda.is_available() else torch.device("cpu")

    def _log(self, msg: str) -> None:
        if getattr(self, "verbose", True):
            print(f"[grounded_ae] {msg}", flush=True)
