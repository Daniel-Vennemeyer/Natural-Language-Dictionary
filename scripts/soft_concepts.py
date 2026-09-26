"""Natural Language Dictionary: learn ONLY K soft concept prompts through the frozen judge.

Nothing is finetuned -- no LoRA, no policy LM, no judge updates. The ONLY trainable parameters
are K soft prompts E [K, n_tokens, d] sitting in the judge's concept slot. Training is plain
backprop (no RL): P = presence_grid_soft(E, texts) is differentiable, and the loss is the
SET-LEVEL coverage -J_cv(P, Y) (cross-fitted in-batch trace-R^2 of the target) plus a
token-manifold pull. Collapse is a neutral plateau here, not an attractor: a duplicated soft
concept adds a collinear presence column -> zero marginal J -> no gradient toward it.

Target (--target): s = the de-confounded scalar axis (rank-1 read-off ruler); acts_wc = top
--acts-dims PCs of the WITHIN-positive-class de-confounded residual -- the principled multi-dim
"ways of exhibiting the behavior" target, where variance-explained REQUIRES a distinct
multi-atom basis by construction. --lambda-fid adds the second irreducible objective as a
differentiable term: each atom's UNIQUE presence component (residualized on the other atoms)
must be linearly readable from the activations (cross-fitted direction correlation), so every
atom is forced to be a mechanistically real, non-duplicate feature DURING training, not just
at post-hoc read-off. --eval-judge re-scores the final named taxonomy under an independent
judge model, breaking train/eval judge circularity.

Because a soft prompt is a behavioral probe in QUESTION space (it can express presence patterns
no sentence produces), this is a STRUCTURE-FIRST method: discovery is language-free; language
enters once, at the end, via SNAP-TO-TEXT -- each frozen soft concept is named by the candidate
sentence whose HARD judged presence best correlates with the soft presence on a probe set
(extensional matching, not embedding cosine). The per-atom snap correlation is its
verbalizability score; the soft-vs-named coverage gap on test is the price of language.

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    python scripts/soft_concepts.py --config configs/.../l20_resp.yaml \
      --units $U/units_last_l20.npz --prompt-acts $U/prompt_last_l20.npz \
      --init-concepts $U/init_behavior.npz --num-concepts 6 --steps 300 \
      --out $U/soft_concepts.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import yaml

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

from taxonomy_discovery.utils.frozen_presence import FrozenJudge
from taxonomy_discovery.utils.runlock import acquire, gpu_guard


def _auc(sc, y):
    y = (np.asarray(y) > 0.5).astype(int); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return 0.5
    o = np.argsort(sc); r = np.empty(len(sc)); r[o] = np.arange(1, len(sc) + 1)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _shrinkc(S, a):
    return (1 - a) * S + a * (np.trace(S) / S.shape[0]) * np.eye(S.shape[0])


def _pcs(fit, allx, P):
    _, _, Vt = np.linalg.svd(fit, full_matrices=False); W = Vt[:min(P, Vt.shape[0])].T
    return allx @ W


def _pcs_w(fit, allx, P):
    _, _, Vt = np.linalg.svd(fit, full_matrices=False); W = Vt[:min(P, Vt.shape[0])].T
    return allx @ W, W


def _ridge_np(P, y):
    X = np.concatenate([P, np.ones((len(P), 1))], 1)
    lam = 1e-2 * np.trace(X.T @ X) / X.shape[1]; L = lam * np.eye(X.shape[1]); L[-1, -1] = 0
    return np.linalg.solve(X.T @ X + L, X.T @ y)


def _r2_np(w, P, y):
    pred = np.concatenate([P, np.ones((len(P), 1))], 1) @ w
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def _effk(P):
    act = P[P.std(1) > 1e-6]
    if len(act) == 0:
        return 0.0
    ev = np.linalg.eigvalsh(np.atleast_2d(np.corrcoef(act)))
    return float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-9))


def _spear(a, b):                                              # Spearman rank correlation (paper Fidelity)
    def rk(x):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
        return r
    ra, rb = rk(a) - (len(a) - 1) / 2, rk(b) - (len(b) - 1) / 2
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-9))


def _phrase(t):
    t = (t or "").split("\n")[0].strip().strip('"').strip()
    for cut in ("<|", "GROUP", "Group", "Note:", "note:"):
        if cut in t:
            t = t.split(cut)[0].strip()
    t = t.rstrip(".;,: ").lstrip("-* ")
    return " ".join(t.split()[:24])


def _fwd_select(Pc, Yp, K, min_gain=1e-3):
    """Set-level naming: forward-select concepts whose JOINT presence best reconstructs the target
    on the probe set (symmetric split-half cross-fit). The SET is the unit, not the atom --
    per-atom extensional snap picks each atom's best match blind to what the rest of the set
    already covers, and verbal_ceiling measured that costing ~2.6x recon at matched K."""
    ev = np.arange(Pc.shape[1]) % 2 == 0
    sel, best = [], -1e9
    for _ in range(K):
        gains = []
        for c in range(len(Pc)):
            if c in sel:
                continue
            cand = sel + [c]
            v = 0.5 * (_r2_np(_ridge_np(Pc[cand][:, ev].T, Yp[ev]), Pc[cand][:, ~ev].T, Yp[~ev])
                       + _r2_np(_ridge_np(Pc[cand][:, ~ev].T, Yp[~ev]), Pc[cand][:, ev].T, Yp[ev]))
            gains.append((v, c))
        v, c = max(gains)
        if v < best + min_gain:
            break
        best = v; sel.append(c)
    return sel


def _jcv_t(P, Y, ridge):
    """Differentiable symmetric cross-fitted trace-R^2 of target Y [M, D] from presence P [K, M].
    D=1 recovers the scalar-axis objective; for acts_wc the Frobenius form variance-weights the
    within-class PCs, so atoms are pulled toward the dominant ways the behavior varies."""
    M = P.shape[1]; Am = torch.arange(M, device=P.device) % 2 == 0

    def half(trm, tem):
        X = torch.cat([P[:, trm].T, torch.ones(int(trm.sum()), 1, device=P.device)], 1)
        Xt = torch.cat([P[:, tem].T, torch.ones(int(tem.sum()), 1, device=P.device)], 1)
        lm = ridge * torch.trace(X.T @ X) / X.shape[1]
        L = lm * torch.eye(X.shape[1], device=P.device); L[-1, -1] = 0
        w = torch.linalg.solve(X.T @ X + L, X.T @ Y[trm])
        pred = Xt @ w; yt = Y[tem]
        return 1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-9)
    return 0.5 * (half(Am, ~Am) + half(~Am, Am))


def _jdm_t(P, Y, R, Wy_t, ridge):
    """DM-TIED differentiable recon: the judgments->activation->prediction pipeline as the
    OBJECTIVE. Each atom's activation signature is its presence-weighted DiffMean over R
    (differentiable in P, not a free parameter), mapped into Y coords by the fixed basis Wy_t;
    only K scalars are fit. Cross-fitted like _jcv_t: build dirs + fit gammas on one half,
    score the other, symmetrize. K params instead of K*D -- the interpretable model, trained.
    """
    M = P.shape[1]; dev = P.device
    Am = torch.arange(M, device=dev) % 2 == 0

    def half(trm, tem):
        w = P[:, trm].clamp(0, 1)                                # [K, Mtr] presence weights
        sw, sc = w.sum(1, keepdim=True) + 1e-6, (1 - w).sum(1, keepdim=True) + 1e-6
        V = (w / sw) @ R[trm] - ((1 - w) / sc) @ R[trm]          # [K, pc] presence-weighted DiffMean
        V = V / (V.norm(dim=1, keepdim=True) + 1e-9)
        VY = V @ Wy_t                                            # [K, D] signatures in Y coords
        muY, muP = Y[trm].mean(0), P[:, trm].mean(1)
        A = (P[:, trm] - muP[:, None]).T[:, :, None] * VY[None, :, :]      # [Mtr, K, D]
        Af = A.permute(0, 2, 1).reshape(-1, A.shape[1])                   # [Mtr*D, K]
        yf = (Y[trm] - muY).reshape(-1)
        G = Af.T @ Af + ridge * torch.eye(A.shape[1], device=dev) * (torch.trace(Af.T @ Af) / A.shape[1])
        gam = torch.linalg.solve(G, Af.T @ yf)                   # [K] the only fitted params
        At = (P[:, tem] - muP[:, None]).T[:, :, None] * VY[None, :, :]     # [Mte, K, D]
        pred = muY + (At * gam[None, :, None]).sum(1)
        yt = Y[tem]
        return 1 - ((yt - pred) ** 2).sum() / (((yt - yt.mean(0)) ** 2).sum() + 1e-9)
    return 0.5 * (half(Am, ~Am) + half(~Am, Am))


def _fid_cv_t(P, R, ridge):
    """Differentiable cross-fitted interpretation fidelity per atom -> [K].

    For each atom j: residualize its presence on the OTHER atoms (fit on one half) to get its
    unique component u_j, ridge-fit an activation direction v_j to u_j on that half, and score
    corr(R @ v_j, u_j) on the other half; symmetric average. A duplicate atom has no unique
    component -> fid ~ 0 (anti-collapse pressure is intrinsic, same as LOO-recon); the corr is
    scaled by the unique-variance share so a near-duplicate cannot harvest fidelity from
    amplified residual noise."""
    K, M = P.shape; dev = P.device
    Am = torch.arange(M, device=dev) % 2 == 0

    def half(trm, tem):
        Xr = R[trm]; lm = ridge * torch.trace(Xr.T @ Xr) / Xr.shape[1]
        G = Xr.T @ Xr + lm * torch.eye(R.shape[1], device=dev)
        fids = []
        for j in range(K):
            o = torch.cat([P[:j], P[j + 1:]], 0)
            Xo = torch.cat([o[:, trm].T, torch.ones(int(trm.sum()), 1, device=dev)], 1)
            lo = 1e-2 * torch.trace(Xo.T @ Xo) / Xo.shape[1]
            Lo = lo * torch.eye(Xo.shape[1], device=dev); Lo[-1, -1] = 0
            b = torch.linalg.solve(Xo.T @ Xo + Lo, Xo.T @ P[j, trm])
            u_tr = P[j, trm] - Xo @ b
            u_te = P[j, tem] - torch.cat([o[:, tem].T, torch.ones(int(tem.sum()), 1, device=dev)], 1) @ b
            v = torch.linalg.solve(G, Xr.T @ u_tr)
            pr = R[tem] @ v
            uc = u_te - u_te.mean(); pc = pr - pr.mean()
            corr = (uc * pc).sum() / (uc.norm() * pc.norm() + 1e-6)
            share = (u_te.std() / (P[j, tem].std() + 1e-6)).clamp(max=1.0)
            fids.append(corr * share)
        return torch.stack(fids)
    return 0.5 * (half(Am, ~Am) + half(~Am, Am))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True); ap.add_argument("--acts-key", default="resp_acts")
    ap.add_argument("--init-concepts", default=None,
                    help="npz with 'names': on-manifold warm start (one distinct name per atom) + snap candidates")
    ap.add_argument("--num-concepts", type=int, default=6); ap.add_argument("--n-tokens", type=int, default=6)
    ap.add_argument("--target", choices=["s", "acts_wc", "acts_wc_axisw", "acts_pc"], default="s",
                    help="training/eval target. s = de-confounded scalar axis (rank-1). acts_wc = top "
                         "--acts-dims PCs of the WITHIN-positive-class de-confounded residual (fit on "
                         "positive-train, projected onto everyone): the multi-dim 'ways of exhibiting the "
                         "behavior' variance. Rank-1 class-diff does NOT make it rank-1, so reconstructing "
                         "it requires a distinct multi-atom basis -- distinctness+grounding become intrinsic "
                         "to the coverage objective. The axis AUC stays a read-off either way.")
    ap.add_argument("--acts-dims", type=int, default=32)
    ap.add_argument("--canon-acts-dims", type=int, default=32,
                    help="dimensionality of the --eval-canonical yardstick target. Held at the "
                         "canonical 32 independently of --acts-dims so a dims ablation is scored "
                         "against the SAME target as every other row.")
    ap.add_argument("--lambda-fid", type=float, default=0.0,
                    help="weight of the differentiable cross-fitted interpretation-fidelity term: each "
                         "atom's UNIQUE presence component must be linearly readable from the activation "
                         "residual Res. recon+fid is the irreducible NLD objective pair (rl_compose's "
                         "acts_wc lesson) -- here both are exact gradients, no RL. Try 0.5-1.5.")
    ap.add_argument("--lambda-recon", type=float, default=1.0,
                    help="weight of the recon term. 0 = the causal-only ablation arm: prediction is "
                         "rank-1 collapse (the class contrast is ~rank-1, so the causal gain is "
                         "saturated by ONE axis-aligned atom; K near-copies of agree/validate). Its "
                         "failure is the empirical justification for the composed objective.")
    ap.add_argument("--lambda-causal", type=float, default=0.0,
                    help="CAUSAL CONTROL term, fully differentiable (no RL, no generation, no judge in "
                         "the loop): build each atom's extensional DiffMean direction from the CURRENT "
                         "soft presences, QR the set, intervene on the layer-L residual stream during "
                         "TEACHER-FORCED forwards of real D+/D- responses, and reward the symmetric "
                         "contrastive gain -- pushing coordinates toward the positive-class values must "
                         "raise per-token likelihood of positive responses MORE than negatives (and the "
                         "reverse for the negative push). Composes with recon+fid: atoms must describe "
                         "the data, be readable in activations, AND control the behavior. Try 0.5-1.0.")
    ap.add_argument("--causal-mode", choices=["contrast", "reft", "matrix"], default="contrast",
                    help="contrast: fixed class-contrast shift + D+/D- contrastive gain (v1). "
                         "reft: ReFT-r1-style gated multi-direction intervention, contrastive loss. "
                         "matrix: the teacher-forced TWIN OF THE AIE TABLE -- push each atom's direction "
                         "alone, measure likelihood gains on rows that EXPRESS each concept (per-atom "
                         "hi/lo presence rows from the current batch), train for DIAGONAL DOMINANCE: "
                         "G_jj up (own-concept gain = AIE surrogate), |G_jk| down (cross-gain = "
                         "selectivity surrogate). Duplicates have maximal off-diagonals, so distinctness "
                         "pressure is built in; --lambda-recon 0 is defensible under this mode.")
    ap.add_argument("--causal-atoms", type=int, default=2,
                    help="atoms pushed per step in matrix mode (random subset; unbiased). Memory scales "
                         "linearly (direct grad forwards): 2 atoms x 16 rows fits comfortably; raise "
                         "with care.")
    ap.add_argument("--causal-cross", type=float, default=1.0, help="off-diagonal penalty weight (matrix)")
    ap.add_argument("--causal-hilo", type=int, default=2, help="hi/lo rows per concept per step (matrix)")
    ap.add_argument("--causal-topk", type=int, default=8, help="tokens pooled into each gate (reft mode)")
    ap.add_argument("--causal-l1", type=float, default=0.001, help="L1 on non-TopK latents (reft mode)")
    ap.add_argument("--causal-sym", action="store_true",
                    help="restore the symmetric (push+ AND push-) causal gain. v1 measured push- as "
                         "inert (-0.005/-0.001 vs push+ -0.049/-0.059); the one-sided default S = "
                         "E_D+[dlogp|push+] - E_D-[dlogp|push+] halves causal compute per step -- "
                         "reinvest it in --causal-batch.")
    ap.add_argument("--causal-interv", choices=["add", "replace"], default="add",
                    help="add: h += (alpha/2) * Q(t+ - t-) (push along the class contrast INSIDE the "
                         "learned subspace). replace: h' = (I-QQ^T)h + Q t+/- (set subspace coords to "
                         "class-typical values -- stronger claim, more violent).")
    ap.add_argument("--causal-alpha", type=float, default=2.0)
    ap.add_argument("--causal-batch", type=int, default=16, help="responses per side per causal step")
    ap.add_argument("--causal-pool", type=int, default=384,
                    help="fixed pos/neg train rows whose base likelihoods are cached once at startup")
    ap.add_argument("--causal-clip", type=float, default=2.0,
                    help="clip per-response likelihood gains (nats/token) -- caps the suppress-the-"
                         "negatives escape and generic-fluency outliers")
    ap.add_argument("--causal-drop", type=float, default=0.2,
                    help="slot dropout on dictionary columns per causal step: stochastic unique-"
                         "contribution pressure (a column adding nothing earns nothing)")
    ap.add_argument("--causal-resp-tokens", type=int, default=128)
    ap.add_argument("--causal-prompt-tokens", type=int, default=192)
    ap.add_argument("--interv-prompts", type=int, default=0,
                    help="FINAL-EVAL causal read-off (works with or without --lambda-causal): steer the "
                         "base model along each atom's fidelity direction on this many held-out prompts, "
                         "judge all atom names on steered vs base generations, report per-atom AIE / "
                         "selectivity / length-ratio. 0 = off.")
    ap.add_argument("--interv-alpha", type=float, default=4.0)
    ap.add_argument("--axbench-factors", default="",
                    help="AxBench-style steering eval (Wu et al. 2025): comma-separated steering "
                         "factors (e.g. '1,2,4,6,8'). Per atom: generate under each factor on held-out "
                         "prompts, judge 3 subscores 0-9 (concept incorporation / instruction-following "
                         "/ fluency), HARMONIC-mean them (any failing axis kills the score -- the "
                         "principled len-ratio guard), select the best factor on half the prompts, "
                         "report on the other half. Empty = off.")
    ap.add_argument("--axbench-prompts", type=int, default=16)
    ap.add_argument("--axbench-max-new", type=int, default=96)
    ap.add_argument("--eval-judge", default=None,
                    help="HF model name of an INDEPENDENT judge: re-scores the final named taxonomy "
                         "(R^2 / AUC / Fidelity on TEST, response grain) after freeing the training judge. "
                         "Breaks the train/eval judge circularity; report this ladder in the paper.")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.003,
                    help="Adam moves each coordinate ~lr/step regardless of scale; token-embedding entries are "
                         "~0.01-0.03, so lr 0.02 exits the token manifold in ONE step (support crashes). Keep <=0.005.")
    ap.add_argument("--batch-texts", type=int, default=128,
                    help="texts per gradient step (halved for cross-fit). At 64 the K+1-param cross-fit is "
                         "NEGATIVE at init even with real signal -> the gradient's best move is to kill all "
                         "presence variance. 128 + --fit-ridge 0.1 keeps the objective positive-mean.")
    ap.add_argument("--fit-ridge", type=float, default=0.1,
                    help="ridge multiplier inside the differentiable cross-fit (x trace/d). 1e-2 overfits "
                         "half-batches -> negative J at init -> dead-feature collapse.")
    ap.add_argument("--judge-temp", type=float, default=1.5, help="scorer temperature (>1 softens saturated digits; "
                    "at saturation the presence gradient vanishes and dead atoms become absorbing)")
    ap.add_argument("--lambda-support", type=float, default=1.0,
                    help="hinge relu(tau - support_j) per atom: closes the 'zero all features -> R2=0 beats "
                         "negative' escape that killed the first run")
    ap.add_argument("--support-tau", type=float, default=0.05)
    ap.add_argument("--lambda-lang", type=float, default=0.1,
                    help="pull each soft token toward the token-embedding manifold: limits off-manifold judge "
                         "exploitation and keeps atoms snap-able. The snap score reports whatever survives.")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--fit-eval-n", type=int, default=512); ap.add_argument("--val-eval-n", type=int, default=256)
    ap.add_argument("--keep", type=int, default=0,
                    help="if >0 and < num-concepts: over-provision atoms, then backward-eliminate to this K by "
                         "val J (cheap insurance against bad local optima / redundant atoms)")
    ap.add_argument("--project-every", type=int, default=0,
                    help="PROJECTED GRADIENT onto the language manifold (0=off). Every N steps: snap each atom "
                         "to its best sentence (extensional match over init names + frozen-LM slot decodes + "
                         "contrastive proposals), RESET the slot to that sentence's token embeddings, clear "
                         "Adam state. The learned object is language THROUGHOUT training; soft excursions "
                         "between projections are optimizer internals. The best sentence set across "
                         "projections (by hard val J) is reported as the PROJECTED taxonomy. This is "
                         "rl_compose's slots+readout without RL: the judge path carries exact gradients and "
                         "the frozen-LM decode is just the projection's candidate generator. Try 75.")
    ap.add_argument("--probe-n", type=int, default=512, help="probe texts for snap-to-text extensional matching")
    ap.add_argument("--name-proposals", type=int, default=6, help="contrastive naming proposals per atom")
    ap.add_argument("--contrast-n", type=int, default=6); ap.add_argument("--example-chars", type=int, default=500)
    ap.add_argument("--spec-min", type=float, default=0.55, help="specificity gate on name candidates (0=off)")
    ap.add_argument("--gen-temp", type=float, default=0.9); ap.add_argument("--gen-max-new", type=int, default=32)
    ap.add_argument("--joint-scoring", action="store_true",
                    help="score all K concepts in ONE judge prompt (taxonomy context, two-pass "
                         "self-conditioned readout, per-step slot-order shuffling). Explaining-away "
                         "distinctness becomes part of the MEASUREMENT; ~Kx cheaper per step. The "
                         "final ladder also reports the independent-scored SOFT rung (deployment "
                         "condition + comparability) and an order-consistency diagnostic.")
    ap.add_argument("--judge-grain", choices=["response", "sentence"], default="response",
                    help="judge granularity. sentence: split each response into sentences, judge each "
                         "'prompt + one sentence' unit, aggregate per response with --sent-agg (max = "
                         "'the behavior occurred somewhere', differentiable). Behaviors are LOCAL, so "
                         "sentence units sharpen the judge task; pair with --acts-key sent_acts for the "
                         "activation side. Costs ~mean-sentences-per-response x judge calls.")
    ap.add_argument("--max-sents", type=int, default=6)
    ap.add_argument("--sent-agg", choices=["max", "mean"], default="max")
    ap.add_argument("--len-gate", type=float, default=0.45,
                    help="drop naming candidates whose judged presence correlates with the length "
                         "basis above this on the probe set (measurement-side length de-confound; "
                         "0.45 catches only blatant length thermometers; 0.3 ate empathy atoms. 0 = off.")
    ap.add_argument("--recon-mode", choices=["free", "dmtied"], default="free",
                    help="ABLATION: free (default) = cross-fitted ridge from presence to Y (K*D "
                         "free coefficients). dmtied = the INTERPRETABLE model as the objective: "
                         "each atom's signature is its presence-weighted DiffMean, only K scalars "
                         "fit -- trains for exactly what the DM-TIED ladder line reports.")
    ap.add_argument("--no-prompt-residual", action="store_true",
                    help="ABLATION: skip the prompt-ridge residualization (train target built from "
                         "raw response PCs; prompt-explainable variance stays in)")
    ap.add_argument("--eval-canonical", action="store_true",
                    help="ABLATION HARNESS: whatever prep/target the TRAINING used, score the FINAL "
                         "ladder (and PCA ceiling, fidelity, set selection) against the CANONICAL "
                         "target (prompt-residualized + len-deconfounded acts_wc_axisw) -- makes R^2 "
                         "comparable across ablation rows")
    ap.add_argument("--no-len-deconf", action="store_true",
                    help="disable the response-length de-confound (reproduce pre-2026-07-16 rows). "
                         "verbal_ceiling showed a length/truncation factor is load-bearing in acts_wc "
                         "(greedy k=2 pick); by default log-length joins the prompt regressors so the "
                         "target is behavior, not verbosity.")
    ap.add_argument("--pc-dim", type=int, default=256); ap.add_argument("--shrink", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--max-text-chars", type=int, default=600); ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    ap.add_argument("--judge-text-batch", type=int, default=0,
                    help="texts per judge chunk (0 = auto: scales 48 down with --max-text-chars). "
                         "The checkpointed soft-grid backward recomputes one chunk; 48 chunks of "
                         "1600-char texts OOM an 80GB card.")
    ap.add_argument("--max-n", type=int, default=0,
                    help="subsample the joined dataset to this many responses before splitting (0 = all). "
                         "Large corpora (longtok: 62k) make the final hard-judged ladder absurdly "
                         "expensive with no statistical benefit -- 6000 matches the sycophancy scale.")
    ap.add_argument("--lambda-div", type=float, default=0.0,
                    help="behavior-redundancy penalty: mean relu(|corr(P_j,P_k)| - div-tau)^2 over atom "
                         "pairs. Hinged so natural co-occurrence below tau is free -- the aim is "
                         "IDENTIFIABILITY (pin the basis within the stable subspace; init-restart r1 "
                         "showed rotation degeneracy) plus a mild entailment-ceiling lift, not corr=0.")
    ap.add_argument("--div-tau", type=float, default=0.2)
    ap.add_argument("--lambda-cyc", type=float, default=0.0,
                    help="CYCLE-CONSISTENT sayability: every --cyc-every steps, self-decode each atom "
                         "(Patchscopes splice), anchor it to its best-corr decode (distinct-gamma "
                         "selection, hysteresis), and reward corr(soft presence, anchor's hard "
                         "presence) each step. Pulls atoms toward describable FUNCTIONS -- the "
                         "principled replacement for lambda-lang's embedding-geometry proxy (which "
                         "overshoots into function-word space). Gradient flows into E only; anchors "
                         "are constants between refreshes. 0.2-0.3 typical.")
    ap.add_argument("--cyc-every", type=int, default=25); ap.add_argument("--cyc-warmup", type=int, default=100)
    ap.add_argument("--cyc-accept", choices=["named", "corr"], default="named",
                    help="anchor acceptance criterion. corr: challenger beats incumbent's own-atom "
                         "corr on the held split (proxy: blind to Y-relevance and cross-name error "
                         "redundancy -- the square-law killers of named R^2). named: challenger must "
                         "INCREASE the joint named R^2 (ridge from all anchors to Y, fit FITe, scored "
                         "on the held CYCV split) while keeping a minimum own-corr -- coordinate "
                         "ascent on the deliverable itself.")
    ap.add_argument("--resume-from", default=None,
                    help="STAGE 2: initialize soft prompts from a trained checkpoint's soft_prompts "
                         "and continue training (typically with --lambda-cyc and --cyc-warmup 0: "
                         "sayability annealing as a pure post-process on the validated stage-1 atoms; "
                         "same seed = same splits, so stage-1 vs stage-2 is a controlled comparison).")
    ap.add_argument("--cyc-samples", type=int, default=8); ap.add_argument("--cyc-gamma", type=float, default=0.5)
    ap.add_argument("--cyc-corr-floor", type=float, default=0.6,
                    help="named acceptance: candidate own-corr (CYCV) must be >= min(max(incumbent, "
                         "0.25), THIS). Below the floor own-corr may only RISE; above it, trades down "
                         "to the floor are allowed. Replaces the old incumbent-0.05 rule, whose decay "
                         "compounded across swaps and walked fidelity down (K=8 fid_name +0.52).")
    ap.add_argument("--pca-ceiling", default=None,
                    help="e.g. '4,8': print the PCA-K ceiling of the target (max ladder R^2 for ANY "
                         "K linear features) and exit before loading the judge. Reference denominators "
                         "for SOFT/NAMED R^2.")
    ap.add_argument("--init-seed", type=int, default=-1,
                    help="dedicated rng for the soft-prompt initialization (-1 = follow --seed). "
                         "Varies the init while --seed keeps splits/probe/eval identical: the "
                         "random-restart identifiability test. With --init-concepts it shuffles the "
                         "warm-start assignment; without, each value gives fresh random-vocab inits.")
    args = ap.parse_args()
    acquire("soft_" + args.out.replace("/", "_"))                # same --out twice = refused
    gpu_guard(18.0)                                              # don't start on a full GPU
    dev = torch.device("cuda"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    jb = args.judge_text_batch or max(8, (48 * 600) // args.max_text_chars)
    print(f"[soft] judge text_batch = {jb}", flush=True)
    cfg = yaml.safe_load(open(args.config)); mname = cfg["model"]["name"]; dt = cfg["model"].get("dtype", "bfloat16")

    # ---- join response acts + prompt acts + labels + texts (identical to rl_compose/nl_dictionary) ----
    u = np.load(args.units, allow_pickle=True)
    if args.acts_key == "sent_acts":
        # one row per SENTENCE (sent_resp = parent response text) -> mean-pool sentence acts per
        # response. NB: setdefault-style joining would silently keep only the FIRST sentence.
        Rmap = {}
        for h, p in zip(np.asarray(u["sent_acts"], np.float64), [str(x) for x in u["sent_resp"]]):
            Rmap.setdefault(" ".join(p.split()), []).append(h)
        Rmap = {k: np.mean(v, 0) for k, v in Rmap.items()}
        print(f"[soft] sent_acts: pooled {len(u['sent_acts'])} sentence acts -> {len(Rmap)} responses", flush=True)
    else:
        respkey = {"resp_acts": "resp_text"}[args.acts_key]
        Rmap = {}
        for h, t in zip(np.asarray(u[args.acts_key], np.float64), [str(x) for x in u[respkey]]):
            Rmap.setdefault(" ".join(t.split()), h)
    pa = np.load(args.prompt_acts, allow_pickle=True); Hp_all = np.asarray(pa["prompt_acts"], np.float64)
    rtxt = [str(x) for x in pa["response_text"]]; ptxt = [str(x) for x in pa["prompt_text"]]; lab = np.asarray(pa["label"], np.int64)
    Hr, Hp, y, resp, prm = [], [], [], [], []
    for hp, rt, pt, l in zip(Hp_all, rtxt, ptxt, lab):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); Hp.append(hp); y.append(int(l)); resp.append(rt); prm.append(pt)
    Hr, Hp, y = np.array(Hr), np.array(Hp), np.array(y); n = len(y)
    if args.max_n and n > args.max_n:
        keep = rng.permutation(n)[:args.max_n]
        Hr, Hp, y = Hr[keep], Hp[keep], y[keep]
        resp = [resp[i] for i in keep]; prm = [prm[i] for i in keep]; n = args.max_n
        print(f"[soft] --max-n: subsampled to {n} responses (label balance {y.mean():.2f})", flush=True)
    perm = rng.permutation(n); te = np.zeros(n, bool); te[perm[:max(1, int(n * args.heldout_frac))]] = True; tr = ~te

    mu_r = Hr[tr].mean(0)
    Zr, Wr = _pcs_w((Hr - mu_r)[tr], Hr - mu_r, args.pc_dim)     # keep Wr: pc-dirs -> hidden space
    Zp = _pcs((Hp - Hp[tr].mean(0))[tr], Hp - Hp[tr].mean(0), args.pc_dim)
    A = Zp[tr]; lam = args.ridge * np.trace(A.T @ A) / A.shape[1]
    Wm = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zr[tr])
    if args.no_prompt_residual:
        Res = Zr.copy()
        print("[soft] ABLATION: prompt-ridge residualization OFF (Res = response PCs)", flush=True)
    else:
        Res = Zr - Zp @ Wm
    if not args.no_len_deconf:
        # EXACT projection (OLS fit on train), not a ridge column: inside the prompt ridge the
        # trace-scaled lambda crushes a unit-variance regressor (first attempt left corr 0.40).
        # Basis includes the TRUNCATION INDICATOR -- the judge sees cut text iff len > max chars,
        # a threshold effect no smooth length term captures.
        rl = np.array([len(r) for r in resp], np.float64)
        L = np.stack([np.log1p(rl), rl, (rl > args.max_text_chars).astype(np.float64)], 1)
        L = (L - L[tr].mean(0)) / (L[tr].std(0) + 1e-9)
        L1 = np.concatenate([L, np.ones((len(L), 1))], 1)
        B = np.linalg.lstsq(L1[tr], Res[tr], rcond=None)[0]
        Res = Res - L1 @ B
        lc = max(abs(float(np.corrcoef(L[tr][:, k], Res[tr, j])[0, 1]))
                 for j in range(8) for k in range(3) if L[tr][:, k].std() > 1e-9)
        print(f"[soft] length de-confound ON (exact, 3-basis): max |corr(length, Res PC1-8)| = "
              f"{lc:.3f} after removal", flush=True)
    p, q = Res[tr][y[tr] == 1], Res[tr][y[tr] == 0]
    wl = np.linalg.solve(_shrinkc(np.cov(p, rowvar=False) + np.cov(q, rowvar=False), args.shrink), p.mean(0) - q.mean(0))
    s = Res @ wl; s = (s - s[tr].mean()) / (s[tr].std() + 1e-9)
    if np.corrcoef(s[tr], y[tr])[0, 1] < 0:
        s = -s
    if args.target.startswith("acts_wc"):
        # within-class multi-dim target (nl_dictionary/rl_compose acts_wc): PC basis fit on
        # TRAIN positive-class residuals, projected onto everyone.
        sy = np.where(tr & (y == 1))[0]; mu1 = Res[sy].mean(0)
        Y, Wy = _pcs_w(Res[sy] - mu1, Res - mu1, args.acts_dims)   # keep Wy: Res -> Y basis
        print(f"[soft] {args.target} target: within-class PCs (fit on {len(sy)} pos-train), dims={Y.shape[1]}", flush=True)
    elif args.target == "acts_pc":
        # ABLATION target: plain train-fit PCs of Res, NO class conditioning -- "just the
        # activation variance", the minimal-prep target
        mu_t = Res[tr].mean(0)
        Y, Wy = _pcs_w(Res[tr] - mu_t, Res - mu_t, args.acts_dims)
        print(f"[soft] acts_pc target: plain PCs (no class conditioning), dims={Y.shape[1]}", flush=True)
    else:
        Y = s[:, None]
    w_axn = np.ones(Y.shape[1]) if Y.shape[1] > 1 else None
    if args.target == "acts_wc_axisw":
        w_ax = np.abs(np.array([float(np.corrcoef(Y[tr][:, d], s[tr])[0, 1]) for d in range(Y.shape[1])]))
        w_axn = w_ax / (w_ax.max() + 1e-9)
        Y = Y * w_axn[None, :]                                   # relevance-weighted target (train-fit weights)
        print(f"[soft] axisw: PCs weighted by |corr(PC, axis)| -- mean weight "
              f"{(w_ax / (w_ax.max() + 1e-9)).mean():.2f}, top dim corr {w_ax.max():.2f}", flush=True)

    if args.eval_canonical:
        # CANONICAL target for the final ladder, independent of ablation flags: prompt-residual +
        # exact length deconfound + within-class PCs + axisw. Training sees the ABLATED (Res,s,Y);
        # the final eval swaps these in so every ablation row shares one yardstick.
        Res_c = Zr - Zp @ Wm
        rl_c = np.array([len(r) for r in resp], np.float64)
        Lc = np.stack([np.log1p(rl_c), rl_c, (rl_c > args.max_text_chars).astype(np.float64)], 1)
        Lc = (Lc - Lc[tr].mean(0)) / (Lc[tr].std(0) + 1e-9)
        L1c = np.concatenate([Lc, np.ones((len(Lc), 1))], 1)
        Res_c = Res_c - L1c @ np.linalg.lstsq(L1c[tr], Res_c[tr], rcond=None)[0]
        pc_, qc_ = Res_c[tr][y[tr] == 1], Res_c[tr][y[tr] == 0]
        wlc = np.linalg.solve(_shrinkc(np.cov(pc_, rowvar=False) + np.cov(qc_, rowvar=False),
                                       args.shrink), pc_.mean(0) - qc_.mean(0))
        s_c = Res_c @ wlc; s_c = (s_c - s_c[tr].mean()) / (s_c[tr].std() + 1e-9)
        if np.corrcoef(s_c[tr], y[tr])[0, 1] < 0:
            s_c = -s_c
        syc = np.where(tr & (y == 1))[0]; mu1c = Res_c[syc].mean(0)
        # the canonical yardstick keeps ITS OWN dimensionality: reading it off args.acts_dims
        # would make an --acts-dims ablation score against a target that moved with it, i.e.
        # no shared yardstick at all (the one thing this flag exists to provide)
        Y_c, Wy_c = _pcs_w(Res_c[syc] - mu1c, Res_c - mu1c, args.canon_acts_dims)
        waxc = np.abs(np.array([float(np.corrcoef(Y_c[tr][:, d_], s_c[tr])[0, 1])
                                for d_ in range(Y_c.shape[1])]))
        waxc_n = waxc / (waxc.max() + 1e-9)
        Y_c = Y_c * waxc_n[None, :]
        print("[soft] eval-canonical ON: final ladder will score against the canonical "
              "acts_wc_axisw target regardless of training prep", flush=True)

    texts = [f"User asked: {prm[i][:args.max_prompt_chars]}\n\nResponse: {resp[i][:args.max_text_chars]}" for i in range(n)]

    # ---- sentence-grain judging units: each response -> prompt+sentence units, aggregated back per response ----
    units_by_resp, unit_text = [], []
    if args.judge_grain == "sentence":
        import re
        for i in range(n):
            ss = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", resp[i][:args.max_text_chars])
                  if len(x.strip()) > 2][:args.max_sents] or [resp[i][:args.max_text_chars]]
            ix = []
            for t in ss:
                ix.append(len(unit_text))
                unit_text.append(f"User asked: {prm[i][:args.max_prompt_chars]}\n\nOne sentence from the response: {t}")
            units_by_resp.append(np.array(ix))
        print(f"[soft] sentence grain: {len(unit_text)} units / {n} responses "
              f"(mean {len(unit_text) / n:.1f} per resp, agg={args.sent_agg})", flush=True)

    def _judge_texts(idx):                                       # -> (flat unit texts, response segment bounds|None)
        if args.judge_grain == "response":
            return [texts[i] for i in idx], None
        segs = np.cumsum([0] + [len(units_by_resp[i]) for i in idx])
        return [unit_text[u] for i in idx for u in units_by_resp[i]], segs

    def _agg_np(P, segs):                                        # [K, units] -> [K, responses]
        out = np.zeros((P.shape[0], len(segs) - 1), np.float32)
        for r in range(len(segs) - 1):
            seg = P[:, segs[r]:segs[r + 1]]
            out[:, r] = seg.max(1) if args.sent_agg == "max" else seg.mean(1)
        return out

    pv = rng.permutation(np.where(tr)[0])
    VALe = pv[:min(args.val_eval_n, len(pv) // 4)]
    FITe = pv[len(VALe):len(VALe) + min(args.fit_eval_n, len(pv) - len(VALe))]
    TRAIN = pv[len(VALe):]                                       # gradient batches sample from here (excl. VALe)
    TE = np.where(te)[0]
    PROBE = TRAIN[rng.permutation(len(TRAIN))[:min(args.probe_n, len(TRAIN))]]
    CYCV = np.setdiff1d(TRAIN, PROBE)[:256]                      # cyc-anchor ACCEPTANCE split:
                                                                 # disjoint from the selection probe
    print(f"[soft] n={n} train={len(TRAIN)} val={len(VALe)} test={len(TE)}  axis AUC(s,label)={_auc(s[TE], y[TE]):.3f}",
          flush=True)

    # PCA-K CEILING: best possible ladder R^2 for ANY K scalar features. A K-feature linear
    # predictor has rank <= K, and the best rank-K linear predictor of Y is projection onto Y's
    # own top-K train PCs -- an oracle no presence-based method can beat. Denominator matches
    # _r2_np (test-mean centered).
    def _pca_ceiling(k_):
        fit_ix = np.concatenate([FITe, VALe])
        mu_y = Y[fit_ix].mean(0)
        _, _, Vt = np.linalg.svd(Y[fit_ix] - mu_y, full_matrices=False)
        W = Vt[:k_]; Yt = Y[TE]
        pred = mu_y + (Yt - mu_y) @ W.T @ W
        return float(1 - ((Yt - pred) ** 2).sum() / (((Yt - Yt.mean(0)) ** 2).sum() + 1e-9))

    if args.pca_ceiling:
        for k_ in [int(x) for x in args.pca_ceiling.split(",")]:
            print(f"[soft] PCA-{k_} ceiling (TEST): R^2={_pca_ceiling(k_):.3f}  "
                  f"(max for any {k_} linear features)", flush=True)
        return

    judge = FrozenJudge(mname, device=dev, dtype=dt, max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)
    tok = judge.tok; d = judge.embed.weight.shape[1]; K = args.num_concepts

    # ---- causal-control machinery (teacher-forced intervened likelihood; all differentiable) ----
    if args.lambda_causal > 0:
        layer = int(cfg["model"].get("layer", 20))
        Hc_t = torch.tensor(Hr - mu_r, dtype=torch.float32, device=dev)      # centered hidden acts [n, D]
        Wr_t = torch.tensor(Wr, dtype=torch.float32, device=dev)             # pc->hidden basis [D, pc]
        cstate = {"d": None, "Q": None, "z": None, "V": None, "am": None}    # hook payload

        def _chook(_m, _inp, out):
            o = out[0] if isinstance(out, tuple) else out
            if cstate["d"] is not None:                                      # add: constant shift
                o = o + cstate["d"].to(o.dtype)
            elif cstate["Q"] is not None:                                    # replace: swap subspace coords
                Q = cstate["Q"].to(o.dtype)
                o = o - (o @ Q) @ Q.T + (Q @ cstate["z"].to(o.dtype))
            elif cstate["V"] is not None:                                    # reft: detection-gated sum of
                V = cstate["V"]; of = o.float()                              # ALL K directions at once
                a = torch.relu(of @ V.T) / cstate["sig"][None, None, :]      # [B, T, K] token latents
                if cstate["am"] is not None:
                    a = a * cstate["am"][:, :, None].to(a.dtype)
                kk = min(cstate["topk"], a.shape[1])
                topv = a.topk(kk, dim=1).values                              # TopK over tokens
                g = topv.mean(1)                                             # [B, K] per-example gates
                cstate["l1_out"] = (a.sum() - topv.sum()) / max(a.numel() - topv.numel(), 1)
                o = (of + cstate["alpha_g"] * (g @ V)[:, None, :]).to(o.dtype)
            else:
                return None
            return (o,) + out[1:] if isinstance(out, tuple) else o
        judge.layers[layer].register_forward_hook(_chook)

        pos_pool = np.array([i for i in TRAIN if y[i] == 1][:args.causal_pool])
        neg_pool = np.array([i for i in TRAIN if y[i] == 0][:args.causal_pool])
        _ctok = {}                                                           # row -> (ids, prompt_len)

        def _crow(i):
            if i not in _ctok:
                pid = tok(tok.apply_chat_template([{"role": "user", "content": prm[i][:args.max_prompt_chars]}],
                                                  tokenize=False, add_generation_prompt=True),
                          add_special_tokens=False).input_ids[:args.causal_prompt_tokens]
                rid = tok(resp[i], add_special_tokens=False).input_ids[:args.causal_resp_tokens]
                _ctok[i] = (pid + rid, len(pid))
            return _ctok[i]

        def _clogp(rows, grad=False):
            """Mean per-token log p of each row's RESPONSE under the (possibly hooked) model."""
            seqs = [_crow(i) for i in rows]
            Lm = max(len(sq) for sq, _ in seqs)
            ids = torch.full((len(rows), Lm), tok.pad_token_id, dtype=torch.long, device=dev)
            am = torch.zeros((len(rows), Lm), dtype=torch.long, device=dev)
            rmask = torch.zeros((len(rows), Lm), dtype=torch.bool, device=dev)
            for r, (sq, lp) in enumerate(seqs):
                ids[r, :len(sq)] = torch.tensor(sq, device=dev); am[r, :len(sq)] = 1
                rmask[r, lp:len(sq)] = True
            if cstate["V"] is not None:
                cstate["am"] = am
            if grad is None:                                     # ambient mode: required under
                import contextlib                                # torch.utils.checkpoint (no-grad
                ctx = contextlib.nullcontext()                   # forward, grad during recompute);
            else:                                                # forcing enable_grad there builds a
                ctx = torch.enable_grad() if grad else torch.no_grad()   # second live graph -> double
            with ctx:                                            # backward
                outs = []                                        # row-chunked: full-vocab logits are the
                for c0 in range(0, len(rows), 12):               # memory bound ([rows, T, 152k] x fp32
                    sl = slice(c0, c0 + 12)                      # copies, retained for backward)
                    lg = judge.model(input_ids=ids[sl], attention_mask=am[sl]).logits.float()
                    lp_tok = torch.log_softmax(lg[:, :-1], -1).gather(-1, ids[sl][:, 1:, None]).squeeze(-1)
                    m = rmask[sl][:, 1:].float()
                    outs.append((lp_tok * m).sum(1) / (m.sum(1) + 1e-9))
                return torch.cat(outs, 0)

        base_lp = {}                                                         # base likelihoods, cached once
        allp = np.concatenate([pos_pool, neg_pool])
        print(f"[soft] causal: caching base likelihoods for {len(allp)} responses ...", flush=True)
        for c0 in range(0, len(allp), 16):
            rows = allp[c0:c0 + 16]
            for i, v in zip(rows, _clogp(rows).cpu().tolist()):
                base_lp[int(i)] = v
        print(f"[soft] causal ready: interv={args.causal_interv} alpha={args.causal_alpha} "
              f"pool={len(pos_pool)}+{len(neg_pool)} layer={layer}", flush=True)

        def _causal_gain(P_batch, mb, grad=True, nb=None, rows=None):
            """Symmetric contrastive teacher-forced gain of the CURRENT dictionary (differentiable
            through presences -> extensional dirs -> QR -> intervention)."""
            nb = nb or args.causal_batch
            w = P_batch.clamp(0, 1)                                          # [K, B] current presences
            Rb = Rt[torch.as_tensor(mb, device=dev)]
            posm = (w / (w.sum(1, keepdim=True) + 1e-6)) @ Rb
            negm = ((1 - w) / ((1 - w).sum(1, keepdim=True) + 1e-6)) @ Rb
            Vh = F.normalize((posm - negm) @ Wr_t.T, dim=1)                  # [K, D] hidden-space dirs
            if args.causal_drop > 0 and grad:
                keepm = torch.rand(Vh.shape[0], device=dev) > args.causal_drop
                if keepm.any():
                    Vh = Vh[keepm]
            if args.causal_mode == "matrix":
                # per-atom gain matrix with diagonal dominance (the trainable AIE/selectivity twin)
                sig = torch.stack([(Hc_t @ Vh[j]).std() for j in range(Vh.shape[0])]).clamp_min(1e-3)
                Kc = Vh.shape[0]
                Pb_np = P_batch.detach().cpu().numpy()
                # EXCLUSIVE hi rows: top of P_k - mean(P_others). Plain top-P_k rows overlap heavily
                # across atoms (strong responses express several concepts), so cross-gain conflated
                # row CO-OCCURRENCE with causal off-target -- the v1 confound.
                excl = Pb_np - (Pb_np.sum(0, keepdims=True) - Pb_np) / max(Pb_np.shape[0] - 1, 1)
                hi = [mb[np.argsort(excl[k])[-args.causal_hilo:]] for k in range(Kc)]
                lo = [mb[np.argsort(Pb_np[k])[:args.causal_hilo]] for k in range(Kc)]
                rows_all = np.array([i for k in range(Kc) for i in list(hi[k]) + list(lo[k])])
                need_b = [int(i) for i in rows_all if int(i) not in base_lp]
                if need_b:                                       # lazy base-likelihood cache
                    with torch.no_grad():
                        for c0 in range(0, len(need_b), 16):
                            rr = need_b[c0:c0 + 16]
                            for i2, v2 in zip(rr, _clogp(np.array(rr)).cpu().tolist()):
                                base_lp[int(i2)] = v2
                push = (rng.choice(Kc, min(args.causal_atoms, Kc), replace=False)
                        if grad else np.arange(Kc))
                diag, cross = [], []
                for j in push:
                    # direct grad forwards (the reft-mode pattern, proven): torch.utils.checkpoint
                    # around a forward whose HOOK modifies outputs produced double-backward errors;
                    # memory is bounded by --causal-atoms/--causal-hilo instead (2x16 rows < reft 2x32)
                    cstate["d"] = args.causal_alpha * sig[j] * Vh[j]
                    lp_all = _clogp(rows_all, grad=grad)
                    cstate["d"] = None
                    dl = torch.stack([lp_all[k2] - base_lp[int(i)] for k2, i in enumerate(rows_all)]
                                     ).clamp(-args.causal_clip, args.causal_clip)
                    G_row = []
                    for k in range(Kc):                          # G_jk = gain(hi_k) - gain(lo_k)
                        o0 = 2 * args.causal_hilo * k
                        G_row.append(dl[o0:o0 + args.causal_hilo].mean()
                                     - dl[o0 + args.causal_hilo:o0 + 2 * args.causal_hilo].mean())
                    G_row = torch.stack(G_row)
                    diag.append(G_row[j])
                    cross.append(torch.cat([G_row[:j], G_row[j + 1:]]).abs().mean())
                diag = torch.stack(diag).mean(); cross = torch.stack(cross).mean()
                gains = {"pP": diag, "pN": cross}                # pP=diag, pN=cross in the logs
                return diag - args.causal_cross * cross, gains
            if args.causal_mode == "reft":
                sig = torch.stack([(Hc_t @ Vh[j]).std() for j in range(Vh.shape[0])]).clamp_min(1e-3)
                cstate.update(V=Vh, sig=sig, alpha_g=args.causal_alpha, topk=args.causal_topk, l1_out=None)
                # CONTRASTIVE gated loss: v1 (positive-only) was exploited by SELF-AMPLIFICATION --
                # gates fire on what a text already contains, and amplifying detected features raises
                # likelihood of ANY feature-congruent text (TEST: D+ +0.06 but D- +0.18). Scoring the
                # SAME gated push on D- and subtracting cancels the exploit; specific gain survives.
                rp = rows[0] if rows is not None else rng.choice(pos_pool, nb, replace=False)
                rn = rows[1] if rows is not None else rng.choice(neg_pool, nb, replace=False)
                lp = _clogp(rp, grad=grad)
                l1p = cstate["l1_out"]; cstate["l1_out"] = None
                gp = torch.stack([lp[k2] - base_lp[int(i)] for k2, i in enumerate(rp)]
                                 ).clamp(-args.causal_clip, args.causal_clip).mean()
                lpn = _clogp(rn, grad=grad)
                l1n = cstate["l1_out"]
                gn = torch.stack([lpn[k2] - base_lp[int(i)] for k2, i in enumerate(rn)]
                                 ).clamp(-args.causal_clip, args.causal_clip).mean()
                zero = torch.zeros((), device=dev)
                l1 = 0.5 * ((l1p if l1p is not None else zero) + (l1n if l1n is not None else zero))
                gains = {"pP": gp, "pN": gn, "l1": l1}
                cstate["V"] = None; cstate["am"] = None
                return (gp - gn) - args.causal_l1 * l1, gains
            Q, _ = torch.linalg.qr(Vh.T)                                     # [D, K'] orthonormal
            tp = (Hc_t[torch.as_tensor(pos_pool, device=dev)] @ Q).mean(0)   # class-typical coords
            tn = (Hc_t[torch.as_tensor(neg_pool, device=dev)] @ Q).mean(0)
            rp, rn = rows if rows is not None else (rng.choice(pos_pool, nb, replace=False),
                                                     rng.choice(neg_pool, nb, replace=False))
            gains = {}
            signs = ((+1, "p"), (-1, "n")) if args.causal_sym else ((+1, "p"),)
            for sgn, tag in signs:
                if args.causal_interv == "add":
                    cstate["d"] = sgn * (args.causal_alpha / 2) * (Q @ (tp - tn)); cstate["Q"] = None
                else:
                    cstate["Q"] = Q; cstate["z"] = tp if sgn > 0 else tn; cstate["d"] = None
                for rows, rtag in ((rp, "P"), (rn, "N")):
                    lp = _clogp(rows, grad=grad)
                    dl = torch.stack([lp[k] - base_lp[int(i)] for k, i in enumerate(rows)])
                    gains[tag + rtag] = dl.clamp(-args.causal_clip, args.causal_clip).mean()
                cstate["d"] = cstate["Q"] = cstate["z"] = None
            if args.causal_sym:
                return 0.5 * ((gains["pP"] - gains["pN"]) + (gains["nN"] - gains["nP"])), gains
            return gains["pP"] - gains["pN"], gains

    # ---- the ONLY trainable parameters: K soft concept prompts, warm-started from distinct names ----
    E = torch.zeros(K, args.n_tokens, d)
    names0 = ([str(c) for c in np.load(args.init_concepts, allow_pickle=True)["names"]
               if "control" not in str(c).lower()] if args.init_concepts else [])
    irng = np.random.default_rng(args.init_seed) if args.init_seed >= 0 else rng
    if names0 and args.init_seed >= 0:
        names0 = [names0[i] for i in irng.permutation(len(names0))]
    with torch.no_grad():
        for j in range(K):
            if names0:
                ids = tok(names0[j % len(names0)], add_special_tokens=False).input_ids[:args.n_tokens]
            else:
                ids = irng.choice(tok.vocab_size, args.n_tokens).tolist()
            e = judge.embed(torch.tensor(ids, device=dev)).detach().float().cpu()
            E[j, :len(e)] = e
            if len(e) < args.n_tokens:
                E[j, len(e):] = e[-1]
    if args.resume_from:
        _rk = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        _Er = _rk["soft_prompts"].float()
        assert tuple(_Er.shape) == tuple(E.shape), \
            f"--resume-from shape {tuple(_Er.shape)} != configured {tuple(E.shape)}"
        E = _Er.clone()
        print(f"[soft] resumed soft prompts from {args.resume_from}", flush=True)
    E = torch.nn.Parameter(E.to(dev)); opt = torch.optim.Adam([E], lr=args.lr)
    with torch.no_grad():                                        # vocab manifold (for the lang pull), bf16 to fit
        En = F.normalize(judge.embed.weight.float(), dim=1).to(torch.bfloat16)
    print(f"[soft] trainable params: {K}x{args.n_tokens}x{d} soft tokens = {E.numel()} (nothing else)", flush=True)

    Rt = torch.tensor(Res, dtype=torch.float32, device=dev)     # activation residual for the diff fid term
    Wy_t = torch.tensor((Wy * (w_axn[None, :] if w_axn is not None else 1.0)) if Y.shape[1] > 1
                        else np.zeros((Res.shape[1], 1)),
                        dtype=torch.float32, device=dev)         # Res -> Y coords (fixed basis)

    @torch.no_grad()
    def _softP(Emat, idx, joint=None):                           # hard-eval soft presence (no grad)
        tl, segs = _judge_texts(idx)
        joint = args.joint_scoring if joint is None else joint
        if joint:
            Kc = Emat.shape[0]                                   # average two mirrored slot orders
            P = 0.5 * (judge.presence_grid_soft_joint(Emat, tl, text_batch=jb, temp=args.judge_temp,
                                                      order=list(range(Kc)))
                       + judge.presence_grid_soft_joint(Emat, tl, text_batch=jb, temp=args.judge_temp,
                                                        order=list(range(Kc))[::-1]))
            P = P.float().cpu().numpy()
        else:
            P = judge.presence_grid_soft(Emat, tl, text_batch=jb,
                                         temp=args.judge_temp, mode="scale").float().cpu().numpy()
        return P if segs is None else _agg_np(P, segs)

    def _fidelity(P_dir, dir_idx, r_eval, eval_idx):
        """Paper Fidelity per atom: v_j = presence-weighted DiffMean over Res[dir_idx] (the same
        protocol as the expert-taxonomy baseline), a_j = <norm h, norm v_j> on the DISJOINT eval
        split, Fidelity_j = spearman(a_j, r_eval_j). r_eval = the judged behavior score on the
        eval split (soft presence during training; the snapped NAME's hard presence at final).
        Returns (fidelities [K'], directions [K', pc])."""
        Rf, Re = Res[dir_idx], Res[eval_idx]
        Ren = Re / (np.linalg.norm(Re, axis=1, keepdims=True) + 1e-9)
        fids = np.zeros(len(P_dir)); dirs = np.zeros((len(P_dir), Res.shape[1]))
        for j in range(len(P_dir)):
            w = np.clip(P_dir[j], 0, 1)
            if w.sum() < 1e-6 or (1 - w).sum() < 1e-6 or r_eval[j].std() < 1e-9:
                continue
            v = (w / w.sum()) @ Rf - ((1 - w) / (1 - w).sum()) @ Rf
            nv = np.linalg.norm(v)
            if nv < 1e-9:
                continue
            dirs[j] = v / nv
            fids[j] = _spear(Ren @ dirs[j], r_eval[j])
        return fids, dirs

    def _jhard(Pf, Pv, yf, yv):                                  # np: fit FITe -> R^2 on VALe
        return _r2_np(_ridge_np(Pf.T, yf), Pv.T, yv)

    # ---- language machinery: generation, slot decode (rl_compose's readout, frozen), snap ----
    @torch.no_grad()
    def _gen(prompts, chunk=12):
        outs = []
        tok.padding_side = "left"
        for i in range(0, len(prompts), chunk):
            enc = tok(prompts[i:i + chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=3584, add_special_tokens=False).to(dev)
            out = judge.model.generate(**enc, do_sample=args.gen_temp > 0, temperature=max(args.gen_temp, 1e-3),
                                       top_p=0.95, max_new_tokens=args.gen_max_new, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g, skip_special_tokens=True) for g in out[:, enc["input_ids"].shape[1]:]]
        tok.padding_side = "right"
        return outs

    _nl_ids = {i for i, t in enumerate(tok.convert_ids_to_tokens(list(range(tok.vocab_size))))
               if t and ("Ċ" in t or "\n" in t)}
    _readout_ids = torch.tensor(tok(" — in replying, the assistant tends to", add_special_tokens=False).input_ids,
                                device=dev)

    @torch.no_grad()
    def _decode_slots(Emat, max_new=24):                         # slot -> phrase via the frozen LM (no training)
        B = Emat.shape[0]; emb = judge.embed
        cur = torch.cat([Emat.to(emb.weight.dtype), emb(_readout_ids)[None].expand(B, -1, -1)], 1)
        am = torch.ones(B, cur.shape[1], dtype=torch.long, device=dev)
        out = [[] for _ in range(B)]; done = torch.zeros(B, dtype=torch.bool, device=dev)
        for _ in range(max_new):
            lg = judge.model(inputs_embeds=cur, attention_mask=am).logits[:, -1]
            nxt = lg.argmax(-1)
            for b in range(B):
                if not done[b]:
                    t = int(nxt[b])
                    if t == tok.eos_token_id or (t in _nl_ids and out[b]):
                        done[b] = True
                    elif t not in _nl_ids:
                        out[b].append(t)
            cur = torch.cat([cur, emb(nxt).unsqueeze(1)], 1)
            am = torch.cat([am, torch.ones(B, 1, dtype=torch.long, device=dev)], 1)
            if bool(done.all()):
                break
        return [_phrase(tok.decode(g)) for g in out]

    hardc = {"P": {}, "F": {}, "V": {}, "T": {}, "C": {}}        # hard presence: judge each (name, split) ONCE ever
    _sidx = {"P": PROBE, "F": FITe, "V": VALe, "T": TE, "C": CYCV}

    def _hardP(cs, split):
        need = [c for c in dict.fromkeys(cs) if c not in hardc[split]]
        if need:
            tl, segs = _judge_texts(_sidx[split])
            Ph = judge.presence_grid(need, tl, text_batch=jb, grain="response", mode="scale")
            if segs is not None:
                Ph = _agg_np(Ph, segs)
            for c, row in zip(need, Ph):
                hardc[split][c] = row.astype(np.float32)
        return np.stack([hardc[split][c] for c in cs])

    def _snap_atoms(Emat, atoms):
        """Name atoms by EXTENSIONAL match: the candidate whose HARD judged presence best correlates
        with the atom's soft presence on the probe set. Pool = init names + frozen-LM slot decodes +
        per-atom contrastive proposals. Returns (names, snap corrs, soft probe presence [K, probe])."""
        Pp = _softP(Emat, PROBE)
        pool = list(dict.fromkeys(names0 + _decode_slots(Emat)))
        prompts = []
        for j in atoms:
            order = np.argsort(-Pp[j]); pl = max(4 * args.contrast_n, 24)
            for _m in range(args.name_proposals):
                Aix = rng.choice(order[:pl], args.contrast_n, replace=False)
                Bix = rng.choice(order[-pl:], args.contrast_n, replace=False)
                # random WINDOW, not the head: head-truncation at example_chars means the namer
                # never sees the middle/end of long texts -> every proposal describes trace
                # OPENINGS ("start responses with...") -- the longtok naming failure mode
                def _win(t):
                    w = args.example_chars
                    return t if len(t) <= w else t[rng.integers(0, len(t) - w):][:w]
                exA = "\n\n".join(f"[A{i+1}] {_win(texts[PROBE[a]])}" for i, a in enumerate(Aix))
                exB = "\n\n".join(f"[B{i+1}] {_win(texts[PROBE[b]])}" for i, b in enumerate(Bix))
                prompts.append(
                    judge.u_open + "Below are replies by an AI assistant (each shown with the user message it "
                    "answers). The GROUP A replies share ONE specific, observable response behavior that the "
                    "GROUP B replies lack.\n\nGROUP A:\n" + exA + "\n\nGROUP B:\n" + exB + "\n\n"
                    "Name that ONE behavior. Be specific and behavioral (what the reply DOES, not what topic it "
                    "is about). Answer with a single short clause completing \"the assistant tends to ...\" -- "
                    "output only the clause." + judge.a_open + "the assistant tends to")
        pool += [_phrase(t) for t in _gen(prompts)]
        pool = [c for c in dict.fromkeys(pool) if c and len(c.split()) >= 3]
        if args.spec_min > 0 and pool:
            sp = judge.coherence_scale(pool)
            gated = [c for c, v in zip(pool, sp) if v >= args.spec_min]
            pool = gated or pool
        Ph = _hardP(pool, "P")
        if args.len_gate > 0:                                    # measurement-side length de-confound:
            rlp = np.array([len(resp[i]) for i in PROBE], np.float64)   # a length/truncation detector is
            Lb = np.stack([np.log1p(rlp), (rlp > args.max_text_chars).astype(np.float64)], 1)   # not a
            keepc = []                                           # behavioral concept
            for c in range(len(pool)):
                lc = max(abs(float(np.corrcoef(Ph[c], Lb[:, k])[0, 1])) for k in range(2)
                         if Lb[:, k].std() > 1e-9) if Ph[c].std() > 1e-6 else 0.0
                keepc.append(lc <= args.len_gate)
            if sum(keepc) >= len(atoms):                         # never gate below one name per atom
                pool = [c for c, k in zip(pool, keepc) if k]; Ph = Ph[np.array(keepc)]
        # SIGNED correlations (a name must describe what the atom fires ON -- an anti-correlated
        # candidate describes the complement; abs() produced sign-flipped names with negative paper
        # fidelity), assigned ONE-TO-ONE greedily (strongest atom picks first) so extensional
        # near-duplicate atoms get distinct names -- and the projection reset then pushes them apart.
        C = np.zeros((len(atoms), len(pool)))
        for ai, j in enumerate(atoms):
            for c in range(len(pool)):
                C[ai, c] = float(np.corrcoef(Pp[j], Ph[c])[0, 1]) if (Ph[c].std() > 1e-6
                                                                     and Pp[j].std() > 1e-6) else 0.0
        names = [""] * len(atoms); snaps = np.zeros(len(atoms)); used = set()
        for ai in np.argsort(-C.max(1)):
            pick = next((c for c in np.argsort(-C[ai]) if c not in used), int(np.argmax(C[ai])))
            used.add(pick); names[ai] = pool[pick]; snaps[ai] = C[ai, pick]
        return names, snaps, Pp, pool, Ph

    def _named_jval(nms):                                        # hard sentence-set J: fit FITe -> R^2 on VALe
        uq = list(dict.fromkeys(nms))
        return _jhard(_hardP(uq, "F"), _hardP(uq, "V"), Y[FITe], Y[VALe])

    # ---- cycle-consistency state: per-atom language anchors ----
    cyc_anchors = [None] * K; cyc_acorr = np.full(K, -1e9); cyc_anchor_P = None
    cyc_name_cache = {}                                          # name -> hard presence over ALL texts

    def _cyc_decode(Emat):
        """Patchscopes splice: one batched generation per atom -> candidate clauses."""
        pre_s = (judge.u_open + "The following tokens describe ONE specific, observable behavior "
                 "of an AI assistant's responses: \"")
        suf_s = ("\". Restate that behavior in plain English. Be specific and behavioral. Answer "
                 "with a single short clause completing \"the assistant tends to ...\" -- output "
                 "only the clause." + judge.a_open + "the assistant tends to")
        outs = {}
        with torch.no_grad():
            pe = judge.embed(tok(pre_s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev))[0]
            se = judge.embed(tok(suf_s, add_special_tokens=False, return_tensors="pt").input_ids.to(dev))[0]
            for j in range(K):
                ej = Emat[j].to(dev, dtype=pe.dtype)
                seq = torch.cat([pe, ej, se], 0)[None].expand(args.cyc_samples, -1, -1)
                am_ = torch.ones(seq.shape[:2], dtype=torch.long, device=dev)
                try:
                    out = judge.model.generate(inputs_embeds=seq, attention_mask=am_, do_sample=True,
                                               temperature=0.8, top_p=0.95, max_new_tokens=28,
                                               pad_token_id=tok.pad_token_id)
                except ValueError:
                    # multimodal wrappers (Gemma4Unified...) refuse generate(inputs_embeds);
                    # manual sampling loop -- uniform unpadded batch, so defaults suffice
                    o = judge.model(inputs_embeds=seq, attention_mask=am_, use_cache=True)
                    ids = []
                    for _t in range(28):
                        lg = o.logits[:, -1] / 0.8
                        pr = torch.softmax(lg.float(), -1)
                        sp, si = torch.sort(pr, descending=True)
                        keep = (sp.cumsum(-1) - sp) <= 0.95      # top-p
                        pr = torch.zeros_like(pr).scatter(1, si, sp * keep)
                        nxt = torch.multinomial(pr / pr.sum(-1, keepdim=True), 1)
                        ids.append(nxt)
                        am_ = torch.cat([am_, torch.ones_like(nxt)], 1)
                        o = judge.model(input_ids=nxt, attention_mask=am_,
                                        past_key_values=o.past_key_values, use_cache=True)
                    out = torch.cat(ids, 1)
                outs[j] = [c for c in dict.fromkeys(_phrase(tok.decode(g, skip_special_tokens=True))
                                                    for g in out) if c and len(c.split()) >= 3]
        return outs

    def _cnp(a, b):
        a = a - a.mean(); b = b - b.mean()
        dn = np.linalg.norm(a) * np.linalg.norm(b)
        return float((a * b).sum() / dn) if dn > 1e-9 else 0.0

    # ---- training: backprop through the frozen judge; nothing else moves ----
    best_j, best_E = -1e9, E.detach().clone()
    best_anchors = [None] * K                                    # cyc anchors AT the best-state step
    best_named, best_named_j = None, -1e9                        # best PROJECTED sentence set (hard val J)
    for step in range(args.steps):
        if args.lambda_cyc > 0 and step >= args.cyc_warmup and step % args.cyc_every == 0:
            with torch.no_grad():
                Pp_c = _softP(E.detach(), PROBE)                 # SELECTION split (which candidate)
                Pv_c = _softP(E.detach(), CYCV)                  # ACCEPTANCE split (disjoint): the
            cands = _cyc_decode(E.detach())                      # cycle loss optimizes toward the
            allc = list(dict.fromkeys(c for cs in cands.values() for c in cs))   # anchor, so accepting
            if allc:                                             # on the selection probe would feed
                Phc = _hardP(allc, "P")                          # selection optimism INTO training
                rowc = {c: Phc[i] for i, c in enumerate(allc)}
                changed = False
                for j in range(K):                               # re-score the INCUMBENT vs the
                    if cyc_anchors[j]:                           # CURRENT atom on the acceptance
                        cyc_acorr[j] = _cnp(                     # split (name presence cached over
                            cyc_name_cache[cyc_anchors[j]][CYCV], Pv_c[j])   # all texts)
                hC = len(CYCV) // 2                          # SPLIT-HALF acceptance: an accepted swap
                def _njval_cyc(nms):                         # must improve joint named R^2 on BOTH
                    uq = list(dict.fromkeys(nm for nm in nms if nm))   # CYCV halves (fit FITe). One
                    if not uq:                               # split let noise-fit swaps through -- the
                        return np.array([-1e9, -1e9])        # K=8 held-out drop (cyc .346 -> .306
                    Pf_, Pc_ = _hardP(uq, "F"), _hardP(uq, "C")        # while snap ROSE to .361).
                    return np.array([_jhard(Pf_, Pc_[:, :hC], Y[FITe], Y[CYCV[:hC]]),
                                     _jhard(Pf_, Pc_[:, hC:], Y[FITe], Y[CYCV[hC:]])])
                base_nj = _njval_cyc(cyc_anchors) if args.cyc_accept == "named" else None
                for j in range(K):
                    ranked = []                                  # rank by probe DISTINCT score
                    for cnd in cands.get(j, []):
                        own = _cnp(rowc[cnd], Pp_c[j])
                        oth = max((_cnp(rowc[cnd], Pp_c[k]) for k in range(K) if k != j), default=0.0)
                        ranked.append((own - args.cyc_gamma * max(oth, 0.0), cnd))
                    ranked.sort(reverse=True)
                    for _ds, bc in ranked[:2 if args.cyc_accept == "named" else 1]:
                        if bc == cyc_anchors[j]:
                            continue
                        cv = _cnp(_hardP([bc], "C")[0], Pv_c[j])
                        if args.cyc_accept == "corr":
                            if cv > cyc_acorr[j] + 1e-3:
                                cyc_anchors[j], cyc_acorr[j] = bc, cv; changed = True
                                print(f"[soft]   cyc anchor {j} <- \"{bc}\"", flush=True)
                                break
                        else:                                    # named: Delta joint-R^2, fidelity floor
                            # below the floor own-corr may only RISE (no decaying-incumbent walk-down);
                            # above it, trades down to the floor are allowed. Names must DESCRIBE.
                            if cv < min(max(cyc_acorr[j], 0.25), args.cyc_corr_floor):
                                continue
                            trial = list(cyc_anchors); trial[j] = bc
                            njv = _njval_cyc(trial)
                            if (njv > base_nj + 1e-3).all():
                                cyc_anchors[j], cyc_acorr[j] = bc, cv
                                base_nj = njv; changed = True
                                print(f"[soft]   cyc anchor {j} <- \"{bc}\"  "
                                      f"(joint R2 {base_nj[0]:.3f}/{base_nj[1]:.3f})", flush=True)
                                break
                if changed or cyc_anchor_P is None:
                    for nm in set(filter(None, cyc_anchors)) - set(cyc_name_cache):
                        cyc_name_cache[nm] = judge.presence_grid(
                            [nm], texts, text_batch=jb, grain="response",
                            mode="scale")[0].astype(np.float32)
                    cyc_anchor_P = np.stack([cyc_name_cache[nm] if nm else
                                             np.zeros(len(texts), np.float32) for nm in cyc_anchors])
                print(f"[soft]   cyc anchors (heldout corr): " +
                      " ".join(f"{v:+.2f}" if v > -1e8 else " -- " for v in cyc_acorr), flush=True)
        mb = TRAIN[rng.choice(len(TRAIN), args.batch_texts, replace=False)]
        tl, segs = _judge_texts(mb)
        if args.joint_scoring:
            P = judge.presence_grid_soft_joint(E, tl, text_batch=jb, temp=args.judge_temp,
                                               order=rng.permutation(K).tolist())
        else:
            P = judge.presence_grid_soft(E, tl, text_batch=jb, temp=args.judge_temp, mode="scale")
        if segs is not None:                                     # sentence units -> response (max is differentiable:
            P = torch.stack([(P[:, segs[r]:segs[r + 1]].max(1).values if args.sent_agg == "max"
                              else P[:, segs[r]:segs[r + 1]].mean(1)) for r in range(len(mb))], 1)
        Yb = torch.tensor(Y[mb], device=dev, dtype=torch.float32)
        if args.recon_mode == "dmtied":
            jcv = _jdm_t(P.float(), Yb, Rt[torch.as_tensor(mb, device=dev)], Wy_t, args.fit_ridge)
        else:
            jcv = _jcv_t(P.float(), Yb, args.fit_ridge)
        loss = -args.lambda_recon * jcv
        fid_t = None
        if args.lambda_fid > 0:
            fid_t = _fid_cv_t(P.float(), Rt[torch.as_tensor(mb, device=dev)], args.fit_ridge)
            loss = loss - args.lambda_fid * fid_t.mean()
        cg_t = None
        if args.lambda_causal > 0:
            cg_t, _ = _causal_gain(P.float(), mb, grad=True)
            loss = loss - args.lambda_causal * cg_t
        if args.lambda_support > 0:                              # anti-dead: zeroing all features is NOT an escape
            loss = loss + args.lambda_support * F.relu(args.support_tau - P.float().mean(1)).mean()
        if args.lambda_lang > 0:
            sln = F.normalize(E.reshape(-1, d), dim=1)
            loss = loss + args.lambda_lang * (1.0 - (sln.to(torch.bfloat16) @ En.T).float().max(1).values).mean()
        if args.lambda_div > 0:                                  # hinged redundancy: only EXCESS co-firing
            Pc_ = P.float() - P.float().mean(1, keepdim=True)
            Cd = (Pc_ @ Pc_.T) / (Pc_.norm(dim=1)[:, None] * Pc_.norm(dim=1)[None, :] + 1e-9)
            off = Cd - torch.diag_embed(Cd.diagonal())
            loss = loss + args.lambda_div * (F.relu(off.abs() - args.div_tau) ** 2).sum() / (K * (K - 1))
        ccy_val = None
        if args.lambda_cyc > 0 and cyc_anchor_P is not None:     # stay describable: match the anchor's
            Pa = torch.tensor(cyc_anchor_P[:, mb], device=dev, dtype=torch.float32)   # hard presence
            pz = P.float() - P.float().mean(1, keepdim=True)
            az = Pa - Pa.mean(1, keepdim=True)
            ccy = (pz * az).sum(1) / (pz.norm(dim=1) * az.norm(dim=1) + 1e-9)
            loss = loss + args.lambda_cyc * (1.0 - ccy).mean()
            ccy_val = float(ccy.detach().mean())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_([E], 1.0); opt.step()
        if step % 5 == 0:
            with torch.no_grad():
                sup = P.float().mean(1)
                Pc = P.float() - P.float().mean(1, keepdim=True)
                pn = Pc.norm(dim=1) + 1e-9
                Cm = (Pc @ Pc.T) / (pn[:, None] * pn[None, :]); Cm.fill_diagonal_(0)
            ftxt = f" fid={fid_t.mean().item():+.2f}" if fid_t is not None else ""
            ftxt += f" cg={cg_t.item():+.3f}" if cg_t is not None else ""
            ftxt += f" cyc={ccy_val:+.2f}" if ccy_val is not None else ""
            print(f"[soft] step {step:>4} Jcv={jcv.item():.3f}{ftxt} maxcorr={Cm.abs().max().item():.2f} "
                  f"support={sup.mean().item():.2f} minsup={sup.min().item():.2f}", flush=True)
        if step % args.eval_every == 0:
            Pf_now = _softP(E.detach(), FITe); Pv_now = _softP(E.detach(), VALe)
            jh = _jhard(Pf_now, Pv_now, Y[FITe], Y[VALe])
            fids, _ = _fidelity(Pf_now, FITe, Pv_now, VALe)      # v from FITe, spearman on VALe (disjoint)
            cval = 0.0
            if args.lambda_causal > 0:
                cval = float(_causal_gain(torch.tensor(Pf_now, dtype=torch.float32, device=dev),
                                          FITe, grad=False, nb=32)[0])
            cycval = 0.0
            if args.lambda_cyc > 0 and cyc_anchor_P is not None:   # best-state must respect the cycle
                Pv_soft = _softP(E.detach(), VALe)                 # objective or stage 2 reverts to
                cycval = float(np.mean([_cnp(Pv_soft[j], cyc_anchor_P[j, VALe])   # pre-cyc states
                                        for j in range(K) if cyc_anchors[j]] or [0.0]))
            sel = args.lambda_recon * jh + args.lambda_fid * fids.mean() + args.lambda_causal * cval \
                + args.lambda_cyc * cycval
            ctxt = f" cg_val={cval:+.3f}" if args.lambda_causal > 0 else ""
            print(f"[soft]   eval step {step}: J_val={jh:.3f} fid={fids.mean():+.2f} "
                  f"(min {fids.min():+.2f}){ctxt}{'  (new best)' if sel > best_j else ''}", flush=True)
            if args.lambda_cyc > 0 and any(cyc_anchors):         # anchors ARE the current names;
                hix = [j for j in range(K) if cyc_anchors[j]]    # split presences cached, ~free
                anms = [cyc_anchors[j] for j in hix]
                PfN, PvN = _hardP(anms, "F"), _hardP(anms, "V")
                nj = _jhard(PfN, PvN, Y[FITe], Y[VALe])
                nfids, _ = _fidelity(PfN, FITe, PvN, VALe)
                ncorr = np.array([_cnp(PvN[i], Pv_now[j]) for i, j in enumerate(hix)])
                print(f"[soft]   named({len(hix)}/{K}): R2={nj:.3f} fid={nfids.mean():+.2f} "
                      f"corr={ncorr.mean():+.2f} (min {ncorr.min():+.2f}) supp={PvN.mean():.2f}",
                      flush=True)
            if sel > best_j:
                best_j, best_E = sel, E.detach().clone()
                best_anchors = list(cyc_anchors)                 # names travel WITH the state they describe
            dead = np.where(Pf_now.mean(1) < 0.02)[0]            # SAE-style resampling: dead atoms have no
            if len(dead) and step < args.steps - args.eval_every:  # gradient (saturated scorer) -> re-seed them
                with torch.no_grad():
                    for j in dead:
                        ids = (tok(names0[int(rng.integers(len(names0)))], add_special_tokens=False).input_ids
                               [:args.n_tokens] if names0 else rng.choice(tok.vocab_size, args.n_tokens).tolist())
                        e = judge.embed(torch.tensor(ids, device=dev)).detach().float()
                        E.data[j, :len(e)] = e + 0.01 * torch.randn_like(e)
                        if len(e) < args.n_tokens:
                            E.data[j, len(e):] = E.data[j, len(e) - 1]
                opt.state.clear()                                # stale Adam moments would re-kill them
                print(f"[soft]   resampled dead atoms {dead.tolist()}", flush=True)
        if args.project_every > 0 and step > 0 and step % args.project_every == 0:
            # PROJECT onto the language manifold: snap -> reset slots to the sentences' embeddings.
            names_p, snap_p, _, _, _ = _snap_atoms(E.detach(), list(range(K)))
            jn = _named_jval(names_p)
            new = jn > best_named_j
            if new:
                best_named_j, best_named = jn, (list(names_p), snap_p.copy())
            print(f"[soft]   PROJECT step {step}: hard J_val={jn:.3f}{'  (new best set)' if new else ''}", flush=True)
            for nm, sc in zip(names_p, snap_p):
                print(f"[soft]     x{sc:+.2f} {nm}", flush=True)
            with torch.no_grad():
                for j, nm in enumerate(names_p):
                    ids = tok(nm, add_special_tokens=False).input_ids[:args.n_tokens] or [tok.eos_token_id]
                    e = judge.embed(torch.tensor(ids, device=dev)).detach().float()
                    E.data[j, :len(e)] = e
                    if len(e) < args.n_tokens:
                        E.data[j, len(e):] = E.data[j, len(e) - 1]
            opt.state.clear()
    final_anchors = list(cyc_anchors)                            # named acceptance is monotone on CYCV,
    E = best_E                                                   # so end-of-run anchors are the best set
    if args.eval_canonical:                                      # closures (_fidelity/_pca_ceiling/
        Res, s, Y = Res_c, s_c, Y_c                              # _named_jval) read these at call time
        Wy, w_axn = Wy_c, waxc_n
        print("[soft] FINAL EVAL on the CANONICAL target (training target was ablated)", flush=True)
    Pf_e, Pv_e = _softP(E, FITe), _softP(E, VALe)

    # ---- optional backward elimination to --keep atoms (val J decides) ----
    keep = list(range(K))
    if 0 < args.keep < K:
        while len(keep) > args.keep:
            drop, bv = None, -1e9
            for j in keep:
                cand = [i for i in keep if i != j]
                v = _jhard(Pf_e[cand], Pv_e[cand], Y[FITe], Y[VALe])
                if v > bv:
                    bv, drop = v, j
            print(f"[soft] eliminate atom {drop} (J_val without it: {bv:.3f})", flush=True)
            keep.remove(drop)
    Kk = len(keep)

    # ---- snap-to-text on the best soft state (extensional matching; see _snap_atoms) ----
    names, snap, Pp, npool, nPh = _snap_atoms(E, keep)

    # ---- final ladder on TEST: soft structure ceiling vs snapped (verbalized) taxonomy ----
    # R^2 is against the training target Y (recon); the axis AUC is always a READ-OFF via a
    # separate presence->s fit, never a training force (grounding as read-off, not objective).
    YFV, sFV = np.concatenate([Y[FITe], Y[VALe]]), np.concatenate([s[FITe], s[VALe]])

    def _ladder(Pf_, Pv_, Pt_):
        PfvT = np.concatenate([Pf_, Pv_], 1).T
        j = _r2_np(_ridge_np(PfvT, YFV), Pt_.T, Y[TE])
        auc = _auc(np.concatenate([Pt_.T, np.ones((len(TE), 1))], 1) @ _ridge_np(PfvT, sFV), y[TE])
        return j, auc

    Pt = _softP(E, TE)[keep]; Pf_k, Pv_k = Pf_e[keep], Pv_e[keep]
    j_soft, auc_soft = _ladder(Pf_k, Pv_k, Pt)
    if args.joint_scoring:
        tlT, segsT = _judge_texts(TE)
        Pa = judge.presence_grid_soft_joint(E, tlT, text_batch=jb, temp=args.judge_temp,
                                            order=list(range(K))).float().cpu().numpy()
        Pb_ = judge.presence_grid_soft_joint(E, tlT, text_batch=jb, temp=args.judge_temp,
                                             order=list(range(K))[::-1]).float().cpu().numpy()
        if segsT is not None:
            Pa, Pb_ = _agg_np(Pa, segsT), _agg_np(Pb_, segsT)
        oc = np.array([_spear(Pa[j], Pb_[j]) for j in keep if Pa[j].std() > 1e-6 and Pb_[j].std() > 1e-6])
        print(f"[soft] joint order-consistency (spearman across mirrored slot orders, TEST): "
              f"mean {oc.mean():+.2f} min {oc.min():+.2f}  [multi-attribute judge reliability]", flush=True)
        Pf_i, Pv_i, Pt_i = (_softP(E, FITe, joint=False)[keep], _softP(E, VALe, joint=False)[keep],
                            _softP(E, TE, joint=False)[keep])
        j_ind, auc_ind = _ladder(Pf_i, Pv_i, Pt_i)
        print(f"[soft]   SOFT (independent-scored, deployment condition): R^2={j_ind:.3f}  "
              f"AUC={auc_ind:.3f}", flush=True)
    uniq = list(dict.fromkeys(names))
    Phf, Phv, Pht = _hardP(uniq, "F"), _hardP(uniq, "V"), _hardP(uniq, "T")
    j_named, auc_named = _ladder(Phf, Phv, Pht)
    # ---- paper Fidelity on TEST: v_j from FITe+VALe soft presence (presence-weighted DiffMean);
    # fid_soft = spearman(<h,v_j>, soft presence)  [direction <-> soft feature];
    # fid_name = spearman(<h,v_j>, hard presence of the snapped NAME)  [the paper metric, verbatim]
    dir_idx = np.concatenate([FITe, VALe])
    fid_soft, dirs = _fidelity(np.concatenate([Pf_k, Pv_k], 1), dir_idx, Pt, TE)
    r_named = np.stack([Pht[uniq.index(names[i])] for i in range(Kk)])
    fid_name, _ = _fidelity(np.concatenate([Pf_k, Pv_k], 1), dir_idx, r_named, TE)
    pc_ceil = _pca_ceiling(Kk)
    print(f"\n[soft] FINAL test ladder:  axis AUC={_auc(s[TE], y[TE]):.3f}  "
          f"PCA-{Kk} ceiling R^2={pc_ceil:.3f} (max for any {Kk} linear features)", flush=True)
    print(f"[soft]   SOFT structure : R^2={j_soft:.3f}  AUC={auc_soft:.3f}  K={Kk} effK={_effk(Pt):.2f}  "
          f"Fidelity(soft)={fid_soft.mean():+.2f}  ({j_soft / max(pc_ceil, 1e-9):.0%} of PCA ceiling)", flush=True)
    # DM-TIED recon: the judgments->activation->prediction pipeline as a NESTED model. Each
    # atom's activation signature is FIXED to its presence-weighted DiffMean (the fidelity dirs,
    # FIT+VAL-estimated); only K scalars are fit: Yhat = c + sum_j gamma_j * P_j * v_jY. Ratio
    # to the free ridge = how much of SOFT R^2 the interpretable geometric story explains.
    if Y.shape[1] > 1:
        vY = (dirs @ Wy) * w_axn[None, :]                        # dirs [K, pc] -> Y coords [K, D]
        P_fv = np.concatenate([Pf_k, Pv_k], 1)                   # [K, Mfv]
        fv_idx = np.concatenate([FITe, VALe])
        muY, muP = Y[fv_idx].mean(0), P_fv.mean(1)
        A_ = np.stack([(P_fv[j] - muP[j])[:, None] * vY[j][None, :] for j in range(Kk)], -1)
        gam = np.linalg.lstsq(A_.reshape(-1, Kk), (Y[fv_idx] - muY).reshape(-1), rcond=None)[0]
        pred = muY + sum(gam[j] * (Pt[j] - muP[j])[:, None] * vY[j][None, :] for j in range(Kk))
        j_dm = float(1 - ((Y[TE] - pred) ** 2).sum() / (((Y[TE] - Y[TE].mean(0)) ** 2).sum() + 1e-9))
        print(f"[soft]   DM-TIED recon  : R^2={j_dm:.3f}  (K scalars on frozen DiffMean dirs; "
              f"{j_dm / max(j_soft, 1e-9):.0%} of the free ridge)", flush=True)
    print(f"[soft]   NAMED taxonomy : R^2={j_named:.3f}  AUC={auc_named:.3f}  K={len(uniq)} effK={_effk(Pht):.2f}  "
          f"Fidelity(named)={fid_name.mean():+.2f}  (soft-named gap = price of language)", flush=True)
    # SET-NAMED: choose the K names as a SET by forward selection on the probe split (the atom-
    # aligned snap names above stay for the per-atom fidelity table; this is the paper taxonomy).
    sel = _fwd_select(nPh, Y[PROBE], Kk)
    set_names = list(dict.fromkeys(npool[c] for c in sel))
    Pgf, Pgv, Pgt = _hardP(set_names, "F"), _hardP(set_names, "V"), _hardP(set_names, "T")
    j_set, auc_set = _ladder(Pgf, Pgv, Pgt)
    fid_set, _ = _fidelity(np.concatenate([Pgf, Pgv], 1), dir_idx, Pgt, TE)
    print(f"[soft]   SET-NAMED tax. : R^2={j_set:.3f}  AUC={auc_set:.3f}  K={len(set_names)} "
          f"effK={_effk(Pgt):.2f}  Fidelity={fid_set.mean():+.2f}  (set-level forward selection)", flush=True)
    for nm in set_names:
        print(f"        {nm}", flush=True)
    # CYC-NAMED: the anchors stage-2 actually optimized -- without this the ratcheted names are
    # discarded and the ladder re-snaps from scratch. Two sets: the snapshot at the best-state
    # step (paired with the reported E) and the END-of-run set, which is monotone-best on CYCV
    # under named acceptance (an anchor set's named R^2 does not depend on E, so it stays valid).
    cyc_out, cyc_sets = {}, []
    if args.lambda_cyc > 0:
        cyc_sets = [("", "best-state anchors", best_anchors)]
        if final_anchors != best_anchors:
            cyc_sets.append(("_final", "final anchors, monotone-best on CYCV", final_anchors))
        cyc_sets = [(sfx, tag, a) for sfx, tag, a in cyc_sets if any(a)]
    for sfx, tag, anch in cyc_sets:
        cnames = [anch[j] or names[i] for i, j in enumerate(keep)]
        uqc = list(dict.fromkeys(cnames))
        Pcf, Pcv, Pct = _hardP(uqc, "F"), _hardP(uqc, "V"), _hardP(uqc, "T")
        j_cyc, auc_cyc = _ladder(Pcf, Pcv, Pct)
        r_cyc = np.stack([Pct[uqc.index(cnames[i])] for i in range(Kk)])
        fid_cyc, _ = _fidelity(np.concatenate([Pf_k, Pv_k], 1), dir_idx, r_cyc, TE)
        ccorr = np.array([_cnp(r_cyc[i], Pt[i]) for i in range(Kk)])
        print(f"[soft]   CYC-NAMED tax. : R^2={j_cyc:.3f}  AUC={auc_cyc:.3f}  K={len(uqc)} "
              f"effK={_effk(Pct):.2f}  Fidelity={fid_cyc.mean():+.2f}  corr(soft)={ccorr.mean():+.2f} "
              f"(min {ccorr.min():+.2f})  ({tag})", flush=True)
        for i in range(Kk):
            print(f"        c{ccorr[i]:+.2f} / F{fid_cyc[i]:+.2f}  {cnames[i]}", flush=True)
        cyc_out.update({f"cyc_anchors{sfx}": cnames, f"j_cyc_named{sfx}": j_cyc,
                        f"auc_cyc_named{sfx}": auc_cyc, f"fidelity_cyc_named{sfx}": fid_cyc,
                        f"cyc_corr_test{sfx}": ccorr})
    proj = {}
    if args.project_every > 0:
        jn_final = _named_jval(names)                            # the final snap set competes too
        if jn_final > best_named_j:
            best_named_j, best_named = jn_final, (list(names), snap.copy())
    if best_named is not None:
        pn, psnap = best_named; uqp = list(dict.fromkeys(pn))
        Pnf, Pnv, Pnt = _hardP(uqp, "F"), _hardP(uqp, "V"), _hardP(uqp, "T")
        j_proj, auc_proj = _ladder(Pnf, Pnv, Pnt)
        fid_proj, _ = _fidelity(np.concatenate([Pnf, Pnv], 1), dir_idx, Pnt, TE)
        proj = {"projected_names": pn, "projected_snap": psnap, "j_projected": j_proj,
                "auc_projected": auc_proj, "fidelity_projected": fid_proj}
        print(f"[soft]   PROJECTED tax.: R^2={j_proj:.3f}  AUC={auc_proj:.3f}  K={len(uqp)} effK={_effk(Pnt):.2f}  "
              f"Fidelity={fid_proj.mean():+.2f}  (best sentence set across projections, val-selected)", flush=True)
        for nm in uqp:
            print(f"        {nm}", flush=True)
    print("[soft] atoms (snap / fid_soft / fid_name / support : name)  [fid_name = paper Fidelity: "
          "spearman(<h,v_j>, judged presence of the name) on TEST]:", flush=True)
    for i, j in enumerate(keep):
        dup = "" if names[i] in uniq[:i + 1] and names.index(names[i]) == i else "  [duplicate name]"
        print(f"    x{snap[i]:+.2f} / f{fid_soft[i]:+.2f} / F{fid_name[i]:+.2f} / {Pp[j].mean():.2f}  "
              f"{names[i]}{dup}", flush=True)

    # ---- FINAL causal metrics ----
    causal_out = {}
    if args.lambda_causal > 0:
        # held-out value of the training objective: teacher-forced contrastive gain on TEST rows
        # (class-coordinate targets stay train-estimated; only the scored responses are held out)
        pos_te = np.array([i for i in TE if y[i] == 1][:96]); neg_te = np.array([i for i in TE if y[i] == 0][:96])
        for c0 in range(0, len(np.concatenate([pos_te, neg_te])), 16):
            rows_ = np.concatenate([pos_te, neg_te])[c0:c0 + 16]
            for i, v in zip(rows_, _clogp(rows_).cpu().tolist()):
                base_lp[int(i)] = v
        Pfk_t = torch.tensor(Pf_k, dtype=torch.float32, device=dev)
        if args.causal_mode == "matrix":
            sM, gM = _causal_gain(Pfk_t, FITe, grad=False)
            print(f"[soft] CAUSAL matrix (teacher-forced, FIT rows, nats/token): "
                  f"S={float(sM):+.3f}  diag={float(gM['pP']):+.3f}  |cross|={float(gM['pN']):+.3f}",
                  flush=True)
            causal_out.update({"S_matrix": float(sM), "matrix_diag": float(gM["pP"]),
                               "matrix_cross": float(gM["pN"])})
        m = min(len(pos_te), len(neg_te)); Ss, comps = [], []
        for c0 in (range(0, m, 16) if args.causal_mode != "matrix" else []):
            sT, g = _causal_gain(Pfk_t, FITe, grad=False,
                                 rows=(pos_te[c0:c0 + 16], neg_te[c0:c0 + 16]))
            Ss.append(float(sT)); comps.append({k: float(v) for k, v in g.items()})
        cm = {k: float(np.mean([c[k] for c in comps])) for k in comps[0]} if comps else {}
        Ss = Ss or [0.0]
        neg = (f" | push-: D+ {cm['nP']:+.3f} D- {cm['nN']:+.3f}" if "nP" in cm else "")
        cm = cm or {"pP": 0.0, "pN": 0.0}
        print(f"[soft] CAUSAL gain (teacher-forced, TEST, nats/token): S={np.mean(Ss):+.3f}  "
              f"push+: D+ {cm['pP']:+.3f} D- {cm['pN']:+.3f}{neg}", flush=True)
        causal_out.update({"S_test": float(np.mean(Ss)), **{f"gain_{k}": v for k, v in cm.items()}})
    if args.interv_prompts > 0:
        # generation-based read-off (CNLD protocol): steer each atom's fidelity direction, judge
        # all atom names on steered vs base generations
        if args.lambda_causal <= 0:                              # no hook registered yet
            layer = int(cfg["model"].get("layer", 20))
            cstate = {"d": None, "Q": None, "z": None}

            def _chook(_m, _inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if cstate["d"] is None:
                    return None
                o = o + cstate["d"].to(o.dtype)
                return (o,) + out[1:] if isinstance(out, tuple) else o
            judge.layers[layer].register_forward_hook(_chook)
        Hd = (Wr @ dirs.T).T
        Hd = Hd / (np.linalg.norm(Hd, axis=1, keepdims=True) + 1e-9)
        spread = np.array([np.std((Hr - mu_r) @ Hd[j]) for j in range(Kk)])
        pidx = TE[rng.permutation(len(TE))[:args.interv_prompts]]
        chats = [tok.apply_chat_template([{"role": "user", "content": prm[i][:args.max_prompt_chars]}],
                                         tokenize=False, add_generation_prompt=True) for i in pidx]

        def _fmt_o(outs):
            return [f"User asked: {prm[i][:args.max_prompt_chars]}\n\nResponse: {o[:args.max_text_chars]}"
                    for i, o in zip(pidx, outs)]
        cstate["d"] = None
        base_out = _gen(chats, chunk=8)
        base_len = np.mean([len(o) for o in base_out]) + 1e-9
        Pb_c = judge.presence_grid(names, _fmt_o(base_out), text_batch=32, mode="scale")
        dY = np.zeros((Kk, Kk, len(pidx))); len_ratio = np.ones(Kk)
        for jf in range(Kk):
            cstate["d"] = torch.tensor(args.interv_alpha * spread[jf] * Hd[jf], dtype=torch.float32, device=dev)
            so = _gen(chats, chunk=8)
            cstate["d"] = None                                   # judge scoring runs UNsteered
            len_ratio[jf] = np.mean([len(o) for o in so]) / base_len
            dY[jf] = judge.presence_grid(names, _fmt_o(so), text_batch=32, mode="scale") - Pb_c
        aie = np.abs(dY.mean(2))
        sel_c = np.array([aie[j, j] / max(np.delete(aie[j], j).max(), 1e-6) for j in range(Kk)])
        print(f"[soft] CAUSAL read-off ({len(pidx)} prompts, alpha={args.interv_alpha} x spread)"
              f"  [AIE / selectivity / len-ratio]:", flush=True)
        for jf in range(Kk):
            print(f"    AIE={aie[jf, jf]:.3f}  sel={sel_c[jf]:.2f}  len={len_ratio[jf]:.2f}  {names[jf]}",
                  flush=True)
        causal_out.update({"aie": aie, "selectivity_causal": sel_c, "len_ratio": len_ratio, "dY": dY})
    if args.axbench_factors:
        # ---- AxBench-style steering eval: factor sweep, 3-subscore harmonic mean, split selection ----
        factors = [float(x) for x in args.axbench_factors.split(",")]
        if args.interv_prompts <= 0:                             # machinery not built yet
            if args.lambda_causal <= 0:
                layer = int(cfg["model"].get("layer", 20))
                cstate = {"d": None, "Q": None, "z": None}

                def _chook(_m, _inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    if cstate["d"] is None:
                        return None
                    o = o + cstate["d"].to(o.dtype)
                    return (o,) + out[1:] if isinstance(out, tuple) else o
                judge.layers[layer].register_forward_hook(_chook)
            Hd = (Wr @ dirs.T).T
            Hd = Hd / (np.linalg.norm(Hd, axis=1, keepdims=True) + 1e-9)
            spread = np.array([np.std((Hr - mu_r) @ Hd[j]) for j in range(Kk)])
        aidx = TE[rng.permutation(len(TE))[args.interv_prompts:args.interv_prompts + args.axbench_prompts]]
        achats = [tok.apply_chat_template([{"role": "user", "content": prm[i][:args.max_prompt_chars]}],
                                          tokenize=False, add_generation_prompt=True) for i in aidx]
        gm_save, gm = args.gen_max_new, args.axbench_max_new

        def _rate(texts_r, question):                            # 0-9 digit rating -> [0,1], batched
            self_t = judge
            prompts = [f"{judge.u_open}{question}\n\n{t[:900]}\n\nRate 0-9. Reply with ONLY one "
                       f"digit.{judge.a_open}" for t in texts_r]
            outv = np.zeros(len(prompts), np.float32)
            for c0 in range(0, len(prompts), 32):
                grp = prompts[c0:c0 + 32]
                enc = tok(grp, return_tensors="pt", padding=True, truncation=True, max_length=1024,
                          add_special_tokens=False).to(dev)
                with torch.no_grad():
                    lg = self_t.model(**enc).logits
                last = enc["attention_mask"].sum(1) - 1
                lgl = lg[torch.arange(len(grp), device=dev), last]
                outv[c0:c0 + 32] = self_t._score_scale(lgl).float().cpu().numpy()
            return outv

        nsel = len(aidx) // 2
        ax_rows = []
        args.gen_max_new = gm
        for jf in range(Kk):
            per_factor = []                                      # (hm per prompt [M], subscores)
            for f in factors:
                cstate["d"] = torch.tensor(f * spread[jf] * Hd[jf], dtype=torch.float32, device=dev)
                so = _gen(achats, chunk=8)
                cstate["d"] = None
                pair = [f"Instruction: {prm[i][:args.max_prompt_chars]}\n\nResponse: {o}"
                        for i, o in zip(aidx, so)]
                c_sc = _rate(pair, f"How strongly does the Response display this behavior: "
                                   f"\"the assistant tends to {names[jf]}\"?")
                i_sc = _rate(pair, "How well does the Response address the Instruction?")
                f_sc = _rate([f"Response: {o}" for o in so], "How fluent and coherent is the Response?")
                hm = 3.0 / (1.0 / np.maximum(c_sc, 1e-3) + 1.0 / np.maximum(i_sc, 1e-3)
                            + 1.0 / np.maximum(f_sc, 1e-3))
                per_factor.append((hm, c_sc, i_sc, f_sc))
            best_f = int(np.argmax([pf[0][:nsel].mean() for pf in per_factor]))
            hm, c_sc, i_sc, f_sc = per_factor[best_f]
            ax_rows.append((factors[best_f], hm[nsel:].mean(), c_sc[nsel:].mean(),
                            i_sc[nsel:].mean(), f_sc[nsel:].mean()))
        args.gen_max_new = gm_save
        print(f"[soft] AXBENCH-style steering eval ({len(aidx)} prompts, {len(factors)} factors, "
              f"factor selected on {nsel}, scored on {len(aidx) - nsel}; scores 0-1)"
              f"  [factor / overall(hm) / concept / instruct / fluency]:", flush=True)
        for jf, (bf, o, c, i_, fl) in enumerate(ax_rows):
            print(f"    f={bf:<4} hm={o:.2f}  c={c:.2f} i={i_:.2f} fl={fl:.2f}  {names[jf]}", flush=True)
        print(f"[soft]   mean steering score (hm) = {np.mean([r[1] for r in ax_rows]):.3f}", flush=True)
        causal_out.update({"axbench": np.array(ax_rows, np.float32), "axbench_factors": np.array(factors)})
    ej_out = {}
    if args.eval_judge:
        # HELD-OUT JUDGE: the taxonomy was learned AND selected under the training judge, so its
        # fidelity/recon numbers can reflect judge-gaming. Re-score the named taxonomy end-to-end
        # under an independent judge (response grain). Free the training judge first (one GPU).
        # Wrapped: a judge failure (gated repo, incompatible transformers, OOM) must NEVER lose
        # a finished run -- save the checkpoint without the held-out rows and re-score later.
        try:
            del judge.model, judge
            torch.cuda.empty_cache()
            ej = FrozenJudge(args.eval_judge, device=dev, dtype=dt,
                             max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)

            def _ejp(cs, idx):
                return ej.presence_grid(cs, [texts[i] for i in idx], text_batch=jb,
                                        grain="response", mode="scale").astype(np.float32)
            Ef, Ev, Et = _ejp(uniq, FITe), _ejp(uniq, VALe), _ejp(uniq, TE)
            j_ej, auc_ej = _ladder(Ef, Ev, Et)
            Pdir_ej = np.stack([np.concatenate([Ef[uniq.index(names[i])], Ev[uniq.index(names[i])]])
                                for i in range(Kk)])
            r_ej = np.stack([Et[uniq.index(names[i])] for i in range(Kk)])
            fid_ej, _ = _fidelity(Pdir_ej, dir_idx, r_ej, TE)
            print(f"[soft]   HELD-OUT JUDGE ({args.eval_judge}): R^2={j_ej:.3f}  AUC={auc_ej:.3f}  "
                  f"Fidelity(named)={fid_ej.mean():+.2f}  [independent of the training judge]", flush=True)
            Egf, Egv, Egt = _ejp(set_names, FITe), _ejp(set_names, VALe), _ejp(set_names, TE)
            j_ejs, auc_ejs = _ladder(Egf, Egv, Egt)
            print(f"[soft]   HELD-OUT JUDGE (set-named): R^2={j_ejs:.3f}  AUC={auc_ejs:.3f}", flush=True)
            ej_out = {"eval_judge": args.eval_judge, "j_eval_judge": j_ej, "auc_eval_judge": auc_ej,
                      "fidelity_eval_judge": fid_ej, "j_eval_judge_set": j_ejs, "auc_eval_judge_set": auc_ejs}
            for sfx, tag, anch in cyc_sets:
                uqc = list(dict.fromkeys(cyc_out[f"cyc_anchors{sfx}"]))
                Cf, Cv, Ct = _ejp(uqc, FITe), _ejp(uqc, VALe), _ejp(uqc, TE)
                j_ejc, auc_ejc = _ladder(Cf, Cv, Ct)
                print(f"[soft]   HELD-OUT JUDGE (cyc-named, {tag}): R^2={j_ejc:.3f}  AUC={auc_ejc:.3f}",
                      flush=True)
                ej_out.update({f"j_eval_judge_cyc{sfx}": j_ejc, f"auc_eval_judge_cyc{sfx}": auc_ejc})
        except Exception as e:  # noqa: BLE001
            print(f"[soft]   HELD-OUT JUDGE FAILED ({args.eval_judge}): {e} -- saving without it; "
                  f"re-score later via a names-only rerun or heldout_ladder", flush=True)
            ej_out = {"eval_judge_error": str(e)}
    # PROVENANCE: the invocation this checkpoint came from, so a cell can be
    # checked against the flags it was BUILT with (--max-prompt-chars above all)
    # instead of inferred from mtimes and the battery script's git history.
    _prov = {"args": vars(args), "argv": " ".join(_sys.argv)}
    torch.save({**_prov, "soft_prompts": E[keep].cpu(), "names": names, "snap_corr": snap,
                "set_names": set_names, "j_set": j_set, "auc_set": auc_set, "fidelity_set": fid_set,
                "fidelity_soft": fid_soft, "fidelity_named": fid_name, "dirs_pc": dirs,
                "j_soft": j_soft, "auc_soft": auc_soft, "j_named": j_named, "auc_named": auc_named,
                "kept": keep, "n_tokens": args.n_tokens, "target": args.target,
                "pca_ceiling": pc_ceil, **cyc_out, **proj, **causal_out, **ej_out}, args.out)


if __name__ == "__main__":
    main()
