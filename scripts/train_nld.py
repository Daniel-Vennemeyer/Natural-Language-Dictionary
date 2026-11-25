"""Train a natural language dictionary against a frozen judge.

Each of the K entries is a soft prompt that the frozen judge reads as a concept; its concept
score on a response is the judge's graded presence rating. Stage 1 trains the soft prompts on
the explanation objective J (cross-fitted R^2 of the search target Y from the concept scores)
plus the fidelity objective F_k, the support, language and diversity regularizers. Stage 2
resumes from Stage 1 and adds cycle-consistency: each entry is decoded into candidate
descriptions, a description is accepted as the entry's anchor when it raises the named R^2, and
the soft concept score is pulled toward the anchor's judged score. The final block reports the
dictionary on the test split and saves the descriptions and soft prompts.

Example:
    python scripts/train_nld.py --config CONFIG.yaml --units UNITS.npz --prompt-acts PROMPTS.npz \\
        --seed 0 --max-prompt-chars 400 --acts-key sent_acts --num-concepts 8 \\
        --target acts_wc_axisw --lambda-fid 1 --lambda-div 2 --lambda-lang 0.4 --fit-ridge 0.2 \\
        --batch-texts 256 --probe-n 512 --judge-temp 1.5 --gen-temp 0.8 --steps 400 --out stage1.pt
    python scripts/train_nld.py <same flags> --steps 100 --resume-from stage1.pt \\
        --lambda-cyc 0.9 --cyc-warmup 0 --cyc-every 1 --out dictionary.pt
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from nld.utils.frozen_presence import FrozenJudge
from nld.utils.runlock import acquire, gpu_guard


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


def _spear(a, b):                                              # Spearman rank correlation (Fidelity)
    def rk(x):
        o = np.argsort(x); r = np.empty(len(x)); r[o] = np.arange(len(x))
        return r
    ra, rb = rk(a) - (len(a) - 1) / 2, rk(b) - (len(b) - 1) / 2
    return float((ra * rb).sum() / (np.sqrt((ra ** 2).sum() * (rb ** 2).sum()) + 1e-9))


def _cnp(a, b):                                                # Pearson correlation, 0 if constant
    a = a - a.mean(); b = b - b.mean()
    dn = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / dn) if dn > 1e-9 else 0.0


def _phrase(t):
    """First line of a generation, trimmed to a short clause."""
    t = (t or "").split("\n")[0].strip().strip('"').strip()
    for cut in ("<|", "GROUP", "Group", "Note:", "note:"):
        if cut in t:
            t = t.split(cut)[0].strip()
    t = t.rstrip(".;,: ").lstrip("-* ")
    return " ".join(t.split()[:24])


def _fwd_select(Pc, Yp, K, min_gain=1e-3):
    """Choose descriptions as a set: forward-select candidates whose joint concept scores best
    reconstruct the target on the probe split (symmetric split-half cross-fit). Selecting the
    set, rather than the best match per entry, accounts for what the other descriptions cover."""
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
    """Explanation objective J: differentiable, symmetric cross-fitted R^2 of the search target
    Y [M, D] from the concept scores P [K, M]. The Frobenius form weights each target dimension
    by its variance."""
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


def _fid_cv_t(P, R, ridge):
    """Fidelity objective F_k: differentiable, cross-fitted, one value per entry -> [K].

    For entry k: residualize its concept score on the other entries (fit on one half) to get its
    unique component u_k, ridge-fit an activation direction v_k to u_k on that half, and score
    corr(R @ v_k, u_k) on the other half; average both directions. A duplicate entry has no
    unique component, so its fidelity is near 0. The correlation is multiplied by the uniqueness
    weight (unique share of the score's spread) so a near-duplicate cannot gain fidelity from
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True)
    ap.add_argument("--acts-key", choices=["sent_acts"], default="sent_acts",
                    help="response activations: sentence activations mean-pooled per response")
    ap.add_argument("--target", choices=["acts_wc_axisw"], default="acts_wc_axisw",
                    help="search target Y: within-positive-class PCs of the de-confounded activation "
                         "residual, each weighted by |corr| with the behavior axis")
    ap.add_argument("--acts-dims", type=int, default=32, help="number of target PCs")
    ap.add_argument("--num-concepts", type=int, default=6, help="dictionary size K")
    ap.add_argument("--n-tokens", type=int, default=6, help="soft tokens per entry")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.003,
                    help="Adam moves each coordinate ~lr per step and token embeddings are ~0.01-0.03 "
                         "in scale, so a much larger lr leaves the token manifold in one step")
    ap.add_argument("--batch-texts", type=int, default=128,
                    help="texts per gradient step (split in half for the cross-fit)")
    ap.add_argument("--fit-ridge", type=float, default=0.1,
                    help="ridge multiplier (x trace/d) inside the differentiable cross-fits; a small "
                         "ridge overfits the half-batches and makes J negative at initialization")
    ap.add_argument("--prefix-cache-gb", type=float, default=0.0,
                    help="host memory for encoded text prefixes reused across training steps (about 100 GB "
                         "holds a full sycophancy discovery split for Qwen3-4B); 0 re-encodes every step")
    ap.add_argument("--judge-temp", type=float, default=1.5,
                    help="judge scoring temperature; >1 softens saturated digits, where the concept "
                         "score gradient vanishes")
    ap.add_argument("--lambda-fid", type=float, default=0.0, help="weight of the fidelity objective F_k")
    ap.add_argument("--lambda-support", type=float, default=1.0,
                    help="support regularizer: hinge relu(tau - mean concept score) per entry, so "
                         "switching every entry off is not a way to avoid a negative J")
    ap.add_argument("--support-tau", type=float, default=0.05)
    ap.add_argument("--lambda-lang", type=float, default=0.1,
                    help="language regularizer: pull each soft token toward its nearest token embedding")
    ap.add_argument("--lambda-div", type=float, default=0.0,
                    help="diversity penalty: mean relu(|corr(P_j, P_k)| - div-tau)^2 over entry pairs; "
                         "co-occurrence below tau is free")
    ap.add_argument("--div-tau", type=float, default=0.2)
    ap.add_argument("--lambda-cyc", type=float, default=0.0,
                    help="cycle-consistency weight (Stage 2): reward corr(soft concept score, the "
                         "anchor description's judged score)")
    ap.add_argument("--cyc-every", type=int, default=25, help="steps between anchor refreshes")
    ap.add_argument("--cyc-warmup", type=int, default=100, help="first step with an anchor refresh")
    ap.add_argument("--cyc-samples", type=int, default=8, help="decoded candidates per entry per refresh")
    ap.add_argument("--cyc-gamma", type=float, default=0.5,
                    help="candidate ranking: own corr minus gamma x max corr with the other entries")
    ap.add_argument("--cyc-corr-floor", type=float, default=0.6,
                    help="a candidate's own corr must be >= min(max(incumbent, 0.25), floor)")
    ap.add_argument("--resume-from", default=None,
                    help="Stage 2: initialize the soft prompts from a Stage 1 checkpoint")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--fit-eval-n", type=int, default=512); ap.add_argument("--val-eval-n", type=int, default=256)
    ap.add_argument("--probe-n", type=int, default=512, help="probe texts for matching descriptions")
    ap.add_argument("--name-proposals", type=int, default=6, help="contrastive description proposals per entry")
    ap.add_argument("--contrast-n", type=int, default=6); ap.add_argument("--example-chars", type=int, default=500)
    ap.add_argument("--spec-min", type=float, default=0.55, help="specificity gate on candidate descriptions")
    ap.add_argument("--gen-temp", type=float, default=0.9); ap.add_argument("--gen-max-new", type=int, default=32)
    ap.add_argument("--len-gate", type=float, default=0.45,
                    help="drop candidate descriptions whose judged score correlates with response "
                         "length or truncation above this on the probe split (0 = off)")
    ap.add_argument("--pc-dim", type=int, default=256); ap.add_argument("--shrink", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--max-text-chars", type=int, default=600); ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acquire("soft_" + args.out.replace("/", "_"))                # refuse a second run with the same --out
    gpu_guard(18.0)
    dev = torch.device("cuda"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    jb = max(8, (48 * 600) // args.max_text_chars)               # judge text batch; long texts use smaller chunks
    print(f"[soft] judge text_batch = {jb}", flush=True)
    cfg = yaml.safe_load(open(args.config)); mname = cfg["model"]["name"]; dt = cfg["model"].get("dtype", "bfloat16")

    # ---- join response activations, prompt activations, labels and texts ----
    u = np.load(args.units, allow_pickle=True)
    # one row per sentence (sent_resp = parent response text); mean-pool per response
    Rmap = {}
    for h, p in zip(np.asarray(u["sent_acts"], np.float64), [str(x) for x in u["sent_resp"]]):
        Rmap.setdefault(" ".join(p.split()), []).append(h)
    Rmap = {k: np.mean(v, 0) for k, v in Rmap.items()}
    print(f"[soft] sent_acts: pooled {len(u['sent_acts'])} sentence acts -> {len(Rmap)} responses", flush=True)
    pa = np.load(args.prompt_acts, allow_pickle=True); Hp_all = np.asarray(pa["prompt_acts"], np.float64)
    rtxt = [str(x) for x in pa["response_text"]]; ptxt = [str(x) for x in pa["prompt_text"]]; lab = np.asarray(pa["label"], np.int64)
    Hr, Hp, y, resp, prm = [], [], [], [], []
    for hp, rt, pt, l in zip(Hp_all, rtxt, ptxt, lab):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); Hp.append(hp); y.append(int(l)); resp.append(rt); prm.append(pt)
    Hr, Hp, y = np.array(Hr), np.array(Hp), np.array(y); n = len(y)
    perm = rng.permutation(n); te = np.zeros(n, bool); te[perm[:max(1, int(n * args.heldout_frac))]] = True; tr = ~te

    # ---- activation residual: response PCs minus what the prompt predicts, minus length ----
    mu_r = Hr[tr].mean(0)
    Zr = _pcs((Hr - mu_r)[tr], Hr - mu_r, args.pc_dim)
    Zp = _pcs((Hp - Hp[tr].mean(0))[tr], Hp - Hp[tr].mean(0), args.pc_dim)
    A = Zp[tr]; lam = args.ridge * np.trace(A.T @ A) / A.shape[1]
    Wm = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zr[tr])
    Res = Zr - Zp @ Wm
    # remove log-length, length and truncation effects (fit on the discovery split)
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

    # ---- behavior axis s (shrunk LDA) and search target Y ----
    p, q = Res[tr][y[tr] == 1], Res[tr][y[tr] == 0]
    wl = np.linalg.solve(_shrinkc(np.cov(p, rowvar=False) + np.cov(q, rowvar=False), args.shrink), p.mean(0) - q.mean(0))
    s = Res @ wl; s = (s - s[tr].mean()) / (s[tr].std() + 1e-9)
    if np.corrcoef(s[tr], y[tr])[0, 1] < 0:
        s = -s
    # within-class PCs: basis fit on positive-class discovery residuals, projected onto everyone
    sy = np.where(tr & (y == 1))[0]; mu1 = Res[sy].mean(0)
    Y = _pcs(Res[sy] - mu1, Res - mu1, args.acts_dims)
    print(f"[soft] {args.target} target: within-class PCs (fit on {len(sy)} pos-train), dims={Y.shape[1]}", flush=True)
    # weight each PC by its relevance to the behavior axis (weights fit on the discovery split)
    w_ax = np.abs(np.array([float(np.corrcoef(Y[tr][:, d], s[tr])[0, 1]) for d in range(Y.shape[1])]))
    w_axn = w_ax / (w_ax.max() + 1e-9)
    Y = Y * w_axn[None, :]
    print(f"[soft] axisw: PCs weighted by |corr(PC, axis)| -- mean weight "
          f"{(w_ax / (w_ax.max() + 1e-9)).mean():.2f}, top dim corr {w_ax.max():.2f}", flush=True)

    texts = [f"User asked: {prm[i][:args.max_prompt_chars]}\n\nResponse: {resp[i][:args.max_text_chars]}" for i in range(n)]

    # ---- splits: the discovery split is divided into VAL / FIT / TRAIN; PROBE and CYCV come from TRAIN ----
    pv = rng.permutation(np.where(tr)[0])
    VALe = pv[:min(args.val_eval_n, len(pv) // 4)]
    FITe = pv[len(VALe):len(VALe) + min(args.fit_eval_n, len(pv) - len(VALe))]
    TRAIN = pv[len(VALe):]                                       # gradient batches sample from here (excl. VALe)
    TE = np.where(te)[0]                                         # test split
    PROBE = TRAIN[rng.permutation(len(TRAIN))[:min(args.probe_n, len(TRAIN))]]
    CYCV = np.setdiff1d(TRAIN, PROBE)[:256]                      # anchor acceptance split, disjoint from PROBE
    print(f"[soft] n={n} train={len(TRAIN)} val={len(VALe)} test={len(TE)}  axis AUC(s,label)={_auc(s[TE], y[TE]):.3f}",
          flush=True)

    def _pca_ceiling(k_):
        """Best test R^2 any K linear features can reach: projection onto Y's own top-K PCs."""
        fit_ix = np.concatenate([FITe, VALe])
        mu_y = Y[fit_ix].mean(0)
        _, _, Vt = np.linalg.svd(Y[fit_ix] - mu_y, full_matrices=False)
        W = Vt[:k_]; Yt = Y[TE]
        pred = mu_y + (Yt - mu_y) @ W.T @ W
        return float(1 - ((Yt - pred) ** 2).sum() / (((Yt - Yt.mean(0)) ** 2).sum() + 1e-9))

    judge = FrozenJudge(mname, device=dev, dtype=dt, max_text_chars=args.max_text_chars + args.max_prompt_chars + 40,
                        prefix_cache_gb=args.prefix_cache_gb)
    tok = judge.tok; d = judge.embed.weight.shape[1]; K = args.num_concepts

    # ---- trainable parameters: K soft prompts, initialized from random vocabulary tokens ----
    E = torch.zeros(K, args.n_tokens, d)
    with torch.no_grad():
        # drawn in Stage 2 too (then overwritten), so both stages consume the same random draws
        for j in range(K):
            ids = rng.choice(tok.vocab_size, args.n_tokens).tolist()
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
    with torch.no_grad():                                        # token-embedding manifold for the language regularizer
        En = F.normalize(judge.embed.weight.float(), dim=1).to(torch.bfloat16)
    print(f"[soft] trainable params: {K}x{args.n_tokens}x{d} soft tokens = {E.numel()} (nothing else)", flush=True)

    Rt = torch.tensor(Res, dtype=torch.float32, device=dev)     # activation residual for F_k

    @torch.no_grad()
    def _softP(Emat, idx):                                       # soft concept scores, no gradient
        return judge.presence_grid_soft(Emat, [texts[i] for i in idx], text_batch=jb,
                                        temp=args.judge_temp, mode="scale").float().cpu().numpy()

    def _fidelity(P_dir, dir_idx, r_eval, eval_idx):
        """Fidelity per entry: v_k = concept-score-weighted DiffMean over Res[dir_idx],
        a_k = <normalized h, v_k> on the disjoint eval split, Fidelity_k = spearman(a_k, r_eval_k),
        where r_eval is the concept score on the eval split (soft during training; the
        description's judged score at the end). Returns (fidelities [K'], directions [K', pc])."""
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

    def _jhard(Pf, Pv, yf, yv):                                  # fit on FITe -> R^2 on VALe
        return _r2_np(_ridge_np(Pf.T, yf), Pv.T, yv)

    # ---- describing entries: generation, greedy slot decoding, matching ----
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
    def _decode_slots(Emat, max_new=24):                         # soft prompt -> phrase, greedy
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

    hardc = {"P": {}, "F": {}, "V": {}, "T": {}, "C": {}}        # judged score of each (description, split), cached
    _sidx = {"P": PROBE, "F": FITe, "V": VALe, "T": TE, "C": CYCV}

    def _hardP(cs, split):
        need = [c for c in dict.fromkeys(cs) if c not in hardc[split]]
        if need:
            Ph = judge.presence_grid(need, [texts[i] for i in _sidx[split]], text_batch=jb,
                                     grain="response", mode="scale")
            for c, row in zip(need, Ph):
                hardc[split][c] = row.astype(np.float32)
        return np.stack([hardc[split][c] for c in cs])

    def _snap_atoms(Emat):
        """Describe each entry by matching: the candidate description whose judged score best
        correlates with the entry's soft concept score on the probe split. Candidates are greedy
        slot decodes plus contrastive proposals per entry. Returns (descriptions, match corrs,
        soft probe scores [K, probe], candidate pool, pool judged scores)."""
        Pp = _softP(Emat, PROBE)
        pool = list(dict.fromkeys(_decode_slots(Emat)))
        prompts = []
        for j in range(K):
            order = np.argsort(-Pp[j]); pl = max(4 * args.contrast_n, 24)
            for _m in range(args.name_proposals):
                Aix = rng.choice(order[:pl], args.contrast_n, replace=False)
                Bix = rng.choice(order[-pl:], args.contrast_n, replace=False)
                # a random window rather than the head, so the proposer also sees the middle and
                # end of long responses
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
        if args.len_gate > 0:                                    # a length or truncation detector is
            rlp = np.array([len(resp[i]) for i in PROBE], np.float64)   # not a behavioral concept
            Lb = np.stack([np.log1p(rlp), (rlp > args.max_text_chars).astype(np.float64)], 1)
            keepc = []
            for c in range(len(pool)):
                lc = max(abs(float(np.corrcoef(Ph[c], Lb[:, k])[0, 1])) for k in range(2)
                         if Lb[:, k].std() > 1e-9) if Ph[c].std() > 1e-6 else 0.0
                keepc.append(lc <= args.len_gate)
            if sum(keepc) >= K:                                  # keep at least one candidate per entry
                pool = [c for c, k in zip(pool, keepc) if k]; Ph = Ph[np.array(keepc)]
        # Signed correlations: a description must match what the entry fires on, not its
        # complement. Assignment is one-to-one and greedy (strongest entry first), so
        # near-duplicate entries receive distinct descriptions.
        C = np.zeros((K, len(pool)))
        for j in range(K):
            for c in range(len(pool)):
                C[j, c] = float(np.corrcoef(Pp[j], Ph[c])[0, 1]) if (Ph[c].std() > 1e-6
                                                                    and Pp[j].std() > 1e-6) else 0.0
        names = [""] * K; snaps = np.zeros(K); used = set()
        for j in np.argsort(-C.max(1)):
            pick = next((c for c in np.argsort(-C[j]) if c not in used), int(np.argmax(C[j])))
            used.add(pick); names[j] = pool[pick]; snaps[j] = C[j, pick]
        return names, snaps, Pp, pool, Ph

    # ---- cycle-consistency state: one anchor description per entry ----
    cyc_anchors = [None] * K; cyc_acorr = np.full(K, -1e9); cyc_anchor_P = None
    cyc_name_cache = {}                                          # description -> judged score over all texts

    def _cyc_decode(Emat):
        """Splice each soft prompt into a restatement prompt and sample candidate descriptions."""
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
                    # some multimodal model wrappers reject generate(inputs_embeds=...): sample
                    # manually (uniform unpadded batch, so the defaults suffice)
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

    # ---- training: backpropagate through the frozen judge into the soft prompts ----
    best_j, best_E = -1e9, E.detach().clone()
    best_anchors = [None] * K                                    # anchors at the best-state step
    for step in range(args.steps):
        if args.lambda_cyc > 0 and step >= args.cyc_warmup and step % args.cyc_every == 0:
            # Candidates are ranked on PROBE and accepted on the disjoint CYCV split: the cycle
            # loss trains toward the accepted anchor, so accepting on the ranking split would
            # feed selection optimism into training.
            with torch.no_grad():
                Pp_c = _softP(E.detach(), PROBE)
                Pv_c = _softP(E.detach(), CYCV)
            cands = _cyc_decode(E.detach())
            allc = list(dict.fromkeys(c for cs in cands.values() for c in cs))
            if allc:
                Phc = _hardP(allc, "P")
                rowc = {c: Phc[i] for i, c in enumerate(allc)}
                changed = False
                for j in range(K):                               # re-score incumbents against the current entry
                    if cyc_anchors[j]:
                        cyc_acorr[j] = _cnp(cyc_name_cache[cyc_anchors[j]][CYCV], Pv_c[j])
                hC = len(CYCV) // 2

                def _njval_cyc(nms):
                    # named R^2 (fit FITe) on each CYCV half; a swap must improve both halves
                    uq = list(dict.fromkeys(nm for nm in nms if nm))
                    if not uq:
                        return np.array([-1e9, -1e9])
                    Pf_, Pc_ = _hardP(uq, "F"), _hardP(uq, "C")
                    return np.array([_jhard(Pf_, Pc_[:, :hC], Y[FITe], Y[CYCV[:hC]]),
                                     _jhard(Pf_, Pc_[:, hC:], Y[FITe], Y[CYCV[hC:]])])
                base_nj = _njval_cyc(cyc_anchors)
                for j in range(K):
                    ranked = []                                  # own corr minus overlap with other entries
                    for cnd in cands.get(j, []):
                        own = _cnp(rowc[cnd], Pp_c[j])
                        oth = max((_cnp(rowc[cnd], Pp_c[k]) for k in range(K) if k != j), default=0.0)
                        ranked.append((own - args.cyc_gamma * max(oth, 0.0), cnd))
                    ranked.sort(reverse=True)
                    for _ds, bc in ranked[:2]:
                        if bc == cyc_anchors[j]:
                            continue
                        cv = _cnp(_hardP([bc], "C")[0], Pv_c[j])
                        # below the floor own-corr may only rise; above it, trades down to the
                        # floor are allowed, so a description keeps describing its entry
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
                print("[soft]   cyc anchors (heldout corr): " +
                      " ".join(f"{v:+.2f}" if v > -1e8 else " -- " for v in cyc_acorr), flush=True)
        mb = TRAIN[rng.choice(len(TRAIN), args.batch_texts, replace=False)]
        P = judge.presence_grid_soft(E, [texts[i] for i in mb], text_batch=jb, temp=args.judge_temp, mode="scale")
        Yb = torch.tensor(Y[mb], device=dev, dtype=torch.float32)
        jcv = _jcv_t(P.float(), Yb, args.fit_ridge)
        loss = -jcv
        fid_t = None
        if args.lambda_fid > 0:
            fid_t = _fid_cv_t(P.float(), Rt[torch.as_tensor(mb, device=dev)], args.fit_ridge)
            loss = loss - args.lambda_fid * fid_t.mean()
        if args.lambda_support > 0:
            loss = loss + args.lambda_support * F.relu(args.support_tau - P.float().mean(1)).mean()
        if args.lambda_lang > 0:
            sln = F.normalize(E.reshape(-1, d), dim=1)
            loss = loss + args.lambda_lang * (1.0 - (sln.to(torch.bfloat16) @ En.T).float().max(1).values).mean()
        if args.lambda_div > 0:
            Pc_ = P.float() - P.float().mean(1, keepdim=True)
            Cd = (Pc_ @ Pc_.T) / (Pc_.norm(dim=1)[:, None] * Pc_.norm(dim=1)[None, :] + 1e-9)
            off = Cd - torch.diag_embed(Cd.diagonal())
            loss = loss + args.lambda_div * (F.relu(off.abs() - args.div_tau) ** 2).sum() / (K * (K - 1))
        ccy_val = None
        if args.lambda_cyc > 0 and cyc_anchor_P is not None:     # cycle-consistency with the anchors
            Pa = torch.tensor(cyc_anchor_P[:, mb], device=dev, dtype=torch.float32)
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
            ftxt += f" cyc={ccy_val:+.2f}" if ccy_val is not None else ""
            print(f"[soft] step {step:>4} Jcv={jcv.item():.3f}{ftxt} maxcorr={Cm.abs().max().item():.2f} "
                  f"support={sup.mean().item():.2f} minsup={sup.min().item():.2f}", flush=True)
        if step % args.eval_every == 0:
            Pf_now = _softP(E.detach(), FITe); Pv_now = _softP(E.detach(), VALe)
            jh = _jhard(Pf_now, Pv_now, Y[FITe], Y[VALe])
            fids, _ = _fidelity(Pf_now, FITe, Pv_now, VALe)      # direction from FITe, spearman on VALe
            cycval = 0.0
            if args.lambda_cyc > 0 and cyc_anchor_P is not None:   # the best state must also respect cycle-consistency
                cycval = float(np.mean([_cnp(Pv_now[j], cyc_anchor_P[j, VALe])
                                        for j in range(K) if cyc_anchors[j]] or [0.0]))
            sel = jh + args.lambda_fid * fids.mean() + args.lambda_cyc * cycval
            print(f"[soft]   eval step {step}: J_val={jh:.3f} fid={fids.mean():+.2f} "
                  f"(min {fids.min():+.2f}){'  (new best)' if sel > best_j else ''}", flush=True)
            if args.lambda_cyc > 0 and any(cyc_anchors):         # anchors are the current descriptions
                hix = [j for j in range(K) if cyc_anchors[j]]
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
                best_anchors = list(cyc_anchors)                 # anchors are saved with the state they describe
            # Dead entries get no gradient (saturated judge), so re-initialize them from random
            # tokens and clear the Adam state, whose stale moments would switch them off again.
            dead = np.where(Pf_now.mean(1) < 0.02)[0]
            if len(dead) and step < args.steps - args.eval_every:
                with torch.no_grad():
                    for j in dead:
                        ids = rng.choice(tok.vocab_size, args.n_tokens).tolist()
                        e = judge.embed(torch.tensor(ids, device=dev)).detach().float()
                        E.data[j, :len(e)] = e + 0.01 * torch.randn_like(e)
                        if len(e) < args.n_tokens:
                            E.data[j, len(e):] = E.data[j, len(e) - 1]
                opt.state.clear()
                print(f"[soft]   resampled dead atoms {dead.tolist()}", flush=True)
    final_anchors = list(cyc_anchors)
    E = best_E
    Pf_k, Pv_k = _softP(E, FITe), _softP(E, VALe)

    # ---- describe the best soft state ----
    names, snap, Pp, npool, nPh = _snap_atoms(E)

    # ---- final evaluation on the test split ----
    # R^2 is against the search target Y; the AUC is a read-off through a separate
    # concept-score -> s fit, never a training term.
    YFV, sFV = np.concatenate([Y[FITe], Y[VALe]]), np.concatenate([s[FITe], s[VALe]])

    def _ladder(Pf_, Pv_, Pt_):
        PfvT = np.concatenate([Pf_, Pv_], 1).T
        j = _r2_np(_ridge_np(PfvT, YFV), Pt_.T, Y[TE])
        auc = _auc(np.concatenate([Pt_.T, np.ones((len(TE), 1))], 1) @ _ridge_np(PfvT, sFV), y[TE])
        return j, auc

    Pt = _softP(E, TE)
    j_soft, auc_soft = _ladder(Pf_k, Pv_k, Pt)
    uniq = list(dict.fromkeys(names))
    Phf, Phv, Pht = _hardP(uniq, "F"), _hardP(uniq, "V"), _hardP(uniq, "T")
    j_named, auc_named = _ladder(Phf, Phv, Pht)
    # Fidelity on the test split: v_k from the FITe+VALe soft concept scores.
    # fid_soft: against the soft concept score; fid_name: against the description's judged score.
    dir_idx = np.concatenate([FITe, VALe])
    fid_soft, dirs = _fidelity(np.concatenate([Pf_k, Pv_k], 1), dir_idx, Pt, TE)
    r_named = np.stack([Pht[uniq.index(names[i])] for i in range(K)])
    fid_name, _ = _fidelity(np.concatenate([Pf_k, Pv_k], 1), dir_idx, r_named, TE)
    pc_ceil = _pca_ceiling(K)
    print(f"\n[soft] FINAL test ladder:  axis AUC={_auc(s[TE], y[TE]):.3f}  "
          f"PCA-{K} ceiling R^2={pc_ceil:.3f} (max for any {K} linear features)", flush=True)
    print(f"[soft]   SOFT structure : R^2={j_soft:.3f}  AUC={auc_soft:.3f}  K={K} effK={_effk(Pt):.2f}  "
          f"Fidelity(soft)={fid_soft.mean():+.2f}  ({j_soft / max(pc_ceil, 1e-9):.0%} of PCA ceiling)", flush=True)
    print(f"[soft]   NAMED taxonomy : R^2={j_named:.3f}  AUC={auc_named:.3f}  K={len(uniq)} effK={_effk(Pht):.2f}  "
          f"Fidelity(named)={fid_name.mean():+.2f}  (soft-named gap = price of language)", flush=True)
    # descriptions chosen as a set by forward selection on the probe split
    sel = _fwd_select(nPh, Y[PROBE], K)
    set_names = list(dict.fromkeys(npool[c] for c in sel))
    Pgf, Pgv, Pgt = _hardP(set_names, "F"), _hardP(set_names, "V"), _hardP(set_names, "T")
    j_set, auc_set = _ladder(Pgf, Pgv, Pgt)
    fid_set, _ = _fidelity(np.concatenate([Pgf, Pgv], 1), dir_idx, Pgt, TE)
    print(f"[soft]   SET-NAMED tax. : R^2={j_set:.3f}  AUC={auc_set:.3f}  K={len(set_names)} "
          f"effK={_effk(Pgt):.2f}  Fidelity={fid_set.mean():+.2f}  (set-level forward selection)", flush=True)
    for nm in set_names:
        print(f"        {nm}", flush=True)
    # Stage 2 descriptions: the anchors at the best-state step (paired with the saved soft
    # prompts) and, if different, the end-of-run anchors (acceptance only ever raises the named
    # R^2 on CYCV, and that R^2 does not depend on the soft prompts).
    cyc_out, cyc_sets = {}, []
    if args.lambda_cyc > 0:
        cyc_sets = [("", "best-state anchors", best_anchors)]
        if final_anchors != best_anchors:
            cyc_sets.append(("_final", "final anchors, monotone-best on CYCV", final_anchors))
        cyc_sets = [(sfx, tag, a) for sfx, tag, a in cyc_sets if any(a)]
    for sfx, tag, anch in cyc_sets:
        cnames = [anch[j] or names[j] for j in range(K)]         # entries without an anchor keep their matched description
        uqc = list(dict.fromkeys(cnames))
        Pcf, Pcv, Pct = _hardP(uqc, "F"), _hardP(uqc, "V"), _hardP(uqc, "T")
        j_cyc, auc_cyc = _ladder(Pcf, Pcv, Pct)
        r_cyc = np.stack([Pct[uqc.index(cnames[i])] for i in range(K)])
        fid_cyc, _ = _fidelity(np.concatenate([Pf_k, Pv_k], 1), dir_idx, r_cyc, TE)
        ccorr = np.array([_cnp(r_cyc[i], Pt[i]) for i in range(K)])
        print(f"[soft]   CYC-NAMED tax. : R^2={j_cyc:.3f}  AUC={auc_cyc:.3f}  K={len(uqc)} "
              f"effK={_effk(Pct):.2f}  Fidelity={fid_cyc.mean():+.2f}  corr(soft)={ccorr.mean():+.2f} "
              f"(min {ccorr.min():+.2f})  ({tag})", flush=True)
        for i in range(K):
            print(f"        c{ccorr[i]:+.2f} / F{fid_cyc[i]:+.2f}  {cnames[i]}", flush=True)
        cyc_out.update({f"cyc_anchors{sfx}": cnames, f"j_cyc_named{sfx}": j_cyc,
                        f"auc_cyc_named{sfx}": auc_cyc, f"fidelity_cyc_named{sfx}": fid_cyc,
                        f"cyc_corr_test{sfx}": ccorr})
    print("[soft] atoms (snap / fid_soft / fid_name / support : name)  [fid_name = paper Fidelity: "
          "spearman(<h,v_j>, judged presence of the name) on TEST]:", flush=True)
    for i in range(K):
        dup = "" if names[i] in uniq[:i + 1] and names.index(names[i]) == i else "  [duplicate name]"
        print(f"    x{snap[i]:+.2f} / f{fid_soft[i]:+.2f} / F{fid_name[i]:+.2f} / {Pp[i].mean():.2f}  "
              f"{names[i]}{dup}", flush=True)

    torch.save({"args": vars(args), "argv": " ".join(sys.argv),
                "soft_prompts": E.cpu(), "names": names, "snap_corr": snap,
                "set_names": set_names, "j_set": j_set, "auc_set": auc_set, "fidelity_set": fid_set,
                "fidelity_soft": fid_soft, "fidelity_named": fid_name, "dirs_pc": dirs,
                "j_soft": j_soft, "auc_soft": auc_soft, "j_named": j_named, "auc_named": auc_named,
                "kept": list(range(K)), "n_tokens": args.n_tokens, "target": args.target,
                "pca_ceiling": pc_ceil, **cyc_out}, args.out)


if __name__ == "__main__":
    main()
