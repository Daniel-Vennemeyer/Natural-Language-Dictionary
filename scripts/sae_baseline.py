"""SAE BASELINE: a TopK sparse autoencoder's features as the taxonomy, same eval protocol.

The standard mech-interp comparison (AxBench): do SAE features -- unsupervised, no judge in
the loop -- match the learned taxonomy on recon/fidelity, and do their decoder directions
steer selectively? No pretrained residual SAE exists for Qwen3-4B-Instruct-2507 (Qwen-Scope
starts at 8B-Base), so by default this trains a TopK SAE on the SAME L20 sentence meanpool
states the taxonomy methods use (train-split responses only) -- if anything a FAVORABLE
handicap for the SAE (in-domain data, matched grain). --sae-from loads external weights
(W_enc [D,d], b_enc [D], W_dec [d,D], mu [d]) if a pretrained one appears.

Pipeline: train/load SAE -> response-level features (mean over the response's sentence
feature activations) -> forward-select K features on FIT->VAL against the SAME target Y ->
  SAE-STRUCT  : R^2/AUC/effK of the K raw features on TEST + Fidelity(feat)  [paper protocol]
  SAE-NAMED   : name each feature (contrastive GROUP A/B captioning + probe-corr argmax --
                the snap protocol), judge the names, R^2/AUC/effK + Fidelity(named)
Saves {names, dirs_causal: decoder rows} -- causal_diag then steers the RAW SAE directions
(--dirs causal, sweep) and/or distills from the names, so selectivity rows are comparable.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/sae_baseline.py \
    --config configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml \
    --units $U/units_l20.npz --acts-key sent_acts --prompt-acts $U/prompt_last_l20.npz \
    --num-concepts 4 --out outputs/runs/soft_wc/sae_base_k4.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
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


def _ridge(P, y, lam_mult=1e-2):
    X = np.concatenate([P, np.ones((len(P), 1))], 1)
    lam = lam_mult * np.trace(X.T @ X) / X.shape[1]; L = lam * np.eye(X.shape[1]); L[-1, -1] = 0
    return np.linalg.solve(X.T @ X + L, X.T @ y)


def _r2(w, P, y):
    pred = np.concatenate([P, np.ones((len(P), 1))], 1) @ w
    return float(1 - ((y - pred) ** 2).sum() / (((y - y.mean(0)) ** 2).sum() + 1e-9))


def _spear(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64); rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.linalg.norm(ra) * np.linalg.norm(rb)
    return float((ra * rb).sum() / d) if d > 1e-9 else 0.0


def _effk(P):
    act = P[P.std(1) > 1e-6]
    if len(act) == 0:
        return 0.0
    ev = np.linalg.eigvalsh(np.atleast_2d(np.corrcoef(act)))
    return float((ev.sum() ** 2) / ((ev ** 2).sum() + 1e-9))


def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float((a * b).sum() / d) if d > 1e-9 else 0.0


def _phrase(t, cap=28):
    t = (t or "").strip().strip('"').strip().split("\n")[0]
    for cut in ("<|", "GROUP", "Group", "Note:", "note:"):
        if cut in t:
            t = t.split(cut)[0].strip()
    return " ".join(t.rstrip(".;,: ").lstrip("-* ").split()[:cap])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True); ap.add_argument("--acts-key", default="sent_acts")
    ap.add_argument("--target", choices=["acts_wc", "acts_wc_axisw"], default="acts_wc_axisw")
    ap.add_argument("--acts-dims", type=int, default=32)
    ap.add_argument("--num-concepts", type=int, default=4)
    ap.add_argument("--dict", type=int, default=4096); ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--sae-epochs", type=int, default=200); ap.add_argument("--sae-lr", type=float, default=1e-3)
    ap.add_argument("--sae-batch", type=int, default=1024)
    ap.add_argument("--sae-from", default=None,
                    help="load SAE weights (.pt with W_enc/b_enc/W_dec/mu) instead of training -- "
                         "the hook for a pretrained SAE, should one appear for this model+layer")
    ap.add_argument("--support-min", type=float, default=0.01,
                    help="drop features active on < this fraction of train responses")
    ap.add_argument("--names-from", default=None,
                    help=".pt with 'names': skip captioning and evaluate THESE names against the "
                         "selected features (atom-aligned fidelity: v from feature acts, spearman "
                         "vs name presence) -- e.g. fuzzing explanations of the same features")
    ap.add_argument("--name-samples", type=int, default=8, help="caption candidates per feature")
    ap.add_argument("--contrast-n", type=int, default=6); ap.add_argument("--example-chars", type=int, default=500)
    ap.add_argument("--probe-n", type=int, default=256, help="probe texts for name-candidate scoring")
    ap.add_argument("--gen-temp", type=float, default=0.9); ap.add_argument("--gen-max-new", type=int, default=32)
    ap.add_argument("--fit-eval-n", type=int, default=512); ap.add_argument("--val-eval-n", type=int, default=256)
    ap.add_argument("--pc-dim", type=int, default=256); ap.add_argument("--shrink", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--no-len-deconf", action="store_true")
    ap.add_argument("--max-text-chars", type=int, default=600); ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--struct-only", action="store_true",
                    help="stop after SAE-STRUCT. The whole structural ladder -- R^2/AUC/effK and "
                         "Fidelity(feat) -- is judge-free, so a CONFIGURATION sweep never needs "
                         "to load the judge or caption anything. Naming would also confound the "
                         "SAE config with captioner noise, which is not what such a sweep is "
                         "asking about.")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acquire("saeb_" + args.out.replace("/", "_"))
    gpu_guard(12.0)
    dev = torch.device("cuda"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config)); mname = cfg["model"]["name"]; dt = cfg["model"].get("dtype", "bfloat16")

    # ---- data join + de-confound + splits: byte-compatible with soft_concepts (same seed) ----
    u = np.load(args.units, allow_pickle=True)
    sent_H = np.asarray(u["sent_acts"], np.float64)
    sent_R = [" ".join(str(x).split()) for x in u["sent_resp"]]
    Rmap = {}
    for h, p in zip(sent_H, sent_R):
        Rmap.setdefault(p, []).append(h)
    Rmap = {k: np.mean(v, 0) for k, v in Rmap.items()}
    pa = np.load(args.prompt_acts, allow_pickle=True); Hp_all = np.asarray(pa["prompt_acts"], np.float64)
    rtxt = [str(x) for x in pa["response_text"]]; ptxt = [str(x) for x in pa["prompt_text"]]
    lab = np.asarray(pa["label"], np.int64)
    Hr, Hp, y, resp, prm, rkeys = [], [], [], [], [], []
    for hp, rt, pt, l in zip(Hp_all, rtxt, ptxt, lab):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); Hp.append(hp); y.append(int(l)); resp.append(rt); prm.append(pt)
            rkeys.append(k)
    Hr, Hp, y = np.array(Hr), np.array(Hp), np.array(y); n = len(y)
    perm = rng.permutation(n); te = np.zeros(n, bool); te[perm[:max(1, int(n * args.heldout_frac))]] = True
    tr = ~te
    Zr = _pcs((Hr - Hr[tr].mean(0))[tr], Hr - Hr[tr].mean(0), args.pc_dim)
    Zp = _pcs((Hp - Hp[tr].mean(0))[tr], Hp - Hp[tr].mean(0), args.pc_dim)
    A = Zp[tr]; lam = args.ridge * np.trace(A.T @ A) / A.shape[1]
    Wm = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zr[tr]); Res = Zr - Zp @ Wm
    if not args.no_len_deconf:
        rl = np.array([len(r) for r in resp], np.float64)
        L = np.stack([np.log1p(rl), rl, (rl > args.max_text_chars).astype(np.float64)], 1)
        L = (L - L[tr].mean(0)) / (L[tr].std(0) + 1e-9)
        L1 = np.concatenate([L, np.ones((len(L), 1))], 1)
        B = np.linalg.lstsq(L1[tr], Res[tr], rcond=None)[0]
        Res = Res - L1 @ B
    p_, q_ = Res[tr][y[tr] == 1], Res[tr][y[tr] == 0]
    wl = np.linalg.solve(_shrinkc(np.cov(p_, rowvar=False) + np.cov(q_, rowvar=False), args.shrink),
                         p_.mean(0) - q_.mean(0))
    s = Res @ wl; s = (s - s[tr].mean()) / (s[tr].std() + 1e-9)
    if np.corrcoef(s[tr], y[tr])[0, 1] < 0:
        s = -s
    sy = np.where(tr & (y == 1))[0]; mu1 = Res[sy].mean(0)
    Y = _pcs(Res[sy] - mu1, Res - mu1, args.acts_dims)
    if args.target == "acts_wc_axisw":
        w_ax = np.abs(np.array([float(np.corrcoef(Y[tr][:, d_], s[tr])[0, 1]) for d_ in range(Y.shape[1])]))
        Y = Y * (w_ax / (w_ax.max() + 1e-9))[None, :]
    texts = [f"User asked: {prm[i][:args.max_prompt_chars]}\n\nResponse: {resp[i][:args.max_text_chars]}"
             for i in range(n)]
    pv = rng.permutation(np.where(tr)[0])
    VALe = pv[:min(args.val_eval_n, len(pv) // 4)]
    FITe = pv[len(VALe):len(VALe) + min(args.fit_eval_n, len(pv) - len(VALe))]
    TRAIN = pv[len(VALe):]
    TE = np.where(te)[0]
    print(f"[sae] n={n} fit={len(FITe)} val={len(VALe)} test={len(TE)}  axis AUC={_auc(s[TE], y[TE]):.3f}",
          flush=True)

    # ---- TopK SAE on sentence states of TRAIN responses only (test never seen) ----
    tr_keys = {rkeys[i] for i in np.where(tr)[0]}
    X_tr = np.stack([h for h, p in zip(sent_H, sent_R) if p in tr_keys])
    d = X_tr.shape[1]; D = args.dict
    mu = X_tr.mean(0); sd_g = float(X_tr.std())                  # global scale keeps geometry
    Xt = torch.tensor((X_tr - mu) / sd_g, dtype=torch.float32, device=dev)
    _FVU = float("nan")      # stays NaN on the --sae-from path, which trains nothing
    if args.sae_from:
        sw = torch.load(args.sae_from, map_location="cpu", weights_only=False)
        W_enc = torch.tensor(np.asarray(sw["W_enc"]), dtype=torch.float32, device=dev)
        b_enc = torch.tensor(np.asarray(sw["b_enc"]), dtype=torch.float32, device=dev)
        W_dec = torch.tensor(np.asarray(sw["W_dec"]), dtype=torch.float32, device=dev)
        mu = np.asarray(sw.get("mu", mu)); sd_g = float(sw.get("scale", sd_g)); D = W_enc.shape[0]
        print(f"[sae] loaded external SAE from {args.sae_from} (D={D})", flush=True)
    else:
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        W_dec = torch.randn(d, D, generator=g).to(dev); W_dec /= W_dec.norm(dim=0, keepdim=True)
        W_enc = W_dec.T.clone().contiguous(); b_enc = torch.zeros(D, device=dev)
        W_enc.requires_grad_(True); b_enc.requires_grad_(True); W_dec.requires_grad_(True)
        opt = torch.optim.Adam([W_enc, b_enc, W_dec], lr=args.sae_lr)
        for ep in range(args.sae_epochs):
            ix = torch.randperm(len(Xt), device=dev)
            tot = 0.0
            for c0 in range(0, len(Xt), args.sae_batch):
                xb = Xt[ix[c0:c0 + args.sae_batch]]
                z = torch.relu(xb @ W_enc.T + b_enc)
                tv = torch.topk(z, args.topk, dim=1)
                zs = torch.zeros_like(z).scatter_(1, tv.indices, tv.values)
                loss = ((xb - zs @ W_dec.T) ** 2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    W_dec /= W_dec.norm(dim=0, keepdim=True) + 1e-9
                tot += float(loss.detach()) * len(xb)
            _FVU = tot / len(Xt) / float(Xt.var())
            if ep % 25 == 0 or ep == args.sae_epochs - 1:
                print(f"[sae] epoch {ep:>3}  FVU={_FVU:.3f}", flush=True)
        W_enc = W_enc.detach(); b_enc = b_enc.detach(); W_dec = W_dec.detach()

    # ---- response-level features: mean over the response's sentence feature activations ----
    @torch.no_grad()
    def _feats(H):
        Xa = torch.tensor((H - mu) / sd_g, dtype=torch.float32, device=dev)
        out = []
        for c0 in range(0, len(Xa), 4096):
            z = torch.relu(Xa[c0:c0 + 4096] @ W_enc.T + b_enc)
            tv = torch.topk(z, args.topk, dim=1)
            out.append(torch.zeros_like(z).scatter_(1, tv.indices, tv.values).cpu().numpy())
        return np.concatenate(out)
    sent_F = _feats(sent_H)
    Fmap = {}
    for f, p in zip(sent_F, sent_R):
        Fmap.setdefault(p, []).append(f)
    F = np.stack([np.mean(Fmap[k], 0) for k in rkeys])           # [n, D] response features

    # ---- forward-select K features on FIT->VAL against Y (identical selection protocol) ----
    supp = (F[np.where(tr)[0]] > 1e-6).mean(0)
    alive = np.where(supp >= args.support_min)[0]
    print(f"[sae] {len(alive)}/{D} features above support {args.support_min}", flush=True)
    # Support scales with topk/dict, so a FIXED threshold empties the pool for sparse configs:
    # at topk=8 of dict=4096 almost no feature fires in 5% of responses. That is a real property
    # of the configuration -- it cannot supply K adequately-supported features -- so report it
    # as an outcome with the numbers behind it and exit cleanly, rather than dying in max() on
    # an empty sequence and taking the rest of a sweep down with it.
    if len(alive) < args.num_concepts:
        q = np.quantile(supp, [0.5, 0.9, 0.99, 1.0])
        print(f"[sae] SAE-UNUSABLE: only {len(alive)} features reach support "
              f"{args.support_min} but K={args.num_concepts} are needed. "
              f"support median={q[0]:.4f} p90={q[1]:.4f} p99={q[2]:.4f} max={q[3]:.4f}; "
              f"n_train={int(tr.sum())} (support {args.support_min} = "
              f"{int(args.support_min * tr.sum())} responses)", flush=True)
        torch.save({"args": vars(args), "argv": " ".join(_sys.argv), "unusable": True,
                    "n_alive": int(len(alive)), "support": supp, "fvu": _FVU}, args.out)
        print(f"[sae] wrote {args.out}  (unusable configuration)", flush=True)
        return
    Ff, Fv, Ft = F[FITe], F[VALe], F[TE]
    YFV = np.concatenate([Y[FITe], Y[VALe]]); sFV = np.concatenate([s[FITe], s[VALe]])
    # SPLIT-HALF selection: with thousands of candidate features and a 256-row val, plain
    # argmax-on-val harvests selection optimism (first run: val 0.175 -> test 0.064). Score
    # each candidate by its WORSE half of VALe -- only replicating gains survive.
    hv = len(VALe) // 2
    sel = []
    for k in range(args.num_concepts):
        gains = []
        for c in alive:
            if c in sel:
                continue
            w = _ridge(Ff[:, sel + [c]], Y[FITe])
            gains.append((min(_r2(w, Fv[:hv, sel + [c]], Y[VALe[:hv]]),
                              _r2(w, Fv[hv:, sel + [c]], Y[VALe[hv:]])), int(c)))
        r2v_k, cbest = max(gains)
        sel.append(cbest)
        print(f"[sae] k={k + 1}: feature {cbest} (val worse-half R^2={r2v_k:.3f}, "
              f"support {supp[cbest]:.2f})", flush=True)
    Kk = len(sel)
    # SAE-FULL: the whole alive dictionary as a pool (lambda picked on VAL). High full + low K
    # = the axis is DISTRIBUTED across many features (superposition shattered by the sparsity
    # prior, no small readable subset); low full = the dictionary lost it outright.
    bestl = max((_r2(_ridge(Ff[:, alive], Y[FITe], lm), Fv[:, alive], Y[VALe]), lm)
                for lm in (1e-3, 1e-2, 0.1, 0.3, 1.0))
    lmf = bestl[1]
    wf = _ridge(np.concatenate([Ff[:, alive], Fv[:, alive]], 0), YFV, lmf)
    j_full = _r2(wf, Ft[:, alive], Y[TE])
    auc_full = _auc(np.concatenate([Ft[:, alive], np.ones((len(TE), 1))], 1)
                    @ _ridge(np.concatenate([Ff[:, alive], Fv[:, alive]], 0), sFV, lmf), y[TE])
    print(f"[sae] SAE-FULL ({len(alive)} features): R^2={j_full:.3f}  AUC={auc_full:.3f}  "
          f"(lambda {lmf}; val {bestl[0]:.3f})", flush=True)

    PdF = np.concatenate([Ff[:, sel].T, Fv[:, sel].T], 1)        # [K, fit+val]
    wk = _ridge(PdF.T, YFV)
    j_struct = _r2(wk, Ft[:, sel], Y[TE])
    auc_struct = _auc(np.concatenate([Ft[:, sel], np.ones((len(TE), 1))], 1) @ _ridge(PdF.T, sFV), y[TE])

    dir_idx = np.concatenate([FITe, VALe])
    Rf_, Re_ = Res[dir_idx], Res[TE]
    Ren = Re_ / (np.linalg.norm(Re_, axis=1, keepdims=True) + 1e-9)

    def _fid(P_dir, r_eval):
        """paper Fidelity: v = presence-weighted DiffMean over Res[FIT+VAL], spearman on TEST."""
        fids = np.zeros(len(P_dir))
        for j in range(len(P_dir)):
            w = np.clip(P_dir[j] / (P_dir[j].max() + 1e-9), 0, 1)
            if w.sum() < 1e-6 or (1 - w).sum() < 1e-6 or r_eval[j].std() < 1e-9:
                continue
            v = (w / w.sum()) @ Rf_ - ((1 - w) / (1 - w).sum()) @ Rf_
            nv = np.linalg.norm(v)
            if nv > 1e-9:
                fids[j] = _spear(Ren @ (v / nv), r_eval[j])
        return fids
    fid_feat = _fid(PdF, Ft[:, sel].T)
    print(f"\n[sae] SAE-STRUCT ({Kk} features): R^2={j_struct:.3f}  AUC={auc_struct:.3f}  "
          f"effK={_effk(Ft[:, sel].T):.2f}  Fidelity(feat)={fid_feat.mean():+.2f}", flush=True)
    # SAE HEALTH, for a configuration sweep. L0 is not reported because a TopK SAE fixes it at
    # --topk by construction; what varies is how much of the dictionary is actually used.
    alive = int((sent_F > 0).any(0).sum())
    print(f"[sae] SAE-HEALTH: dict={D} topk={args.topk} alive={alive}/{D} "
          f"dead={1 - alive / D:.3f} FVU={_FVU:.3f} "
          f"mean_support={float(supp.mean()):.3f}", flush=True)
    if args.struct_only:
        _prov2 = {"args": vars(args), "argv": " ".join(_sys.argv)}
        torch.save({**_prov2, "names": [], "feature_idx": sel,
                    "dirs_causal": W_dec[:, sel].T.cpu().numpy().astype(np.float32),
                    "j_struct": j_struct, "auc_struct": auc_struct, "fidelity_feat": fid_feat,
                    "alive": alive, "fvu": _FVU, "struct_only": True}, args.out)
        print(f"[sae] wrote {args.out}  (--struct-only: no names, no causal legs)", flush=True)
        return

    # ---- name the selected features: contrastive captioning + probe-corr argmax (snap) ----
    judge = FrozenJudge(mname, device=dev, dtype=dt,
                        max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)
    tok = judge.tok

    @torch.no_grad()
    def _gen(prompts, chunk=12):
        outs = []
        tok.padding_side = "left"
        for i in range(0, len(prompts), chunk):
            enc = tok(prompts[i:i + chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=3584, add_special_tokens=False).to(dev)
            out = judge.model.generate(**enc, do_sample=True, temperature=args.gen_temp, top_p=0.95,
                                       max_new_tokens=args.gen_max_new, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g_, skip_special_tokens=True) for g_ in out[:, enc["input_ids"].shape[1]:]]
        tok.padding_side = "right"
        return outs
    PRB = TRAIN[rng.permutation(len(TRAIN))[:args.probe_n]]
    if args.names_from:
        ext = torch.load(args.names_from, map_location="cpu", weights_only=False)
        names = [str(x) for x in ext["names"]][:Kk]
        Pprb = judge.presence_grid(names, [texts[i] for i in PRB], text_batch=48,
                                   grain="response", mode="scale")
        name_corr = [_corr(Pprb[j], F[PRB, sel[j]]) for j in range(Kk)]
        print(f"[sae] evaluating {Kk} external names from {args.names_from}", flush=True)
    else:
        names, name_corr = [], []
    for j, c in enumerate(sel if not args.names_from else []):
        sc = F[TRAIN, c]
        o = np.argsort(sc); pl = max(4 * args.contrast_n, 24)
        prompts = []
        for _m in range(args.name_samples):
            Aix = rng.choice(o[-pl:], args.contrast_n, replace=False)
            Bix = rng.choice(o[:pl], args.contrast_n, replace=False)
            exA = "\n\n".join(f"[A{i+1}] {texts[TRAIN[a]][:args.example_chars]}" for i, a in enumerate(Aix))
            exB = "\n\n".join(f"[B{i+1}] {texts[TRAIN[b]][:args.example_chars]}" for i, b in enumerate(Bix))
            prompts.append(
                judge.u_open + "Below are replies by an AI assistant (each shown with the user "
                "message it answers). The GROUP A replies share ONE specific, observable response "
                "behavior that the GROUP B replies lack.\n\nGROUP A:\n" + exA + "\n\nGROUP B:\n" + exB
                + "\n\nName that ONE behavior. Be specific and behavioral. Answer with a single short "
                "clause completing \"the assistant tends to ...\" -- output only the clause."
                + judge.a_open + "the assistant tends to")
        cands = [c_ for c_ in dict.fromkeys(_phrase(t) for t in _gen(prompts))
                 if c_ and len(c_.split()) >= 3]
        if not cands:
            names.append(f"SAE feature {c}"); name_corr.append(0.0)
            continue
        Ph = judge.presence_grid(cands, [texts[i] for i in PRB], text_batch=48,
                                 grain="response", mode="scale")
        cor = [(_corr(Ph[i], F[PRB, c]), cands[i]) for i in range(len(cands))]
        bc, bn = max(cor)
        names.append(bn); name_corr.append(bc)
        print(f"[sae] feature {c}: snap corr {bc:+.2f}  [pool {len(cands)}/{args.name_samples}]  {bn[:80]}", flush=True)

    # ---- SAE-NAMED ladder (judged presences of the names, same rungs) ----
    uniq = list(dict.fromkeys(names))
    EV = np.concatenate([FITe, VALe, TE])
    Pn = judge.presence_grid(uniq, [texts[i] for i in EV], text_batch=48,
                             grain="response", mode="scale").astype(np.float32)
    nf, nv = len(FITe), len(VALe)
    Pnf, Pnv, Pnt = Pn[:, :nf], Pn[:, nf:nf + nv], Pn[:, nf + nv:]
    PdN = np.concatenate([Pnf, Pnv], 1)
    j_named = _r2(_ridge(PdN.T, YFV), Pnt.T, Y[TE])
    auc_named = _auc(np.concatenate([Pnt.T, np.ones((len(TE), 1))], 1) @ _ridge(PdN.T, sFV), y[TE])
    r_named = np.stack([Pnt[uniq.index(names[i])] for i in range(Kk)])
    fid_named = _fid(PdF, r_named)                                # dirs from the FEATURES (the method)
    print(f"\n[sae] SAE-NAMED ({len(uniq)} names): R^2={j_named:.3f}  AUC={auc_named:.3f}  "
          f"effK={_effk(Pnt):.2f}  Fidelity(named)={fid_named.mean():+.2f}", flush=True)
    for i in range(Kk):
        print(f"    c{name_corr[i]:+.2f} / F{fid_named[i]:+.2f} / supp{supp[sel[i]]:.2f}  {names[i]}",
              flush=True)

    dec = W_dec[:, sel].T.cpu().numpy()                          # [K, d] unit decoder rows
    # PROVENANCE: the invocation this checkpoint came from, so a cell can be
    # checked against the flags it was BUILT with (--max-prompt-chars above all)
    # instead of inferred from mtimes and the battery script's git history.
    _prov = {"args": vars(args), "argv": " ".join(_sys.argv)}
    torch.save({**_prov, "names": names, "dirs_causal": dec.astype(np.float32), "feature_idx": sel,
                "snap_corr": np.array(name_corr), "j_struct": j_struct, "auc_struct": auc_struct,
                "fidelity_feat": fid_feat, "j_named": j_named, "auc_named": auc_named,
                "fidelity_named": fid_named, "support": supp[sel],
                "sae": {"W_enc": W_enc.cpu().numpy(), "b_enc": b_enc.cpu().numpy(),
                        "W_dec": W_dec.cpu().numpy(), "mu": mu, "scale": sd_g},
                "dict": D, "topk": args.topk}, args.out)
    print(f"[sae] wrote {args.out}  (dirs_causal = decoder rows; causal_diag-ready)", flush=True)


if __name__ == "__main__":
    main()
