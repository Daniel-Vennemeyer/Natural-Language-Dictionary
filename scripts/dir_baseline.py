"""DIRECTION BASELINES: turn ANY K directions into the standard evaluable taxonomy package.

Directions from --method diffmean_pca (built here: [DiffMean] + top-(K-1) within-class PC
directions, the classic AxBench-style baselines; K=1 = pure DiffMean) or --dirs-npy (any
external [K, d] hidden-space basis: make_reft_basis output, a pretrained SAE's decoder rows,
probing dirs, ...). Then the SAME protocol as sae_baseline/soft_concepts:

  unit scores  = residualized-activation projections onto each direction (response level)
  names        = contrastive GROUP A/B captioning from each unit's top/bottom responses,
                 snap-selected by probe corr
  DIR-STRUCT   = R^2/AUC/effK of the K raw scores on TEST vs the canonical target
  DIR-NAMED    = judged names ladder + Fidelity(named) [atom-aligned: v from unit scores]
  checkpoint   = {names, dirs_causal} -> causal_diag (ceiling/distill/sweep) for AIE/selectivity

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/dir_baseline.py \
    --config configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml \
    --units $U/units_l20.npz --prompt-acts $U/prompt_last_l20.npz \
    --method diffmean_pca --num-concepts 5 --out outputs/runs/soft_wc/dmpca_k5.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

from taxonomy_discovery.utils.concept_bank import build_bank
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


def _pcs_w(fit, allx, P):
    _, _, Vt = np.linalg.svd(fit, full_matrices=False); W = Vt[:min(P, Vt.shape[0])].T
    return allx @ W, W


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
    if len(act) < 2:
        return float(len(act))
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
    ap.add_argument("--method", choices=["diffmean_pca", "external"], default="diffmean_pca")
    ap.add_argument("--dirs-npy", default=None,
                    help="external [K, d] hidden-space directions (reft basis, SAE decoder rows...); "
                         "implies --method external")
    ap.add_argument("--names-from", default=None,
                    help="a .pt with 'names' (e.g. an expert_taxonomy checkpoint): use those "
                         "names verbatim INSTEAD of contrastive captioning. The expert "
                         "taxonomy supplies its own labels, so re-captioning its directions "
                         "would measure our captioner, not the expert taxonomy -- but the "
                         "STRUCT/NAMED/Fidelity ladder must still be computed by this same "
                         "code so the row is comparable to dmpca/bsae.")
    ap.add_argument("--num-concepts", type=int, default=4)
    ap.add_argument("--target", choices=["acts_wc", "acts_wc_axisw"], default="acts_wc_axisw")
    ap.add_argument("--acts-dims", type=int, default=32)
    ap.add_argument("--name-samples", type=int, default=8); ap.add_argument("--contrast-n", type=int, default=6)
    ap.add_argument("--example-chars", type=int, default=500); ap.add_argument("--probe-n", type=int, default=256)
    ap.add_argument("--gen-temp", type=float, default=0.9); ap.add_argument("--gen-max-new", type=int, default=32)
    ap.add_argument("--fit-eval-n", type=int, default=512); ap.add_argument("--val-eval-n", type=int, default=256)
    ap.add_argument("--pc-dim", type=int, default=256); ap.add_argument("--shrink", type=float, default=0.15)
    ap.add_argument("--ridge", type=float, default=1.0); ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--no-len-deconf", action="store_true")
    ap.add_argument("--max-text-chars", type=int, default=600); ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--cand-bank", type=int, default=0, metavar="VOCAB",
                    help="NAME SOURCE: snap each direction to its nearest concept in the "
                         "pipeline's VOCAB-phrase bank instead of captioning it. The snap is "
                         "IDENTICAL (argmax corr with the unit score on the same probe set), so "
                         "only the candidate pool changes: generated vs retrieved. 0 = caption.")
    ap.add_argument("--no-bank-cache", action="store_true",
                    help="recompute the --cand-bank presence grid instead of reusing the cache")
    ap.add_argument("--sign", choices=["auto", "asbuilt"], default=None,
                    help="direction orientation. 'auto' flips each direction toward the behavior "
                         "axis (default when names are captioned here); 'asbuilt' keeps the "
                         "supplied signs (default under --names-from, whose dirs carry their own "
                         "orientation). Set EXPLICITLY to compare NAMING procedures: the arms must "
                         "share one orientation or Fidelity(named) inverts between them.")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acquire("dirb_" + args.out.replace("/", "_"))
    gpu_guard(12.0)
    dev = torch.device("cuda"); torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config)); mname = cfg["model"]["name"]; dt = cfg["model"].get("dtype", "bfloat16")

    # ---- prep: byte-compatible with soft_concepts/sae_baseline (same seed) ----
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
    Hr, Hp, y, resp, prm = [], [], [], [], []
    for hp, rt, pt, l in zip(Hp_all, rtxt, ptxt, lab):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); Hp.append(hp); y.append(int(l)); resp.append(rt); prm.append(pt)
    Hr, Hp, y = np.array(Hr), np.array(Hp), np.array(y); n = len(y)
    perm = rng.permutation(n); te = np.zeros(n, bool); te[perm[:max(1, int(n * args.heldout_frac))]] = True
    tr = ~te
    mu_r = Hr[tr].mean(0)
    Zr, Wr = _pcs_w((Hr - mu_r)[tr], Hr - mu_r, args.pc_dim)     # Wr: hidden -> pc basis [d, P]
    Zp, _ = _pcs_w((Hp - Hp[tr].mean(0))[tr], Hp - Hp[tr].mean(0), args.pc_dim)
    A = Zp[tr]; lam = args.ridge * np.trace(A.T @ A) / A.shape[1]
    Wm = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Zr[tr]); Res = Zr - Zp @ Wm
    if not args.no_len_deconf:
        rl = np.array([len(r) for r in resp], np.float64)
        L = np.stack([np.log1p(rl), rl, (rl > args.max_text_chars).astype(np.float64)], 1)
        L = (L - L[tr].mean(0)) / (L[tr].std(0) + 1e-9)
        L1 = np.concatenate([L, np.ones((len(L), 1))], 1)
        Res = Res - L1 @ np.linalg.lstsq(L1[tr], Res[tr], rcond=None)[0]
    p_, q_ = Res[tr][y[tr] == 1], Res[tr][y[tr] == 0]
    wl = np.linalg.solve(_shrinkc(np.cov(p_, rowvar=False) + np.cov(q_, rowvar=False), args.shrink),
                         p_.mean(0) - q_.mean(0))
    s = Res @ wl; s = (s - s[tr].mean()) / (s[tr].std() + 1e-9)
    if np.corrcoef(s[tr], y[tr])[0, 1] < 0:
        s = -s
    sy = np.where(tr & (y == 1))[0]; mu1 = Res[sy].mean(0)
    Y, Wy = _pcs_w(Res[sy] - mu1, Res - mu1, args.acts_dims)     # Wy: pc -> Y basis [P, D]
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
    K = args.num_concepts
    print(f"[dirb] n={n} fit={len(FITe)} val={len(VALe)} test={len(TE)}  "
          f"axis AUC={_auc(s[TE], y[TE]):.3f}", flush=True)

    # ---- directions (pc basis for scoring, hidden space for steering) ----
    if args.dirs_npy:
        args.method = "external"
        Dh = np.asarray(np.load(args.dirs_npy), np.float64)[:K]  # [K, d] hidden
        Dh = Dh / (np.linalg.norm(Dh, axis=1, keepdims=True) + 1e-9)
        Dpc = Dh @ Wr                                            # project into the Res PC basis
        print(f"[dirb] external dirs from {args.dirs_npy}: {Dh.shape}", flush=True)
    else:                                                        # diffmean_pca: DiffMean + top PCs
        dm_pc = Res[tr][y[tr] == 1].mean(0) - Res[tr][y[tr] == 0].mean(0)
        dirs_pc = [dm_pc / (np.linalg.norm(dm_pc) + 1e-9)]
        for d_ in range(K - 1):                                  # within-class PC directions
            v = Wy[:, d_]
            dirs_pc.append(v / (np.linalg.norm(v) + 1e-9))
        Dpc = np.stack(dirs_pc)                                  # [K, P]
        Dh = Dpc @ Wr.T                                          # back to hidden space for steering
        Dh = Dh / (np.linalg.norm(Dh, axis=1, keepdims=True) + 1e-9)
        print(f"[dirb] diffmean_pca dirs: 1 DiffMean + {K - 1} within-class PCs", flush=True)
    S = (Res @ Dpc.T).T.astype(np.float32)                       # [K, n] residualized projections
    # SIGN. For captioned directions (diffmean_pca, PCs) the sign is arbitrary, so orient
    # each toward the behavior axis. But directions supplied WITH their own names -- built
    # as cov(acts, concept score) by expert_taxonomy / bank_topk -- already point toward
    # "more of this concept", and flipping an ANTI-behavior concept (e.g. "blunt judgmental
    # tone", or Ekman's sadness/anger on a positive-valence axis) inverts exactly what
    # Fidelity(named) measures: the bank row reported Fidelity -0.35 with |F| 0.48-0.58,
    # i.e. faithful names scored as unfaithful. Keep the as-built orientation there, so
    # fidelity and R^2 are on the same footing as the soft-concept ladder (whose atoms are
    # never flipped against the axis either).
    sign_mode = args.sign or ("asbuilt" if args.names_from else "auto")
    if sign_mode == "asbuilt":
        print(f"[dirb] sign=asbuilt: keeping the as-built direction signs (names fix the "
              f"orientation); no axis flip", flush=True)
    else:
        print(f"[dirb] sign=auto: orienting each direction toward the behavior axis", flush=True)
        for j in range(K):                                       # sign: positive pole = axis-positive
            if _corr(S[j][TRAIN], s[TRAIN]) < 0:
                S[j], Dpc[j], Dh[j] = -S[j], -Dpc[j], -Dh[j]

    # ---- DIR-STRUCT ladder ----
    YFV = np.concatenate([Y[FITe], Y[VALe]]); sFV = np.concatenate([s[FITe], s[VALe]])
    PdF = np.concatenate([S[:, FITe], S[:, VALe]], 1)
    j_struct = _r2(_ridge(PdF.T, YFV), S[:, TE].T, Y[TE])
    auc_struct = _auc(np.concatenate([S[:, TE].T, np.ones((len(TE), 1))], 1) @ _ridge(PdF.T, sFV), y[TE])
    print(f"\n[dirb] DIR-STRUCT ({args.method}, K={K}): R^2={j_struct:.3f}  AUC={auc_struct:.3f}  "
          f"effK={_effk(S[:, TE]):.2f}", flush=True)

    # ---- name each direction: contrastive captioning + probe-corr snap ----
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
    names, name_corr = [], []
    if args.names_from:
        import torch as _t
        _nm = [str(x) for x in _t.load(args.names_from, map_location="cpu",
                                       weights_only=False)["names"]][:K]
        if len(_nm) < K:
            raise SystemExit(f"--names-from has {len(_nm)} names but K={K}")
        names = _nm
        _P = judge.presence_grid(names, [texts[i] for i in PRB], text_batch=48,
                                 grain="response", mode="scale")
        name_corr = [float(_corr(_P[j], S[j][PRB])) for j in range(K)]
        for j in range(K):
            print(f"[dirb] dir {j}: SUPPLIED name, probe corr {name_corr[j]:+.2f}  "
                  f"{names[j][:80]}", flush=True)
    BANK = None
    if args.cand_bank and not args.names_from:
        from taxonomy_discovery.core.types import BehaviorExample
        exs = [BehaviorExample(example_id=str(i), behavior_family="x", dataset_name="x",
                               split="train", prompt=prm[i], response=resp[i]) for i in TRAIN]
        BANK, bsrc = build_bank(cfg, exs, args.cand_bank)
        # Score the bank ONCE: the grid is concepts x probe texts and does not depend on j, so
        # all K directions snap out of one pass instead of K passes over ~1000 concepts.
        # It also does not depend on K AT ALL -- no rng is consumed between the split
        # permutation and PRB -- so a K-sweep would recompute the identical ~1024x256 grid
        # once per K. Cache it, keyed by a hash of the inputs and VERIFIED on load against the
        # stored concept list and probe indices, so a stale or drifted bank recomputes instead
        # of silently naming the directions off the wrong grid.
        import hashlib
        import os                              # module header binds it as _os
        key = hashlib.sha1("|".join([
            os.path.abspath(args.units), os.path.abspath(args.config), str(args.acts_key),
            str(args.max_prompt_chars), str(args.max_text_chars), str(args.probe_n),
            str(args.seed), str(args.cand_bank), str(mname)]).encode()).hexdigest()[:16]
        cpath = os.path.join("outputs", "cache", f"bankgrid_{key}.npz")
        BP = None
        if not args.no_bank_cache and os.path.exists(cpath):
            try:
                z = np.load(cpath, allow_pickle=True)
                if list(z["concepts"]) == list(BANK) and np.array_equal(z["prb"], PRB):
                    BP = z["grid"].astype(np.float64)
                    print(f"[dirb] cand-bank: reusing cached grid {cpath}", flush=True)
                else:
                    print(f"[dirb] cand-bank: cache {cpath} does not match this bank/probe "
                          f"set; recomputing", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[dirb] cand-bank: cache unreadable ({e}); recomputing", flush=True)
        if BP is None:
            BP = np.asarray(judge.presence_grid(BANK, [texts[i] for i in PRB], text_batch=32,
                                                mode="scale"), dtype=np.float64)
            if not args.no_bank_cache:
                os.makedirs(os.path.dirname(cpath), exist_ok=True)
                np.savez(cpath, concepts=np.array(BANK, dtype=object), prb=PRB,
                         grid=BP.astype(np.float32))
        print(f"[dirb] cand-bank: {len(BANK)} concepts from the {bsrc}, scored on "
              f"{len(PRB)} probe responses", flush=True)
    for j in range(K if not args.names_from else 0):
        if BANK is not None:                                     # RETRIEVE: same argmax, fixed pool
            bc, bn = max((_corr(BP[i], S[j][PRB]), BANK[i]) for i in range(len(BANK)))
            names.append(bn); name_corr.append(bc)
            print(f"[dirb] dir {j}: snap corr {bc:+.2f}  [pool {len(BANK)}/{len(BANK)}]  "
                  f"{bn[:80]}", flush=True)
            continue
        sc = S[j][TRAIN]; o = np.argsort(sc); pl = max(4 * args.contrast_n, 24)
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
            names.append(f"{args.method} direction {j}"); name_corr.append(0.0)
            continue
        Ph = judge.presence_grid(cands, [texts[i] for i in PRB], text_batch=48,
                                 grain="response", mode="scale")
        bc, bn = max((_corr(Ph[i], S[j][PRB]), cands[i]) for i in range(len(cands)))
        names.append(bn); name_corr.append(bc)
        print(f"[dirb] dir {j}: snap corr {bc:+.2f}  [pool {len(cands)}/{args.name_samples}]  {bn[:80]}", flush=True)

    # ---- DIR-NAMED ladder + Fidelity (atom-aligned: v from unit scores) ----
    uniq = list(dict.fromkeys(names))
    EV = np.concatenate([FITe, VALe, TE])
    Pn = judge.presence_grid(uniq, [texts[i] for i in EV], text_batch=48,
                             grain="response", mode="scale").astype(np.float32)
    nf, nv = len(FITe), len(VALe)
    Pnf, Pnv, Pnt = Pn[:, :nf], Pn[:, nf:nf + nv], Pn[:, nf + nv:]
    PdN = np.concatenate([Pnf, Pnv], 1)
    j_named = _r2(_ridge(PdN.T, YFV), Pnt.T, Y[TE])
    auc_named = _auc(np.concatenate([Pnt.T, np.ones((len(TE), 1))], 1) @ _ridge(PdN.T, sFV), y[TE])
    dir_idx = np.concatenate([FITe, VALe])
    Rf_, Re_ = Res[dir_idx], Res[TE]
    Ren = Re_ / (np.linalg.norm(Re_, axis=1, keepdims=True) + 1e-9)
    fid_named = np.zeros(K)
    r_named = np.stack([Pnt[uniq.index(names[i])] for i in range(K)])
    for j in range(K):
        w = S[j][dir_idx]
        w = np.clip((w - w.min()) / (w.max() - w.min() + 1e-9), 0, 1)
        if w.sum() < 1e-6 or (1 - w).sum() < 1e-6 or r_named[j].std() < 1e-9:
            continue
        v = (w / w.sum()) @ Rf_ - ((1 - w) / (1 - w).sum()) @ Rf_
        nv_ = np.linalg.norm(v)
        if nv_ > 1e-9:
            fid_named[j] = _spear(Ren @ (v / nv_), r_named[j])
    print(f"\n[dirb] DIR-NAMED ({len(uniq)} names): R^2={j_named:.3f}  AUC={auc_named:.3f}  "
          f"effK={_effk(Pnt):.2f}  Fidelity(named)={fid_named.mean():+.2f}", flush=True)
    for j in range(K):
        print(f"    c{name_corr[j]:+.2f} / F{fid_named[j]:+.2f}  {names[j]}", flush=True)

    # PROVENANCE: the invocation this checkpoint came from, so a cell can be
    # checked against the flags it was BUILT with (--max-prompt-chars above all)
    # instead of inferred from mtimes and the battery script's git history.
    _prov = {"args": vars(args), "argv": " ".join(_sys.argv)}
    torch.save({**_prov, "names": names, "dirs_causal": Dh.astype(np.float32), "method": args.method,
                "snap_corr": np.array(name_corr), "j_struct": j_struct, "auc_struct": auc_struct,
                "j_named": j_named, "auc_named": auc_named, "fidelity_named": fid_named},
               args.out)
    print(f"[dirb] wrote {args.out}  (dirs_causal ready for causal_diag)", flush=True)


if __name__ == "__main__":
    main()
