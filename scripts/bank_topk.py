#!/usr/bin/env python
"""NO-METHOD ABLATION: skip NLD entirely; just take the top-K concepts off the bank.

The pipeline's candidate bank is ~1024 concept phrases clustered from the extraction cache.
This script asks the obvious question a reviewer will: does any of the soft-concept
machinery earn its keep, or would picking the K bank concepts that correlate best with the
behavior label do just as well?

  1. rebuild the same bank the pipeline uses (vocab_source: extract -> 1024 representatives;
     falls back to the built-in concept list if no extraction cache exists)
  2. score every bank concept on TRAIN responses with the frozen judge (0-9 presence)
  3. rank by point-biserial correlation with the behavior label
  4. select K, build each direction as cov(acts, that concept's score) -- the SAME
     construction expert_taxonomy and --router balanced use
  5. save {names, dirs_causal} so causal_diag / routed_acme treat it like any checkpoint

--select top     : the K highest-correlating concepts (the naive baseline, as asked)
--select greedy  : forward selection on held-out R^2 of the label (controls for the
                   redundancy that plain top-K suffers -- the job lambda_div does in NLD)
Both are computed and printed; --select picks which one is saved.

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/bank_topk.py \
    --config configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml \
    --units $U/units_l20.npz --prompt-acts $U/prompt_last_l20.npz \
    --num-concepts 5 --out outputs/runs/soft_wc/bat_bank_k5.pt
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


_bank = build_bank    # shared with dir_baseline --cand-bank: same bank, provably


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True)
    ap.add_argument("--num-concepts", type=int, default=5)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--bank-n", type=int, default=384,
                    help="train responses used to score the bank (cost is bank x this)")
    ap.add_argument("--select", choices=["top", "greedy"], default="top")
    ap.add_argument("--max-text-chars", type=int, default=600)
    ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acquire("bank_" + args.out.replace("/", "_"))
    gpu_guard(10.0)
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config))

    # ---- battery-identical prep ----
    u = np.load(args.units, allow_pickle=True)
    Rmap = {}
    for h_, p in zip(np.asarray(u["sent_acts"], np.float64), [str(x) for x in u["sent_resp"]]):
        Rmap.setdefault(" ".join(p.split()), []).append(h_)
    Rmap = {k: np.mean(v, 0) for k, v in Rmap.items()}
    pa = np.load(args.prompt_acts, allow_pickle=True)
    Hr, prm, resp, yl = [], [], [], []
    for rt, pt, l_ in zip([str(x) for x in pa["response_text"]],
                          [str(x) for x in pa["prompt_text"]],
                          np.asarray(pa["label"], np.int64)):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); prm.append(pt); resp.append(rt); yl.append(int(l_))
    Hr = np.array(Hr); yl = np.array(yl); n = len(Hr)
    perm = rng.permutation(n); te = np.zeros(n, bool)
    te[perm[:max(1, int(n * args.heldout_frac))]] = True
    TRI = np.where(~te)[0]
    rows = TRI[rng.permutation(len(TRI))[:args.bank_n]]

    from taxonomy_discovery.core.types import BehaviorExample
    exs = [BehaviorExample(example_id=str(i), behavior_family="x", dataset_name="x",
                           split="train", prompt=prm[i], response=resp[i]) for i in TRI]
    bank, src = _bank(cfg, exs, args.vocab_size)
    print(f"[bank] {len(bank)} concepts from the {src}; scoring on {len(rows)} train "
          f"responses (K={args.num_concepts}, select={args.select})", flush=True)

    judge = FrozenJudge(cfg["model"]["name"], device=torch.device("cuda"),
                        dtype=cfg["model"].get("dtype", "bfloat16"),
                        max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)
    texts = [f"User asked: {prm[i][:args.max_prompt_chars]}\n\n"
             f"Response: {resp[i][:args.max_text_chars]}" for i in rows]
    P = np.asarray(judge.presence_grid(bank, texts, text_batch=32, mode="scale"),
                   dtype=np.float64)                              # [M, n_rows]
    y = yl[rows].astype(np.float64)

    # ---- rank by |point-biserial corr| with the behavior label ----
    Pc = P - P.mean(1, keepdims=True)
    yc = y - y.mean()
    den = (np.linalg.norm(Pc, axis=1) * np.linalg.norm(yc)) + 1e-9
    corr = (Pc @ yc) / den
    order = np.argsort(-np.abs(corr))
    top = order[:args.num_concepts]
    print("[bank] TOP by |corr| with the label:", flush=True)
    for r, j in enumerate(order[:max(args.num_concepts, 10)]):
        mark = "*" if j in set(top.tolist()) else " "
        print(f"   {mark} {r + 1:>2}. r={corr[j]:+.3f}  {bank[j][:88]}", flush=True)

    # ---- greedy forward selection on held-out label R^2 (redundancy-aware reference) ----
    ev, od = np.arange(0, len(rows), 2), np.arange(1, len(rows), 2)

    def _r2(idx, fit, sc):
        X = np.c_[P[idx][:, fit].T, np.ones(len(fit))]
        w, *_ = np.linalg.lstsq(X, y[fit], rcond=None)
        Xs = np.c_[P[idx][:, sc].T, np.ones(len(sc))]
        r = y[sc] - Xs @ w
        return 1.0 - float(r @ r) / float(((y[sc] - y[sc].mean()) ** 2).sum() + 1e-9)

    greedy: list[int] = []
    for _ in range(args.num_concepts):
        best = max((j for j in range(len(bank)) if j not in greedy),
                   key=lambda j: _r2(greedy + [j], ev, od))
        greedy.append(best)
    print(f"[bank] greedy (held-out label R^2 {_r2(greedy, ev, od):.3f} vs "
          f"top-K {_r2(list(top), ev, od):.3f}):", flush=True)
    for r, j in enumerate(greedy):
        print(f"     {r + 1:>2}. r={corr[j]:+.3f}  {bank[j][:88]}", flush=True)

    sel = list(top) if args.select == "top" else greedy
    # redundancy of the selection: mean |corr| between the chosen concepts' score vectors
    S = P[sel]
    Sc = S - S.mean(1, keepdims=True)
    C = (Sc @ Sc.T) / (np.outer(np.linalg.norm(Sc, axis=1), np.linalg.norm(Sc, axis=1)) + 1e-9)
    offd = C[~np.eye(len(sel), dtype=bool)]
    if len(offd):
        print(f"[bank] selected {len(sel)} ({args.select}); pairwise |corr| between their "
              f"presence scores: mean {np.abs(offd).mean():.3f} max {np.abs(offd).max():.3f} "
              f"(high = redundant, which is what lambda_div exists to prevent)", flush=True)
    else:
        print(f"[bank] selected 1 ({args.select}); redundancy undefined at K=1", flush=True)

    Xc = Hr[rows] - Hr[rows].mean(0)
    dirs = []
    for j in sel:
        s = P[j] - P[j].mean()
        v = (s[:, None] * Xc).mean(0)
        dirs.append(v / (np.linalg.norm(v) + 1e-9))
    dirs = np.array(dirs)
    torch.save({"names": [bank[j] for j in sel], "dirs_causal": dirs.astype(np.float32),
                "bank_size": len(bank), "bank_source": src, "select": args.select,
                "corr": corr[sel], "sel_idx": np.array(sel),
                "top_idx": np.array(top), "greedy_idx": np.array(greedy)}, args.out)
    print(f"[bank] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
