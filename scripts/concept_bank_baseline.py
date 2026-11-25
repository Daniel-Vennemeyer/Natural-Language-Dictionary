#!/usr/bin/env python
"""Build a concept-bank dictionary: the K bank concepts most correlated with the label.

This baseline uses no dictionary training. The judge scores every concept in the candidate
bank (BANK_FILE, one concept per line from concept_bank_generate.py, else a built-in list)
on discovery responses; the K concepts with the largest |point-biserial correlation| with
the behavior label become the entries. Each entry's direction is the covariance between
response activations and that concept score. Saves {names, dirs_causal} for selectivity.py.

  BANK_FILE=<bank.txt> python scripts/concept_bank_baseline.py --config <config.yaml> \
    --units <units.npz> --prompt-acts <prompt_acts.npz> --seed 0 \
    --num-concepts 5 --select top --max-prompt-chars 400 --out <dictionary.pt>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from nld.utils.concept_bank import build_bank
from nld.utils.frozen_presence import FrozenJudge
from nld.utils.runlock import acquire, gpu_guard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True)
    ap.add_argument("--num-concepts", type=int, default=5)
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--bank-n", type=int, default=384,
                    help="discovery responses used to score the bank (cost is bank size x this)")
    ap.add_argument("--select", choices=["top"], default="top",
                    help="selection rule: the K concepts with the largest |corr| with the label")
    ap.add_argument("--max-text-chars", type=int, default=600)
    ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acquire("bank_" + args.out.replace("/", "_"))
    gpu_guard(10.0)
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config))

    # ---- join response activations to prompts and labels; discovery/test split identical to training ----
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

    from nld.core.types import BehaviorExample
    exs = [BehaviorExample(example_id=str(i), behavior_family="x", dataset_name="x",
                           split="train", prompt=prm[i], response=resp[i]) for i in TRI]
    bank, src = build_bank(cfg, exs, args.vocab_size)
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

    sel = list(top)
    # redundancy of the selection: |corr| between the chosen concepts' score vectors
    S = P[sel]
    Sc = S - S.mean(1, keepdims=True)
    C = (Sc @ Sc.T) / (np.outer(np.linalg.norm(Sc, axis=1), np.linalg.norm(Sc, axis=1)) + 1e-9)
    offd = C[~np.eye(len(sel), dtype=bool)]
    if len(offd):
        print(f"[bank] selected {len(sel)} ({args.select}); pairwise |corr| between their "
              f"presence scores: mean {np.abs(offd).mean():.3f} max {np.abs(offd).max():.3f} "
              f"(high = redundant)", flush=True)
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
                "top_idx": np.array(top)}, args.out)
    print(f"[bank] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
