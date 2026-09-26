#!/usr/bin/env python
"""PAIRED ARM-vs-ARM comparison of two routed_acme runs (no GPU, no re-judging).

Two racme runs on the same config/seed see the SAME pinned prompts and produce the SAME
baseline generations, so their per-prompt judged scores are directly pairable: the contrast
A_steered - B_steered is measured within prompt, which removes prompt difficulty (the
dominant variance) and is far tighter than comparing two independently-reported means.

Reports, per alpha and per metric: mean difference, paired SE, effect in sd, and W/L/T.

  PYTHONPATH=src python scripts/racme_compare.py \
    --a outputs/runs/soft_wc/racme_soft_k5_axis96.pt \
    --b outputs/runs/soft_wc/racme_soft_k5_none96.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def _key(a):
    return f"a{a:g}_s1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="arm A .pt (e.g. axis)")
    ap.add_argument("--b", required=True, help="arm B .pt (e.g. none)")
    ap.add_argument("--label-a", default=None); ap.add_argument("--label-b", default=None)
    args = ap.parse_args()
    A = torch.load(args.a, map_location="cpu", weights_only=False)
    B = torch.load(args.b, map_location="cpu", weights_only=False)
    la = args.label_a or A.get("router_type", "A")
    lb = args.label_b or B.get("router_type", "B")

    for k in ("per_prompt", "base_per_prompt"):
        for nm, D in (("--a", A), ("--b", B)):
            if k not in D:
                raise SystemExit(f"{nm} lacks '{k}': rerun it with routed_acme >= 8230492")

    pa = A["generations"]["prompts"] if "generations" in A else None
    pb = B["generations"]["prompts"] if "generations" in B else None
    if pa is not None and pb is not None and list(pa) != list(pb):
        raise SystemExit("the two runs used DIFFERENT prompts -- not pairable (same seed?)")
    ba, bb = A["base_per_prompt"]["target"], B["base_per_prompt"]["target"]
    same_base = bool(np.allclose(np.asarray(ba), np.asarray(bb)))
    print(f"[cmp] {la} vs {lb}   n={len(ba)} prompts   "
          f"baselines {'IDENTICAL' if same_base else 'DIFFER (report is still paired)'}",
          flush=True)

    alphas = sorted({a for (a, s) in A["per_prompt"] if s > 0}
                    & {a for (a, s) in B["per_prompt"] if s > 0})
    if not alphas:
        # arms may have been swept at different alphas: compare every A alpha to every B one
        alphas = []
        print("[cmp] no shared alpha; comparing each arm at its own operating point",
              flush=True)
        aa = max((a for (a, s) in A["per_prompt"] if s > 0),
                 key=lambda a: A["by_alpha"][a]["acme"])
        bb_ = max((a for (a, s) in B["per_prompt"] if s > 0),
                  key=lambda a: B["by_alpha"][a]["acme"])
        pairs = [(aa, bb_)]
    else:
        pairs = [(a, a) for a in alphas]

    for a_a, a_b in pairs:
        PA, PB = A["per_prompt"][(a_a, 1.0)], B["per_prompt"][(a_b, 1.0)]
        mets = [m for m in PA if m in PB]
        tag = f"alpha {a_a:g}" if a_a == a_b else f"{la}@{a_a:g} vs {lb}@{a_b:g}"
        print(f"\n[cmp] === {tag} ===", flush=True)
        for m in ["target"] + [x for x in mets if x != "target"]:
            d = np.asarray(PA[m], np.float64) - np.asarray(PB[m], np.float64)
            se = d.std(ddof=1) / np.sqrt(len(d))
            w, l_, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
            star = "  *" if abs(d.mean()) > 2 * se else ""
            print(f"[cmp]   {m:<14} {la} - {lb} = {d.mean():+.3f} +-{se:.3f} "
                  f"({d.mean() / max(se, 1e-9):+.1f} sd)  {w}W/{l_}L/{t}T{star}", flush=True)


if __name__ == "__main__":
    main()
