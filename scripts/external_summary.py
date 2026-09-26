#!/usr/bin/env python
"""Unified numerical summary of the external validations (AUROC + AUPRC + tests).

Reads liarsbench_val .json and aita_nta .pt results. Where per-example predictions were
stored, BOTH metrics are recomputed offline (no re-judging) and paired bootstrap tests are
run against the best arm of each artifact type; otherwise stored AUROCs are shown and the
AUPRC column reads '--'. AUPRC is reported against the class prior, which is the
no-skill baseline and differs sharply between tests (AITA's NTA target is ~87% positive,
LiarsBench is balanced by construction).

  python scripts/external_summary.py --files outputs/runs/soft_wc/liars_instructed_both.json \
      outputs/runs/soft_wc/aita_full.pt --paired
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(y) & ~np.isnan(s)
    return float(roc_auc_score(y[m], s[m])) if m.sum() > 9 and len(set(y[m])) > 1 else float("nan")


def _ap(y, s):
    from sklearn.metrics import average_precision_score
    m = ~np.isnan(y) & ~np.isnan(s)
    return (float(average_precision_score(y[m], s[m])) if m.sum() > 9
            and len(set(y[m])) > 1 else float("nan"))


def _load(path):
    if path.endswith(".json"):
        d = json.load(open(path))
    else:
        import torch
        d = torch.load(path, map_location="cpu", weights_only=False)
        d = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}
    return d


def _paired(y, pa, pb, M, reps, rng):
    idx = [rng.integers(0, len(y), len(y)) for _ in range(reps)]
    b = np.array([M(y[i], pa[i]) - M(y[i], pb[i]) for i in idx
                  if len(set(y[i])) > 1])
    lo, hi = np.percentile(b, [2.5, 97.5])
    return M(y, pa) - M(y, pb), lo, hi, 2 * min((b <= 0).mean(), (b >= 0).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--paired", action="store_true",
                    help="paired bootstrap of every arm against the best arm per artifact")
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    for path in args.files:
        if not os.path.exists(path):
            print(f"[ext] SKIP (missing): {path}"); continue
        d = _load(path)
        rows, preds = d.get("rows", {}), d.get("preds") or {}
        y = np.asarray(d.get("y_odd") or d.get("y") or [], np.float64)
        prior = float(y.mean()) if len(y) else d.get("pos_rate", float("nan"))
        floor = (rows.get("random_dirs") or {}).get("mean", float("nan"))
        ref = rows.get("skyline", float("nan"))
        print(f"\n=== {os.path.basename(path)} ===")
        print(f"n(scored)={len(y) if len(y) else '?'}  class prior={prior:.3f}"
              + (f"  random-dirs floor={floor:.3f}" if floor == floor else "")
              + (f"  supervised ref={ref:.3f}" if ref == ref else ""))
        print(f"{'arm':<30}{'AUROC':>8}{'AUPRC':>8}{'AP-prior':>10}")
        tab = []
        for tag, r in rows.items():
            if not isinstance(r, dict) or "lr" not in r:
                continue
            p_ = preds.get(tag)
            if p_ is not None and len(y):
                p_ = np.asarray(p_, np.float64)
                au, apr = _auc(y, p_), _ap(y, p_)
            else:
                au, apr = r.get("lr", float("nan")), r.get("ap", float("nan"))
            tab.append((tag, au, apr))
        for tag, au, apr in sorted(tab, key=lambda t: -(t[1] if t[1] == t[1] else 0)):
            lift = apr - prior if apr == apr and prior == prior else float("nan")
            print(f"{tag:<30}{au:>8.3f}"
                  + (f"{apr:>8.3f}" if apr == apr else f"{'--':>8}")
                  + (f"{lift:>+10.3f}" if lift == lift else f"{'--':>10}"))
        if not args.paired or not preds or not len(y):
            continue
        for arm in ("dirs", "judged"):
            cand = [(t, np.asarray(p, np.float64)) for t, p in preds.items()
                    if p is not None and t.endswith(arm)]
            if len(cand) < 2:
                continue
            for metric, M in (("AUROC", _auc), ("AUPRC", _ap)):
                best = max(cand, key=lambda c: M(y, c[1]))
                print(f"\n  paired {metric} vs best {arm} arm: {best[0]} "
                      f"({M(y, best[1]):.3f})")
                for t, p_ in sorted(cand, key=lambda c: -M(y, c[1])):
                    if t == best[0]:
                        continue
                    dlt, lo, hi, pv = _paired(y, best[1], p_, M, args.reps, rng)
                    star = " *" if (lo > 0 or hi < 0) else ""
                    print(f"    vs {t:<28}{M(y, p_):>7.3f}  delta {dlt:+.3f} "
                          f"[{lo:+.3f}, {hi:+.3f}]  p={pv:.3f}{star}")


if __name__ == "__main__":
    main()
