#!/usr/bin/env python
"""VADER ACME for the emotion domain: re-score saved generations with a LEXICON, not a judge.

routed_acme stores every condition's generations, so the valence effect can be re-measured
post hoc with VADER -- deterministic, reproducible by anyone, and with no rubric, no
prompt-format sensitivity and no threshold to calibrate. For a domain whose behavior IS
sentiment, that removes the judge from the headline number entirely.

Reports, per alpha, the paired change in VADER compound (same prompts, baseline vs steered)
with SE and W/L/T, plus the positive/negative component shifts.

It also cross-checks the INSTRUMENT: when the .pt carries per-prompt judge scores, the
correlation between VADER compound and the judge's positivity is printed. High agreement
validates the judged emotion column; low agreement means one of the two is not measuring
valence, which is worth knowing before either goes in a table.

  python scripts/vader_acme.py --pt outputs/runs/soft_wc/ebat_soft_k5_racme_balanced.pt
  python scripts/vader_acme.py --pt <...> --out outputs/runs/soft_wc/vader_emo.pt
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import torch


def _analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer(), "vaderSentiment"
    except Exception:
        pass
    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        try:
            return SentimentIntensityAnalyzer(), "nltk"
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)
            return SentimentIntensityAnalyzer(), "nltk (downloaded lexicon)"
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"VADER unavailable ({e}).\n  pip install vaderSentiment\n"
            "  (or: pip install nltk && python -c \"import nltk; "
            "nltk.download('vader_lexicon')\")")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="routed_acme .pt with saved generations")
    ap.add_argument("--keys", default="", help="comma list of steered keys (default: all _s1)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sia, backend = _analyzer()
    ck = torch.load(args.pt, map_location="cpu", weights_only=False)
    gen = ck.get("generations")
    if not gen:
        raise SystemExit(f"no generations in {args.pt}")
    base = gen["base"]
    keys = ([k.strip() for k in args.keys.split(",") if k.strip()]
            or sorted(k for k in gen["steered"] if k.endswith("_s1")))

    def score(texts):
        s = [sia.polarity_scores(t.strip()[:2000]) for t in texts]
        return {k: np.array([x[k] for x in s], dtype=np.float64)
                for k in ("compound", "pos", "neg", "neu")}

    B = score(base)
    print(f"[vader] {backend}; {len(base)} prompts; router={ck.get('router_type')}", flush=True)
    print(f"[vader] baseline compound {B['compound'].mean():+.3f} "
          f"(pos {B['pos'].mean():.3f} neg {B['neg'].mean():.3f})", flush=True)
    print("\n[vader] === VADER ACME (paired per prompt, vs unsteered baseline) ===", flush=True)
    res = {"baseline": B}
    for k in keys:
        S = score(gen["steered"][k])
        d = S["compound"] - B["compound"]
        se = d.std(ddof=1) / np.sqrt(len(d))
        w, l_, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
        res[k] = {"scores": S, "d_compound": d}
        print(f"[vader]   {k:<10} compound {S['compound'].mean():+.3f}  dCOMPOUND "
              f"{d.mean():+.3f} +-{se:.3f} ({d.mean() / max(se, 1e-9):+.1f} sd)  "
              f"{w}W/{l_}L/{t}T   dpos {S['pos'].mean() - B['pos'].mean():+.3f}  "
              f"dneg {S['neg'].mean() - B['neg'].mean():+.3f}", flush=True)

    # ---- instrument cross-check: VADER vs the LLM judge on the SAME texts ----
    bp = ck.get("base_per_prompt") or {}
    pp = ck.get("per_prompt") or {}
    def _corr(a_, b_):
        """None when either side is (near-)constant -- correlation is undefined there,
        and NaN printed as a number invites a false 'they disagree' reading."""
        if len(a_) < 3 or a_.std() < 1e-9 or b_.std() < 1e-9:
            return None
        return float(np.corrcoef(a_, b_)[0, 1])

    if bp.get("target") is not None:
        jb = np.asarray(bp["target"], np.float64)
        if len(jb) == len(B["compound"]):
            r = _corr(jb, B["compound"])
            print(f"\n[vader] INSTRUMENT CHECK (baseline texts): corr(VADER compound, judge "
                  f"positivity) = " + (f"{r:+.3f}" if r is not None
                                       else "undefined (one side has no variance)"), flush=True)
            ds = []
            for (a, s), d_ in pp.items():
                key = f"a{a:g}_s{int(s)}"
                if s > 0 and key in res and d_.get("target") is not None:
                    jd = np.asarray(d_["target"], np.float64) - jb
                    ds.append((key, _corr(jd, res[key]["d_compound"])))
            for key, rr in ds:
                print(f"[vader]   {key}: corr(dVADER, dJUDGE) = "
                      + (f"{rr:+.3f}" if rr is not None else "undefined (no variance)"),
                      flush=True)
            ok = [x[1] for x in ds if x[1] is not None]
            if ok and np.mean(ok) < 0.2:
                print("[vader]   WARNING: the two instruments disagree on the CHANGE -- at "
                      "least one is not tracking valence; do not average them.", flush=True)
    else:
        print("\n[vader] (no per-prompt judge scores in this .pt -- rerun that racme cell "
              "with the per_prompt commit to get the instrument cross-check)", flush=True)

    if args.out:
        torch.save({"res": res, "pt": args.pt, "backend": backend}, args.out)
        print(f"\n[vader] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
