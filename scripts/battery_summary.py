"""Summarize battery_syco.sh results into one method x K table.

Parses outputs/runs/soft_wc/bat_<method>_k<K>{,_causal,_sweep}.log for:
  struct R^2 / named R^2 / Fidelity(named)      (build log)
  oracle AIE + sel (prompting ceiling), distilled AIE + sel @ a8   (causal log)
  honest per-atom-alpha median selectivity      (sweep log)

  python scripts/battery_summary.py [--dir outputs/runs/soft_wc] [--prefix bat_]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PAT = {
    "struct": re.compile(r"(?:SOFT structure|SAE-STRUCT|DIR-STRUCT)[^:]*:\s+R\^2=([-\d.]+)"),
    "named": re.compile(r"(?:NAMED taxonomy|SAE-NAMED|DIR-NAMED)[^:]*:\s+R\^2=([-\d.]+)"),
    "fid": re.compile(r"Fidelity\(named\)=([+-][\d.]+)"),
    # soft's REPORTED names are the cycle anchors (same set the causal legs read AIE/sel off);
    # first CYC-NAMED line = best-state anchors = the ckpt's cyc_anchors key
    "cyc": re.compile(r"CYC-NAMED tax\. : R\^2=([-\d.]+)\s+AUC=[-\d.]+\s+K=\d+\s*"
                      r"effK=[-\d.]+\s+Fidelity=([+-][\d.]+)"),
    "ceil": re.compile(r"PROMPTING CEILING[^:]*: mean own-AIE=([\d.]+)\s+median sel=([\d.nan]+)"),
    "dist": re.compile(r"DISTILLED[^:]*: mean own-AIE=([\d.]+)\s+median sel=([\d.nan]+)"),
    "honest": re.compile(r"PER-ATOM-ALPHA:(?: steerable (\d+/\d+))?\s*oracle median sel=([\d.nan]+)"
                         r"(?: \(passing-only [\d.nan]+\))?\s+HONEST median sel=([\d.nan]+)"),
    "acme": re.compile(r"ACME \(expert-taxonomy\): mean ([+-][\d.]+)\s+median ([+-][\d.]+)"),
    "racme": re.compile(r"ROUTED ACME: mean ([+-][\d.]+)\s+min ([+-][\d.]+)"),
}


def _last(pat, text):
    m = pat.findall(text)
    return m[-1] if m else None


def collect(args, prefix):
    """Scrape one battery prefix into {(method, K): row}. Factored out of main so the
    cross-seed mode can collect several prefixes (one per training seed) and compare."""
    ks = {int(x) for x in args.k.split(",")} if args.k else None
    meths = {x.strip() for x in args.method.split(",")} if args.method else None
    root = Path(args.dir)
    rows = {}
    for f in sorted(root.glob(f"{prefix}*_k*.log")):
        # battery_dec/emo (and syco's router legs) write ${tag}_racme_${ROUTER}.log. The old
        # pattern required the optional group to be followed immediately by .log, so those
        # files matched NOTHING and were skipped -- leaving the ACME column empty for every
        # deception and emotion cell rather than merely unlabelled.
        m = re.match(rf"{prefix}(\w+?)_k(\d+)(_causal|_sweep|_parent|_racme(?:_\w+)?)?\.log",
                     f.name)
        if not m:
            continue
        method, k, leg = m.group(1), int(m.group(2)), m.group(3) or "_build"
        if ks is not None and k not in ks:
            continue
        if meths is not None and method not in meths:
            continue
        router = None
        if leg.startswith("_racme"):
            router = leg[len("_racme"):].lstrip("_") or None
            leg = "_racme"
        txt = f.read_text(errors="replace")
        r = rows.setdefault((method, k), {})
        if leg == "_build":
            r["struct"] = _last(PAT["struct"], txt)
            r["named"] = _last(PAT["named"], txt)
            r["fid"] = _last(PAT["fid"], txt)
            cyc = PAT["cyc"].findall(txt)
            if cyc:                                              # anchors = the reported name set
                r["named"], r["fid"] = cyc[0]
                r["namesrc"] = "cyc"
        elif leg == "_causal":
            c = _last(PAT["ceil"], txt); d = _last(PAT["dist"], txt)
            if c:
                r["o_aie"], r["o_sel"] = c
            if d:
                r["d_aie"], r["d_sel"] = d
        elif leg == "_sweep":
            h = _last(PAT["honest"], txt)
            if h:
                r["steer"], r["sw_oracle"], r["sw_honest"] = h
            a = _last(PAT["acme"], txt)
            if a:
                r["acme_mean"], r["acme_med"] = a
        elif leg == "_parent":
            a = _last(PAT["acme"], txt)
            if a:
                r["acme_mean"], r["acme_med"] = a
        elif leg == "_racme":
            a = _last(PAT["racme"], txt)
            if a:
                r.setdefault("racme_by", {})[router or "default"] = a
                # sigma-unit ACME: the racme .pt saves base_per_prompt -- sigma = std of the
                # baseline per-prompt judged scores (the caption's "units of baseline sd")
                pt = f.with_suffix(".pt")
                if pt.exists():
                    try:
                        import numpy as _np
                        import torch as _torch
                        ck = _torch.load(pt, map_location="cpu", weights_only=False)
                        bp = (ck.get("base_per_prompt") or {}).get("target")
                        if bp is not None and len(bp) > 1:
                            sd = float(_np.asarray(bp, _np.float64).std(ddof=1))
                            if sd > 1e-9:
                                r.setdefault("sigma_by", {})[router or "default"] = sd
                        # PAIRED t over prompts: the racme arms score the SAME prompt set before
                        # and after, so the per-prompt DIFFERENCE is the unit of evidence and its
                        # own sd is the denominator -- NOT the baseline sd that ACME(sd) reports,
                        # which is an effect-SIZE scale and says nothing about n.
                        from bootstrap_cis import load_racme                  # one definition of
                        rc = load_racme(str(pt))                              # "which alpha* row"
                        dvec = _np.asarray(rc["steered"], _np.float64) - _np.asarray(rc["base"],
                                                                                     _np.float64)
                        nP = len(dvec); sdd = float(dvec.std(ddof=1))
                        if nP > 1 and sdd > 1e-9:
                            r.setdefault("t_by", {})[router or "default"] = (
                                float(dvec.mean() / (sdd / _np.sqrt(nP))), nP)
                    except Exception:  # noqa: BLE001  (old-format .pt: column prints --)
                        pass
    # routed ACME (all atoms + learned router) supersedes the per-atom parent number. With
    # several routers present prefer balanced (the maximin solver) over axis, and SAY which
    # one the cell came from -- silently mixing routers across rows would make the column
    # incomparable.
    for r in rows.values():
        by = r.get("racme_by") or {}
        for pick in ("balanced", "axis", "default"):
            if pick in by:
                r["racme"], r["racme_min"] = by[pick]
                r["router"] = pick
                if args.acme_source == "racme":
                    r["acme_med"] = r["racme"]
                    sd = (r.get("sigma_by") or {}).get(pick)
                    if sd:
                        r["acme_sd"] = f"{float(r['racme']) / sd:+.1f}σ"
                    tt = (r.get("t_by") or {}).get(pick)
                    if tt:
                        r["acme_t"] = f"{tt[0]:+.2f}"
                        r["acme_df"] = tt[1] - 1
                else:
                    r["router"] = "peratom±8"
                break

    return rows


SEED_COLS = [("named", "namedR2"), ("fid", "fid(named)"), ("d_aie", "distAIE@a8"),
             ("sw_honest", "honestSel"), ("acme_med", "ACME"), ("acme_t", "ACME(t)")]


def seed_report(args, prefixes):
    """CROSS-SEED ROBUSTNESS: the same method/K trained under different seeds, one battery
    prefix each. Reports mean +/- half-range (the spread across seeds, NOT a sampling CI --
    with 2-3 seeds a percentile interval would be fiction) plus the raw per-seed values, so a
    difference between methods can be read against how far one method moves on a reseed."""
    per = {p: collect(args, p) for p in prefixes}
    have = {p: len(r) for p, r in per.items()}
    for p, n in have.items():
        print(f"[seed] {p:<10} {n} cells" + ("" if n else "   <-- MISSING (not run yet?)"))
    cells = sorted({c for r in per.values() for c in r})
    if not cells:
        raise SystemExit(f"no logs under {args.dir} for any of {prefixes}")
    hdr = ["method", "K", "seeds"] + [h for _, h in SEED_COLS]
    print("\n" + " | ".join(f"{h:>18}" for h in hdr))
    print("-|-".join("-" * 18 for _ in hdr))
    for method, k in cells:
        out = [method, str(k)]
        vals_by_col = []
        for key, _ in SEED_COLS:
            vs = []
            for p in prefixes:
                v = (per[p].get((method, k)) or {}).get(key)
                try:
                    vs.append(float(v))
                except (TypeError, ValueError):
                    pass
            vals_by_col.append(vs)
        n_seeds = max((len(v) for v in vals_by_col), default=0)
        out.append(str(n_seeds))
        for vs in vals_by_col:
            if not vs:
                out.append("--")
            elif len(vs) == 1:
                out.append(f"{vs[0]:+.3f}")
            else:
                mu = sum(vs) / len(vs)
                out.append(f"{mu:+.3f}+-{(max(vs) - min(vs)) / 2:.3f}")
        print(" | ".join(f"{v:>18}" for v in out))
    print("\nper-seed values:")
    for method, k in cells:
        for key, lab in SEED_COLS:
            vs = [(p, (per[p].get((method, k)) or {}).get(key)) for p in prefixes]
            if all(v is None for _, v in vs):
                continue
            print(f"  {method}_k{k:<3} {lab:<12} " +
                  "  ".join(f"{p}={'--' if v is None else v}" for p, v in vs))
    print("\nnotes: spread here is TRAINING-seed variation at fixed data and dose protocol. "
          "It is the right yardstick for 'is this gap real?': a method difference smaller "
          "than the reseed half-range should not be interpreted. Atom-level (not metric-level) "
          "seed agreement is a separate measurement -- scripts/stability_eval.py, which "
          "matches atoms across two checkpoints by signed presence correlation against a "
          "permutation null.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/runs/soft_wc")
    ap.add_argument("--prefix", default="bat_")
    ap.add_argument("--method", default=None,
                    help="restrict to these methods (comma-separated), e.g. reft or soft,reft")
    ap.add_argument("--k", default=None,
                    help="restrict to these K (comma-separated), e.g. 9 or 1,5,9")
    ap.add_argument("--acme-source", choices=["racme", "peratom"], default="racme",
                    help="which measurement fills the ACME column: racme = routed "
                         "whole-taxonomy ACME (supersedes when present); peratom = the "
                         "per-atom +/-a8 sign-weighted parent experiment (the ORIGINAL "
                         "paper-table AIE convention)")
    ap.add_argument("--latex", action="store_true",
                    help="emit paper-table rows: method & K & namedR2 & fid & distAIE@a8 & honestSel & ACME")
    ap.add_argument("--seeds", default=None, metavar="PFX1,PFX2,...",
                    help="CROSS-SEED mode: collect these prefixes (one battery run per "
                         "training seed, e.g. bat_,bat_s1_,bat_s2_) and report each cell as "
                         "mean +/- half-range with the per-seed values, instead of one table")
    args = ap.parse_args()
    if args.seeds:
        return seed_report(args, [p for p in args.seeds.split(",") if p])
    rows = collect(args, args.prefix)
    if not rows:
        raise SystemExit(f"no {args.prefix}*_k*.log files under {args.dir}")
    cols = ["struct", "named", "fid", "acme_med", "acme_sd", "acme_t", "racme_min", "router",
            "o_aie", "o_sel", "d_aie", "d_sel", "sw_honest", "steer"]
    hdr = ["method", "K", "structR2", "namedR2", "fid(named)", "ACME", "ACME(sd)", "ACME(t)",
           "ACMEmin", "router", "oracleAIE", "oracleSel", "distAIE@a8", "distSel@a8",
           "honestSel", "steerable"]
    if args.latex:
        # paper table rows: method & K & namedR2 & fid & AIE & Sel (distilled @ a8; honest sel)
        NAME = {"soft": "NLD (ours)", "bsae": "Behavior SAE", "dmpca": "DiffMean PCA",
                "psae": "Pretrained SAE", "reft": "ReFT"}
        def _f(v, scale=1.0, nd=2):
            try:
                return f"{float(v) * scale:.{nd}f}"
            except (TypeError, ValueError):
                return "---"
        for (method, k) in sorted(rows):
            r = rows[(method, k)]
            print(f"{NAME.get(method, method)} & {k} & {_f(r.get('named'))} & {_f(r.get('fid'))} & "
                  f"{_f(r.get('d_aie'))} & {_f(r.get('sw_honest'))} & {_f(r.get('acme_med'))} \\\\")
        return
    print(" | ".join(f"{h:>10}" for h in hdr))
    print("-|-".join("-" * 10 for _ in hdr))
    for (method, k) in sorted(rows):
        r = rows[(method, k)]
        vals = [method, str(k)] + [str(r.get(c) if r.get(c) is not None else "--") for c in cols]
        print(" | ".join(f"{v:>10}" for v in vals))
    print("\nnotes: K=1 selectivity is undefined (nan); fid(named) is the atom-aligned paper "
          "protocol from each build log; honestSel = per-atom alpha, select-even/score-odd. "
          "For runs with cycle anchors (soft), namedR2/fid report the CYC-NAMED (best-state "
          "anchors) line -- the same name set the causal legs steer/read off -- not the "
          "re-snapped NAMED line. ACME = median per-atom sign-weighted change in the EXPERT-"
          "taxonomy parent judgments (overall syco + validation/indirectness/framing) at +/-a8 "
          "-- the top-level behavior, not the atom's own name. ACME(t) = PAIRED t over the "
          "interv-prompts (mean per-prompt change / SE of that change, df = n-1); ACME(sd) is "
          "the effect SIZE in baseline-sd units and does not scale with n. The t assumes "
          "prompt independence and a roughly symmetric difference distribution -- judged "
          "scores are banded, so bootstrap_cis.py's percentile CI is the primary interval and "
          "this column is the quick significance read.")


if __name__ == "__main__":
    main()
