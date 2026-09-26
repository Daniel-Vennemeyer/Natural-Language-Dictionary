#!/usr/bin/env python
"""Dump the FINAL learned behavior labels (atom names) for every method x K x domain.

Which names are "final" depends on the method, and the battery's reporting convention is
followed exactly:
  soft   -> cyc_anchors (the stage-2 cycle anchors; these are what the causal legs steer
            and what CYC-NAMED scores). Falls back to snap `names` if a checkpoint has no
            cycle stage, and the source is printed per row so the two are never conflated.
  bsae   -> names (contrastive captions of each SAE feature)
  dmpca  -> names (contrastive captions of each DiffMean/PC direction)

Reads checkpoints only -- no GPU, no judge, no regeneration.

  PYTHONPATH=src python scripts/dump_names.py                       # all domains, text
  PYTHONPATH=src python scripts/dump_names.py --format md --out names.md
  PYTHONPATH=src python scripts/dump_names.py --domain syco --method soft --k 5
  PYTHONPATH=src python scripts/dump_names.py --format json --out names.json
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re

import torch

DOMAINS = {"syco": ("bat_", "sycophancy"), "dec": ("dbat_", "deception"),
           "emo": ("ebat_", "emotion")}


def _names(ck):
    """-> (list[str], source_key). Cycle anchors win: they are what the battery reports."""
    base = [str(x) for x in (ck.get("names") or [])]
    anc = ck.get("cyc_anchors")
    if anc:
        out, used = [], False
        for j in range(max(len(base), len(anc))):
            a = anc[j] if j < len(anc) else None
            if a:
                out.append(str(a)); used = True
            else:
                out.append(base[j] if j < len(base) else "")
        if used:
            return out, "cyc_anchors"
    return base, "names"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="outputs/runs/soft_wc")
    ap.add_argument("--domain", default=None, choices=list(DOMAINS))
    ap.add_argument("--method", default=None, help="soft | bsae | dmpca | ... (substring ok)")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--format", default="text", choices=["text", "md", "json", "csv"])
    ap.add_argument("--max-chars", type=int, default=0,
                    help="truncate each label for display (0 = full text)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for dom, (pfx, full) in DOMAINS.items():
        if args.domain and dom != args.domain:
            continue
        pat = os.path.join(args.runs, f"{pfx}*_k*.pt")
        for path in sorted(glob.glob(pat)):
            m = re.match(rf".*/{pfx}([a-z]+)_k(\d+)\.pt$", path)   # skip _s1/_causal/_sweep/_racme
            if not m:
                continue
            meth, K = m.group(1), int(m.group(2))
            if args.method and args.method not in meth:
                continue
            if args.k is not None and K != args.k:
                continue
            try:
                ck = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as e:  # noqa: BLE001
                print(f"[names] SKIP {path}: {e}")
                continue
            nm, src = _names(ck)
            if not nm:
                continue
            # mtime: "most recent" is a real question here -- deception checkpoints written
            # before the --max-prompt-chars 800 fix describe runs where 85% of prompts lost
            # the question, and nothing distinguishes them by filename.
            mt = datetime.datetime.fromtimestamp(os.path.getmtime(path))
            rows.append({"domain": full, "domain_key": dom, "method": meth, "K": K,
                         "source": src, "checkpoint": os.path.basename(path),
                         "mtime": mt.strftime("%Y-%m-%d %H:%M"),
                         "names": [" ".join(str(x).split()) for x in nm]})
    rows.sort(key=lambda r: (r["domain_key"], r["method"], r["K"]))

    def cut(s):
        return s if not args.max_chars else (s[:args.max_chars] + ("..." if len(s) > args.max_chars else ""))

    if args.format == "json":
        text = json.dumps(rows, indent=2)
    elif args.format == "csv":
        import csv
        import io
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["domain", "method", "K", "atom", "label", "source", "mtime"])
        for r in rows:
            for j, n in enumerate(r["names"]):
                w.writerow([r["domain"], r["method"], r["K"], j, n, r["source"], r["mtime"]])
        text = buf.getvalue()
    elif args.format == "md":
        L = []
        for r in rows:
            L.append(f"\n### {r['domain']} · {r['method']} · K={r['K']}  "
                     f"<sub>({r['source']}, {r['checkpoint']}, {r['mtime']})</sub>\n")
            for j, n in enumerate(r["names"]):
                L.append(f"{j}. {cut(n)}")
        text = "\n".join(L)
    else:
        L = []
        for r in rows:
            L.append(f"\n=== {r['domain']} / {r['method']} / K={r['K']}   "
                     f"[{r['source']}]  {r['checkpoint']}  ({r['mtime']})")
            for j, n in enumerate(r["names"]):
                L.append(f"  {j}  {cut(n)}")
        text = "\n".join(L)

    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        n_lab = sum(len(r["names"]) for r in rows)
        print(f"[names] wrote {args.out}: {len(rows)} taxonomies, {n_lab} labels")
        cyc = sum(1 for r in rows if r["source"] == "cyc_anchors")
        print(f"[names] {cyc}/{len(rows)} use cycle anchors; the rest fall back to snap names")
    else:
        print(text)


if __name__ == "__main__":
    main()
