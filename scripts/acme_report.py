#!/usr/bin/env python
"""FULL ACME readout from the saved routed_acme logs -- every router, every alpha, every category.

battery_summary collapses ACME to one number per cell (mean, preferred router). That is the right
summary and the wrong thing to read when the question is whether an intervention actually worked:
the mean can cancel when an intervention TRADES categories, the operating-point alpha is chosen
under a length guard that may not have passed, and axis-vs-balanced is the comparison the whole
router argument turns on. This prints all of it, from logs already on disk -- no GPU, no judge.

  PYTHONPATH=src python scripts/acme_report.py --prefix dbat_
  PYTHONPATH=src python scripts/acme_report.py --prefix ebat_ --k 9 --all-alphas
  PYTHONPATH=src python scripts/acme_report.py --prefix dbat_ --format csv --out acme_dec.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import re
from pathlib import Path

# Exactly the lines routed_acme prints (see its reporting block).
BASE = re.compile(r"baseline (\w+) \(([^)]*)\) rate/intensity ([\d.]+)\s+\(([^)]*)\)")
ALPHA = re.compile(                                  # re.M: the trailing $ must mean end-of-LINE,
    r"alpha \+([\d.]+): ACME ([+-][\d.]+) \+-([\d.]+) \(paired SE, ([+-][\d.]+) sd; "
    r"(\d+)W/(\d+)L/(\d+)T\)(.*?)\s+\(len ([^)]*)\)\s*(.*)$",
    re.M)                                            # else only the final line can ever match
CAT = re.compile(r"d(\w+)=([+-][\d.]+)\+-([\d.]+)\(([+-][\d.]+)sd\)")
FINAL = re.compile(r"ROUTED ACME: mean ([+-][\d.]+)\s+min ([+-][\d.]+)\s+\(alpha ([\d.]+), "
                   r"len ([^,]*), baseline rate ([\d.]+)(; NO alpha passed the len guard)?\)")
WORST = re.compile(r"WORST category cos ([+-][\d.]+) -> ([+-][\d.]+)")
CONE = re.compile(r"cos\(axis, atom SPAN\) ([\d.]+)\s+cos\(axis, NONNEG cone\) ([\d.]+)")
NAME = re.compile(r"^(\w+?)_k(\d+)_racme(?:_(\w+))?\.log$")


def parse(path: Path) -> dict | None:
    txt = path.read_text(errors="replace")
    fin = FINAL.findall(txt)
    if not fin:
        return None
    mean, dmin, astar, lstar, brate, noguard = fin[-1]
    b = BASE.search(txt)
    alphas = []
    for m in ALPHA.finditer(txt):
        a, acme, se, sd, w, l_, t, _extra, ln, cats = m.groups()
        alphas.append({"alpha": float(a), "acme": float(acme), "se": float(se), "sd": float(sd),
                       "w": int(w), "l": int(l_), "t": int(t), "len": ln,
                       "cats": [{"name": c[0], "d": float(c[1]), "se": float(c[2]),
                                 "sd": float(c[3])} for c in CAT.findall(cats)]})
    worst = WORST.findall(txt)
    cone = CONE.findall(txt)
    return {"mean": float(mean), "min": float(dmin), "alpha_star": float(astar),
            "len_star": lstar.strip(), "baseline": float(brate),
            "guard_failed": bool(noguard),
            "domain": b.group(1) if b else "?",
            "base_cats": b.group(4) if b else "",
            "alphas": alphas,
            "worst_cos": (float(worst[-1][0]), float(worst[-1][1])) if worst else None,
            "cone": (float(cone[-1][0]), float(cone[-1][1])) if cone else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/runs/soft_wc")
    ap.add_argument("--prefix", required=True, help="dbat_ | ebat_ | bat_ | ...")
    ap.add_argument("--method", default=None,
                    help="restrict to these methods (comma-separated), e.g. reft")
    ap.add_argument("--k", default=None, help="restrict to these K (comma-separated)")
    ap.add_argument("--all-alphas", action="store_true",
                    help="every evaluated alpha, not just the operating point")
    ap.add_argument("--format", default="text", choices=["text", "csv"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ks = {int(x) for x in args.k.split(",")} if args.k else None
    meths = {x.strip() for x in args.method.split(",")} if args.method else None

    cells = []
    for f in sorted(Path(args.dir).glob(f"{args.prefix}*_k*_racme*.log")):
        m = NAME.match(f.name[len(args.prefix):])
        if not m:
            continue
        meth, K, router = m.group(1), int(m.group(2)), m.group(3) or "default"
        if ks is not None and K not in ks:
            continue
        if meths is not None and meth not in meths:
            continue
        r = parse(f)
        if r:
            cells.append({"method": meth, "K": K, "router": router, "log": f.name, **r})
    if not cells:
        raise SystemExit(f"no parseable {args.prefix}*_racme*.log under {args.dir}")
    cells.sort(key=lambda c: (c["method"], c["K"], c["router"]))

    if args.format == "csv":
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["domain", "method", "K", "router", "alpha", "acme", "paired_se", "sd",
                    "W", "L", "T", "len", "category", "d", "cat_se", "cat_sd",
                    "operating_point", "guard_failed", "baseline"])
        for c in cells:
            for a in c["alphas"]:
                star = abs(a["alpha"] - c["alpha_star"]) < 1e-9
                if not (args.all_alphas or star):
                    continue
                for cat in (a["cats"] or [{"name": "", "d": "", "se": "", "sd": ""}]):
                    w.writerow([c["domain"], c["method"], c["K"], c["router"], a["alpha"],
                                a["acme"], a["se"], a["sd"], a["w"], a["l"], a["t"], a["len"],
                                cat["name"], cat["d"], cat["se"], cat["sd"],
                                int(star), int(c["guard_failed"]), c["baseline"]])
        text = buf.getvalue()
    else:
        L = [f"\n=== FULL ACME  ({args.prefix}) ===",
             "ACME = delta in the expert-taxonomy judgment vs unsteered, same prompts (paired).",
             "The operating point is the largest-ACME alpha whose length ratio stays in "
             "[0.85, 1.15];",
             "'LEN GUARD FAILED' means no alpha did, so the number is confounded with response "
             "length.\n"]
        for c in cells:
            star = "  <-- operating point"
            L.append(f"--- {c['domain']} / {c['method']} / K={c['K']} / router={c['router']}"
                     + ("   [LEN GUARD FAILED]" if c["guard_failed"] else ""))
            L.append(f"    baseline {c['baseline']:.3f}   ({c['base_cats']})")
            if c["cone"]:
                L.append(f"    geometry: cos(axis, SPAN) {c['cone'][0]:.3f}  "
                         f"cos(axis, NONNEG cone) {c['cone'][1]:.3f}")
            if c["worst_cos"]:
                L.append(f"    worst-category cos: axis {c['worst_cos'][0]:+.3f} -> "
                         f"balanced {c['worst_cos'][1]:+.3f}")
            for a in c["alphas"]:
                is_star = abs(a["alpha"] - c["alpha_star"]) < 1e-9
                if not (args.all_alphas or is_star):
                    continue
                L.append(f"    alpha +{a['alpha']:g}  ACME {a['acme']:+.3f} +-{a['se']:.3f} "
                         f"({a['sd']:+.1f} sd)  {a['w']}W/{a['l']}L/{a['t']}T  len {a['len']}"
                         + (star if is_star else ""))
                for cat in a["cats"]:
                    L.append(f"        d{cat['name']:<22} {cat['d']:+.3f} +-{cat['se']:.3f} "
                             f"({cat['sd']:+.1f} sd)")
            L.append(f"    reported: mean {c['mean']:+.3f}  min {c['min']:+.3f}  "
                     f"(alpha {c['alpha_star']:g}, len {c['len_star']})")
        text = "\n".join(L)

    if args.out:
        Path(args.out).write_text(text)
        print(f"[acme] wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
