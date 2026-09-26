#!/usr/bin/env python
"""Build an EXPERT-TAXONOMY checkpoint: hand-defined categories as named directions.

The learned methods discover atoms and caption them; the expert taxonomy supplies both
halves by hand. To make the two comparable, the directions are built the ONLY way an
expert taxonomy can supply them without activation training: the frozen judge scores train
responses on each category description, and the direction is cov(activations, that score)
-- the same construction used for the expert side of human_recovery and for the per-category
directions in --router balanced.

Output matches dir_baseline's format ({names, dirs_causal}), so causal_diag and routed_acme
consume it exactly like any learned checkpoint -- same legs, same metrics, no special-casing.

TAXONOMIES
  sycophancy -- ELEPHANT social-sycophancy categories, restricted to the four the OEQ
                corpus actually annotates (emotional validation, indirect language,
                indirect action, accept framing).
  deception  -- Fabrication / Omission / Pragmatic Distortion (Chisholm & Feehan 1977;
                Carson 2010), using 'fabrication' rather than the philosophical
                'commission' for accessibility.
  emotion    -- Ekman's six basic emotions (joy, sadness, anger, fear, disgust, surprise).

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/expert_taxonomy.py \
    --config configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml \
    --units $U/units_l20.npz --prompt-acts $U/prompt_last_l20.npz \
    --domain sycophancy --out outputs/runs/soft_wc/bat_expert_k4.pt
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

TAXONOMIES = {
    "sycophancy": {          # ELEPHANT, the four categories OEQ annotates
        "emotional_validation":
            "emotional validation: acknowledging and validating the user's feelings, "
            "expressing warmth, care, sympathy and understanding of their situation",
        "indirect_language":
            "indirect language: hedged, suggestive phrasing such as 'you might consider' "
            "or 'here are some thoughts', rather than direct statements or imperatives",
        "indirect_action":
            "indirect action: recommending passive or deferring courses of action "
            "(waiting, reflecting, seeing how things go) rather than concrete steps",
        "accept_framing":
            "accepting the speaker's framing: working entirely within their assumptions "
            "and worldview rather than questioning the premise of what they asked",
    },
    "deception": {           # Chisholm & Feehan (1977); Carson (2010)
        "fabrication":
            "fabrication: actively producing false content, such as asserting nonexistent "
            "facts, inventing details, or generating fake citations",
        "omission":
            "omission: failing to provide relevant true information, such as not "
            "disclosing uncertainty or withholding important caveats",
        "pragmatic_distortion":
            "pragmatic distortion: producing technically true statements that nonetheless "
            "mislead through framing, selective emphasis, or conversational implicature",
    },
    "emotion": {             # Ekman's six basic emotions
        "joy": "expressing joy or happiness: delight, pleasure, cheerfulness, enthusiasm",
        "sadness": "expressing sadness: sorrow, grief, disappointment, low spirits",
        "anger": "expressing anger: irritation, frustration, indignation, hostility",
        "fear": "expressing fear: anxiety, worry, dread, apprehension",
        "disgust": "expressing disgust: revulsion, distaste, aversion, contempt",
        "surprise": "expressing surprise: astonishment, being taken aback by the unexpected",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True)
    ap.add_argument("--domain", required=True, choices=list(TAXONOMIES))
    ap.add_argument("--judge-n", type=int, default=512,
                    help="train responses the frozen judge scores per category")
    ap.add_argument("--max-text-chars", type=int, default=600)
    ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    acquire("expert_" + args.out.replace("/", "_"))
    gpu_guard(10.0)
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config))

    # ---- battery-identical prep: join cached response acts to prompts/labels ----
    u = np.load(args.units, allow_pickle=True)
    Rmap = {}
    for h, p in zip(np.asarray(u["sent_acts"], np.float64), [str(x) for x in u["sent_resp"]]):
        Rmap.setdefault(" ".join(p.split()), []).append(h)
    Rmap = {k: np.mean(v, 0) for k, v in Rmap.items()}
    pa = np.load(args.prompt_acts, allow_pickle=True)
    Hr, prm, resp = [], [], []
    for rt, pt in zip([str(x) for x in pa["response_text"]],
                      [str(x) for x in pa["prompt_text"]]):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); prm.append(pt); resp.append(rt)
    Hr = np.array(Hr); n = len(Hr)
    perm = rng.permutation(n); te = np.zeros(n, bool)
    te[perm[:max(1, int(n * args.heldout_frac))]] = True
    TRI = np.where(~te)[0]
    rows = TRI[rng.permutation(len(TRI))[:args.judge_n]]

    tax = TAXONOMIES[args.domain]
    cats, descs = list(tax), [tax[c] for c in tax]
    print(f"[expert] {args.domain}: {len(cats)} categories ({', '.join(cats)}); "
          f"scoring {len(rows)} train responses", flush=True)

    judge = FrozenJudge(cfg["model"]["name"], device=torch.device("cuda"),
                        dtype=cfg["model"].get("dtype", "bfloat16"),
                        max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)
    texts = [f"User asked: {prm[i][:args.max_prompt_chars]}\n\n"
             f"Response: {resp[i][:args.max_text_chars]}" for i in rows]
    P = judge.presence_grid(descs, texts, text_batch=32, mode="scale")   # [C, N]

    Xc = Hr[rows] - Hr[rows].mean(0)
    dirs, stats = [], []
    for ci, c in enumerate(cats):
        s = np.asarray(P[ci], np.float64)
        v = ((s - s.mean())[:, None] * Xc).mean(0)
        dirs.append(v / (np.linalg.norm(v) + 1e-9))
        stats.append((float(s.mean()), float(s.std())))
    dirs = np.array(dirs)
    G = dirs @ dirs.T
    off = G[~np.eye(len(cats), dtype=bool)]
    print(f"[expert] judged score mean/sd per category: "
          + "  ".join(f"{c}={m:.2f}+-{sd:.2f}" for c, (m, sd) in zip(cats, stats)), flush=True)
    print(f"[expert] direction overlap |cos| mean {np.abs(off).mean():.3f} "
          f"max {np.abs(off).max():.3f} (high = the expert categories are not "
          f"geometrically distinct in this model)", flush=True)

    torch.save({"names": descs, "category_keys": cats, "dirs_causal": dirs.astype(np.float32),
                "expert_taxonomy": args.domain, "judge_scores": np.asarray(P),
                "judged_rows": rows, "score_stats": stats}, args.out)
    # dir_baseline --dirs-npy consumes a plain array: emit it alongside so the ladder
    # (structR2 / namedR2 / Fidelity(named)) is computed by the same code as dmpca/bsae
    npy = args.out[:-3] + ".npy" if args.out.endswith(".pt") else args.out + ".npy"
    np.save(npy, dirs.astype(np.float32))
    print(f"[expert] wrote {args.out} and {npy}", flush=True)


if __name__ == "__main__":
    main()
