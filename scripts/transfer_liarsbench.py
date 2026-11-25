#!/usr/bin/env python
"""Transfer of a dictionary to LiarsBench lie detection.

Uses the instructed-deception subset. With --strip-system the system prompt (which carries
the honest/deceptive instruction) is dropped, so detection must come from the response;
with --balance-by-model honest and deceptive responses are sampled equally within each
generating model, so model identity carries no label information; with --len-deconf
response length is regressed out of the activations. Each dictionary is tested twice: by
its direction scores (response activations projected on its directions) and by its concept
scores (the judge's rating of each entry's description on each response). Logistic
regression is fit on the even-indexed examples and AUROC is reported on the odd ones.
Baselines: random directions at each dictionary size, random descriptions, and a supervised
difference-of-means direction as a reference.

  python scripts/transfer_liarsbench.py --config CONFIG.yaml --subset instructed-deception \
      --strip-system --balance-by-model --len-deconf \
      --checkpoints DICT.pt [OTHER_DICT.pt ...] --out liarsbench.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os

import numpy as np
import torch
import yaml

DISTRACTORS = [
    "use contractions such as \"don't\" and \"it's\" throughout the response",
    "mention specific numbers, dates, or quantities",
    "include a rhetorical question somewhere in the response",
    "use metaphors or similes drawn from nature",
    "reference technology, apps, or the internet",
    "begin sentences with conjunctions like \"And\" or \"But\"",
    "use parentheses to add side remarks",
    "mention food, cooking, or eating",
    "include a numbered or bulleted list",
    "spell out abbreviations in full instead of shortening them",
    "use British spelling conventions",
    "quote or paraphrase what the user said before responding to it",
]


def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    m = ~np.isnan(y) & ~np.isnan(s)
    if m.sum() < 10 or len(set(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m], s[m]))


def _best_single(y, F):
    a = [max(v, 1 - v) if not np.isnan(v := _auc(y, F[:, j])) else np.nan
         for j in range(F.shape[1])]
    ok = [(v, j) for j, v in enumerate(a) if not np.isnan(v)]
    return max(ok) if ok else (float("nan"), -1)


def _ap(y, s):
    """Average precision (AUPRC); the class-prior baseline is mean(y)."""
    from sklearn.metrics import average_precision_score
    m = ~np.isnan(y) & ~np.isnan(s)
    if m.sum() < 10 or len(set(y[m])) < 2:
        return float("nan")
    return float(average_precision_score(y[m], s[m]))


def _lr_pred(ytr, Ftr, Fte):
    """Held-out predicted probabilities (paired tests need per-example scores)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if len(set(ytr)) < 2:
        return None
    sc = StandardScaler().fit(Ftr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(
        sc.transform(Ftr), ytr)
    return lr.predict_proba(sc.transform(Fte))[:, 1]


def _load_ckpt(path):
    """Entry descriptions (cycle anchors override names) and unit-norm directions; a
    checkpoint without directions uses those in <stem>_causal.pt."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    names = [str(x) for x in ck.get("names", [])]
    if ck.get("cyc_anchors"):
        names = [str(a) if a else names[j] for j, a in enumerate(ck["cyc_anchors"][:len(names)])]
    D = ck.get("dirs_causal")
    if D is None:
        cpath = path[:-3] + "_causal.pt"
        if os.path.exists(cpath):
            dd = torch.load(cpath, map_location="cpu", weights_only=False)
            src = dd.get("refine") or dd.get("distill")
            D = src["dirs"] if src else None
    if D is not None:
        D = np.asarray(D, np.float64)
        D = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return names, D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="subject-model config (layer, name)")
    ap.add_argument("--dataset", default="ventura1900/liarsbenchneurips")
    ap.add_argument("--subset", default=None, help="HF config name (instructed subset)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--rand-draws", type=int, default=10,
                    help="independent random-direction draws per dictionary size (the spread "
                         "over draws is the reported interval on the baseline)")
    ap.add_argument("--len-deconf", action="store_true",
                    help="regress response length out of the activations before scoring "
                         "(the same three length features describe_directions removes); the "
                         "length-only AUROC is always printed")
    ap.add_argument("--balance-by-model", action="store_true",
                    help="sample equal deceptive/honest counts within each generating model, "
                         "so model identity carries no label information (otherwise it lifts "
                         "every score, random directions included)")
    ap.add_argument("--strip-system", action="store_true",
                    help="drop system turns from the prompt (activations and judging): "
                         "removes the honest/deceptive instruction, so detection must come "
                         "from the response")
    ap.add_argument("--checkpoints", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=600, help="examples (balanced per class)")
    ap.add_argument("--max-prompt-chars", type=int, default=800)
    ap.add_argument("--max-text-chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config))
    layer = int(cfg["model"].get("layer", 20))
    dev = torch.device("cuda")

    # ---- dataset ----
    from datasets import load_dataset
    ds = load_dataset(args.dataset, args.subset, split=args.split,
                      verification_mode="no_checks")
    cols = ds.column_names
    print(f"[liars] {args.dataset}/{args.subset} [{args.split}]: n={len(ds)} cols={cols}",
          flush=True)
    lcol = next((c for c in ["label", "deceptive", "is_lie", "lie", "is_deceptive", "labels"]
                 if c in cols), None)
    if lcol is None or "messages" not in cols:
        raise SystemExit(f"[liars] expected a 'messages' column and a label column in {cols}")

    def _split_chat(v):
        """Prompt = the non-assistant turns (system turns dropped with --strip-system);
        response = the last assistant turn."""
        ms = v if isinstance(v, list) else ast.literal_eval(str(v))
        skip = {"assistant"} | ({"system"} if args.strip_system else set())
        pr = "\n".join(str(m.get("content", "")) for m in ms if m.get("role") not in skip)
        asst = [str(m.get("content", "")) for m in ms if m.get("role") == "assistant"]
        return pr, (asst[-1] if asst else "")
    pairs = [_split_chat(v) for v in ds["messages"]]
    prm_all = [p for p, _ in pairs]
    resp_all = [r for _, r in pairs]
    keep = [i for i, (_, r) in enumerate(pairs) if r.strip()]
    print(f"[liars] parsed 'messages' transcripts: {len(keep)}/{len(pairs)} rows with "
          f"an assistant turn", flush=True)

    def _lab(v):
        return 1 if str(v).strip().lower() in ("1", "true", "deceptive", "lie", "yes") else 0
    labels_all = ds[lcol]
    prm = [prm_all[i][:args.max_prompt_chars] for i in keep]
    resp = [resp_all[i][:args.max_text_chars] for i in keep]
    y = np.array([_lab(labels_all[i]) for i in keep], np.float64)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    if not len(pos) or not len(neg):
        raise SystemExit(f"[liars] single-class labels (pos={len(pos)} neg={len(neg)})")
    if args.balance_by_model and "model" in cols:
        # responses come from several models; if model identity correlates with the label,
        # any feature set (random directions included) can detect the model instead of the
        # lie, so each model contributes equal deceptive and honest counts
        mdl_all = ds["model"]
        mdl = np.array([str(mdl_all[i]) for i in keep])
        per_model, picks = max(1, args.n // (2 * max(len(set(mdl)), 1))), []
        for m in sorted(set(mdl)):
            mp = np.where((mdl == m) & (y == 1))[0]
            mn = np.where((mdl == m) & (y == 0))[0]
            h = min(per_model, len(mp), len(mn))
            if not h:
                print(f"[liars]   model {m[:38]:<38} SKIPPED (pos={len(mp)} neg={len(mn)})",
                      flush=True)
                continue
            picks.append(np.concatenate([rng.permutation(mp)[:h], rng.permutation(mn)[:h]]))
            print(f"[liars]   model {m[:38]:<38} {h}/class", flush=True)
        if not picks:
            raise SystemExit("[liars] --balance-by-model: no model has both classes")
        sel = rng.permutation(np.concatenate(picks))
        half = len(sel) // 2
    else:
        half = min(args.n // 2, len(pos), len(neg))
        sel = (np.concatenate([rng.permutation(pos)[:half], rng.permutation(neg)[:half]])
               if half else rng.permutation(np.concatenate([pos, neg]))[:args.n])
        sel = rng.permutation(sel)
    prm, resp, y = [prm[i] for i in sel], [resp[i] for i in sel], y[sel]
    print(f"[liars] sampled {len(sel)} ({half}/class"
          + ("; balanced within model)" if args.balance_by_model else ")"), flush=True)

    # ---- activations: mean over response tokens at the config layer ----
    from nld.utils.frozen_presence import FrozenJudge
    fj = FrozenJudge(cfg["model"]["name"], device=dev,
                     dtype=cfg["model"].get("dtype", "bfloat16"),
                     max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)
    tok, model = fj.tok, fj.model
    H = np.zeros((len(sel), model.config.hidden_size))
    with torch.no_grad():
        for c0 in range(0, len(sel), 8):
            texts, plens = [], []
            for i in range(c0, min(c0 + 8, len(sel))):
                texts.append(tok.apply_chat_template(
                    [{"role": "user", "content": prm[i]},
                     {"role": "assistant", "content": resp[i]}], tokenize=False))
                plens.append(len(tok(tok.apply_chat_template(
                    [{"role": "user", "content": prm[i]}], tokenize=False,
                    add_generation_prompt=True), add_special_tokens=False).input_ids))
            enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                      max_length=1400, add_special_tokens=False).to(dev)
            hs = model(**enc, output_hidden_states=True).hidden_states[layer + 1].float()
            am = enc.attention_mask
            for r_, pl in enumerate(plens):
                T = int(am[r_].sum())
                H[c0 + r_] = (hs[r_, pl:T].mean(0) if T > pl
                              else hs[r_, :T].mean(0)).cpu().numpy()
            if c0 % 160 == 0:
                print(f"[liars] extracted {c0 + len(plens)}/{len(sel)}", flush=True)

    # deceptive and honest responses can differ in length alone: report the length-only
    # AUROC, and with --len-deconf regress length out as describe_directions does
    ev, od = np.arange(0, len(sel), 2), np.arange(1, len(sel), 2)
    rl = np.array([len(r) for r in resp], np.float64)
    len_auc = _auc(y[od], rl[od])
    print(f"\n[liars] LENGTH-ONLY baseline AUROC {max(len_auc, 1 - len_auc):.3f} "
          f"(direction-corrected; mean chars {rl.mean():.0f}, "
          f"deceptive {rl[y == 1].mean():.0f} vs honest {rl[y == 0].mean():.0f})", flush=True)
    if args.len_deconf:
        L = np.stack([np.log1p(rl), rl, (rl > args.max_text_chars).astype(np.float64)], 1)
        L = (L - L[ev].mean(0)) / (L[ev].std(0) + 1e-9)
        L1 = np.concatenate([L, np.ones((len(L), 1))], 1)
        H = H - L1 @ np.linalg.lstsq(L1[ev], H[ev], rcond=None)[0]
        print("[liars] length de-confound applied to activations (fit on even half)",
              flush=True)
    mu = H[ev].mean(0)
    bodies = [f"User asked: {p}\n\nResponse: {r}" for p, r in zip(prm, resp)]
    res = {"n": len(sel), "rows": {}}

    def _row(tag, F, extra=""):
        p_ = _lr_pred(y[ev], F[ev], F[od])
        r = {"lr": _auc(y[od], p_) if p_ is not None else float("nan"),
             "ap": _ap(y[od], p_) if p_ is not None else float("nan")}
        r["single"], r["single_ix"] = _best_single(y[od], F[od])
        r["max"] = _auc(y[od], F[od].max(1))
        # per-example held-out predictions: every score set is evaluated on the same
        # examples, so transfer_summary.py can run paired tests
        res.setdefault("preds", {})[tag] = (p_.tolist() if p_ is not None else None)
        res["y_odd"] = y[od].tolist()
        res["rows"][tag] = r
        print(f"[liars]   {tag:<34} AUROC {r['lr']:.3f}  AUPRC {r['ap']:.3f}  "
              f"max-agg {r['max']:.3f}  single {r['single']:.3f} (#{r['single_ix']})"
              f"{extra}", flush=True)

    print("\n[liars] === TARGET: instructed-deception lie label (AUROC, fit-even/"
          "score-odd) ===", flush=True)
    for path in args.checkpoints:
        if not os.path.exists(path):
            print(f"[liars] SKIP (missing): {path}", flush=True)
            continue
        names, D = _load_ckpt(path)
        base = os.path.basename(path).replace(".pt", "")
        if D is not None:
            _row(f"{base} dirs", (H - mu) @ D[:len(names)].T)
        P = fj.presence_grid(names, bodies, text_batch=32, mode="scale")
        P = np.asarray(P.detach().float().cpu() if torch.is_tensor(P) else P)
        res["rows"].setdefault("_judged", {})[base] = P.tolist()
        _row(f"{base} judged", P.T)

    # baselines. Random directions are drawn at every dictionary size K in use, since a
    # random K-dimensional projection scores higher as K grows; the spread is over
    # --rand-draws independent draws, and the median draw is stored for paired tests.
    ks = sorted({len(_load_ckpt(p)[0]) for p in args.checkpoints if os.path.exists(p)}) \
        or [5]
    for K in ks:
        a, pr_ = [], []
        for _ in range(args.rand_draws):
            Dr = rng.normal(size=(K, H.shape[1]))
            Dr /= np.linalg.norm(Dr, axis=1, keepdims=True)
            F = (H - mu) @ Dr.T
            p_ = _lr_pred(y[ev], F[ev], F[od])
            a.append(_auc(y[od], p_)); pr_.append(p_)
        med = int(np.argsort(a)[len(a) // 2])
        tag = f"RANDOM dirs k{K}"
        res["rows"][tag] = {"lr": float(np.mean(a)), "ap": _ap(y[od], pr_[med]),
                            "sd": float(np.std(a)), "draws": [float(v) for v in a],
                            "single": float(np.max(a)), "single_ix": -1}
        res.setdefault("preds", {})[tag] = pr_[med].tolist()
        print(f"[liars]   {tag + f' (x{args.rand_draws})':<34} AUROC {np.mean(a):.3f} "
              f"+- {2 * np.std(a):.3f} (2sd over draws)  [median draw stored for tests]",
              flush=True)
        if K == ks[0]:
            res["rows"]["random_dirs"] = {"mean": float(np.mean(a)),
                                          "sd": float(np.std(a))}
    Pr = fj.presence_grid(list(rng.choice(DISTRACTORS, 5, replace=False)), bodies,
                          text_batch=32, mode="scale")
    Pr = np.asarray(Pr.detach().float().cpu() if torch.is_tensor(Pr) else Pr)
    _row("RANDOM concepts (K=5)", Pr.T)

    dm = H[ev][y[ev] == 1].mean(0) - H[ev][y[ev] == 0].mean(0)
    sky = _auc(y[od], (H[od] - mu) @ (dm / (np.linalg.norm(dm) + 1e-9)))
    print(f"[liars]   {'SKYLINE supervised DiffMean':<34} AUROC {sky:.3f}  "
          f"(fit on even-half labels)", flush=True)
    res["rows"]["skyline"] = sky

    json.dump(res, open(args.out, "w"), indent=1)
    print(f"\n[liars] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
