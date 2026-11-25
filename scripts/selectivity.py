"""Measure per-entry steering effects and selectivity for a dictionary.

Each entry j is steered along a direction v_j and every entry's description is judged on
the steered responses. The target effect is the change in entry j's own concept score;
the cross-effects are the changes in the other entries' scores. Selectivity is the target
effect divided by the largest cross-effect (floored at --sel-floor).

Two experiments (--only):
  distill  build each entry's direction from its description: the difference of mean
           activations between responses generated with the description as a system
           instruction and unprompted responses; then steer at a fixed --alpha.
  sweep    steer each direction at every strength in --alphas and pick a per-entry
           steering strength (select on even prompts, score on odd prompts).

Steering prompts come from the test split, which is rebuilt from --seed exactly as in
training, so --seed must match the checkpoint's training seed.

  python scripts/selectivity.py --config <config.yaml> --units <units.npz> \\
    --acts-key sent_acts --prompt-acts <prompt_acts.npz> --max-prompt-chars 400 --seed 0 \\
    --checkpoint <dictionary.pt> --only distill --alpha 8 --interv-prompts 48 \\
    --out <dictionary>_causal.pt
  python scripts/selectivity.py ... --checkpoint <dictionary.pt> --only sweep \\
    --dirs-from <dictionary>_causal.pt --alphas 2,4,6,8,12,16 --interv-prompts 96 \\
    --out <dictionary>_sweep.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from nld.utils.frozen_presence import FrozenJudge
from nld.utils.runlock import acquire, gpu_guard

EXPERIMENTS = {"distill", "sweep"}


def _pcs_w(fit, allx, P):
    _, _, Vt = np.linalg.svd(fit, full_matrices=False); W = Vt[:min(P, Vt.shape[0])].T
    return allx @ W, W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--units", required=True)
    ap.add_argument("--prompt-acts", required=True); ap.add_argument("--acts-key", default="resp_acts")
    ap.add_argument("--checkpoint", required=True,
                    help="dictionary .pt: uses dirs_causal, else dirs_pc, else names only")
    ap.add_argument("--dirs", choices=["causal", "obs"], default="causal",
                    help="causal = the checkpoint's steering directions (dirs_causal); "
                         "obs = its reading directions (dirs_pc) mapped back to activation space")
    ap.add_argument("--only", default="distill", help="comma subset of {distill,sweep}")
    ap.add_argument("--dirs-from", default=None,
                    help="a previous selectivity .pt: steer its distilled directions "
                         "instead of the checkpoint directions")
    ap.add_argument("--alphas", default="1,2,4,8,16",
                    help="sweep steering strengths, in units of the activation spread along each direction")
    ap.add_argument("--sel-floor", type=float, default=0.05,
                    help="selectivity denominator floor on the 0-1 concept-score scale: cross-effects "
                         "below judge noise count as the noise floor, not as ~0")
    ap.add_argument("--len-hi", type=float, default=1.5,
                    help="upper length-ratio guard for the per-entry steering strength: much longer, "
                         "degenerate text inflates the target effect just as truncation does "
                         "(the lower guard is 0.85)")
    ap.add_argument("--alpha", type=float, default=4.0, help="steering strength for distill")
    ap.add_argument("--interv-prompts", type=int, default=24)
    ap.add_argument("--gen-prompts", default=None,
                    help="file with one generation prompt per line: replaces the dataset prompts "
                         "for steering and judging. Needed for domains whose dataset prompt is a "
                         "constant frame, where greedy generation would yield one effective sample. "
                         "Sampled with rng(seed+2), so the set is fixed per seed.")
    ap.add_argument("--gen-max-new", type=int, default=160); ap.add_argument("--gen-temp", type=float, default=0.0)
    ap.add_argument("--pc-dim", type=int, default=256)
    ap.add_argument("--heldout-frac", type=float, default=0.25)
    ap.add_argument("--max-text-chars", type=int, default=600); ap.add_argument("--max-prompt-chars", type=int, default=400)
    ap.add_argument("--names-from", default=None,
                    help="a dictionary .pt whose final descriptions (cyc_anchors, else names) "
                         "replace the checkpoint's names")
    ap.add_argument("--seed", type=int, default=0, help="must match the checkpoint's training seed")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run = set(args.only.split(","))
    if run - EXPERIMENTS:
        raise SystemExit(f"[cdiag] unknown experiment(s) {sorted(run - EXPERIMENTS)}; "
                         f"choose from {sorted(EXPERIMENTS)}")
    acquire("cdiag_" + args.out.replace("/", "_"))
    gpu_guard(18.0)
    dev = torch.device("cuda"); rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(args.config)); mname = cfg["model"]["name"]; dt = cfg["model"].get("dtype", "bfloat16")
    layer = int(cfg["model"].get("layer", 20))

    # ---- prompts, discovery/test split and activation spread, identical to training ----
    u = np.load(args.units, allow_pickle=True)
    if args.acts_key == "sent_acts":
        Rmap = {}
        for h, p in zip(np.asarray(u["sent_acts"], np.float64), [str(x) for x in u["sent_resp"]]):
            Rmap.setdefault(" ".join(p.split()), []).append(h)
        Rmap = {k: np.mean(v, 0) for k, v in Rmap.items()}
    else:
        Rmap = {}
        for h, t in zip(np.asarray(u[args.acts_key], np.float64), [str(x) for x in u["resp_text"]]):
            Rmap.setdefault(" ".join(t.split()), h)
    pa = np.load(args.prompt_acts, allow_pickle=True)
    Hr, resp, prm = [], [], []
    for hp, rt, pt in zip(np.asarray(pa["prompt_acts"], np.float64),
                          [str(x) for x in pa["response_text"]], [str(x) for x in pa["prompt_text"]]):
        k = " ".join(rt.split())
        if k in Rmap:
            Hr.append(Rmap[k]); resp.append(rt); prm.append(pt)
    Hr = np.array(Hr); n = len(Hr)
    perm = rng.permutation(n); te = np.zeros(n, bool); te[perm[:max(1, int(n * args.heldout_frac))]] = True
    tr = ~te; mu_r = Hr[tr].mean(0)
    TE = np.where(te)[0]

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    names = [str(x) for x in ck["names"]]; Kk = len(names)
    if args.names_from:
        sd = torch.load(args.names_from, map_location="cpu", weights_only=False)
        src_names = sd["cyc_anchors"] if "cyc_anchors" in sd else sd.get("names", [])
        for j, nm in enumerate(src_names[:Kk]):
            if nm:
                names[j] = str(nm)
        print(f"[cdiag] names replaced from {args.names_from}", flush=True)
    if args.dirs == "causal" and "dirs_causal" in ck:
        Hd = np.asarray(ck["dirs_causal"], np.float64)
    elif "dirs_pc" in ck:
        _, Wr = _pcs_w((Hr - mu_r)[tr], Hr - mu_r, args.pc_dim)
        Hd = np.asarray(ck["dirs_pc"], np.float64) @ Wr.T
        args.dirs = "obs"
    else:                                                        # names-only checkpoint:
        Hd = None; args.dirs = "none"                            # distill, or sweep with --dirs-from
    if args.dirs_from:
        dd = torch.load(args.dirs_from, map_location="cpu", weights_only=False)
        Hd = np.asarray(dd["distill"]["dirs"], np.float64); args.dirs = f"from:{args.dirs_from}"
    if Hd is not None:
        Hd = Hd / (np.linalg.norm(Hd, axis=1, keepdims=True) + 1e-9)
    elif "sweep" in run:
        raise SystemExit("[cdiag] checkpoint has no directions: sweep needs --dirs-from")
    print(f"[cdiag] {Kk} atoms, dirs={args.dirs}, experiments={sorted(run)}", flush=True)
    for nm in names:
        print(f"    {nm[:90]}", flush=True)

    judge = FrozenJudge(mname, device=dev, dtype=dt,
                        max_text_chars=args.max_text_chars + args.max_prompt_chars + 40)
    tok = judge.tok
    pidx = TE[rng.permutation(len(TE))[:args.interv_prompts]]
    if args.gen_prompts:
        gp = [ln.strip() for ln in open(args.gen_prompts) if len(ln.strip()) >= 8]
        gsel = np.random.default_rng(args.seed + 2).permutation(len(gp))[:args.interv_prompts]
        prm = [gp[int(i)] for i in gsel]                         # _chats/_fmt read prm[pidx]
        pidx = np.arange(len(prm))
        print(f"[cdiag] generation prompts OVERRIDDEN: {len(prm)} of {len(gp)} "
              f"from {args.gen_prompts}", flush=True)

    # Prompts longer than --max-prompt-chars lose their tail, which is often where the
    # question is, so neither the model nor the judge would see what was asked.
    _tr = [len(prm[i]) > args.max_prompt_chars for i in pidx]
    if sum(_tr):
        _mx = max(len(prm[i]) for i in pidx)
        print(f"[cdiag] WARNING: {100 * sum(_tr) / len(_tr):.0f}% of eval prompts exceed "
              f"--max-prompt-chars {args.max_prompt_chars} (longest {_mx}) -- the TAIL is cut, and that is "
              f"where the question lives. Raise it (e.g. --max-prompt-chars {min(_mx + 50, 2000)}).",
              flush=True)

    def _chats(sys=None):
        return [tok.apply_chat_template(
            ([{"role": "system", "content": sys}] if sys else [])
            + [{"role": "user", "content": prm[i][:args.max_prompt_chars]}],
            tokenize=False, add_generation_prompt=True) for i in pidx]
    chats0 = _chats()
    blocks = judge.layers
    state = {"d": None}

    def hook(_m, _inp, out):
        if state["d"] is None:
            return
        h = out[0] if isinstance(out, tuple) else out
        h = h + state["d"].to(h.dtype)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    hd = blocks[layer].register_forward_hook(hook)

    @torch.no_grad()
    def _gen(prompts, chunk=8):
        outs = []
        tok.padding_side = "left"
        for i in range(0, len(prompts), chunk):
            enc = tok(prompts[i:i + chunk], return_tensors="pt", padding=True, truncation=True,
                      max_length=3584, add_special_tokens=False).to(dev)
            out = judge.model.generate(**enc, do_sample=args.gen_temp > 0,
                                       temperature=max(args.gen_temp, 1e-3), top_p=0.95,
                                       max_new_tokens=args.gen_max_new, pad_token_id=tok.pad_token_id)
            outs += [tok.decode(g, skip_special_tokens=True) for g in out[:, enc["input_ids"].shape[1]:]]
        tok.padding_side = "right"
        return outs

    def _gen_steered(vec, chats=None):
        state["d"] = None if vec is None else torch.tensor(vec, dtype=torch.float32, device=dev)
        outs = _gen(chats or chats0)
        state["d"] = None                                        # judging must run unsteered
        return outs

    def _fmt(outs):
        return [f"User asked: {prm[i][:args.max_prompt_chars]}\n\nResponse: {o[:args.max_text_chars]}"
                for i, o in zip(pidx, outs)]

    base_out = _gen_steered(None)
    base_len = np.mean([len(o) for o in base_out]) + 1e-9
    Pb = judge.presence_grid(names, _fmt(base_out), text_batch=32, mode="scale")
    res = {"names": names, "dirs": args.dirs, "checkpoint": args.checkpoint}

    def _table(gen_fn, tag):
        """K x K effect table for one set of interventions; gen_fn(j) -> steered outputs.
        Row j holds the change in every entry's concept score when entry j is steered."""
        dY = np.zeros((Kk, Kk, len(pidx))); len_ratio = np.ones(Kk)
        for j in range(Kk):
            so = gen_fn(j)
            len_ratio[j] = np.mean([len(o) for o in so]) / base_len
            dY[j] = judge.presence_grid(names, _fmt(so), text_batch=32, mode="scale") - Pb
        aie = np.abs(dY.mean(2))
        sel = np.array([aie[j, j] / max(np.delete(aie[j], j).max(initial=0.0), args.sel_floor)
                        if Kk > 1 else np.nan for j in range(Kk)])   # K=1: no cross-effect, undefined
        print(f"\n[cdiag] {tag}  [own-AIE / selectivity / len-ratio]:", flush=True)
        for j in range(Kk):
            print(f"    AIE={aie[j, j]:.3f}  sel={sel[j]:.2f}  len={len_ratio[j]:.2f}  {names[j][:70]}",
                  flush=True)
        print(f"[cdiag]   {tag}: mean own-AIE={aie.diagonal().mean():.3f}  "
              f"median sel={np.median(sel):.2f}", flush=True)
        return {"aie": aie, "sel": sel, "len_ratio": len_ratio, "dY": dY.astype(np.float32)}

    # ---- sweep: the full effect table at every steering strength ----
    if "sweep" in run:
        spread = np.array([float(np.std((Hr - mu_r) @ Hd[j])) or 1.0 for j in range(Kk)])
        alphas = sorted(float(a) for a in args.alphas.split(","))
        res["sweep"] = {}
        for a in alphas:
            res["sweep"][a] = _table(lambda j, a=a: _gen_steered(a * spread[j] * Hd[j]),
                                     f"SWEEP alpha={a:+g}")
        print("\n[cdiag] SWEEP summary (mean own-AIE / median sel / min len-ratio):", flush=True)
        for a in alphas:
            r_ = res["sweep"][a]
            print(f"    a{a:+g}: {r_['aie'].diagonal().mean():.3f} / {np.median(r_['sel']):.2f} / "
                  f"{r_['len_ratio'].min():.2f}", flush=True)
        # Per-entry steering strength: entries differ in the strength they need and where the
        # text degenerates. Guards: length ratio in [0.85, --len-hi] (degenerate text inflates
        # the target effect) and target effect >= 0.05 (below that, selectivity is a ratio of
        # noise). The oracle choice maximizes selectivity on all prompts; the honest choice
        # selects the strength on even prompts and reports selectivity on odd prompts.
        npx = res["sweep"][alphas[0]]["dY"].shape[2]
        ev, od = np.arange(0, npx, 2), np.arange(1, npx, 2)

        def _sel_half(a, j, half):
            aie_h = np.abs(res["sweep"][a]["dY"][j][:, half].mean(1))
            off = np.delete(aie_h, j).max(initial=0.0)
            return (float(aie_h[j] / max(off, args.sel_floor)) if Kk > 1 else float("nan")), float(aie_h[j])
        print(f"\n[cdiag] PER-ATOM BEST ALPHA (guards: 0.85<=len<={args.len_hi:g}, own-AIE>=0.05)  "
              "[oracle | honest=select-even/score-odd]:", flush=True)
        o_sel, h_sel = [], []
        for j in range(Kk):
            cand = [a for a in alphas if 0.85 <= res["sweep"][a]["len_ratio"][j] <= args.len_hi
                    and res["sweep"][a]["aie"][j, j] >= 0.05]
            if not cand:
                print(f"    atom {j}: no alpha passes guards   {names[j][:56]}", flush=True)
                o_sel.append(0.0); h_sel.append(0.0)
                continue
            ao = max(cand, key=lambda a: res["sweep"][a]["sel"][j])
            ah = max(cand, key=lambda a: _sel_half(a, j, ev)[0] if _sel_half(a, j, ev)[1] >= 0.05
                     else -1e9)
            so, ho = float(res["sweep"][ao]["sel"][j]), _sel_half(ah, j, od)
            o_sel.append(so); h_sel.append(ho[0])
            print(f"    a{ao:+g} sel={so:.2f} AIE={res['sweep'][ao]['aie'][j, j]:.3f} | "
                  f"a{ah:+g} sel={ho[0]:.2f} AIE={ho[1]:.3f}   {names[j][:56]}", flush=True)
        # Steerability (entries passing the guards at some strength) is reported separately
        # from the selectivity of the steerable entries, since guard failures count as 0.0.
        n_pass = sum(1 for v in h_sel if v > 0)
        h_pass = [v for v in h_sel if v > 0] or [0.0]
        o_pass = [v for v in o_sel if v > 0] or [0.0]
        print(f"[cdiag]   PER-ATOM-ALPHA: steerable {n_pass}/{Kk}  "
              f"oracle median sel={np.median(o_sel):.2f} (passing-only {np.median(o_pass):.2f})  "
              f"HONEST median sel={np.median(h_sel):.2f} (passing-only {np.median(h_pass):.2f})",
              flush=True)
        res["sweep_best"] = {"oracle_sel": np.array(o_sel), "honest_sel": np.array(h_sel),
                             "n_steerable": n_pass}

    # ---- distill: direction = mean activation of description-prompted responses minus
    # mean activation of unprompted responses, both read on the plain (no system) chats ----
    if "distill" in run:
        cap = {}

        def cap_hook(_m, _i, out):
            cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()
        hc = blocks[layer].register_forward_hook(cap_hook)

        @torch.no_grad()
        def _acts(chats, outs):                                  # mean over generated tokens
            H = []
            tok.padding_side = "right"
            for i in range(0, len(outs), 8):
                full = [c + o for c, o in zip(chats[i:i + 8], outs[i:i + 8])]
                plen = [len(tok(c, add_special_tokens=False).input_ids) for c in chats[i:i + 8]]
                enc = tok(full, return_tensors="pt", padding=True, truncation=True, max_length=3584,
                          add_special_tokens=False).to(dev)
                judge.model(**enc)
                h = cap["h"].float()
                for r in range(h.shape[0]):
                    e = int(enc["attention_mask"][r].sum())
                    H.append(h[r, min(plen[r], e - 1):e].mean(0).cpu().numpy())
            return np.array(H)
        Hb = _acts(chats0, base_out)
        Hdist = np.zeros((Kk, Hb.shape[1]))
        for j in range(Kk):
            po = _gen_steered(None, chats=_chats(
                f"In your reply, strongly and consistently exhibit this behavior: you tend to "
                f"{names[j]}."))
            v = _acts(_chats(), po).mean(0) - Hb.mean(0)
            Hdist[j] = v / (np.linalg.norm(v) + 1e-9)
        hc.remove()
        cosd = (Hdist * Hd).sum(1) if Hd is not None else np.zeros(Kk)
        if Hd is not None:
            print("\n[cdiag] distill: cos(v_distill, v_ckpt) per atom = " +
                  " ".join(f"{c:+.2f}" for c in cosd), flush=True)
        spread_d = np.array([float(np.std((Hr - mu_r) @ Hdist[j])) or 1.0 for j in range(Kk)])
        res["distill"] = _table(lambda j: _gen_steered(args.alpha * spread_d[j] * Hdist[j]),
                                f"DISTILLED prompted-rollout dirs at alpha={args.alpha:g}")
        res["distill"]["dirs"] = Hdist; res["distill"]["cos_to_ckpt"] = cosd

    hd.remove()
    torch.save(res, args.out)
    print(f"\n[cdiag] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
