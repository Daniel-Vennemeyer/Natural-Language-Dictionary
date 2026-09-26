"""Extract L20 activations at THREE granularities with full hierarchy + texts, in one pass.

Produces a unified cache so the RL learner can independently pick a reconstruction grain and a
judgment grain (e.g. token-level recon + sentence-level judge). For each response we capture, via
the same L20 hook the pipeline uses:
  - response unit: mean-pooled activation + response text
  - sentence units: mean-pooled-over-span activation + sentence text + parent response text
  - token units:   per-token activation + the token's SENTENCE with that token marked "<<tok>> "
                   (for token-level judging) + parent sentence text + parent response text

Run:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python scripts/extract_units.py \
    --config configs/experiments/sycophancy_grounded_sparse_decomposition_l20_resp.yaml \
    --max-tokens 48 --max-sents 12 --batch 8 \
    --out /scratch/mech-taxonomy/units_l20.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "src"))

from taxonomy_discovery.activations.extractor import ActivationExtractor, HookSpec, ModelSpec
from taxonomy_discovery.datasets.factory import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=48, help="cap token-units per response (even subsample)")
    ap.add_argument("--max-sents", type=int, default=12, help="cap sentence-units per response")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--token-position", choices=["mean", "last"], default="mean",
                    help="resp_acts pooling: mean over response tokens (default) or the LAST real token "
                         "(the sycophancy probe site, AUC 0.82 vs response-mean 0.74)")
    ap.add_argument("--resp-only", action="store_true",
                    help="extract ONLY response-level acts (skip sentence/token units) -- for last-token analysis")
    ap.add_argument("--dataset", default=None, help="dataset name (default: config train_dataset; e.g. aita, ss)")
    ap.add_argument("--data-dir", default=None, help="data path for --dataset (default: config eval_data_dirs / train)")
    ap.add_argument("--split", default=None, help="dataset split to load (default: train_split, or 'test' for eval sets)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); m = cfg["model"]
    ext = ActivationExtractor(
        ModelSpec(model_name=m["name"], device="cuda", dtype=m.get("dtype", "bfloat16"),
                  batch_size=args.batch, max_length=args.max_length),
        HookSpec(layer=int(m["layer"]), stream=m.get("stream", "resid_post"),
                 token_selector="all_tokens", normalize=False))
    ext._load_model_and_tokenizer()
    tok = ext.tokenizer; mod = ext._resolve_hook_module()
    dsname = args.dataset or cfg["behavior"]["train_dataset"]
    if args.data_dir:
        data_dir = args.data_dir
    elif args.dataset and args.dataset != cfg["behavior"]["train_dataset"]:
        data_dir = cfg["behavior"].get("eval_data_dirs", {}).get(args.dataset)
        if not data_dir:
            raise SystemExit(f"no eval_data_dirs entry for {args.dataset!r}; pass --data-dir")
    else:
        data_dir = cfg["behavior"]["train_data_dir"]
    split = args.split or (cfg["behavior"].get("train_split", "train") if dsname == cfg["behavior"]["train_dataset"]
                           else "test")
    print(f"[units] dataset={dsname} split={split} data_dir={data_dir}", flush=True)
    ds = build_dataset(dataset_name=dsname, behavior_family=cfg["behavior"]["family"], data_dir=data_dir)
    examples = ds.load(split=split)

    R = {"acts": [], "text": []}
    S = {"acts": [], "text": [], "resp": []}
    T = {"acts": [], "marked": [], "sent": [], "resp": []}

    def batches(xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i + n]

    done = 0
    for batch in batches(examples, args.batch):
        texts = [ext._format_example_text(ex)["text"] for ex in batch]
        enc = tok(texts, return_tensors="pt", return_offsets_mapping=True, padding=True,
                  truncation=True, max_length=args.max_length)
        offs = enc.pop("offset_mapping")
        enc = {k: v.to(ext.device) for k, v in enc.items()}
        captured: dict = {}
        h = mod.register_forward_hook(ext._make_hook_fn(captured))
        try:
            with torch.no_grad():
                ext.model(**enc)
        finally:
            h.remove()
        hidden = captured["hidden"]                                # [B, Tk, d]
        mask = enc["attention_mask"].bool()
        for i, ex in enumerate(batch):
            full = texts[i]; off = offs[i].tolist()
            real = [t for t in range(len(off)) if mask[i, t] and off[t][1] > off[t][0]]
            if not real:
                continue
            hid = hidden[i]
            pooled = hid[real].mean(0) if args.token_position == "mean" else hid[real[-1]]   # mean-pool or LAST token
            R["acts"].append(pooled.float().cpu().numpy()); R["text"].append(ex.response or ex.prompt or "")
            if args.resp_only:                                 # last-token analysis needs only response-level acts
                continue
            rtext = ex.response or ex.prompt or ""             # SS-style sets have no response -> use the prompt/sentence
            keep = real
            if args.max_tokens and len(real) > args.max_tokens:    # even subsample token-units
                sel = np.linspace(0, len(real) - 1, args.max_tokens).round().astype(int)
                keep = [real[s] for s in sorted(set(sel.tolist()))]
            keep = set(keep)
            for sent in ext._split_sentences(rtext)[:args.max_sents]:
                span = ext._find_char_span(full, sent)
                if span is None:
                    continue
                cs, ce = span
                spos = [t for t in real if off[t][0] < ce and off[t][1] > cs]
                if not spos:
                    continue
                S["acts"].append(hid[spos].mean(0).float().cpu().numpy()); S["text"].append(sent); S["resp"].append(rtext)
                for t in spos:
                    if t not in keep:
                        continue
                    cs_t, ce_t = off[t]
                    a = max(0, cs_t - cs); b = max(0, ce_t - cs)
                    marked = sent[:a] + "<<" + sent[a:b] + ">> " + sent[b:]
                    T["acts"].append(hid[t].float().cpu().numpy())
                    T["marked"].append(marked); T["sent"].append(sent); T["resp"].append(rtext)
        done += len(batch)
        if done % (args.batch * 20) == 0:
            print(f"[units] {done}/{len(examples)} responses  (sent={len(S['acts'])} tok={len(T['acts'])})", flush=True)

    if args.resp_only:
        np.savez(args.out, resp_acts=np.stack(R["acts"]).astype(np.float32),
                 resp_text=np.array(R["text"], dtype=object))
    else:
        np.savez(args.out,
                 resp_acts=np.stack(R["acts"]).astype(np.float32), resp_text=np.array(R["text"], dtype=object),
                 sent_acts=np.stack(S["acts"]).astype(np.float32), sent_text=np.array(S["text"], dtype=object),
                 sent_resp=np.array(S["resp"], dtype=object),
                 tok_acts=np.stack(T["acts"]).astype(np.float32), tok_marked=np.array(T["marked"], dtype=object),
                 tok_sent=np.array(T["sent"], dtype=object), tok_resp=np.array(T["resp"], dtype=object))
    print(f"[units] saved {args.out}  responses={len(R['acts'])} sentences={len(S['acts'])} tokens={len(T['acts'])} "
          f"(token-position={args.token_position}{', resp-only' if args.resp_only else ''})", flush=True)


if __name__ == "__main__":
    main()
