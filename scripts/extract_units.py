"""Extract residual-stream activations for training responses and their sentences.

Each response is run through the subject model; activations at the config layer are
mean-pooled over the response's tokens (resp_acts, resp_text) and over each of its first
--max-sents sentences (sent_acts, sent_text, sent_resp = the parent response).

  python scripts/extract_units.py --config CONFIG.yaml --out units.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import yaml

from nld.activations.extractor import ActivationExtractor, HookSpec, ModelSpec
from nld.datasets.factory import build_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-sents", type=int, default=12, help="cap sentence-units per response")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=1024)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); m = cfg["model"]
    ext = ActivationExtractor(
        ModelSpec(model_name=m["name"], device="cuda", dtype=m.get("dtype", "bfloat16"),
                  batch_size=args.batch, max_length=args.max_length),
        HookSpec(layer=int(m["layer"]), stream=m.get("stream", "resid_post"),
                 token_selector="all_tokens", normalize=False))
    ext._load_model_and_tokenizer()
    tok = ext.tokenizer; mod = ext._resolve_hook_module()
    dsname = cfg["behavior"]["train_dataset"]
    data_dir = cfg["behavior"]["train_data_dir"]
    split = cfg["behavior"].get("train_split", "train")
    print(f"[units] dataset={dsname} split={split} data_dir={data_dir}", flush=True)
    ds = build_dataset(dataset_name=dsname, behavior_family=cfg["behavior"]["family"], data_dir=data_dir)
    examples = ds.load(split=split)

    R = {"acts": [], "text": []}
    S = {"acts": [], "text": [], "resp": []}

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
            # real tokens only: padding and zero-width special tokens have empty offsets
            real = [t for t in range(len(off)) if mask[i, t] and off[t][1] > off[t][0]]
            if not real:
                continue
            hid = hidden[i]
            rtext = ex.response or ex.prompt or ""             # sets without a response use the prompt
            R["acts"].append(hid[real].mean(0).float().cpu().numpy()); R["text"].append(rtext)
            for sent in ext._split_sentences(rtext)[:args.max_sents]:
                span = ext._find_char_span(full, sent)
                if span is None:
                    continue
                cs, ce = span
                spos = [t for t in real if off[t][0] < ce and off[t][1] > cs]
                if not spos:
                    continue
                S["acts"].append(hid[spos].mean(0).float().cpu().numpy()); S["text"].append(sent); S["resp"].append(rtext)
        done += len(batch)
        if done % (args.batch * 20) == 0:
            print(f"[units] {done}/{len(examples)} responses  (sent={len(S['acts'])})", flush=True)

    np.savez(args.out,
             resp_acts=np.stack(R["acts"]).astype(np.float32), resp_text=np.array(R["text"], dtype=object),
             sent_acts=np.stack(S["acts"]).astype(np.float32), sent_text=np.array(S["text"], dtype=object),
             sent_resp=np.array(S["resp"], dtype=object))
    print(f"[units] saved {args.out}  responses={len(R['acts'])} sentences={len(S['acts'])}", flush=True)


if __name__ == "__main__":
    main()
