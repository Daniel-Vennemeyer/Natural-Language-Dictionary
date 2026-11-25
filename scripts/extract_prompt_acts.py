"""Extract residual-stream activations for the training prompts.

Each prompt is mean-pooled over its tokens at the config layer and saved with its
response text and behaviour label (keys prompt_acts, response_text, prompt_text, label).

  python scripts/extract_prompt_acts.py --config CONFIG.yaml --out prompts.npz
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
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config)); m = cfg["model"]
    ext = ActivationExtractor(
        ModelSpec(model_name=m["name"], device="cuda", dtype=m.get("dtype", "bfloat16"),
                  batch_size=args.batch, max_length=args.max_length),
        HookSpec(layer=int(m["layer"]), stream=m.get("stream", "resid_post"),
                 token_selector="all_tokens", normalize=False))
    ext._load_model_and_tokenizer(); tok = ext.tokenizer; mod = ext._resolve_hook_module()
    ds = build_dataset(dataset_name=cfg["behavior"]["train_dataset"], behavior_family=cfg["behavior"]["family"],
                       data_dir=cfg["behavior"]["train_data_dir"])
    examples = ds.load(split=cfg["behavior"].get("train_split", "train"))
    examples = [e for e in examples if (e.prompt and e.response is not None and e.label is not None)]

    acts, rtext, ptext, labels = [], [], [], []

    def batches(xs, n):
        for i in range(0, len(xs), n):
            yield xs[i:i + n]

    done = 0
    for batch in batches(examples, args.batch):
        texts = [str(e.prompt) for e in batch]                     # prompt only: what the response is about
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=args.max_length)
        enc = {k: v.to(ext.device) for k, v in enc.items()}
        captured: dict = {}
        h = mod.register_forward_hook(ext._make_hook_fn(captured))
        try:
            with torch.no_grad():
                ext.model(**enc)
        finally:
            h.remove()
        hidden = captured["hidden"]; mask = enc["attention_mask"].bool()
        for i, e in enumerate(batch):
            mk = mask[i]
            if int(mk.sum()) == 0:
                continue
            row = hidden[i][mk]                                    # real prompt-token activations
            acts.append(row.mean(0).float().cpu().numpy())
            rtext.append(e.response or ""); ptext.append(str(e.prompt)); labels.append(int(e.label))
        done += len(batch)
        if done % (args.batch * 20) == 0:
            print(f"[prompt] {done}/{len(examples)}", flush=True)

    np.savez(args.out, prompt_acts=np.stack(acts).astype(np.float32),
             response_text=np.array(rtext, dtype=object), prompt_text=np.array(ptext, dtype=object),
             label=np.array(labels, dtype=np.int64))
    print(f"[prompt] saved {args.out}  n={len(acts)}  (syco={int(np.sum(labels))} non={len(labels) - int(np.sum(labels))})",
          flush=True)


if __name__ == "__main__":
    main()
