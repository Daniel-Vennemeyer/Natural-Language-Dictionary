"""Extract L20 mean-pooled PROMPT activations per OEQ example -- for TOPIC de-confounding.

The response activations encode prompt+response (relation); the prompt activations encode TOPIC alone.
Residualizing response-on-prompt strips topic, isolating response-specific (behavioral) signal. Saves
prompt_acts [N,d], response_text [N] (join key to units resp_text/sent_resp), label [N], prompt_text [N].

  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    python scripts/extract_prompt_acts.py --config configs/.../l20_resp.yaml \
      --out /scratch/mech-taxonomy/prompt_acts_l20.npz
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
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--token-position", choices=["mean", "last"], default="mean",
                    help="prompt pooling: mean over prompt tokens (default) or the LAST real token "
                         "(match the response --token-position used in extract_units)")
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
        texts = [str(e.prompt) for e in batch]                     # PROMPT only -> topic representation
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
            pooled = row.mean(0) if args.token_position == "mean" else row[-1]   # mean or LAST token
            acts.append(pooled.float().cpu().numpy())
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
