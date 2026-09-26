#!/usr/bin/env python
"""Generate a DOMAIN concept bank with an API model (DeepSeek V4 Flash by default).

The sycophancy bank is clustered from the extraction cache; deception/emotion have no
cache, and the built-in fallback list is sycophancy-flavored. This script generates a
~1024-phrase candidate bank for a domain: verb-first, one-line behavioral phrases that
complete "the assistant tends to ___" (the exact shape the presence judge and bank_topk
expect). Each request is grounded with a fresh sample of REAL responses from the domain's
prompt-acts cache so phrases describe behaviors that actually occur, and generation rounds
repeat with deduplication until --size unique phrases exist. Output: one phrase per line;
consumed everywhere via BANK_FILE=<path> (concept_bank.build_bank priority).

  PYTHONPATH=src python scripts/gen_bank.py \
    --prompt-acts $U/prompt_last_rpdec_l20.npz \
    --domain "deception in roleplay dialogue: an assistant playing a character who may \
mislead, misdirect, omit, or fabricate" \
    --out $U/bank_deception.txt
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from taxonomy_discovery.utils.api_presence import _load_key  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-acts", required=True,
                    help="domain npz (response_text used to ground generation)")
    ap.add_argument("--domain", required=True, help="one-sentence domain description")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--per-call", type=int, default=40)
    ap.add_argument("--examples-per-call", type=int, default=6)
    ap.add_argument("--max-rounds", type=int, default=60)
    ap.add_argument("--api-model", default="deepseek-v4-flash")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--key-var", default="DEEPSEEK_API_KEY")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="large: reasoning models spend budget thinking before the answer")
    ap.add_argument("--max-text-chars", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from openai import OpenAI
    client = OpenAI(api_key=_load_key(var=args.key_var), base_url=args.base_url)
    rng = np.random.default_rng(args.seed)
    resp = [str(x) for x in np.load(args.prompt_acts, allow_pickle=True)["response_text"]]
    print(f"[genbank] {len(resp)} responses for grounding; target {args.size} phrases "
          f"via {args.api_model}", flush=True)

    bank, seen = [], set()
    for rnd in range(args.max_rounds):
        ex = "\n\n".join(f"RESPONSE {i + 1}:\n{resp[j][:args.max_text_chars]}"
                         for i, j in enumerate(rng.integers(0, len(resp),
                                                            args.examples_per_call)))
        avoid = "; ".join(rng.permutation(bank)[:15]) if bank else "(none yet)"
        prompt = (
            f"Domain: {args.domain}\n\n"
            f"Here are real responses from this domain:\n\n{ex}\n\n"
            f"Generate {args.per_call} DISTINCT candidate behavior concepts for a taxonomy "
            "of how responses in this domain can behave. Each concept must:\n"
            "- be ONE line, verb-first, completing the sentence 'the assistant tends to ...'\n"
            "- describe HOW a response behaves (style, rhetorical move, stance, framing), "
            "not its topic\n"
            "- be concrete and judgeable on a single response\n"
            "- cover a SPREAD: common and rare, benign and problematic, coarse and fine\n"
            f"Do not repeat these existing concepts: {avoid}\n\n"
            "Output ONLY the concepts, one per line, no numbering, no commentary.")
        kw = dict(model=args.api_model, temperature=args.temperature,
                  max_tokens=args.max_tokens,
                  messages=[{"role": "user", "content": prompt}])
        if not getattr(main, "_no_disable", False):
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            r = client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            if "thinking" in str(e).lower() and "extra_body" in kw:
                main._no_disable = True                            # endpoint rejects the knob
                kw.pop("extra_body")
                try:
                    r = client.chat.completions.create(**kw)
                except Exception as e2:  # noqa: BLE001
                    print(f"[genbank] round {rnd} failed ({e2}), continuing", flush=True)
                    continue
            else:
                print(f"[genbank] round {rnd} failed ({type(e).__name__}: {e}), continuing",
                      flush=True)
                continue
        msg = r.choices[0].message
        txt = (msg.content or "") or (getattr(msg, "reasoning_content", None) or "")
        added = 0
        for ln in txt.splitlines():
            p = re.sub(r"^[\s\-\*\d\.\)]+", "", ln).strip().rstrip(".")
            p = re.sub(r"^(the assistant tends to|tends to)\s+", "", p, flags=re.I)
            key = " ".join(p.lower().split())
            if 15 <= len(p) <= 160 and key not in seen:
                seen.add(key); bank.append(p); added += 1
        if added < 3:                                              # make silent failures visible
            print(f"[genbank]   low yield (finish={r.choices[0].finish_reason}, "
                  f"content_len={len(msg.content or '')}): "
                  f"{(txt or '(empty)')[:200]!r}", flush=True)
        print(f"[genbank] round {rnd + 1}: {len(bank)}/{args.size}", flush=True)
        if len(bank) >= args.size:
            break
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(f"# domain bank ({args.api_model}): {args.domain}\n")
        for p in bank[:args.size]:
            f.write(p + "\n")
    print(f"[genbank] wrote {min(len(bank), args.size)} phrases -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
