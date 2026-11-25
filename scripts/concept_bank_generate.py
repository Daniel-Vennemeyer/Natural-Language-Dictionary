#!/usr/bin/env python
"""Generate a candidate concept bank for a domain with an API model.

Each round shows the model a fresh random sample of real responses from the domain and
asks for one-line, verb-first behavior concepts that complete "the assistant tends to
...". Rounds repeat, with deduplication, until --size unique concepts exist. The output
has one concept per line and is read by concept_bank_baseline.py through BANK_FILE=<path>.

  python scripts/concept_bank_generate.py --prompt-acts <prompt_acts.npz> \
    --domain "<one-sentence domain description>" --out <bank.txt>
"""
from __future__ import annotations

import argparse
import re

import numpy as np

from nld.utils.api_presence import _load_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-acts", required=True,
                    help="domain prompt-activation .npz; its response_text grounds generation")
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
                    help="kept large because reasoning models spend tokens before the answer")
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
    disable_thinking = True
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
        if disable_thinking:
            kw["extra_body"] = {"thinking": {"type": "disabled"}}
        try:
            r = client.chat.completions.create(**kw)
        except Exception as e:  # noqa: BLE001
            if "thinking" in str(e).lower() and "extra_body" in kw:
                disable_thinking = False                           # endpoint rejects the option
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
        if added < 3:                                              # surface empty or truncated replies
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
