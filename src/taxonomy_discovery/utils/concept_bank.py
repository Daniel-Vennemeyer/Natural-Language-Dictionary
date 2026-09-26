"""The pipeline's candidate concept bank, in one place.

~1024 concept phrases clustered from the extraction cache (falling back to the built-in
sycophancy list when no cache exists). Two callers need it and must provably get the SAME bank,
or the comparison between them is not a comparison:

  bank_topk.py      -- no-method ablation: rank the bank against the behavior LABEL, take top-K
  dir_baseline.py   -- --cand-bank: snap each DIRECTION to its nearest bank concept

Kept as a function rather than a cached constant because the extraction-cache bank depends on the
examples passed in.
"""
from __future__ import annotations

from typing import Any


def build_bank(cfg: dict, examples: list, size: int) -> tuple[list[str], str]:
    """Return (concepts, source_description). Never raises: falls back to the built-in list.

    Priority: BANK_FILE env / cfg method.bank_file (one phrase per line -- e.g. a
    gen_bank.py domain bank) > extraction-cache clustering > built-in sycophancy list."""
    import os
    from taxonomy_discovery.pipelines.bottom_up.grounded_concept_ae import (
        GroundedConceptAEMethod,
    )
    from taxonomy_discovery.pipelines.bottom_up.language_sae import (
        DEFAULT_SYCOPHANCY_CONCEPTS,
    )
    bf = os.environ.get("BANK_FILE") or (cfg.get("method", {}) or {}).get("bank_file")
    if bf:
        phrases = [ln.strip() for ln in open(bf, encoding="utf-8")
                   if ln.strip() and not ln.startswith("#")]
        if phrases:
            return phrases[:size], f"bank file {bf} ({len(phrases)} phrases)"
        print(f"[bank] bank file {bf} empty -- falling through", flush=True)
    mcfg: dict[str, Any] = dict(cfg.get("method", {}) or {})
    mcfg.setdefault("vocab_size", size)
    h = GroundedConceptAEMethod(config={**mcfg, "verbose": False})
    h.verbose = False
    try:
        b = h._bank_from_extraction(examples, mcfg, size)
    except Exception as e:  # noqa: BLE001
        print(f"[bank] extraction-cache bank unavailable ({e})", flush=True)
        b = None
    if b:
        return list(b), "extraction cache"
    return list(DEFAULT_SYCOPHANCY_CONCEPTS), "built-in default list"
