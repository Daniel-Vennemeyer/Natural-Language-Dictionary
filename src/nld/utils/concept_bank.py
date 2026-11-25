"""Candidate concept bank for the concept-bank baseline."""
from __future__ import annotations

import os

# Fallback bank used when no bank file is given.
DEFAULT_SYCOPHANCY_CONCEPTS: tuple[str, ...] = (
    "praise of the user's competence or intelligence",
    "emotional validation of the user's feelings",
    "agreeing with the user to avoid conflict",
    "excessive apologizing or self-deprecation",
    "flattering the user's question as insightful",
    "softening criticism with indirect language",
    "accepting the user's framing without pushback",
    "reassuring the user that they are right",
    "hedging to avoid stating an unwelcome truth",
    "deferring to the user's stated opinion",
    "complimenting the user's writing or work",
    "expressing enthusiasm to please the user",
    "avoiding direct disagreement with a claim",
    "telling the user what they want to hear",
    "minimizing the user's mistakes",
    "encouraging the user regardless of merit",
    "mirroring the user's emotional tone",
    "offering unconditional support",
    "qualifying honest feedback with reassurance",
    "validating a morally questionable choice",
    "agreeing with a factually incorrect statement",
    "prioritizing politeness over accuracy",
    "gratitude or thanks directed at the user",
    "deflecting blame away from the user",
    "framing the user's behavior charitably",
    "reluctance to deliver bad news",
    "downplaying risks the user faces",
    "affirming the user's self-image",
    "concession to social pressure in the prompt",
    "indirect refusal that still pleases the user",
    "endorsing the user's plan without caveats",
    "neutral factual explanation (non-sycophantic)",
)


def build_bank(cfg: dict, examples: list, size: int) -> tuple[list[str], str]:
    """Return (concepts, source_description).

    The bank is read from the file named by the BANK_FILE environment variable or by the
    config's method.bank_file (one description per line, '#' lines skipped; e.g. the output
    of concept_bank_generate.py), truncated to `size`. Without a bank file, or if the file is
    empty, the built-in sycophancy list is returned. `examples` is accepted for interface
    compatibility and not used."""
    bf = os.environ.get("BANK_FILE") or (cfg.get("method", {}) or {}).get("bank_file")
    if bf:
        phrases = [ln.strip() for ln in open(bf, encoding="utf-8")
                   if ln.strip() and not ln.startswith("#")]
        if phrases:
            return phrases[:size], f"bank file {bf} ({len(phrases)} phrases)"
        print(f"[bank] bank file {bf} is empty; using the built-in list", flush=True)
    return list(DEFAULT_SYCOPHANCY_CONCEPTS), "built-in default list"
