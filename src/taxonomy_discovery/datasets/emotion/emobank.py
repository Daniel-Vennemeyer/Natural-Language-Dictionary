"""EmoBank emotion domain: high- vs low-VALENCE sentences (VAD-annotated corpus).

emobank.csv (JULIELab EmoBank; the Kaggle 'emobank' dataset redistributes the same corpus)
holds ~10k sentences with reader/writer-averaged Valence/Arousal/Dominance ratings on a
1-5 scale. The behavioral classes are the VALENCE TAILS: label 1 = V >= mean + 1 sd (high
valence / positive affect), label 0 = V <= mean - 1 sd (low valence / negative affect);
mid-valence sentences are dropped. Mean/sd are computed over the full usable corpus
(mu ~2.98, sd ~0.35 -> ~1.2k high / ~1.5k low).

These are FOUND texts, not model generations (the difraud pattern): the prompt is one
constant frame, so prompt-ridge residualization reduces to centering while keeping the
pipeline's prep identical across domains. The corpus's own train/dev/test split is kept
in metadata; the pipeline draws its own splits.

data_dir: the emobank.csv path (or a directory containing it; defaults to this module's
directory).
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from taxonomy_discovery.core.types import BehaviorExample
from taxonomy_discovery.datasets.base import BaseDataset

FRAME = "A short everyday text follows."


class EmoBankDataset(BaseDataset):
    def load(self, split: str) -> list[BehaviorExample]:
        ref = Path(self.data_dir or Path(__file__).parent)
        path = ref if ref.is_file() else ref / "emobank.csv"
        if not path.exists():
            raise FileNotFoundError(f"emobank.csv not found at {path}")

        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = [r for r in csv.DictReader(f)
                    if len(" ".join(str(r.get("text", "")).split())) >= 8]
        V = [float(r["V"]) for r in rows]
        mu, sd = statistics.mean(V), statistics.pstdev(V)
        hi, lo = mu + sd, mu - sd

        examples: list[BehaviorExample] = []
        for r, v in zip(rows, V):
            if lo < v < hi:                                      # mid-valence: dropped
                continue
            lab = 1 if v >= hi else 0
            examples.append(BehaviorExample(
                example_id=str(r["id"]),
                behavior_family=self.behavior_family,
                dataset_name=self.dataset_name,
                split=split,
                prompt=FRAME,
                response=" ".join(str(r["text"]).split()),
                label=lab,
                metadata={"V": v, "A": float(r["A"]), "D": float(r["D"]),
                          "orig_split": r.get("split")},
            ))
        npos = sum(1 for e in examples if e.label == 1)
        print(f"[emobank] {len(examples)} examples (V mean {mu:.3f} sd {sd:.3f}; "
              f"high>= {hi:.3f}: {npos}, low<= {lo:.3f}: {len(examples) - npos}; "
              f"{len(rows) - len(examples)} mid-valence dropped)", flush=True)
        return examples
