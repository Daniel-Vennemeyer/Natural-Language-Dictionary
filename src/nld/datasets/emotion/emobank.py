"""EmoBank positive-emotion dataset: high- vs low-valence sentences.

emobank.csv (EmoBank, Buechel and Hahn) holds about 10k sentences with Valence/Arousal/
Dominance ratings on a 1-5 scale. The classes are the valence tails: label 1 = V >= mean + 1 sd
(positive affect), label 0 = V <= mean - 1 sd (negative affect); sentences in between are
dropped. Mean and sd are computed over all usable sentences. The texts are not model
generations, so every example shares one constant prompt frame. The corpus's own split is
kept in metadata.

data_dir: the emobank.csv path, or a directory containing it.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from nld.core.types import BehaviorExample
from nld.datasets.base import BaseDataset

FRAME = "A short everyday text follows."


class EmoBankDataset(BaseDataset):
    def load(self, split: str) -> list[BehaviorExample]:
        ref = Path(self.data_dir)
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
            if lo < v < hi:                                      # mid-valence
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
