"""Roleplay deception dataset: scenarios with paired honest and deceptive completions.

The YAML file is a list of {scenario, question, answer_prefix, honest_completion,
deceptive_completion}. Both completions of a scenario share one prompt
(scenario + question + speaker tag), so the pair differs only in the response.
label: 1 = deceptive completion, 0 = honest completion.
data_dir: path to the YAML file, or a directory holding dataset.yaml.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from nld.core.types import BehaviorExample
from nld.datasets.base import BaseDataset


class DeceptionRoleplayDataset(BaseDataset):
    def load(self, split: str) -> list[BehaviorExample]:
        path = Path(self.data_dir)
        if path.is_dir():
            path = path / "dataset.yaml"
        items = yaml.safe_load(open(path, encoding="utf-8"))
        examples: list[BehaviorExample] = []
        for i, it in enumerate(items):
            scen = " ".join(str(it.get("scenario", "")).split())
            q = " ".join(str(it.get("question", "")).split())
            pref = " ".join(str(it.get("answer_prefix", "")).split())
            prompt = f"{scen}\n\n{q}\n{pref}".strip()
            for key, lab in (("deceptive_completion", 1), ("honest_completion", 0)):
                text = " ".join(str(it.get(key, "")).split()).strip().strip('"').strip()
                if len(text) < 8:
                    continue
                examples.append(BehaviorExample(
                    example_id=f"rp{i}_{'dec' if lab else 'hon'}",
                    behavior_family=self.behavior_family,
                    dataset_name=self.dataset_name,
                    split=split,
                    prompt=prompt,
                    response=text,
                    label=lab,
                    metadata={"answer_prefix": pref, "scenario_id": i},
                ))
        print(f"[roleplay-deception] {len(examples)} examples from {len(items)} paired scenarios",
              flush=True)
        return examples
