"""Roleplay deception dataset: scenarios with PAIRED honest vs deceptive completions.

dataset.yaml items: {scenario, question, answer_prefix, honest_completion, deceptive_completion}
(371 scenarios -> 742 examples). Unlike DIFrauD's found texts, these are response-shaped
completions to a real prompt, so the pipeline's "assistant response" framing fits natively.

The pair shares one prompt (scenario + question + speaker tag), so the prompt-ridge
residualization removes scenario/topic EXACTLY (perfectly paired contrastive design -- the
OEQ setup with a guaranteed on-topic counterfactual). Completion lengths are balanced
(honest median 283 chars vs deceptive 272), so the length deconfound has little to do.

label: 1 = deceptive completion, 0 = honest completion.
data_dir: path to dataset.yaml (defaults to the file next to this module).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from taxonomy_discovery.core.types import BehaviorExample
from taxonomy_discovery.datasets.base import BaseDataset


class DeceptionRoleplayDataset(BaseDataset):
    def load(self, split: str) -> list[BehaviorExample]:
        path = Path(self.data_dir or Path(__file__).parent / "dataset.yaml")
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
