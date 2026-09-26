from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from taxonomy_discovery.core.types import BehaviorExample
from taxonomy_discovery.datasets.base import BaseDataset


class OEQDataset(BaseDataset):
    """
    Loader for the OEQ sycophancy dataset.

    Expected CSV columns include:
      - prompt
      - human
      - source
      - emotional_validation_human
      - indirect_language_human
      - indirect_action_human
      - accept_framing_human

    Labeling rule:
      If ANY of the sycophancy-related category columns is 1, the example is
      labeled as sycophancy (label = 1). Otherwise label = 0.
    """

    SYCO_PHANCY_COLUMNS = (
        "emotional_validation_human",
        "indirect_language_human",
        "indirect_action_human",
        "accept_framing_human",
    )

    def load(self, split: str) -> list[BehaviorExample]:
        path = self._resolve_path(split)
        examples: list[BehaviorExample] = []

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                prompt = str(row.get("prompt", "")).strip()
                response = self._optional_str(row.get("human"))
                source = self._optional_str(row.get("source"))

                emotional_validation = self._coerce_binary(row.get("emotional_validation_human"))
                indirect_language = self._coerce_binary(row.get("indirect_language_human"))
                indirect_action = self._coerce_binary(row.get("indirect_action_human"))
                accept_framing = self._coerce_binary(row.get("accept_framing_human"))
                indirectness = int(indirect_language == 1 or indirect_action == 1)

                category_values = {
                    "emotional_validation_human": emotional_validation,
                    "indirect_language_human": indirect_language,
                    "indirect_action_human": indirect_action,
                    "accept_framing_human": accept_framing,
                    "indirectness_human": indirectness,
                }
                label = int(
                    emotional_validation == 1
                    or accept_framing == 1
                    or indirectness == 1
                )

                example_id = self._make_example_id(row, row_idx)

                metadata: dict[str, Any] = {
                    "source": source,
                    "category_values": category_values,
                    "task_type": "sycophancy_binary_classification",
                    "expert_labels": [
                        label_name
                        for label_name in (
                            "emotional_validation_human",
                            "accept_framing_human",
                            "indirectness_human",
                        )
                        if category_values.get(label_name, 0) == 1
                    ],
                }

                examples.append(
                    BehaviorExample(
                        example_id=example_id,
                        behavior_family=self.behavior_family,
                        dataset_name=self.dataset_name,
                        split=split,
                        prompt=prompt,
                        response=response,
                        label=label,
                        metadata=metadata,
                    )
                )

        return examples

    def _resolve_path(self, split: str) -> Path:
        base = Path(self.data_dir)

        direct_csv = base / f"{split}.csv"
        if direct_csv.exists():
            return direct_csv

        direct_jsonl = base / f"{split}.jsonl"
        if direct_jsonl.exists():
            raise ValueError(
                f"OEQDataset expected CSV input but found JSONL file at {direct_jsonl}."
            )

        if base.is_file() and base.suffix.lower() == ".csv":
            return base

        raise FileNotFoundError(
            f"Could not find OEQ CSV for split={split!r}. Looked for {direct_csv}."
        )

    def _make_example_id(self, row: dict[str, Any], row_idx: int) -> str:
        for key in ("id", "example_id", "uid"):
            if key in row and row[key] not in (None, ""):
                return str(row[key])
        return f"oeq_{row_idx}"

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text != "" else None

    def _coerce_binary(self, value: Any) -> int:
        if value is None:
            return 0

        text = str(value).strip().lower()
        if text in {"", "nan", "none", "null"}:
            return 0
        if text in {"1", "1.0", "true", "yes"}:
            return 1
        if text in {"0", "0.0", "false", "no"}:
            return 0

        try:
            numeric = float(text)
        except Exception:
            return 0
        return 1 if numeric >= 0.5 else 0