from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from taxonomy_discovery.core.types import BehaviorExample
from taxonomy_discovery.datasets.base import BaseDataset


class AITADataset(BaseDataset):
    """
    Loader for the AITA social-sycophancy evaluation dataset.

    Expected CSV columns include:
      - prompt
      - top_comment
      - is_asshole
      - ytanta
      - validation_human
      - indirectness_human
      - framing_human

    Intended usage:
      - prompt -> BehaviorExample.prompt
      - top_comment -> gold/reference response in BehaviorExample.response
      - human rubric columns are stored in metadata for analysis
      - dataset is primarily for generation-time evaluation: feed the prompt,
        then score the generated response with the sycophancy evaluator.
    """

    HUMAN_METRIC_COLUMNS = (
        "validation_human",
        "indirectness_human",
        "framing_human",
    )

    def load(self, split: str) -> list[BehaviorExample]:
        path = self._resolve_path(split)
        examples: list[BehaviorExample] = []

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                prompt = str(row.get("prompt", "")).strip()
                reference_response = self._optional_str(row.get("top_comment"))

                human_metric_values = {
                    col: self._coerce_binary(row.get(col))
                    for col in self.HUMAN_METRIC_COLUMNS
                }

                # If any human metric is 1, treat the reference response as sycophantic.
                label = int(any(v == 1 for v in human_metric_values.values()))

                example_id = self._make_example_id(row, row_idx)

                metadata: dict[str, Any] = {
                    "reference_response": reference_response,
                    "is_asshole": self._optional_str(row.get("is_asshole")),
                    "ytanta": self._optional_str(row.get("ytanta")),
                    "human_metric_values": human_metric_values,
                    "task_type": "sycophancy_generation_eval",
                }

                examples.append(
                    BehaviorExample(
                        example_id=example_id,
                        behavior_family=self.behavior_family,
                        dataset_name=self.dataset_name,
                        split=split,
                        prompt=prompt,
                        response=reference_response,
                        label=label,
                        metadata=metadata,
                    )
                )

        return examples

    def _resolve_path(self, split: str) -> Path:
        base = Path(self.data_dir)

        # If the config already points directly to a CSV file, use it as-is.
        if base.suffix.lower() == ".csv":
            if base.exists():
                return base
            raise FileNotFoundError(
                f"AITA CSV file does not exist: {base}"
            )

        direct_csv = base / f"{split}.csv"
        if direct_csv.exists():
            return direct_csv

        raise FileNotFoundError(
            f"Could not find AITA CSV for split={split!r}. Looked for {direct_csv}."
        )

    def _make_example_id(self, row: dict[str, Any], row_idx: int) -> str:
        for key in ("id", "example_id", "uid"):
            if key in row and row[key] not in (None, ""):
                return str(row[key])
        return f"aita_{row_idx}"

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