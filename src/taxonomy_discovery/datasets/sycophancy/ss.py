from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from taxonomy_discovery.core.types import BehaviorExample
from taxonomy_discovery.datasets.base import BaseDataset


class SSDataset(BaseDataset):
    """
    Loader for the SS sycophancy dataset.

    Expected CSV columns include:
      - sentence
      - self_attitude
      - other_attitude

    Labeling rule:
      Treat the example as sycophantic if either attitude field is present and
      non-empty. This keeps the loader aligned with the social-sycophancy setup:
      we feed the sentence as the prompt and evaluate the generated response with
      the sycophancy evaluator.

    Notes:
      - self_attitude / other_attitude are preserved in metadata.
      - prompt <- sentence
      - response <- None
      - label is a weak binary indicator derived from whether the row has any
        attitude annotation at all.
    """

    ATTITUDE_COLUMNS = (
        "self_attitude",
        "other_attitude",
    )

    def load(self, split: str) -> list[BehaviorExample]:
        path = self._resolve_path(split)
        examples: list[BehaviorExample] = []

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                prompt = str(row.get("sentence", "")).strip()

                attitude_values = {
                    col: self._optional_str(row.get(col))
                    for col in self.ATTITUDE_COLUMNS
                }

                # Weak binary label: if either attitude field is present, mark as 1.
                label = int(any(v is not None for v in attitude_values.values()))

                example_id = self._make_example_id(row, row_idx)

                metadata: dict[str, Any] = {
                    "self_attitude": attitude_values["self_attitude"],
                    "other_attitude": attitude_values["other_attitude"],
                    "task_type": "sycophancy_generation_eval",
                }

                examples.append(
                    BehaviorExample(
                        example_id=example_id,
                        behavior_family=self.behavior_family,
                        dataset_name=self.dataset_name,
                        split=split,
                        prompt=prompt,
                        response=None,
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

        if base.is_file() and base.suffix.lower() == ".csv":
            return base

        raise FileNotFoundError(
            f"Could not find SS CSV for split={split!r}. Looked for {direct_csv}."
        )

    def _make_example_id(self, row: dict[str, Any], row_idx: int) -> str:
        for key in ("id", "example_id", "uid"):
            if key in row and row[key] not in (None, ""):
                return str(row[key])
        return f"ss_{row_idx}"

    def _optional_str(self, value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text if text != "" else None