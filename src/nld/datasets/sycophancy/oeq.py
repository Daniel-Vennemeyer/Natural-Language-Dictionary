"""OEQ social-sycophancy dataset: open-ended advice questions with human-labeled responses.

CSV columns: prompt, human (the response), source, and the binary expert labels
emotional_validation_human, indirect_language_human, indirect_action_human,
accept_framing_human. label = 1 if the response shows emotional validation, accepts the
user's framing, or uses indirect language or indirect action; else 0.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from nld.core.types import BehaviorExample
from nld.datasets.base import BaseDataset


class OEQDataset(BaseDataset):
    def load(self, split: str) -> list[BehaviorExample]:
        path = self._resolve_path(split)
        examples: list[BehaviorExample] = []

        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                prompt = str(row.get("prompt", "")).strip()
                response = self._optional_str(row.get("human"))

                emotional_validation = self._coerce_binary(row.get("emotional_validation_human"))
                indirect_language = self._coerce_binary(row.get("indirect_language_human"))
                indirect_action = self._coerce_binary(row.get("indirect_action_human"))
                accept_framing = self._coerce_binary(row.get("accept_framing_human"))
                indirectness = int(indirect_language == 1 or indirect_action == 1)
                label = int(emotional_validation == 1 or accept_framing == 1 or indirectness == 1)

                examples.append(
                    BehaviorExample(
                        example_id=self._make_example_id(row, row_idx),
                        behavior_family=self.behavior_family,
                        dataset_name=self.dataset_name,
                        split=split,
                        prompt=prompt,
                        response=response,
                        label=label,
                        metadata={"source": self._optional_str(row.get("source"))},
                    )
                )

        return examples

    def _resolve_path(self, split: str) -> Path:
        """data_dir is the CSV itself, or a directory holding <split>.csv."""
        base = Path(self.data_dir)
        if base.is_file() and base.suffix.lower() == ".csv":
            return base
        direct_csv = base / f"{split}.csv"
        if direct_csv.exists():
            return direct_csv
        raise FileNotFoundError(f"Could not find OEQ CSV for split={split!r}. Looked for {direct_csv}.")

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
