from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BehaviorExample:
    """One prompt, optionally with a response, and its behavior label."""
    example_id: str
    behavior_family: str
    dataset_name: str
    split: str
    prompt: str
    response: str | None = None
    label: int | float | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
