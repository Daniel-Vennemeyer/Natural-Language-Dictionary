from __future__ import annotations

from abc import ABC, abstractmethod

from taxonomy_discovery.core.types import BehaviorExample


class BaseEvaluator(ABC):
    @abstractmethod
    def score_batch(
        self,
        examples: list[BehaviorExample],
        outputs: list[str],
    ) -> list[dict[str, float]]:
        raise NotImplementedError