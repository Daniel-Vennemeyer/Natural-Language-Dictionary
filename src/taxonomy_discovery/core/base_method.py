from __future__ import annotations

from abc import ABC, abstractmethod

from taxonomy_discovery.core.types import ActivationBatch, BasisResult, BehaviorExample


class BaseMethod(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def fit(
        self,
        examples: list[BehaviorExample],
        activations: ActivationBatch,
    ) -> BasisResult:
        raise NotImplementedError