from __future__ import annotations

from abc import ABC, abstractmethod

from taxonomy_discovery.core.types import BehaviorExample


class BaseDataset(ABC):
    def __init__(self, dataset_name: str, behavior_family: str, data_dir: str):
        self.dataset_name = dataset_name
        self.behavior_family = behavior_family
        self.data_dir = data_dir

    @abstractmethod
    def load(self, split: str) -> list[BehaviorExample]:
        raise NotImplementedError