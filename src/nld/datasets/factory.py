from __future__ import annotations

from nld.datasets.deception.roleplay import DeceptionRoleplayDataset
from nld.datasets.emotion.emobank import EmoBankDataset
from nld.datasets.sycophancy.oeq import OEQDataset

DATASET_REGISTRY = {
    "oeq": OEQDataset,                                 # social sycophancy
    "deception_roleplay": DeceptionRoleplayDataset,    # deception
    "emobank": EmoBankDataset,                         # positive emotion
}


def build_dataset(dataset_name: str, behavior_family: str, data_dir: str):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name](
        dataset_name=dataset_name,
        behavior_family=behavior_family,
        data_dir=data_dir,
    )
