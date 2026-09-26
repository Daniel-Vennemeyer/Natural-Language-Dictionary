from __future__ import annotations

from taxonomy_discovery.datasets.sycophancy.oeq import OEQDataset
from taxonomy_discovery.datasets.sycophancy.aita import AITADataset
from taxonomy_discovery.datasets.sycophancy.ss import SSDataset

from taxonomy_discovery.datasets.deception.roleplay import DeceptionRoleplayDataset

from taxonomy_discovery.datasets.emotion.emobank import EmoBankDataset

DATASET_REGISTRY = {
    # social sycophancy: OEQ = discovery corpus, AITA/SS = transfer / eval sets
    "oeq": OEQDataset,
    "aita": AITADataset,
    "ss": SSDataset,

    # deception: paired honest/deceptive roleplay completions (discovery corpus)
    "deception_roleplay": DeceptionRoleplayDataset,

    # emotion: EmoBank valence tails
    "emobank": EmoBankDataset,
}


def build_dataset(dataset_name: str, behavior_family: str, data_dir: str):
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASET_REGISTRY[dataset_name](
        dataset_name=dataset_name,
        behavior_family=behavior_family,
        data_dir=data_dir,
    )
