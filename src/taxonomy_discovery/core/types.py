from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BehaviorExample:
    example_id: str
    behavior_family: str
    dataset_name: str
    split: str
    prompt: str
    response: str | None = None
    label: int | float | str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActivationBatch:
    example_ids: list[str]
    activations: Any  # typically np.ndarray or torch.Tensor with shape [n, d]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSet:
    candidates: Any  # shape [m, d]
    candidate_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BasisResult:
    method_name: str
    behavior_family: str
    train_dataset: str
    model_name: str
    layer: int
    directions: Any  # shape [k, d]
    direction_names: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fit_config: dict[str, Any] = field(default_factory=dict)
    artifact_paths: dict[str, str] = field(default_factory=dict)


@dataclass
class InterventionRecord:
    example_id: str
    alpha: float
    direction_idx: int | str | None
    output_text: str | None
    scores: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterventionResult:
    basis_id: str
    eval_dataset: str
    records: list[InterventionRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SummaryMetrics:
    run_id: str
    method_name: str
    behavior_family: str
    train_dataset: str
    eval_dataset: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentConfig:
    experiment_name: str
    model: dict[str, Any]
    behavior: dict[str, Any]
    method: dict[str, Any]
    intervention: dict[str, Any]
    metrics: dict[str, Any]
    output_root: str = "outputs/runs"
    activation_cache_dir: str = "outputs/caches/activations"
    seed: int = 0
    force_recompute_activations: bool = False

@dataclass
class ConditionSummary:
    direction_idx: int | None
    alpha: float
    n_examples: int
    target_mean: float
    cross_means: dict[str, float]
    deltas: dict[str, float]
