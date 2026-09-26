"""Feature, training, prediction, and evaluation models for classification."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping

from ..knowledge_base.models import Fact, FactStatus
from ..uncertainty.models import BeliefReport, ProbabilityDistribution

FeatureValue = float | int | str | bool


def validate_feature_mapping(features: Mapping[str, Any]) -> Mapping[str, FeatureValue]:
    """Validate and freeze a structured, scalar-valued feature vector."""

    if not features:
        raise ValueError("A feature vector must contain at least one feature")
    checked: dict[str, FeatureValue] = {}
    for name, value in features.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Feature names must be non-empty strings")
        if isinstance(value, bool):
            checked[name] = value
        elif isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"Numeric feature {name!r} must be finite")
            checked[name] = number
        elif isinstance(value, str) and value.strip():
            checked[name] = value
        else:
            raise TypeError(
                f"Feature {name!r} must be a finite number, boolean, or non-empty category"
            )
    return MappingProxyType(checked)


@dataclass(frozen=True, slots=True)
class FeatureFactSpec:
    """How to read one feature or a family of features from KB facts.

    For a fixed feature, set ``feature_name`` and point ``value_argument_index``
    at its value. For a family such as ``sensor_feature(window, name, value)``,
    set ``name_argument_index`` and leave ``feature_name`` unset.
    """

    predicate: str
    feature_name: str | None = None
    entity_argument_index: int = 0
    value_argument_index: int = 1
    name_argument_index: int | None = None
    accepted_statuses: tuple[FactStatus, ...] = (
        FactStatus.CONFIRMED,
        FactStatus.PROPOSED,
    )

    def __post_init__(self) -> None:
        if not self.predicate or not self.predicate.strip():
            raise ValueError("Feature predicate must not be empty")
        if (self.feature_name is None) == (self.name_argument_index is None):
            raise ValueError("Set exactly one of feature_name or name_argument_index")
        indices = (self.entity_argument_index, self.value_argument_index)
        if self.name_argument_index is not None:
            indices += (self.name_argument_index,)
        if any(not isinstance(index, int) or index < 0 for index in indices):
            raise ValueError("Feature argument indices must be non-negative integers")
        statuses = tuple(
            status if isinstance(status, FactStatus) else FactStatus(status)
            for status in self.accepted_statuses
        )
        if not statuses:
            raise ValueError("At least one Knowledge Base fact status must be accepted")
        object.__setattr__(self, "accepted_statuses", statuses)


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Features extracted for one sensor window or other domain entity."""

    entity_id: Hashable
    values: Mapping[str, FeatureValue]
    fact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", validate_feature_mapping(self.values))
        object.__setattr__(self, "fact_ids", tuple(self.fact_ids))


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One labeled feature vector; group_id identifies its source recording."""

    features: Mapping[str, FeatureValue]
    label: str
    group_id: str | None = None
    label_fact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("Training labels must be non-empty strings")
        if self.group_id is not None and (
            not isinstance(self.group_id, str) or not self.group_id.strip()
        ):
            raise ValueError("group_id must be a non-empty string when supplied")
        object.__setattr__(self, "features", validate_feature_mapping(self.features))


@dataclass(frozen=True, slots=True)
class FeatureContribution:
    """Linear model contribution to the selected class score."""

    feature_name: str
    contribution: float


@dataclass(frozen=True, slots=True)
class ClassifierPrediction:
    """One prediction with its full distribution and reviewable KB records."""

    entity_id: Hashable | None
    prediction_id: str | None
    label: str
    confidence: float
    probabilities: ProbabilityDistribution
    review_required: bool
    review_threshold: float
    model_name: str
    feature_contributions: tuple[FeatureContribution, ...] = ()
    feature_fact_ids: tuple[str, ...] = ()
    prediction_fact: Fact | None = None
    probability_facts: tuple[Fact, ...] = ()
    uncertainty_report: BeliefReport | None = None
    probability_note: str = (
        "These are model predict_proba estimates; empirical calibration must be checked on labeled holdout data."
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "feature_contributions", tuple(self.feature_contributions))
        object.__setattr__(self, "feature_fact_ids", tuple(self.feature_fact_ids))
        object.__setattr__(self, "probability_facts", tuple(self.probability_facts))


@dataclass(frozen=True, slots=True)
class ClassifierEvaluation:
    """Holdout metrics; confusion-matrix rows are true classes, columns predicted."""

    labels: tuple[str, ...]
    sample_count: int
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_class_precision: Mapping[str, float]
    per_class_recall: Mapping[str, float]
    confusion_matrix: tuple[tuple[int, ...], ...]
    primary_metric: str = "macro_f1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "per_class_precision", MappingProxyType(dict(self.per_class_precision)))
        object.__setattr__(self, "per_class_recall", MappingProxyType(dict(self.per_class_recall)))
        object.__setattr__(
            self,
            "confusion_matrix",
            tuple(tuple(int(value) for value in row) for row in self.confusion_matrix),
        )
