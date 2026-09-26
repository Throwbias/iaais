"""Immutable probability and reporting models for the IAAIS Uncertainty Module."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Hashable, Mapping


def _freeze_mapping(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True, slots=True)
class ProbabilityDistribution:
    """A normalized categorical distribution over hashable outcomes."""

    probabilities: Mapping[Hashable, float]

    def __post_init__(self) -> None:
        if not self.probabilities:
            raise ValueError("A probability distribution must contain at least one outcome")
        normalized: dict[Hashable, float] = {}
        for outcome, raw_probability in self.probabilities.items():
            if isinstance(raw_probability, bool):
                raise TypeError("Probabilities must be numeric, not boolean")
            probability = float(raw_probability)
            if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
                raise ValueError("Probabilities must be finite values between 0 and 1")
            normalized[outcome] = probability
        total = sum(normalized.values())
        if not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"Probabilities must sum to 1 (received {total:.12g})")
        object.__setattr__(self, "probabilities", _freeze_mapping(normalized))

    @classmethod
    def from_weights(cls, weights: Mapping[Hashable, float]) -> ProbabilityDistribution:
        """Normalize non-negative weights into a probability distribution."""

        if not weights:
            raise ValueError("At least one weight is required")
        checked: dict[Hashable, float] = {}
        for outcome, raw_weight in weights.items():
            if isinstance(raw_weight, bool):
                raise TypeError("Weights must be numeric, not boolean")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("Weights must be finite and non-negative")
            checked[outcome] = weight
        total = sum(checked.values())
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Weights must have a finite, positive total")
        return cls({outcome: weight / total for outcome, weight in checked.items()})

    @property
    def most_likely(self) -> Hashable:
        """Return the first outcome with the greatest probability."""

        return max(self.probabilities, key=self.probabilities.__getitem__)

    @property
    def entropy_bits(self) -> float:
        """Shannon entropy of the distribution, in bits."""

        return -sum(
            probability * math.log2(probability)
            for probability in self.probabilities.values()
            if probability > 0.0
        )


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Traceable evidence used to set or update a belief."""

    kind: str
    source: str
    observation: Any = None
    evidence_probability: float | None = None
    log_evidence: float | None = None
    information_gain_bits: float = 0.0
    fact_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind or not self.kind.strip():
            raise ValueError("Evidence kind must not be empty")
        if not self.source or not self.source.strip():
            raise ValueError("Evidence source must not be empty")
        if self.evidence_probability is not None and (
            not math.isfinite(float(self.evidence_probability)) or self.evidence_probability < 0.0
        ):
            raise ValueError("Evidence probability must be finite and non-negative")
        if not math.isfinite(float(self.information_gain_bits)) and not math.isinf(
            float(self.information_gain_bits)
        ):
            raise ValueError("Information gain must be a number")
        if self.information_gain_bits < 0.0:
            raise ValueError("Information gain cannot be negative")
        object.__setattr__(self, "fact_ids", tuple(self.fact_ids))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class CategoricalBelief:
    """Prior, current posterior, and evidence history for one variable."""

    variable: str
    prior: ProbabilityDistribution
    posterior: ProbabilityDistribution
    evidence: tuple[EvidenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.variable or not self.variable.strip():
            raise ValueError("Belief variable name must not be empty")
        if self.prior.probabilities.keys() != self.posterior.probabilities.keys():
            raise ValueError("Prior and posterior must use the same outcome space")
        object.__setattr__(self, "evidence", tuple(self.evidence))


@dataclass(frozen=True, slots=True)
class BeliefReport:
    """Human-readable posterior summary with an explicit evidence trail."""

    variable: str
    posterior: ProbabilityDistribution
    most_likely: Hashable
    most_likely_probability: float
    entropy_bits: float
    normalized_entropy: float
    evidence_count: int
    information_gain_bits: float
    evidence: tuple[EvidenceRecord, ...]
    calibration_note: str = (
        "Information gain measures posterior shift under the supplied model, not source reliability. "
        "A normalized posterior is not proof of calibration; compare predictions with labeled outcomes."
    )


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """Top-label confidence and accuracy summary for one probability bin."""

    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Empirical calibration metrics calculated from labeled predictions."""

    variable: str
    sample_count: int
    brier_score: float | None
    log_loss: float | None
    expected_calibration_error: float | None
    bins: tuple[CalibrationBin, ...]
    note: str = (
        "Metrics summarize recorded labeled cases; they do not establish future or clinical validity."
    )
