"""Belief storage, exact updates, planning utilities, and Knowledge Base bridge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Hashable, Mapping

from ..knowledge_base.models import FactStatus, Pattern
from .bayes import (
    ImpossibleEvidenceError,
    bayesian_update,
    expected_utility as calculate_expected_utility,
    information_gain_bits,
)
from .calibration import CalibrationTracker
from .models import (
    BeliefReport,
    CalibrationReport,
    CategoricalBelief,
    EvidenceRecord,
    ProbabilityDistribution,
)
from .particle_filter import ParticleFilter, ParticleFilterReport


class KnowledgeImportStatus(str, Enum):
    """Outcome of consulting deterministic evidence in the Knowledge Base."""

    IMPORTED = "imported"
    UNKNOWN = "unknown"
    CONFLICTED = "conflicted"
    UNCONFIRMED = "unconfirmed"
    INCONSISTENT = "inconsistent"


@dataclass(frozen=True, slots=True)
class KnowledgeImportResult:
    """Result of translating a confirmed Knowledge Base query into evidence."""

    status: KnowledgeImportStatus
    report: BeliefReport | None = None
    fact_ids: tuple[str, ...] = ()
    message: str = ""


class UncertaintyModule:
    """Maintain inspectable probabilistic beliefs separately from KB facts.

    Categorical variables use exact Bayesian updates. Continuous or dependent
    state can use an attached :class:`ParticleFilter`. This module never
    rewrites Knowledge Base facts: confirmed facts can condition a belief, while
    proposed, missing, or conflicted facts remain reviewable evidence states.
    """

    def __init__(self) -> None:
        self._beliefs: dict[str, CategoricalBelief] = {}
        self._particle_filters: dict[str, ParticleFilter[Hashable]] = {}
        self._calibration: dict[str, CalibrationTracker] = {}

    @property
    def beliefs(self) -> Mapping[str, CategoricalBelief]:
        """Read-only mapping of the current categorical belief states."""

        return MappingProxyType(dict(self._beliefs))

    def set_prior(
        self,
        variable: str,
        probabilities: ProbabilityDistribution | Mapping[Hashable, float],
    ) -> BeliefReport:
        """Create the initial categorical prior for a variable."""

        if variable in self._beliefs:
            raise ValueError(f"A belief already exists for {variable!r}; update it with evidence")
        distribution = (
            probabilities
            if isinstance(probabilities, ProbabilityDistribution)
            else ProbabilityDistribution(probabilities)
        )
        self._beliefs[variable] = CategoricalBelief(variable, distribution, distribution)
        return self.report(variable)

    def observe_classifier_prediction(
        self,
        variable: str,
        probabilities: ProbabilityDistribution | Mapping[Hashable, float],
        *,
        source: str = "classifier",
        metadata: Mapping[str, object] | None = None,
    ) -> BeliefReport:
        """Store a classifier's full output distribution as its current belief.

        The classifier distribution replaces the current posterior because it
        is already a model output. It is not multiplied by the prior as if it
        were an independent likelihood, which could double-count evidence.
        Pass every class probability, not only the winning class confidence.
        """

        distribution = (
            probabilities
            if isinstance(probabilities, ProbabilityDistribution)
            else ProbabilityDistribution(probabilities)
        )
        current = self._beliefs.get(variable)
        if current is not None and set(current.posterior.probabilities) != set(
            distribution.probabilities
        ):
            raise ValueError("Classifier and existing belief must use the same outcome space")
        prior = current.prior if current is not None else distribution
        information_gain = (
            information_gain_bits(distribution, current.posterior) if current is not None else 0.0
        )
        record = EvidenceRecord(
            kind="classifier_prediction",
            source=source,
            observation=distribution.most_likely,
            information_gain_bits=information_gain,
            metadata={
                **dict(metadata or {}),
                "full_distribution": dict(distribution.probabilities),
            },
        )
        evidence = current.evidence + (record,) if current is not None else (record,)
        self._beliefs[variable] = CategoricalBelief(variable, prior, distribution, evidence)
        return self.report(variable)

    def update_categorical(
        self,
        variable: str,
        observation: object,
        likelihoods: Mapping[Hashable, float],
        *,
        source: str,
        kind: str = "categorical_observation",
        fact_ids: tuple[str, ...] = (),
        metadata: Mapping[str, object] | None = None,
    ) -> BeliefReport:
        """Apply exact Bayes updating from ``P(observation | state)``."""

        if variable not in self._beliefs:
            raise KeyError(f"Set a prior before updating {variable!r}")
        current = self._beliefs[variable]
        posterior, evidence_probability = bayesian_update(current.posterior, likelihoods)
        record = EvidenceRecord(
            kind=kind,
            source=source,
            observation=observation,
            evidence_probability=evidence_probability,
            log_evidence=math.log(evidence_probability),
            information_gain_bits=information_gain_bits(posterior, current.posterior),
            fact_ids=fact_ids,
            metadata=metadata or {},
        )
        self._beliefs[variable] = CategoricalBelief(
            variable,
            current.prior,
            posterior,
            current.evidence + (record,),
        )
        return self.report(variable)

    def import_knowledge_base_evidence(
        self,
        knowledge_base: object,
        variable: str,
        pattern: Pattern,
        *,
        true_value: Hashable = True,
        false_value: Hashable = False,
        source: str = "knowledge_base",
    ) -> KnowledgeImportResult:
        """Condition a belief on a confirmed, non-conflicted KB fact.

        Unknown or proposed-only facts are not converted into certainty. A KB
        conflict blocks the import until the facts are reviewed. If no belief
        exists yet, a uniform Boolean prior is used for the supplied values.
        """

        if true_value == false_value:
            raise ValueError("true_value and false_value must be distinct")
        query = getattr(knowledge_base, "query", None)
        if not callable(query):
            raise TypeError("knowledge_base must provide query(pattern)")
        result = query(pattern)
        status = getattr(getattr(result, "status", None), "value", getattr(result, "status", None))
        if status == "conflicted":
            return KnowledgeImportResult(
                KnowledgeImportStatus.CONFLICTED,
                message="Conflicting positive and negative KB evidence must be reviewed first.",
            )
        if status == "unknown":
            return KnowledgeImportResult(
                KnowledgeImportStatus.UNKNOWN,
                message="The Knowledge Base has no active evidence for this query.",
            )

        supports = result.matches if status == "entailed" else result.counter_matches
        confirmed = tuple(
            fact
            for fact in supports
            if getattr(getattr(fact, "status", None), "value", getattr(fact, "status", None))
            == FactStatus.CONFIRMED.value
        )
        if not confirmed:
            return KnowledgeImportResult(
                KnowledgeImportStatus.UNCONFIRMED,
                message="Only proposed or otherwise non-confirmed facts support this result.",
            )

        asserted_value = true_value if status == "entailed" else false_value
        if variable not in self._beliefs:
            self.set_prior(variable, {true_value: 0.5, false_value: 0.5})
        belief = self._beliefs[variable]
        if set(belief.posterior.probabilities) != {true_value, false_value}:
            return KnowledgeImportResult(
                KnowledgeImportStatus.INCONSISTENT,
                message="The existing belief does not use the supplied true/false outcome values.",
            )

        fact_ids = tuple(fact.fact_id for fact in confirmed if fact.fact_id)
        fact_metadata = {
            "pattern": str(pattern),
            "fact_sources": tuple(fact.source for fact in confirmed),
            "fact_confidences": tuple(float(fact.confidence) for fact in confirmed),
            "fact_provenance": tuple(tuple(fact.provenance) for fact in confirmed),
        }
        likelihoods = {
            true_value: 1.0 if asserted_value == true_value else 0.0,
            false_value: 1.0 if asserted_value == false_value else 0.0,
        }
        try:
            report = self.update_categorical(
                variable,
                observation=f"confirmed KB evidence for {pattern}",
                likelihoods=likelihoods,
                source=source,
                kind="confirmed_kb_fact",
                fact_ids=fact_ids,
                metadata=fact_metadata,
            )
        except ImpossibleEvidenceError:
            return KnowledgeImportResult(
                KnowledgeImportStatus.INCONSISTENT,
                fact_ids=fact_ids,
                message=(
                    "The confirmed KB fact has zero probability under the current belief; "
                    "the evidence sources disagree."
                ),
            )
        return KnowledgeImportResult(
            KnowledgeImportStatus.IMPORTED,
            report=report,
            fact_ids=fact_ids,
            message="Confirmed KB evidence was recorded as a hard probabilistic constraint.",
        )

    def expected_utility(self, variable: str, utilities: Mapping[Hashable, float]) -> float:
        """Return expected utility under the variable's current posterior."""

        return calculate_expected_utility(self._beliefs[variable].posterior, utilities)

    def create_particle_filter(
        self,
        variable: str,
        particles: list[Hashable] | tuple[Hashable, ...],
        weights: list[float] | tuple[float, ...] | None = None,
        *,
        resample_threshold: float = 0.5,
        seed: int | None = None,
    ) -> ParticleFilter[Hashable]:
        """Attach a particle filter for a continuous or dependent variable."""

        if not variable or not variable.strip():
            raise ValueError("Variable name must not be empty")
        if variable in self._particle_filters:
            raise ValueError(f"A particle filter already exists for {variable!r}")
        particle_filter: ParticleFilter[Hashable] = ParticleFilter(
            particles,
            weights,
            resample_threshold=resample_threshold,
            seed=seed,
        )
        self._particle_filters[variable] = particle_filter
        return particle_filter

    def particle_filter(self, variable: str) -> ParticleFilter[Hashable]:
        """Return the attached particle filter for a variable."""

        return self._particle_filters[variable]

    def record_calibration_case(
        self,
        variable: str,
        probabilities: ProbabilityDistribution | Mapping[Hashable, float],
        actual_outcome: Hashable,
        *,
        bin_count: int = 10,
    ) -> CalibrationReport:
        """Record one classifier forecast after its actual class is known."""

        distribution = (
            probabilities
            if isinstance(probabilities, ProbabilityDistribution)
            else ProbabilityDistribution(probabilities)
        )
        tracker = self._calibration.get(variable)
        if tracker is None:
            tracker = self._calibration[variable] = CalibrationTracker(
                variable, bin_count=bin_count
            )
        elif tracker.bin_count != bin_count:
            raise ValueError("bin_count cannot change after calibration cases are recorded")
        tracker.record(distribution, actual_outcome)
        return tracker.report()

    def calibration_report(self, variable: str) -> CalibrationReport:
        """Return calibration metrics, or an empty report before labels arrive."""

        tracker = self._calibration.get(variable)
        if tracker is None:
            return CalibrationTracker(variable).report()
        return tracker.report()

    def report(self, variable: str) -> BeliefReport:
        """Return posterior probabilities, concentration, and evidence strength."""

        belief = self._beliefs[variable]
        posterior = belief.posterior
        entropy = posterior.entropy_bits
        outcome_count = len(posterior.probabilities)
        normalized_entropy = entropy / math.log2(outcome_count) if outcome_count > 1 else 0.0
        likely = posterior.most_likely
        return BeliefReport(
            variable=variable,
            posterior=posterior,
            most_likely=likely,
            most_likely_probability=posterior.probabilities[likely],
            entropy_bits=entropy,
            normalized_entropy=normalized_entropy,
            evidence_count=len(belief.evidence),
            information_gain_bits=sum(
                record.information_gain_bits for record in belief.evidence
            ),
            evidence=belief.evidence,
        )
