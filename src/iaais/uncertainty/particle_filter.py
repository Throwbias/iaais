"""A small bootstrap particle filter for continuous or dependent states."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Generic, Hashable, Iterable, TypeVar

from .models import EvidenceRecord

State = TypeVar("State", bound=Hashable)
Observation = TypeVar("Observation")


@dataclass(frozen=True, slots=True)
class ParticleFilterReport(Generic[State]):
    """Posterior particle sample and scalar sensor summary."""

    particles: tuple[State, ...]
    weights: tuple[float, ...]
    mean: float | tuple[float, ...] | None
    variance: float | tuple[float, ...] | None
    effective_sample_size: float
    evidence_count: int
    last_log_evidence: float | None
    evidence: tuple[EvidenceRecord, ...]


class ParticleFilter(Generic[State]):
    """Bootstrap filter with likelihood weighting and systematic resampling.

    State values must be hashable. ``report`` computes moments for scalar
    numeric states and fixed-length numeric tuple states. Other state shapes
    remain available in ``particles`` with ``mean`` and ``variance`` set to
    ``None`` so a caller can apply a domain-specific summary.
    """

    def __init__(
        self,
        particles: Iterable[State],
        weights: Iterable[float] | None = None,
        *,
        resample_threshold: float = 0.5,
        seed: int | None = None,
    ) -> None:
        states = tuple(particles)
        if not states:
            raise ValueError("A particle filter requires at least one particle")
        if not 0.0 <= resample_threshold <= 1.0:
            raise ValueError("resample_threshold must be between 0 and 1")
        raw_weights = tuple(weights) if weights is not None else (1.0,) * len(states)
        if len(raw_weights) != len(states):
            raise ValueError("Particle and weight counts must match")
        checked: list[float] = []
        for weight in raw_weights:
            if isinstance(weight, bool):
                raise TypeError("Particle weights must be numeric, not boolean")
            number = float(weight)
            if not math.isfinite(number) or number < 0.0:
                raise ValueError("Particle weights must be finite and non-negative")
            checked.append(number)
        total = sum(checked)
        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Particle weights must have a positive finite total")
        self._particles = states
        self._weights = tuple(weight / total for weight in checked)
        self.resample_threshold = resample_threshold
        self._rng = random.Random(seed)
        self._evidence: list[EvidenceRecord] = []
        self._last_log_evidence: float | None = None

    @property
    def particles(self) -> tuple[State, ...]:
        return self._particles

    @property
    def weights(self) -> tuple[float, ...]:
        return self._weights

    @property
    def effective_sample_size(self) -> float:
        return 1.0 / sum(weight * weight for weight in self._weights)

    def advance(self, transition: Callable[[State], State]) -> None:
        """Propagate each particle through a transition callback."""

        self._particles = tuple(transition(state) for state in self._particles)

    def update(
        self,
        observation: Observation,
        likelihood: Callable[[State, Observation], float],
        *,
        source: str = "sensor",
        metadata: dict[str, object] | None = None,
    ) -> ParticleFilterReport[State]:
        """Weight particles by ``P(observation | state)`` and resample as needed."""

        prior_weights = self._weights
        log_weights: list[float] = []
        likelihood_values: list[float] = []
        for state, weight in zip(self._particles, prior_weights):
            raw_likelihood = likelihood(state, observation)
            if isinstance(raw_likelihood, bool):
                raise TypeError("Likelihood must be numeric, not boolean")
            value = float(raw_likelihood)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("Particle likelihoods must be finite and non-negative")
            likelihood_values.append(value)
            log_weights.append(
                math.log(weight) + math.log(value)
                if weight > 0.0 and value > 0.0
                else -math.inf
            )

        max_log_weight = max(log_weights)
        if max_log_weight == -math.inf:
            raise ValueError("Observation has zero likelihood for every particle")
        scaled = [
            math.exp(log_weight - max_log_weight) if log_weight > -math.inf else 0.0
            for log_weight in log_weights
        ]
        scaled_total = sum(scaled)
        posterior_weights = tuple(value / scaled_total for value in scaled)
        log_evidence = max_log_weight + math.log(scaled_total)

        gain = 0.0
        for prior, posterior in zip(prior_weights, posterior_weights):
            if posterior == 0.0:
                continue
            if prior == 0.0:
                gain = math.inf
                break
            gain += posterior * math.log2(posterior / prior)
        self._weights = posterior_weights
        self._last_log_evidence = log_evidence
        self._evidence.append(
            EvidenceRecord(
                kind="particle_observation",
                source=source,
                observation=observation,
                log_evidence=log_evidence,
                information_gain_bits=max(0.0, gain),
                metadata=metadata or {},
            )
        )
        if self.effective_sample_size < self.resample_threshold * len(self._particles):
            self.systematic_resample()
        return self.report()

    def systematic_resample(self) -> None:
        """Replace the weighted population with an equally weighted sample."""

        count = len(self._particles)
        step = 1.0 / count
        start = self._rng.random() * step
        cumulative: list[float] = []
        total = 0.0
        for weight in self._weights:
            total += weight
            cumulative.append(total)
        selected: list[State] = []
        index = 0
        for particle_index in range(count):
            position = start + particle_index * step
            while index < count - 1 and position > cumulative[index]:
                index += 1
            selected.append(self._particles[index])
        self._particles = tuple(selected)
        self._weights = (step,) * count

    def report(self) -> ParticleFilterReport[State]:
        """Return weighted mean, variance, effective sample size, and evidence."""

        mean: float | tuple[float, ...] | None = None
        variance: float | tuple[float, ...] | None = None
        if all(isinstance(state, (int, float)) and not isinstance(state, bool) for state in self._particles):
            numeric_particles = tuple(float(state) for state in self._particles)
            scalar_mean = sum(value * weight for value, weight in zip(numeric_particles, self._weights))
            scalar_variance = sum(
                weight * (value - scalar_mean) ** 2
                for value, weight in zip(numeric_particles, self._weights)
            )
            mean, variance = scalar_mean, scalar_variance
        elif all(
            isinstance(state, tuple)
            and all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in state)
            for state in self._particles
        ):
            dimensions = len(self._particles[0])
            if all(len(state) == dimensions for state in self._particles):
                values = tuple(tuple(float(value) for value in state) for state in self._particles)
                vector_mean = tuple(
                    sum(state[index] * weight for state, weight in zip(values, self._weights))
                    for index in range(dimensions)
                )
                vector_variance = tuple(
                    sum(
                        weight * (state[index] - vector_mean[index]) ** 2
                        for state, weight in zip(values, self._weights)
                    )
                    for index in range(dimensions)
                )
                mean, variance = vector_mean, vector_variance
        return ParticleFilterReport(
            particles=self._particles,
            weights=self._weights,
            mean=mean,
            variance=variance,
            effective_sample_size=self.effective_sample_size,
            evidence_count=len(self._evidence),
            last_log_evidence=self._last_log_evidence,
            evidence=tuple(self._evidence),
        )
