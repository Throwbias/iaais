"""Exact categorical Bayesian updating and probability propagation."""

from __future__ import annotations

import math
from typing import Hashable, Mapping

from .models import ProbabilityDistribution


class ImpossibleEvidenceError(ValueError):
    """Raised when the supplied model assigns zero probability to evidence."""


def bayesian_update(
    prior: ProbabilityDistribution | Mapping[Hashable, float],
    likelihoods: Mapping[Hashable, float],
) -> tuple[ProbabilityDistribution, float]:
    """Return ``P(state | evidence)`` and the predictive evidence probability.

    ``likelihoods[state]`` is ``P(evidence | state)``. The likelihood mapping
    must cover exactly the prior's outcomes, preventing silent omission of a
    hypothesis during normalization.
    """

    prior_distribution = (
        prior if isinstance(prior, ProbabilityDistribution) else ProbabilityDistribution(prior)
    )
    if set(likelihoods) != set(prior_distribution.probabilities):
        raise ValueError("Likelihoods must have exactly the same outcomes as the prior")

    weighted: dict[Hashable, float] = {}
    for outcome, raw_likelihood in likelihoods.items():
        if isinstance(raw_likelihood, bool):
            raise TypeError("Likelihoods must be numeric, not boolean")
        likelihood = float(raw_likelihood)
        if not math.isfinite(likelihood) or not 0.0 <= likelihood <= 1.0:
            raise ValueError("Categorical likelihoods must be finite values between 0 and 1")
        weighted[outcome] = prior_distribution.probabilities[outcome] * likelihood

    evidence_probability = sum(weighted.values())
    if evidence_probability <= 0.0 or not math.isfinite(evidence_probability):
        raise ImpossibleEvidenceError("Evidence has zero probability under the current belief")
    posterior = ProbabilityDistribution(
        {outcome: value / evidence_probability for outcome, value in weighted.items()}
    )
    return posterior, evidence_probability


def information_gain_bits(
    posterior: ProbabilityDistribution,
    prior: ProbabilityDistribution,
) -> float:
    """Calculate ``KL(posterior || prior)`` in bits."""

    if set(posterior.probabilities) != set(prior.probabilities):
        raise ValueError("Prior and posterior must use the same outcome space")
    terms: list[float] = []
    for outcome, post_probability in posterior.probabilities.items():
        prior_probability = prior.probabilities[outcome]
        if post_probability == 0.0:
            continue
        if prior_probability == 0.0:
            return math.inf
        terms.append(post_probability * math.log2(post_probability / prior_probability))
    return max(0.0, sum(terms))


def expected_utility(
    distribution: ProbabilityDistribution | Mapping[Hashable, float],
    utilities: Mapping[Hashable, float],
) -> float:
    """Weight each outcome's utility by its probability and return the mean."""

    belief = (
        distribution
        if isinstance(distribution, ProbabilityDistribution)
        else ProbabilityDistribution(distribution)
    )
    missing = set(belief.probabilities) - set(utilities)
    if missing:
        raise ValueError(f"Utility values are missing outcomes: {sorted(map(repr, missing))}")
    total = 0.0
    for outcome, probability in belief.probabilities.items():
        utility = float(utilities[outcome])
        if not math.isfinite(utility):
            raise ValueError("Utilities must be finite numbers")
        total += probability * utility
    return total


def propagate_categorical(
    prior: ProbabilityDistribution | Mapping[Hashable, float],
    transitions: Mapping[Hashable, Mapping[Hashable, float]],
) -> ProbabilityDistribution:
    """Push a categorical belief through a conditional transition model.

    Each row is ``P(next_state | current_state)`` and must be normalized.
    """

    belief = prior if isinstance(prior, ProbabilityDistribution) else ProbabilityDistribution(prior)
    if set(transitions) != set(belief.probabilities):
        raise ValueError("Transition rows must cover exactly the prior outcomes")
    accumulated: dict[Hashable, float] = {}
    for current_state, current_probability in belief.probabilities.items():
        row = ProbabilityDistribution(transitions[current_state])
        for next_state, transition_probability in row.probabilities.items():
            accumulated[next_state] = accumulated.get(next_state, 0.0) + (
                current_probability * transition_probability
            )
    return ProbabilityDistribution(accumulated)
