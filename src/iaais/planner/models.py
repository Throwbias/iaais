"""Planning state, action, goal, and result models for IAAIS."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

from iaais.knowledge_base import Fact, Polarity


FactKey = tuple[str, tuple[Hashable, ...], Polarity]


def fact_key(
    predicate: str,
    arguments: tuple[Hashable, ...] = (),
    polarity: Polarity = Polarity.POSITIVE,
) -> FactKey:
    """Build the immutable fact signature used by planning states."""

    return predicate, tuple(arguments), polarity


def fact_key_from_fact(fact: Fact) -> FactKey:
    """Convert a Knowledge Base fact into a planning-state signature."""

    return fact.key


@dataclass(frozen=True, slots=True)
class PlanningState:
    """Immutable symbolic world state used by deterministic search."""

    facts: frozenset[FactKey] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "facts", frozenset(self.facts))


# Friendly alias for callers that refer to the state as a PlannerState.
PlannerState = PlanningState


@dataclass(frozen=True, slots=True)
class GoalSpec:
    """Conjunctive goal with explicit positive and negative facts."""

    positive: frozenset[FactKey] = frozenset()
    negative: frozenset[FactKey] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "positive", frozenset(self.positive))
        object.__setattr__(self, "negative", frozenset(self.negative))

    def is_satisfied(self, state: PlanningState) -> bool:
        return self.positive.issubset(state.facts) and self.negative.issubset(state.facts)

    def missing(self, state: PlanningState) -> frozenset[FactKey]:
        return frozenset((self.positive | self.negative) - state.facts)

    def describe(self) -> str:
        facts = [str(item) for item in sorted(self.positive | self.negative, key=str)]
        return " AND ".join(facts) if facts else "empty goal"


@dataclass(frozen=True, slots=True)
class PlanningAction:
    """Grounded deterministic action with STRIPS-style effects."""

    name: str
    preconditions: frozenset[FactKey] = frozenset()
    add_effects: frozenset[FactKey] = frozenset()
    delete_effects: frozenset[FactKey] = frozenset()
    cost: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Planning action names must not be empty")
        object.__setattr__(self, "preconditions", frozenset(self.preconditions))
        object.__setattr__(self, "add_effects", frozenset(self.add_effects))
        object.__setattr__(self, "delete_effects", frozenset(self.delete_effects))
        if not math.isfinite(float(self.cost)) or self.cost < 0:
            raise ValueError("Planning action cost must be finite and non-negative")

    def applicable(self, state: PlanningState) -> bool:
        return self.preconditions.issubset(state.facts)

    def apply(self, state: PlanningState) -> PlanningState:
        if not self.applicable(state):
            raise ValueError(f"Action {self.name!r} is not applicable in the supplied state")

        next_facts = set(state.facts)
        next_facts.difference_update(self.delete_effects)

        # Adding one polarity supersedes its explicit opposite in the
        # hypothetical world state. The historical evidence remains in the KB.
        for predicate, arguments, polarity in self.add_effects:
            opposite = Polarity.NEGATIVE if polarity is Polarity.POSITIVE else Polarity.POSITIVE
            next_facts.discard((predicate, arguments, opposite))
        next_facts.update(self.add_effects)
        return PlanningState(frozenset(next_facts))


class PlannerStatus(str, Enum):
    """Outcome of deterministic planning."""

    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    LIMIT_REACHED = "limit_reached"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True, slots=True)
class PlannerResult:
    """Plan result plus the evidence written to the Knowledge Base."""

    plan_id: str
    status: PlannerStatus
    goal: GoalSpec
    initial_state: PlanningState
    final_state: PlanningState | None = None
    actions: tuple[PlanningAction, ...] = ()
    total_cost: float = 0.0
    expanded_nodes: int = 0
    written_fact_ids: tuple[str, ...] = ()
    message: str = ""
    optimality_guaranteed: bool = False
    goal_gaps: tuple[FactKey, ...] = ()
    review_reasons: tuple[str, ...] = ()
    replanned_from: str | None = None

    @property
    def success(self) -> bool:
        return self.status is PlannerStatus.SUCCESS

    @property
    def requires_review(self) -> bool:
        """Whether the result must remain provisional for human review."""

        return self.status is PlannerStatus.REVIEW_REQUIRED or bool(self.review_reasons)


@dataclass(frozen=True, slots=True)
class StochasticOutcome:
    """One possible outcome of an MDP action."""

    next_state: Hashable
    probability: float
    reward: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.probability) <= 1.0 or not math.isfinite(float(self.probability)):
            raise ValueError("Outcome probability must be between 0 and 1")
        if not math.isfinite(float(self.reward)):
            raise ValueError("Outcome reward must be finite")


@dataclass(frozen=True, slots=True)
class StochasticAction:
    """Action with probabilistic outcomes for optional MDP planning."""

    name: str
    outcomes: tuple[StochasticOutcome, ...]

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Stochastic action names must not be empty")
        object.__setattr__(self, "outcomes", tuple(self.outcomes))
        total = sum(outcome.probability for outcome in self.outcomes)
        if not self.outcomes or not math.isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Stochastic action outcome probabilities must sum to 1")


@dataclass(frozen=True, slots=True)
class MDPResult:
    """Value function and policy returned by value iteration."""

    values: Mapping[Hashable, float]
    policy: Mapping[Hashable, StochasticAction | None]
    iterations: int
    converged: bool
