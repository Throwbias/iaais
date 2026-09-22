"""Public data models for the IAAIS Search Engine.

The search engine deliberately knows nothing about workouts, sensors, or
Knowledge Base rules.  A caller supplies a start state, a goal test, and an
action function that yields :class:`SearchAction` objects.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Generic, Hashable, Iterable, Mapping, TypeVar


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


def _identity(state: StateT) -> Hashable:
    """Use the state itself as its duplicate-detection key."""

    return state  # type: ignore[return-value]


def _zero_heuristic(state: StateT) -> float:
    """Default heuristic, making A* behave like uniform-cost search."""

    del state
    return 0.0


class SearchAlgorithm(str, Enum):
    """Strategies supported by the general-purpose search engine."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    UNIFORM_COST = "uniform_cost"
    GREEDY_BEST_FIRST = "greedy_best_first"
    ASTAR = "astar"


class SearchStatus(str, Enum):
    """Terminal status reported by a search run."""

    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    LIMIT_REACHED = "limit_reached"


@dataclass(frozen=True, slots=True)
class SearchAction(Generic[StateT, ActionT]):
    """One possible transition from a state.

    ``action`` is the domain-level label or object.  ``next_state`` is the
    result of applying that action.  Keeping both values lets later modules
    inspect the action sequence without coupling the engine to a particular
    transition representation.
    """

    action: ActionT
    next_state: StateT
    cost: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            numeric_cost = float(self.cost)
        except (TypeError, ValueError) as exc:
            raise ValueError("Search action cost must be a finite non-negative number") from exc

        if not math.isfinite(numeric_cost) or numeric_cost < 0:
            raise ValueError("Search action cost must be a finite non-negative number")


GoalTest = Callable[[StateT], bool]
ActionFunction = Callable[[StateT], Iterable[SearchAction[StateT, ActionT]]]
HeuristicFunction = Callable[[StateT], float]
StateKeyFunction = Callable[[StateT], Hashable]


@dataclass(frozen=True, slots=True)
class SearchProblem(Generic[StateT, ActionT]):
    """Uniform problem interface consumed by :class:`SearchEngine`.

    A problem consists of exactly the pieces required by the Chapter 2
    integration contract: a start state, a goal test, and an action function.
    The optional heuristic and state-key function allow callers to add domain
    knowledge without changing the engine.
    """

    initial_state: StateT
    goal_test: GoalTest[StateT]
    actions: ActionFunction[StateT, ActionT]
    heuristic: HeuristicFunction[StateT] = _zero_heuristic
    state_key: StateKeyFunction[StateT] = _identity


@dataclass(frozen=True, slots=True)
class PathStep(Generic[StateT, ActionT]):
    """One action and resulting state in a returned solution path."""

    action: ActionT
    state: StateT
    step_cost: float
    cumulative_cost: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[StateT, ActionT]):
    """Complete, inspectable result of one search run."""

    status: SearchStatus
    algorithm: SearchAlgorithm
    initial_state: StateT
    path: tuple[PathStep[StateT, ActionT], ...] = ()
    final_state: StateT | None = None
    total_cost: float = 0.0
    expanded_nodes: int = 0
    generated_nodes: int = 0
    max_frontier_size: int = 0
    optimality_guaranteed: bool = False
    message: str = ""

    @property
    def success(self) -> bool:
        """Whether the search reached a goal state."""

        return self.status is SearchStatus.SUCCESS

    @property
    def states(self) -> tuple[StateT, ...]:
        """Return the initial state followed by each state on the path."""

        return (self.initial_state,) + tuple(step.state for step in self.path)
