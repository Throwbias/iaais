"""Bridge between the Knowledge Base and the generic Search Engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Hashable, Iterable

from iaais.search_engine import SearchAction, SearchProblem

from .engine import KnowledgeBase
from .models import ConstraintCheck, ConstraintDisposition, Explanation


ActionProvider = Callable[[Any], Iterable[SearchAction[Any, Any]]]
ConstraintProvider = Callable[[Any, SearchAction[Any, Any]], ConstraintCheck]
CostProvider = Callable[[Any, Any], float]
HeuristicProvider = Callable[[Any, Any], float]


class KnowledgeBaseSearchAdapter:
    """Expose Knowledge Base-aware actions through the Search Engine API.

    The adapter deliberately accepts an ``action_provider``.  The Knowledge
    Base stores facts and rules, but a domain module still decides how a
    particular state is translated into candidate transitions.  This avoids
    duplicating exercise-specific state logic inside the generic Knowledge
    Base while giving constraints, uncertainty, and provenance a central
    place to influence search.
    """

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        action_provider: ActionProvider,
        *,
        constraint_provider: ConstraintProvider | None = None,
        cost_provider: CostProvider | None = None,
        heuristic_provider: HeuristicProvider | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.action_provider = action_provider
        self.constraint_provider = constraint_provider
        self.cost_provider = cost_provider
        self.heuristic_provider = heuristic_provider

    def get_applicable_actions(self, state: Any) -> tuple[SearchAction[Any, Any], ...]:
        """Return allowed candidate actions with review penalties applied."""

        applicable: list[SearchAction[Any, Any]] = []
        for action in self.action_provider(state):
            check = self.check_constraints(state, action)
            if check.disposition is ConstraintDisposition.REJECT:
                continue

            base_cost = float(action.cost)
            if self.cost_provider is not None:
                base_cost += float(self.cost_provider(state, action.next_state))
            metadata = dict(action.metadata)
            metadata["constraint_disposition"] = check.disposition.value
            if check.reason:
                metadata["constraint_reason"] = check.reason
            if check.penalty:
                metadata["uncertainty_penalty"] = check.penalty
            applicable.append(replace(action, cost=base_cost + check.penalty, metadata=metadata))
        return tuple(applicable)

    def check_constraints(self, state: Any, action: SearchAction[Any, Any]) -> ConstraintCheck:
        """Evaluate hard constraints or return an unconstrained default."""

        if self.constraint_provider is None:
            return ConstraintCheck()
        return self.constraint_provider(state, action)

    def estimate_remaining_cost(self, state: Any, goal: Any) -> float:
        """Return a lower-bound estimate when the domain supplies one."""

        if self.heuristic_provider is None:
            return 0.0
        estimate = float(self.heuristic_provider(state, goal))
        if estimate < 0:
            raise ValueError("Knowledge Base heuristic must be non-negative")
        return estimate

    def make_problem(
        self,
        initial_state: Any,
        goal_test: Callable[[Any], bool],
        goal: Any,
        *,
        state_key: Callable[[Any], Hashable] | None = None,
    ) -> SearchProblem[Any, Any]:
        """Create a SearchProblem wired to this adapter."""

        heuristic = lambda state: self.estimate_remaining_cost(state, goal)
        if state_key is None:
            return SearchProblem(initial_state, goal_test, self.get_applicable_actions, heuristic)
        return SearchProblem(
            initial_state,
            goal_test,
            self.get_applicable_actions,
            heuristic,
            state_key,
        )

    def record_search_result(
        self,
        path: Iterable[Any],
        goal_result: Any,
        unexplained_observations: Iterable[Any] = (),
        *,
        run_id: str | None = None,
    ) -> tuple[Any, ...]:
        """Persist a Search Engine result through the Knowledge Base."""

        return self.knowledge_base.record_search_result(
            path,
            goal_result,
            unexplained_observations,
            run_id=run_id,
        )

    def explain_failed_search(self, state: Any, goal: Any) -> Explanation:
        """Return the Knowledge Base's structured failure explanation."""

        return self.knowledge_base.explain_failed_search(state, goal)
