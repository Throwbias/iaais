"""Bounded graph-search algorithms for IAAIS.

The default strategy is A*.  With an admissible heuristic, non-negative
transition costs, and normal graph-search duplicate handling, A* returns a
least-cost solution.  If no heuristic is supplied, the default zero heuristic
reduces A* to uniform-cost search.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
import math
from typing import Generic, TypeVar

from .models import (
    SearchAction,
    SearchAlgorithm,
    SearchProblem,
    SearchResult,
    SearchStatus,
    PathStep,
)


StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass(slots=True)
class _Node(Generic[StateT, ActionT]):
    state: StateT
    state_key: object
    parent: _Node[StateT, ActionT] | None
    transition: SearchAction[StateT, ActionT] | None
    path_cost: float
    depth: int
    order: int


class SearchEngine(Generic[StateT, ActionT]):
    """Solve finite state-space problems with a bounded graph search.

    Parameters
    ----------
    algorithm:
        One of the supported :class:`SearchAlgorithm` values.  A* is the
        default because IAAIS will later search small but structured spaces
        where domain knowledge can reduce expansions.
    max_expansions:
        Safety bound on expanded nodes.  ``None`` disables this bound.
    max_depth:
        Optional maximum solution depth.  ``None`` allows any depth.
    heuristic_is_admissible:
        Declaration from the caller.  The engine cannot prove a heuristic is
        admissible, so this flag controls whether a successful A* result is
        labeled as optimality-guaranteed.
    """

    def __init__(
        self,
        algorithm: SearchAlgorithm | str = SearchAlgorithm.ASTAR,
        *,
        max_expansions: int | None = 10_000,
        max_depth: int | None = None,
        heuristic_is_admissible: bool = False,
    ) -> None:
        try:
            self.algorithm = (
                algorithm
                if isinstance(algorithm, SearchAlgorithm)
                else SearchAlgorithm(algorithm)
            )
        except ValueError as exc:
            supported = ", ".join(item.value for item in SearchAlgorithm)
            raise ValueError(f"Unknown search algorithm {algorithm!r}; choose from {supported}") from exc

        if max_expansions is not None and (not isinstance(max_expansions, int) or max_expansions < 0):
            raise ValueError("max_expansions must be a non-negative integer or None")
        if max_depth is not None and (not isinstance(max_depth, int) or max_depth < 0):
            raise ValueError("max_depth must be a non-negative integer or None")

        self.max_expansions = max_expansions
        self.max_depth = max_depth
        self.heuristic_is_admissible = heuristic_is_admissible

    def search(self, problem: SearchProblem[StateT, ActionT]) -> SearchResult[StateT, ActionT]:
        """Search ``problem`` and return an inspectable path result."""

        root_key = self._validated_key(problem, problem.initial_state)
        sequence = count()
        root = _Node(
            state=problem.initial_state,
            state_key=root_key,
            parent=None,
            transition=None,
            path_cost=0.0,
            depth=0,
            order=next(sequence),
        )

        if problem.goal_test(root.state):
            return self._success_result(root, problem.initial_state, expanded=0, generated=0, frontier=1)

        if self.algorithm in (SearchAlgorithm.BREADTH_FIRST, SearchAlgorithm.DEPTH_FIRST):
            frontier: deque[_Node[StateT, ActionT]] | list[_Node[StateT, ActionT]]
            if self.algorithm is SearchAlgorithm.BREADTH_FIRST:
                frontier = deque([root])
            else:
                frontier = [root]
            discovered = {root_key}
            best_cost: dict[object, float] | None = None
        else:
            frontier_heap: list[tuple[float, int, _Node[StateT, ActionT]]] = []
            heappush(frontier_heap, (self._priority(problem, root), root.order, root))
            frontier = frontier_heap  # type: ignore[assignment]
            discovered = set()
            best_cost = {root_key: 0.0}

        expanded = 0
        generated = 0
        max_frontier = 1
        depth_cutoff = False

        while frontier:
            if self.algorithm is SearchAlgorithm.BREADTH_FIRST:
                node = frontier.popleft()  # type: ignore[union-attr]
            elif self.algorithm is SearchAlgorithm.DEPTH_FIRST:
                node = frontier.pop()  # type: ignore[union-attr]
            else:
                _, _, node = heappop(frontier)  # type: ignore[arg-type]
                assert best_cost is not None
                if node.path_cost > best_cost.get(node.state_key, math.inf):
                    continue

            if problem.goal_test(node.state):
                return self._success_result(
                    node,
                    problem.initial_state,
                    expanded=expanded,
                    generated=generated,
                    frontier=max_frontier,
                )

            if self.max_expansions is not None and expanded >= self.max_expansions:
                return self._limit_result(
                    problem.initial_state,
                    expanded=expanded,
                    generated=generated,
                    frontier=max_frontier,
                    message=f"Expansion limit of {self.max_expansions} reached before a goal was found",
                )

            expanded += 1
            if self.max_depth is not None and node.depth >= self.max_depth:
                depth_cutoff = True
                continue

            transitions = tuple(problem.actions(node.state))
            if self.algorithm is SearchAlgorithm.DEPTH_FIRST:
                transitions = tuple(reversed(transitions))

            for transition in transitions:
                if not isinstance(transition, SearchAction):
                    raise TypeError(
                        "The action function must yield SearchAction instances; "
                        f"received {type(transition).__name__}"
                    )

                child_key = self._validated_key(problem, transition.next_state)
                child_cost = node.path_cost + float(transition.cost)

                if self.algorithm in (SearchAlgorithm.BREADTH_FIRST, SearchAlgorithm.DEPTH_FIRST):
                    if child_key in discovered:
                        continue
                    discovered.add(child_key)
                else:
                    assert best_cost is not None
                    previous_cost = best_cost.get(child_key, math.inf)
                    if child_cost >= previous_cost:
                        continue
                    best_cost[child_key] = child_cost

                child = _Node(
                    state=transition.next_state,
                    state_key=child_key,
                    parent=node,
                    transition=transition,
                    path_cost=child_cost,
                    depth=node.depth + 1,
                    order=next(sequence),
                )
                generated += 1

                if self.algorithm is SearchAlgorithm.BREADTH_FIRST:
                    frontier.append(child)  # type: ignore[union-attr]
                elif self.algorithm is SearchAlgorithm.DEPTH_FIRST:
                    frontier.append(child)  # type: ignore[union-attr]
                else:
                    heappush(frontier, (self._priority(problem, child), child.order, child))  # type: ignore[arg-type]

            max_frontier = max(max_frontier, len(frontier))

        if depth_cutoff:
            return self._limit_result(
                problem.initial_state,
                expanded=expanded,
                generated=generated,
                frontier=max_frontier,
                message=f"Depth limit of {self.max_depth} reached before a goal was found",
            )

        return SearchResult(
            status=SearchStatus.NO_SOLUTION,
            algorithm=self.algorithm,
            initial_state=problem.initial_state,
            expanded_nodes=expanded,
            generated_nodes=generated,
            max_frontier_size=max_frontier,
            optimality_guaranteed=self._optimality_guaranteed(),
            message="The frontier was exhausted without reaching a goal state",
        )

    def _priority(self, problem: SearchProblem[StateT, ActionT], node: _Node[StateT, ActionT]) -> float:
        if self.algorithm is SearchAlgorithm.UNIFORM_COST:
            return node.path_cost
        if self.algorithm is SearchAlgorithm.GREEDY_BEST_FIRST:
            return self._validated_heuristic(problem, node.state)
        if self.algorithm is SearchAlgorithm.ASTAR:
            return node.path_cost + self._validated_heuristic(problem, node.state)
        raise AssertionError("FIFO/stack algorithms do not use a heap priority")

    @staticmethod
    def _validated_key(problem: SearchProblem[StateT, ActionT], state: StateT) -> object:
        key = problem.state_key(state)
        try:
            hash(key)
        except TypeError as exc:
            raise TypeError("Search state keys must be hashable") from exc
        return key

    @staticmethod
    def _validated_heuristic(problem: SearchProblem[StateT, ActionT], state: StateT) -> float:
        try:
            value = float(problem.heuristic(state))
        except (TypeError, ValueError) as exc:
            raise ValueError("The heuristic must return a finite non-negative number") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("The heuristic must return a finite non-negative number")
        return value

    def _optimality_guaranteed(self) -> bool:
        return self.algorithm is SearchAlgorithm.UNIFORM_COST or (
            self.algorithm is SearchAlgorithm.ASTAR and self.heuristic_is_admissible
        )

    def _success_result(
        self,
        node: _Node[StateT, ActionT],
        initial_state: StateT,
        *,
        expanded: int,
        generated: int,
        frontier: int,
    ) -> SearchResult[StateT, ActionT]:
        return SearchResult(
            status=SearchStatus.SUCCESS,
            algorithm=self.algorithm,
            initial_state=initial_state,
            path=self._path_from(node),
            final_state=node.state,
            total_cost=node.path_cost,
            expanded_nodes=expanded,
            generated_nodes=generated,
            max_frontier_size=frontier,
            optimality_guaranteed=self._optimality_guaranteed(),
            message="Goal reached",
        )

    def _limit_result(
        self,
        initial_state: StateT,
        *,
        expanded: int,
        generated: int,
        frontier: int,
        message: str,
    ) -> SearchResult[StateT, ActionT]:
        return SearchResult(
            status=SearchStatus.LIMIT_REACHED,
            algorithm=self.algorithm,
            initial_state=initial_state,
            expanded_nodes=expanded,
            generated_nodes=generated,
            max_frontier_size=frontier,
            optimality_guaranteed=self._optimality_guaranteed(),
            message=message,
        )

    @staticmethod
    def _path_from(node: _Node[StateT, ActionT]) -> tuple[PathStep[StateT, ActionT], ...]:
        steps: list[PathStep[StateT, ActionT]] = []
        current = node
        while current.parent is not None and current.transition is not None:
            transition = current.transition
            steps.append(
                PathStep(
                    action=transition.action,
                    state=current.state,
                    step_cost=float(transition.cost),
                    cumulative_cost=current.path_cost,
                    metadata=transition.metadata,
                )
            )
            current = current.parent
        steps.reverse()
        return tuple(steps)
