"""Heuristics for deterministic IAAIS planning."""

from __future__ import annotations

from math import inf
from typing import Iterable

from .models import FactKey, GoalSpec, PlanningAction, PlanningState


class RelaxedProblemHeuristic:
    """Delete-relaxed lower bound for grounded forward planning.

    Delete effects are ignored.  The estimate propagates the cheapest known
    optimistic cost to each fact and returns the maximum cost among the goal
    facts.  Using a maximum rather than a sum avoids over-counting one action
    that achieves multiple goals, keeping the estimate admissible.
    """

    def __init__(self, actions: Iterable[PlanningAction], goal: GoalSpec) -> None:
        self.actions = tuple(actions)
        self.goal = goal

    def __call__(self, state: PlanningState) -> float:
        costs: dict[FactKey, float] = {fact: 0.0 for fact in state.facts}
        changed = True

        while changed:
            changed = False
            for action in self.actions:
                if not action.preconditions.issubset(costs):
                    continue
                precondition_cost = max(
                    (costs[fact] for fact in action.preconditions),
                    default=0.0,
                )
                candidate_cost = precondition_cost + float(action.cost)
                for effect in action.add_effects:
                    if candidate_cost < costs.get(effect, inf):
                        costs[effect] = candidate_cost
                        changed = True

        goal_facts = self.goal.positive | self.goal.negative
        return max((costs[fact] for fact in goal_facts if fact in costs), default=0.0)
