"""Optional value-iteration planner for stochastic finite state spaces."""

from __future__ import annotations

from typing import Callable, Hashable, Iterable

from .models import MDPResult, StochasticAction


class ValueIterationPlanner:
    """Compute a stationary policy for a finite discounted MDP."""

    def __init__(
        self,
        states: Iterable[Hashable],
        actions: Callable[[Hashable], Iterable[StochasticAction]],
        *,
        terminal_states: Iterable[Hashable] = (),
        discount: float = 0.9,
        tolerance: float = 1e-8,
        max_iterations: int = 1_000,
    ) -> None:
        self.states = tuple(dict.fromkeys(states))
        self.actions = actions
        self.terminal_states = frozenset(terminal_states)
        if not 0.0 <= discount < 1.0:
            raise ValueError("discount must be between 0 inclusive and 1 exclusive")
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        if max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        self.discount = discount
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self) -> MDPResult:
        """Run value iteration until convergence or the iteration bound."""

        values = {state: 0.0 for state in self.states}
        policy: dict[Hashable, StochasticAction | None] = {
            state: None for state in self.states
        }

        for iteration in range(1, self.max_iterations + 1):
            next_values: dict[Hashable, float] = {}
            max_delta = 0.0

            for state in self.states:
                if state in self.terminal_states:
                    next_values[state] = 0.0
                    policy[state] = None
                    continue

                candidates = tuple(self.actions(state))
                if not candidates:
                    next_values[state] = 0.0
                    policy[state] = None
                    continue

                best_action = max(
                    candidates,
                    key=lambda action: self._action_value(action, values),
                )
                best_value = self._action_value(best_action, values)
                next_values[state] = best_value
                policy[state] = best_action
                max_delta = max(max_delta, abs(best_value - values[state]))

            values = next_values
            if max_delta <= self.tolerance:
                return MDPResult(values, dict(policy), iteration, True)

        return MDPResult(values, dict(policy), self.max_iterations, False)

    def rollout(
        self,
        initial_state: Hashable,
        result: MDPResult,
        *,
        terminal_test: Callable[[Hashable], bool] | None = None,
        max_steps: int = 50,
    ) -> tuple[StochasticAction, ...]:
        """Return policy actions until a terminal state or loop bound.

        In a stochastic environment this is the policy, not a guaranteed
        deterministic action sequence.  The rollout follows the most likely
        outcome only for demonstration and inspection.
        """

        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        state = initial_state
        actions: list[StochasticAction] = []
        seen: set[Hashable] = set()
        for _ in range(max_steps):
            if state in seen or (terminal_test is not None and terminal_test(state)):
                break
            seen.add(state)
            action = result.policy.get(state)
            if action is None:
                break
            actions.append(action)
            state = max(action.outcomes, key=lambda outcome: outcome.probability).next_state
        return tuple(actions)

    def _action_value(self, action: StochasticAction, values: dict[Hashable, float]) -> float:
        return sum(
            outcome.probability
            * (outcome.reward + self.discount * values.get(outcome.next_state, 0.0))
            for outcome in action.outcomes
        )
