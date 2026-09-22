"""Tests for the optional stochastic planning component."""

from __future__ import annotations

from iaais.planner import StochasticAction, StochasticOutcome, ValueIterationPlanner


def test_value_iteration_returns_a_policy_for_a_finite_mdp():
    states = ("start", "safe", "goal", "fail")

    def actions(state):
        if state == "start":
            return (
                StochasticAction(
                    "risky",
                    (
                        StochasticOutcome("goal", 0.5, reward=10),
                        StochasticOutcome("fail", 0.5, reward=0),
                    ),
                ),
                StochasticAction(
                    "safe",
                    (StochasticOutcome("safe", 1.0, reward=1),),
                ),
            )
        if state == "safe":
            return (StochasticAction("finish-safely", (StochasticOutcome("goal", 1.0, reward=2),)),)
        return ()

    planner = ValueIterationPlanner(
        states,
        actions,
        terminal_states=("goal", "fail"),
        discount=0.9,
    )
    result = planner.solve()

    assert result.converged is True
    assert result.iterations < 100
    assert result.policy["start"] is not None
    assert result.policy["start"].name == "risky"
    assert result.values["start"] > 4


def test_mdp_rollout_follows_the_policy_most_likely_outcome_for_inspection():
    states = ("start", "goal", "fail")

    def actions(state):
        if state == "start":
            return (
                StochasticAction(
                    "try-action",
                    (
                        StochasticOutcome("goal", 0.8, reward=1),
                        StochasticOutcome("fail", 0.2, reward=-1),
                    ),
                ),
            )
        return ()

    planner = ValueIterationPlanner(states, actions, terminal_states=("goal", "fail"))
    result = planner.solve()

    rollout = planner.rollout("start", result, terminal_test=lambda state: state in {"goal", "fail"})

    assert [action.name for action in rollout] == ["try-action"]
