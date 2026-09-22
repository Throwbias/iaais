"""Behavioral tests for the Chapter 2 IAAIS Search Engine."""

from __future__ import annotations

import pytest

from iaais.search_engine import (
    SearchAction,
    SearchAlgorithm,
    SearchEngine,
    SearchProblem,
    SearchStatus,
)


def make_graph_problem(heuristic=None) -> SearchProblem[str, str]:
    graph = {
        "S": [
            SearchAction("S-to-A", "A", cost=2),
            SearchAction("S-to-B", "B", cost=1),
        ],
        "A": [SearchAction("A-to-G", "G", cost=2)],
        "B": [SearchAction("B-to-C", "C", cost=2)],
        "C": [SearchAction("C-to-G", "G", cost=2)],
        "G": [],
    }

    return SearchProblem(
        initial_state="S",
        goal_test=lambda state: state == "G",
        actions=lambda state: graph[state],
        heuristic=heuristic or (lambda state: 0.0),
    )


def action_names(result):
    return [step.action for step in result.path]


def test_astar_returns_a_lowest_cost_path_with_an_admissible_heuristic():
    heuristic_values = {"S": 3, "A": 2, "B": 3, "C": 2, "G": 0}
    problem = make_graph_problem(lambda state: heuristic_values[state])

    result = SearchEngine(
        SearchAlgorithm.ASTAR,
        heuristic_is_admissible=True,
    ).search(problem)

    assert result.status is SearchStatus.SUCCESS
    assert action_names(result) == ["S-to-A", "A-to-G"]
    assert result.total_cost == 4
    assert result.optimality_guaranteed is True
    assert result.states == ("S", "A", "G")


def test_uniform_cost_search_prefers_lower_cost_over_shallower_depth():
    graph = {
        "S": [
            SearchAction("direct-but-expensive", "G", cost=10),
            SearchAction("detour", "A", cost=1),
        ],
        "A": [SearchAction("finish-detour", "G", cost=1)],
        "G": [],
    }
    problem = SearchProblem("S", lambda state: state == "G", lambda state: graph[state])

    result = SearchEngine(SearchAlgorithm.UNIFORM_COST).search(problem)

    assert action_names(result) == ["detour", "finish-detour"]
    assert result.total_cost == 2
    assert result.optimality_guaranteed is True


def test_breadth_first_search_finds_a_shallow_path_but_not_necessarily_a_cheap_path():
    graph = {
        "S": [
            SearchAction("direct-but-expensive", "G", cost=10),
            SearchAction("cheap-detour", "A", cost=1),
        ],
        "A": [SearchAction("finish-detour", "G", cost=1)],
        "G": [],
    }
    problem = SearchProblem("S", lambda state: state == "G", lambda state: graph[state])

    result = SearchEngine(SearchAlgorithm.BREADTH_FIRST).search(problem)

    assert action_names(result) == ["direct-but-expensive"]
    assert result.total_cost == 10
    assert result.optimality_guaranteed is False


def test_cost_search_reopens_a_state_when_a_cheaper_path_is_found():
    graph = {
        "S": [
            SearchAction("expensive-to-A", "A", cost=5),
            SearchAction("cheap-to-B", "B", cost=1),
        ],
        "B": [SearchAction("B-to-A", "A", cost=1)],
        "A": [SearchAction("A-to-G", "G", cost=1)],
        "G": [],
    }
    problem = SearchProblem("S", lambda state: state == "G", lambda state: graph[state])

    result = SearchEngine(SearchAlgorithm.UNIFORM_COST).search(problem)

    assert action_names(result) == ["cheap-to-B", "B-to-A", "A-to-G"]
    assert result.total_cost == 3


def test_depth_first_search_is_available_and_deterministic():
    problem = make_graph_problem()

    result = SearchEngine(SearchAlgorithm.DEPTH_FIRST).search(problem)

    assert result.status is SearchStatus.SUCCESS
    assert action_names(result) == ["S-to-A", "A-to-G"]


def test_no_solution_reports_failure_without_faking_a_path():
    problem = SearchProblem(
        initial_state="S",
        goal_test=lambda state: state == "G",
        actions=lambda state: {"S": [SearchAction("dead-end", "D")], "D": []}[state],
    )

    result = SearchEngine().search(problem)

    assert result.status is SearchStatus.NO_SOLUTION
    assert result.path == ()
    assert result.final_state is None
    assert result.total_cost == 0


def test_expansion_bound_is_explicitly_reported():
    problem = make_graph_problem()

    result = SearchEngine(max_expansions=0).search(problem)

    assert result.status is SearchStatus.LIMIT_REACHED
    assert "Expansion limit" in result.message


def test_depth_bound_is_explicitly_reported():
    problem = make_graph_problem()

    result = SearchEngine(max_depth=1).search(problem)

    assert result.status is SearchStatus.LIMIT_REACHED
    assert "Depth limit" in result.message


def test_initial_goal_returns_an_empty_successful_path():
    problem = SearchProblem("already-done", lambda state: state == "already-done", lambda _: [])

    result = SearchEngine().search(problem)

    assert result.status is SearchStatus.SUCCESS
    assert result.path == ()
    assert result.total_cost == 0
    assert result.expanded_nodes == 0


def test_invalid_transition_cost_is_rejected():
    with pytest.raises(ValueError, match="finite non-negative"):
        SearchAction("bad", "next", cost=-1)


def test_workout_style_metadata_survives_in_the_returned_path():
    problem = SearchProblem(
        initial_state=("segment-1", "unexplained"),
        goal_test=lambda state: state[1] == "review-ready",
        actions=lambda state: [
            SearchAction(
                "propose-back-squat",
                ("segment-1", "review-ready"),
                cost=0.25,
                metadata={"confidence": 0.91, "status": "proposed"},
            )
        ]
        if state[1] == "unexplained"
        else [],
    )

    result = SearchEngine(SearchAlgorithm.ASTAR).search(problem)

    assert result.success is True
    assert result.path[0].metadata == {"confidence": 0.91, "status": "proposed"}
