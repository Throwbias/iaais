"""Behavioral tests for deterministic IAAIS planning."""

from __future__ import annotations

from iaais.knowledge_base import FactStatus, KnowledgeBase, P, Polarity, TruthStatus
from iaais.planner import (
    GoalSpec,
    Planner,
    PlannerStatus,
    PlanningAction,
    RelaxedProblemHeuristic,
    fact_key,
)


def planning_fixture():
    session_open = fact_key("session_open", ("S1",))
    context_ready = fact_key("context_ready", ("S1",))
    segment_ready = fact_key("segment_ready", ("S1", "SEG1"))
    workout_logged = fact_key("workout_logged", ("S1",))

    actions = (
        PlanningAction(
            "open-session-context",
            preconditions=frozenset({session_open}),
            add_effects=frozenset({context_ready}),
            cost=1,
        ),
        PlanningAction(
            "prepare-segment",
            preconditions=frozenset({context_ready}),
            add_effects=frozenset({segment_ready}),
            cost=1,
        ),
        PlanningAction(
            "record-workout",
            preconditions=frozenset({segment_ready}),
            add_effects=frozenset({workout_logged}),
            cost=1,
        ),
    )
    return session_open, context_ready, segment_ready, workout_logged, actions


def test_planner_reads_kb_state_finds_a_forward_plan_and_writes_result():
    session_open, _, _, workout_logged, actions = planning_fixture()
    kb = KnowledgeBase()
    kb.assert_fact("session_open", ("S1",), fact_id="session-open", source="session")
    planner = Planner(kb)

    result = planner.plan(
        actions,
        GoalSpec(positive=frozenset({workout_logged})),
        plan_id="plan-1",
    )

    assert result.status is PlannerStatus.SUCCESS
    assert result.success
    assert [action.name for action in result.actions] == [
        "open-session-context",
        "prepare-segment",
        "record-workout",
    ]
    assert result.initial_state.facts == frozenset({session_open})
    assert result.final_state is not None
    assert workout_logged in result.final_state.facts
    assert result.total_cost == 3
    assert result.optimality_guaranteed is True
    assert kb.query(P("plan_result", "plan-1", "success", tuple(action.name for action in result.actions))).entailed


def test_planner_returns_no_solution_without_inventing_actions():
    _, _, _, workout_logged, actions = planning_fixture()
    kb = KnowledgeBase()
    planner = Planner(kb)

    result = planner.plan(actions, GoalSpec(positive=frozenset({workout_logged})))

    assert result.status is PlannerStatus.NO_SOLUTION
    assert result.actions == ()
    assert result.final_state is None
    assert kb.query(P("plan_result", "plan-00001", "no_solution", ())).entailed


def test_relaxed_problem_heuristic_ignores_delete_effects_and_remains_optimistic():
    session_open, context_ready, segment_ready, workout_logged, actions = planning_fixture()
    heuristic = RelaxedProblemHeuristic(actions, GoalSpec(positive=frozenset({workout_logged})))

    estimate = heuristic(type("State", (), {"facts": frozenset({session_open})})())

    assert estimate == 3
    assert context_ready not in frozenset({session_open})
    assert segment_ready not in frozenset({session_open})


def test_execution_update_records_action_and_effects_as_proposed_facts():
    _, _, _, workout_logged, actions = planning_fixture()
    kb = KnowledgeBase()
    planner = Planner(kb)

    records = planner.apply_execution_update(actions[-1], execution_id="exec-1")

    assert len(records) == 2
    assert kb.query(P("action_executed", "exec-1", "record-workout")).entailed
    effect = kb.query(P("workout_logged", "S1"))
    assert effect.status is TruthStatus.ENTAILED
    assert effect.matches[0].status.value == "proposed"


def test_execution_update_records_delete_effect_as_explicit_negative_evidence():
    open_segment = fact_key("segment_open", ("S1",))
    closed_segment = fact_key("segment_closed", ("S1",))
    action = PlanningAction(
        "close-segment",
        preconditions=frozenset({open_segment}),
        add_effects=frozenset({closed_segment}),
        delete_effects=frozenset({open_segment}),
    )
    kb = KnowledgeBase()
    planner = Planner(kb)

    planner.apply_execution_update(action, execution_id="exec-close")

    assert kb.query(P("segment_closed", "S1")).status is TruthStatus.ENTAILED
    open_result = kb.query(P("segment_open", "S1"))
    assert open_result.status is TruthStatus.CONTRADICTED
    assert open_result.counter_matches[0].polarity is Polarity.NEGATIVE


def test_planner_marks_a_successful_plan_as_provisional_when_using_proposed_evidence():
    candidate = fact_key("exercise_candidate", ("SEG1", "back_squat"))
    logged = fact_key("workout_logged", ("S1",))
    action = PlanningAction(
        "log-candidate",
        preconditions=frozenset({candidate}),
        add_effects=frozenset({logged}),
    )
    kb = KnowledgeBase()
    kb.assert_fact(
        "exercise_candidate",
        ("SEG1", "back_squat"),
        fact_id="candidate-1",
        status=FactStatus.PROPOSED,
        source="exercise-model",
    )

    result = Planner(kb).plan(
        (action,),
        GoalSpec(positive=frozenset({logged})),
        plan_id="plan-provisional",
    )

    assert result.status is PlannerStatus.SUCCESS
    assert result.requires_review is True
    assert "candidate-1" in " ".join(result.review_reasons)


def test_conflicting_active_evidence_blocks_planning_until_review():
    candidate = fact_key("exercise_candidate", ("SEG1", "back_squat"))
    logged = fact_key("workout_logged", ("S1",))
    action = PlanningAction(
        "log-candidate",
        preconditions=frozenset({candidate}),
        add_effects=frozenset({logged}),
    )
    kb = KnowledgeBase()
    kb.assert_fact("exercise_candidate", ("SEG1", "back_squat"), fact_id="candidate-positive")
    kb.assert_fact(
        "exercise_candidate",
        ("SEG1", "back_squat"),
        fact_id="candidate-negative",
        polarity=Polarity.NEGATIVE,
    )

    result = Planner(kb).plan(
        (action,),
        GoalSpec(positive=frozenset({logged})),
        plan_id="plan-conflict",
    )

    assert result.status is PlannerStatus.REVIEW_REQUIRED
    assert result.actions == ()
    assert result.requires_review is True
    assert kb.query(P("plan_result", "plan-conflict", "review_required", ())).entailed


def test_replan_uses_corrected_kb_evidence_and_links_to_previous_plan():
    squat = fact_key("exercise_candidate", ("SEG1", "back_squat"))
    lunge = fact_key("exercise_candidate", ("SEG1", "lunge"))
    logged = fact_key("workout_logged", ("S1",))
    actions = (
        PlanningAction("log-squat", preconditions=frozenset({squat}), add_effects=frozenset({logged})),
        PlanningAction("log-lunge", preconditions=frozenset({lunge}), add_effects=frozenset({logged})),
    )
    goal = GoalSpec(positive=frozenset({logged}))
    kb = KnowledgeBase()
    kb.assert_fact(
        "exercise_candidate",
        ("SEG1", "back_squat"),
        fact_id="candidate-squat",
        status=FactStatus.PROPOSED,
        source="exercise-model",
    )
    planner = Planner(kb)

    first = planner.plan(actions, goal, plan_id="plan-before-review")
    assert [action.name for action in first.actions] == ["log-squat"]

    kb.retract_fact(
        "candidate-squat",
        reason="Trainer corrected the exercise interpretation",
    )
    kb.assert_fact(
        "exercise_candidate",
        ("SEG1", "lunge"),
        fact_id="candidate-lunge",
        status=FactStatus.CONFIRMED,
        source="trainer-review",
    )

    second = planner.replan(
        actions,
        goal,
        previous_plan_id=first.plan_id,
        reason="Replan after trainer correction",
    )

    assert second.status is PlannerStatus.SUCCESS
    assert [action.name for action in second.actions] == ["log-lunge"]
    assert second.replanned_from == first.plan_id
    assert second.requires_review is False
