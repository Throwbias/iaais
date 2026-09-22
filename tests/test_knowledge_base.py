"""Behavioral tests for the IAAIS Knowledge Base module."""

from __future__ import annotations

import pytest

from iaais.knowledge_base import (
    ConstraintCheck,
    FactStatus,
    KnowledgeBase,
    P,
    Polarity,
    Rule,
    TruthStatus,
    V,
)


def candidate_rule(output_status=FactStatus.PROPOSED) -> Rule:
    return Rule(
        name="supported_logging_candidate",
        premises=(
            P("session_profile", V("session"), V("profile")),
            P("activity_observation", V("session"), V("segment")),
            P("exercise_segment", V("segment"), V("exercise")),
            P("supported_exercise", V("exercise"), V("profile")),
        ),
        conclusion=P("candidate_log", V("session"), V("segment"), V("exercise")),
        description="A supported, sufficiently evidenced segment becomes a reviewable candidate.",
        output_status=output_status,
    )


def populated_candidate_kb() -> KnowledgeBase:
    kb = KnowledgeBase([candidate_rule()])
    kb.assert_fact("session_profile", ("S1", "gyro-v1"), fact_id="profile-1", source="session")
    kb.assert_fact(
        "activity_observation",
        ("S1", "SEG1"),
        fact_id="activity-1",
        confidence=0.91,
        source="activity-model",
    )
    kb.assert_fact(
        "exercise_segment",
        ("SEG1", "back_squat"),
        fact_id="segment-1",
        confidence=0.88,
        status=FactStatus.PROPOSED,
        source="exercise-model",
    )
    kb.assert_fact(
        "supported_exercise",
        ("back_squat", "gyro-v1"),
        fact_id="catalog-1",
        source="exercise-catalog",
    )
    return kb


def test_forward_chaining_derives_a_reviewable_candidate():
    kb = populated_candidate_kb()

    result = kb.query(P("candidate_log", "S1", "SEG1", "back_squat"))

    assert result.status is TruthStatus.ENTAILED
    assert len(result.matches) == 1
    assert result.matches[0].status is FactStatus.PROPOSED
    assert result.matches[0].confidence == pytest.approx(0.88)


def test_explanation_contains_rule_and_supporting_fact_ids():
    kb = populated_candidate_kb()

    explanation = kb.query(P("candidate_log", "S1", "SEG1", "back_squat")).explanation_text

    assert "supported_logging_candidate" in explanation
    assert "profile-1" in explanation
    assert "activity-1" in explanation
    assert "segment-1" in explanation
    assert "catalog-1" in explanation


def test_open_world_absence_is_unknown_not_contradiction():
    kb = KnowledgeBase()

    result = kb.query(P("supported_exercise", "deadlift", "gyro-v1"))

    assert result.status is TruthStatus.UNKNOWN
    assert result.matches == ()
    assert result.counter_matches == ()


def test_explicit_negative_evidence_contradicts_a_positive_query():
    kb = KnowledgeBase()
    kb.assert_fact(
        "supported_exercise",
        ("deadlift", "gyro-v1"),
        fact_id="not-supported-1",
        polarity=Polarity.NEGATIVE,
        source="sensor-capability-catalog",
    )

    result = kb.query(P("supported_exercise", "deadlift", "gyro-v1"))

    assert result.status is TruthStatus.CONTRADICTED
    assert result.matches == ()
    assert result.counter_matches[0].fact_id == "not-supported-1"


def test_positive_and_negative_evidence_are_reported_as_conflicted():
    kb = KnowledgeBase()
    kb.assert_fact("supported_exercise", ("deadlift", "gyro-v1"), fact_id="positive-1")
    kb.assert_fact(
        "supported_exercise",
        ("deadlift", "gyro-v1"),
        fact_id="negative-1",
        polarity=Polarity.NEGATIVE,
    )

    result = kb.query(P("supported_exercise", "deadlift", "gyro-v1"))

    assert result.status is TruthStatus.CONFLICTED
    assert {fact.fact_id for fact in result.matches} == {"positive-1"}
    assert {fact.fact_id for fact in result.counter_matches} == {"negative-1"}


def test_rejected_and_unknown_facts_do_not_support_rules():
    kb = KnowledgeBase(
        [
            Rule(
                "derive-ready",
                (P("observed", V("item")),),
                P("ready", V("item")),
            )
        ]
    )
    kb.assert_fact("observed", ("rejected-item",), status=FactStatus.REJECTED)
    kb.assert_fact("observed", ("unknown-item",), status=FactStatus.UNKNOWN)

    assert kb.query(P("ready", "rejected-item")).status is TruthStatus.UNKNOWN
    assert kb.query(P("ready", "unknown-item")).status is TruthStatus.UNKNOWN


def test_confirmed_rule_output_is_downgraded_when_support_is_proposed():
    kb = KnowledgeBase(
        [
            Rule(
                "confirmed-looking-result",
                (P("observation", V("item")),),
                P("result", V("item")),
                output_status=FactStatus.CONFIRMED,
            )
        ]
    )
    kb.assert_fact("observation", ("item-1",), status=FactStatus.PROPOSED)

    result = kb.query(P("result", "item-1"))

    assert result.matches[0].status is FactStatus.PROPOSED


def test_rules_chain_across_multiple_inference_rounds():
    kb = KnowledgeBase(
        [
            Rule("step-one", (P("raw", V("item")),), P("prepared", V("item"))),
            Rule("step-two", (P("prepared", V("item")),), P("usable", V("item"))),
        ]
    )
    kb.assert_fact("raw", ("record-1",))

    result = kb.query(P("usable", "record-1"))

    assert result.status is TruthStatus.ENTAILED
    assert {fact.source for fact in result.matches} == {"rule:step-two"}


def test_retracting_evidence_invalidates_derived_conclusions_without_deleting_history():
    kb = KnowledgeBase(
        [
            Rule(
                "derive-ready",
                (P("observed", V("item")),),
                P("ready", V("item")),
            )
        ]
    )
    kb.assert_fact("observed", ("record-1",), fact_id="observation-1")

    assert kb.query(P("ready", "record-1")).status is TruthStatus.ENTAILED

    rejected = kb.retract_fact(
        "observation-1",
        reason="Trainer rejected the sensor observation",
    )

    assert rejected.status is FactStatus.REJECTED
    assert kb.get_fact("observation-1") is not None
    assert kb.query(P("observed", "record-1"), infer=False).status is TruthStatus.UNKNOWN
    assert kb.query(P("ready", "record-1"), infer=False).status is TruthStatus.UNKNOWN


def test_invalid_rule_with_unbound_conclusion_variable_is_rejected():
    with pytest.raises(ValueError, match="unbound variable"):
        Rule("bad", (P("raw", V("item")),), P("result", V("missing")))


def test_search_results_are_persisted_as_reviewable_facts():
    kb = KnowledgeBase()
    path = ["open-set", "append-repetition"]

    records = kb.record_search_result(path, "partial", ["interval-3"], run_id="search-1")

    assert len(records) == 2
    assert kb.query(P("search_run", "search-1", "partial", ("'open-set'", "'append-repetition'"))).entailed
    assert kb.query(P("unexplained_observation", "search-1", "'interval-3'"), infer=False).entailed


def test_fact_and_constraint_validation_rejects_invalid_confidence_or_penalty():
    with pytest.raises(ValueError, match="confidence"):
        KnowledgeBase().assert_fact("bad", (), confidence=1.1)
    with pytest.raises(ValueError, match="penalty"):
        ConstraintCheck(penalty=-0.1)
