"""Tests for the Uncertainty Module's confirmed-fact Knowledge Base bridge."""

from __future__ import annotations

from iaais.knowledge_base import FactStatus, KnowledgeBase, P, Polarity
from iaais.uncertainty import KnowledgeImportStatus, UncertaintyModule


def test_confirmed_fact_conditions_belief_and_retains_fact_provenance():
    knowledge_base = KnowledgeBase()
    fact = knowledge_base.assert_fact(
        "sensor_available",
        ("wrist-unit",),
        status=FactStatus.CONFIRMED,
        source="device_check",
    )
    uncertainty = UncertaintyModule()

    result = uncertainty.import_knowledge_base_evidence(
        knowledge_base,
        "sensor_available",
        P("sensor_available", "wrist-unit"),
        true_value="available",
        false_value="unavailable",
    )

    assert result.status is KnowledgeImportStatus.IMPORTED
    assert result.fact_ids == (fact.fact_id,)
    assert result.report is not None
    assert result.report.posterior.probabilities == {"available": 1.0, "unavailable": 0.0}
    assert result.report.evidence[0].fact_ids == (fact.fact_id,)


def test_proposed_fact_is_not_silently_treated_as_deterministic():
    knowledge_base = KnowledgeBase()
    knowledge_base.assert_fact(
        "activity",
        ("squat",),
        status=FactStatus.PROPOSED,
        source="classifier",
        confidence=0.73,
    )
    result = UncertaintyModule().import_knowledge_base_evidence(
        knowledge_base,
        "activity",
        P("activity", "squat"),
    )

    assert result.status is KnowledgeImportStatus.UNCONFIRMED
    assert result.report is None


def test_conflicted_kb_evidence_blocks_belief_update():
    knowledge_base = KnowledgeBase()
    knowledge_base.assert_fact("activity", ("squat",), status=FactStatus.CONFIRMED)
    knowledge_base.assert_fact(
        "activity",
        ("squat",),
        status=FactStatus.CONFIRMED,
        polarity=Polarity.NEGATIVE,
    )
    uncertainty = UncertaintyModule()
    prior = uncertainty.set_prior("activity", {True: 0.5, False: 0.5})

    result = uncertainty.import_knowledge_base_evidence(
        knowledge_base,
        "activity",
        P("activity", "squat"),
    )

    assert result.status is KnowledgeImportStatus.CONFLICTED
    assert uncertainty.report("activity").posterior == prior.posterior
