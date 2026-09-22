"""Cross-module tests for Knowledge Base actions entering Search Engine."""

from __future__ import annotations

from iaais.knowledge_base import (
    ConstraintCheck,
    ConstraintDisposition,
    KnowledgeBase,
    KnowledgeBaseSearchAdapter,
    P,
    TruthStatus,
    V,
)
from iaais.search_engine import SearchAction, SearchAlgorithm, SearchEngine


def test_adapter_filters_blocked_actions_and_searches_kb_transitions():
    kb = KnowledgeBase()
    kb.assert_fact("transition", ("start", "middle"), confidence=0.95, fact_id="t-1")
    kb.assert_fact("transition", ("middle", "goal"), confidence=0.90, fact_id="t-2")
    kb.assert_fact("transition", ("start", "goal"), confidence=0.99, fact_id="t-3")
    kb.assert_fact("blocked_transition", ("start", "goal"), fact_id="blocked-1")

    def action_provider(state):
        result = kb.query(P("transition", state, V("next")))
        for fact in result.matches:
            yield SearchAction(
                action=fact.fact_id,
                next_state=fact.arguments[1],
                cost=1.0,
                metadata={"confidence": fact.confidence},
            )

    def constraint_provider(state, action):
        blocked = kb.query(
            P("blocked_transition", state, action.next_state),
            infer=False,
        )
        if blocked.status is TruthStatus.ENTAILED:
            return ConstraintCheck(
                disposition=ConstraintDisposition.REJECT,
                reason="explicitly blocked transition",
            )
        if float(action.metadata.get("confidence", 1.0)) < 0.8:
            return ConstraintCheck(
                disposition=ConstraintDisposition.REVIEW,
                penalty=0.5,
                reason="low-confidence transition",
            )
        return ConstraintCheck()

    adapter = KnowledgeBaseSearchAdapter(
        kb,
        action_provider,
        constraint_provider=constraint_provider,
    )
    problem = adapter.make_problem("start", lambda state: state == "goal", "goal")
    result = SearchEngine(SearchAlgorithm.UNIFORM_COST).search(problem)

    assert result.success
    assert [step.action for step in result.path] == ["t-1", "t-2"]
    assert result.final_state == "goal"
    assert all(step.metadata["constraint_disposition"] == "allow" for step in result.path)


def test_adapter_persists_search_result_and_explains_failure():
    kb = KnowledgeBase()
    adapter = KnowledgeBaseSearchAdapter(kb, lambda state: [])

    records = adapter.record_search_result([], "no_solution", ["segment-2"], run_id="search-9")
    explanation = adapter.explain_failed_search("start", "goal")

    assert len(records) == 2
    assert "No supported search path" in explanation.text
    assert kb.query(P("search_run", "search-9", "no_solution", ())).entailed


def test_adapter_can_supply_a_domain_remaining_cost_heuristic():
    kb = KnowledgeBase()
    adapter = KnowledgeBaseSearchAdapter(
        kb,
        lambda state: [],
        heuristic_provider=lambda state, goal: 2.0 if state != goal else 0.0,
    )

    assert adapter.estimate_remaining_cost("start", "goal") == 2.0
    assert adapter.estimate_remaining_cost("goal", "goal") == 0.0
