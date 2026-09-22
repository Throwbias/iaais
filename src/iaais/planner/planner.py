"""Deterministic IAAIS Planner integrating the Knowledge Base and Search Engine."""

from __future__ import annotations

from itertools import count
from typing import Callable, Iterable

from iaais.knowledge_base import (
    Fact,
    FactStatus,
    KnowledgeBase,
    KnowledgeBaseSearchAdapter,
    Polarity,
)
from iaais.search_engine import SearchAction, SearchAlgorithm, SearchEngine, SearchStatus

from .heuristics import RelaxedProblemHeuristic
from .models import (
    FactKey,
    GoalSpec,
    PlannerResult,
    PlannerStatus,
    PlanningAction,
    PlanningState,
)


class Planner:
    """Generate goal-directed action sequences from the current KB snapshot."""

    BOOKKEEPING_PREDICATES = frozenset(
        {
            "action_executed",
            "plan_result",
            "search_run",
            "unexplained_observation",
        }
    )

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        *,
        search_engine: SearchEngine[PlanningState, PlanningAction] | None = None,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.search_engine = search_engine or SearchEngine(
            SearchAlgorithm.ASTAR,
            heuristic_is_admissible=True,
        )
        self._plan_ids = count(1)

    def current_state(
        self,
        predicate_filter: Callable[[Fact], bool] | None = None,
        *,
        include_proposed: bool = True,
        include_bookkeeping: bool = False,
    ) -> PlanningState:
        """Read domain evidence into a state.

        Planner/search bookkeeping is excluded by default.  Those records
        describe how the system reasoned; they are not facts about the
        exercise session itself and must not influence later replanning.
        """

        allowed_statuses = {FactStatus.CONFIRMED}
        if include_proposed:
            allowed_statuses.add(FactStatus.PROPOSED)
        facts = (
            fact
            for fact in self.knowledge_base.facts
            if fact.status in allowed_statuses
            and (include_bookkeeping or fact.predicate not in self.BOOKKEEPING_PREDICATES)
            and (predicate_filter is None or predicate_filter(fact))
        )
        return PlanningState(frozenset(fact.key for fact in facts))

    def plan(
        self,
        actions: Iterable[PlanningAction],
        goal: GoalSpec,
        *,
        initial_state: PlanningState | None = None,
        plan_id: str | None = None,
        predicate_filter: Callable[[Fact], bool] | None = None,
        unexplained_observations: Iterable[object] = (),
        include_proposed: bool = True,
        include_bookkeeping: bool = False,
        block_on_conflict: bool = True,
        replan_of: str | None = None,
        replan_reason: str | None = None,
    ) -> PlannerResult:
        """Plan from a KB snapshot and persist the proposed result in the KB."""

        action_catalog = tuple(actions)
        start = initial_state or self.current_state(
            predicate_filter,
            include_proposed=include_proposed,
            include_bookkeeping=include_bookkeeping,
        )
        plan_id = plan_id or f"plan-{next(self._plan_ids):05d}"
        heuristic = RelaxedProblemHeuristic(action_catalog, goal)

        def action_provider(state: PlanningState):
            for action in action_catalog:
                if action.applicable(state):
                    yield SearchAction(
                        action=action,
                        next_state=action.apply(state),
                        cost=action.cost,
                        metadata={"planning_action": action.name, **dict(action.metadata)},
                    )

        adapter = KnowledgeBaseSearchAdapter(
            self.knowledge_base,
            action_provider,
            heuristic_provider=lambda state, _goal: heuristic(state),
        )

        review_reasons, has_conflict = self._review_reasons(
            start,
            predicate_filter=predicate_filter,
        )
        initial_goal_gaps = self._sorted_fact_keys(goal.missing(start))

        if has_conflict and block_on_conflict:
            review_reasons = review_reasons + (
                "Planning was blocked because the current state contains conflicting active evidence.",
            )
            written = list(
                adapter.record_search_result(
                    (),
                    PlannerStatus.REVIEW_REQUIRED.value,
                    unexplained_observations,
                    run_id=plan_id,
                )
            )
            plan_fact = self.knowledge_base.assert_fact(
                "plan_result",
                (plan_id, PlannerStatus.REVIEW_REQUIRED.value, ()),
                status=FactStatus.PROPOSED,
                source="planner",
                metadata=self._plan_metadata(
                    goal=goal,
                    total_cost=0.0,
                    expanded_nodes=0,
                    goal_gaps=initial_goal_gaps,
                    review_reasons=review_reasons,
                    replan_of=replan_of,
                    replan_reason=replan_reason,
                ),
            )
            written.append(plan_fact)
            return PlannerResult(
                plan_id=plan_id,
                status=PlannerStatus.REVIEW_REQUIRED,
                goal=goal,
                initial_state=start,
                final_state=None,
                total_cost=0.0,
                expanded_nodes=0,
                written_fact_ids=tuple(fact.fact_id for fact in written),
                message="Planning is blocked until conflicting evidence is reviewed",
                optimality_guaranteed=False,
                goal_gaps=initial_goal_gaps,
                review_reasons=review_reasons,
                replanned_from=replan_of,
            )

        problem = adapter.make_problem(start, goal.is_satisfied, goal)
        search_result = self.search_engine.search(problem)

        if search_result.status is SearchStatus.SUCCESS:
            status = PlannerStatus.SUCCESS
            planned_actions = tuple(step.action for step in search_result.path)
            final_state = search_result.final_state
            goal_result = "satisfied"
            message = "Goal reached by the proposed action sequence"
        elif search_result.status is SearchStatus.LIMIT_REACHED:
            status = PlannerStatus.LIMIT_REACHED
            planned_actions = ()
            final_state = None
            goal_result = "limit_reached"
            message = search_result.message
        else:
            status = PlannerStatus.NO_SOLUTION
            planned_actions = ()
            final_state = None
            goal_result = "no_solution"
            message = search_result.message

        goal_gaps = self._sorted_fact_keys(
            goal.missing(final_state if final_state is not None else start)
        )
        if review_reasons:
            message = f"{message}; review required: {'; '.join(review_reasons)}"

        written = list(
            adapter.record_search_result(
                search_result.path,
                goal_result,
                unexplained_observations,
                run_id=plan_id,
            )
        )
        plan_fact = self.knowledge_base.assert_fact(
            "plan_result",
            (plan_id, status.value, tuple(action.name for action in planned_actions)),
            status=FactStatus.PROPOSED,
            source="planner",
            metadata=self._plan_metadata(
                goal=goal,
                total_cost=search_result.total_cost,
                expanded_nodes=search_result.expanded_nodes,
                goal_gaps=goal_gaps,
                review_reasons=review_reasons,
                replan_of=replan_of,
                replan_reason=replan_reason,
            ),
        )
        written.append(plan_fact)

        return PlannerResult(
            plan_id=plan_id,
            status=status,
            goal=goal,
            initial_state=start,
            final_state=final_state,
            actions=planned_actions,
            total_cost=search_result.total_cost,
            expanded_nodes=search_result.expanded_nodes,
            written_fact_ids=tuple(fact.fact_id for fact in written),
            message=message,
            optimality_guaranteed=search_result.optimality_guaranteed,
            goal_gaps=goal_gaps,
            review_reasons=review_reasons,
            replanned_from=replan_of,
        )

    def replan(
        self,
        actions: Iterable[PlanningAction],
        goal: GoalSpec,
        *,
        previous_plan_id: str,
        reason: str,
        plan_id: str | None = None,
        initial_state: PlanningState | None = None,
        predicate_filter: Callable[[Fact], bool] | None = None,
        unexplained_observations: Iterable[object] = (),
        include_proposed: bool = True,
        include_bookkeeping: bool = False,
        block_on_conflict: bool = True,
    ) -> PlannerResult:
        """Generate a new plan from the current KB after evidence changes."""

        if not previous_plan_id or not previous_plan_id.strip():
            raise ValueError("previous_plan_id must not be empty")
        if not reason or not reason.strip():
            raise ValueError("A replan requires a non-empty reason")

        return self.plan(
            actions,
            goal,
            initial_state=initial_state,
            plan_id=plan_id or f"replan-{next(self._plan_ids):05d}",
            predicate_filter=predicate_filter,
            unexplained_observations=unexplained_observations,
            include_proposed=include_proposed,
            include_bookkeeping=include_bookkeeping,
            block_on_conflict=block_on_conflict,
            replan_of=previous_plan_id,
            replan_reason=reason,
        )

    def _review_reasons(
        self,
        state: PlanningState,
        *,
        predicate_filter: Callable[[Fact], bool] | None,
    ) -> tuple[tuple[str, ...], bool]:
        active_facts = tuple(
            fact
            for fact in self.knowledge_base.facts
            if fact.active
            and fact.key in state.facts
            and (predicate_filter is None or predicate_filter(fact))
        )
        reasons = [
            f"Proposed evidence {fact.fact_id} is being used in the initial planning state."
            for fact in active_facts
            if fact.status is FactStatus.PROPOSED
        ]

        polarities: dict[tuple[str, tuple[object, ...]], set[Polarity]] = {}
        for fact in active_facts:
            base_key = (fact.predicate, fact.arguments)
            polarities.setdefault(base_key, set()).add(fact.polarity)

        has_conflict = False
        for (predicate, arguments), values in sorted(polarities.items(), key=str):
            if len(values) < 2:
                continue
            has_conflict = True
            labels = ", ".join(sorted(polarity.value for polarity in values))
            reasons.append(
                f"Conflicting active evidence for {predicate}{arguments}: {labels}."
            )
        return tuple(reasons), has_conflict

    @staticmethod
    def _sorted_fact_keys(facts: Iterable[FactKey]) -> tuple[FactKey, ...]:
        return tuple(sorted(facts, key=str))

    @staticmethod
    def _plan_metadata(
        *,
        goal: GoalSpec,
        total_cost: float,
        expanded_nodes: int,
        goal_gaps: tuple[FactKey, ...],
        review_reasons: tuple[str, ...],
        replan_of: str | None,
        replan_reason: str | None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "goal": goal.describe(),
            "total_cost": total_cost,
            "expanded_nodes": expanded_nodes,
            "goal_gaps": tuple(map(str, goal_gaps)),
            "review_reasons": review_reasons,
        }
        if replan_of is not None:
            metadata["replanned_from"] = replan_of
        if replan_reason is not None:
            metadata["replan_reason"] = replan_reason
        return metadata

    def apply_execution_update(
        self,
        action: PlanningAction,
        *,
        execution_id: str,
        source: str = "decision_agent",
    ) -> tuple[Fact, ...]:
        """Write a completed information-processing action to the KB.

        This method does not control a person or device. It records the
        symbolic effects of an action after a future Decision Agent reports
        that the action occurred. Existing evidence is preserved; new facts
        are marked proposed until the relevant module or human reviewer
        confirms them.
        """

        records: list[Fact] = []
        records.append(
            self.knowledge_base.assert_fact(
                "action_executed",
                (execution_id, action.name),
                status=FactStatus.PROPOSED,
                source=source,
            )
        )

        for predicate, arguments, polarity in action.add_effects:
            records.append(
                self.knowledge_base.assert_fact(
                    predicate,
                    arguments,
                    status=FactStatus.PROPOSED,
                    source=source,
                    provenance=(records[0].fact_id,),
                    polarity=polarity,
                )
            )

        for predicate, arguments, polarity in action.delete_effects:
            opposite = Polarity.NEGATIVE if polarity is Polarity.POSITIVE else Polarity.POSITIVE
            records.append(
                self.knowledge_base.assert_fact(
                    predicate,
                    arguments,
                    status=FactStatus.PROPOSED,
                    source=source,
                    provenance=(records[0].fact_id,),
                    polarity=opposite,
                )
            )
        return tuple(records)
