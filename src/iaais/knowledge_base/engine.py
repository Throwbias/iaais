"""Production-rule Knowledge Base for IAAIS.

The first implementation uses a small Horn-style rule language with explicit
positive and negative facts.  Forward chaining derives reviewable conclusions
to a fixed point; query results retain the facts and rules that support them.
Unknown is never inferred from absence.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import count
from typing import Any, Hashable, Iterable, Mapping

from .models import (
    Explanation,
    Fact,
    FactStatus,
    Pattern,
    Polarity,
    QueryResult,
    Rule,
    TruthStatus,
    Variable,
)


Bindings = dict[str, Hashable]


def match_pattern(
    pattern: Pattern,
    fact: Fact,
    bindings: Mapping[str, Hashable] | None = None,
) -> Bindings | None:
    """Match a pattern against a fact and return variable bindings.

    Existing bindings are respected, which allows the Knowledge Base to join
    multiple rule premises while preserving shared variables.
    """

    if pattern.predicate != fact.predicate or pattern.polarity is not fact.polarity:
        return None
    if len(pattern.arguments) != len(fact.arguments):
        return None

    result: Bindings = dict(bindings or {})
    for term, value in zip(pattern.arguments, fact.arguments):
        if isinstance(term, Variable):
            if term.name in result and result[term.name] != value:
                return None
            result[term.name] = value
        elif term != value:
            return None
    return result


class InferenceLimitExceeded(RuntimeError):
    """Raised when a rule set does not reach a fixed point in time."""


class KnowledgeBase:
    """Store facts and rules, answer queries, and preserve explanations."""

    def __init__(self, rules: Iterable[Rule] = ()) -> None:
        self._facts: dict[str, Fact] = {}
        self._rules: dict[str, Rule] = {}
        self._fact_ids = count(1)
        self._search_run_ids = count(1)
        self._derivation_keys: set[tuple[str, tuple[Any, ...], tuple[str, ...]]] = set()
        for rule in rules:
            self.add_rule(rule)

    @property
    def facts(self) -> tuple[Fact, ...]:
        """Facts in insertion order, including derived facts."""

        return tuple(self._facts.values())

    @property
    def rules(self) -> tuple[Rule, ...]:
        """Registered production rules in insertion order."""

        return tuple(self._rules.values())

    def add_fact(self, fact: Fact) -> Fact:
        """Store a fact and assign an ID when the caller omitted one."""

        if not isinstance(fact, Fact):
            raise TypeError("KnowledgeBase.add_fact expects a Fact")

        if not fact.fact_id:
            fact = replace(fact, fact_id=f"fact-{next(self._fact_ids):05d}")
        if fact.fact_id in self._facts and self._facts[fact.fact_id] != fact:
            raise ValueError(f"Fact ID already exists with different content: {fact.fact_id}")
        self._facts[fact.fact_id] = fact
        return fact

    def assert_fact(
        self,
        predicate: str,
        arguments: Iterable[Hashable] = (),
        *,
        fact_id: str = "",
        confidence: float = 1.0,
        status: FactStatus = FactStatus.CONFIRMED,
        source: str = "unknown",
        observed_at: str | None = None,
        interval: tuple[float, float] | None = None,
        provenance: Iterable[str] = (),
        polarity: Polarity = Polarity.POSITIVE,
        metadata: Mapping[str, Any] | None = None,
    ) -> Fact:
        """Create and store a grounded fact using keyword-friendly fields."""

        return self.add_fact(
            Fact(
                predicate=predicate,
                arguments=tuple(arguments),
                fact_id=fact_id,
                confidence=confidence,
                status=status,
                source=source,
                observed_at=observed_at,
                interval=interval,
                provenance=tuple(provenance),
                polarity=polarity,
                metadata=dict(metadata or {}),
            )
        )

    def add_rule(self, rule: Rule) -> Rule:
        """Register a production rule by unique name."""

        if not isinstance(rule, Rule):
            raise TypeError("KnowledgeBase.add_rule expects a Rule")
        if rule.name in self._rules and self._rules[rule.name] != rule:
            raise ValueError(f"Rule name already exists with different content: {rule.name}")
        self._rules[rule.name] = rule
        return rule

    def get_fact(self, fact_id: str) -> Fact | None:
        """Retrieve a fact by stable ID."""

        return self._facts.get(fact_id)

    def retract_fact(
        self,
        fact_id: str,
        *,
        reason: str,
        source: str = "reviewer",
    ) -> Fact:
        """Reject a fact without deleting its evidence or stale conclusions.

        A corrected sensor interpretation must not silently remain active in
        the Knowledge Base.  Retraction changes the fact's epistemic status
        to ``REJECTED`` and invalidates active derived facts whose provenance
        depends on it.  The original fact IDs and content remain inspectable,
        which preserves the review history without allowing stale conclusions
        to participate in later queries or plans.
        """

        if not reason or not reason.strip():
            raise ValueError("A fact retraction requires a non-empty reason")

        original = self._facts.get(fact_id)
        if original is None:
            raise KeyError(f"Unknown fact ID: {fact_id}")

        rejected = replace(
            original,
            status=FactStatus.REJECTED,
            source=source,
            metadata={
                **dict(original.metadata),
                "retracted_reason": reason,
                "retracted_from_status": original.status.value,
            },
        )
        self._facts[fact_id] = rejected

        invalidated = {fact_id}
        changed = True
        while changed:
            changed = False
            for candidate in tuple(self._facts.values()):
                if not candidate.active or candidate.fact_id in invalidated:
                    continue
                if not candidate.source.startswith("rule:"):
                    continue
                if not invalidated.intersection(candidate.provenance):
                    continue
                self._facts[candidate.fact_id] = replace(
                    candidate,
                    status=FactStatus.REJECTED,
                    source="knowledge_base_retraction",
                    metadata={
                        **dict(candidate.metadata),
                        "invalidated_by": fact_id,
                        "retracted_reason": reason,
                    },
                )
                invalidated.add(candidate.fact_id)
                changed = True

        return rejected

    def infer(self, *, max_rounds: int = 100) -> tuple[Fact, ...]:
        """Forward-chain rules until no new facts can be derived.

        Derived facts retain the minimum confidence of their supporting facts
        and rule.  A conclusion supported by any proposed evidence remains
        proposed even when the rule itself is configured as confirmed.
        """

        if not isinstance(max_rounds, int) or max_rounds < 1:
            raise ValueError("max_rounds must be a positive integer")

        newly_derived: list[Fact] = []
        for round_index in range(max_rounds):
            active_facts = tuple(fact for fact in self._facts.values() if fact.active)
            round_facts: list[Fact] = []

            for rule in self._rules.values():
                for bindings, supports in self._join_premises(rule.premises, active_facts):
                    conclusion_arguments = self._instantiate(rule.conclusion, bindings)
                    support_ids = tuple(fact.fact_id for fact in supports)
                    derivation_key = (rule.name, (conclusion_arguments, rule.conclusion.polarity), support_ids)
                    if derivation_key in self._derivation_keys:
                        continue
                    self._derivation_keys.add(derivation_key)

                    confidence = min(
                        (float(rule.confidence),)
                        + tuple(float(fact.confidence) for fact in supports)
                    )
                    status = rule.output_status
                    if status is FactStatus.CONFIRMED and any(
                        fact.status is not FactStatus.CONFIRMED for fact in supports
                    ):
                        status = FactStatus.PROPOSED

                    derived = self.add_fact(
                        Fact(
                            predicate=rule.conclusion.predicate,
                            arguments=conclusion_arguments,
                            confidence=confidence,
                            status=status,
                            source=f"rule:{rule.name}",
                            provenance=support_ids + (f"rule:{rule.name}",),
                            polarity=rule.conclusion.polarity,
                            metadata={"rule": rule.name, "round": round_index + 1},
                        )
                    )
                    round_facts.append(derived)

            if not round_facts:
                return tuple(newly_derived)
            newly_derived.extend(round_facts)

        raise InferenceLimitExceeded(
            f"Knowledge Base inference did not reach a fixed point within {max_rounds} rounds"
        )

    def query(self, pattern: Pattern, *, infer: bool = True) -> QueryResult:
        """Answer a pattern query under the open-world assumption.

        ``ENTAILED`` requires an active fact matching the requested polarity.
        ``CONTRADICTED`` requires an active fact with the opposite polarity.
        With neither, the result is ``UNKNOWN``; with both, it is
        ``CONFLICTED``.  Rejected and unknown-status facts do not participate.
        """

        if infer:
            self.infer()

        matches = self._matching_facts(pattern)
        counter_matches = self._matching_facts(
            pattern.with_polarity(
                Polarity.NEGATIVE if pattern.polarity is Polarity.POSITIVE else Polarity.POSITIVE
            )
        )

        if matches and counter_matches:
            status = TruthStatus.CONFLICTED
        elif matches:
            status = TruthStatus.ENTAILED
        elif counter_matches:
            status = TruthStatus.CONTRADICTED
        else:
            status = TruthStatus.UNKNOWN

        bindings = tuple(
            match_pattern(pattern, fact) or {} for fact in matches
        )
        explanations: list[Explanation] = []
        for fact in matches:
            explanations.append(self._explain_fact(pattern, fact, set()))
        for fact in counter_matches:
            explanations.append(
                self._explain_fact(pattern.with_polarity(fact.polarity), fact, set())
            )
        if not explanations:
            explanations.append(
                Explanation(
                    query=pattern,
                    status=TruthStatus.UNKNOWN,
                    message=f"Unknown: no active fact or rule supports {pattern}.",
                )
            )

        return QueryResult(
            query=pattern,
            status=status,
            matches=tuple(matches),
            counter_matches=tuple(counter_matches),
            bindings=bindings,
            explanations=tuple(explanations),
        )

    def explain(self, pattern: Pattern, *, infer: bool = True) -> tuple[Explanation, ...]:
        """Return the explanation tree(s) for a query."""

        return self.query(pattern, infer=infer).explanations

    def record_search_result(
        self,
        path: Iterable[Any],
        goal_result: Any,
        unexplained_observations: Iterable[Any] = (),
        *,
        run_id: str | None = None,
    ) -> tuple[Fact, ...]:
        """Persist a Search Engine result as reviewable Knowledge Base facts."""

        if run_id is None:
            run_id = f"search-{next(self._search_run_ids):05d}"
        result_value = getattr(goal_result, "value", goal_result)
        actions = tuple(repr(getattr(step, "action", step)) for step in path)
        observations = tuple(repr(item) for item in unexplained_observations)
        records = [
            self.assert_fact(
                "search_run",
                (run_id, str(result_value), actions),
                status=FactStatus.PROPOSED,
                source="search_engine",
                metadata={"path_length": len(actions), "unexplained_count": len(observations)},
            )
        ]
        for observation in observations:
            records.append(
                self.assert_fact(
                    "unexplained_observation",
                    (run_id, observation),
                    status=FactStatus.PROPOSED,
                    source="search_engine",
                    provenance=(records[0].fact_id,),
                )
            )
        return tuple(records)

    def explain_failed_search(self, state: Any, goal: Any) -> Explanation:
        """Return a structured, reviewable explanation for a failed search."""

        return Explanation(
            query=Pattern("search_goal", (repr(state), repr(goal))),
            status=TruthStatus.UNKNOWN,
            message=(
                f"No supported search path was found from {state!r} toward {goal!r}. "
                "The state remains unresolved; unknown evidence was not treated as false."
            ),
        )

    def _matching_facts(self, pattern: Pattern) -> tuple[Fact, ...]:
        return tuple(
            fact
            for fact in self._facts.values()
            if fact.active and match_pattern(pattern, fact) is not None
        )

    @staticmethod
    def _instantiate(pattern: Pattern, bindings: Mapping[str, Hashable]) -> tuple[Hashable, ...]:
        values: list[Hashable] = []
        for term in pattern.arguments:
            if isinstance(term, Variable):
                if term.name not in bindings:
                    raise ValueError(
                        f"Cannot instantiate unbound variable ?{term.name} in {pattern}"
                    )
                values.append(bindings[term.name])
            else:
                values.append(term)
        return tuple(values)

    @staticmethod
    def _join_premises(
        premises: tuple[Pattern, ...],
        facts: tuple[Fact, ...],
    ) -> tuple[tuple[Bindings, tuple[Fact, ...]], ...]:
        partials: list[tuple[Bindings, tuple[Fact, ...]]] = [(dict(), tuple())]
        for premise in premises:
            next_partials: list[tuple[Bindings, tuple[Fact, ...]]] = []
            for bindings, supports in partials:
                for fact in facts:
                    matched = match_pattern(premise, fact, bindings)
                    if matched is not None:
                        next_partials.append((matched, supports + (fact,)))
            partials = next_partials
            if not partials:
                break
        return tuple(partials)

    def _explain_fact(
        self,
        query: Pattern,
        fact: Fact,
        visited: set[str],
    ) -> Explanation:
        if fact.fact_id in visited:
            return Explanation(
                query=query,
                status=TruthStatus.ENTAILED,
                conclusion=fact,
                message=f"Cycle detected while explaining {fact.fact_id}.",
            )
        visited = set(visited)
        visited.add(fact.fact_id)

        if fact.source.startswith("rule:"):
            rule_name = fact.source.split(":", 1)[1]
            supporting_facts = tuple(
                self._facts[fact_id]
                for fact_id in fact.provenance
                if fact_id in self._facts and fact_id != fact.fact_id
            )
            children = tuple(
                self._explain_fact(
                    Pattern(item.predicate, item.arguments, item.polarity),
                    item,
                    visited,
                )
                for item in supporting_facts
            )
            return Explanation(
                query=query,
                status=TruthStatus.ENTAILED,
                conclusion=fact,
                rule_name=rule_name,
                supporting_facts=supporting_facts,
                children=children,
            )

        return Explanation(
            query=query,
            status=TruthStatus.ENTAILED,
            conclusion=fact,
        )
