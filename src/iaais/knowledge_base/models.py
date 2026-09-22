"""Facts, rules, and explanation models for the IAAIS Knowledge Base."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping


class FactStatus(str, Enum):
    """Epistemic status of a stored fact."""

    CONFIRMED = "confirmed"
    PROPOSED = "proposed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"


class Polarity(str, Enum):
    """Whether a fact asserts or explicitly denies a predicate."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


class TruthStatus(str, Enum):
    """Result of asking whether a pattern is supported."""

    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Variable:
    """A named variable used in a rule or query pattern."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Variable names must not be empty")

    def __str__(self) -> str:
        return f"?{self.name}"


def V(name: str) -> Variable:
    """Short helper for creating a rule/query variable."""

    return Variable(name)


Term = Hashable | Variable


@dataclass(frozen=True, slots=True)
class Pattern:
    """Predicate pattern containing constants and optional variables."""

    predicate: str
    arguments: tuple[Term, ...] = ()
    polarity: Polarity = Polarity.POSITIVE

    def __post_init__(self) -> None:
        if not self.predicate or not self.predicate.strip():
            raise ValueError("Pattern predicates must not be empty")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not isinstance(self.polarity, Polarity):
            object.__setattr__(self, "polarity", Polarity(self.polarity))

    def with_polarity(self, polarity: Polarity) -> Pattern:
        """Return the same pattern with a different polarity."""

        return Pattern(self.predicate, self.arguments, polarity)

    def __str__(self) -> str:
        arguments = ", ".join(str(argument) for argument in self.arguments)
        prefix = "not " if self.polarity is Polarity.NEGATIVE else ""
        return f"{prefix}{self.predicate}({arguments})"


def P(predicate: str, *arguments: Term, polarity: Polarity = Polarity.POSITIVE) -> Pattern:
    """Short helper for creating a predicate pattern."""

    return Pattern(predicate, tuple(arguments), polarity)


@dataclass(frozen=True, slots=True)
class Fact:
    """A grounded, provenance-bearing domain fact.

    Absence of a positive fact does not create a negative fact.  Explicit
    negative facts use ``polarity=Polarity.NEGATIVE`` so the Knowledge Base
    can distinguish contradiction from lack of evidence.
    """

    predicate: str
    arguments: tuple[Hashable, ...] = ()
    fact_id: str = ""
    confidence: float = 1.0
    status: FactStatus = FactStatus.CONFIRMED
    source: str = "unknown"
    observed_at: str | None = None
    interval: tuple[float, float] | None = None
    provenance: tuple[str, ...] = ()
    polarity: Polarity = Polarity.POSITIVE
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.predicate or not self.predicate.strip():
            raise ValueError("Fact predicates must not be empty")
        object.__setattr__(self, "arguments", tuple(self.arguments))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if not 0.0 <= float(self.confidence) <= 1.0 or not math.isfinite(float(self.confidence)):
            raise ValueError("Fact confidence must be a finite number between 0 and 1")
        if not isinstance(self.status, FactStatus):
            object.__setattr__(self, "status", FactStatus(self.status))
        if not isinstance(self.polarity, Polarity):
            object.__setattr__(self, "polarity", Polarity(self.polarity))
        if self.interval is not None:
            start, end = self.interval
            if start > end:
                raise ValueError("Fact interval start must not be after its end")
        for argument in self.arguments:
            try:
                hash(argument)
            except TypeError as exc:
                raise TypeError("Fact arguments must be hashable") from exc

    @property
    def key(self) -> tuple[str, tuple[Hashable, ...], Polarity]:
        """Predicate, arguments, and polarity used for matching."""

        return self.predicate, self.arguments, self.polarity

    @property
    def active(self) -> bool:
        """Whether this fact can participate in a rule premise."""

        return self.status in (FactStatus.CONFIRMED, FactStatus.PROPOSED)

    def __str__(self) -> str:
        pattern = Pattern(self.predicate, self.arguments, self.polarity)
        return f"{pattern} [{self.status.value}, confidence={float(self.confidence):.2f}]"


@dataclass(frozen=True, slots=True)
class Rule:
    """A production rule with conjunctive premises and one conclusion."""

    name: str
    premises: tuple[Pattern, ...]
    conclusion: Pattern
    description: str = ""
    confidence: float = 1.0
    output_status: FactStatus = FactStatus.PROPOSED

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Rule names must not be empty")
        object.__setattr__(self, "premises", tuple(self.premises))
        if not 0.0 <= float(self.confidence) <= 1.0 or not math.isfinite(float(self.confidence)):
            raise ValueError("Rule confidence must be a finite number between 0 and 1")
        if not isinstance(self.output_status, FactStatus):
            object.__setattr__(self, "output_status", FactStatus(self.output_status))

        premise_variables = {
            term.name
            for premise in self.premises
            for term in premise.arguments
            if isinstance(term, Variable)
        }
        conclusion_variables = {
            term.name
            for term in self.conclusion.arguments
            if isinstance(term, Variable)
        }
        unbound = conclusion_variables - premise_variables
        if unbound:
            names = ", ".join(sorted(unbound))
            raise ValueError(f"Rule conclusion contains unbound variable(s): {names}")


@dataclass(frozen=True, slots=True)
class Explanation:
    """Inspectable explanation tree for a query result."""

    query: Pattern
    status: TruthStatus
    conclusion: Fact | None = None
    rule_name: str | None = None
    supporting_facts: tuple[Fact, ...] = ()
    children: tuple[Explanation, ...] = ()
    message: str = ""

    @property
    def text(self) -> str:
        """Render the explanation as readable indented text."""

        return self.render()

    def render(self, indent: int = 0) -> str:
        prefix = " " * indent
        if self.message:
            line = self.message
        elif self.status is TruthStatus.UNKNOWN:
            line = f"Unknown: no active fact or rule supports {self.query}."
        elif self.conclusion is not None and self.rule_name:
            support_ids = ", ".join(fact.fact_id for fact in self.supporting_facts)
            line = f"{self.conclusion} because rule '{self.rule_name}' used [{support_ids}]."
        elif self.conclusion is not None:
            line = f"{self.conclusion} supplied by source '{self.conclusion.source}'."
        else:
            line = f"{self.status.value.title()}: {self.query}."

        lines = [prefix + line]
        for child in self.children:
            lines.append(child.render(indent + 2))
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class QueryResult:
    """Truth status, matches, and explanations returned for a query."""

    query: Pattern
    status: TruthStatus
    matches: tuple[Fact, ...] = ()
    counter_matches: tuple[Fact, ...] = ()
    bindings: tuple[Mapping[str, Hashable], ...] = ()
    explanations: tuple[Explanation, ...] = ()

    @property
    def entailed(self) -> bool:
        return self.status is TruthStatus.ENTAILED

    @property
    def explanation_text(self) -> str:
        return "\n".join(explanation.text for explanation in self.explanations)


class ConstraintDisposition(str, Enum):
    """How a Knowledge Base constraint affects a candidate search action."""

    ALLOW = "allow"
    REVIEW = "review"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class ConstraintCheck:
    """Structured constraint outcome consumed by the Search adapter."""

    disposition: ConstraintDisposition = ConstraintDisposition.ALLOW
    penalty: float = 0.0
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.disposition, ConstraintDisposition):
            object.__setattr__(self, "disposition", ConstraintDisposition(self.disposition))
        if self.penalty < 0 or not math.isfinite(float(self.penalty)):
            raise ValueError("Constraint penalty must be a finite non-negative number")

    @property
    def allowed(self) -> bool:
        return self.disposition is not ConstraintDisposition.REJECT
