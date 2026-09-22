"""Knowledge Base module for structured facts, rules, and explanations."""

from .engine import InferenceLimitExceeded, KnowledgeBase, match_pattern
from .models import (
    ConstraintCheck,
    ConstraintDisposition,
    Explanation,
    Fact,
    FactStatus,
    P,
    Pattern,
    Polarity,
    QueryResult,
    Rule,
    TruthStatus,
    V,
    Variable,
)
from .search_adapter import KnowledgeBaseSearchAdapter

__all__ = [
    "ConstraintCheck",
    "ConstraintDisposition",
    "Explanation",
    "Fact",
    "FactStatus",
    "InferenceLimitExceeded",
    "KnowledgeBase",
    "KnowledgeBaseSearchAdapter",
    "P",
    "Pattern",
    "Polarity",
    "QueryResult",
    "Rule",
    "TruthStatus",
    "V",
    "Variable",
    "match_pattern",
]
