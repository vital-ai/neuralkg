from dataclasses import dataclass, field
from typing import List, Optional
from .literal import Literal, Comparison, AggregateSpec

@dataclass(slots=True)
class Rule:
    """
    A Datalog rule that can include:
      1. A (possibly aggregate) head Literal
      2. A body containing:
           - positive or negated Literals
           - Comparisons (built-ins)
           - At most one AggregateSpec (applies to the head)

    Example 1 (negation + comparison):
        sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y, not married(X, Y).

    Example 2 (aggregate):
        total_spent(C, SumVal) :-
            purchase(C, O, Amt),
            SumVal = sum([Amt]) GROUP BY (C).
    """
    head: Literal
    body_literals: list[Literal] = field(default_factory=list)
    comparisons: list[Comparison] = field(default_factory=list)
    aggregate: Optional[AggregateSpec] = None
    _has_aggregates: bool = field(default=False)
    _available_vars_after_agg: set = field(default_factory=set)
    
    @property
    def has_aggregates(self) -> bool:
        """Returns True if this rule contains aggregate predicates."""
        return self._has_aggregates
    
    @has_aggregates.setter
    def has_aggregates(self, value: bool):
        self._has_aggregates = value
    
    @property
    def available_vars_after_agg(self) -> set:
        """Returns the set of variables available after aggregation."""
        return self._available_vars_after_agg
    
    @available_vars_after_agg.setter
    def available_vars_after_agg(self, value: set):
        self._available_vars_after_agg = value

    def __repr__(self) -> str:
        parts: list[str] = []
        for lit in self.body_literals:
            parts.append(repr(lit))
        for cmp_ in self.comparisons:
            parts.append(repr(cmp_))
        if self.aggregate:
            parts.append(repr(self.aggregate))

        if parts:
            body_str = ", ".join(parts)
            return f"{repr(self.head)} :- {body_str}."
        else:
            return f"{repr(self.head)}."
