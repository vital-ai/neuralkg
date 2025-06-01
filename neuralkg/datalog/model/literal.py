from dataclasses import dataclass, field
from typing import Any, Optional
from .terms import Term, Variable, Constant

@dataclass(frozen=True, slots=True)
class Literal:
    """
    A (possibly negated) predicate applied to terms.
      - predicate: name of the relation, e.g. "parent"
      - terms: tuple of Term (Variable or Constant)
      - negated: if True, this is a negative literal `not predicate(...)`
    """
    predicate: str
    terms: tuple[Term, ...]
    negated: bool = False

    def arity(self) -> int:
        return len(self.terms)

    def is_ground(self) -> bool:
        return all(not t.is_variable() for t in self.terms)

    def __repr__(self) -> str:
        inner = ", ".join(repr(t) for t in self.terms)
        if self.negated:
            return f"not {self.predicate}({inner})"
        else:
            return f"{self.predicate}({inner})"

@dataclass(frozen=True, slots=True)
class Comparison:
    """
    A built-in comparison in the rule body, e.g. X < Y or X != 3.
      left: Term (Variable or Constant)
      op: one of "=", "!=", "<", "<=", ">", ">="
      right: Term (Variable or Constant)
    At evaluation time, this translates to a DataFrame filter.
    """
    left: Term
    op: str
    right: Term

    def __repr__(self) -> str:
        return f"{self.left} {self.op} {self.right}"

@dataclass(frozen=True, slots=True)
class AggregateSpec:
    """
    Specification of an aggregate function in the head of a rule.
      func: one of
         "count","sum","avg","min","max","median","std","var",
         "count_distinct","collect",
         "first","last","product","mode","string_agg"
      group_by: a tuple of Variables that we group on
      target: the Variable whose values we aggregate
      result: the head Variable that captures the aggregate output
    """
    func: str
    group_by: tuple[Variable, ...]
    target: Variable
    result: Variable

    def __repr__(self) -> str:
        gb = ", ".join(v.name for v in self.group_by)
        return f"{self.func}([{self.target.name}]) GROUP BY ({gb}) AS {self.result.name}"
