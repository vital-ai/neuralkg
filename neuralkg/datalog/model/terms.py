from dataclasses import dataclass, field
from typing import Any

@dataclass(frozen=True, slots=True)
class Term:
    """
    Base class for Datalog terms (either a Constant or a Variable).
    """
    name: str

    def is_variable(self) -> bool:
        return isinstance(self, Variable)

    def is_constant(self) -> bool:
        return isinstance(self, Constant)

    def __repr__(self) -> str:
        return self.name

@dataclass(frozen=True, slots=True)
class Constant(Term):
    """
    A Datalog constant, e.g. "john", 42, "hello".
    The `value` is the actual value;
    but we set `name = str(value)` for printing/consistency.
    """
    value: Any = field(compare=False)
    name: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "name", str(self.value))

    def __repr__(self) -> str:
        # Render with quotes if string
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)

@dataclass(frozen=True, slots=True)
class Variable(Term):
    """
    A Datalog variable, e.g. X, Y, Z. The `name` is the variable's identifier.
    """
    def __repr__(self) -> str:
        return self.name
