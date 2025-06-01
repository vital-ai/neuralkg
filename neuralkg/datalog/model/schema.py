from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class RelationSchema:
    """
    Relation schema: predicate name, arity, and column names.
    """
    predicate: str
    arity: int
    colnames: tuple[str, ...]

    def __post_init__(self):
        if len(self.colnames) != self.arity:
            raise ValueError("RelationSchema: colnames length must match arity.")
