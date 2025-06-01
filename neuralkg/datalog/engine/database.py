from typing import Any
from ..model.schema import RelationSchema
from .frame import Frame, make_frame
from ..model.rule import Rule
from ..model.literal import Literal
from ..model.terms import Variable, Constant

class DatalogDatabase:
    """
    Holds:
      - _relations: Dict[predicate -> (RelationSchema, Frame)]
      - rules: List[Rule]
    Provides methods to declare relations, add facts/rules, query, etc.
    """
    def __init__(self) -> None:
        self._relations: dict[str, tuple[RelationSchema, Frame]] = {}
        self.rules: list[Rule] = []

    def create_relation(self, predicate: str, arity: int, colnames: tuple[str, ...] = None) -> None:
        """
        Ensure a relation named `predicate` of the specified `arity` exists.
        If it already exists, verify the arity matches. Otherwise, create an
        empty Frame with the appropriate column names (defaults to arg* for facts, or variable names for rules).
        """
        if predicate in self._relations:
            schema, _ = self._relations[predicate]
            if schema.arity != arity:
                raise ValueError(
                    f"create_relation: '{predicate}' exists with arity {schema.arity}, not {arity}."
                )
            return

        if colnames is None:
            # Default column names: arg0, arg1, ..., arg{arity-1}
            colnames = tuple(f"arg{i}" for i in range(arity))
        schema = RelationSchema(predicate=predicate, arity=arity, colnames=colnames)
        empty_frame = make_frame.empty(list(colnames))
        self._relations[predicate] = (schema, empty_frame)

    def add_fact(self, atom: Literal) -> None:
        """
        Insert a ground fact (must be non-negated and ground) into the corresponding relation.
        E.g. `add_fact(Literal("edge",(Constant("a"),Constant("b")),negated=False))`.
        If the relation does not exist, create it. Avoid duplicates.
        """
        if atom.negated:
            raise ValueError("Cannot add a negative literal as a fact.")

        if not atom.is_ground():
            raise ValueError(f"Fact must be ground; got {atom}.")

        pred = atom.predicate
        arity = atom.arity()
        self.create_relation(pred, arity)

        schema, frame = self._relations[pred]
        row = {schema.colnames[i]: atom.terms[i].value for i in range(arity)}
        candidate = make_frame.from_dicts([row], list(schema.colnames))

        if frame.num_rows() > 0:
            merged = frame.merge(candidate, how="inner",
                                 left_on=list(schema.colnames),
                                 right_on=list(schema.colnames))
            if merged.num_rows() > 0:
                return  # duplicate found

        appended = frame.concat([candidate])
        new_full = appended.drop_duplicates()
        self._relations[pred] = (schema, new_full)

    def get_relation(self, predicate: str) -> Frame:
        """
        Return a *copy* of the Frame for the given predicate. Raises KeyError if missing.
        """
        if predicate not in self._relations:
            raise KeyError(f"get_relation: no relation named '{predicate}'.")
        schema, frame = self._relations[predicate]
        return frame.copy()

    def relation_arity(self, predicate: str) -> int:
        """
        Return the arity of a given predicate. Raises KeyError if missing.
        """
        if predicate not in self._relations:
            raise KeyError(f"relation_arity: no relation named '{predicate}'.")
        schema, _ = self._relations[predicate]
        return schema.arity

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule (IDB). Ensures the relation for the head predicate exists with correct schema columns.
        """
        head_pred = rule.head.predicate
        arity = rule.head.arity()
        # Use variable names from head as schema columns
        head_vars = tuple(
            t.name if isinstance(t, Variable) else f"__const_{i}"
            for i, t in enumerate(rule.head.terms)
        )
        self.create_relation(head_pred, arity, colnames=head_vars)
        self.rules.append(rule)

    def all_predicates(self) -> list[str]:
        """
        Return the set of all predicates (from existing EDB relations and from rule heads/bodies).
        """
        preds = set(self._relations.keys())
        for r in self.rules:
            preds.add(r.head.predicate)
            for lit in r.body_literals:
                preds.add(lit.predicate)
        return list(preds)

    def query(self, goal: Literal) -> Frame:
        """
        Once the database is at fixpoint, retrieve all tuples satisfying `goal`.
        If `goal` is ground (no variables), returns a 0- or 1-row Frame.
        If `goal` has variables, returns a Frame whose columns are the variable names.
        """
        if goal.negated:
            raise ValueError("Cannot query a negated literal directly.")

        pred = goal.predicate
        if pred not in self._relations:
            # No such relation: no results
            return make_frame.empty([])

        schema, frame = self._relations[pred]
        df = frame.copy()

        # 1) Filter on any constant positions in goal
        for i, t in enumerate(goal.terms):
            match t:
                case Constant() as c:
                    df = df.filter_equals(schema.colnames[i], c.value)
                case Variable():
                    continue
                case _:
                    raise ValueError(f"Unexpected term in query: {t}")

        if goal.is_ground():
            return df  # 0 or 1 row

        # 2) Project onto variable columns only
        var_positions = [(i, t.name) for i, t in enumerate(goal.terms) if isinstance(t, Variable)]
        select_cols: list[str] = []
        rename_map: dict[str, str] = {}
        for pos, var_name in var_positions:
            col = schema.colnames[pos]
            select_cols.append(col)
            rename_map[col] = var_name

        records = df.to_records()
        projected = [{rename_map[k]: row[k] for k in select_cols} for row in records]
        return make_frame.from_dicts(projected, [v for _, v in var_positions])

    def reset(self) -> None:
        """
        Clear all stored facts (empty all Frames) and drop all rules.
        Schemas remain declared.
        """
        for pred, (schema, _) in self._relations.items():
            empty_frame = make_frame.empty(list(schema.colnames))
            self._relations[pred] = (schema, empty_frame)
        self.rules.clear()
