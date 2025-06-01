print('EDIT TEST: evaluator.py')
from typing import Any
from .frame import Frame, make_frame, PandasFrame
from .database import DatalogDatabase
from ..model.rule import Rule
from ..model.literal import Literal, Comparison
from ..model.terms import Variable, Constant

class BottomUpEvaluator:
    """
    A stratified, semi-naïve, bottom-up Datalog evaluator supporting:
      - Stratified negation
      - Literal comparisons in the body
      - Aggregates (count, sum, avg, min, max, median, std, var,
                   count_distinct, collect,
                   first, last, product, mode, string_agg) in the head
    """
    def __init__(self, db: DatalogDatabase) -> None:
        self.db = db

        # Two maps per predicate:
        #   self._full[pred]  = Frame of all facts known so far (EDB + IDB from lower/equal strata)
        #   self._delta[pred] = Frame of facts derived in the *last* iteration (for semi-naïve)
        self._full: dict[str, Frame] = {}
        self._delta: dict[str, Frame] = {}

        # Initialize from any EDB in db._relations
        for pred, (_, frame) in db._relations.items():
            uniq = frame.copy().drop_duplicates()
            self._full[pred] = uniq
            self._delta[pred] = uniq.copy()

    def _get_full(self, pred: str) -> Frame:
        """
        Return the Frame of all known facts for `pred`. If missing, create an empty one.
        """
        if pred not in self._full:
            ar = self.db.relation_arity(pred)
            colnames = tuple(f"arg{i}" for i in range(ar))
            empty = make_frame.empty(list(colnames))
            self._full[pred] = empty
            self._delta[pred] = empty.copy()
        return self._full[pred]

    def _get_delta(self, pred: str) -> Frame:
        """
        Return the Frame of facts derived in the *last* iteration for `pred`.
        If missing, return an empty Frame.
        """
        if pred not in self._delta:
            ar = self.db.relation_arity(pred)
            colnames = tuple(f"arg{i}" for i in range(ar))
            self._delta[pred] = make_frame.empty(list(colnames))
        return self._delta[pred]

    def _evaluate_positive_body(self, body_literals: list[Literal]) -> tuple[Frame, dict[Variable, str]]:
        """
        Given a list of positive (non-negated) Literals, compute the join of their
        current ‘full’ tables. Returns (combined_frame, var_to_column_map).
        var_to_column_map maps each Variable → column name in the combined_frame.
        """
        tables: list[Frame] = []
        var_maps: list[dict[Variable, str]] = []

        for idx, lit in enumerate(body_literals):
            pred = lit.predicate
            ar = lit.arity()
            schema, _ = self.db._relations[pred]
            colnames = schema.colnames

            base_full = self._get_full(pred)
            # 1) Rename columns to “b{idx}_arg{j}”
            renamed_map = {colnames[j]: f"b{idx}_arg{j}" for j in range(ar)}
            tbl_renamed = base_full.rename(renamed_map)

            # 2) Filter out any rows that do NOT satisfy ground constant arguments in lit
            for j, t in enumerate(lit.terms):
                match t:
                    case Constant() as c:
                        col_j = renamed_map[colnames[j]]
                        tbl_renamed = tbl_renamed.filter_equals(col_j, c.value)
                    case Variable():
                        pass
                    case _:
                        raise ValueError(f"Unexpected term in body: {t}")

            # 3) Build var_map for this literal: maps Variable → renamed column
            mapping: dict[Variable, str] = {}
            for j, t in enumerate(lit.terms):
                if isinstance(t, Variable):
                    mapping[t] = renamed_map[colnames[j]]

            tables.append(tbl_renamed)
            var_maps.append(mapping)

        # If no positive literals, return an “all-true” one-row Frame with no columns
        if not tables:
            dummy = make_frame.empty([])
            return dummy, {}

        # 4) Iteratively join all tables on shared Variables
        combined = tables[0]
        combined_varmap = var_maps[0].copy()

        for idx in range(1, len(tables)):
            right_tbl = tables[idx]
            right_map = var_maps[idx]

            shared_vars = set(combined_varmap.keys()) & set(right_map.keys())
            left_on = [combined_varmap[v] for v in shared_vars]
            right_on = [right_map[v] for v in shared_vars]

            if shared_vars:
                combined = combined.merge(
                    right_tbl,
                    how="inner",
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=("", "")
                )
            else:
                combined = combined.assign(_tmp_key=1).merge(
                    right_tbl.assign(_tmp_key=1),
                    how="inner",
                    left_on=["_tmp_key"],
                    right_on=["_tmp_key"],
                    suffixes=("", "")
                ).drop(columns=["_tmp_key"])

            for v, colname in right_map.items():
                if v not in combined_varmap:
                    combined_varmap[v] = colname

        return combined, combined_varmap

    def _filter_negative_literals(
        self,
        combined: Frame,
        combined_varmap: dict[Variable, str],
        negated_literals: list[Literal]
    ) -> Frame:
        """
        Given a Frame `combined` (from positive join), remove any rows that
        violate a negated literal. Each negated literal is of the form
        `not p(t1,...,tn)`. We do an anti-join between `combined` and the FULL
        relation for `p` on the columns that bind t1,...,tn.
        """
        result = combined
        for lit in negated_literals:
            pred = lit.predicate
            if pred not in self._full:
                continue  # No facts for p → nothing to remove

            schema, _ = self.db._relations[pred]
            full_tbl = self._get_full(pred)

            join_left_cols: list[str] = []
            join_right_cols: list[str] = []
            for i, t in enumerate(lit.terms):
                if isinstance(t, Constant):
                    # If lit is ground-constant, anti-join removes rows only if
                    # combined binding matches that constant and full_tbl contains it.
                    # (We skip explicit ground-constant handling here.)
                    continue
                elif isinstance(t, Variable):
                    if t not in combined_varmap:
                        join_left_cols = []
                        break
                    left_col = combined_varmap[t]
                    right_col = schema.colnames[i]
                    join_left_cols.append(left_col)
                    join_right_cols.append(right_col)
                else:
                    raise ValueError(f"Unexpected term: {t}")

            if not join_left_cols:
                continue

            merged = result.merge(
                full_tbl,
                how="left",
                left_on=join_left_cols,
                right_on=join_right_cols,
                indicator=True
            )
            filtered = merged.filter_equals("_merge", "left_only").drop(columns=["_merge"] + join_right_cols)
            result = filtered

        return result

    def _filter_comparisons(
        self,
        combined: Frame,
        combined_varmap: dict[Variable, str],
        comparisons: list[Comparison]
    ) -> Frame:
        """
        For each Comparison (left op right), filter the combined Frame accordingly.
        """
        df = combined
        for cmp_ in comparisons:
            left_term = cmp_.left
            right_term = cmp_.right
            op = cmp_.op

            left_col = None
            right_val = None
            if isinstance(left_term, Variable):
                left_col = combined_varmap[left_term]
            else:
                left_val = left_term.value

            if isinstance(right_term, Variable):
                right_col = combined_varmap[right_term]
            else:
                right_val = right_term.value

            # Only support variable-column vs. constant comparisons for now
            if left_col is not None and right_term.is_constant():
                if op == "=":
                    df = df.filter_equals(left_col, right_term.value)
                elif op == "!=":
                    df = df.filter_equals(left_col, lambda x: x != right_term.value)
                elif op == "<":
                    df = df.filter_equals(left_col, lambda x: x < right_term.value)
                elif op == "<=":
                    df = df.filter_equals(left_col, lambda x: x <= right_term.value)
                elif op == ">":
                    df = df.filter_equals(left_col, lambda x: x > right_term.value)
                elif op == ">=":
                    df = df.filter_equals(left_col, lambda x: x >= right_term.value)
                else:
                    raise ValueError(f"Unsupported comparison operator: {op}")
            else:
                raise NotImplementedError("Only variable-column vs. constant comparisons supported.")
        return df

    def evaluate(self):
        print("EVALUATE METHOD ENTERED")
        """
        Perform bottom-up evaluation to a fixpoint, applying all rules (including aggregates)
        until no new facts are derived. Updates the database's relations for all derived predicates.
        """
        changed = True
        iter_count = 0
        print("[DEBUG] Rules loaded:")
        for i, rule in enumerate(self.db.rules):
            print(f"  Rule {i}: head={rule.head.predicate}, body={[lit.predicate for lit in rule.body_literals]}, aggregate={getattr(rule, 'aggregate', None)}")
        print("[DEBUG] EDB facts loaded:")
        for pred, (schema, frame) in self.db._relations.items():
            print(f"  {pred}: {frame.num_rows()} rows, columns={schema.colnames}")
            if frame.num_rows() > 0:
                print(f"    Sample: {list(frame.to_records())[:3]}")
        while changed:
            changed = False
            iter_count += 1
            print(f"\n[DEBUG] Evaluation Iteration {iter_count}")
            # For each rule in the database
            for rule_idx, rule in enumerate(self.db.rules):
                head_pred = rule.head.predicate
                arity = rule.head.arity()
                schema, _ = self.db._relations[head_pred]
                print(f"[DEBUG]  Applying Rule {rule_idx}: head={head_pred}")
                # 1. Evaluate positive body
                pos_body = [lit for lit in rule.body_literals if not lit.negated]
                neg_body = [lit for lit in rule.body_literals if lit.negated]
                combined, varmap = self._evaluate_positive_body(pos_body)
                print(f"[DEBUG]    Combined frame after positive body: {combined.num_rows()} rows, columns={getattr(combined, '_df', getattr(combined, 'columns', []))}")
                if combined.num_rows() > 0:
                    print(f"[DEBUG]      Sample: {list(combined.to_records())[:3]}")
                print(f"[DEBUG]    varmap: {varmap}")
                # 2. Filter negative literals
                if neg_body:
                    combined = self._filter_negative_literals(combined, varmap, neg_body)
                    print(f"[DEBUG]    After filtering negated literals: {combined.num_rows()} rows")
                # 3. Filter comparisons
                if rule.comparisons:
                    combined = self._filter_comparisons(combined, varmap, rule.comparisons)
                    print(f"[DEBUG]    After filtering comparisons: {combined.num_rows()} rows")
                # 4. Handle aggregates
                if rule.aggregate:
                    agg = rule.aggregate
                    group_cols = [varmap[v] for v in agg.group_by]
                    target_col = varmap[agg.target]
                    result_col = agg.result.name
                    func = agg.func
                    df = combined
                    pdf = df._df if hasattr(df, '_df') else None
                    if pdf is not None:
                        grouped = pdf.groupby(group_cols, dropna=False)
                        if func == "sum":
                            agg_df = grouped[target_col].sum().reset_index().rename(columns={target_col: result_col})
                        elif func == "min":
                            agg_df = grouped[target_col].min().reset_index().rename(columns={target_col: result_col})
                        elif func == "max":
                            agg_df = grouped[target_col].max().reset_index().rename(columns={target_col: result_col})
                        elif func == "avg":
                            agg_df = grouped[target_col].mean().reset_index().rename(columns={target_col: result_col})
                        elif func == "median":
                            agg_df = grouped[target_col].median().reset_index().rename(columns={target_col: result_col})
                        elif func == "std":
                            agg_df = grouped[target_col].std().reset_index().rename(columns={target_col: result_col})
                        elif func == "var":
                            agg_df = grouped[target_col].var().reset_index().rename(columns={target_col: result_col})
                        elif func == "count":
                            agg_df = grouped[target_col].count().reset_index().rename(columns={target_col: result_col})
                        elif func == "count_distinct":
                            agg_df = grouped[target_col].nunique().reset_index().rename(columns={target_col: result_col})
                        elif func == "collect":
                            agg_df = grouped[target_col].apply(list).reset_index().rename(columns={target_col: result_col})
                        elif func == "first":
                            agg_df = grouped[target_col].first().reset_index().rename(columns={target_col: result_col})
                        elif func == "last":
                            agg_df = grouped[target_col].last().reset_index().rename(columns={target_col: result_col})
                        elif func == "product":
                            import numpy as np
                            agg_df = grouped[target_col].apply(np.prod).reset_index().rename(columns={target_col: result_col})
                        elif func == "mode":
                            agg_df = grouped[target_col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None).reset_index().rename(columns={target_col: result_col})
                        elif func == "string_agg":
                            agg_df = grouped[target_col].apply(lambda x: ','.join(map(str, x))).reset_index().rename(columns={target_col: result_col})
                        else:
                            raise NotImplementedError(f"Aggregate function {func} not implemented.")
                        out_rows = []
                        for _, row in agg_df.iterrows():
                            out = {}
                            for i, v in enumerate(agg.group_by):
                                out[schema.colnames[i]] = row[group_cols[i]]
                            out[schema.colnames[len(agg.group_by)]] = row[result_col]
                            out_rows.append(out)
                        out_frame = PandasFrame.from_dicts(out_rows, list(schema.colnames))
                    else:
                        out_frame = make_frame.empty(list(schema.colnames))
                    print(f"[DEBUG]    Aggregate out_frame: {out_frame.num_rows()} rows")
                else:
                    head_vars = [t for t in rule.head.terms if isinstance(t, Variable)]
                    head_cols = [varmap[v] for v in head_vars]
                    out_rows = []
                    for rec in combined.to_records():
                        out = {}
                        for i, v in enumerate(head_vars):
                            out[schema.colnames[i]] = rec[head_cols[i]]
                        out_rows.append(out)
                    out_frame = PandasFrame.from_dicts(out_rows, list(schema.colnames))
                    print(f"[DEBUG]    Non-aggregate out_frame: {out_frame.num_rows()} rows")
                prev_full = self._get_full(head_pred)
                before_rows = prev_full.num_rows()
                appended = prev_full.concat([out_frame])
                new_full = appended.drop_duplicates()
                after_rows = new_full.num_rows()
                print(f"[DEBUG]    {head_pred}: {before_rows} -> {after_rows} rows after rule application")
                if out_frame.num_rows() > 0:
                    print(f"[DEBUG]    Sample out_frame rows: {list(out_frame.to_records())[:3]}")
                if new_full.num_rows() > prev_full.num_rows():
                    changed = True
                self._full[head_pred] = new_full
                self._delta[head_pred] = out_frame
                self.db._relations[head_pred] = (schema, new_full)
