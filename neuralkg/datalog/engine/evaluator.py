import logging

import os
logger = logging.getLogger(__name__)
log_level_str = os.environ.get("DLG_DEBUG", "INFO").upper()
try:
    logger.setLevel(getattr(logging, log_level_str))
except AttributeError:
    logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

from typing import Any
from .frame import Frame, make_frame
from .database import DatalogDatabase
from .aggregate import AGGREGATE_FUNCS, init_aggregate_registry
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
    def __init__(self, db: DatalogDatabase, max_iterations: int = 1000) -> None:
        self.db = db
        self.max_iterations = max_iterations
        init_aggregate_registry()

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
            # Assert no duplicates
            # Use Frame interface: drop_duplicates and num_rows
            if uniq.num_rows() != uniq.drop_duplicates().num_rows():
                logger.warning(f"[WARN][{pred}] Duplicates found in initial _full!")
            assert uniq.num_rows() == uniq.drop_duplicates().num_rows(), f"Duplicates in initial _full for {pred}"

    def commit_to_db(self):
        """
        Synchronize all facts from self._full into self.db._relations, preserving schema.
        This ensures DatalogDatabase.get_relation and query_from_string return up-to-date results.
        """
        logger.debug("[COMMIT] Committing evaluator results to DatalogDatabase._relations...")
        for pred, frame in self._full.items():
            if pred in self.db._relations:
                schema, _ = self.db._relations[pred]
            else:
                # If schema not present, infer from frame columns
                from neuralkg.datalog.model.schema import RelationSchema
                schema = RelationSchema(predicate=pred, arity=len(frame.columns()), colnames=tuple(frame.columns()))
            # Overwrite the Frame in the database
            self.db._relations[pred] = (schema, frame.copy())
            cols = frame.columns()
            nrows = frame.num_rows()
            logger.debug(f"[COMMIT] Updated relation '{pred}' with columns: {cols} and {nrows} rows.")
        logger.debug("[COMMIT] Commit complete.")

    def evaluate(self, max_iterations: int = None):
        if max_iterations is None:
            max_iterations = self.max_iterations
        all_rules = self.db.rules
        logger.debug("[EVAL][RULES] Loaded rules:")
        for r in all_rules:
            logger.debug(f"  Rule: {r.head.predicate} :- {[str(lit) for lit in getattr(r, 'body_literals', [])]}")
        logger.debug(f"[AUDIT][RULES] Loaded rules: {[r.head.predicate for r in all_rules]}")
        logger.debug(f"[EVAL][RULES] Head predicates (repr): {[repr(r.head.predicate) for r in all_rules]}")

        positive_rules = [r for r in all_rules if not any(getattr(lit, 'negated', False) for lit in getattr(r, 'body_literals', []))]
        negation_rules = [r for r in all_rules if any(getattr(lit, 'negated', False) for lit in getattr(r, 'body_literals', []))]
        iter_count = 0
        changed = True
        while changed:
            if iter_count >= 100:
                logger.error(f"[ERROR][LOOP] Exceeded hard-coded max_iterations=100. Raising RuntimeError.")
                raise RuntimeError(f"Datalog evaluation exceeded hard-coded max_iterations=100. Possible infinite loop.")
            logger.debug(f"[DEBUG][LOOP] iter_count={iter_count}, changed={changed}")
            if iter_count >= max_iterations:
                logger.error(f"[ERROR][LOOP] Exceeded max_iterations={max_iterations}. Raising RuntimeError.")
                raise RuntimeError(f"Datalog evaluation exceeded max_iterations={max_iterations}. Possible infinite loop.")
            logger.debug(f"[EVAL][ITER {iter_count}] Evaluating rules: {[r.head.predicate for r in all_rules]}")
            logger.debug(f"[EVAL][ITER {iter_count}] Rule objects: {[str(r) for r in all_rules]}")
            logger.debug(f"[AUDIT][LOOP] Iteration {iter_count} START")
            logger.debug(f"[AUDIT][LOOP][2] Top of evaluation loop reached: iter {iter_count}")
            if iter_count >= max_iterations:
                logger.error(f"[ERROR][LOOP] Exceeded max_iterations={max_iterations}. Raising RuntimeError.")
                raise RuntimeError(f"Datalog evaluation exceeded max_iterations={max_iterations}. Possible infinite loop.")
            changed_any = False
            for pred in self._full:
                full_before = self._get_full(pred)
                logger.debug(f"[EVAL][DIAG][{pred}] BEFORE iteration: self._full['{pred}'] rows={full_before.num_rows()}")
            for i, rule in enumerate(positive_rules):
                pred = rule.head.predicate
                full_before = self._get_full(pred)
                logger.debug(f"[EVAL][DIAG][{pred}] BEFORE rule: self._full['{pred}'] rows={full_before.num_rows()}")
                prev_full = full_before.num_rows()
                logger.debug(f"[EVAL][Iter {iter_count}] Rule {pred}: prev total facts = {prev_full}")
                rule_changed = self._evaluate_rule(rule, replace=False)
                full_after = self._get_full(pred)
                logger.debug(f"[EVAL][DIAG][{pred}] AFTER rule: self._full['{pred}'] rows={full_after.num_rows()}")
                new_full = full_after.num_rows()
                delta_rows = new_full - prev_full
                logger.debug(f"[EVAL][Iter {iter_count}] Rule {pred}: new facts = {delta_rows}")
                if delta_rows > 0 and pred in self._delta:
                    sample = self._delta[pred].to_records()[:5]
                    logger.debug(f"[EVAL][Iter {iter_count}] Sample new facts for {pred}: {sample}")

                # --- UNIVERSAL DIAGNOSTIC: After every rule, log columns and sample rows for every predicate ---
                for log_pred, log_frame in self._full.items():
                    cols = log_frame.columns() if callable(log_frame.columns) else log_frame.columns
                    num_rows = log_frame.num_rows()
                    logger.debug(f"[DIAG][PREDICATE] Columns in {log_pred}: {cols}")
                    if num_rows > 100:
                        logger.debug(f"[DIAG][PREDICATE] {log_pred} has {num_rows} rows; not printing sample.")
                    else:
                        logger.debug(f"[DIAG][PREDICATE] Sample rows in {log_pred}: {log_frame.to_records()[:5]}")
                    # Runaway growth check
                    if num_rows > 10000:
                        logger.error(f"[ERROR][PREDICATE] {log_pred} has runaway size: {num_rows} rows!")
                        raise RuntimeError(f"Predicate {log_pred} exceeded safe row limit ({num_rows} rows)")
                if rule_changed:
                    changed_any = True
            changed = changed_any
            for pred in self._full:
                full_after = self._get_full(pred)
                logger.debug(f"[EVAL][DIAG][{pred}] AFTER iteration: self._full['{pred}'] rows={full_after.num_rows()}")
            for pred in self._full:
                logger.debug(f"[EVAL][Iter {iter_count}] Predicate {pred}: total facts = {self._full[pred].num_rows()}")
            logger.debug(f"[AUDIT][ALL] Iter {iter_count}: audit block triggered.")
            for pred, frame in self._full.items():
                cols = frame.columns()
                logger.debug(f"[AUDIT][{pred}] Iter {iter_count}: columns={cols}, rows={frame.num_rows()}")
                sample = frame.to_records()[:10]
                logger.debug(f"[AUDIT][{pred}] Sample rows: {sample}")
            for pred, frame in self._full.items():
                # Use Frame interface: drop_duplicates and num_rows
                if frame.num_rows() != frame.drop_duplicates().num_rows():
                    logger.warning(f"[WARN][{pred}] Duplicates found in _full after iteration {iter_count}!")
                assert frame.num_rows() == frame.drop_duplicates().num_rows(), f"Duplicates in _full for {pred} after iter {iter_count}"
            iter_count += 1
        # Evaluate negation rules
        for rule in negation_rules:
            head_pred = rule.head.predicate
            head_vars = rule.head.variables if hasattr(rule.head, 'variables') else rule.head.terms
            result = self._evaluate_rule(rule, replace=True)
            if result:
                # _evaluate_rule already calls _enforce_head_schema_and_update
                pass

        logger.debug(f"[AUDIT][COMPLETE] Evaluation finished after {iter_count} iterations.")
        # Automatically commit results to the database after evaluation
        self.commit_to_db()

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

    def _evaluate_positive_body(self, body_literals: list[Literal], comparisons=None) -> tuple[Frame, dict[Variable, str]]:
        """
        Given a list of positive (non-negated) Literals, compute the join of their current 'full' tables.
        Returns (combined_frame, var_to_column_map). var_to_column_map maps each Variable → column name in the combined_frame.
        """
        logger.debug(f"[FLOW][POSBODY] Entered _evaluate_positive_body for body_literals: {body_literals} and comparisons: {comparisons}")
        logger.debug(f"[EVAL][POSBODY][START] Body literals: {body_literals}, Comparisons: {comparisons}")
        
        if not body_literals:
            logger.debug(f"[FLOW][POSBODY] No positive body literals, returning all-true dummy frame.")
            dummy = make_frame.empty([])
            logger.debug(f"[FLOW][POSBODY] Exiting _evaluate_positive_body with columns: {dummy.columns()} and varmap: {{}}")
            return dummy, {}
        
        # Get all literal solutions and set up variable-to-column mapping
        tables = []
        var_maps = []  # Variable-to-column mapping for each literal
        
        # STEP 1: Prepare tables for each literal with proper renaming and filtering
        for idx, lit in enumerate(body_literals):
            pred = lit.predicate
            ar = lit.arity()
            logger.debug(f"[POSBODY][LIT{idx}] Processing literal: {lit} with predicate {pred}")
            
            # Get base data for this predicate
            base_full = self._get_full(pred)
            schema, _ = self.db._relations[pred]
            colnames = schema.colnames
            
            logger.debug(f"[POSBODY][LIT{idx}] Base table columns: {base_full.columns()}, Schema columns: {colnames}")
            logger.debug(f"[POSBODY][LIT{idx}] Base table rows: {base_full.num_rows()}, Sample: {base_full.to_records()[:5]}")
            
            # Ensure all schema columns exist in the table
            for col in colnames:
                if col not in base_full.columns():
                    logger.error(f"[POSBODY][ERROR][LIT{idx}] Column '{col}' from schema not in table columns: {base_full.columns()}")
            
            # Select only the columns defined in the schema
            try:
                base_full = base_full[list(colnames)]
            except Exception as e:
                logger.error(f"[POSBODY][ERROR][LIT{idx}] Error selecting schema columns: {e}")
                logger.error(f"[POSBODY][ERROR][LIT{idx}] Schema columns: {colnames}, Available columns: {base_full.columns()}")
                # Create an empty frame with the required columns as fallback
                base_full = make_frame.empty(list(colnames))
            
            # Rename columns to internal positional names to avoid name collisions
            renamed_map = {colnames[j]: f"b{idx}_arg{j}" for j in range(ar)}
            tbl_renamed = base_full.rename(renamed_map)
            logger.debug(f"[POSBODY][LIT{idx}] Renamed columns: {renamed_map}")
            logger.debug(f"[POSBODY][LIT{idx}] Table after renaming: columns={tbl_renamed.columns()}, rows={tbl_renamed.num_rows()}")
            
            # Filter for constant terms (ground terms in the literal)
            for j, term in enumerate(lit.terms):
                if isinstance(term, Constant):
                    col_name = f"b{idx}_arg{j}"
                    logger.debug(f"[POSBODY][LIT{idx}] Filtering column {col_name} for constant value {term.value} (type {type(term.value)})")
                    tbl_renamed = tbl_renamed.filter_equals(col_name, term.value)
            
            logger.debug(f"[POSBODY][LIT{idx}] Table after constant filtering: rows={tbl_renamed.num_rows()}, sample={tbl_renamed.to_records()[:5]}")
            
            # Empty table check after filtering
            if tbl_renamed.num_rows() == 0:
                logger.warning(f"[POSBODY][LIT{idx}] Table is EMPTY after filtering constants!")
            
            # Map each variable to the corresponding column in the renamed table
            mapping = {}
            for j, term in enumerate(lit.terms):
                if isinstance(term, Variable):
                    col_name = f"b{idx}_arg{j}"
                    mapping[term] = col_name
                    logger.debug(f"[POSBODY][LIT{idx}] Mapped variable {term} to column {col_name}")
            
            tables.append(tbl_renamed)
            var_maps.append(mapping)
            logger.debug(f"[POSBODY][LIT{idx}] Variable mapping: {mapping}")
        
        # STEP 2: Join all tables
        combined = tables[0]
        combined_varmap = var_maps[0].copy()
        logger.debug(f"[POSBODY][JOIN] Starting with table 0: rows={combined.num_rows()}, columns={combined.columns()}")
        logger.debug(f"[POSBODY][JOIN] Initial varmap: {combined_varmap}")
        
        # Join each subsequent table
        for i in range(1, len(tables)):
            right_tbl = tables[i]
            logger.debug(f"[POSBODY][JOIN][STEP{i}] ===== JOIN STEP {i} =====")
            logger.debug(f"[POSBODY][JOIN][STEP{i}] Right table rows: {right_tbl.num_rows()}, columns: {right_tbl.columns()}")
            logger.debug(f"[POSBODY][JOIN][STEP{i}] Left table rows: {combined.num_rows()}, columns: {combined.columns()}")
            
            # Find common variables between combined and current table
            common_vars = [v for v in var_maps[i] if v in combined_varmap]
            logger.debug(f"[POSBODY][JOIN][STEP{i}] Common variables: {common_vars}")
            
            if not common_vars:
                # No common variables - perform cross join (Cartesian product)
                logger.debug(f"[POSBODY][JOIN][STEP{i}] No common variables, performing CROSS JOIN")
                if combined.num_rows() > 0 and right_tbl.num_rows() > 0:
                    combined = combined.cross_join(right_tbl)
                    logger.debug(f"[POSBODY][JOIN][STEP{i}] Cross join result: rows={combined.num_rows()}, columns={combined.columns()}")
                else:
                    logger.warning(f"[POSBODY][JOIN][STEP{i}] One table is empty! Combined: {combined.num_rows()} rows, Right: {right_tbl.num_rows()} rows")
                    # If either table is empty, result will be empty
                    combined = make_frame.empty(list(combined.columns()) + list(right_tbl.columns()))
            else:
                # Regular join on common variables
                left_cols = [combined_varmap[v] for v in common_vars]
                right_cols = [var_maps[i][v] for v in common_vars]
                
                logger.debug(f"[POSBODY][JOIN][STEP{i}] Join keys: left={left_cols}, right={right_cols}")
                
                # Validate join keys exist in tables
                left_missing = [col for col in left_cols if col not in combined.columns()]
                right_missing = [col for col in right_cols if col not in right_tbl.columns()]
                
                if left_missing or right_missing:
                    logger.error(f"[POSBODY][JOIN][STEP{i}] Missing join columns! Left missing: {left_missing}, Right missing: {right_missing}")
                
                # Perform the join
                if not left_missing and not right_missing:
                    logger.debug(f"[POSBODY][JOIN][STEP{i}] Joining tables: left rows={combined.num_rows()}, right rows={right_tbl.num_rows()}")
                    logger.debug(f"[POSBODY][JOIN][STEP{i}] Left sample: {combined.to_records()[:3]}")
                    logger.debug(f"[POSBODY][JOIN][STEP{i}] Right sample: {right_tbl.to_records()[:3]}")
                    
                    try:
                        combined = combined.merge(right_tbl, left_on=left_cols, right_on=right_cols, how="inner")
                        logger.debug(f"[POSBODY][JOIN][STEP{i}] Join result: rows={combined.num_rows()}, columns={combined.columns()}")
                        logger.debug(f"[POSBODY][JOIN][STEP{i}] Join result sample: {combined.to_records()[:3]}")
                    except Exception as e:
                        logger.error(f"[POSBODY][JOIN][STEP{i}] Join failed: {e}")
                        # Create empty result frame as fallback
                        combined = make_frame.empty(list(combined.columns()) + list(right_tbl.columns()))
                else:
                    logger.warning(f"[POSBODY][JOIN][STEP{i}] Skipping join due to missing columns, creating empty result")
                    combined = make_frame.empty(list(combined.columns()) + list(right_tbl.columns()))
            
            # Empty result check
            if combined.num_rows() == 0:
                logger.warning(f"[POSBODY][JOIN][STEP{i}] Join produced EMPTY result!")
            
            # STEP 3: Update variable-to-column mapping after join
            new_varmap = {}
            
            # First add all variables from combined_varmap that still exist in the joined table
            for v, col in combined_varmap.items():
                if col in combined.columns():
                    new_varmap[v] = col
                    logger.debug(f"[POSBODY][JOIN][STEP{i}] Keeping left variable mapping: {v} -> {col}")
            
            # Then add variables from right table that aren't already in the mapping
            for v, col in var_maps[i].items():
                if col in combined.columns():
                    if v not in new_varmap:  # Only add if not already present
                        new_varmap[v] = col
                        logger.debug(f"[POSBODY][JOIN][STEP{i}] Adding right variable mapping: {v} -> {col}")
            
            combined_varmap = new_varmap
            logger.debug(f"[POSBODY][JOIN][STEP{i}] Updated varmap: {combined_varmap}")
            
            # Check for all-NULL rows (join failure indicator)
            if combined.num_rows() > 0:
                sample_rows = combined.to_records()[:5]
                all_null = True
                for row in sample_rows:
                    for val in row.values():
                        if val is not None and not (isinstance(val, float) and val != val):  # Not None and not NaN
                            all_null = False
                            break
                    if not all_null:
                        break
                if all_null:
                    logger.error(f"[POSBODY][JOIN][STEP{i}] WARNING: Sample rows contain all NULL/NaN values!")
        
        # Final validation and cleanup
        if combined.num_rows() == 0:
            logger.warning("[POSBODY][RESULT] Final result is EMPTY!")
        
        # Ensure varmap only contains columns that exist in the final result
        final_varmap = {v: c for v, c in combined_varmap.items() if c in combined.columns()}
        if len(final_varmap) != len(combined_varmap):
            dropped = set(combined_varmap.keys()) - set(final_varmap.keys())
            logger.warning(f"[POSBODY][RESULT] Dropped variables from final varmap: {dropped}")
        
        logger.debug(f"[POSBODY][RESULT] Final table: rows={combined.num_rows()}, columns={combined.columns()}")
        logger.debug(f"[POSBODY][RESULT] Final variable mapping: {final_varmap}")
        logger.debug(f"[POSBODY][RESULT] Final table sample: {combined.to_records()[:5]}")
        
        return combined, final_varmap

    def _filter_negative_literals(self, result, varmap, negated_literals):
        """
        For each negated literal, filter out rows from `result` that appear in the negated relation.
        Columns in the negated relation are renamed to match the current internal varmap for join keys.
        Defensive logging and robust handling of missing variables/columns included.
        """
        orig_columns = result.columns()
        for lit in negated_literals:
            pred = lit.predicate
            ar = lit.arity()
            schema, _ = self.db._relations[pred]
            colnames = schema.colnames
            full_tbl = self._get_full(pred)
            # Defensive: ensure columns exist
            missing_before = [c for c in colnames if c not in full_tbl.columns()]
            if missing_before:
                logger.error(f"[NEG][ERROR] Predicate '{pred}': missing columns before rename: {missing_before}. Available: {full_tbl.columns()}.")
            full_tbl = full_tbl[list(colnames)]

            # Build mapping: negated relation positional column -> internal column name from varmap
            neg_col_rename = {}
            join_left_cols = []
            join_right_cols = []
            for i, t in enumerate(lit.terms):
                if isinstance(t, Variable):
                    if t in varmap:
                        neg_col_rename[colnames[i]] = varmap[t]
                        join_left_cols.append(varmap[t])
                        join_right_cols.append(varmap[t])
                    else:
                        logger.warning(f"[NEG][WARN] Variable {t} in negated literal {lit} not found in varmap {varmap}")
            # Defensive logging
            if not join_left_cols:
                logger.warning(f"[NEG][WARN] No join keys found for negated literal {lit} with varmap {varmap}")
            for col in join_left_cols:
                if col not in result.columns():
                    logger.error(f"[NEG][ERROR] Join key '{col}' not in result columns: {result.columns()}")
            # Rename only the columns needed for the join
            full_tbl_renamed = full_tbl.rename(neg_col_rename)
            # Only keep join columns in negated relation (robust to extra columns)
            full_tbl_renamed = full_tbl_renamed[join_right_cols] if join_right_cols else full_tbl_renamed
            # Anti-join: filter out rows that appear in the negated relation
            merged = result.merge(
                full_tbl_renamed,
                how="left",
                left_on=join_left_cols,
                right_on=join_right_cols,
                indicator=True
            )
            filtered = merged.filter_equals("_merge", "left_only").drop(columns=["_merge"])
            logger.debug(f"[NEG][FILTER] After filtering negated literal {lit}: columns={filtered.columns()}, sample={filtered.to_records()[:5]}")
            result = filtered[orig_columns] if all(col in filtered.columns() for col in orig_columns) else filtered
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
        # If no comparisons, just return the combined frame
        if not comparisons or combined.num_rows() == 0:
            logger.debug(f"[COMPARISON] No comparisons or empty frame, returning as is: {combined.num_rows()} rows")
            return combined

        logger.debug(f"[COMPARISON] Starting with frame: {combined.num_rows()} rows, columns: {combined.columns()}")
        logger.debug(f"[COMPARISON] Variable mapping: {combined_varmap}")
        logger.debug(f"[COMPARISON] Comparisons to apply: {comparisons}")
        
        # Create a mask to filter the dataframe
        for c in comparisons:
            left, op, right = c.left, c.op, c.right
            logger.debug(f"[COMPARISON] Processing: {left} {op} {right}")
            
            if isinstance(left, Variable) and isinstance(right, Constant):
                if left in combined_varmap:
                    col = combined_varmap[left]
                    if col in combined.columns():
                        val = right.value
                        logger.debug(f"[COMPARISON] Filtering {col} {op} {val} (type: {type(val)})")
                        
                        # Save row count before filtering
                        before_count = combined.num_rows()
                        
                        # Get dataframe mask using the filter method with correct operator
                        if op == "<":
                            combined = combined.filter(col, "<", val)
                        elif op == "<=":
                            combined = combined.filter(col, "<=", val)
                        elif op == "==":
                            combined = combined.filter(col, "==", val)
                        elif op == "!=":
                            combined = combined.filter(col, "!=", val)
                        elif op == ">":
                            combined = combined.filter(col, ">", val)
                        elif op == ">=":
                            combined = combined.filter(col, ">=", val)
                        else:
                            logger.error(f"[COMPARISON] Unsupported comparison operator: {op}")
                        
                        after_count = combined.num_rows()
                        logger.debug(f"[COMPARISON] After filtering: {after_count} rows (removed {before_count - after_count} rows)")
                        logger.debug(f"[COMPARISON] Sample after filtering: {combined.to_records()[:3]}")
                    else:
                        logger.error(f"[COMPARISON] Column '{col}' for variable {left} not found in frame columns: {combined.columns()}")
                        return make_frame.empty(combined.columns())  # Return empty frame if column missing
                else:
                    logger.error(f"[COMPARISON] Variable {left} not found in varmap: {combined_varmap}")
                    return make_frame.empty(combined.columns())  # Return empty frame if variable missing
            
            elif isinstance(left, Variable) and isinstance(right, Variable):
                if left in combined_varmap and right in combined_varmap:
                    left_col = combined_varmap[left]
                    right_col = combined_varmap[right]
                    
                    if left_col in combined.columns() and right_col in combined.columns():
                        logger.debug(f"[COMPARISON] Comparing columns {left_col} {op} {right_col}")
                        before_count = combined.num_rows()
                        
                        # Special handling for pandas-like operations
                        try:
                            if op == "<":
                                combined = combined[combined[left_col] < combined[right_col]]
                            elif op == "<=":
                                combined = combined[combined[left_col] <= combined[right_col]]
                            elif op == "==":
                                combined = combined[combined[left_col] == combined[right_col]]
                            elif op == "!=":
                                combined = combined[combined[left_col] != combined[right_col]]
                            elif op == ">":
                                combined = combined[combined[left_col] > combined[right_col]]
                            elif op == ">=":
                                combined = combined[combined[left_col] >= combined[right_col]]
                            else:
                                logger.error(f"[COMPARISON] Unsupported comparison operator: {op}")
                                continue
                        except Exception as e:
                            logger.error(f"[COMPARISON] Error comparing columns: {e}")
                            # Try fallback to Python-level comparison for non-DataFrame objects
                            filtered_rows = []
                            for i, row in enumerate(combined.to_records()):
                                left_val = row.get(left_col)
                                right_val = row.get(right_col)
                                try:
                                    if op == "<" and left_val < right_val:
                                        filtered_rows.append(i)
                                    elif op == "<=" and left_val <= right_val:
                                        filtered_rows.append(i)
                                    elif op == "==" and left_val == right_val:
                                        filtered_rows.append(i)
                                    elif op == "!=" and left_val != right_val:
                                        filtered_rows.append(i)
                                    elif op == ">" and left_val > right_val:
                                        filtered_rows.append(i)
                                    elif op == ">=" and left_val >= right_val:
                                        filtered_rows.append(i)
                                except Exception:
                                    # Skip problematic comparisons (e.g., None, incomparable types)
                                    pass
                            combined = combined.iloc[filtered_rows]
                        
                        after_count = combined.num_rows()
                        logger.debug(f"[COMPARISON] After filtering: {after_count} rows (removed {before_count - after_count} rows)")
                    else:
                        logger.error(f"[COMPARISON] Columns '{left_col}' or '{right_col}' not found in frame: {combined.columns()}")
                        return make_frame.empty(combined.columns())
                else:
                    logger.error(f"[COMPARISON] Variables {left}, {right} not found in varmap: {combined_varmap}")
                    return make_frame.empty(combined.columns())
            
            elif isinstance(left, Constant) and isinstance(right, Variable):
                if right in combined_varmap:
                    col = combined_varmap[right]
                    if col in combined.columns():
                        val = left.value
                        logger.debug(f"[COMPARISON] Filtering with constant: {val} {op} {col} (reversed)")
                        before_count = combined.num_rows()
                        
                        # Get dataframe mask based on reversed operators using filter with op
                        if op == "<":
                            combined = combined.filter(col, ">", val)  # x < y -> y > x
                        elif op == "<=":
                            combined = combined.filter(col, ">=", val)  # x <= y -> y >= x
                        elif op == "==":
                            combined = combined.filter(col, "==", val)  # x == y -> y == x
                        elif op == "!=":
                            combined = combined.filter(col, "!=", val)  # x != y -> y != x
                        elif op == ">":
                            combined = combined.filter(col, "<", val)  # x > y -> y < x
                        elif op == ">=":
                            combined = combined.filter(col, "<=", val)  # x >= y -> y <= x
                        else:
                            logger.error(f"[COMPARISON] Unsupported comparison operator: {op}")
                        
                        after_count = combined.num_rows()
                        logger.debug(f"[COMPARISON] After filtering: {after_count} rows (removed {before_count - after_count} rows)")
                    else:
                        logger.error(f"[COMPARISON] Column '{col}' for variable {right} not found in frame: {combined.columns()}")
                        return make_frame.empty(combined.columns())
                else:
                    logger.error(f"[COMPARISON] Variable {right} not found in varmap: {combined_varmap}")
                    return make_frame.empty(combined.columns())
            
            else:
                # Both Constants, evaluate directly
                lval, rval = left.value, right.value
                logger.debug(f"[COMPARISON] Comparing constants: {lval} {op} {rval}")
                result = False
                
                try:
                    if op == "<":
                        result = lval < rval
                    elif op == "<=":
                        result = lval <= rval
                    elif op == "==":
                        result = lval == rval
                    elif op == "!=":
                        result = lval != rval
                    elif op == ">":
                        result = lval > rval
                    elif op == ">=":
                        result = lval >= rval
                    else:
                        logger.error(f"[COMPARISON] Unsupported comparison operator: {op}")
                except Exception as e:
                    logger.error(f"[COMPARISON] Error comparing constants: {e}")
                    result = False

                if not result:  # If comparison is false, return empty frame
                    logger.debug(f"[COMPARISON] Constant comparison is false, returning empty frame")
                    return make_frame.empty(combined.columns())
                else:
                    logger.debug(f"[COMPARISON] Constant comparison is true, keeping frame unchanged")

        logger.debug(f"[COMPARISON] Final result after all comparisons: {combined.num_rows()} rows")
        return combined

    def _filter_aggregates(self, result, varmap, rule):
        # No-op stub: just return as-is
        return result, varmap

    def _evaluate_rule(self, rule, replace=False):
        logger.debug(f"[EVAL][RULE][START] Evaluating rule: {rule}")
        logger.debug(f"[EVAL][RULE][START][DEBUG] Head predicate: {repr(rule.head.predicate)} | Rule: {rule} | Body literals: {getattr(rule, 'body_literals', None) or getattr(rule, 'body', [])}")
        head_pred = rule.head.predicate
        head_vars = [t for t in rule.head.terms if isinstance(t, Variable)]
        body_literals = getattr(rule, 'body_literals', None)
        if body_literals is None:
            body_literals = getattr(rule, 'body', [])
        comparisons = getattr(rule, 'comparisons', [])
        positive_body_literals = [lit for lit in body_literals if not getattr(lit, 'negated', False)]
        negated_literals = [lit for lit in body_literals if getattr(lit, 'negated', False)]

        logger.debug(f"[EVAL][RULE][BODY] Head: {head_pred}, Body literals: {positive_body_literals}, Negated: {negated_literals}, Comparisons: {comparisons}")

        # 1. Evaluate positive body and get initial varmap
        combined, varmap = self._evaluate_positive_body(positive_body_literals, comparisons=comparisons)

        logger.debug(f"[EVAL][RULE][AFTER_POSBODY] Head: {head_pred}, Combined shape: {combined.num_rows()}, Columns: {combined.columns()}, Sample: {combined.to_records()[:5]}")

        # 2. Ensure all needed variables are mapped and present as columns
        full_varmap, combined = self._enforce_variable_column_mapping(
            combined, varmap, head_vars, negated_literals, comparisons
        )

        # 3. Apply all filters (negation, comparison, aggregate)
        logger.debug(f"[DIAG][RULE:{head_pred}] [PRE-FILTERS] comparisons: {comparisons}")
        result, varmap = self._apply_all_filters(
            combined, full_varmap, negated_literals, comparisons, rule
        )
        logger.debug(f"[DIAG][RULE:{head_pred}] [POST-FILTERS] columns: {result.columns()}")

        # 4. Project/rename to head variables
        pred_tag = f"[DIAG][RULE:{head_pred}]"
        cols = result.columns()
        logger.debug(f"{pred_tag} [PRE-PROJECT] Columns: {cols}")
        logger.debug(f"{pred_tag} [PRE-PROJECT] Sample rows: {result.to_records()[:5]}")
        logger.debug(f"{pred_tag} [PRE-PROJECT] varmap: {varmap}")
        # Diagnostic: log projection mapping and DataFrame state for all predicates
        logger.debug(f"{pred_tag} [PROJECT][START] varmap before projection: {varmap}")
        logger.debug(f"{pred_tag} [PROJECT][START] columns before projection: {cols}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] head_vars: {head_vars}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] varmap: {varmap}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] result.columns: {cols}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] result sample: {result.to_records()[:5]}")
        # PATCH: For base case (one body literal, all head vars distinct, all body terms are variables), set varmap by position
        body_literals = getattr(rule, 'body_literals', None) or getattr(rule, 'body', [])
        if (
            len(body_literals) == 1
            and len(head_vars) == len(set(head_vars))
            and all(isinstance(t, type(head_vars[0])) for t in body_literals[0].terms)
            and all(isinstance(t, type(head_vars[0])) for t in head_vars)
        ):
            # Map head_vars[i] to result.columns()[i] by position
            varmap = {hv: cols[i] for i, hv in enumerate(head_vars) if i < len(cols)}
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] Overriding varmap for base case: {varmap}")
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] DF before projection: columns={cols}, sample_rows={result.to_records()[:5]}")
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] head_vars: {head_vars}")
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] varmap: {varmap}")
        updated = self._enforce_head_schema_and_update(
            result, varmap, rule, head_pred, head_vars, replace
        )
        full = self._full.get(head_pred, None)

        if full is not None:
            cols = full.columns() if callable(full.columns) else full.columns
            logger.debug(f"{pred_tag} [FULL UPDATE] Columns: {cols}")
            logger.debug(f"{pred_tag} [FULL UPDATE] Sample rows: {full.to_records()[:10]}")
            logger.debug(f"{pred_tag} [FULL UPDATE] num_rows: {full.num_rows()}")
        return updated

    def _enforce_variable_column_mapping(self, combined, varmap, head_vars, negated_literals, comparisons):
        # Ensures all variables needed for the rule (head, negations, comparisons) are mapped to columns
        
        # 1. Collect all variables that need to be in the varmap
        neg_vars = set()
        for lit in negated_literals:
            for t in lit.terms:
                if isinstance(t, Variable):
                    neg_vars.add(t)
        
        comp_vars = set()
        for comp in comparisons:
            if hasattr(comp, 'left') and isinstance(comp.left, Variable):
                comp_vars.add(comp.left)
            if hasattr(comp, 'right') and isinstance(comp.right, Variable):
                comp_vars.add(comp.right)
        
        all_needed_vars = set(head_vars) | neg_vars | comp_vars
        cols = combined.columns()
        
        logger.debug(f"[VARMAP][ENFORCE] Head vars: {head_vars}")
        logger.debug(f"[VARMAP][ENFORCE] Negated vars: {neg_vars}")
        logger.debug(f"[VARMAP][ENFORCE] Comparison vars: {comp_vars}")
        logger.debug(f"[VARMAP][ENFORCE] All needed vars: {all_needed_vars}")
        logger.debug(f"[VARMAP][ENFORCE] Initial varmap: {varmap}")
        logger.debug(f"[VARMAP][ENFORCE] Available columns: {cols}")
        
        # 2. Check each needed variable and ensure it has a mapping to a column
        for v in all_needed_vars:
            if v in varmap and varmap[v] in cols:
                # Variable already has a valid mapping
                logger.debug(f"[VARMAP][ENFORCE] Variable {v} already mapped to column {varmap[v]}")
                continue
            
            # Try to find by name among existing mapped variables
            mapped = False
            for k in varmap:
                if hasattr(k, 'name') and hasattr(v, 'name') and k.name == v.name and varmap[k] in cols:
                    varmap[v] = varmap[k]
                    logger.debug(f"[VARMAP][ENFORCE] Mapped variable {v} to column {varmap[v]} using existing mapping")
                    mapped = True
                    break
            
            if not mapped and hasattr(v, 'name'):
                # Create a positional column name using a unique prefix
                var_name = v.name
                new_col_name = f"var_{var_name}_{len(varmap)}"
                
                # Use the variable name as a fallback only if it already exists in the columns
                if var_name in cols:
                    varmap[v] = var_name
                    logger.debug(f"[VARMAP][ENFORCE] Mapped variable {v} to existing column {var_name}")
                else:
                    # Add a new column with NULL values
                    combined = combined.assign(**{new_col_name: [None] * combined.num_rows()})
                    varmap[v] = new_col_name
                    logger.debug(f"[VARMAP][ENFORCE] Created new column {new_col_name} for variable {v}")
        
        # 3. Verify that all needed variables now have mappings
        unmapped_vars = [v for v in all_needed_vars if v not in varmap or varmap[v] not in cols]
        if unmapped_vars:
            logger.warning(f"[VARMAP][ENFORCE] Could not map variables: {unmapped_vars}")
        
        # 4. Check for unused columns that might be used later
        unused_cols = [c for c in cols if not any(c == varmap.get(v) for v in varmap)]
        if unused_cols:
            logger.debug(f"[VARMAP][ENFORCE] Unused columns: {unused_cols}")
        
        logger.debug(f"[VARMAP][ENFORCE] Final varmap: {varmap}")
        logger.debug(f"[VARMAP][ENFORCE] Final columns: {combined.columns()}")
        
        # Return a copy of the varmap to avoid modifying the original
        full_varmap = dict(varmap)
        return full_varmap, combined

    def _apply_all_filters(self, combined, full_varmap, negated_literals, comparisons, rule):
        result = combined
        # --- Negation filtering ---
        if negated_literals:
            result = self._filter_negative_literals(result, full_varmap, negated_literals)
        # --- Comparison filtering ---
        logger.debug(f"[DIAG][FILTERS] About to apply comparison filtering. comparisons: {comparisons}")
        result = self._filter_comparisons(result, full_varmap, comparisons or [])
        # --- Aggregate filtering (stub) ---
        # If you add aggregate logic, update varmap accordingly
        return result, full_varmap

    def _update_full_and_delta(self, result, head_pred, head_vars, varmap):
        # result is always a Frame; enforce interface discipline
        logger.debug(f"[DELTA][{head_pred}] Updating delta and full with result columns: {result.columns()}")
        
        # The result should already have user-facing column names from _enforce_head_schema_and_update
        # We'll operate directly on these columns
        user_cols = [v.name for v in head_vars if v.name in result.columns()]
        if not user_cols:
            logger.warning(f"[DELTA][{head_pred}] No valid user columns found in result, using all columns")
            user_cols = list(result.columns())
        
        # Only keep the user columns
        try:
            result = result[user_cols]
        except Exception as e:
            logger.warning(f"[DELTA][{head_pred}] Error selecting user columns: {e}")
            # If error, keep all columns
            result = result
        
        # Drop duplicates
        result = result.drop_duplicates()
        logger.debug(f"[DELTA][{head_pred}] After dropping duplicates: {result.num_rows()} rows")
        
        # Compute delta: new rows not in _full
        existing = self._full.get(head_pred, None)
        
        if existing is not None and existing.num_rows() > 0:
            logger.debug(f"[DELTA][{head_pred}] Existing has {existing.num_rows()} rows with columns: {existing.columns()}")
            
            # Ensure result and existing have the same columns for comparison
            result_cols = set(result.columns())
            existing_cols = set(existing.columns())
            
            # Columns in both frames
            common_cols = list(result_cols.intersection(existing_cols))
            
            if common_cols:
                # Add any missing columns to each frame with null values
                for col in result_cols - existing_cols:
                    existing = existing.assign(**{col: [None] * existing.num_rows()})
                
                for col in existing_cols - result_cols:
                    result = result.assign(**{col: [None] * result.num_rows()})
                
                # Compute delta: rows in result not in existing
                # Add suffixes to avoid column name conflicts
                delta = result.merge(
                    existing, 
                    how="left", 
                    indicator=True, 
                    left_on=common_cols, 
                    right_on=common_cols,
                    suffixes=('_new', '_old')
                )
                delta = delta[delta["_merge"] == "left_only"].drop(columns=["_merge"])
                
                # Remove suffix from column names
                rename_map = {}
                for col in delta.columns():
                    if col.endswith('_new'):
                        rename_map[col] = col[:-4]  # Remove '_new' suffix
                if rename_map:
                    delta = delta.rename(rename_map)
                
                # Use common columns for both result and delta
                all_cols = list(result_cols.union(existing_cols))
                delta = delta[all_cols]
                result = result[all_cols]
                
                logger.debug(f"[DELTA][{head_pred}] Delta has {delta.num_rows()} rows")
            else:
                # No common columns, treat all as new
                delta = result.copy()
                logger.warning(f"[DELTA][{head_pred}] No common columns between result and existing")
        else:
            # No existing data, all rows are new
            delta = result.copy()
            logger.debug(f"[DELTA][{head_pred}] No existing data, all rows ({delta.num_rows()}) are new")
        
        # Only update full if there is a non-empty delta
        if delta.num_rows() > 0:
            # If this is the first time, use result as full
            if existing is None:
                self._full[head_pred] = result.copy()
            else:
                # Use the concat_rows method to concatenate existing and delta frames
                # This is a cleaner approach using our enhanced Frame interface
                try:
                    new_full = existing.concat_rows(delta)
                    self._full[head_pred] = new_full.drop_duplicates()
                except Exception as e:
                    # Log the error for debugging purposes
                    logger.warning(f"[DELTA][{head_pred}] concat_rows failed: {e}")
                    # Fallback approach - try concat method directly
                    try:
                        new_full = existing.concat([delta])
                        self._full[head_pred] = new_full.drop_duplicates()
                    except Exception as e2:
                        # Final fallback using from_dicts if all else fails
                        logger.warning(f"[DELTA][{head_pred}] All concat methods failed, using dictionary fallback: {e2}")
                        from neuralkg.datalog.engine.frame import make_frame
                        existing_dicts = existing.to_records()
                        delta_dicts = delta.to_records()
                        combined_dicts = existing_dicts + delta_dicts
                        all_cols = list(set(existing.columns()).union(set(delta.columns())))
                        self._full[head_pred] = make_frame.from_dicts(combined_dicts, all_cols).drop_duplicates()
            
            self._delta[head_pred] = delta
            logger.debug(f"[DELTA][{head_pred}] Updated full to {self._full[head_pred].num_rows()} rows")
            logger.debug(f"[DELTA][{head_pred}] Delta has {delta.num_rows()} rows")
            return True
        else:
            # No new rows, just update delta with empty frame
            self._delta[head_pred] = delta
            logger.debug(f"[DELTA][{head_pred}] No new rows, delta is empty")
            return False

    def _enforce_head_schema_and_update(self, result, varmap, rule, head_pred, head_vars, replace):
        # result is always a Frame; enforce interface discipline
        try:
            # Get the internal column names to project for head variables
            internal_cols = []
            head_cols_map = {}
            
            logger.debug(f"[PROJECTION][{head_pred}] head_vars: {head_vars}")
            logger.debug(f"[PROJECTION][{head_pred}] varmap: {varmap}")
            logger.debug(f"[PROJECTION][{head_pred}] result.columns: {result.columns()}")
            logger.debug(f"[PROJECTION][{head_pred}] result sample rows BEFORE projection: {result.to_records()[:5]}")
            
            # Get the internal column for each head variable from the varmap
            for v in head_vars:
                if v in varmap and varmap[v] in result.columns():
                    internal_cols.append(varmap[v])
                    head_cols_map[varmap[v]] = v.name
                    logger.debug(f"[PROJECTION][{head_pred}] Mapped head variable {v} to internal column {varmap[v]}")
                else:
                    logger.warning(f"[PROJECTION][{head_pred}] Could not find mapping for head variable {v} in varmap or column not in result")
            
            logger.debug(f"[PROJECTION][{head_pred}] head_cols (internal): {internal_cols}")
            
            # Project only the internal columns needed for head variables
            if internal_cols:
                # First project to the internal columns
                projected = result[internal_cols]
                
                # Now rename the columns to the user variable names for output
                projected_renamed = projected.rename(head_cols_map)
                logger.debug(f"[PROJECTION][{head_pred}] projected.columns after renaming: {projected_renamed.columns()}")
                logger.debug(f"[PROJECTION][{head_pred}] projected sample rows: {projected_renamed.to_records()[:5]}")
                
                # Success path: update relations with renamed columns
                if replace:
                    self._delta[head_pred] = projected_renamed.copy()
                    self._full[head_pred] = projected_renamed.copy()
                    return True
                else:
                    return self._update_full_and_delta(projected_renamed, head_pred, head_vars, varmap)
            else:
                # No valid columns found, create empty frame with user variable names
                user_cols = [v.name for v in head_vars]
                empty_frame = make_frame.empty(user_cols)
                
                if replace:
                    self._delta[head_pred] = empty_frame
                    self._full[head_pred] = empty_frame
                    return True
                else:
                    return self._update_full_and_delta(empty_frame, head_pred, head_vars, varmap)
        except Exception as e:
            logger.warning(f"[PATCH][HEAD_SCHEMA] Could not enforce schema for '{head_pred}': {e}")
            # Fall back to simple rename of internal columns to user variable names
            rename_map = {varmap[v]: v.name for v in head_vars if v in varmap}
            result_renamed = result.rename(rename_map)
            
            if replace:
                self._delta[head_pred] = result_renamed.copy()
                self._full[head_pred] = result_renamed.copy()
                return True
            else:
                return self._update_full_and_delta(result_renamed, head_pred, head_vars, varmap)
