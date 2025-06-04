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

    def _evaluate_positive_body(
        self, body_literals: list[Literal], comparisons: list[Comparison] = None
    ) -> tuple[Frame, dict]:
        """Evaluate the positive portion of a rule body.

        Args:
            body_literals: list[Literal]
            comparisons: list[Comparison] (optional)

        Returns:
            Tuple of (result frame, varmap)
        """
        # Filter out aggregate predicates - they will be processed later
        regular_literals = [lit for lit in body_literals if lit.predicate not in AGGREGATE_FUNCS]
        logger.debug(f"[FLOW][POSBODY] Entered _evaluate_positive_body for body_literals: {body_literals} and comparisons: {comparisons}")
        logger.debug(f"[EVAL][POSBODY][START] Body literals: {regular_literals}, Comparisons: {comparisons}")
        
        if len(regular_literals) == 0:
            logger.debug(f"[FLOW][POSBODY] No regular (non-aggregate) body literals, returning all-true dummy frame.")
            dummy = make_frame.empty([])
            logger.debug(f"[FLOW][POSBODY] Exiting _evaluate_positive_body with columns: {dummy.columns()} and varmap: {{}}")
            return dummy, {}
        
        # Get all literal solutions and set up variable-to-column mapping
        tables = []
        var_maps = []  # Variable-to-column mapping for each literal
        
        # STEP 1: Prepare tables for each literal with proper renaming and filtering
        for idx, lit in enumerate(regular_literals):
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
        """Filter result based on negated literals.

        Parameters
        ----------
        result : Frame
            Current result table.
        varmap : dict
            Mapping from variables to columns.
        negated_literals : list
            List of negated literals.

        Returns
        -------
        Frame
            Filtered result.
        """
        # For each negated literal, perform anti-join filtering
        orig_columns = result.columns()  # Remember original columns for later
        
        for lit in negated_literals:
            # Skip if result is empty, nothing to filter
            if result.num_rows() == 0:
                logger.debug(f"[NEG] Skipping negation filtering for {lit}, result is empty")
                continue
                
            logger.debug(f"[NEG][FILTER] Filtering by negated literal: {lit}")

            # Get the full relation for the negated literal
            full_tbl = self._get_full(lit.predicate)
            if full_tbl is None:
                logger.warning(f"[NEG][WARN] Negated predicate {lit.predicate} not found in database")
                continue
            
            # Get the predicate signature (arity)
            pred_sig = [f"arg{i}" for i in range(lit.arity())]
            
            # If the full table is empty, nothing is excluded by negation
            if full_tbl.num_rows() == 0:
                logger.debug(f"[NEG][EMPTY] Negated table {lit.predicate} is empty, no filtering needed")
                continue
            
            # Extract shared variables between result and negated relation
            # These are variables that appear in both positive and negated literals
            shared_vars = []
            join_cols_left = []  # Columns from result table
            join_cols_right = []  # Columns from negated table
            
            # Get the existing column names of the full table
            existing_colnames = full_tbl.columns()
            logger.debug(f"[NEG][JOIN] Negated table original columns: {existing_colnames}")
            
            # We'll use the existing column names from the full table
            # No need to rename at this stage, we'll map when creating join conditions
            
            # Map variables in the negated literal to their position in the relation
            var_positions = {}
            for i, term in enumerate(lit.terms):
                if isinstance(term, Variable):
                    var_positions[term] = i
            
            # CRITICAL FIX: For each variable in the negated literal, only use it for joining
            # if it's in the result columns (meaning it appeared in positive literals)
            for term, pos in var_positions.items():
                if term in varmap:
                    col_name = varmap[term]  # Column name in result table
                    # Only use variables that are actually in the result columns from positive literals
                    if col_name in result.columns() and not col_name.startswith('var_') and pos < len(existing_colnames):
                        shared_vars.append(term)
                        join_cols_left.append(col_name)  # Column in result table
                        join_cols_right.append(existing_colnames[pos])  # Actual column in negated table
                        logger.debug(f"[NEG][JOIN] Found shared variable {term} for join: {col_name} = {existing_colnames[pos]}")
                    else:
                        logger.debug(f"[NEG][JOIN] Skipping variable {term}: not in result columns, synthetic column, or position out of range")
            
            # If no shared variables, there's nothing to filter on
            if not shared_vars:
                logger.warning(f"[NEG][WARN] No shared variables between result and negated literal {lit}")
                if full_tbl.num_rows() > 0:
                    # This is a ground fact - if present, it negates everything
                    logger.debug(f"[NEG][WARN] Negated relation has rows but no join keys. Result will be empty.")
                    return result.from_records([])  # Return empty frame
                continue
            
            logger.debug(f"[NEG][JOIN] Anti-join on: left cols={join_cols_left}, right cols={join_cols_right}")
            logger.debug(f"[NEG][JOIN] Left (result) sample: {result.to_records()[:3]}")
            logger.debug(f"[NEG][JOIN] Right (negated) sample: {full_tbl.to_records()[:3]}")
            
            # Perform anti-join: left outer join with indicator and keep rows marked as 'left_only'
            merged = result.merge(
                full_tbl,
                how="left",
                left_on=join_cols_left,
                right_on=join_cols_right,
                indicator=True
            )
            
            # Keep only rows where there was no match in the negated relation
            filtered = merged.filter_equals("_merge", "left_only")
            logger.debug(f"[NEG][JOIN] Filtered rows: {filtered.num_rows()} out of original {result.num_rows()}")
            
            # Keep only the original columns from our result table to clean up the result
            result_cols = [col for col in orig_columns if col in filtered.columns()]
            if filtered.num_rows() > 0 and result_cols:
                result = filtered[result_cols]
                logger.debug(f"[NEG][JOIN] After projection: {result.num_rows()} rows, columns={result.columns()}")
                logger.debug(f"[NEG][JOIN] Sample after negation: {result.to_records()[:3]}")
            else:
                # If filtering resulted in empty result, return empty frame
                logger.debug(f"[NEG][JOIN] No results after negation filtering")
                result = filtered.from_records([])  # Empty frame with same columns
                
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

    def _filter_aggregates(self, result: Frame, varmap: dict, rule) -> tuple[Frame, dict]:
        """
        Apply aggregation operations to the result Frame, updating
        varmap accordingly.
        
        Args:
            result: Frame to aggregate
            varmap: mapping of Variable -> column name
            rule: the Rule being evaluated
        
        Returns:
            Tuple of (aggregated_frame, updated_varmap)
        """
        
        # Extract aggregation predicates from rule
        aggregates = []
        for lit in rule.body_literals:
            if not lit.negated and lit.predicate in AGGREGATE_FUNCS:
                aggregates.append(lit)
        
        if not aggregates:
            return result, varmap
            
        # If we have aggregates but empty input, return empty frame with result structure
        if result.num_rows() == 0:
            # Create an empty frame with the expected structure
            result_cols = list(varmap.values()) + [agg.terms[-1].name for agg in aggregates]
            return make_frame.empty(result_cols), varmap
            
        # For each aggregate predicate, compute the aggregation
        for agg_lit in aggregates:
            logger.debug(f"[AGGREGATE] Processing aggregate: {agg_lit}")
            agg_func = AGGREGATE_FUNCS[agg_lit.predicate]
            
            # Last arg is result variable, second-to-last is value variable
            # All preceding args are grouping variables
            result_var = agg_lit.terms[-1]
            value_var = agg_lit.terms[-2]
            group_vars = agg_lit.terms[:-2]
            
            logger.debug(f"[AGGREGATE] Result var: {result_var}, Value var: {value_var}, Group vars: {group_vars}")
            
            # Convert variables to column names using varmap
            if value_var not in varmap:
                logger.error(f"[AGGREGATE] Value variable {value_var} not in varmap {varmap}")
                continue
                
            value_col = varmap[value_var]
            group_cols = []
            
            for var in group_vars:
                if var not in varmap:
                    logger.error(f"[AGGREGATE] Group variable {var} not in varmap {varmap}")
                    continue
                group_cols.append(varmap[var])
            
            logger.debug(f"[AGGREGATE] Value column: {value_col}, Group columns: {group_cols}")
            
            # Group by group_cols and apply agg_func to value_col
            logger.debug(f"[AGGREGATE] Input frame rows: {result.num_rows()}, columns: {result.columns()}")
            
            # Prepare result structures
            agg_results = []
            result_cols = list(group_cols)
            result_cols.append(result_var.name)
            
            try:
                # Get all rows as dictionaries
                records = result.to_records()
                
                if group_cols:
                    # Manual grouping by group_cols
                    groups = {}
                    for record in records:
                        # Create a group key from group column values
                        group_key = tuple(record[col] for col in group_cols)
                        # Initialize group if not seen before
                        if group_key not in groups:
                            groups[group_key] = []
                        # Add value to group
                        groups[group_key].append(record[value_col])
                    
                    # Apply aggregation to each group
                    for group_key, values in groups.items():
                        row = {}
                        # Add group columns to result
                        for i, col in enumerate(group_cols):
                            row[col] = group_key[i]
                        # Add aggregated value
                        row[result_var.name] = agg_func(values)
                        agg_results.append(row)
                else:
                    # No grouping, aggregate all values
                    values = [record[value_col] for record in records]
                    agg_value = agg_func(values)
                    agg_results.append({result_var.name: agg_value})
            except Exception as e:
                logger.error(f"[AGGREGATE] Error during aggregation: {e}")
                return result, varmap
                
            # Create result frame and update varmap
            logger.debug(f"[AGGREGATE] Creating result frame with columns {result_cols} from {agg_results}")
            result = make_frame.from_dicts(agg_results, result_cols)
            
            # Create new varmap containing only group variables and result variable
            new_varmap = {}
            
            # Add group variables to the new varmap with their column names in the result frame
            for i, var in enumerate(group_vars):
                new_varmap[var] = group_cols[i]
            
            # Add result variable to the new varmap
            new_varmap[result_var] = result_var.name
            
            logger.debug(f"[AGGREGATE] Old varmap: {varmap}")
            logger.debug(f"[AGGREGATE] New varmap: {new_varmap}")
            
            # Replace the varmap with the new one
            varmap = new_varmap
            
            # Log which variables are available after aggregation
            logger.debug(f"[AGGREGATE] Available variables after aggregation: {set(varmap.keys())}")
            
            # Store available variables in rule properties
            rule.has_aggregates = True
            rule.available_vars_after_agg = set(varmap.keys())
            
            logger.debug(f"[AGGREGATE] Updated rule properties: has_aggregates={rule.has_aggregates}, available_vars_after_agg={rule.available_vars_after_agg}")
            
            # Continue with next aggregate predicate
        return result, varmap

    def _evaluate_rule(self, rule, replace=False):
        '''Evaluate a single rule and return whether the head relation changed.'''
        logger.debug(f"[EVAL][RULE] Evaluating rule: {rule.head.predicate} :- {[str(lit) for lit in getattr(rule, 'body_literals', [])]}")
        
        # Step 1: Find positive literals that aren't aggregates
        positive_body_literals = []
        agg_predicates = []
        for lit in rule.body_literals:
            if not lit.negated and lit.predicate not in AGGREGATE_FUNCS:
                positive_body_literals.append(lit)
            elif not lit.negated and lit.predicate in AGGREGATE_FUNCS:
                agg_predicates.append(lit.predicate)
                rule.has_aggregates = True
        
        # Step 2: Evaluate positive body without aggregates
        result, varmap = self._evaluate_positive_body(positive_body_literals)
        
        # Early exit if empty result -- rule can't fire
        if result.num_rows() == 0:  
            logger.debug(f"[EVAL][RULE][EARLY EXIT] Empty positive body result")
            return False
            
        head_pred = rule.head.predicate
        head_vars = rule.head.terms
        
        # 3. Identify any negative literals and comparison literals
        negated_literals = [lit for lit in rule.body_literals if getattr(lit, 'negated', False)]
        comparisons = rule.comparisons if hasattr(rule, 'comparisons') else []
        
        logger.debug(f"[EVAL][RULE][AFTER_POSBODY] Head: {head_pred}, Combined shape: {result.num_rows()}, Columns: {result.columns()}, Sample: {result.to_records()[:5]}")

        # 4. Ensure all needed variables are mapped and present as columns
        full_varmap, result = self._enforce_variable_column_mapping(
            result, varmap, head_vars, negated_literals, comparisons
        )

        # 3. Apply all filters (negation, comparison, aggregate)
        logger.debug(f"[DIAG][RULE:{head_pred}] [PRE-FILTERS] comparisons: {comparisons}")
        result, full_varmap = self._apply_all_filters(
            result, full_varmap, negated_literals, comparisons, rule
        )
        logger.debug(f"[DIAG][RULE:{head_pred}] [POST-FILTERS] columns: {result.columns()}")

        # 4. Project/rename to head variables
        pred_tag = f"[DIAG][RULE:{head_pred}]"
        cols = result.columns()
        logger.debug(f"{pred_tag} [PRE-PROJECT] Columns: {cols}")
        logger.debug(f"{pred_tag} [PRE-PROJECT] Sample rows: {result.to_records()[:5]}")
        logger.debug(f"{pred_tag} [PRE-PROJECT] varmap: {full_varmap}")
        # Diagnostic: log projection mapping and DataFrame state for all predicates
        logger.debug(f"{pred_tag} [PROJECT][START] varmap before projection: {full_varmap}")
        logger.debug(f"{pred_tag} [PROJECT][START] columns before projection: {cols}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] head_vars: {head_vars}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] varmap: {full_varmap}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] result.columns: {cols}")
        logger.debug(f"[DIAG][PROJECTION][{head_pred}] result sample: {result.to_records()[:5]}")
        # PATCH: For base case (one body literal, all head vars distinct, all body terms are variables), set varmap by position
        body_literals = getattr(rule, 'body_literals', None) or getattr(rule, 'body', [])
        if (
            len(body_literals) == 1
            and len(head_vars) == len(set(head_vars))
            and len(head_vars) > 0  # Ensure head_vars is not empty
            and not rule.has_aggregates  # Don't override varmap if we have aggregates
            and all(isinstance(t, type(head_vars[0])) for t in body_literals[0].terms)
            and all(isinstance(t, type(head_vars[0])) for t in head_vars)
        ):
            # Map head_vars[i] to result.columns()[i] by position
            full_varmap = {hv: cols[i] for i, hv in enumerate(head_vars) if i < len(cols)}
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] Overriding varmap for base case: {full_varmap}")
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] DF before projection: columns={cols}, sample_rows={result.to_records()[:5]}")
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] head_vars: {head_vars}")
            logger.debug(f"[PATCH][PROJECTION][{head_pred}] varmap: {full_varmap}")
        updated = self._enforce_head_schema_and_update(
            result, full_varmap, rule, head_pred, head_vars, replace
        )
        full = self._full.get(head_pred, None)

        if full is not None:
            cols = full.columns() if callable(full.columns) else full.columns
            logger.debug(f"{pred_tag} [FULL UPDATE] Columns: {cols}")
            logger.debug(f"{pred_tag} [FULL UPDATE] Sample rows: {full.to_records()[:10]}")
            logger.debug(f"{pred_tag} [FULL UPDATE] num_rows: {full.num_rows()}")
        return updated

    def _enforce_variable_column_mapping(self, result, varmap, head_vars, negated_literals=None, comparisons=None):
        """Ensure that all variables in head_vars and comparisons are mapped to columns.
        This may require creating new columns with null values in some cases.
        
        Args:
            result (Frame): The current partial result
            varmap (dict): Current mapping of Variables to column names
            head_vars (list): Variables in the head that need to be preserved
            negated_literals (list, optional): Negated literals in the rule
            comparisons (list, optional): Comparison literals in the rule
            
        Returns:
            tuple: (updated_varmap, updated_result)
        """
        varmap = dict(varmap)  # Make a copy to avoid modifying the original
        negated_literals = negated_literals or []
        comparisons = comparisons or []
        
        # Log the original state for debugging
        logger.debug(f"[VARMAP] Starting enforcement with varmap: {varmap}")
        logger.debug(f"[VARMAP] Result columns: {result.columns()}")
        logger.debug(f"[VARMAP] Head vars: {head_vars}")
        
        # Collect all variables that need to be in the result
        all_needed_vars = set()        
        
        # Add head variables
        for var in head_vars:
            # Skip dummy variables or non-variable terms
            if var.is_variable() and var.name != "__dummy__":
                all_needed_vars.add(var)
        
        # Add variables from comparisons
        for comp in comparisons:
            # For each term in the comparison that is a variable
            for term in [comp.left, comp.right]:
                if hasattr(term, 'is_variable') and term.is_variable():
                    all_needed_vars.add(term)
                    
        # Add variables from negated literals that also appear in varmap
        # (only the shared variables need to exist in both positive and negative literals)
        for lit in negated_literals:
            for term in lit.terms:
                if hasattr(term, 'is_variable') and term.is_variable():
                    for v in varmap:
                        if term == v:
                            all_needed_vars.add(term)
                            break
        
        logger.debug(f"[VARMAP] All needed vars: {all_needed_vars}")
        
        # First make sure all columns in the result have a variable mapping
        # This helps ensure we don't lose data when we map variables to columns
        result_cols = result.columns()
        for col in result_cols:
            # If this column isn't mapped to any variable, create a mapping
            if col != '__dummy__' and col not in varmap.values():
                # For columns that seem to be actual variables and not synthetic ones
                if not col.startswith('var_'):
                    # Find a variable to map to this column or create one
                    matching_var = None
                    for var in all_needed_vars:
                        if var.name == col and var not in varmap:
                            matching_var = var
                            break
                    
                    if not matching_var:
                        # Create a new variable for this column
                        # Variable is already imported at the top of the file
                        matching_var = Variable(col)
                        
                    varmap[matching_var] = col
                    logger.debug(f"[VARMAP] Mapped existing column {col} to variable {matching_var}")
        
        # Find variables that are not yet mapped to columns
        unmapped_vars = set()
        for var in all_needed_vars:
            if var not in varmap or varmap[var] not in result.columns():
                unmapped_vars.add(var)
                
        if unmapped_vars:
            logger.debug(f"[VARMAP] Found {len(unmapped_vars)} unmapped variables: {unmapped_vars}")
            
            # Determine which unmapped vars are likely aggregate result variables 
            # (those end with Sum, Count, Amount, etc.)
            agg_result_vars = set()
            for var in unmapped_vars:
                var_name = var.name
                if any(var_name.endswith(suffix) for suffix in ["Sum", "Count", "Avg", "Min", "Max", "Total", "Amount"]):
                    agg_result_vars.add(var)
                    logger.debug(f"[VARMAP] Detected aggregate result variable: {var}")
                    
            for var in unmapped_vars:
                # Skip warning for special cases to reduce noise
                if var.name == "__dummy__" or var in agg_result_vars:
                    new_col_name = var.name
                else:
                    # For head variables that are not mapped, show a warning
                    if var in head_vars:
                        logger.warning(f"[VARMAP] Head variable {var} not mapped to any column. Creating new column.")
                    else:
                        logger.debug(f"[VARMAP] Variable {var} not mapped to any column. Creating new column.")
                    new_col_name = f"{var.name}"
                
                # Add a new column with null/NA values
                logger.debug(f"[VARMAP] Adding column {new_col_name} to result frame")
                varmap[var] = new_col_name
                # Use assign to add a new column (compatible with Frame interface)
                result = result.assign(**{new_col_name: None})
        
        logger.debug(f"[VARMAP] Final varmap: {varmap}")
        logger.debug(f"[VARMAP] Final result columns: {result.columns()}")
        
        return varmap, result

    def _apply_all_filters(self, combined, full_varmap, negated_literals, comparisons, rule):
        result = combined
        # --- Negation filtering ---
        if negated_literals:
            result = self._filter_negative_literals(result, full_varmap, negated_literals)
        # --- Comparison filtering ---
        logger.debug(f"[DIAG][FILTERS] About to apply comparison filtering. comparisons: {comparisons}")
        result = self._filter_comparisons(result, full_varmap, comparisons or [])
        # --- Aggregate filtering ---
        # Apply aggregations and update varmap accordingly
        result, full_varmap = self._filter_aggregates(result, full_varmap, rule)
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

    def _project_to_head(self, rule, result, varmap):
        head_vars = rule.head.terms
        logger.debug(f"[PROJECTION][{rule.head.predicate}] head_vars: {head_vars}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] varmap: {varmap}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] result.columns: {result.columns()}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] result sample: {result.to_records()[:2] if result.num_rows() > 0 else []}")

        if result.num_rows() == 0:
            # Create an empty frame with the right structure
            schema = [t.name for t in head_vars]
            return make_frame.empty(schema)
            
        # If we have processed aggregates, we need to filter the head vars to only those available
        filtered_head_vars = head_vars
        # Check if this rule has aggregates and filter head vars accordingly
        if rule.has_aggregates:
            available_vars = rule.available_vars_after_agg
            if available_vars:
                filtered_head_vars = [var for var in head_vars if var in available_vars]
                logger.debug(f"[PROJECTION][{rule.head.predicate}] Filtered head_vars due to aggregation: {filtered_head_vars}")

        logger.debug(f"[PROJECTION][{rule.head.predicate}] head_vars: {filtered_head_vars}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] varmap: {varmap}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] result.columns: {result.columns()}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] result sample rows BEFORE projection: {result.to_records()[:2] if result.num_rows() > 0 else []}")

        head_cols = []
        head_var_names = []

        for var in filtered_head_vars:
            if var.is_variable():
                if var in varmap:
                    head_cols.append(varmap[var])
                    head_var_names.append(var.name)
                else:
                    # Variable not found in varmap, use a dummy value
                    result = result.assign(**{var.name: None})
                    head_cols.append(var.name)
                    head_var_names.append(var.name)

        logger.debug(f"[PROJECTION][{rule.head.predicate}] head_cols (internal): {head_cols}")

        # Select the relevant columns and rename to external names
        projected = result.rename({src: dst for src, dst in zip(head_cols, head_var_names)})

        logger.debug(f"[PROJECTION][{rule.head.predicate}] projected.columns after renaming: {projected.columns()}")
        logger.debug(f"[PROJECTION][{rule.head.predicate}] projected sample rows: {projected.to_records()[:2] if projected.num_rows() > 0 else []}")

        return projected
        
    def _enforce_head_schema_and_update(self, result, varmap, rule, head_pred, head_vars, replace):
        """Project and rename columns in result to match head variable names, then update relations.
        
        Args:
            result: Combined Frame with query results
            varmap: Mapping from Variables to column names
            rule: The Rule being evaluated
            head_pred: Head predicate name
            head_vars: List of Variable terms in head
            replace: If True, overwrite existing relation; otherwise update incrementally
            
        Returns:
            bool: True if relation was modified, False otherwise
        """
        try:
            logger.debug(f"[PROJECTION][{head_pred}] head_vars: {head_vars}")
            logger.debug(f"[PROJECTION][{head_pred}] varmap: {varmap}")
            logger.debug(f"[PROJECTION][{head_pred}] result.columns: {result.columns()}")
            logger.debug(f"[PROJECTION][{head_pred}] result sample rows BEFORE projection: {result.to_records()[:5]}")
            
            # Special case for ground queries with "proved" variable
            if len(head_vars) == 1 and head_vars[0].name == 'proved':
                logger.debug(f"[PROJECTION][{head_pred}] Handling ground query result")
                if result.num_rows() > 0:
                    # Create a frame with a single row and "true" value
                    projected = make_frame.from_dicts([{"proved": True}], ["proved"])
                    logger.debug(f"[PROJECTION][{head_pred}] Ground query proved true, rows: {projected.num_rows()}, columns: {projected.columns()}")
                else:
                    # Empty result means the query is not proved
                    projected = make_frame.empty(['proved'])
                    logger.debug(f"[PROJECTION][{head_pred}] Ground query not proved, returning empty frame")
                    
                if replace:
                    self._delta[head_pred] = projected.copy()
                    self._full[head_pred] = projected.copy()
                    return True
                else:
                    return self._update_full_and_delta(projected, head_pred, head_vars, varmap)
            
            # Special case for __dummy__ - we need to add meaningful columns instead
            elif len(head_vars) == 1 and head_vars[0].name == '__dummy__':
                if result.num_rows() > 0:
                    # Get all the meaningful columns (excluding synthetic ones)
                    cols = result.columns()
                    relevant_cols = [c for c in cols if not c.startswith('var_')]
                    
                    # If the only meaningful column is __dummy__, just use it
                    if not relevant_cols or (len(relevant_cols) == 1 and relevant_cols[0] == '__dummy__'):
                        projected = result
                    else:
                        # Project to the meaningful columns
                        projected = result[relevant_cols]
                        
                    logger.debug(f"[PROJECTION][{head_pred}] Special __dummy__ case, using columns: {projected.columns()}")
                else:
                    # Empty result with __dummy__ variable
                    projected = make_frame.empty(['__dummy__'])
                    
                if replace:
                    self._delta[head_pred] = projected.copy()
                    self._full[head_pred] = projected.copy()
                    return True
                else:
                    return self._update_full_and_delta(projected, head_pred, head_vars, varmap)
            
            # Normal case: map head variables to columns
            internal_cols = []
            head_cols_map = {}
            user_cols = []
            
            for v in head_vars:
                user_cols.append(v.name)
                if v in varmap and varmap[v] in result.columns():
                    internal_cols.append(varmap[v])
                    head_cols_map[varmap[v]] = v.name
                    logger.debug(f"[PROJECTION][{head_pred}] Mapped head variable {v} to internal column {varmap[v]}")
                else:
                    logger.warning(f"[PROJECTION][{head_pred}] Could not find mapping for head variable {v} in varmap or column not in result")
            
            logger.debug(f"[PROJECTION][{head_pred}] head_cols (internal): {internal_cols}")
            logger.debug(f"[PROJECTION][{head_pred}] user_cols: {user_cols}")
            
            if internal_cols:
                # Project only to columns we need
                projected = result[internal_cols]
                
                # Rename columns to match user variable names
                projected_renamed = projected.rename(head_cols_map)
                logger.debug(f"[PROJECTION][{head_pred}] projected.columns after renaming: {projected_renamed.columns()}")
                logger.debug(f"[PROJECTION][{head_pred}] sample: {projected_renamed.to_records()[:5]}")
                
                if replace:
                    self._delta[head_pred] = projected_renamed.copy()
                    self._full[head_pred] = projected_renamed.copy()
                    return True
                else:
                    return self._update_full_and_delta(projected_renamed, head_pred, head_vars, varmap)
            else:
                # No valid columns found, create empty frame with user variable names
                empty_frame = make_frame.empty(user_cols)
                
                if replace:
                    self._delta[head_pred] = empty_frame
                    self._full[head_pred] = empty_frame
                    return True
                else:
                    return self._update_full_and_delta(empty_frame, head_pred, head_vars, varmap)
                    
        except Exception as e:
            logger.warning(f"[PATCH][HEAD_SCHEMA] Could not enforce schema for '{head_pred}': {e}")
            
            # Fallback: use a simple rename based on varmap
            try:
                rename_map = {}
                for v in head_vars:
                    if v in varmap:
                        rename_map[varmap[v]] = v.name
                
                # If we can't remap columns, try to at least select the best columns from what's available
                all_cols = result.columns()
                if not rename_map and all_cols:
                    cols_to_keep = [c for c in all_cols if not (c.startswith('var_') or c == '__dummy__')]
                    if cols_to_keep:
                        result = result[cols_to_keep]
                    
                result_renamed = result.rename(rename_map) if rename_map else result
                
                if replace:
                    self._delta[head_pred] = result_renamed.copy()
                    self._full[head_pred] = result_renamed.copy() 
                    return True
                else:
                    return self._update_full_and_delta(result_renamed, head_pred, head_vars, varmap)
            except Exception as e2:
                logger.error(f"[PATCH][HEAD_SCHEMA] Even fallback failed for '{head_pred}': {e2}")
                return False
