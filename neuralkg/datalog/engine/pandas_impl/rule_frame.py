"""
Rule frame implementation using pandas.

This module provides a pandas-based implementation of the RuleFrame interface
for storing and managing logical rules in a structured tabular format.
"""

import pandas as pd
import json
import uuid
import logging
from typing import Any, Dict, List, Set, Tuple, Optional, Iterator, Union, Iterable
from collections import defaultdict
import networkx as nx

from ..frame_interfaces import RuleFrame, BaseFrame
from ...model.rule import Rule
from ...model.literal import Literal

logger = logging.getLogger(__name__)

class PandasRuleFrame(RuleFrame):
    """
    A RuleFrame implementation using pandas.DataFrame for storing rules.
    
    This implementation serializes Rule objects into a structured DataFrame
    with columns for rule_id, head predicate, body literals, negation info,
    and other rule properties.
    """
    __slots__ = ("_df",)
    
    # Define the columns used in the rule frame
    RULE_ID_COL = "rule_id"
    HEAD_PRED_COL = "head_predicate"
    HEAD_TERMS_COL = "head_terms"
    HEAD_VAR_COLS = "head_variables" 
    BODY_PREDS_COL = "body_predicates"
    BODY_TERMS_COL = "body_terms"
    BODY_VARS_COL = "body_variables"
    NEGATED_LITS_COL = "negated_literals"
    COMPARISON_COL = "comparisons"
    AGGREGATES_COL = "aggregates"
    METADATA_COL = "metadata"
    
    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize with a pandas DataFrame.
        
        Args:
            df: DataFrame containing serialized rules
        """
        self._df = df
        
    @classmethod
    def empty(cls, columns: List[str] = None) -> 'PandasRuleFrame':
        """Create an empty rule frame.
        
        Args:
            columns: Optional list of columns (ignored, included for interface compatibility)
            
        Returns:
            Empty PandasRuleFrame
        """
        columns = [
            cls.RULE_ID_COL,
            cls.HEAD_PRED_COL,
            cls.HEAD_TERMS_COL,
            cls.HEAD_VAR_COLS,
            cls.BODY_PREDS_COL,
            cls.BODY_TERMS_COL,
            cls.BODY_VARS_COL,
            cls.NEGATED_LITS_COL,
            cls.COMPARISON_COL,
            cls.AGGREGATES_COL,
            cls.METADATA_COL
        ]
        df = pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
        return cls(df)
        
    def copy(self) -> 'PandasRuleFrame':
        """Return a copy of this rule frame.
        
        Returns:
            Copy of this PandasRuleFrame
        """
        return PandasRuleFrame(self._df.copy().reset_index(drop=True))
    
    def to_records(self) -> List[Dict[str, Any]]:
        """Convert frame to a list of dictionaries.
        
        Returns:
            List of dictionaries, one per rule
        """
        return self._df.to_dict(orient="records")
    
    def num_rows(self) -> int:
        """Return the number of rows (rules) in the frame.
        
        Returns:
            Number of rules in the frame
        """
        return len(self._df)
    
    def columns(self) -> List[str]:
        """Return the column names in this frame.
        
        Returns:
            List of column names
        """
        return list(self._df.columns)
    
    @classmethod
    def from_dicts(cls, records: List[Dict[str, Any]], columns: List[str] = None) -> 'PandasRuleFrame':
        """Create a frame from a list of dictionaries.
        
        Args:
            records: List of rule records as dictionaries
            columns: Optional list of columns (ignored, mandatory columns are used)
            
        Returns:
            PandasRuleFrame containing the rule records
        """
        if not records:
            return cls.empty()
        
        # Ensure all required columns are present
        for record in records:
            if cls.RULE_ID_COL not in record:
                record[cls.RULE_ID_COL] = str(uuid.uuid4())
            if cls.METADATA_COL not in record:
                record[cls.METADATA_COL] = {}
        
        df = pd.DataFrame(records)
        return cls(df)
    
    @classmethod
    def from_rules(cls, rules: List[Rule]) -> 'PandasRuleFrame':
        """Create a RuleFrame from Rule objects.
        
        Args:
            rules: List of Rule objects
            
        Returns:
            RuleFrame containing the rules
        """
        if not rules:
            return cls.empty()
        
        records = []
        for rule in rules:
            record = cls._serialize_rule(rule)
            records.append(record)
        
        return cls.from_dicts(records)
    
    @classmethod
    def _serialize_rule(cls, rule: Rule) -> Dict[str, Any]:
        """Serialize a Rule object to a dictionary representation.
        
        Args:
            rule: Rule object to serialize
            
        Returns:
            Dictionary representation of the rule
        """
        # Extract head predicate and terms
        head_pred = rule.head.predicate
        head_terms = [str(term) for term in rule.head.terms]
        
        # Identify variables - they are Variable objects, not just strings starting with '?'
        from ...model.terms import Variable
        head_vars = [str(term) for term in rule.head.terms if isinstance(term, Variable)]
        
        # Extract body predicates, terms, and negation info
        body_preds = []
        body_terms = []
        body_vars_set = set()
        negated_indices = []
        
        for i, lit in enumerate(rule.body_literals):
            body_preds.append(lit.predicate)
            lit_terms = [str(term) for term in lit.terms]
            body_terms.append(lit_terms)
            
            # Track variables
            # Identify variables properly
            vars_in_lit = [str(term) for term in lit.terms if isinstance(term, Variable)]
            body_vars_set.update(vars_in_lit)
            
            # Track negated literals
            if getattr(lit, 'negated', False):
                negated_indices.append(i)
        
        # Handle comparisons and aggregates
        comparisons = []
        if hasattr(rule, 'comparisons') and rule.comparisons:
            for comp in rule.comparisons:
                comp_dict = {
                    'left': str(comp.left),
                    'op': comp.op,
                    'right': str(comp.right)
                }
                comparisons.append(comp_dict)
        
        aggregates = []
        if hasattr(rule, 'aggregates') and rule.aggregates:
            for agg in rule.aggregates:
                agg_dict = {
                    'op': agg.op,
                    'var': str(agg.var),
                    'result_var': str(agg.result_var),
                    'group_by': [str(v) for v in agg.group_by]
                }
                aggregates.append(agg_dict)
        
        # Assemble record
        record = {
            cls.RULE_ID_COL: getattr(rule, 'rule_id', str(uuid.uuid4())),
            cls.HEAD_PRED_COL: head_pred,
            cls.HEAD_TERMS_COL: head_terms,
            cls.HEAD_VAR_COLS: head_vars,
            cls.BODY_PREDS_COL: body_preds,
            cls.BODY_TERMS_COL: body_terms,
            cls.BODY_VARS_COL: list(body_vars_set),
            cls.NEGATED_LITS_COL: negated_indices,
            cls.COMPARISON_COL: comparisons,
            cls.AGGREGATES_COL: aggregates,
            cls.METADATA_COL: getattr(rule, 'metadata', {}) or {}
        }
        
        return record
    
    def _deserialize_rule(self, record: Dict[str, Any]) -> Rule:
        """Deserialize a rule record back to a Rule object.
        
        Args:
            record: Dictionary representation of a rule
            
        Returns:
            Reconstructed Rule object
        """
        from ...model.literal import Literal, Comparison, AggregateSpec
        from ...model.terms import Variable, Constant
        
        # Convert head and body variables to sets for quick lookup
        head_vars = set(record[self.HEAD_VAR_COLS])
        body_vars = set(record[self.BODY_VARS_COL])
        
        # Helper function to convert string terms to proper Term objects
        def convert_term(term):
            # If term is in the variables list, create a Variable
            term_str = str(term)
            if term_str in head_vars or term_str in body_vars or \
               (isinstance(term_str, str) and term_str.startswith('?')):
                # Extract variable name - remove '?' prefix if present
                var_name = term_str[1:] if term_str.startswith('?') else term_str
                return Variable(var_name)
            else:
                # Otherwise create a Constant with the original value
                return Constant(term)
        
        # Reconstruct the head literal with proper terms
        head_terms = [convert_term(term) for term in record[self.HEAD_TERMS_COL]]
        head = Literal(record[self.HEAD_PRED_COL], tuple(head_terms))
        
        # Reconstruct body literals with proper terms
        body_literals = []
        for i, (pred, terms) in enumerate(zip(record[self.BODY_PREDS_COL], record[self.BODY_TERMS_COL])):
            body_terms = [convert_term(term) for term in terms]
            negated = i in record[self.NEGATED_LITS_COL]
            lit = Literal(pred, tuple(body_terms), negated=negated)
            body_literals.append(lit)
        
        # Reconstruct comparisons
        comparisons = []
        for comp_dict in record[self.COMPARISON_COL]:
            comp = Comparison(comp_dict['left'], comp_dict['op'], comp_dict['right'])
            comparisons.append(comp)
        
        # Reconstruct aggregates
        aggregates = []
        for agg_dict in record[self.AGGREGATES_COL]:
            agg = AggregateSpec(
                op=agg_dict['op'],
                var=agg_dict['var'],
                result_var=agg_dict['result_var'],
                group_by=agg_dict['group_by']
            )
            aggregates.append(agg)
        
        # Create the rule - handle cases where Rule doesn't accept aggregates parameter
        try:
            # Try with aggregates parameter first
            rule = Rule(
                head=head,
                body_literals=body_literals,
                comparisons=comparisons,
                aggregates=aggregates
            )
        except TypeError:
            # Fall back to older Rule constructor without aggregates
            rule = Rule(
                head=head,
                body_literals=body_literals,
                comparisons=comparisons
            )
            
            # Store aggregates as metadata if available
            if aggregates:
                if not hasattr(rule, 'metadata'):
                    rule.metadata = {}
                rule.metadata['aggregates'] = aggregates
        
        # Add rule_id and metadata
        rule.rule_id = record[self.RULE_ID_COL]
        rule.metadata = record[self.METADATA_COL].copy() if record[self.METADATA_COL] else {}
        
        return rule
        
    def to_rules(self) -> List[Rule]:
        """Convert the RuleFrame back to Rule objects.
        
        Returns:
            List of Rule objects
        """
        return [self._deserialize_rule(record) for record in self.to_records()]
    
    def filter_by_head(self, predicate: str) -> 'PandasRuleFrame':
        """Filter rules by head predicate name.
        
        Args:
            predicate: Head predicate name to filter by
            
        Returns:
            RuleFrame with only rules having the specified head predicate
        """
        mask = self._df[self.HEAD_PRED_COL] == predicate
        return PandasRuleFrame(self._df[mask].reset_index(drop=True))
    
    def filter_by_body(self, predicate: str) -> 'PandasRuleFrame':
        """Filter rules that have a given predicate in their body.
        
        Args:
            predicate: Body predicate name to filter by
            
        Returns:
            RuleFrame with only rules having the specified predicate in their body
        """
        # Use pandas' apply to check if predicate is in the body_predicates list
        mask = self._df[self.BODY_PREDS_COL].apply(lambda preds: predicate in preds)
        return PandasRuleFrame(self._df[mask].reset_index(drop=True))
    
    def get_dependencies(self) -> Dict[str, Set[str]]:
        """Get predicate dependency map (which predicates depend on which).
        
        Returns:
            Dictionary mapping from each head predicate to the set of predicates it depends on
        """
        dependencies = defaultdict(set)
        
        for _, row in self._df.iterrows():
            head_pred = row[self.HEAD_PRED_COL]
            body_preds = set(row[self.BODY_PREDS_COL])
            dependencies[head_pred].update(body_preds)
        
        return dict(dependencies)
    
    def has_negation(self) -> bool:
        """Check if any rules contain negated literals.
        
        Returns:
            True if at least one rule contains negation, False otherwise
        """
        return any(len(neg_indices) > 0 for neg_indices in self._df[self.NEGATED_LITS_COL])
    
    def stratify(self) -> List['PandasRuleFrame']:
        """Stratify rules based on dependencies and negation.
        
        Returns:
            List of RuleFrames, one per stratum, ordered from lowest to highest stratum
        """
        # Get dependency graph
        dependencies = self.get_dependencies()
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add all predicates as nodes
        all_preds = set(dependencies.keys())
        for deps in dependencies.values():
            all_preds.update(deps)
        
        for pred in all_preds:
            G.add_node(pred)
        
        # Add edges from head to body predicates
        for head_pred, body_preds in dependencies.items():
            for body_pred in body_preds:
                G.add_edge(head_pred, body_pred)
        
        # Add negation edges (with special weight)
        for _, row in self._df.iterrows():
            head_pred = row[self.HEAD_PRED_COL]
            negated_indices = row[self.NEGATED_LITS_COL]
            if negated_indices:
                body_preds = row[self.BODY_PREDS_COL]
                for neg_idx in negated_indices:
                    neg_pred = body_preds[neg_idx]
                    # Set weight=2 for negation edges (regular edges have default weight=1)
                    # This helps distinguish negation edges when detecting strongly connected components
                    G.add_edge(head_pred, neg_pred, weight=2)
        
        # Check for strongly connected components with negation edges
        # If found, the program may not be stratifiable
        sccs = list(nx.strongly_connected_components(G))
        
        for scc in sccs:
            if len(scc) > 1:  # Non-trivial SCC
                # Check if the SCC contains a negation edge
                for u in scc:
                    for v in G.successors(u):
                        if v in scc and G.get_edge_data(u, v, {}).get('weight', 1) > 1:
                            logger.warning(f"Found negation cycle in rules: {scc}")
                            # Program may not be stratifiable, but we'll continue anyway
        
        # Do topological sort to get strata
        try:
            strata_order = list(nx.topological_sort(G))
            strata_order.reverse()  # Reverse so that independent predicates come first
        except nx.NetworkXUnfeasible:
            # Graph has cycles, so use SCCs as strata
            strata_order = []
            for scc in sccs:
                strata_order.extend(list(scc))
        
        # Group rules by stratum
        strata = []
        visited_rules = set()
        
        for pred in strata_order:
            # Get rules with this predicate in the head
            stratum_rules = []
            mask = self._df[self.HEAD_PRED_COL] == pred
            for idx, row in self._df[mask].iterrows():
                rule_id = row[self.RULE_ID_COL]
                if rule_id not in visited_rules:
                    stratum_rules.append(self._deserialize_rule(row))
                    visited_rules.add(rule_id)
            
            if stratum_rules:
                strata.append(PandasRuleFrame.from_rules(stratum_rules))
        
        # Add any rules not yet visited (should not happen, but just in case)
        remaining_rules = []
        for idx, row in self._df.iterrows():
            rule_id = row[self.RULE_ID_COL]
            if rule_id not in visited_rules:
                remaining_rules.append(self._deserialize_rule(row))
                visited_rules.add(rule_id)
        
        if remaining_rules:
            strata.append(PandasRuleFrame.from_rules(remaining_rules))
        
        return strata
    
    def get_rules_for_predicate(self, predicate: str) -> 'PandasRuleFrame':
        """Get all rules that define a specific predicate.
        
        Args:
            predicate: Predicate name to filter by
            
        Returns:
            RuleFrame with only rules defining the specified predicate
        """
        return self.filter_by_head(predicate)
    
    def set_metadata(self, rule_id: str, key: str, value: Any) -> 'PandasRuleFrame':
        """Set metadata for a specific rule.
        
        Args:
            rule_id: ID of the rule to modify
            key: Metadata key
            value: Metadata value
            
        Returns:
            Updated RuleFrame
        """
        df = self._df.copy()
        mask = df[self.RULE_ID_COL] == rule_id
        
        if not mask.any():
            logger.warning(f"Rule ID {rule_id} not found in RuleFrame")
            return self.copy()
        
        # Update metadata dictionary for the rule
        for idx, row in df[mask].iterrows():
            metadata = row[self.METADATA_COL].copy() if row[self.METADATA_COL] else {}
            metadata[key] = value
            df.at[idx, self.METADATA_COL] = metadata
        
        return PandasRuleFrame(df)
    
    def get_metadata(self, rule_id: str, key: str) -> Any:
        """Get metadata for a specific rule.
        
        Args:
            rule_id: ID of the rule
            key: Metadata key
            
        Returns:
            Value of the metadata key, or None if not set
        """
        mask = self._df[self.RULE_ID_COL] == rule_id
        if not mask.any():
            return None
        
        row = self._df[mask].iloc[0]
        metadata = row[self.METADATA_COL]
        
        if metadata and key in metadata:
            return metadata[key]
        return None
        
    def concat(self, other: 'RuleFrame') -> 'PandasRuleFrame':
        """Concatenate this RuleFrame with another.
        
        Args:
            other: Another RuleFrame to concatenate with this one
            
        Returns:
            Combined PandasRuleFrame
        """
        if not isinstance(other, PandasRuleFrame):
            # Convert to PandasRuleFrame if it's another implementation
            rules = other.to_rules()
            other = PandasRuleFrame.from_rules(rules)
            
        # Concatenate the underlying DataFrames
        result_df = pd.concat([self._df, other._df], ignore_index=True)
        
        # Deduplicate rules by rule_id
        result_df = result_df.drop_duplicates(subset=[self.RULE_ID_COL], keep='last')
        
        return PandasRuleFrame(result_df)
