"""
Frame interfaces for the Datalog engine.

This module defines the base interfaces for different types of frames used in the Datalog engine:
- BaseFrame: Common base interface for all frame types
- FactFrame: Interface for frames that store facts (relations)
- RuleFrame: Interface for frames that store logical rules
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Set, Tuple, Iterable, Union

class BaseFrame(ABC):
    """Abstract base class for all frame-like structures."""
    
    @abstractmethod
    def copy(self):
        """Return a copy of this frame."""
        pass
    
    @abstractmethod
    def to_records(self) -> List[Dict[str, Any]]:
        """Convert frame to a list of dictionaries."""
        pass
    
    @abstractmethod
    def num_rows(self) -> int:
        """Return the number of rows in the frame."""
        pass
    
    @abstractmethod
    def columns(self) -> List[str]:
        """Return the column names in this frame."""
        pass
    
    @classmethod
    @abstractmethod
    def empty(cls, columns: List[str]):
        """Create an empty frame with the given columns."""
        pass
    
    @classmethod
    @abstractmethod
    def from_dicts(cls, records: List[Dict[str, Any]], columns: List[str] = None):
        """Create a frame from a list of dictionaries."""
        pass


class FactFrame(BaseFrame):
    """Interface for frame structures that store facts/relations."""
    
    @abstractmethod
    def filter(self, column: str, op: str, value: Any) -> 'FactFrame':
        """
        Filter the frame by a column condition.
        
        Args:
            column: Column name to filter on
            op: Operator (e.g., "==", "<", ">", "!=", "in", "not in")
            value: Value to compare with
            
        Returns:
            Filtered FactFrame
        """
        pass
    
    @abstractmethod
    def filter_equals(self, column: str, value: Any) -> 'FactFrame':
        """
        Filter for rows where column equals value.
        
        Args:
            column: Column name to filter on
            value: Value to compare with
            
        Returns:
            Filtered FactFrame
        """
        pass
    
    @abstractmethod
    def merge(self, other: 'FactFrame', how: str, left_on: List[str], 
              right_on: List[str], **kwargs) -> 'FactFrame':
        """
        Merge this frame with another.
        
        Args:
            other: Frame to merge with
            how: Type of join ("inner", "outer", "left", "right")
            left_on: Columns from this frame to join on
            right_on: Columns from other frame to join on
            **kwargs: Additional merge parameters
            
        Returns:
            Merged FactFrame
        """
        pass
    
    @abstractmethod
    def concat(self, others: Iterable['FactFrame']) -> 'FactFrame':
        """
        Concatenate multiple frames.
        
        Args:
            others: Frames to concatenate with this one
            
        Returns:
            Concatenated FactFrame
        """
        pass
    
    @abstractmethod
    def concat_rows(self, other: 'FactFrame') -> 'FactFrame':
        """
        Concatenate rows from another frame.
        
        Args:
            other: Frame whose rows to append
            
        Returns:
            FactFrame with appended rows
        """
        pass
    
    @abstractmethod
    def drop(self, columns: List[str]) -> 'FactFrame':
        """
        Drop specified columns.
        
        Args:
            columns: Columns to drop
            
        Returns:
            FactFrame with columns removed
        """
        pass
    
    @abstractmethod
    def drop_duplicates(self) -> 'FactFrame':
        """
        Remove duplicate rows.
        
        Returns:
            FactFrame with duplicates removed
        """
        pass
    
    @abstractmethod
    def rename(self, columns_map: Dict[str, str]) -> 'FactFrame':
        """
        Rename columns according to the mapping.
        
        Args:
            columns_map: Mapping from old column names to new ones
            
        Returns:
            FactFrame with renamed columns
        """
        pass
    
    @abstractmethod
    def assign(self, **kwargs) -> 'FactFrame':
        """
        Assign new columns.
        
        Args:
            **kwargs: Column name -> value mappings
            
        Returns:
            FactFrame with new columns
        """
        pass
    
    def filter_by_truth_value(self, value) -> 'FactFrame':
        """
        Filter facts by their truth value (for Well-founded semantics).
        
        Args:
            value: Truth value to filter by
            
        Returns:
            FactFrame containing only facts with the specified truth value
        """
        # Default implementation for backward compatibility
        # Implementations supporting Well-founded semantics should override this
        return self.copy()


class RuleFrame(BaseFrame):
    """Interface for frame structures that store logical rules."""
    
    @abstractmethod
    def filter_by_head(self, predicate: str) -> 'RuleFrame':
        """
        Filter rules by head predicate name.
        
        Args:
            predicate: Head predicate name to filter by
            
        Returns:
            RuleFrame with only rules having the specified head predicate
        """
        pass
    
    @abstractmethod
    def filter_by_body(self, predicate: str) -> 'RuleFrame':
        """
        Filter rules that have a given predicate in their body.
        
        Args:
            predicate: Body predicate name to filter by
            
        Returns:
            RuleFrame with only rules having the specified predicate in their body
        """
        pass
    
    @abstractmethod
    def get_dependencies(self) -> Dict[str, Set[str]]:
        """
        Get predicate dependency map (which predicates depend on which).
        
        Returns:
            Dictionary mapping from each head predicate to the set of predicates it depends on
        """
        pass
    
    @abstractmethod
    def has_negation(self) -> bool:
        """
        Check if any rules contain negated literals.
        
        Returns:
            True if at least one rule contains negation, False otherwise
        """
        pass
    
    @abstractmethod
    def stratify(self) -> List['RuleFrame']:
        """
        Stratify rules based on dependencies and negation.
        
        Returns:
            List of RuleFrames, one per stratum, ordered from lowest to highest stratum
        """
        pass
    
    @abstractmethod
    def get_rules_for_predicate(self, predicate: str) -> 'RuleFrame':
        """
        Get all rules that define a specific predicate.
        
        Args:
            predicate: Predicate name to filter by
            
        Returns:
            RuleFrame with only rules defining the specified predicate
        """
        pass
    
    @abstractmethod
    def set_metadata(self, rule_id: str, key: str, value: Any) -> 'RuleFrame':
        """
        Set metadata for a specific rule.
        
        Args:
            rule_id: ID of the rule to modify
            key: Metadata key
            value: Metadata value
            
        Returns:
            Updated RuleFrame
        """
        pass
    
    @abstractmethod
    def get_metadata(self, rule_id: str, key: str) -> Any:
        """
        Get metadata for a specific rule.
        
        Args:
            rule_id: ID of the rule
            key: Metadata key
            
        Returns:
            Value of the metadata key, or None if not set
        """
        pass
    
    @abstractmethod
    def to_rules(self):
        """
        Convert the RuleFrame back to Rule objects.
        
        Returns:
            List of Rule objects
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_rules(cls, rules):
        """
        Create a RuleFrame from Rule objects.
        
        Args:
            rules: List of Rule objects
            
        Returns:
            RuleFrame containing the rules
        """
        pass
