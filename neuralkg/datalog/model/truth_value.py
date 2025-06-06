"""
Truth value enumeration for Datalog evaluation, supporting well-founded semantics.
"""
from enum import Enum, auto
from typing import Optional, Union


class TruthValue(Enum):
    """
    Enumeration of truth values used in well-founded semantics.
    
    THREE-VALUED LOGIC:
    - TRUE: Definitely true facts
    - FALSE: Definitely false facts (closed-world assumption)
    - UNKNOWN: Facts with undefined truth value (e.g. in recursive negation)
    """
    TRUE = auto()
    FALSE = auto()
    UNKNOWN = auto()
    
    @classmethod
    def from_bool(cls, value: Optional[bool]) -> 'TruthValue':
        """Convert a boolean or None to a TruthValue.
        
        Args:
            value: Boolean value (or None for UNKNOWN)
            
        Returns:
            Corresponding TruthValue
        """
        if value is None:
            return cls.UNKNOWN
        return cls.TRUE if value else cls.FALSE
    
    def to_bool(self) -> Optional[bool]:
        """Convert TruthValue to a boolean or None.
        
        Returns:
            True for TRUE, False for FALSE, None for UNKNOWN
        """
        if self == TruthValue.TRUE:
            return True
        elif self == TruthValue.FALSE:
            return False
        return None
    
    def __str__(self) -> str:
        return self.name
    
    def __repr__(self) -> str:
        return f"TruthValue.{self.name}"


# Logical operations for three-valued logic
def logical_not(value: TruthValue) -> TruthValue:
    """Logical NOT operation for three-valued logic.
    
    Args:
        value: Input truth value
        
    Returns:
        NOT value
    """
    if value == TruthValue.TRUE:
        return TruthValue.FALSE
    elif value == TruthValue.FALSE:
        return TruthValue.TRUE
    return TruthValue.UNKNOWN


def logical_and(a: TruthValue, b: TruthValue) -> TruthValue:
    """Logical AND operation for three-valued logic.
    
    Args:
        a: First truth value
        b: Second truth value
        
    Returns:
        a AND b
    """
    if a == TruthValue.FALSE or b == TruthValue.FALSE:
        return TruthValue.FALSE
    elif a == TruthValue.TRUE and b == TruthValue.TRUE:
        return TruthValue.TRUE
    return TruthValue.UNKNOWN


def logical_or(a: TruthValue, b: TruthValue) -> TruthValue:
    """Logical OR operation for three-valued logic.
    
    Args:
        a: First truth value
        b: Second truth value
        
    Returns:
        a OR b
    """
    if a == TruthValue.TRUE or b == TruthValue.TRUE:
        return TruthValue.TRUE
    elif a == TruthValue.FALSE and b == TruthValue.FALSE:
        return TruthValue.FALSE
    return TruthValue.UNKNOWN
