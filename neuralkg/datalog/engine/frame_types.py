"""
Enumerations for Frame types and implementations used throughout the Datalog engine.
"""
from enum import Enum, auto

class FrameImplementation(str, Enum):
    """
    Available Frame implementations for the Datalog engine.
    
    This enum provides all supported backend implementations and should be used
    when specifying the implementation to use with FrameFactory.
    """
    PANDAS = "pandas"
    CUDF = "cudf"
    MNMG = "mnmg"
    SCALLOP = "scallop"
    
    @classmethod
    def get_all_implementations(cls) -> list[str]:
        """Return a list of all available implementation names."""
        return [impl.value for impl in cls]
        
    @classmethod
    def is_valid(cls, implementation: str) -> bool:
        """Check if an implementation name is valid."""
        return implementation in [impl.value for impl in cls]
