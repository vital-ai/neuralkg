from abc import ABC, abstractmethod
from typing import Any, Iterable, List, Dict, Type, TypeVar

FrameT = TypeVar("FrameT", bound="Frame")

class Frame(ABC):
    """
    Abstract base class for a DataFrame-like object. Any backend must
    implement these methods.
    
    The Frame interface defines operations on tabular data that are needed
    by the Datalog evaluation engine. Different implementations can provide
    optimized backends (e.g., pandas, cuDF, MNMGDatalog).
    """
    @classmethod
    @abstractmethod
    def empty(cls, columns: List[str]) -> 'Frame':
        ...

    @classmethod
    @abstractmethod
    def from_dicts(cls, rows: List[Dict[str, Any]], columns: List[str]) -> 'Frame':
        ...

    @abstractmethod
    def copy(self) -> 'Frame':
        ...

    @abstractmethod
    def rename(self, columns_map: Dict[str, str]) -> 'Frame':
        ...

    @abstractmethod
    def filter_equals(self, column: str, value: Any) -> 'Frame':
        ...

    @abstractmethod
    def assign(self, **kwargs: Any) -> 'Frame':
        ...

    @abstractmethod
    def merge(
        self,
        other: 'Frame',
        how: str,
        left_on: List[str],
        right_on: List[str],
        suffixes: tuple[str, str] = ("", ""),
        **kwargs
    ) -> 'Frame':
        ...

    @abstractmethod
    def concat(self, others: Iterable['Frame']) -> 'Frame':
        ...
        
    @abstractmethod
    def concat_rows(self, other: 'Frame') -> 'Frame':
        """Concatenate rows from another frame to this frame.
        
        This is a convenience method for concat([other]).
        """
        ...

    @abstractmethod
    def drop(self, columns: List[str]) -> 'Frame':
        ...

    @abstractmethod
    def drop_duplicates(self) -> 'Frame':
        ...

    @abstractmethod
    def to_records(self) -> List[Dict[str, Any]]:
        ...

    @abstractmethod
    def columns(self) -> List[str]:
        ...

    @abstractmethod
    def num_rows(self) -> int:
        ...
        
    @abstractmethod
    def has_only_nan_values(self, column: str) -> bool:
        """Check if a column contains only NaN values.
        
        Args:
            column: Column name to check
            
        Returns:
            True if all values in the column are NaN, False otherwise
        """
        ...


# Define a module-level object that will be properly initialized later
class MakeFrameProxy:
    """A proxy object that will be initialized with the actual implementation's factory methods."""
    def __getattr__(self, name):
        # Lazily import to avoid circular imports
        from .frame_factory import FrameFactory
        # Get the implementation and delegate the method call
        impl = FrameFactory.get_implementation()
        return getattr(impl, name)

# Create a proxy object that delegates to the real implementation
# This avoids circular imports while maintaining the API
make_frame = MakeFrameProxy()
