import pandas as pd
from abc import ABC, abstractmethod
from typing import Any, Iterable, Type, TypeVar

FrameT = TypeVar("FrameT", bound="Frame")

class Frame(ABC):
    """
    Abstract base class for a DataFrame-like object. Any backend must
    implement these methods.
    """
    @classmethod
    @abstractmethod
    def empty(cls, columns: list[str]) -> 'Frame':
        ...

    @classmethod
    @abstractmethod
    def from_dicts(cls, rows: list[dict[str, Any]], columns: list[str]) -> 'Frame':
        ...

    @abstractmethod
    def copy(self) -> 'Frame':
        ...

    @abstractmethod
    def rename(self, columns_map: dict[str, str]) -> 'Frame':
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
        left_on: list[str],
        right_on: list[str],
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
    def drop(self, columns: list[str]) -> 'Frame':
        ...

    @abstractmethod
    def drop_duplicates(self) -> 'Frame':
        ...

    @abstractmethod
    def to_records(self) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def columns(self) -> list[str]:
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

class PandasFrame(Frame):
    """
    A Frame implementation using pandas.DataFrame under the hood.
    """
    __slots__ = ("_df",)

    def __getitem__(self, key):
        import pandas as pd
        # Column selection
        if isinstance(key, list):
            return PandasFrame(self._df[key])
        elif isinstance(key, str):
            return self._df[key]
        # Boolean row mask
        elif isinstance(key, pd.Series) and key.dtype == bool:
            return PandasFrame(self._df[key])
        else:
            raise TypeError(f"Invalid key type for PandasFrame: {type(key)}")

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def empty(cls, columns: list[str]) -> 'PandasFrame':
        df = pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
        return cls(df)

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]], columns: list[str]) -> 'PandasFrame':
        if rows is None or len(rows) == 0:
            return cls.empty(columns)
        df = pd.DataFrame(rows, columns=columns)
        for col in columns:
            df[col] = df[col].astype(object)
        return cls(df)

    def copy(self) -> 'PandasFrame':
        return PandasFrame(self._df.copy().reset_index(drop=True))

    def rename(self, columns_map: dict[str, str]) -> 'PandasFrame':
        return PandasFrame(self._df.rename(columns=columns_map))

    def filter(self, column: str, op: str, value: Any) -> 'PandasFrame':
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"[PandasFrame.filter] Filtering: column={column}, op={op}, value={value}")
        logger.debug(f"[PandasFrame.filter] Column dtype: {self._df[column].dtype}")
        logger.debug(f"[PandasFrame.filter] Sample values: {self._df[column].head(10).tolist()}")
        # Attempt type coercion if needed
        series = self._df[column]
        try:
            if op in ("<", "<=", ">", ">="):
                series = pd.to_numeric(series, errors='coerce')
                value = float(value)
        except Exception as e:
            logger.warning(f"[PandasFrame.filter] Type coercion failed: {e}")
        if op == '==':
            mask = series == value
        elif op in ('!=', '<>'):
            mask = series != value
        elif op == '<':
            mask = series < value
        elif op == '<=':
            mask = series <= value
        elif op == '>':
            mask = series > value
        elif op == '>=':
            mask = series >= value
        else:
            raise ValueError(f"Unsupported operator: {op}")
        logger.debug(f"[PandasFrame.filter] Filter mask: {mask.head(10).tolist()}")
        filtered = self._df[mask].reset_index(drop=True)
        logger.debug(f"[PandasFrame.filter] Filtered DataFrame shape: {filtered.shape}")
        return PandasFrame(filtered)

    def filter_equals(self, column: str, value: Any) -> 'PandasFrame':
        if callable(value):
            mask = self._df[column].apply(value)
            return PandasFrame(self._df[mask].reset_index(drop=True))
        else:
            return self.filter(column, '==', value)

    def assign(self, **kwargs: Any) -> 'PandasFrame':
        return PandasFrame(self._df.assign(**kwargs))

    def merge(
        self,
        other: 'Frame',
        how: str,
        left_on: list[str],
        right_on: list[str],
        suffixes: tuple[str, str] = ("", ""),
        **kwargs
    ) -> 'PandasFrame':
        # Accept any Frame, convert to PandasFrame if needed
        if not isinstance(other, PandasFrame):
            other = PandasFrame.from_dicts(other.to_records(), other.columns())
        # Upcast merge columns to object dtype in both DataFrames
        for col in left_on:
            if col in self._df.columns:
                self._df[col] = self._df[col].astype(object)
        for col in right_on:
            if col in other._df.columns:
                other._df[col] = other._df[col].astype(object)
        merged = self._df.merge(
            other._df,
            how=how,
            left_on=left_on,
            right_on=right_on,
            suffixes=suffixes,
            **kwargs
        )
        return PandasFrame(merged.reset_index(drop=True))

    def concat(self, others: Iterable['Frame']) -> 'PandasFrame':
        frames = [self._df] + [o._df for o in others if isinstance(o, PandasFrame)]
        concatenated = pd.concat(frames, ignore_index=True)
        return PandasFrame(concatenated)
        
    def concat_rows(self, other: 'Frame') -> 'PandasFrame':
        """Concatenate rows from another frame to this frame.
        
        This is a convenience method for concat([other]).
        """
        if not isinstance(other, PandasFrame):
            # Convert other frame to PandasFrame if needed
            other = PandasFrame.from_dicts(other.to_records(), other.columns())
        
        return self.concat([other])

    def drop(self, columns: list[str]) -> 'PandasFrame':
        return PandasFrame(self._df.drop(columns=columns))

    def drop_duplicates(self) -> 'PandasFrame':
        df = self._df.copy()
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        return PandasFrame(df.drop_duplicates().reset_index(drop=True))

    def to_records(self) -> list[dict[str, Any]]:
        return self._df.to_dict(orient="records")

    def columns(self) -> list[str]:
        import pandas as pd
        assert isinstance(self._df, pd.DataFrame), f"PandasFrame._df must be a pd.DataFrame, got {type(self._df)}"
        cols = self._df.columns
        if not isinstance(cols, list):
            cols = list(cols)
        assert all(isinstance(c, str) for c in cols), f"PandasFrame.columns() must return list of str, got: {cols}"
        return cols

    def num_rows(self) -> int:
        return len(self._df)

    def duplicated(self, *args, **kwargs):
        # Returns a boolean Series indicating duplicate rows
        return self._df.duplicated(*args, **kwargs)

    def to_dict(self, *args, **kwargs):
        # Delegate to the underlying DataFrame
        return self._df.to_dict(*args, **kwargs)

    def __len__(self):
        return len(self._df)
        
    def has_only_nan_values(self, column: str) -> bool:
        """Check if a column contains only NaN values.
        
        Args:
            column: Column name to check
            
        Returns:
            True if all values in the column are NaN, False otherwise
        """
        if column not in self._df.columns:
            return False
            
        # Use pandas isna() to check for NaN values in all rows
        return self._df[column].isna().all()

# By default, we use PandasFrame. To switch, assign PolarsFrame here (once implemented).
make_frame: Type[Frame] = PandasFrame
