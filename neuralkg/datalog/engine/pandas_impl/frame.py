import pandas as pd
from typing import Any, Iterable, List, Dict, Optional, Set, Tuple, Callable, Union
import numpy as np
import logging
from copy import copy

from ..frame import Frame

logger = logging.getLogger(__name__)

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
    def configure(cls, config: Dict[str, Any]) -> None:
        """Configure pandas-specific settings.
        
        Args:
            config: Configuration dictionary for pandas implementation
        """
        # Simply log that configuration was received
        # PandasFrame uses default configuration
        if config:
            logger.info(f"Received PandasFrame configuration: {config}")
        else:
            logger.info("Using default PandasFrame configuration")
    
    @classmethod
    def empty(cls, columns: List[str]) -> 'PandasFrame':
        df = pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
        return cls(df)

    @classmethod
    def from_dicts(cls, rows: List[Dict[str, Any]], columns: List[str] = None) -> 'PandasFrame':
        if rows is None or len(rows) == 0:
            return cls.empty(columns if columns is not None else [])
        df = pd.DataFrame(rows, columns=columns)
        if columns:
            for col in columns:
                df[col] = df[col].astype(object)
        return cls(df)

    def copy(self) -> 'PandasFrame':
        return PandasFrame(self._df.copy().reset_index(drop=True))

    def rename(self, columns_map: Dict[str, str]) -> 'PandasFrame':
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
            if op == "==":
                mask = series == value
            elif op == "!=":
                mask = series != value
            elif op == "<":
                mask = series < value
            elif op == "<=":
                mask = series <= value
            elif op == ">":
                mask = series > value
            elif op == ">=":
                mask = series >= value
            elif op == "in":
                mask = series.isin(value)
            elif op == "not in":
                mask = ~series.isin(value)
            else:
                raise ValueError(f"Unsupported operator: {op}")
        except TypeError as e:
            logger.warning(f"[PandasFrame.filter] TypeError during comparison: {e}")
            logger.warning(f"[PandasFrame.filter] Attempting to convert types for comparison")
            # Try converting the series to same type as value
            try:
                if isinstance(value, str):
                    series = series.astype(str)
                elif isinstance(value, int):
                    series = series.astype(int)
                elif isinstance(value, float):
                    series = series.astype(float)
                    
                if op == "==":
                    mask = series == value
                elif op == "!=":
                    mask = series != value
                elif op == "<":
                    mask = series < value
                elif op == "<=":
                    mask = series <= value
                elif op == ">":
                    mask = series > value
                elif op == ">=":
                    mask = series >= value
                elif op == "in":
                    mask = series.isin(value)
                elif op == "not in":
                    mask = ~series.isin(value)
                else:
                    raise ValueError(f"Unsupported operator: {op}")
            except Exception as e2:
                logger.error(f"[PandasFrame.filter] Failed to convert types: {e2}")
                # Fall back to empty result
                return PandasFrame(self._df.head(0))
                
        return PandasFrame(self._df[mask].reset_index(drop=True))

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
        left_on: List[str],
        right_on: List[str],
        suffixes: tuple[str, str] = ("", ""),
        **kwargs
    ) -> 'PandasFrame':
        # Accept any Frame, convert to PandasFrame if needed
        if not isinstance(other, PandasFrame):
            from ..frame_factory import FrameFactory
            other = FrameFactory.from_dicts(other.to_records(), other.columns())
            if not isinstance(other, PandasFrame):
                # If still not PandasFrame (different implementation), convert directly
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
        frames = [self._df]
        for o in others:
            if isinstance(o, PandasFrame):
                frames.append(o._df)
            else:
                # Convert to PandasFrame if from different implementation
                from ..frame_factory import FrameFactory
                o_pandas = FrameFactory.from_dicts(o.to_records(), o.columns())
                if isinstance(o_pandas, PandasFrame):
                    frames.append(o_pandas._df)
                else:
                    # Direct conversion if FrameFactory returns a different implementation
                    frames.append(pd.DataFrame(o.to_records()))
                    
        concatenated = pd.concat(frames, ignore_index=True)
        return PandasFrame(concatenated)
        
    def concat_rows(self, other: 'Frame') -> 'PandasFrame':
        """Concatenate rows from another frame to this frame.
        
        This is a convenience method for concat([other]).
        """
        return self.concat([other])

    def drop(self, columns: List[str]) -> 'PandasFrame':
        return PandasFrame(self._df.drop(columns=columns))

    def drop_duplicates(self) -> 'PandasFrame':
        df = self._df.copy()
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, list)).any():
                df[col] = df[col].apply(lambda x: tuple(x) if isinstance(x, list) else x)
        return PandasFrame(df.drop_duplicates().reset_index(drop=True))

    def to_records(self) -> List[Dict[str, Any]]:
        return self._df.to_dict(orient="records")

    def columns(self) -> List[str]:
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
