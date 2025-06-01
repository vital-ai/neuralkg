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

class PandasFrame(Frame):
    """
    A Frame implementation using pandas.DataFrame under the hood.
    """
    __slots__ = ("_df",)

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    @classmethod
    def empty(cls, columns: list[str]) -> 'PandasFrame':
        df = pd.DataFrame({col: pd.Series(dtype="object") for col in columns})
        return cls(df)

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]], columns: list[str]) -> 'PandasFrame':
        if not rows:
            return cls.empty(columns)
        df = pd.DataFrame(rows, columns=columns)
        for col in columns:
            df[col] = df[col].astype(object)
        return cls(df)

    def copy(self) -> 'PandasFrame':
        return PandasFrame(self._df.copy().reset_index(drop=True))

    def rename(self, columns_map: dict[str, str]) -> 'PandasFrame':
        return PandasFrame(self._df.rename(columns=columns_map))

    def filter_equals(self, column: str, value: Any) -> 'PandasFrame':
        return PandasFrame(self._df[self._df[column] == value].reset_index(drop=True))

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
        assert isinstance(other, PandasFrame)
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
        return list(self._df.columns)

    def num_rows(self) -> int:
        return len(self._df)

# By default, we use PandasFrame. To switch, assign PolarsFrame here (once implemented).
make_frame: Type[Frame] = PandasFrame
