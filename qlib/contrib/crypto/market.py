"""The daily market a perpetual backtest needs, read from a qlib directory that
``jarvis-data dump qlib`` wrote.

Each frame is days by instruments, on the qlib calendar:

- ``close``: the last-price close. NaN on a day without a bar or without a trade; a day with a
  NaN close is a day the instrument cannot be traded and keeps its last mark.
- ``vwap``: the day's volume-weighted price, the deal price of ``deal_price="vwap"``.
- ``funding``: the funding rates settled in (d 00:00, d+1 00:00] UTC, summed. Positive: longs pay.
  NaN where the data does not know it (the current month, a missing record).
- ``amount``: the quote volume, for liquidity tiers and impact.
- ``delist``: True on the last day of a listing span that ends before the data does: a position
  still open then is closed at that day's close, as the exchange settles it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd

from ...data import D
from ...log import get_module_logger

logger = get_module_logger("crypto.market")

FIELDS = {"close": "$close", "vwap": "$vwap", "funding": "$funding", "amount": "$amount"}


@dataclass
class PerpMarket:
    close: pd.DataFrame
    vwap: pd.DataFrame
    funding: pd.DataFrame
    amount: pd.DataFrame
    delist: pd.DataFrame

    def __post_init__(self) -> None:
        frames = {name: getattr(self, name) for name in ("close", "vwap", "funding", "amount", "delist")}
        index, columns = self.close.index, self.close.columns
        for name, frame in frames.items():
            if not frame.index.equals(index) or not frame.columns.equals(columns):
                raise ValueError(f"PerpMarket.{name} is not on the days and instruments of close")
        if not index.is_monotonic_increasing or index.has_duplicates:
            raise ValueError("PerpMarket days must be increasing and unique")
        if columns.has_duplicates:
            raise ValueError("PerpMarket instruments repeat")

    @property
    def days(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.close.index)

    @property
    def instruments(self) -> pd.Index:
        return self.close.columns

    @classmethod
    def load(
        cls,
        instruments: Iterable[str],
        start_time: Any,
        end_time: Any,
        freq: str = "day",
    ) -> "PerpMarket":
        """Read the instruments' fields from qlib (``qlib.init`` first) for [start_time, end_time],
        whatever universe they are in: a position kept after it leaves the universe still needs
        its prices."""
        names = sorted({str(name) for name in instruments})
        if not names:
            raise ValueError("No instruments to load")
        days = pd.DatetimeIndex(D.calendar(start_time=start_time, end_time=end_time, freq=freq))
        if days.empty:
            raise ValueError(f"The calendar has no day from {start_time} to {end_time}")
        frames = {}
        for name, field in FIELDS.items():
            frames[name] = _field(names, field, days, freq)
        if frames["funding"].isna().all().all():
            logger.warning("No $funding in the data: funding is not charged")
        delist = pd.DataFrame(False, index=days, columns=pd.Index(names, name="instrument"))
        spans = D.list_instruments(D.instruments("all"), as_list=False, freq=freq)
        data_end = pd.Timestamp(D.calendar(freq=freq)[-1])
        for name in names:
            for begin, end in spans.get(name, []):
                end = pd.Timestamp(end)
                if end < data_end and days[0] <= end <= days[-1]:
                    delist.loc[end, name] = True
        return cls(delist=delist, **frames)


def _field(names: List[str], field: str, days: pd.DatetimeIndex, freq: str) -> pd.DataFrame:
    try:
        data = D.features(names, [field], start_time=days[0], end_time=days[-1], freq=freq)
    except (ValueError, KeyError, FileNotFoundError) as error:
        logger.warning(f"{field} cannot be read ({error}): it is NaN")
        data = None
    if data is None or data.empty:
        frame = pd.DataFrame(np.nan, index=days, columns=names)
    else:
        frame = data.iloc[:, 0].unstack(level="instrument")
        frame = frame.reindex(index=days, columns=names)
    frame = frame.astype("float64")
    frame.index.name = "datetime"
    frame.columns = pd.Index(names, name="instrument")
    return frame


def rolling_adv(amount: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """Mean daily quote volume over the ``window`` days before each day (the day itself is not
    in it); a day without a bar is not counted."""
    return amount.rolling(window, min_periods=min_periods).mean().shift(1)


def rolling_sigma(close: pd.DataFrame, window: int, min_periods: int) -> pd.DataFrame:
    """Standard deviation of daily close-to-close returns over the ``window`` days before each day."""
    returns = close / close.shift(1) - 1
    return returns.rolling(window, min_periods=min_periods).std().shift(1)


def benchmark_returns(close: Optional[pd.Series]) -> Optional[pd.Series]:
    if close is None:
        return None
    return (close / close.shift(1) - 1).fillna(0.0)
