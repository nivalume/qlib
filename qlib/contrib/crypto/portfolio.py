"""Target weights of a long-short book from one day's cross-sectional scores.

The same object builds the targets of a backtest (:mod:`qlib.contrib.crypto.backtest`) and of
live trading (:mod:`qlib.contrib.crypto.live`), so a live day trades what the backtest would.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

WEIGHTINGS = ("equal", "rank")
_EPS = 1e-12


@dataclass(frozen=True)
class LongShortPortfolio:
    """Long the best-scored instruments and short the worst.

    Sizes, one of:

    - ``n_long`` and ``n_short``: that many names a side (0 for a one-sided book).
    - ``quantile``: the top and bottom fraction of the scored names, at least one a side.

    When fewer names are scored than the sides need, each side takes at most half of them.
    When fewer than ``min_names`` are scored, the target is flat: nothing is held.

    Weights: the long side sums to ``(gross + net) / 2`` and the short side to
    ``-(gross - net) / 2``, as fractions of the account's equity. ``gross = 1, net = 0`` is a
    dollar-neutral book with half the equity long and half short. ``weighting`` spreads a side
    equally, or by rank (the best name of a side gets ``k`` parts, the last one 1 part).
    ``max_weight`` caps a name's absolute weight; what the cap cuts off is not given to the
    others, so a capped book holds less than ``gross``.

    ``buffer`` keeps turnover down: a name held on a side stays there while its rank on that side
    is within ``k + buffer``; the free places go to the best names not yet held. With
    ``buffer = 0`` the target depends on the scores alone.

    Equal scores are ranked by instrument name, so a target never depends on input order.
    """

    n_long: Optional[int] = None
    n_short: Optional[int] = None
    quantile: Optional[float] = None
    gross: float = 1.0
    net: float = 0.0
    weighting: str = "equal"
    buffer: int = 0
    max_weight: Optional[float] = None
    min_names: int = 10

    def __post_init__(self) -> None:
        counts = self.n_long is not None or self.n_short is not None
        if counts == (self.quantile is not None):
            raise ValueError("Give n_long and n_short, or quantile (not both)")
        if counts:
            if self.n_long is None or self.n_short is None:
                raise ValueError("Give both n_long and n_short")
            for name in ("n_long", "n_short"):
                value = getattr(self, name)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"{name} must be an integer >= 0, not {value!r}")
            if self.n_long + self.n_short == 0:
                raise ValueError("n_long and n_short cannot both be 0")
        else:
            if not isinstance(self.quantile, (int, float)) or not 0 < self.quantile <= 0.5:
                raise ValueError(f"quantile must be in (0, 0.5], not {self.quantile!r}")
        if not (isinstance(self.gross, (int, float)) and math.isfinite(self.gross) and self.gross > 0):
            raise ValueError(f"gross must be a positive number, not {self.gross!r}")
        if not (isinstance(self.net, (int, float)) and math.isfinite(self.net)) or abs(self.net) > self.gross + _EPS:
            raise ValueError(f"net must be within [-gross, gross], not {self.net!r}")
        if counts and self.n_long == 0 and self.long_total > _EPS:
            raise ValueError("n_long is 0, so the long side is empty: set net = -gross")
        if counts and self.n_short == 0 and self.short_total > _EPS:
            raise ValueError("n_short is 0, so the short side is empty: set net = gross")
        if self.weighting not in WEIGHTINGS:
            raise ValueError(f"weighting must be one of {WEIGHTINGS}, not {self.weighting!r}")
        if isinstance(self.buffer, bool) or not isinstance(self.buffer, int) or self.buffer < 0:
            raise ValueError(f"buffer must be an integer >= 0, not {self.buffer!r}")
        if self.max_weight is not None and not (
            isinstance(self.max_weight, (int, float)) and math.isfinite(self.max_weight) and self.max_weight > 0
        ):
            raise ValueError(f"max_weight must be a positive number, not {self.max_weight!r}")
        if isinstance(self.min_names, bool) or not isinstance(self.min_names, int) or self.min_names < 1:
            raise ValueError(f"min_names must be an integer >= 1, not {self.min_names!r}")

    @property
    def long_total(self) -> float:
        return (self.gross + self.net) / 2

    @property
    def short_total(self) -> float:
        return (self.gross - self.net) / 2

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def sizes(self, scored: int) -> "tuple[int, int]":
        """The names a side takes when ``scored`` names have a score."""
        if scored < self.min_names:
            return 0, 0
        if self.quantile is not None:
            k = max(1, int(math.floor(scored * self.quantile + 1e-9)))
            n_long, n_short = k, k
        else:
            assert self.n_long is not None and self.n_short is not None
            n_long, n_short = self.n_long, self.n_short
        if n_long + n_short > scored:
            if n_long == 0:
                n_short = scored
            elif n_short == 0:
                n_long = scored
            else:
                n_long = min(n_long, scored // 2)
                n_short = min(n_short, scored - n_long, scored // 2)
        return n_long, n_short

    def boundary_tied(self, scores: pd.Series) -> bool:
        """Whether two scores are equal at the edge of the long or the short side: which of the
        equal names is held there is then decided by instrument name, not by the scores. A model
        that outputs few distinct values (a handful of trees) does this."""
        values = pd.Series(scores, dtype="float64").to_numpy()
        values = np.sort(values[np.isfinite(values)])
        n_long, n_short = self.sizes(len(values))
        if 0 < n_long < len(values) and values[-n_long] == values[-n_long - 1]:
            return True
        return 0 < n_short < len(values) and values[n_short - 1] == values[n_short]

    def target(self, scores: pd.Series, current: Optional[pd.Series] = None) -> pd.Series:
        """The target weight of each instrument to hold, from one day's ``scores`` (indexed by
        instrument; NaN and infinite scores are left out). ``current`` holds the weights held now:
        only their signs are used, by ``buffer``. Instruments not in the result are held at 0."""
        ranked = rank(scores)
        n_long, n_short = self.sizes(len(ranked))
        if n_long == 0 and n_short == 0:
            return _weights({})
        held_long: set = set()
        held_short: set = set()
        if current is not None and len(current):
            current = pd.Series(current, dtype="float64")
            held_long = set(current.index[current > _EPS])
            held_short = set(current.index[current < -_EPS])
        longs = self._pick(ranked, n_long, held_long, exclude=set())
        shorts = self._pick(ranked[::-1], n_short, held_short, exclude=set(longs))
        weights: Dict[str, float] = {}
        weights.update(self._spread(longs, self.long_total))
        weights.update(self._spread(shorts, -self.short_total))
        return _weights(weights)

    def _pick(self, ranked: List[str], k: int, held: set, exclude: set) -> List[str]:
        """The ``k`` names a side takes from ``ranked`` (best first), keeping held ones within
        ``k + buffer``; in rank order."""
        if k == 0:
            return []
        limit = k + self.buffer
        chosen = [name for name in ranked[:limit] if name in held and name not in exclude][:k]
        taken = set(chosen)
        for name in ranked:
            if len(chosen) >= k:
                break
            if name not in exclude and name not in taken:
                chosen.append(name)
                taken.add(name)
        position = {name: i for i, name in enumerate(ranked)}
        return sorted(chosen, key=position.__getitem__)

    def _spread(self, names: Sequence[str], total: float) -> Mapping[str, float]:
        if not names or abs(total) <= _EPS:
            return {}
        if self.weighting == "equal":
            parts = np.ones(len(names))
        else:
            parts = np.arange(len(names), 0, -1, dtype="float64")
        weights = total * parts / parts.sum()
        if self.max_weight is not None:
            weights = np.clip(weights, -self.max_weight, self.max_weight)
        return dict(zip(names, weights.tolist()))


def rank(scores: pd.Series) -> List[str]:
    """Instrument names with a finite score, best first; equal scores by name."""
    values = pd.Series(scores, dtype="float64")
    values = values[np.isfinite(values.to_numpy())]
    if values.index.has_duplicates:
        raise ValueError(f"Scores repeat instruments: {sorted(set(values.index[values.index.duplicated()]))[:5]}")
    names = np.asarray([str(name) for name in values.index], dtype=object)
    order = np.lexsort((names, -values.to_numpy()))
    return names[order].tolist()


def _weights(values: Mapping[str, float]) -> pd.Series:
    series = pd.Series(dict(values), dtype="float64", name="weight")
    series = series[series.abs() > _EPS]
    series.index.name = "instrument"
    return series.sort_index()
