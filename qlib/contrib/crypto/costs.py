"""What a trade in a perpetual costs: the exchange fee, the spread and slippage of its liquidity
tier, and market impact.

For a trade of notional ``|v|`` in an instrument whose mean daily quote volume over the
``window`` days before the trade is ``adv``, and whose daily return has standard deviation
``sigma`` over those days, the cost is ``|v| * rate`` with

    rate = fee + spread(adv) + impact * sigma * sqrt(|v| / adv)

- ``fee``: the taker fee rate of a side (Binance USD-M VIP 0 is 0.05%); ``fees`` overrides it
  for single instruments.
- ``spread_tiers``: ``(min_adv, rate)`` pairs, the half spread and slippage of a market order in
  instruments of that liquidity. An instrument without a known ``adv`` pays the last tier's.
- ``impact``: the coefficient of the square-root impact law; 0 turns it off. It makes costs grow
  with the account, so set the backtest's ``account`` to the equity that will trade.
- ``max_participation``: the most of ``adv`` one day's trade in an instrument may be. A larger
  rebalance trade is cut to it, so the position reaches its target over several days. A forced
  close (a delisting) is never cut.

The defaults are starting points, not measurements: calibrate them with your own fills.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

DEFAULT_SPREAD_TIERS: Tuple[Tuple[float, float], ...] = (
    (1e9, 0.0001),  # 1 bp: BTC, ETH and the largest coins
    (1e8, 0.0003),
    (2e7, 0.0008),
    (0.0, 0.0020),  # 20 bp: the thinnest
)


@dataclass(frozen=True)
class CostModel:
    fee: float = 0.0005
    fees: Mapping[str, float] = field(default_factory=dict)
    spread_tiers: Sequence[Sequence[float]] = DEFAULT_SPREAD_TIERS
    impact: float = 0.5
    window: int = 30
    max_participation: Optional[float] = None

    def __post_init__(self) -> None:
        _rate("fee", self.fee)
        for name, value in dict(self.fees).items():
            _rate(f"fees[{name!r}]", value)
        tiers = [tuple(float(x) for x in tier) for tier in self.spread_tiers]
        if not tiers or any(len(tier) != 2 for tier in tiers):
            raise ValueError("spread_tiers is a list of (min_adv, rate) pairs")
        for threshold, value in tiers:
            if not math.isfinite(threshold) or threshold < 0:
                raise ValueError(f"A spread tier's min_adv must be >= 0, not {threshold!r}")
            _rate("a spread tier's rate", value)
        if min(threshold for threshold, _ in tiers) != 0:
            raise ValueError("spread_tiers needs a tier with min_adv 0, so that every instrument has one")
        object.__setattr__(self, "spread_tiers", tuple(sorted(tiers, key=lambda tier: -tier[0])))
        object.__setattr__(self, "fees", dict(self.fees))
        _rate("impact", self.impact)
        if isinstance(self.window, bool) or not isinstance(self.window, int) or self.window < 2:
            raise ValueError(f"window must be an integer >= 2, not {self.window!r}")
        if self.max_participation is not None and not (
            isinstance(self.max_participation, (int, float))
            and math.isfinite(self.max_participation)
            and self.max_participation > 0
        ):
            raise ValueError(f"max_participation must be a positive number, not {self.max_participation!r}")

    def to_dict(self) -> Dict[str, object]:
        return {
            "fee": self.fee,
            "fees": dict(self.fees),
            "spread_tiers": [list(tier) for tier in self.spread_tiers],
            "impact": self.impact,
            "window": self.window,
            "max_participation": self.max_participation,
        }

    @property
    def min_periods(self) -> int:
        return max(2, self.window // 3)

    def fee_rates(self, instruments: Sequence[str]) -> np.ndarray:
        return np.asarray([self.fees.get(name, self.fee) for name in instruments], dtype="float64")

    def spread(self, adv: np.ndarray) -> np.ndarray:
        """The spread rate of each instrument's liquidity tier; the last tier where ``adv`` is
        not known."""
        adv = np.asarray(adv, dtype="float64")
        out = np.full(adv.shape, self.spread_tiers[-1][1])
        known = np.isfinite(adv)
        for threshold, value in reversed(self.spread_tiers):
            out[known & (adv >= threshold)] = value
        return out

    def rates(self, trade: np.ndarray, fee: np.ndarray, adv: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """The cost rate of trades of notional ``trade`` (signed or not)."""
        size = np.abs(np.asarray(trade, dtype="float64"))
        rate = fee + self.spread(adv)
        if self.impact > 0:
            usable = np.isfinite(adv) & (adv > 0) & np.isfinite(sigma)
            share = np.zeros_like(size)
            share[usable] = size[usable] / adv[usable]
            rate = rate + np.where(usable, self.impact * np.where(usable, sigma, 0.0) * np.sqrt(share), 0.0)
        return rate

    def cap(self, trade: np.ndarray, adv: np.ndarray) -> np.ndarray:
        """``trade`` cut to ``max_participation`` of ``adv``; no trade where ``adv`` is not known."""
        if self.max_participation is None:
            return trade
        limit = np.where(np.isfinite(adv) & (adv > 0), self.max_participation * np.where(np.isfinite(adv), adv, 0), 0.0)
        return np.clip(trade, -limit, limit)


def _rate(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a number >= 0, not {value!r}")
