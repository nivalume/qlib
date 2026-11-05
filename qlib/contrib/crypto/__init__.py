# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Crypto perpetuals in qlib: a long-short backtest with funding and per-instrument costs, a
workflow record for it, and the daily targets of live trading.

The data is a qlib directory written by ``jarvis-data dump qlib`` (``$close $vwap $amount
$funding`` and the rest), initialized with ``qlib.init(region="crypto")``.
"""

from .backtest import BacktestResult, backtest_long_short, to_scores
from .costs import CostModel
from .market import PerpMarket
from .portfolio import LongShortPortfolio

__all__ = [
    "BacktestResult",
    "CostModel",
    "LongShortPortfolio",
    "PerpMarket",
    "backtest_long_short",
    "to_scores",
]
