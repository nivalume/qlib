"""``PerpPortAnaRecord``: the long-short perpetual backtest as a step of a qlib workflow.

In a ``qrun`` config, after ``SignalRecord``:

.. code-block:: yaml

    record:
      - class: SignalRecord
        module_path: qlib.workflow.record_temp
        kwargs: {model: <MODEL>, dataset: <DATASET>}
      - class: PerpPortAnaRecord
        module_path: qlib.contrib.crypto.record
        kwargs:
          config:
            portfolio: {quantile: 0.1, gross: 1.0, net: 0.0, buffer: 2}
            costs: {fee: 0.0005, impact: 0.5}
            backtest: {account: 1000000, delay: 1, deal_price: close, benchmark: null}

It writes, under ``portfolio_analysis/``, the files ``PortAnaRecord`` writes for a daily
backtest, so qlib's report graphs read them:

- ``report_normal_1day.pkl``: the daily report (``account return cost bench turnover`` and the
  perpetual's own columns: ``funding fee slippage`` and the exposures)
- ``port_analysis_1day.pkl``: the risk analysis without and with costs, over 365 days a year

and its own:

- ``weights_1day.pkl``: the weight of each instrument held at each close
- ``trades_1day.pkl``: every trade, with its cost
- ``perp_config.pkl``: the settings of the run; ``qlib.contrib.crypto.live`` builds live targets
  with the same portfolio settings

Metrics are logged as ``PortAnaRecord`` logs them (``1day.excess_return_with_cost.annualized_return``
and the rest), and ``1day.perp.*`` for turnover, costs, funding and exposures.
"""

from __future__ import annotations

from pprint import pprint
from typing import Any, Dict, Optional

from ...log import get_module_logger
from ...utils import flatten_dict
from ...utils.data import deepcopy_basic_type
from ...workflow.record_temp import ACRecordTemp, SignalRecord
from .backtest import DAYS_PER_YEAR, backtest_long_short
from .costs import CostModel
from .portfolio import LongShortPortfolio

logger = get_module_logger("crypto.record")

SECTIONS = ("portfolio", "costs", "backtest")
BACKTEST_KEYS = (
    "start_time",
    "end_time",
    "account",
    "delay",
    "rebalance",
    "deal_price",
    "benchmark",
    "funding_missing",
)
FREQ = "1day"


def build_portfolio(config: Optional[Dict]) -> LongShortPortfolio:
    return LongShortPortfolio(**(dict(config) if config else {"quantile": 0.1}))


def build_costs(config: Optional[Dict]) -> CostModel:
    config = dict(config or {})
    if "spread_tiers" in config:
        config["spread_tiers"] = tuple(tuple(tier) for tier in config["spread_tiers"])
    return CostModel(**config)


class PerpPortAnaRecord(ACRecordTemp):
    artifact_path = "portfolio_analysis"
    depend_cls = SignalRecord

    def __init__(self, recorder: Any, config: Optional[Dict] = None, skip_existing: bool = False, **kwargs: Any):
        super().__init__(recorder=recorder, skip_existing=skip_existing, **kwargs)
        config = deepcopy_basic_type(config or {})
        unknown = sorted(set(config) - set(SECTIONS))
        if unknown:
            raise ValueError(f"Unknown PerpPortAnaRecord config section {unknown[0]!r}; use {', '.join(SECTIONS)}")
        self.portfolio = build_portfolio(config.get("portfolio"))
        self.costs = build_costs(config.get("costs"))
        self.backtest_config = dict(config.get("backtest") or {})
        unknown = sorted(set(self.backtest_config) - set(BACKTEST_KEYS))
        if unknown:
            raise ValueError(f"Unknown backtest setting {unknown[0]!r}; use {', '.join(BACKTEST_KEYS)}")

    def _generate(self, **kwargs: Any) -> Dict[str, Any]:
        pred = self.load("pred.pkl")
        result = backtest_long_short(pred, portfolio=self.portfolio, costs=self.costs, **self.backtest_config)
        analysis = result.analysis(N=DAYS_PER_YEAR)
        metrics = flatten_dict(analysis["risk"].unstack().T.to_dict())
        metrics.update({f"perp.{name}": value for name, value in result.summary(N=DAYS_PER_YEAR).items()})
        self.recorder.log_metrics(**{f"{FREQ}.{name}": value for name, value in metrics.items()})
        pprint(f"Long-short perpetual backtest {result.config['start_time']} to {result.config['end_time']}:")
        pprint(analysis)
        pprint(result.summary(N=DAYS_PER_YEAR))
        return {
            f"report_normal_{FREQ}.pkl": result.report,
            f"port_analysis_{FREQ}.pkl": analysis,
            f"weights_{FREQ}.pkl": result.weights,
            f"trades_{FREQ}.pkl": result.trades,
            "perp_config.pkl": result.config,
        }

    def list(self) -> list:
        return [
            f"report_normal_{FREQ}.pkl",
            f"port_analysis_{FREQ}.pkl",
            f"weights_{FREQ}.pkl",
            f"trades_{FREQ}.pkl",
            "perp_config.pkl",
        ]
