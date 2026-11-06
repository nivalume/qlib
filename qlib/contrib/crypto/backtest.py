"""A daily backtest of a long-short book of USD-M perpetuals.

qlib's own backtest (Exchange, TopkDropoutStrategy) holds stocks: long only, no funding, one fee
rate. This one holds perpetuals: a signed notional per instrument against the account's equity,
the funding they settle, and a cost per trade from :class:`~qlib.contrib.crypto.costs.CostModel`.

Time
----
Day ``t`` is the UTC day whose bar closes at 00:00 of ``t + 1``. A score of day ``t`` is known at
that close; its targets trade ``delay`` days later (1 by default):

- ``deal_price="close"``: at the close of ``t + delay``. With ``delay = 1`` that is the label
  ``Ref($close, -2) / Ref($close, -1) - 1``: the position earns from the close of ``t + 1`` to
  the close of ``t + 2`` and pays the funding of day ``t + 2``.
- ``deal_price="vwap"``: during day ``t + delay``, at its vwap; the position traded earns from
  the vwap to the close, and the day's funding is charged on the mean of the positions before
  and after the trade.

Each day, in order
------------------
1. Mark: a position earns ``quantity x (close - last mark)``. On a day without a close
   (no bar, or no trade) the position keeps its last mark and cannot trade.
2. Funding: a position held through the day pays ``notional x $funding`` (shorts receive when
   positive). A NaN ``$funding`` (not published yet, a missing record) charges nothing and is
   counted in ``funding_missing``; ``funding_missing="raise"`` stops instead.
3. Trade: on a rebalance day, the targets of :class:`LongShortPortfolio` from the decision day's
   scores, as fractions of the equity, minus what is held. An instrument that cannot trade today
   keeps its position. Instruments delisted today are left out of the targets.
4. Delisting: a position open on the last day of its listing span is closed at that day's close.
5. Costs: every trade pays :class:`CostModel`'s rate on its notional.

The report has qlib's columns, so ``risk_analysis`` and qlib's report graphs read it: ``return``
is the price and funding P&L over the previous equity, before trading costs; ``cost`` is the
trading costs; ``return - cost`` is the account's return.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ...log import get_module_logger
from ..evaluate import risk_analysis
from .costs import CostModel
from .market import PerpMarket, benchmark_returns, rolling_adv, rolling_sigma
from .portfolio import LongShortPortfolio

logger = get_module_logger("crypto.backtest")

DEAL_PRICES = ("close", "vwap")
FUNDING_POLICIES = ("zero", "raise")
DAYS_PER_YEAR = 365
_DUST = 1e-9  # a trade below this fraction of the equity is not made


@dataclass
class BacktestResult:
    report: pd.DataFrame
    weights: pd.DataFrame
    trades: pd.DataFrame
    config: Dict[str, object] = field(default_factory=dict)

    def analysis(self, N: int = DAYS_PER_YEAR) -> pd.DataFrame:
        """qlib's risk analysis of the excess return without and with costs, as ``PortAnaRecord``
        writes it, annualized over ``N`` days."""
        report = self.report
        frames = {
            "excess_return_without_cost": risk_analysis(report["return"] - report["bench"], N=N, freq=None),
            "excess_return_with_cost": risk_analysis(
                report["return"] - report["bench"] - report["cost"], N=N, freq=None
            ),
        }
        return pd.concat(frames)

    def summary(self, N: int = DAYS_PER_YEAR) -> Dict[str, float]:
        report = self.report
        net = report["return"] - report["cost"]
        held = report["n_long"] + report["n_short"]
        return {
            "days": float(len(report)),
            "annualized_return": float(net.mean() * N),
            "information_ratio": (
                float(net.mean() / net.std(ddof=1) * math.sqrt(N)) if net.std(ddof=1) > 0 else float("nan")
            ),
            "max_drawdown": float((net.cumsum() - net.cumsum().cummax()).min()),  # qlib's: of summed returns
            "account_drawdown": float((report["account"] / report["account"].cummax() - 1).min()),
            "annualized_cost": float(report["cost"].mean() * N),
            "annualized_funding": float(report["funding"].mean() * N),
            "turnover": float(report["turnover"].mean()),
            "gross_exposure": float(report["gross_exposure"].mean()),
            "net_exposure": float(report["net_exposure"].mean()),
            "funding_missing": float(report["funding_missing"].sum() / held.sum()) if held.sum() else 0.0,
            "tied_boundary": float(report["tied_boundary"].sum() / max(1, report["rebalanced"].sum())),
            "final_account": float(report["account"].iloc[-1]),
        }


def to_scores(signal: Union[pd.Series, pd.DataFrame]) -> pd.Series:
    """A score series indexed by (datetime, instrument) from a prediction: a Series, or a frame
    with a ``score`` column or a single column (qlib's ``pred.pkl``)."""
    if isinstance(signal, pd.DataFrame):
        if "score" in signal.columns:
            signal = signal["score"]
        elif signal.shape[1] == 1:
            signal = signal.iloc[:, 0]
        else:
            raise ValueError(f"The signal has columns {list(signal.columns)}: give a 'score' column")
    if not isinstance(signal, pd.Series) or not isinstance(signal.index, pd.MultiIndex):
        raise ValueError("The signal is a Series indexed by (datetime, instrument)")
    names = list(signal.index.names)
    if set(names) >= {"datetime", "instrument"}:
        signal = signal.reorder_levels(["datetime", "instrument"])
    elif signal.index.nlevels != 2:
        raise ValueError(f"The signal's index has levels {names}: use (datetime, instrument)")
    signal = signal.astype("float64").copy()
    signal.index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(signal.index.get_level_values(0)), signal.index.get_level_values(1).astype(str)],
        names=["datetime", "instrument"],
    )
    if signal.index.has_duplicates:
        raise ValueError("The signal repeats (datetime, instrument) pairs")
    return signal.sort_index()


def backtest_long_short(
    signal: Union[pd.Series, pd.DataFrame],
    *,
    portfolio: Optional[LongShortPortfolio] = None,
    costs: Optional[CostModel] = None,
    market: Optional[PerpMarket] = None,
    start_time: Any = None,
    end_time: Any = None,
    account: float = 1_000_000.0,
    delay: int = 1,
    rebalance: int = 1,
    deal_price: str = "close",
    benchmark: Optional[str] = None,
    funding_missing: str = "zero",
) -> BacktestResult:
    """Backtest the long-short book of ``portfolio`` on ``signal``; see the module's docstring.

    signal : scores indexed by (datetime, instrument), such as qlib's ``pred.pkl``. Only the
        instruments scored on a day can be chosen that day, so a signal from a qlib dataset on a
        point-in-time universe (``top100``) trades that universe.
    market : the prices, funding and delistings; read from qlib (``qlib.init`` first) when None.
    start_time, end_time : the days of the report; by default from the first trade to the day
        after the last.
    account : the starting equity, in USDT. Impact costs grow with it.
    delay : days from a score to its trade (>= 1 with ``deal_price="vwap"``).
    rebalance : decide every ``rebalance`` days, counted from the first scored day; a decision
        day without scores does not trade, and positions drift with prices in between.
    benchmark : an instrument whose close-to-close return is the report's ``bench``; 0 when None.
    """
    portfolio = portfolio or LongShortPortfolio(quantile=0.1)
    costs = costs or CostModel()
    if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
        raise ValueError(f"delay must be an integer >= 0, not {delay!r}")
    if isinstance(rebalance, bool) or not isinstance(rebalance, int) or rebalance < 1:
        raise ValueError(f"rebalance must be an integer >= 1, not {rebalance!r}")
    if deal_price not in DEAL_PRICES:
        raise ValueError(f"deal_price must be one of {DEAL_PRICES}, not {deal_price!r}")
    if deal_price == "vwap" and delay < 1:
        raise ValueError("deal_price='vwap' trades through the day, so it needs delay >= 1")
    if delay == 0:
        logger.warning("delay=0 trades at the close the scores were computed from: that cannot be done live")
    if funding_missing not in FUNDING_POLICIES:
        raise ValueError(f"funding_missing must be one of {FUNDING_POLICIES}, not {funding_missing!r}")
    if not (isinstance(account, (int, float)) and math.isfinite(account) and account > 0):
        raise ValueError(f"account must be a positive number, not {account!r}")

    scores = to_scores(signal)
    if end_time is not None:
        # Scores after the report's last day cannot trade inside it.
        late = scores.index.get_level_values("datetime") > pd.Timestamp(end_time)
        if late.any():
            logger.info("Ignoring %d scores dated after end_time %s", int(late.sum()), pd.Timestamp(end_time).date())
            scores = scores[~late]
    if scores.empty:
        raise ValueError("The signal is empty")
    signal_days = pd.DatetimeIndex(scores.index.get_level_values("datetime").unique()).sort_values()
    if market is None:
        wanted = set(scores.index.get_level_values("instrument"))
        if benchmark is not None:
            wanted.add(benchmark)
        load_start = signal_days[0] - pd.Timedelta(costs.window + 5, unit="D")
        load_end = (
            pd.Timestamp(end_time) if end_time is not None else signal_days[-1] + pd.Timedelta(delay + 1, unit="D")
        )
        market = PerpMarket.load(wanted, load_start, load_end)
    days = market.days
    names = [str(name) for name in market.instruments]
    position = {day: i for i, day in enumerate(days)}
    missing_days = [day for day in signal_days if day not in position]
    if missing_days:
        raise ValueError(f"Signal days are not in the market's calendar: {[str(d.date()) for d in missing_days[:5]]}")
    unknown = sorted(set(scores.index.get_level_values("instrument")) - set(names))
    if unknown:
        raise ValueError(f"Scored instruments are not in the market: {unknown[:5]}")
    if benchmark is not None and benchmark not in names:
        raise ValueError(f"The benchmark {benchmark!r} is not in the market")

    decisions = sorted(position[day] for day in signal_days)
    first = decisions[0]
    by_day = {day: group.droplevel("datetime") for day, group in scores.groupby(level="datetime")}
    executions: Dict[int, pd.Timestamp] = {}
    for p in decisions:
        if (p - first) % rebalance == 0 and p + delay < len(days):
            executions[p + delay] = days[p]
    if not executions:
        raise ValueError("No scored day can be traded: the market ends before the first trade")
    start_pos = _locate(days, start_time, side="left") if start_time is not None else min(executions)
    if end_time is not None:
        end_pos = _locate(days, end_time, side="right")
    else:
        end_pos = min(max(executions) + 1, len(days) - 1)
    if start_pos > end_pos:
        raise ValueError(f"No day between start_time {start_time} and end_time {end_time}")

    close = market.close.to_numpy(dtype="float64")
    vwap = market.vwap.to_numpy(dtype="float64")
    funding = market.funding.to_numpy(dtype="float64")
    delist = market.delist.to_numpy(dtype=bool)
    adv = rolling_adv(market.amount, costs.window, costs.min_periods).to_numpy(dtype="float64")
    sigma = rolling_sigma(market.close, costs.window, costs.min_periods).to_numpy(dtype="float64")
    fee = costs.fee_rates(names)
    column = {name: j for j, name in enumerate(names)}
    bench = benchmark_returns(market.close[benchmark]) if benchmark is not None else None

    n = len(names)
    quantity = np.zeros(n)
    mark = np.full(n, np.nan)
    nav = float(account)
    ruined = False
    rows: List[Dict[str, float]] = []
    weight_rows: List[np.ndarray] = []
    trade_rows: List[Dict[str, object]] = []

    def targets(t: int, equity: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        """The trade toward the targets of the scores decided for day t: its notional at the deal
        price, its quantity, and whether scores tie at the edge of a side. The quantity is the
        target quantity less the held one, so a position sold out is exactly 0."""
        day_scores = by_day[executions[t]]
        leaving = [names[j] for j in np.flatnonzero(delist[t])]
        if leaving:
            day_scores = day_scores.drop(labels=[name for name in leaving if name in day_scores.index])
        weights = np.zeros(n)
        held = quantity != 0
        now = pd.Series(quantity[held] * mark[held] / equity, index=[names[j] for j in np.flatnonzero(held)])
        for name, weight in portfolio.target(day_scores, current=now).items():
            weights[column[name]] = weight
        deal = close[t] if deal_price == "close" else vwap[t]
        tradable = np.isfinite(close[t]) & np.isfinite(deal) & (deal > 0) & ~delist[t]
        price = np.where(tradable, deal, 1.0)
        change = np.where(tradable, weights * equity / price - quantity, 0.0)
        notional = change * price
        change[np.abs(notional) < _DUST * equity] = 0.0
        notional = change * price
        capped = costs.cap(notional, adv[t])
        scale = np.divide(capped, notional, out=np.zeros(n), where=notional != 0)
        change = np.where(capped == notional, change, change * scale)
        return capped, change, portfolio.boundary_tied(day_scores)

    for t in range(start_pos, end_pos + 1):
        day = days[t]
        previous = nav
        held = quantity != 0
        value = np.where(held, quantity * np.where(held, mark, 0.0), 0.0)  # at the last marks
        price = close[t]
        priced = np.isfinite(price)
        deal = price if deal_price == "close" else vwap[t]
        trade = np.zeros(n)
        change = np.zeros(n)  # the quantity traded
        tied = False
        rebalanced = t in executions and not ruined

        if rebalanced and deal_price == "vwap":
            trade, change, tied = targets(t, previous)

        # 1. mark at the close
        moved = held & priced
        pnl = float(np.sum(quantity[moved] * (price[moved] - mark[moved])))
        if deal_price == "vwap":
            traded = change != 0
            pnl += float(np.sum(change[traded] * (price[traded] - deal[traded])))

        # 2. funding on what was held through the day
        rate = funding[t]
        known = np.isfinite(rate)
        through = value + trade / 2 if deal_price == "vwap" else value
        exposed = through != 0
        missing = exposed & ~known
        if missing.any() and funding_missing == "raise":
            raise ValueError(f"{day.date()}: no $funding for held {[names[j] for j in np.flatnonzero(missing)][:5]}")
        funding_pnl = -float(np.sum(through[exposed & known] * rate[exposed & known]))

        if deal_price == "vwap":
            quantity = quantity + change
        mark = np.where(priced, price, mark)
        equity = previous + pnl + funding_pnl

        # 3. trade toward the targets at the close
        if rebalanced and deal_price == "close" and equity > 0:
            trade, change, tied = targets(t, equity)
            quantity = quantity + change

        # 4. close what is delisted today, at its last mark
        forced = delist[t] & (quantity != 0)
        closing = np.where(forced, -quantity * np.where(forced, mark, 0.0), 0.0)
        quantity = np.where(forced, 0.0, quantity)

        # 5. costs
        notional = np.abs(trade) + np.abs(closing)
        rates = costs.rates(notional, fee, adv[t], sigma[t])
        paid = rates * notional
        cost = float(paid.sum())
        fee_paid = float(np.sum(fee * notional))
        nav = equity - cost

        if nav <= 0 and not ruined:
            logger.error(
                f"{day.date()}: the account's equity reached {nav:.2f}: everything is closed and trading stops"
            )
            ruined = True
            quantity = np.zeros(n)
            nav = 0.0

        for j in np.flatnonzero(notional):
            for amount, kind in ((trade[j], "rebalance"), (closing[j], "delist")):
                if amount != 0:
                    trade_rows.append(
                        {
                            "datetime": day,
                            "instrument": names[j],
                            "value": float(amount),
                            "price": float(deal[j] if kind == "rebalance" else mark[j]),
                            "cost": float(rates[j] * abs(amount)),
                            "kind": kind,
                        }
                    )

        after = np.where(quantity != 0, quantity * np.where(quantity != 0, mark, 0.0), 0.0)
        weights = after / nav if nav > 0 else np.zeros(n)
        base = previous if previous > 0 else 1.0
        rows.append(
            {
                "account": nav,
                "return": (pnl + funding_pnl) / base if previous > 0 else 0.0,
                "cost": cost / base if previous > 0 else 0.0,
                "bench": float(bench.iloc[t]) if bench is not None else 0.0,
                "turnover": float(notional.sum()) / base if previous > 0 else 0.0,
                "funding": funding_pnl / base if previous > 0 else 0.0,
                "fee": fee_paid / base if previous > 0 else 0.0,
                "slippage": (cost - fee_paid) / base if previous > 0 else 0.0,
                "long_exposure": float(weights[weights > 0].sum()),
                "short_exposure": float(weights[weights < 0].sum()),
                "gross_exposure": float(np.abs(weights).sum()),
                "net_exposure": float(weights.sum()),
                "n_long": int((weights > 0).sum()),
                "n_short": int((weights < 0).sum()),
                "funding_missing": int(missing.sum()),
                "rebalanced": bool(rebalanced),
                "tied_boundary": bool(tied),
            }
        )
        weight_rows.append(weights)

    index = pd.DatetimeIndex(days[start_pos : end_pos + 1], name="datetime")
    report = pd.DataFrame(rows, index=index)
    weights = pd.DataFrame(np.vstack(weight_rows), index=index, columns=pd.Index(names, name="instrument"))
    weights = weights.loc[:, (weights != 0).any(axis=0)]
    trades = pd.DataFrame(trade_rows, columns=["datetime", "instrument", "value", "price", "cost", "kind"])
    if int(report["funding_missing"].sum()):
        logger.warning(
            f"$funding was missing for {int(report['funding_missing'].sum())} position-days "
            f"(the last on {report.index[report['funding_missing'] > 0][-1].date()}): they were charged nothing"
        )
    tied_days = int(report["tied_boundary"].sum())
    if tied_days:
        logger.warning(
            f"On {tied_days} of {int(report['rebalanced'].sum())} rebalance days two scores were equal at the edge "
            "of the long or short side: the names held there were chosen by instrument name, not by the model"
        )
    config = {
        "portfolio": portfolio.to_dict(),
        "costs": costs.to_dict(),
        "account": float(account),
        "delay": delay,
        "rebalance": rebalance,
        "deal_price": deal_price,
        "benchmark": benchmark,
        "funding_missing": funding_missing,
        "start_time": str(index[0].date()),
        "end_time": str(index[-1].date()),
    }
    return BacktestResult(report, weights, trades, config)


def _locate(days: pd.DatetimeIndex, moment: Any, side: str) -> int:
    """The first day at or after ``moment`` (side='left'), or the last at or before (side='right')."""
    moment = pd.Timestamp(moment)
    if side == "left":
        found = int(days.searchsorted(moment, side="left"))
        if found >= len(days):
            raise ValueError(f"No market day at or after {moment.date()}")
        return found
    found = int(days.searchsorted(moment, side="right")) - 1
    if found < 0:
        raise ValueError(f"No market day at or before {moment.date()}")
    return found
