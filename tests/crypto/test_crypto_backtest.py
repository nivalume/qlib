"""The perpetual long-short backtest, checked against sums done by hand."""

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.crypto import BacktestResult, CostModel, LongShortPortfolio, PerpMarket, backtest_long_short

FREE = CostModel(fee=0.0, spread_tiers=((0.0, 0.0),), impact=0.0)
PAIR = LongShortPortfolio(n_long=1, n_short=1, min_names=2)


def market(close, *, vwap=None, funding=None, amount=None, delist=None, start="2024-01-01"):
    """A market from {instrument: [one value a day]} frames."""
    close = pd.DataFrame(close, dtype="float64")
    days = pd.date_range(start, periods=len(close), freq="D", name="datetime")
    close.index = days
    close.columns = pd.Index(close.columns, name="instrument")

    def frame(values, fill):
        out = pd.DataFrame(fill, index=days, columns=close.columns, dtype="float64")
        for name, column in (values or {}).items():
            out[name] = column
        return out

    flags = pd.DataFrame(False, index=days, columns=close.columns)
    for name, day in (delist or {}).items():
        flags.loc[pd.Timestamp(day), name] = True
    return PerpMarket(
        close=close,
        vwap=frame(vwap, np.nan) if vwap is not None else close.copy(),
        funding=frame(funding, 0.0),
        amount=frame(amount, 1e12),
        delist=flags,
    )


def signal(days, **scores):
    """Scores for each day of ``days``: {instrument: score} the same every day."""
    index = pd.MultiIndex.from_product([pd.DatetimeIndex(days), sorted(scores)], names=["datetime", "instrument"])
    return pd.Series([scores[name] for _ in days for name in sorted(scores)], index=index, dtype="float64")


def run(mkt, sig, **kwargs) -> BacktestResult:
    kwargs.setdefault("portfolio", PAIR)
    kwargs.setdefault("costs", FREE)
    return backtest_long_short(sig, market=mkt, **kwargs)


def test_a_score_trades_at_the_next_close_and_earns_the_day_after():
    mkt = market(
        {"A": [100, 110, 121, 121], "B": [100, 100, 90, 99]}, funding={"A": [0, 0, 0.001, 0], "B": [0, 0, 0.002, 0]}
    )
    result = run(mkt, signal(["2024-01-01", "2024-01-02"], A=1.0, B=-1.0))
    report = result.report
    assert list(report.index.strftime("%m-%d")) == ["01-02", "01-03", "01-04"]
    # 01-02: the score of 01-01 buys A and sells B at the close; nothing was held through the day
    assert report["return"].iloc[0] == 0
    assert result.weights.iloc[0].to_dict() == pytest.approx({"A": 0.5, "B": -0.5})
    # 01-03: A +10%, B -10%: 0.5 * 0.1 + 0.5 * 0.1; funding: the long pays 0.1%, the short gets 0.2%
    assert report["return"].iloc[1] == pytest.approx(0.1 + (-0.5 * 0.001 + 0.5 * 0.002))
    assert report["funding"].iloc[1] == pytest.approx(0.0005)
    # 01-04: rebalanced to 0.5 / -0.5 of 1,100,500 at the close of 01-03; A flat, B +10%
    assert report["return"].iloc[2] == pytest.approx(-0.05)
    assert report["account"].iloc[-1] == pytest.approx(1_100_500 * 0.95)
    assert result.config["end_time"] == "2024-01-04"


def test_the_return_is_the_label_of_the_targets():
    """Daily rebalancing without costs: the day's return is the targets' label, with funding:
    Ref($close, -2) / Ref($close, -1) - 1 - Ref($funding, -2) of the decision day."""
    rng = np.random.default_rng(7)
    names = [f"C{i:02d}" for i in range(20)]
    days = 40
    close = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.03, (days, 20)), axis=0)), columns=names)
    rates = pd.DataFrame(rng.normal(0.0001, 0.0005, (days, 20)), columns=names)
    mkt = market(close.to_dict("list"), funding=rates.to_dict("list"))
    index = pd.MultiIndex.from_product([mkt.days[:-2], names], names=["datetime", "instrument"])
    scores = pd.Series(rng.normal(size=len(index)), index=index)
    book = LongShortPortfolio(quantile=0.2, gross=1.5, net=0.1, weighting="rank", min_names=5)
    result = run(mkt, scores, portfolio=book)
    c, f = mkt.close, mkt.funding
    label = (c.shift(-2) / c.shift(-1) - 1) - f.shift(-2)
    for t, day in enumerate(mkt.days[:-2]):
        weights = book.target(scores.loc[day])
        expected = float((weights * label.loc[day, weights.index]).sum())
        trade_day = mkt.days[t + 2]
        net = result.report.loc[trade_day, "return"]
        assert net == pytest.approx(expected, abs=1e-12), day
    # a position that is sold leaves nothing behind: 4 names a side, counted exactly
    assert (result.report["n_long"] == 4).all() and (result.report["n_short"] == 4).all()
    assert not ((result.weights.abs() > 0) & (result.weights.abs() < 1e-9)).to_numpy().any()
    assert result.report["gross_exposure"].iloc[:-1].tolist() == pytest.approx([1.5] * (len(result.report) - 1))
    assert result.report["net_exposure"].iloc[:-1].tolist() == pytest.approx([0.1] * (len(result.report) - 1))


def test_costs_are_charged_on_what_trades():
    mkt = market({"A": [100, 100, 100], "B": [100, 100, 100]})
    costs = CostModel(fee=0.0005, fees={"B": 0.001}, spread_tiers=((0.0, 0.0002),), impact=0.0)
    result = run(mkt, signal(["2024-01-01", "2024-01-02"], A=1.0, B=-1.0), costs=costs)
    first = result.report.iloc[0]
    # 500,000 each side: A pays 0.05% + 0.02%, B 0.1% + 0.02%
    assert first["cost"] == pytest.approx((500_000 * 0.0007 + 500_000 * 0.0012) / 1_000_000)
    assert first["fee"] == pytest.approx((500_000 * 0.0005 + 500_000 * 0.001) / 1_000_000)
    assert first["turnover"] == pytest.approx(1.0)
    # the next day trades only back to target, on the equity the costs left
    second = result.report.iloc[1]
    equity = 1_000_000 * (1 - first["cost"])
    assert second["turnover"] == pytest.approx(2 * (500_000 - equity / 2) / equity)
    trades = result.trades
    assert set(trades["kind"]) == {"rebalance"} and len(trades) == 4


def test_impact_grows_with_the_trade():
    costs = CostModel(fee=0.0, spread_tiers=((1e8, 0.0001), (0.0, 0.002)), impact=0.5)
    adv = np.array([2e8, 1e7, np.nan])
    sigma = np.array([0.04, 0.04, 0.04])
    rates = costs.rates(np.array([2e6, 1e5, 1e5]), np.zeros(3), adv, sigma)
    assert rates[0] == pytest.approx(0.0001 + 0.5 * 0.04 * np.sqrt(2e6 / 2e8))
    assert rates[1] == pytest.approx(0.002 + 0.5 * 0.04 * np.sqrt(1e5 / 1e7))
    assert rates[2] == pytest.approx(0.002)  # no known volume: the last tier, no impact


def test_a_trade_is_capped_by_its_share_of_volume():
    mkt = market({"A": [100] * 40, "B": [100] * 40}, amount={"A": [1e6] * 40, "B": [1e9] * 40})
    costs = CostModel(fee=0.0, spread_tiers=((0.0, 0.0),), impact=0.0, max_participation=0.1)
    days = mkt.days[31:33]
    result = run(mkt, signal(days, A=1.0, B=-1.0), costs=costs)
    weights = result.weights
    # A trades 10% of its 1,000,000 a day: 100,000, then 100,000 more; B reaches its target at once
    assert weights["A"].iloc[0] == pytest.approx(0.1)
    assert weights["A"].iloc[1] == pytest.approx(0.2)
    assert weights["B"].iloc[0] == pytest.approx(-0.5)


def test_a_day_without_a_price_keeps_the_mark_and_does_not_trade():
    mkt = market({"A": [100, 100, np.nan, 120, 120], "B": [100, 100, 100, 100, 100]})
    result = run(mkt, signal(mkt.days[:4], A=1.0, B=-1.0))
    report = result.report.set_index(result.report.index.strftime("%m-%d"))
    assert report.loc["01-03", "return"] == 0  # A has no close: no move, no trade
    assert result.weights.loc["2024-01-03", "A"] == pytest.approx(0.5)
    # 01-04: A is marked from 100 to 120 at last
    assert report.loc["01-04", "return"] == pytest.approx(0.5 * 0.2)


def test_a_delisted_position_is_closed_at_its_last_close():
    mkt = market(
        {"A": [100, 100, 80, np.nan, np.nan], "B": [100, 100, 100, 100, 100], "C": [50, 50, 50, 50, 50]},
        delist={"A": "2024-01-03"},
    )
    sig = signal(mkt.days[:4], A=3.0, C=2.0, B=1.0)
    book = LongShortPortfolio(n_long=1, n_short=1, min_names=2)
    result = run(mkt, sig, portfolio=book, costs=CostModel(fee=0.001, spread_tiers=((0.0, 0.0),), impact=0.0))
    report = result.report.set_index(result.report.index.strftime("%m-%d"))
    # long 500,000 of A from 01-02 (sized before that day's 0.1% fee); it falls 20% on 01-03, its
    # last day, and is closed at that close
    assert report.loc["01-03", "return"] == pytest.approx(500_000 * -0.2 / (1_000_000 * 0.999))
    delisted = result.trades[result.trades["kind"] == "delist"]
    assert delisted["instrument"].tolist() == ["A"] and delisted["price"].tolist() == [80]
    assert delisted["value"].iloc[0] == pytest.approx(-500_000 * 0.8)
    # nor is A a target on its last day: C takes the long side then
    assert "A" not in result.weights.columns[result.weights.loc["2024-01-03"] != 0]
    assert result.weights.loc["2024-01-03", "C"] > 0
    assert result.weights.loc["2024-01-04", "A"] == 0


def test_vwap_trades_through_the_day():
    mkt = market(
        {"A": [100, 100, 110, 110], "B": [100, 100, 100, 100]},
        vwap={"A": [100, 105, 108, 110], "B": [100, 100, 100, 100]},
        funding={"A": [0, 0.001, 0, 0], "B": [0, 0, 0, 0]},
    )
    result = run(mkt, signal(mkt.days[:2], A=1.0, B=-1.0), deal_price="vwap")
    report = result.report.set_index(result.report.index.strftime("%m-%d"))
    # 01-02: buys 500,000 of A at 105 (4761.9 coins), marked at 100 at the close; funding on half
    assert report.loc["01-02", "return"] == pytest.approx((500_000 / 105 * (100 - 105) - 250_000 * 0.001) / 1e6)
    # 01-03: the coins held earn from the close of 100 to 110; the target is half the equity in
    # coins at that day's vwap of 108, and the coins sold or bought at 108 are marked to 110
    equity = 1e6 * (1 + report.loc["01-02", "return"])
    held_a = 500_000 / 105
    target_a = 0.5 * equity / 108
    expected = held_a * (110 - 100) + (target_a - held_a) * (110 - 108)
    assert report.loc["01-03", "return"] == pytest.approx(expected / equity)
    with pytest.raises(ValueError, match="needs delay >= 1"):
        run(mkt, signal(mkt.days[:2], A=1.0, B=-1.0), deal_price="vwap", delay=0)


def test_a_position_sold_out_at_vwap_leaves_nothing_behind():
    mkt = market(
        {"A": [100, 100, 110, 110], "B": [100] * 4, "C": [100] * 4},
        vwap={"A": [100, 105, 108, 110], "B": [100] * 4, "C": [100] * 4},
    )
    index = pd.MultiIndex.from_product([mkt.days[:2], ["A", "B", "C"]], names=["datetime", "instrument"])
    scores = pd.Series([3.0, 1.0, 2.0, 2.0, 1.0, 3.0], index=index)  # long A, then long C: A is sold out
    result = run(mkt, scores, deal_price="vwap")
    assert result.weights.loc["2024-01-03", "A"] == 0
    assert (result.report["n_long"] == 1).all() and (result.report["n_short"] == 1).all()
    sold = result.trades[(result.trades["instrument"] == "A") & (result.trades["datetime"] == "2024-01-03")]
    assert sold["value"].iloc[0] == pytest.approx(-500_000 / 105 * 108)  # all the coins, at the vwap of 108


def test_ties_at_the_edge_of_a_side_are_reported():
    mkt = market({f"S{i}": [100] * 6 for i in range(6)})
    flat = signal(mkt.days[:3], **{f"S{i}": 1.0 for i in range(6)})
    result = run(mkt, flat, portfolio=LongShortPortfolio(n_long=2, n_short=2, min_names=4))
    rebalanced = result.report["rebalanced"]
    assert result.report.loc[rebalanced, "tied_boundary"].all() and result.summary()["tied_boundary"] == 1.0
    distinct = signal(mkt.days[:3], **{f"S{i}": float(i) for i in range(6)})
    result = run(mkt, distinct, portfolio=LongShortPortfolio(n_long=2, n_short=2, min_names=4))
    assert not result.report["tied_boundary"].any() and result.summary()["tied_boundary"] == 0.0


def test_positions_drift_between_rebalances():
    mkt = market({"A": [100, 100, 200, 200, 200], "B": [100, 100, 100, 100, 100]})
    result = run(mkt, signal(mkt.days[:4], A=1.0, B=-1.0), rebalance=2)
    rebalanced = result.report["rebalanced"].tolist()
    assert rebalanced == [True, False, True, False]
    # A doubles on 01-03: no rebalance that day, so it is 1.0 / 1.5 of the equity at the close
    assert result.weights.loc["2024-01-03", "A"] == pytest.approx(1.0 / 1.5)
    assert result.weights.loc["2024-01-04", "A"] == pytest.approx(0.5)


def test_missing_funding_is_counted_or_refused():
    mkt = market({"A": [100] * 4, "B": [100] * 4}, funding={"A": [0, 0, np.nan, 0], "B": [0, 0, 0.001, 0]})
    result = run(mkt, signal(mkt.days[:3], A=1.0, B=-1.0))
    assert result.report["funding_missing"].tolist() == [0, 1, 0]
    assert result.report.loc["2024-01-03", "funding"] == pytest.approx(0.5 * 0.001)
    with pytest.raises(ValueError, match="no \\$funding for held"):
        run(mkt, signal(mkt.days[:3], A=1.0, B=-1.0), funding_missing="raise")


def test_too_few_names_hold_nothing():
    mkt = market({"A": [100] * 4, "B": [100] * 4})
    result = run(mkt, signal(mkt.days[:3], A=1.0, B=-1.0), portfolio=LongShortPortfolio(n_long=1, n_short=1))
    assert (result.report["gross_exposure"] == 0).all() and result.trades.empty


def test_the_benchmark_and_the_analysis():
    mkt = market({"A": [100, 100, 110, 121], "B": [100, 100, 100, 100], "BTC": [10, 10, 11, 11]})
    result = run(mkt, signal(mkt.days[:2], A=1.0, B=-1.0), benchmark="BTC")
    assert result.report["bench"].tolist() == pytest.approx([0.0, 0.1, 0.0])
    analysis = result.analysis()
    assert set(analysis.index.get_level_values(0)) == {"excess_return_without_cost", "excess_return_with_cost"}
    net = result.report["return"] - result.report["bench"]
    assert analysis.loc[("excess_return_without_cost", "annualized_return"), "risk"] == pytest.approx(net.mean() * 365)
    summary = result.summary()
    assert summary["days"] == 3 and summary["final_account"] == pytest.approx(result.report["account"].iloc[-1])
    account = result.report["account"]
    assert summary["account_drawdown"] == pytest.approx((account / account.cummax() - 1).min())


def test_bad_arguments_are_refused():
    mkt = market({"A": [100] * 3, "B": [100] * 3})
    sig = signal(mkt.days[:2], A=1.0, B=-1.0)
    with pytest.raises(ValueError, match="deal_price"):
        run(mkt, sig, deal_price="open")
    with pytest.raises(ValueError, match="rebalance"):
        run(mkt, sig, rebalance=0)
    with pytest.raises(ValueError, match="not in the market"):
        run(mkt, signal(mkt.days[:2], A=1.0, Z=-1.0))
    with pytest.raises(ValueError, match="calendar"):
        run(mkt, signal(["2023-12-31"], A=1.0, B=-1.0))
    with pytest.raises(ValueError, match="benchmark"):
        run(mkt, sig, benchmark="ETH")
