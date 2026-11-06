"""The crypto region: a daily market that trades every day, in fractional amounts, without a
price limit, and annualizes over 365 days."""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import qlib
from qlib.backtest.decision import Order, OrderDir
from qlib.backtest.exchange import Exchange
from qlib.config import C
from qlib.constant import REG_CN, REG_CRYPTO
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from qlib.data.cache import H
from qlib.utils.time import get_min_cal, is_single_value


@pytest.fixture
def crypto_market(tmp_path):
    """Two perpetuals over 14 calendar days, weekends included. BTCUSDT gains 40% on the 6th day."""
    calendar = pd.date_range("2024-01-01", periods=14, freq="D")
    provider_uri = tmp_path / "crypto"
    (provider_uri / "calendars").mkdir(parents=True)
    (provider_uri / "calendars" / "day.txt").write_text("".join(f"{day:%Y-%m-%d}\n" for day in calendar))
    (provider_uri / "instruments").mkdir()
    spans = "".join(f"{symbol}\t2024-01-01\t2024-01-14\n" for symbol in ("BTCUSDT", "ETHUSDT"))
    (provider_uri / "instruments" / "all.txt").write_text(spans)
    btc = np.full(len(calendar), 40000.0)
    btc[5:] = 56000.0
    closes = {"BTCUSDT": btc, "ETHUSDT": np.linspace(2000.0, 2600.0, len(calendar))}
    for symbol, close in closes.items():
        change = np.r_[np.nan, close[1:] / close[:-1] - 1]
        values = {
            "close": close,
            "open": close,
            "vwap": close,
            "volume": np.full(len(calendar), 1000.0),
            "factor": np.ones(len(calendar)),
            "change": change,
        }
        directory = provider_uri / "features" / symbol.lower()
        directory.mkdir(parents=True)
        for field, array in values.items():
            np.concatenate(([0], array)).astype("<f4").tofile(directory / f"{field}.day.bin")
    previous = deepcopy(C)
    qlib.init(provider_uri=str(provider_uri), region=REG_CRYPTO, expression_cache=None, dataset_cache=None)
    try:
        yield calendar
    finally:
        H.clear()
        C.set_conf_from_C(previous)
        if previous.registered:
            C.register()


def test_the_region_disables_trade_units_and_price_limits(crypto_market):
    assert C.region == REG_CRYPTO
    assert C.trade_unit is None and C.limit_threshold is None and C.deal_price == "close"
    calendar = D.calendar(freq="day")
    assert len(calendar) == 14 and {day.weekday() for day in calendar} == set(range(7))
    exchange = Exchange(freq="day", start_time="2024-01-01", end_time="2024-01-14", codes="all", min_cost=0.0)
    assert exchange.trade_unit is None and exchange.limit_type == Exchange.LT_NONE
    jump = pd.Timestamp("2024-01-06")
    # a 40% day is not a limit: the instrument can be bought and sold
    assert exchange.is_stock_tradable("BTCUSDT", jump, jump, direction=OrderDir.BUY)
    assert exchange.is_stock_tradable("BTCUSDT", jump, jump, direction=OrderDir.SELL)
    # a fractional amount is not rounded away
    assert exchange.round_amount_by_trade_unit(0.0123, factor=1.0) == pytest.approx(0.0123)
    order = Order("BTCUSDT", 0.0123, OrderDir.BUY, jump, jump)
    value, cost, price = exchange.deal_order(order, position=_cash(1_000_000))
    assert order.deal_amount == pytest.approx(0.0123)
    assert price == pytest.approx(56000.0) and value == pytest.approx(0.0123 * 56000.0)
    assert cost == pytest.approx(value * exchange.open_cost)


def _cash(amount):
    from qlib.backtest.position import Position

    return Position(cash=amount)


def test_risk_analysis_annualizes_over_365_days(crypto_market):
    returns = pd.Series([0.01, -0.005, 0.002, 0.0, 0.004])
    crypto = risk_analysis(returns, freq="day")
    assert crypto.loc["annualized_return", "risk"] == pytest.approx(returns.mean() * 365)
    assert risk_analysis(returns, freq="1min").loc["annualized_return", "risk"] == pytest.approx(
        returns.mean() * 365 * 1440
    )
    C["region"] = REG_CN
    assert risk_analysis(returns, freq="day").loc["annualized_return", "risk"] == pytest.approx(returns.mean() * 238)


def test_the_minute_calendar_is_the_whole_day():
    minutes = get_min_cal(region=REG_CRYPTO)
    assert len(minutes) == 1440 and minutes[0].strftime("%H:%M") == "00:00"
    assert minutes[-1].strftime("%H:%M") == "23:59"
    assert get_min_cal(shift=1, region=REG_CRYPTO)[0].strftime("%H:%M") == "23:59"
    start = pd.Timestamp("2024-01-01 23:59")
    assert is_single_value(start, start + pd.Timedelta("5min"), pd.Timedelta("1min"), REG_CRYPTO)
    start = pd.Timestamp("2024-01-01 12:00")
    assert not is_single_value(start, start + pd.Timedelta("5min"), pd.Timedelta("1min"), REG_CRYPTO)
