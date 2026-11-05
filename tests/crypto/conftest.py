"""A small crypto market in a qlib directory, as ``jarvis-data dump qlib`` writes one, and an
isolated qlib session on it."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


def write_market(path, calendar, columns, spans, pools):
    """``columns``: {instrument: {field: array on the calendar}}; ``spans``: {instrument: [(first,
    last)]} for all.txt; ``pools``: {name: {instrument: [(first, last)]}}."""
    (path / "calendars").mkdir(parents=True)
    days = "".join(f"{day:%Y-%m-%d}\n" for day in calendar)
    (path / "calendars" / "day.txt").write_text(days)
    future = calendar.append(pd.date_range(calendar[-1] + pd.Timedelta(1, unit="D"), periods=30, freq="D"))
    (path / "calendars" / "day_future.txt").write_text("".join(f"{day:%Y-%m-%d}\n" for day in future))
    (path / "instruments").mkdir()

    def lines(table):
        return "".join(
            f"{name}\t{pd.Timestamp(a):%Y-%m-%d}\t{pd.Timestamp(b):%Y-%m-%d}\n"
            for name in sorted(table)
            for a, b in table[name]
        )

    (path / "instruments" / "all.txt").write_text(lines(spans))
    for name, table in pools.items():
        (path / "instruments" / f"{name}.txt").write_text(lines(table))
    for name, fields in columns.items():
        directory = path / "features" / name.lower()
        directory.mkdir(parents=True)
        for field, values in fields.items():
            np.concatenate(([0], values)).astype("<f4").tofile(directory / f"{field}.day.bin")


def synthetic_columns(names, days, seed=0, delisted=None):
    """Prices with a momentum signal in them, so a model has something to learn."""
    rng = np.random.default_rng(seed)
    columns = {}
    for i, name in enumerate(names):
        drift = rng.normal(0, 0.002)
        noise = rng.normal(0, 0.03, days)
        returns = drift + noise
        returns[1:] += 0.2 * noise[:-1]
        close = 10.0 * (i + 1) * np.exp(np.cumsum(returns))
        opening = close * (1 + rng.normal(0, 0.005, days))
        volume = rng.uniform(1e5, 2e5, days)
        amount = volume * close
        funding = rng.normal(0.0001, 0.0002, days)
        if delisted is not None and name in delisted:
            stop = delisted[name]
            for values in (close, opening, volume, amount, funding):
                values[stop + 1 :] = np.nan
        columns[name] = {
            "open": opening,
            "close": close,
            "high": np.fmax(opening, close) * 1.01,
            "low": np.fmin(opening, close) * 0.99,
            "vwap": (opening + close) / 2,
            "volume": volume,
            "amount": amount,
            "funding": funding,
            "factor": np.ones(days),
            "change": np.r_[np.nan, close[1:] / close[:-1] - 1],
        }
    return columns


@pytest.fixture
def crypto_session(tmp_path, monkeypatch):
    import qlib
    from qlib.config import C
    from qlib.constant import REG_CRYPTO
    from qlib.data.cache import H
    from qlib.workflow import R

    days = 240
    calendar = pd.date_range("2024-01-01", periods=days, freq="D")
    names = [f"C{i:02d}USDT" for i in range(24)]
    delisted = {"C05USDT": 200}
    columns = synthetic_columns(names, days, delisted=delisted)
    spans = {name: [(calendar[0], calendar[delisted.get(name, days - 1)])] for name in names}
    pools = {"top20": {name: spans[name] for name in names[:20]}}
    provider = tmp_path / "crypto_um"
    write_market(provider, calendar, columns, spans, pools)
    previous = deepcopy(C)
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    uri = "file:" + str((tmp_path / "mlruns").resolve())
    qlib.init(
        provider_uri=str(provider),
        region=REG_CRYPTO,
        kernels=1,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": uri, "default_exp_name": "crypto"},
        },
    )
    try:
        yield SimpleNamespace(
            provider=provider, calendar=calendar, names=names, delisted=delisted, mlruns=tmp_path / "mlruns", uri=uri
        )
    finally:
        R.end_exp()
        H.clear()
        C.set_conf_from_C(previous)
        if previous.registered:
            C.register()
