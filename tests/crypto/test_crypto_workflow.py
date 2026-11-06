"""The crypto loop inside qlib: the market from a dumped directory, a workflow that trains, predicts
and backtests the long-short book, and the live targets of the trained run."""

import json

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.crypto import LongShortPortfolio, PerpMarket, backtest_long_short
from qlib.contrib.crypto import live
from qlib.data import D

FEATURES = (
    ["$close/Ref($close,1)-1", "Mean($close,5)/$close-1", "$volume/Mean($volume,10)", "$funding"],
    ["RET1", "MA5", "VOL10", "FUND"],
)
LABEL = (["Ref($close,-2)/Ref($close,-1)-1-Ref($funding,-2)"], ["LABEL0"])


def task(calendar, record_config):
    day = lambda i: f"{calendar[i]:%Y-%m-%d}"  # noqa: E731
    handler = {
        "class": "DataHandlerLP",
        "module_path": "qlib.data.dataset.handler",
        "kwargs": {
            "start_time": day(0),
            "end_time": day(-1),
            "instruments": "top20",
            "data_loader": {
                "class": "QlibDataLoader",
                "module_path": "qlib.data.dataset.loader",
                "kwargs": {"config": {"feature": FEATURES, "label": LABEL}},
            },
            "infer_processors": [
                {
                    "class": "RobustZScoreNorm",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {
                        "fields_group": "feature",
                        "clip_outlier": True,
                        "fit_start_time": day(0),
                        "fit_end_time": day(119),
                    },
                },
                {
                    "class": "Fillna",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {"fields_group": "feature"},
                },
            ],
            "learn_processors": [
                {"class": "DropnaLabel", "module_path": "qlib.data.dataset.processor"},
                {
                    "class": "CSRankNorm",
                    "module_path": "qlib.data.dataset.processor",
                    "kwargs": {"fields_group": "label"},
                },
            ],
        },
    }
    return {
        "model": {"class": "LinearModel", "module_path": "qlib.contrib.model.linear", "kwargs": {"estimator": "ols"}},
        "dataset": {
            "class": "DatasetH",
            "module_path": "qlib.data.dataset",
            "kwargs": {
                "handler": handler,
                "segments": {
                    "train": (day(10), day(119)),
                    "valid": (day(120), day(159)),
                    "test": (day(160), day(-1)),
                },
            },
        },
        "record": [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            },
            {
                "class": "PerpPortAnaRecord",
                "module_path": "qlib.contrib.crypto.record",
                "kwargs": {"config": record_config},
            },
        ],
    }


RECORD = {
    "portfolio": {"quantile": 0.2, "min_names": 5, "buffer": 1},
    "costs": {"fee": 0.0005, "spread_tiers": [[0, 0.0002]], "impact": 0.0},
    "backtest": {"account": 1_000_000, "benchmark": "C00USDT"},
}


def test_the_market_reads_a_dumped_directory(crypto_session):
    calendar = crypto_session.calendar
    market = PerpMarket.load(["C05USDT", "C01USDT"], calendar[190], calendar[-1])
    assert market.days[0] == calendar[190] and market.days[-1] == calendar[-1]
    assert market.delist["C05USDT"].sum() == 1 and market.delist.loc[calendar[200], "C05USDT"]
    assert not market.delist["C01USDT"].any()  # listed to the end of the data: not delisted
    assert market.close.loc[calendar[201] :, "C05USDT"].isna().all()
    assert np.isfinite(market.funding["C01USDT"]).all() and np.isfinite(market.vwap["C01USDT"]).all()


def test_a_backtest_reads_its_market_from_qlib(crypto_session):
    calendar = crypto_session.calendar
    scores = D.features(D.instruments("top20"), ["-1 * ($close / Ref($close, 5) - 1)"], calendar[150], calendar[230])
    scores = scores.iloc[:, 0].swaplevel().sort_index()
    result = backtest_long_short(scores, portfolio=LongShortPortfolio(quantile=0.2, min_names=5))
    # the first score trades a day later; the last trade is held one day
    assert result.report.index[0] == calendar[151] and result.report.index[-1] == calendar[232]
    assert (result.report["gross_exposure"] > 0.9).all()
    delisted = result.trades[result.trades["kind"] == "delist"]
    if "C05USDT" in result.weights.columns and result.weights.loc[calendar[199], "C05USDT"] != 0:
        assert delisted["instrument"].tolist() == ["C05USDT"]
    assert (result.weights.loc[calendar[201] :, result.weights.columns == "C05USDT"] == 0).all().all()


@pytest.fixture
def trained(crypto_session):
    from qlib.model.trainer import task_train

    recorder = task_train(task(crypto_session.calendar, RECORD), experiment_name="crypto")
    return crypto_session, recorder


def test_the_workflow_backtests_the_predictions(trained):
    session, recorder = trained
    report = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    config = recorder.load_object("portfolio_analysis/perp_config.pkl")
    pred = recorder.load_object("pred.pkl")
    assert {"account", "return", "cost", "bench", "turnover", "funding"} <= set(report.columns)
    assert report.index[0] == session.calendar[161]  # the first prediction trades a day later
    assert config["portfolio"]["quantile"] == 0.2 and config["delay"] == 1
    metrics = recorder.list_metrics()
    assert "1day.excess_return_with_cost.annualized_return" in metrics and "1day.perp.turnover" in metrics
    expected = backtest_long_short(
        pred,
        portfolio=LongShortPortfolio(quantile=0.2, min_names=5, buffer=1),
        costs=__import__("qlib.contrib.crypto", fromlist=["CostModel"]).CostModel(
            fee=0.0005, spread_tiers=((0, 0.0002),), impact=0.0
        ),
        benchmark="C00USDT",
    )
    pd.testing.assert_frame_equal(report, expected.report)
    assert analysis.loc[("excess_return_with_cost", "annualized_return"), "risk"] == pytest.approx(
        ((report["return"] - report["bench"] - report["cost"]).mean()) * 365
    )


def test_live_targets_are_the_backtests(trained):
    session, recorder = trained
    last = session.calendar[-1]
    pred = recorder.load_object("pred.pkl")["score"].xs(last, level="datetime")
    targets = live.generate_targets(recorder, trusted=True, now=last + pd.Timedelta(1, unit="D"))
    assert targets.ok, targets.problems
    assert targets.decision_date == last
    # the live scores are the workflow's own predictions of the day
    pd.testing.assert_series_equal(
        targets.scores.sort_index(), pred.sort_index(), check_names=False, check_dtype=False, rtol=1e-6
    )
    book = LongShortPortfolio(quantile=0.2, min_names=5, buffer=1)
    pd.testing.assert_series_equal(targets.weights, book.target(pred))
    held = pd.Series({targets.weights.index[0]: 0.1})
    with_held = live.generate_targets(recorder, trusted=True, now=last + pd.Timedelta(1, unit="D"), current=held)
    pd.testing.assert_series_equal(with_held.weights, book.target(pred, current=held))
    document = targets.document(equity=50_000)
    assert document["trade"] == {"deal_price": "close", "at": f"{last + pd.Timedelta(2, unit='D'):%Y-%m-%d}T00:00:00Z"}
    assert document["notional"] == pytest.approx({k: v * 50_000 for k, v in document["weights"].items()})
    assert document["checks"]["universe_members"] == 20 - 1  # C05USDT left with its delisting


def test_live_refuses_stale_or_untrusted_inputs(trained):
    session, recorder = trained
    last = session.calendar[-1]
    stale = live.generate_targets(recorder, trusted=True, now=last + pd.Timedelta(5, unit="D"))
    assert not stale.ok and "days old" in stale.problems[0] and stale.weights.empty
    with pytest.raises(ValueError, match="trusted"):
        live.generate_targets(recorder, trusted=False, now=last)
    # a dump where most members have no bar on the decision day
    from qlib.data.cache import H

    for name in session.names[:12]:
        path = session.provider / "features" / name.lower() / "close.day.bin"
        values = np.fromfile(path, dtype="<f4")
        values[-1] = np.nan
        values.tofile(path)
    H.clear()
    partial = live.generate_targets(recorder, trusted=True, now=last + pd.Timedelta(1, unit="D"))
    assert not partial.ok
    assert any("have a close" in problem for problem in partial.problems)


def test_live_refuses_scores_that_tie_at_the_edge_of_a_side(trained, monkeypatch):
    from qlib.contrib.model.linear import LinearModel

    session, recorder = trained
    last = session.calendar[-1]
    original = LinearModel.predict

    def constant(self, dataset, segment="test"):
        return original(self, dataset, segment) * 0 + 1.0  # a model that cannot tell names apart

    monkeypatch.setattr(LinearModel, "predict", constant)
    now = last + pd.Timedelta(1, unit="D")
    refused = live.generate_targets(recorder, trusted=True, now=now)
    assert not refused.ok and "equal at the edge" in refused.problems[0] and refused.weights.empty
    assert refused.checks["boundary_tied"] is True
    allowed = live.generate_targets(recorder, trusted=True, now=now, allow_ties=True)
    assert allowed.ok and allowed.checks["boundary_tied"] is True and len(allowed.weights) > 0


def test_the_live_command(trained, tmp_path):
    session, recorder = trained
    out = tmp_path / "targets"
    current = tmp_path / "current.json"
    current.write_text(json.dumps({"C01USDT": 0.05}))
    arguments = [
        "--provider-uri",
        str(session.provider),
        "--mlruns",
        str(session.mlruns),
        "--experiment",
        "crypto",
        "--out",
        str(out),
        "--current",
        str(current),
        "--equity",
        "100000",
        "--max-age-days",
        "-1",
        "--trusted",
    ]
    assert live.main(arguments) == 0
    written = json.loads((out / f"targets-{session.calendar[-1]:%Y-%m-%d}.json").read_text())
    assert written == {**json.loads((out / "latest.json").read_text()), "generated_at": written["generated_at"]}
    assert written["recorder"]["id"] == recorder.info["id"]
    assert abs(sum(written["weights"].values())) < 1e-9 and written["equity"] == 100000
    assert not list(out.glob(".*"))  # no temporary file left
    stale = [a if a != "-1" else "0" for a in arguments]
    (out / "latest.json").unlink()
    assert live.main(stale) == live.REFUSED
    assert not (out / "latest.json").exists()
