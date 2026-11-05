"""Daily target weights for live trading, from a trained workflow run.

    python -m qlib.contrib.crypto.live \\
        --provider-uri ~/.qlib/qlib_data/crypto_um --mlruns ./mlruns --experiment crypto \\
        --out ./targets --trusted

Each day, after ``jarvis-data sync`` and ``jarvis-data dump qlib``:

1. Load the run's model (``params.pkl``), its dataset settings (``dataset``) and the backtest's
   settings (``portfolio_analysis/perp_config.pkl``, written by ``PerpPortAnaRecord``). The model
   and the dataset are pickles: ``--trusted`` says they are yours (qlib refuses otherwise).
2. Score the decision day (by default the last day of the data) with the run's dataset, on the
   run's universe, from ``lookback_days`` of history.
3. Check the inputs, and refuse to write targets when one fails:

   - the data is no older than ``max_age_days`` (a sync that did not run);
   - at least ``min_coverage`` of the universe's members have a close on the decision day and
     a finite score (a partial dump);
   - no raw feature is NaN for more than ``max_feature_nan`` of the members (a field the data
     does not have yet, such as ``$funding`` before the month's archive).

4. Build the targets with the backtest's ``LongShortPortfolio``, from the weights held now
   (``--current``, for its ``buffer``), and write them atomically to ``targets-<day>.json`` and
   ``latest.json``.

A refusal writes nothing and exits with 2, so the trading side keeps its last targets.

The targets are fractions of the account's equity (``--equity`` adds USDT notionals). They are
the book the backtest holds after its trade: with ``deal_price="close"`` and ``delay = 1`` the
decision day ``d`` trades at the close of ``d + 1``, that is 00:00 UTC of ``d + 2`` (``trade_at``).
Trading at another time is a different strategy from the one backtested.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from ...data import D
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger
from ...utils.exceptions import LoadObjectError
from .portfolio import LongShortPortfolio
from .record import build_portfolio

logger = get_module_logger("crypto.live")

REFUSED = 2
CONFIG_ARTIFACT = "portfolio_analysis/perp_config.pkl"


@dataclass
class Targets:
    decision_date: pd.Timestamp
    weights: pd.Series
    scores: pd.Series
    portfolio: LongShortPortfolio
    delay: int
    deal_price: str
    checks: Dict[str, object] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)
    recorder: Dict[str, object] = field(default_factory=dict)
    calendar_end: Optional[pd.Timestamp] = None

    @property
    def ok(self) -> bool:
        return not self.problems

    def execution(self) -> Dict[str, str]:
        day = self.decision_date + pd.Timedelta(self.delay, unit="D")
        if self.deal_price == "vwap":
            return {"deal_price": "vwap", "from": _iso(day), "until": _iso(day + pd.Timedelta(1, unit="D"))}
        return {"deal_price": "close", "at": _iso(day + pd.Timedelta(1, unit="D"))}

    def document(self, equity: Optional[float] = None) -> Dict[str, object]:
        weights = {name: float(value) for name, value in self.weights.items()}
        doc: Dict[str, object] = {
            "decision_date": f"{self.decision_date.date()}",
            "trade": self.execution(),
            "generated_at": _iso(pd.Timestamp.now(tz="UTC")),
            "calendar_end": f"{self.calendar_end.date()}" if self.calendar_end is not None else None,
            "recorder": self.recorder,
            "portfolio": self.portfolio.to_dict(),
            "weights": weights,
            "gross": float(np.abs(self.weights).sum()),
            "net": float(self.weights.sum()),
            "n_long": int((self.weights > 0).sum()),
            "n_short": int((self.weights < 0).sum()),
            "checks": self.checks,
        }
        if equity is not None:
            doc["equity"] = float(equity)
            doc["notional"] = {name: float(value * equity) for name, value in weights.items()}
        return doc


def _iso(moment: pd.Timestamp) -> str:
    moment = pd.Timestamp(moment)
    if moment.tzinfo is None:
        moment = moment.tz_localize("UTC")
    return moment.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def latest_recorder(experiment_name: str) -> Any:
    """The latest finished recorder of an experiment."""
    from ...workflow import R  # pylint: disable=C0415

    experiment = R.get_exp(experiment_name=experiment_name, create=False)
    recorders = experiment.list_recorders(rtype="list", status="FINISHED")
    if not recorders:
        raise ValueError(f"The experiment {experiment_name!r} has no finished recorder")
    return recorders[0]


def generate_targets(
    recorder: Any,
    *,
    date: Any = None,
    current: Optional[pd.Series] = None,
    portfolio: Optional[LongShortPortfolio] = None,
    lookback_days: int = 400,
    min_coverage: float = 0.9,
    max_feature_nan: float = 0.5,
    max_age_days: Optional[int] = 2,
    trusted: bool = False,
    now: Optional[pd.Timestamp] = None,
) -> Targets:
    """The targets of the decision day ``date`` (the data's last day when None); see the module's
    docstring. ``recorder`` is a finished qlib workflow recorder with ``PerpPortAnaRecord``'s
    output (or pass ``portfolio``). Check ``Targets.ok`` before trading them."""
    if not 0 < min_coverage <= 1:
        raise ValueError(f"min_coverage must be in (0, 1], not {min_coverage!r}")
    if not 0 <= max_feature_nan <= 1:
        raise ValueError(f"max_feature_nan must be in [0, 1], not {max_feature_nan!r}")
    try:
        config = recorder.load_object(CONFIG_ARTIFACT)
    except LoadObjectError:
        if portfolio is None:
            raise ValueError(
                f"The recorder has no {CONFIG_ARTIFACT}: run PerpPortAnaRecord in the workflow, or pass a portfolio"
            ) from None
        config = {}
    portfolio = portfolio or build_portfolio(config.get("portfolio"))
    delay = int(config.get("delay", 1))
    deal_price = str(config.get("deal_price", "close"))
    try:
        model = recorder.load_object("params.pkl", trusted=trusted)
        dataset = recorder.load_object("dataset", trusted=trusted)
    except LoadObjectError as error:
        if not trusted:
            raise ValueError(
                "The model and dataset are pickles that qlib loads only when told they are trusted: "
                "pass trusted=True (--trusted) for a recorder you trained"
            ) from error
        raise

    calendar = pd.DatetimeIndex(D.calendar(freq="day"))
    calendar_end = calendar[-1]
    day = pd.Timestamp(date) if date is not None else calendar_end
    if day not in calendar:
        raise ValueError(f"{day.date()} is not in the data's calendar (it ends {calendar_end.date()})")
    start = day - pd.Timedelta(lookback_days, unit="D")
    dataset.config(handler_kwargs={"start_time": start, "end_time": day}, segments={"test": (day, day)})
    dataset.setup_data(handler_kwargs={"init_type": DataHandlerLP.IT_LS})
    predicted = model.predict(dataset, segment="test")
    if isinstance(predicted, pd.DataFrame):
        predicted = predicted.iloc[:, 0]
    scores = predicted.xs(day, level="datetime") if "datetime" in predicted.index.names else predicted
    scores = pd.Series(scores, dtype="float64")
    scores.index = scores.index.astype(str)

    problems: List[str] = []
    checks: Dict[str, object] = {"decision_date": f"{day.date()}", "calendar_end": f"{calendar_end.date()}"}
    clock = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    if clock.tzinfo is None:
        clock = clock.tz_localize("UTC")
    age = (clock.tz_convert("UTC").tz_localize(None).normalize() - day).days
    checks["age_days"] = int(age)
    if max_age_days is not None and age > max_age_days:
        problems.append(f"the decision day {day.date()} is {age} days old (max {max_age_days}): is the data synced?")

    universe = _universe(dataset)
    members = D.list_instruments(universe, start_time=day, end_time=day, as_list=True)
    checks["universe_members"] = len(members)
    if not members:
        problems.append(f"the universe has no member on {day.date()}")
    else:
        closes = D.features(members, ["$close"], start_time=day, end_time=day)
        priced = int(np.isfinite(closes.iloc[:, 0].to_numpy()).sum()) if not closes.empty else 0
        scored = int(np.isfinite(scores.reindex(members).to_numpy()).sum())
        checks.update({"priced": priced, "scored": scored, "min_coverage": min_coverage})
        if priced < min_coverage * len(members):
            problems.append(f"only {priced} of {len(members)} members have a close on {day.date()}")
        if scored < min_coverage * len(members):
            problems.append(f"only {scored} of {len(members)} members have a score on {day.date()}")
    worst = _worst_feature(dataset, day)
    if worst is not None:
        checks["worst_feature"] = {"name": worst[0], "nan_share": round(worst[1], 4), "max": max_feature_nan}
        if worst[1] > max_feature_nan:
            problems.append(f"the feature {worst[0]} is NaN for {worst[1]:.0%} of the members on {day.date()}")

    weights = portfolio.target(scores, current=current) if not problems else pd.Series(dtype="float64")
    info = recorder.info if hasattr(recorder, "info") else {}
    return Targets(
        decision_date=day,
        weights=weights,
        scores=scores.sort_index(),
        portfolio=portfolio,
        delay=delay,
        deal_price=deal_price,
        checks=checks,
        problems=problems,
        recorder={"id": info.get("id"), "experiment_id": info.get("experiment_id"), "name": info.get("name")},
        calendar_end=calendar_end,
    )


def _universe(dataset: Any) -> Any:
    instruments = dataset.handler.instruments
    if isinstance(instruments, str):
        return D.instruments(instruments)
    return instruments


def _worst_feature(dataset: Any, day: pd.Timestamp) -> Optional[Tuple[str, float]]:
    """The raw feature with the largest share of NaN on ``day``, and that share."""
    try:
        raw = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_R)
    except (KeyError, AttributeError, TypeError) as error:
        logger.warning(f"The raw features cannot be checked: {error}")
        return None
    if raw is None or raw.empty:
        return ("<all>", 1.0)
    if "datetime" in raw.index.names:
        raw = raw.xs(day, level="datetime")
    share = raw.isna().mean()
    if share.empty:
        return None
    name = share.idxmax()
    return (str(name[-1] if isinstance(name, tuple) else name), float(share.max()))


def read_weights(path: Path) -> pd.Series:
    """Weights from a JSON file: ``{"BTCUSDT": 0.05, ...}``, or a targets file of this module."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, Mapping) and "weights" in data:
        data = data["weights"]
    if not isinstance(data, Mapping):
        raise ValueError(f"{path}: expected an object of instrument weights")
    weights = pd.Series({str(k): float(v) for k, v in data.items()}, dtype="float64")
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError(f"{path}: weights must be finite numbers")
    return weights


def write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write ``document`` to ``path`` so that a reader sees the old file or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w") as stream:
            json.dump(document, stream, indent=2, sort_keys=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m qlib.contrib.crypto.live", description=__doc__.split("\n", maxsplit=1)[0]
    )
    parser.add_argument("--provider-uri", required=True, help="the qlib directory jarvis-data dumped")
    parser.add_argument("--mlruns", default="mlruns", help="the workflow's mlruns directory (default ./mlruns)")
    parser.add_argument("--experiment", required=True, help="the workflow's experiment name")
    parser.add_argument("--recorder", help="a recorder id (default: the experiment's latest finished one)")
    parser.add_argument("--out", required=True, type=Path, help="directory for targets-<day>.json and latest.json")
    parser.add_argument("--date", help="the decision day, YYYY-MM-DD (default: the data's last day)")
    parser.add_argument("--current", type=Path, help="JSON of the weights held now (for the portfolio's buffer)")
    parser.add_argument("--equity", type=float, help="the account's equity in USDT, to add notionals")
    parser.add_argument("--lookback-days", type=int, default=400, help="days of history for the features")
    parser.add_argument("--min-coverage", type=float, default=0.9, help="share of members that must be scored")
    parser.add_argument("--max-feature-nan", type=float, default=0.5, help="largest NaN share of a raw feature")
    parser.add_argument("--max-age-days", type=int, default=2, help="oldest decision day accepted; -1 for any")
    parser.add_argument("--trusted", action="store_true", help="load the recorder's model and dataset pickles")
    args = parser.parse_args(argv)

    import qlib  # pylint: disable=C0415
    from qlib.constant import REG_CRYPTO  # pylint: disable=C0415
    from qlib.workflow import R  # pylint: disable=C0415

    if args.equity is not None and not (math.isfinite(args.equity) and args.equity > 0):
        parser.error("--equity must be a positive number")
    qlib.init(
        provider_uri=str(Path(args.provider_uri).expanduser()),
        region=REG_CRYPTO,
        expression_cache=None,
        dataset_cache=None,
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": "file:" + str(Path(args.mlruns).expanduser().resolve()),
                "default_exp_name": "Experiment",
            },
        },
    )
    if args.recorder:
        recorder = R.get_recorder(recorder_id=args.recorder, experiment_name=args.experiment)
    else:
        recorder = latest_recorder(args.experiment)
    current = read_weights(args.current) if args.current else None
    targets = generate_targets(
        recorder,
        date=args.date,
        current=current,
        lookback_days=args.lookback_days,
        min_coverage=args.min_coverage,
        max_feature_nan=args.max_feature_nan,
        max_age_days=None if args.max_age_days < 0 else args.max_age_days,
        trusted=args.trusted,
    )
    if not targets.ok:
        for problem in targets.problems:
            logger.error(f"refused: {problem}")
        logger.error("No targets written: keep trading the last ones")
        return REFUSED
    document = targets.document(equity=args.equity)
    name = f"targets-{targets.decision_date.date()}.json"
    write_json(args.out / name, document)
    write_json(args.out / "latest.json", document)
    logger.info(
        f"{name}: {document['n_long']} long, {document['n_short']} short, gross {document['gross']:.3f}, "
        f"net {document['net']:.3f}; trade {document['trade']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
