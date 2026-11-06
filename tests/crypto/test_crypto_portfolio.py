"""Targets of the long-short book from one day's scores."""

import numpy as np
import pandas as pd
import pytest

from qlib.contrib.crypto import LongShortPortfolio
from qlib.contrib.crypto.portfolio import rank

SCORES = pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 2.0, "E": 1.0, "F": np.nan, "G": np.inf})


def test_the_best_go_long_and_the_worst_short():
    book = LongShortPortfolio(n_long=2, n_short=2, min_names=4)
    assert book.target(SCORES).to_dict() == pytest.approx({"A": 0.25, "B": 0.25, "D": -0.25, "E": -0.25})


def test_gross_and_net_set_the_sides():
    book = LongShortPortfolio(n_long=2, n_short=1, gross=2.0, net=0.5, min_names=4)
    target = book.target(SCORES)
    assert target[target > 0].sum() == pytest.approx(1.25)
    assert target[target < 0].sum() == pytest.approx(-0.75)
    long_only = LongShortPortfolio(n_long=2, n_short=0, gross=1.0, net=1.0, min_names=1)
    assert long_only.target(SCORES).to_dict() == pytest.approx({"A": 0.5, "B": 0.5})


def test_quantile_sizes_follow_the_scored_names():
    book = LongShortPortfolio(quantile=0.2, min_names=3)
    assert book.sizes(5) == (1, 1) and book.sizes(23) == (4, 4) and book.sizes(2) == (0, 0)
    assert book.target(SCORES).to_dict() == pytest.approx({"A": 0.5, "E": -0.5})
    # fewer names than the sides need: each side takes at most half
    assert LongShortPortfolio(n_long=4, n_short=4, min_names=1).sizes(5) == (2, 2)


def test_too_few_names_is_flat():
    assert LongShortPortfolio(n_long=1, n_short=1).target(SCORES).empty  # 5 scored, min_names 10


def test_rank_weighting_and_the_cap():
    book = LongShortPortfolio(n_long=2, n_short=2, weighting="rank", min_names=4)
    assert book.target(SCORES).to_dict() == pytest.approx({"A": 1 / 3, "B": 1 / 6, "D": -1 / 6, "E": -1 / 3})
    capped = LongShortPortfolio(n_long=1, n_short=1, max_weight=0.2, min_names=2)
    assert capped.target(SCORES).to_dict() == pytest.approx({"A": 0.2, "E": -0.2})


def test_equal_scores_rank_by_name_whatever_the_order():
    tied = pd.Series({"Z": 1.0, "M": 1.0, "A": 1.0, "Q": 0.0})
    assert rank(tied) == ["A", "M", "Z", "Q"]
    assert rank(tied.iloc[::-1]) == ["A", "M", "Z", "Q"]
    with pytest.raises(ValueError, match="repeat"):
        rank(pd.Series([1.0, 2.0], index=["A", "A"]))


def test_ties_at_the_edge_of_a_side_are_detected():
    book = LongShortPortfolio(n_long=2, n_short=2, min_names=4)
    assert not book.boundary_tied(SCORES)
    assert book.boundary_tied(pd.Series({"A": 5.0, "B": 4.0, "C": 4.0, "D": 2.0, "E": 1.0}))  # 2nd and 3rd best
    assert book.boundary_tied(pd.Series({"A": 5.0, "B": 4.0, "C": 3.0, "D": 3.0, "E": 1.0}))  # 2nd and 3rd worst
    # equal scores inside a side are harmless: both are held either way
    assert not book.boundary_tied(pd.Series({"A": 5.0, "B": 5.0, "C": 3.0, "D": 1.0, "E": 1.0}))
    assert not book.boundary_tied(pd.Series({"A": 1.0, "B": 1.0}))  # too few names: nothing is held


def test_the_buffer_keeps_held_names():
    book = LongShortPortfolio(n_long=2, n_short=2, buffer=1, min_names=4)
    held = pd.Series({"C": 0.25, "D": -0.25})
    # C ranks 3rd, within 2 + 1: it stays long; D is 2nd from the bottom: it stays short
    assert book.target(SCORES, current=held).to_dict() == pytest.approx({"A": 0.25, "C": 0.25, "D": -0.25, "E": -0.25})
    # without the buffer the scores alone decide
    plain = LongShortPortfolio(n_long=2, n_short=2, min_names=4)
    assert plain.target(SCORES, current=held).index.tolist() == ["A", "B", "D", "E"]
    # past the buffer a held name goes: D ranks 4th, past 2 + 1, so it is not kept long
    target = book.target(SCORES, current=pd.Series({"D": 0.25}))
    assert target[target > 0].index.tolist() == ["A", "B"]


def test_the_buffer_lowers_turnover():
    rng = np.random.default_rng(3)
    names = [f"N{i:02d}" for i in range(60)]
    base = rng.normal(size=60)
    plain = LongShortPortfolio(quantile=0.1, min_names=10)
    buffered = LongShortPortfolio(quantile=0.1, min_names=10, buffer=6)

    def turnover(book):
        held, total = pd.Series(dtype="float64"), 0.0
        for _ in range(100):
            scores = pd.Series(base + rng.normal(scale=0.5, size=60), index=names)
            target = book.target(scores, current=held)
            total += target.sub(held, fill_value=0).abs().sum()
            held = target
        return total

    assert turnover(buffered) < 0.8 * turnover(plain)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(), "Give n_long and n_short, or quantile"),
        (dict(n_long=1, n_short=1, quantile=0.1), "not both"),
        (dict(n_long=1), "Give both"),
        (dict(n_long=-1, n_short=1), "n_long must be"),
        (dict(n_long=0, n_short=0), "cannot both be 0"),
        (dict(quantile=0.6), "quantile must be"),
        (dict(quantile=0.1, gross=0), "gross must be"),
        (dict(quantile=0.1, net=2.0), "net must be"),
        (dict(n_long=0, n_short=2), "set net = -gross"),
        (dict(n_long=2, n_short=0), "set net = gross"),
        (dict(quantile=0.1, weighting="score"), "weighting"),
        (dict(quantile=0.1, buffer=-1), "buffer"),
        (dict(quantile=0.1, max_weight=0), "max_weight"),
        (dict(quantile=0.1, min_names=0), "min_names"),
    ],
)
def test_bad_settings_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LongShortPortfolio(**kwargs)
