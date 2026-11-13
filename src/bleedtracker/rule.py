"""The frozen rule: a hand-written score and a reason code for every page.

action_score = severity × log1p(impressions) × (1 + underclick) × (1 + 0.5 × reachable)

- severity: how far the second half of the month fell below 0.8 of the first half,
  from 0 (not sliding) to 1 (no second-half impressions).
- underclick: 1 when the page's CTR is below the median CTR of its position band.
- reachable: 1 when the average position is between 1 and 20.

The score has no fitted weights. Its one fitted quantity is the median CTR per
position band, learned on training clients only.
"""

import numpy as np
import pandas as pd

POSITION_EDGES = [0, 3, 10, 20, np.inf]
POSITION_BANDS = ["1-3", "4-10", "11-20", "21+"]
SLIP_GATE = 0.8
REACHABLE_POSITION = 20
UNDERCLICK_WEIGHT = 1.0
REACHABLE_WEIGHT = 0.5


def position_band(pos: pd.Series) -> pd.Series:
    """Band of each average position. Edges are (0, 3], (3, 10], (10, 20] and above 20."""
    return pd.cut(pos, POSITION_EDGES, labels=POSITION_BANDS)


def band_median_ctr(train: pd.DataFrame) -> pd.Series:
    """Median CTR of each position band, learned on the rows given (training clients only)."""
    return train.groupby(position_band(train.pos_mar), observed=True)["ctr_mar"].median()


def signals(frame: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """The rule's three signals for every page.

    A page whose band has no learned median is never marked as underclicking.
    """
    band_median = position_band(frame.pos_mar).astype(object).map(medians.to_dict())
    return pd.DataFrame(
        {
            "severity": (1 - frame.momentum_in_month / SLIP_GATE).clip(0, 1),
            "underclick": (frame.ctr_mar < band_median.astype(float)).astype(int),
            "reachable": frame.pos_mar.between(1, REACHABLE_POSITION).astype(int),
        },
        index=frame.index,
    )


def action_score(frame: pd.DataFrame, medians: pd.Series) -> pd.Series:
    """The rule's score. Zero for every page that isn't sliding."""
    s = signals(frame, medians)
    return (
        s.severity
        * np.log1p(frame.imp)
        * (1 + UNDERCLICK_WEIGHT * s.underclick)
        * (1 + REACHABLE_WEIGHT * s.reachable)
    )


def reason_code(frame: pd.DataFrame, medians: pd.Series) -> pd.Series:
    """The reason an editor reads next to each page."""
    s = signals(frame, medians)
    codes = np.select(
        [
            s.severity == 0,
            (s.underclick == 1) & (s.reachable == 1),
            s.reachable == 1,
        ],
        ["holding_steady", "slipping_and_underclicking", "slipping_reachable_page"],
        default="slipping_deep_page",
    )
    return pd.Series(codes, index=frame.index)
