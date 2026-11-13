"""Constants of the data contract, in one place."""

import calendar
import math
from datetime import date

RELEASE = "hf://datasets/FlyRank/internship-warehouse"

# A client is eligible when its search history starts on or before this day.
# Every feature month is then a full month of history.
MIN_HISTORY = date(2026, 1, 1)

# A page needs this many impressions in the feature month. Below it, a 20% drop is noise.
MIN_IMPRESSIONS = 100

# A page is declining when next month's impressions are more than 20% below this month's.
DECLINE_DROP = 0.20

FEATURES = ["log_impressions", "active_days", "position", "ctr", "momentum"]


def days_in_month(month: str) -> int:
    """Number of days in a month written as "YYYY-MM"."""
    year, mon = map(int, month.split("-"))
    return calendar.monthrange(year, mon)[1]


def second_half_start(month: str) -> date:
    """First day of the month's second half.

    The first half gets the extra day in a month with an odd number of days.
    A 31-day month splits 16/15. A 30-day month splits 15/15.
    """
    year, mon = map(int, month.split("-"))
    return date(year, mon, math.ceil(days_in_month(month) / 2) + 1)
