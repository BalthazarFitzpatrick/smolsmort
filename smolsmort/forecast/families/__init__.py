"""built-in model families; registration never imports optional libraries"""

from smolsmort.forecast.families.base import (
    Family,
    Fitted,
    Param,
    available_families,
    get_family,
    register,
)
from smolsmort.forecast.families.lightgbm import LightgbmFamily
from smolsmort.forecast.families.ridge import RidgeFamily
from smolsmort.forecast.families.stats import StatsFamily
from smolsmort.forecast.families.xgboost import XgboostFamily

register(XgboostFamily())
register(LightgbmFamily())
register(RidgeFamily())
for name in ("ets", "theta", "arima", "snaive"):
    register(StatsFamily(name))

__all__ = ["Family", "Fitted", "Param", "available_families", "get_family", "register"]
