"""built-in model families; registration never imports optional libraries"""

from smolsmort.forecast.families.base import (
    Family,
    Fitted,
    Param,
    available_families,
    get_family,
    register,
)
from smolsmort.forecast.families.xgboost import XgboostFamily

register(XgboostFamily())

__all__ = ["Family", "Fitted", "Param", "available_families", "get_family", "register"]
