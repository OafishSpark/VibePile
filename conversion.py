"""Приведение столбцов DataFrame к float64 (для расчётов) и к строкам (для таблицы)."""
from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

SECONDS_PER_DAY = 86400.0


def is_datetime(s: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(s)


def to_float(s: pd.Series) -> np.ndarray:
    """Числа или даты -> float64. Даты переводятся в дни matplotlib (mdates.date2num)."""
    if is_datetime(s):
        return np.asarray(mdates.date2num(s.to_numpy()), dtype=np.float64)
    return s.to_numpy(dtype=np.float64)


def duration_to_float(s: pd.Series, time_is_datetime: bool, unit: str = "s") -> np.ndarray:
    """Длительность -> float64 в единицах оси времени.

    Время — даты: результат в днях; timedelta переводится точно, число берётся в единицах unit.
    Время — числа: timedelta переводится в секунды, число остаётся как есть.
    """
    if pd.api.types.is_timedelta64_dtype(s):
        sec = s.dt.total_seconds().to_numpy(dtype=np.float64)
        return sec / SECONDS_PER_DAY if time_is_datetime else sec
    values = s.to_numpy(dtype=np.float64)
    if time_is_datetime:
        return values * (pd.Timedelta(1, unit=unit) / pd.Timedelta(days=1))
    return values


def to_text(s: pd.Series, decimals: int = 3) -> list[str]:
    """Столбец -> строки для таблицы.

    Дробные числа округляются до decimals знаков, целые выводятся как есть,
    даты — с миллисекундами, длительности — в секундах.
    """
    if is_datetime(s):
        return s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3].fillna("").tolist()
    if pd.api.types.is_timedelta64_dtype(s):
        return [f"{v:.{decimals}f}" for v in s.dt.total_seconds().to_numpy(dtype=np.float64)]
    if pd.api.types.is_integer_dtype(s):
        return s.astype(str).tolist()
    if pd.api.types.is_numeric_dtype(s):
        return [f"{v:.{decimals}f}" for v in s.to_numpy(dtype=np.float64)]
    return s.astype(str).tolist()
