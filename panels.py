"""Панели — описание того, что и как рисуется на одном графике.

Чтобы добавить свой тип графика, напишите класс с тем же интерфейсом, что у Panel;
менять EventPlot и EventViewer не нужно.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Union

import numpy as np
import pandas as pd
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection

from .conversion import duration_to_float


class Panel(Protocol):
    """Интерфейс панели.

    geometry() возвращает по отрезку на событие: float64-массив (n, 2, 2)
    [событие, начало/конец, x/y] в координатах данных; точка — отрезок нулевой длины.
    По этой геометрии работают клик и выделение окном.

    draw() рисует события и возвращает артистов; set_color(colors) у каждого артиста
    с colors формы (n, 4) должен красить события в порядке строк.
    """

    @property
    def label(self) -> str: ...

    def geometry(self, df: pd.DataFrame, t: np.ndarray, time_is_datetime: bool) -> np.ndarray: ...

    def draw(self, ax: Axes, geom: np.ndarray, color) -> list[Artist]: ...


@dataclass(frozen=True)
class Points:
    """Точки: значение колонки y в момент события."""

    y: str
    name: str | None = None  # подпись оси Y; по умолчанию — имя колонки
    size: float = 14         # размер маркера, pt^2

    @property
    def label(self) -> str:
        return self.name or self.y

    def geometry(self, df: pd.DataFrame, t: np.ndarray, time_is_datetime: bool) -> np.ndarray:
        p = np.column_stack((t, df[self.y].to_numpy(dtype=np.float64)))
        return np.stack((p, p), axis=1)

    def draw(self, ax: Axes, geom: np.ndarray, color) -> list[Artist]:
        return [ax.scatter(geom[:, 0, 0], geom[:, 0, 1], s=self.size, linewidths=0, color=color)]


@dataclass(frozen=True)
class Segments:
    """Отрезки от (t, start) до (t + duration, end) и точка в центре каждого.

    duration — timedelta или число. Если время — даты, число берётся в единицах unit
    ("s", "ms", "min", ...). Если время числовое, длительность должна быть в тех же
    единицах (timedelta переводится в секунды).
    """

    start: str
    end: str
    duration: str
    unit: str = "s"
    name: str | None = None  # подпись оси Y; по умолчанию — имя колонки start
    linewidth: float = 1.5
    size: float = 14         # размер точки в центре, pt^2
    center: bool = True      # рисовать ли точку в центре

    @property
    def label(self) -> str:
        return self.name or self.start

    def geometry(self, df: pd.DataFrame, t: np.ndarray, time_is_datetime: bool) -> np.ndarray:
        dur = duration_to_float(df[self.duration], time_is_datetime, self.unit)
        a = np.column_stack((t, df[self.start].to_numpy(dtype=np.float64)))
        b = np.column_stack((t + dur, df[self.end].to_numpy(dtype=np.float64)))
        return np.stack((a, b), axis=1)

    def draw(self, ax: Axes, geom: np.ndarray, color) -> list[Artist]:
        # transform задан явно: отрезки всегда в координатах данных и следуют за зумом
        lines = LineCollection(geom, colors=[color], linewidths=self.linewidth, transform=ax.transData)
        artists = [ax.add_collection(lines, autolim=False)]
        if self.center:
            mid = geom.mean(axis=1)
            artists.append(ax.scatter(mid[:, 0], mid[:, 1], s=self.size, linewidths=0,
                                      color=color, zorder=3))
        return artists


PanelSpec = Union[str, Panel]  # строка — сокращение для Points(строка)


def as_panel(spec: PanelSpec) -> Panel:
    return Points(spec) if isinstance(spec, str) else spec
