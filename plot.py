"""Панели событий с общей осью времени, toolbar matplotlib и выделением мышью."""
from __future__ import annotations

import functools
import tkinter as tk
from tkinter import ttk
from typing import Callable, Sequence

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, ScalarFormatter
from matplotlib.widgets import RectangleSelector

from .conversion import is_datetime, to_float
from .geometry import seg_dist2, seg_in_rect
from .panels import PanelSpec, as_panel


def _padded(a: np.ndarray, frac: float = 0.05) -> tuple[float, float]:
    """Границы оси с небольшим запасом."""
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    pad = (hi - lo) * frac or 0.5
    return lo - pad, hi + pad


class EventPlot(ttk.Frame):
    """Графики панелей друг под другом с общей осью времени и стандартным toolbar.

    События адресуются позицией строки в DataFrame, переданном в set_data().
    Выделение здесь не хранится, о действиях мыши сообщают колбэки:
      on_toggle(row)  — клик рядом с событием;
      on_select(mask) — окно задело события (mask — bool-массив длины n).
    Показать выделение — set_selected(mask).
    """

    click_tol = 5  # px: если мышь сдвинулась меньше — это клик, а не окно
    pick_tol = 8   # px: радиус поиска события при клике

    def __init__(self, master, panels: Sequence[PanelSpec], time_label: str = "",
                 on_toggle: Callable[[int], None] | None = None,
                 on_select: Callable[[np.ndarray], None] | None = None,
                 color="tab:blue", sel_color="tab:red", **kw):
        super().__init__(master, **kw)
        self.panels = [as_panel(p) for p in panels]
        self.on_toggle = on_toggle
        self.on_select = on_select
        self._palette = np.array([to_rgba(color), to_rgba(sel_color)])  # (2, 4)
        self._geom: list[np.ndarray] = []  # для каждой панели (n, 2, 2) float64
        self._artists = []                 # все артисты всех панелей
        self._press = None                 # координаты нажатия мыши, px

        self.fig = Figure(figsize=(8, 6), layout="constrained")
        self.axes = list(self.fig.subplots(len(self.panels), 1, sharex=True, squeeze=False)[:, 0])
        for ax, panel in zip(self.axes, self.panels):
            ax.set_ylabel(panel.label)
            ax.grid(alpha=0.3)
        self.axes[-1].set_xlabel(time_label)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self, pack_toolbar=False)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # во время зума/панорамирования селекторы отключаются автоматически (widgetlock)
        self._selectors = [
            RectangleSelector(ax, functools.partial(self._on_rect, i), useblit=True,
                              button=[MouseButton.LEFT],
                              props=dict(facecolor=sel_color, alpha=0.2))
            for i, ax in enumerate(self.axes)
        ]
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    @property
    def n_events(self) -> int:
        return len(self._geom[0]) if self._geom else 0

    # ---------- данные и отображение
    def set_data(self, df: pd.DataFrame, time_col: str) -> None:
        """Нарисовать события df; порядок строк df задаёт номера событий."""
        time_is_dt = is_datetime(df[time_col])
        t = to_float(df[time_col])
        self._geom = [p.geometry(df, t, time_is_dt) for p in self.panels]

        for a in self._artists:
            a.remove()
        self._artists = [a for p, ax, g in zip(self.panels, self.axes, self._geom)
                         for a in p.draw(ax, g, self._palette[0])]

        if len(t):
            self._autoscale()
        self._set_time_format(time_is_dt)
        self.toolbar.update()  # сброс истории навигации: Home вернёт к новому виду
        self.canvas.draw_idle()

    def set_selected(self, mask: np.ndarray) -> None:
        """Покрасить события: mask[i] = True — выделено."""
        colors = self._palette[mask.astype(np.intp)]
        for a in self._artists:
            a.set_color(colors)
        self.canvas.draw_idle()

    def _autoscale(self) -> None:
        # ось X общая для всех графиков и учитывает концы отрезков
        self.axes[0].set_xlim(_padded(np.concatenate([g[:, :, 0].ravel() for g in self._geom])))
        for ax, g in zip(self.axes, self._geom):
            ax.set_ylim(_padded(g[:, :, 1]))

    def _set_time_format(self, is_datetime_axis: bool) -> None:
        xaxis = self.axes[0].xaxis  # у осей с sharex общие локатор и форматтер
        if is_datetime_axis:
            loc = mdates.AutoDateLocator()
            xaxis.set_major_locator(loc)
            xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        else:
            xaxis.set_major_locator(AutoLocator())
            xaxis.set_major_formatter(ScalarFormatter())

    # ---------- мышь
    def _on_press(self, event) -> None:
        self._press = (event.x, event.y) if event.button == MouseButton.LEFT else None

    def _on_release(self, event) -> None:
        """Клик: сообщить о ближайшем событии."""
        press, self._press = self._press, None
        if (press is None or self.toolbar.mode or self.on_toggle is None
                or event.inaxes not in self.axes or not self.n_events
                or np.hypot(event.x - press[0], event.y - press[1]) > self.click_tol):
            return
        geom = self._geom[self.axes.index(event.inaxes)]
        px = event.inaxes.transData.transform(geom.reshape(-1, 2)).reshape(geom.shape)
        d2 = seg_dist2(px, (event.x, event.y))
        row = int(d2.argmin())
        if d2[row] <= self.pick_tol ** 2:
            self.on_toggle(row)

    def _on_rect(self, i: int, eclick, erelease) -> None:
        """Окно: сообщить о задетых событиях."""
        if (self.toolbar.mode or self.on_select is None or not self.n_events
                or np.hypot(erelease.x - eclick.x, erelease.y - eclick.y) <= self.click_tol):
            return  # слишком маленькое окно — это клик, его обработает _on_release
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        mask = seg_in_rect(self._geom[i], x0, x1, y0, y1)
        if mask.any():
            self.on_select(mask)
