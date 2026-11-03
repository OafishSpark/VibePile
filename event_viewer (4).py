"""Просмотр последовательности событий: графики от времени + таблица.

Зависимости: numpy, pandas, matplotlib (>= 3.6), tkinter.
"""
import functools
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from tkinter import filedialog, messagebox, ttk

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.ticker import AutoLocator, ScalarFormatter
from matplotlib.widgets import RectangleSelector

# Шрифт таблицы и кнопок: None — системный TkDefaultFont, либо, например, ("Segoe UI", 10).
# Можно переопределить для конкретного виджета: EventViewer(..., font=("Arial", 11)).
FONT = None
MARKER_SIZE = 14  # размер точек, pt^2

CLICK_TOL = 5  # px: если мышь сдвинулась меньше — это клик, а не выделение окном
PICK_TOL = 8   # px: радиус поиска ближайшего события при клике


# ---------------------------------------------------------------- утилиты
def to_float(s: pd.Series) -> np.ndarray:
    """Столбец (числа или даты) -> float64 для вычислений и отрисовки."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return mdates.date2num(s.to_numpy())
    return s.to_numpy(dtype=np.float64)


def to_text(s: pd.Series) -> list[str]:
    """Столбец -> строки для таблицы (дробные числа округляются до тысячных).

    Длительности (timedelta) выводятся в секундах.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.strftime("%Y-%m-%d %H:%M:%S.%f").str[:-3].fillna("").tolist()
    if pd.api.types.is_timedelta64_dtype(s):
        return [f"{v:.3f}" for v in s.dt.total_seconds().to_numpy(dtype=np.float64)]
    if pd.api.types.is_integer_dtype(s):
        return s.astype(str).tolist()
    if pd.api.types.is_numeric_dtype(s):
        return [f"{v:.3f}" for v in s.to_numpy(dtype=np.float64)]
    return s.astype(str).tolist()


def padded(a: np.ndarray, frac: float = 0.05) -> tuple[float, float]:
    """Границы оси с небольшим запасом."""
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    pad = (hi - lo) * frac or 0.5
    return lo - pad, hi + pad


# ---------------------------------------------------------------- геометрия
# Каждое событие на панели — отрезок seg[i] = [[x0, y0], [x1, y1]].
# Точка — отрезок нулевой длины, поэтому клик и окно работают одинаково для всех панелей.

def seg_dist2(seg: np.ndarray, p) -> np.ndarray:
    """Квадрат расстояния от точки p до каждого отрезка (n, 2, 2)."""
    a, ab = seg[:, 0], seg[:, 1] - seg[:, 0]
    ap = np.asarray(p, dtype=np.float64) - a
    l2 = (ab ** 2).sum(axis=1)
    u = np.divide((ap * ab).sum(axis=1), l2, out=np.zeros_like(l2), where=l2 > 0)
    d = ap - np.clip(u, 0.0, 1.0)[:, None] * ab
    d2 = (d ** 2).sum(axis=1)
    return np.where(np.isnan(d2), np.inf, d2)


def seg_in_rect(seg: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Маска отрезков, у которых хотя бы часть внутри прямоугольника (Лианг — Барски)."""
    a, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    ok = np.isfinite(seg).all(axis=(1, 2))
    t0, t1 = np.zeros(len(seg)), np.ones(len(seg))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-d[:, 0], a[:, 0] - x0), (d[:, 0], x1 - a[:, 0]),
                     (-d[:, 1], a[:, 1] - y0), (d[:, 1], y1 - a[:, 1])):
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
            ok &= ~((p == 0) & (q < 0))  # параллелен стороне и снаружи
    return ok & (t0 <= t1)


# ---------------------------------------------------------------- панели
# Панель должна иметь: label, geometry(df, t, t_is_dt) -> (n, 2, 2) и
# draw(ax, geom, color) -> список артистов, у каждого set_color((n, 4)) красит события.
@dataclass(frozen=True)
class Points:
    """Панель-точки: значение колонки y в момент события."""
    y: str

    @property
    def label(self) -> str:
        return self.y

    def geometry(self, df: pd.DataFrame, t: np.ndarray, t_is_dt: bool) -> np.ndarray:
        p = np.column_stack((t, df[self.y].to_numpy(dtype=np.float64)))
        return np.stack((p, p), axis=1)

    def draw(self, ax, geom: np.ndarray, color) -> list:
        return [ax.scatter(geom[:, 0, 0], geom[:, 0, 1], s=MARKER_SIZE, linewidths=0, color=color)]


@dataclass(frozen=True)
class Segments:
    """Панель-отрезки: от (t, start) до (t + duration, end).

    duration — timedelta или число. Если время — даты, число берётся в единицах unit;
    если время числовое, длительность должна быть в тех же единицах (timedelta -> секунды).
    """
    start: str
    end: str
    duration: str
    unit: str = "s"
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.start

    def _duration(self, df: pd.DataFrame, t_is_dt: bool) -> np.ndarray:
        d = df[self.duration]
        if pd.api.types.is_timedelta64_dtype(d):
            sec = d.dt.total_seconds().to_numpy(dtype=np.float64)
            return sec / 86400.0 if t_is_dt else sec  # даты в matplotlib — в днях
        dur = d.to_numpy(dtype=np.float64)
        return dur * (pd.Timedelta(1, unit=self.unit) / pd.Timedelta(days=1)) if t_is_dt else dur

    def geometry(self, df: pd.DataFrame, t: np.ndarray, t_is_dt: bool) -> np.ndarray:
        a = np.column_stack((t, df[self.start].to_numpy(dtype=np.float64)))
        b = np.column_stack((t + self._duration(df, t_is_dt), df[self.end].to_numpy(dtype=np.float64)))
        return np.stack((a, b), axis=1)

    def draw(self, ax, geom: np.ndarray, color) -> list:
        # transform задан явно: отрезки всегда в координатах данных и следуют за зумом
        lc = LineCollection(geom, colors=[color], linewidths=1.5, transform=ax.transData)
        mid = geom.mean(axis=1)  # точка в центре отрезка
        pts = ax.scatter(mid[:, 0], mid[:, 1], s=MARKER_SIZE, linewidths=0, color=color, zorder=3)
        return [ax.add_collection(lc, autolim=False), pts]


# ---------------------------------------------------------------- таблица
class EventTable(ttk.Frame):
    """Прокручиваемая таблица (ttk.Treeview) с подсветкой выделенных строк.

    Строки адресуются позицией: iid = '0', '1', ...
    on_click(row: int) вызывается при клике по строке.
    font — шрифт таблицы (None — TkDefaultFont).
    """

    def __init__(self, master, on_click=None, font=None, **kw):
        super().__init__(master, **kw)
        self.on_click = on_click

        # шрифты явно привязаны к self: иначе при нескольких tk.Tk() они создаются не в том окне
        self._font = tkfont.Font(self, font=font or "TkDefaultFont")
        self._bold = tkfont.Font(self, font=self._font)
        self._bold.configure(weight="bold")

        # собственный стиль, чтобы не менять остальные Treeview в приложении
        style_name = f"T{id(self)}.Treeview"
        style = ttk.Style(self)
        style.configure(style_name, font=self._font,
                        rowheight=self._font.metrics("linespace") + 6)  # высота строки под шрифт
        style.configure(f"{style_name}.Heading", font=self._bold)

        self.tree = ttk.Treeview(self, show="headings", selectmode="none", style=style_name)
        self.tree.bind("<Button-1>", self._on_click)
        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.tree.tag_configure("sel", font=self._bold, background="#ffd9d9")
        self._n = 0

    def set_data(self, columns: list[tuple[str, list[str]]]) -> None:
        """columns: [(заголовок, список строк), ...], все списки одной длины."""
        self.tree.delete(*self.tree.get_children())
        ids = [str(i) for i in range(len(columns))]  # id колонок не зависят от заголовков
        self.tree["columns"] = ids
        for cid, (name, values) in zip(ids, columns):
            # ширина по самому длинному значению жирным шрифтом -> цифры не наезжают
            longest = max(values, key=len, default="")
            width = max(self._bold.measure(name), self._bold.measure(longest)) + 16
            self.tree.heading(cid, text=name)
            self.tree.column(cid, width=width, minwidth=width, anchor=tk.E, stretch=False)

        rows = list(zip(*(values for _, values in columns)))
        for i, row in enumerate(rows):
            self.tree.insert("", tk.END, iid=str(i), values=row)
        self._n = len(rows)

    def _on_click(self, event) -> None:
        if self.on_click is None or self.tree.identify_region(event.x, event.y) != "cell":
            return  # клик по заголовку, разделителю или пустому месту
        row = self.tree.identify_row(event.y)
        if row:
            self.on_click(int(row))

    def mark(self, rows: np.ndarray, flags: np.ndarray) -> None:
        """Обновить подсветку только у изменившихся строк."""
        for r, f in zip(rows.tolist(), flags.tolist()):
            self.tree.item(str(r), tags=("sel",) if f else ())

    def scroll_to(self, row: int) -> None:
        """Прокрутить так, чтобы строка row оказалась наверху."""
        if self._n:
            self.tree.yview_moveto(row / self._n)


# ---------------------------------------------------------------- основной виджет
class EventViewer(ttk.Frame):
    """Панели «величина(время)» + таблица событий с общим выделением.

    panels     — список панелей: строка (= Points) или Points / Segments.
    table_cols — колонки таблицы; None — все колонки DataFrame.
    font       — шрифт таблицы и кнопок (по умолчанию FONT).
    Создаётся один раз, данные меняются через load(df).
    """

    def __init__(self, master, time_col: str, panels: list, table_cols: list[str] | None = None,
                 color="tab:blue", sel_color="tab:red", font=FONT, **kw):
        super().__init__(master, **kw)
        self.time_col = time_col
        self.panels = [Points(p) if isinstance(p, str) else p for p in panels]
        self.table_cols = None if table_cols is None else list(table_cols)
        self.font = tkfont.Font(self, font=font or "TkDefaultFont")
        self._palette = np.array([to_rgba(color), to_rgba(sel_color)])  # (2, 4)

        self.df = pd.DataFrame()
        self.selected = np.zeros(0, dtype=bool)
        self._geom: list[np.ndarray] = []  # для каждой панели (n, 2, 2) float64
        self._artists = []                 # все артисты всех панелей (scatter, LineCollection)
        self._press = None                 # координаты нажатия мыши, px

        self._next = tk.BooleanVar(self, value=False)  # для wait_next()
        self._alive = True
        self.bind("<Destroy>", self._on_destroy, add="+")

        # PanedWindow — границу между графиками и таблицей можно тянуть мышью
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        plot_frame = ttk.Frame(paned)
        self.table = EventTable(paned, on_click=self._toggle, font=self.font)
        paned.add(plot_frame, weight=3)
        paned.add(self.table, weight=1)

        self._build_plot(plot_frame)

    # ---------- построение интерфейса
    def _build_plot(self, parent) -> None:
        self.fig = Figure(figsize=(8, 6), layout="constrained")
        axs = self.fig.subplots(len(self.panels), 1, sharex=True, squeeze=False)
        self.axes = list(axs[:, 0])
        for ax, panel in zip(self.axes, self.panels):
            ax.set_ylabel(panel.label)
            ax.grid(alpha=0.3)
        self.axes[-1].set_xlabel(self.time_col)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self._add_buttons(self.toolbar)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # во время зума/панорамирования селекторы отключаются автоматически (widgetlock)
        self._selectors = [
            RectangleSelector(ax, functools.partial(self._on_rect, i), useblit=True,
                              button=[MouseButton.LEFT],
                              props=dict(facecolor="tab:red", alpha=0.2))
            for i, ax in enumerate(self.axes)
        ]
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _add_buttons(self, tb) -> None:
        btn_style = f"V{id(self)}.TButton"
        ttk.Style(self).configure(btn_style, font=self.font)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(tb, text="Снять выделение", style=btn_style, command=self.clear_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Сохранить CSV", style=btn_style, command=self.save_selected).pack(side=tk.LEFT, padx=2)
        self._next_btn = ttk.Button(tb, text="Далее ▶", style=btn_style, state=tk.DISABLED,
                                    command=lambda: self._next.set(True))
        self._next_btn.pack(side=tk.LEFT, padx=2)
        self._count = tk.StringVar(self, value="Выделено: 0")
        ttk.Label(tb, textvariable=self._count, font=self.font).pack(side=tk.LEFT, padx=8)

    # ---------- данные
    def load(self, df: pd.DataFrame, selected=None) -> None:
        """Показать новый DataFrame (виджеты переиспользуются).

        selected — индексы (метки df.index) событий, выделенных изначально.
        """
        self.df = df.sort_values(self.time_col, kind="stable")
        t_is_dt = pd.api.types.is_datetime64_any_dtype(self.df[self.time_col])
        t = to_float(self.df[self.time_col])
        self._geom = [p.geometry(self.df, t, t_is_dt) for p in self.panels]
        self.selected = np.zeros(len(t), dtype=bool)

        for a in self._artists:
            a.remove()
        self._artists = [a for p, ax, g in zip(self.panels, self.axes, self._geom)
                         for a in p.draw(ax, g, self._palette[0])]

        # ось X общая для всех графиков и учитывает концы отрезков
        self.axes[0].set_xlim(padded(np.concatenate([g[:, :, 0].ravel() for g in self._geom])))
        for ax, g in zip(self.axes, self._geom):
            ax.set_ylim(padded(g[:, :, 1]))
        self._set_time_format(t_is_dt)

        cols = self.table_cols if self.table_cols is not None else list(self.df.columns)
        self.table.set_data([(str(self.df.index.name or "index"), to_text(self.df.index.to_series())),
                             *[(str(c), to_text(self.df[c])) for c in cols]])

        self._count.set("Выделено: 0")
        self.toolbar.update()  # сброс истории навигации, Home вернёт к новому виду
        self.canvas.draw_idle()

        if selected is not None:
            mask = self.df.index.isin(list(selected))
            self._apply(mask, scroll_row=int(mask.argmax()) if mask.any() else None)

    def _set_time_format(self, is_datetime: bool) -> None:
        xaxis = self.axes[0].xaxis  # у осей с sharex общие локатор и форматтер
        if is_datetime:
            loc = mdates.AutoDateLocator()
            xaxis.set_major_locator(loc)
            xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
        else:
            xaxis.set_major_locator(AutoLocator())
            xaxis.set_major_formatter(ScalarFormatter())

    @property
    def selected_df(self) -> pd.DataFrame:
        return self.df.iloc[np.flatnonzero(self.selected)]

    # ---------- последовательная разметка
    def wait_next(self) -> pd.Index | None:
        """Ждать нажатия «Далее», не блокируя окно.

        Возвращает индексы выделенных событий или None, если окно закрыли.
        """
        self._next.set(False)
        self._next_btn.state(["!disabled"])
        self.wait_variable(self._next)
        if not self._alive:
            return None
        self._next_btn.state(["disabled"])
        return self.selected_df.index

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._alive = False
            self._next.set(True)  # прервать wait_next()

    # ---------- выделение
    def _apply(self, new: np.ndarray, scroll_row: int | None = None) -> None:
        """Единая точка изменения выделения: цвета, таблица, счётчик."""
        changed = np.flatnonzero(new != self.selected)
        if changed.size:
            self.selected = new
            colors = self._palette[new.astype(np.intp)]
            for a in self._artists:
                a.set_color(colors)
            self.table.mark(changed, new[changed])
            self._count.set(f"Выделено: {int(new.sum())}")
            self.canvas.draw_idle()
        if scroll_row is not None:
            self.table.scroll_to(scroll_row)

    def _toggle(self, row: int, scroll: bool = False) -> None:
        """Переключить выделение одного события (по позиции строки)."""
        new = self.selected.copy()
        new[row] = not new[row]
        self._apply(new, scroll_row=row if scroll else None)

    def clear_selection(self) -> None:
        self._apply(np.zeros_like(self.selected))

    def _on_press(self, event) -> None:
        self._press = (event.x, event.y) if event.button == MouseButton.LEFT else None

    def _on_release(self, event) -> None:
        """Клик: переключить выделение ближайшего события."""
        press, self._press = self._press, None
        if (press is None or self.toolbar.mode or event.inaxes not in self.axes
                or not self.selected.size
                or np.hypot(event.x - press[0], event.y - press[1]) > CLICK_TOL):
            return
        geom = self._geom[self.axes.index(event.inaxes)]
        px = event.inaxes.transData.transform(geom.reshape(-1, 2)).reshape(geom.shape)
        d2 = seg_dist2(px, (event.x, event.y))
        j = int(d2.argmin())
        if d2[j] <= PICK_TOL ** 2:
            self._toggle(j, scroll=True)

    def _on_rect(self, i: int, eclick, erelease) -> None:
        """Окно: только добавляет события в выделение."""
        if self.toolbar.mode or np.hypot(erelease.x - eclick.x, erelease.y - eclick.y) <= CLICK_TOL:
            return  # слишком маленькое окно — это клик, его обработает _on_release
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        inside = seg_in_rect(self._geom[i], x0, x1, y0, y1)
        if inside.any():
            self._apply(self.selected | inside, scroll_row=int(inside.argmax()))

    # ---------- сохранение
    def save_selected(self) -> None:
        if not self.selected.any():
            messagebox.showinfo("Сохранение", "Нет выделенных событий.", parent=self)
            return
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self.selected_df.to_csv(path)


# ---------------------------------------------------------------- пример
def make_demo(seed: int, n: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {"time": pd.Timestamp("2026-01-01") + pd.to_timedelta(np.sort(rng.uniform(0, 3600, n)), unit="s"),
         "dur": pd.to_timedelta(rng.uniform(1, 10, n), unit="s")},
        index=pd.Index(np.arange(1, n + 1), name="event_id"),
    )
    for k in range(4):
        df[f"p{k}"] = rng.normal(size=n) * 10 ** k
    df["p2_end"] = df["p2"] + rng.normal(scale=50, size=n)
    df["p3_end"] = df["p3"] + rng.normal(scale=500, size=n)
    df["channel"] = rng.choice(["A", "B", "C"], size=n)  # есть только в таблице
    return df


if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1300x750")
    viewer = EventViewer(root, time_col="time", panels=[
        "p0",
        "p1",
        Segments(start="p2", end="p2_end", duration="dur"),
        Segments(start="p3", end="p3_end", duration="dur"),
    ])
    viewer.pack(fill=tk.BOTH, expand=True)

    results = {}
    for i in range(3):
        root.title(f"Набор {i + 1} из 3")
        viewer.load(make_demo(i), selected=[5, 10, 500] if i == 0 else None)
        sel = viewer.wait_next()
        if sel is None:  # окно закрыли
            break
        results[i] = sel
        print(f"Набор {i + 1}: выделено {len(sel)}")
    else:
        root.destroy()
