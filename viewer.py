"""EventViewer — графики + таблица с общим выделением, сохранение и последовательная разметка."""
from __future__ import annotations

import functools
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .conversion import to_text
from .panels import PanelSpec
from .plot import EventPlot
from .table import EventTable


class EventViewer(ttk.Frame):
    """Графики событий и таблица с общим выделением.

    time_col   — колонка времени (числа или даты).
    panels     — панели сверху вниз: строка (= Points) или Points / Segments / своя панель.
    table_cols — колонки таблицы; None — все колонки DataFrame. Индекс показывается всегда.
    font       — шрифт таблицы и кнопок, например ("Arial", 11); None — TkDefaultFont.

    Виджет создаётся один раз, данные меняются через load(). Внутри события адресуются
    позицией строки в self.df (DataFrame, отсортированный по времени).
    """

    def __init__(self, master, time_col: str, panels: Sequence[PanelSpec],
                 table_cols: Sequence[str] | None = None, font=None,
                 color="tab:blue", sel_color="tab:red", **kw):
        super().__init__(master, **kw)
        self.time_col = time_col
        self.table_cols = None if table_cols is None else list(table_cols)
        self.font = tkfont.Font(self, font=font or "TkDefaultFont")
        self.df = pd.DataFrame()
        self._selected = np.zeros(0, dtype=bool)

        self._next = tk.BooleanVar(self, value=False)  # для wait_next()
        self._alive = True
        self.bind("<Destroy>", self._on_destroy, add="+")

        # PanedWindow — границу между графиками и таблицей можно тянуть мышью
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        self.plot = EventPlot(paned, panels, time_label=time_col,
                              on_toggle=functools.partial(self.toggle, scroll=True),
                              on_select=self._add, color=color, sel_color=sel_color)
        self.table = EventTable(paned, on_click=self.toggle, font=self.font)
        paned.add(self.plot, weight=3)
        paned.add(self.table, weight=1)
        self._add_buttons(self.plot.toolbar)

    def _add_buttons(self, tb) -> None:
        style = f"V{id(self)}.TButton"
        ttk.Style(self).configure(style, font=self.font)
        ttk.Separator(tb, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)
        ttk.Button(tb, text="Снять выделение", style=style,
                   command=self.clear_selection).pack(side=tk.LEFT, padx=2)
        ttk.Button(tb, text="Сохранить CSV", style=style,
                   command=self.save_selected).pack(side=tk.LEFT, padx=2)
        self._next_btn = ttk.Button(tb, text="Далее ▶", style=style, state=tk.DISABLED,
                                    command=lambda: self._next.set(True))
        self._next_btn.pack(side=tk.LEFT, padx=2)
        self._count = tk.StringVar(self, value="Выделено: 0")
        ttk.Label(tb, textvariable=self._count, font=self.font).pack(side=tk.LEFT, padx=8)

    # ---------- данные
    def load(self, df: pd.DataFrame, selected: Iterable | None = None) -> None:
        """Показать новый DataFrame. selected — метки df.index, выделенные изначально."""
        self.df = df.sort_values(self.time_col, kind="stable")
        self._selected = np.zeros(len(self.df), dtype=bool)
        self.plot.set_data(self.df, self.time_col)
        self.table.set_data(self._table_columns())
        self._count.set("Выделено: 0")
        if selected is not None:
            self.select(selected)

    def _table_columns(self) -> list[tuple[str, list[str]]]:
        cols = self.table_cols if self.table_cols is not None else list(self.df.columns)
        index = (str(self.df.index.name or "index"), to_text(self.df.index.to_series()))
        return [index, *[(str(c), to_text(self.df[c])) for c in cols]]

    # ---------- выделение (публичный API)
    @property
    def selected(self) -> np.ndarray:
        """Копия маски выделения (по строкам self.df)."""
        return self._selected.copy()

    @property
    def selected_index(self) -> pd.Index:
        return self.df.index[self._selected]

    @property
    def selected_df(self) -> pd.DataFrame:
        return self.df.iloc[np.flatnonzero(self._selected)]

    def select(self, labels: Iterable, scroll: bool = True) -> None:
        """Добавить в выделение события с метками индекса labels (отсутствующие пропускаются)."""
        self._add(self.df.index.isin(list(labels)), scroll=scroll)

    def toggle(self, row: int, scroll: bool = False) -> None:
        """Переключить выделение события по позиции строки."""
        new = self._selected.copy()
        new[row] = not new[row]
        self._set_selection(new, scroll_row=row if scroll else None)

    def clear_selection(self) -> None:
        self._set_selection(np.zeros_like(self._selected))

    def _add(self, mask: np.ndarray, scroll: bool = True) -> None:
        first = int(mask.argmax()) if scroll and mask.any() else None
        self._set_selection(self._selected | mask, scroll_row=first)

    def _set_selection(self, new: np.ndarray, scroll_row: int | None = None) -> None:
        """Единая точка изменения выделения: графики, таблица, счётчик."""
        changed = np.flatnonzero(new != self._selected)
        if changed.size:
            self._selected = new
            self.plot.set_selected(new)
            self.table.mark(changed, new[changed])
            self._count.set(f"Выделено: {int(new.sum())}")
        if scroll_row is not None:
            self.table.scroll_to(scroll_row)

    # ---------- сохранение
    def save_selected(self, path: str | None = None) -> str | None:
        """Сохранить выделенные события в CSV. Без path спрашивает файл в диалоге.

        Возвращает путь к файлу или None, если сохранение отменено.
        """
        if path is None:
            if not self._selected.any():
                messagebox.showinfo("Сохранение", "Нет выделенных событий.", parent=self)
                return None
            path = filedialog.asksaveasfilename(parent=self, defaultextension=".csv",
                                                filetypes=[("CSV", "*.csv")])
            if not path:
                return None
        self.selected_df.to_csv(path)
        return path

    # ---------- последовательная разметка
    def wait_next(self) -> pd.Index | None:
        """Ждать нажатия «Далее», не блокируя окно.

        Возвращает метки выделенных событий или None, если окно закрыли.
        """
        self._next.set(False)
        self._next_btn.state(["!disabled"])
        self.wait_variable(self._next)
        if not self._alive:
            return None
        self._next_btn.state(["disabled"])
        return self.selected_index

    def _on_destroy(self, event) -> None:
        if event.widget is self:
            self._alive = False
            self._next.set(True)  # прервать wait_next()
