"""Прокручиваемая таблица событий с подсветкой выделенных строк."""
from __future__ import annotations

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk
from typing import Callable

import numpy as np


class EventTable(ttk.Frame):
    """Таблица на ttk.Treeview. Строки адресуются позицией: iid = '0', '1', ...

    on_click(row) вызывается при клике по строке. Само выделение таблица не хранит,
    а только показывает его: mark() и scroll_to().
    """

    sel_background = "#ffd9d9"

    def __init__(self, master, on_click: Callable[[int], None] | None = None, font=None, **kw):
        super().__init__(master, **kw)
        self.on_click = on_click
        self._n = 0

        # шрифты явно привязаны к self: иначе при нескольких tk.Tk() они создаются не в том окне
        self.font = tkfont.Font(self, font=font or "TkDefaultFont")
        self.bold = tkfont.Font(self, font=self.font)
        self.bold.configure(weight="bold")

        # собственный стиль, чтобы не менять остальные Treeview в приложении
        style_name = f"T{id(self)}.Treeview"
        style = ttk.Style(self)
        style.configure(style_name, font=self.font, rowheight=self.font.metrics("linespace") + 6)
        style.configure(f"{style_name}.Heading", font=self.bold)

        self.tree = ttk.Treeview(self, show="headings", selectmode="none", style=style_name)
        self.tree.tag_configure("sel", font=self.bold, background=self.sel_background)
        self.tree.bind("<Button-1>", self._on_click)

        ysb = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self.tree.yview)
        xsb = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=ysb.set, xscrollcommand=xsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        ysb.grid(row=0, column=1, sticky="ns")
        xsb.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def __len__(self) -> int:
        return self._n

    def set_data(self, columns: list[tuple[str, list[str]]]) -> None:
        """columns: [(заголовок, список строк), ...], все списки одной длины."""
        self.tree.delete(*self.tree.get_children())
        ids = [str(i) for i in range(len(columns))]  # id колонок не зависят от заголовков
        self.tree["columns"] = ids
        for cid, (name, values) in zip(ids, columns):
            # ширина по самому длинному значению жирным шрифтом -> текст не наезжает
            longest = max(values, key=len, default="")
            width = max(self.bold.measure(name), self.bold.measure(longest)) + 16
            self.tree.heading(cid, text=name)
            self.tree.column(cid, width=width, minwidth=width, anchor=tk.E, stretch=False)

        rows = list(zip(*(values for _, values in columns)))
        for i, row in enumerate(rows):
            self.tree.insert("", tk.END, iid=str(i), values=row)
        self._n = len(rows)

    def mark(self, rows: np.ndarray, flags: np.ndarray) -> None:
        """Обновить подсветку только у перечисленных строк."""
        for r, f in zip(rows.tolist(), flags.tolist()):
            self.tree.item(str(r), tags=("sel",) if f else ())

    def scroll_to(self, row: int) -> None:
        """Прокрутить так, чтобы строка row оказалась наверху."""
        if self._n:
            self.tree.yview_moveto(row / self._n)

    def _on_click(self, event) -> None:
        if self.on_click is None or self.tree.identify_region(event.x, event.y) != "cell":
            return  # клик по заголовку, разделителю или пустому месту
        row = self.tree.identify_row(event.y)
        if row:
            self.on_click(int(row))
