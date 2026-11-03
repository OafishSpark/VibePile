"""Векторная геометрия отрезков для выделения событий.

Каждое событие — отрезок seg[i] = [[x0, y0], [x1, y1]], массив seg имеет форму (n, 2, 2).
Точка — отрезок нулевой длины, поэтому функции одинаково работают для точек и отрезков.
"""
from __future__ import annotations

import numpy as np


def seg_dist2(seg: np.ndarray, p) -> np.ndarray:
    """Квадрат расстояния от точки p = (x, y) до каждого отрезка. NaN -> inf."""
    a, ab = seg[:, 0], seg[:, 1] - seg[:, 0]
    ap = np.asarray(p, dtype=np.float64) - a
    l2 = (ab ** 2).sum(axis=1)
    u = np.divide((ap * ab).sum(axis=1), l2, out=np.zeros_like(l2), where=l2 > 0)
    d = ap - np.clip(u, 0.0, 1.0)[:, None] * ab
    d2 = (d ** 2).sum(axis=1)
    return np.where(np.isnan(d2), np.inf, d2)


def seg_in_rect(seg: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    """Маска отрезков, хотя бы частично попавших в прямоугольник (алгоритм Лианга — Барски).

    Требуется x0 <= x1 и y0 <= y1. Отрезки с NaN никогда не попадают.
    """
    a, d = seg[:, 0], seg[:, 1] - seg[:, 0]
    ok = np.isfinite(seg).all(axis=(1, 2))
    t0, t1 = np.zeros(len(seg)), np.ones(len(seg))
    with np.errstate(divide="ignore", invalid="ignore"):
        for p, q in ((-d[:, 0], a[:, 0] - x0), (d[:, 0], x1 - a[:, 0]),
                     (-d[:, 1], a[:, 1] - y0), (d[:, 1], y1 - a[:, 1])):
            r = q / p
            t0 = np.where(p < 0, np.maximum(t0, r), t0)
            t1 = np.where(p > 0, np.minimum(t1, r), t1)
            ok &= ~((p == 0) & (q < 0))  # параллелен стороне и лежит снаружи
    return ok & (t0 <= t1)
