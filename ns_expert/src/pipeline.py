"""Leak-free preprocessing.

Every fitted transformation --- missing-value imputation, feature scaling, and
feature selection --- is estimated on the *training* rows only and then applied
to validation and test rows.  This module exists because the earlier code
imputed with full-dataset medians before splitting, which leaks test
information into training.

``Preprocessor`` is deliberately sklearn-shaped (``fit``/``transform``) so it
composes with the model wrappers, and it is a plain object rather than an
sklearn ``Pipeline`` because the expert knowledge base is expressed in *raw*
units and must be mapped through the same bounds the scaler learned.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Preprocessor:
    """Fit imputation / scaling / selection on train, apply to val+test.

    Parameters
    ----------
    scale:
        ``"minmax"`` (default) maps each feature to ``[0, 1]`` using training
        bounds; ``"standard"`` standardises with training mean/std; ``"none"``
        leaves values untouched (used by datasets whose loader already emits the
        canonical scale, such as the discrete MONK / tic-tac-toe encodings, so
        the encoding cannot drift with the training fold).
    select:
        ``None`` keeps all features; an int keeps the top-``k`` features by
        training-set association with the label; ``"corr"`` drops features whose
        absolute training correlation with another retained feature exceeds
        ``corr_threshold``.
    """

    scale: str = "minmax"
    select: int | str | None = None
    corr_threshold: float = 0.98
    clip: tuple[float, float] = (0.0, 1.0)

    medians_: pd.Series | None = None
    lo_: pd.Series | None = None
    hi_: pd.Series | None = None
    mean_: pd.Series | None = None
    std_: pd.Series | None = None
    kept_: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "Preprocessor":
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Preprocessor.fit expects a DataFrame")
        X = X.astype(float)
        # imputation: training medians only
        self.medians_ = X.median()
        Xf = X.fillna(self.medians_)
        self._fit_scaling(Xf)
        Z = self._scale(Xf)
        self.kept_ = self._select(Z, y)
        return self

    def _fit_scaling(self, Xf: pd.DataFrame) -> None:
        if self.scale == "minmax":
            self.lo_, self.hi_ = Xf.min(), Xf.max()
        elif self.scale == "standard":
            self.mean_, self.std_ = Xf.mean(), Xf.std(ddof=0).replace(0, 1.0)
        elif self.scale == "none":
            # canonical scale already supplied by the loader; identity transform
            self.lo_ = pd.Series(0.0, index=Xf.columns)
            self.hi_ = pd.Series(1.0, index=Xf.columns)
        else:
            raise ValueError(f"unknown scale {self.scale!r}")

    def _scale(self, Xf: pd.DataFrame) -> pd.DataFrame:
        if self.scale == "minmax":
            span = (self.hi_ - self.lo_).replace(0, 1.0)
            Z = (Xf - self.lo_) / span
            return Z.clip(*self.clip)
        if self.scale == "none":
            return Xf.copy()
        Z = (Xf - self.mean_) / self.std_
        return Z

    def _select(self, Z: pd.DataFrame, y: np.ndarray | None) -> list[str]:
        cols = list(Z.columns)
        if self.select is None or y is None:
            return cols
        yv = np.asarray(y, dtype=float)
        if self.select == "corr":
            keep: list[str] = []
            for c in cols:
                v = Z[c].to_numpy()
                if np.std(v) == 0:
                    continue
                if any(abs(np.corrcoef(v, Z[k].to_numpy())[0, 1]) > self.corr_threshold
                       for k in keep):
                    continue
                keep.append(c)
            return keep or cols
        # univariate association with the label, measured on training rows
        scores = {}
        for c in cols:
            v = Z[c].to_numpy()
            if np.std(v) == 0:
                scores[c] = 0.0
                continue
            scores[c] = abs(float(np.corrcoef(v, yv)[0, 1]))
        ranked = sorted(cols, key=lambda c: scores[c], reverse=True)
        return ranked[: int(self.select)]

    # ------------------------------------------------------------------ #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.medians_ is None:
            raise RuntimeError("call fit before transform")
        Xf = X.astype(float).fillna(self.medians_)
        Z = self._scale(Xf)
        return Z[self.kept_].reset_index(drop=True)

    def fit_transform(self, X: pd.DataFrame, y: np.ndarray | None = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    # ------------------------------------------------------------------ #
    def raw_bounds(self) -> tuple[pd.Series, pd.Series]:
        """Raw-space (lo, hi) per feature, for mapping expert thresholds."""
        return self.lo_, self.hi_
