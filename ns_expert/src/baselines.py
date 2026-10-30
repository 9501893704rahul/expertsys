"""Classical machine-learning baselines with a common fit/predict interface."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class SklearnModel:
    def __init__(self, name: str, est, seed: int = 0):
        self.name = name
        self.est = est
        self.seed = seed

    def fit(self, X, y, X_val=None, y_val=None):
        self.est.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.est.predict_proba(X)[:, 1]

    def predict(self, X, thr: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= thr).astype(int)

    def rule_confidence(self):
        return None


def build_baselines(seed: int = 0) -> dict[str, SklearnModel]:
    """Standard tabular baselines.

    ``class_weight="balanced"`` is set wherever the estimator supports it, to
    match the class weighting used by the neural models on the imbalanced
    datasets. Gradient boosting has no such option and is left unweighted.
    """
    return {
        "LogReg": SklearnModel("LogReg", LogisticRegression(max_iter=2000, C=1.0,
                                                            class_weight="balanced")),
        "DecisionTree": SklearnModel(
            "DecisionTree", DecisionTreeClassifier(max_depth=4, min_samples_leaf=30,
                                                   class_weight="balanced", random_state=seed)
        ),
        "RandomForest": SklearnModel(
            "RandomForest",
            RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5,
                                   class_weight="balanced", n_jobs=-1, random_state=seed),
        ),
        "GradientBoosting": SklearnModel(
            "GradientBoosting",
            GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=3,
                                       random_state=seed),
        ),
    }
