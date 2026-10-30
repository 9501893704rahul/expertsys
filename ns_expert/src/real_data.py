"""Real small datasets, with expert knowledge bases built from published rules.

Three datasets are used, chosen because they are small, standard, and each
supports a different part of the evaluation:

* ``heart``  - UCI Cleveland heart disease (303 rows).  A clinical expert rule
  base is written from published cut-offs (hypertension >=140 mmHg, high
  cholesterol >=240 mg/dl, ST depression >=1.0, etc.).  The knowledge base is
  *approximate*: clinical guidelines are not the data-generating process, which
  is exactly the realistic regime.
* ``pima``   - Pima Indians diabetes (768 rows).  Expert rule base from
  ADA/WHO cut-offs (impaired fasting glucose >=126 mg/dl, obesity BMI >=30).
* ``ttt``    - Tic-tac-toe endgame (958 rows).  This dataset has a *genuine*
  ground-truth logic: a board is a win iff some row, column or diagonal is all
  ``x``.  That makes real-data rule recovery measurable against true rules,
  bridging the synthetic and real studies.

Features are min-max normalised to ``[0, 1]`` using training-split statistics, so
the same differentiable-logic machinery (bandwidth ``tau = 0.1``) applies.
Expert thresholds are mapped through the same transform, which keeps the
knowledge base in the units the models actually see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .knowledge import Domain, Literal, Rule

DATA = Path(__file__).resolve().parent.parent / "data"


# --------------------------------------------------------------------------- #
# raw loaders
# --------------------------------------------------------------------------- #
HEART_COLS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
              "thalach", "exang", "oldpeak", "slope", "ca", "thal"]


def load_heart_raw() -> tuple[pd.DataFrame, np.ndarray]:
    """Raw features with missing values *left as NaN*.

    Imputation is a fitted transformation and must happen inside the training
    fold (see :mod:`src.pipeline`); filling here with a full-dataset median
    would leak test information into training.
    """
    df = pd.read_csv(DATA / "heart_cleveland.csv", header=None, na_values="?")
    X = df.iloc[:, :13].copy()
    X.columns = HEART_COLS
    y = (df.iloc[:, 13] > 0).astype(int).to_numpy()
    X = X.apply(pd.to_numeric, errors="coerce")
    return X, y


PIMA_COLS = ["pregnancies", "glucose", "bloodpressure", "skinthickness",
             "insulin", "bmi", "pedigree", "age"]


def load_pima_raw() -> tuple[pd.DataFrame, np.ndarray]:
    """Raw Pima features; zero sentinels become NaN, imputation is deferred."""
    df = pd.read_csv(DATA / "pima.csv", header=None)
    X = df.iloc[:, :8].copy()
    X.columns = PIMA_COLS
    y = df.iloc[:, 8].astype(int).to_numpy()
    # zeros are missing-value sentinels in several Pima columns
    for c in ["glucose", "bloodpressure", "skinthickness", "insulin", "bmi"]:
        X.loc[X[c] == 0, c] = np.nan
    return X, y


TTT_CELLS = [f"c{i}{j}" for i in range(3) for j in range(3)]
TTT_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),          # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),          # columns
    (0, 4, 8), (2, 4, 6),                     # diagonals
]


MONK_COLS = ["a1", "a2", "a3", "a4", "a5", "a6"]


def load_monk_raw(version: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """MONK's problems: discrete, rule-governed, with *published* ground truth.

    Unlike heart/pima (where the expert KB is approximate) and like tic-tac-toe,
    the MONK target is an exact logical formula, so this is a second dataset on
    which structural rule recovery is measurable against a genuine truth.
    """
    frames = []
    for split in ("train", "test"):
        df = pd.read_csv(DATA / f"monks-{version}.{split}", sep=r"\s+", header=None)
        frames.append(df)
    allf = pd.concat(frames, ignore_index=True)
    # file layout is: class a1 a2 a3 a4 a5 a6 Id
    y = allf.iloc[:, 0].astype(int).to_numpy()
    X = allf.iloc[:, 1:7].copy()
    X.columns = MONK_COLS
    X = X.astype(float)
    return X, y


def load_ttt_raw() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(DATA / "tictactoe.data", header=None)
    enc = {"x": 1.0, "o": 0.0, "b": 0.5}
    X = df.iloc[:, :9].apply(lambda s: s.map(enc)).astype(float)
    X.columns = TTT_CELLS
    y = (df.iloc[:, 9].str.strip() == "positive").astype(int).to_numpy()
    return X, y


WBC_COLS = ["clump_thickness", "uniformity_cell_size", "uniformity_cell_shape",
            "marginal_adhesion", "single_epithelial_size", "bare_nuclei",
            "bland_chromatin", "normal_nucleoli", "mitoses"]


def load_wbc_raw() -> tuple[pd.DataFrame, np.ndarray]:
    """Wisconsin Breast Cancer (original). Missing values left as NaN.

    This is the dataset the KBANN breast-cancer knowledge base was published
    for (Towell & Shavlik, 1994).  Values are the standard cytology 1-10 scale;
    class 2 (benign) -> 0, class 4 (malignant) -> 1.
    """
    df = pd.read_csv(DATA / "breast_cancer_wisconsin.data", header=None,
                     na_values="?")
    X = df.iloc[:, 1:10].copy()
    X.columns = WBC_COLS
    X = X.apply(pd.to_numeric, errors="coerce")
    y = (df.iloc[:, 10].astype(int) == 4).astype(int).to_numpy()
    return X, y


# --------------------------------------------------------------------------- #
# expert knowledge bases (raw units)
# --------------------------------------------------------------------------- #
# body literals: (feature, op, raw_threshold, kind)
HEART_KB = [
    ("hypertension", [("trestbps", ">=", 140, "feature")], 0.90),
    ("hyperlipidemia", [("chol", ">=", 240, "feature")], 0.85),
    # clinical risk factors from published guidelines (not tuned on the data)
    ("risk", [("hypertension", ">=", 0.5, "concept"), ("age", ">=", 55, "feature")], 0.90),
    ("risk", [("exang", "==", 1, "feature"), ("oldpeak", ">=", 1.0, "feature")], 0.90),
    ("risk", [("hyperlipidemia", ">=", 0.5, "concept"), ("exang", "==", 1, "feature")], 0.85),
    ("risk", [("ca", ">=", 1, "feature"), ("oldpeak", ">=", 1.0, "feature")], 0.85),
    ("risk", [("sex", "==", 1, "feature"), ("age", ">=", 55, "feature"),
              ("exang", "==", 1, "feature")], 0.85),
    ("risk", [("thalach", "<=", 120, "feature"), ("oldpeak", ">=", 1.0, "feature")], 0.85),
]
HEART_CONCEPTS = ("hypertension", "hyperlipidemia")

PIMA_KB = [
    ("hyperglycemia", [("glucose", ">=", 140, "feature")], 0.90),
    ("obesity", [("bmi", ">=", 30, "feature")], 0.85),
    ("diabetes", [("hyperglycemia", ">=", 0.5, "concept"), ("age", ">=", 45, "feature")], 0.90),
    ("diabetes", [("glucose", ">=", 126, "feature"), ("obesity", ">=", 0.5, "concept")], 0.90),
    ("diabetes", [("glucose", ">=", 180, "feature")], 0.95),
    ("diabetes", [("bmi", ">=", 35, "feature"), ("age", ">=", 40, "feature")], 0.80),
]
PIMA_CONCEPTS = ("hyperglycemia", "obesity")

# tic-tac-toe: the eight winning lines are the genuine ground truth
TTT_KB = [("win", [(TTT_CELLS[i], ">=", 0.75, "feature") for i in line], 1.0)
          for line in TTT_LINES]
TTT_CONCEPTS: tuple[str, ...] = ()

# MONK-1: (a1 = a2) OR (a5 = 1).  Each disjunct is a conjunction of equalities,
# so the formula is exactly in the fragment the rule layer represents.  Equality
# is expanded into one rule per shared value: (a1 = a2) <=> OR_v (a1 = v AND a2 = v).
MONK1_KB = [
    ("monk", [("a1", "==", v, "feature"), ("a2", "==", v, "feature")], 1.0)
    for v in (1, 2, 3)
] + [("monk", [("a5", "==", 1, "feature")], 1.0)]
MONK1_CONCEPTS: tuple[str, ...] = ()

# MONK-3: (a5 = 3 AND a4 = 1) OR (a5 /= 4 AND a2 /= 3)   [5% class noise in train]
# Inequalities expand to a disjunction of equalities: a5 /= 4 <=> a5 in {1,2,3},
# a2 /= 3 <=> a2 in {1,2}.
MONK3_KB = [("monk", [("a5", "==", 3, "feature"), ("a4", "==", 1, "feature")], 1.0)]
for _a5 in (1, 2, 3):
    for _a2 in (1, 2):
        MONK3_KB.append(("monk", [("a5", "==", _a5, "feature"),
                                  ("a2", "==", _a2, "feature")], 1.0))
MONK3_CONCEPTS: tuple[str, ...] = ()

# Wisconsin breast cancer: KBANN-style hierarchical KB (Towell & Shavlik, 1994;
# Clark & Niblett, 1989).  Intermediate concepts are defined by published
# cytological criteria; the label is malignant if the relevant concept fires.
# Thresholds are the standard 1-10 cytology cut-points, not tuned on the data.
WBC_KB = [
    ("atypia", [("clump_thickness", ">=", 4, "feature"),
                ("uniformity_cell_size", ">=", 3, "feature")], 0.90),
    ("atypia", [("uniformity_cell_shape", ">=", 4, "feature"),
                ("marginal_adhesion", ">=", 3, "feature")], 0.85),
    ("nuclei", [("bare_nuclei", ">=", 3, "feature")], 0.85),
    ("nuclei", [("single_epithelial_size", ">=", 4, "feature")], 0.80),
    ("chromatin", [("bland_chromatin", ">=", 5, "feature")], 0.85),
    ("chromatin", [("normal_nucleoli", ">=", 4, "feature")], 0.80),
    ("malignant", [("atypia", ">=", 0.5, "concept"), ("nuclei", ">=", 0.5, "concept")], 0.90),
    ("malignant", [("atypia", ">=", 0.5, "concept"), ("chromatin", ">=", 0.5, "concept")], 0.85),
    ("malignant", [("clump_thickness", ">=", 6, "feature"),
                   ("uniformity_cell_size", ">=", 5, "feature")], 0.90),
    ("malignant", [("bare_nuclei", ">=", 6, "feature"),
                   ("bland_chromatin", ">=", 6, "feature")], 0.85),
]
WBC_CONCEPTS = ("atypia", "nuclei", "chromatin")

DATASETS = {
    "heart": dict(loader=load_heart_raw, kb=HEART_KB, concepts=HEART_CONCEPTS,
                  binary=("sex", "fbs", "exang"), label="risk", depth=3),
    "pima": dict(loader=load_pima_raw, kb=PIMA_KB, concepts=PIMA_CONCEPTS,
                 binary=(), label="diabetes", depth=3),
    "ttt": dict(loader=load_ttt_raw, kb=TTT_KB, concepts=TTT_CONCEPTS,
                binary=(), label="win", depth=5),
    "monk1": dict(loader=lambda: load_monk_raw(1), kb=MONK1_KB,
                  concepts=MONK1_CONCEPTS, binary=(), label="monk", depth=4),
    "monk3": dict(loader=lambda: load_monk_raw(3), kb=MONK3_KB,
                  concepts=MONK3_CONCEPTS, binary=(), label="monk", depth=5),
    "wbc": dict(loader=load_wbc_raw, kb=WBC_KB, concepts=WBC_CONCEPTS,
                binary=(), label="malignant", depth=3),
}


def ground_truth_rules(name: str) -> tuple[Rule, ...] | None:
    """Published ground-truth logic, where one exists.

    Tic-tac-toe (the eight winning lines) and MONK-1/3 (their published logical
    formulas) have a genuine ground truth, so rule recovery there measures
    recovery of the true logic rather than of the expert KB.
    """
    if name == "ttt":
        return tuple(Rule("win", tuple(Literal(TTT_CELLS[i], ">=", 0.75, "feature")
                                       for i in line)) for line in TTT_LINES)
    if name in ("monk1", "monk3"):
        spec = DATASETS[name]
        return tuple(Rule("monk", tuple(Literal(f, op, t, "feature")
                                        for f, op, t, _ in body), cf)
                     for _, body, cf in spec["kb"])
    return None


# --------------------------------------------------------------------------- #
# normalisation + domain construction
# --------------------------------------------------------------------------- #
def feature_support(X: pd.DataFrame, max_unique: int = 12) -> dict[str, np.ndarray]:
    """Observed value sets for low-cardinality features (for literal matching).

    On discrete-support features such as the tic-tac-toe cells the semantic
    literal matcher must compare partitions on the values that actually occur,
    otherwise ``>0.5`` and ``>=0.75`` look different although they select the
    same rows.  Continuous features are left out so the matcher falls back to a
    dense grid.
    """
    sup = {}
    for c in X.columns:
        u = np.unique(X[c].to_numpy(dtype=float))
        if 0 < len(u) <= max_unique:
            sup[c] = u
    return sup


def normalize(raw: pd.DataFrame, lo: pd.Series, hi: pd.Series) -> pd.DataFrame:
    span = (hi - lo).replace(0, 1.0)
    return ((raw - lo) / span).clip(0.0, 1.0)


def _map_thr(thr: float, lo: float, hi: float) -> float:
    span = hi - lo if hi > lo else 1.0
    return float(np.clip((thr - lo) / span, 0.0, 1.0))


def build_domain(name: str, lo: pd.Series, hi: pd.Series,
                 features: list[str] | None = None) -> Domain:
    """Build the domain, optionally restricted to the selected feature set.

    When feature selection drops a feature, every expert rule that references it
    is dropped too, and a concept with no surviving rule disappears from
    ``domain.concepts``.  The returned ``domain.features`` order matches the
    column order of the transformed frames, which is what the rule layer indexes
    into.
    """
    spec = DATASETS[name]
    feats = tuple(features) if features is not None else tuple(lo.index)
    keep = set(feats)

    # concept rules survive only if all their feature literals are available
    concept_rules: dict[str, list[Rule]] = {}
    for head, body, cf in spec["kb"]:
        if head not in spec["concepts"]:
            continue
        if any(kind == "feature" and f not in keep for f, _, _, kind in body):
            continue
        lits = tuple(Literal(f, op, _map_thr(t, lo[f], hi[f]), kind)
                     for f, op, t, kind in body)
        concept_rules.setdefault(head, []).append(Rule(head, lits, cf))
    concepts = tuple(c for c in spec["concepts"] if c in concept_rules)

    rules: list[Rule] = []
    for head, body, cf in spec["kb"]:
        if head in spec["concepts"]:
            continue
        ok = True
        lits = []
        for f, op, t, kind in body:
            if kind == "feature":
                if f not in keep:
                    ok = False
                    break
                lits.append(Literal(f, op, _map_thr(t, lo[f], hi[f]), "feature"))
            else:
                if f not in concepts:
                    ok = False
                    break
                lits.append(Literal(f, op, 0.5, "concept"))
        if ok:
            rules.append(Rule(head, tuple(lits), cf))
    for head in concepts:
        rules.extend(concept_rules[head])

    return Domain(
        name=name,
        features=feats,
        concepts=concepts,
        binary_features=tuple(b for b in spec["binary"] if b in keep),
        true_rules=tuple(rules),      # on real data the expert KB is all we have
        expert_rules=tuple(rules),
        label=spec["label"],
        label_noise=0.0,
        meta={"real": True, "n_features": len(feats)},
    )


def load_real(name: str, seed: int = 0, fractions=(0.6, 0.2, 0.2)):
    """Return (domain, train, val, test) with normalised frames and label arrays.

    Imputation and scaling are fitted on the training rows only, then applied to
    validation and test.  ``domain`` is built from those same training bounds,
    so expert thresholds live in the units the models actually see.
    """
    from .data import Dataset
    from .pipeline import Preprocessor

    Xraw, y = DATASETS[name]["loader"]()
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    a = int(fractions[0] * len(idx))
    b = int((fractions[0] + fractions[1]) * len(idx))
    tr_i, va_i, te_i = idx[:a], idx[a:b], idx[b:]

    prep = Preprocessor().fit(Xraw.iloc[tr_i], y[tr_i])
    lo, hi = prep.raw_bounds()
    domain = build_domain(name, lo, hi, features=prep.kept_)

    def mk(ix):
        Xn = prep.transform(Xraw.iloc[ix])
        return Dataset(Xn, y[ix], pd.DataFrame(index=range(len(ix))), domain)

    return domain, mk(tr_i), mk(va_i), mk(te_i)


def make_oof_folds(name: str, n_folds: int = 5, seed: int = 0,
                   stratify: bool = True):
    """Yield leak-free (domain, train, val, test) splits for nested CV.

    Each fold is a disjoint test block covering the whole dataset exactly once
    (out-of-fold), so no observation is reused across folds and seed-level
    pseudo-replication is impossible.  The validation block used for tuning is
    carved out of the *training* rows only.
    """
    from sklearn.model_selection import StratifiedKFold

    from .data import Dataset
    from .pipeline import Preprocessor

    Xraw, y = DATASETS[name]["loader"]()
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (tr_all, te_i) in enumerate(splitter.split(Xraw, y)):
        # inner validation carved from training rows only
        inner = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed + 100)
        tr_i, va_i = next(inner.split(Xraw.iloc[tr_all], y[tr_all]))
        tr_i, va_i = tr_all[tr_i], tr_all[va_i]

        prep = Preprocessor().fit(Xraw.iloc[tr_i], y[tr_i])
        lo, hi = prep.raw_bounds()
        domain = build_domain(name, lo, hi, features=prep.kept_)

        def mk(ix):
            return Dataset(prep.transform(Xraw.iloc[ix]), y[ix],
                           pd.DataFrame(index=range(len(ix))), domain)

        yield dict(fold=fold, prep=prep, domain=domain,
                   train=mk(tr_i), val=mk(va_i), test=mk(te_i),
                   tr_i=tr_i, va_i=va_i, te_i=te_i)
